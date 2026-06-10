"""Paper trading engine for the first safe implementation."""

from __future__ import annotations

import json
import random
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import pandas as pd

from broker import build_broker
from commentary import generate_buy_comment, generate_no_trade_comment, generate_sell_comment
from market_sections import market_section_allowed
from market_regime import classify_market_regime, dynamic_exposure_policy, dynamic_exposure_target
from ml.backtest_exit_ai import apply_exit_ai_to_plan, exit_ai_trade_fields, get_exit_ai_advisor, update_unrealized_extrema
from ml.data_loader import JQuantsDataLoader
from ml.portfolio_manager_sizing import (
    DEFAULT_PM_DATASET,
    DEFAULT_PM_MODEL_DIR,
    DEFAULT_PM_V3_DATASET,
    DEFAULT_PM_V3_MODEL_DIR,
    EXPECTED_CLEAN_FEATURE_COUNT,
    EXPECTED_PM_V3_FEATURE_COUNT,
    PortfolioManagerSizingAdvisor,
    PortfolioManagerV3SizingAdvisor,
)
from ml.portfolio_manager_score_based_rule import apply_score_based_pm_rule, is_score_based_rule
from safety import can_trade, safety_event
from tax import calculate_period_profit_summary


_PM_SIZING_ADVISOR_CACHE: dict[tuple[Any, ...], PortfolioManagerSizingAdvisor | PortfolioManagerV3SizingAdvisor] = {}
_EXIT_AI_V2_GATE_ADVISOR_CACHE: dict[tuple[str, str, str], "ExitAIV2GateAdvisor"] = {}
_BEAR_PM_BOOSTER_REGIME_CACHE: dict[str, dict[str, str]] = {}


def execute_paper_trades(
    scoring_log: dict[str, Any],
    trade_decision_log: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    cash = float(config["portfolio"]["initial_cash"])
    initial_cash = float(config["portfolio"]["initial_cash"])
    max_allocation = float(config["portfolio"]["max_allocation_per_symbol"])
    allocation_limit = initial_cash * max_allocation
    stop_loss_pct = float(config["risk"]["stop_loss_pct"])
    take_profit_pct = float(config["risk"]["take_profit_pct"])
    max_holding_days = int(config["risk"]["max_holding_business_days"])
    selected = scoring_log["selected"]
    trade_date = trade_decision_log["date"]
    broker = build_broker(state, config)

    orders = []
    positions = []
    closed_trades = []
    realized_pnl = 0.0

    for item in selected:
        allocation = min(allocation_limit, cash)
        quantity = int(allocation // item["close_price"])
        if quantity <= 0:
            continue

        entry_price = float(item["close_price"])
        notional = quantity * entry_price
        cash -= notional
        order = {
            "order_id": f"{trade_decision_log['run_id']}-{item['code']}-BUY",
            "code": item["code"],
            "name": item["name"],
            "side": "BUY",
            "quantity": quantity,
            "price": entry_price,
            "notional": round(notional, 2),
            "allocation_limit": round(allocation_limit, 2),
            "status": "FILLED",
            "entry_date": trade_date,
            "reason": item["selection_reason"],
        }
        orders.append(order)

        mark_price = _simulate_mark_price(trade_decision_log["run_id"], item["code"], entry_price)
        unrealized_pct = (mark_price - entry_price) / entry_price
        holding_days = 1
        exit_reason = _exit_reason(
            unrealized_pct,
            stop_loss_pct,
            take_profit_pct,
            holding_days=holding_days,
            max_holding_days=max_holding_days,
        )
        position = {
            "code": item["code"],
            "name": item["name"],
            "quantity": quantity,
            "entry_date": trade_date,
            "entry_price": entry_price,
            "current_price": mark_price,
            "market_value": round(quantity * mark_price, 2),
            "unrealized_pnl": round(quantity * (mark_price - entry_price), 2),
            "unrealized_return_pct": round(unrealized_pct, 4),
            "holding_business_days": holding_days,
            "buy_reason": item["selection_reason"],
        }

        if exit_reason:
            proceeds = quantity * mark_price
            pnl = proceeds - notional
            cash += proceeds
            realized_pnl += pnl
            closed_trades.append(
                {
                    "code": item["code"],
                    "name": item["name"],
                    "quantity": quantity,
                    "entry_date": trade_date,
                    "exit_date": trade_date,
                    "holding_days": holding_days,
                    "entry_price": entry_price,
                    "exit_price": mark_price,
                    "buy_reason": item["selection_reason"],
                    "sell_reason": exit_reason,
                    "profit": round(pnl, 2),
                    "profit_rate": round(pnl / notional, 4),
                    "result": _trade_result(pnl),
                }
            )
        else:
            positions.append(position)

    portfolio_market_value = round(sum(position["market_value"] for position in positions), 2)
    total_assets = round(cash + portfolio_market_value, 2)
    cumulative_pnl = round(total_assets - initial_cash, 2)

    return {
        "run_id": trade_decision_log["run_id"],
        "date": trade_decision_log["date"],
        "dealer_id": config["dealer"]["id"],
        "orders": orders,
        "positions": positions,
        "closed_trades": closed_trades,
        "pnl": {
            "initial_cash": initial_cash,
            "cash": round(cash, 2),
            "portfolio_market_value": portfolio_market_value,
            "total_assets": total_assets,
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(sum(position["unrealized_pnl"] for position in positions), 2),
            "cumulative_pnl": cumulative_pnl,
            "cumulative_return_pct": round(cumulative_pnl / initial_cash, 4),
        },
    }


def initial_paper_state(config: dict[str, Any]) -> dict[str, Any]:
    initial_cash = float(config["portfolio"]["initial_cash"])
    return {
        "cash": initial_cash,
        "positions": [],
        "closed_trades": [],
        "asset_history": [],
    }


def execute_paper_trade_day(
    scoring_log: dict[str, Any],
    trade_decision_log: dict[str, Any],
    config: dict[str, Any],
    state: dict[str, Any],
    day_number: int,
) -> dict[str, Any]:
    """Execute one simulation day.

    Day1 only opens positions. From Day2 onward, existing positions are marked,
    exit rules are evaluated, then empty slots are filled with new buys.
    """
    cash = float(state["cash"])
    initial_cash = float(config["portfolio"]["initial_cash"])
    max_positions = int(config["portfolio"]["max_positions"])
    max_allocation = float(config["portfolio"]["max_allocation_per_symbol"])
    allocation_limit = initial_cash * max_allocation
    stop_loss_pct = float(config["risk"]["stop_loss_pct"])
    take_profit_pct = float(config["risk"]["take_profit_pct"])
    max_holding_days = int(config["risk"]["max_holding_business_days"])
    trade_date = trade_decision_log["date"]
    broker = build_broker(state, config)

    orders = []
    order_attempts = []
    safety_events = []
    day_closed_trades = []
    next_positions = []

    for position in state["positions"]:
        updated = _update_position_price(position, trade_decision_log["run_id"], day_number)
        exit_reason = ""
        if day_number >= 2:
            exit_reason = _exit_reason(
                updated["unrealized_return_pct"],
                stop_loss_pct,
                take_profit_pct,
                holding_days=updated["holding_business_days"],
                max_holding_days=max_holding_days,
            )

        if exit_reason:
            proceeds = updated["quantity"] * updated["current_price"]
            entry_notional = updated["quantity"] * updated["entry_price"]
            stop_loss_fields = _demo_stop_loss_fields(
                exit_reason,
                float(updated["entry_price"]),
                float(updated["current_price"]),
                stop_loss_pct,
                trade_date,
            )
            closed = _apply_exit_costs(
                {
                    "trade_id": f"{updated['entry_date']}_{trade_date}_{updated['code']}",
                    "action": "SELL",
                    "code": updated["code"],
                    "name": updated["name"],
                    "entry_date": updated["entry_date"],
                    "exit_date": trade_date,
                    "holding_days": updated["holding_business_days"],
                    "entry_price": updated["entry_price"],
                    "exit_price": updated["current_price"],
                    "shares": updated["quantity"],
                    "exit_reason": exit_reason,
                    "buy_reason": updated["buy_reason"],
                    **_position_feature_snapshot(updated),
                    **stop_loss_fields,
                },
                entry_notional,
                proceeds,
                float(updated.get("buy_commission", 0)),
                config,
            )
            closed["profit"] = closed["net_profit"]
            closed["profit_rate"] = closed["net_profit_rate"]
            closed["result"] = _trade_result(closed["gross_profit"])
            closed["dealer_comment"] = generate_sell_comment(closed, config)
            validation = can_trade(closed, _safety_portfolio(state, order_attempts + day_closed_trades, closed), config)
            if not validation["allowed"]:
                event = safety_event(trade_date, closed, validation)
                safety_events.append(event)
                order_attempts.append(_safety_rejected_order(closed, validation))
                next_positions.append(updated)
                continue
            filled = broker.place_sell_order(closed)
            cash += proceeds
            day_closed_trades.append(filled)
        else:
            next_positions.append(updated)

    held_codes = {position["code"] for position in next_positions}
    slots = max_positions - len(next_positions)
    for item in scoring_log["scores"]:
        if slots <= 0:
            break
        if item["code"] in held_codes:
            continue
        if item["confidence"] < float(config["scoring"]["confidence_min_for_buy"]):
            continue

        allocation = min(allocation_limit, cash)
        quantity, skipped_reason = _calculate_buy_shares(float(item["close_price"]), allocation, config)
        if quantity <= 0:
            order_attempts.append(
                _skipped_buy_attempt(
                    trade_id=f"{trade_decision_log['run_id']}-D{day_number:03d}-{item['code']}-SKIP",
                    action="SKIP_BUY",
                    code=item["code"],
                    name=item["name"],
                    trade_date=trade_date,
                    price=float(item["close_price"]),
                    allocation_limit=allocation,
                    score=item.get("total_score"),
                    reason=item.get("selection_reason") or "空き枠補充のため採点上位から採用",
                    skipped_reason=skipped_reason,
                    config=config,
                )
            )
            continue

        entry_price = float(item["close_price"])
        notional = quantity * entry_price
        buy_commission = _calculate_commission(notional, config)
        order = {
            "order_id": f"{trade_decision_log['run_id']}-D{day_number:03d}-{item['code']}-BUY",
            "action": "BUY",
            "code": item["code"],
            "name": item["name"],
            "side": "BUY",
            "quantity": quantity,
            "price": entry_price,
            "notional": round(notional, 2),
            "buy_commission": buy_commission,
            "allocation_limit": round(allocation_limit, 2),
            "status": "FILLED",
            "entry_date": trade_date,
            "reason": item["selection_reason"] or "空き枠補充のため採点上位から採用",
            "round_lot_size": _round_lot_size(config),
            "use_round_lot": _use_round_lot(config),
            "skipped_reason": "",
            "dealer_comment": generate_buy_comment(item, config),
            **_technical_snapshot(item),
        }
        validation = can_trade(order, _safety_portfolio(state, order_attempts + day_closed_trades, order), config)
        if not validation["allowed"]:
            event = safety_event(trade_date, order, validation)
            safety_events.append(event)
            order_attempts.append(_safety_rejected_order(order, validation))
            continue

        cash -= notional + buy_commission
        held_codes.add(item["code"])
        slots -= 1
        filled = broker.place_buy_order(order)
        orders.append(filled)
        order_attempts.append(filled)
        next_positions.append(
            {
                "code": item["code"],
                "name": item["name"],
                "quantity": quantity,
                "entry_date": trade_date,
                "entry_price": entry_price,
                "current_price": entry_price,
                "market_value": round(quantity * entry_price, 2),
                "buy_commission": buy_commission,
                "unrealized_pnl": 0.0,
                "unrealized_return_pct": 0.0,
                "holding_business_days": 1,
                "buy_reason": filled["reason"],
                **_technical_snapshot(item),
            }
        )

    state["cash"] = round(cash, 2)
    state["positions"] = next_positions
    state["closed_trades"].extend(day_closed_trades)

    portfolio_market_value = round(sum(position["market_value"] for position in next_positions), 2)
    total_assets = round(state["cash"] + portfolio_market_value, 2)
    state["asset_history"].append(total_assets)
    period_profit = calculate_period_profit_summary(state["closed_trades"], config)
    realized_pnl = period_profit["net_cumulative_profit"]
    gross_cumulative_profit = period_profit["gross_cumulative_profit"]
    net_cumulative_profit = period_profit["net_cumulative_profit"]
    total_commission = period_profit["total_commission"]
    estimated_tax_total = period_profit["estimated_tax_total"]
    cumulative_pnl = round(total_assets - initial_cash, 2)

    return {
        "run_id": trade_decision_log["run_id"],
        "date": trade_date,
        "day_number": day_number,
        "dealer_id": config["dealer"]["id"],
        "orders": orders,
        "order_attempts": order_attempts,
        "positions": next_positions,
        "closed_trades": day_closed_trades,
        "safety_events": safety_events,
        "all_closed_trades": list(state["closed_trades"]),
        "asset_history": list(state["asset_history"]),
        "pnl": {
            "initial_cash": initial_cash,
            "cash": state["cash"],
            "portfolio_market_value": portfolio_market_value,
            "total_assets": total_assets,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": round(sum(position["unrealized_pnl"] for position in next_positions), 2),
            "cumulative_pnl": cumulative_pnl,
            "cumulative_return_pct": round(cumulative_pnl / initial_cash, 4),
            "gross_cumulative_profit": gross_cumulative_profit,
            "net_cumulative_profit": net_cumulative_profit,
            "total_commission": total_commission,
            "estimated_tax_total": estimated_tax_total,
            "net_total_assets": round(initial_cash + net_cumulative_profit, 2),
        },
    }


def _simulate_mark_price(run_id: str, code: str, entry_price: float) -> float:
    rng = random.Random(f"{run_id}:{code}:paper")
    move = rng.choice([-0.035, -0.018, 0.012, 0.028, 0.064])
    return round(entry_price * (1 + move), 2)


def _update_position_price(position: dict[str, Any], run_id: str, day_number: int) -> dict[str, Any]:
    rng = random.Random(f"{run_id}:{position['code']}:day:{day_number}")
    move = rng.choice([-0.028, -0.015, -0.006, 0.008, 0.017, 0.031, 0.045])
    current_price = round(float(position["current_price"]) * (1 + move), 2)
    quantity = int(position["quantity"])
    entry_price = float(position["entry_price"])
    holding_days = int(position["holding_business_days"]) + 1
    return {
        **position,
        "current_price": current_price,
        "market_value": round(quantity * current_price, 2),
        "unrealized_pnl": round(quantity * (current_price - entry_price), 2),
        "unrealized_return_pct": round((current_price - entry_price) / entry_price, 4),
        "holding_business_days": holding_days,
    }


def _exit_reason(
    return_pct: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    holding_days: int,
    max_holding_days: int,
) -> str:
    if holding_days < 2:
        return ""
    if return_pct <= stop_loss_pct:
        return "損切りルールに到達"
    if return_pct >= take_profit_pct:
        return "利確ルールに到達"
    if holding_days >= max_holding_days:
        return "最大保有期間に到達"
    return ""


def _trade_result(pnl: float) -> str:
    if pnl > 0:
        return "WIN"
    if pnl < 0:
        return "LOSS"
    return "EVEN"


def _calculate_commission(notional: float, config: dict[str, Any]) -> float:
    costs = config.get("costs", {})
    if not costs.get("enabled", False):
        return 0.0
    commission = notional * float(costs.get("commission_rate", 0.0))
    min_commission = float(costs.get("min_commission", 0))
    if commission > 0:
        commission = max(commission, min_commission)
    return round(commission, 2)


def _apply_exit_costs(
    trade: dict[str, Any],
    entry_notional: float,
    exit_notional: float,
    buy_commission: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    gross_profit = exit_notional - entry_notional
    sell_commission = _calculate_commission(exit_notional, config)
    total_commission = buy_commission + sell_commission
    taxable_profit = max(gross_profit - total_commission, 0)
    costs = config.get("costs", {})
    estimated_tax = taxable_profit * float(costs.get("tax_rate", 0.0)) if costs.get("apply_tax_on_profit", True) else 0.0
    net_profit = gross_profit - total_commission - estimated_tax
    return {
        **trade,
        "gross_profit": round(gross_profit, 2),
        "gross_profit_rate": round(gross_profit / entry_notional, 4) if entry_notional else 0,
        "buy_commission": round(buy_commission, 2),
        "sell_commission": round(sell_commission, 2),
        "total_commission": round(total_commission, 2),
        "taxable_profit": round(taxable_profit, 2),
        "estimated_tax": round(estimated_tax, 2),
        "net_profit": round(net_profit, 2),
        "net_profit_rate": round(net_profit / entry_notional, 4) if entry_notional else 0,
    }


def _calculate_buy_shares(price: float, allocation: float, config: dict[str, Any]) -> tuple[int, str]:
    if price <= 0 or allocation <= 0:
        return 0, "insufficient_available_cash"
    if _use_round_lot(config):
        lot_size = _round_lot_size(config)
        minimum_amount = price * lot_size
        if minimum_amount > allocation:
            return 0, "round_lot_unaffordable"
        lots = int(allocation // minimum_amount)
        return lots * lot_size, ""
    if not bool(config.get("trading", {}).get("allow_fractional_shares", False)):
        shares = int(allocation // price)
        if shares <= 0:
            return 0, "round_lot_unaffordable"
        return shares, ""
    shares = int(allocation // price)
    if shares <= 0:
        return 0, "round_lot_unaffordable"
    return shares, ""


def _scaled_buy_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("scaled_buy")
    return policy if isinstance(policy, dict) else {}


def _scaled_buy_enabled(config: dict[str, Any]) -> bool:
    return bool(_scaled_buy_policy(config).get("enabled", False))


def _ai_purchase_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("ai_purchase_policy")
    return policy if isinstance(policy, dict) else {}


def _ai_purchase_enabled(config: dict[str, Any]) -> bool:
    return bool(_ai_purchase_policy(config).get("enabled", False))


def _purchase_audit_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("purchase_audit")
    return policy if isinstance(policy, dict) else {}


def _purchase_audit_enabled(config: dict[str, Any]) -> bool:
    return bool(_purchase_audit_policy(config).get("enabled", False) or _ai_purchase_enabled(config))


def _portfolio_manager_sizing_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("portfolio_manager_ai_sizing")
    return policy if isinstance(policy, dict) else {}


def _portfolio_manager_sizing_enabled(config: dict[str, Any]) -> bool:
    return bool(_portfolio_manager_sizing_policy(config).get("enabled", False))


def _portfolio_manager_disabled_equal_weight_enabled(config: dict[str, Any]) -> bool:
    policy = _portfolio_manager_sizing_policy(config)
    rule = str(policy.get("rule") or "")
    return _portfolio_manager_sizing_enabled(config) and rule in {
        "disabled_equal_weight",
        "pm_disabled_equal_weight",
    }


def _portfolio_manager_score_based_rule_enabled(config: dict[str, Any]) -> bool:
    policy = _portfolio_manager_sizing_policy(config)
    return _portfolio_manager_sizing_enabled(config) and is_score_based_rule(str(policy.get("rule") or ""))


def _relative_allocator_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("relative_allocator")
    return policy if isinstance(policy, dict) else {}


def _relative_allocator_enabled(config: dict[str, Any]) -> bool:
    return bool(_relative_allocator_policy(config).get("enabled", False))


def _portfolio_manager_low_score_skip_policy(config: dict[str, Any]) -> tuple[bool, float]:
    policy = _portfolio_manager_sizing_policy(config)
    enabled = bool(policy.get("low_score_skip_enabled", False))
    threshold = float(policy.get("low_score_skip_threshold", -0.20))
    return enabled, threshold


def _portfolio_manager_per_code_exposure_cap_policy(config: dict[str, Any]) -> tuple[bool, float]:
    policy = _portfolio_manager_sizing_policy(config)
    enabled = bool(policy.get("per_code_exposure_cap_enabled", False))
    rate = float(policy.get("per_code_exposure_cap_rate", 0.0) or 0.0)
    return enabled and rate > 0, rate


def _portfolio_manager_buy_ordering_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = _portfolio_manager_sizing_policy(config)
    mode = str(policy.get("buy_ordering_mode") or "default")
    return {
        "mode": mode,
        "pm_order_weight": float(policy.get("pm_order_weight", 0.0) or 0.0),
        "fallback_to_next_affordable_selected": bool(policy.get("fallback_to_next_affordable_selected", False)),
        "fallback_min_pm_score": float(policy.get("fallback_min_pm_score", 0.0) or 0.0),
        "fallback_min_pm_multiplier": float(policy.get("fallback_min_pm_multiplier", 1.0) or 1.0),
    }


def _portfolio_manager_pm_aware_ordering_enabled(config: dict[str, Any]) -> bool:
    policy = _portfolio_manager_buy_ordering_policy(config)
    return _portfolio_manager_sizing_enabled(config) and policy["mode"] == "pm_aware"


def _high_pm_min_hold_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = _portfolio_manager_sizing_policy(config)
    return {
        "enabled": bool(policy.get("high_pm_min_hold_enabled", False)),
        "minimum_hold_days": int(policy.get("high_pm_min_hold_days", 0) or 0),
        "min_pm_multiplier": float(policy.get("high_pm_min_hold_min_multiplier", 1.15) or 1.15),
    }


def _high_pm_min_hold_trade_fields(source: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "high_pm_min_hold_enabled",
        "high_pm_min_hold_days",
        "high_pm_min_hold_applied",
        "high_pm_min_hold_blocked_exit",
        "high_pm_min_hold_blocked_exit_count",
        "high_pm_min_hold_exit_reason_original",
        "high_pm_min_hold_release_date",
        "holding_days_at_exit_signal",
    ]
    return {key: source.get(key) for key in keys if key in source}


def _bear_pm_booster_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("bear_pm_booster", {}) if isinstance(config, dict) else {}
    return {
        "enabled": bool(policy.get("bear_pm_booster_enabled", policy.get("enabled", False))),
        "min_pm_multiplier": float(policy.get("min_pm_multiplier", 1.15) or 1.15),
        "booster_multiplier": float(policy.get("booster_multiplier", 1.5) or 1.5),
    }


def _bear_pm_booster_regime_for_date(trade_date: str, config: dict[str, Any]) -> str:
    if not _bear_pm_booster_policy(config)["enabled"]:
        return ""
    profile_id = str(config.get("profile_id") or "")
    cache_key = f"{profile_id}:2022-01-01:2026-12-31"
    if cache_key not in _BEAR_PM_BOOSTER_REGIME_CACHE:
        _BEAR_PM_BOOSTER_REGIME_CACHE[cache_key] = _load_topix_ma_regime_map("2022-01-01", "2026-12-31")
    return _BEAR_PM_BOOSTER_REGIME_CACHE.get(cache_key, {}).get(str(trade_date), "Unknown")


def _load_topix_ma_regime_map(start_date: str, end_date: str) -> dict[str, str]:
    try:
        cache_root = Path(__file__).resolve().parents[1] / "data" / "cache" / "jquants"
        topix = JQuantsDataLoader(cache_root).load_topix(start_date, end_date)
    except Exception:
        return {}
    if topix.empty or not {"date", "close"}.issubset(topix.columns):
        return {}
    frame = topix.copy().dropna(subset=["date", "close"]).sort_values("date")
    frame = frame.drop_duplicates("date", keep="last")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    if frame.empty:
        return {}
    frame["ma25"] = frame["close"].rolling(25, min_periods=1).mean()
    frame["ma75"] = frame["close"].rolling(75, min_periods=1).mean()
    frame["market_regime"] = frame.apply(_classify_topix_ma_regime_row, axis=1)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return dict(zip(frame["date"], frame["market_regime"]))


def _classify_topix_ma_regime_row(row: pd.Series) -> str:
    close = pd.to_numeric(pd.Series([row.get("close")]), errors="coerce").iloc[0]
    ma25 = pd.to_numeric(pd.Series([row.get("ma25")]), errors="coerce").iloc[0]
    ma75 = pd.to_numeric(pd.Series([row.get("ma75")]), errors="coerce").iloc[0]
    if pd.isna(close) or pd.isna(ma25) or pd.isna(ma75):
        return "Unknown"
    if close < ma75 and ma25 < ma75:
        return "Bear"
    if close > ma75 and ma25 > ma75:
        return "Bull"
    return "Neutral"


def _bear_pm_booster_trade_fields(source: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "market_regime",
        "bear_pm_booster_enabled",
        "bear_pm_booster_applied",
        "bear_pm_booster_multiplier",
        "bear_pm_booster_reason",
        "bear_pm_booster_before_amount",
        "bear_pm_booster_after_amount",
        "bear_pm_booster_limited_by",
        "bear_pm_booster_pm_multiplier",
        "bear_pm_booster_pm_score",
    ]
    return {key: source.get(key) for key in keys if key in source}


def _exit_ai_v2_gate_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("ml_exit_ai_v2_gate", {}) if isinstance(config, dict) else {}
    return {
        "enabled": bool(policy.get("enabled", False)),
        "model_dir": str(policy.get("model_dir") or "models/ml/exit_ai_v2/candidate_v2_api_only"),
        "dataset_path": str(policy.get("dataset_path") or "data/ml/exit_ai_v2/exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet"),
        "score_threshold": float(policy.get("score_threshold", 0.20) or 0.20),
        "high_pm_safe_mode": bool(policy.get("high_pm_safe_mode", False)),
        "high_pm_min_multiplier": float(policy.get("high_pm_min_multiplier", 1.15) or 1.15),
        "high_pm_score_threshold": float(policy.get("high_pm_score_threshold", 0.25) or 0.25),
        "model_version": str(policy.get("model_version") or "exit_ai_v2_candidate_v2_api_only"),
        "selected_count_in_day_forbidden": bool(policy.get("selected_count_in_day_forbidden", True)),
    }


def _exit_ai_v2_gate_trade_fields(source: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "exit_ai_v2_enabled",
        "exit_ai_v2_prediction_available",
        "exit_ai_v2_score",
        "exit_ai_v2_threshold",
        "exit_ai_v2_gate_signal",
        "exit_ai_v2_gate_reason",
        "exit_ai_v2_feature_missing_count",
        "exit_ai_v2_model_version",
        "exit_ai_v2_used_as_exit_trigger",
        "exit_ai_v2_high_pm_safe_mode",
        "exit_ai_v2_high_pm_threshold",
    ]
    return {key: source.get(key) for key in keys if key in source}


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return Path(__file__).resolve().parents[1] / value


def _normalize_stock_code(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


class ExitAIV2GateAdvisor:
    """Read-only scorer for the Phase 5-F candidate Exit AI v2 model."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        dataset_path: str | Path,
        score_threshold: float,
        high_pm_safe_mode: bool,
        high_pm_min_multiplier: float,
        high_pm_score_threshold: float,
        model_version: str,
        selected_count_in_day_forbidden: bool = True,
    ) -> None:
        self.model_dir = _repo_path(model_dir)
        self.dataset_path = _repo_path(dataset_path)
        self.score_threshold = float(score_threshold)
        self.high_pm_safe_mode = bool(high_pm_safe_mode)
        self.high_pm_min_multiplier = float(high_pm_min_multiplier)
        self.high_pm_score_threshold = float(high_pm_score_threshold)
        self.model_version = model_version
        self.selected_count_in_day_forbidden = bool(selected_count_in_day_forbidden)
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.preprocess: dict[str, Any] = {}
        self.feature_columns: list[str] = []
        self.dataset: pd.DataFrame | None = None
        self.available = False
        self.unavailable_reason = ""
        self._load()

    def _load(self) -> None:
        try:
            import joblib

            metadata_path = self.model_dir / "model_metadata.json"
            preprocess_path = self.model_dir / "preprocess.json"
            model_path = self.model_dir / "exit_quality_top_decile_classifier.joblib"
            if not metadata_path.exists() or not preprocess_path.exists() or not model_path.exists():
                self.unavailable_reason = "model_bundle_missing"
                return
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
            metadata_features = self.metadata.get("feature_columns")
            preprocess_features = self.preprocess.get("feature_columns")
            if not isinstance(metadata_features, list) or not isinstance(preprocess_features, list):
                self.unavailable_reason = "feature_schema_missing"
                return
            if metadata_features != preprocess_features:
                self.unavailable_reason = "feature_schema_mismatch"
                return
            if self.selected_count_in_day_forbidden and "selected_count_in_day" in metadata_features:
                self.unavailable_reason = "selected_count_in_day_forbidden"
                return
            self.feature_columns = [str(column) for column in metadata_features]
            columns = ["code", "as_of_date", *self.feature_columns]
            self.dataset = pd.read_parquet(self.dataset_path, columns=columns)
            self.dataset["code"] = self.dataset["code"].map(_normalize_stock_code)
            self.dataset["as_of_date"] = pd.to_datetime(self.dataset["as_of_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            self.dataset = self.dataset.dropna(subset=["code", "as_of_date"]).set_index(["code", "as_of_date"], drop=False)
            self.model = joblib.load(model_path)
            self.available = True
        except Exception as exc:  # pragma: no cover - defensive fallback for backtest safety
            self.unavailable_reason = f"load_failed:{type(exc).__name__}"
            self.available = False

    def decision_for(self, *, code: Any, trade_date: str, pm_multiplier: Any) -> dict[str, Any]:
        fields = {
            "exit_ai_v2_enabled": True,
            "exit_ai_v2_prediction_available": False,
            "exit_ai_v2_score": None,
            "exit_ai_v2_threshold": self.score_threshold,
            "exit_ai_v2_gate_signal": False,
            "exit_ai_v2_gate_reason": self.unavailable_reason,
            "exit_ai_v2_feature_missing_count": None,
            "exit_ai_v2_model_version": self.model_version,
            "exit_ai_v2_used_as_exit_trigger": False,
            "exit_ai_v2_high_pm_safe_mode": self.high_pm_safe_mode,
            "exit_ai_v2_high_pm_threshold": self.high_pm_score_threshold if self.high_pm_safe_mode else None,
        }
        if not self.available or self.model is None or self.dataset is None:
            return fields
        key = (_normalize_stock_code(code), str(trade_date))
        if key not in self.dataset.index:
            fields["exit_ai_v2_gate_reason"] = "dataset_row_missing"
            return fields
        row = self.dataset.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        feature_row = pd.DataFrame([row[self.feature_columns].to_dict()])
        missing_count = int(feature_row.isna().sum(axis=1).iloc[0])
        transformed = self._transform(feature_row)
        score = float(self.model.predict_proba(transformed)[:, 1][0])
        pm_value = _optional_float(pm_multiplier)
        threshold = self.score_threshold
        if self.high_pm_safe_mode and pm_value is not None and pm_value >= self.high_pm_min_multiplier:
            threshold = self.high_pm_score_threshold
        fields.update(
            {
                "exit_ai_v2_prediction_available": True,
                "exit_ai_v2_score": score,
                "exit_ai_v2_threshold": threshold,
                "exit_ai_v2_gate_signal": score >= threshold,
                "exit_ai_v2_gate_reason": "score_above_threshold" if score >= threshold else "score_below_threshold",
                "exit_ai_v2_feature_missing_count": missing_count,
            }
        )
        return fields

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed: dict[str, Any] = {}
        medians = self.preprocess.get("medians", {})
        modes = self.preprocess.get("modes", {})
        for column in self.preprocess.get("numeric_columns", []):
            transformed[column] = pd.to_numeric(frame[column], errors="coerce").fillna(medians.get(column, 0.0))
        for column in self.preprocess.get("categorical_columns", []):
            filled = frame[column].fillna(modes.get(column, "")).astype("category")
            transformed[column] = filled.cat.codes.astype(float)
        output = pd.DataFrame(transformed, index=frame.index)
        for column in self.preprocess.get("missing_indicator_columns", []):
            output[f"{column}_missing"] = frame[column].isna().astype(int)
        return output


def _exit_ai_v2_gate_advisor(config: dict[str, Any]) -> ExitAIV2GateAdvisor | None:
    policy = _exit_ai_v2_gate_policy(config)
    if not policy["enabled"]:
        return None
    key = (policy["model_dir"], policy["dataset_path"], str(policy))
    if key not in _EXIT_AI_V2_GATE_ADVISOR_CACHE:
        _EXIT_AI_V2_GATE_ADVISOR_CACHE[key] = ExitAIV2GateAdvisor(
            model_dir=policy["model_dir"],
            dataset_path=policy["dataset_path"],
            score_threshold=policy["score_threshold"],
            high_pm_safe_mode=policy["high_pm_safe_mode"],
            high_pm_min_multiplier=policy["high_pm_min_multiplier"],
            high_pm_score_threshold=policy["high_pm_score_threshold"],
            model_version=policy["model_version"],
            selected_count_in_day_forbidden=policy["selected_count_in_day_forbidden"],
        )
    return _EXIT_AI_V2_GATE_ADVISOR_CACHE[key]


def _apply_exit_ai_v2_gate_to_plan(
    exit_plan: dict[str, Any],
    advisor: Any | None,
    *,
    position: dict[str, Any],
    trade_date: str,
    current_price: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if advisor is None:
        fields = {
            "exit_ai_v2_enabled": False,
            "exit_ai_v2_prediction_available": False,
            "exit_ai_v2_score": None,
            "exit_ai_v2_threshold": None,
            "exit_ai_v2_gate_signal": False,
            "exit_ai_v2_gate_reason": "disabled",
            "exit_ai_v2_feature_missing_count": None,
            "exit_ai_v2_model_version": None,
            "exit_ai_v2_used_as_exit_trigger": False,
            "exit_ai_v2_high_pm_safe_mode": False,
            "exit_ai_v2_high_pm_threshold": None,
        }
        return exit_plan, fields
    fields = advisor.decision_for(
        code=position.get("code"),
        trade_date=trade_date,
        pm_multiplier=position.get("pm_multiplier"),
    )
    if exit_plan.get("exit_reason"):
        return exit_plan, fields
    if not fields.get("exit_ai_v2_gate_signal"):
        return exit_plan, fields
    updated = dict(exit_plan)
    updated.update(
        {
            "exit_reason": "exit_ai_v2_gate",
            "exit_price": current_price,
            "intended_exit_price": current_price,
            "execute_now": True,
        }
    )
    fields["exit_ai_v2_used_as_exit_trigger"] = True
    return updated, fields


def _apply_high_pm_min_hold_exit_guard(
    exit_plan: dict[str, Any],
    position: dict[str, Any],
    config: dict[str, Any],
    trade_date: str,
    holding_days: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = _high_pm_min_hold_policy(config)
    fields: dict[str, Any] = {
        "high_pm_min_hold_enabled": policy["enabled"],
        "high_pm_min_hold_days": policy["minimum_hold_days"] if policy["enabled"] else None,
        "high_pm_min_hold_applied": False,
        "high_pm_min_hold_blocked_exit": False,
        "high_pm_min_hold_blocked_exit_count": int(float(position.get("high_pm_min_hold_blocked_exit_count") or 0)),
        "high_pm_min_hold_exit_reason_original": "",
        "high_pm_min_hold_release_date": "",
        "holding_days_at_exit_signal": holding_days if exit_plan.get("exit_reason") else None,
    }
    if not policy["enabled"] or policy["minimum_hold_days"] <= 0:
        return exit_plan, fields
    pm_multiplier = _optional_float(position.get("pm_multiplier"))
    high_pm = pm_multiplier is not None and pm_multiplier >= policy["min_pm_multiplier"]
    fields["high_pm_min_hold_applied"] = high_pm
    if not high_pm:
        return exit_plan, fields
    fields["high_pm_min_hold_release_date"] = _business_date_after_entry(str(position.get("entry_date") or ""), policy["minimum_hold_days"])
    if holding_days >= policy["minimum_hold_days"]:
        return exit_plan, fields
    if not bool(exit_plan.get("exit_ai_triggered")):
        return exit_plan, fields
    original_reason = str(exit_plan.get("exit_reason") or "")
    updated = dict(exit_plan)
    updated.update(
        {
            "exit_reason": "",
            "exit_price": None,
            "intended_exit_price": None,
            "execute_now": False,
            "exit_ai_triggered": False,
            "exit_ai_warning": (
                f"{str(exit_plan.get('exit_ai_warning') or '')};high_pm_min_hold_blocked"
                if exit_plan.get("exit_ai_warning")
                else "high_pm_min_hold_blocked"
            ),
        }
    )
    fields.update(
        {
            "high_pm_min_hold_blocked_exit": True,
            "high_pm_min_hold_blocked_exit_count": int(float(position.get("high_pm_min_hold_blocked_exit_count") or 0)) + 1,
            "high_pm_min_hold_exit_reason_original": original_reason,
            "holding_days_at_exit_signal": holding_days,
        }
    )
    return updated, fields


def _portfolio_manager_sizing_advisor(config: dict[str, Any]) -> PortfolioManagerSizingAdvisor | PortfolioManagerV3SizingAdvisor:
    policy = _portfolio_manager_sizing_policy(config)
    rule = str(policy.get("rule") or "")
    if rule in {"disabled_equal_weight", "pm_disabled_equal_weight"}:
        raise RuntimeError("PM sizing advisor is disabled by profile rule")
    if is_score_based_rule(rule):
        raise RuntimeError("PM sizing advisor is disabled for score-based PM rule")
    if rule == "pm_ai_v3_candidate":
        model_dir = str(policy.get("model_dir") or DEFAULT_PM_V3_MODEL_DIR)
        dataset_path = str(policy.get("dataset_path") or DEFAULT_PM_V3_DATASET)
        expected_feature_count = int(policy.get("expected_feature_count") or EXPECTED_PM_V3_FEATURE_COUNT)
        mapping_name = str(policy.get("pm_v3_mapping") or "mapping_a_rank_score_only")
        key = ("pm_v3", model_dir, dataset_path, expected_feature_count, mapping_name)
        if key not in _PM_SIZING_ADVISOR_CACHE:
            _PM_SIZING_ADVISOR_CACHE[key] = PortfolioManagerV3SizingAdvisor(
                model_dir=model_dir,
                dataset_path=dataset_path,
                expected_feature_count=expected_feature_count,
                mapping_name=mapping_name,
            )
        return _PM_SIZING_ADVISOR_CACHE[key]
    model_dir = str(policy.get("model_dir") or DEFAULT_PM_MODEL_DIR)
    dataset_path = str(policy.get("dataset_path") or DEFAULT_PM_DATASET)
    expected_feature_count = int(policy.get("expected_feature_count") or EXPECTED_CLEAN_FEATURE_COUNT)
    calibration_rule = str(policy.get("pm_calibration_rule") or policy.get("calibration_rule") or "")
    calibration_thresholds = policy.get("pm_calibration_thresholds") or policy.get("calibration_thresholds") or {}
    if not isinstance(calibration_thresholds, dict):
        calibration_thresholds = {}
    key = (model_dir, dataset_path, expected_feature_count, calibration_rule, str(sorted(calibration_thresholds.items())))
    if key not in _PM_SIZING_ADVISOR_CACHE:
        _PM_SIZING_ADVISOR_CACHE[key] = PortfolioManagerSizingAdvisor(
            model_dir=model_dir,
            dataset_path=dataset_path,
            expected_feature_count=expected_feature_count,
            calibration_rule=calibration_rule,
            calibration_thresholds=calibration_thresholds,
        )
    return _PM_SIZING_ADVISOR_CACHE[key]


def _portfolio_manager_disabled_equal_weight_fields(
    *,
    base_shares: int | None = None,
    entry_price: float | None = None,
    cash: float | None = None,
) -> dict[str, Any]:
    shares = int(base_shares or 0)
    price = float(entry_price or 0.0)
    base_amount = max(0.0, shares * price)
    fields: dict[str, Any] = {
        "pm_ai_enabled": False,
        "pm_status": "disabled",
        "pm_missing_reason": "pm_disabled",
        "pm_feature_count": 0,
        "pm_high_conviction_proba": None,
        "pm_avoid_proba": None,
        "pm_score": 0.0,
        "pm_multiplier": 1.0,
        "pm_model_version": "disabled",
        "pm_feature_found": False,
        "pm_warning": "pm_disabled",
        "pm_model_path": "",
        "pm_api_only_candidate_enabled": False,
        "pm_multiplier_source": "pm_disabled_equal_weight",
    }
    if base_shares is not None:
        target_amount = min(max(0.0, float(cash or 0.0)), base_amount) if cash is not None else base_amount
        fields.update(
            {
                "pm_base_planned_shares": shares,
                "pm_base_planned_amount": round(base_amount, 2),
                "pm_target_amount": round(base_amount, 2),
                "pm_cash_capped_target_amount": round(target_amount, 2),
                "pm_resized_shares": shares,
                "pm_resized_amount": round(base_amount, 2),
                "pm_resize_reason": "pm_disabled_equal_weight",
            }
        )
    return fields


def _ensure_portfolio_manager_decision_fields(item: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if not _portfolio_manager_sizing_enabled(config):
        return {}
    if _portfolio_manager_disabled_equal_weight_enabled(config):
        fields = _portfolio_manager_disabled_equal_weight_fields()
        item.update(fields)
        return fields
    if _portfolio_manager_score_based_rule_enabled(config):
        if "pm_score" in item and "pm_multiplier" in item and item.get("pm_multiplier_source") == "score_based_pm_rule":
            return _portfolio_manager_trade_fields(item)
        fields = {
            "pm_ai_enabled": False,
            "pm_status": "missing",
            "pm_missing_reason": "score_based_pm_fields_missing",
            "pm_feature_count": 0,
            "pm_high_conviction_proba": None,
            "pm_avoid_proba": None,
            "pm_score": 0.0,
            "pm_multiplier": 1.0,
            "pm_model_version": "score_based_pm_missing",
            "pm_feature_found": False,
            "pm_warning": "score_based_pm_fields_missing",
            "pm_model_path": "",
            "pm_api_only_candidate_enabled": False,
            "pm_multiplier_source": "score_based_pm_rule",
        }
        item.update(fields)
        return fields
    if "pm_score" in item and "pm_multiplier" in item and "pm_feature_found" in item:
        return _portfolio_manager_trade_fields(item)
    try:
        advisor = _portfolio_manager_sizing_advisor(config)
        decision = advisor.decision_for(str(item.get("signal_date") or item.get("date") or ""), str(item.get("code") or ""))
        fields = decision.as_fields()
    except Exception as exc:
        fields = {
            "pm_ai_enabled": True,
            "pm_status": "error",
            "pm_missing_reason": f"pm_decision_error:{type(exc).__name__}",
            "pm_feature_count": "",
            "pm_high_conviction_proba": None,
            "pm_avoid_proba": None,
            "pm_score": 0.0,
            "pm_multiplier": 1.0,
            "pm_model_version": "",
            "pm_feature_found": False,
            "pm_warning": f"pm_decision_error:{type(exc).__name__}",
        }
    item.update(fields)
    return fields


def _apply_portfolio_manager_sizing(
    *,
    item: dict[str, Any],
    trade_date: str,
    shares: int,
    entry_price: float,
    cash: float,
    config: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    if not _portfolio_manager_sizing_enabled(config):
        return shares, {}
    base_shares = int(shares or 0)
    base_amount = max(0.0, base_shares * float(entry_price or 0))
    fields: dict[str, Any] = {
        "pm_base_planned_shares": base_shares,
        "pm_base_planned_amount": round(base_amount, 2),
    }
    if _portfolio_manager_disabled_equal_weight_enabled(config):
        fields.update(
            _portfolio_manager_disabled_equal_weight_fields(
                base_shares=base_shares,
                entry_price=entry_price,
                cash=cash,
            )
        )
        item.update(fields)
        return shares, fields
    if base_shares <= 0 or entry_price <= 0:
        fields.update(
            {
                "pm_ai_enabled": True,
                "pm_status": "missing",
                "pm_missing_reason": "pm_no_base_shares",
                "pm_feature_count": "",
                "pm_high_conviction_proba": None,
                "pm_avoid_proba": None,
                "pm_score": 0.0,
                "pm_multiplier": 1.0,
                "pm_model_version": "",
                "pm_feature_found": False,
                "pm_warning": "pm_no_base_shares",
            }
        )
        item.update(fields)
        return shares, fields
    if _portfolio_manager_score_based_rule_enabled(config):
        score_fields = _portfolio_manager_trade_fields(item)
        fields.update(
            {
                "pm_ai_enabled": False,
                "pm_status": score_fields.get("pm_status", "ok"),
                "pm_missing_reason": score_fields.get("pm_missing_reason", ""),
                "pm_feature_count": score_fields.get("pm_feature_count", 0),
                "pm_high_conviction_proba": None,
                "pm_avoid_proba": None,
                "pm_score": score_fields.get("pm_score", 0.0),
                "pm_multiplier": score_fields.get("pm_multiplier", 1.0),
                "pm_model_version": score_fields.get("pm_model_version", "score_based_pm_missing"),
                "pm_feature_found": score_fields.get("pm_feature_found", False),
                "pm_warning": score_fields.get("pm_warning", ""),
                "pm_model_path": "",
                "pm_api_only_candidate_enabled": False,
                "pm_multiplier_source": "score_based_pm_rule",
            }
        )
    elif _relative_allocator_enabled(config):
        multiplier = float(item.get("relative_multiplier") or item.get("pm_multiplier") or 1.0)
        relative_score = _optional_float(item.get("relative_score"))
        fields.update(
            {
                "pm_ai_enabled": True,
                "pm_status": "ok" if item.get("relative_multiplier") is not None else "missing",
                "pm_missing_reason": "" if item.get("relative_multiplier") is not None else "relative_allocator_fields_missing",
                "pm_feature_count": 0,
                "pm_high_conviction_proba": None,
                "pm_avoid_proba": None,
                "pm_score": 0.0 if relative_score is None else relative_score,
                "pm_multiplier": multiplier,
                "pm_model_version": "relative_allocator_v1",
                "pm_feature_found": item.get("relative_multiplier") is not None,
                "pm_warning": "" if item.get("relative_multiplier") is not None else "relative_allocator_fields_missing",
                "pm_model_path": "",
                "pm_api_only_candidate_enabled": False,
                "pm_multiplier_source": "relative_allocator",
            }
        )
    else:
        try:
            advisor = _portfolio_manager_sizing_advisor(config)
            decision = advisor.decision_for(str(item.get("signal_date") or item.get("date") or trade_date), str(item.get("code") or ""))
            fields.update(decision.as_fields())
        except Exception as exc:  # Defensive: PM sizing must not break the base trading engine.
            fields.update(
                {
                    "pm_ai_enabled": True,
                    "pm_status": "error",
                    "pm_missing_reason": f"pm_sizing_error:{type(exc).__name__}",
                    "pm_feature_count": "",
                    "pm_high_conviction_proba": None,
                    "pm_avoid_proba": None,
                    "pm_score": 0.0,
                    "pm_multiplier": 1.0,
                    "pm_model_version": "",
                    "pm_feature_found": False,
                    "pm_warning": f"pm_sizing_error:{type(exc).__name__}",
                }
            )
    low_score_skip_enabled, low_score_skip_threshold = _portfolio_manager_low_score_skip_policy(config)
    if (
        low_score_skip_enabled
        and bool(fields.get("pm_feature_found"))
        and float(fields.get("pm_score") or 0.0) < low_score_skip_threshold
    ):
        fields.update(
            {
                "pm_status": "skipped",
                "pm_missing_reason": "",
                "pm_target_amount": 0.0,
                "pm_cash_capped_target_amount": 0.0,
                "pm_resized_shares": 0,
                "pm_resized_amount": 0.0,
                "pm_resize_reason": "pm_low_score_skip",
            }
        )
        item.update(fields)
        return 0, fields
    multiplier = float(fields.get("pm_multiplier") or 1.0)
    target_amount = max(0.0, min(float(cash or 0), base_amount * multiplier))
    resized_shares, resize_reason = _calculate_buy_shares(entry_price, target_amount, config)
    fields.update(
        {
            "pm_target_amount": round(base_amount * multiplier, 2),
            "pm_cash_capped_target_amount": round(target_amount, 2),
            "pm_resized_shares": resized_shares,
            "pm_resized_amount": round(resized_shares * entry_price, 2),
            "pm_resize_reason": resize_reason,
        }
    )
    item.update(fields)
    return resized_shares, fields


def _apply_bear_pm_booster(
    *,
    item: dict[str, Any],
    trade_date: str,
    shares: int,
    entry_price: float,
    cash: float,
    config: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    policy = _bear_pm_booster_policy(config)
    before_amount = max(0.0, int(shares or 0) * float(entry_price or 0))
    pm_multiplier = _optional_float(item.get("pm_multiplier"))
    pm_score = _optional_float(item.get("pm_score"))
    market_regime = _bear_pm_booster_regime_for_date(trade_date, config) if policy["enabled"] else str(item.get("market_regime") or "")
    fields: dict[str, Any] = {
        "market_regime": market_regime,
        "bear_pm_booster_enabled": bool(policy["enabled"]),
        "bear_pm_booster_applied": False,
        "bear_pm_booster_multiplier": policy["booster_multiplier"] if policy["enabled"] else "",
        "bear_pm_booster_reason": "",
        "bear_pm_booster_before_amount": round(before_amount, 2),
        "bear_pm_booster_after_amount": round(before_amount, 2),
        "bear_pm_booster_limited_by": "",
        "bear_pm_booster_pm_multiplier": pm_multiplier,
        "bear_pm_booster_pm_score": pm_score,
    }
    if not policy["enabled"]:
        fields["bear_pm_booster_reason"] = "disabled"
        item.update(fields)
        return shares, fields
    if shares <= 0 or entry_price <= 0:
        fields["bear_pm_booster_reason"] = "no_positive_shares"
        item.update(fields)
        return shares, fields
    if market_regime != "Bear":
        fields["bear_pm_booster_reason"] = "not_bear" if market_regime else "regime_unknown"
        item.update(fields)
        return shares, fields
    if pm_multiplier is None or pm_multiplier < policy["min_pm_multiplier"]:
        fields["bear_pm_booster_reason"] = "pm_multiplier_below_threshold"
        item.update(fields)
        return shares, fields

    desired_amount = before_amount * policy["booster_multiplier"]
    target_amount = min(max(0.0, float(cash or 0)), desired_amount)
    boosted_shares, resize_reason = _calculate_buy_shares(entry_price, target_amount, config)
    after_amount = max(0.0, boosted_shares * float(entry_price or 0))
    limited_by: list[str] = []
    if target_amount < desired_amount:
        limited_by.append("available_cash")
    if after_amount < desired_amount:
        limited_by.append("round_lot")
    fields.update(
        {
            "bear_pm_booster_applied": boosted_shares > shares,
            "bear_pm_booster_reason": "applied" if boosted_shares > shares else resize_reason or "not_increased_after_round_lot",
            "bear_pm_booster_after_amount": round(after_amount, 2),
            "bear_pm_booster_limited_by": ",".join(dict.fromkeys(limited_by)),
        }
    )
    item.update(fields)
    return max(shares, boosted_shares), fields


def _mark_bear_pm_booster_limited_by(item: dict[str, Any], fields: dict[str, Any], reason: str) -> dict[str, Any]:
    if not fields or not fields.get("bear_pm_booster_enabled"):
        return fields
    existing = str(fields.get("bear_pm_booster_limited_by") or "")
    parts = [part for part in existing.split(",") if part]
    if reason and reason not in parts:
        parts.append(reason)
    fields["bear_pm_booster_limited_by"] = ",".join(parts)
    item["bear_pm_booster_limited_by"] = fields["bear_pm_booster_limited_by"]
    return fields


def _portfolio_manager_trade_fields(source: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pm_ai_enabled",
        "pm_status",
        "pm_missing_reason",
        "pm_feature_count",
        "pm_high_conviction_proba",
        "pm_avoid_proba",
        "pm_score",
        "pm_multiplier",
        "pm_model_version",
        "pm_feature_found",
        "pm_warning",
        "pm_model_path",
        "pm_api_only_candidate_enabled",
        "pm_calibration_rule",
        "pm_calibration_thresholds",
        "pm_candidate_high_conviction_proba",
        "pm_candidate_avoid_proba",
        "pm_candidate_score",
        "pm_candidate_multiplier",
        "pm_candidate_multiplier_raw",
        "pm_candidate_multiplier_calibrated",
        "pm_candidate_feature_missing_count",
        "pm_candidate_prediction_available",
        "pm_candidate_fallback_reason",
        "pm_base_planned_shares",
        "pm_base_planned_amount",
        "pm_target_amount",
        "pm_cash_capped_target_amount",
        "pm_resized_shares",
        "pm_resized_amount",
        "pm_resize_reason",
        "pm_per_code_cap_enabled",
        "pm_per_code_cap_rate",
        "pm_per_code_current_exposure",
        "pm_per_code_max_exposure",
        "pm_per_code_allowed_additional_buy",
        "pm_per_code_cap_original_shares",
        "pm_per_code_cap_original_amount",
        "pm_per_code_cap_shares",
        "pm_per_code_cap_amount",
        "pm_per_code_cap_reduced",
        "pm_per_code_cap_skip",
        "pm_per_code_cap_reason",
        "relative_allocator_enabled",
        "relative_allocator_rule",
        "relative_candidate_count",
        "relative_rank",
        "relative_percentile",
        "relative_score",
        "relative_source_score",
        "relative_multiplier",
        "relative_multiplier_reason",
        "score_based_pm_enabled",
        "score_based_pm_rule",
        "score_based_pm_threshold_variant",
        "score_based_pm_weight_variant",
        "score_based_pm_candidate_count",
        "score_based_pm_rank",
        "score_based_pm_score",
        "score_based_pm_multiplier",
        "pm_rule_score",
        "pm_rule_score_percentile",
        "pm_rule_source",
        "pm_rule_threshold_variant",
        "pm_rule_weight_variant",
        "pm_rule_bucket",
        "pm_rule_risk_adjusted_score_percentile",
        "pm_rule_expected_return_percentile",
        "pm_rule_stock_selection_rank_score_percentile",
        "pm_rule_candidate_strength_percentile",
        "pm_multiplier_source",
        "buy_ordering_mode",
        "original_candidate_order",
        "pm_aware_candidate_order",
        "buy_priority_score",
        "fallback_enabled",
        "fallback_triggered",
        "fallback_from_code",
        "fallback_to_code",
        "fallback_from_reason",
        "fallback_to_pm_score",
        "fallback_to_pm_multiplier",
        "fallback_to_order",
        "skipped_by_fallback_quality_filter",
    ]
    fields = {key: source.get(key) for key in keys if key in source}
    fields.update(_high_pm_min_hold_trade_fields(source))
    fields.update(_bear_pm_booster_trade_fields(source))
    return fields


def _position_market_value(position: dict[str, Any]) -> float:
    value = position.get("market_value")
    if value not in {None, ""}:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    shares = float(position.get("shares") or position.get("quantity") or 0)
    price = float(position.get("current_price") or position.get("entry_price") or 0)
    return shares * price


def _current_code_exposure(positions: list[dict[str, Any]], code: str) -> float:
    target = str(code)
    return float(sum(_position_market_value(position) for position in positions if str(position.get("code") or "") == target))


def _apply_per_code_exposure_cap(
    *,
    item: dict[str, Any],
    shares: int,
    entry_price: float,
    positions: list[dict[str, Any]],
    total_assets: float,
    config: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    enabled, cap_rate = _portfolio_manager_per_code_exposure_cap_policy(config)
    if not enabled:
        return shares, {}
    code = str(item.get("code") or "")
    original_shares = int(shares or 0)
    original_amount = max(0.0, original_shares * float(entry_price or 0))
    current_exposure = _current_code_exposure(positions, code)
    max_exposure = max(0.0, float(total_assets or 0) * cap_rate)
    allowed = max_exposure - current_exposure
    fields: dict[str, Any] = {
        "pm_per_code_cap_enabled": True,
        "pm_per_code_cap_rate": cap_rate,
        "pm_per_code_current_exposure": round(current_exposure, 2),
        "pm_per_code_max_exposure": round(max_exposure, 2),
        "pm_per_code_allowed_additional_buy": round(allowed, 2),
        "pm_per_code_cap_original_shares": original_shares,
        "pm_per_code_cap_original_amount": round(original_amount, 2),
        "pm_per_code_cap_shares": original_shares,
        "pm_per_code_cap_amount": round(original_amount, 2),
        "pm_per_code_cap_reduced": False,
        "pm_per_code_cap_skip": False,
        "pm_per_code_cap_reason": "",
    }
    if original_shares <= 0 or entry_price <= 0:
        item.update(fields)
        return shares, fields
    if allowed <= 0:
        fields.update(
            {
                "pm_per_code_cap_shares": 0,
                "pm_per_code_cap_amount": 0.0,
                "pm_per_code_cap_reduced": True,
                "pm_per_code_cap_skip": True,
                "pm_per_code_cap_reason": "per_code_exposure_cap",
            }
        )
        item.update(fields)
        return 0, fields
    if original_amount <= allowed:
        item.update(fields)
        return shares, fields
    lot_size = _round_lot_size(config) if _use_round_lot(config) else 1
    capped_lots = int(allowed // (entry_price * lot_size))
    capped_shares = capped_lots * lot_size
    reason = "per_code_exposure_cap"
    if capped_shares <= 0:
        reason = "per_code_exposure_cap_scaled_below_round_lot"
    fields.update(
        {
            "pm_per_code_cap_shares": capped_shares,
            "pm_per_code_cap_amount": round(capped_shares * entry_price, 2),
            "pm_per_code_cap_reduced": True,
            "pm_per_code_cap_skip": capped_shares <= 0,
            "pm_per_code_cap_reason": reason,
        }
    )
    item.update(fields)
    return capped_shares, fields


def _daily_buy_limit_info(config: dict[str, Any], total_assets: float | None = None) -> dict[str, Any]:
    scaled_policy = _scaled_buy_policy(config)
    mode = str(scaled_policy.get("limit_mode") or "fixed")
    if mode == "unlimited":
        return {"limit": 0.0, "type": "unlimited", "ratio": None}
    if mode == "asset_ratio":
        ratio = float(scaled_policy.get("daily_buy_limit_ratio", 0) or 0)
        if ratio <= 0:
            return {"limit": 0.0, "type": "unlimited", "ratio": ratio}
        assets = float(total_assets or 0)
        return {"limit": max(0.0, assets * ratio), "type": "asset_ratio", "ratio": ratio}
    policy = _ai_purchase_policy(config)
    value = scaled_policy.get("daily_buy_limit")
    if value is None:
        value = policy.get("daily_buy_limit")
    if value is None:
        value = config.get("safety", {}).get("max_daily_buy_amount")
    return {"limit": float(value or 0), "type": "fixed", "ratio": None}


def _configured_daily_buy_limit(config: dict[str, Any], total_assets: float | None = None) -> float:
    return float(_daily_buy_limit_info(config, total_assets).get("limit") or 0)


def _daily_buy_amount_used(today_orders: list[dict[str, Any]]) -> float:
    return sum(
        float(order.get("amount") or order.get("notional") or 0)
        for order in today_orders
        if str(order.get("action") or order.get("side") or "").upper() == "BUY" and not order.get("rejected")
    )


def _daily_buy_limit_remaining(config: dict[str, Any], today_orders: list[dict[str, Any]], total_assets: float | None = None) -> float:
    limit = _configured_daily_buy_limit(config, total_assets)
    if limit <= 0:
        return float("inf")
    return max(0.0, limit - _daily_buy_amount_used(today_orders))


def _scale_buy_to_daily_limit(
    *,
    shares: int,
    price: float,
    config: dict[str, Any],
    today_orders: list[dict[str, Any]],
    total_assets: float | None = None,
) -> tuple[int, dict[str, Any]]:
    if not _scaled_buy_enabled(config) or shares <= 0 or price <= 0:
        return shares, {}
    limit_info = _daily_buy_limit_info(config, total_assets)
    max_daily_buy = float(limit_info.get("limit") or 0)
    if max_daily_buy <= 0:
        return shares, {}
    already_planned = _daily_buy_amount_used(today_orders)
    available_daily_buy = max(0.0, max_daily_buy - already_planned)
    original_amount = shares * price
    if original_amount <= available_daily_buy:
        return shares, {}
    lot_size = _round_lot_size(config) if _use_round_lot(config) else 1
    scaled_lots = int(available_daily_buy // (price * lot_size))
    scaled_shares = scaled_lots * lot_size
    if scaled_shares <= 0:
        return 0, {
            "scaled_buy_triggered": True,
            "original_planned_shares": shares,
            "scaled_shares": 0,
            "original_amount": round(original_amount, 2),
            "scaled_amount": 0.0,
            "scale_reason": "daily_buy_limit",
            "daily_buy_limit_type": limit_info.get("type"),
            "daily_buy_limit_ratio": limit_info.get("ratio"),
            "daily_buy_limit_applied": round(max_daily_buy, 2),
        }
    return scaled_shares, {
        "scaled_buy_triggered": True,
        "original_planned_shares": shares,
        "scaled_shares": scaled_shares,
        "original_amount": round(original_amount, 2),
        "scaled_amount": round(scaled_shares * price, 2),
        "scale_reason": "daily_buy_limit",
        "daily_buy_limit_type": limit_info.get("type"),
        "daily_buy_limit_ratio": limit_info.get("ratio"),
        "daily_buy_limit_applied": round(max_daily_buy, 2),
    }


def _candidate_rank_value(item: dict[str, Any], fallback: int | None = None) -> int:
    value = item.get("daily_score_rank")
    if value in {None, ""}:
        value = item.get("rank")
    try:
        rank = int(value)
    except (TypeError, ValueError):
        rank = int(fallback or 999999)
    return rank if rank > 0 else int(fallback or 999999)


def _candidate_risk_adjusted_score(item: dict[str, Any]) -> float:
    value = item.get("risk_adjusted_score")
    if value is not None and value != "":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    expected = item.get("expected_return_10d")
    bad_entry = item.get("bad_entry_probability_10d")
    try:
        expected_value = float(expected)
        bad_entry_value = float(bad_entry)
    except (TypeError, ValueError):
        return float(item.get("total_score") or item.get("score") or 0)
    return expected_value - 0.5 * bad_entry_value


def _ai_rank_position_ratio(config: dict[str, Any], rank: int) -> float:
    policy = _ai_purchase_policy(config)
    tiers = policy.get("rank_position_ratio_tiers")
    if isinstance(tiers, list):
        for tier in tiers:
            if not isinstance(tier, dict):
                continue
            max_rank = tier.get("max_rank")
            ratio = tier.get("ratio")
            if ratio is None:
                continue
            if max_rank in {None, ""} or rank <= int(max_rank):
                return float(ratio)
    ratios = policy.get("rank_position_ratios")
    if isinstance(ratios, dict):
        if rank <= 1 and ratios.get("rank1") is not None:
            return float(ratios["rank1"])
        if rank <= 3 and ratios.get("rank2_3") is not None:
            return float(ratios["rank2_3"])
        if ratios.get("rank4_plus") is not None:
            return float(ratios["rank4_plus"])
    return float(policy.get("max_position_amount_ratio", 0.3) or 0.3)


def _ai_purchase_allocation(
    *,
    cash: float,
    initial_cash: float,
    state: dict[str, Any],
    config: dict[str, Any],
    today_orders: list[dict[str, Any]],
    rank: int,
) -> tuple[float, str, dict[str, Any]]:
    policy = _ai_purchase_policy(config)
    if not _ai_purchase_enabled(config):
        return 0.0, "", {}
    daily_remaining = _daily_buy_limit_remaining(config, today_orders)
    total_assets = float(state.get("total_assets") or (cash + _current_market_exposure(state)) or initial_cash)
    rank_ratio = _ai_rank_position_ratio(config, rank)
    max_position_ratio = float(policy.get("max_position_amount_ratio", rank_ratio) or rank_ratio)
    max_position_abs = float(policy.get("max_position_amount_abs", 0) or 0)
    caps = [
        max(0.0, cash),
        daily_remaining if _configured_daily_buy_limit(config) > 0 else max(0.0, cash),
        max(0.0, total_assets * rank_ratio),
        max(0.0, total_assets * max_position_ratio),
    ]
    if max_position_abs > 0:
        caps.append(max_position_abs)
    allocation = max(0.0, min(caps))
    return allocation, "ai_purchase_policy", {
        "ai_purchase_enabled": True,
        "ai_purchase_rank_ratio": rank_ratio,
        "ai_purchase_max_position_amount_ratio": max_position_ratio,
        "ai_purchase_max_position_amount_abs": max_position_abs,
        "ai_purchase_total_assets": round(total_assets, 2),
        "daily_buy_limit_remaining_before": round(daily_remaining, 2),
    }


def _capital_utilization_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("capital_utilization_policy")
    return policy if isinstance(policy, dict) else {}


def _capital_utilization_enabled(config: dict[str, Any]) -> bool:
    return bool(_capital_utilization_policy(config).get("enabled", False))


def _current_market_exposure(state: dict[str, Any]) -> float:
    total = 0.0
    for position in state.get("positions", []) or []:
        if not isinstance(position, dict):
            continue
        value = position.get("market_value")
        if value is None:
            value = float(position.get("shares") or position.get("quantity") or 0) * float(
                position.get("current_price") or position.get("entry_price") or 0
            )
        total += float(value or 0)
    return total


def _available_buy_budget(
    cash: float,
    initial_cash: float,
    state: dict[str, Any],
    config: dict[str, Any],
    pending_buy_amount: float = 0.0,
    allocation_budget: float | None = None,
    market_context: dict[str, Any] | None = None,
) -> tuple[float, str]:
    base_allocation = initial_cash * float(config["portfolio"]["max_allocation_per_symbol"])
    policy = _capital_utilization_policy(config)
    if not bool(policy.get("enabled", False)):
        return max(0.0, min(base_allocation, cash - pending_buy_amount)), "legacy_allocation_limit"

    min_cash_buffer = float(policy.get("min_cash_buffer", 0) or 0)
    cash_available = cash - pending_buy_amount - min_cash_buffer
    if cash_available <= 0:
        return 0.0, "insufficient_available_cash"

    total_assets = float(state.get("total_assets") or (cash + _current_market_exposure(state)) or initial_cash)
    max_position_value_rate = policy.get("max_position_value_rate")
    if max_position_value_rate is not None:
        position_cap = total_assets * float(max_position_value_rate)
    else:
        position_cap = float(
            policy.get("max_position_size")
            or config.get("safety", {}).get("max_single_order_amount")
            or base_allocation
            or 0
        )
    if position_cap <= 0:
        position_cap = base_allocation
    if not bool(policy.get("buy_as_much_as_possible") or policy.get("allow_budget_reallocation", False)):
        position_cap = min(position_cap, base_allocation)

    target_exposure = policy.get("target_exposure")
    if target_exposure is not None:
        dynamic_regime = _dynamic_exposure_regime(market_context or {})
        target_exposure, _dynamic_triggered = dynamic_exposure_target(config, dynamic_regime, target_exposure)
    if target_exposure is not None:
        pending_for_exposure = 0.0 if str(policy.get("allocation_strategy") or "") == "relaxed_pending_target_exposure" else pending_buy_amount
        remaining_exposure = total_assets * float(target_exposure) - _current_market_exposure(state) - pending_for_exposure
        if remaining_exposure <= 0:
            return 0.0, "target_exposure_limit"
        position_cap = min(position_cap, remaining_exposure)

    if allocation_budget is not None:
        position_cap = min(position_cap, float(allocation_budget))
        return max(0.0, min(cash_available, position_cap)), "same_day_allocation_budget"

    return max(0.0, min(cash_available, position_cap)), "capital_utilization_policy"


def _allocation_strategy(config: dict[str, Any]) -> str:
    policy = _capital_utilization_policy(config)
    return str(policy.get("allocation_strategy") or "sequential").strip() or "sequential"


def _affordable_fallback_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("affordable_fallback_buy")
    return policy if isinstance(policy, dict) else {}


def _affordable_fallback_enabled(config: dict[str, Any]) -> bool:
    return bool(_affordable_fallback_policy(config).get("enabled", False))


def _affordable_fallback_replace_enabled(config: dict[str, Any]) -> bool:
    policy = _affordable_fallback_policy(config)
    return bool(policy.get("replace_unaffordable_selected", True))


def _affordable_fallback_surplus_enabled(config: dict[str, Any]) -> bool:
    return bool(_affordable_fallback_policy(config).get("surplus_after_selection", False))


def _affordable_fallback_max_buys_per_day(config: dict[str, Any]) -> int:
    policy = _affordable_fallback_policy(config)
    value = policy.get("max_fallback_buys_per_day", 999999)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 999999


def _affordable_fallback_float_setting(policy: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = policy.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _dynamic_exposure_regime(item: dict[str, Any]) -> str:
    existing = str(item.get("dynamic_exposure_regime") or item.get("classified_market_regime") or "")
    if existing:
        return existing
    return classify_market_regime(
        item.get("advance_ratio"),
        item.get("market_average_change_rate", item.get("average_change_rate")),
        item.get("market_regime"),
    )


def _dynamic_exposure_log_fields(config: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    policy = dynamic_exposure_policy(config)
    enabled = bool(policy.get("enabled", False))
    default_target = _capital_utilization_policy(config).get("target_exposure")
    regime = _dynamic_exposure_regime(item)
    target, triggered = dynamic_exposure_target(config, regime, default_target)
    return {
        "dynamic_exposure_enabled": enabled,
        "dynamic_exposure_regime": regime,
        "dynamic_target_exposure": target,
        "dynamic_exposure_triggered": bool(triggered),
        "dynamic_exposure_source_date": item.get("dynamic_exposure_source_date", ""),
        "dynamic_exposure_source_date_mode": item.get("dynamic_exposure_source_date_mode", "previous_trading_day"),
        "dynamic_exposure_source_lag_days": item.get("dynamic_exposure_source_lag_days"),
        "dynamic_exposure_source_fallback_used": bool(item.get("dynamic_exposure_source_fallback_used", False)),
        "dynamic_exposure_same_day_context_used": bool(item.get("dynamic_exposure_same_day_context_used", False)),
        "market_average_change_rate": item.get("market_average_change_rate", item.get("average_change_rate")),
        "classified_market_regime": item.get("classified_market_regime") or regime,
    }


def _candidate_round_lot_amount(item: dict[str, Any], config: dict[str, Any]) -> float:
    price = _candidate_entry_price(item)
    return price * _round_lot_size(config)


def _available_cash_after_buffer(cash: float, pending_buy_amount: float, config: dict[str, Any]) -> float:
    min_cash_buffer = float(_capital_utilization_policy(config).get("min_cash_buffer", 0) or 0)
    return max(0.0, cash - pending_buy_amount - min_cash_buffer)


def _is_affordable_fallback_skip_reason(reason: str) -> bool:
    return reason in {
        "selected_but_not_affordable",
        "round_lot_unaffordable",
        "insufficient_available_cash",
        "target_exposure_limit",
        "daily_buy_limit_scaled_below_round_lot",
        "per_code_exposure_cap_scaled_below_round_lot",
        "below_round_lot_after_pm_sizing",
        "pm_sizing_scaled_below_round_lot",
    }


def _phase3l_selected_fallback_enabled(config: dict[str, Any]) -> bool:
    return bool(_portfolio_manager_buy_ordering_policy(config)["fallback_to_next_affordable_selected"])


def _phase3l_apply_selected_fallback_fields(
    item: dict[str, Any],
    config: dict[str, Any],
    pending: dict[str, Any] | None,
) -> dict[str, Any]:
    enabled = _phase3l_selected_fallback_enabled(config)
    item["fallback_enabled"] = enabled
    if not pending:
        item.setdefault("fallback_triggered", False)
        item.setdefault("skipped_by_fallback_quality_filter", False)
        return {}
    _ensure_portfolio_manager_decision_fields(item, config)
    fields = {
        "fallback_enabled": enabled,
        "fallback_triggered": True,
        "fallback_from_code": pending.get("from_code", ""),
        "fallback_to_code": item.get("code", ""),
        "fallback_from_reason": pending.get("from_reason", ""),
        "fallback_to_pm_score": item.get("pm_score"),
        "fallback_to_pm_multiplier": item.get("pm_multiplier"),
        "fallback_to_order": item.get("pm_aware_candidate_order") or item.get("original_candidate_order") or _candidate_rank_value(item),
        "skipped_by_fallback_quality_filter": False,
    }
    item.update(fields)
    return fields


def _phase3l_selected_fallback_quality_allowed(item: dict[str, Any], config: dict[str, Any]) -> bool:
    policy = _portfolio_manager_buy_ordering_policy(config)
    pm_score = item.get("pm_score")
    pm_multiplier = item.get("pm_multiplier")
    try:
        if pm_score is not None and float(pm_score) >= float(policy["fallback_min_pm_score"]):
            return True
    except (TypeError, ValueError):
        pass
    try:
        if pm_multiplier is not None and float(pm_multiplier) >= float(policy["fallback_min_pm_multiplier"]):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _find_affordable_fallback_candidate(
    scored_candidates: list[dict[str, Any]],
    original: dict[str, Any],
    config: dict[str, Any],
    *,
    allocation_limit: float,
    cash: float,
    pending_buy_amount: float,
    held_codes: set[str],
    pending_buy_codes: set[str],
    same_day_regular_selected_codes: set[str] | None = None,
    diagnostics: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    if not _affordable_fallback_enabled(config) or allocation_limit <= 0:
        return None
    policy = _affordable_fallback_policy(config)
    selection = config.get("selection", {})
    regular_min_score = float(selection.get("min_score", 0) or 0)
    fallback_min_score = float(selection.get("fallback_min_score", selection.get("top_pick_min_score", regular_min_score)) or 0)
    configured_min_score = policy.get("min_total_score")
    configured_min_score = float(configured_min_score) if configured_min_score is not None else None
    configured_min_risk_adjusted = _affordable_fallback_float_setting(
        policy,
        "min_risk_adjusted_score",
        "fallback_min_risk_adjusted_score",
    )
    configured_min_expected_return = _affordable_fallback_float_setting(
        policy,
        "min_expected_return_10d",
        "fallback_min_expected_return_10d",
    )
    configured_max_bad_entry = _affordable_fallback_float_setting(
        policy,
        "max_bad_entry_probability_10d",
        "fallback_max_bad_entry_probability",
    )
    max_rank_in_day = policy.get("max_rank_in_day")
    max_rank_in_day = int(max_rank_in_day) if max_rank_in_day is not None else None
    fallback_top_k = int(policy.get("fallback_top_k", 50) or 50)
    min_turnover_value = float(policy.get("min_turnover_value", config.get("ml_backtest", {}).get("min_turnover_value", 0)) or 0)
    require_prediction = bool(policy.get("require_walk_forward_prediction", False))
    sort_key = str(policy.get("ranking") or config.get("ml_backtest", {}).get("ranking") or "risk_adjusted_score")
    min_confidence = float(selection.get("min_confidence", config.get("scoring", {}).get("confidence_min_for_buy", 0.0)) or 0)
    signal_date = str(original.get("signal_date") or original.get("date") or "")
    available_cash = _available_cash_after_buffer(cash, pending_buy_amount, config)
    blocked_codes = set(held_codes) | set(pending_buy_codes) | set(same_day_regular_selected_codes or set()) | {str(original.get("code") or "")}
    candidates = []
    for candidate in scored_candidates:
        if not isinstance(candidate, dict):
            continue
        code = str(candidate.get("code") or "")
        if not code or code in blocked_codes:
            continue
        candidate_signal_date = str(candidate.get("signal_date") or candidate.get("date") or "")
        if signal_date and candidate_signal_date != signal_date:
            continue
        if require_prediction and not candidate.get("ml_prediction_found", False):
            if diagnostics is not None:
                diagnostics["fallback_missing_prediction_count"] = diagnostics.get("fallback_missing_prediction_count", 0) + 1
            continue
        turnover = candidate.get("turnover_value")
        try:
            turnover_value = float(turnover)
        except (TypeError, ValueError):
            turnover_value = 0.0
        if min_turnover_value and turnover_value < min_turnover_value:
            if diagnostics is not None:
                diagnostics["fallback_low_turnover_count"] = diagnostics.get("fallback_low_turnover_count", 0) + 1
            continue
        if not market_section_allowed(candidate, config):
            continue
        if candidate.get("entry_price_available") is False:
            continue
        score = float(candidate.get("total_score") or candidate.get("score") or 0)
        if configured_min_score is not None and score < configured_min_score:
            if diagnostics is not None:
                diagnostics["fallback_score_below_min_count"] = diagnostics.get("fallback_score_below_min_count", 0) + 1
            continue
        rank_value = candidate.get("daily_score_rank") if candidate.get("daily_score_rank") is not None else candidate.get("rank")
        rank = int(rank_value) if rank_value not in {None, ""} else None
        if max_rank_in_day is not None and (rank is None or rank > max_rank_in_day):
            if diagnostics is not None:
                diagnostics["fallback_rank_out_of_range_count"] = diagnostics.get("fallback_rank_out_of_range_count", 0) + 1
            continue
        confidence = float(candidate.get("confidence") or 0)
        if confidence < min_confidence or score < fallback_min_score:
            continue
        risk_adjusted = _candidate_risk_adjusted_score(candidate)
        if configured_min_risk_adjusted is not None and risk_adjusted < configured_min_risk_adjusted:
            if diagnostics is not None:
                diagnostics["fallback_risk_adjusted_below_min_count"] = diagnostics.get("fallback_risk_adjusted_below_min_count", 0) + 1
            continue
        expected_return = candidate.get("expected_return_10d")
        try:
            expected_return_value = float(expected_return)
        except (TypeError, ValueError):
            expected_return_value = None
        if configured_min_expected_return is not None and (
            expected_return_value is None or expected_return_value < configured_min_expected_return
        ):
            if diagnostics is not None:
                diagnostics["fallback_expected_return_below_min_count"] = diagnostics.get("fallback_expected_return_below_min_count", 0) + 1
            continue
        bad_entry = candidate.get("bad_entry_probability_10d")
        try:
            bad_entry_value = float(bad_entry)
        except (TypeError, ValueError):
            bad_entry_value = None
        if configured_max_bad_entry is not None and (bad_entry_value is None or bad_entry_value > configured_max_bad_entry):
            if diagnostics is not None:
                diagnostics["fallback_bad_entry_above_max_count"] = diagnostics.get("fallback_bad_entry_above_max_count", 0) + 1
            continue
        round_lot_amount = _candidate_round_lot_amount(candidate, config)
        if round_lot_amount <= 0 or round_lot_amount > allocation_limit or round_lot_amount > available_cash:
            continue
        regular_priority = 0 if score >= regular_min_score else 1
        rank_sort = rank if rank is not None else 999999
        primary_sort = -risk_adjusted if sort_key == "risk_adjusted_score" else -score
        candidates.append((regular_priority, primary_sort, rank_sort, round_lot_amount, code, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    fallback_rank = 0
    selected_candidate = None
    for fallback_rank, candidate_tuple in enumerate(candidates[:fallback_top_k], start=1):
        selected_candidate = dict(candidate_tuple[5])
        break
    if selected_candidate is None:
        return None
    selected_candidate["candidate_source"] = "fallback"
    selected_candidate["selection_source"] = "affordable_fallback_buy"
    selected_candidate["fallback_rank"] = fallback_rank
    selected_candidate["raw_candidate_rank"] = _candidate_rank_value(original)
    return selected_candidate


def _find_surplus_affordable_fallback_candidate(
    scored_candidates: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    allocation_limit: float,
    cash: float,
    pending_buy_amount: float,
    held_codes: set[str],
    pending_buy_codes: set[str],
    same_day_regular_selected_codes: set[str],
    diagnostics: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    if not _affordable_fallback_enabled(config) or not _affordable_fallback_surplus_enabled(config):
        return None
    if allocation_limit <= 0:
        return None
    policy = _affordable_fallback_policy(config)
    selection = config.get("selection", {})
    regular_min_score = float(selection.get("min_score", 0) or 0)
    configured_min_score = policy.get("min_total_score")
    min_score = float(configured_min_score) if configured_min_score is not None else regular_min_score
    configured_min_risk_adjusted = _affordable_fallback_float_setting(
        policy,
        "min_risk_adjusted_score",
        "fallback_min_risk_adjusted_score",
    )
    configured_min_expected_return = _affordable_fallback_float_setting(
        policy,
        "min_expected_return_10d",
        "fallback_min_expected_return_10d",
    )
    configured_max_bad_entry = _affordable_fallback_float_setting(
        policy,
        "max_bad_entry_probability_10d",
        "fallback_max_bad_entry_probability",
    )
    max_rank_in_day = policy.get("max_rank_in_day")
    max_rank_in_day = int(max_rank_in_day) if max_rank_in_day is not None else None
    fallback_top_k = int(policy.get("fallback_top_k", 50) or 50)
    min_turnover_value = float(policy.get("min_turnover_value", config.get("ml_backtest", {}).get("min_turnover_value", 0)) or 0)
    require_prediction = bool(policy.get("require_walk_forward_prediction", False))
    sort_key = str(policy.get("ranking") or config.get("ml_backtest", {}).get("ranking") or "risk_adjusted_score")
    min_confidence = float(selection.get("min_confidence", config.get("scoring", {}).get("confidence_min_for_buy", 0.0)) or 0)
    available_cash = _available_cash_after_buffer(cash, pending_buy_amount, config)
    blocked_codes = set(held_codes) | set(pending_buy_codes) | set(same_day_regular_selected_codes)
    candidates = []
    for candidate in scored_candidates:
        if not isinstance(candidate, dict):
            continue
        code = str(candidate.get("code") or "")
        if not code or code in blocked_codes:
            continue
        if candidate.get("selected"):
            continue
        if require_prediction and not candidate.get("ml_prediction_found", False):
            if diagnostics is not None:
                diagnostics["missing_prediction"] = diagnostics.get("missing_prediction", 0) + 1
            continue
        turnover = candidate.get("turnover_value")
        try:
            turnover_value = float(turnover)
        except (TypeError, ValueError):
            turnover_value = 0.0
        if min_turnover_value and turnover_value < min_turnover_value:
            if diagnostics is not None:
                diagnostics["low_turnover"] = diagnostics.get("low_turnover", 0) + 1
            continue
        if not market_section_allowed(candidate, config):
            if diagnostics is not None:
                diagnostics["market_not_allowed"] = diagnostics.get("market_not_allowed", 0) + 1
            continue
        if candidate.get("entry_price_available") is False:
            if diagnostics is not None:
                diagnostics["entry_price_missing"] = diagnostics.get("entry_price_missing", 0) + 1
            continue
        score = float(candidate.get("total_score") or candidate.get("score") or 0)
        if score < min_score:
            if diagnostics is not None:
                diagnostics["score_below_min"] = diagnostics.get("score_below_min", 0) + 1
            continue
        confidence = float(candidate.get("confidence") or 0)
        if confidence < min_confidence:
            if diagnostics is not None:
                diagnostics["confidence_below_min"] = diagnostics.get("confidence_below_min", 0) + 1
            continue
        risk_adjusted = _candidate_risk_adjusted_score(candidate)
        if configured_min_risk_adjusted is not None and risk_adjusted < configured_min_risk_adjusted:
            if diagnostics is not None:
                diagnostics["risk_adjusted_below_min"] = diagnostics.get("risk_adjusted_below_min", 0) + 1
            continue
        expected_return = candidate.get("expected_return_10d")
        try:
            expected_return_value = float(expected_return)
        except (TypeError, ValueError):
            expected_return_value = None
        if configured_min_expected_return is not None and (
            expected_return_value is None or expected_return_value < configured_min_expected_return
        ):
            if diagnostics is not None:
                diagnostics["expected_return_below_min"] = diagnostics.get("expected_return_below_min", 0) + 1
            continue
        bad_entry = candidate.get("bad_entry_probability_10d")
        try:
            bad_entry_value = float(bad_entry)
        except (TypeError, ValueError):
            bad_entry_value = None
        if configured_max_bad_entry is not None and (bad_entry_value is None or bad_entry_value > configured_max_bad_entry):
            if diagnostics is not None:
                diagnostics["bad_entry_above_max"] = diagnostics.get("bad_entry_above_max", 0) + 1
            continue
        rank_value = candidate.get("daily_score_rank") if candidate.get("daily_score_rank") is not None else candidate.get("rank")
        rank = int(rank_value) if rank_value not in {None, ""} else None
        if max_rank_in_day is not None and (rank is None or rank > max_rank_in_day):
            if diagnostics is not None:
                diagnostics["rank_out_of_range"] = diagnostics.get("rank_out_of_range", 0) + 1
            continue
        round_lot_amount = _candidate_round_lot_amount(candidate, config)
        if round_lot_amount <= 0 or round_lot_amount > allocation_limit or round_lot_amount > available_cash:
            if diagnostics is not None:
                diagnostics["not_affordable"] = diagnostics.get("not_affordable", 0) + 1
            continue
        rank_sort = rank if rank is not None else 999999
        primary_sort = -risk_adjusted if sort_key == "risk_adjusted_score" else -score
        candidates.append((primary_sort, rank_sort, round_lot_amount, code, candidate))
    if diagnostics is not None:
        diagnostics["candidate_count"] = len(candidates)
    if not candidates or fallback_top_k <= 0:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    selected_candidate = dict(candidates[:fallback_top_k][0][4])
    selected_candidate["candidate_source"] = "fallback"
    selected_candidate["selection_source"] = "affordable_fallback_buy"
    selected_candidate["fallback_rank"] = 1
    selected_candidate["raw_candidate_rank"] = ""
    return selected_candidate


def _sort_selected_candidates(selected: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    if _ai_purchase_enabled(config):
        if _portfolio_manager_pm_aware_ordering_enabled(config):
            return _sort_selected_candidates_pm_aware(selected, config)
        return sorted(
            selected,
            key=lambda item: (
                _candidate_rank_value(item),
                -_candidate_risk_adjusted_score(item),
                str(item.get("code") or ""),
            ),
        )
    strategy = _allocation_strategy(config)
    if strategy != "round_lot_priority_near_score":
        return sorted(selected, key=lambda item: (float(item["total_score"]), float(item["confidence"])), reverse=True)

    tolerance = float(_capital_utilization_policy(config).get("round_lot_priority_score_tolerance", 3) or 3)

    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        left_score = float(left["total_score"])
        right_score = float(right["total_score"])
        if abs(left_score - right_score) <= tolerance:
            left_lot = _candidate_round_lot_amount(left, config)
            right_lot = _candidate_round_lot_amount(right, config)
            if left_lot != right_lot:
                return -1 if left_lot < right_lot else 1
        if left_score != right_score:
            return -1 if left_score > right_score else 1
        left_confidence = float(left["confidence"])
        right_confidence = float(right["confidence"])
        if left_confidence != right_confidence:
            return -1 if left_confidence > right_confidence else 1
        left_code = str(left.get("code") or "")
        right_code = str(right.get("code") or "")
        if left_code == right_code:
            return 0
        return -1 if left_code < right_code else 1

    return sorted(selected, key=cmp_to_key(compare))


def _sort_selected_candidates_pm_aware(selected: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    policy = _portfolio_manager_buy_ordering_policy(config)
    weight = float(policy["pm_order_weight"])
    items = [dict(item) for item in selected]
    for index, item in enumerate(items, start=1):
        original_order = _candidate_rank_value(item, index)
        item["buy_ordering_mode"] = "pm_aware"
        item["original_candidate_order"] = original_order
        _ensure_portfolio_manager_decision_fields(item, config)
    risk_values = [_candidate_risk_adjusted_score(item) for item in items]
    pm_values = [float(item.get("pm_score") or 0.0) for item in items]

    def normalize(value: float, values: list[float]) -> float:
        if not values:
            return 0.0
        low = min(values)
        high = max(values)
        if high == low:
            return 0.0
        return (value - low) / (high - low)

    for item in items:
        risk_norm = normalize(_candidate_risk_adjusted_score(item), risk_values)
        pm_norm = normalize(float(item.get("pm_score") or 0.0), pm_values)
        item["buy_priority_score"] = round(risk_norm + weight * pm_norm, 8)
    ordered = sorted(
        items,
        key=lambda item: (
            -float(item.get("buy_priority_score") or 0.0),
            _candidate_rank_value(item),
            str(item.get("code") or ""),
        ),
    )
    for index, item in enumerate(ordered, start=1):
        item["pm_aware_candidate_order"] = index
    return ordered


def _apply_relative_allocator_to_candidates(selected: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    policy = _relative_allocator_policy(config)
    enabled = bool(policy.get("enabled", False))
    if not selected:
        return selected
    if not enabled:
        for item in selected:
            item.setdefault("relative_allocator_enabled", False)
        return selected
    rule = str(policy.get("rule") or "blended_relative_score")
    items = [dict(item) for item in selected]
    score_columns = ["risk_adjusted_score", "expected_return_10d", "bad_entry_probability_10d"]
    percentiles = {column: _relative_percentiles(items, column, lower_is_better=(column == "bad_entry_probability_10d")) for column in score_columns}
    for item in items:
        code = str(item.get("code") or "")
        relative_score = _relative_allocator_score(code, rule, percentiles)
        rank = _relative_rank_for_score(relative_score, [score for score in (_relative_allocator_score(str(row.get("code") or ""), rule, percentiles) for row in items) if score is not None])
        candidate_count = len(items)
        percentile = 1.0 if candidate_count <= 1 else 1.0 - ((rank - 1.0) / float(candidate_count - 1))
        multiplier, reason = _relative_multiplier_for(rule, rank, percentile)
        source_score = _relative_source_score(item, rule)
        item.update(
            {
                "relative_allocator_enabled": True,
                "relative_allocator_rule": rule,
                "relative_candidate_count": candidate_count,
                "relative_rank": rank,
                "relative_percentile": percentile,
                "relative_score": relative_score,
                "relative_source_score": source_score,
                "relative_multiplier": multiplier,
                "relative_multiplier_reason": reason,
                "pm_multiplier_source": "relative_allocator",
                "pm_ai_enabled": True,
                "pm_status": "ok",
                "pm_missing_reason": "",
                "pm_feature_count": 0,
                "pm_high_conviction_proba": None,
                "pm_avoid_proba": None,
                "pm_score": relative_score,
                "pm_multiplier": multiplier,
                "pm_model_version": "relative_allocator_v1",
                "pm_feature_found": True,
                "pm_warning": "",
                "pm_model_path": "",
                "pm_api_only_candidate_enabled": False,
            }
        )
    return items


def _apply_score_based_pm_rule_to_candidates(selected: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    if not selected:
        return selected
    if not _portfolio_manager_score_based_rule_enabled(config):
        for item in selected:
            item.setdefault("score_based_pm_enabled", False)
        return selected
    policy = _portfolio_manager_sizing_policy(config)
    return apply_score_based_pm_rule(selected, policy)


def _relative_percentiles(items: list[dict[str, Any]], column: str, *, lower_is_better: bool = False) -> dict[str, float]:
    values: list[tuple[str, float]] = []
    for item in items:
        value = _optional_float(item.get(column))
        if value is not None:
            values.append((str(item.get("code") or ""), value))
    if not values:
        return {}
    ordered = sorted(values, key=lambda pair: (pair[1], pair[0]), reverse=not lower_is_better)
    count = len(ordered)
    out: dict[str, float] = {}
    for index, (code, _value) in enumerate(ordered, start=1):
        out[code] = 1.0 if count <= 1 else 1.0 - ((index - 1.0) / float(count - 1))
    return out


def _relative_allocator_score(code: str, rule: str, percentiles: dict[str, dict[str, float]]) -> float | None:
    rule = rule.strip().lower()
    if rule == "risk_adjusted_score_rank":
        return percentiles.get("risk_adjusted_score", {}).get(code)
    if rule == "expected_return_10d_rank":
        return percentiles.get("expected_return_10d", {}).get(code)
    risk = percentiles.get("risk_adjusted_score", {}).get(code)
    expected = percentiles.get("expected_return_10d", {}).get(code)
    bad_entry = percentiles.get("bad_entry_probability_10d", {}).get(code)
    parts = [(risk, 0.45), (expected, 0.35), (bad_entry, 0.20)]
    available = [(float(value), weight) for value, weight in parts if value is not None]
    if not available:
        return None
    weight_sum = sum(weight for _value, weight in available)
    return sum(value * weight for value, weight in available) / weight_sum if weight_sum else None


def _relative_rank_for_score(score: float | None, scores: list[float]) -> int:
    if score is None or not scores:
        return 999999
    better = sum(1 for value in scores if value > score)
    return int(better + 1)


def _relative_multiplier_for(rule: str, rank: int, percentile: float | None) -> tuple[float, str]:
    if percentile is None:
        return 1.0, "relative_score_missing"
    rule = rule.strip().lower()
    if rule == "no_pm_baseline":
        return 1.0, "no_pm_baseline"
    if rule == "conservative_blend":
        if rank == 1:
            return 1.30, "rank_1"
        if rank == 2 or percentile >= 0.70:
            return 1.15, "rank_2_or_top30pct"
        if percentile <= 0.30:
            return 0.80, "bottom30pct"
        return 1.00, "middle"
    if rank == 1 or percentile >= 0.90:
        return 1.30, "rank_1_or_top10pct"
    if percentile >= 0.75:
        return 1.15, "top25pct"
    if percentile <= 0.25:
        return 0.80, "bottom25pct"
    return 1.00, "middle"


def _relative_source_score(item: dict[str, Any], rule: str) -> Any:
    rule = rule.strip().lower()
    if rule == "risk_adjusted_score_rank":
        return item.get("risk_adjusted_score")
    if rule == "expected_return_10d_rank":
        return item.get("expected_return_10d")
    if rule == "no_pm_baseline":
        return ""
    return json.dumps(
        {
            "risk_adjusted_score": item.get("risk_adjusted_score"),
            "expected_return_10d": item.get("expected_return_10d"),
            "bad_entry_probability_10d": item.get("bad_entry_probability_10d"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _purchase_audit_event(
    *,
    item: dict[str, Any],
    trade_date: str,
    config: dict[str, Any],
    cash_before: float,
    cash_after: float,
    daily_buy_limit_remaining_before: float,
    daily_buy_limit_remaining_after: float,
    max_positions_remaining_before: int,
    planned_shares: int = 0,
    planned_amount: float = 0.0,
    scaled_shares: int = 0,
    scaled_amount: float = 0.0,
    final_shares: int = 0,
    final_amount: float = 0.0,
    decision: str = "SKIP",
    skip_reason: str = "",
    reject_reason: str = "",
    scale_reason: str = "",
    allocation_limit: float = 0.0,
    allocation_reason: str = "",
    daily_buy_limit_type: str = "",
    daily_buy_limit_ratio: float | None = None,
    daily_buy_limit_applied: float | None = None,
) -> dict[str, Any]:
    rank = _candidate_rank_value(item)
    score_rank = item.get("daily_score_rank") if item.get("daily_score_rank") not in {None, ""} else item.get("rank")
    expected_return = item.get("expected_return_10d")
    bad_entry = item.get("bad_entry_probability_10d")
    risk_adjusted = item.get("risk_adjusted_score")
    if risk_adjusted in {None, ""}:
        risk_adjusted = _candidate_risk_adjusted_score(item)
    candidate_source = item.get("candidate_source") or ("fallback" if item.get("affordable_fallback_buy_selected") else "selected")
    event = {
        "trade_id": f"{trade_date}_{item.get('code')}_PURCHASE_AUDIT",
        "action": "PURCHASE_AUDIT",
        "profile_id": config.get("profile_id", ""),
        "profile_name": config.get("profile_name", ""),
        "signal_date": item.get("signal_date") or item.get("date") or trade_date,
        "entry_date": trade_date,
        "code": item.get("code"),
        "name": item.get("name", ""),
        "candidate_source": candidate_source,
        "fallback_rank": item.get("fallback_rank", ""),
        "raw_candidate_rank": item.get("raw_candidate_rank", ""),
        "candidate_rank": rank,
        "score_rank": score_rank if score_rank is not None else rank,
        "risk_adjusted_score": risk_adjusted,
        "expected_return_10d": expected_return,
        "bad_entry_probability_10d": bad_entry,
        "cash_before": round(cash_before, 2),
        "cash_after": round(cash_after, 2),
        "daily_buy_limit_remaining_before": round(daily_buy_limit_remaining_before, 2),
        "daily_buy_limit_remaining_after": round(daily_buy_limit_remaining_after, 2),
        "daily_buy_limit_type": daily_buy_limit_type,
        "daily_buy_limit_ratio": daily_buy_limit_ratio,
        "daily_buy_limit_applied": round(daily_buy_limit_applied, 2) if daily_buy_limit_applied is not None else "",
        "max_positions_remaining_before": max_positions_remaining_before,
        "planned_shares": planned_shares,
        "planned_amount": round(planned_amount, 2),
        "scaled_shares": scaled_shares,
        "scaled_amount": round(scaled_amount, 2),
        "final_shares": final_shares,
        "final_amount": round(final_amount, 2),
        "decision": decision,
        "skip_reason": skip_reason,
        "reject_reason": reject_reason,
        "scale_reason": scale_reason,
        "allocation_limit": round(allocation_limit, 2),
        "allocation_reason": allocation_reason,
    }
    for key in [
        "buy_ordering_mode",
        "original_candidate_order",
        "pm_aware_candidate_order",
        "buy_priority_score",
        "fallback_enabled",
        "fallback_triggered",
        "fallback_from_code",
        "fallback_to_code",
        "fallback_from_reason",
        "fallback_to_pm_score",
        "fallback_to_pm_multiplier",
        "fallback_to_order",
        "skipped_by_fallback_quality_filter",
    ]:
        event[key] = item.get(key, "")
    for key in [
        "pm_ai_enabled",
        "pm_status",
        "pm_missing_reason",
        "pm_feature_count",
        "pm_high_conviction_proba",
        "pm_avoid_proba",
        "pm_score",
        "pm_multiplier",
        "pm_model_version",
        "pm_feature_found",
        "pm_warning",
        "pm_model_path",
        "pm_api_only_candidate_enabled",
        "pm_calibration_rule",
        "pm_calibration_thresholds",
        "pm_candidate_high_conviction_proba",
        "pm_candidate_avoid_proba",
        "pm_candidate_score",
        "pm_candidate_multiplier",
        "pm_candidate_multiplier_raw",
        "pm_candidate_multiplier_calibrated",
        "pm_candidate_feature_missing_count",
        "pm_candidate_prediction_available",
        "pm_candidate_fallback_reason",
        "pm_base_planned_shares",
        "pm_base_planned_amount",
        "pm_target_amount",
        "pm_cash_capped_target_amount",
        "pm_resized_shares",
        "pm_resized_amount",
        "pm_resize_reason",
        "pm_per_code_cap_enabled",
        "pm_per_code_cap_rate",
        "pm_per_code_current_exposure",
        "pm_per_code_max_exposure",
        "pm_per_code_allowed_additional_buy",
        "pm_per_code_cap_original_shares",
        "pm_per_code_cap_original_amount",
        "pm_per_code_cap_shares",
        "pm_per_code_cap_amount",
        "pm_per_code_cap_reduced",
        "pm_per_code_cap_skip",
        "pm_per_code_cap_reason",
        "relative_allocator_enabled",
        "relative_allocator_rule",
        "relative_candidate_count",
        "relative_rank",
        "relative_percentile",
        "relative_score",
        "relative_source_score",
        "relative_multiplier",
        "relative_multiplier_reason",
        "score_based_pm_enabled",
        "score_based_pm_rule",
        "score_based_pm_threshold_variant",
        "score_based_pm_weight_variant",
        "score_based_pm_candidate_count",
        "score_based_pm_rank",
        "score_based_pm_score",
        "score_based_pm_multiplier",
        "pm_rule_score",
        "pm_rule_score_percentile",
        "pm_rule_source",
        "pm_rule_threshold_variant",
        "pm_rule_weight_variant",
        "pm_rule_bucket",
        "pm_rule_risk_adjusted_score_percentile",
        "pm_rule_expected_return_percentile",
        "pm_rule_stock_selection_rank_score_percentile",
        "pm_rule_candidate_strength_percentile",
        "pm_multiplier_source",
        "bear_pm_booster_enabled",
        "bear_pm_booster_applied",
        "bear_pm_booster_multiplier",
        "bear_pm_booster_reason",
        "bear_pm_booster_before_amount",
        "bear_pm_booster_after_amount",
        "bear_pm_booster_limited_by",
        "bear_pm_booster_pm_multiplier",
        "bear_pm_booster_pm_score",
    ]:
        event[key] = item.get(key, "")
    return event


def _eligible_same_day_buy_candidates(
    selected: list[dict[str, Any]],
    next_positions: list[dict[str, Any]],
    pending_buy_codes: set[str],
    max_positions: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    held_codes = {position["code"] for position in next_positions}
    slots = max(0, max_positions - len(next_positions) - len(pending_buy_codes))
    eligible: list[dict[str, Any]] = []
    for item in selected:
        if len(eligible) >= slots:
            break
        code = str(item.get("code") or "")
        if code in held_codes or code in pending_buy_codes:
            continue
        if item.get("entry_price_available") is False:
            continue
        if not market_section_allowed(item, config):
            continue
        eligible.append(item)
    return eligible


def _same_day_allocation_budgets(
    selected: list[dict[str, Any]],
    cash: float,
    initial_cash: float,
    state: dict[str, Any],
    config: dict[str, Any],
    next_positions: list[dict[str, Any]],
    pending_buy_codes: set[str],
    max_positions: int,
) -> dict[str, float]:
    if _allocation_strategy(config) != "same_day_equal_budget":
        return {}
    eligible = _eligible_same_day_buy_candidates(selected, next_positions, pending_buy_codes, max_positions, config)
    if not eligible:
        return {}
    total_budget, _reason = _available_buy_budget(cash, initial_cash, state, config, pending_buy_amount=0.0)
    per_candidate = total_budget / len(eligible) if eligible else 0.0
    return {str(item.get("code") or ""): per_candidate for item in eligible}


def _skipped_buy_attempt(
    trade_id: str,
    action: str,
    code: str,
    name: str,
    trade_date: str,
    price: float,
    allocation_limit: float,
    score: Any,
    reason: str,
    skipped_reason: str,
    config: dict[str, Any],
    allocation_reason: str = "",
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "trade_id": trade_id,
        "action": action,
        "code": code,
        "name": name,
        "entry_date": trade_date,
        "entry_price": price,
        "shares": 0,
        "amount": 0,
        "allocation_limit": round(allocation_limit, 2),
        "allocation_reason": allocation_reason,
        "allocation_strategy": _allocation_strategy(config),
        "score": score,
        "reason": reason,
        "round_lot_size": _round_lot_size(config),
        "use_round_lot": _use_round_lot(config),
        "skipped_reason": skipped_reason,
        "dealer_comment": generate_no_trade_comment(skipped_reason, config),
    }
    if extra_fields:
        event.update(extra_fields)
    return event


def _use_round_lot(config: dict[str, Any]) -> bool:
    return bool(config.get("trading", {}).get("use_round_lot", False))


def _round_lot_size(config: dict[str, Any]) -> int:
    return int(config.get("trading", {}).get("round_lot_size", 100))


def _relative_strength_score_snapshot_value(item: dict[str, Any]) -> Any:
    if "relative_strength_score" not in item:
        return None
    value = item.get("relative_strength_score")
    return 0.0 if value is None else value


def _technical_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    selected_reason = item.get("selection_reason") or item.get("selected_reason") or item.get("reason", "")
    return {
        "rsi": item.get("rsi"),
        "volume_ratio": item.get("volume_ratio"),
        "total_score": item.get("total_score"),
        "technical_score": item.get("technical_score"),
        "selected_reason": selected_reason,
        "ml_prediction_found": item.get("ml_prediction_found"),
        "ml_prediction_source": item.get("ml_prediction_source"),
        "ml_turnover_filter_pass": item.get("ml_turnover_filter_pass"),
        "expected_return_10d": item.get("expected_return_10d"),
        "expected_max_return_20d": item.get("expected_max_return_20d"),
        "swing_success_probability_20d": item.get("swing_success_probability_20d"),
        "bad_entry_probability_10d": item.get("bad_entry_probability_10d"),
        "risk_adjusted_score": item.get("risk_adjusted_score"),
        "sector_name": item.get("sector_name"),
        "section": item.get("section"),
        "market_section": item.get("market_section"),
        "listing_market": item.get("listing_market"),
        "sector_momentum_score": item.get("sector_momentum_score"),
        "sector_rank": item.get("sector_rank"),
        "sector_comment": item.get("sector_comment"),
        "sector_score_adjustment": item.get("sector_score_adjustment"),
        "ma_score": item.get("ma_score") or item.get("trend_score"),
        "trend_score": item.get("trend_score"),
        "volume_score": item.get("volume_score"),
        "rsi_score": item.get("rsi_score"),
        "candlestick_score": item.get("candlestick_score"),
        "market_context_score": item.get("market_context_score"),
        "winner_loser_rule_score": item.get("winner_loser_rule_score"),
        "winner_loser_rule_name": item.get("winner_loser_rule_name"),
        "winner_loser_rule_reason": item.get("winner_loser_rule_reason"),
        "sector_score": item.get("sector_score") or item.get("sector_score_adjustment"),
        "penalty_score": item.get("penalty_score"),
        "score_components": item.get("score_components", {}),
        "score_components_total": item.get("score_components_total"),
        "score_components_match": item.get("score_components_match"),
        "candle_type": item.get("candle_type"),
        "candlestick_signals": item.get("candlestick_signals", []),
        "ma5": item.get("ma5"),
        "ma25": item.get("ma25"),
        "previous_ma25": item.get("previous_ma25"),
        "macd_hist": item.get("macd_hist"),
        "bb_position": item.get("bb_position"),
        "atr": item.get("atr"),
        "stock_return_5d": item.get("stock_return_5d"),
        "stock_return_10d": item.get("stock_return_10d"),
        "stock_return_20d": item.get("stock_return_20d"),
        "benchmark_source": item.get("benchmark_source"),
        "benchmark_return_5d": item.get("benchmark_return_5d"),
        "benchmark_return_10d": item.get("benchmark_return_10d"),
        "benchmark_return_20d": item.get("benchmark_return_20d"),
        "relative_strength_5d": item.get("relative_strength_5d"),
        "relative_strength_10d": item.get("relative_strength_10d"),
        "relative_strength_20d": item.get("relative_strength_20d"),
        "relative_strength_score": _relative_strength_score_snapshot_value(item),
        "investor_context_source": item.get("investor_context_source"),
        "investor_context_week": item.get("investor_context_week"),
        "overseas_net_buy": item.get("overseas_net_buy"),
        "overseas_net_buy_4w_sum": item.get("overseas_net_buy_4w_sum"),
        "overseas_net_buy_4w_trend": item.get("overseas_net_buy_4w_trend"),
        "overseas_buy_sell_ratio": item.get("overseas_buy_sell_ratio"),
        "individual_net_buy": item.get("individual_net_buy"),
        "institution_net_buy": item.get("institution_net_buy"),
        "trust_bank_net_buy": item.get("trust_bank_net_buy"),
        "proprietary_net_buy": item.get("proprietary_net_buy"),
        "investor_context_score": item.get("investor_context_score"),
        "market_filter_applied": item.get("market_filter_applied", False),
        "market_regime": item.get("market_regime"),
        "entry_market_regime": item.get("entry_market_regime") or item.get("market_regime"),
        "advance_ratio": item.get("advance_ratio"),
        "market_average_change_rate": item.get("market_average_change_rate"),
        "classified_market_regime": item.get("classified_market_regime"),
        "dynamic_exposure_enabled": item.get("dynamic_exposure_enabled"),
        "dynamic_exposure_regime": item.get("dynamic_exposure_regime"),
        "dynamic_target_exposure": item.get("dynamic_target_exposure"),
        "dynamic_exposure_triggered": item.get("dynamic_exposure_triggered"),
        "dynamic_exposure_source_date": item.get("dynamic_exposure_source_date"),
        "dynamic_exposure_source_date_mode": item.get("dynamic_exposure_source_date_mode"),
        "dynamic_exposure_source_lag_days": item.get("dynamic_exposure_source_lag_days"),
        "dynamic_exposure_source_fallback_used": item.get("dynamic_exposure_source_fallback_used"),
        "dynamic_exposure_same_day_context_used": item.get("dynamic_exposure_same_day_context_used"),
        "affordable_fallback_buy_selected": item.get("affordable_fallback_buy_selected", False),
        "candidate_source": item.get("candidate_source") or ("fallback" if item.get("affordable_fallback_buy_selected") else "selected"),
        "fallback_rank": item.get("fallback_rank"),
        "raw_candidate_rank": item.get("raw_candidate_rank"),
        "affordable_fallback_original_code": item.get("affordable_fallback_original_code"),
        "affordable_fallback_original_name": item.get("affordable_fallback_original_name"),
        "affordable_fallback_reason": item.get("affordable_fallback_reason"),
        "affordable_fallback_round_lot_amount": item.get("affordable_fallback_round_lot_amount"),
        "market_filter_reason": item.get("market_filter_reason", ""),
        "earnings_filter_checked": item.get("earnings_filter_checked", False),
        "earnings_filter_blocked": item.get("earnings_filter_blocked", False),
        "earnings_filter_reason": item.get("earnings_filter_reason", ""),
        "earnings_announcement_date": item.get("earnings_announcement_date"),
        "earnings_calendar_records_count": item.get("earnings_calendar_records_count"),
        "earnings_info_found": item.get("earnings_info_found", False),
        "earnings_candidate_date": item.get("earnings_candidate_date"),
        "earnings_days_until_earnings": item.get("earnings_days_until_earnings"),
        **_portfolio_manager_trade_fields(item),
    }


def _position_feature_snapshot(position: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rsi",
        "volume_ratio",
        "total_score",
        "technical_score",
        "selected_reason",
        "sector_name",
        "section",
        "market_section",
        "listing_market",
        "sector_momentum_score",
        "sector_rank",
        "sector_comment",
        "sector_score_adjustment",
        "ma_score",
        "trend_score",
        "volume_score",
        "rsi_score",
        "candlestick_score",
        "market_context_score",
        "winner_loser_rule_score",
        "winner_loser_rule_name",
        "winner_loser_rule_reason",
        "sector_score",
        "penalty_score",
        "score_components",
        "score_components_total",
        "score_components_match",
        "candle_type",
        "candlestick_signals",
        "ma5",
        "ma25",
        "previous_ma25",
        "macd_hist",
        "bb_position",
        "atr",
        "stock_return_5d",
        "stock_return_10d",
        "stock_return_20d",
        "benchmark_source",
        "benchmark_return_5d",
        "benchmark_return_10d",
        "benchmark_return_20d",
        "relative_strength_5d",
        "relative_strength_10d",
        "relative_strength_20d",
        "relative_strength_score",
        "investor_context_source",
        "investor_context_week",
        "overseas_net_buy",
        "overseas_net_buy_4w_sum",
        "overseas_net_buy_4w_trend",
        "overseas_buy_sell_ratio",
        "individual_net_buy",
        "institution_net_buy",
        "trust_bank_net_buy",
        "proprietary_net_buy",
        "investor_context_score",
        "market_filter_applied",
        "market_regime",
        "entry_market_regime",
        "advance_ratio",
        "market_average_change_rate",
        "classified_market_regime",
        "dynamic_exposure_enabled",
        "dynamic_exposure_regime",
        "dynamic_target_exposure",
        "dynamic_exposure_triggered",
        "dynamic_exposure_source_date",
        "dynamic_exposure_source_date_mode",
        "dynamic_exposure_source_lag_days",
        "dynamic_exposure_source_fallback_used",
        "dynamic_exposure_same_day_context_used",
        "candidate_source",
        "fallback_rank",
        "raw_candidate_rank",
        "affordable_fallback_buy_selected",
        "affordable_fallback_original_code",
        "affordable_fallback_original_name",
        "affordable_fallback_reason",
        "affordable_fallback_round_lot_amount",
        "entry_score",
        "ml_prediction_found",
        "ml_prediction_source",
        "ml_turnover_filter_pass",
        "expected_return_10d",
        "expected_max_return_20d",
        "swing_success_probability_20d",
        "bad_entry_probability_10d",
        "risk_adjusted_score",
        "holding_signal_status",
        "holding_entry_score",
        "holding_current_score",
        "holding_score_drop",
        "holding_effective_max_days",
        "holding_reselected",
        "holding_extended",
        "holding_signal_lost_streak",
        "holding_signal_lost_exit_avoided",
        "holding_signal_lost_exit_avoided_count",
        "holding_extension_count",
        "holding_extension_eligible",
        "holding_extension_reason",
        "holding_unrealized_profit_rate",
        "conditional_hold_extension_applied",
        "conditional_hold_extension_count",
        "conditional_hold_extension_reason",
        "conditional_hold_extension_trigger_profit_rate",
        "extension_profit_rate",
        "extension_exit_guard_triggered",
        "extension_exit_guard_reason",
        "conditional_hold_extension_rejected",
        "conditional_hold_extension_rejected_reason",
        "market_filter_reason",
        "earnings_filter_checked",
        "earnings_filter_blocked",
        "earnings_filter_reason",
        "earnings_announcement_date",
        "earnings_calendar_records_count",
        "earnings_info_found",
        "earnings_candidate_date",
        "earnings_days_until_earnings",
        "scaled_buy_triggered",
        "original_planned_shares",
        "scaled_shares",
        "original_amount",
        "scaled_amount",
        "scale_reason",
        "pm_ai_enabled",
        "pm_status",
        "pm_missing_reason",
        "pm_feature_count",
        "pm_high_conviction_proba",
        "pm_avoid_proba",
        "pm_score",
        "pm_multiplier",
        "pm_model_version",
        "pm_feature_found",
        "pm_warning",
        "pm_model_path",
        "pm_api_only_candidate_enabled",
        "pm_calibration_rule",
        "pm_calibration_thresholds",
        "pm_candidate_high_conviction_proba",
        "pm_candidate_avoid_proba",
        "pm_candidate_score",
        "pm_candidate_multiplier",
        "pm_candidate_multiplier_raw",
        "pm_candidate_multiplier_calibrated",
        "pm_candidate_feature_missing_count",
        "pm_candidate_prediction_available",
        "pm_candidate_fallback_reason",
        "pm_base_planned_shares",
        "pm_base_planned_amount",
        "pm_target_amount",
        "pm_cash_capped_target_amount",
        "pm_resized_shares",
        "pm_resized_amount",
        "pm_resize_reason",
        "pm_per_code_cap_enabled",
        "pm_per_code_cap_rate",
        "pm_per_code_current_exposure",
        "pm_per_code_max_exposure",
        "pm_per_code_allowed_additional_buy",
        "pm_per_code_cap_original_shares",
        "pm_per_code_cap_original_amount",
        "pm_per_code_cap_shares",
        "pm_per_code_cap_amount",
        "pm_per_code_cap_reduced",
        "pm_per_code_cap_skip",
        "pm_per_code_cap_reason",
        "relative_allocator_enabled",
        "relative_allocator_rule",
        "relative_candidate_count",
        "relative_rank",
        "relative_percentile",
        "relative_score",
        "relative_source_score",
        "relative_multiplier",
        "relative_multiplier_reason",
        "score_based_pm_enabled",
        "score_based_pm_rule",
        "score_based_pm_threshold_variant",
        "score_based_pm_weight_variant",
        "score_based_pm_candidate_count",
        "score_based_pm_rank",
        "score_based_pm_score",
        "score_based_pm_multiplier",
        "pm_rule_score",
        "pm_rule_score_percentile",
        "pm_rule_source",
        "pm_rule_threshold_variant",
        "pm_rule_weight_variant",
        "pm_rule_bucket",
        "pm_rule_risk_adjusted_score_percentile",
        "pm_rule_expected_return_percentile",
        "pm_rule_stock_selection_rank_score_percentile",
        "pm_rule_candidate_strength_percentile",
        "pm_multiplier_source",
        "bear_pm_booster_enabled",
        "bear_pm_booster_applied",
        "bear_pm_booster_multiplier",
        "bear_pm_booster_reason",
        "bear_pm_booster_before_amount",
        "bear_pm_booster_after_amount",
        "bear_pm_booster_limited_by",
        "bear_pm_booster_pm_multiplier",
        "bear_pm_booster_pm_score",
        "high_pm_min_hold_enabled",
        "high_pm_min_hold_days",
        "high_pm_min_hold_applied",
        "high_pm_min_hold_blocked_exit",
        "high_pm_min_hold_blocked_exit_count",
        "high_pm_min_hold_exit_reason_original",
        "high_pm_min_hold_release_date",
        "holding_days_at_exit_signal",
    ]
    snapshot = {key: position.get(key) for key in keys if key in position}
    if not snapshot.get("market_regime") and snapshot.get("entry_market_regime"):
        snapshot["market_regime"] = snapshot["entry_market_regime"]
    if "relative_strength_score" in snapshot and snapshot["relative_strength_score"] is None:
        snapshot["relative_strength_score"] = 0.0
    return snapshot


def initial_live_paper_state(config: dict[str, Any]) -> dict[str, Any]:
    initial_cash = float(config["portfolio"]["initial_cash"])
    return {
        "cash": initial_cash,
        "positions": [],
        "total_assets": initial_cash,
        "cumulative_profit": 0.0,
        "gross_cumulative_profit": 0.0,
        "net_cumulative_profit": 0.0,
        "total_commission": 0.0,
        "estimated_tax_total": 0.0,
        "closed_trades": [],
        "pending_orders": [],
        "asset_history": [initial_cash],
        "current_day": 0,
        "processed_dates": [],
    }


def _candidate_entry_price(item: dict[str, Any]) -> float:
    value = item.get("entry_price")
    if value is None:
        value = item.get("close")
    if value is None:
        return 0.0
    return float(value)


def _candidate_market_price(item: dict[str, Any], fallback: float | None = None) -> float:
    value = item.get("close")
    if value is None:
        value = fallback
    if value is None:
        return 0.0
    return float(value)


def _execution_timing_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "signal_date": item.get("signal_date") or item.get("date"),
        "entry_price_source": item.get("entry_price_source"),
        "signal_close_price": item.get("signal_close_price"),
        "entry_open_price": item.get("entry_open_price"),
        "entry_gap_rate": item.get("entry_gap_rate"),
    }


def execute_real_data_paper_trade(
    scored_candidates: list[dict[str, Any]],
    state: dict[str, Any],
    config: dict[str, Any],
    trade_date: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    cash = float(state["cash"])
    initial_cash = float(config["portfolio"]["initial_cash"])
    max_positions = int(config["portfolio"]["max_positions"])
    allocation_limit = initial_cash * float(config["portfolio"]["max_allocation_per_symbol"])
    take_profit_pct = float(config["risk"]["take_profit_pct"])
    stop_loss_pct = float(config["risk"]["stop_loss_pct"])
    max_holding_days = int(config["risk"]["max_holding_business_days"])
    broker = build_broker(state, config)
    next_day_execution = bool(config.get("execution", {}).get("use_next_day_open_execution", False))
    stop_loss_execution = str(config.get("execution", {}).get("stop_loss_execution", "next_day_open"))
    exit_ai_advisor = get_exit_ai_advisor(config)
    exit_ai_v2_gate_advisor = _exit_ai_v2_gate_advisor(config)

    trades = []
    safety_events = []
    next_positions = list(state.get("positions", []))
    closed_today = []
    price_by_code = {str(item["code"]): item for item in scored_candidates}
    selection = config.get("selection", {})
    min_score = float(selection.get("fallback_min_score", selection.get("top_pick_min_score", 70)))
    min_confidence = float(selection.get("min_confidence", config["scoring"].get("confidence_min_for_buy", 0.7)))
    selected = [
        item
        for item in scored_candidates
        if item.get("selected") and float(item["total_score"]) >= min_score and float(item["confidence"]) >= min_confidence
    ]
    selected = _sort_selected_candidates(selected, config)
    selected = _apply_score_based_pm_rule_to_candidates(selected, config)
    selected = _apply_relative_allocator_to_candidates(selected, config)
    selected_codes_for_hold = {str(item.get("code") or "") for item in selected if item.get("code")}
    pending_orders = list(state.get("pending_orders", []))
    due_pending, future_pending = _split_due_pending_orders(pending_orders, trade_date)

    for pending in due_pending:
        market = price_by_code.get(pending["code"], {})
        executed_price = _execution_price(market, pending)
        if pending["action"] == "BUY":
            scaled_shares, scaled_buy_fields = _scale_buy_to_daily_limit(
                shares=int(pending["shares"]),
                price=executed_price,
                config=config,
                today_orders=trades,
                total_assets=float(state.get("total_assets") or initial_cash),
            )
            if scaled_buy_fields:
                pending = {**pending, **scaled_buy_fields, "shares": scaled_shares}
            if scaled_shares <= 0:
                rejected = {
                    **pending,
                    "action": "SKIP_BUY",
                    "entry_date": trade_date,
                    "entry_price": executed_price,
                    "executed_price": executed_price,
                    "amount": 0,
                    "skipped_reason": "日次買付上限内で単元株を買えないため見送り",
                    "order_status": "REJECTED",
                }
                trades.append(rejected)
                continue
            amount = int(pending["shares"]) * executed_price
            buy_commission = _calculate_commission(amount, config)
            if cash < amount + buy_commission:
                rejected = {
                    **pending,
                    "action": "SKIP_BUY",
                    "entry_date": trade_date,
                    "entry_price": executed_price,
                    "executed_price": executed_price,
                    "amount": 0,
                    "skipped_reason": "翌営業日寄り付き約定時点で買付余力が不足",
                    "order_status": "REJECTED",
                }
                trades.append(rejected)
                continue
            buy_log = {
                **pending,
                "trade_id": pending.get("trade_id") or pending["order_id"],
                "action": "BUY",
                "entry_date": trade_date,
                "entry_price": executed_price,
                "executed_price": executed_price,
                "price": executed_price,
                "amount": round(amount, 2),
                "buy_commission": buy_commission,
                "status": "FILLED",
                **_slippage_fields(float(pending["intended_price"]), executed_price),
            }
            validation = can_trade(buy_log, _safety_portfolio(state, trades, buy_log), config)
            if not validation["allowed"]:
                event = safety_event(trade_date, buy_log, validation)
                safety_events.append(event)
                trades.append(_safety_rejected_order(buy_log, validation))
                continue
            cash -= amount + buy_commission
            filled = broker.place_buy_order(buy_log)
            trades.append(filled)
            next_positions.append(
                {
                    "code": pending["code"],
                    "name": pending["name"],
                    "sector_name": pending.get("sector_name", ""),
                    "signal_date": pending.get("signal_date"),
                    "entry_date": trade_date,
                    "entry_price": executed_price,
                    "entry_price_source": pending.get("entry_price_source"),
                    "signal_close_price": pending.get("signal_close_price"),
                    "entry_open_price": pending.get("entry_open_price"),
                    "entry_gap_rate": pending.get("entry_gap_rate"),
                    "current_price": executed_price,
                    "shares": int(pending["shares"]),
                    **scaled_buy_fields,
                    "market_value": round(amount, 2),
                    "buy_commission": buy_commission,
                    "holding_days": 1,
                    "max_unrealized_return_so_far": 0.0,
                    "min_unrealized_return_so_far": 0.0,
                    "score": pending.get("score"),
                    "entry_score": pending.get("entry_score") or pending.get("score"),
                    "reason": pending.get("reason", ""),
                    **_position_feature_snapshot(pending),
                    "unrealized_profit": 0.0,
                    "unrealized_profit_rate": 0.0,
                }
            )
        elif pending["action"] == "SELL":
            position = _pop_position(next_positions, pending["code"])
            if not position:
                trades.append({**pending, "order_status": "REJECTED", "rejected_reason": "売却対象の保有銘柄がありません"})
                continue
            proceeds = int(position["shares"]) * executed_price
            entry_notional = int(position["shares"]) * float(position["entry_price"])
            closed = _apply_exit_costs(
                {
                    **pending,
                    "trade_id": pending.get("trade_id") or pending["order_id"],
                    "action": "SELL",
                    "signal_date": position.get("signal_date"),
                    "entry_date": position["entry_date"],
                    "exit_date": trade_date,
                    "holding_days": int(position.get("holding_days", 0)) + 1,
                    "entry_price": position["entry_price"],
                    "entry_price_source": position.get("entry_price_source"),
                    "signal_close_price": position.get("signal_close_price"),
                    "entry_open_price": position.get("entry_open_price"),
                    "entry_gap_rate": position.get("entry_gap_rate"),
                    "exit_price": executed_price,
                    "executed_price": executed_price,
                    "actual_exit_price": executed_price,
                    "shares": position["shares"],
                    "buy_reason": position.get("reason", ""),
                    **_position_feature_snapshot(position),
                    **_slippage_fields(float(pending["intended_price"]), executed_price),
                    **_pending_stop_loss_execution_fields(pending, executed_price),
                },
                entry_notional,
                proceeds,
                float(position.get("buy_commission", 0)),
                config,
            )
            closed["profit"] = closed["net_profit"]
            closed["profit_rate"] = closed["net_profit_rate"]
            closed["result"] = _trade_result(closed["gross_profit"])
            closed["dealer_comment"] = generate_sell_comment(closed, config)
            validation = can_trade(closed, _safety_portfolio(state, trades, closed), config)
            if not validation["allowed"]:
                event = safety_event(trade_date, closed, validation)
                safety_events.append(event)
                trades.append(_safety_rejected_order(closed, validation))
                next_positions.append(position)
                continue
            cash += proceeds
            filled = broker.place_sell_order(closed)
            trades.append(filled)
            closed_today.append(filled)

    next_positions_after_pending = []
    pending_sell_codes = {order["code"] for order in future_pending if order.get("action") == "SELL"}

    for position in next_positions:
        market = price_by_code.get(str(position["code"]))
        current_price = float(market["close"]) if market else float(position["current_price"])
        holding_days = int(position["holding_days"]) if position.get("entry_date") == trade_date else int(position["holding_days"]) + 1
        holding_signal = _holding_signal_revaluation(
            position,
            market,
            selected_codes_for_hold,
            config,
            holding_days,
            max_holding_days,
        )
        exit_plan = _real_exit_plan(
            position=position,
            market=market or {},
            trade_date=trade_date,
            current_price=current_price,
            holding_days=holding_days,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_days=int(holding_signal.get("holding_effective_max_days") or max_holding_days),
            stop_loss_execution=stop_loss_execution,
        )
        exit_plan = _apply_holding_revaluation_exit_plan(
            exit_plan,
            holding_signal,
            config,
            current_price,
            holding_days,
        )
        exit_plan = apply_exit_ai_to_plan(
            exit_plan,
            exit_ai_advisor if market else None,
            position=position,
            market=market or {},
            trade_date=trade_date,
            current_price=current_price,
            holding_days=holding_days,
        )
        exit_plan, exit_ai_v2_gate_fields = _apply_exit_ai_v2_gate_to_plan(
            exit_plan,
            exit_ai_v2_gate_advisor if market else None,
            position=position,
            trade_date=trade_date,
            current_price=current_price,
        )
        exit_plan, high_pm_min_hold_fields = _apply_high_pm_min_hold_exit_guard(
            exit_plan,
            position,
            config,
            trade_date,
            holding_days,
        )
        profit_rate = exit_plan["mark_profit_rate"]
        exit_reason = exit_plan["exit_reason"]
        planned_exit_price = float(exit_plan["exit_price"] or current_price)
        max_unrealized_return_so_far, min_unrealized_return_so_far = update_unrealized_extrema(position, profit_rate)
        updated_position = {
            **position,
            "current_price": current_price,
            "market_value": round(int(position["shares"]) * current_price, 2),
            "holding_days": holding_days,
            "unrealized_profit": round(int(position["shares"]) * (current_price - float(position["entry_price"])), 2),
            "unrealized_profit_rate": round(profit_rate, 4),
            "max_unrealized_return_so_far": max_unrealized_return_so_far,
            "min_unrealized_return_so_far": min_unrealized_return_so_far,
            "drawdown_from_peak": profit_rate - max_unrealized_return_so_far if max_unrealized_return_so_far is not None else None,
            **holding_signal,
            **exit_ai_v2_gate_fields,
            **high_pm_min_hold_fields,
        }
        if exit_reason and position["code"] not in pending_sell_codes:
            execute_now = bool(exit_plan.get("execute_now", False))
            proceeds = int(position["shares"]) * planned_exit_price
            entry_notional = int(position["shares"]) * float(position["entry_price"])
            closed = _apply_exit_costs(
                {
                    "trade_id": f"{position['entry_date']}_{trade_date}_{position['code']}",
                    "action": "SELL",
                    "code": position["code"],
                    "name": position["name"],
                    "sector_name": position.get("sector_name", ""),
                    "signal_date": position.get("signal_date"),
                    "entry_date": position["entry_date"],
                    "exit_date": trade_date,
                    "holding_days": holding_days,
                    "entry_price": position["entry_price"],
                    "entry_price_source": position.get("entry_price_source"),
                    "signal_close_price": position.get("signal_close_price"),
                    "entry_open_price": position.get("entry_open_price"),
                    "entry_gap_rate": position.get("entry_gap_rate"),
                    "exit_price": planned_exit_price,
                    "actual_exit_price": planned_exit_price,
                    "shares": position["shares"],
                    "exit_reason": exit_reason,
                    "buy_reason": position.get("reason", ""),
                    **_position_feature_snapshot(position),
                    **holding_signal,
                    **exit_ai_v2_gate_fields,
                    **high_pm_min_hold_fields,
                    **_stop_loss_trade_fields(exit_plan, planned_exit_price),
                    **exit_ai_trade_fields(exit_plan),
                    **_exit_ai_v2_gate_trade_fields(exit_ai_v2_gate_fields),
                },
                entry_notional,
                proceeds,
                float(position.get("buy_commission", 0)),
                config,
            )
            closed["profit"] = closed["net_profit"]
            closed["profit_rate"] = closed["net_profit_rate"]
            closed["result"] = _trade_result(closed["gross_profit"])
            closed["dealer_comment"] = generate_sell_comment(closed, config)
            validation = can_trade(closed, _safety_portfolio(state, trades, closed), config)
            if not validation["allowed"]:
                event = safety_event(trade_date, closed, validation)
                safety_events.append(event)
                trades.append(_safety_rejected_order(closed, validation))
                next_positions_after_pending.append(updated_position)
                continue
            if next_day_execution and not execute_now:
                pending_sell = _pending_order_from_trade(closed, trade_date, action="SELL")
                future_pending.append(pending_sell)
                trades.append(pending_sell)
                next_positions_after_pending.append(updated_position)
            else:
                filled = broker.place_sell_order(closed)
                cash += proceeds
                trades.append(filled)
                closed_today.append(filled)
        else:
            next_positions_after_pending.append(updated_position)

    next_positions = next_positions_after_pending

    held_codes = {position["code"] for position in next_positions}
    pending_buy_codes = {order["code"] for order in future_pending if order.get("action") == "BUY"}
    same_day_regular_selected_codes = {str(item.get("code") or "") for item in selected if item.get("code")}
    same_day_allocation_budgets = _same_day_allocation_budgets(
        selected,
        cash,
        initial_cash,
        state,
        config,
        next_positions,
        pending_buy_codes,
        max_positions,
    )
    buy_candidates = []
    ai_purchase_active = _ai_purchase_enabled(config)
    purchase_audit_active = _purchase_audit_enabled(config)
    fallback_buys_today = 0
    max_fallback_buys_per_day = _affordable_fallback_max_buys_per_day(config)
    selected_fallback_pending: dict[str, Any] | None = None
    for item_index, item in enumerate(selected, start=1):
        dynamic_exposure_fields = _dynamic_exposure_log_fields(config, item)
        phase3l_fallback_fields = _phase3l_apply_selected_fallback_fields(
            item,
            config,
            selected_fallback_pending if _phase3l_selected_fallback_enabled(config) else None,
        )
        dynamic_exposure_fields.update(phase3l_fallback_fields)
        item.update(dynamic_exposure_fields)
        candidate_rank = _candidate_rank_value(item, item_index)
        current_total_assets = float(state.get("total_assets") or (cash + _current_market_exposure({"positions": next_positions})) or initial_cash)
        daily_limit_info = _daily_buy_limit_info(config, current_total_assets)
        daily_remaining_before = _daily_buy_limit_remaining(config, trades, current_total_assets)
        max_positions_remaining_before = max(0, max_positions - len(next_positions) - len(pending_buy_codes))
        cash_before_candidate = cash
        if (
            _phase3l_selected_fallback_enabled(config)
            and selected_fallback_pending
            and not _phase3l_selected_fallback_quality_allowed(item, config)
        ):
            item["skipped_by_fallback_quality_filter"] = True
            item["fallback_to_pm_score"] = item.get("pm_score")
            item["fallback_to_pm_multiplier"] = item.get("pm_multiplier")
            dynamic_exposure_fields["skipped_by_fallback_quality_filter"] = True
            trades.append(
                _skipped_buy_attempt(
                    trade_id=f"{trade_date}_{item['code']}_SKIP_BUY",
                    action="SKIP_BUY",
                    code=item["code"],
                    name=item["name"],
                    trade_date=trade_date,
                    price=_candidate_entry_price(item),
                    allocation_limit=0,
                    score=item.get("total_score"),
                    reason=item.get("selection_reason") or item.get("selected_reason") or item["reason"],
                    skipped_reason="fallback_quality_filter",
                    config=config,
                    allocation_reason="fallback_quality_filter",
                    extra_fields=dynamic_exposure_fields,
                )
            )
            if purchase_audit_active:
                trades.append(
                    _purchase_audit_event(
                        item=item,
                        trade_date=trade_date,
                        config=config,
                        cash_before=cash_before_candidate,
                        cash_after=cash,
                        daily_buy_limit_remaining_before=daily_remaining_before,
                        daily_buy_limit_remaining_after=_daily_buy_limit_remaining(config, trades, current_total_assets),
                        daily_buy_limit_type=str(daily_limit_info.get("type") or ""),
                        daily_buy_limit_ratio=daily_limit_info.get("ratio"),
                        daily_buy_limit_applied=float(daily_limit_info.get("limit") or 0),
                        max_positions_remaining_before=max_positions_remaining_before,
                        decision="SKIP",
                        skip_reason="fallback_quality_filter",
                        allocation_reason="fallback_quality_filter",
                    )
                )
            continue
        if not market_section_allowed(item, config):
            trades.append(
                _skipped_buy_attempt(
                    trade_id=f"{trade_date}_{item['code']}_SKIP_BUY",
                    action="SKIP_BUY",
                    code=item["code"],
                    name=item["name"],
                    trade_date=trade_date,
                    price=_candidate_entry_price(item),
                    allocation_limit=0,
                    score=item.get("total_score"),
                    reason=item.get("selection_reason") or item.get("selected_reason") or item["reason"],
                    skipped_reason="market_filter_excluded",
                    config=config,
                    allocation_reason="market_filter_excluded",
                    extra_fields=dynamic_exposure_fields,
                )
            )
            if purchase_audit_active:
                trades.append(
                    _purchase_audit_event(
                        item=item,
                        trade_date=trade_date,
                        config=config,
                        cash_before=cash_before_candidate,
                        cash_after=cash,
                        daily_buy_limit_remaining_before=daily_remaining_before,
                        daily_buy_limit_remaining_after=_daily_buy_limit_remaining(config, trades, current_total_assets),
                        daily_buy_limit_type=str(daily_limit_info.get("type") or ""),
                        daily_buy_limit_ratio=daily_limit_info.get("ratio"),
                        daily_buy_limit_applied=float(daily_limit_info.get("limit") or 0),
                        max_positions_remaining_before=max_positions_remaining_before,
                        decision="SKIP",
                        skip_reason="market_filter_excluded",
                        allocation_reason="market_filter_excluded",
                    )
                )
            continue
        if len(next_positions) + len(pending_buy_codes) >= max_positions:
            trades.append(
                _skipped_buy_attempt(
                    trade_id=f"{trade_date}_{item['code']}_SKIP_BUY",
                    action="SKIP_BUY",
                    code=item["code"],
                    name=item["name"],
                    trade_date=trade_date,
                    price=_candidate_entry_price(item),
                    allocation_limit=0,
                    score=item.get("total_score"),
                    reason=item.get("selection_reason") or item.get("selected_reason") or item["reason"],
                    skipped_reason="max_positions_limit",
                    config=config,
                    allocation_reason="max_positions_limit",
                    extra_fields=dynamic_exposure_fields,
                )
            )
            if purchase_audit_active:
                trades.append(
                    _purchase_audit_event(
                        item=item,
                        trade_date=trade_date,
                        config=config,
                        cash_before=cash_before_candidate,
                        cash_after=cash,
                        daily_buy_limit_remaining_before=daily_remaining_before,
                        daily_buy_limit_remaining_after=_daily_buy_limit_remaining(config, trades, current_total_assets),
                        daily_buy_limit_type=str(daily_limit_info.get("type") or ""),
                        daily_buy_limit_ratio=daily_limit_info.get("ratio"),
                        daily_buy_limit_applied=float(daily_limit_info.get("limit") or 0),
                        max_positions_remaining_before=max_positions_remaining_before,
                        decision="SKIP",
                        skip_reason="max_positions_limit",
                        allocation_reason="max_positions_limit",
                    )
                )
            continue
        if item["code"] in held_codes or item["code"] in pending_buy_codes:
            if purchase_audit_active:
                trades.append(
                    _purchase_audit_event(
                        item=item,
                        trade_date=trade_date,
                        config=config,
                        cash_before=cash_before_candidate,
                        cash_after=cash,
                        daily_buy_limit_remaining_before=daily_remaining_before,
                        daily_buy_limit_remaining_after=_daily_buy_limit_remaining(config, trades, current_total_assets),
                        daily_buy_limit_type=str(daily_limit_info.get("type") or ""),
                        daily_buy_limit_ratio=daily_limit_info.get("ratio"),
                        daily_buy_limit_applied=float(daily_limit_info.get("limit") or 0),
                        max_positions_remaining_before=max_positions_remaining_before,
                        decision="SKIP",
                        skip_reason="duplicate_holding",
                        allocation_reason="duplicate_holding",
                    )
                )
            continue
        if item.get("entry_price_available") is False:
            trades.append(
                _skipped_buy_attempt(
                    trade_id=f"{trade_date}_{item['code']}_SKIP_BUY",
                    action="SKIP_BUY",
                    code=item["code"],
                    name=item["name"],
                    trade_date=trade_date,
                    price=float(item.get("close") or 0),
                    allocation_limit=0,
                    score=item.get("total_score"),
                    reason=item.get("selection_reason") or item.get("selected_reason") or item["reason"],
                    skipped_reason="entry_dateの価格データがないため買付見送り",
                    config=config,
                    allocation_reason="next_day_entry_missing",
                    extra_fields=dynamic_exposure_fields,
                )
            )
            if purchase_audit_active:
                trades.append(
                    _purchase_audit_event(
                        item=item,
                        trade_date=trade_date,
                        config=config,
                        cash_before=cash_before_candidate,
                        cash_after=cash,
                        daily_buy_limit_remaining_before=daily_remaining_before,
                        daily_buy_limit_remaining_after=_daily_buy_limit_remaining(config, trades, current_total_assets),
                        daily_buy_limit_type=str(daily_limit_info.get("type") or ""),
                        daily_buy_limit_ratio=daily_limit_info.get("ratio"),
                        daily_buy_limit_applied=float(daily_limit_info.get("limit") or 0),
                        max_positions_remaining_before=max_positions_remaining_before,
                        decision="SKIP",
                        skip_reason="next_day_entry_missing",
                        allocation_reason="next_day_entry_missing",
                    )
                )
            continue
        if ai_purchase_active:
            allocation, allocation_reason, ai_purchase_fields = _ai_purchase_allocation(
                cash=cash,
                initial_cash=initial_cash,
                state=state,
                config=config,
                today_orders=trades,
                rank=candidate_rank,
            )
            pending_buy_amount = 0.0
        else:
            allocation_budget = same_day_allocation_budgets.get(str(item.get("code") or ""))
            pending_buy_amount = 0.0 if allocation_budget is not None else sum(float(order.get("estimated_amount") or order.get("amount") or 0) for order in buy_candidates)
            allocation, allocation_reason = _available_buy_budget(
                cash,
                initial_cash,
                state,
                config,
                pending_buy_amount=pending_buy_amount,
                allocation_budget=allocation_budget,
                market_context=item,
            )
            ai_purchase_fields = {}
        entry_price = _candidate_entry_price(item)
        current_price = _candidate_market_price(item, entry_price)
        shares, skipped_reason = _calculate_buy_shares(entry_price, allocation, config)
        planned_shares = shares
        planned_amount = shares * entry_price if shares > 0 else 0.0
        pm_sizing_fields: dict[str, Any] = {}
        bear_pm_booster_fields: dict[str, Any] = {}
        per_code_cap_fields: dict[str, Any] = {}
        if shares > 0:
            shares, pm_sizing_fields = _apply_portfolio_manager_sizing(
                item=item,
                trade_date=trade_date,
                shares=shares,
                entry_price=entry_price,
                cash=cash,
                config=config,
            )
            if shares <= 0:
                skipped_reason = str(pm_sizing_fields.get("pm_resize_reason") or "pm_sizing_scaled_below_round_lot")
        if shares > 0:
            shares, bear_pm_booster_fields = _apply_bear_pm_booster(
                item=item,
                trade_date=trade_date,
                shares=shares,
                entry_price=entry_price,
                cash=cash,
                config=config,
            )
        if shares > 0:
            shares, per_code_cap_fields = _apply_per_code_exposure_cap(
                item=item,
                shares=shares,
                entry_price=entry_price,
                positions=next_positions,
                total_assets=current_total_assets,
                config=config,
            )
            if per_code_cap_fields.get("pm_per_code_cap_reduced"):
                _mark_bear_pm_booster_limited_by(item, bear_pm_booster_fields, "per_code_exposure_cap")
            if shares <= 0:
                skipped_reason = str(per_code_cap_fields.get("pm_per_code_cap_reason") or "per_code_exposure_cap")
        scaled_buy_fields: dict[str, Any] = {}
        if shares <= 0 and allocation_reason in {"insufficient_available_cash", "target_exposure_limit"}:
            skipped_reason = allocation_reason
        elif (
            shares <= 0
            and _capital_utilization_enabled(config)
            and not ai_purchase_active
            and skipped_reason not in {"pm_low_score_skip", "pm_sizing_scaled_below_round_lot", "per_code_exposure_cap", "per_code_exposure_cap_scaled_below_round_lot"}
        ):
            skipped_reason = "selected_but_not_affordable"
        if shares <= 0:
            fallback_item = None
            fallback_diagnostics: dict[str, int] = {}
            if (
                _is_affordable_fallback_skip_reason(skipped_reason)
                and _affordable_fallback_replace_enabled(config)
                and fallback_buys_today < max_fallback_buys_per_day
            ):
                fallback_item = _find_affordable_fallback_candidate(
                    scored_candidates,
                    item,
                    config,
                    allocation_limit=allocation,
                    cash=cash,
                    pending_buy_amount=pending_buy_amount,
                    held_codes=held_codes,
                    pending_buy_codes=pending_buy_codes,
                    same_day_regular_selected_codes=same_day_regular_selected_codes,
                    diagnostics=fallback_diagnostics,
                )
            if fallback_item:
                trades.append(
                    _skipped_buy_attempt(
                        trade_id=f"{trade_date}_{item['code']}_SKIP_BUY",
                        action="SKIP_BUY",
                        code=item["code"],
                        name=item["name"],
                        trade_date=trade_date,
                        price=entry_price,
                        allocation_limit=allocation,
                        score=item.get("total_score"),
                        reason=item.get("selection_reason") or item.get("selected_reason") or item["reason"],
                        skipped_reason=skipped_reason,
                        config=config,
                        allocation_reason=allocation_reason,
                        extra_fields={
                            **dynamic_exposure_fields,
                            **pm_sizing_fields,
                            **bear_pm_booster_fields,
                            **per_code_cap_fields,
                            "affordable_fallback_attempted": True,
                            "affordable_fallback_replaced_by_code": fallback_item.get("code"),
                            "affordable_fallback_replaced_by_name": fallback_item.get("name"),
                            "affordable_fallback_reason": "selected_unaffordable_replaced",
                            **fallback_diagnostics,
                        },
                    )
                )
                fallback_item["selected"] = True
                fallback_item["affordable_fallback_buy_selected"] = True
                fallback_item["affordable_fallback_original_code"] = item.get("code")
                fallback_item["affordable_fallback_original_name"] = item.get("name")
                fallback_item["affordable_fallback_reason"] = skipped_reason
                fallback_item["affordable_fallback_round_lot_amount"] = _candidate_round_lot_amount(fallback_item, config)
                fallback_item["selection_reason"] = (
                    f"affordable_fallback_buy: {item.get('code')} が {skipped_reason} のため、"
                    f"買付可能な候補を繰り上げ"
                )
                fallback_item["selected_reason"] = fallback_item["selection_reason"]
                fallback_item["reason"] = fallback_item["selection_reason"]
                item = fallback_item
                dynamic_exposure_fields = _dynamic_exposure_log_fields(config, item)
                item.update(dynamic_exposure_fields)
                entry_price = _candidate_entry_price(item)
                current_price = _candidate_market_price(item, entry_price)
                shares, skipped_reason = _calculate_buy_shares(entry_price, allocation, config)
                pm_sizing_fields = {}
                bear_pm_booster_fields = {}
                per_code_cap_fields = {}
                if shares > 0:
                    shares, pm_sizing_fields = _apply_portfolio_manager_sizing(
                        item=item,
                        trade_date=trade_date,
                        shares=shares,
                        entry_price=entry_price,
                        cash=cash,
                        config=config,
                    )
                    if shares <= 0:
                        skipped_reason = str(pm_sizing_fields.get("pm_resize_reason") or "pm_sizing_scaled_below_round_lot")
                if shares > 0:
                    shares, bear_pm_booster_fields = _apply_bear_pm_booster(
                        item=item,
                        trade_date=trade_date,
                        shares=shares,
                        entry_price=entry_price,
                        cash=cash,
                        config=config,
                    )
                if shares > 0:
                    shares, per_code_cap_fields = _apply_per_code_exposure_cap(
                        item=item,
                        shares=shares,
                        entry_price=entry_price,
                        positions=next_positions,
                        total_assets=current_total_assets,
                        config=config,
                    )
                    if per_code_cap_fields.get("pm_per_code_cap_reduced"):
                        _mark_bear_pm_booster_limited_by(item, bear_pm_booster_fields, "per_code_exposure_cap")
                    if shares <= 0:
                        skipped_reason = str(per_code_cap_fields.get("pm_per_code_cap_reason") or "per_code_exposure_cap")
                if shares <= 0:
                    fallback_item = None
            if fallback_item:
                pass
            else:
                no_fallback_extra = {
                    **dynamic_exposure_fields,
                    **pm_sizing_fields,
                    **bear_pm_booster_fields,
                    **per_code_cap_fields,
                    "affordable_fallback_attempted": _is_affordable_fallback_skip_reason(skipped_reason)
                    and _affordable_fallback_enabled(config),
                    "affordable_fallback_no_candidate": _is_affordable_fallback_skip_reason(skipped_reason)
                    and _affordable_fallback_enabled(config),
                    **fallback_diagnostics,
                }
                trades.append(
                    _skipped_buy_attempt(
                        trade_id=f"{trade_date}_{item['code']}_SKIP_BUY",
                        action="SKIP_BUY",
                        code=item["code"],
                        name=item["name"],
                        trade_date=trade_date,
                        price=entry_price,
                        allocation_limit=allocation,
                        score=item.get("total_score"),
                        reason=item.get("selection_reason") or item.get("selected_reason") or item["reason"],
                        skipped_reason=skipped_reason,
                        config=config,
                        allocation_reason=allocation_reason,
                        extra_fields=no_fallback_extra,
                    )
                )
                if purchase_audit_active:
                    trades.append(
                        _purchase_audit_event(
                            item=item,
                            trade_date=trade_date,
                            config=config,
                            cash_before=cash_before_candidate,
                            cash_after=cash,
                            daily_buy_limit_remaining_before=daily_remaining_before,
                            daily_buy_limit_remaining_after=_daily_buy_limit_remaining(config, trades, current_total_assets),
                            daily_buy_limit_type=str(daily_limit_info.get("type") or ""),
                            daily_buy_limit_ratio=daily_limit_info.get("ratio"),
                            daily_buy_limit_applied=float(daily_limit_info.get("limit") or 0),
                            max_positions_remaining_before=max_positions_remaining_before,
                            planned_shares=planned_shares,
                            planned_amount=planned_amount,
                            decision="SKIP",
                            skip_reason=skipped_reason,
                            allocation_limit=allocation,
                            allocation_reason=allocation_reason,
                        )
                    )
                if _phase3l_selected_fallback_enabled(config) and _is_affordable_fallback_skip_reason(skipped_reason):
                    selected_fallback_pending = {"from_code": item.get("code"), "from_reason": skipped_reason}
                continue
        if shares > 0:
            shares, scaled_buy_fields = _scale_buy_to_daily_limit(
                shares=shares,
                price=entry_price,
                config=config,
                today_orders=trades,
                total_assets=current_total_assets,
            )
            if shares <= 0:
                skipped_reason = "daily_buy_limit_scaled_below_round_lot"
            if scaled_buy_fields.get("scaled_buy_triggered"):
                _mark_bear_pm_booster_limited_by(item, bear_pm_booster_fields, "daily_buy_limit")
        if shares <= 0:
            trades.append(
                _skipped_buy_attempt(
                    trade_id=f"{trade_date}_{item['code']}_SKIP_BUY",
                    action="SKIP_BUY",
                    code=item["code"],
                    name=item["name"],
                    trade_date=trade_date,
                    price=entry_price,
                    allocation_limit=allocation,
                    score=item.get("total_score"),
                    reason=item.get("selection_reason") or item.get("selected_reason") or item["reason"],
                    skipped_reason=skipped_reason,
                    config=config,
                    allocation_reason=allocation_reason,
                    extra_fields={**dynamic_exposure_fields, **pm_sizing_fields, **bear_pm_booster_fields, **per_code_cap_fields},
                )
            )
            if purchase_audit_active:
                trades.append(
                    _purchase_audit_event(
                        item=item,
                        trade_date=trade_date,
                        config=config,
                        cash_before=cash_before_candidate,
                        cash_after=cash,
                        daily_buy_limit_remaining_before=daily_remaining_before,
                        daily_buy_limit_remaining_after=_daily_buy_limit_remaining(config, trades, current_total_assets),
                        daily_buy_limit_type=str(daily_limit_info.get("type") or ""),
                        daily_buy_limit_ratio=daily_limit_info.get("ratio"),
                        daily_buy_limit_applied=float(daily_limit_info.get("limit") or 0),
                        max_positions_remaining_before=max_positions_remaining_before,
                        planned_shares=planned_shares,
                        planned_amount=planned_amount,
                        scaled_shares=int(scaled_buy_fields.get("scaled_shares") or 0),
                        scaled_amount=float(scaled_buy_fields.get("scaled_amount") or 0),
                        decision="SKIP",
                        skip_reason=skipped_reason,
                        scale_reason=str(scaled_buy_fields.get("scale_reason") or ""),
                        allocation_limit=allocation,
                        allocation_reason=allocation_reason,
                    )
                )
            if _phase3l_selected_fallback_enabled(config) and _is_affordable_fallback_skip_reason(skipped_reason):
                selected_fallback_pending = {"from_code": item.get("code"), "from_reason": skipped_reason}
            continue
        amount = shares * entry_price
        buy_commission = _calculate_commission(amount, config)
        position = {
            "code": item["code"],
            "name": item["name"],
            "sector_name": item.get("sector_name", ""),
            "signal_date": item.get("signal_date") or item.get("date"),
            "entry_date": trade_date,
            "entry_price": entry_price,
            "entry_price_source": item.get("entry_price_source"),
            "signal_close_price": item.get("signal_close_price"),
            "entry_open_price": item.get("entry_open_price"),
            "entry_gap_rate": item.get("entry_gap_rate"),
            "current_price": current_price,
            "shares": shares,
            "market_value": round(shares * current_price, 2),
            "buy_commission": buy_commission,
            "holding_days": 1,
            "score": item["total_score"],
            "entry_score": item["total_score"],
            "reason": item.get("selection_reason") or item.get("selected_reason") or item["reason"],
            **scaled_buy_fields,
            **ai_purchase_fields,
            **pm_sizing_fields,
            **bear_pm_booster_fields,
            **per_code_cap_fields,
            **_technical_snapshot(item),
            "unrealized_profit": round(shares * (current_price - entry_price), 2),
            "unrealized_profit_rate": round((current_price - entry_price) / entry_price, 4) if entry_price else 0.0,
        }
        buy_log = {
            "trade_id": f"{trade_date}_{item['code']}_BUY",
            "action": "BUY",
            "code": item["code"],
            "name": item["name"],
            "sector_name": item.get("sector_name", ""),
            **_execution_timing_fields(item),
            "entry_date": trade_date,
            "entry_price": entry_price,
            "shares": shares,
            "amount": round(amount, 2),
            "allocation_limit": round(allocation, 2),
            "allocation_reason": allocation_reason,
            "allocation_strategy": _allocation_strategy(config),
            "buy_commission": buy_commission,
            **scaled_buy_fields,
            **ai_purchase_fields,
            **pm_sizing_fields,
            **bear_pm_booster_fields,
            **per_code_cap_fields,
            "score": item["total_score"],
            "entry_score": item["total_score"],
            "rank": item.get("rank"),
            "daily_score_rank": item.get("daily_score_rank") or item.get("rank"),
            "reason": item.get("selection_reason") or item.get("selected_reason") or item["reason"],
            **_technical_snapshot(item),
            **dynamic_exposure_fields,
            "round_lot_size": _round_lot_size(config),
            "use_round_lot": _use_round_lot(config),
            "skipped_reason": "",
            "dealer_comment": generate_buy_comment(item, config),
        }
        validation = can_trade(buy_log, _safety_portfolio(state, trades, buy_log), config)
        if not validation["allowed"]:
            event = safety_event(trade_date, buy_log, validation)
            safety_events.append(event)
            trades.append(_safety_rejected_order(buy_log, validation))
            if purchase_audit_active:
                trades.append(
                    _purchase_audit_event(
                        item=item,
                        trade_date=trade_date,
                        config=config,
                        cash_before=cash_before_candidate,
                        cash_after=cash,
                        daily_buy_limit_remaining_before=daily_remaining_before,
                        daily_buy_limit_remaining_after=_daily_buy_limit_remaining(config, trades, current_total_assets),
                        daily_buy_limit_type=str(daily_limit_info.get("type") or ""),
                        daily_buy_limit_ratio=daily_limit_info.get("ratio"),
                        daily_buy_limit_applied=float(daily_limit_info.get("limit") or 0),
                        max_positions_remaining_before=max_positions_remaining_before,
                        planned_shares=planned_shares,
                        planned_amount=planned_amount,
                        scaled_shares=int(scaled_buy_fields.get("scaled_shares") or shares),
                        scaled_amount=float(scaled_buy_fields.get("scaled_amount") or amount),
                        final_shares=0,
                        final_amount=0.0,
                        decision="SKIP",
                        reject_reason=str(validation.get("reason") or validation.get("message") or "safety_rejected"),
                        scale_reason=str(scaled_buy_fields.get("scale_reason") or ""),
                        allocation_limit=allocation,
                        allocation_reason=allocation_reason,
                    )
                )
            continue
        held_codes.add(item["code"])
        pending_buy_codes.add(item["code"])
        if item.get("fallback_triggered"):
            selected_fallback_pending = None
        if item.get("affordable_fallback_buy_selected"):
            fallback_buys_today += 1
        if next_day_execution:
            pending_buy = _pending_order_from_trade(buy_log, trade_date, action="BUY")
            future_pending.append(pending_buy)
            trades.append(pending_buy)
            buy_candidates.append(pending_buy)
        else:
            next_positions.append(position)
            cash -= amount + buy_commission
            filled = broker.place_buy_order(buy_log)
            trades.append(filled)
            buy_candidates.append(filled)
        if purchase_audit_active:
            decision = "SCALED_BUY" if scaled_buy_fields else "BUY"
            trades.append(
                _purchase_audit_event(
                    item=item,
                    trade_date=trade_date,
                    config=config,
                    cash_before=cash_before_candidate,
                    cash_after=cash,
                    daily_buy_limit_remaining_before=daily_remaining_before,
                    daily_buy_limit_remaining_after=_daily_buy_limit_remaining(config, trades, current_total_assets),
                    daily_buy_limit_type=str(daily_limit_info.get("type") or ""),
                    daily_buy_limit_ratio=daily_limit_info.get("ratio"),
                    daily_buy_limit_applied=float(daily_limit_info.get("limit") or 0),
                    max_positions_remaining_before=max_positions_remaining_before,
                    planned_shares=planned_shares,
                    planned_amount=planned_amount,
                    scaled_shares=int(scaled_buy_fields.get("scaled_shares") or shares),
                    scaled_amount=float(scaled_buy_fields.get("scaled_amount") or amount),
                    final_shares=shares,
                    final_amount=amount,
                    decision=decision,
                    scale_reason=str(scaled_buy_fields.get("scale_reason") or ""),
                    allocation_limit=allocation,
                    allocation_reason=allocation_reason,
                )
            )

    while (
        _affordable_fallback_surplus_enabled(config)
        and fallback_buys_today < max_fallback_buys_per_day
        and len(next_positions) + len(pending_buy_codes) < max_positions
    ):
        pending_buy_amount = (
            sum(float(order.get("estimated_amount") or order.get("amount") or 0) for order in buy_candidates)
            if next_day_execution
            else 0.0
        )
        allocation, allocation_reason = _available_buy_budget(
            cash,
            initial_cash,
            state,
            config,
            pending_buy_amount=pending_buy_amount,
            allocation_budget=None,
            market_context={},
        )
        diagnostics: dict[str, int] = {}
        fallback_item = _find_surplus_affordable_fallback_candidate(
            scored_candidates,
            config,
            allocation_limit=allocation,
            cash=cash,
            pending_buy_amount=pending_buy_amount,
            held_codes=held_codes,
            pending_buy_codes=pending_buy_codes,
            same_day_regular_selected_codes=same_day_regular_selected_codes,
            diagnostics=diagnostics,
        )
        if not fallback_item:
            break
        fallback_item["selected"] = True
        fallback_item["affordable_fallback_buy_selected"] = True
        fallback_item["affordable_fallback_original_code"] = ""
        fallback_item["affordable_fallback_original_name"] = ""
        fallback_item["affordable_fallback_reason"] = "surplus_available_cash"
        fallback_item["affordable_fallback_round_lot_amount"] = _candidate_round_lot_amount(fallback_item, config)
        fallback_item["affordable_fallback_candidate_count"] = diagnostics.get("candidate_count", 0)
        fallback_item["selection_source"] = "affordable_fallback_buy"
        fallback_item["selection_reason"] = "affordable_fallback_buy: 余剰資金で買付可能な高順位候補を追加"
        fallback_item["selected_reason"] = fallback_item["selection_reason"]
        fallback_item["reason"] = fallback_item["selection_reason"]
        dynamic_exposure_fields = _dynamic_exposure_log_fields(config, fallback_item)
        fallback_item.update(dynamic_exposure_fields)
        entry_price = _candidate_entry_price(fallback_item)
        current_price = _candidate_market_price(fallback_item, entry_price)
        shares, skipped_reason = _calculate_buy_shares(entry_price, allocation, config)
        scaled_buy_fields = {}
        current_total_assets = float(state.get("total_assets") or (cash + _current_market_exposure({"positions": next_positions})) or initial_cash)
        daily_limit_info = _daily_buy_limit_info(config, current_total_assets)
        daily_remaining_before = _daily_buy_limit_remaining(config, trades, current_total_assets)
        cash_before_candidate = cash
        planned_shares = shares
        planned_amount = shares * entry_price if shares > 0 else 0.0
        pm_sizing_fields: dict[str, Any] = {}
        bear_pm_booster_fields: dict[str, Any] = {}
        per_code_cap_fields: dict[str, Any] = {}
        if shares > 0:
            shares, pm_sizing_fields = _apply_portfolio_manager_sizing(
                item=fallback_item,
                trade_date=trade_date,
                shares=shares,
                entry_price=entry_price,
                cash=cash,
                config=config,
            )
            if shares <= 0:
                skipped_reason = str(pm_sizing_fields.get("pm_resize_reason") or "pm_sizing_scaled_below_round_lot")
        if shares > 0:
            shares, bear_pm_booster_fields = _apply_bear_pm_booster(
                item=fallback_item,
                trade_date=trade_date,
                shares=shares,
                entry_price=entry_price,
                cash=cash,
                config=config,
            )
        if shares > 0:
            shares, per_code_cap_fields = _apply_per_code_exposure_cap(
                item=fallback_item,
                shares=shares,
                entry_price=entry_price,
                positions=next_positions,
                total_assets=current_total_assets,
                config=config,
            )
            if per_code_cap_fields.get("pm_per_code_cap_reduced"):
                _mark_bear_pm_booster_limited_by(fallback_item, bear_pm_booster_fields, "per_code_exposure_cap")
            if shares <= 0:
                skipped_reason = str(per_code_cap_fields.get("pm_per_code_cap_reason") or "per_code_exposure_cap")
        max_positions_remaining_before = max(0, max_positions - len(next_positions) - len(pending_buy_codes))
        if shares > 0:
            shares, scaled_buy_fields = _scale_buy_to_daily_limit(
                shares=shares,
                price=entry_price,
                config=config,
                today_orders=trades,
                total_assets=current_total_assets,
            )
            if shares <= 0:
                skipped_reason = "daily_buy_limit_scaled_below_round_lot"
            if scaled_buy_fields.get("scaled_buy_triggered"):
                _mark_bear_pm_booster_limited_by(fallback_item, bear_pm_booster_fields, "daily_buy_limit")
        if shares <= 0:
            break
        amount = shares * entry_price
        buy_commission = _calculate_commission(amount, config)
        position = {
            "code": fallback_item["code"],
            "name": fallback_item["name"],
            "sector_name": fallback_item.get("sector_name", ""),
            "signal_date": fallback_item.get("signal_date") or fallback_item.get("date"),
            "entry_date": trade_date,
            "entry_price": entry_price,
            "entry_price_source": fallback_item.get("entry_price_source"),
            "signal_close_price": fallback_item.get("signal_close_price"),
            "entry_open_price": fallback_item.get("entry_open_price"),
            "entry_gap_rate": fallback_item.get("entry_gap_rate"),
            "current_price": current_price,
            "shares": shares,
            "market_value": round(shares * current_price, 2),
            "buy_commission": buy_commission,
            "holding_days": 1,
            "score": fallback_item["total_score"],
            "reason": fallback_item.get("selection_reason") or fallback_item.get("selected_reason") or fallback_item["reason"],
            **scaled_buy_fields,
            **pm_sizing_fields,
            **bear_pm_booster_fields,
            **per_code_cap_fields,
            **_technical_snapshot(fallback_item),
            "unrealized_profit": round(shares * (current_price - entry_price), 2),
            "unrealized_profit_rate": round((current_price - entry_price) / entry_price, 4) if entry_price else 0.0,
        }
        buy_log = {
            "trade_id": f"{trade_date}_{fallback_item['code']}_BUY",
            "action": "BUY",
            "code": fallback_item["code"],
            "name": fallback_item["name"],
            "sector_name": fallback_item.get("sector_name", ""),
            **_execution_timing_fields(fallback_item),
            "entry_date": trade_date,
            "entry_price": entry_price,
            "shares": shares,
            "amount": round(amount, 2),
            "allocation_limit": round(allocation, 2),
            "allocation_reason": allocation_reason,
            "allocation_strategy": _allocation_strategy(config),
            "buy_commission": buy_commission,
            **scaled_buy_fields,
            **pm_sizing_fields,
            **bear_pm_booster_fields,
            **per_code_cap_fields,
            "score": fallback_item["total_score"],
            "rank": fallback_item.get("rank"),
            "daily_score_rank": fallback_item.get("daily_score_rank") or fallback_item.get("rank"),
            "reason": fallback_item.get("selection_reason") or fallback_item.get("selected_reason") or fallback_item["reason"],
            **_technical_snapshot(fallback_item),
            **dynamic_exposure_fields,
            "round_lot_size": _round_lot_size(config),
            "use_round_lot": _use_round_lot(config),
            "skipped_reason": "",
            "dealer_comment": generate_buy_comment(fallback_item, config),
        }
        validation = can_trade(buy_log, _safety_portfolio(state, trades, buy_log), config)
        if not validation["allowed"]:
            event = safety_event(trade_date, buy_log, validation)
            safety_events.append(event)
            trades.append(_safety_rejected_order(buy_log, validation))
            if purchase_audit_active:
                trades.append(
                    _purchase_audit_event(
                        item=fallback_item,
                        trade_date=trade_date,
                        config=config,
                        cash_before=cash_before_candidate,
                        cash_after=cash,
                        daily_buy_limit_remaining_before=daily_remaining_before,
                        daily_buy_limit_remaining_after=_daily_buy_limit_remaining(config, trades, current_total_assets),
                        daily_buy_limit_type=str(daily_limit_info.get("type") or ""),
                        daily_buy_limit_ratio=daily_limit_info.get("ratio"),
                        daily_buy_limit_applied=float(daily_limit_info.get("limit") or 0),
                        max_positions_remaining_before=max_positions_remaining_before,
                        planned_shares=planned_shares,
                        planned_amount=planned_amount,
                        scaled_shares=int(scaled_buy_fields.get("scaled_shares") or shares),
                        scaled_amount=float(scaled_buy_fields.get("scaled_amount") or amount),
                        final_shares=0,
                        final_amount=0.0,
                        decision="SKIP",
                        reject_reason=str(validation.get("reason") or validation.get("message") or "safety_rejected"),
                        scale_reason=str(scaled_buy_fields.get("scale_reason") or ""),
                        allocation_limit=allocation,
                        allocation_reason=allocation_reason,
                    )
                )
            break
        held_codes.add(fallback_item["code"])
        pending_buy_codes.add(fallback_item["code"])
        fallback_buys_today += 1
        same_day_regular_selected_codes.add(str(fallback_item.get("code") or ""))
        if next_day_execution:
            pending_buy = _pending_order_from_trade(buy_log, trade_date, action="BUY")
            future_pending.append(pending_buy)
            trades.append(pending_buy)
            buy_candidates.append(pending_buy)
        else:
            next_positions.append(position)
            cash -= amount + buy_commission
            filled = broker.place_buy_order(buy_log)
            trades.append(filled)
            buy_candidates.append(filled)
        if purchase_audit_active:
            decision = "SCALED_BUY" if scaled_buy_fields else "BUY"
            trades.append(
                _purchase_audit_event(
                    item=fallback_item,
                    trade_date=trade_date,
                    config=config,
                    cash_before=cash_before_candidate,
                    cash_after=cash,
                    daily_buy_limit_remaining_before=daily_remaining_before,
                    daily_buy_limit_remaining_after=_daily_buy_limit_remaining(config, trades, current_total_assets),
                    daily_buy_limit_type=str(daily_limit_info.get("type") or ""),
                    daily_buy_limit_ratio=daily_limit_info.get("ratio"),
                    daily_buy_limit_applied=float(daily_limit_info.get("limit") or 0),
                    max_positions_remaining_before=max_positions_remaining_before,
                    planned_shares=planned_shares,
                    planned_amount=planned_amount,
                    scaled_shares=int(scaled_buy_fields.get("scaled_shares") or shares),
                    scaled_amount=float(scaled_buy_fields.get("scaled_amount") or amount),
                    final_shares=shares,
                    final_amount=amount,
                    decision=decision,
                    scale_reason=str(scaled_buy_fields.get("scale_reason") or ""),
                    allocation_limit=allocation,
                    allocation_reason=allocation_reason,
                )
            )

    if not selected:
        no_trade_reason = "本日は買付対象なし"
        trades.append(
            {
                "action": "NO_BUY",
                "date": trade_date,
                "reason": no_trade_reason,
                "dealer_comment": generate_no_trade_comment(no_trade_reason, config),
            }
        )

    positions_value = round(sum(float(position["market_value"]) for position in next_positions), 2)
    total_assets = round(cash + positions_value, 2)
    previous_assets = float(state.get("total_assets", initial_cash))
    cumulative_profit = round(total_assets - initial_cash, 2)
    state["cash"] = round(cash, 2)
    state["positions"] = next_positions
    state["total_assets"] = total_assets
    state["cumulative_profit"] = cumulative_profit
    state["closed_trades"].extend(closed_today)
    state["pending_orders"] = future_pending
    period_profit = calculate_period_profit_summary(state["closed_trades"], config)
    state["gross_cumulative_profit"] = period_profit["gross_cumulative_profit"]
    state["net_cumulative_profit"] = period_profit["net_cumulative_profit"]
    state["total_commission"] = period_profit["total_commission"]
    state["estimated_tax_total"] = period_profit["estimated_tax_total"]
    state.setdefault("asset_history", [initial_cash]).append(total_assets)
    state["current_day"] = int(state.get("current_day", 0)) + 1
    processed_dates = state.setdefault("processed_dates", [])
    if trade_date not in processed_dates:
        processed_dates.append(trade_date)

    summary = _real_portfolio_summary(state, trade_date, positions_value, previous_assets, initial_cash)
    summary["safety_events"] = safety_events
    summary["pending_orders_count"] = len(future_pending)
    summary["pending_orders"] = future_pending
    summary["executed_orders"] = [trade for trade in trades if trade.get("order_status") == "FILLED"]
    return state, summary, trades


def _real_exit_reason(
    profit_rate: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    holding_days: int,
    max_holding_days: int,
) -> str:
    if holding_days < 2:
        return ""
    if profit_rate >= take_profit_pct:
        return "利確"
    if profit_rate <= stop_loss_pct:
        return "損切り"
    if holding_days >= max_holding_days:
        return "最大保有期間到達"
    return ""


def _real_exit_plan(
    position: dict[str, Any],
    market: dict[str, Any],
    trade_date: str,
    current_price: float,
    holding_days: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_days: int,
    stop_loss_execution: str,
) -> dict[str, Any]:
    entry_price = float(position["entry_price"])
    mark_profit_rate = (current_price - entry_price) / entry_price if entry_price else 0.0
    trigger_price = round(entry_price * (1 + stop_loss_pct), 4)
    low_price = _optional_float(market.get("low"))
    stop_hit_intraday = holding_days >= 2 and low_price is not None and low_price <= trigger_price
    if stop_loss_execution in {"intraday_stop", "conservative_intraday_stop"} and stop_hit_intraday:
        exit_price = trigger_price
        if stop_loss_execution == "conservative_intraday_stop":
            exit_price = min(trigger_price, current_price)
        return {
            "exit_reason": "損切り",
            "exit_price": exit_price,
            "mark_profit_rate": mark_profit_rate,
            "stop_loss_rate": stop_loss_pct,
            "stop_loss_trigger_price": trigger_price,
            "stop_loss_triggered_date": trade_date,
            "intended_exit_price": trigger_price,
            "execute_now": True,
        }

    exit_reason = _real_exit_reason(mark_profit_rate, take_profit_pct, stop_loss_pct, holding_days, max_holding_days)
    plan = {
        "exit_reason": exit_reason,
        "exit_price": current_price if exit_reason else None,
        "mark_profit_rate": mark_profit_rate,
        "stop_loss_rate": stop_loss_pct,
        "stop_loss_trigger_price": None,
        "stop_loss_triggered_date": None,
        "intended_exit_price": current_price if exit_reason else None,
        "execute_now": False,
    }
    if exit_reason == "損切り":
        plan["stop_loss_trigger_price"] = trigger_price
        plan["stop_loss_triggered_date"] = trade_date
    return plan


def _holding_revaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("holding_revaluation") or config.get("holding_signal_revaluation") or {}
    return value if isinstance(value, dict) else {}


def _conditional_hold_extension_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("conditional_hold_extension") or {}
    return value if isinstance(value, dict) else {}


def _holding_revaluation_enabled(config: dict[str, Any]) -> bool:
    return bool(_holding_revaluation_config(config).get("enabled")) or bool(_conditional_hold_extension_config(config).get("enabled"))


def _holding_signal_revaluation_enabled(config: dict[str, Any]) -> bool:
    return bool(_holding_revaluation_config(config).get("enabled"))


def _bool_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _holding_signal_revaluation(
    position: dict[str, Any],
    market: dict[str, Any] | None,
    selected_codes: set[str],
    config: dict[str, Any],
    holding_days: int,
    default_max_holding_days: int,
) -> dict[str, Any]:
    cfg = _holding_revaluation_config(config)
    conditional_cfg = _conditional_hold_extension_config(config)
    enabled = bool(cfg.get("enabled")) or bool(conditional_cfg.get("enabled"))
    code = str(position.get("code") or "")
    market = market or {}
    row_score = _optional_float(market.get("total_score"))
    entry_score = _optional_float(position.get("entry_score"))
    if entry_score is None:
        entry_score = _optional_float(position.get("score"))
    if entry_score is None:
        entry_score = _optional_float(position.get("total_score"))
    score_drop = None
    if entry_score is not None and row_score is not None:
        score_drop = round(entry_score - row_score, 4)
    entry_price = _optional_float(position.get("entry_price"))
    current_price = _optional_float(market.get("close")) or _optional_float(position.get("current_price"))
    unrealized_profit_rate = None
    if entry_price and current_price is not None:
        unrealized_profit_rate = round((current_price - entry_price) / entry_price, 4)
    previous_signal_lost_streak = int(position.get("holding_signal_lost_streak") or 0)
    previous_extension_count = int(position.get("holding_extension_count") or 0)
    previous_avoided_count = int(position.get("holding_signal_lost_exit_avoided_count") or 0)
    previous_extended = _bool_config(position.get("holding_extended"), False)
    if not enabled:
        return {
            "holding_signal_status": "",
            "holding_entry_score": entry_score,
            "holding_current_score": row_score,
            "holding_score_drop": score_drop,
            "holding_effective_max_days": default_max_holding_days,
            "holding_reselected": False,
            "holding_extended": False,
            "holding_signal_lost_streak": 0,
            "holding_signal_lost_exit_avoided": False,
            "holding_signal_lost_exit_avoided_count": previous_avoided_count,
            "holding_extension_count": previous_extension_count,
            "holding_extension_eligible": False,
            "holding_extension_reason": "",
            "holding_unrealized_profit_rate": unrealized_profit_rate,
            "conditional_hold_extension_applied": False,
            "conditional_hold_extension_count": 0,
            "conditional_hold_extension_reason": "",
            "conditional_hold_extension_trigger_profit_rate": None,
            "extension_profit_rate": None,
            "extension_exit_guard_triggered": False,
            "extension_exit_guard_reason": "",
            "conditional_hold_extension_rejected": False,
            "conditional_hold_extension_rejected_reason": "",
        }

    if code in selected_codes:
        status = "reselected"
    elif market:
        status = "still_candidate"
    else:
        status = "signal_lost"

    signal_lost_streak = previous_signal_lost_streak + 1 if status == "signal_lost" else 0
    if _holding_signal_revaluation_enabled(config):
        threshold = _optional_float(cfg.get("score_drop_exit_threshold"))
        score_below = _optional_float(cfg.get("score_drop_exit_score_below"))
        score_lost_days = int(cfg.get("score_drop_exit_requires_signal_lost_days") or 0)
        score_only_when_not_profitable = _bool_config(cfg.get("score_drop_exit_only_when_not_profitable"), False)
        score_exit_allowed = True
        if score_below is not None and (row_score is None or row_score >= score_below):
            score_exit_allowed = False
        if score_lost_days and signal_lost_streak < score_lost_days:
            score_exit_allowed = False
        if score_only_when_not_profitable and unrealized_profit_rate is not None and unrealized_profit_rate > 0:
            score_exit_allowed = False
        if market and threshold is not None and score_drop is not None and score_drop >= threshold and score_exit_allowed:
            status = "score_deteriorated"

    effective_max_days = default_max_holding_days
    reselected = status == "reselected"
    extended = False
    extension_eligible = False
    extension_reason = ""
    extension_count = previous_extension_count
    if reselected and bool(cfg.get("hold_reselection_enabled", False)):
        extension_eligible, extension_reason = _holding_extension_eligible(position, market, cfg, score_drop, unrealized_profit_rate)
        max_extension_count = int(cfg.get("hold_extension_max_count") or 0)
        may_start_extension = max_extension_count <= 0 or previous_extension_count < max_extension_count
        if previous_extended or (extension_eligible and may_start_extension):
            effective_max_days = max(default_max_holding_days, int(cfg.get("hold_extension_max_days") or default_max_holding_days))
        if effective_max_days > default_max_holding_days and holding_days >= default_max_holding_days and not previous_extended:
            extension_count = previous_extension_count + 1
        extended = effective_max_days > default_max_holding_days and holding_days >= default_max_holding_days
    conditional_applied = _bool_config(position.get("conditional_hold_extension_applied"), False)
    conditional_count = int(position.get("conditional_hold_extension_count") or 0)
    conditional_reason = str(position.get("conditional_hold_extension_reason") or "")
    conditional_trigger_profit_rate = _optional_float(position.get("conditional_hold_extension_trigger_profit_rate"))
    extension_profit_rate = _optional_float(position.get("extension_profit_rate"))
    if extension_profit_rate is None:
        extension_profit_rate = conditional_trigger_profit_rate
    conditional_rejected = False
    conditional_rejected_reason = ""
    conditional_profit_rate_at_max_holding = None
    conditional_close_at_max_holding = None
    conditional_ma25_at_max_holding = None
    conditional_previous_ma25_at_max_holding = None
    conditional_relative_strength_score_at_max_holding = None
    if _bool_config(conditional_cfg.get("enabled"), False):
        min_profit_rate = _optional_float(
            conditional_cfg.get("min_unrealized_profit_rate")
            if conditional_cfg.get("min_unrealized_profit_rate") is not None
            else conditional_cfg.get("minimum_profit_for_extension")
        )
        if min_profit_rate is None:
            min_profit_rate = 0.03
        extension_max_days = int(conditional_cfg.get("max_holding_days") or conditional_cfg.get("extend_to_max_holding_days") or default_max_holding_days)
        max_extension_count = int(conditional_cfg.get("max_extension_count") or 1)
        may_start_extension = max_extension_count <= 0 or conditional_count < max_extension_count
        if holding_days >= default_max_holding_days:
            conditional_profit_rate_at_max_holding = unrealized_profit_rate
            conditional_close_at_max_holding = _optional_float(market.get("close")) or current_price
            conditional_ma25_at_max_holding = _optional_float(market.get("ma25"))
            conditional_previous_ma25_at_max_holding = _optional_float(market.get("previous_ma25"))
            conditional_relative_strength_score_at_max_holding = _optional_float(market.get("relative_strength_score"))
        allowed, decision_reason, rejected_reasons = _conditional_hold_extension_decision(
            market,
            entry_price,
            current_price,
            unrealized_profit_rate,
            conditional_cfg,
            min_profit_rate,
            conditional_count,
            may_start_extension,
        )
        if conditional_applied or (holding_days >= default_max_holding_days and allowed):
            effective_max_days = max(default_max_holding_days, extension_max_days)
            extended = effective_max_days > default_max_holding_days and holding_days >= default_max_holding_days
            if not conditional_applied and holding_days >= default_max_holding_days:
                conditional_applied = True
                conditional_count += 1
                conditional_reason = decision_reason or f"unrealized_profit_rate>={min_profit_rate:.4f}"
                conditional_trigger_profit_rate = unrealized_profit_rate
                extension_profit_rate = unrealized_profit_rate
            if conditional_applied:
                extension_eligible = True
                extension_reason = conditional_reason or "conditional_profit_extension"
        elif holding_days >= default_max_holding_days:
            conditional_rejected = True
            conditional_rejected_reason = "+".join(rejected_reasons) if rejected_reasons else "unknown"

    return {
        "holding_signal_status": status,
        "holding_entry_score": entry_score,
        "holding_current_score": row_score,
        "holding_score_drop": score_drop,
        "holding_effective_max_days": effective_max_days,
        "holding_reselected": reselected,
        "holding_extended": extended,
        "holding_signal_lost_streak": signal_lost_streak,
        "holding_signal_lost_exit_avoided": False,
        "holding_signal_lost_exit_avoided_count": previous_avoided_count,
        "holding_extension_count": extension_count,
        "holding_extension_eligible": extension_eligible,
        "holding_extension_reason": extension_reason,
        "holding_unrealized_profit_rate": unrealized_profit_rate,
        "conditional_hold_extension_applied": conditional_applied,
        "conditional_hold_extension_count": conditional_count,
        "conditional_hold_extension_reason": conditional_reason,
        "conditional_hold_extension_trigger_profit_rate": conditional_trigger_profit_rate,
        "extension_profit_rate": extension_profit_rate,
        "extension_exit_guard_triggered": False,
        "extension_exit_guard_reason": "",
        "conditional_hold_extension_rejected": conditional_rejected,
        "conditional_hold_extension_rejected_reason": conditional_rejected_reason,
        "conditional_hold_extension_profit_rate_at_max_holding": conditional_profit_rate_at_max_holding,
        "conditional_hold_extension_close_at_max_holding": conditional_close_at_max_holding,
        "conditional_hold_extension_ma25_at_max_holding": conditional_ma25_at_max_holding,
        "conditional_hold_extension_previous_ma25_at_max_holding": conditional_previous_ma25_at_max_holding,
        "conditional_hold_extension_relative_strength_score_at_max_holding": conditional_relative_strength_score_at_max_holding,
    }


def _conditional_hold_extension_decision(
    market: dict[str, Any],
    entry_price: float | None,
    current_price: float | None,
    unrealized_profit_rate: float | None,
    cfg: dict[str, Any],
    min_profit_rate: float,
    conditional_count: int,
    may_start_extension: bool,
) -> tuple[bool, str, list[str]]:
    reasons: list[str] = []
    if not may_start_extension or conditional_count > 0:
        reasons.append("already_extended")
    if unrealized_profit_rate is None:
        reasons.append("missing_indicator")
    elif unrealized_profit_rate < min_profit_rate:
        reasons.append(_conditional_hold_extension_profit_reject_reason(cfg))
    if entry_price is None or current_price is None:
        if "missing_indicator" not in reasons:
            reasons.append("missing_indicator")
    elif current_price <= entry_price:
        reasons.append(_conditional_hold_extension_profit_reject_reason(cfg))
    if _bool_config(cfg.get("require_trend_continuation"), False):
        close = _optional_float(market.get("close"))
        ma5 = _optional_float(market.get("ma5"))
        ma25 = _optional_float(market.get("ma25"))
        previous_ma25 = _optional_float(market.get("previous_ma25"))
        relative_strength_score = _optional_float(market.get("relative_strength_score"))
        if close is None:
            reasons.append("missing_indicator")
        if not _bool_config(cfg.get("skip_ma5_condition"), False):
            if ma5 is None:
                reasons.append("missing_indicator")
            elif close is not None and close < ma5:
                reasons.append("below_ma5")
        if ma25 is None:
            reasons.append("missing_indicator")
        elif close is not None and close < ma25:
            reasons.append("below_ma25")
        if _bool_config(cfg.get("require_ma25_uptrend"), False):
            if ma25 is None or previous_ma25 is None:
                reasons.append("missing_indicator")
            elif ma25 <= previous_ma25:
                reasons.append("ma25_not_uptrend")
        min_relative_strength = _optional_float(cfg.get("min_relative_strength_score"))
        if min_relative_strength is None:
            min_relative_strength = _optional_float(cfg.get("minimum_relative_strength_score"))
        if min_relative_strength is None:
            min_relative_strength = 60.0
        if relative_strength_score is None:
            reasons.append("missing_indicator")
        elif relative_strength_score < min_relative_strength:
            reasons.append(_conditional_hold_extension_relative_strength_reject_reason(cfg))
    if reasons:
        deduped = list(dict.fromkeys(reasons))
        return False, "", deduped
    return True, _conditional_hold_extension_reason(cfg, min_profit_rate), []


def _conditional_hold_extension_profit_reject_reason(cfg: dict[str, Any]) -> str:
    if cfg.get("profit_reject_reason"):
        return str(cfg.get("profit_reject_reason"))
    return "low_profit_rate"


def _conditional_hold_extension_relative_strength_reject_reason(cfg: dict[str, Any]) -> str:
    if cfg.get("relative_strength_reject_reason"):
        return str(cfg.get("relative_strength_reject_reason"))
    return "low_relative_strength"


def _conditional_hold_extension_reason(cfg: dict[str, Any], min_profit_rate: float) -> str:
    if _bool_config(cfg.get("require_trend_continuation"), False):
        min_relative_strength = _optional_float(cfg.get("min_relative_strength_score"))
        if min_relative_strength is None:
            min_relative_strength = _optional_float(cfg.get("minimum_relative_strength_score"))
        if min_relative_strength is None:
            min_relative_strength = 60.0
        if _bool_config(cfg.get("require_ma25_uptrend"), False):
            return f"trend_continuation_profit>={min_profit_rate:.4f}_rs>={min_relative_strength:.1f}_ma25_uptrend"
        return f"trend_continuation_profit>={min_profit_rate:.4f}_rs>={min_relative_strength:.1f}"
    return f"unrealized_profit_rate>={min_profit_rate:.4f}"


def _holding_extension_eligible(
    position: dict[str, Any],
    market: dict[str, Any],
    cfg: dict[str, Any],
    score_drop: float | None,
    unrealized_profit_rate: float | None,
) -> tuple[bool, str]:
    if not _bool_config(cfg.get("hold_extension_require_confirmation"), False):
        return True, "reselected"
    reasons = []
    min_profit_rate = _optional_float(cfg.get("hold_extension_min_unrealized_profit_rate"))
    if min_profit_rate is not None and unrealized_profit_rate is not None and unrealized_profit_rate >= min_profit_rate:
        reasons.append("unrealized_profit")
    max_score_drop = _optional_float(cfg.get("hold_extension_max_score_drop"))
    if max_score_drop is not None and score_drop is not None and score_drop <= max_score_drop:
        reasons.append("score_maintained")
    max_rank = _optional_int(cfg.get("hold_extension_max_rank"))
    rank_value = market.get("daily_score_rank") if market.get("daily_score_rank") is not None else market.get("rank")
    rank = _optional_int(rank_value)
    if max_rank is not None and rank is not None and rank <= max_rank:
        reasons.append("top_rank")
    return bool(reasons), "+".join(reasons)


def _apply_holding_revaluation_exit_plan(
    exit_plan: dict[str, Any],
    holding_signal: dict[str, Any],
    config: dict[str, Any],
    current_price: float,
    holding_days: int,
) -> dict[str, Any]:
    if not _holding_revaluation_enabled(config):
        return exit_plan
    cfg = _holding_revaluation_config(config)
    if exit_plan.get("exit_reason") in {"損切り", "利確"}:
        return exit_plan
    guard_plan = _extension_exit_guard_plan(exit_plan, holding_signal, config, current_price)
    if guard_plan is not None:
        return guard_plan
    if not _holding_signal_revaluation_enabled(config):
        return exit_plan
    if bool(cfg.get("stop_loss_always_priority", True)) and exit_plan.get("exit_reason") == "損切り":
        return exit_plan
    if holding_days < 2:
        return exit_plan
    status = str(holding_signal.get("holding_signal_status") or "")
    reason = ""
    unrealized_profit_rate = _optional_float(holding_signal.get("holding_unrealized_profit_rate"))
    if status == "signal_lost" and _bool_config(cfg.get("early_exit_on_signal_lost"), False):
        min_lost_days = int(cfg.get("signal_lost_exit_min_consecutive_days") or 1)
        lost_streak = int(holding_signal.get("holding_signal_lost_streak") or 0)
        loss_threshold = _optional_float(cfg.get("signal_lost_exit_unrealized_loss_rate_threshold"))
        loss_override = loss_threshold is not None and unrealized_profit_rate is not None and unrealized_profit_rate <= loss_threshold
        suppress_profit = _bool_config(cfg.get("suppress_signal_lost_exit_when_unrealized_profit"), False)
        if suppress_profit and unrealized_profit_rate is not None and unrealized_profit_rate > 0:
            holding_signal["holding_signal_lost_exit_avoided"] = True
            holding_signal["holding_signal_lost_exit_avoided_count"] = int(holding_signal.get("holding_signal_lost_exit_avoided_count") or 0) + 1
        elif lost_streak >= min_lost_days or loss_override:
            reason = "シグナル消失"
        else:
            holding_signal["holding_signal_lost_exit_avoided"] = True
            holding_signal["holding_signal_lost_exit_avoided_count"] = int(holding_signal.get("holding_signal_lost_exit_avoided_count") or 0) + 1
    elif status == "score_deteriorated" and cfg.get("score_drop_exit_threshold") is not None:
        if _bool_config(cfg.get("suppress_score_drop_exit_when_unrealized_profit"), False) and unrealized_profit_rate is not None and unrealized_profit_rate > 0:
            holding_signal["holding_signal_lost_exit_avoided"] = True
            holding_signal["holding_signal_lost_exit_avoided_count"] = int(holding_signal.get("holding_signal_lost_exit_avoided_count") or 0) + 1
            return exit_plan
        reason = "スコア低下"
    if not reason:
        return exit_plan
    updated = dict(exit_plan)
    updated.update(
        {
            "exit_reason": reason,
            "exit_price": current_price,
            "intended_exit_price": current_price,
            "execute_now": False,
        }
    )
    return updated


def _extension_exit_guard_plan(
    exit_plan: dict[str, Any],
    holding_signal: dict[str, Any],
    config: dict[str, Any],
    current_price: float,
) -> dict[str, Any] | None:
    conditional_cfg = _conditional_hold_extension_config(config)
    guard_cfg = conditional_cfg.get("extension_exit_guard") or {}
    if not isinstance(guard_cfg, dict) or not _bool_config(guard_cfg.get("enabled"), False):
        return None
    if not _truthy_holding_value(holding_signal.get("conditional_hold_extension_applied")):
        return None
    extension_profit_rate = _optional_float(holding_signal.get("extension_profit_rate"))
    if extension_profit_rate is None:
        extension_profit_rate = _optional_float(holding_signal.get("conditional_hold_extension_trigger_profit_rate"))
    current_profit_rate = _optional_float(holding_signal.get("holding_unrealized_profit_rate"))
    if extension_profit_rate is None or current_profit_rate is None:
        return None
    reasons: list[str] = []
    max_pullback = _optional_float(guard_cfg.get("max_profit_pullback_points"))
    if max_pullback is not None and extension_profit_rate - current_profit_rate >= max_pullback:
        reasons.append("profit_pullback_exceeded")
    min_remaining_profit_rate = _optional_float(guard_cfg.get("min_remaining_profit_rate"))
    if min_remaining_profit_rate is not None and current_profit_rate < min_remaining_profit_rate:
        reasons.append("remaining_profit_below_min")
    if not reasons:
        return None
    reason_text = "+".join(reasons)
    holding_signal["extension_exit_guard_triggered"] = True
    holding_signal["extension_exit_guard_reason"] = reason_text
    holding_signal["extension_profit_rate"] = extension_profit_rate
    updated = dict(exit_plan)
    updated.update(
        {
            "exit_reason": str(guard_cfg.get("exit_reason") or "延長後失速撤退"),
            "exit_price": current_price,
            "intended_exit_price": current_price,
            "execute_now": False,
        }
    )
    return updated


def _truthy_holding_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _stop_loss_trade_fields(exit_plan: dict[str, Any], actual_exit_price: float) -> dict[str, Any]:
    trigger_price = exit_plan.get("stop_loss_trigger_price")
    stop_loss_slippage_rate = None
    if trigger_price:
        stop_loss_slippage_rate = round((actual_exit_price - float(trigger_price)) / float(trigger_price), 4)
    intended_exit_price = exit_plan.get("intended_exit_price")
    gap_slippage_rate = None
    if intended_exit_price:
        gap_slippage_rate = round((actual_exit_price - float(intended_exit_price)) / float(intended_exit_price), 4)
    return {
        "stop_loss_rate": exit_plan.get("stop_loss_rate"),
        "stop_loss_trigger_price": trigger_price,
        "stop_loss_triggered_date": exit_plan.get("stop_loss_triggered_date"),
        "intended_exit_price": intended_exit_price,
        "actual_exit_price": actual_exit_price,
        "gap_slippage_rate": gap_slippage_rate,
        "stop_loss_slippage_rate": stop_loss_slippage_rate,
    }


def _demo_stop_loss_fields(
    exit_reason: str,
    entry_price: float,
    actual_exit_price: float,
    stop_loss_pct: float,
    trade_date: str,
) -> dict[str, Any]:
    if exit_reason != "損切り":
        return {
            "stop_loss_rate": None,
            "stop_loss_trigger_price": None,
            "stop_loss_triggered_date": None,
            "intended_exit_price": actual_exit_price,
            "actual_exit_price": actual_exit_price,
            "gap_slippage_rate": None,
            "stop_loss_slippage_rate": None,
        }
    trigger_price = round(entry_price * (1 + stop_loss_pct), 4)
    return _stop_loss_trade_fields(
        {
            "stop_loss_rate": stop_loss_pct,
            "stop_loss_trigger_price": trigger_price,
            "stop_loss_triggered_date": trade_date,
            "intended_exit_price": actual_exit_price,
        },
        actual_exit_price,
    )


def _pending_stop_loss_execution_fields(pending: dict[str, Any], actual_exit_price: float) -> dict[str, Any]:
    trigger_price = pending.get("stop_loss_trigger_price")
    intended_exit_price = pending.get("intended_exit_price") or pending.get("intended_price")
    stop_loss_slippage_rate = None
    if trigger_price:
        stop_loss_slippage_rate = round((actual_exit_price - float(trigger_price)) / float(trigger_price), 4)
    gap_slippage_rate = None
    if intended_exit_price:
        gap_slippage_rate = round((actual_exit_price - float(intended_exit_price)) / float(intended_exit_price), 4)
    return {
        "stop_loss_rate": pending.get("stop_loss_rate"),
        "stop_loss_trigger_price": trigger_price,
        "stop_loss_triggered_date": pending.get("stop_loss_triggered_date"),
        "intended_exit_price": intended_exit_price,
        "actual_exit_price": actual_exit_price,
        "gap_slippage_rate": gap_slippage_rate,
        "stop_loss_slippage_rate": stop_loss_slippage_rate,
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _split_due_pending_orders(pending_orders: list[dict[str, Any]], trade_date: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    due = []
    future = []
    for order in pending_orders:
        if order.get("status") != "PENDING":
            continue
        if str(order.get("scheduled_execution_date", "")) <= trade_date:
            due.append(order)
        else:
            future.append(order)
    return due, future


def _pending_order_from_trade(trade: dict[str, Any], created_date: str, action: str) -> dict[str, Any]:
    intended_price = float(trade.get("entry_price") or trade.get("exit_price") or trade.get("price") or 0)
    order_id = trade.get("order_id") or trade.get("trade_id") or f"{created_date}_{trade.get('code')}_{action}"
    return {
        **trade,
        "order_id": str(order_id).replace("_BUY", "_PENDING_BUY").replace("_SELL", "_PENDING_SELL"),
        "action": action,
        "created_date": created_date,
        "scheduled_execution_date": _next_business_date(created_date),
        "intended_price": intended_price,
        "status": "PENDING",
        "order_status": "PENDING",
        "executed_price": None,
        "slippage_amount": None,
        "slippage_rate": None,
    }


def _next_business_date(date_text: str) -> str:
    from datetime import date, timedelta

    current = date.fromisoformat(date_text) + timedelta(days=1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current.isoformat()


def _business_date_after_entry(entry_date_text: str, holding_days: int) -> str:
    from datetime import date, timedelta

    if not entry_date_text:
        return ""
    current = date.fromisoformat(entry_date_text)
    for _ in range(max(0, holding_days - 1)):
        current += timedelta(days=1)
        while current.weekday() >= 5:
            current += timedelta(days=1)
    return current.isoformat()


def _execution_price(market: dict[str, Any], pending_order: dict[str, Any]) -> float:
    price = market.get("open") or market.get("close") or pending_order.get("intended_price")
    return float(price)


def _slippage_fields(intended_price: float, executed_price: float) -> dict[str, float]:
    slippage_amount = executed_price - intended_price
    return {
        "intended_price": intended_price,
        "executed_price": executed_price,
        "slippage_amount": round(slippage_amount, 2),
        "slippage_rate": round(slippage_amount / intended_price, 4) if intended_price else 0.0,
    }


def _pop_position(positions: list[dict[str, Any]], code: str) -> dict[str, Any] | None:
    for index, position in enumerate(positions):
        if position.get("code") == code:
            return positions.pop(index)
    return None


def _real_portfolio_summary(
    state: dict[str, Any],
    trade_date: str,
    positions_value: float,
    previous_assets: float,
    initial_cash: float,
) -> dict[str, Any]:
    closed_count = len(state["closed_trades"])
    wins = sum(1 for trade in state["closed_trades"] if trade["result"] == "WIN")
    win_rate = round(wins / closed_count, 4) if closed_count else None
    max_drawdown = _max_drawdown(state.get("asset_history", [initial_cash]))
    return {
        "date": trade_date,
        "day": state["current_day"],
        "cash": state["cash"],
        "positions_value": positions_value,
        "total_assets": state["total_assets"],
        "daily_profit": round(state["total_assets"] - previous_assets, 2),
        "cumulative_profit": state["cumulative_profit"],
        "cumulative_profit_rate": round(state["cumulative_profit"] / initial_cash, 4),
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "open_positions_count": len(state["positions"]),
        "closed_trades_count": closed_count,
        "gross_cumulative_profit": state.get("gross_cumulative_profit", 0.0),
        "net_cumulative_profit": state.get("net_cumulative_profit", 0.0),
        "total_commission": state.get("total_commission", 0.0),
        "estimated_tax_total": state.get("estimated_tax_total", 0.0),
        "net_total_assets": round(initial_cash + state.get("net_cumulative_profit", 0.0), 2),
    }


def _max_drawdown(asset_history: list[float]) -> float:
    peak = asset_history[0]
    max_drawdown = 0.0
    for asset in asset_history:
        peak = max(peak, asset)
        max_drawdown = min(max_drawdown, (asset - peak) / peak)
    return round(max_drawdown, 4)


def _safety_portfolio(state: dict[str, Any], today_orders: list[dict[str, Any]], pending_order: dict[str, Any]) -> dict[str, Any]:
    return {
        "cash": state.get("cash"),
        "total_assets": state.get("total_assets"),
        "max_drawdown": state.get("max_drawdown", 0),
        "daily_profit_rate": state.get("daily_profit_rate", 0),
        "today_orders": today_orders,
        "_pending_order": pending_order,
    }


def _safety_rejected_order(order: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    return {
        **order,
        "action": order.get("action") or order.get("side"),
        "status": "REJECTED",
        "order_status": "REJECTED",
        "broker_provider": "paper",
        "live_trading": False,
        "safety_checked": True,
        "rejected": True,
        "rejected_reason": validation.get("reason"),
        "safety_rule": validation.get("safety_rule"),
        "skipped_reason": validation.get("reason"),
    }
