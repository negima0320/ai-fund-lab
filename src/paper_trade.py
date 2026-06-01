"""Paper trading engine for the first safe implementation."""

from __future__ import annotations

import random
from typing import Any

from broker import build_broker
from commentary import generate_buy_comment, generate_no_trade_comment, generate_sell_comment
from market_sections import market_section_allowed
from safety import can_trade, safety_event
from tax import calculate_period_profit_summary


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
        return 0, "買付余力が不足しているため買付不可"
    if _use_round_lot(config):
        lot_size = _round_lot_size(config)
        minimum_amount = price * lot_size
        if minimum_amount > allocation:
            return 0, f"{lot_size}株購入に必要な金額が1銘柄上限を超えるため買付不可"
        lots = int(allocation // minimum_amount)
        return lots * lot_size, ""
    if not bool(config.get("trading", {}).get("allow_fractional_shares", False)):
        shares = int(allocation // price)
        if shares <= 0:
            return 0, "1株購入に必要な金額が1銘柄上限を超えるため買付不可"
        return shares, ""
    shares = int(allocation // price)
    if shares <= 0:
        return 0, "買付可能株数が0のため買付不可"
    return shares, ""


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
) -> dict[str, Any]:
    return {
        "trade_id": trade_id,
        "action": action,
        "code": code,
        "name": name,
        "entry_date": trade_date,
        "entry_price": price,
        "shares": 0,
        "amount": 0,
        "allocation_limit": round(allocation_limit, 2),
        "score": score,
        "reason": reason,
        "round_lot_size": _round_lot_size(config),
        "use_round_lot": _use_round_lot(config),
        "skipped_reason": skipped_reason,
        "dealer_comment": generate_no_trade_comment(skipped_reason, config),
    }


def _use_round_lot(config: dict[str, Any]) -> bool:
    return bool(config.get("trading", {}).get("use_round_lot", False))


def _round_lot_size(config: dict[str, Any]) -> int:
    return int(config.get("trading", {}).get("round_lot_size", 100))


def _technical_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    selected_reason = item.get("selection_reason") or item.get("selected_reason") or item.get("reason", "")
    return {
        "rsi": item.get("rsi"),
        "volume_ratio": item.get("volume_ratio"),
        "total_score": item.get("total_score"),
        "technical_score": item.get("technical_score"),
        "selected_reason": selected_reason,
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
        "sector_score": item.get("sector_score") or item.get("sector_score_adjustment"),
        "penalty_score": item.get("penalty_score"),
        "score_components": item.get("score_components", {}),
        "score_components_total": item.get("score_components_total"),
        "score_components_match": item.get("score_components_match"),
        "candle_type": item.get("candle_type"),
        "candlestick_signals": item.get("candlestick_signals", []),
        "ma5": item.get("ma5"),
        "ma25": item.get("ma25"),
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
        "relative_strength_score": item.get("relative_strength_score"),
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
        "advance_ratio": item.get("advance_ratio"),
        "market_filter_reason": item.get("market_filter_reason", ""),
        "earnings_filter_checked": item.get("earnings_filter_checked", False),
        "earnings_filter_blocked": item.get("earnings_filter_blocked", False),
        "earnings_filter_reason": item.get("earnings_filter_reason", ""),
        "earnings_announcement_date": item.get("earnings_announcement_date"),
        "earnings_calendar_records_count": item.get("earnings_calendar_records_count"),
        "earnings_info_found": item.get("earnings_info_found", False),
        "earnings_candidate_date": item.get("earnings_candidate_date"),
        "earnings_days_until_earnings": item.get("earnings_days_until_earnings"),
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
        "sector_score",
        "penalty_score",
        "score_components",
        "score_components_total",
        "score_components_match",
        "candle_type",
        "candlestick_signals",
        "ma5",
        "ma25",
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
        "advance_ratio",
        "market_filter_reason",
        "earnings_filter_checked",
        "earnings_filter_blocked",
        "earnings_filter_reason",
        "earnings_announcement_date",
        "earnings_calendar_records_count",
        "earnings_info_found",
        "earnings_candidate_date",
        "earnings_days_until_earnings",
    ]
    return {key: position.get(key) for key in keys if key in position}


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
    return float(value)


def _candidate_market_price(item: dict[str, Any], fallback: float | None = None) -> float:
    value = item.get("close")
    if value is None:
        value = fallback
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

    trades = []
    safety_events = []
    next_positions = list(state.get("positions", []))
    closed_today = []
    price_by_code = {item["code"]: item for item in scored_candidates}
    pending_orders = list(state.get("pending_orders", []))
    due_pending, future_pending = _split_due_pending_orders(pending_orders, trade_date)

    for pending in due_pending:
        market = price_by_code.get(pending["code"], {})
        executed_price = _execution_price(market, pending)
        if pending["action"] == "BUY":
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
                    "market_value": round(amount, 2),
                    "buy_commission": buy_commission,
                    "holding_days": 1,
                    "score": pending.get("score"),
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
        market = price_by_code.get(position["code"])
        current_price = float(market["close"]) if market else float(position["current_price"])
        holding_days = int(position["holding_days"]) if position.get("entry_date") == trade_date else int(position["holding_days"]) + 1
        exit_plan = _real_exit_plan(
            position=position,
            market=market or {},
            trade_date=trade_date,
            current_price=current_price,
            holding_days=holding_days,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_days=max_holding_days,
            stop_loss_execution=stop_loss_execution,
        )
        profit_rate = exit_plan["mark_profit_rate"]
        exit_reason = exit_plan["exit_reason"]
        planned_exit_price = float(exit_plan["exit_price"] or current_price)
        updated_position = {
            **position,
            "current_price": current_price,
            "market_value": round(int(position["shares"]) * current_price, 2),
            "holding_days": holding_days,
            "unrealized_profit": round(int(position["shares"]) * (current_price - float(position["entry_price"])), 2),
            "unrealized_profit_rate": round(profit_rate, 4),
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
                    **_stop_loss_trade_fields(exit_plan, planned_exit_price),
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
    selection = config.get("selection", {})
    min_score = float(selection.get("fallback_min_score", selection.get("top_pick_min_score", 70)))
    min_confidence = float(selection.get("min_confidence", config["scoring"].get("confidence_min_for_buy", 0.7)))
    selected = [
        item
        for item in scored_candidates
        if item.get("selected") and float(item["total_score"]) >= min_score and float(item["confidence"]) >= min_confidence
    ]
    selected.sort(key=lambda item: (float(item["total_score"]), float(item["confidence"])), reverse=True)
    buy_candidates = []
    for item in selected:
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
                )
            )
            continue
        if len(next_positions) + len(pending_buy_codes) >= max_positions:
            break
        if item["code"] in held_codes or item["code"] in pending_buy_codes:
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
                )
            )
            continue
        allocation = min(allocation_limit, cash)
        entry_price = _candidate_entry_price(item)
        current_price = _candidate_market_price(item, entry_price)
        shares, skipped_reason = _calculate_buy_shares(entry_price, allocation, config)
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
                )
            )
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
            "reason": item.get("selection_reason") or item.get("selected_reason") or item["reason"],
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
            "buy_commission": buy_commission,
            "score": item["total_score"],
            "reason": item.get("selection_reason") or item.get("selected_reason") or item["reason"],
            **_technical_snapshot(item),
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
            continue
        held_codes.add(item["code"])
        pending_buy_codes.add(item["code"])
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
