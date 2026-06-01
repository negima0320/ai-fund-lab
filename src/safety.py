"""Safety guard checks for paper and future live trading."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_order(order: dict[str, Any], portfolio: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    safety = config.get("safety", {})
    action = _action(order)
    amount = _order_amount(order)

    if safety.get("mode", "paper") != "paper" and not safety.get("allow_live_trading", False):
        return _reject("live_trading_disabled", "allow_live_trading が false のため実売買注文は実行できません")

    emergency = check_emergency_stop(config)
    if emergency["stopped"] and action == "BUY":
        return _reject("emergency_stop", "STOP_TRADING ファイルが存在するため新規買付停止")

    max_single = float(safety.get("max_single_order_amount", 0) or 0)
    if not _single_order_amount_limit_disabled(config) and max_single > 0 and amount > max_single:
        return _reject("max_single_order_amount", "1注文上限を超えています")

    drawdown = check_drawdown_limit(portfolio, config)
    if drawdown["stopped"]:
        return _reject("drawdown_limit", drawdown["reason"])

    daily_loss = check_daily_loss_limit(portfolio, config)
    if daily_loss["stopped"]:
        return _reject("daily_loss_limit", daily_loss["reason"])

    return _ok()


def validate_daily_limits(today_orders: list[dict[str, Any]], portfolio: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    safety = config.get("safety", {})
    max_orders = int(safety.get("max_orders_per_day", 0) or 0)
    executable_orders = [order for order in today_orders if not order.get("rejected") and _action(order) in {"BUY", "SELL"}]
    if max_orders > 0 and len(executable_orders) >= max_orders:
        return _reject("max_orders_per_day", "1日の注文数上限を超えています")

    buy_total = sum(_order_amount(order) for order in executable_orders if _action(order) == "BUY")
    sell_total = sum(_order_amount(order) for order in executable_orders if _action(order) == "SELL")
    next_order = portfolio.get("_pending_order", {})
    next_amount = _order_amount(next_order)
    action = _action(next_order)

    max_buy = float(safety.get("max_daily_buy_amount", 0) or 0)
    if action == "BUY" and max_buy > 0 and buy_total + next_amount > max_buy:
        return _reject("max_daily_buy_amount", "1日の買付上限を超えています")

    max_sell = float(safety.get("max_daily_sell_amount", 0) or 0)
    if action == "SELL" and max_sell > 0 and sell_total + next_amount > max_sell:
        return _reject("max_daily_sell_amount", "1日の売却上限を超えています")

    return _ok()


def check_emergency_stop(config: dict[str, Any]) -> dict[str, Any]:
    path = _emergency_stop_path(config)
    exists = path.exists()
    return {
        "stopped": exists,
        "path": str(path),
        "reason": "STOP_TRADING ファイルが存在します" if exists else "",
    }


def check_drawdown_limit(portfolio: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    threshold = float(config.get("safety", {}).get("stop_trading_if_drawdown_exceeds", 0) or 0)
    current = float(portfolio.get("max_drawdown", 0) or 0)
    stopped = threshold < 0 and current <= threshold
    return {
        "stopped": stopped,
        "reason": f"最大ドローダウンが停止基準を超えています ({current:.2%})" if stopped else "",
    }


def check_daily_loss_limit(portfolio: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    threshold = float(config.get("safety", {}).get("stop_trading_if_daily_loss_rate_exceeds", 0) or 0)
    current = float(portfolio.get("daily_profit_rate", portfolio.get("day_change_pct", 0)) or 0)
    stopped = threshold < 0 and current <= threshold
    return {
        "stopped": stopped,
        "reason": f"日次損失率が停止基準を超えています ({current:.2%})" if stopped else "",
    }


def can_trade(order: dict[str, Any], portfolio: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    order_check = validate_order(order, portfolio, config)
    if not order_check["allowed"]:
        return order_check
    daily_check = validate_daily_limits(portfolio.get("today_orders", []), {**portfolio, "_pending_order": order}, config)
    if not daily_check["allowed"]:
        return daily_check
    return _ok()


def _single_order_amount_limit_disabled(config: dict[str, Any]) -> bool:
    policy = config.get("capital_utilization_policy", {})
    return bool(
        config.get("disable_single_order_amount_limit")
        or config.get("safety", {}).get("disable_single_order_amount_limit")
        or (isinstance(policy, dict) and policy.get("disable_single_order_amount_limit"))
    )


def safety_event(date: str, order: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    order_payload = {
        **order,
        "order_status": "REJECTED",
        "broker_provider": order.get("broker_provider") or "paper",
        "live_trading": bool(order.get("live_trading", False)),
        "safety_checked": True,
    }
    return {
        "date": date,
        "order": order_payload,
        "rejected": True,
        "rejected_reason": validation.get("reason", "セーフティガードにより拒否"),
        "safety_rule": validation.get("safety_rule", "unknown"),
    }


def _emergency_stop_path(config: dict[str, Any]) -> Path:
    path = Path(config.get("safety", {}).get("emergency_stop_file", "storage/STOP_TRADING"))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path


def _action(order: dict[str, Any]) -> str:
    return str(order.get("action") or order.get("side") or "").upper()


def _order_amount(order: dict[str, Any]) -> float:
    amount = order.get("amount") or order.get("notional")
    if amount is not None:
        return float(amount)
    shares = order.get("shares") or order.get("quantity") or 0
    price = order.get("entry_price") or order.get("exit_price") or order.get("price") or 0
    return float(shares) * float(price)


def _ok() -> dict[str, Any]:
    return {"allowed": True, "reason": "", "safety_rule": ""}


def _reject(rule: str, reason: str) -> dict[str, Any]:
    return {"allowed": False, "reason": reason, "safety_rule": rule}
