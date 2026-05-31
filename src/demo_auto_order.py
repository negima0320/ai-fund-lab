"""Demo auto-order guardrails for Tachibana demo environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from broker import BaseBroker, LiveTradingDisabledError


class DemoAutoOrderBlocked(RuntimeError):
    """Raised when demo auto-order safety checks block execution."""


def load_operation_schedule(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def latest_order_preview_path(root: Path, profile_id: str) -> Path:
    preview_dir = root / "reports" / profile_id / "order_previews"
    candidates = sorted(preview_dir.glob("order_preview_*.json"))
    if not candidates:
        raise FileNotFoundError(f"order preview not found: {preview_dir}")
    return candidates[-1]


def validate_demo_auto_order(
    config: dict[str, Any],
    schedule: dict[str, Any],
    orders: list[dict[str, Any]],
    balance: dict[str, Any],
    positions: list[dict[str, Any]],
    existing_orders: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    broker = config.get("broker", {})
    tachibana = config.get("tachibana", {})
    policy = schedule.get("execution_policy", {})
    schedule_safety = schedule.get("safety", {})
    safety = config.get("safety", {})

    if str(tachibana.get("environment", "demo")) == "live":
        raise DemoAutoOrderBlocked("env=live のためdemo自動発注を停止します")
    if str(broker.get("provider", "paper")) == "tachibana_live":
        raise DemoAutoOrderBlocked("broker=tachibana_live のためdemo自動発注を停止します")
    if bool(schedule_safety.get("forbid_live_auto_order", True)) and str(broker.get("provider")) == "tachibana_live":
        raise DemoAutoOrderBlocked("forbid_live_auto_order=true のためlive自動発注は禁止です")
    if str(broker.get("provider")) != "tachibana_demo":
        raise DemoAutoOrderBlocked("broker が tachibana_demo ではありません")
    if str(policy.get("broker")) != "tachibana_demo":
        raise DemoAutoOrderBlocked("operation_schedule.execution_policy.broker が tachibana_demo ではありません")
    if not bool(policy.get("auto_order_enabled", False)):
        raise DemoAutoOrderBlocked("auto_order_enabled=false のため自動発注を停止します")
    if str(policy.get("execution_mode")) != "auto_demo":
        raise DemoAutoOrderBlocked("execution_mode が auto_demo ではありません")

    buy_orders = [order for order in orders if _action(order) == "BUY"]
    held_codes = {str(item.get("code")) for item in positions}
    order_codes = {str(item.get("code")) for item in existing_orders or []}
    cash = float(balance.get("cash") or 0.0)
    total_buy_amount = sum(_order_amount(order) for order in buy_orders)

    if total_buy_amount > cash:
        raise DemoAutoOrderBlocked("cash不足のため自動発注を停止します")
    duplicate_holding = [order for order in buy_orders if str(order.get("code")) in held_codes]
    if duplicate_holding:
        raise DemoAutoOrderBlocked("同一銘柄保有中のため自動発注を停止します")
    duplicate_order = [order for order in buy_orders if str(order.get("code")) in order_codes]
    if duplicate_order:
        raise DemoAutoOrderBlocked("同一銘柄の当日注文があるため自動発注を停止します")

    max_positions = int(config.get("portfolio", {}).get("max_positions", 0) or 0)
    if max_positions > 0 and len(positions) + len(buy_orders) > max_positions:
        raise DemoAutoOrderBlocked("max_positions超過のため自動発注を停止します")

    max_orders = int(safety.get("max_orders_per_day", 0) or 0)
    if max_orders > 0 and len(existing_orders or []) + len(orders) > max_orders:
        raise DemoAutoOrderBlocked("当日注文上限超過のため自動発注を停止します")

    max_daily_buy = float(safety.get("max_daily_buy_amount", 0) or 0)
    if max_daily_buy > 0 and total_buy_amount > max_daily_buy:
        raise DemoAutoOrderBlocked("max_daily_buy_amount超過のため自動発注を停止します")

    max_single = float(safety.get("max_single_order_amount", 0) or 0)
    oversized = [order for order in orders if max_single > 0 and _order_amount(order) > max_single]
    if oversized:
        raise DemoAutoOrderBlocked("max_single_order_amount超過のため自動発注を停止します")

    return {
        "allowed": True,
        "broker": "tachibana_demo",
        "order_count": len(orders),
        "buy_count": len(buy_orders),
        "estimated_buy_amount": round(total_buy_amount, 2),
    }


def execute_demo_auto_orders(
    config: dict[str, Any],
    schedule: dict[str, Any],
    orders: list[dict[str, Any]],
    broker: BaseBroker,
) -> dict[str, Any]:
    if broker.provider != "tachibana_demo":
        raise DemoAutoOrderBlocked("broker が tachibana_demo ではありません")
    balance = broker.get_account_balance()
    positions = broker.get_positions()
    existing_orders = broker.get_orders()
    validation = validate_demo_auto_order(config, schedule, orders, balance, positions, existing_orders)
    results = [broker.place_order(order) for order in orders]
    return {
        "status": "ordered",
        "validation": validation,
        "orders": results,
    }


def _action(order: dict[str, Any]) -> str:
    return str(order.get("action") or order.get("side") or "").upper()


def _order_amount(order: dict[str, Any]) -> float:
    amount = order.get("estimated_amount") or order.get("amount") or order.get("notional")
    if amount is not None:
        return float(amount)
    shares = order.get("shares") or order.get("quantity") or 0
    price = order.get("estimated_price") or order.get("entry_price") or order.get("price") or 0
    return float(shares) * float(price)
