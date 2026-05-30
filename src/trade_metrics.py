"""Shared trade aggregation rules for analyze, compare, and backtest summaries."""

from __future__ import annotations

from typing import Any


EXCLUDED_ORDER_STATUSES = {"PENDING", "REJECTED", "CANCELLED", "PREVIEW"}
EXCLUDED_ACTIONS = {"SKIP_BUY", "NO_BUY"}


def is_filled_trade(row: dict[str, Any]) -> bool:
    if row.get("action") not in {"BUY", "SELL"}:
        return False
    return str(row.get("order_status") or row.get("status") or "").upper() == "FILLED"


def is_closed_trade_for_metrics(row: dict[str, Any]) -> bool:
    return (
        is_filled_trade(row)
        and row.get("action") == "SELL"
        and row.get("result") in {"WIN", "LOSS"}
    )


def is_excluded_order_event(row: dict[str, Any]) -> bool:
    status = str(row.get("order_status") or row.get("status") or "").upper()
    action = str(row.get("action") or "").upper()
    return status in EXCLUDED_ORDER_STATUSES or action in EXCLUDED_ACTIONS


def profit_factor_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in rows if is_closed_trade_for_metrics(row)]
    wins = [row for row in closed if row.get("result") == "WIN"]
    losses = [row for row in closed if row.get("result") == "LOSS"]
    gross_profits = [float(row.get("gross_profit") or row.get("profit") or 0) for row in closed]
    realized_profit_total = round(sum(gross_profits), 2)
    gross_win_total = round(sum(value for value in gross_profits if value > 0), 2)
    gross_loss_total = round(sum(value for value in gross_profits if value < 0), 2)
    return {
        "closed_trades": closed,
        "wins": wins,
        "losses": losses,
        "closed_trade_count": len(closed),
        "win_count": len(wins),
        "loss_count": len(losses),
        "excluded_order_event_count": sum(1 for row in rows if is_excluded_order_event(row)),
        "realized_profit_total": realized_profit_total,
        "gross_profit_total": gross_win_total,
        "gross_win_total": gross_win_total,
        "gross_loss_total": gross_loss_total,
        "profit_factor": round(gross_win_total / abs(gross_loss_total), 4) if gross_loss_total < 0 else None,
    }
