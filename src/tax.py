"""Shared simplified tax calculations for paper trading analysis."""

from __future__ import annotations

from typing import Any


def calculate_period_estimated_tax(
    gross_cumulative_profit: float,
    total_commission: float,
    config: dict[str, Any],
) -> float:
    """Estimate tax after netting gains and losses for the whole period."""
    costs = config.get("costs", {})
    if not costs.get("apply_tax_on_profit", True):
        return 0.0
    if gross_cumulative_profit <= 0:
        return 0.0

    tax_rate = float(costs.get("tax_rate", 0.0) or 0.0)
    taxable_profit = max(gross_cumulative_profit - total_commission, 0.0)
    return round(taxable_profit * tax_rate, 2)


def calculate_period_profit_summary(
    closed_trades: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, float]:
    """Summarize realized profit using period-level tax netting."""
    gross_cumulative_profit = round(
        sum(float(trade.get("gross_profit", trade.get("profit", 0)) or 0) for trade in closed_trades),
        2,
    )
    total_commission = round(sum(float(trade.get("total_commission", 0) or 0) for trade in closed_trades), 2)
    estimated_tax_total = calculate_period_estimated_tax(gross_cumulative_profit, total_commission, config)
    net_cumulative_profit = round(gross_cumulative_profit - estimated_tax_total - total_commission, 2)
    return {
        "gross_cumulative_profit": gross_cumulative_profit,
        "net_cumulative_profit": net_cumulative_profit,
        "total_commission": total_commission,
        "estimated_tax_total": estimated_tax_total,
    }
