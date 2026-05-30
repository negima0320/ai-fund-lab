from __future__ import annotations

from tax import calculate_period_estimated_tax, calculate_period_profit_summary


def test_period_estimated_tax_is_zero_when_gross_profit_is_negative(config_copy: dict) -> None:
    assert calculate_period_estimated_tax(-1000.0, 0.0, config_copy) == 0.0


def test_period_profit_summary_uses_netting_before_tax(config_copy: dict) -> None:
    config_copy["costs"]["tax_rate"] = 0.20315
    summary = calculate_period_profit_summary(
        [
            {"gross_profit": 1000.0, "total_commission": 100.0, "estimated_tax": 182.84},
            {"gross_profit": -1500.0, "total_commission": 100.0, "estimated_tax": 0.0},
        ],
        config_copy,
    )

    assert summary["gross_cumulative_profit"] == -500.0
    assert summary["total_commission"] == 200.0
    assert summary["estimated_tax_total"] == 0.0
    assert summary["net_cumulative_profit"] == -700.0


def test_period_profit_summary_taxes_only_positive_period_profit(config_copy: dict) -> None:
    config_copy["costs"]["tax_rate"] = 0.2
    summary = calculate_period_profit_summary(
        [
            {"gross_profit": 2000.0, "total_commission": 100.0},
            {"gross_profit": -500.0, "total_commission": 100.0},
        ],
        config_copy,
    )

    assert summary["gross_cumulative_profit"] == 1500.0
    assert summary["total_commission"] == 200.0
    assert summary["estimated_tax_total"] == 260.0
    assert summary["net_cumulative_profit"] == 1040.0
