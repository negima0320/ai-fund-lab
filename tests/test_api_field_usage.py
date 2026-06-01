from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main as main_module
from indicators import calculate_indicators


def test_normalize_daily_price_preserves_adjusted_and_limit_fields() -> None:
    row = main_module._normalize_daily_price(
        {
            "Code": "10010",
            "Date": "20260105",
            "O": 100,
            "H": 110,
            "L": 95,
            "C": 108,
            "Vo": 1000,
            "Va": 108000,
            "AdjO": 50,
            "AdjH": 55,
            "AdjL": 47.5,
            "AdjC": 54,
            "AdjVo": 2000,
            "UL": "1",
            "LL": "0",
        }
    )

    assert row["open"] == 100
    assert row["close"] == 108
    assert row["adjusted_close"] == 54
    assert row["adjusted_volume"] == 2000
    assert row["adjusted_price_usage"] == "available_not_used"
    assert row["limit_up_flag"] is True
    assert row["limit_down_flag"] is False
    assert row["turnover_value"] == 108000
    assert row["direct_turnover_value_source"] == "api_va"


def test_indicators_preserve_api_audit_fields_without_changing_score_inputs() -> None:
    rows = []
    for index in range(25):
        rows.append(
            {
                "code": "1001",
                "date": f"2026-01-{index + 1:02d}",
                "open": 100 + index,
                "high": 101 + index,
                "low": 99 + index,
                "close": 100 + index,
                "volume": 1000 + index,
                "adjusted_close": 50 + index,
                "adjusted_volume": 2000 + index,
                "adjusted_price_usage": "available_not_used",
                "limit_up_flag": index == 24,
                "limit_down_flag": False,
                "turnover_value": 123456,
            }
        )

    indicators, excluded = calculate_indicators(
        rows,
        {"1001": "Sample"},
        "2026-01-25",
        stock_metadata={
            "1001": {
                "scale_category": "TOPIX Small 1",
                "margin_type": "Loan",
                "product_category": "Common Stock",
                "sector17_code": "1",
                "sector33_code": "0050",
            }
        },
        indicator_mode="minimal",
    )

    assert excluded == 0
    item = indicators[0]
    assert item["close"] == 124
    assert item["adjusted_close"] == 74
    assert item["limit_up_flag"] is True
    assert item["direct_turnover_value"] == 123456
    assert item["direct_turnover_value_source"] == "api_va_available_not_used"
    assert item["scale_category"] == "TOPIX Small 1"
    assert item["product_category"] == "Common Stock"
