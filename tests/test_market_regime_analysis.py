from __future__ import annotations

import csv
import json
from pathlib import Path

from market_regime import classify_market_regime, effective_market_context_for_signal
from market_regime_analysis import build_market_regime_analysis, render_market_regime_analysis_markdown


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_market_regime_classifier_boundaries() -> None:
    assert classify_market_regime(0.70, 0.008, "risk_on") == "strong_bull"
    assert classify_market_regime(0.58, 0.002, "neutral") == "bull"
    assert classify_market_regime(0.30, -0.008, "risk_off") == "strong_bear"
    assert classify_market_regime(0.42, -0.002, "neutral") == "bear"
    assert classify_market_regime(0.50, 0.0, "neutral") == "range"
    assert classify_market_regime(None, None, "risk_on") == "bull"


def test_effective_market_context_uses_previous_day_even_when_same_day_exists() -> None:
    contexts = {
        "2026-01-05": {"advance_ratio": 0.20, "average_change_rate": -0.01, "market_regime": "risk_off"},
        "2026-01-06": {"advance_ratio": 0.80, "average_change_rate": 0.02, "market_regime": "risk_on"},
    }

    resolved = effective_market_context_for_signal("2026-01-06", contexts)

    assert resolved["source_date"] == "2026-01-05"
    assert resolved["regime"] == "strong_bear"
    assert resolved["same_day_used"] is False


def test_effective_market_context_falls_back_farther_back_when_previous_missing() -> None:
    contexts = {
        "2026-01-02": {"advance_ratio": 0.75, "average_change_rate": 0.01, "market_regime": "risk_on"},
        "2026-01-06": {"advance_ratio": 0.20, "average_change_rate": -0.01, "market_regime": "risk_off"},
    }

    resolved = effective_market_context_for_signal("2026-01-06", contexts)

    assert resolved["source_date"] == "2026-01-02"
    assert resolved["regime"] == "strong_bull"
    assert resolved["fallback_used"] is True


def test_market_regime_analysis_uses_existing_context_and_summary(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs/backtests/test_profile/2021-01-01_to_2021-01-05"
    _write_csv(
        log_dir / "trades.csv",
        [
            {
                "action": "SELL",
                "entry_date": "2021-01-04",
                "gross_profit": 1000,
                "gross_profit_rate": 0.02,
                "market_regime": "risk_on",
            },
            {
                "action": "SELL",
                "entry_date": "2021-01-05",
                "gross_profit": -500,
                "gross_profit_rate": -0.01,
                "market_regime": "risk_off",
            },
        ],
    )
    _write_csv(
        log_dir / "summary.csv",
        [
            {
                "date": "2021-01-04",
                "cash": 200000,
                "positions_value": 800000,
                "total_assets": 1000000,
                "daily_profit": 1000,
                "max_drawdown": 0,
                "open_positions_count": 2,
            },
            {
                "date": "2021-01-05",
                "cash": 600000,
                "positions_value": 400000,
                "total_assets": 1000000,
                "daily_profit": -500,
                "max_drawdown": -0.01,
                "open_positions_count": 1,
            },
        ],
    )
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    (processed / "market_context_2021-01-04.json").write_text(
        json.dumps({"advance_ratio": 0.8, "average_change_rate": 0.01, "market_regime": "risk_on"}),
        encoding="utf-8",
    )
    (processed / "market_context_2021-01-05.json").write_text(
        json.dumps({"advance_ratio": 0.2, "average_change_rate": -0.01, "market_regime": "risk_off"}),
        encoding="utf-8",
    )

    analysis = build_market_regime_analysis(tmp_path, "test_profile", "2021-01-01", "2021-01-05")

    assert analysis["trade_performance_by_regime"]["strong_bull"]["trade_count"] == 1
    assert analysis["trade_performance_by_regime"]["strong_bear"]["trade_count"] == 1
    assert analysis["capital_utilization_by_regime"]["strong_bull"]["average_market_exposure"] == 0.8
    assert "case_a" in analysis["exposure_improvement_simulation"]

    markdown = render_market_regime_analysis_markdown(analysis)
    assert "## 1. 市場局面分類" in markdown
    assert "## 4. 仮想エクスポージャー改善シミュレーション" in markdown
