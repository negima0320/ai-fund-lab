from __future__ import annotations

from pathlib import Path

import pandas as pd

from profile_loader import load_profile
from paper_trade import _apply_portfolio_manager_sizing, _apply_relative_allocator_to_candidates, _purchase_audit_event


def _config() -> dict:
    return {
        "profile_id": "test_relative_allocator",
        "profile_name": "test",
        "portfolio_manager_ai_sizing": {
            "enabled": True,
            "relative_allocator_enabled": True,
            "low_score_skip_enabled": False,
            "per_code_exposure_cap_enabled": True,
            "per_code_exposure_cap_rate": 0.38,
        },
        "relative_allocator": {"enabled": True, "rule": "blended_relative_score"},
        "trading": {"use_round_lot": True, "round_lot_size": 100},
    }


def _items() -> list[dict]:
    return [
        {
            "code": "11110",
            "name": "A",
            "signal_date": "2023-01-04",
            "total_score": 50,
            "confidence": 0.9,
            "risk_adjusted_score": 0.90,
            "expected_return_10d": 0.10,
            "bad_entry_probability_10d": 0.02,
        },
        {
            "code": "22220",
            "name": "B",
            "signal_date": "2023-01-04",
            "total_score": 49,
            "confidence": 0.9,
            "risk_adjusted_score": 0.20,
            "expected_return_10d": -0.01,
            "bad_entry_probability_10d": 0.50,
        },
        {
            "code": "33330",
            "name": "C",
            "signal_date": "2023-01-04",
            "total_score": 48,
            "confidence": 0.9,
            "risk_adjusted_score": 0.55,
            "expected_return_10d": 0.04,
            "bad_entry_probability_10d": 0.20,
        },
    ]


def test_v292_profile_loads_and_aliases() -> None:
    profile = load_profile("rookie_dealer_02_v2_92_relative_allocator_cap38")
    alias = load_profile("rookie_dealer_02_v2_92")
    dot_alias = load_profile("rookie_dealer_02_v2.92")

    assert profile["profile_id"] == "rookie_dealer_02_v2_92_relative_allocator_cap38"
    assert alias["relative_allocator"]["rule"] == "blended_relative_score"
    assert dot_alias["portfolio_manager_ai_sizing"]["per_code_exposure_cap_rate"] == 0.38
    assert profile["relative_allocator"]["enabled"] is True
    assert profile["portfolio_manager_ai_sizing"]["rule"] == "relative_allocator"


def test_relative_allocator_assigns_only_prediction_time_candidate_fields() -> None:
    assigned = _apply_relative_allocator_to_candidates(_items(), _config())
    by_code = {item["code"]: item for item in assigned}

    assert by_code["11110"]["relative_rank"] == 1
    assert by_code["11110"]["relative_multiplier"] == 1.30
    assert by_code["22220"]["relative_multiplier"] == 0.80
    assert by_code["11110"]["pm_multiplier_source"] == "relative_allocator"
    assert "selected_count_in_day" not in by_code["11110"]
    assert "cash_after" not in by_code["11110"]


def test_relative_allocator_sizing_does_not_load_pm_model() -> None:
    item = _apply_relative_allocator_to_candidates(_items(), _config())[0]
    shares, fields = _apply_portfolio_manager_sizing(
        item=item,
        trade_date="2023-01-05",
        shares=10,
        entry_price=1000.0,
        cash=100000.0,
        config=_config(),
    )

    assert shares == 0  # 13,000 yen target is below one 100-share lot.
    assert fields["pm_multiplier"] == 1.30
    assert fields["pm_model_version"] == "relative_allocator_v1"
    assert fields["pm_model_path"] == ""
    assert fields["pm_multiplier_source"] == "relative_allocator"


def test_purchase_audit_contains_relative_allocator_columns() -> None:
    item = _apply_relative_allocator_to_candidates(_items(), _config())[0]
    event = _purchase_audit_event(
        item=item,
        trade_date="2023-01-05",
        config=_config(),
        cash_before=100000,
        cash_after=90000,
        daily_buy_limit_remaining_before=100000,
        daily_buy_limit_remaining_after=90000,
        max_positions_remaining_before=5,
        decision="BUY",
    )

    assert event["relative_allocator_enabled"] is True
    assert event["relative_allocator_rule"] == "blended_relative_score"
    assert event["relative_rank"] == 1
    assert event["pm_multiplier_source"] == "relative_allocator"


def test_phase8h_report_builder_handles_fixture(tmp_path: Path, monkeypatch) -> None:
    import scripts.ml.report_phase8h_relative_allocator_backtest as report_script

    log_dir = tmp_path / "logs/backtests/rookie_dealer_02_v2_92_relative_allocator_cap38/2023-01-01_to_2026-05-31"
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2023-01-05", "total_assets": 1000000, "positions_value": 0},
            {"date": "2023-01-06", "total_assets": 1010000, "positions_value": 500000},
        ]
    ).to_csv(log_dir / "summary.csv", index=False)
    pd.DataFrame(
        [
            {"action": "SELL", "code": "11110", "exit_date": "2023-01-06", "net_profit": 10000, "holding_days": 2, "pm_multiplier": 1.3}
        ]
    ).to_csv(log_dir / "trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "relative_allocator_enabled": True,
                "relative_rank": 1,
                "relative_candidate_count": 3,
                "relative_score": 1.0,
                "pm_multiplier": 1.3,
            }
        ]
    ).to_csv(log_dir / "purchase_audit.csv", index=False)
    (log_dir / "backtest_summary.json").write_text(
        '{"initial_capital": 1000000, "final_assets": 1010000, "net_cumulative_profit": 10000, "profit_factor": 2.5, "max_drawdown": -0.01, "win_rate": 1.0, "total_trades": 1}',
        encoding="utf-8",
    )

    monkeypatch.setattr(report_script, "ROOT", tmp_path)
    report = report_script.build_report()
    v292 = next(row for row in report["relative_allocator_metrics"] if row["label"] == "v2_92_relative_allocator_cap38")
    assert v292["relative_allocator_operationally_valid"] is True
