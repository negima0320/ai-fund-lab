from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase6f_cap_constraint_audit import BOOSTER_PROFILE, Phase6FCapConstraintAudit


PROFILE = "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
PERIOD = "2023-01-01_to_2026-05-31"


def _write_fixture(root: Path) -> None:
    log_dir = root / "logs" / "backtests" / PROFILE / PERIOD
    log_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2023-01-04", periods=100)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "code": "11110",
                "entry_date": dates[70].strftime("%Y-%m-%d"),
                "exit_date": dates[74].strftime("%Y-%m-%d"),
                "net_profit": 30_000,
                "net_profit_rate": 0.10,
                "holding_days": 4,
                "pm_multiplier": 1.30,
            },
            {
                "action": "SELL",
                "code": "22220",
                "entry_date": dates[72].strftime("%Y-%m-%d"),
                "exit_date": dates[76].strftime("%Y-%m-%d"),
                "net_profit": -5_000,
                "net_profit_rate": -0.05,
                "holding_days": 4,
                "pm_multiplier": 1.15,
            },
        ]
    ).to_csv(log_dir / "trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "entry_date": dates[70].strftime("%Y-%m-%d"),
                "code": "11110",
                "decision": "BUY",
                "final_amount": 300_000,
                "pm_multiplier": 1.30,
                "pm_per_code_cap_rate": 0.30,
                "pm_per_code_current_exposure": 0,
                "pm_per_code_max_exposure": 300_000,
                "pm_per_code_cap_original_amount": 500_000,
                "pm_per_code_cap_amount": 300_000,
                "pm_per_code_cap_reduced": True,
                "pm_per_code_cap_skip": False,
            },
            {
                "entry_date": dates[72].strftime("%Y-%m-%d"),
                "code": "22220",
                "decision": "BUY",
                "final_amount": 200_000,
                "pm_multiplier": 1.15,
                "pm_per_code_cap_rate": 0.30,
                "pm_per_code_current_exposure": 100_000,
                "pm_per_code_max_exposure": 300_000,
                "pm_per_code_cap_original_amount": 400_000,
                "pm_per_code_cap_amount": 200_000,
                "pm_per_code_cap_reduced": True,
                "pm_per_code_cap_skip": False,
            },
            {
                "entry_date": dates[74].strftime("%Y-%m-%d"),
                "code": "33330",
                "decision": "BUY",
                "final_amount": 100_000,
                "pm_multiplier": 0.80,
                "pm_per_code_cap_rate": 0.30,
                "pm_per_code_current_exposure": 0,
                "pm_per_code_max_exposure": 300_000,
                "pm_per_code_cap_original_amount": 100_000,
                "pm_per_code_cap_amount": 100_000,
                "pm_per_code_cap_reduced": False,
                "pm_per_code_cap_skip": False,
            },
        ]
    ).to_csv(log_dir / "purchase_audit.csv", index=False)

    booster_dir = root / "logs" / "backtests" / BOOSTER_PROFILE / PERIOD
    booster_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "entry_date": dates[70].strftime("%Y-%m-%d"),
                "code": "11110",
                "final_amount": 300_000,
                "pm_multiplier": 1.30,
                "pm_per_code_cap_rate": 0.30,
                "pm_per_code_current_exposure": 0,
                "pm_per_code_max_exposure": 300_000,
                "pm_per_code_cap_original_amount": 500_000,
                "pm_per_code_cap_amount": 300_000,
                "bear_pm_booster_applied": True,
                "bear_pm_booster_before_amount": 300_000,
                "bear_pm_booster_after_amount": 450_000,
            }
        ]
    ).to_csv(booster_dir / "purchase_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "code": "11110",
                "entry_date": dates[70].strftime("%Y-%m-%d"),
                "exit_date": dates[74].strftime("%Y-%m-%d"),
                "net_profit": 30_000,
                "net_profit_rate": 0.10,
            }
        ]
    ).to_csv(booster_dir / "trades.csv", index=False)

    topix_dir = root / "data" / "cache" / "jquants" / "topix_prices"
    topix_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, day in enumerate(pd.bdate_range("2022-10-03", periods=180)):
        close = 1000 + idx * 2 if idx < 105 else 1210 - (idx - 105) * 8
        records.append({"date": day.strftime("%Y-%m-%d"), "open": close, "high": close, "low": close, "close": close})
    (topix_dir / "2022-10-03_to_2023-06-09.json").write_text(json.dumps({"records": records}), encoding="utf-8")


def test_phase6f_builds_cap_constraint_audit(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = Phase6FCapConstraintAudit(tmp_path).build_report()

    assert result["metadata"]["audit_only"] is True
    assert result["metadata"]["full_backtest_executed"] is False
    assert result["coverage"]["cap_hit_count"] == 2
    assert len(result["cap_rate_audit"]) == 3
    assert result["cap_rate_audit"][0]["newly_allowed_amount"] > 0
    assert result["bear_booster_cap_relation"][0]["booster_blocked_by_cap"] > 0
    assert result["pm_high_score_cap_audit"][0]["trade_count"] == 2
    assert result["virtual_comparison"]
    assert "cap_is_current_bottleneck" in result["verdict"]


def test_phase6f_saves_reports(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = Phase6FCapConstraintAudit(tmp_path)

    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 6-F" in paths.markdown.read_text(encoding="utf-8")
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["sources"]["approximation_note"].startswith("Additional profit")
