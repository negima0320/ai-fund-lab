from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase3d_detail_audit import PortfolioManagerPhase3DDetailAudit


PERIOD = "2023-01-01_to_2026-05-31"


def _write_profile(root: Path, profile: str, *, pm_multiplier: float = 1.3, code: str = "67400") -> None:
    path = root / "logs" / "backtests" / profile / PERIOD
    path.mkdir(parents=True)
    summary = {
        "initial_capital": 1_000_000,
        "final_assets": 1_100_000,
        "net_cumulative_profit": 100_000,
        "profit_factor": 2.0,
        "max_drawdown": -0.05,
        "win_rate": 0.5,
        "closed_trade_count": 2,
        "daily_asset_curve": [
            {"date": "2023-01-10", "total_assets": 1_000_000},
            {"date": "2023-01-20", "total_assets": 1_100_000},
        ],
        "all_trades": [
            {
                "action": "BUY",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "code": code,
                "pm_high_conviction_proba": 0.9,
                "pm_avoid_proba": 0.45,
                "pm_score": 0.45,
                "pm_multiplier": pm_multiplier,
                "pm_model_version": "test",
                "pm_feature_count": 68,
            }
        ],
    }
    (path / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "exit_date": "2023-01-10",
                "code": code,
                "shares": 100,
                "entry_price": 1000,
                "net_profit": 10_000,
                "holding_days": 5,
                "pm_high_conviction_proba": 0.9,
                "pm_avoid_proba": 0.45,
                "pm_multiplier": pm_multiplier,
                "pm_model_version": "test",
                "pm_feature_count": 68,
                "pm_score": 0.45,
            },
            {
                "action": "SELL",
                "signal_date": "2023-01-11",
                "entry_date": "2023-01-12",
                "exit_date": "2023-01-20",
                "code": "12340",
                "shares": 100,
                "entry_price": 1000,
                "net_profit": -5_000,
                "holding_days": 8,
                "pm_high_conviction_proba": 0.4,
                "pm_avoid_proba": 0.7,
                "pm_multiplier": 0.6,
                "pm_model_version": "test",
                "pm_feature_count": 68,
                "pm_score": -0.3,
            },
        ]
    ).to_csv(path / "trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "code": code,
                "decision": "BUY",
                "skip_reason": "",
                "final_amount": 100_000,
                "pm_high_conviction_proba": 0.9,
                "pm_avoid_proba": 0.45,
                "pm_score": 0.45,
                "pm_multiplier": pm_multiplier,
                "pm_model_version": "test",
                "pm_feature_count": 68,
            },
            {
                "signal_date": "2023-01-11",
                "entry_date": "2023-01-12",
                "code": "12340",
                "decision": "SKIP",
                "skip_reason": "selected_but_not_affordable",
                "final_amount": 0,
            },
        ]
    ).to_csv(path / "purchase_audit.csv", index=False)


def test_detail_audit_builds_multiplier_and_score_summaries(tmp_path: Path) -> None:
    _write_profile(tmp_path, "baseline", pm_multiplier=1.0)
    _write_profile(tmp_path, "phase3d", pm_multiplier=1.3)

    result = PortfolioManagerPhase3DDetailAudit(
        root=tmp_path,
        baseline_profile="baseline",
        phase3d_profile="phase3d",
        period=PERIOD,
    ).build()

    assert result["pm_multiplier_summary"]
    assert result["pm_score_band_summary"]
    assert result["focus_code_dependency"]["phase3d_profit"] == 10_000
    assert result["pm_log_check"]["trades_csv_sell_pm_non_null_rows"] == 2
    assert result["pm_log_check"]["all_trades_buy_pm_non_null_rows"] == 1


def test_detail_audit_report_can_be_saved(tmp_path: Path) -> None:
    _write_profile(tmp_path, "baseline", pm_multiplier=1.0)
    _write_profile(tmp_path, "phase3d", pm_multiplier=1.3)
    auditor = PortfolioManagerPhase3DDetailAudit(
        root=tmp_path,
        baseline_profile="baseline",
        phase3d_profile="phase3d",
        period=PERIOD,
    )
    paths = auditor.save(auditor.build())
    assert paths.markdown.exists()
    assert paths.json.exists()
