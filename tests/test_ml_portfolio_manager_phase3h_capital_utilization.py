from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase3h_capital_utilization import PortfolioManagerPhase3HCapitalUtilizationAudit


PERIOD = "2023-01-01_to_2026-05-31"


def _write_profile(root: Path, profile: str, *, profit: float = 1000.0) -> None:
    base = root / "logs" / "backtests" / profile / PERIOD
    base.mkdir(parents=True)
    (base / "backtest_summary.json").write_text(
        json.dumps(
            {
                "net_cumulative_profit": profit,
                "profit_factor": 1.5,
                "max_drawdown": -0.1,
                "win_rate": 0.5,
                "closed_trades_count": 2,
                "daily_asset_curve": [
                    {"date": "2026-01-05", "total_assets": 1_000_000},
                    {"date": "2026-01-06", "total_assets": 1_010_000},
                ],
                "all_trades": [],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"day": 1, "date": "2026-01-05", "cash": 800_000, "positions_value": 200_000, "total_assets": 1_000_000, "daily_profit": 0, "open_positions_count": 1},
            {"day": 2, "date": "2026-01-06", "cash": 900_000, "positions_value": 110_000, "total_assets": 1_010_000, "daily_profit": 10_000, "open_positions_count": 1},
        ]
    ).to_csv(base / "summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "action": "BUY",
                "signal_date": "2026-01-04",
                "entry_date": "2026-01-05",
                "exit_date": "",
                "code": "1111",
                "entry_price": 100,
                "shares": 1000,
                "net_profit": "",
                "final_amount": 100_000,
            },
            {
                "action": "SELL",
                "signal_date": "2026-01-04",
                "entry_date": "2026-01-05",
                "exit_date": "2026-01-06",
                "code": "1111",
                "entry_price": 100,
                "exit_price": 110,
                "shares": 1000,
                "net_profit": profit,
                "final_amount": 100_000,
            },
        ]
    ).to_csv(base / "trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "decision": "BUY",
                "signal_date": "2026-01-04",
                "entry_date": "2026-01-05",
                "code": "1111",
                "final_amount": 100_000,
                "pm_per_code_cap_reduced": True,
                "pm_per_code_cap_reason": "per_code_exposure_cap",
            },
            {
                "decision": "SKIP",
                "signal_date": "2026-01-05",
                "entry_date": "2026-01-06",
                "code": "2222",
                "skip_reason": "pm_low_score_skip",
            },
        ]
    ).to_csv(base / "purchase_audit.csv", index=False)


def _make_audit(tmp_path: Path) -> PortfolioManagerPhase3HCapitalUtilizationAudit:
    profiles = {
        "baseline": "v2_73",
        "phase3d": "v2_75",
        "phase3e": "v2_76",
        "phase3g": "v2_77_020",
        "phase3h": "v2_77_030",
    }
    for profile in profiles.values():
        _write_profile(tmp_path, profile)
    return PortfolioManagerPhase3HCapitalUtilizationAudit(
        root=tmp_path,
        baseline_profile=profiles["baseline"],
        phase3d_profile=profiles["phase3d"],
        phase3e_profile=profiles["phase3e"],
        phase3g_profile=profiles["phase3g"],
        phase3h_profile=profiles["phase3h"],
    )


def test_phase3h_builds_daily_monthly_and_low_utilization_sections(tmp_path: Path) -> None:
    audit = _make_audit(tmp_path)

    result = audit.build()

    distribution = result["capital_utilization_distribution"]["v2_77_cap_030"]
    assert distribution["days_below_50pct"] == 2
    assert distribution["average_cash"] > 0
    assert result["monthly_capital_utilization"]["v2_77_cap_030"][0]["pm_low_score_skip_count"] == 1
    assert result["target_low_utilization_days"][0]["dominant_reason"] in {"unknown", "exit_only_day", "no_candidates"}
    assert result["target_low_utilization_days"][1]["dominant_reason"] == "candidates_all_low_pm_skipped"
    assert result["target_skip_reason_utilization"][0]["skip_reason"] == "pm_low_score_skip"
    assert any(row["flag"] == "low_utilization_due_to_pm_skip" for row in result["bottleneck_flags"])
    assert result["constraints"]["selected_count_in_day_used"] is False
    assert result["constraints"]["api_refetch"] is False


def test_phase3h_saves_markdown_and_json(tmp_path: Path) -> None:
    audit = _make_audit(tmp_path)
    result = audit.build()
    paths = audit.save(result)

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 3-H" in paths.markdown.read_text(encoding="utf-8")
