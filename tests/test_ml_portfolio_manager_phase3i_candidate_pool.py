from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase3i_candidate_pool import PortfolioManagerPhase3ICandidatePoolAudit


PERIOD = "2023-01-01_to_2026-05-31"


def _profile_yaml(root: Path, profile: str, max_selected: int) -> None:
    path = root / "config" / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{profile}.yaml").write_text(
        "\n".join(
            [
                f"profile_id: {profile}",
                "selection:",
                f"  max_selected: {max_selected}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_profile(root: Path, profile: str, *, profit: float, util_value: float, no_candidate_day: bool = False) -> None:
    base = root / "logs" / "backtests" / profile / PERIOD
    base.mkdir(parents=True)
    (base / "backtest_summary.json").write_text(
        json.dumps(
            {
                "net_cumulative_profit": profit,
                "profit_factor": 2.4,
                "max_drawdown": -0.08,
                "win_rate": 0.55,
                "closed_trades_count": 1,
                "daily_asset_curve": [],
                "all_trades": [],
            }
        ),
        encoding="utf-8",
    )
    final_amount = 1_000_000 * util_value
    pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "cash": 1_000_000 * (1 - util_value),
                "positions_value": 1_000_000 * util_value,
                "total_assets": 1_000_000,
                "daily_profit": 0,
                "open_positions_count": 1,
            },
            {
                "date": "2026-01-06",
                "cash": 900_000,
                "positions_value": 100_000,
                "total_assets": 1_000_000,
                "daily_profit": 0,
                "open_positions_count": 1,
            },
        ]
    ).to_csv(base / "summary.csv", index=False)
    pd.DataFrame(
        [
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
                "final_amount": final_amount,
            }
        ]
    ).to_csv(base / "trades.csv", index=False)
    audit_rows = [] if no_candidate_day else [
        {
            "decision": "SKIP",
            "signal_date": "2026-01-05",
            "entry_date": "2026-01-06",
            "code": "2222",
            "skip_reason": "pm_low_score_skip",
        }
    ]
    pd.DataFrame(audit_rows).to_csv(base / "purchase_audit.csv", index=False)


def test_phase3i_compares_candidate_pool_variants(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    profiles = {
        "current": ("current_profile", 10, 1_000_000.0, 0.50),
        "x2": ("pool_x2", 20, 1_200_000.0, 0.62),
        "x3": ("pool_x3", 30, 900_000.0, 0.70),
    }
    for _, (profile, max_selected, profit, util) in profiles.items():
        _profile_yaml(tmp_path, profile, max_selected)
        _write_profile(tmp_path, profile, profit=profit, util_value=util)

    audit = PortfolioManagerPhase3ICandidatePoolAudit(
        root=tmp_path,
        current_profile="current_profile",
        pool_x2_profile="pool_x2",
        pool_x3_profile="pool_x3",
    )
    result = audit.build()
    paths = audit.save(result)

    assert result["candidate_pool_settings"]["current"]["max_selected"] == 10
    assert result["candidate_pool_settings"]["candidate_pool_x2"]["max_selected"] == 20
    assert result["comparison"][1]["capital_utilization_delta_vs_current"] > 0
    assert result["constraints"]["selected_count_in_day_used"] is False
    assert paths.markdown.exists()
    assert paths.json.exists()
