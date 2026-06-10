from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase11e_exit_dd_guard import Phase11EExitDDGuard, Phase11EOptions


def _write_fixture(root: Path) -> None:
    simulation_path = root / "data/ml/valuation_engine/phase11c_allocation_simulation_2025.parquet"
    dataset_path = root / "data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet"
    phase11d_report = root / "reports/ml/phase11d_combined_backtest_2025.json"
    simulation_path.parent.mkdir(parents=True, exist_ok=True)
    phase11d_report.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2025-01-07", periods=28)
    simulation_rows = []
    dataset_rows = []
    for day_index, date in enumerate(dates):
        for rank in range(8):
            code = f"{10000 + rank}"
            entry_price = 100 + rank * 5
            close = entry_price
            if rank == 0 and day_index >= 3:
                close = 90.0
            if rank == 1 and day_index >= 4:
                close = 112.0
            proba = 0.95 - rank * 0.05
            if rank == 1 and day_index >= 2:
                proba = 0.40
            simulation_rows.append(
                {
                    "rule": "equal_weight_top5",
                    "date": date,
                    "code": code,
                    "opportunity_top_decile_proba": proba,
                    "opportunity_score_proba_rank": proba,
                    "future_return_20d": 0.05 - rank * 0.01,
                    "future_max_return_20d": 0.12 - rank * 0.01,
                    "future_max_drawdown_20d": -0.08,
                    "opportunity_value_20d": 0.04 - rank * 0.005,
                    "opportunity_top_decile_20d": 1 if rank < 2 else 0,
                }
            )
            dataset_rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": close,
                    "turnover_value": 1_000_000 + rank,
                }
            )

    pd.DataFrame(simulation_rows).to_parquet(simulation_path, index=False)
    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)
    phase11d_report.write_text(
        json.dumps(
            {
                "strategy_results": [
                    {
                        "strategy": "candidate_valuation_top5",
                        "net_profit": 100_000,
                        "PF": 1.6,
                        "DD": -0.16,
                        "win_rate": 0.6,
                        "total_trades": 10,
                        "final_assets": 1_100_000,
                        "capital_utilization": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_phase11e_runs_variants_and_keeps_leakage_low(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    runner = Phase11EExitDDGuard(tmp_path, options=Phase11EOptions(daily_buy_budget=900_000, max_positions=5))
    report = runner.build_report()

    assert report["metadata"]["limited_2025_only"] is True
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["metadata"]["stop_loss_uses_future_low"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert {row["variant"] for row in report["variant_results"]} >= {
        "E0_no_guard",
        "E1_stop_loss_8pct",
        "E2_stop_loss_5pct",
        "E3_opportunity_disappeared",
        "E4_stop_loss_8pct_plus_opportunity",
    }
    assert any(row.get("stop_loss", 0) > 0 for row in report["exit_reason_counts"])
    assert any(row.get("opportunity_rank_below_floor", 0) > 0 for row in report["exit_reason_counts"])


def test_phase11e_saves_reports(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    paths = Phase11EExitDDGuard(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "11-E"
    assert loaded["leakage_checklist"]["future_columns_used_as_features"] == []
    assert loaded["skipped_variants"] == []
