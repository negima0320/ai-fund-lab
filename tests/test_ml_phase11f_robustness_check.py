from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase11f_robustness_check import Phase11FRobustnessCheck


def _write_fixture(root: Path) -> None:
    simulation_path = root / "data/ml/valuation_engine/phase11c_allocation_simulation_2025.parquet"
    dataset_path = root / "data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet"
    simulation_path.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2025-01-07", periods=35)
    simulation_rows = []
    dataset_rows = []
    for day_index, date in enumerate(dates):
        for rank in range(8):
            code = f"{10000 + rank}"
            entry_price = 100 + rank * 4
            close = entry_price
            if rank == 0 and day_index >= 5:
                close = 88.0
            if rank == 1 and day_index >= 4:
                close = 114.0
            proba = 0.95 - rank * 0.05
            if rank == 1 and day_index >= 3:
                proba = 0.42
            simulation_rows.append(
                {
                    "rule": "equal_weight_top5",
                    "date": date,
                    "code": code,
                    "opportunity_top_decile_proba": proba,
                    "opportunity_score_proba_rank": proba,
                    "future_return_20d": 0.04 - rank * 0.005,
                    "future_max_return_20d": 0.12 - rank * 0.01,
                    "future_max_drawdown_20d": -0.08,
                    "opportunity_value_20d": 0.03 - rank * 0.004,
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


def test_phase11f_builds_robustness_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    report = Phase11FRobustnessCheck(tmp_path).build_report()

    assert report["metadata"]["limited_2025_only"] is True
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert [row["cost_rate"] for row in report["cost_sensitivity"]] == [0.0, 0.001, 0.002, 0.003]
    assert {row["threshold_profile"] for row in report["threshold_sensitivity"]} == {"loose", "baseline", "strict"}
    assert "same_code_reentry_count" in report["overtrading_check"]
    assert "cost_02_pf_at_least_1_8" in report["combined_robustness_summary"]


def test_phase11f_saves_reports(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    paths = Phase11FRobustnessCheck(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "11-F"
    assert loaded["leakage_checklist"]["future_columns_used_as_features"] == []
