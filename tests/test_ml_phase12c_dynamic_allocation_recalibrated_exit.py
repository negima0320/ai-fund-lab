from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase12c_dynamic_allocation_recalibrated_exit import Phase12CDynamicAllocationRecalibratedExit


EXPECTED_STRATEGIES = {
    "C0_baseline_equal_allocation",
    "C1_dynamic_raw_B5_2_exit",
    "C2_dynamic_normalized_B5_2_exit",
    "C3_partial_normalized_30_B5_2_exit",
    "C4_partial_normalized_50_B5_2_exit",
}


def _write_artifact(root: Path) -> None:
    path = root / ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for day_index, date in enumerate(pd.bdate_range("2025-01-07", periods=45)):
        for rank in range(8):
            quality = (7 - rank) / 7
            close = 100 + rank * 4
            if rank == 0:
                close = 100 + min(day_index, 10) * 2
                if day_index > 10:
                    close = 120 - (day_index - 10)
            if rank == 1 and day_index >= 5:
                close = 90
            rows.append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "close": float(close),
                    "turnover_value": 1_000_000 + rank,
                    "stock_selection_rank_score": quality,
                    "risk_adjusted_score": quality,
                    "expected_return": quality / 10,
                    "candidate_strength": quality,
                    "opportunity_proba": 0.05 + quality * 0.5,
                    "downside_bad_proba": 0.40 if rank < 2 else 0.10,
                    "opportunity_rank_percentile": (rank + 1) / 8,
                    "downside_rank_percentile": 0.95 if rank == 1 else 0.30 + rank / 10,
                    "confidence": 0.5,
                    "future_return_20d": 0.05 - rank * 0.004,
                    "future_max_return_20d": 0.12 - rank * 0.006,
                    "future_max_drawdown_20d": -0.12 if rank == 1 else -0.04,
                    "opportunity_value_20d": 0.06 - rank * 0.004,
                    "opportunity_top_decile_20d": 1 if rank < 2 else 0,
                    "downside_bad_20d": 1 if rank == 1 else 0,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase12c_builds_integration_report(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    report = Phase12CDynamicAllocationRecalibratedExit(tmp_path).build_report()

    assert report["metadata"]["phase"] == "12-C"
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert {row["strategy"] for row in report["strategy_results"]} == EXPECTED_STRATEGIES
    assert "best_strategy" in report["strategy_comparison"]
    assert "ready_for_phase13" in report["recommendation"]


def test_phase12c_saves_reports(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    paths = Phase12CDynamicAllocationRecalibratedExit(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "12-C"
    assert loaded["leakage_checklist"]["existing_model_overwritten"] is False
