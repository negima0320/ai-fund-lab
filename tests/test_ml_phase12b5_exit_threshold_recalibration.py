from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase12b5_exit_threshold_recalibration import Phase12B5ExitThresholdRecalibration


EXPECTED_VARIANTS = {
    "B5_0_baseline",
    "B5_1_rank_floor_lower",
    "B5_2_proba_drop_larger",
    "B5_3_both_relaxed",
    "B5_4_confirmation_2days",
    "B5_5_confirmation_3days",
    "B5_6_profit_protected_exit",
    "B5_7_loss_only_hard_exit",
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


def test_phase12b5_builds_recalibration_report(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    report = Phase12B5ExitThresholdRecalibration(tmp_path).build_report()

    assert report["metadata"]["phase"] == "12-B5"
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert {row["variant"] for row in report["variant_results"]} == EXPECTED_VARIANTS
    assert {row["variant"] for row in report["hold_exit_quality"]} == EXPECTED_VARIANTS
    assert "best_variant" in report["variant_comparison"]


def test_phase12b5_saves_reports(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    paths = Phase12B5ExitThresholdRecalibration(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "12-B5"
    assert loaded["leakage_checklist"]["existing_model_overwritten"] is False
