from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase12c2_utilization_without_dd_explosion import Phase12C2UtilizationWithoutDDExplosion


EXPECTED_VARIANTS = {
    "C2_base_dynamic_normalized_B5_2_exit",
    "C2a_normalized_cap_20pct",
    "C2b_normalized_cap_15pct",
    "C2c_normalized_downside_penalty_squared",
    "C2d_normalized_top_weight_cap_30pct",
    "C2e_normalized_cash_reserve_80pct",
}


def _write_artifact(root: Path) -> None:
    path = root / ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for day_index, date in enumerate(pd.bdate_range("2025-01-07", periods=50)):
        for rank in range(8):
            quality = (7 - rank) / 7
            close = 100.0 + rank * 5
            if rank == 0:
                close = 100.0 + min(day_index, 8) * 3
                if day_index > 8:
                    close = 124.0 - (day_index - 8) * 1.5
            if rank == 1:
                close = 120.0 - max(day_index - 5, 0) * 1.4
            downside_rank = 0.30 if rank == 0 else 0.60 if rank == 1 else 0.78 if rank == 2 else 0.90
            rows.append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "close": close,
                    "turnover_value": 1_000_000 + rank,
                    "opportunity_proba": 0.08 + quality * 0.50,
                    "downside_bad_proba": 0.45 if rank == 1 else 0.18,
                    "opportunity_rank_percentile": (rank + 1) / 8,
                    "downside_rank_percentile": downside_rank,
                    "confidence": 0.6,
                    "future_return_20d": 0.05 - rank * 0.005,
                    "future_max_return_20d": 0.12 - rank * 0.006,
                    "future_max_drawdown_20d": -0.12 if rank == 1 else -0.04,
                    "opportunity_value_20d": 0.06 - rank * 0.004,
                    "opportunity_top_decile_20d": 1 if rank < 2 else 0,
                    "downside_bad_20d": 1 if rank == 1 else 0,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase12c2_builds_dd_audit_report(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    report = Phase12C2UtilizationWithoutDDExplosion(tmp_path).build_report()

    assert report["metadata"]["phase"] == "12-C2"
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert {row["strategy"] for row in report["variant_results"]} == EXPECTED_VARIANTS
    assert "top20_loss_trades" in report["dd_attribution_audit"]
    assert "largest_position_weight_max" in report["concentration_audit"]
    assert "by_downside_bucket" in report["downside_exposure_audit"]
    assert "main_dd_cause" in report["recommendation"]


def test_phase12c2_saves_reports(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    paths = Phase12C2UtilizationWithoutDDExplosion(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "12-C2"
    assert loaded["leakage_checklist"]["existing_model_overwritten"] is False
