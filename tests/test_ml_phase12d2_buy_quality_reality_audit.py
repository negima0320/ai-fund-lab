from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase12d2_buy_quality_reality_audit import Phase12D2BuyQualityRealityAudit


EXPECTED_LAYERS = {
    "L0_candidate_universe",
    "L1_stock_selection_top5",
    "L2_opportunity_top5",
    "L3_opportunity_downside_top5_A3_3",
    "L4_dynamic_normalized_C2c_candidates",
}

EXPECTED_STRATEGIES = {
    "S0_stock_selection_only",
    "S1_stock_selection_plus_B5_2_exit",
    "S2_opportunity_only",
    "S3_opportunity_plus_B5_2_exit",
    "S4_opportunity_downside_dynamic_allocation_B5_2_exit",
}


def _write_artifact(root: Path) -> None:
    path = root / ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for day_index, date in enumerate(pd.bdate_range("2025-01-07", periods=45)):
        for rank in range(8):
            quality = (7 - rank) / 7
            close = 100.0 + rank * 5
            if rank == 0:
                close = 100.0 + min(day_index, 8) * 2
                if day_index > 12:
                    close = 116.0 - (day_index - 12) * 0.8
            downside_rank = 0.30 if rank == 0 else 0.60 if rank == 1 else 0.78 if rank == 2 else 0.90
            rows.append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "close": close,
                    "turnover_value": 1_000_000 + rank,
                    "stock_selection_rank_score": quality,
                    "risk_adjusted_score": quality,
                    "expected_return": quality / 10,
                    "candidate_strength": quality,
                    "opportunity_proba": 0.08 + quality * 0.50,
                    "downside_bad_proba": 0.42 if rank == 1 else 0.16,
                    "opportunity_rank_percentile": quality,
                    "downside_rank_percentile": downside_rank,
                    "confidence": 0.6,
                    "future_return_20d": 0.05 - rank * 0.004,
                    "future_max_return_20d": 0.12 - rank * 0.006,
                    "future_max_drawdown_20d": -0.12 if rank == 1 else -0.04,
                    "opportunity_value_20d": 0.06 - rank * 0.004,
                    "opportunity_top_decile_20d": 1 if rank < 2 else 0,
                    "downside_bad_20d": 1 if rank == 1 else 0,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase12d2_builds_reality_audit(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    report = Phase12D2BuyQualityRealityAudit(tmp_path).build_report()

    assert report["metadata"]["phase"] == "12-D2"
    assert report["metadata"]["new_model_trained"] is False
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["new_model_trained"] is False
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert {row["layer"] for row in report["buy_quality_layer_comparison"]} == EXPECTED_LAYERS
    assert {row["strategy"] for row in report["strategy_layer_comparison"]} == EXPECTED_STRATEGIES
    assert "stock_selection_effect" in report["layer_contribution_summary"]
    assert "buy_quality_good_enough" in report["judgments"]
    assert "recommended_next_phase" in report["recommendation"]


def test_phase12d2_saves_reports(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    paths = Phase12D2BuyQualityRealityAudit(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "12-D2"
    assert loaded["leakage_checklist"]["existing_model_overwritten"] is False
