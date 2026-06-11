from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase12e1_stock_selection_reality_audit import Phase12E1StockSelectionRealityAudit


EXPECTED_STRATEGIES = {
    "S0_stock_selection_rank_score_top5_equal_20d",
    "S1_risk_adjusted_score_top5_equal_20d",
    "S2_expected_return_top5_equal_20d",
    "S3_candidate_strength_top5_equal_20d",
    "S4_opportunity_top5_equal_20d",
    "S5_stock_top5_prefilter_opportunity_downside_dynamic_B5_2_exit",
    "S6_no_stock_prefilter_opportunity_downside_dynamic_B5_2_exit",
    "S7_stock_top20_prefilter_opportunity_downside_dynamic_B5_2_exit",
}


def _write_artifact(root: Path) -> None:
    path = root / ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    dates = pd.bdate_range("2025-01-07", periods=45)
    for day_index, date in enumerate(dates):
        for rank in range(12):
            stock_quality = (11 - rank) / 11
            opportunity_quality = 1.0 if rank in {2, 3} else stock_quality * 0.6
            close = 100.0 + rank * 4
            if rank == 2:
                close = 100.0 + min(day_index, 12) * 2
            downside_rank = 0.25 if rank == 2 else 0.95 if rank == 3 else 0.60
            downside_proba = 0.12 if rank == 2 else 0.60 if rank == 3 else 0.25
            rows.append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "close": close,
                    "turnover_value": 1_000_000 + rank,
                    "stock_selection_rank_score": stock_quality,
                    "risk_adjusted_score": stock_quality * 0.9,
                    "expected_return": stock_quality / 10,
                    "candidate_strength": stock_quality,
                    "opportunity_proba": 0.05 + opportunity_quality * 0.60,
                    "downside_bad_proba": downside_proba,
                    "opportunity_rank_percentile": opportunity_quality,
                    "downside_rank_percentile": downside_rank,
                    "confidence": 0.6,
                    "future_return_20d": 0.08 if rank == 2 else -0.02 if rank == 3 else 0.02 - rank * 0.001,
                    "future_max_return_20d": 0.16 if rank == 2 else 0.03,
                    "future_max_drawdown_20d": -0.03 if rank == 2 else -0.14 if rank == 3 else -0.06,
                    "opportunity_value_20d": 0.10 if rank == 2 else -0.03 if rank == 3 else 0.01,
                    "opportunity_top_decile_20d": 1 if rank == 2 else 0,
                    "downside_bad_20d": 1 if rank == 3 else 0,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase12e1_builds_stock_selection_reality_audit(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    report = Phase12E1StockSelectionRealityAudit(tmp_path).build_report()

    assert report["metadata"]["phase"] == "12-E1"
    assert report["metadata"]["new_model_trained"] is False
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert report["leakage_checklist"]["new_model_trained"] is False
    assert {row["strategy"] for row in report["strategy_comparison_table"]} == EXPECTED_STRATEGIES
    assert any(row["selection"] == "candidate_universe" for row in report["rank_quality_table"])
    assert any(row["score"] == "stock_selection_rank_score" for row in report["score_monotonicity_table"])
    assert any(row["layer"] == "F_universe_then_opportunity_downside_top5" for row in report["layer_interaction_summary"])
    assert "stock_selection_adds_value" in report["final_judgment"]
    assert "recommended_next_phase" in report["recommendation"]


def test_phase12e1_saves_reports(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    paths = Phase12E1StockSelectionRealityAudit(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "12-E1"
    assert loaded["leakage_checklist"]["existing_model_overwritten"] is False
