from __future__ import annotations

import json
from pathlib import Path

from ml.phase12e2_stock_selection_architecture_audit import Phase12E2StockSelectionArchitectureAudit


def _write_evidence(root: Path) -> None:
    model_dir = root / "models/ml/walk_forward/current/2025-01"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "feature_columns.json").write_text(
        json.dumps(
            [
                "close",
                "return_5d",
                "ma25_gap",
                "volume_ratio_5d",
                "body_ratio",
                "EPS",
                "topix_return_5d",
                "sector_name",
            ]
        ),
        encoding="utf-8",
    )
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "future_10d_return_regression": {"rmse": 0.1, "mae": 0.05},
                "bad_entry_10d_classification": {"auc": 0.6, "precision": 0.2},
            }
        ),
        encoding="utf-8",
    )
    report_dir = root / "reports/ml"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase12d3_prediction_lineage_oos_audit.json").write_text(
        json.dumps({"final_trust_decision": {"stock_selection_strict_oos_for_2025": True}}),
        encoding="utf-8",
    )
    (report_dir / "walk_forward_model_audit_5y_enriched_v2.json").write_text(
        json.dumps(
            {
                "fold_rows": [
                    {
                        "month": "2025-01",
                        "train_start": "2023-01-04",
                        "effective_train_end": "2024-12-02",
                        "test_start": "2025-01-01",
                        "test_end": "2025-01-31",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "phase12e1_stock_selection_reality_audit_2025.json").write_text(
        json.dumps(
            {
                "final_judgment": {
                    "stock_selection_adds_value": False,
                    "stock_selection_top5_valid": False,
                    "stock_selection_prefilter_hurts_valuation": True,
                },
                "rank_quality_table": [
                    {"selection": "candidate_universe", "opportunity_top_decile_20d_rate": 0.10},
                    {"selection": "stock_selection_rank_score_top5", "opportunity_top_decile_20d_rate": 0.08},
                    {"selection": "candidate_strength_top5", "opportunity_top_decile_20d_rate": 0.20},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_phase12e2_builds_architecture_audit(tmp_path: Path) -> None:
    _write_evidence(tmp_path)

    report = Phase12E2StockSelectionArchitectureAudit(tmp_path).build_report()

    assert report["metadata"]["phase"] == "12-E2"
    assert report["metadata"]["new_model_trained"] is False
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["stock_selection_architecture_summary"]["model_family"].startswith("LightGBM")
    assert report["feature_category_summary"]["Price"]["count"] >= 1
    assert report["feature_category_summary"]["Financial"]["count"] >= 1
    assert report["label_summary"]["feature_exclusion_columns"]
    assert any(row["column"] == "stock_selection_rank_score" for row in report["output_column_meaning"])
    assert all(row["strict_oos_for_2025"] for row in report["model_lineage_summary"])
    assert report["suspected_failure_reason"]["stock_selection_ai_current_objective_misaligned_with_phase12"] is True
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert report["leakage_checklist"]["new_model_trained"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["recommendation"]["recommended_next_phase"] == "Phase12-E3 Remove Stock Selection Prefilter Test"


def test_phase12e2_saves_reports(tmp_path: Path) -> None:
    _write_evidence(tmp_path)

    paths = Phase12E2StockSelectionArchitectureAudit(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "12-E2"
    assert loaded["leakage_checklist"]["existing_model_overwritten"] is False
