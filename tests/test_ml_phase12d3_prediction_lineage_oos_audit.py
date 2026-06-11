from __future__ import annotations

import json
from pathlib import Path

from ml.phase12d3_prediction_lineage_oos_audit import PHASE12_REPORT_STEMS, Phase12D3PredictionLineageOOSAudit


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_minimal_fixture(root: Path) -> None:
    split = {
        "train": {"start": "2023-01-04", "end": "2023-12-31"},
        "validation": {"start": "2024-01-01", "end": "2024-12-31"},
        "test": {"start": "2025-01-01", "end": "2025-12-31"},
        "strict_model_oos": True,
        "train_validation_test_overlap": False,
    }
    _write_json(
        root / "models/ml/valuation_engine/research_phase11b3_downside/model_metadata.json",
        {"phase": "11-B3", "strict_model_oos": True, "split": split, "existing_model_overwritten": False},
    )
    _write_json(
        root / "models/ml/valuation_engine/research_phase11i_strict_oos/model_metadata.json",
        {"phase": "11-I", "strict_model_oos": True, "split": split, "existing_model_overwritten": False},
    )
    _write_json(
        root / "models/ml/valuation_engine/candidate_phase11b/model_metadata.json",
        {"phase": "11-B", "train_period": {"start": "2023-01-04", "end": "2024-12-31"}, "test_period": {"start": "2025-01-01", "end": "2025-12-31"}},
    )
    fold_rows = [
        {
            "month": f"2025-{month:02d}",
            "effective_train_end_before_test_start": True,
        }
        for month in range(1, 13)
    ]
    _write_json(
        root / "reports/ml/walk_forward_model_audit_5y_enriched_v2.json",
        {
            "summary": {
                "prediction_root_is_walk_forward": True,
                "fold_specific_model_used_by_code_path": True,
                "current_model_used_by_walk_forward_code_path": False,
                "metadata_missing_files": 12,
            },
            "fold_rows": fold_rows,
        },
    )
    _write_json(root / "reports/ml/phase11i_strict_walk_forward_oos_2025.json", {"metadata": {"phase": "11-I"}, "conditions": {"strict_model_oos": True}, "leakage_checklist": {"blocking_issues": []}})
    _write_json(root / "reports/ml/phase11b3_expected_downside_model_2025.json", {"metadata": {"phase": "11-B3"}, "conditions": {"strict_model_oos": True}, "leakage_checklist": {"blocking_issues": []}})
    for stem in PHASE12_REPORT_STEMS:
        _write_json(
            root / f"reports/ml/{stem}.json",
            {
                "metadata": {"phase": "12-X", "new_model_trained": False, "historical_predictions_regenerated": False, "existing_model_overwritten": False, "profile_changed": False, "full_backtest_executed": False},
                "conditions": {"period": {"start": "2025-01-01", "end": "2025-12-31"}},
                "dataset_summary": {"date_range": {"min": "2025-01-07", "max": "2025-12-29"}},
                "leakage_checklist": {"blocking_issues": [], "new_model_trained": False, "historical_predictions_regenerated": False, "existing_model_overwritten": False, "profile_changed": False, "full_backtest_executed": False},
                "dummy": "stock_selection opportunity_proba downside_bad_proba",
            },
        )
    source = root / "src/ml/phase12a_dynamic_capital_allocation.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("PHASE11B3_MODEL_DIR = 'models/ml/valuation_engine/research_phase11b3_downside'", encoding="utf-8")


def test_phase12d3_trusts_strict_oos_fixture(tmp_path: Path) -> None:
    _write_minimal_fixture(tmp_path)

    report = Phase12D3PredictionLineageOOSAudit(tmp_path).build_report()

    assert report["metadata"]["phase"] == "12-D3"
    assert report["integrity_checklist"]["new_model_trained"] is False
    assert report["integrity_checklist"]["stock_selection_lineage_unknown"] is False
    assert report["integrity_checklist"]["valuation_lineage_unknown"] is False
    assert report["integrity_checklist"]["downside_lineage_unknown"] is False
    assert report["final_trust_decision"]["stock_selection_strict_oos_for_2025"] is True
    assert report["final_trust_decision"]["valuation_strict_oos_for_2025"] is True
    assert report["final_trust_decision"]["downside_strict_oos_for_2025"] is True
    assert report["final_trust_decision"]["phase12_results_trustworthy"] is True


def test_phase12d3_saves_reports(tmp_path: Path) -> None:
    _write_minimal_fixture(tmp_path)

    paths = Phase12D3PredictionLineageOOSAudit(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "12-D3"
    assert loaded["integrity_checklist"]["existing_model_overwritten"] is False
