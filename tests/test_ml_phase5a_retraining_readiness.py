from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase5a_retraining_readiness import Phase5ARetrainingReadinessAudit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_model(root: Path, path: str, feature_columns: list[str], targets: dict[str, dict]) -> None:
    model_dir = root / path
    _write_json(
        model_dir / "model_metadata.json",
        {
            "model_profile": model_dir.name,
            "train_start": "2023-01-01",
            "train_end": "2025-12-31",
            "valid_start": "2026-01-01",
            "valid_end": "2026-05-31",
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "targets": targets,
            "leakage_guard": "fixture chronological split",
        },
    )
    _write_json(model_dir / "metrics.json", {name: {"auc": 0.6} for name in targets})
    (model_dir / "feature_columns.json").write_text(json.dumps(feature_columns), encoding="utf-8")


def _write_fixture_tree(root: Path, *, pm_forbidden: bool = False) -> None:
    _write_model(
        root,
        "models/ml/current_enriched_v2",
        ["return_5d", "bad_entry_probability_10d"],
        {"bad_entry_10d_classification": {"target": "bad_entry_10d", "task": "classification"}},
    )
    _write_model(
        root,
        "models/ml/exit/current_v2_66",
        ["holding_days", "unrealized_return", "risk_adjusted_score"],
        {"avoid_loss_5d_classification": {"target": "avoid_loss_5d", "task": "classification"}},
    )
    pm_features = ["risk_adjusted_score", "candidate_count_in_day"]
    if pm_forbidden:
        pm_features.append("selected_count_in_day")
    _write_model(
        root,
        "models/ml/portfolio_manager/current_v2_73_phase3b_clean",
        pm_features,
        {
            "high_conviction_target_classification": {"target": "high_conviction_target", "task": "classification"},
            "avoid_target_classification": {"target": "avoid_target", "task": "classification"},
        },
    )

    (root / "data/ml/walk_forward_predictions").mkdir(parents=True, exist_ok=True)
    (root / "data/ml/features").mkdir(parents=True, exist_ok=True)
    (root / "data/ml/labels").mkdir(parents=True, exist_ok=True)
    (root / "data/ml/datasets").mkdir(parents=True, exist_ok=True)
    (root / "data/ml/exit_datasets").mkdir(parents=True, exist_ok=True)
    (root / "data/ml/portfolio_manager").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"code": "11110"}]).to_parquet(root / "data/ml/walk_forward_predictions/predictions_2023-01-04.parquet")
    pd.DataFrame([{"code": "11110"}]).to_parquet(root / "data/ml/features/features_2023-01-04.parquet")
    pd.DataFrame([{"code": "11110"}]).to_parquet(root / "data/ml/labels/labels_2023-01-04.parquet")
    pd.DataFrame([{"date": "2023-01-04", "code": "11110", "bad_entry_10d": False}]).to_parquet(root / "data/ml/datasets/ml_dataset.parquet")
    pd.DataFrame(
        [
            {
                "trade_id": "t",
                "code": "11110",
                "current_date": "2023-01-05",
                "future_remaining_return_5d": 0.01,
                "future_remaining_return_10d": 0.02,
                "hold_better_5d": True,
                "should_exit_now_5d": False,
                "avoid_loss_5d": False,
            }
        ]
    ).to_parquet(root / "data/ml/exit_datasets/exit_dataset_v2_66_2023-01_to_2026-05.parquet")
    pd.DataFrame(
        [
            {
                "signal_date": "2023-01-04",
                "code": "11110",
                "high_conviction_target": True,
                "avoid_target": False,
                "realized_return": 0.03,
            }
        ]
    ).to_parquet(root / "data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet")


def test_phase5a_builds_inventory_and_prioritizes_exit_ai(tmp_path: Path) -> None:
    _write_fixture_tree(tmp_path)

    result = Phase5ARetrainingReadinessAudit(tmp_path).build_report()

    assert result["metadata"]["audit_only"] is True
    assert len(result["model_inventory"]) == 3
    assert result["exit_ai_readiness"]["required_data_available"] is True
    assert result["exit_ai_readiness"]["recommended_exit_label"].startswith("API-only exit_quality_score")
    assert result["retraining_priority"]["recommended_order"][0] == "Exit AI retraining"
    assert result["recommended_next_phase"] == "Phase 5-B Exit AI v2 API-Only Dataset Design"
    assert result["metadata"]["api_only_retraining_data_policy"] is True
    assert result["exit_ai_readiness"]["existing_dataset_retraining_safe"] is False


def test_phase5a_leakage_audit_blocks_forbidden_pm_feature(tmp_path: Path) -> None:
    _write_fixture_tree(tmp_path, pm_forbidden=True)

    result = Phase5ARetrainingReadinessAudit(tmp_path).build_report()

    assert result["leakage_audit"]["blocking_issues"]
    assert result["retraining_priority"]["recommended_order"][0] == "Market Regime Audit before retraining"


def test_phase5a_saves_report(tmp_path: Path) -> None:
    _write_fixture_tree(tmp_path)

    audit = Phase5ARetrainingReadinessAudit(tmp_path)
    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 5-A" in paths.markdown.read_text(encoding="utf-8")
