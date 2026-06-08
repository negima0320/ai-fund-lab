from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase7b_pm_ai_leakage_forensics import Phase7BPMLeakageForensics, classify_pm_column


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_fixture(root: Path, *, leaked_feature: bool = False) -> None:
    model_dir = root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean"
    dataset_path = root / "data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet"
    features = ["return_5d", "risk_adjusted_score", "topix_return_5d"]
    if leaked_feature:
        features.append("cash_before")
    _write_json(model_dir / "feature_columns.json", features)
    _write_json(
        model_dir / "model_metadata.json",
        {
            "model_profile": "portfolio_manager_v2_73_phase3b_clean",
            "feature_count": len(features),
            "feature_columns": features,
            "targets": {
                "high_conviction_target_classification": {"target": "high_conviction_target", "task": "binary"},
                "realized_return_regression": {"target": "realized_return", "task": "regression"},
            },
            "leakage_guard": "label/audit/result columns are excluded from feature_columns",
        },
    )
    _write_json(model_dir / "metrics.json", {"high_conviction_target_classification": {"auc": 0.6}})
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "signal_date": "2023-01-04",
                "code": "11110",
                "return_5d": 0.01,
                "risk_adjusted_score": 0.2,
                "topix_return_5d": 0.01,
                "future_10d_return": 0.05,
                "realized_return": 0.05,
                "high_conviction_target": True,
                "actual_net_profit": 5000,
                "cash_before": 1_000_000,
            },
            {
                "signal_date": "2026-05-28",
                "code": "22220",
                "return_5d": -0.01,
                "risk_adjusted_score": 0.1,
                "topix_return_5d": -0.01,
                "future_10d_return": -0.03,
                "realized_return": -0.03,
                "high_conviction_target": False,
                "actual_net_profit": -1000,
                "cash_before": 800_000,
            },
        ]
    ).to_parquet(dataset_path)


def test_classify_pm_column_groups_backtest_and_model_prediction_columns() -> None:
    assert classify_pm_column("actual_net_profit") == "backtest_trade_outcome"
    assert classify_pm_column("cash_before") == "backtest_cash_state"
    assert classify_pm_column("risk_adjusted_score") == "model_prediction_feature"
    assert classify_pm_column("topix_return_5d") == "api_market_feature"


def test_phase7b_distinguishes_dataset_columns_from_training_features(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = Phase7BPMLeakageForensics(tmp_path).build_report()

    assert "actual_net_profit" in result["dataset_column_audit"]["forbidden_columns"]
    assert result["training_feature_audit"]["forbidden_columns_used_as_features"] == []
    assert result["target_label_audit"]["target_labels_forbidden_for_retraining"] == []
    assert result["v2_82_safety_impact_assessment"]["v282_result_trust_level"] == "medium_trust"


def test_phase7b_flags_forbidden_training_feature(tmp_path: Path) -> None:
    _write_fixture(tmp_path, leaked_feature=True)

    result = Phase7BPMLeakageForensics(tmp_path).build_report()

    assert result["training_feature_audit"]["forbidden_columns_used_as_features"] == ["cash_before"]
    assert result["v2_82_safety_impact_assessment"]["v282_result_trust_level"] == "low_trust"
    assert result["retraining_judgement"]["next_phase_recommended"] == "Phase 7-C PM AI Feature Leakage Fix"


def test_phase7b_saves_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = Phase7BPMLeakageForensics(tmp_path)

    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 7-B" in paths.markdown.read_text(encoding="utf-8")

