from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase7b_prime_pm_ai_leakage_fix import Phase7BPrimePMLeakageFixAudit, classify_pm_column_prime


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_fixture(root: Path, *, forbidden_feature: bool = False) -> None:
    model_dir = root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean"
    dataset_path = root / "data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet"
    features = ["close_position", "return_5d", "candidate_count_in_day", "rank_in_day"]
    if forbidden_feature:
        features.append("actual_net_profit")
    _write_json(model_dir / "feature_columns.json", features)
    _write_json(
        model_dir / "model_metadata.json",
        {
            "model_profile": "portfolio_manager_v2_73_phase3b_clean",
            "feature_columns": features,
            "feature_count": len(features),
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
                "close_position": 0.7,
                "return_5d": 0.01,
                "candidate_count_in_day": 25,
                "rank_in_day": 1,
                "future_10d_return": 0.04,
                "realized_return": 0.04,
                "high_conviction_target": True,
                "actual_net_profit": 1000,
            },
            {
                "signal_date": "2026-05-28",
                "code": "22220",
                "close_position": 0.2,
                "return_5d": -0.01,
                "candidate_count_in_day": 30,
                "rank_in_day": 3,
                "future_10d_return": -0.02,
                "realized_return": -0.02,
                "high_conviction_target": False,
                "actual_net_profit": -500,
            },
        ]
    ).to_parquet(dataset_path)


def test_prime_classifier_treats_close_position_as_safe_price_feature() -> None:
    assert classify_pm_column_prime("close_position") == "api_price_feature"
    assert classify_pm_column_prime("body_ratio") == "api_price_feature"
    assert classify_pm_column_prime("candidate_count_in_day") == "candidate_list_dependent"
    assert classify_pm_column_prime("actual_net_profit") == "backtest_trade_outcome"


def test_prime_reaudit_separates_suspicious_candidate_features_from_forbidden(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = Phase7BPrimePMLeakageFixAudit(tmp_path).build_report()

    feature_audit = result["training_feature_audit"]
    judgement = result["final_judgement"]
    assert feature_audit["forbidden_columns_used_as_features"] == []
    assert "candidate_count_in_day" in feature_audit["suspicious_columns_used_as_features"]
    assert "close_position" not in feature_audit["forbidden_columns_used_as_features"]
    assert judgement["feature_leakage_confirmed"] is False
    assert judgement["feature_leakage_suspected"] is True
    assert judgement["v282_result_trust_level"] == "medium_trust"


def test_prime_reaudit_confirms_forbidden_feature_if_actual_outcome_is_used(tmp_path: Path) -> None:
    _write_fixture(tmp_path, forbidden_feature=True)

    result = Phase7BPrimePMLeakageFixAudit(tmp_path).build_report()

    assert result["training_feature_audit"]["forbidden_columns_used_as_features"] == ["actual_net_profit"]
    assert result["final_judgement"]["feature_leakage_confirmed"] is True
    assert result["final_judgement"]["v282_result_trust_level"] == "low_trust"


def test_prime_reaudit_saves_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = Phase7BPrimePMLeakageFixAudit(tmp_path)

    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 7-B'" in paths.markdown.read_text(encoding="utf-8")

