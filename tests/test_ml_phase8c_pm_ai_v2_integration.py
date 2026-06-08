from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.phase7d_pm_ai_api_only_dataset_builder import is_candidate_list_feature
from ml.portfolio_manager_sizing import PortfolioManagerSizingAdvisor
from profile_loader import load_profile


class _FixedProbaModel:
    def __init__(self, positive_probability: float, feature_names: list[str] | None = None) -> None:
        self.positive_probability = float(positive_probability)
        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names, dtype=object)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), self.positive_probability)
        return np.column_stack([1.0 - positive, positive])


def test_v290_profile_loads_candidate_pm_without_changing_current_pm() -> None:
    config = load_profile("rookie_dealer_02_v2_90")
    alias_config = load_profile("rookie_dealer_02_v2.90")
    base = load_profile("rookie_dealer_02_v2_82")

    pm = config["portfolio_manager_ai_sizing"]
    assert config["profile_id"] == "rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38"
    assert alias_config["profile_id"] == config["profile_id"]
    assert pm["model_dir"] == "models/ml/portfolio_manager/candidate_v2_api_only"
    assert pm["dataset_path"] == "data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05.parquet"
    assert pm["expected_feature_count"] == 40
    assert base["portfolio_manager_ai_sizing"]["model_dir"] == "models/ml/portfolio_manager/current_v2_73_phase3b_clean"
    assert config["ml_exit_ai"]["model_dir"] == "models/ml/exit/current_v2_66"


def _write_candidate_fixture(root: Path) -> tuple[Path, Path]:
    model_dir = root / "models" / "ml" / "portfolio_manager" / "candidate_v2_api_only"
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = ["close", "volume_ratio_5d"]
    model_columns = ["close", "volume_ratio_5d", "close_missing"]
    (model_dir / "feature_columns.json").write_text(json.dumps(feature_columns), encoding="utf-8")
    (model_dir / "model_metadata.json").write_text(
        json.dumps({"model_profile": "pm_ai_candidate_v2_api_only"}),
        encoding="utf-8",
    )
    (model_dir / "preprocess.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "numeric_columns": feature_columns,
                "categorical_columns": [],
                "medians": {"close": 100.0, "volume_ratio_5d": 1.0},
                "missing_indicator_columns": ["close"],
            }
        ),
        encoding="utf-8",
    )
    joblib.dump(_FixedProbaModel(0.80, model_columns), model_dir / "high_conviction_target_classifier.joblib")
    joblib.dump(_FixedProbaModel(0.20, model_columns), model_dir / "avoid_target_classifier.joblib")

    dataset = root / "data" / "ml" / "portfolio_manager_api_only" / "pm.parquet"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "as_of_date": "2023-01-04",
                "code": "11110",
                "close": None,
                "volume_ratio_5d": 2.0,
            }
        ]
    ).to_parquet(dataset, index=False)
    return model_dir, dataset


def test_candidate_pm_advisor_recreates_missing_indicators_and_candidate_fields(tmp_path: Path) -> None:
    model_dir, dataset = _write_candidate_fixture(tmp_path)

    advisor = PortfolioManagerSizingAdvisor(
        root=tmp_path,
        model_dir=model_dir,
        dataset_path=dataset,
        expected_feature_count=2,
    )
    decision = advisor.decision_for("2023-01-04", "11110")
    fields = decision.as_fields()

    assert decision.feature_found is True
    assert decision.multiplier == 1.30
    assert fields["pm_api_only_candidate_enabled"] is True
    assert fields["pm_candidate_prediction_available"] is True
    assert fields["pm_candidate_feature_missing_count"] == 1
    assert fields["pm_candidate_multiplier"] == 1.30


def test_candidate_pm_advisor_falls_back_when_prediction_unavailable(tmp_path: Path) -> None:
    model_dir, dataset = _write_candidate_fixture(tmp_path)
    advisor = PortfolioManagerSizingAdvisor(
        root=tmp_path,
        model_dir=model_dir,
        dataset_path=dataset,
        expected_feature_count=2,
    )

    decision = advisor.decision_for("2023-01-05", "99990")
    fields = decision.as_fields()

    assert decision.feature_found is False
    assert decision.multiplier == 1.0
    assert fields["pm_candidate_prediction_available"] is False
    assert fields["pm_candidate_fallback_reason"] == "pm_feature_row_missing"


def test_real_candidate_feature_columns_are_api_only() -> None:
    path = Path("models/ml/portfolio_manager/candidate_v2_api_only/feature_columns.json")
    features = json.loads(path.read_text(encoding="utf-8"))

    assert "selected_count_in_day" not in features
    assert not any(is_candidate_list_feature(feature) for feature in features)

