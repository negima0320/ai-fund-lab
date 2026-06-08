from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.portfolio_manager_sizing import PortfolioManagerSizingAdvisor, multiplier_from_high_minus_avoid
from profile_loader import load_profile


class _FixedProbaModel:
    def __init__(self, positive_probability: float, feature_names: list[str] | None = None) -> None:
        self.positive_probability = float(positive_probability)
        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names, dtype=object)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), self.positive_probability)
        return np.column_stack([1.0 - positive, positive])


def test_v291_profile_loads_calibrated_candidate_pm_without_changing_current_pm() -> None:
    config = load_profile("rookie_dealer_02_v2_91")
    alias_config = load_profile("rookie_dealer_02_v2.91")
    base = load_profile("rookie_dealer_02_v2_82")

    pm = config["portfolio_manager_ai_sizing"]
    assert config["profile_id"] == "rookie_dealer_02_v2_91_pm_ai_v2_calibrated_rule_e_cap38"
    assert alias_config["profile_id"] == config["profile_id"]
    assert pm["model_dir"] == "models/ml/portfolio_manager/candidate_v2_api_only"
    assert pm["dataset_path"] == "data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05.parquet"
    assert pm["pm_calibration_rule"] == "quantile_match_current_pm_distribution"
    assert pm["pm_calibration_thresholds"]["pm130_score_min"] == -0.12284356890271281
    assert pm["expected_feature_count"] == 40
    assert pm["per_code_exposure_cap_rate"] == 0.38
    assert config["ml_exit_ai"]["model_dir"] == "models/ml/exit/current_v2_66"
    assert base["portfolio_manager_ai_sizing"]["model_dir"] == "models/ml/portfolio_manager/current_v2_73_phase3b_clean"
    assert "pm_calibration_rule" not in base["portfolio_manager_ai_sizing"]


def _write_candidate_fixture(root: Path, high: float, avoid: float) -> tuple[Path, Path]:
    model_dir = root / "models/ml/portfolio_manager/candidate_v2_api_only"
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = ["close"]
    model_columns = ["close"]
    (model_dir / "feature_columns.json").write_text(json.dumps(feature_columns), encoding="utf-8")
    (model_dir / "model_metadata.json").write_text(json.dumps({"model_profile": "pm_ai_candidate_v2_api_only"}), encoding="utf-8")
    (model_dir / "preprocess.json").write_text(
        json.dumps({"feature_columns": feature_columns, "medians": {"close": 100.0}, "missing_indicator_columns": []}),
        encoding="utf-8",
    )
    joblib.dump(_FixedProbaModel(high, model_columns), model_dir / "high_conviction_target_classifier.joblib")
    joblib.dump(_FixedProbaModel(avoid, model_columns), model_dir / "avoid_target_classifier.joblib")
    dataset = root / "data/ml/portfolio_manager_api_only/pm.parquet"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"as_of_date": "2023-01-04", "code": "11110", "close": 100.0}]).to_parquet(dataset, index=False)
    return model_dir, dataset


def test_rule_e_calibration_keeps_raw_and_calibrated_multiplier_fields(tmp_path: Path) -> None:
    model_dir, dataset = _write_candidate_fixture(tmp_path, high=0.30, avoid=0.40)
    raw = multiplier_from_high_minus_avoid(0.30, 0.40)
    assert raw == 0.80

    advisor = PortfolioManagerSizingAdvisor(
        root=tmp_path,
        model_dir=model_dir,
        dataset_path=dataset,
        expected_feature_count=1,
        calibration_rule="quantile_match_current_pm_distribution",
        calibration_thresholds={
            "pm130_score_min": -0.12,
            "pm115_score_min": -0.14,
            "pm080_score_max": -0.21,
        },
    )
    decision = advisor.decision_for("2023-01-04", "11110")
    fields = decision.as_fields()

    assert decision.score == pytest.approx(-0.10)
    assert fields["pm_multiplier"] == 1.30
    assert fields["pm_candidate_multiplier_raw"] == 0.80
    assert fields["pm_candidate_multiplier_calibrated"] == 1.30
    assert fields["pm_calibration_rule"] == "quantile_match_current_pm_distribution"
    assert fields["pm_candidate_prediction_available"] is True
