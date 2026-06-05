from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import (
    CATEGORICAL_FEATURE_COLUMNS,
    ML_FEATURES_ROOT,
    ML_MODEL_CURRENT_ROOT,
    ML_PREDICTIONS_ROOT,
    MODEL_FILENAMES,
    PREDICTION_COLUMNS,
)


class Predictor:
    """Run daily predictions from saved current ML models."""

    def __init__(
        self,
        feature_root: str | Path = ML_FEATURES_ROOT,
        model_root: str | Path = ML_MODEL_CURRENT_ROOT,
        prediction_root: str | Path = ML_PREDICTIONS_ROOT,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.model_root = Path(model_root)
        self.prediction_root = Path(prediction_root)
        self.models: dict[str, Any] = {}
        self.feature_columns: list[str] = []

    def load_current_models(self) -> dict[str, Any]:
        feature_columns_path = self.model_root / "feature_columns.json"
        self.feature_columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
        self.models = {}
        for name, filename in MODEL_FILENAMES.items():
            path = self.model_root / filename
            if path.exists():
                self.models[name] = self._load_model(path)
        return self.models

    def predict_daily(self, target_date: str) -> pd.DataFrame:
        if not self.models or not self.feature_columns:
            self.load_current_models()

        features = self._read_features(target_date)
        if features.empty:
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        prepared = self._prepare_features(features)

        output = features[["date", "code"]].copy()
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
        output["code"] = output["code"].astype("string")
        for column in CATEGORICAL_FEATURE_COLUMNS:
            output[column] = features[column].astype("string") if column in features.columns else pd.NA
        output["expected_return_5d"] = self._predict_values(self.models["future_5d_return_regression"], prepared)
        output["expected_return_10d"] = self._predict_values(self.models["future_10d_return_regression"], prepared)
        output["upside_probability_10d"] = self._predict_probability(self.models["upside_10d_classification"], prepared)
        output["bad_entry_probability_10d"] = self._predict_probability(self.models["bad_entry_10d_classification"], prepared)
        output["expected_max_return_10d"] = self._predict_optional_values(
            "future_max_return_10d_regression",
            prepared,
            len(output),
        )
        output["expected_max_return_20d"] = self._predict_optional_values(
            "future_max_return_20d_regression",
            prepared,
            len(output),
        )
        output["swing_success_probability_20d"] = self._predict_optional_probability(
            "future_swing_success_20d_classification",
            prepared,
            len(output),
        )
        output["entry_risk_label"] = output["bad_entry_probability_10d"].map(self._risk_label)
        output["ml_score"] = (
            output["expected_return_10d"] * 100
            + output["upside_probability_10d"] * 10
            - output["bad_entry_probability_10d"] * 15
        )
        return output[PREDICTION_COLUMNS].reset_index(drop=True)

    def save_predictions(self, df: pd.DataFrame, target_date: str) -> Path:
        path = self.prediction_root / f"predictions_{target_date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def _read_features(self, target_date: str) -> pd.DataFrame:
        return pd.read_parquet(self.feature_root / f"features_{target_date}.parquet")

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        prepared = features.copy()
        for column in self.feature_columns:
            if column not in prepared.columns:
                prepared[column] = pd.NA
        prepared = prepared[self.feature_columns]
        for column in CATEGORICAL_FEATURE_COLUMNS:
            if column in prepared.columns:
                prepared[column] = prepared[column].astype("category")
        return prepared

    def _load_model(self, path: Path) -> Any:
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("joblib is required to load ML models. Install requirements.txt first.") from exc
        return joblib.load(path)

    def _predict_values(self, model: Any, features: pd.DataFrame) -> list[float]:
        return [float(value) for value in model.predict(features)]

    def _predict_optional_values(self, model_name: str, features: pd.DataFrame, rows: int) -> list[float | None]:
        model = self.models.get(model_name)
        if model is None:
            return [None for _ in range(rows)]
        return self._predict_values(model, features)

    def _predict_probability(self, model: Any, features: pd.DataFrame) -> list[float]:
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(features)
                return [float(row[1]) for row in probabilities]
            except AttributeError:
                pass
        return [float(value) for value in model.predict(features)]

    def _predict_optional_probability(self, model_name: str, features: pd.DataFrame, rows: int) -> list[float | None]:
        model = self.models.get(model_name)
        if model is None:
            return [None for _ in range(rows)]
        return self._predict_probability(model, features)

    def _risk_label(self, probability: float) -> str:
        if probability < 0.25:
            return "safe"
        if probability < 0.40:
            return "watch"
        return "danger"
