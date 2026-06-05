from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import (
    CATEGORICAL_FEATURE_COLUMNS,
    LIGHTGBM_CLASSIFICATION_PARAMS,
    LIGHTGBM_REGRESSION_PARAMS,
    ML_MODEL_ARCHIVE_ROOT,
    ML_MODEL_CURRENT_ROOT,
    MODEL_EXCLUDE_COLUMNS,
    MODEL_TARGETS,
)


class ModelTrainer:
    """Train and save the configured LightGBM models."""

    def __init__(
        self,
        archive_root: str | Path = ML_MODEL_ARCHIVE_ROOT,
        current_root: str | Path = ML_MODEL_CURRENT_ROOT,
        timestamp: str | None = None,
    ) -> None:
        self.archive_root = Path(archive_root)
        self.current_root = Path(current_root)
        self.timestamp = timestamp
        self.feature_columns: list[str] = []
        self.warnings: list[str] = []

    def load_dataset(self, path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def train_all(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, Any]:
        train_prepared, valid_prepared, feature_columns = self._prepare_frames(train_df, valid_df)
        self.feature_columns = feature_columns
        models: dict[str, Any] = {}
        metrics: dict[str, Any] = {}

        for name, spec in MODEL_TARGETS.items():
            target = spec["target"]
            task = spec["task"]
            if target not in train_prepared.columns:
                self.warnings.append(f"{name} skipped because target column is missing: {target}")
                continue
            target_train, target_valid = self._target_frames(train_prepared, valid_prepared, target)
            if target_train.empty:
                self.warnings.append(f"{name} skipped because train target has no non-null rows: {target}")
                continue
            if task == "regression":
                model, model_metrics = self.train_regression(target, target_train, target_valid, feature_columns)
            else:
                model, model_metrics = self.train_classification(target, target_train, target_valid, feature_columns)
            models[name] = model
            metrics[name] = model_metrics

        if self.warnings:
            metrics["warnings"] = list(self.warnings)
        return {"models": models, "metrics": metrics, "feature_columns": feature_columns}

    def train_regression(
        self,
        target_col: str,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> tuple[Any, dict[str, float]]:
        features = feature_columns or self.extract_feature_columns(train_df)
        model = self._make_regression_model()
        model.fit(train_df[features], train_df[target_col], eval_set=[(valid_df[features], valid_df[target_col])])
        predictions = model.predict(valid_df[features])
        actual = valid_df[target_col]
        metrics = {
            "rmse": self._rmse(actual, predictions),
            "mae": self._mae(actual, predictions),
        }
        return model, metrics

    def train_classification(
        self,
        target_col: str,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> tuple[Any, dict[str, float]]:
        features = feature_columns or self.extract_feature_columns(train_df)
        model = self._make_classification_model()
        y_train = train_df[target_col].astype(int)
        y_valid = valid_df[target_col].astype(int)
        model.fit(train_df[features], y_train, eval_set=[(valid_df[features], y_valid)])
        probabilities = self._predict_positive_probability(model, valid_df[features])
        predictions = [1 if value >= 0.5 else 0 for value in probabilities]
        metrics = {
            "auc": self._auc(y_valid, probabilities),
            "accuracy": self._accuracy(y_valid, predictions),
            "precision": self._precision(y_valid, predictions),
            "recall": self._recall(y_valid, predictions),
        }
        return model, metrics

    def save_models(self, models: dict[str, Any], metrics: dict[str, Any]) -> Path:
        archive_dir = self.archive_root / (self.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
        archive_dir.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            self._dump_model(model, archive_dir / f"{name}.joblib")
        self._write_json(archive_dir / "feature_columns.json", self.feature_columns)
        self._write_json(archive_dir / "metrics.json", metrics)

        if self.current_root.exists():
            shutil.rmtree(self.current_root)
        self.current_root.mkdir(parents=True, exist_ok=True)
        for path in archive_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, self.current_root / path.name)
        return archive_dir

    def extract_feature_columns(self, df: pd.DataFrame) -> list[str]:
        excluded = set(MODEL_EXCLUDE_COLUMNS)
        return [column for column in df.columns if column not in excluded]

    def _prepare_frames(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        train = train_df.copy()
        if valid_df.empty:
            valid = train.copy()
            self.warnings.append("validation dataset is empty; using train dataset as validation for smoke/training continuity")
        else:
            valid = valid_df.copy()
        for frame in [train, valid]:
            for column in CATEGORICAL_FEATURE_COLUMNS:
                if column in frame.columns:
                    frame[column] = frame[column].astype("category")
        feature_columns = self.extract_feature_columns(train)
        return train, valid, feature_columns

    def _target_frames(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = train_df.dropna(subset=[target_col]).copy()
        valid = valid_df.dropna(subset=[target_col]).copy()
        if valid.empty and not train.empty:
            valid = train.copy()
            self.warnings.append(f"{target_col}: validation target is empty; using train rows for validation")
        return train, valid

    def _make_regression_model(self) -> Any:
        try:
            from lightgbm import LGBMRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("lightgbm is required to train ML models. Install requirements.txt first.") from exc
        return LGBMRegressor(**LIGHTGBM_REGRESSION_PARAMS)

    def _make_classification_model(self) -> Any:
        try:
            from lightgbm import LGBMClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("lightgbm is required to train ML models. Install requirements.txt first.") from exc
        return LGBMClassifier(**LIGHTGBM_CLASSIFICATION_PARAMS)

    def _dump_model(self, model: Any, path: Path) -> None:
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("joblib is required to save ML models. Install requirements.txt first.") from exc
        joblib.dump(model, path)

    def _predict_positive_probability(self, model: Any, features: pd.DataFrame) -> list[float]:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            return [float(row[1]) for row in probabilities]
        return [float(value) for value in model.predict(features)]

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _rmse(self, actual: pd.Series, predicted: Any) -> float:
        errors = [float(a) - float(p) for a, p in zip(actual, predicted)]
        return math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else float("nan")

    def _mae(self, actual: pd.Series, predicted: Any) -> float:
        errors = [abs(float(a) - float(p)) for a, p in zip(actual, predicted)]
        return sum(errors) / len(errors) if errors else float("nan")

    def _accuracy(self, actual: pd.Series, predicted: list[int]) -> float:
        pairs = [(int(a), int(p)) for a, p in zip(actual, predicted)]
        return sum(1 for a, p in pairs if a == p) / len(pairs) if pairs else float("nan")

    def _precision(self, actual: pd.Series, predicted: list[int]) -> float:
        pairs = [(int(a), int(p)) for a, p in zip(actual, predicted)]
        true_positive = sum(1 for a, p in pairs if a == 1 and p == 1)
        predicted_positive = sum(1 for _, p in pairs if p == 1)
        return true_positive / predicted_positive if predicted_positive else 0.0

    def _recall(self, actual: pd.Series, predicted: list[int]) -> float:
        pairs = [(int(a), int(p)) for a, p in zip(actual, predicted)]
        true_positive = sum(1 for a, p in pairs if a == 1 and p == 1)
        actual_positive = sum(1 for a, _ in pairs if a == 1)
        return true_positive / actual_positive if actual_positive else 0.0

    def _auc(self, actual: pd.Series, probabilities: list[float]) -> float:
        pairs = sorted((float(score), int(label)) for label, score in zip(actual, probabilities))
        positives = sum(label for _, label in pairs)
        negatives = len(pairs) - positives
        if positives == 0 or negatives == 0:
            return float("nan")
        rank_sum = sum(rank for rank, (_, label) in enumerate(pairs, start=1) if label == 1)
        return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
