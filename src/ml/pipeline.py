from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.config import ML_MODEL_CURRENT_ROOT, MODEL_FILENAMES
from ml.feature_builder import FeatureBuilder
from ml.label_generator import LabelGenerator
from ml.predictor import Predictor


class DailyMLPipeline:
    """Run the lightweight daily ML workflow."""

    def __init__(
        self,
        feature_builder: FeatureBuilder | None = None,
        predictor: Predictor | None = None,
        label_generator: LabelGenerator | None = None,
        model_root: str | Path = ML_MODEL_CURRENT_ROOT,
    ) -> None:
        self.feature_builder = feature_builder or FeatureBuilder()
        self.predictor = predictor or Predictor()
        self.label_generator = label_generator or LabelGenerator()
        self.model_root = Path(model_root)

    def run_daily_pipeline(self, target_date: str) -> dict[str, Any]:
        warnings: list[str] = []

        features = self.feature_builder.build_daily_features(target_date)
        features_path = self.feature_builder.save_daily_features(features, target_date)

        predictions_path = None
        if self._current_models_available():
            predictions = self.predictor.predict_daily(target_date)
            predictions_path = self.predictor.save_predictions(predictions, target_date)
        else:
            warnings.append("current ML models are missing; skipped prediction")

        labels_paths = self.label_generator.update_available_labels(as_of_date=target_date)
        return {
            "features_path": features_path,
            "predictions_path": predictions_path,
            "labels_paths": labels_paths,
            "warnings": warnings,
        }

    def _current_models_available(self) -> bool:
        required_paths = [self.model_root / "feature_columns.json"]
        required_paths.extend(self.model_root / filename for filename in MODEL_FILENAMES.values())
        return all(path.exists() for path in required_paths)


def run_daily_pipeline(target_date: str) -> dict[str, Any]:
    return DailyMLPipeline().run_daily_pipeline(target_date)
