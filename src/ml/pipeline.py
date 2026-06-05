from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.config import ML_MODEL_CURRENT_ROOT, MODEL_FILENAMES
from ml.daily_candidates import DailyAICandidateExporter, ENRICHED_V2_REQUIRED_FEATURES
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
        candidate_exporter: DailyAICandidateExporter | None = None,
        model_root: str | Path = ML_MODEL_CURRENT_ROOT,
    ) -> None:
        self.feature_builder = feature_builder or FeatureBuilder()
        self.predictor = predictor or Predictor()
        self.label_generator = label_generator or LabelGenerator()
        self.candidate_exporter = candidate_exporter or DailyAICandidateExporter()
        self.model_root = Path(model_root)

    def run_daily_pipeline(
        self,
        target_date: str,
        export_candidates: bool = True,
        candidate_top_n: int = 10,
        min_turnover_value: float = 50_000_000,
        max_bad_entry_probability: float | None = None,
    ) -> dict[str, Any]:
        warnings: list[str] = []

        features = self.feature_builder.build_daily_features(target_date)
        features_path = self.feature_builder.save_daily_features(features, target_date)

        predictions_path = None
        candidate_csv_path = None
        candidate_md_path = None
        if self._current_models_available():
            model_warning = self._current_model_profile_warning()
            if model_warning:
                warnings.append(model_warning)
            predictions = self.predictor.predict_daily(target_date)
            predictions_path = self.predictor.save_predictions(predictions, target_date)
        else:
            warnings.append("current ML models are missing; skipped prediction")

        if export_candidates:
            if predictions_path is None:
                warnings.append("prediction was skipped; skipped AI candidate export")
            else:
                try:
                    candidates = self.candidate_exporter.build_candidates(
                        target_date,
                        top_n=candidate_top_n,
                        min_turnover_value=min_turnover_value,
                        max_bad_entry_probability=max_bad_entry_probability,
                    )
                    candidate_csv_path = self.candidate_exporter.save_csv(candidates, target_date)
                    candidate_md_path = self.candidate_exporter.save_markdown(candidates, target_date)
                except (FileNotFoundError, ValueError) as exc:
                    warnings.append(f"AI candidate export skipped: {exc}")

        labels_paths = self.label_generator.update_available_labels(as_of_date=target_date)
        return {
            "features_path": features_path,
            "predictions_path": predictions_path,
            "candidate_csv_path": candidate_csv_path,
            "candidate_md_path": candidate_md_path,
            "labels_paths": labels_paths,
            "warnings": warnings,
        }

    def _current_models_available(self) -> bool:
        required_paths = [self.model_root / "feature_columns.json"]
        required_paths.extend(self.model_root / filename for filename in MODEL_FILENAMES.values())
        return all(path.exists() for path in required_paths)

    def _current_model_profile_warning(self) -> str | None:
        feature_path = self.model_root / "feature_columns.json"
        try:
            feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "current ML model profile could not be checked; expected enriched_v2 for daily AI candidates"
        feature_set = {str(column) for column in feature_columns}
        missing = sorted(ENRICHED_V2_REQUIRED_FEATURES - feature_set)
        if missing:
            return (
                "current ML model does not look like enriched_v2; "
                f"missing enriched features: {', '.join(missing)}"
            )
        return None


def run_daily_pipeline(
    target_date: str,
    export_candidates: bool = True,
    candidate_top_n: int = 10,
    min_turnover_value: float = 50_000_000,
    max_bad_entry_probability: float | None = None,
) -> dict[str, Any]:
    return DailyMLPipeline().run_daily_pipeline(
        target_date,
        export_candidates=export_candidates,
        candidate_top_n=candidate_top_n,
        min_turnover_value=min_turnover_value,
        max_bad_entry_probability=max_bad_entry_probability,
    )
