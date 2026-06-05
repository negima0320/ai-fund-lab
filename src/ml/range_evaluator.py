from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import ML_REPORTS_ROOT
from ml.dataset_builder import DatasetBuilder
from ml.evaluator import PredictionEvaluator
from ml.feature_builder import FeatureBuilder
from ml.label_generator import LabelGenerator
from ml.model_trainer import ModelTrainer
from ml.predictor import Predictor


class RangeSmokeRunner:
    """Run a small multi-day ML smoke loop from cached local data."""

    def __init__(
        self,
        feature_builder: FeatureBuilder | None = None,
        label_generator: LabelGenerator | None = None,
        dataset_builder: DatasetBuilder | None = None,
        model_trainer: ModelTrainer | None = None,
        predictor: Predictor | None = None,
        evaluator: PredictionEvaluator | None = None,
        report_root: str | Path = ML_REPORTS_ROOT,
    ) -> None:
        self.feature_builder = feature_builder or FeatureBuilder()
        self.label_generator = label_generator or LabelGenerator()
        self.dataset_builder = dataset_builder or DatasetBuilder()
        self.model_trainer = model_trainer or ModelTrainer()
        self.predictor = predictor or Predictor()
        self.evaluator = evaluator or PredictionEvaluator()
        self.report_root = Path(report_root)

    def run(self, start_date: str, end_date: str, train_end: str, valid_end: str, top_n: int = 10) -> dict[str, Any]:
        warnings: list[str] = []
        processed_dates: list[str] = []
        skipped_dates: list[dict[str, str]] = []
        labels_by_date: dict[str, pd.DataFrame] = {}
        features_total_rows = 0
        labels_total_rows = 0

        for date_text in self._date_texts(start_date, end_date):
            features = self.feature_builder.build_daily_features(date_text)
            self.feature_builder.save_daily_features(features, date_text)
            labels = self.label_generator.generate_labels(date_text)
            if not labels.empty:
                self.label_generator.save_labels(labels, date_text)

            reasons = []
            if features.empty:
                reasons.append("features empty")
            if labels.empty:
                reasons.append("labels empty")
            if reasons:
                skipped_dates.append({"date": date_text, "reason": ", ".join(reasons)})
                continue

            processed_dates.append(date_text)
            labels_by_date[date_text] = labels
            features_total_rows += int(len(features))
            labels_total_rows += int(len(labels))

        dataset = self.dataset_builder.build_dataset(start_date, end_date)
        dataset_path = self.dataset_builder.save_dataset(dataset, "ml_dataset")
        train, valid, test = self.dataset_builder.split_by_time(dataset, train_end, valid_end)
        train_path = self.dataset_builder.save_dataset(train, "train")
        valid_path = self.dataset_builder.save_dataset(valid, "valid")
        test_path = self.dataset_builder.save_dataset(test, "test")

        model_path: Path | None = None
        metrics: dict[str, Any] = {}
        if train.empty:
            warnings.append("training skipped because train dataset is empty")
        else:
            training = self.model_trainer.train_all(train, valid)
            model_path = self.model_trainer.save_models(training["models"], training["metrics"])
            metrics = training["metrics"]
            warnings.extend(metrics.get("warnings", []))

        prediction_rows_total = 0
        joined_frames: list[pd.DataFrame] = []
        evaluation_paths: list[Path] = []
        prediction_paths: list[Path] = []
        if model_path is not None:
            for date_text in processed_dates:
                try:
                    predictions = self.predictor.predict_daily(date_text)
                    prediction_path = self.predictor.save_predictions(predictions, date_text)
                    prediction_paths.append(prediction_path)
                except Exception as exc:
                    warnings.append(f"{date_text}: prediction failed: {exc}")
                    continue

                prediction_rows_total += int(len(predictions))
                labels = labels_by_date.get(date_text, pd.DataFrame())
                joined = self.evaluator.join_predictions_labels(predictions, labels)
                if joined.empty:
                    skipped_dates.append({"date": date_text, "reason": "prediction/label join empty"})
                    continue

                joined_frames.append(joined)
                evaluation = self.evaluator.evaluate_joined(joined, date_text, top_n=top_n)
                evaluation_paths.append(self.evaluator.save_report(evaluation, date_text))

        joined_all = pd.concat(joined_frames, ignore_index=True) if joined_frames else pd.DataFrame()
        range_evaluation = self.evaluator.evaluate_joined(joined_all, f"{start_date}_to_{end_date}", top_n=top_n)
        range_report_path = self.save_range_report(range_evaluation, start_date, end_date, processed_dates, skipped_dates, warnings)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "train_end": train_end,
            "valid_end": valid_end,
            "processed_dates": processed_dates,
            "skipped_dates": skipped_dates,
            "features_total_rows": features_total_rows,
            "labels_total_rows": labels_total_rows,
            "dataset_rows": int(len(dataset)),
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "test_rows": int(len(test)),
            "prediction_rows_total": prediction_rows_total,
            "joined_evaluation_rows_total": int(range_evaluation["joined_rows"]),
            "risk_bad_entry_rates": self._risk_bad_entry_rates(range_evaluation),
            "top_n_future_10d_return_mean": range_evaluation["top_n_summary"]["future_10d_return_mean"],
            "expected_vs_future_10d_corr": range_evaluation["expected_vs_future_10d_corr"],
            "dataset_path": dataset_path,
            "train_path": train_path,
            "valid_path": valid_path,
            "test_path": test_path,
            "model_path": model_path,
            "prediction_paths": prediction_paths,
            "evaluation_paths": evaluation_paths,
            "range_report_path": range_report_path,
            "warnings": warnings,
        }

    def save_range_report(
        self,
        evaluation: dict[str, Any],
        start_date: str,
        end_date: str,
        processed_dates: list[str],
        skipped_dates: list[dict[str, str]],
        warnings: list[str],
    ) -> Path:
        path = self.report_root / f"range_smoke_{start_date}_to_{end_date}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.format_range_report(evaluation, start_date, end_date, processed_dates, skipped_dates, warnings),
            encoding="utf-8",
        )
        return path

    def format_range_report(
        self,
        evaluation: dict[str, Any],
        start_date: str,
        end_date: str,
        processed_dates: list[str],
        skipped_dates: list[dict[str, str]],
        warnings: list[str],
    ) -> str:
        lines = [
            f"# ML Range Smoke Evaluation {start_date} to {end_date}",
            "",
            "This is a short smoke check. Do not treat it as model quality evidence.",
            "",
            "## Overview",
            "",
            f"- processed_dates: {', '.join(processed_dates) if processed_dates else 'none'}",
            f"- skipped_dates: {self._format_skipped(skipped_dates)}",
            f"- joined_evaluation_rows_total: {evaluation['joined_rows']}",
            f"- top_n: {evaluation['top_n']}",
            f"- top_n_future_10d_return_mean: {self._fmt(evaluation['top_n_summary']['future_10d_return_mean'])}",
            f"- expected_vs_future_10d_corr: {self._fmt(evaluation['expected_vs_future_10d_corr'])}",
            "",
            "## Entry Risk Label Summary",
            "",
            self.evaluator._group_table(evaluation["risk_label_summary"], "entry_risk_label"),
            "",
            "## Bad Entry Probability Bands",
            "",
            self.evaluator._group_table(evaluation["bad_entry_probability_bands"], "band"),
            "",
            "## Top Rows",
            "",
            self.evaluator._top_rows_table(evaluation["top_rows"]),
            "",
            "## Warnings",
            "",
            "\n".join(f"- {warning}" for warning in warnings) if warnings else "_None._",
            "",
        ]
        return "\n".join(lines)

    def _risk_bad_entry_rates(self, evaluation: dict[str, Any]) -> dict[str, float | None]:
        return {
            row["entry_risk_label"]: row["bad_entry_10d_rate"]
            for row in evaluation["risk_label_summary"]
        }

    def _date_texts(self, start_date: str, end_date: str) -> list[str]:
        return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]

    def _format_skipped(self, skipped_dates: list[dict[str, str]]) -> str:
        if not skipped_dates:
            return "none"
        return ", ".join(f"{item['date']} ({item['reason']})" for item in skipped_dates)

    def _fmt(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)
