from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import ML_MODELS_ROOT, ML_REPORTS_ROOT


EXIT_MODEL_TARGETS = {
    "future_remaining_return_5d_regression": {"target": "future_remaining_return_5d", "task": "regression"},
    "future_remaining_return_10d_regression": {"target": "future_remaining_return_10d", "task": "regression"},
    "hold_better_5d_classification": {"target": "hold_better_5d", "task": "classification"},
    "should_exit_now_5d_classification": {"target": "should_exit_now_5d", "task": "classification"},
    "avoid_loss_5d_classification": {"target": "avoid_loss_5d", "task": "classification"},
}

EXIT_FEATURE_COLUMNS = [
    "holding_days",
    "entry_price",
    "current_close",
    "unrealized_return",
    "max_unrealized_return_so_far",
    "min_unrealized_return_so_far",
    "drawdown_from_peak",
    "remaining_days_to_actual_exit",
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "risk_adjusted_score",
    "volume",
    "turnover_value",
    "return_5d",
    "return_10d",
    "ma25_gap",
    "daily_range_ratio",
]

DEPLOYABLE_EXCLUDED_FEATURES = ["remaining_days_to_actual_exit"]


@dataclass(frozen=True)
class ExitModelPaths:
    model_dir: Path
    markdown: Path
    json: Path


class ExitModelTrainer:
    """Train Exit AI models from held-day exit datasets."""

    def __init__(
        self,
        model_root: str | Path = ML_MODELS_ROOT / "exit" / "current_v2_66",
        report_root: str | Path = ML_REPORTS_ROOT,
    ) -> None:
        self.model_root = Path(model_root)
        self.report_root = Path(report_root)
        self.feature_columns: list[str] = []

    def load_dataset(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        df["current_date"] = pd.to_datetime(df["current_date"], errors="coerce")
        return df

    def split_by_time(
        self,
        df: pd.DataFrame,
        train_end: str = "2025-12-31",
        valid_start: str = "2026-01-01",
        valid_end: str = "2026-05-31",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = df[df["current_date"].le(pd.Timestamp(train_end))].copy()
        valid = df[df["current_date"].between(pd.Timestamp(valid_start), pd.Timestamp(valid_end))].copy()
        return train, valid

    def train_all(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, include_remaining_days: bool = False) -> dict[str, Any]:
        feature_columns = self._feature_columns(include_remaining_days)
        self.feature_columns = feature_columns
        train = self._prepare_features(train_df, feature_columns)
        valid = self._prepare_features(valid_df, feature_columns)
        models: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        predictions: dict[str, pd.DataFrame] = {}
        for model_name, spec in EXIT_MODEL_TARGETS.items():
            target = spec["target"]
            task = spec["task"]
            train_target = train.dropna(subset=[target]).copy()
            valid_target = valid.dropna(subset=[target]).copy()
            if train_target.empty or valid_target.empty:
                metrics[model_name] = {"skipped": True, "reason": "missing train or validation target rows"}
                continue
            if task == "regression":
                model, model_metrics, pred_df = self._train_regression(model_name, target, train_target, valid_target, feature_columns)
            else:
                model, model_metrics, pred_df = self._train_classification(model_name, target, train_target, valid_target, feature_columns)
            models[model_name] = model
            metrics[model_name] = model_metrics
            predictions[model_name] = pred_df
        return {"models": models, "metrics": metrics, "feature_columns": feature_columns, "predictions": predictions}

    def compare_feature_sets(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, Any]:
        without = self.train_all(train_df, valid_df, include_remaining_days=False)
        with_remaining = self.train_all(train_df, valid_df, include_remaining_days=True)
        self.feature_columns = without["feature_columns"]
        return {
            "without_remaining_days": self._strip_models(without),
            "with_remaining_days": self._strip_models(with_remaining),
            "selected_feature_set": "without_remaining_days",
        }

    def save_models(self, train_result: dict[str, Any], metadata: dict[str, Any]) -> Path:
        if self.model_root.exists():
            shutil.rmtree(self.model_root)
        self.model_root.mkdir(parents=True, exist_ok=True)
        for name, model in train_result["models"].items():
            self._dump_model(model, self.model_root / f"{name}.joblib")
        self._write_json(self.model_root / "feature_columns.json", train_result["feature_columns"])
        self._write_json(self.model_root / "metrics.json", train_result["metrics"])
        self._write_json(self.model_root / "model_metadata.json", metadata)
        return self.model_root

    def save_report(self, report: dict[str, Any]) -> ExitModelPaths:
        self.report_root.mkdir(parents=True, exist_ok=True)
        markdown = self.report_root / "exit_model_training_v2_66_2023-01_to_2026-05.md"
        json_path = self.report_root / "exit_model_training_v2_66_2023-01_to_2026-05.json"
        markdown.write_text(self.format_markdown(report), encoding="utf-8")
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return ExitModelPaths(model_dir=self.model_root, markdown=markdown, json=json_path)

    def build_metadata(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
        return {
            "model_profile": "exit_v2_66",
            "source_profile": "rookie_dealer_02_v2_66_ml_ranked",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "train_start": self._date_min(train_df),
            "train_end": self._date_max(train_df),
            "valid_start": self._date_min(valid_df),
            "valid_end": self._date_max(valid_df),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "excluded_for_deployability": DEPLOYABLE_EXCLUDED_FEATURES,
            "targets": EXIT_MODEL_TARGETS,
            "leakage_guard": "time split by current_date; deployable feature set excludes remaining_days_to_actual_exit",
        }

    def format_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Exit Model Training v2_66",
            "",
            f"- dataset_path: `{report.get('dataset_path')}`",
            f"- model_dir: `{report.get('model_dir')}`",
            f"- train_rows: {report.get('train_rows')}",
            f"- valid_rows: {report.get('valid_rows')}",
            f"- selected_feature_set: `{report.get('selected_feature_set')}`",
            f"- feature_count: {report.get('feature_count')}",
            "",
            "## Metrics",
            "",
            self._metrics_table(report["metrics"]),
            "",
            "## Feature Set Comparison",
            "",
            self._comparison_table(report["feature_set_comparison"]),
            "",
            "## Decile Analysis",
            "",
        ]
        for model_name, rows in report.get("decile_analysis", {}).items():
            lines.extend([f"### {model_name}", "", self._table(rows, list(rows[0].keys()) if rows else []), ""])
        lines.extend(
            [
                "## Leakage Notes",
                "",
                "- Train/valid split is chronological by `current_date`.",
                "- Features use held-day state and walk-forward prediction values already present in the dataset.",
                "- `remaining_days_to_actual_exit` is excluded from the selected deployable model because it is not known in live operation.",
                "",
            ]
        )
        return "\n".join(lines)

    def _train_regression(
        self,
        model_name: str,
        target: str,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        features: list[str],
    ) -> tuple[Any, dict[str, Any], pd.DataFrame]:
        model = self._make_regression_model()
        model.fit(train[features], train[target], eval_set=[(valid[features], valid[target])])
        predicted = pd.Series(model.predict(valid[features]), index=valid.index)
        metrics = {
            "rmse": self._rmse(valid[target], predicted),
            "mae": self._mae(valid[target], predicted),
            "correlation": self._correlation(valid[target], predicted),
            "decile_analysis": self._decile_analysis(predicted, valid[target], target, higher_is_better=True),
        }
        pred_df = pd.DataFrame({"actual": valid[target], "predicted": predicted})
        return model, metrics, pred_df

    def _train_classification(
        self,
        model_name: str,
        target: str,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        features: list[str],
    ) -> tuple[Any, dict[str, Any], pd.DataFrame]:
        model = self._make_classification_model()
        y_train = train[target].astype(int)
        y_valid = valid[target].astype(int)
        model.fit(train[features], y_train, eval_set=[(valid[features], y_valid)])
        probabilities = pd.Series(self._predict_positive_probability(model, valid[features]), index=valid.index)
        predictions = probabilities.ge(0.5).astype(int)
        metrics = {
            "auc": self._auc(y_valid, probabilities),
            "accuracy": self._accuracy(y_valid, predictions),
            "precision": self._precision(y_valid, predictions),
            "recall": self._recall(y_valid, predictions),
            "decile_analysis": self._decile_analysis(probabilities, y_valid, target, higher_is_better=True),
        }
        pred_df = pd.DataFrame({"actual": y_valid, "predicted": probabilities})
        return model, metrics, pred_df

    def _feature_columns(self, include_remaining_days: bool) -> list[str]:
        excluded = set() if include_remaining_days else set(DEPLOYABLE_EXCLUDED_FEATURES)
        return [column for column in EXIT_FEATURE_COLUMNS if column not in excluded]

    def _prepare_features(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        prepared = df.copy()
        for column in feature_columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        return prepared

    def _make_regression_model(self) -> Any:
        try:
            from lightgbm import LGBMRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("lightgbm is required to train Exit AI models. Install requirements.txt first.") from exc
        return LGBMRegressor(
            objective="regression",
            metric="rmse",
            learning_rate=0.05,
            num_leaves=15,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_data_in_leaf=20,
            n_estimators=100,
            verbose=-1,
        )

    def _make_classification_model(self) -> Any:
        try:
            from lightgbm import LGBMClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("lightgbm is required to train Exit AI models. Install requirements.txt first.") from exc
        return LGBMClassifier(
            objective="binary",
            metric="auc",
            learning_rate=0.05,
            num_leaves=15,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_data_in_leaf=20,
            n_estimators=100,
            verbose=-1,
        )

    def _dump_model(self, model: Any, path: Path) -> None:
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("joblib is required to save Exit AI models. Install requirements.txt first.") from exc
        joblib.dump(model, path)

    def _predict_positive_probability(self, model: Any, features: pd.DataFrame) -> list[float]:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            return [float(row[1]) for row in probabilities]
        return [float(value) for value in model.predict(features)]

    def _strip_models(self, result: dict[str, Any]) -> dict[str, Any]:
        return {"metrics": result["metrics"], "feature_columns": result["feature_columns"]}

    def _metrics_table(self, metrics: dict[str, Any]) -> str:
        rows = []
        for model_name, payload in metrics.items():
            row = {"model": model_name}
            for key in ["rmse", "mae", "correlation", "auc", "accuracy", "precision", "recall"]:
                if key in payload:
                    row[key] = payload[key]
            rows.append(row)
        return self._table(rows, ["model", "rmse", "mae", "correlation", "auc", "accuracy", "precision", "recall"])

    def _comparison_table(self, comparison: dict[str, Any]) -> str:
        rows = []
        for feature_set, payload in comparison.items():
            if feature_set == "selected_feature_set":
                continue
            for model_name, metrics in payload.get("metrics", {}).items():
                row = {"feature_set": feature_set, "model": model_name}
                for key in ["rmse", "mae", "correlation", "auc", "accuracy", "precision", "recall"]:
                    if key in metrics:
                        row[key] = metrics[key]
                rows.append(row)
        return self._table(rows, ["feature_set", "model", "rmse", "mae", "correlation", "auc", "accuracy", "precision", "recall"])

    def _decile_analysis(self, predicted: pd.Series, actual: pd.Series, target: str, higher_is_better: bool) -> list[dict[str, Any]]:
        df = pd.DataFrame({"predicted": predicted, "actual": actual}).dropna()
        if df.empty:
            return []
        rank = df["predicted"].rank(method="first", ascending=True)
        df["decile"] = pd.qcut(rank, q=min(10, len(df)), labels=False, duplicates="drop") + 1
        rows = []
        for decile, group in df.groupby("decile"):
            rows.append(
                {
                    "decile": int(decile),
                    "count": int(len(group)),
                    "predicted_mean": float(group["predicted"].mean()),
                    "actual_mean": float(group["actual"].mean()),
                    "actual_positive_rate": float(group["actual"].astype(float).gt(0).mean()),
                }
            )
        return rows

    def _rmse(self, actual: pd.Series, predicted: pd.Series) -> float:
        errors = actual.astype(float) - predicted.astype(float)
        return math.sqrt(float((errors * errors).mean()))

    def _mae(self, actual: pd.Series, predicted: pd.Series) -> float:
        return float((actual.astype(float) - predicted.astype(float)).abs().mean())

    def _correlation(self, actual: pd.Series, predicted: pd.Series) -> float:
        if actual.nunique(dropna=True) < 2 or predicted.nunique(dropna=True) < 2:
            return float("nan")
        return float(actual.astype(float).corr(predicted.astype(float)))

    def _accuracy(self, actual: pd.Series, predicted: pd.Series) -> float:
        return float((actual.astype(int) == predicted.astype(int)).mean())

    def _precision(self, actual: pd.Series, predicted: pd.Series) -> float:
        true_positive = int(((actual.astype(int) == 1) & (predicted.astype(int) == 1)).sum())
        predicted_positive = int((predicted.astype(int) == 1).sum())
        return true_positive / predicted_positive if predicted_positive else 0.0

    def _recall(self, actual: pd.Series, predicted: pd.Series) -> float:
        true_positive = int(((actual.astype(int) == 1) & (predicted.astype(int) == 1)).sum())
        actual_positive = int((actual.astype(int) == 1).sum())
        return true_positive / actual_positive if actual_positive else 0.0

    def _auc(self, actual: pd.Series, probabilities: pd.Series) -> float:
        pairs = sorted((float(score), int(label)) for label, score in zip(actual, probabilities))
        positives = sum(label for _, label in pairs)
        negatives = len(pairs) - positives
        if positives == 0 or negatives == 0:
            return float("nan")
        rank_sum = sum(rank for rank, (_, label) in enumerate(pairs, start=1) if label == 1)
        return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    def _date_min(self, df: pd.DataFrame) -> str | None:
        if df.empty:
            return None
        return str(pd.to_datetime(df["current_date"]).min().date())

    def _date_max(self, df: pd.DataFrame) -> str | None:
        if df.empty:
            return None
        return str(pd.to_datetime(df["current_date"]).max().date())

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(self._format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            return f"{value:.4f}"
        return str(value)
