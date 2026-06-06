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
from ml.portfolio_manager_dataset import FEATURE_COLUMNS


PORTFOLIO_MANAGER_TARGETS = {
    "ideal_weight_bucket_classification": {"target": "ideal_weight_bucket", "task": "multiclass"},
    "high_conviction_target_classification": {"target": "high_conviction_target", "task": "binary"},
    "avoid_target_classification": {"target": "avoid_target", "task": "binary"},
    "realized_return_regression": {"target": "realized_return", "task": "regression"},
    "ideal_cash_reserve_bucket_classification": {"target": "ideal_cash_reserve_bucket", "task": "multiclass"},
}

PHASE3A_EXCLUDED_FEATURES = [
    "expected_max_return_20d",
    "swing_success_probability_20d",
]


@dataclass(frozen=True)
class PortfolioManagerTrainingPaths:
    model_dir: Path
    markdown: Path
    json: Path


class PortfolioManagerTrainer:
    """Train lightweight Portfolio Manager AI models from candidate-level datasets."""

    def __init__(
        self,
        model_root: str | Path = ML_MODELS_ROOT / "portfolio_manager" / "current_v2_73_phase3a",
        report_root: str | Path = ML_REPORTS_ROOT,
        feature_candidates: list[str] | None = None,
        excluded_features: list[str] | None = None,
        report_name: str = "portfolio_manager_phase3a_training_2023-01_to_2026-05",
        model_profile: str = "portfolio_manager_v2_73_phase3a",
    ) -> None:
        self.model_root = Path(model_root)
        self.report_root = Path(report_root)
        self.feature_candidates = feature_candidates or FEATURE_COLUMNS
        self.excluded_features = excluded_features if excluded_features is not None else PHASE3A_EXCLUDED_FEATURES
        self.report_name = report_name
        self.model_profile = model_profile
        self.feature_columns: list[str] = []
        self.class_maps: dict[str, dict[str, int]] = {}

    def load_dataset(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
        return df

    def split_by_time(
        self,
        df: pd.DataFrame,
        train_end: str = "2025-12-31",
        test_start: str = "2026-01-01",
        test_end: str = "2026-05-31",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = df[df["signal_date"].le(pd.Timestamp(train_end))].copy()
        test = df[df["signal_date"].between(pd.Timestamp(test_start), pd.Timestamp(test_end))].copy()
        return train, test

    def extract_feature_columns(self, df: pd.DataFrame) -> list[str]:
        columns = []
        for column in self.feature_candidates:
            if column in self.excluded_features:
                continue
            if column not in df.columns:
                continue
            if df[column].notna().sum() == 0:
                continue
            columns.append(column)
        return columns

    def train_all(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
        feature_columns = self.extract_feature_columns(train_df)
        self.feature_columns = feature_columns
        train = self._prepare_features(train_df, feature_columns)
        test = self._prepare_features(test_df, feature_columns)
        models: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        prediction_frames: dict[str, pd.DataFrame] = {}

        for model_name, spec in PORTFOLIO_MANAGER_TARGETS.items():
            target = spec["target"]
            task = spec["task"]
            if target not in train.columns:
                metrics[model_name] = {"skipped": True, "reason": f"missing target column: {target}"}
                continue
            train_target = train.dropna(subset=[target]).copy()
            test_target = test.dropna(subset=[target]).copy()
            if train_target.empty or test_target.empty:
                metrics[model_name] = {"skipped": True, "reason": "missing train or test target rows"}
                continue
            if task == "regression":
                model, model_metrics, pred_df = self._train_regression(target, train_target, test_target, feature_columns)
            elif task == "binary":
                model, model_metrics, pred_df = self._train_binary(model_name, target, train_target, test_target, feature_columns)
            else:
                model, model_metrics, pred_df = self._train_multiclass(model_name, target, train_target, test_target, feature_columns)
            models[model_name] = model
            metrics[model_name] = model_metrics
            prediction_frames[model_name] = pred_df
        return {"models": models, "metrics": metrics, "feature_columns": feature_columns, "predictions": prediction_frames}

    def build_metadata(
        self,
        dataset_path: str | Path,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: list[str],
    ) -> dict[str, Any]:
        return {
            "model_profile": self.model_profile,
            "source_profile": "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue",
            "dataset_path": str(dataset_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "train_start": self._date_min(train_df),
            "train_end": self._date_max(train_df),
            "test_start": self._date_min(test_df),
            "test_end": self._date_max(test_df),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "excluded_features": self.excluded_features,
            "targets": PORTFOLIO_MANAGER_TARGETS,
            "leakage_guard": "chronological split by signal_date; label/audit/result columns are excluded from feature_columns",
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

    def build_report(
        self,
        dataset_path: str | Path,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_result: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "dataset_path": str(dataset_path),
            "model_dir": str(self.model_root),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "feature_count": int(len(train_result["feature_columns"])),
            "feature_columns": train_result["feature_columns"],
            "excluded_features": self.excluded_features,
            "target_class_distribution": {
                target: self._value_counts(train_df, target)
                for target in ["ideal_weight_bucket", "high_conviction_target", "avoid_target", "ideal_cash_reserve_bucket"]
            },
            "metrics": train_result["metrics"],
            "decile_analysis": self._selected_deciles(train_result["metrics"]),
            "metadata": metadata,
            "leakage_notes": [
                "Feature columns are taken from Portfolio Manager Phase 2 FEATURE_COLUMNS only.",
                "Label/audit/result columns are not present in feature_columns.",
                "Split is chronological: train <= 2025-12-31, test is 2026-01-01 to 2026-05-31.",
                f"Excluded features: {', '.join(self.excluded_features) if self.excluded_features else 'none'}.",
            ],
        }

    def save_report(self, report: dict[str, Any]) -> PortfolioManagerTrainingPaths:
        self.report_root.mkdir(parents=True, exist_ok=True)
        markdown = self.report_root / f"{self.report_name}.md"
        json_path = self.report_root / f"{self.report_name}.json"
        markdown.write_text(self.format_markdown(report), encoding="utf-8")
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerTrainingPaths(model_dir=self.model_root, markdown=markdown, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        metric_rows = []
        for model_name, metrics in report["metrics"].items():
            row = {"model": model_name}
            for key in ["rmse", "mae", "correlation", "auc", "accuracy", "precision", "recall"]:
                if key in metrics:
                    row[key] = metrics[key]
            metric_rows.append(row)
        lines = [
            f"# Portfolio Manager AI Training ({self.model_profile})",
            "",
            f"- dataset_path: `{report['dataset_path']}`",
            f"- model_dir: `{report['model_dir']}`",
            f"- train_rows: {report['train_rows']}",
            f"- test_rows: {report['test_rows']}",
            f"- feature_count: {report['feature_count']}",
            f"- excluded_features: {', '.join(report['excluded_features'])}",
            "",
            "## Metrics",
            "",
            self._table(metric_rows, ["model", "rmse", "mae", "correlation", "auc", "accuracy", "precision", "recall"]),
            "",
            "## Decile Analysis",
            "",
        ]
        for model_name, rows in report["decile_analysis"].items():
            lines.extend([f"### {model_name}", "", self._table(rows, list(rows[0].keys()) if rows else []), ""])
        lines.extend(["## Confusion Matrices", ""])
        for model_name, metrics in report["metrics"].items():
            if "confusion_matrix" in metrics:
                lines.extend([f"### {model_name}", "", "```json", json.dumps(metrics["confusion_matrix"], ensure_ascii=False, indent=2), "```", ""])
        lines.extend(["## Feature Importance Top 30", ""])
        for model_name, metrics in report["metrics"].items():
            lines.extend(
                [
                    f"### {model_name}",
                    "",
                    self._table(metrics.get("feature_importance_top30", []), ["feature", "importance"]),
                    "",
                ]
            )
        lines.extend(["## Leakage Notes", ""])
        for item in report["leakage_notes"]:
            lines.append(f"- {item}")
        lines.append("")
        return "\n".join(lines)

    def _train_regression(
        self,
        target: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list[str],
    ) -> tuple[Any, dict[str, Any], pd.DataFrame]:
        model = self._make_regression_model()
        model.fit(train[features], train[target], eval_set=[(test[features], test[target])])
        predicted = pd.Series(model.predict(test[features]), index=test.index, name="predicted")
        actual = test[target].astype(float)
        metrics = {
            "rmse": self._rmse(actual, predicted),
            "mae": self._mae(actual, predicted),
            "correlation": self._correlation(actual, predicted),
            "feature_importance_top30": self._feature_importance(model, features),
            "decile_analysis": self._decile_analysis(predicted, test["realized_return"], actual_binary=None),
        }
        return model, metrics, pd.DataFrame({"actual": actual, "predicted": predicted})

    def _train_binary(
        self,
        model_name: str,
        target: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list[str],
    ) -> tuple[Any, dict[str, Any], pd.DataFrame]:
        model = self._make_binary_model()
        y_train = train[target].astype(bool).astype(int)
        y_test = test[target].astype(bool).astype(int)
        model.fit(train[features], y_train, eval_set=[(test[features], y_test)])
        probabilities = pd.Series(self._positive_probability(model, test[features]), index=test.index, name="predicted")
        predicted = probabilities.ge(0.5).astype(int)
        metrics = {
            "auc": self._auc(y_test, probabilities),
            "accuracy": self._accuracy(y_test, predicted),
            "precision": self._precision(y_test, predicted),
            "recall": self._recall(y_test, predicted),
            "class_distribution": self._class_distribution(y_test),
            "confusion_matrix": self._confusion_matrix(y_test, predicted, ["false", "true"]),
            "feature_importance_top30": self._feature_importance(model, features),
            "decile_analysis": self._decile_analysis(probabilities, test["realized_return"], actual_binary=y_test),
        }
        return model, metrics, pd.DataFrame({"actual": y_test, "predicted": probabilities})

    def _train_multiclass(
        self,
        model_name: str,
        target: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list[str],
    ) -> tuple[Any, dict[str, Any], pd.DataFrame]:
        labels = sorted(train[target].dropna().astype(str).unique())
        mapping = {label: index for index, label in enumerate(labels)}
        self.class_maps[model_name] = mapping
        y_train = train[target].astype(str).map(mapping).astype(int)
        y_test = test[target].astype(str).map(mapping)
        keep = y_test.notna()
        test = test[keep].copy()
        y_test = y_test[keep].astype(int)
        model = self._make_multiclass_model(len(labels))
        model.fit(train[features], y_train, eval_set=[(test[features], y_test)])
        probabilities = model.predict_proba(test[features])
        predicted = pd.Series(probabilities.argmax(axis=1), index=test.index)
        metrics = {
            "auc": self._multiclass_auc(y_test, probabilities, list(range(len(labels)))),
            "accuracy": self._accuracy(y_test, predicted),
            "precision": self._macro_precision(y_test, predicted, list(range(len(labels)))),
            "recall": self._macro_recall(y_test, predicted, list(range(len(labels)))),
            "class_labels": labels,
            "class_distribution": self._class_distribution(y_test),
            "confusion_matrix": self._confusion_matrix(y_test, predicted, labels),
            "feature_importance_top30": self._feature_importance(model, features),
        }
        if "strong" in mapping:
            strong_probability = pd.Series(probabilities[:, mapping["strong"]], index=test.index)
            metrics["strong_probability_decile_analysis"] = self._decile_analysis(strong_probability, test["realized_return"], actual_binary=None)
        if "aggressive" in mapping:
            aggressive_probability = pd.Series(probabilities[:, mapping["aggressive"]], index=test.index)
            metrics["aggressive_probability_decile_analysis"] = self._decile_analysis(aggressive_probability, test["realized_return"], actual_binary=None)
        return model, metrics, pd.DataFrame({"actual": y_test, "predicted": predicted})

    def _prepare_features(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        data = df.copy()
        for column in features:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        return data

    def _make_regression_model(self) -> Any:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            metric="rmse",
            learning_rate=0.05,
            num_leaves=15,
            min_data_in_leaf=10,
            n_estimators=120,
            verbose=-1,
        )

    def _make_binary_model(self) -> Any:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            metric="auc",
            learning_rate=0.05,
            num_leaves=15,
            min_data_in_leaf=10,
            n_estimators=120,
            verbose=-1,
        )

    def _make_multiclass_model(self, class_count: int) -> Any:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=class_count,
            metric="multi_logloss",
            learning_rate=0.05,
            num_leaves=15,
            min_data_in_leaf=10,
            n_estimators=120,
            verbose=-1,
        )

    def _dump_model(self, model: Any, path: Path) -> None:
        import joblib

        joblib.dump(model, path)

    def _selected_deciles(self, metrics: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        selected = {}
        if "ideal_weight_bucket_classification" in metrics:
            selected["predicted_strong_probability"] = metrics["ideal_weight_bucket_classification"].get(
                "strong_probability_decile_analysis", []
            )
        if "avoid_target_classification" in metrics:
            selected["predicted_avoid_probability"] = metrics["avoid_target_classification"].get("decile_analysis", [])
        if "realized_return_regression" in metrics:
            selected["predicted_realized_return"] = metrics["realized_return_regression"].get("decile_analysis", [])
        return selected

    def _positive_probability(self, model: Any, features: pd.DataFrame) -> list[float]:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            return [float(row[1]) for row in probabilities]
        return [float(value) for value in model.predict(features)]

    def _feature_importance(self, model: Any, features: list[str]) -> list[dict[str, Any]]:
        values = getattr(model, "feature_importances_", None)
        if values is None:
            return []
        rows = [
            {"feature": feature, "importance": int(importance)}
            for feature, importance in zip(features, values)
        ]
        return sorted(rows, key=lambda row: row["importance"], reverse=True)[:30]

    def _decile_analysis(
        self,
        predicted: pd.Series,
        realized_return: pd.Series,
        actual_binary: pd.Series | None,
    ) -> list[dict[str, Any]]:
        data = pd.DataFrame({"predicted": predicted, "realized_return": realized_return})
        if actual_binary is not None:
            data["actual_positive"] = actual_binary
        data = data.dropna(subset=["predicted", "realized_return"])
        if data.empty:
            return []
        rank = data["predicted"].rank(method="first")
        data["decile"] = pd.qcut(rank, q=min(10, len(data)), labels=False, duplicates="drop") + 1
        rows = []
        for decile, group in data.groupby("decile"):
            row = {
                "decile": int(decile),
                "count": int(len(group)),
                "predicted_mean": float(group["predicted"].mean()),
                "realized_return_mean": float(group["realized_return"].mean()),
                "realized_return_median": float(group["realized_return"].median()),
                "positive_return_rate": float(group["realized_return"].gt(0).mean()),
            }
            if "actual_positive" in group.columns:
                row["actual_target_rate"] = float(group["actual_positive"].astype(bool).mean())
            rows.append(row)
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

    def _macro_precision(self, actual: pd.Series, predicted: pd.Series, labels: list[int]) -> float:
        values = []
        for label in labels:
            true_positive = int(((actual == label) & (predicted == label)).sum())
            predicted_positive = int((predicted == label).sum())
            values.append(true_positive / predicted_positive if predicted_positive else 0.0)
        return float(sum(values) / len(values)) if values else float("nan")

    def _macro_recall(self, actual: pd.Series, predicted: pd.Series, labels: list[int]) -> float:
        values = []
        for label in labels:
            true_positive = int(((actual == label) & (predicted == label)).sum())
            actual_positive = int((actual == label).sum())
            values.append(true_positive / actual_positive if actual_positive else 0.0)
        return float(sum(values) / len(values)) if values else float("nan")

    def _auc(self, actual: pd.Series, probabilities: pd.Series) -> float:
        pairs = sorted((float(score), int(label)) for label, score in zip(actual, probabilities))
        positives = sum(label for _, label in pairs)
        negatives = len(pairs) - positives
        if positives == 0 or negatives == 0:
            return float("nan")
        rank_sum = sum(rank for rank, (_, label) in enumerate(pairs, start=1) if label == 1)
        return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    def _multiclass_auc(self, actual: pd.Series, probabilities: Any, labels: list[int]) -> float:
        try:
            from sklearn.metrics import roc_auc_score

            return float(roc_auc_score(actual, probabilities, labels=labels, multi_class="ovr", average="macro"))
        except Exception:
            return float("nan")

    def _confusion_matrix(self, actual: pd.Series, predicted: pd.Series, labels: list[str]) -> dict[str, Any]:
        label_indices = list(range(len(labels)))
        matrix = []
        for actual_value in label_indices:
            row = []
            for predicted_value in label_indices:
                row.append(int(((actual.astype(int) == actual_value) & (predicted.astype(int) == predicted_value)).sum()))
            matrix.append(row)
        return {"labels": labels, "matrix": matrix}

    def _class_distribution(self, values: pd.Series) -> list[dict[str, Any]]:
        counts = values.astype(str).value_counts()
        total = int(counts.sum())
        return [{"value": value, "count": int(count), "rate": float(count / total) if total else None} for value, count in counts.items()]

    def _value_counts(self, dataset: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        if column not in dataset.columns:
            return []
        counts = dataset[column].fillna("<NA>").astype(str).value_counts()
        total = int(counts.sum())
        return [{"value": value, "count": int(count), "rate": float(count / total) if total else None} for value, count in counts.items()]

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    def _date_min(self, df: pd.DataFrame) -> str | None:
        if df.empty:
            return None
        return str(pd.to_datetime(df["signal_date"]).min().date())

    def _date_max(self, df: pd.DataFrame) -> str | None:
        if df.empty:
            return None
        return str(pd.to_datetime(df["signal_date"]).max().date())

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
