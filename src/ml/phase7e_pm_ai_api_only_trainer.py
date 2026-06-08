"""Phase 7-E PM AI API-only trainer prototype.

The prototype trains only from the Phase 7-D API-only dataset when explicitly
requested. Dry-run mode performs schema and leakage checks only. Current PM AI
model directories are never overwritten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase7d_pm_ai_api_only_dataset_builder import is_candidate_list_feature


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase7e_pm_ai_api_only_trainer_2021_to_2026"
DATASET_PATH = Path("data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05.parquet")
MODEL_DIR = Path("models/ml/portfolio_manager/candidate_v2_api_only")

KEY_COLUMNS = {"as_of_date", "code", "split"}
PRIMARY_CLASSIFICATION_TARGETS = ["high_conviction_target", "avoid_target"]
SECONDARY_REGRESSION_TARGETS = ["risk_adjusted_future_return", "future_10d_return"]
ALL_TARGET_COLUMNS = set(PRIMARY_CLASSIFICATION_TARGETS + SECONDARY_REGRESSION_TARGETS + ["future_5d_return"])
FORBIDDEN_EXACT_COLUMNS = {
    "selected_count_in_day",
    "candidate_count_in_day",
    "rank_in_day",
    "score_rank_in_day",
    "candidate_rank",
    "score_rank",
    "max_positions_remaining_before",
    "cash_before",
    "cash_after",
    "decision",
    "exit_reason",
    "skip_reason",
}
FORBIDDEN_PREFIXES = ("actual_", "realized_", "profit_", "cash_", "portfolio_", "position_")


@dataclass(frozen=True)
class TrainOptions:
    dry_run: bool = True
    sample_rows: int | None = None
    train_full: bool = False


@dataclass(frozen=True)
class Phase7EPaths:
    model_dir: Path | None
    markdown: Path
    json: Path


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _is_forbidden(column: str) -> bool:
    lower = column.lower()
    return column in FORBIDDEN_EXACT_COLUMNS or lower.startswith(FORBIDDEN_PREFIXES) or is_candidate_list_feature(column)


def _is_label_like(column: str) -> bool:
    lower = column.lower()
    return (
        column in ALL_TARGET_COLUMNS
        or lower.startswith("future_")
        or lower.endswith("_target")
        or "target" in lower
        or "label" in lower
    )


def _feature_columns(columns: list[str]) -> list[str]:
    features = []
    for column in columns:
        if column in KEY_COLUMNS:
            continue
        if _is_forbidden(column) or _is_label_like(column):
            continue
        features.append(column)
    return features


def _stats(series: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p10": float(clean.quantile(0.10)),
        "p90": float(clean.quantile(0.90)),
    }


class Phase7EPMAIAPIOnlyTrainerPrototype:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        dataset_path: Path | None = None,
        model_dir: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.dataset_path = self._root(dataset_path or DATASET_PATH)
        self.model_dir = self._root(model_dir or MODEL_DIR)

    def run(self, options: TrainOptions | None = None) -> dict[str, Any]:
        options = options or TrainOptions()
        dataset = _read_parquet(self.dataset_path)
        dataset = self._limit_rows(dataset, options)
        feature_columns = _feature_columns(list(dataset.columns))
        leakage = self._leakage_check(dataset, feature_columns)
        report = self._base_report(dataset, feature_columns, options, leakage)
        if options.dry_run and not options.sample_rows and not options.train_full:
            report["next_recommended_action"] = "Run sample training with --sample-rows 50000"
            return report
        if leakage["blocking_issues"]:
            report["training_status"] = {"trained": False, "model_saved": False, "reason": "leakage blocking issues"}
            return report
        train = dataset[dataset["split"].eq("train")].copy()
        valid = dataset[dataset["split"].eq("validation")].copy()
        test = dataset[dataset["split"].eq("test")].copy()
        if train.empty or valid.empty or test.empty:
            report["training_status"] = {"trained": False, "model_saved": False, "reason": "missing train/validation/test rows"}
            return report
        preprocess = self._fit_preprocess(train, feature_columns)
        x_train = self._transform(train, preprocess)
        x_valid = self._transform(valid, preprocess)
        x_test = self._transform(test, preprocess)
        models: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        for target in PRIMARY_CLASSIFICATION_TARGETS:
            y_train = train[target].astype(int)
            y_valid = valid[target].astype(int)
            y_test = test[target].astype(int)
            model = self._make_classifier()
            model.fit(x_train, y_train)
            models[target] = model
            metrics[target] = {
                "validation": self._classification_metrics(y_valid, self._predict_scores(model, x_valid)),
                "test": self._classification_metrics(y_test, self._predict_scores(model, x_test)),
            }
        model_saved = False
        model_path = None
        if options.train_full:
            model_path = self._save_candidate_models(models, preprocess, feature_columns)
            model_saved = True
        report.update(
            {
                "training_status": {
                    "trained": True,
                    "models": {target: "sklearn.ensemble.HistGradientBoostingClassifier" for target in models},
                    "model_saved": model_saved,
                    "model_save_path": str(model_path) if model_path else None,
                    "model_save_note": "Saved only to candidate PM AI API-only directory." if model_saved else "Dry/sample prototype does not save models.",
                },
                "preprocess": self._preprocess_report(preprocess),
                "metrics": metrics,
                "feature_importance": self._feature_importance(models, list(x_train.columns)),
                "next_recommended_action": "Review sample metrics, then decide whether full candidate training is warranted." if not options.train_full else "Proceed to PM AI API-only integration audit.",
            }
        )
        report["model_output"]["model_saved"] = model_saved
        report["model_output"]["model_dir"] = str(self.model_dir) if model_saved else None
        return report

    def save_report(self, report: dict[str, Any]) -> Phase7EPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase7EPaths(model_dir=Path(report["model_output"]["model_dir"]) if report["model_output"].get("model_dir") else None, markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        metric_rows = []
        for target, by_split in report.get("metrics", {}).items():
            for split, values in by_split.items():
                metric_rows.append({"target": target, "split": split, **values})
        return "\n".join(
            [
                "# AI Retraining Phase 7-E PM AI API-Only Trainer Prototype",
                "",
                "## Scope",
                "",
                "- trainer prototype only",
                "- current PM AI model overwrite forbidden",
                "- no backtest/profile/live order integration",
                "",
                "## Dataset",
                "",
                self._table([report["dataset"]], ["path", "rows_used", "feature_count", "train_rows", "validation_rows", "test_rows"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_check"]], ["forbidden_columns_in_features", "candidate_list_dependent_columns_in_features", "label_like_columns_in_features", "selected_count_in_day_found", "split_overlap", "leakage_risk", "blocking_issues"]),
                "",
                "## Metrics",
                "",
                self._table(metric_rows, ["target", "split", "auc", "pr_auc", "precision_at_top10pct", "recall_at_top10pct", "top_decile_lift", "positive_rate"]),
                "",
                f"Next recommended action: `{report.get('next_recommended_action')}`",
                "",
            ]
        )

    def _base_report(self, dataset: pd.DataFrame, features: list[str], options: TrainOptions, leakage: dict[str, Any]) -> dict[str, Any]:
        split_counts = dataset["split"].value_counts().to_dict() if "split" in dataset.columns else {}
        return {
            "metadata": {
                "phase": "7-E",
                "prototype": True,
                "dry_run": options.dry_run and not options.sample_rows and not options.train_full,
                "sample_rows": options.sample_rows,
                "train_full": options.train_full,
                "full_backtest_executed": False,
                "profile_added": False,
                "current_model_overwritten": False,
                "live_order_placement_executed": False,
            },
            "dataset": {
                "path": str(self.dataset_path),
                "rows_used": int(len(dataset)),
                "columns": int(len(dataset.columns)),
                "feature_count": int(len(features)),
                "train_rows": int(split_counts.get("train", 0)),
                "validation_rows": int(split_counts.get("validation", 0)),
                "test_rows": int(split_counts.get("test", 0)),
            },
            "target_design": {
                "primary_classification_targets": PRIMARY_CLASSIFICATION_TARGETS,
                "secondary_regression_targets": SECONDARY_REGRESSION_TARGETS,
                "trained_in_prototype": PRIMARY_CLASSIFICATION_TARGETS,
                "regression_targets_metadata_only": True,
            },
            "feature_policy": {
                "feature_source": "Phase 7-D API-only dataset feature_columns",
                "feature_columns": features,
                "candidate_list_dependent_features_forbidden": True,
                "stock_selection_walk_forward_predictions_allowed": True,
            },
            "preprocess": {
                "strategy": "train split median/mode imputation; validation/test use train-fitted values only",
                "missing_indicator_threshold": 0.05,
                "scaler": "none",
            },
            "leakage_check": leakage,
            "training_status": {"trained": False, "model_saved": False},
            "metrics": {},
            "model_output": {
                "candidate_model_dir": str(self.model_dir),
                "model_saved": False,
                "current_model_overwrite_forbidden": True,
                "current_model_dir_untouched": str(self.root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean"),
            },
        }

    def _limit_rows(self, dataset: pd.DataFrame, options: TrainOptions) -> pd.DataFrame:
        if dataset.empty or options.train_full or not options.sample_rows:
            return dataset.copy()
        rows = []
        per_split = max(1, options.sample_rows // 3)
        for split_name in ["train", "validation", "test"]:
            rows.append(dataset[dataset["split"].eq(split_name)].head(per_split))
        sample = pd.concat(rows, ignore_index=True)
        if len(sample) < options.sample_rows:
            used = set(sample.index)
            remainder = dataset.loc[~dataset.index.isin(used)].head(options.sample_rows - len(sample))
            sample = pd.concat([sample, remainder], ignore_index=True)
        return sample.head(options.sample_rows).copy()

    def _fit_preprocess(self, train: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
        numeric = [column for column in feature_columns if pd.api.types.is_numeric_dtype(train[column])]
        categorical = [column for column in feature_columns if column not in set(numeric)]
        medians = {column: float(pd.to_numeric(train[column], errors="coerce").median()) for column in numeric}
        modes = {}
        for column in categorical:
            mode = train[column].mode(dropna=True)
            modes[column] = mode.iloc[0] if not mode.empty else ""
        missing_rates = train[feature_columns].isna().mean()
        indicators = [column for column, rate in missing_rates.items() if rate >= 0.05]
        return {"feature_columns": feature_columns, "numeric_columns": numeric, "categorical_columns": categorical, "medians": medians, "modes": modes, "missing_indicator_columns": indicators}

    def _transform(self, frame: pd.DataFrame, preprocess: dict[str, Any]) -> pd.DataFrame:
        source = frame[preprocess["feature_columns"]].copy()
        transformed: dict[str, Any] = {}
        for column in preprocess["numeric_columns"]:
            transformed[column] = pd.to_numeric(source[column], errors="coerce").fillna(preprocess["medians"].get(column, 0.0))
        for column in preprocess["categorical_columns"]:
            filled = source[column].fillna(preprocess["modes"].get(column, "")).astype("category")
            transformed[column] = filled.cat.codes.astype(float)
        output = pd.DataFrame(transformed, index=frame.index)
        for column in preprocess["missing_indicator_columns"]:
            output[f"{column}_missing"] = source[column].isna().astype(int)
        return output

    def _make_classifier(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.08, max_leaf_nodes=31, random_state=42)

    def _predict_scores(self, model: Any, features: pd.DataFrame) -> pd.Series:
        if hasattr(model, "predict_proba"):
            return pd.Series(model.predict_proba(features)[:, 1], index=features.index)
        return pd.Series(model.predict(features), index=features.index)

    def _classification_metrics(self, y_true: pd.Series, scores: pd.Series) -> dict[str, Any]:
        from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

        y = y_true.astype(int)
        score = pd.to_numeric(scores, errors="coerce").fillna(0.0)
        top_n = max(1, int(len(score) * 0.10))
        top_index = score.sort_values(ascending=False).head(top_n).index
        top_pred = pd.Series(0, index=score.index)
        top_pred.loc[top_index] = 1
        positive_rate = float(y.mean()) if len(y) else 0.0
        auc = float(roc_auc_score(y, score)) if y.nunique() > 1 else None
        pr_auc = float(average_precision_score(y, score)) if y.nunique() > 1 else None
        top_rate = float(y.loc[top_index].mean()) if len(top_index) else 0.0
        return {
            "auc": auc,
            "pr_auc": pr_auc,
            "precision_at_top10pct": float(precision_score(y, top_pred, zero_division=0)),
            "recall_at_top10pct": float(recall_score(y, top_pred, zero_division=0)),
            "top_decile_lift": (top_rate / positive_rate) if positive_rate else None,
            "positive_rate": positive_rate,
            "prediction_score": _stats(score),
            "calibration_by_decile": self._calibration_by_decile(y, score),
        }

    def _calibration_by_decile(self, y_true: pd.Series, scores: pd.Series) -> list[dict[str, Any]]:
        frame = pd.DataFrame({"actual": y_true.astype(int), "score": scores})
        frame["decile"] = pd.qcut(frame["score"].rank(method="first"), 10, labels=False, duplicates="drop")
        rows = []
        for decile, group in frame.groupby("decile", dropna=False):
            rows.append({"decile": int(decile) if pd.notna(decile) else None, "rows": int(len(group)), "score_mean": float(group["score"].mean()), "actual_positive_rate": float(group["actual"].mean())})
        return rows

    def _feature_importance(self, models: dict[str, Any], columns: list[str]) -> dict[str, Any]:
        return {"available": False, "reason": "HistGradientBoostingClassifier does not expose impurity feature_importances_.", "models": list(models.keys()), "feature_columns": columns}

    def _save_candidate_models(self, models: dict[str, Any], preprocess: dict[str, Any], feature_columns: list[str]) -> Path:
        import joblib

        self.model_dir.mkdir(parents=True, exist_ok=True)
        for target, model in models.items():
            joblib.dump(model, self.model_dir / f"{target}_classifier.joblib")
        metadata = {
            "model_profile": "pm_ai_candidate_v2_api_only",
            "feature_columns": feature_columns,
            "primary_classification_targets": PRIMARY_CLASSIFICATION_TARGETS,
            "secondary_regression_targets": SECONDARY_REGRESSION_TARGETS,
            "model": "sklearn.ensemble.HistGradientBoostingClassifier",
            "current_model_overwrite_forbidden": True,
        }
        (self.model_dir / "preprocess.json").write_text(json.dumps(preprocess, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (self.model_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (self.model_dir / "feature_columns.json").write_text(json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.model_dir

    def _preprocess_report(self, preprocess: dict[str, Any]) -> dict[str, Any]:
        return {
            "numeric_strategy": "median",
            "categorical_strategy": "mode",
            "fit_scope": "train split only",
            "numeric_columns": preprocess["numeric_columns"],
            "categorical_columns": preprocess["categorical_columns"],
            "columns_to_impute": sorted([*preprocess["numeric_columns"], *preprocess["categorical_columns"]]),
            "missing_indicator_columns": preprocess["missing_indicator_columns"],
            "scaler": "none",
        }

    def _leakage_check(self, dataset: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
        forbidden = [column for column in feature_columns if _is_forbidden(column)]
        candidate = [column for column in feature_columns if is_candidate_list_feature(column)]
        candidate_in_dataset = [column for column in dataset.columns if is_candidate_list_feature(column)]
        label_like = [column for column in feature_columns if _is_label_like(column)]
        selected_count = "selected_count_in_day" in dataset.columns or "selected_count_in_day" in feature_columns
        split_overlap = self._split_overlap(dataset)
        blocking = []
        if forbidden:
            blocking.append("Forbidden columns remain in features.")
        if candidate:
            blocking.append("Candidate-list dependent columns remain in features.")
        if candidate_in_dataset:
            blocking.append("Candidate-list dependent columns are present in dataset.")
        if label_like:
            blocking.append("Label-like columns remain in features.")
        if selected_count:
            blocking.append("selected_count_in_day is present.")
        if split_overlap:
            blocking.append("Split date ranges overlap.")
        return {
            "forbidden_columns_in_features": forbidden,
            "candidate_list_dependent_columns_in_features": sorted(set(candidate + candidate_in_dataset)),
            "label_like_columns_in_features": label_like,
            "future_return_in_features": [column for column in feature_columns if column.startswith("future_")],
            "target_columns_in_features": [column for column in feature_columns if "target" in column.lower()],
            "selected_count_in_day_found": selected_count,
            "split_overlap": split_overlap,
            "train_threshold_only": True,
            "leakage_risk": "high" if blocking else "low",
            "blocking_issues": blocking,
        }

    def _split_overlap(self, dataset: pd.DataFrame) -> bool:
        if dataset.empty or "split" not in dataset.columns or "as_of_date" not in dataset.columns:
            return False
        ranges = []
        dates = pd.to_datetime(dataset["as_of_date"], errors="coerce")
        for split, group in dataset.assign(_date=dates).groupby("split", dropna=False):
            ranges.append((group["_date"].min(), group["_date"].max(), split))
        for index, left in enumerate(ranges):
            for right in ranges[index + 1 :]:
                if pd.notna(left[0]) and pd.notna(right[0]) and left[0] <= right[1] and right[0] <= left[1]:
                    return True
        return False

    def _root(self, path: Path) -> Path:
        if path.is_absolute():
            try:
                return self.root / path.relative_to(ROOT)
            except ValueError:
                return path
        return self.root / path

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value[:8])
                    if len(row.get(column, [])) > 8:
                        value += ", ..."
                values.append(str(value).replace("\n", " "))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)


def run(root: Path | str = ROOT, options: TrainOptions | None = None) -> Phase7EPaths:
    trainer = Phase7EPMAIAPIOnlyTrainerPrototype(root)
    return trainer.save_report(trainer.run(options))
