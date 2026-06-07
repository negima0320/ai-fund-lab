"""Phase 5-E Exit AI v2 trainer prototype.

The prototype uses the Phase 5-C API-only dataset and trains a lightweight
ranking-style top-decile classifier only when requested. It never overwrites the
current Exit AI model path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase5e_exit_ai_v2_trainer_prototype_2021-06_to_2026-05"
DATASET_PATH = ROOT / "data" / "ml" / "exit_ai_v2" / "exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet"
MODEL_DIR = ROOT / "models" / "ml" / "exit_ai_v2" / "candidate_v2_api_only"

DROP_MISSING_30PCT_FEATURES = {"BPS", "OP_growth", "FEPS_growth", "FSales_growth", "FOP_growth"}
KEY_COLUMNS = {"code", "as_of_date", "split"}
LABEL_COLUMNS = {
    "future_return_3d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "avoid_loss_5d",
    "miss_profit_5d",
    "exit_quality_score",
    "exit_quality_score_risk_adjusted",
    "future_max_drawdown_5d",
    "future_max_drawdown_10d",
    "future_max_return_5d",
    "future_max_return_10d",
}
FORBIDDEN_COLUMNS = {
    "trade_id",
    "actual_exit_date",
    "actual_sell_price",
    "realized_profit",
    "realized_return",
    "win",
    "loss",
    "holding_days",
    "remaining_days_to_actual_exit",
    "exit_reason",
    "selected_count_in_day",
    "portfolio_cash",
    "total_assets",
    "market_value",
    "profile_id",
}


@dataclass(frozen=True)
class TrainOptions:
    dry_run: bool = True
    sample_rows: int | None = None
    train_full: bool = False


@dataclass(frozen=True)
class Phase5EPaths:
    model_dir: Path | None
    markdown: Path
    json: Path


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _is_forbidden(column: str) -> bool:
    lower = column.lower()
    return column in FORBIDDEN_COLUMNS or "backtest" in lower or "profile" in lower or "realized" in lower


def _is_label_like(column: str) -> bool:
    lower = column.lower()
    return (
        column in LABEL_COLUMNS
        or lower.startswith(("future_return_", "future_max_", "avoid_loss_", "miss_profit_"))
        or "exit_quality_score" in lower
        or "target" in lower
        or "label" in lower
    )


def _feature_columns(columns: list[str]) -> list[str]:
    features = []
    for column in columns:
        if column in KEY_COLUMNS or column in DROP_MISSING_30PCT_FEATURES:
            continue
        if _is_forbidden(column) or _is_label_like(column):
            continue
        features.append(column)
    return features


def _series_stats(series: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p10": float(clean.quantile(0.10)),
        "p90": float(clean.quantile(0.90)),
    }


class Phase5EExitAIV2TrainerPrototype:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        dataset_path: Path | None = None,
        model_dir: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.dataset_path = self._root_path(dataset_path or DATASET_PATH)
        self.model_dir = self._root_path(model_dir or MODEL_DIR)

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
            report["next_recommended_action"] = "Fix leakage blocking issues before training"
            return report
        train = dataset[dataset["split"].eq("train")].copy()
        valid = dataset[dataset["split"].eq("validation")].copy()
        test = dataset[dataset["split"].eq("test")].copy()
        if train.empty or valid.empty or test.empty:
            report["training_status"] = {"trained": False, "reason": "missing train/validation/test split rows"}
            return report
        threshold = float(train["exit_quality_score"].quantile(0.90))
        train = self._assign_target(train, threshold)
        valid = self._assign_target(valid, threshold)
        test = self._assign_target(test, threshold)
        preprocess = self._fit_preprocess(train, feature_columns)
        x_train = self._transform(train, preprocess)
        x_valid = self._transform(valid, preprocess)
        x_test = self._transform(test, preprocess)
        y_train = train["exit_quality_top_decile"].astype(int)
        y_valid = valid["exit_quality_top_decile"].astype(int)
        y_test = test["exit_quality_top_decile"].astype(int)
        model = self._make_model()
        model.fit(x_train, y_train)
        valid_scores = self._predict_scores(model, x_valid)
        test_scores = self._predict_scores(model, x_test)
        model_saved = False
        model_save_path = None
        model_save_reason = "Phase 5-E prototype sample training does not save models."
        if options.train_full:
            model_save_path = self._save_candidate_model(
                model,
                preprocess,
                {
                    "label_source": "exit_quality_score",
                    "target_column": "exit_quality_top_decile",
                    "train_top_decile_threshold": threshold,
                    "threshold_fit_scope": "train split only",
                    "feature_columns": feature_columns,
                    "model": "sklearn.ensemble.HistGradientBoostingClassifier",
                    "current_model_overwrite_forbidden": True,
                },
            )
            model_saved = True
            model_save_reason = "Saved only to candidate Exit AI v2 directory; current model path was not touched."
        report.update(
            {
                "training_status": {
                    "trained": True,
                    "model": "sklearn.ensemble.HistGradientBoostingClassifier",
                    "model_saved": model_saved,
                    "model_save_path": str(model_save_path) if model_save_path else None,
                    "model_save_note": model_save_reason,
                },
                "target": {
                    "label_source": "exit_quality_score",
                    "target_column": "exit_quality_top_decile",
                    "train_top_decile_threshold": threshold,
                    "threshold_fit_scope": "train split only",
                    "train_positive_rate": float(y_train.mean()),
                    "validation_positive_rate": float(y_valid.mean()),
                    "test_positive_rate": float(y_test.mean()),
                },
                "preprocess": self._preprocess_report(preprocess),
                "metrics": {
                    "validation": self._metrics(y_valid, valid_scores),
                    "test": self._metrics(y_test, test_scores),
                },
                "feature_importance": self._feature_importance(model, list(x_train.columns)),
                "next_recommended_action": "Proceed to Exit AI v2 Prediction / Integration Audit." if options.train_full else "Review sample metrics, then decide whether Phase 5-F full walk-forward training is warranted.",
            }
        )
        report["model_output"]["model_saved"] = model_saved
        report["model_output"]["model_dir"] = str(self.model_dir) if model_saved else None
        return report

    def save_report(self, report: dict[str, Any]) -> Phase5EPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase5EPaths(model_dir=None, markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# AI Retraining Phase 5-E Exit AI v2 Trainer Prototype",
            "",
            "## Scope",
            "",
            "- trainer prototype only",
            "- current model overwrite forbidden",
            "- no backtest/profile/live order integration",
            "",
            "## Dataset",
            "",
            self._table([report["dataset"]], ["path", "rows_used", "feature_count", "train_rows", "validation_rows", "test_rows"]),
            "",
            "## Leakage",
            "",
            self._table([report["leakage_check"]], ["forbidden_columns_in_features", "label_like_columns_in_features", "split_overlap", "train_threshold_only", "leakage_risk", "blocking_issues"]),
            "",
            "## Target",
            "",
            self._table([report.get("target", {})], ["label_source", "target_column", "train_top_decile_threshold", "threshold_fit_scope", "train_positive_rate", "validation_positive_rate", "test_positive_rate"]),
            "",
            "## Metrics",
            "",
            self._table([{"split": split, **values} for split, values in report.get("metrics", {}).items()], ["split", "auc", "pr_auc", "precision_at_top10pct", "recall_at_top10pct", "top_decile_lift", "positive_rate"]),
            "",
            f"Next recommended action: `{report.get('next_recommended_action')}`",
            "",
        ]
        return "\n".join(lines)

    def _base_report(self, dataset: pd.DataFrame, features: list[str], options: TrainOptions, leakage: dict[str, Any]) -> dict[str, Any]:
        split_counts = dataset["split"].value_counts().to_dict() if "split" in dataset.columns else {}
        return {
            "metadata": {
                "phase": "5-E",
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
            "feature_policy": {
                "feature_set": "feature_set_drop_missing_30pct",
                "dropped_features": sorted(DROP_MISSING_30PCT_FEATURES),
                "feature_columns": features,
            },
            "target": {
                "label_source": "exit_quality_score",
                "target_column": "exit_quality_top_decile",
                "threshold_fit_scope": "train split only",
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
            },
        }

    def _limit_rows(self, dataset: pd.DataFrame, options: TrainOptions) -> pd.DataFrame:
        if dataset.empty or options.train_full or not options.sample_rows:
            return dataset.copy()
        rows = []
        per_split = max(1, options.sample_rows // 3)
        for split_name in ["train", "validation", "test"]:
            part = dataset[dataset["split"].eq(split_name)].head(per_split)
            rows.append(part)
        sample = pd.concat(rows, ignore_index=True)
        if len(sample) < options.sample_rows:
            remainder = dataset.drop(sample.index, errors="ignore").head(options.sample_rows - len(sample))
            sample = pd.concat([sample, remainder], ignore_index=True)
        return sample.head(options.sample_rows).copy()

    def _assign_target(self, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
        result = frame.copy()
        result["exit_quality_top_decile"] = pd.to_numeric(result["exit_quality_score"], errors="coerce").ge(threshold).astype(int)
        return result

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
        return {
            "feature_columns": feature_columns,
            "numeric_columns": numeric,
            "categorical_columns": categorical,
            "medians": medians,
            "modes": modes,
            "missing_indicator_columns": indicators,
        }

    def _transform(self, frame: pd.DataFrame, preprocess: dict[str, Any]) -> pd.DataFrame:
        columns = preprocess["feature_columns"]
        result = frame[columns].copy()
        transformed: dict[str, Any] = {}
        for column in preprocess["numeric_columns"]:
            transformed[column] = pd.to_numeric(result[column], errors="coerce").fillna(preprocess["medians"].get(column, 0.0))
        for column in preprocess["categorical_columns"]:
            filled = result[column].fillna(preprocess["modes"].get(column, "")).astype("category")
            transformed[column] = filled.cat.codes.astype(float)
        output = pd.DataFrame(transformed, index=frame.index)
        for column in preprocess["missing_indicator_columns"]:
            output[f"{column}_missing"] = result[column].isna().astype(int)
        return output

    def _make_model(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.08, max_leaf_nodes=31, random_state=42)

    def _predict_scores(self, model: Any, features: pd.DataFrame) -> pd.Series:
        if hasattr(model, "predict_proba"):
            return pd.Series(model.predict_proba(features)[:, 1], index=features.index)
        return pd.Series(model.predict(features), index=features.index)

    def _metrics(self, y_true: pd.Series, scores: pd.Series) -> dict[str, Any]:
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
        precision_top = float(precision_score(y, top_pred, zero_division=0))
        recall_top = float(recall_score(y, top_pred, zero_division=0))
        top_rate = float(y.loc[top_index].mean()) if len(top_index) else 0.0
        deciles = self._calibration_by_decile(y, score)
        return {
            "auc": auc,
            "pr_auc": pr_auc,
            "precision_at_top10pct": precision_top,
            "recall_at_top10pct": recall_top,
            "top_decile_lift": (top_rate / positive_rate) if positive_rate else None,
            "positive_rate": positive_rate,
            "prediction_score": _series_stats(score),
            "calibration_by_decile": deciles,
        }

    def _calibration_by_decile(self, y_true: pd.Series, scores: pd.Series) -> list[dict[str, Any]]:
        frame = pd.DataFrame({"actual": y_true.astype(int), "score": scores})
        frame["decile"] = pd.qcut(frame["score"].rank(method="first"), 10, labels=False, duplicates="drop")
        rows = []
        for decile, group in frame.groupby("decile", dropna=False):
            rows.append(
                {
                    "decile": int(decile) if pd.notna(decile) else None,
                    "rows": int(len(group)),
                    "score_mean": float(group["score"].mean()),
                    "actual_positive_rate": float(group["actual"].mean()),
                }
            )
        return rows

    def _feature_importance(self, model: Any, columns: list[str]) -> dict[str, Any]:
        if hasattr(model, "feature_importances_"):
            values = getattr(model, "feature_importances_")
            rows = sorted(
                [{"feature": column, "importance": float(value)} for column, value in zip(columns, values)],
                key=lambda row: row["importance"],
                reverse=True,
            )
            return {"available": True, "top20": rows[:20]}
        return {"available": False, "reason": "HistGradientBoostingClassifier does not expose impurity feature_importances_."}

    def _save_candidate_model(self, model: Any, preprocess: dict[str, Any], metadata: dict[str, Any]) -> Path:
        import joblib

        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "exit_quality_top_decile_classifier.joblib"
        joblib.dump(model, model_path)
        (self.model_dir / "preprocess.json").write_text(json.dumps(preprocess, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (self.model_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (self.model_dir / "feature_columns.json").write_text(json.dumps(metadata["feature_columns"], ensure_ascii=False, indent=2), encoding="utf-8")
        return model_path

    def _preprocess_report(self, preprocess: dict[str, Any]) -> dict[str, Any]:
        return {
            "numeric_strategy": "median",
            "categorical_strategy": "mode",
            "fit_scope": "train split only",
            "numeric_columns": preprocess["numeric_columns"],
            "categorical_columns": preprocess["categorical_columns"],
            "columns_to_impute": sorted(
                [*preprocess["numeric_columns"], *preprocess["categorical_columns"]]
            ),
            "missing_indicator_columns": preprocess["missing_indicator_columns"],
            "scaler": "none",
        }

    def _leakage_check(self, dataset: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
        forbidden = [column for column in feature_columns if _is_forbidden(column)]
        label_like = [column for column in feature_columns if _is_label_like(column)]
        selected_count = "selected_count_in_day" in dataset.columns or "selected_count_in_day" in feature_columns
        split_overlap = self._split_overlap(dataset)
        blocking = []
        if forbidden:
            blocking.append("Forbidden columns remain in features.")
        if label_like:
            blocking.append("Label-like columns remain in features.")
        if selected_count:
            blocking.append("selected_count_in_day is present.")
        if split_overlap:
            blocking.append("Split date ranges overlap.")
        return {
            "forbidden_columns_in_features": forbidden,
            "label_like_columns_in_features": label_like,
            "future_return_in_features": [column for column in feature_columns if column.startswith("future_return_")],
            "target_or_label_in_features": [column for column in feature_columns if "target" in column.lower() or "label" in column.lower()],
            "selected_count_in_day_found": selected_count,
            "backtest_result_profile_columns_found": [
                column for column in dataset.columns if "backtest" in column.lower() or "profile" in column.lower()
            ],
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

    def _root_path(self, path: Path) -> Path:
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


def run(root: Path | str = ROOT, options: TrainOptions | None = None) -> Phase5EPaths:
    trainer = Phase5EExitAIV2TrainerPrototype(root)
    return trainer.save_report(trainer.run(options))
