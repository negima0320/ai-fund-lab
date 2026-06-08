"""Phase 9-D PM AI v3 trainer prototype.

Research-only trainer for the clean PM AI v3 dataset. It trains candidate
models under a leakage guard and saves only to the Phase 9-D candidate
directory. It does not integrate with strategy profiles or overwrite current PM
AI / Exit AI artifacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.portfolio_manager_v3_dataset_builder import (
    CONDITIONAL_RELATIVE_FEATURES,
    FINANCIAL_FEATURES,
    FORBIDDEN_TOKENS,
    LABEL_COLUMNS,
    MARKET_FEATURES,
    PRICE_VOLUME_FEATURES,
    STOCK_PREDICTION_FEATURES,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase9d_pm_ai_v3_trainer_2023-01_to_2026-05"
DATASET_PATH = Path("data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_2023-01_to_2026-05.parquet")
MODEL_DIR = Path("models/ml/portfolio_manager_v3/candidate_phase9d")

MANDATORY_DROP_FEATURES = [
    "days_after_earnings",
    "days_to_earnings",
    "FOP_growth",
    "FSales_growth",
    "FEPS_growth",
    "PayoutRatioAnn",
]
KEY_COLUMNS = {"prediction_date", "code", "market_date", "market_regime_key", "data_source", "relative_feature_timing"}
MODEL_TARGETS = {
    "model_a_candidate_ranking_regressor": {
        "target": "relative_future_utility_percentile_in_day",
        "kind": "regression",
        "filename": "model_a_candidate_ranking_regressor.joblib",
    },
    "model_b_downside_utility_regressor": {
        "target": "downside_penalized_return_10d",
        "kind": "regression",
        "filename": "model_b_downside_utility_regressor.joblib",
    },
    "model_c_top_utility_classifier": {
        "target": "top_decile_future_utility_in_day",
        "kind": "classification",
        "filename": "model_c_top_utility_classifier.joblib",
    },
}


@dataclass(frozen=True)
class Phase9DTrainPaths:
    model_dir: Path
    markdown: Path
    json: Path


@dataclass(frozen=True)
class Phase9DTrainOptions:
    save_models: bool = True
    include_market_comparison: bool = True


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if getattr(series, "dtype", None) == bool:
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _safe_corr(left: pd.Series | None, right: pd.Series | None, method: str = "pearson") -> float | None:
    frame = pd.DataFrame({"left": _numeric(left), "right": _numeric(right)}).dropna()
    if len(frame) < 3 or frame["left"].nunique() <= 1 or frame["right"].nunique() <= 1:
        return None
    value = frame["left"].corr(frame["right"], method=method)
    return None if pd.isna(value) else float(value)


class PMAIV3TrainerPrototype:
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

    def run(self, options: Phase9DTrainOptions | None = None) -> dict[str, Any]:
        options = options or Phase9DTrainOptions()
        dataset = _read_parquet(self.dataset_path)
        feature_plan = self._feature_plan(dataset)
        feature_columns = feature_plan["feature_columns"]
        splits = self._time_split(dataset)
        split_check = self._split_overlap_check(splits)
        leakage = self._leakage_check(feature_columns)
        metrics: dict[str, Any] = {}
        models: dict[str, Any] = {}
        predictions: dict[str, dict[str, pd.Series]] = {}
        if not dataset.empty and not leakage["blocking_issues"] and feature_columns:
            for model_name, spec in MODEL_TARGETS.items():
                target = spec["target"]
                metrics[model_name], models[model_name], predictions[model_name] = self._train_one(
                    dataset,
                    splits,
                    feature_columns,
                    target,
                    spec["kind"],
                )
        market_comparison = self._market_feature_comparison(dataset, splits, feature_columns) if options.include_market_comparison else {}
        feature_importance = self._feature_importance(dataset, splits, feature_columns, predictions)
        multiplier = self._multiplier_prototype(dataset, splits, predictions)
        saved_paths = self._save_models(models, feature_columns, feature_plan, splits, leakage, metrics) if options.save_models and models else {}
        report = {
            "metadata": {
                "phase": "9-D",
                "trainer_prototype": True,
                "training_executed": bool(models),
                "strategy_integration_executed": False,
                "backtest_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_model_regenerated_historical_predictions": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
            },
            "input_paths": {"dataset": str(self.dataset_path)},
            "model_output": {
                "model_dir": str(self.model_dir),
                "saved_model_paths": saved_paths,
                "candidate_only": True,
            },
            "feature_plan": feature_plan,
            "split": self._split_report(splits, split_check),
            "leakage_checklist": leakage,
            "metrics": metrics,
            "with_market_without_market_comparison": market_comparison,
            "feature_importance": feature_importance,
            "multiplier_prototype": multiplier,
            "verdict": self._verdict(metrics, leakage, multiplier),
        }
        return report

    def save_report(self, report: dict[str, Any]) -> Phase9DTrainPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9DTrainPaths(model_dir=self.model_dir, markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        metric_rows = []
        for model, by_split in report.get("metrics", {}).items():
            for split, values in by_split.items():
                metric_rows.append({"model": model, "split": split, **values})
        importance_rows = []
        for model, rows in report.get("feature_importance", {}).items():
            for row in rows[:10]:
                importance_rows.append({"model": model, **row})
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 9-D PM AI v3 Trainer Prototype",
                "",
                "## Scope",
                "",
                "- research candidate model training/evaluation only",
                "- no strategy integration, no backtest, no current artifact overwrite",
                "",
                "## Feature Plan",
                "",
                self._table([report["feature_plan"]], ["feature_count_after_drops", "dropped_features", "forbidden_feature_count", "label_columns_in_features"]),
                "",
                "## Split",
                "",
                self._table(report["split"]["rows"], ["split", "rows", "date_min", "date_max"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_checklist"]], ["forbidden_feature_count", "forbidden_feature_columns", "label_columns_in_features", "split_overlap", "leakage_risk"]),
                "",
                "## Metrics",
                "",
                self._table(metric_rows, ["model", "split", "mae", "rmse", "correlation", "spearman", "auc", "pr_auc", "precision_at_top10pct", "recall_at_top10pct", "top_decile_lift", "predicted_top10_actual_downside_mean", "overall_downside_mean"]),
                "",
                "## With Market / Without Market",
                "",
                self._table(report.get("with_market_without_market_comparison", {}).get("rows", []), ["model", "split", "with_market_primary_metric", "without_market_primary_metric", "delta"]),
                "",
                "## Feature Importance Top",
                "",
                self._table(importance_rows, ["model", "feature", "importance", "category"]),
                "",
                "## Multiplier Prototype",
                "",
                self._table(report["multiplier_prototype"].get("distribution", []), ["multiplier", "count", "rate", "actual_downside_mean", "actual_rank_percentile_mean"]),
                "",
                "## Verdict",
                "",
                self._table([report["verdict"]], ["phase9e_integration_audit_worth_testing", "recommended_next_phase", "reason"]),
                "",
            ]
        )

    def _root(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def _feature_plan(self, dataset: pd.DataFrame) -> dict[str, Any]:
        if dataset.empty:
            return {"feature_columns": [], "feature_count_after_drops": 0, "dropped_features": [], "forbidden_feature_count": 0, "label_columns_in_features": []}
        feature_candidates = [
            column
            for column in dataset.columns
            if column not in LABEL_COLUMNS
            and column not in KEY_COLUMNS
            and column not in {"market_regime_key", "data_source", "relative_feature_timing"}
        ]
        dropped: dict[str, str] = {}
        features: list[str] = []
        for column in feature_candidates:
            if column in MANDATORY_DROP_FEATURES:
                dropped[column] = "phase9c_high_missing_required_drop"
                continue
            if self._has_forbidden_token(column):
                dropped[column] = "forbidden_token"
                continue
            if self._is_label_like(column):
                dropped[column] = "label_like"
                continue
            values = _numeric(dataset[column])
            if values.isna().mean() > 0.50:
                dropped[column] = "missing_rate_gt_0.50"
                continue
            if values.dropna().empty or values.nunique(dropna=True) <= 1:
                dropped[column] = "constant_or_empty"
                continue
            if values.isin([float("inf"), float("-inf")]).any():
                dropped[column] = "infinite_value"
                continue
            if not pd.api.types.is_numeric_dtype(dataset[column]) and dataset[column].dtype != bool:
                dropped[column] = "non_numeric"
                continue
            features.append(column)
        return {
            "feature_columns": features,
            "feature_count_after_drops": len(features),
            "dropped_features": sorted(dropped),
            "dropped_feature_reasons": dropped,
            "forbidden_feature_count": len([column for column in features if self._has_forbidden_token(column)]),
            "label_columns_in_features": [column for column in features if column in LABEL_COLUMNS or self._is_label_like(column)],
        }

    def _time_split(self, dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if dataset.empty:
            return {"train": dataset, "validation": dataset, "test": dataset}
        data = dataset.copy()
        dates = pd.to_datetime(data["prediction_date"], errors="coerce")
        return {
            "train": data[(dates >= "2023-01-04") & (dates <= "2024-12-31")].copy(),
            "validation": data[(dates >= "2025-01-01") & (dates <= "2025-12-31")].copy(),
            "test": data[(dates >= "2026-01-01") & (dates <= "2026-04-27")].copy(),
        }

    def _split_overlap_check(self, splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
        keys = {
            split: set(zip(frame.get("prediction_date", pd.Series(dtype=str)).astype(str), frame.get("code", pd.Series(dtype=str)).astype(str)))
            for split, frame in splits.items()
        }
        overlap = {
            "train_validation": len(keys["train"] & keys["validation"]),
            "train_test": len(keys["train"] & keys["test"]),
            "validation_test": len(keys["validation"] & keys["test"]),
        }
        return {"overlap_counts": overlap, "split_overlap": any(value > 0 for value in overlap.values())}

    def _train_one(
        self,
        dataset: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
        features: list[str],
        target: str,
        kind: str,
    ) -> tuple[dict[str, Any], Any, dict[str, pd.Series]]:
        train = splits["train"].dropna(subset=[target]).copy()
        valid = splits["validation"].dropna(subset=[target]).copy()
        test = splits["test"].dropna(subset=[target]).copy()
        if train.empty or valid.empty or test.empty:
            return {"training_status": "missing_split_rows"}, None, {}
        x_train = train[features]
        y_train = _numeric(train[target])
        if kind == "classification":
            y_train = y_train.astype(int)
            model = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.06, random_state=42)
        else:
            model = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.06, random_state=42)
        model.fit(x_train, y_train)
        metrics = {
            split: self._evaluate_split(frame.dropna(subset=[target]), features, target, model, kind)
            for split, frame in {"train": train, "validation": valid, "test": test}.items()
        }
        preds = {
            split: pd.Series(self._predict(model, frame[features], kind), index=frame.index)
            for split, frame in {"train": train, "validation": valid, "test": test}.items()
        }
        return metrics, model, preds

    def _evaluate_split(self, frame: pd.DataFrame, features: list[str], target: str, model: Any, kind: str) -> dict[str, Any]:
        y = _numeric(frame[target])
        pred = pd.Series(self._predict(model, frame[features], kind), index=frame.index)
        downside = _numeric(frame.get("downside_penalized_return_10d"))
        top10_mask = self._top_fraction_mask(pred, 0.10)
        base = {
            "rows": int(len(frame)),
            "target_null_dropped_count": 0,
            "predicted_top10_actual_downside_mean": float(downside[top10_mask].mean()) if top10_mask.any() else None,
            "overall_downside_mean": float(downside.mean()) if not downside.dropna().empty else None,
        }
        if kind == "classification":
            y_int = y.astype(int)
            pred_label = top10_mask.astype(int)
            positive_rate = float(y_int.mean()) if len(y_int) else None
            return {
                **base,
                "auc": self._safe_auc(y_int, pred),
                "pr_auc": self._safe_pr_auc(y_int, pred),
                "precision_at_top10pct": float(precision_score(y_int, pred_label, zero_division=0)),
                "recall_at_top10pct": float(recall_score(y_int, pred_label, zero_division=0)),
                "top_decile_lift": (float(y_int[top10_mask].mean()) / positive_rate) if top10_mask.any() and positive_rate else None,
                "positive_rate": positive_rate,
            }
        error = pred - y
        return {
            **base,
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(math.sqrt(mean_squared_error(y, pred))),
            "correlation": _safe_corr(pred, y),
            "spearman": _safe_corr(pred, y, method="spearman"),
            **self._top_rank_quality(frame, pred),
        }

    def _top_rank_quality(self, frame: pd.DataFrame, pred: pd.Series) -> dict[str, Any]:
        work = frame[["prediction_date", "relative_future_utility_rank_in_day", "downside_penalized_return_10d"]].copy()
        work["_pred"] = pred
        top1 = []
        top3 = []
        for _, group in work.groupby("prediction_date"):
            ranked = group.sort_values("_pred", ascending=False)
            if not ranked.empty:
                top1.append(float(ranked.iloc[0]["downside_penalized_return_10d"]))
            top3.extend(pd.to_numeric(ranked.head(3)["downside_penalized_return_10d"], errors="coerce").dropna().tolist())
        return {
            "top1_hit_quality_mean": float(pd.Series(top1).mean()) if top1 else None,
            "top3_hit_quality_mean": float(pd.Series(top3).mean()) if top3 else None,
        }

    def _market_feature_comparison(self, dataset: pd.DataFrame, splits: dict[str, pd.DataFrame], features: list[str]) -> dict[str, Any]:
        without_market = [feature for feature in features if feature not in MARKET_FEATURES and not feature.startswith("topix_") and not feature.startswith("market_")]
        rows = []
        if not without_market or len(without_market) == len(features):
            return {"rows": rows, "note": "market feature comparison skipped or no market features present"}
        for model_name, spec in MODEL_TARGETS.items():
            full_metrics, _, _ = self._train_one(dataset, splits, features, spec["target"], spec["kind"])
            no_metrics, _, _ = self._train_one(dataset, splits, without_market, spec["target"], spec["kind"])
            for split in ["validation", "test"]:
                full = self._primary_metric(full_metrics.get(split, {}), spec["kind"])
                no = self._primary_metric(no_metrics.get(split, {}), spec["kind"])
                rows.append(
                    {
                        "model": model_name,
                        "split": split,
                        "with_market_primary_metric": full,
                        "without_market_primary_metric": no,
                        "delta": (None if full is None or no is None else full - no),
                    }
                )
        return {"rows": rows, "note": "primary metric is spearman for regressors and PR-AUC for classifier"}

    def _feature_importance(
        self,
        dataset: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
        features: list[str],
        predictions: dict[str, dict[str, pd.Series]],
    ) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        valid = splits.get("validation", pd.DataFrame())
        for model_name, by_split in predictions.items():
            pred = by_split.get("validation")
            if pred is None or valid.empty:
                result[model_name] = []
                continue
            rows = []
            for feature in features:
                importance = abs(_safe_corr(valid[feature], pred) or 0.0)
                rows.append({"feature": feature, "importance": float(importance), "category": self._feature_category(feature)})
            result[model_name] = sorted(rows, key=lambda row: row["importance"], reverse=True)[:30]
        return result

    def _multiplier_prototype(self, dataset: pd.DataFrame, splits: dict[str, pd.DataFrame], predictions: dict[str, dict[str, pd.Series]]) -> dict[str, Any]:
        test = splits.get("test", pd.DataFrame()).copy()
        if test.empty:
            return {"distribution": []}
        rank_pred = predictions.get("model_a_candidate_ranking_regressor", {}).get("test")
        utility_pred = predictions.get("model_b_downside_utility_regressor", {}).get("test")
        top_pred = predictions.get("model_c_top_utility_classifier", {}).get("test")
        if rank_pred is None or utility_pred is None or top_pred is None:
            return {"distribution": []}
        work = test.copy()
        work["_rank_pred"] = rank_pred
        work["_utility_pred"] = utility_pred
        work["_top_pred"] = top_pred
        pct = work["_rank_pred"].rank(pct=True)
        utility_med = work["_utility_pred"].median()
        work["pm_v3_multiplier_prototype"] = 1.00
        work.loc[pct >= 0.90, "pm_v3_multiplier_prototype"] = 1.15
        work.loc[(pct >= 0.90) & (work["_utility_pred"] >= utility_med) & (work["_top_pred"] >= work["_top_pred"].quantile(0.70)), "pm_v3_multiplier_prototype"] = 1.30
        work.loc[pct <= 0.25, "pm_v3_multiplier_prototype"] = 0.80
        work.loc[(pct <= 0.10) | (work["_utility_pred"] <= work["_utility_pred"].quantile(0.10)), "pm_v3_multiplier_prototype"] = 0.60
        rows = []
        for multiplier, group in work.groupby("pm_v3_multiplier_prototype"):
            rows.append(
                {
                    "multiplier": float(multiplier),
                    "count": int(len(group)),
                    "rate": float(len(group) / len(work)),
                    "actual_downside_mean": float(_numeric(group.get("downside_penalized_return_10d")).mean()),
                    "actual_rank_percentile_mean": float(_numeric(group.get("relative_future_utility_percentile_in_day")).mean()),
                }
            )
        top10 = self._top_fraction_mask(work["_rank_pred"], 0.10)
        return {
            "distribution": sorted(rows, key=lambda row: row["multiplier"], reverse=True),
            "predicted_top10_actual_downside_penalized_return_10d": float(_numeric(work.loc[top10, "downside_penalized_return_10d"]).mean()) if top10.any() else None,
            "overall_actual_downside_penalized_return_10d": float(_numeric(work["downside_penalized_return_10d"]).mean()),
            "prototype_for_research_only": True,
        }

    def _save_models(
        self,
        models: dict[str, Any],
        features: list[str],
        feature_plan: dict[str, Any],
        splits: dict[str, pd.DataFrame],
        leakage: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, str]:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for model_name, model in models.items():
            if model is None:
                continue
            path = self.model_dir / MODEL_TARGETS[model_name]["filename"]
            joblib.dump(model, path)
            paths[model_name] = str(path)
        (self.model_dir / "feature_columns.json").write_text(json.dumps(features, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata = {
            "phase": "9-D",
            "dataset": str(self.dataset_path),
            "feature_count": len(features),
            "dropped_features": feature_plan["dropped_features"],
            "split_rows": {split: int(len(frame)) for split, frame in splits.items()},
            "leakage_checklist": leakage,
            "metrics": metrics,
            "current_pm_ai_overwritten": False,
            "current_exit_ai_overwritten": False,
            "v2_82_profile_overwritten": False,
        }
        (self.model_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        paths["feature_columns"] = str(self.model_dir / "feature_columns.json")
        paths["training_metadata"] = str(self.model_dir / "training_metadata.json")
        return paths

    def _split_report(self, splits: dict[str, pd.DataFrame], check: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for split, frame in splits.items():
            dates = pd.to_datetime(frame.get("prediction_date", pd.Series(dtype=str)), errors="coerce")
            rows.append(
                {
                    "split": split,
                    "rows": int(len(frame)),
                    "date_min": dates.min().strftime("%Y-%m-%d") if not dates.dropna().empty else None,
                    "date_max": dates.max().strftime("%Y-%m-%d") if not dates.dropna().empty else None,
                }
            )
        return {"rows": rows, **check}

    def _leakage_check(self, features: list[str]) -> dict[str, Any]:
        forbidden = [feature for feature in features if self._has_forbidden_token(feature)]
        label_like = [feature for feature in features if self._is_label_like(feature)]
        return {
            "forbidden_feature_count": len(forbidden),
            "forbidden_feature_columns": forbidden,
            "label_columns_in_features": label_like,
            "split_overlap": False,
            "leakage_risk": "high" if forbidden or label_like else "low",
            "blocking_issues": [issue for issue, present in [("forbidden_feature", bool(forbidden)), ("label_in_feature", bool(label_like))] if present],
        }

    def _predict(self, model: Any, x: pd.DataFrame, kind: str) -> Any:
        if kind == "classification":
            if hasattr(model, "predict_proba"):
                return model.predict_proba(x)[:, 1]
        return model.predict(x)

    def _safe_auc(self, y: pd.Series, pred: pd.Series) -> float | None:
        return None if y.nunique() < 2 else float(roc_auc_score(y, pred))

    def _safe_pr_auc(self, y: pd.Series, pred: pd.Series) -> float | None:
        return None if y.nunique() < 2 else float(average_precision_score(y, pred))

    def _top_fraction_mask(self, score: pd.Series, fraction: float) -> pd.Series:
        if score.empty:
            return pd.Series(False, index=score.index)
        n = max(1, math.ceil(len(score) * fraction))
        cutoff = score.rank(method="first", ascending=False).le(n)
        return cutoff.reindex(score.index).fillna(False)

    def _primary_metric(self, metrics: dict[str, Any], kind: str) -> float | None:
        return metrics.get("pr_auc") if kind == "classification" else metrics.get("spearman")

    def _feature_category(self, feature: str) -> str:
        if feature in STOCK_PREDICTION_FEATURES:
            return "stock_selection_prediction"
        if feature in MARKET_FEATURES or feature.startswith("topix_") or feature.startswith("market_"):
            return "market"
        if feature in PRICE_VOLUME_FEATURES:
            return "price_volume"
        if feature in FINANCIAL_FEATURES:
            return "financial"
        if feature in CONDITIONAL_RELATIVE_FEATURES:
            return "conditional_relative"
        return "other"

    def _verdict(self, metrics: dict[str, Any], leakage: dict[str, Any], multiplier: dict[str, Any]) -> dict[str, Any]:
        top10_downside = multiplier.get("predicted_top10_actual_downside_penalized_return_10d")
        overall = multiplier.get("overall_actual_downside_penalized_return_10d")
        worth = bool(leakage["leakage_risk"] == "low" and top10_downside is not None and overall is not None and top10_downside > overall)
        return {
            "phase9e_integration_audit_worth_testing": worth,
            "recommended_next_phase": "Phase 9-E: PM AI v3 Candidate Integration Audit" if worth else "Phase 9-D2: Trainer/label refinement",
            "reason": "predicted top10 downside utility beats dataset average" if worth else "top10 utility lift is not strong enough yet or leakage blocked",
        }

    def _has_forbidden_token(self, column: str) -> bool:
        lowered = column.lower()
        return any(token in lowered for token in FORBIDDEN_TOKENS)

    def _is_label_like(self, column: str) -> bool:
        lowered = column.lower()
        return lowered.startswith("future_") or "label" in lowered or "target" in lowered or column in LABEL_COLUMNS

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(self._format(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, list):
            return ", ".join(str(item) for item in value[:12])
        if isinstance(value, dict):
            return ", ".join(f"{key}:{val}" for key, val in value.items())
        return str(value).replace("\n", " ")


def train_phase9d_pm_ai_v3(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3TrainerPrototype(root).run()

