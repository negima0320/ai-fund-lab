"""Phase 11-B Valuation Engine prototype trainer.

This module trains a lightweight, research-only valuation prototype from the
Phase 11-A dataset. It does not run strategy backtests, add profiles, overwrite
current models, regenerate historical predictions, or use backtest/trade
outcomes as features.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11a_valuation_dataset_audit import FEATURE_CANDIDATES, STOCK_SELECTION_BASE_FEATURES


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = Path("data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet")
MODEL_DIR = Path("models/ml/valuation_engine/candidate_phase11b")
REPORT_STEM = "phase11b_valuation_engine_prototype_2025_holdout"

TRAIN_START = "2023-01-04"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

REGRESSION_TARGET = "opportunity_value_20d"
CLASSIFICATION_TARGET = "opportunity_top_decile_20d"
OUTPUT_QUALITY_LABELS = [
    "opportunity_value_20d",
    "future_max_return_20d",
    "future_max_drawdown_20d",
    "opportunity_top_decile_20d",
]

TARGET_COLUMNS = {
    "future_return_20d",
    "future_max_return_20d",
    "future_max_drawdown_20d",
    "opportunity_value_20d",
    "opportunity_top_decile_20d",
}

FORBIDDEN_TOKENS = {
    "backtest",
    "trade",
    "trades",
    "profit",
    "loss",
    "cash",
    "portfolio",
    "position",
    "selected",
    "bought",
    "affordable",
    "skip",
    "exit",
    "final_assets",
    "pm_multiplier",
    "current_pm",
    "future_return",
    "future_max_return",
    "future_max_drawdown",
    "opportunity_value",
    "opportunity_top_decile",
}
ALWAYS_FORBIDDEN_COLUMNS = {"selected_count_in_day", "current_pm_multiplier", "pm_multiplier"}


@dataclass(frozen=True)
class Phase11BOptions:
    max_train_rows: int = 250_000
    max_test_rows: int | None = None
    random_state: int = 42
    max_iter: int = 80
    learning_rate: float = 0.06
    save_model: bool = True


@dataclass(frozen=True)
class Phase11BPaths:
    markdown: Path
    json: Path
    model_dir: Path | None


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _safe_corr(left: pd.Series | None, right: pd.Series | None, *, method: str = "pearson") -> float | None:
    if left is None or right is None:
        return None
    frame = pd.DataFrame({"left": _numeric(left), "right": _numeric(right)}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    value = frame["left"].corr(frame["right"], method=method)
    return _safe_float(value)


class Phase11BValuationEnginePrototype:
    def __init__(self, root: Path | str = ROOT, *, options: Phase11BOptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase11BOptions()

    def run(self) -> Phase11BPaths:
        report, models = self.build_report_and_models()
        return self.save_outputs(report, models)

    def build_report_and_models(self) -> tuple[dict[str, Any], dict[str, Any]]:
        dataset = self.load_dataset()
        feature_columns = self.extract_feature_columns(dataset)
        leakage = self.leakage_checklist(feature_columns)
        train, test = self.split_dataset(dataset)
        report_base = {
            "metadata": self.metadata(),
            "dataset": self.dataset_summary(dataset, train, test),
            "feature_policy": {
                "feature_columns": feature_columns,
                "feature_count": len(feature_columns),
                "numeric_only": True,
                "target_columns_excluded": sorted(TARGET_COLUMNS),
            },
            "leakage_checklist": leakage,
        }
        if leakage["blocking_issues"]:
            report_base["recommendation"] = self.recommendation(False, leakage, [], {})
            return report_base, {}

        prepared_train, prepared_test = self.prepare_frames(train, test, feature_columns)
        models = self.train_models(prepared_train, feature_columns)
        predictions = self.predict(models, prepared_test, feature_columns)
        regression = self.regression_metrics(prepared_test, predictions)
        classification = self.classification_metrics(prepared_test, predictions)
        output_quality = self.output_quality(prepared_test, predictions)
        report = {
            **report_base,
            "model_config": {
                "regression_model": "HistGradientBoostingRegressor",
                "classification_model": "HistGradientBoostingClassifier",
                "max_iter": self.options.max_iter,
                "learning_rate": self.options.learning_rate,
                "max_train_rows": self.options.max_train_rows,
                "max_test_rows": self.options.max_test_rows,
                "random_state": self.options.random_state,
            },
            "valuation_output_definition": self.valuation_output_definition(),
            "regression_metrics": regression,
            "classification_metrics": classification,
            "output_quality": output_quality,
            "recommendation": self.recommendation(True, leakage, output_quality, classification),
        }
        return report, models

    def load_dataset(self) -> pd.DataFrame:
        path = self.root / DATASET_PATH
        data = pd.read_parquet(path)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        if "code" in data.columns:
            data["code"] = data["code"].astype("string")
        return data.dropna(subset=["date", "code", REGRESSION_TARGET, CLASSIFICATION_TARGET]).reset_index(drop=True)

    def split_dataset(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = dataset[(dataset["date"] >= pd.Timestamp(TRAIN_START)) & (dataset["date"] <= pd.Timestamp(TRAIN_END))].copy()
        test = dataset[(dataset["date"] >= pd.Timestamp(TEST_START)) & (dataset["date"] <= pd.Timestamp(TEST_END))].copy()
        if self.options.max_train_rows and len(train) > self.options.max_train_rows:
            train = train.sample(n=self.options.max_train_rows, random_state=self.options.random_state).sort_values(["date", "code"])
        if self.options.max_test_rows and len(test) > self.options.max_test_rows:
            test = test.sample(n=self.options.max_test_rows, random_state=self.options.random_state).sort_values(["date", "code"])
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def extract_feature_columns(self, dataset: pd.DataFrame) -> list[str]:
        allowed_candidates = self.phase11a_allowed_feature_candidates()
        feature_columns: list[str] = []
        for column in dataset.columns:
            if column in {"date", "code"} or column in TARGET_COLUMNS:
                continue
            if column not in allowed_candidates:
                continue
            if self.is_forbidden_column(column):
                continue
            dtype = dataset[column].dtype
            if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
                feature_columns.append(column)
        return feature_columns

    def phase11a_allowed_feature_candidates(self) -> set[str]:
        columns: set[str] = set()
        for group_columns in FEATURE_CANDIDATES.values():
            columns.update(group_columns)
        for base in STOCK_SELECTION_BASE_FEATURES:
            columns.update(
                {
                    f"{base}_rank_in_day",
                    f"{base}_percentile_in_day",
                    f"{base}_gap_to_best",
                }
            )
        return columns

    def leakage_checklist(self, feature_columns: list[str]) -> dict[str, Any]:
        future_columns = [column for column in feature_columns if any(token in column.lower() for token in ["future_return", "future_max_return", "future_max_drawdown"])]
        forbidden = [column for column in feature_columns if self.is_forbidden_column(column)]
        backtest = [column for column in feature_columns if "backtest" in column.lower()]
        trade = [column for column in feature_columns if any(token in column.lower() for token in ["trade", "profit", "loss"])]
        cash_portfolio = [column for column in feature_columns if any(token in column.lower() for token in ["cash", "portfolio", "position"])]
        blocking = []
        if forbidden:
            blocking.append("forbidden columns selected as features")
        if future_columns:
            blocking.append("future columns selected as features")
        if "selected_count_in_day" in feature_columns:
            blocking.append("selected_count_in_day selected as feature")
        if "pm_multiplier" in feature_columns or "current_pm_multiplier" in feature_columns:
            blocking.append("current PM multiplier selected as feature")
        return {
            "future_columns_in_features": future_columns,
            "forbidden_columns_in_features": forbidden,
            "backtest_columns_in_features": backtest,
            "trade_result_columns_in_features": trade,
            "cash_or_portfolio_columns_in_features": cash_portfolio,
            "current_pm_multiplier_used": "pm_multiplier" in feature_columns or "current_pm_multiplier" in feature_columns,
            "selected_count_in_day_used": "selected_count_in_day" in feature_columns,
            "historical_predictions_regenerated": False,
            "leakage_risk": "high" if blocking else "low",
            "blocking_issues": blocking,
        }

    def is_forbidden_column(self, column: str) -> bool:
        lowered = column.lower()
        if lowered in ALWAYS_FORBIDDEN_COLUMNS:
            return True
        return any(token in lowered for token in FORBIDDEN_TOKENS)

    def prepare_frames(self, train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        prepared = []
        for frame in [train, test]:
            data = frame.copy()
            for column in feature_columns:
                if pd.api.types.is_bool_dtype(data[column].dtype):
                    data[column] = data[column].astype(int)
                else:
                    data[column] = _numeric(data[column])
            data[feature_columns] = data[feature_columns].replace([float("inf"), float("-inf")], pd.NA)
            prepared.append(data)
        return prepared[0], prepared[1]

    def train_models(self, train: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        x_train = train[feature_columns]
        regressor = HistGradientBoostingRegressor(
            max_iter=self.options.max_iter,
            learning_rate=self.options.learning_rate,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            early_stopping=True,
            random_state=self.options.random_state,
        )
        classifier = HistGradientBoostingClassifier(
            max_iter=self.options.max_iter,
            learning_rate=self.options.learning_rate,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            early_stopping=True,
            random_state=self.options.random_state,
        )
        regressor.fit(x_train, _numeric(train[REGRESSION_TARGET]))
        classifier.fit(x_train, train[CLASSIFICATION_TARGET].astype(int))
        return {"regressor": regressor, "classifier": classifier, "feature_columns": feature_columns}

    def predict(self, models: dict[str, Any], test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        x_test = test[feature_columns]
        pred_value = pd.Series(models["regressor"].predict(x_test), index=test.index, name="predicted_opportunity_value")
        proba = models["classifier"].predict_proba(x_test)[:, 1]
        pred_proba = pd.Series(proba, index=test.index, name="opportunity_top_decile_proba")
        score = pred_value.rank(pct=True, method="average") * 100.0
        confidence = (pred_proba - 0.5).abs() * 200.0
        return pd.DataFrame(
            {
                "predicted_opportunity_value": pred_value,
                "opportunity_top_decile_proba": pred_proba,
                "opportunity_score": score.clip(0, 100),
                "expected_upside": pd.NA,
                "expected_downside": pd.NA,
                "confidence": confidence.clip(0, 100),
            }
        )

    def regression_metrics(self, test: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
        actual = _numeric(test[REGRESSION_TARGET])
        pred = _numeric(predictions["predicted_opportunity_value"])
        errors = pred - actual
        deciles = self.prediction_deciles(pred, test, REGRESSION_TARGET, "predicted_opportunity_value")
        return {
            "target": REGRESSION_TARGET,
            "mae": _safe_float(errors.abs().mean()),
            "rmse": _safe_float(math.sqrt(float((errors**2).mean()))),
            "pearson": _safe_corr(pred, actual, method="pearson"),
            "spearman": _safe_corr(pred, actual, method="spearman"),
            "prediction_deciles": deciles,
        }

    def classification_metrics(self, test: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
        from sklearn.metrics import average_precision_score, roc_auc_score

        actual = test[CLASSIFICATION_TARGET].astype(int)
        proba = _numeric(predictions["opportunity_top_decile_proba"])
        auc = None
        pr_auc = None
        if actual.nunique() >= 2:
            auc = _safe_float(roc_auc_score(actual, proba))
            pr_auc = _safe_float(average_precision_score(actual, proba))
        top_n = max(1, int(len(proba) * 0.10))
        top_index = proba.sort_values(ascending=False).head(top_n).index
        deciles = self.prediction_deciles(proba, test, CLASSIFICATION_TARGET, "opportunity_top_decile_proba")
        return {
            "target": CLASSIFICATION_TARGET,
            "auc": auc,
            "pr_auc": pr_auc,
            "precision_at_top10pct": _safe_float(actual.loc[top_index].mean()) if len(top_index) else None,
            "base_positive_rate": _safe_float(actual.mean()),
            "prediction_deciles": deciles,
        }

    def output_quality(self, test: pd.DataFrame, predictions: pd.DataFrame) -> list[dict[str, Any]]:
        frame = pd.concat([test[OUTPUT_QUALITY_LABELS].reset_index(drop=True), predictions[["opportunity_score"]].reset_index(drop=True)], axis=1)
        frame = frame.dropna(subset=["opportunity_score"])
        if frame.empty or frame["opportunity_score"].nunique() < 2:
            return []
        frame["decile"] = pd.qcut(frame["opportunity_score"], q=10, labels=False, duplicates="drop") + 1
        rows = []
        for decile, group in frame.groupby("decile", dropna=True):
            rows.append(
                {
                    "opportunity_score_decile": int(decile),
                    "count": int(len(group)),
                    "actual_opportunity_value_20d_mean": _safe_float(_numeric(group["opportunity_value_20d"]).mean()),
                    "actual_future_max_return_20d_mean": _safe_float(_numeric(group["future_max_return_20d"]).mean()),
                    "actual_future_max_drawdown_20d_mean": _safe_float(_numeric(group["future_max_drawdown_20d"]).mean()),
                    "opportunity_top_decile_20d_rate": _safe_float(_numeric(group["opportunity_top_decile_20d"]).mean()),
                }
            )
        return rows

    def prediction_deciles(self, prediction: pd.Series, test: pd.DataFrame, label: str, prediction_name: str) -> list[dict[str, Any]]:
        frame = pd.DataFrame({"prediction": _numeric(prediction), "actual": _numeric(test[label])}).dropna()
        if frame.empty or frame["prediction"].nunique() < 2:
            return []
        frame["decile"] = pd.qcut(frame["prediction"], q=10, labels=False, duplicates="drop") + 1
        rows = []
        for decile, group in frame.groupby("decile", dropna=True):
            rows.append(
                {
                    "prediction": prediction_name,
                    "decile": int(decile),
                    "count": int(len(group)),
                    "actual_mean": _safe_float(group["actual"].mean()),
                    "prediction_mean": _safe_float(group["prediction"].mean()),
                }
            )
        return rows

    def save_outputs(self, report: dict[str, Any], models: dict[str, Any]) -> Phase11BPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        model_dir = None
        if self.options.save_model and models and not report["leakage_checklist"]["blocking_issues"]:
            model_dir = self.root / MODEL_DIR
            model_dir.mkdir(parents=True, exist_ok=True)
            self.save_models(model_dir, models, report)
        return Phase11BPaths(markdown=md_path, json=json_path, model_dir=model_dir)

    def save_models(self, model_dir: Path, models: dict[str, Any], report: dict[str, Any]) -> None:
        import joblib

        joblib.dump(models["regressor"], model_dir / "opportunity_value_20d_regressor.joblib")
        joblib.dump(models["classifier"], model_dir / "opportunity_top_decile_20d_classifier.joblib")
        feature_columns = models["feature_columns"]
        (model_dir / "feature_columns.json").write_text(json.dumps(feature_columns, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata = {
            "phase": "11-B",
            "model_profile": "candidate_phase11b",
            "research_only": True,
            "train_period": {"start": TRAIN_START, "end": TRAIN_END},
            "test_period": {"start": TEST_START, "end": TEST_END},
            "feature_count": len(feature_columns),
            "regression_target": REGRESSION_TARGET,
            "classification_target": CLASSIFICATION_TARGET,
            "leakage_risk": report["leakage_checklist"]["leakage_risk"],
            "blocking_issues": report["leakage_checklist"]["blocking_issues"],
        }
        (model_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (model_dir / "valuation_output_schema.json").write_text(
            json.dumps(self.valuation_output_definition(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "11-B",
            "prototype_only": True,
            "training_executed": True,
            "backtest_executed": False,
            "profile_added": False,
            "profile_modified": False,
            "current_model_overwritten": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
            "live_order_executed": False,
            "dataset_path": str(self.root / DATASET_PATH),
            "model_dir": str(self.root / MODEL_DIR),
        }

    def dataset_summary(self, dataset: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(dataset)),
            "unique_codes": int(dataset["code"].nunique()) if "code" in dataset else 0,
            "date_range": {
                "min": dataset["date"].min().strftime("%Y-%m-%d") if not dataset.empty else None,
                "max": dataset["date"].max().strftime("%Y-%m-%d") if not dataset.empty else None,
            },
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_period": {"start": TRAIN_START, "end": TRAIN_END},
            "test_period": {"start": TEST_START, "end": TEST_END},
        }

    def valuation_output_definition(self) -> dict[str, Any]:
        return {
            "opportunity_score": "percentile rank of predicted_opportunity_value within the evaluated test frame, scaled to 0-100",
            "predicted_opportunity_value": "regression output for opportunity_value_20d",
            "opportunity_top_decile_proba": "classification probability for opportunity_top_decile_20d",
            "expected_upside": None,
            "expected_upside_status": "not implemented in Phase 11-B to keep the prototype lightweight; add a future_max_return_20d regressor in a later phase",
            "expected_downside": None,
            "expected_downside_status": "not implemented in Phase 11-B to keep the prototype lightweight; add a future_max_drawdown_20d regressor in a later phase",
            "confidence": "abs(opportunity_top_decile_proba - 0.5) * 200, clipped to 0-100",
        }

    def recommendation(
        self,
        ready: bool,
        leakage: dict[str, Any],
        output_quality: list[dict[str, Any]],
        classification: dict[str, Any],
    ) -> dict[str, Any]:
        top_decile = output_quality[-1] if output_quality else {}
        positive_rate = classification.get("base_positive_rate") if classification else None
        top_rate = top_decile.get("opportunity_top_decile_20d_rate")
        lift = None
        if positive_rate not in (None, 0) and top_rate is not None:
            lift = float(top_rate) / float(positive_rate)
        return {
            "ready_for_phase11c": bool(ready and leakage.get("leakage_risk") == "low" and not leakage.get("blocking_issues")),
            "candidate_model_saved": bool(ready and not leakage.get("blocking_issues")),
            "recommended_next_phase": "Phase 11-C Capital Allocation Engine Prototype" if ready and not leakage.get("blocking_issues") else "Fix leakage/blocking issues before Phase 11-C",
            "suggested_allocation_input": ["opportunity_score", "opportunity_top_decile_proba", "confidence"],
            "top_opportunity_decile_lift_vs_base": _safe_float(lift),
            "known_risks": [
                "expected_upside and expected_downside are not yet modeled as separate outputs",
                "training uses a deterministic sample of the train period by default to avoid long jobs",
                "this is not a strategy backtest and does not prove capital allocation performance",
            ],
        }

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Phase 11-B Valuation Engine Prototype",
                "",
                "## Scope",
                "",
                "- lightweight prototype training only",
                "- no strategy backtest, no profile addition, no existing model overwrite",
                "- Phase 11-A dataset is used; historical predictions are not regenerated",
                "",
                "## Dataset",
                "",
                self._table([report["dataset"]], ["rows", "unique_codes", "date_range", "train_rows", "test_rows", "train_period", "test_period"]),
                "",
                "## Feature Policy",
                "",
                self._table([report["feature_policy"]], ["feature_count", "numeric_only", "target_columns_excluded", "feature_columns"]),
                "",
                "## Leakage Checklist",
                "",
                self._table([report["leakage_checklist"]], ["future_columns_in_features", "forbidden_columns_in_features", "backtest_columns_in_features", "trade_result_columns_in_features", "cash_or_portfolio_columns_in_features", "current_pm_multiplier_used", "selected_count_in_day_used", "historical_predictions_regenerated", "leakage_risk", "blocking_issues"]),
                "",
                "## Valuation Output Definition",
                "",
                self._table([report.get("valuation_output_definition", {})], ["opportunity_score", "predicted_opportunity_value", "opportunity_top_decile_proba", "expected_upside_status", "expected_downside_status", "confidence"]),
                "",
                "## Regression",
                "",
                self._table([report.get("regression_metrics", {})], ["target", "mae", "rmse", "pearson", "spearman"]),
                "",
                "### Regression Prediction Deciles",
                "",
                self._table((report.get("regression_metrics") or {}).get("prediction_deciles", []), ["prediction", "decile", "count", "actual_mean", "prediction_mean"]),
                "",
                "## Classification",
                "",
                self._table([report.get("classification_metrics", {})], ["target", "auc", "pr_auc", "precision_at_top10pct", "base_positive_rate"]),
                "",
                "### Classification Prediction Deciles",
                "",
                self._table((report.get("classification_metrics") or {}).get("prediction_deciles", []), ["prediction", "decile", "count", "actual_mean", "prediction_mean"]),
                "",
                "## Output Quality",
                "",
                self._table(report.get("output_quality", []), ["opportunity_score_decile", "count", "actual_opportunity_value_20d_mean", "actual_future_max_return_20d_mean", "actual_future_max_drawdown_20d_mean", "opportunity_top_decile_20d_rate"]),
                "",
                "## Recommendation",
                "",
                self._table([report.get("recommendation", {})], ["ready_for_phase11c", "candidate_model_saved", "recommended_next_phase", "suggested_allocation_input", "top_opportunity_decile_lift_vs_base", "known_risks"]),
                "",
            ]
        )

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in rows:
            values = [self._format_cell(row.get(column)) for column in columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def _format_cell(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False, default=str)
        if value is None:
            return ""
        return str(value).replace("\n", " ")
