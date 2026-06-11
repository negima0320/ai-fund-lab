"""Phase 11-B3 expected downside model prototype.

This research-only prototype trains strict OOS classifiers for:
- opportunity_top_decile_20d
- downside_bad_20d = future_max_drawdown_20d <= -10%

It only performs model-quality and BUY-quality audits for 2025. It does not run
a strategy backtest, overwrite existing models, change profiles, or regenerate
historical predictions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11b_valuation_engine_prototype import CLASSIFICATION_TARGET, DATASET_PATH, Phase11BValuationEnginePrototype
from ml.phase11e_exit_dd_guard import FUTURE_EVAL_COLUMNS, _numeric, _safe_float
from ml.phase11i_strict_oos import TEST_END, TEST_START, TRAIN_END, TRAIN_START, VALIDATION_END, VALIDATION_START


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path("models/ml/valuation_engine/research_phase11b3_downside")
REPORT_STEM = "phase11b3_expected_downside_model_2025"
DOWNSIDE_TARGET = "downside_bad_20d"
DOWNSIDE_BAD_THRESHOLD = -0.10
TOP_N = 5


@dataclass(frozen=True)
class Phase11B3Options:
    max_train_rows: int = 250_000
    random_state: int = 42
    max_iter: int = 80
    learning_rate: float = 0.06
    save_model: bool = True


@dataclass(frozen=True)
class Phase11B3Paths:
    markdown: Path
    json: Path
    model_dir: Path | None


class Phase11B3ExpectedDownsideModel:
    def __init__(self, root: Path | str = ROOT, *, options: Phase11B3Options | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase11B3Options()

    def run(self) -> Phase11B3Paths:
        report, models = self.build_report_and_models()
        return self.save_outputs(report, models)

    def build_report_and_models(self) -> tuple[dict[str, Any], dict[str, Any]]:
        dataset = self.load_dataset()
        feature_columns = Phase11BValuationEnginePrototype(self.root).extract_feature_columns(dataset)
        leakage = self.leakage_checklist(feature_columns)
        if leakage["blocking_issues"]:
            report = {
                "metadata": self.metadata(),
                "split": self.split_definition(),
                "feature_policy": {"feature_columns": feature_columns, "feature_count": len(feature_columns)},
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], leakage),
            }
            return report, {}

        train, validation, test = self.split_dataset(dataset)
        train_prepared, validation_prepared, test_prepared = self.prepare_frames(train, validation, test, feature_columns)
        opportunity_model = self.train_classifier(train_prepared, feature_columns, CLASSIFICATION_TARGET)
        downside_model = self.train_classifier(train_prepared, feature_columns, DOWNSIDE_TARGET)
        validation_scored = self.attach_predictions(
            validation_prepared,
            self.predict(opportunity_model, validation_prepared, feature_columns),
            self.predict(downside_model, validation_prepared, feature_columns),
        )
        test_scored = self.attach_predictions(
            test_prepared,
            self.predict(opportunity_model, test_prepared, feature_columns),
            self.predict(downside_model, test_prepared, feature_columns),
        )
        buy_quality = self.combined_ranking_audit(test_scored)
        report = {
            "metadata": self.metadata(),
            "split": self.split_definition(),
            "dataset_summary": self.dataset_summary(dataset, train, validation, test),
            "feature_policy": {"feature_columns": feature_columns, "feature_count": len(feature_columns)},
            "model_config": {
                "opportunity_model": "HistGradientBoostingClassifier",
                "downside_model": "HistGradientBoostingClassifier",
                "max_iter": self.options.max_iter,
                "learning_rate": self.options.learning_rate,
                "max_train_rows": self.options.max_train_rows,
                "random_state": self.options.random_state,
            },
            "opportunity_model_quality": {
                "validation": self.binary_model_quality(validation_scored, "opportunity_proba", CLASSIFICATION_TARGET, "actual_positive_rate"),
                "test": self.binary_model_quality(test_scored, "opportunity_proba", CLASSIFICATION_TARGET, "actual_positive_rate"),
            },
            "downside_model_quality": {
                "validation": self.binary_model_quality(validation_scored, "downside_bad_proba", DOWNSIDE_TARGET, "actual_downside_rate"),
                "test": self.binary_model_quality(test_scored, "downside_bad_proba", DOWNSIDE_TARGET, "actual_downside_rate"),
            },
            "combined_ranking_audit": buy_quality,
            "pass_fail": self.pass_fail(buy_quality),
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(buy_quality, leakage),
        }
        models = {"opportunity_model": opportunity_model, "downside_model": downside_model, "feature_columns": feature_columns}
        return report, models

    def load_dataset(self) -> pd.DataFrame:
        data = pd.read_parquet(self.root / DATASET_PATH)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        data[DOWNSIDE_TARGET] = (_numeric(data["future_max_drawdown_20d"]) <= DOWNSIDE_BAD_THRESHOLD).astype(int)
        required = ["date", "code", CLASSIFICATION_TARGET, DOWNSIDE_TARGET, "future_max_drawdown_20d"]
        return data.dropna(subset=required).reset_index(drop=True)

    def split_dataset(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = dataset[(dataset["date"] >= TRAIN_START) & (dataset["date"] <= TRAIN_END)].copy()
        validation = dataset[(dataset["date"] >= VALIDATION_START) & (dataset["date"] <= VALIDATION_END)].copy()
        test = dataset[(dataset["date"] >= TEST_START) & (dataset["date"] <= TEST_END)].copy()
        if self.options.max_train_rows and len(train) > self.options.max_train_rows:
            train = train.sample(n=self.options.max_train_rows, random_state=self.options.random_state).sort_values(["date", "code"])
        return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)

    def prepare_frames(self, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        frames = []
        for frame in [train, validation, test]:
            data = frame.copy()
            for column in feature_columns:
                if pd.api.types.is_bool_dtype(data[column].dtype):
                    data[column] = data[column].astype(int)
                else:
                    data[column] = _numeric(data[column])
            data[feature_columns] = data[feature_columns].replace([float("inf"), float("-inf")], pd.NA)
            frames.append(data)
        return frames[0], frames[1], frames[2]

    def train_classifier(self, train: pd.DataFrame, feature_columns: list[str], target: str) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier

        classifier = HistGradientBoostingClassifier(
            max_iter=self.options.max_iter,
            learning_rate=self.options.learning_rate,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            early_stopping=True,
            random_state=self.options.random_state,
        )
        classifier.fit(train[feature_columns], train[target].astype(int))
        return classifier

    def predict(self, model: Any, frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
        return pd.Series(model.predict_proba(frame[feature_columns])[:, 1], index=frame.index)

    def attach_predictions(self, frame: pd.DataFrame, opportunity_proba: pd.Series, downside_proba: pd.Series) -> pd.DataFrame:
        data = frame.copy()
        data["opportunity_proba"] = opportunity_proba
        data["downside_bad_proba"] = downside_proba
        data["opportunity_rank"] = data.groupby("date")["opportunity_proba"].rank(method="average", pct=True)
        data["downside_rank"] = data.groupby("date")["downside_bad_proba"].rank(method="average", pct=True)
        data["score_v1"] = data["opportunity_proba"] - data["downside_bad_proba"]
        data["score_v2"] = data["opportunity_proba"] * (1.0 - data["downside_bad_proba"])
        data["score_v3"] = data["opportunity_rank"] - data["downside_rank"]
        return data.dropna(subset=["date", "code", "close"]).sort_values(["date", "code"]).reset_index(drop=True)

    def binary_model_quality(self, scored: pd.DataFrame, proba_column: str, target: str, decile_rate_key: str) -> dict[str, Any]:
        from sklearn.metrics import average_precision_score, roc_auc_score

        actual = scored[target].astype(int)
        proba = _numeric(scored[proba_column])
        top_n = max(1, int(len(scored) * 0.10))
        top_index = proba.sort_values(ascending=False).head(top_n).index
        deciles = self.prediction_deciles(scored, proba_column, target, decile_rate_key)
        return {
            "target": target,
            "AUC": _safe_float(roc_auc_score(actual, proba)) if actual.nunique() > 1 else None,
            "PR_AUC": _safe_float(average_precision_score(actual, proba)) if actual.nunique() > 1 else None,
            "precision_at_top10": _safe_float(actual.loc[top_index].mean()) if top_n else None,
            "base_positive_rate": _safe_float(actual.mean()),
            "prediction_decile_actual_rate": deciles,
        }

    def prediction_deciles(self, scored: pd.DataFrame, proba_column: str, target: str, rate_key: str) -> list[dict[str, Any]]:
        data = scored[[proba_column, target]].copy()
        data["prediction_decile"] = pd.qcut(data[proba_column].rank(method="first"), 10, labels=False, duplicates="drop") + 1
        rows = []
        for decile, group in data.groupby("prediction_decile", sort=True):
            rows.append(
                {
                    "prediction_decile": int(decile),
                    "count": int(len(group)),
                    rate_key: _safe_float(group[target].mean()),
                    "proba_mean": _safe_float(group[proba_column].mean()),
                }
            )
        return rows

    def combined_ranking_audit(self, scored: pd.DataFrame) -> list[dict[str, Any]]:
        specs = [
            ("opportunity_only_top5", "opportunity_proba"),
            ("score_v1_top5", "score_v1"),
            ("score_v2_top5", "score_v2"),
            ("score_v3_top5", "score_v3"),
        ]
        rows = []
        for label, score_column in specs:
            top = self.daily_top(scored, score_column, TOP_N)
            rows.append(self.buy_quality(label, top, score_column))
        return rows

    def daily_top(self, scored: pd.DataFrame, score_column: str, n: int) -> pd.DataFrame:
        return (
            scored.sort_values(["date", score_column, "turnover_value", "code"], ascending=[True, False, False, True])
            .groupby("date", group_keys=False)
            .head(n)
            .reset_index(drop=True)
        )

    def buy_quality(self, label: str, frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
        drawdown = _numeric(frame["future_max_drawdown_20d"])
        return {
            "candidate_set": label,
            "score_column": score_column,
            "rows": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()),
            "avg_opportunity_proba": self.mean(frame, "opportunity_proba"),
            "avg_downside_bad_proba": self.mean(frame, "downside_bad_proba"),
            "avg_score": self.mean(frame, score_column),
            "future_return_20d_mean": self.mean(frame, "future_return_20d"),
            "future_max_return_20d_mean": self.mean(frame, "future_max_return_20d"),
            "future_max_drawdown_20d_mean": self.mean(frame, "future_max_drawdown_20d"),
            "opportunity_value_20d_mean": self.mean(frame, "opportunity_value_20d"),
            "opportunity_top_decile_20d_rate": self.mean(frame, "opportunity_top_decile_20d"),
            "downside_bad_rate": _safe_float((drawdown <= DOWNSIDE_BAD_THRESHOLD).mean()),
        }

    def mean(self, frame: pd.DataFrame, column: str) -> float | None:
        values = _numeric(frame[column]) if column in frame.columns else pd.Series(dtype=float)
        return _safe_float(values.mean()) if not values.empty else None

    def pass_fail(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        opportunity = next((row for row in rows if row["candidate_set"] == "opportunity_only_top5"), {})
        candidates = [row for row in rows if row["candidate_set"] != "opportunity_only_top5"]
        passed = []
        ideal = []
        for row in candidates:
            downside_rate = _safe_float(row.get("downside_bad_rate")) or 1.0
            top_rate = _safe_float(row.get("opportunity_top_decile_20d_rate")) or 0.0
            if downside_rate <= 0.25:
                passed.append(row["candidate_set"])
            if downside_rate <= 0.25 and top_rate >= 0.24:
                ideal.append(row["candidate_set"])
        best = self.best_candidate(rows)
        return {
            "baseline_downside_bad_rate": opportunity.get("downside_bad_rate"),
            "baseline_top_decile_rate": opportunity.get("opportunity_top_decile_20d_rate"),
            "downside_target_passed_sets": passed,
            "ideal_passed_sets": ideal,
            "best_candidate_set": best.get("candidate_set") if best else None,
            "best_candidate_downside_bad_rate": best.get("downside_bad_rate") if best else None,
            "best_candidate_top_decile_rate": best.get("opportunity_top_decile_20d_rate") if best else None,
            "any_downside_passed": bool(passed),
            "any_ideal_passed": bool(ideal),
        }

    def best_candidate(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        candidates = [row for row in rows if row["candidate_set"] != "opportunity_only_top5"]
        if not candidates:
            return None
        candidates.sort(key=lambda row: (-(row.get("downside_bad_rate") or 1.0), row.get("opportunity_top_decile_20d_rate") or 0.0, row.get("opportunity_value_20d_mean") or -10**9), reverse=True)
        return candidates[0]

    def leakage_checklist(self, feature_columns: list[str]) -> dict[str, Any]:
        helper = Phase11BValuationEnginePrototype(self.root)
        future = [column for column in feature_columns if column.startswith("future_") or column.startswith("opportunity_value") or column in {CLASSIFICATION_TARGET, DOWNSIDE_TARGET}]
        forbidden = [column for column in feature_columns if helper.is_forbidden_column(column)]
        blocking = []
        if future:
            blocking.append("future columns used as features")
        if forbidden:
            blocking.append("forbidden columns used as features")
        return {
            "future_columns_used_as_features": future,
            "future_columns_used_only_as_labels": ["future_max_drawdown_20d", CLASSIFICATION_TARGET, DOWNSIDE_TARGET],
            "backtest_columns_used_as_features": [column for column in feature_columns if "backtest" in column.lower()],
            "trade_result_columns_used_as_features": [column for column in feature_columns if any(token in column.lower() for token in ["trade", "profit", "loss"])],
            "existing_model_overwritten": False,
            "profile_changed": False,
            "strict_model_oos": True,
            "leakage_risk": "high" if blocking else "low",
            "blocking_issues": blocking,
        }

    def recommendation(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"ready_for_phase11b4": False, "recommended_next_phase": "Fix leakage blockers"}
        status = self.pass_fail(rows) if rows else {"any_downside_passed": False, "any_ideal_passed": False, "best_candidate_set": None}
        if status["any_ideal_passed"]:
            next_phase = "Phase11-B4 strict OOS combined ranking strategy check"
        elif status["any_downside_passed"]:
            next_phase = "Phase11-B4 combined ranking threshold tuning"
        else:
            next_phase = "Phase11-B3b downside model feature improvement"
        return {
            "ready_for_phase11b4": bool(status["any_downside_passed"]),
            "ideal_condition_passed": bool(status["any_ideal_passed"]),
            "recommended_next_phase": next_phase,
            "reason": "Proceed only if combined ranking reduces downside_bad_rate to <= 25%; ideal also keeps opportunity_top_decile_rate >= 24%.",
        }

    def dataset_summary(self, dataset: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(dataset)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "train_date_range": self.date_range(train),
            "validation_date_range": self.date_range(validation),
            "test_date_range": self.date_range(test),
            "train_downside_rate": self.mean(train, DOWNSIDE_TARGET),
            "validation_downside_rate": self.mean(validation, DOWNSIDE_TARGET),
            "test_downside_rate": self.mean(test, DOWNSIDE_TARGET),
        }

    def date_range(self, frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "min": frame["date"].min().date().isoformat() if not frame.empty else None,
            "max": frame["date"].max().date().isoformat() if not frame.empty else None,
        }

    def split_definition(self) -> dict[str, Any]:
        return {
            "train": {"start": TRAIN_START, "end": TRAIN_END},
            "validation": {"start": VALIDATION_START, "end": VALIDATION_END},
            "test": {"start": TEST_START, "end": TEST_END},
            "strict_model_oos": True,
            "train_validation_test_overlap": False,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "11-B3",
            "research_only": True,
            "strategy_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
            "model_dir": str(self.root / MODEL_DIR),
        }

    def save_outputs(self, report: dict[str, Any], models: dict[str, Any]) -> Phase11B3Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        model_dir = self.root / MODEL_DIR if self.options.save_model and models else None
        if model_dir:
            self.save_model_bundle(model_dir, models, report)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase11B3Paths(markdown=markdown_path, json=json_path, model_dir=model_dir)

    def save_model_bundle(self, model_dir: Path, models: dict[str, Any], report: dict[str, Any]) -> None:
        import joblib

        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(models["opportunity_model"], model_dir / "opportunity_top_decile_20d_classifier.joblib")
        joblib.dump(models["downside_model"], model_dir / "downside_bad_20d_classifier.joblib")
        (model_dir / "feature_columns.json").write_text(json.dumps(models["feature_columns"], ensure_ascii=False, indent=2), encoding="utf-8")
        metadata = {
            "phase": "11-B3",
            "research_only": True,
            "strict_model_oos": True,
            "split": report["split"],
            "feature_count": len(models["feature_columns"]),
            "targets": [CLASSIFICATION_TARGET, DOWNSIDE_TARGET],
            "existing_model_overwritten": False,
        }
        (model_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 11-B3 Expected Downside Model Prototype",
            "",
            "## Scope",
            "",
            "- strict split: train 2023, validation 2024, test 2025",
            "- research-only Opportunity + Downside classifiers",
            "- BUY-quality audit only; no strategy backtest",
            "",
            "## Downside Model Quality 2025",
            "",
            self.table([report.get("downside_model_quality", {}).get("test", {})], ["AUC", "PR_AUC", "precision_at_top10", "base_positive_rate"]),
            "",
            "## Combined Ranking Audit",
            "",
            self.table(report.get("combined_ranking_audit", []), ["candidate_set", "rows", "candidate_days", "avg_opportunity_proba", "avg_downside_bad_proba", "avg_score", "future_return_20d_mean", "future_max_return_20d_mean", "future_max_drawdown_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_rate", "downside_bad_rate"]),
            "",
            "## Pass / Fail",
            "",
            self.table([report.get("pass_fail", {})], ["baseline_downside_bad_rate", "baseline_top_decile_rate", "downside_target_passed_sets", "ideal_passed_sets", "best_candidate_set", "best_candidate_downside_bad_rate", "best_candidate_top_decile_rate", "any_downside_passed", "any_ideal_passed"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_as_labels", "backtest_columns_used_as_features", "trade_result_columns_used_as_features", "existing_model_overwritten", "profile_changed", "strict_model_oos", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["ready_for_phase11b4", "ideal_condition_passed", "recommended_next_phase", "reason"]),
            "",
        ]
        return "\n".join(lines)

    def table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(self.format_value(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def format_value(self, value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return f"{value:.4f}"
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, list):
            return ", ".join(map(str, value))
        if value is None:
            return ""
        return str(value)
