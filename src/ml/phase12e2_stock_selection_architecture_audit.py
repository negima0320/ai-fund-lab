"""Phase 12-E2 Stock Selection architecture audit.

This audit is read-only. It explains what the current Stock Selection layer
uses as features, labels, models, prediction code path, and output columns. It
does not train models, regenerate predictions, change profiles, overwrite
models, call external APIs, or run a full backtest.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.config import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FINANCIAL_FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MODEL_EXCLUDE_COLUMNS,
    MODEL_FILENAMES,
    MODEL_TARGETS,
    PREDICTION_COLUMNS,
    TECHNICAL_FEATURE_COLUMNS,
    TOPIX_FEATURE_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase12e2_stock_selection_architecture_audit"


@dataclass(frozen=True)
class Phase12E2Paths:
    markdown: Path
    json: Path


class Phase12E2StockSelectionArchitectureAudit:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)

    def run(self) -> Phase12E2Paths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        feature_columns = self.load_feature_columns()
        metrics = self.load_metrics()
        walk_forward = self.load_json("reports/ml/walk_forward_model_audit_5y_enriched_v2.json")
        d3 = self.load_json("reports/ml/phase12d3_prediction_lineage_oos_audit.json")
        e1 = self.load_json("reports/ml/phase12e1_stock_selection_reality_audit_2025.json")
        code_paths = self.code_path_summary()
        model_lineage = self.model_lineage_summary(feature_columns, metrics, walk_forward, d3)
        output_meaning = self.output_column_meaning()
        failure_reason = self.suspected_failure_reason(e1, output_meaning)
        leakage = self.leakage_checklist(feature_columns)
        recommendation = self.recommendation(failure_reason, e1, leakage)
        return {
            "metadata": self.metadata(),
            "executive_summary": self.executive_summary(model_lineage, failure_reason, recommendation),
            "stock_selection_architecture_summary": self.architecture_summary(feature_columns),
            "feature_category_summary": self.feature_category_summary(feature_columns),
            "label_summary": self.label_summary(),
            "output_column_meaning": output_meaning,
            "model_lineage_summary": model_lineage,
            "code_path_summary": code_paths,
            "suspected_failure_reason": failure_reason,
            "phase12e1_connection": self.phase12e1_connection(e1),
            "leakage_checklist": leakage,
            "recommendation": recommendation,
        }

    def load_json(self, relative: str) -> dict[str, Any]:
        path = self.root / relative
        if not path.exists():
            return {"missing": True, "path": str(path)}
        return json.loads(path.read_text(encoding="utf-8"))

    def load_feature_columns(self) -> list[str]:
        candidates = [
            self.root / "models/ml/walk_forward/current/2025-01/feature_columns.json",
            self.root / "models/ml/walk_forward/archive/walk_forward_202501/feature_columns.json",
        ]
        for path in candidates:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        return list(FEATURE_COLUMNS[2:])

    def load_metrics(self) -> dict[str, Any]:
        candidates = [
            self.root / "models/ml/walk_forward/current/2025-01/metrics.json",
            self.root / "models/ml/walk_forward/archive/walk_forward_202501/metrics.json",
        ]
        for path in candidates:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["_source_path"] = str(path)
                return payload
        return {"missing": True}

    def architecture_summary(self, feature_columns: list[str]) -> dict[str, Any]:
        return {
            "system_name": "Stock Selection AI / walk-forward ML",
            "training_code": "src/ml/model_trainer.py::ModelTrainer.train_all",
            "prediction_code": "src/ml/predictor.py::Predictor.predict_daily",
            "walk_forward_code": "src/ml/walk_forward.py::MLWalkForwardRunner",
            "feature_builder_code": "src/ml/feature_builder.py::FeatureBuilder",
            "model_family": "LightGBM LGBMRegressor and LGBMClassifier",
            "model_tasks": self.model_tasks(),
            "feature_count": len(feature_columns),
            "prediction_outputs": list(PREDICTION_COLUMNS),
            "ranking_derivation_source": "Phase11A derives risk_adjusted_score, expected_return, stock_selection_rank_score, candidate_strength from prediction-time walk-forward outputs.",
        }

    def model_tasks(self) -> list[dict[str, Any]]:
        return [
            {
                "model_name": name,
                "target": spec["target"],
                "task": spec["task"],
                "filename": MODEL_FILENAMES.get(name),
            }
            for name, spec in MODEL_TARGETS.items()
        ]

    def feature_category_summary(self, feature_columns: list[str]) -> dict[str, Any]:
        categories = {
            "Price": [],
            "Candle": [],
            "Volume": [],
            "Financial": [],
            "Market": [],
            "Other": [],
        }
        for column in feature_columns:
            lower = column.lower()
            if column in FINANCIAL_FEATURE_COLUMNS or lower in {"eps", "bps", "eqar"} or "growth" in lower or "payout" in lower:
                categories["Financial"].append(column)
            elif column in TOPIX_FEATURE_COLUMNS or "topix" in lower or "relative_return" in lower or column in CATEGORICAL_FEATURE_COLUMNS:
                categories["Market"].append(column)
            elif "volume" in lower or "turnover" in lower:
                categories["Volume"].append(column)
            elif column in TECHNICAL_FEATURE_COLUMNS or any(token in lower for token in ["return_", "ma", "close"]):
                categories["Price"].append(column)
            elif any(token in lower for token in ["body", "shadow", "gap", "range", "close_position"]):
                categories["Candle"].append(column)
            else:
                categories["Other"].append(column)
        return {
            name: {
                "count": len(columns),
                "columns": columns,
            }
            for name, columns in categories.items()
        }

    def label_summary(self) -> dict[str, Any]:
        rows = []
        for name, spec in MODEL_TARGETS.items():
            rows.append(
                {
                    "model_name": name,
                    "target": spec["target"],
                    "task": spec["task"],
                    "label_role": "training_label",
                    "allowed_as_feature": spec["target"] not in MODEL_EXCLUDE_COLUMNS,
                }
            )
        return {
            "label_columns": list(LABEL_COLUMNS),
            "model_targets": rows,
            "future_labels_used_for_training": [spec["target"] for spec in MODEL_TARGETS.values()],
            "feature_exclusion_columns": list(MODEL_EXCLUDE_COLUMNS),
            "label_safety_note": "future/label columns are excluded by MODEL_EXCLUDE_COLUMNS before training features are extracted.",
        }

    def output_column_meaning(self) -> list[dict[str, Any]]:
        return [
            {
                "column": "expected_return_5d",
                "meaning": "Regression prediction for future_5d_return.",
                "source": "future_5d_return_regression model",
                "higher_is_better": True,
            },
            {
                "column": "expected_return_10d",
                "meaning": "Regression prediction for future_10d_return; Phase12/Valuation aliases this to expected_return.",
                "source": "future_10d_return_regression model",
                "higher_is_better": True,
            },
            {
                "column": "upside_probability_10d",
                "meaning": "Probability of upside_10d positive label.",
                "source": "upside_10d_classification model",
                "higher_is_better": True,
            },
            {
                "column": "bad_entry_probability_10d",
                "meaning": "Probability of bad_entry_10d label; used as risk penalty.",
                "source": "bad_entry_10d_classification model",
                "higher_is_better": False,
            },
            {
                "column": "expected_max_return_20d",
                "meaning": "Regression prediction for future_max_return_20d.",
                "source": "future_max_return_20d_regression model",
                "higher_is_better": True,
            },
            {
                "column": "swing_success_probability_20d",
                "meaning": "Probability of future_swing_success_20d.",
                "source": "future_swing_success_20d_classification model",
                "higher_is_better": True,
            },
            {
                "column": "ml_score",
                "meaning": "Post-processed score: expected_return_10d * 100 + upside_probability_10d * 10 - bad_entry_probability_10d * 15.",
                "source": "src/ml/predictor.py::Predictor.predict_daily",
                "higher_is_better": True,
            },
            {
                "column": "risk_adjusted_score",
                "meaning": "Derived score: expected_return_10d - 0.5 * bad_entry_probability_10d.",
                "source": "src/ml/phase11a_valuation_dataset_audit.py::_derive_stock_selection_scores",
                "higher_is_better": True,
            },
            {
                "column": "expected_return",
                "meaning": "Alias of expected_return_10d in Phase11/12 valuation artifacts.",
                "source": "src/ml/phase11a_valuation_dataset_audit.py::_derive_stock_selection_scores",
                "higher_is_better": True,
            },
            {
                "column": "stock_selection_rank_score",
                "meaning": "Derived rank score from ml_score when no native column exists; sorted descending.",
                "source": "src/ml/phase11a_valuation_dataset_audit.py::_derive_stock_selection_scores",
                "higher_is_better": True,
            },
            {
                "column": "candidate_strength",
                "meaning": "Derived composite: expected_max_return_20d + swing_success_probability_20d - bad_entry_probability_10d.",
                "source": "src/ml/phase11a_valuation_dataset_audit.py::_derive_stock_selection_scores",
                "higher_is_better": True,
            },
        ]

    def model_lineage_summary(
        self,
        feature_columns: list[str],
        metrics: dict[str, Any],
        walk_forward: dict[str, Any],
        d3: dict[str, Any],
    ) -> list[dict[str, Any]]:
        source_path = metrics.get("_source_path")
        d3_stock = d3.get("stock_selection_lineage", {}) if isinstance(d3, dict) else {}
        fold_rows = walk_forward.get("fold_rows", []) if isinstance(walk_forward, dict) else []
        fold_2025 = [row for row in fold_rows if str(row.get("month", "")).startswith("2025")]
        rows = []
        for name, spec in MODEL_TARGETS.items():
            metric = metrics.get(name, {}) if isinstance(metrics, dict) else {}
            rows.append(
                {
                    "model_path": str(self.root / "models/ml/walk_forward/current/2025-01" / (MODEL_FILENAMES.get(name) or "")),
                    "model_name": name,
                    "model_type": "LightGBM LGBMRegressor" if spec["task"] == "regression" else "LightGBM LGBMClassifier",
                    "target": spec["target"],
                    "feature_count": len(feature_columns),
                    "metadata_available": bool(metric),
                    "metric_summary": {key: value for key, value in metric.items() if isinstance(value, (int, float))},
                    "metrics_source": source_path,
                    "train_start": self.first_non_null(fold_2025, "train_start") or d3_stock.get("train_start"),
                    "train_end": self.first_non_null(fold_2025, "effective_train_end") or d3_stock.get("train_end"),
                    "validation_start": "last train-window split inside ModelTrainer/MLWalkForwardRunner",
                    "validation_end": "last train-window split inside ModelTrainer/MLWalkForwardRunner",
                    "test_start": self.first_non_null(fold_2025, "test_start") or "2025-01-01",
                    "test_end": self.last_non_null(fold_2025, "test_end") or "2025-12-31",
                    "strict_oos_for_2025": bool(d3.get("final_trust_decision", {}).get("stock_selection_strict_oos_for_2025", False)),
                }
            )
        return rows

    def first_non_null(self, rows: list[dict[str, Any]], key: str) -> Any:
        for row in rows:
            if row.get(key) is not None:
                return row.get(key)
        return None

    def last_non_null(self, rows: list[dict[str, Any]], key: str) -> Any:
        for row in reversed(rows):
            if row.get(key) is not None:
                return row.get(key)
        return None

    def code_path_summary(self) -> dict[str, Any]:
        files = {
            "feature_build": "src/ml/feature_builder.py",
            "dataset_build": "src/ml/dataset_builder.py",
            "training": "src/ml/model_trainer.py",
            "walk_forward": "src/ml/walk_forward.py",
            "prediction": "src/ml/predictor.py",
            "phase11a_join_and_derive": "src/ml/phase11a_valuation_dataset_audit.py",
            "profile_prediction_root": "config/profiles/* ml_backtest.prediction_root=data/ml/walk_forward_predictions",
        }
        evidence = {}
        for key, relative in files.items():
            path = self.root / relative if "*" not in relative else None
            evidence[key] = {
                "path": relative,
                "exists": path.exists() if path else True,
            }
        return {
            "where_features_are_built": "FeatureBuilder.build_daily_features reads cached J-Quants price/financial/listed/TOPIX data and writes data/ml/features/features_YYYY-MM-DD.parquet.",
            "where_models_are_trained": "MLWalkForwardRunner.run builds expanding-window datasets and calls ModelTrainer.train_all for each monthly fold.",
            "where_predictions_are_generated": "MLWalkForwardRunner._predict_month loads fold-specific current models and Predictor.save_predictions writes data/ml/walk_forward_predictions/predictions_YYYY-MM-DD.parquet.",
            "where_stock_selection_columns_are_derived": "Phase11A joins features and walk-forward predictions, then derives risk_adjusted_score, expected_return, stock_selection_rank_score, candidate_strength.",
            "output_columns": list(PREDICTION_COLUMNS),
            "evidence_files": evidence,
        }

    def phase12e1_connection(self, e1: dict[str, Any]) -> dict[str, Any]:
        if e1.get("missing"):
            return {"available": False, "path": e1.get("path")}
        rank_rows = {row.get("selection"): row for row in e1.get("rank_quality_table", [])}
        return {
            "available": True,
            "stock_selection_adds_value": e1.get("final_judgment", {}).get("stock_selection_adds_value"),
            "stock_selection_top5_valid": e1.get("final_judgment", {}).get("stock_selection_top5_valid"),
            "stock_selection_prefilter_hurts_valuation": e1.get("final_judgment", {}).get("stock_selection_prefilter_hurts_valuation"),
            "candidate_universe_top_decile_rate": self.safe(rank_rows.get("candidate_universe", {}).get("opportunity_top_decile_20d_rate")),
            "stock_selection_rank_score_top5_top_decile_rate": self.safe(rank_rows.get("stock_selection_rank_score_top5", {}).get("opportunity_top_decile_20d_rate")),
            "candidate_strength_top5_top_decile_rate": self.safe(rank_rows.get("candidate_strength_top5", {}).get("opportunity_top_decile_20d_rate")),
            "opportunity_top5_reference": 0.2400,
        }

    def suspected_failure_reason(self, e1: dict[str, Any], output_meaning: list[dict[str, Any]]) -> dict[str, Any]:
        connection = self.phase12e1_connection(e1)
        reasons = [
            "stock_selection_rank_score is a hand-composed ml_score, not a direct 20d opportunity objective.",
            "Core Stock Selection targets are 5d/10d return, 10d upside/bad-entry, and max/swing labels; Phase12 judged 20d opportunity/downside quality.",
            "risk_adjusted_score penalizes bad_entry_probability_10d and may favor safety over opportunity top-decile capture.",
            "candidate_strength includes expected_max_return_20d and swing_success_probability_20d, which better matches the Phase12 opportunity objective, explaining why it was relatively stronger in E1.",
            "Valuation/Downside models are trained directly on Phase11 opportunity/downside labels, so they can replace the older stock-selection composite for this research objective.",
        ]
        return {
            "main_hypothesis": "Objective mismatch: Stock Selection is a short-horizon composite selector, while Phase12 is evaluating 20d opportunity plus downside risk.",
            "reasons": reasons,
            "e1_evidence": connection,
            "stock_selection_ai_current_objective_misaligned_with_phase12": True,
            "valuation_engine_likely_replaces_stock_selection_prefilter_for_phase12": bool(connection.get("stock_selection_prefilter_hurts_valuation")),
        }

    def recommendation(self, failure_reason: dict[str, Any], e1: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"recommended_next_phase": "Fix Phase12-E2 leakage blockers"}
        connection = self.phase12e1_connection(e1)
        if connection.get("stock_selection_prefilter_hurts_valuation"):
            next_phase = "Phase12-E3 Remove Stock Selection Prefilter Test"
        elif failure_reason.get("stock_selection_ai_current_objective_misaligned_with_phase12"):
            next_phase = "Phase12-E3 Stock Selection Rebuild Design"
        else:
            next_phase = "Phase12-D4 Exit AI Dataset Audit"
        return {
            "recommended_next_phase": next_phase,
            "reason": failure_reason.get("main_hypothesis"),
            "candidate_alternatives": [
                "Phase12-E3 Remove Stock Selection Prefilter Test",
                "Phase12-E3 Candidate Strength Rebase",
                "Phase12-E3 Stock Selection Rebuild Design",
                "Phase12-D4 Exit AI Dataset Audit",
            ],
        }

    def leakage_checklist(self, feature_columns: list[str]) -> dict[str, Any]:
        future_features = [column for column in feature_columns if column.startswith("future_")]
        forbidden_feature_tokens = ("backtest", "trade", "profit", "loss", "cash", "portfolio", "selected", "bought")
        forbidden = [column for column in feature_columns if any(token in column.lower() for token in forbidden_feature_tokens)]
        blocking = []
        if future_features:
            blocking.append("future_columns_present_in_stock_selection_features")
        if forbidden:
            blocking.append("forbidden_backtest_or_trade_columns_present_in_stock_selection_features")
        return {
            "future_columns_used_as_features": future_features,
            "future_columns_used_only_as_labels": [spec["target"] for spec in MODEL_TARGETS.values() if spec["target"].startswith("future_")],
            "backtest_columns_used_as_features": [column for column in forbidden if "backtest" in column.lower()],
            "trade_result_columns_used_as_features": [column for column in forbidden if any(token in column.lower() for token in ("trade", "profit", "loss"))],
            "new_model_trained": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
            "leakage_risk": "low" if not blocking else "high",
            "blocking_issues": blocking,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "12-E2",
            "scope": "Stock Selection architecture audit only",
            "new_model_trained": False,
            "full_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_added": False,
            "profile_modified": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
        }

    def executive_summary(self, model_lineage: list[dict[str, Any]], failure_reason: dict[str, Any], recommendation: dict[str, Any]) -> str:
        strict = all(row.get("strict_oos_for_2025") for row in model_lineage) if model_lineage else False
        return (
            "Stock Selection AI is a LightGBM walk-forward ensemble of short-horizon return/upside/risk models. "
            "Its Phase12 columns are mostly derived composites from prediction-time outputs, not direct 20d opportunity labels. "
            f"strict_oos_for_2025={strict}. "
            f"Main hypothesis: {failure_reason.get('main_hypothesis')} "
            f"Recommended next phase: {recommendation.get('recommended_next_phase')}."
        )

    def save_outputs(self, report: dict[str, Any]) -> Phase12E2Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12E2Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-E2 Stock Selection Architecture Audit",
            "",
            "## Executive Summary",
            "",
            report.get("executive_summary", ""),
            "",
            "## Architecture",
            "",
            self.table([report["stock_selection_architecture_summary"]], ["system_name", "model_family", "feature_count", "training_code", "prediction_code", "walk_forward_code", "ranking_derivation_source"]),
            "",
            "## Feature Categories",
            "",
            self.table(self.category_rows(report["feature_category_summary"]), ["category", "count", "columns"]),
            "",
            "## Labels",
            "",
            self.table(report["label_summary"]["model_targets"], ["model_name", "target", "task", "label_role", "allowed_as_feature"]),
            "",
            "## Output Column Meaning",
            "",
            self.table(report["output_column_meaning"], ["column", "meaning", "source", "higher_is_better"]),
            "",
            "## Model Lineage",
            "",
            self.table(report["model_lineage_summary"], ["model_name", "model_type", "target", "feature_count", "train_start", "train_end", "test_start", "test_end", "strict_oos_for_2025", "metadata_available", "metric_summary"]),
            "",
            "## Code Path",
            "",
            self.table([report["code_path_summary"]], ["where_features_are_built", "where_models_are_trained", "where_predictions_are_generated", "where_stock_selection_columns_are_derived"]),
            "",
            "## E1 Connection",
            "",
            self.table([report["phase12e1_connection"]], ["stock_selection_adds_value", "stock_selection_top5_valid", "stock_selection_prefilter_hurts_valuation", "candidate_universe_top_decile_rate", "stock_selection_rank_score_top5_top_decile_rate", "candidate_strength_top5_top_decile_rate", "opportunity_top5_reference"]),
            "",
            "## Suspected Failure Reason",
            "",
            self.table([report["suspected_failure_reason"]], ["main_hypothesis", "stock_selection_ai_current_objective_misaligned_with_phase12", "valuation_engine_likely_replaces_stock_selection_prefilter_for_phase12", "reasons"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_as_labels", "backtest_columns_used_as_features", "trade_result_columns_used_as_features", "new_model_trained", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "historical_predictions_regenerated", "jquants_api_called", "openai_api_called", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["recommended_next_phase", "reason", "candidate_alternatives"]),
            "",
        ]
        return "\n".join(lines)

    def category_rows(self, categories: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"category": key, **value} for key, value in categories.items()]

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
        text = str(value)
        return re.sub(r"\s+", " ", text)

    def safe(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            result = float(value)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(result) else result
