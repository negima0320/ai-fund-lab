"""Phase 7-B Portfolio Manager AI leakage forensics.

This audit is read-only. It distinguishes three separate questions that were
intentionally conflated in the broader Phase 7-A inventory:

* Does the PM dataset contain backtest/outcome columns?
* Were those columns actually used as model features?
* Are the target labels API-derived future-return labels or executed-trade
  outcomes?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase7b_pm_ai_leakage_forensics_2023-01_to_2026-05"

PM_MODEL_DIR = Path("models/ml/portfolio_manager/current_v2_73_phase3b_clean")
PM_DATASET = Path("data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet")

TARGET_COLUMNS = {
    "ideal_weight_bucket",
    "high_conviction_target",
    "avoid_target",
    "realized_return",
    "ideal_cash_reserve_bucket",
    "positive_trade",
}

FUTURE_LABEL_COLUMNS = {
    "future_5d_return",
    "future_10d_return",
}

IDENTIFIER_COLUMNS = {
    "signal_date",
    "code",
    "entry_date",
    "name",
    "profile_id",
    "profile_name",
    "trade_id",
}

BACKTEST_POSITION_STATE_COLUMNS = {
    "actual_shares",
    "actual_holding_days",
    "planned_shares",
    "planned_amount",
    "scaled_shares",
    "scaled_amount",
    "final_shares",
    "final_amount",
    "current_capital_utilization",
    "current_positions_count",
}

BACKTEST_CASH_STATE_COLUMNS = {
    "cash_before",
    "cash_after",
    "cash_before_ratio",
    "daily_buy_limit_remaining_before",
    "daily_buy_limit_remaining_after",
    "max_positions_remaining_before",
}

BACKTEST_TRADE_OUTCOME_COLUMNS = {
    "actual_buy_amount",
    "actual_net_profit",
}

BACKTEST_DECISION_COLUMNS = {
    "decision",
    "skip_reason",
    "exit_reason",
    "reject_reason",
    "scale_reason",
    "allocation_limit",
    "allocation_reason",
}

MODEL_PREDICTION_COLUMNS = {
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "risk_adjusted_score",
    "rank_in_day",
    "score_rank_in_day",
    "risk_adjusted_score_percentile_in_day",
    "expected_return_percentile_in_day",
    "expected_max_return_percentile_in_day",
    "swing_success_percentile_in_day",
    "bad_entry_percentile_in_day",
    "score_gap_to_best",
    "expected_return_gap_to_best",
    "expected_max_return_gap_to_best",
    "swing_success_gap_to_best",
    "bad_entry_gap_to_best",
    "candidate_count_in_day",
    "day_avg_risk_adjusted_score",
    "day_max_risk_adjusted_score",
    "day_avg_expected_return_10d",
    "day_avg_expected_max_return_20d",
    "day_avg_swing_success_probability_20d",
    "day_avg_bad_entry_probability",
    "day_candidate_strength",
    "day_risk_level",
    "prediction_source",
    "prediction_joined",
}

API_PRICE_FEATURE_PREFIXES = (
    "return_",
    "ma",
    "body_",
    "upper_",
    "lower_",
    "close_position",
    "gap_up",
    "daily_range",
    "volume",
    "turnover",
)

API_FINANCIAL_COLUMNS = {
    "EPS",
    "BPS",
    "EqAR",
    "Sales_growth",
    "OP_growth",
    "NP_growth",
    "FEPS_growth",
    "FSales_growth",
    "FOP_growth",
    "PayoutRatioAnn",
    "days_to_earnings",
    "is_near_earnings",
}

API_MARKET_PREFIXES = ("topix_", "relative_")

FORBIDDEN_FEATURE_CATEGORIES = {
    "backtest_position_state",
    "backtest_cash_state",
    "backtest_trade_outcome",
    "backtest_decision_or_reason",
    "future_label",
    "target_label",
    "identifier",
}

SUSPICIOUS_FEATURE_CATEGORIES = {
    "model_prediction_feature",
    "unknown",
}


@dataclass(frozen=True)
class Phase7BPaths:
    markdown: Path
    json: Path


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value[:10]) + (", ..." if len(value) > 10 else "")
    return str(value).replace("\n", " ")


def classify_pm_column(column: str) -> str:
    lower = column.lower()
    if column in TARGET_COLUMNS:
        return "target_label"
    if column in FUTURE_LABEL_COLUMNS or lower.startswith("future_") or lower.startswith("return_after_"):
        return "future_label"
    if column in IDENTIFIER_COLUMNS:
        return "identifier"
    if column in BACKTEST_TRADE_OUTCOME_COLUMNS or "profit" in lower or lower in {"win", "loss", "win_loss"}:
        return "backtest_trade_outcome"
    if column in BACKTEST_CASH_STATE_COLUMNS or "cash" in lower or "portfolio" in lower:
        return "backtest_cash_state"
    if column in BACKTEST_POSITION_STATE_COLUMNS or "position" in lower or "holding_days" in lower:
        return "backtest_position_state"
    if column in BACKTEST_DECISION_COLUMNS or lower.endswith("_reason") or lower == "decision":
        return "backtest_decision_or_reason"
    if column == "selected_count_in_day":
        return "backtest_decision_or_reason"
    if column in MODEL_PREDICTION_COLUMNS:
        return "model_prediction_feature"
    if column in API_FINANCIAL_COLUMNS:
        return "api_financial_feature"
    if lower.startswith(API_MARKET_PREFIXES):
        return "api_market_feature"
    if lower.startswith(API_PRICE_FEATURE_PREFIXES) or column in {"close"}:
        return "api_price_feature"
    return "unknown"


class Phase7BPMLeakageForensics:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)

    def build_report(self) -> dict[str, Any]:
        dataset = _read_parquet(self._root(PM_DATASET))
        metadata = _read_json(self._root(PM_MODEL_DIR) / "model_metadata.json") or {}
        feature_columns = self._training_features(metadata)
        column_audit = self._dataset_column_audit(dataset)
        feature_audit = self._training_feature_audit(feature_columns, dataset, metadata)
        target_audit = self._target_label_audit(dataset, metadata)
        artifact_audit = self._model_artifact_audit(feature_columns)
        importance = self._feature_importance_audit(feature_columns)
        safety = self._safety_assessment(feature_audit, target_audit, artifact_audit)
        retraining = self._retraining_judgement(feature_audit, target_audit, safety)
        return {
            "metadata": {
                "phase": "7-B",
                "audit_only": True,
                "model_retraining_executed": False,
                "full_backtest_executed": False,
                "profile_added": False,
                "current_model_overwritten": False,
                "full_pytest_executed": False,
                "jquants_api_refetch": False,
                "openai_used": False,
                "live_order": False,
            },
            "input_paths": {
                "pm_model_dir": str(self._root(PM_MODEL_DIR)),
                "pm_dataset": str(self._root(PM_DATASET)),
            },
            "dataset_summary": self._dataset_summary(dataset),
            "dataset_column_audit": column_audit,
            "training_feature_audit": feature_audit,
            "target_label_audit": target_audit,
            "model_artifact_audit": artifact_audit,
            "feature_importance_audit": importance,
            "v2_82_safety_impact_assessment": safety,
            "retraining_judgement": retraining,
        }

    def save_report(self, result: dict[str, Any]) -> Phase7BPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path = report_dir / f"{REPORT_STEM}.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase7BPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# AI Retraining Phase 7-B PM AI Leakage Forensics",
            "",
            "## Scope",
            "",
            "- forensic audit only",
            "- no retraining, no backtest, no profile addition, no current model overwrite, no full pytest",
            "",
            "## Dataset Summary",
            "",
            self._table([result["dataset_summary"]], ["path", "exists", "rows", "columns", "date_range"]),
            "",
            "## Dataset Column Audit",
            "",
            self._table(
                [result["dataset_column_audit"]],
                ["total_columns", "safe_feature_columns_count", "suspicious_columns_count", "forbidden_columns_count", "target_columns_count", "unknown_columns_count"],
            ),
            "",
            "## Training Feature Audit",
            "",
            self._table(
                [result["training_feature_audit"]],
                [
                    "training_feature_count",
                    "forbidden_columns_used_as_features",
                    "suspicious_columns_used_as_features",
                    "api_only_feature_count",
                    "non_api_feature_count",
                    "unable_to_reconstruct_training_features",
                ],
            ),
            "",
            "## Target Label Audit",
            "",
            self._table(
                result["target_label_audit"]["labels"],
                ["label", "label_source", "api_only", "backtest_derived", "future_return_api_only_label", "realized_trade_result", "leakage_severity"],
            ),
            "",
            "## Model Artifact Audit",
            "",
            self._table(
                [result["model_artifact_audit"]],
                ["model_artifact_complete", "feature_schema_available", "label_schema_available", "training_config_available", "importances_available"],
            ),
            "",
            "## Feature Importance",
            "",
            self._table(result["feature_importance_audit"].get("top_features", []), ["model", "feature", "importance", "category"]),
            "",
            "## v2_82 Safety Impact",
            "",
            self._table(
                [result["v2_82_safety_impact_assessment"]],
                ["current_pm_model_operational_risk", "v282_result_trust_level", "reason"],
            ),
            "",
            "## Retraining Judgement",
            "",
            self._table(
                [result["retraining_judgement"]],
                [
                    "pm_ai_current_model_safe_to_use",
                    "pm_ai_direct_retraining_allowed",
                    "pm_ai_dataset_rebuild_required",
                    "api_only_pm_dataset_rebuild_priority",
                    "next_phase_recommended",
                ],
            ),
            "",
        ]
        return "\n".join(lines)

    def _root(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def _dataset_summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        date_range = ""
        if "signal_date" in dataset.columns and not dataset.empty:
            dates = pd.to_datetime(dataset["signal_date"], errors="coerce").dropna()
            if not dates.empty:
                date_range = f"{dates.min().date()} to {dates.max().date()}"
        return {
            "path": str(self._root(PM_DATASET)),
            "exists": self._root(PM_DATASET).exists(),
            "rows": int(len(dataset)),
            "columns": int(len(dataset.columns)),
            "date_range": date_range,
        }

    def _training_features(self, metadata: dict[str, Any]) -> list[str]:
        if isinstance(metadata.get("feature_columns"), list):
            return [str(item) for item in metadata["feature_columns"]]
        payload = _read_json(self._root(PM_MODEL_DIR) / "feature_columns.json")
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict) and isinstance(payload.get("feature_columns"), list):
            return [str(item) for item in payload["feature_columns"]]
        return []

    def _dataset_column_audit(self, dataset: pd.DataFrame) -> dict[str, Any]:
        rows = []
        by_category: dict[str, list[str]] = {}
        for column in dataset.columns:
            category = classify_pm_column(str(column))
            by_category.setdefault(category, []).append(str(column))
            rows.append({"column": str(column), "category": category})
        safe_categories = {"api_price_feature", "api_financial_feature", "api_market_feature", "derived_from_api_only"}
        suspicious_categories = {"model_prediction_feature", "unknown"}
        forbidden_categories = {
            "backtest_position_state",
            "backtest_cash_state",
            "backtest_trade_outcome",
            "backtest_decision_or_reason",
            "future_label",
        }
        safe = [row["column"] for row in rows if row["category"] in safe_categories]
        suspicious = [row["column"] for row in rows if row["category"] in suspicious_categories]
        forbidden = [row["column"] for row in rows if row["category"] in forbidden_categories]
        targets = [row["column"] for row in rows if row["category"] == "target_label"]
        unknown = by_category.get("unknown", [])
        return {
            "total_columns": int(len(dataset.columns)),
            "columns_by_category": {key: sorted(value) for key, value in sorted(by_category.items())},
            "safe_feature_columns": sorted(safe),
            "safe_feature_columns_count": len(safe),
            "suspicious_columns": sorted(suspicious),
            "suspicious_columns_count": len(suspicious),
            "forbidden_columns": sorted(forbidden),
            "forbidden_columns_count": len(forbidden),
            "target_columns": sorted(targets),
            "target_columns_count": len(targets),
            "unknown_columns": sorted(unknown),
            "unknown_columns_count": len(unknown),
        }

    def _training_feature_audit(self, features: list[str], dataset: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
        categorized = [{"feature": feature, "category": classify_pm_column(feature)} for feature in features]
        forbidden = [row["feature"] for row in categorized if row["category"] in FORBIDDEN_FEATURE_CATEGORIES]
        suspicious = [row["feature"] for row in categorized if row["category"] in SUSPICIOUS_FEATURE_CATEGORIES]
        api_only = [
            row["feature"]
            for row in categorized
            if row["category"] in {"api_price_feature", "api_financial_feature", "api_market_feature", "derived_from_api_only"}
        ]
        missing = [feature for feature in features if feature not in dataset.columns]
        unable = not features
        return {
            "training_feature_count": len(features),
            "training_feature_columns": features,
            "training_feature_categories": categorized,
            "forbidden_columns_used_as_features": sorted(forbidden),
            "suspicious_columns_used_as_features": sorted(suspicious),
            "api_only_feature_count": len(api_only),
            "non_api_feature_count": len(features) - len(api_only),
            "missing_training_features_in_dataset": sorted(missing),
            "unable_to_reconstruct_training_features": unable,
            "reason": "" if not unable else "feature_columns not found in metadata or feature_columns.json",
            "metadata_leakage_guard": metadata.get("leakage_guard", ""),
        }

    def _target_label_audit(self, dataset: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
        target_specs = metadata.get("targets") if isinstance(metadata.get("targets"), dict) else {}
        target_columns = [str(spec.get("target")) for spec in target_specs.values() if isinstance(spec, dict) and spec.get("target")]
        if not target_columns:
            target_columns = ["ideal_weight_bucket", "high_conviction_target", "avoid_target", "realized_return", "ideal_cash_reserve_bucket"]
        labels = [self._audit_target_label(label, dataset) for label in target_columns]
        safe = [row["label"] for row in labels if row["leakage_severity"] in {"low", "medium_name_risk"}]
        forbidden = [row["label"] for row in labels if row["leakage_severity"] in {"high", "block"}]
        return {
            "labels": labels,
            "target_labels_safe_for_retraining": sorted(safe),
            "target_labels_forbidden_for_retraining": sorted(forbidden),
            "target_rebuild_required": bool(forbidden) or any(row["leakage_severity"] != "low" for row in labels),
        }

    def _audit_target_label(self, label: str, dataset: pd.DataFrame) -> dict[str, Any]:
        future_equal = self._series_equal(dataset, label, "future_10d_return")
        source = "unknown"
        severity = "medium"
        api_only = False
        backtest = False
        realized_trade = False
        future_label = False
        if label == "realized_return" and future_equal:
            source = "alias_of_future_10d_return_from_api_price"
            severity = "medium_name_risk"
            api_only = True
            future_label = True
        elif label in {"high_conviction_target", "avoid_target", "ideal_weight_bucket", "ideal_cash_reserve_bucket", "positive_trade"}:
            source = "derived_from_future_10d_return_candidate_day_distribution"
            severity = "medium_universe_dependency"
            api_only = True
            future_label = True
        elif label in {"actual_net_profit", "actual_buy_amount"}:
            source = "executed_trade_outcome"
            severity = "block"
            backtest = True
            realized_trade = True
        return {
            "label": label,
            "label_source": source,
            "api_only": api_only,
            "backtest_derived": backtest,
            "future_return_api_only_label": future_label,
            "realized_trade_result": realized_trade,
            "leakage_severity": severity,
        }

    def _series_equal(self, dataset: pd.DataFrame, left: str, right: str) -> bool:
        if left not in dataset.columns or right not in dataset.columns or dataset.empty:
            return False
        left_values = pd.to_numeric(dataset[left], errors="coerce")
        right_values = pd.to_numeric(dataset[right], errors="coerce")
        mask = left_values.notna() & right_values.notna()
        if not mask.any():
            return False
        return bool(np.allclose(left_values[mask], right_values[mask], equal_nan=True))

    def _model_artifact_audit(self, features: list[str]) -> dict[str, Any]:
        model_dir = self._root(PM_MODEL_DIR)
        model_files = sorted(path.name for path in model_dir.glob("*.joblib")) if model_dir.exists() else []
        metadata = _read_json(model_dir / "model_metadata.json") or {}
        metrics = _read_json(model_dir / "metrics.json") or {}
        return {
            "model_files": model_files,
            "metadata_files": sorted(path.name for path in model_dir.glob("*.json")) if model_dir.exists() else [],
            "preprocess_files": sorted(path.name for path in model_dir.glob("*imput*")) + sorted(path.name for path in model_dir.glob("*scal*")),
            "feature_schema_available": bool(features),
            "label_schema_available": isinstance(metadata.get("targets"), dict),
            "training_config_available": bool(metadata),
            "importances_available": bool(model_files),
            "model_artifact_complete": bool(model_files and features and metrics and metadata),
        }

    def _feature_importance_audit(self, features: list[str]) -> dict[str, Any]:
        try:
            import joblib
        except Exception:
            return {"top_features": [], "suspicious_top_features": [], "forbidden_top_features": [], "reason": "joblib unavailable"}
        top_rows: list[dict[str, Any]] = []
        for path in sorted(self._root(PM_MODEL_DIR).glob("*.joblib")):
            try:
                model = joblib.load(path)
            except Exception:
                continue
            importances = getattr(model, "feature_importances_", None)
            if importances is None:
                continue
            for feature, importance in sorted(zip(features, importances), key=lambda item: float(item[1]), reverse=True)[:10]:
                top_rows.append(
                    {
                        "model": path.stem,
                        "feature": feature,
                        "importance": float(importance),
                        "category": classify_pm_column(feature),
                    }
                )
        suspicious = [row for row in top_rows if row["category"] in SUSPICIOUS_FEATURE_CATEGORIES]
        forbidden = [row for row in top_rows if row["category"] in FORBIDDEN_FEATURE_CATEGORIES]
        return {
            "top_features": top_rows,
            "suspicious_top_features": suspicious,
            "forbidden_top_features": forbidden,
        }

    def _safety_assessment(self, feature_audit: dict[str, Any], target_audit: dict[str, Any], artifact_audit: dict[str, Any]) -> dict[str, Any]:
        forbidden_features = feature_audit["forbidden_columns_used_as_features"]
        forbidden_targets = target_audit["target_labels_forbidden_for_retraining"]
        if forbidden_features:
            risk = "high"
            trust = "low_trust"
            reason = "forbidden backtest/outcome columns were reconstructed in training feature_columns"
        elif forbidden_targets:
            risk = "medium_high"
            trust = "medium_trust"
            reason = "features look safe, but one or more target labels are executed-trade outcomes"
        elif feature_audit["suspicious_columns_used_as_features"]:
            risk = "medium"
            trust = "medium_trust"
            reason = "no outcome/cash/trade features were used, but same-day ranking/model-prediction features require lineage documentation"
        else:
            risk = "low"
            trust = "high_trust"
            reason = "training features exclude forbidden outcome/audit columns and labels are future-return derived"
        if not artifact_audit["feature_schema_available"]:
            risk = "medium_high"
            trust = "low_trust"
            reason = "feature schema could not be reconstructed"
        return {
            "current_pm_model_operational_risk": risk,
            "v282_result_trust_level": trust,
            "reason": reason,
        }

    def _retraining_judgement(self, feature_audit: dict[str, Any], target_audit: dict[str, Any], safety: dict[str, Any]) -> dict[str, Any]:
        direct_allowed = (
            not feature_audit["forbidden_columns_used_as_features"]
            and not target_audit["target_labels_forbidden_for_retraining"]
            and safety["current_pm_model_operational_risk"] in {"low", "medium"}
        )
        rebuild_required = bool(target_audit["target_rebuild_required"] or feature_audit["suspicious_columns_used_as_features"])
        if feature_audit["forbidden_columns_used_as_features"]:
            next_phase = "Phase 7-C PM AI Feature Leakage Fix"
        elif rebuild_required:
            next_phase = "Phase 7-C PM AI API-only Dataset Design"
        else:
            next_phase = "Keep v2_82 but mark PM AI as legacy-risk"
        return {
            "pm_ai_current_model_safe_to_use": safety["v282_result_trust_level"] != "low_trust",
            "pm_ai_direct_retraining_allowed": direct_allowed and not rebuild_required,
            "pm_ai_dataset_rebuild_required": rebuild_required,
            "api_only_pm_dataset_rebuild_priority": "high" if rebuild_required else "medium",
            "next_phase_recommended": next_phase,
        }

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not columns:
            return ""
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)


def build_and_save_phase7b_report(root: Path | str = ROOT) -> Phase7BPaths:
    audit = Phase7BPMLeakageForensics(root)
    return audit.save_report(audit.build_report())

