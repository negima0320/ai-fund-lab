"""Phase 7-B' PM AI leakage forensics fix and re-audit.

This module fixes the Phase 7-B false-positive around ``close_position`` and
separates true forbidden outcome/state features from candidate-list dependent
features that may be reproducible at prediction time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.phase7b_pm_ai_leakage_forensics import (
    API_FINANCIAL_COLUMNS,
    FUTURE_LABEL_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_PREDICTION_COLUMNS,
    PM_DATASET,
    PM_MODEL_DIR,
    ROOT,
    TARGET_COLUMNS,
    Phase7BPMLeakageForensics,
    _format,
    _read_json,
    _read_parquet,
)


REPORT_STEM = "phase7b_prime_pm_ai_leakage_fix_2023-01_to_2026-05"

SAFE_PRICE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "close_position",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "gap_up_ratio",
    "daily_range_ratio",
    "volume",
    "turnover_value",
}

SAFE_PRICE_PREFIXES = (
    "return_",
    "ma",
    "gap_",
    "volume_ratio_",
    "turnover_ratio_",
)

SAFE_MARKET_PREFIXES = ("topix_return_", "relative_return_")

SUSPICIOUS_CANDIDATE_COLUMNS = {
    "candidate_count_in_day",
    "rank_in_day",
    "score_rank_in_day",
    "candidate_rank",
    "score_rank",
    "day_candidate_strength",
    "day_risk_level",
}

FORBIDDEN_EXACT_COLUMNS = {
    "decision",
    "skip_reason",
    "exit_reason",
    "cash_before",
    "cash_after",
    "cash_before_ratio",
    "daily_buy_limit_remaining_before",
    "daily_buy_limit_remaining_after",
    "max_positions_remaining_before",
    "selected_count_in_day",
    "actual_buy_amount",
    "actual_net_profit",
    "actual_shares",
    "actual_holding_days",
    "planned_amount",
    "planned_shares",
    "scaled_amount",
    "scaled_shares",
    "final_amount",
    "final_shares",
    "allocation_limit",
    "allocation_reason",
    "reject_reason",
    "scale_reason",
    "trade_id",
}

FORBIDDEN_PREFIXES = ("actual_", "realized_", "profit_", "win_", "loss_", "portfolio_", "position_")

FORBIDDEN_FEATURE_CATEGORIES_PRIME = {
    "backtest_position_state",
    "backtest_cash_state",
    "backtest_trade_outcome",
    "backtest_decision_or_reason",
    "future_label",
    "target_label",
    "identifier",
}

SUSPICIOUS_FEATURE_CATEGORIES_PRIME = {"candidate_list_dependent", "model_prediction_feature", "unknown"}


@dataclass(frozen=True)
class Phase7BPrimePaths:
    markdown: Path
    json: Path


def classify_pm_column_prime(column: str) -> str:
    lower = column.lower()
    if column in TARGET_COLUMNS:
        return "target_label"
    if column in FUTURE_LABEL_COLUMNS or lower.startswith("future_") or lower.startswith("return_after_"):
        return "future_label"
    if column in IDENTIFIER_COLUMNS:
        return "identifier"
    if column in SAFE_PRICE_COLUMNS or lower.startswith(SAFE_PRICE_PREFIXES):
        return "api_price_feature"
    if lower.startswith(SAFE_MARKET_PREFIXES):
        return "api_market_feature"
    if column in API_FINANCIAL_COLUMNS:
        return "api_financial_feature"
    if column in SUSPICIOUS_CANDIDATE_COLUMNS:
        return "candidate_list_dependent"
    if lower.startswith("day_avg_") or lower.startswith("day_max_"):
        return "candidate_list_dependent"
    if lower.endswith("_percentile_in_day") or lower.endswith("_gap_to_best"):
        return "candidate_list_dependent"
    if "candidate_strength" in lower:
        return "candidate_list_dependent"
    if column in FORBIDDEN_EXACT_COLUMNS or lower.startswith(FORBIDDEN_PREFIXES):
        if "cash" in lower:
            return "backtest_cash_state"
        if lower.startswith("actual_") or lower.startswith("realized_") or "profit" in lower or lower.startswith(("win_", "loss_")):
            return "backtest_trade_outcome"
        if "position" in lower or "shares" in lower or "amount" in lower or "holding_days" in lower:
            return "backtest_position_state"
        return "backtest_decision_or_reason"
    if column in MODEL_PREDICTION_COLUMNS:
        return "model_prediction_feature"
    return "unknown"


class Phase7BPrimePMLeakageFixAudit(Phase7BPMLeakageForensics):
    def build_report(self) -> dict[str, Any]:
        dataset = _read_parquet(self._root(PM_DATASET))
        metadata = _read_json(self._root(PM_MODEL_DIR) / "model_metadata.json") or {}
        feature_columns = self._training_features(metadata)
        column_audit = self._dataset_column_audit(dataset)
        feature_audit = self._training_feature_audit(feature_columns, dataset, metadata)
        target_audit = self._target_label_audit(dataset, metadata)
        artifact_audit = self._model_artifact_audit(feature_columns)
        importance = self._feature_importance_audit(feature_columns)
        candidate_lineage = self._candidate_feature_lineage(feature_columns)
        final = self._final_judgement(feature_audit, target_audit, candidate_lineage)
        return {
            "metadata": {
                "phase": "7-B-prime",
                "audit_only": True,
                "model_retraining_executed": False,
                "full_backtest_executed": False,
                "profile_added": False,
                "current_model_overwritten": False,
                "full_pytest_executed": False,
                "jquants_api_refetch": False,
                "openai_used": False,
                "live_order": False,
                "fix_summary": [
                    "close_position is classified as api_price_feature",
                    "candidate-list dependent columns are suspicious/conditional, not automatically forbidden",
                    "position-state detection uses explicit backtest columns and position_ prefix only",
                ],
            },
            "input_paths": {
                "pm_model_dir": str(self._root(PM_MODEL_DIR)),
                "pm_dataset": str(self._root(PM_DATASET)),
                "phase7b_report_reference": str(self.root / "reports" / "ml" / "phase7b_pm_ai_leakage_forensics_2023-01_to_2026-05.json"),
            },
            "dataset_summary": self._dataset_summary(dataset),
            "dataset_column_audit": column_audit,
            "training_feature_audit": feature_audit,
            "target_label_audit": target_audit,
            "model_artifact_audit": artifact_audit,
            "feature_importance_audit": importance,
            "candidate_feature_lineage_audit": candidate_lineage,
            "final_judgement": final,
        }

    def save_report(self, result: dict[str, Any]) -> Phase7BPrimePaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path = report_dir / f"{REPORT_STEM}.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase7BPrimePaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# AI Retraining Phase 7-B' PM AI Leakage Forensics Fix",
            "",
            "## Scope",
            "",
            "- re-audit only",
            "- no retraining, no backtest, no profile addition, no current model overwrite, no full pytest",
            "",
            "## Fix Summary",
            "",
        ]
        lines.extend(f"- {item}" for item in result["metadata"]["fix_summary"])
        lines.extend(
            [
                "",
                "## Dataset Column Audit",
                "",
                self._table(
                    [result["dataset_column_audit"]],
                    [
                        "total_columns",
                        "safe_feature_columns_count",
                        "suspicious_columns_count",
                        "forbidden_columns_count",
                        "target_columns_count",
                        "unknown_columns_count",
                    ],
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
                    ],
                ),
                "",
                "## Candidate Feature Lineage",
                "",
                self._table(
                    result["candidate_feature_lineage_audit"],
                    ["feature_group", "features", "generated_from", "prediction_time_available", "live_reproducible", "walk_forward_reproducible", "verdict"],
                ),
                "",
                "## Target Label Audit",
                "",
                self._table(
                    result["target_label_audit"]["labels"],
                    ["label", "label_source", "api_only", "backtest_derived", "future_return_api_only_label", "realized_trade_result", "leakage_severity"],
                ),
                "",
                "## Feature Importance",
                "",
                self._table(result["feature_importance_audit"].get("top_features", []), ["model", "feature", "importance", "category"]),
                "",
                "## Final Judgement",
                "",
                self._table(
                    [result["final_judgement"]],
                    [
                        "feature_leakage_confirmed",
                        "feature_leakage_suspected",
                        "feature_leakage_not_confirmed",
                        "current_pm_model_safe_to_use",
                        "v282_result_trust_level",
                        "pm_ai_direct_retraining_allowed",
                        "pm_ai_dataset_rebuild_required",
                        "next_phase_recommended",
                    ],
                ),
                "",
            ]
        )
        return "\n".join(lines)

    def _dataset_column_audit(self, dataset):  # type: ignore[override]
        rows = []
        by_category: dict[str, list[str]] = {}
        for column in dataset.columns:
            category = classify_pm_column_prime(str(column))
            by_category.setdefault(category, []).append(str(column))
            rows.append({"column": str(column), "category": category})
        safe_categories = {"api_price_feature", "api_financial_feature", "api_market_feature", "derived_from_api_only"}
        suspicious_categories = {"candidate_list_dependent", "model_prediction_feature", "unknown"}
        forbidden_categories = {"backtest_position_state", "backtest_cash_state", "backtest_trade_outcome", "backtest_decision_or_reason", "future_label"}
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

    def _training_feature_audit(self, features, dataset, metadata):  # type: ignore[override]
        categorized = [{"feature": feature, "category": classify_pm_column_prime(feature)} for feature in features]
        forbidden = [row["feature"] for row in categorized if row["category"] in FORBIDDEN_FEATURE_CATEGORIES_PRIME]
        suspicious = [row["feature"] for row in categorized if row["category"] in SUSPICIOUS_FEATURE_CATEGORIES_PRIME]
        api_only = [
            row["feature"]
            for row in categorized
            if row["category"] in {"api_price_feature", "api_financial_feature", "api_market_feature", "derived_from_api_only"}
        ]
        return {
            "training_feature_count": len(features),
            "training_feature_columns": features,
            "training_feature_categories": categorized,
            "forbidden_columns_used_as_features": sorted(forbidden),
            "suspicious_columns_used_as_features": sorted(suspicious),
            "api_only_feature_count": len(api_only),
            "non_api_feature_count": len(features) - len(api_only),
            "missing_training_features_in_dataset": sorted(feature for feature in features if feature not in dataset.columns),
            "unable_to_reconstruct_training_features": not features,
            "reason": "" if features else "feature_columns not found in metadata or feature_columns.json",
            "metadata_leakage_guard": metadata.get("leakage_guard", ""),
        }

    def _feature_importance_audit(self, features):  # type: ignore[override]
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
                        "category": classify_pm_column_prime(feature),
                    }
                )
        suspicious = [row for row in top_rows if row["category"] in SUSPICIOUS_FEATURE_CATEGORIES_PRIME]
        forbidden = [row for row in top_rows if row["category"] in FORBIDDEN_FEATURE_CATEGORIES_PRIME]
        return {"top_features": top_rows, "suspicious_top_features": suspicious, "forbidden_top_features": forbidden}

    def _candidate_feature_lineage(self, features: list[str]) -> list[dict[str, Any]]:
        groups = [
            ("candidate_count_in_day", [f for f in features if f == "candidate_count_in_day"]),
            ("rank_in_day", [f for f in features if f in {"rank_in_day", "score_rank_in_day"}]),
            ("percentile_in_day", [f for f in features if f.endswith("_percentile_in_day")]),
            ("gap_to_best", [f for f in features if f.endswith("_gap_to_best")]),
            ("day_aggregate", [f for f in features if f.startswith("day_avg_") or f.startswith("day_max_") or f in {"day_candidate_strength", "day_risk_level"}]),
        ]
        rows = []
        for name, group_features in groups:
            if not group_features:
                continue
            rows.append(
                {
                    "feature_group": name,
                    "features": sorted(group_features),
                    "generated_from": "same-day candidate universe and precomputed stock-selection/walk-forward prediction scores",
                    "prediction_time_available": "yes_if_full_candidate_list_is_built_before_pm_scoring",
                    "live_reproducible": "conditional",
                    "walk_forward_reproducible": "conditional",
                    "verdict": "conditional",
                    "reason": "Not an executed-trade outcome, but direct retraining needs an API-only candidate universe builder with no selected_count_in_day/backtest selection dependency.",
                }
            )
        return rows

    def _final_judgement(self, feature_audit: dict[str, Any], target_audit: dict[str, Any], candidate_lineage: list[dict[str, Any]]) -> dict[str, Any]:
        forbidden_used = feature_audit["forbidden_columns_used_as_features"]
        suspicious_used = feature_audit["suspicious_columns_used_as_features"]
        confirmed = bool(forbidden_used)
        suspected = bool(suspicious_used or candidate_lineage)
        not_confirmed = not confirmed
        if confirmed:
            trust = "low_trust"
            safe_to_use = False
            next_phase = "Phase 7-C PM AI Feature Leakage Fix"
        elif suspected:
            trust = "medium_trust"
            safe_to_use = True
            next_phase = "Phase 7-C PM AI API-only Dataset Design"
        else:
            trust = "high_trust"
            safe_to_use = True
            next_phase = "Keep v2_82 but mark PM AI as legacy-risk"
        return {
            "feature_leakage_confirmed": confirmed,
            "feature_leakage_suspected": suspected,
            "feature_leakage_not_confirmed": not_confirmed,
            "current_pm_model_safe_to_use": safe_to_use,
            "v282_result_trust_level": trust,
            "pm_ai_direct_retraining_allowed": False,
            "pm_ai_dataset_rebuild_required": True,
            "target_rebuild_required": target_audit["target_rebuild_required"],
            "next_phase_recommended": next_phase,
        }

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)


def build_and_save_phase7b_prime_report(root: Path | str = ROOT) -> Phase7BPrimePaths:
    audit = Phase7BPrimePMLeakageFixAudit(root)
    return audit.save_report(audit.build_report())

