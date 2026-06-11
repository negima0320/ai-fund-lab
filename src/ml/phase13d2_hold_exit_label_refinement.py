"""Phase 13-D2 Hold / Exit Label Refinement.

This is a 2025-only, read-only label refinement audit for Hold / Exit AI.
It reuses existing artifacts and Phase13-D label definitions, then checks
overlap, conflict, and mutually exclusive exit action label candidates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase13a_horizon_reality_audit import ROOT
from ml.phase13d_hold_exit_dataset_audit import (
    CANDIDATE_SETS,
    LABEL_DEFINITIONS,
    Phase13DHoldExitDatasetAudit,
)


REPORT_STEM = "phase13d2_hold_exit_label_refinement_2025"
PRIMARY_CANDIDATE_SET = "candidate_strength_top50"
RAW_LABELS = [
    "hold_correct_20d",
    "sell_early_risk_20d",
    "profit_protection_needed_20d",
    "break_even_guard_needed_20d",
    "stop_loss_too_late_20d",
    "high_upside_high_downside_20d",
    "low_upside_low_downside_20d",
    "avoid_candidate",
]
FOCUSED_OVERLAPS = [
    ("profit_protection_needed_20d", "break_even_guard_needed_20d"),
    ("profit_protection_needed_20d", "stop_loss_too_late_20d"),
    ("break_even_guard_needed_20d", "stop_loss_too_late_20d"),
    ("sell_early_risk_20d", "stop_loss_too_late_20d"),
    ("sell_early_risk_20d", "hold_correct_20d"),
    ("high_upside_high_downside_20d", "hold_correct_20d"),
    ("low_upside_low_downside_20d", "avoid_candidate"),
]
REQUIRED_REPORT_KEYS = [
    "hold_exit_label_refinement_ready",
    "recommended_model_task",
    "recommended_next_phase",
    "ready_for_phase13d3",
    "ready_for_phase13e",
    "leakage_risk",
    "blocking_issues",
]


@dataclass(frozen=True)
class Phase13D2Paths:
    markdown: Path
    json: Path


class Phase13D2HoldExitLabelRefinement:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)
        self.phase13d = Phase13DHoldExitDatasetAudit(root)

    def run(self) -> Phase13D2Paths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data, source_info = self.phase13d.phase13a.load_comparison_dataset()
        data = self.add_refined_labels(self.phase13d.add_hold_exit_labels(data))
        leakage = self.leakage_checklist()
        leakage["blocking_issues"] = self.phase13d.blocking_issues(data, source_info)
        leakage["leakage_risk"] = "low" if not leakage["blocking_issues"] else "medium"
        primary = self.primary_frame(data)
        label_rates = self.label_rates(primary)
        overlap = self.label_overlap_audit(primary)
        exclusive = self.exclusive_label_audit(data, primary)
        candidate_set_distribution = self.exclusive_distribution_by_candidate_set(data)
        priority = self.label_priority_design(overlap, label_rates)
        recommendations = self.recommendations(label_rates, overlap, exclusive, candidate_set_distribution, leakage)
        return {
            "metadata": self.metadata(),
            "input_artifact_summary": self.input_artifact_summary(data, source_info, leakage),
            "label_definitions": {**LABEL_DEFINITIONS, "avoid_candidate": "future_max_return_20d < 0.05 and (future_return_20d <= 0 or future_max_drawdown_20d <= -0.10)"},
            "primary_candidate_set": PRIMARY_CANDIDATE_SET,
            "primary_label_rates": label_rates,
            "label_overlap_conflict_audit": overlap,
            "recommended_label_priority": priority["recommended_label_priority"],
            "priority_reason": priority["priority_reason"],
            "conflict_resolution_rule": priority["conflict_resolution_rule"],
            "exclusive_label_definitions": self.exclusive_label_definitions(),
            "exclusive_label_distribution": exclusive["distribution"],
            "exclusive_label_distribution_by_candidate_set": candidate_set_distribution,
            "unlabeled_rate": exclusive["unlabeled_rate"],
            "multi_label_conflict_before_resolution": overlap["multi_label_conflict_before_resolution"],
            "dataset_readiness": self.dataset_readiness(exclusive, candidate_set_distribution),
            "final_recommendation": recommendations,
            "leakage_checklist": leakage,
            **{key: recommendations.get(key) for key in REQUIRED_REPORT_KEYS},
        }

    def add_refined_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        ret = _numeric(result.get("future_return_20d"))
        max_ret = _numeric(result.get("future_max_return_20d"))
        drawdown = _numeric(result.get("future_max_drawdown_20d"))
        result["avoid_candidate"] = ((max_ret < 0.05) & ((ret <= 0) | (drawdown <= -0.10))).astype(float)
        return result

    def primary_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        method = next(method for method in CANDIDATE_SETS if method.name == PRIMARY_CANDIDATE_SET)
        return self.phase13d.method_frame(data, method)

    def label_rates(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for label in RAW_LABELS:
            values = _numeric(frame.get(label))
            rows.append(
                {
                    "label": label,
                    "count": int(values.fillna(0).sum()) if label in frame.columns else 0,
                    "label_rate": _safe_float(values.mean()) if label in frame.columns and not values.dropna().empty else None,
                }
            )
        return rows

    def label_overlap_audit(self, frame: pd.DataFrame) -> dict[str, Any]:
        label_masks = {label: _numeric(frame.get(label)).fillna(0).astype(bool) for label in RAW_LABELS if label in frame.columns}
        pairwise = []
        for left, right in combinations(label_masks, 2):
            pairwise.append(self.overlap_row(left, right, label_masks[left], label_masks[right], len(frame)))
        focused = [
            self.overlap_row(left, right, label_masks[left], label_masks[right], len(frame))
            for left, right in FOCUSED_OVERLAPS
            if left in label_masks and right in label_masks
        ]
        active_count = pd.DataFrame(label_masks).sum(axis=1) if label_masks else pd.Series(dtype=float)
        conflict_mask = active_count > 1
        return {
            "sample_count": int(len(frame)),
            "label_rate": self.label_rates(frame),
            "pairwise_overlaps": pairwise,
            "focused_overlaps": focused,
            "multi_label_conflict_before_resolution": {
                "count": int(conflict_mask.sum()) if not conflict_mask.empty else 0,
                "rate": _safe_float(conflict_mask.mean()) if not conflict_mask.empty else None,
            },
            "max_pairwise_jaccard": max((self.value(row, "pairwise_jaccard") for row in pairwise), default=0.0),
        }

    def overlap_row(self, left: str, right: str, left_mask: pd.Series, right_mask: pd.Series, total: int) -> dict[str, Any]:
        overlap = left_mask & right_mask
        union = left_mask | right_mask
        return {
            "left_label": left,
            "right_label": right,
            "pairwise_overlap_count": int(overlap.sum()),
            "pairwise_overlap_rate": _safe_float(overlap.sum() / total) if total else None,
            "pairwise_jaccard": _safe_float(overlap.sum() / union.sum()) if int(union.sum()) else 0.0,
            "left_count": int(left_mask.sum()),
            "right_count": int(right_mask.sum()),
        }

    def label_priority_design(self, overlap: dict[str, Any], label_rates: list[dict[str, Any]]) -> dict[str, Any]:
        rates = {row["label"]: self.value(row, "label_rate") for row in label_rates}
        priority = [
            "PROFIT_PROTECT",
            "BREAK_EVEN_GUARD",
            "STOP_REWORK",
            "EXTEND_HOLD",
            "HOLD",
            "AVOID",
            "NO_ACTION",
        ]
        reason = (
            "Profit protection comes first because winner-to-loser leakage is direct annual-return leakage. "
            "Break-even guard follows because it is broader and overlaps with profit protection. "
            "Stop rework is applied after those to avoid turning recoverable profit-protection cases into pure stop labels. "
            "Extend/Hold labels are evaluated only after risk exits are resolved."
        )
        if rates.get("stop_loss_too_late_20d", 0.0) > rates.get("break_even_guard_needed_20d", 0.0):
            reason += " Stop-loss rate is high, but overlap handling keeps profit/break-even labels ahead of stop labels."
        return {
            "recommended_label_priority": priority,
            "priority_reason": reason,
            "conflict_resolution_rule": "Assign the first true label in recommended_label_priority; later labels are suppressed for mutually exclusive training targets.",
            "observed_multi_label_conflict_rate": overlap["multi_label_conflict_before_resolution"]["rate"],
        }

    def exclusive_label_definitions(self) -> dict[str, str]:
        return {
            "PROFIT_PROTECT": "future_max_return_20d >= 0.05 and future_return_20d < 0",
            "BREAK_EVEN_GUARD": "future_max_return_20d >= 0.03 and future_return_20d <= 0 and not PROFIT_PROTECT",
            "STOP_REWORK": "future_max_drawdown_20d <= -0.10 and future_return_20d < 0 and not PROFIT_PROTECT and not BREAK_EVEN_GUARD",
            "EXTEND_HOLD": "(future_return_20d > 0.05 or future_max_return_20d > 0.10) and no prior risk label",
            "HOLD": "future_return_20d > 0 and future_max_drawdown_20d > -0.10 and no prior risk/extend label",
            "AVOID": "future_max_return_20d < 0.05 and future_max_drawdown_20d <= -0.05 and no prior label",
            "NO_ACTION": "No exclusive label condition matched",
        }

    def exclusive_label_audit(self, data: pd.DataFrame, primary: pd.DataFrame) -> dict[str, Any]:
        labeled = self.assign_exclusive_labels(primary)
        distribution = self.distribution(labeled["exclusive_exit_action"])
        return {
            "distribution": distribution,
            "unlabeled_rate": next((row["rate"] for row in distribution if row["exclusive_exit_action"] == "NO_ACTION"), 0.0),
        }

    def exclusive_distribution_by_candidate_set(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for method in CANDIDATE_SETS:
            frame = self.phase13d.method_frame(data, method)
            labeled = self.assign_exclusive_labels(frame)
            for item in self.distribution(labeled["exclusive_exit_action"]):
                rows.append({"candidate_set": method.name, **item})
        return rows

    def assign_exclusive_labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        ret = _numeric(result.get("future_return_20d"))
        max_ret = _numeric(result.get("future_max_return_20d"))
        drawdown = _numeric(result.get("future_max_drawdown_20d"))
        action = pd.Series("NO_ACTION", index=result.index, dtype="object")
        unassigned = action.eq("NO_ACTION")
        profit = (max_ret >= 0.05) & (ret < 0)
        action.loc[unassigned & profit] = "PROFIT_PROTECT"
        unassigned = action.eq("NO_ACTION")
        break_even = (max_ret >= 0.03) & (ret <= 0)
        action.loc[unassigned & break_even] = "BREAK_EVEN_GUARD"
        unassigned = action.eq("NO_ACTION")
        stop = (drawdown <= -0.10) & (ret < 0)
        action.loc[unassigned & stop] = "STOP_REWORK"
        unassigned = action.eq("NO_ACTION")
        extend = (ret > 0.05) | (max_ret > 0.10)
        action.loc[unassigned & extend] = "EXTEND_HOLD"
        unassigned = action.eq("NO_ACTION")
        hold = (ret > 0) & (drawdown > -0.10)
        action.loc[unassigned & hold] = "HOLD"
        unassigned = action.eq("NO_ACTION")
        avoid = (max_ret < 0.05) & (drawdown <= -0.05)
        action.loc[unassigned & avoid] = "AVOID"
        result["exclusive_exit_action"] = action
        return result

    def distribution(self, labels: pd.Series) -> list[dict[str, Any]]:
        total = len(labels)
        counts = labels.value_counts(dropna=False)
        rows = []
        for label in ["PROFIT_PROTECT", "BREAK_EVEN_GUARD", "STOP_REWORK", "EXTEND_HOLD", "HOLD", "AVOID", "NO_ACTION"]:
            count = int(counts.get(label, 0))
            rows.append({"exclusive_exit_action": label, "count": count, "rate": _safe_float(count / total) if total else None})
        return rows

    def dataset_readiness(self, exclusive: dict[str, Any], by_set: list[dict[str, Any]]) -> dict[str, Any]:
        distribution = exclusive["distribution"]
        non_no_action = [row for row in distribution if row["exclusive_exit_action"] != "NO_ACTION"]
        min_count = min((row["count"] for row in non_no_action), default=0)
        max_rate = max((self.value(row, "rate") for row in distribution), default=0.0)
        unlabeled_rate = self.value(exclusive, "unlabeled_rate")
        usable_classes = [row["exclusive_exit_action"] for row in non_no_action if row["count"] >= 100]
        return {
            "primary_candidate_set": PRIMARY_CANDIDATE_SET,
            "minimum_non_no_action_class_count": int(min_count),
            "maximum_class_rate": _safe_float(max_rate),
            "unlabeled_rate": _safe_float(unlabeled_rate),
            "usable_classes_count_ge_100": usable_classes,
            "class_balance_warning": max_rate > 0.70 or unlabeled_rate > 0.50,
            "candidate_set_distribution_rows": len(by_set),
            "ready": len(usable_classes) >= 4 and unlabeled_rate <= 0.50,
        }

    def recommendations(
        self,
        label_rates: list[dict[str, Any]],
        overlap: dict[str, Any],
        exclusive: dict[str, Any],
        by_set: list[dict[str, Any]],
        leakage: dict[str, Any],
    ) -> dict[str, Any]:
        readiness = self.dataset_readiness(exclusive, by_set)
        conflict_rate = self.value(overlap["multi_label_conflict_before_resolution"], "rate")
        ready = readiness["ready"] and not leakage["blocking_issues"]
        model_task = "multiclass_exit_action" if ready else "multi_output_labels" if conflict_rate > 0.20 else "insufficient_evidence"
        next_phase = "Phase13-D3 Hold/Exit Rule Prototype" if ready else "Phase13-C Horizon-Aware Valuation Prototype"
        return {
            "hold_exit_label_refinement_ready": ready,
            "recommended_model_task": model_task,
            "recommended_next_phase": next_phase,
            "ready_for_phase13d3": ready,
            "ready_for_phase13e": False,
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
            "reason": (
                f"primary={PRIMARY_CANDIDATE_SET}; conflict_rate={conflict_rate:.4f}; "
                f"unlabeled_rate={self.value(exclusive, 'unlabeled_rate'):.4f}; "
                f"ready={ready}. Rule prototype is recommended before integrated strategy."
            ),
        }

    def input_artifact_summary(self, data: pd.DataFrame, source_info: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_files": source_info["source_files"],
            "row_count": int(len(data)),
            "date_min": data["date"].min().date().isoformat() if not data.empty else None,
            "date_max": data["date"].max().date().isoformat() if not data.empty else None,
            "unique_code_count": int(data["code"].nunique()) if not data.empty else 0,
            "available_score_columns": source_info["available_score_columns"],
            "available_future_columns": source_info["available_future_columns"],
            "missing_columns": source_info["missing_columns"],
            "candidate_sets": [method.name for method in CANDIDATE_SETS],
            "primary_candidate_set": PRIMARY_CANDIDATE_SET,
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_only_for_audit": self.phase13d.phase13a.expected_future_columns(),
            "future_columns_used_as_features": [],
            "backtest_columns_used_as_features": [],
            "trade_result_columns_used_as_features": [],
            "cash_or_portfolio_columns_used_as_features": [],
            "exit_reason_used_as_feature": False,
            "sell_result_used_as_feature": False,
            "current_pm_multiplier_used": False,
            "new_model_trained": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "strategy_backtest_executed": False,
            "full_backtest_executed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
            "period": {"start": "2025-01-01", "end": "2025-12-31"},
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "13-D2",
            "scope": "2025-only hold/exit label refinement audit",
            "primary_candidate_set": PRIMARY_CANDIDATE_SET,
            "new_model_trained": False,
            "strategy_backtest_executed": False,
            "full_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase13D2Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase13D2Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Phase 13-D2 Hold / Exit Label Refinement",
                "",
                "## Input Artifact Summary",
                "",
                self.table([report["input_artifact_summary"]], ["row_count", "date_min", "date_max", "unique_code_count", "available_score_columns", "available_future_columns", "missing_columns", "candidate_sets", "primary_candidate_set", "leakage_risk", "blocking_issues"]),
                "",
                "## Primary Label Rates",
                "",
                self.table(report["primary_label_rates"], ["label", "count", "label_rate"]),
                "",
                "## Focused Label Overlaps",
                "",
                self.table(report["label_overlap_conflict_audit"]["focused_overlaps"], ["left_label", "right_label", "pairwise_overlap_count", "pairwise_overlap_rate", "pairwise_jaccard", "left_count", "right_count"]),
                "",
                "## Label Priority",
                "",
                self.table([report], ["recommended_label_priority", "priority_reason", "conflict_resolution_rule", "multi_label_conflict_before_resolution"]),
                "",
                "## Exclusive Label Definitions",
                "",
                self.table([{"exclusive_exit_action": key, "definition": value} for key, value in report["exclusive_label_definitions"].items()], ["exclusive_exit_action", "definition"]),
                "",
                "## Exclusive Label Distribution",
                "",
                self.table(report["exclusive_label_distribution"], ["exclusive_exit_action", "count", "rate"]),
                "",
                "## Dataset Readiness",
                "",
                self.table([report["dataset_readiness"]], ["primary_candidate_set", "minimum_non_no_action_class_count", "maximum_class_rate", "unlabeled_rate", "usable_classes_count_ge_100", "class_balance_warning", "ready"]),
                "",
                "## Final Recommendation",
                "",
                self.table([report["final_recommendation"]], REQUIRED_REPORT_KEYS + ["reason"]),
                "",
                "## Leakage Checklist",
                "",
                self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_audit", "new_model_trained", "existing_model_overwritten", "profile_changed", "strategy_backtest_executed", "full_backtest_executed", "historical_predictions_regenerated", "jquants_api_called", "openai_api_called", "leakage_risk", "blocking_issues"]),
                "",
            ]
        )

    def table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(self.format_value(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def value(self, row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        try:
            if value is None:
                return 0.0
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if math.isnan(result) else result

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
