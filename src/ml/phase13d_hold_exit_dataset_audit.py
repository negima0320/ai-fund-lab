"""Phase 13-D Hold / Exit AI Dataset Audit.

This is a 2025-only, read-only dataset audit for Hold / Exit label design.
It uses existing artifacts only and does not train models, regenerate
predictions, run strategy backtests, change profiles, overwrite models, or
call external APIs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase13a_horizon_reality_audit import ROOT, Phase13AHorizonRealityAudit
from ml.phase13b_candidate_generation_redesign import CandidateMethod


REPORT_STEM = "phase13d_hold_exit_dataset_audit_2025"
REQUIRED_REPORT_KEYS = [
    "recommended_exit_hold_primary_horizon",
    "recommended_exit_hold_labels",
    "profit_protection_needed",
    "break_even_guard_needed",
    "stop_loss_rework_needed",
    "hold_exit_ai_dataset_ready",
    "recommended_next_phase",
    "ready_for_phase13d2",
    "ready_for_phase13e",
    "leakage_risk",
    "blocking_issues",
]


@dataclass(frozen=True)
class Phase13DPaths:
    markdown: Path
    json: Path


CANDIDATE_SETS = [
    CandidateMethod("candidate_universe_baseline", None, None, "all"),
    CandidateMethod("stock_selection_rank_score_top50", "stock_selection_rank_score", 50),
    CandidateMethod("candidate_strength_top50", "candidate_strength", 50),
    CandidateMethod("valuation_first_top50", "opportunity_proba", 50),
    CandidateMethod("opportunity_downside_top50", "opportunity_downside_score", 50),
    CandidateMethod("candidate_strength_top5", "candidate_strength", 5),
    CandidateMethod("valuation_first_top5", "opportunity_proba", 5),
    CandidateMethod("opportunity_downside_top5", "opportunity_downside_score", 5),
]

LABEL_DEFINITIONS = {
    "hold_correct_20d": "future_return_20d > 0 and future_max_drawdown_20d > -0.10",
    "sell_early_risk_20d": "future_return_20d > 0.05 or future_max_return_20d > 0.10",
    "profit_protection_needed_20d": "future_max_return_20d >= 0.05 and future_return_20d < 0",
    "break_even_guard_needed_20d": "future_max_return_20d >= 0.03 and future_return_20d <= 0",
    "stop_loss_too_late_20d": "future_max_drawdown_20d <= -0.10 and future_return_20d < 0",
    "high_upside_high_downside_20d": "future_max_return_20d >= 0.10 and future_max_drawdown_20d <= -0.10",
    "low_upside_low_downside_20d": "future_max_return_20d < 0.05 and future_max_drawdown_20d > -0.05",
}


class Phase13DHoldExitDatasetAudit:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)
        self.phase13a = Phase13AHorizonRealityAudit(root)

    def run(self) -> Phase13DPaths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data, source_info = self.phase13a.load_comparison_dataset()
        data = self.add_hold_exit_labels(data)
        leakage = self.leakage_checklist()
        leakage["blocking_issues"] = self.blocking_issues(data, source_info)
        leakage["leakage_risk"] = "low" if not leakage["blocking_issues"] else "medium"
        profit_leakage = [self.profit_leakage_for_set(data, method) for method in CANDIDATE_SETS]
        label_summary = [self.label_summary_for_set(data, method) for method in CANDIDATE_SETS]
        failure_modes = [self.failure_mode_for_set(data, method) for method in CANDIDATE_SETS]
        recommendations = self.recommendations(profit_leakage, label_summary, failure_modes, leakage)
        return {
            "metadata": self.metadata(),
            "input_artifact_summary": self.input_artifact_summary(data, source_info, leakage),
            "label_definitions": LABEL_DEFINITIONS,
            "profit_leakage_audit": profit_leakage,
            "hold_correctness_label_summary": label_summary,
            "exit_hold_failure_mode_summary": failure_modes,
            "final_recommendation": recommendations,
            "leakage_checklist": leakage,
            **{key: recommendations.get(key) for key in REQUIRED_REPORT_KEYS},
        }

    def add_hold_exit_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        ret = _numeric(result.get("future_return_20d"))
        max_ret = _numeric(result.get("future_max_return_20d"))
        drawdown = _numeric(result.get("future_max_drawdown_20d"))
        result["hold_correct_20d"] = ((ret > 0) & (drawdown > -0.10)).astype(float)
        result["sell_early_risk_20d"] = ((ret > 0.05) | (max_ret > 0.10)).astype(float)
        result["profit_protection_needed_20d"] = ((max_ret >= 0.05) & (ret < 0)).astype(float)
        result["break_even_guard_needed_20d"] = ((max_ret >= 0.03) & (ret <= 0)).astype(float)
        result["stop_loss_too_late_20d"] = ((drawdown <= -0.10) & (ret < 0)).astype(float)
        result["high_upside_high_downside_20d"] = ((max_ret >= 0.10) & (drawdown <= -0.10)).astype(float)
        result["low_upside_low_downside_20d"] = ((max_ret < 0.05) & (drawdown > -0.05)).astype(float)
        return result

    def profit_leakage_for_set(self, data: pd.DataFrame, method: CandidateMethod) -> dict[str, Any]:
        frame = self.method_frame(data, method)
        peak5_to_loss = self.peak_to_final_loss(frame, 0.05)
        peak10_to_loss = self.peak_to_final_loss(frame, 0.10)
        leakage_values = self.peak_to_final_leakage(frame)
        return {
            "candidate_set": method.name,
            "score_column": method.score_column,
            "top_n": method.top_n,
            "sample_count": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if not frame.empty else 0,
            "avg_candidates_per_day": self.avg_candidates_per_day(frame),
            "mean_future_return_5d": self.mean(frame, "future_return_5d"),
            "mean_future_return_10d": self.mean(frame, "future_return_10d"),
            "mean_future_return_20d": self.mean(frame, "future_return_20d"),
            "mean_future_max_return_20d": self.mean(frame, "future_max_return_20d"),
            "mean_future_max_drawdown_20d": self.mean(frame, "future_max_drawdown_20d"),
            "top_decile_rate_20d": self.mean(frame, "top_decile_20d"),
            "downside_bad_rate_20d": self.mean(frame, "downside_bad_20d"),
            "peak_5pct_to_final_loss_20d_count": peak5_to_loss["count"],
            "peak_5pct_to_final_loss_20d_rate": peak5_to_loss["rate"],
            "peak_10pct_to_final_loss_20d_count": peak10_to_loss["count"],
            "peak_10pct_to_final_loss_20d_rate": peak10_to_loss["rate"],
            "estimated_peak_to_final_leakage_20d": leakage_values["sum"],
            "estimated_peak_to_final_leakage_20d_mean": leakage_values["mean"],
            "estimated_peak_to_final_leakage_basis": "return basis: sum(max(0, future_max_return_20d - future_return_20d))",
        }

    def label_summary_for_set(self, data: pd.DataFrame, method: CandidateMethod) -> dict[str, Any]:
        frame = self.method_frame(data, method)
        row = {
            "candidate_set": method.name,
            "sample_count": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if not frame.empty else 0,
            "avg_candidates_per_day": self.avg_candidates_per_day(frame),
        }
        for label in LABEL_DEFINITIONS:
            row[f"{label}_rate"] = self.mean(frame, label)
        return row

    def failure_mode_for_set(self, data: pd.DataFrame, method: CandidateMethod) -> dict[str, Any]:
        frame = self.method_frame(data, method)
        ret = _numeric(frame.get("future_return_20d"))
        max_ret = _numeric(frame.get("future_max_return_20d"))
        drawdown = _numeric(frame.get("future_max_drawdown_20d"))
        winner_to_loser = (max_ret >= 0.05) & (ret < 0)
        early_sell_risk = (ret > 0.05) | (max_ret > 0.10)
        late_stop_loss = (drawdown <= -0.10) & (ret < 0)
        profit_protection = winner_to_loser | ((max_ret >= 0.10) & (ret < 0.03))
        break_even = (max_ret >= 0.03) & (ret <= 0)
        hold_candidate = (ret > 0) & (drawdown > -0.10)
        avoid_candidate = (max_ret < 0.05) & ((ret <= 0) | (drawdown <= -0.10))
        return {
            "candidate_set": method.name,
            "sample_count": int(len(frame)),
            "winner_to_loser": self.mask_count_rate(winner_to_loser),
            "early_sell_risk": self.mask_count_rate(early_sell_risk),
            "late_stop_loss_risk": self.mask_count_rate(late_stop_loss),
            "profit_protection_candidate": self.mask_count_rate(profit_protection),
            "break_even_guard_candidate": self.mask_count_rate(break_even),
            "hold_candidate": self.mask_count_rate(hold_candidate),
            "avoid_candidate": self.mask_count_rate(avoid_candidate),
        }

    def method_frame(self, data: pd.DataFrame, method: CandidateMethod) -> pd.DataFrame:
        if method.mode == "all":
            return data.copy()
        if method.score_column is None or method.top_n is None:
            raise ValueError(f"Invalid candidate set: {method}")
        return self.phase13a.top_n_by_day(data, method.score_column, method.top_n)

    def recommendations(
        self,
        profit_leakage: list[dict[str, Any]],
        label_summary: list[dict[str, Any]],
        failure_modes: list[dict[str, Any]],
        leakage: dict[str, Any],
    ) -> dict[str, Any]:
        preferred = self.by_name(profit_leakage, "candidate_strength_top50") or (profit_leakage[0] if profit_leakage else {})
        preferred_labels = self.by_name(label_summary, "candidate_strength_top50") or {}
        preferred_failures = self.by_name(failure_modes, "candidate_strength_top50") or {}
        protection_rate = self.value(preferred_labels, "profit_protection_needed_20d_rate")
        break_even_rate = self.value(preferred_labels, "break_even_guard_needed_20d_rate")
        stop_loss_rate = self.value(preferred_labels, "stop_loss_too_late_20d_rate")
        peak_loss_rate = self.value(preferred, "peak_5pct_to_final_loss_20d_rate")
        labels = ["hold_correct_20d", "sell_early_risk_20d"]
        if protection_rate >= 0.05 or peak_loss_rate >= 0.05:
            labels.append("profit_protection_needed_20d")
        if break_even_rate >= 0.08:
            labels.append("break_even_guard_needed_20d")
        if stop_loss_rate >= 0.08:
            labels.append("stop_loss_too_late_20d")
        labels.extend(["high_upside_high_downside_20d", "low_upside_low_downside_20d"])
        hold_exit_ready = not leakage["blocking_issues"] and all(
            preferred.get(column) is not None
            for column in ["mean_future_return_20d", "mean_future_max_return_20d", "mean_future_max_drawdown_20d"]
        )
        needs_d2 = protection_rate >= 0.05 or break_even_rate >= 0.08 or stop_loss_rate >= 0.08
        next_phase = "Phase13-D2 Hold/Exit Label Refinement" if needs_d2 else "Phase13-E Integrated Strategy Prototype"
        return {
            "recommended_exit_hold_primary_horizon": "20d",
            "recommended_exit_hold_labels": labels,
            "profit_protection_needed": protection_rate >= 0.05 or peak_loss_rate >= 0.05,
            "break_even_guard_needed": break_even_rate >= 0.08,
            "stop_loss_rework_needed": stop_loss_rate >= 0.08,
            "hold_exit_ai_dataset_ready": hold_exit_ready,
            "recommended_next_phase": next_phase,
            "ready_for_phase13d2": needs_d2 and hold_exit_ready,
            "ready_for_phase13e": (not needs_d2) and hold_exit_ready,
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
            "primary_candidate_set": "candidate_strength_top50",
            "primary_candidate_profit_leakage": preferred,
            "primary_candidate_label_rates": preferred_labels,
            "primary_candidate_failure_modes": preferred_failures,
            "reason": self.recommendation_reason(preferred, preferred_labels, preferred_failures, labels, next_phase),
        }

    def recommendation_reason(
        self,
        leakage_row: dict[str, Any],
        label_row: dict[str, Any],
        failure_row: dict[str, Any],
        labels: list[str],
        next_phase: str,
    ) -> str:
        return (
            "candidate_strength_top50 is used as the primary Phase13-B candidate set. "
            f"peak_5pct_to_final_loss_20d_rate={self.value(leakage_row, 'peak_5pct_to_final_loss_20d_rate'):.4f}, "
            f"profit_protection_needed_20d_rate={self.value(label_row, 'profit_protection_needed_20d_rate'):.4f}, "
            f"break_even_guard_needed_20d_rate={self.value(label_row, 'break_even_guard_needed_20d_rate'):.4f}, "
            f"stop_loss_too_late_20d_rate={self.value(label_row, 'stop_loss_too_late_20d_rate'):.4f}. "
            f"Recommended labels={', '.join(labels)}. Next phase={next_phase}."
        )

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
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def blocking_issues(self, data: pd.DataFrame, source_info: dict[str, Any]) -> list[str]:
        issues = []
        required = ["future_return_20d", "future_max_return_20d", "future_max_drawdown_20d"]
        missing_required = [column for column in required if column not in data.columns or _numeric(data[column]).dropna().empty]
        if missing_required:
            issues.append(f"missing_required_20d_labels:{','.join(missing_required)}")
        if "candidate_strength" not in data.columns:
            issues.append("missing_candidate_strength_score")
        if data.empty:
            issues.append("empty_2025_dataset")
        if source_info.get("missing_columns"):
            forty_missing = [column for column in source_info["missing_columns"] if "_40d" in column]
            if forty_missing:
                issues.append(f"missing_40d_labels_not_blocking:{','.join(forty_missing)}")
        return [issue for issue in issues if not issue.startswith("missing_40d_labels_not_blocking")]

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_only_for_audit": self.phase13a.expected_future_columns(),
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
            "phase": "13-D",
            "scope": "2025-only hold/exit dataset label audit",
            "primary_metric_context": "Annual Return; this phase audits leakage that may suppress annual return without running a strategy backtest.",
            "new_model_trained": False,
            "strategy_backtest_executed": False,
            "full_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase13DPaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase13DPaths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        final_columns = REQUIRED_REPORT_KEYS + ["primary_candidate_set", "reason"]
        leakage_columns = [
            "future_columns_used_as_features",
            "future_columns_used_only_for_audit",
            "new_model_trained",
            "existing_model_overwritten",
            "profile_changed",
            "strategy_backtest_executed",
            "full_backtest_executed",
            "historical_predictions_regenerated",
            "jquants_api_called",
            "openai_api_called",
            "leakage_risk",
            "blocking_issues",
        ]
        return "\n".join(
            [
                "# Phase 13-D Hold / Exit AI Dataset Audit",
                "",
                "## Input Artifact Summary",
                "",
                self.table([report["input_artifact_summary"]], ["row_count", "date_min", "date_max", "unique_code_count", "available_score_columns", "available_future_columns", "missing_columns", "candidate_sets", "leakage_risk", "blocking_issues"]),
                "",
                "## Label Definitions",
                "",
                self.table([{"label": key, "definition": value} for key, value in report["label_definitions"].items()], ["label", "definition"]),
                "",
                "## Profit Leakage Audit",
                "",
                self.table(report["profit_leakage_audit"], ["candidate_set", "sample_count", "candidate_days", "avg_candidates_per_day", "mean_future_return_20d", "mean_future_max_return_20d", "mean_future_max_drawdown_20d", "top_decile_rate_20d", "downside_bad_rate_20d", "peak_5pct_to_final_loss_20d_count", "peak_5pct_to_final_loss_20d_rate", "peak_10pct_to_final_loss_20d_count", "peak_10pct_to_final_loss_20d_rate", "estimated_peak_to_final_leakage_20d", "estimated_peak_to_final_leakage_20d_mean"]),
                "",
                "## Hold Correctness Label Summary",
                "",
                self.table(report["hold_correctness_label_summary"], ["candidate_set", "sample_count", "hold_correct_20d_rate", "sell_early_risk_20d_rate", "profit_protection_needed_20d_rate", "break_even_guard_needed_20d_rate", "stop_loss_too_late_20d_rate", "high_upside_high_downside_20d_rate", "low_upside_low_downside_20d_rate"]),
                "",
                "## Exit / Hold Failure Mode Summary",
                "",
                self.table(report["exit_hold_failure_mode_summary"], ["candidate_set", "sample_count", "winner_to_loser", "early_sell_risk", "late_stop_loss_risk", "profit_protection_candidate", "break_even_guard_candidate", "hold_candidate", "avoid_candidate"]),
                "",
                "## Final Recommendation",
                "",
                self.table([report["final_recommendation"]], final_columns),
                "",
                "## Leakage Checklist",
                "",
                self.table([report["leakage_checklist"]], leakage_columns),
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

    def mean(self, frame: pd.DataFrame, column: str) -> float | None:
        if column not in frame.columns:
            return None
        values = _numeric(frame[column]).dropna()
        return _safe_float(values.mean()) if not values.empty else None

    def avg_candidates_per_day(self, frame: pd.DataFrame) -> float | None:
        if frame.empty or frame["date"].nunique() == 0:
            return None
        return _safe_float(len(frame) / frame["date"].nunique())

    def peak_to_final_loss(self, frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
        if "future_max_return_20d" not in frame.columns or "future_return_20d" not in frame.columns:
            return {"count": 0, "rate": None}
        mask = (_numeric(frame["future_max_return_20d"]) >= threshold) & (_numeric(frame["future_return_20d"]) < 0)
        return {"count": int(mask.sum()), "rate": _safe_float(mask.mean()) if len(mask) else None}

    def peak_to_final_leakage(self, frame: pd.DataFrame) -> dict[str, Any]:
        if "future_max_return_20d" not in frame.columns or "future_return_20d" not in frame.columns:
            return {"sum": None, "mean": None}
        leakage = (_numeric(frame["future_max_return_20d"]) - _numeric(frame["future_return_20d"])).clip(lower=0)
        leakage = leakage.dropna()
        return {
            "sum": _safe_float(leakage.sum()) if not leakage.empty else None,
            "mean": _safe_float(leakage.mean()) if not leakage.empty else None,
        }

    def mask_count_rate(self, mask: pd.Series) -> dict[str, Any]:
        if mask.empty:
            return {"count": 0, "rate": None}
        return {"count": int(mask.sum()), "rate": _safe_float(mask.mean())}

    def value(self, row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        if isinstance(value, dict):
            value = value.get("rate")
        try:
            if value is None:
                return 0.0
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if math.isnan(result) else result

    def by_name(self, rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        return next((row for row in rows if row.get("candidate_set") == name), None)

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
