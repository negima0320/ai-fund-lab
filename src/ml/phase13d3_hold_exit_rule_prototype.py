"""Phase 13-D3 Hold / Exit Rule Prototype.

This is a 2025-only lightweight rule proxy. It uses future labels only as
evaluation/proxy targets and does not train models, regenerate predictions,
run full backtests, change profiles, overwrite models, or call external APIs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase13a_horizon_reality_audit import ROOT
from ml.phase13d_hold_exit_dataset_audit import CANDIDATE_SETS, Phase13DHoldExitDatasetAudit
from ml.phase13d2_hold_exit_label_refinement import (
    PRIMARY_CANDIDATE_SET,
    Phase13D2HoldExitLabelRefinement,
)


REPORT_STEM = "phase13d3_hold_exit_rule_prototype_2025"
RULES = [
    "baseline_20d",
    "rule_a_profit_protection",
    "rule_b_break_even_guard",
    "rule_c_stop_rework",
    "rule_d_combined_conservative",
    "rule_e_combined_aggressive",
]
EXCLUSIVE_ACTION_PRIORITY = [
    "PROFIT_PROTECT",
    "BREAK_EVEN_GUARD",
    "STOP_REWORK",
    "EXTEND_HOLD",
    "HOLD",
    "AVOID",
    "NO_ACTION",
]
REQUIRED_REPORT_KEYS = [
    "recommended_hold_exit_rule",
    "recommended_rule_reason",
    "hold_exit_rule_prototype_ready",
    "recommended_next_phase",
    "ready_for_phase13e",
    "ready_for_hold_exit_ai_training",
    "leakage_risk",
    "blocking_issues",
]


@dataclass(frozen=True)
class Phase13D3Paths:
    markdown: Path
    json: Path


class Phase13D3HoldExitRulePrototype:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)
        self.phase13d = Phase13DHoldExitDatasetAudit(root)
        self.phase13d2 = Phase13D2HoldExitLabelRefinement(root)

    def run(self) -> Phase13D3Paths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data, source_info = self.phase13d.phase13a.load_comparison_dataset()
        data = self.phase13d2.add_refined_labels(self.phase13d.add_hold_exit_labels(data))
        data = self.phase13d2.assign_exclusive_labels(data)
        leakage = self.leakage_checklist()
        leakage["blocking_issues"] = self.phase13d.blocking_issues(data, source_info)
        leakage["leakage_risk"] = "low" if not leakage["blocking_issues"] else "medium"
        results = self.evaluate_all(data)
        recommendations = self.recommendations(results, leakage)
        return {
            "metadata": self.metadata(),
            "input_artifact_summary": self.input_artifact_summary(data, source_info, leakage),
            "rule_definitions": self.rule_definitions(),
            "exclusive_action_priority": EXCLUSIVE_ACTION_PRIORITY,
            "rule_proxy_note": "This is not a live trading backtest. Future columns are used only for lightweight realized-return proxy evaluation.",
            "rule_proxy_results": results,
            "candidate_strength_top50_summary": self.primary_summary(results),
            "final_recommendation": recommendations,
            "leakage_checklist": leakage,
            **{key: recommendations.get(key) for key in REQUIRED_REPORT_KEYS},
        }

    def evaluate_all(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        candidate_sets = [method for method in CANDIDATE_SETS if method.name != "candidate_universe_baseline"]
        for method in candidate_sets:
            frame = self.phase13d.method_frame(data, method)
            for rule in RULES:
                rows.append(self.evaluate_rule(frame, method.name, rule))
        return rows

    def evaluate_rule(self, frame: pd.DataFrame, candidate_set: str, rule: str) -> dict[str, Any]:
        baseline = _numeric(frame.get("future_return_20d"))
        max_ret = _numeric(frame.get("future_max_return_20d"))
        action = frame.get("exclusive_exit_action", pd.Series("NO_ACTION", index=frame.index)).astype("string")
        realized = self.rule_return_proxy(frame, rule)
        leakage_before = (max_ret - baseline).clip(lower=0)
        leakage_after = (max_ret - realized).clip(lower=0)
        winner_to_loser_before = (max_ret >= 0.05) & (baseline < 0)
        winner_to_loser_after = (max_ret >= 0.05) & (realized < 0)
        trigger = self.rule_trigger_mask(action, rule)
        mean_baseline = self.series_mean(baseline)
        mean_rule = self.series_mean(realized)
        turnover = self.average_turnover_per_year(frame)
        annual_baseline = mean_baseline * turnover if mean_baseline is not None else None
        annual_rule = mean_rule * turnover if mean_rule is not None else None
        return {
            "candidate_set": candidate_set,
            "rule": rule,
            "sample_count": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if not frame.empty else 0,
            "avg_candidates_per_day": self.avg_candidates_per_day(frame),
            "mean_baseline_return_20d": mean_baseline,
            "mean_rule_return_proxy": mean_rule,
            "return_proxy_delta": self.diff(mean_rule, mean_baseline),
            "winner_to_loser_before_rate": self.mask_rate(winner_to_loser_before),
            "winner_to_loser_after_rate": self.mask_rate(winner_to_loser_after),
            "winner_to_loser_reduction": self.diff(self.mask_rate(winner_to_loser_before), self.mask_rate(winner_to_loser_after)),
            "profit_leakage_before_mean": self.series_mean(leakage_before),
            "profit_leakage_after_mean": self.series_mean(leakage_after),
            "profit_leakage_reduction": self.diff(self.series_mean(leakage_before), self.series_mean(leakage_after)),
            "negative_return_before_rate": self.mask_rate(baseline < 0),
            "negative_return_after_rate": self.mask_rate(realized < 0),
            "rule_trigger_count": int(trigger.sum()) if len(trigger) else 0,
            "rule_trigger_rate": self.mask_rate(trigger),
            "annual_return_proxy": _safe_float(annual_rule) if annual_rule is not None else None,
            "annual_return_proxy_delta": self.diff(annual_rule, annual_baseline),
            "average_turnover_per_year_proxy": _safe_float(turnover),
        }

    def rule_return_proxy(self, frame: pd.DataFrame, rule: str) -> pd.Series:
        ret = _numeric(frame.get("future_return_20d")).copy()
        max_ret = _numeric(frame.get("future_max_return_20d"))
        action = frame.get("exclusive_exit_action", pd.Series("NO_ACTION", index=frame.index)).astype("string")
        realized = ret.copy()
        if rule == "baseline_20d":
            return realized
        if rule == "rule_a_profit_protection":
            realized.loc[action.eq("PROFIT_PROTECT")] = realized.loc[action.eq("PROFIT_PROTECT")].clip(lower=0.03)
            return realized
        if rule == "rule_b_break_even_guard":
            realized.loc[action.eq("BREAK_EVEN_GUARD")] = realized.loc[action.eq("BREAK_EVEN_GUARD")].clip(lower=0.00)
            return realized
        if rule == "rule_c_stop_rework":
            realized.loc[action.eq("STOP_REWORK")] = realized.loc[action.eq("STOP_REWORK")].clip(lower=-0.05)
            return realized
        if rule == "rule_d_combined_conservative":
            realized.loc[action.eq("PROFIT_PROTECT")] = realized.loc[action.eq("PROFIT_PROTECT")].clip(lower=0.03)
            realized.loc[action.eq("BREAK_EVEN_GUARD")] = realized.loc[action.eq("BREAK_EVEN_GUARD")].clip(lower=0.00)
            realized.loc[action.eq("STOP_REWORK")] = realized.loc[action.eq("STOP_REWORK")].clip(lower=-0.05)
            return realized
        if rule == "rule_e_combined_aggressive":
            realized.loc[action.eq("PROFIT_PROTECT")] = realized.loc[action.eq("PROFIT_PROTECT")].clip(lower=0.05)
            realized.loc[action.eq("BREAK_EVEN_GUARD")] = realized.loc[action.eq("BREAK_EVEN_GUARD")].clip(lower=0.00)
            realized.loc[action.eq("STOP_REWORK")] = realized.loc[action.eq("STOP_REWORK")].clip(lower=-0.03)
            extend = action.eq("EXTEND_HOLD")
            realized.loc[extend] = max_ret.loc[extend] * 0.6 + ret.loc[extend] * 0.4
            return realized
        raise ValueError(f"Unknown rule: {rule}")

    def rule_trigger_mask(self, action: pd.Series, rule: str) -> pd.Series:
        if rule == "baseline_20d":
            return pd.Series(False, index=action.index)
        if rule == "rule_a_profit_protection":
            return action.eq("PROFIT_PROTECT")
        if rule == "rule_b_break_even_guard":
            return action.eq("BREAK_EVEN_GUARD")
        if rule == "rule_c_stop_rework":
            return action.eq("STOP_REWORK")
        if rule == "rule_d_combined_conservative":
            return action.isin(["PROFIT_PROTECT", "BREAK_EVEN_GUARD", "STOP_REWORK"])
        if rule == "rule_e_combined_aggressive":
            return action.isin(["PROFIT_PROTECT", "BREAK_EVEN_GUARD", "STOP_REWORK", "EXTEND_HOLD"])
        return pd.Series(False, index=action.index)

    def recommendations(self, results: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        primary_rows = [row for row in results if row["candidate_set"] == PRIMARY_CANDIDATE_SET and row["rule"] != "baseline_20d"]
        if leakage["blocking_issues"] or not primary_rows:
            return {
                "recommended_hold_exit_rule": None,
                "recommended_rule_reason": "blocking issues or no primary candidate results",
                "hold_exit_rule_prototype_ready": False,
                "recommended_next_phase": "Phase13-C Horizon-Aware Valuation Prototype",
                "ready_for_phase13e": False,
                "ready_for_hold_exit_ai_training": False,
                "leakage_risk": leakage["leakage_risk"],
                "blocking_issues": leakage["blocking_issues"],
            }
        best = max(primary_rows, key=lambda row: self.score(row))
        improvement = self.value(best, "return_proxy_delta") > 0 and self.value(best, "profit_leakage_reduction") > 0
        next_phase = "Phase13-E Integrated Strategy Prototype" if improvement else "Phase13-D4 Hold/Exit AI Training Dataset Builder"
        return {
            "recommended_hold_exit_rule": best["rule"],
            "recommended_rule_reason": (
                f"Best primary-set proxy score; return_delta={self.value(best, 'return_proxy_delta'):.4f}, "
                f"annual_delta={self.value(best, 'annual_return_proxy_delta'):.4f}, "
                f"leakage_reduction={self.value(best, 'profit_leakage_reduction'):.4f}, "
                f"winner_to_loser_reduction={self.value(best, 'winner_to_loser_reduction'):.4f}."
            ),
            "hold_exit_rule_prototype_ready": True,
            "recommended_next_phase": next_phase,
            "ready_for_phase13e": improvement,
            "ready_for_hold_exit_ai_training": True,
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def score(self, row: dict[str, Any]) -> float:
        return (
            self.value(row, "return_proxy_delta") * 0.45
            + self.value(row, "profit_leakage_reduction") * 0.25
            + self.value(row, "winner_to_loser_reduction") * 0.20
            + self.value(row, "annual_return_proxy_delta") * 0.10
        )

    def primary_summary(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in results if row["candidate_set"] == PRIMARY_CANDIDATE_SET]

    def rule_definitions(self) -> dict[str, str]:
        return {
            "baseline_20d": "realized_return_proxy = future_return_20d",
            "rule_a_profit_protection": "PROFIT_PROTECT -> max(0.03, future_return_20d)",
            "rule_b_break_even_guard": "BREAK_EVEN_GUARD -> max(0.00, future_return_20d)",
            "rule_c_stop_rework": "STOP_REWORK -> max(-0.05, future_return_20d)",
            "rule_d_combined_conservative": "PROFIT_PROTECT -> max(0.03, ret); BREAK_EVEN_GUARD -> max(0.00, ret); STOP_REWORK -> max(-0.05, ret)",
            "rule_e_combined_aggressive": "PROFIT_PROTECT -> max(0.05, ret); BREAK_EVEN_GUARD -> max(0.00, ret); STOP_REWORK -> max(-0.03, ret); EXTEND_HOLD -> future_max_return_20d * 0.6 + future_return_20d * 0.4",
            "annual_return_proxy": "mean_rule_return_proxy * average_turnover_per_year_proxy; proxy only, not a backtest.",
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
            "candidate_sets": [method.name for method in CANDIDATE_SETS if method.name != "candidate_universe_baseline"],
            "primary_candidate_set": PRIMARY_CANDIDATE_SET,
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_only_for_proxy_evaluation": self.phase13d.phase13a.expected_future_columns(),
            "future_columns_used_as_features": [],
            "backtest_columns_used_as_features": [],
            "trade_result_columns_used_as_features": [],
            "cash_or_portfolio_columns_used_as_features": [],
            "new_model_trained": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "strategy_backtest_executed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
            "period": {"start": "2025-01-01", "end": "2025-12-31"},
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "13-D3",
            "scope": "2025-only lightweight hold/exit rule proxy",
            "primary_candidate_set": PRIMARY_CANDIDATE_SET,
            "new_model_trained": False,
            "full_backtest_executed": False,
            "strategy_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase13D3Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase13D3Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Phase 13-D3 Hold / Exit Rule Prototype",
                "",
                "## Input Artifact Summary",
                "",
                self.table([report["input_artifact_summary"]], ["row_count", "date_min", "date_max", "unique_code_count", "candidate_sets", "primary_candidate_set", "leakage_risk", "blocking_issues"]),
                "",
                "## Rule Definitions",
                "",
                self.table([{"rule": key, "definition": value} for key, value in report["rule_definitions"].items()], ["rule", "definition"]),
                "",
                "## Candidate Strength Top50 Summary",
                "",
                self.table(report["candidate_strength_top50_summary"], ["rule", "sample_count", "mean_baseline_return_20d", "mean_rule_return_proxy", "return_proxy_delta", "winner_to_loser_before_rate", "winner_to_loser_after_rate", "winner_to_loser_reduction", "profit_leakage_before_mean", "profit_leakage_after_mean", "profit_leakage_reduction", "negative_return_before_rate", "negative_return_after_rate", "rule_trigger_rate", "annual_return_proxy", "annual_return_proxy_delta"]),
                "",
                "## Rule Proxy Results",
                "",
                self.table(report["rule_proxy_results"], ["candidate_set", "rule", "sample_count", "mean_rule_return_proxy", "return_proxy_delta", "winner_to_loser_after_rate", "profit_leakage_reduction", "rule_trigger_rate", "annual_return_proxy_delta"]),
                "",
                "## Final Recommendation",
                "",
                self.table([report["final_recommendation"]], REQUIRED_REPORT_KEYS),
                "",
                "## Leakage Checklist",
                "",
                self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_proxy_evaluation", "new_model_trained", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "strategy_backtest_executed", "historical_predictions_regenerated", "jquants_api_called", "openai_api_called", "leakage_risk", "blocking_issues"]),
                "",
            ]
        )

    def average_turnover_per_year(self, frame: pd.DataFrame) -> float:
        # Proxy only: a 20d horizon can turn roughly 252/20 times per year.
        if frame.empty:
            return 0.0
        return 252.0 / 20.0

    def avg_candidates_per_day(self, frame: pd.DataFrame) -> float | None:
        if frame.empty or frame["date"].nunique() == 0:
            return None
        return _safe_float(len(frame) / frame["date"].nunique())

    def series_mean(self, series: pd.Series) -> float | None:
        values = _numeric(series).dropna()
        return _safe_float(values.mean()) if not values.empty else None

    def mask_rate(self, mask: pd.Series) -> float | None:
        return _safe_float(mask.mean()) if len(mask) else None

    def diff(self, left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return _safe_float(left - right)

    def value(self, row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        try:
            if value is None:
                return 0.0
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if math.isnan(result) else result

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
