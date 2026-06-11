"""Phase 13-B Candidate Generation Redesign.

This is a 2025-only, read-only audit comparing candidate generation methods
without Stock Selection top5 as a fixed prefilter. It uses existing artifacts
only and does not train models, regenerate predictions, run full backtests,
change profiles, overwrite models, or call external APIs.
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


REPORT_STEM = "phase13b_candidate_generation_redesign_2025"
REQUIRED_REPORT_KEYS = [
    "recommended_candidate_generation_method",
    "recommended_candidate_count",
    "stock_selection_action",
    "valuation_first_ready",
    "candidate_strength_ready",
    "ready_for_phase13c",
    "ready_for_phase13d",
    "leakage_risk",
    "blocking_issues",
]


@dataclass(frozen=True)
class CandidateMethod:
    name: str
    score_column: str | None
    top_n: int | None
    mode: str = "top_n"


@dataclass(frozen=True)
class Phase13BPaths:
    markdown: Path
    json: Path


METHODS = [
    CandidateMethod("candidate_universe_baseline", None, None, "all"),
    CandidateMethod("stock_selection_rank_score_top50", "stock_selection_rank_score", 50),
    CandidateMethod("stock_selection_rank_score_top100", "stock_selection_rank_score", 100),
    CandidateMethod("candidate_strength_top50", "candidate_strength", 50),
    CandidateMethod("candidate_strength_top100", "candidate_strength", 100),
    CandidateMethod("valuation_first_top50", "opportunity_proba", 50),
    CandidateMethod("valuation_first_top100", "opportunity_proba", 100),
    CandidateMethod("opportunity_downside_top50", "opportunity_downside_score", 50),
    CandidateMethod("opportunity_downside_top100", "opportunity_downside_score", 100),
    CandidateMethod("valuation_first_top5", "opportunity_proba", 5),
    CandidateMethod("candidate_strength_top5", "candidate_strength", 5),
    CandidateMethod("opportunity_downside_top5", "opportunity_downside_score", 5),
]


class Phase13BCandidateGenerationRedesign:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)
        self.phase13a = Phase13AHorizonRealityAudit(root)

    def run(self) -> Phase13BPaths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data, source_info = self.phase13a.load_comparison_dataset()
        leakage = self.leakage_checklist()
        method_results = [self.evaluate_method(data, method) for method in METHODS]
        scoring = self.scoring_definition()
        recommendations = self.recommendations(method_results, leakage)
        return {
            "metadata": self.metadata(),
            "input_artifact_summary": self.input_artifact_summary(data, source_info, leakage),
            "scoring_definition": scoring,
            "candidate_generation_results": method_results,
            "comparison_summary": self.comparison_summary(method_results),
            "final_recommendation": recommendations,
            "leakage_checklist": leakage,
            **{key: recommendations.get(key) for key in REQUIRED_REPORT_KEYS},
        }

    def evaluate_method(self, data: pd.DataFrame, method: CandidateMethod) -> dict[str, Any]:
        frame = self.method_frame(data, method)
        row: dict[str, Any] = {
            "method": method.name,
            "score_column": method.score_column,
            "top_n": method.top_n,
            "rows": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if not frame.empty else 0,
            "avg_candidates_per_day": _safe_float(len(frame) / frame["date"].nunique()) if not frame.empty and frame["date"].nunique() else None,
            "mean_future_return_5d": self.mean(frame, "future_return_5d"),
            "top_decile_rate_5d": self.mean(frame, "top_decile_5d"),
            "mean_future_return_10d": self.mean(frame, "future_return_10d"),
            "top_decile_rate_10d": self.mean(frame, "top_decile_10d"),
            "mean_future_return_20d": self.mean(frame, "future_return_20d"),
            "top_decile_rate_20d": self.mean(frame, "top_decile_20d"),
            "future_max_return_20d": self.mean(frame, "future_max_return_20d"),
            "future_max_drawdown_20d": self.mean(frame, "future_max_drawdown_20d"),
            "downside_bad_rate_20d": self.mean(frame, "downside_bad_20d"),
            "peak_5pct_to_final_loss_20d_rate": self.peak_to_loss_rate(frame),
        }
        row.update(self.method_scores(row))
        return row

    def method_frame(self, data: pd.DataFrame, method: CandidateMethod) -> pd.DataFrame:
        if method.mode == "all":
            return data.copy()
        if method.score_column is None or method.top_n is None:
            raise ValueError(f"Invalid method: {method}")
        return self.phase13a.top_n_by_day(data, method.score_column, method.top_n)

    def method_scores(self, row: dict[str, Any]) -> dict[str, Any]:
        top10 = self.value(row, "top_decile_rate_10d")
        top20 = self.value(row, "top_decile_rate_20d")
        downside = self.value(row, "downside_bad_rate_20d")
        peak_loss = self.value(row, "peak_5pct_to_final_loss_20d_rate")
        avg_per_day = self.value(row, "avg_candidates_per_day")
        opportunity_capture = 0.40 * top10 + 0.60 * top20
        downside_control = 1.0 - min(1.0, downside)
        peak_loss_control = 1.0 - min(1.0, peak_loss)
        breadth = min(1.0, avg_per_day / 50.0) if avg_per_day else 0.0
        total = (
            top10 * 0.25
            + top20 * 0.35
            - downside * 0.25
            - peak_loss * 0.15
            + breadth * 0.05
        )
        return {
            "opportunity_capture_score": _safe_float(opportunity_capture),
            "downside_control_score": _safe_float(downside_control),
            "candidate_breadth_score": _safe_float(breadth),
            "peak_loss_control_score": _safe_float(peak_loss_control),
            "phase13b_total_score": _safe_float(total),
        }

    def scoring_definition(self) -> dict[str, Any]:
        return {
            "opportunity_capture_score": "0.40 * top_decile_rate_10d + 0.60 * top_decile_rate_20d",
            "downside_control_score": "1 - downside_bad_rate_20d",
            "candidate_breadth_score": "min(1, avg_candidates_per_day / 50)",
            "peak_loss_control_score": "1 - peak_5pct_to_final_loss_20d_rate",
            "phase13b_total_score": "top_decile_rate_10d * 0.25 + top_decile_rate_20d * 0.35 - downside_bad_rate_20d * 0.25 - peak_5pct_to_final_loss_20d_rate * 0.15 + candidate_breadth_score * 0.05",
            "note": "Future columns are used only for evaluation; decision scores are existing prediction-time columns.",
        }

    def recommendations(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {
                "recommended_candidate_generation_method": None,
                "recommended_candidate_count": None,
                "stock_selection_action": "insufficient_evidence",
                "valuation_first_ready": False,
                "candidate_strength_ready": False,
                "ready_for_phase13c": False,
                "ready_for_phase13d": False,
                "leakage_risk": leakage["leakage_risk"],
                "blocking_issues": leakage["blocking_issues"],
            }
        candidates = [row for row in rows if row["method"] != "candidate_universe_baseline"]
        best = max(candidates, key=lambda row: self.value(row, "phase13b_total_score")) if candidates else {}
        valuation_rows = [row for row in rows if row["method"].startswith("valuation_first")]
        opp_down_rows = [row for row in rows if row["method"].startswith("opportunity_downside")]
        strength_rows = [row for row in rows if row["method"].startswith("candidate_strength")]
        stock_rows = [row for row in rows if row["method"].startswith("stock_selection_rank_score")]
        best_valuation = max(valuation_rows, key=lambda row: self.value(row, "phase13b_total_score")) if valuation_rows else {}
        best_opp_down = max(opp_down_rows, key=lambda row: self.value(row, "phase13b_total_score")) if opp_down_rows else {}
        best_strength = max(strength_rows, key=lambda row: self.value(row, "phase13b_total_score")) if strength_rows else {}
        best_stock = max(stock_rows, key=lambda row: self.value(row, "phase13b_total_score")) if stock_rows else {}
        chosen = best_opp_down if self.value(best_opp_down, "phase13b_total_score") >= self.value(best_valuation, "phase13b_total_score") else best_valuation
        if self.value(best_strength, "phase13b_total_score") > self.value(chosen, "phase13b_total_score"):
            chosen = best_strength
        stock_action = self.stock_selection_action(best_stock, best_strength, best_valuation, best_opp_down)
        return {
            "recommended_candidate_generation_method": chosen.get("method") or best.get("method"),
            "recommended_candidate_count": chosen.get("top_n") or best.get("top_n"),
            "stock_selection_action": stock_action,
            "valuation_first_ready": bool(best_valuation) and self.value(best_valuation, "top_decile_rate_20d") > 0.12,
            "candidate_strength_ready": bool(best_strength) and self.value(best_strength, "top_decile_rate_20d") > 0.12,
            "ready_for_phase13c": bool(chosen) and leakage["leakage_risk"] == "low",
            "ready_for_phase13d": True,
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
            "phase13b_recommendation": "Use the recommended candidate set for Phase13-C horizon-aware valuation prototype, and start Phase13-D exit/hold label audit in parallel.",
            "reason": self.recommendation_reason(chosen, best_stock, best_strength, best_valuation, best_opp_down, stock_action),
        }

    def stock_selection_action(
        self,
        best_stock: dict[str, Any],
        best_strength: dict[str, Any],
        best_valuation: dict[str, Any],
        best_opp_down: dict[str, Any],
    ) -> str:
        best_stock_score = self.value(best_stock, "phase13b_total_score")
        best_strength_score = self.value(best_strength, "phase13b_total_score")
        best_valuation_score = self.value(best_valuation, "phase13b_total_score")
        best_opp_down_score = self.value(best_opp_down, "phase13b_total_score")
        best_non_stock = max(best_strength_score, best_valuation_score, best_opp_down_score)
        if best_opp_down_score >= best_valuation_score and best_opp_down_score >= best_strength_score and best_opp_down_score > best_stock_score:
            return "valuation_first"
        if best_valuation_score >= best_strength_score and best_valuation_score > best_stock_score:
            return "valuation_first"
        if best_strength_score > best_stock_score and best_strength_score >= best_non_stock:
            return "replace_with_candidate_strength"
        if best_stock.get("top_n") == 50 and best_stock_score >= best_non_stock:
            return "widen_to_top50"
        if best_stock.get("top_n") == 100 and best_stock_score >= best_non_stock:
            return "widen_to_top100"
        if best_stock_score < best_non_stock:
            return "remove_prefilter"
        return "insufficient_evidence"

    def recommendation_reason(
        self,
        chosen: dict[str, Any],
        best_stock: dict[str, Any],
        best_strength: dict[str, Any],
        best_valuation: dict[str, Any],
        best_opp_down: dict[str, Any],
        action: str,
    ) -> str:
        return (
            f"recommended={chosen.get('method')} score={self.value(chosen, 'phase13b_total_score'):.4f}; "
            f"best_stock={best_stock.get('method')} score={self.value(best_stock, 'phase13b_total_score'):.4f}; "
            f"best_candidate_strength={best_strength.get('method')} score={self.value(best_strength, 'phase13b_total_score'):.4f}; "
            f"best_valuation={best_valuation.get('method')} score={self.value(best_valuation, 'phase13b_total_score'):.4f}; "
            f"best_opportunity_downside={best_opp_down.get('method')} score={self.value(best_opp_down, 'phase13b_total_score'):.4f}; "
            f"stock_selection_action={action}."
        )

    def comparison_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_method = {row["method"]: row for row in rows}
        ordered = sorted(rows, key=lambda row: self.value(row, "phase13b_total_score"), reverse=True)
        return {
            "top_methods_by_total_score": [row["method"] for row in ordered[:5]],
            "best_by_family": {
                "stock_selection": self.best_name(rows, "stock_selection_rank_score"),
                "candidate_strength": self.best_name(rows, "candidate_strength"),
                "valuation_first": self.best_name(rows, "valuation_first"),
                "opportunity_downside": self.best_name(rows, "opportunity_downside"),
            },
            "baseline": by_method.get("candidate_universe_baseline", {}),
        }

    def best_name(self, rows: list[dict[str, Any]], prefix: str) -> str | None:
        candidates = [row for row in rows if row["method"].startswith(prefix)]
        if not candidates:
            return None
        return max(candidates, key=lambda row: self.value(row, "phase13b_total_score"))["method"]

    def peak_to_loss_rate(self, frame: pd.DataFrame) -> float | None:
        if "future_max_return_20d" not in frame.columns or "future_return_20d" not in frame.columns:
            return None
        mask = (_numeric(frame["future_max_return_20d"]) >= 0.05) & (_numeric(frame["future_return_20d"]) < 0)
        return _safe_float(mask.mean()) if len(mask) else None

    def mean(self, frame: pd.DataFrame, column: str) -> float | None:
        if column not in frame.columns:
            return None
        values = _numeric(frame[column]).dropna()
        return _safe_float(values.mean()) if not values.empty else None

    def value(self, row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        try:
            if value is None:
                return 0.0
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if math.isnan(result) else result

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
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_as_features": [],
            "future_columns_used_only_for_evaluation": self.phase13a.expected_future_columns(),
            "backtest_columns_used_as_features": [],
            "trade_result_columns_used_as_features": [],
            "cash_or_portfolio_columns_used_as_features": [],
            "selected_or_bought_used_as_features": False,
            "current_pm_multiplier_used": False,
            "new_model_trained": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
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
            "phase": "13-B",
            "scope": "2025-only candidate generation redesign audit",
            "new_model_trained": False,
            "full_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase13BPaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase13BPaths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 13-B Candidate Generation Redesign",
            "",
            "## Input Artifact Summary",
            "",
            self.table([report["input_artifact_summary"]], ["row_count", "date_min", "date_max", "unique_code_count", "available_score_columns", "available_future_columns", "missing_columns", "leakage_risk", "blocking_issues"]),
            "",
            "## Scoring Definition",
            "",
            self.table([report["scoring_definition"]], ["opportunity_capture_score", "downside_control_score", "candidate_breadth_score", "peak_loss_control_score", "phase13b_total_score", "note"]),
            "",
            "## Candidate Generation Results",
            "",
            self.table(report["candidate_generation_results"], ["method", "rows", "candidate_days", "avg_candidates_per_day", "mean_future_return_5d", "top_decile_rate_5d", "mean_future_return_10d", "top_decile_rate_10d", "mean_future_return_20d", "top_decile_rate_20d", "future_max_return_20d", "future_max_drawdown_20d", "downside_bad_rate_20d", "peak_5pct_to_final_loss_20d_rate", "opportunity_capture_score", "downside_control_score", "candidate_breadth_score", "phase13b_total_score"]),
            "",
            "## Final Recommendation",
            "",
            self.table([report["final_recommendation"]], REQUIRED_REPORT_KEYS + ["phase13b_recommendation", "reason"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_evaluation", "new_model_trained", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "historical_predictions_regenerated", "jquants_api_called", "openai_api_called", "leakage_risk", "blocking_issues"]),
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
