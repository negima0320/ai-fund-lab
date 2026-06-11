"""Phase 12-A3 top5 downside penalty refinement.

This audit fixes the candidate universe to daily Opportunity top5 and only
changes the Downside penalty. It reads the Phase 12-A scored artifact and does
not run a strategy backtest, change profiles, overwrite models, or regenerate
historical predictions.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11b3_expected_downside_model import DOWNSIDE_TARGET
from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH, EVAL_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase12a3_top5_penalty_refinement_2025"
MAX_POSITIONS_PROXY = 5
MIN_TOP_DECILE_RATE = 0.20
MAX_DOWNSIDE_BAD_RATE = 0.25
IDEAL_TOP_DECILE_RATE = 0.24
IDEAL_DOWNSIDE_BAD_RATE = 0.20
MIN_AVERAGE_WEIGHT_WARNING = 0.10


@dataclass(frozen=True)
class Phase12A3Options:
    max_positions_proxy: int = MAX_POSITIONS_PROXY


@dataclass(frozen=True)
class Phase12A3Paths:
    markdown: Path
    json: Path


class Phase12A3Top5PenaltyRefinement:
    def __init__(self, root: Path | str = ROOT, *, options: Phase12A3Options | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase12A3Options()

    def run(self) -> Phase12A3Paths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        scored = self.load_scored_universe()
        leakage = self.leakage_checklist()
        if leakage["blocking_issues"]:
            return {
                "metadata": self.metadata(),
                "dataset_summary": self.dataset_summary(scored),
                "penalty_rules": self.penalty_rule_definitions(),
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], leakage),
            }
        top5 = self.opportunity_top5(scored)
        rows = self.evaluate_rules(top5)
        summary = self.summary(rows)
        return {
            "metadata": self.metadata(),
            "dataset_summary": self.dataset_summary(scored, top5),
            "candidate_universe": {
                "name": "opportunity_top5",
                "definition": "Daily top 5 by opportunity_proba. Candidate universe is fixed in Phase12-A3.",
            },
            "penalty_rules": self.penalty_rule_definitions(),
            "rule_quality": rows,
            "summary": summary,
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(rows, leakage),
        }

    def load_scored_universe(self) -> pd.DataFrame:
        path = self.root / ARTIFACT_PATH
        data = pd.read_parquet(path)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        scored_columns = [
            "date",
            "code",
            "close",
            "turnover_value",
            "opportunity_proba",
            "downside_bad_proba",
            "opportunity_rank_percentile",
            "downside_rank_percentile",
            "confidence",
            *EVAL_COLUMNS,
        ]
        available = [column for column in scored_columns if column in data.columns]
        required = ["date", "code", "opportunity_proba", "downside_bad_proba", "downside_rank_percentile"]
        return (
            data[available]
            .drop_duplicates(["date", "code"])
            .dropna(subset=[column for column in required if column in available])
            .reset_index(drop=True)
        )

    def opportunity_top5(self, scored: pd.DataFrame) -> pd.DataFrame:
        return (
            scored.sort_values(["date", "opportunity_proba", "turnover_value", "code"], ascending=[True, False, False, True])
            .groupby("date", group_keys=False)
            .head(self.options.max_positions_proxy)
            .copy()
            .reset_index(drop=True)
        )

    def evaluate_rules(self, top5: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for rule_name in [
            "A2_baseline_penalty_rank_medium",
            "A3_1_rank_medium_plus",
            "A3_2_rank_medium_stronger_tail",
            "A3_3_rank_medium_floor_zero",
            "A3_4_hybrid_rank_and_proba",
            "A3_5_hybrid_rank_and_proba_strict",
        ]:
            allocated = top5.copy()
            allocated["allocation_rule"] = rule_name
            allocated["candidate_universe"] = "opportunity_top5"
            allocated["penalty_rule"] = rule_name
            allocated["allocation_weight"] = self.penalty_weight(allocated, rule_name)
            allocated["allocation_bucket"] = allocated["allocation_weight"].map(lambda value: f"w{value:.3f}")
            rows.append(self.rule_quality(allocated))
        return rows

    def penalty_weight(self, frame: pd.DataFrame, rule: str) -> pd.Series:
        rank = _numeric(frame["downside_rank_percentile"])
        proba = _numeric(frame["downside_bad_proba"])
        if rule == "A2_baseline_penalty_rank_medium":
            return self.rank_medium_weight(rank)
        if rule == "A3_1_rank_medium_plus":
            return rank.map(lambda value: 1.0 if value <= 0.35 else 0.55 if value <= 0.65 else 0.25 if value <= 0.80 else 0.05)
        if rule == "A3_2_rank_medium_stronger_tail":
            return rank.map(lambda value: 1.0 if value <= 0.40 else 0.6 if value <= 0.70 else 0.25 if value <= 0.85 else 0.05)
        if rule == "A3_3_rank_medium_floor_zero":
            return rank.map(lambda value: 1.0 if value <= 0.40 else 0.6 if value <= 0.70 else 0.3 if value <= 0.85 else 0.0)
        if rule == "A3_4_hybrid_rank_and_proba":
            weight = self.rank_medium_weight(rank)
            weight = weight.where(proba < 0.45, weight * 0.5)
            weight = weight.where(proba < 0.60, weight * 0.25)
            return weight
        if rule == "A3_5_hybrid_rank_and_proba_strict":
            weight = self.rank_medium_weight(rank)
            weight = weight.where(proba < 0.40, weight * 0.5)
            weight = weight.where(proba < 0.55, weight * 0.2)
            return weight
        raise ValueError(f"Unknown penalty rule: {rule}")

    def rank_medium_weight(self, rank: pd.Series) -> pd.Series:
        return rank.map(lambda value: 1.0 if value <= 0.40 else 0.6 if value <= 0.70 else 0.3 if value <= 0.85 else 0.1)

    def rule_quality(self, frame: pd.DataFrame) -> dict[str, Any]:
        daily_weight = frame.groupby("date")["allocation_weight"].sum()
        average_candidates = len(frame) / frame["date"].nunique() if frame["date"].nunique() else 0.0
        row = {
            "allocation_rule": str(frame["allocation_rule"].iloc[0]) if not frame.empty else None,
            "candidate_universe": "opportunity_top5",
            "penalty_rule": str(frame["penalty_rule"].iloc[0]) if not frame.empty else None,
            "allocated_rows": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if "date" in frame.columns else 0,
            "average_allocated_candidates_per_day": _safe_float(average_candidates),
            "average_weight": self.mean(frame, "allocation_weight"),
            "budget_usage_proxy": _safe_float((daily_weight.clip(upper=self.options.max_positions_proxy) / self.options.max_positions_proxy).mean()) if not daily_weight.empty else None,
            "weight_distribution": dict(sorted(Counter(frame["allocation_bucket"]).items())),
            **self.weighted_quality(frame),
            **self.unweighted_quality(frame),
        }
        row["minimum_target_passed"] = bool((row.get("weighted_opportunity_top_decile_rate") or 0.0) >= MIN_TOP_DECILE_RATE and (row.get("weighted_downside_bad_rate") or 1.0) <= MAX_DOWNSIDE_BAD_RATE)
        row["ideal_target_passed"] = bool((row.get("weighted_opportunity_top_decile_rate") or 0.0) >= IDEAL_TOP_DECILE_RATE and (row.get("weighted_downside_bad_rate") or 1.0) <= IDEAL_DOWNSIDE_BAD_RATE)
        row["average_weight_warning"] = bool((row.get("average_weight") or 0.0) < MIN_AVERAGE_WEIGHT_WARNING)
        return row

    def weighted_quality(self, frame: pd.DataFrame) -> dict[str, Any]:
        weight = _numeric(frame["allocation_weight"])
        return {
            "weighted_future_return_20d": self.weighted_mean(frame, "future_return_20d", weight),
            "weighted_future_max_return_20d": self.weighted_mean(frame, "future_max_return_20d", weight),
            "weighted_future_max_drawdown_20d": self.weighted_mean(frame, "future_max_drawdown_20d", weight),
            "weighted_opportunity_value_20d": self.weighted_mean(frame, "opportunity_value_20d", weight),
            "weighted_opportunity_top_decile_rate": self.weighted_mean(frame, "opportunity_top_decile_20d", weight),
            "weighted_downside_bad_rate": self.weighted_mean(frame, DOWNSIDE_TARGET, weight),
        }

    def unweighted_quality(self, frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "future_return_20d_mean": self.mean(frame, "future_return_20d"),
            "future_max_return_20d_mean": self.mean(frame, "future_max_return_20d"),
            "future_max_drawdown_20d_mean": self.mean(frame, "future_max_drawdown_20d"),
            "opportunity_value_20d_mean": self.mean(frame, "opportunity_value_20d"),
            "opportunity_top_decile_20d_rate": self.mean(frame, "opportunity_top_decile_20d"),
            "downside_bad_rate": self.mean(frame, DOWNSIDE_TARGET),
            "avg_opportunity_proba": self.mean(frame, "opportunity_proba"),
            "avg_downside_bad_proba": self.mean(frame, "downside_bad_proba"),
            "avg_confidence": self.mean(frame, "confidence"),
        }

    def weighted_mean(self, frame: pd.DataFrame, column: str, weight: pd.Series) -> float | None:
        if column not in frame.columns or frame.empty:
            return None
        values = _numeric(frame[column])
        valid = pd.DataFrame({"value": values, "weight": weight}).dropna()
        total_weight = float(valid["weight"].sum()) if not valid.empty else 0.0
        if total_weight <= 0:
            return None
        return _safe_float((valid["value"] * valid["weight"]).sum() / total_weight)

    def mean(self, frame: pd.DataFrame, column: str) -> float | None:
        values = _numeric(frame[column]) if column in frame.columns else pd.Series(dtype=float)
        return _safe_float(values.mean()) if not values.empty else None

    def summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        minimum = [row for row in rows if row["minimum_target_passed"]]
        ideal = [row for row in rows if row["ideal_target_passed"]]
        best = self.best_rule(rows)
        warnings = [row["allocation_rule"] for row in rows if row.get("average_weight_warning")]
        return {
            "best_allocation_rule": best.get("allocation_rule") if best else None,
            "best_rule_reason": self.best_rule_reason(best),
            "rules_meeting_minimum_target": [row["allocation_rule"] for row in minimum],
            "rules_meeting_ideal_target": [row["allocation_rule"] for row in ideal],
            "average_weight_warning": warnings,
            "opportunity_downside_tradeoff_summary": self.tradeoff_summary(rows),
            "ready_for_phase12b": bool(minimum),
            "recommended_next_phase": "Phase12-B limited allocation strategy check" if minimum else "Phase12-A4 allocation penalty refinement",
        }

    def best_rule(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None

        def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
            top = row.get("weighted_opportunity_top_decile_rate") or 0.0
            downside = row.get("weighted_downside_bad_rate") or 1.0
            opp_value = row.get("weighted_opportunity_value_20d") or -10**9
            avg_weight = row.get("average_weight") or 0.0
            pass_bonus = 10.0 if row.get("minimum_target_passed") else 0.0
            target_penalty = max(0.0, MIN_TOP_DECILE_RATE - top) + max(0.0, downside - MAX_DOWNSIDE_BAD_RATE)
            weight_penalty = max(0.0, MIN_AVERAGE_WEIGHT_WARNING - avg_weight)
            return (pass_bonus - target_penalty - weight_penalty, opp_value, top - downside, avg_weight)

        return max(rows, key=sort_key)

    def best_rule_reason(self, row: dict[str, Any] | None) -> str:
        if not row:
            return "No rules were evaluated."
        return (
            f"{row['allocation_rule']} has weighted top-decile rate "
            f"{row.get('weighted_opportunity_top_decile_rate'):.4f}, weighted downside bad rate "
            f"{row.get('weighted_downside_bad_rate'):.4f}, weighted opportunity value "
            f"{row.get('weighted_opportunity_value_20d'):.4f}, and average weight "
            f"{row.get('average_weight'):.4f}."
        )

    def tradeoff_summary(self, rows: list[dict[str, Any]]) -> str:
        compact = [
            f"{row['allocation_rule']}: top={row.get('weighted_opportunity_top_decile_rate'):.4f}, downside={row.get('weighted_downside_bad_rate'):.4f}, avg_w={row.get('average_weight'):.4f}"
            for row in rows
        ]
        return "; ".join(compact)

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_as_features": [],
            "future_columns_used_only_for_evaluation": EVAL_COLUMNS,
            "backtest_columns_used_as_features": [],
            "trade_result_columns_used_as_features": [],
            "cash_or_portfolio_columns_used_as_model_features": [],
            "selected_or_bought_used_as_features": False,
            "current_pm_multiplier_used": False,
            "historical_predictions_regenerated": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "strategy_backtest_executed": False,
            "full_backtest_executed": False,
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def recommendation(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"ready_for_phase12b": False, "recommended_next_phase": "Fix leakage blockers"}
        summary = self.summary(rows)
        return {
            "ready_for_phase12b": summary["ready_for_phase12b"],
            "recommended_next_phase": summary["recommended_next_phase"],
            "reason": "Proceed only if a top5 penalty rule reaches weighted top-decile >= 0.20 and weighted downside <= 0.25 without average weight collapsing below 0.10.",
        }

    def dataset_summary(self, scored: pd.DataFrame, top5: pd.DataFrame | None = None) -> dict[str, Any]:
        summary = {
            "rows": int(len(scored)),
            "candidate_days": int(scored["date"].nunique()) if not scored.empty else 0,
            "date_range": {
                "min": scored["date"].min().date().isoformat() if not scored.empty else None,
                "max": scored["date"].max().date().isoformat() if not scored.empty else None,
            },
            "source_artifact": str(self.root / ARTIFACT_PATH),
        }
        if top5 is not None:
            summary.update(
                {
                    "top5_rows": int(len(top5)),
                    "top5_candidate_days": int(top5["date"].nunique()) if not top5.empty else 0,
                    "top5_average_candidates_per_day": _safe_float(len(top5) / top5["date"].nunique()) if not top5.empty and top5["date"].nunique() else None,
                }
            )
        return summary

    def penalty_rule_definitions(self) -> dict[str, str]:
        return {
            "A2_baseline_penalty_rank_medium": "downside_rank <= .40:1.0, <=.70:.6, <=.85:.3, else:.1",
            "A3_1_rank_medium_plus": "downside_rank <= .35:1.0, <=.65:.55, <=.80:.25, else:.05",
            "A3_2_rank_medium_stronger_tail": "downside_rank <= .40:1.0, <=.70:.6, <=.85:.25, else:.05",
            "A3_3_rank_medium_floor_zero": "downside_rank <= .40:1.0, <=.70:.6, <=.85:.3, else:0.0",
            "A3_4_hybrid_rank_and_proba": "rank_medium base; downside_proba >= .45 multiplies by .5; >= .60 then multiplies by .25",
            "A3_5_hybrid_rank_and_proba_strict": "rank_medium base; downside_proba >= .40 multiplies by .5; >= .55 then multiplies by .2",
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "12-A3",
            "scope": "2025 top5 penalty refinement only",
            "candidate_universe_fixed": "opportunity_top5",
            "strategy_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase12A3Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12A3Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-A3 Top5 Penalty Refinement",
            "",
            "## Summary",
            "",
            self.table(
                [report["summary"]],
                [
                    "best_allocation_rule",
                    "best_rule_reason",
                    "rules_meeting_minimum_target",
                    "rules_meeting_ideal_target",
                    "average_weight_warning",
                    "ready_for_phase12b",
                    "recommended_next_phase",
                ],
            ),
            "",
            "## Rule Quality",
            "",
            self.table(
                report.get("rule_quality", []),
                [
                    "allocation_rule",
                    "allocated_rows",
                    "candidate_days",
                    "average_allocated_candidates_per_day",
                    "average_weight",
                    "budget_usage_proxy",
                    "weighted_future_return_20d",
                    "weighted_future_max_return_20d",
                    "weighted_future_max_drawdown_20d",
                    "weighted_opportunity_value_20d",
                    "weighted_opportunity_top_decile_rate",
                    "weighted_downside_bad_rate",
                    "future_return_20d_mean",
                    "future_max_return_20d_mean",
                    "future_max_drawdown_20d_mean",
                    "opportunity_value_20d_mean",
                    "opportunity_top_decile_20d_rate",
                    "downside_bad_rate",
                    "minimum_target_passed",
                    "ideal_target_passed",
                    "average_weight_warning",
                ],
            ),
            "",
            "## Leakage Checklist",
            "",
            self.table(
                [report["leakage_checklist"]],
                [
                    "future_columns_used_as_features",
                    "future_columns_used_only_for_evaluation",
                    "backtest_columns_used_as_features",
                    "trade_result_columns_used_as_features",
                    "cash_or_portfolio_columns_used_as_model_features",
                    "selected_or_bought_used_as_features",
                    "current_pm_multiplier_used",
                    "historical_predictions_regenerated",
                    "existing_model_overwritten",
                    "profile_changed",
                    "strategy_backtest_executed",
                    "full_backtest_executed",
                    "leakage_risk",
                    "blocking_issues",
                ],
            ),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["ready_for_phase12b", "recommended_next_phase", "reason"]),
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
