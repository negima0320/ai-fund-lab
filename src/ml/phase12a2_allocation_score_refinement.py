"""Phase 12-A2 allocation score refinement.

This audit narrows the candidate universe to Opportunity-led sets and uses
Downside only as a position-size penalty. It reads the Phase 12-A artifact and
does not run a strategy backtest, change profiles, overwrite models, or
regenerate historical predictions.
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
REPORT_STEM = "phase12a2_allocation_score_refinement_2025"
MAX_POSITIONS_PROXY = 5


@dataclass(frozen=True)
class Phase12A2Options:
    max_positions_proxy: int = MAX_POSITIONS_PROXY


@dataclass(frozen=True)
class Phase12A2Paths:
    markdown: Path
    json: Path


class Phase12A2AllocationScoreRefinement:
    def __init__(self, root: Path | str = ROOT, *, options: Phase12A2Options | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase12A2Options()

    def run(self) -> Phase12A2Paths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        scored = self.load_scored_universe()
        leakage = self.leakage_checklist()
        if leakage["blocking_issues"]:
            return {
                "metadata": self.metadata(),
                "dataset_summary": self.dataset_summary(scored),
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], leakage),
            }
        rows = self.evaluate_rules(scored)
        summary = self.summary(rows)
        return {
            "metadata": self.metadata(),
            "dataset_summary": self.dataset_summary(scored),
            "candidate_universes": self.candidate_universe_definitions(),
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
        return data[available].drop_duplicates(["date", "code"]).dropna(subset=["date", "code", "opportunity_proba", "downside_bad_proba"]).reset_index(drop=True)

    def evaluate_rules(self, scored: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for universe_name in ["opportunity_top5", "opportunity_top10", "opportunity_top20", "opportunity_p95", "opportunity_p90"]:
            universe = self.candidate_universe(scored, universe_name)
            for penalty_name in ["penalty_none", "penalty_soft", "penalty_medium", "penalty_rank_soft", "penalty_rank_medium"]:
                allocated = universe.copy()
                allocated["candidate_universe"] = universe_name
                allocated["penalty_rule"] = penalty_name
                allocated["allocation_rule"] = f"{universe_name}__{penalty_name}"
                allocated["allocation_weight"] = self.penalty_weight(allocated, penalty_name)
                allocated["allocation_bucket"] = allocated["allocation_weight"].map(lambda value: f"w{value:.1f}")
                rows.append(self.rule_quality(allocated))
        return rows

    def candidate_universe(self, scored: pd.DataFrame, name: str) -> pd.DataFrame:
        if name.startswith("opportunity_top"):
            n = int(name.replace("opportunity_top", ""))
            return (
                scored.sort_values(["date", "opportunity_proba", "turnover_value", "code"], ascending=[True, False, False, True])
                .groupby("date", group_keys=False)
                .head(n)
                .copy()
            )
        if name == "opportunity_p95":
            return scored[_numeric(scored["opportunity_rank_percentile"]) >= 0.95].copy()
        if name == "opportunity_p90":
            return scored[_numeric(scored["opportunity_rank_percentile"]) >= 0.90].copy()
        raise ValueError(f"Unknown universe: {name}")

    def penalty_weight(self, frame: pd.DataFrame, rule: str) -> pd.Series:
        if rule == "penalty_none":
            return pd.Series(1.0, index=frame.index)
        downside = _numeric(frame["downside_bad_proba"])
        rank = _numeric(frame["downside_rank_percentile"])
        if rule == "penalty_soft":
            return downside.map(lambda value: 1.0 if value < 0.20 else 0.7 if value < 0.35 else 0.4 if value < 0.50 else 0.2)
        if rule == "penalty_medium":
            return downside.map(lambda value: 1.0 if value < 0.15 else 0.6 if value < 0.30 else 0.3 if value < 0.45 else 0.1)
        if rule == "penalty_rank_soft":
            return rank.map(lambda value: 1.0 if value <= 0.50 else 0.7 if value <= 0.75 else 0.4 if value <= 0.90 else 0.2)
        if rule == "penalty_rank_medium":
            return rank.map(lambda value: 1.0 if value <= 0.40 else 0.6 if value <= 0.70 else 0.3 if value <= 0.85 else 0.1)
        raise ValueError(f"Unknown penalty rule: {rule}")

    def rule_quality(self, frame: pd.DataFrame) -> dict[str, Any]:
        daily_weight = frame.groupby("date")["allocation_weight"].sum()
        average_candidates = len(frame) / frame["date"].nunique() if frame["date"].nunique() else 0.0
        row = {
            "allocation_rule": str(frame["allocation_rule"].iloc[0]) if not frame.empty else None,
            "candidate_universe": str(frame["candidate_universe"].iloc[0]) if not frame.empty else None,
            "penalty_rule": str(frame["penalty_rule"].iloc[0]) if not frame.empty else None,
            "allocated_rows": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if "date" in frame.columns else 0,
            "average_allocated_candidates_per_day": _safe_float(average_candidates),
            "average_weight": self.mean(frame, "allocation_weight"),
            "budget_usage_proxy": _safe_float((daily_weight.clip(upper=self.options.max_positions_proxy) / self.options.max_positions_proxy).mean()) if not daily_weight.empty else None,
            "weight_distribution": dict(sorted(Counter(frame["allocation_bucket"]).items())),
            "overbroad_candidate_universe": bool(average_candidates > 20.0),
            **self.weighted_quality(frame),
            **self.unweighted_quality(frame),
        }
        row["minimum_target_passed"] = bool((row.get("weighted_opportunity_top_decile_rate") or 0.0) >= 0.20 and (row.get("weighted_downside_bad_rate") or 1.0) <= 0.25)
        row["ideal_target_passed"] = bool((row.get("weighted_opportunity_top_decile_rate") or 0.0) >= 0.24 and (row.get("weighted_downside_bad_rate") or 1.0) <= 0.20)
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
        return {
            "best_allocation_rule": best.get("allocation_rule") if best else None,
            "best_candidate_universe": best.get("candidate_universe") if best else None,
            "best_penalty_rule": best.get("penalty_rule") if best else None,
            "best_rule_reason": self.best_rule_reason(best),
            "opportunity_downside_tradeoff_summary": self.tradeoff_summary(rows),
            "rules_meeting_minimum_target": [row["allocation_rule"] for row in minimum],
            "rules_meeting_ideal_target": [row["allocation_rule"] for row in ideal],
            "ready_for_phase12b": bool(minimum),
            "recommended_next_phase": "Phase12-B limited allocation strategy check" if minimum else "Phase12-A3 allocation refinement",
        }

    def best_rule(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None

        def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
            top = row.get("weighted_opportunity_top_decile_rate") or 0.0
            downside = row.get("weighted_downside_bad_rate") or 1.0
            opp_value = row.get("weighted_opportunity_value_20d") or -10**9
            avg_candidates = row.get("average_allocated_candidates_per_day") or 10**9
            pass_bonus = 10.0 if row.get("minimum_target_passed") else 0.0
            broad_penalty = 2.0 if avg_candidates > 20.0 else 0.0
            target_penalty = max(0.0, 0.20 - top) + max(0.0, downside - 0.25)
            return (pass_bonus - target_penalty - broad_penalty, opp_value, top - downside, -avg_candidates)

        return max(rows, key=sort_key)

    def best_rule_reason(self, row: dict[str, Any] | None) -> str:
        if not row:
            return "No rules were evaluated."
        caution = " It is overbroad and should be treated cautiously." if row.get("overbroad_candidate_universe") else ""
        return (
            f"{row['allocation_rule']} has weighted top-decile rate "
            f"{row.get('weighted_opportunity_top_decile_rate'):.4f}, weighted downside bad rate "
            f"{row.get('weighted_downside_bad_rate'):.4f}, weighted opportunity value "
            f"{row.get('weighted_opportunity_value_20d'):.4f}, and average candidates/day "
            f"{row.get('average_allocated_candidates_per_day'):.2f}.{caution}"
        )

    def tradeoff_summary(self, rows: list[dict[str, Any]]) -> str:
        compact = [
            f"{row['allocation_rule']}: top={row.get('weighted_opportunity_top_decile_rate'):.4f}, downside={row.get('weighted_downside_bad_rate'):.4f}, avg_n={row.get('average_allocated_candidates_per_day'):.1f}"
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
            "reason": "Proceed only if a rule reaches weighted top-decile >= 0.20 and weighted downside <= 0.25 without an overbroad universe.",
        }

    def dataset_summary(self, scored: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(scored)),
            "candidate_days": int(scored["date"].nunique()) if not scored.empty else 0,
            "date_range": {
                "min": scored["date"].min().date().isoformat() if not scored.empty else None,
                "max": scored["date"].max().date().isoformat() if not scored.empty else None,
            },
            "source_artifact": str(self.root / ARTIFACT_PATH),
        }

    def candidate_universe_definitions(self) -> dict[str, str]:
        return {
            "opportunity_top5": "Daily top 5 by opportunity_proba",
            "opportunity_top10": "Daily top 10 by opportunity_proba",
            "opportunity_top20": "Daily top 20 by opportunity_proba",
            "opportunity_p95": "Daily opportunity_rank_percentile >= 0.95",
            "opportunity_p90": "Daily opportunity_rank_percentile >= 0.90",
        }

    def penalty_rule_definitions(self) -> dict[str, str]:
        return {
            "penalty_none": "weight = 1.0",
            "penalty_soft": "downside_proba < .20:1.0, <.35:.7, <.50:.4, else:.2",
            "penalty_medium": "downside_proba < .15:1.0, <.30:.6, <.45:.3, else:.1",
            "penalty_rank_soft": "downside_rank <= .50:1.0, <=.75:.7, <=.90:.4, else:.2",
            "penalty_rank_medium": "downside_rank <= .40:1.0, <=.70:.6, <=.85:.3, else:.1",
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "12-A2",
            "scope": "2025 allocation score refinement only",
            "strategy_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase12A2Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12A2Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-A2 Allocation Score Refinement",
            "",
            "## Summary",
            "",
            self.table([report["summary"]], ["best_allocation_rule", "best_candidate_universe", "best_penalty_rule", "best_rule_reason", "ready_for_phase12b", "recommended_next_phase"]),
            "",
            "## Rule Quality",
            "",
            self.table(
                report.get("rule_quality", []),
                [
                    "allocation_rule",
                    "candidate_universe",
                    "penalty_rule",
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
                    "minimum_target_passed",
                    "ideal_target_passed",
                    "overbroad_candidate_universe",
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
