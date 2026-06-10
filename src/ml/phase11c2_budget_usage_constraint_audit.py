"""Phase 11-C2 budget usage constraint audit.

This audit reads the Phase 11-C allocation simulation and Phase 11-A dataset
for 2025 only. It does not run strategy backtests, change profiles, regenerate
predictions, or use future labels for allocation decisions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SIMULATION_PATH = Path("data/ml/valuation_engine/phase11c_allocation_simulation_2025.parquet")
DATASET_PATH = Path("data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet")
REPORT_STEM = "phase11c2_budget_usage_constraint_audit_2025"

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
BASE_RULE = "equal_weight_top5"
ROUND_LOT = 100
BASE_DAILY_BUY_BUDGET = 300_000.0
INITIAL_CASH = 1_000_000.0
PER_CODE_CAP_RATE = 0.38

FUTURE_EVAL_COLUMNS = [
    "future_return_20d",
    "future_max_return_20d",
    "future_max_drawdown_20d",
    "opportunity_value_20d",
    "opportunity_top_decile_20d",
]
DECISION_COLUMNS = [
    "opportunity_top_decile_proba",
    "confidence",
    "opportunity_score_proba_rank",
    "close",
    "turnover_value",
]


@dataclass(frozen=True)
class Phase11C2Options:
    daily_buy_budget: float = BASE_DAILY_BUY_BUDGET
    initial_cash: float = INITIAL_CASH
    per_code_cap_rate: float = PER_CODE_CAP_RATE
    round_lot: int = ROUND_LOT


@dataclass(frozen=True)
class Phase11C2Paths:
    markdown: Path
    json: Path


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


class Phase11C2BudgetUsageConstraintAudit:
    def __init__(self, root: Path | str = ROOT, *, options: Phase11C2Options | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase11C2Options()

    def run(self) -> Phase11C2Paths:
        report = self.build_report()
        return self.save_report(report)

    def build_report(self) -> dict[str, Any]:
        frame = self.load_base_frame()
        daily = self.daily_budget_usage_breakdown(frame)
        reason_summary = self.constraint_reason_summary(daily)
        lot_distribution = self.lot_cost_distribution(frame)
        budget_sensitivity = self.budget_sensitivity(frame)
        max_positions_sensitivity = self.max_positions_sensitivity(frame)
        threshold_sensitivity = self.threshold_sensitivity(frame)
        leakage = self.leakage_checklist()
        recommendation = self.recommendation(daily, reason_summary, budget_sensitivity, max_positions_sensitivity, threshold_sensitivity, leakage)
        return {
            "metadata": self.metadata(),
            "dataset_summary": self.dataset_summary(frame),
            "daily_budget_usage_breakdown": daily,
            "constraint_reason_summary": reason_summary,
            "lot_cost_distribution": lot_distribution,
            "budget_sensitivity": budget_sensitivity,
            "max_positions_sensitivity": max_positions_sensitivity,
            "candidate_threshold_sensitivity": threshold_sensitivity,
            "leakage_checklist": leakage,
            "recommendation": recommendation,
        }

    def load_base_frame(self) -> pd.DataFrame:
        simulation = pd.read_parquet(self.root / SIMULATION_PATH)
        simulation["date"] = pd.to_datetime(simulation["date"], errors="coerce")
        simulation["code"] = simulation["code"].astype("string")
        base = simulation[simulation["rule"].eq(BASE_RULE)].copy()
        base = base[(base["date"] >= pd.Timestamp(START_DATE)) & (base["date"] <= pd.Timestamp(END_DATE))]

        dataset = pd.read_parquet(self.root / DATASET_PATH, columns=["date", "code", "close", "turnover_value"])
        dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
        dataset["code"] = dataset["code"].astype("string")
        data = base.merge(dataset, on=["date", "code"], how="left")
        data["close"] = _numeric(data["close"])
        data["turnover_value"] = _numeric(data["turnover_value"])
        data["lot_cost"] = data["close"] * self.options.round_lot
        return data.sort_values(["date", "opportunity_score_proba_rank", "code"], ascending=[True, False, True]).reset_index(drop=True)

    def daily_budget_usage_breakdown(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for date, group in frame.groupby("date", sort=True):
            allocated = group[group["target_buy_amount"] > 0]
            sorted_group = group.sort_values("opportunity_score_proba_rank", ascending=False)
            affordable = group[_numeric(group["lot_cost"]) <= self.options.daily_buy_budget]
            budget_used = float(_numeric(group["target_buy_amount"]).sum())
            row = {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "candidate_count": int(len(group)),
                "allocated_count": int(len(allocated)),
                "zero_allocation_count": int((group["target_buy_amount"] <= 0).sum()),
                "budget_used": budget_used,
                "budget_usage_rate": budget_used / self.options.daily_buy_budget if self.options.daily_buy_budget else None,
                "unused_budget": self.options.daily_buy_budget - budget_used,
                "top_candidate_close": _safe_float(sorted_group["close"].iloc[0]) if not sorted_group.empty else None,
                "top_candidate_lot_cost": _safe_float(sorted_group["lot_cost"].iloc[0]) if not sorted_group.empty else None,
                "median_candidate_lot_cost": _safe_float(_numeric(group["lot_cost"]).median()),
                "min_candidate_lot_cost": _safe_float(_numeric(group["lot_cost"]).min()),
                "max_candidate_lot_cost": _safe_float(_numeric(group["lot_cost"]).max()),
                "affordable_candidate_count": int(len(affordable)),
                "unaffordable_candidate_count": int(len(group) - len(affordable)),
            }
            row["constraint_reason"] = self.classify_constraint_reason(row, group)
            rows.append(row)
        return rows

    def classify_constraint_reason(self, row: dict[str, Any], group: pd.DataFrame) -> str:
        if row["candidate_count"] <= 0:
            return "no_candidates"
        if row["affordable_candidate_count"] <= 0:
            return "no_affordable_candidates"
        if row["allocated_count"] <= 0:
            positive_weights = (_numeric(group["allocation_weight"]) > 0).sum()
            if positive_weights <= 0:
                return "allocation_weight_zero"
            return "top_candidates_too_expensive"
        if row["allocated_count"] >= 5 and row["budget_usage_rate"] >= 0.80:
            return "max_positions_reached"
        if row["unused_budget"] is not None and row["min_candidate_lot_cost"] is not None and row["unused_budget"] < row["min_candidate_lot_cost"]:
            return "budget_left_below_min_lot"
        if row["allocated_count"] < 5 and row["affordable_candidate_count"] >= 5:
            return "rank_filter_too_strict"
        return "normal_underuse"

    def constraint_reason_summary(self, daily_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frame = pd.DataFrame(daily_rows)
        if frame.empty:
            return []
        return [
            {"constraint_reason": reason, "day_count": int(count)}
            for reason, count in frame["constraint_reason"].value_counts().sort_index().items()
        ]

    def lot_cost_distribution(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        subsets = {
            "top5": frame.groupby("date", group_keys=False).head(5),
            "top10": frame.groupby("date", group_keys=False).head(10),
            "p90+": frame[frame["opportunity_score_proba_rank"] >= 0.90],
            "p95+": frame[frame["opportunity_score_proba_rank"] >= 0.95],
        }
        rows = []
        for label, subset in subsets.items():
            lot_cost = _numeric(subset.get("lot_cost"))
            close = _numeric(subset.get("close"))
            rows.append(
                {
                    "subset": label,
                    "count": int(len(subset)),
                    "close_mean": _safe_float(close.mean()),
                    "close_median": _safe_float(close.median()),
                    "close_p90": _safe_float(close.quantile(0.90)) if not close.dropna().empty else None,
                    "lot_cost_mean": _safe_float(lot_cost.mean()),
                    "lot_cost_median": _safe_float(lot_cost.median()),
                    "lot_cost_p90": _safe_float(lot_cost.quantile(0.90)) if not lot_cost.dropna().empty else None,
                    "affordable_rate_under_300k": _safe_float((lot_cost <= 300_000).mean()) if len(lot_cost) else None,
                    "affordable_rate_under_500k": _safe_float((lot_cost <= 500_000).mean()) if len(lot_cost) else None,
                    "affordable_rate_under_900k": _safe_float((lot_cost <= 900_000).mean()) if len(lot_cost) else None,
                }
            )
        return rows

    def budget_sensitivity(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [self.simulation_summary(frame, daily_budget=budget, max_positions=5, threshold="top5") for budget in [300_000.0, 500_000.0, 900_000.0]]

    def max_positions_sensitivity(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [self.simulation_summary(frame, daily_budget=300_000.0, max_positions=max_positions, threshold=f"top{max_positions}") for max_positions in [5, 10]]

    def threshold_sensitivity(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            self.simulation_summary(frame, daily_budget=300_000.0, max_positions=5, threshold=threshold)
            for threshold in ["top5", "top10", "p90+", "p80+"]
        ]

    def simulation_summary(self, frame: pd.DataFrame, *, daily_budget: float, max_positions: int, threshold: str) -> dict[str, Any]:
        simulated = self.simulate(frame, daily_budget=daily_budget, max_positions=max_positions, threshold=threshold)
        daily_budget_used = simulated.groupby("date")["simulated_buy_amount"].sum()
        usage = daily_budget_used / daily_budget
        allocated = simulated[simulated["simulated_buy_amount"] > 0]
        return {
            "daily_buy_budget": daily_budget,
            "max_positions": max_positions,
            "candidate_threshold": threshold,
            "average_budget_usage_rate": _safe_float(usage.mean()),
            "allocated_rows": int(len(allocated)),
            "average_allocated_candidates_per_day": _safe_float(simulated.assign(is_allocated=simulated["simulated_buy_amount"] > 0).groupby("date")["is_allocated"].sum().mean()),
            "weighted_opportunity_value_20d": self.weighted_mean(allocated, "opportunity_value_20d", _numeric(allocated.get("simulated_buy_amount"))),
            "weighted_opportunity_top_decile_rate": self.weighted_mean(allocated, "opportunity_top_decile_20d", _numeric(allocated.get("simulated_buy_amount"))),
        }

    def simulate(self, frame: pd.DataFrame, *, daily_budget: float, max_positions: int, threshold: str) -> pd.DataFrame:
        chunks = []
        cap_amount = self.options.initial_cash * self.options.per_code_cap_rate
        for _date, group in frame.groupby("date", sort=True):
            candidates = self.threshold_candidates(group, threshold).head(max_positions).copy()
            candidates["simulated_buy_amount"] = 0.0
            candidates["simulated_lot_count"] = 0
            if candidates.empty:
                chunks.append(group.assign(simulated_buy_amount=0.0, simulated_lot_count=0))
                continue
            raw_budget = min(daily_budget / max(len(candidates), 1), cap_amount)
            for index, row in candidates.iterrows():
                lot_cost = _safe_float(row.get("lot_cost")) or 0.0
                lots = int(raw_budget // lot_cost) if lot_cost > 0 else 0
                candidates.at[index, "simulated_lot_count"] = lots
                candidates.at[index, "simulated_buy_amount"] = lots * lot_cost
            rest = group.drop(index=candidates.index, errors="ignore").assign(simulated_buy_amount=0.0, simulated_lot_count=0)
            chunks.append(pd.concat([candidates, rest], ignore_index=True))
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def threshold_candidates(self, group: pd.DataFrame, threshold: str) -> pd.DataFrame:
        ranked = group.sort_values("opportunity_score_proba_rank", ascending=False).copy()
        if threshold == "top5":
            return ranked.head(5)
        if threshold == "top10":
            return ranked.head(10)
        if threshold == "p90+":
            return ranked[ranked["opportunity_score_proba_rank"] >= 0.90]
        if threshold == "p80+":
            return ranked[ranked["opportunity_score_proba_rank"] >= 0.80]
        raise ValueError(f"Unknown threshold: {threshold}")

    def weighted_mean(self, frame: pd.DataFrame, column: str, weights: pd.Series) -> float | None:
        if frame.empty or weights.empty or weights.sum() <= 0:
            return None
        values = _numeric(frame[column])
        valid = values.notna() & weights.notna() & (weights > 0)
        if not valid.any():
            return None
        return _safe_float((values[valid] * weights[valid]).sum() / weights[valid].sum())

    def recommendation(
        self,
        daily_rows: list[dict[str, Any]],
        reason_summary: list[dict[str, Any]],
        budget_sensitivity: list[dict[str, Any]],
        max_positions_sensitivity: list[dict[str, Any]],
        threshold_sensitivity: list[dict[str, Any]],
        leakage: dict[str, Any],
    ) -> dict[str, Any]:
        reasons = {row["constraint_reason"]: row["day_count"] for row in reason_summary}
        top_reason = max(reason_summary, key=lambda row: row["day_count"])["constraint_reason"] if reason_summary else None
        best_budget = max(budget_sensitivity, key=lambda row: row.get("average_budget_usage_rate") or 0, default={})
        best_max_positions = max(
            max_positions_sensitivity,
            key=lambda row: (
                row.get("average_budget_usage_rate") or 0,
                row.get("weighted_opportunity_top_decile_rate") or -1,
            ),
            default={},
        )
        best_threshold_quality = max(threshold_sensitivity, key=lambda row: row.get("weighted_opportunity_top_decile_rate") or -1, default={})
        usage_values = [row.get("average_budget_usage_rate") for row in budget_sensitivity if row.get("average_budget_usage_rate") is not None]
        direction_found = bool(usage_values and max(usage_values) > min(usage_values))
        return {
            "main_budget_bottleneck": self.main_bottleneck(top_reason, reasons),
            "recommended_daily_budget": best_budget.get("daily_buy_budget"),
            "recommended_max_positions": best_max_positions.get("max_positions"),
            "recommended_candidate_threshold": best_threshold_quality.get("candidate_threshold"),
            "expected_budget_usage_range": {
                "min": _safe_float(min(usage_values)) if usage_values else None,
                "max": _safe_float(max(usage_values)) if usage_values else None,
            },
            "quality_tradeoff_summary": self.quality_tradeoff_summary(threshold_sensitivity),
            "ready_for_phase11d": bool(direction_found and leakage["leakage_risk"] == "low" and not leakage["blocking_issues"]),
            "recommended_next_phase": "Phase11-D strict limited-scope design" if direction_found else "Phase11-B2 or Phase11-C3",
        }

    def main_bottleneck(self, top_reason: str | None, reasons: dict[str, int]) -> str:
        if top_reason in {"rank_filter_too_strict", "normal_underuse"}:
            return "round_lot_and_top_candidate_affordability_limit_daily_budget_usage"
        if top_reason:
            return top_reason
        return "unknown"

    def quality_tradeoff_summary(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No threshold sensitivity rows."
        compact = [
            f"{row['candidate_threshold']}: usage={row.get('average_budget_usage_rate'):.3f}, top_decile={row.get('weighted_opportunity_top_decile_rate'):.3f}"
            for row in rows
            if row.get("average_budget_usage_rate") is not None and row.get("weighted_opportunity_top_decile_rate") is not None
        ]
        return "; ".join(compact)

    def leakage_checklist(self) -> dict[str, Any]:
        future_eval = FUTURE_EVAL_COLUMNS
        return {
            "future_columns_used_as_features": [],
            "future_columns_used_only_for_evaluation": future_eval,
            "backtest_columns_used": [],
            "trade_result_columns_used": [],
            "cash_or_portfolio_columns_used_as_model_features": [],
            "selected_or_bought_used": False,
            "current_pm_multiplier_used": False,
            "historical_predictions_regenerated": False,
            "strategy_backtest_executed": False,
            "profile_changed": False,
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "11-C2",
            "audit_only": True,
            "period": {"start": START_DATE, "end": END_DATE},
            "strategy_backtest_executed": False,
            "profile_added": False,
            "profile_modified": False,
            "current_model_overwritten": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
            "simulation_path": str(self.root / SIMULATION_PATH),
            "dataset_path": str(self.root / DATASET_PATH),
        }

    def dataset_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(frame)),
            "unique_codes": int(frame["code"].nunique()),
            "candidate_days": int(frame["date"].nunique()),
            "date_range": {
                "min": frame["date"].min().strftime("%Y-%m-%d") if not frame.empty else None,
                "max": frame["date"].max().strftime("%Y-%m-%d") if not frame.empty else None,
            },
        }

    def save_report(self, report: dict[str, Any]) -> Phase11C2Paths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase11C2Paths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Phase 11-C2 Budget Usage Constraint Audit",
                "",
                "## Scope",
                "",
                "- 2025-only budget constraint audit",
                "- no strategy backtest, no profile change, no model overwrite",
                "- future labels are used only for quality evaluation",
                "",
                "## Dataset Summary",
                "",
                self._table([report["dataset_summary"]], ["rows", "unique_codes", "candidate_days", "date_range"]),
                "",
                "## Leakage Checklist",
                "",
                self._table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_evaluation", "backtest_columns_used", "trade_result_columns_used", "cash_or_portfolio_columns_used_as_model_features", "selected_or_bought_used", "current_pm_multiplier_used", "historical_predictions_regenerated", "strategy_backtest_executed", "profile_changed", "leakage_risk", "blocking_issues"]),
                "",
                "## Constraint Reason Summary",
                "",
                self._table(report["constraint_reason_summary"], ["constraint_reason", "day_count"]),
                "",
                "## Daily Budget Usage Breakdown",
                "",
                self._table(report["daily_budget_usage_breakdown"][:40], ["date", "candidate_count", "allocated_count", "budget_used", "budget_usage_rate", "unused_budget", "top_candidate_lot_cost", "median_candidate_lot_cost", "min_candidate_lot_cost", "affordable_candidate_count", "unaffordable_candidate_count", "constraint_reason"]),
                "",
                "## Lot Cost Distribution",
                "",
                self._table(report["lot_cost_distribution"], ["subset", "count", "close_mean", "close_median", "close_p90", "lot_cost_mean", "lot_cost_median", "lot_cost_p90", "affordable_rate_under_300k", "affordable_rate_under_500k", "affordable_rate_under_900k"]),
                "",
                "## Budget Sensitivity",
                "",
                self._table(report["budget_sensitivity"], ["daily_buy_budget", "max_positions", "candidate_threshold", "average_budget_usage_rate", "allocated_rows", "average_allocated_candidates_per_day", "weighted_opportunity_value_20d", "weighted_opportunity_top_decile_rate"]),
                "",
                "## Max Positions Sensitivity",
                "",
                self._table(report["max_positions_sensitivity"], ["daily_buy_budget", "max_positions", "candidate_threshold", "average_budget_usage_rate", "allocated_rows", "average_allocated_candidates_per_day", "weighted_opportunity_value_20d", "weighted_opportunity_top_decile_rate"]),
                "",
                "## Candidate Threshold Sensitivity",
                "",
                self._table(report["candidate_threshold_sensitivity"], ["daily_buy_budget", "max_positions", "candidate_threshold", "average_budget_usage_rate", "allocated_rows", "average_allocated_candidates_per_day", "weighted_opportunity_value_20d", "weighted_opportunity_top_decile_rate"]),
                "",
                "## Recommendation",
                "",
                self._table([report["recommendation"]], ["main_budget_bottleneck", "recommended_daily_budget", "recommended_max_positions", "recommended_candidate_threshold", "expected_budget_usage_range", "quality_tradeoff_summary", "ready_for_phase11d", "recommended_next_phase"]),
                "",
            ]
        )

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in rows:
            values = [self._format_cell(row.get(column)) for column in columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def _format_cell(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False, default=str)
        if value is None:
            return ""
        return str(value).replace("\n", " ")
