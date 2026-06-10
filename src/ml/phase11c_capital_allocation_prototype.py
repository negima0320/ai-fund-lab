"""Phase 11-C Capital Allocation Engine prototype.

This is a lightweight 2025-only allocation-quality simulation. It reads the
Phase 11-B Valuation Engine candidate model and Phase 11-A dataset, but it does
not run a strategy backtest, add or modify profiles, regenerate predictions, or
use future labels for allocation decisions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = Path("data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet")
MODEL_DIR = Path("models/ml/valuation_engine/candidate_phase11b")
REPORT_STEM = "phase11c_capital_allocation_prototype_2025"
SIMULATION_PATH = Path("data/ml/valuation_engine/phase11c_allocation_simulation_2025.parquet")

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
ROUND_LOT = 100

ALLOCATION_INPUT_COLUMNS = [
    "opportunity_top_decile_proba",
    "confidence",
    "opportunity_score_proba_rank",
    "close",
    "turnover_value",
]
FUTURE_EVAL_COLUMNS = [
    "future_return_20d",
    "future_max_return_20d",
    "future_max_drawdown_20d",
    "opportunity_value_20d",
    "opportunity_top_decile_20d",
]
FORBIDDEN_TOKENS = {
    "backtest",
    "trade",
    "profit",
    "loss",
    "cash",
    "portfolio",
    "position",
    "selected",
    "bought",
    "affordable",
    "skip",
    "exit",
    "final_assets",
    "pm_multiplier",
    "current_pm",
}
RULES = [
    "equal_weight_top5",
    "proba_rank_weighted",
    "proba_confidence_weighted",
    "conservative_top_only",
]


@dataclass(frozen=True)
class Phase11COptions:
    initial_cash: float = 1_000_000.0
    daily_buy_budget: float = 300_000.0
    max_positions: int = 5
    per_code_cap_rate: float = 0.38
    round_lot: int = ROUND_LOT
    save_simulation: bool = True


@dataclass(frozen=True)
class Phase11CPaths:
    markdown: Path
    json: Path
    simulation: Path | None


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


class Phase11CCapitalAllocationPrototype:
    def __init__(self, root: Path | str = ROOT, *, options: Phase11COptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase11COptions()

    def run(self) -> Phase11CPaths:
        report, simulation = self.build_report()
        return self.save_outputs(report, simulation)

    def build_report(self) -> tuple[dict[str, Any], pd.DataFrame]:
        data = self.load_valuation_frame()
        leakage = self.leakage_checklist(ALLOCATION_INPUT_COLUMNS)
        if leakage["blocking_issues"]:
            report = {
                "metadata": self.metadata(),
                "simulation_conditions": self.simulation_conditions(),
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], leakage),
            }
            return report, pd.DataFrame()

        simulation = self.simulate_all_rules(data)
        comparison = self.compare_rules(simulation)
        best_rule = self.best_rule(comparison)
        report = {
            "metadata": self.metadata(),
            "simulation_conditions": self.simulation_conditions(),
            "dataset_summary": self.dataset_summary(data),
            "valuation_output_summary": self.valuation_output_summary(data),
            "allocation_coverage": self.allocation_coverage(simulation),
            "capital_usage_proxy": self.capital_usage_proxy(simulation),
            "quality_by_bucket": self.quality_by_bucket(simulation),
            "rule_comparison": comparison,
            "best_rule": best_rule,
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(comparison, leakage),
        }
        return report, simulation

    def load_valuation_frame(self) -> pd.DataFrame:
        import joblib

        model_dir = self.root / MODEL_DIR
        feature_columns = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
        classifier = joblib.load(model_dir / "opportunity_top_decile_20d_classifier.joblib")
        dataset_columns = sorted(set(["date", "code", "close", "turnover_value", *feature_columns, *FUTURE_EVAL_COLUMNS]))
        data = pd.read_parquet(self.root / DATASET_PATH, columns=dataset_columns)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        data = data[(data["date"] >= pd.Timestamp(START_DATE)) & (data["date"] <= pd.Timestamp(END_DATE))].copy()
        for column in feature_columns:
            data[column] = _numeric(data[column])
        proba = np.asarray(classifier.predict_proba(data[feature_columns]))[:, 1]
        data["opportunity_top_decile_proba"] = proba
        data["confidence"] = (data["opportunity_top_decile_proba"] - 0.5).abs() * 2.0
        grouped = data.groupby("date")["opportunity_top_decile_proba"]
        rank = grouped.rank(method="average", pct=True)
        data["opportunity_score_proba_rank"] = rank
        return data.sort_values(["date", "code"]).reset_index(drop=True)

    def simulate_all_rules(self, data: pd.DataFrame) -> pd.DataFrame:
        frames = [self.simulate_rule(data, rule) for rule in RULES]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def simulate_rule(self, data: pd.DataFrame, rule: str) -> pd.DataFrame:
        frames = []
        for date, group in data.groupby("date", sort=True):
            candidates = group.copy()
            candidates["allocation_weight"] = self.rule_weight(candidates, rule)
            candidates["allocation_bucket"] = self.allocation_bucket(candidates, rule)
            candidates = candidates.sort_values(
                ["allocation_weight", "opportunity_top_decile_proba", "turnover_value", "code"],
                ascending=[False, False, False, True],
            )
            positive_index = candidates[candidates["allocation_weight"] > 0].head(self.options.max_positions).index
            selected = candidates.index.isin(positive_index)
            candidates.loc[~selected, "allocation_weight"] = 0.0
            candidates.loc[~selected, "allocation_bucket"] = "zero"
            candidates = self.assign_amounts(candidates)
            candidates["rule"] = rule
            candidates["simulation_date"] = pd.Timestamp(date)
            frames.append(candidates)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def rule_weight(self, frame: pd.DataFrame, rule: str) -> pd.Series:
        rank = _numeric(frame["opportunity_score_proba_rank"]).fillna(0.0)
        confidence = _numeric(frame["confidence"]).fillna(0.0)
        if rule == "equal_weight_top5":
            order = frame["opportunity_top_decile_proba"].rank(method="first", ascending=False)
            return (order <= self.options.max_positions).astype(float)
        if rule == "proba_rank_weighted":
            return rank.map(lambda value: 1.0 if value >= 0.90 else 0.70 if value >= 0.75 else 0.40 if value >= 0.50 else 0.20 if value >= 0.25 else 0.0)
        if rule == "proba_confidence_weighted":
            penalty = confidence.map(lambda value: 0.5 if value < 0.20 else 1.0)
            return rank * confidence * penalty
        if rule == "conservative_top_only":
            return rank.map(lambda value: 1.20 if value >= 0.95 else 1.00 if value >= 0.90 else 0.0)
        raise ValueError(f"Unknown allocation rule: {rule}")

    def allocation_bucket(self, frame: pd.DataFrame, rule: str) -> pd.Series:
        rank = _numeric(frame["opportunity_score_proba_rank"]).fillna(0.0)
        if rule == "equal_weight_top5":
            order = frame["opportunity_top_decile_proba"].rank(method="first", ascending=False)
            return order.map(lambda value: "top5" if value <= self.options.max_positions else "zero")
        return rank.map(lambda value: "p95+" if value >= 0.95 else "p90_95" if value >= 0.90 else "p75_90" if value >= 0.75 else "p50_75" if value >= 0.50 else "p25_50" if value >= 0.25 else "zero")

    def assign_amounts(self, candidates: pd.DataFrame) -> pd.DataFrame:
        result = candidates.copy()
        result["target_buy_amount"] = 0.0
        result["target_lot_count"] = 0
        positive = result["allocation_weight"] > 0
        weight_sum = float(result.loc[positive, "allocation_weight"].sum())
        if weight_sum <= 0:
            return result
        cap_amount = self.options.initial_cash * self.options.per_code_cap_rate
        for index, row in result.loc[positive].iterrows():
            raw_amount = min(self.options.daily_buy_budget * float(row["allocation_weight"]) / weight_sum, cap_amount)
            lot_value = float(row["close"]) * self.options.round_lot if row.get("close") and float(row["close"]) > 0 else 0.0
            lots = int(raw_amount // lot_value) if lot_value > 0 else 0
            amount = lots * lot_value
            result.at[index, "target_lot_count"] = lots
            result.at[index, "target_buy_amount"] = amount
            if amount <= 0:
                result.at[index, "allocation_bucket"] = "zero_unaffordable_lot"
        return result

    def allocation_coverage(self, simulation: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for rule, group in simulation.groupby("rule", sort=False):
            daily = group.groupby("date")
            allocated = group[group["target_buy_amount"] > 0]
            allocated_per_day = group.assign(is_allocated=group["target_buy_amount"] > 0).groupby("date")["is_allocated"].sum()
            rows.append(
                {
                    "rule": rule,
                    "candidate_days": int(group["date"].nunique()),
                    "allocated_days": int(allocated["date"].nunique()),
                    "allocated_rows": int(len(allocated)),
                    "zero_allocation_rows": int((group["target_buy_amount"] <= 0).sum()),
                    "average_allocated_candidates_per_day": _safe_float(allocated_per_day.mean()),
                }
            )
        return rows

    def capital_usage_proxy(self, simulation: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for rule, group in simulation.groupby("rule", sort=False):
            daily_amount = group.groupby("date")["target_buy_amount"].sum()
            usage = daily_amount / self.options.daily_buy_budget
            rows.append(
                {
                    "rule": rule,
                    "average_daily_budget_usage_rate": _safe_float(usage.mean()),
                    "median_daily_budget_usage_rate": _safe_float(usage.median()),
                    "days_below_60pct_budget_usage": int((usage < 0.60).sum()),
                    "days_above_80pct_budget_usage": int((usage >= 0.80).sum()),
                }
            )
        return rows

    def quality_by_bucket(self, simulation: pd.DataFrame) -> list[dict[str, Any]]:
        allocated = simulation[simulation["target_buy_amount"] > 0].copy()
        rows = []
        for (rule, bucket), group in allocated.groupby(["rule", "allocation_bucket"], sort=True):
            rows.append(self.weighted_quality_row(rule, bucket, group))
        return rows

    def compare_rules(self, simulation: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for rule, group in simulation.groupby("rule", sort=False):
            allocated = group[group["target_buy_amount"] > 0].copy()
            daily_amount = group.groupby("date")["target_buy_amount"].sum()
            usage = daily_amount / self.options.daily_buy_budget
            allocated_per_day = group.assign(is_allocated=group["target_buy_amount"] > 0).groupby("date")["is_allocated"].sum()
            row = self.weighted_quality_row(rule, "all_allocated", allocated)
            row.update(
                {
                    "allocated_candidates_per_day": _safe_float(allocated_per_day.mean()),
                    "budget_usage_rate": _safe_float(usage.mean()),
                    "allocated_rows": int(len(allocated)),
                }
            )
            rows.append(row)
        return rows

    def weighted_quality_row(self, rule: str, bucket: str, group: pd.DataFrame) -> dict[str, Any]:
        weights = _numeric(group.get("target_buy_amount"))
        if group.empty or weights.sum() <= 0:
            return {
                "rule": rule,
                "allocation_bucket": bucket,
                "count": 0,
                "weighted_future_return_20d": None,
                "weighted_future_max_return_20d": None,
                "weighted_future_max_drawdown_20d": None,
                "weighted_opportunity_value_20d": None,
                "weighted_opportunity_top_decile_rate": None,
            }
        return {
            "rule": rule,
            "allocation_bucket": bucket,
            "count": int(len(group)),
            "weighted_future_return_20d": self.weighted_mean(group, "future_return_20d", weights),
            "weighted_future_max_return_20d": self.weighted_mean(group, "future_max_return_20d", weights),
            "weighted_future_max_drawdown_20d": self.weighted_mean(group, "future_max_drawdown_20d", weights),
            "weighted_opportunity_value_20d": self.weighted_mean(group, "opportunity_value_20d", weights),
            "weighted_opportunity_top_decile_rate": self.weighted_mean(group, "opportunity_top_decile_20d", weights),
        }

    def weighted_mean(self, frame: pd.DataFrame, column: str, weights: pd.Series) -> float | None:
        values = _numeric(frame[column])
        valid = values.notna() & weights.notna() & (weights > 0)
        if not valid.any():
            return None
        return _safe_float((values[valid] * weights[valid]).sum() / weights[valid].sum())

    def best_rule(self, comparison: list[dict[str, Any]]) -> dict[str, Any]:
        if not comparison:
            return {}
        ranked = sorted(
            comparison,
            key=lambda row: (
                row.get("weighted_opportunity_top_decile_rate") or -1,
                row.get("weighted_opportunity_value_20d") or -999,
                row.get("budget_usage_rate") or -1,
            ),
            reverse=True,
        )
        best = dict(ranked[0])
        best["selection_reason"] = "highest weighted opportunity_top_decile_rate, then opportunity_value and budget usage"
        return best

    def leakage_checklist(self, decision_columns: list[str]) -> dict[str, Any]:
        future_used = [column for column in decision_columns if column.startswith("future_") or column.startswith("opportunity_value")]
        forbidden = [column for column in decision_columns if self.is_forbidden_column(column)]
        blocking = []
        if future_used:
            blocking.append("future/evaluation column used for allocation decision")
        if forbidden:
            blocking.append("forbidden column used for allocation decision")
        return {
            "future_columns_used_as_features": future_used,
            "backtest_columns_used": [column for column in decision_columns if "backtest" in column.lower()],
            "trade_result_columns_used": [column for column in decision_columns if any(token in column.lower() for token in ["trade", "profit", "loss"])],
            "cash_or_portfolio_columns_used_as_model_features": [column for column in decision_columns if any(token in column.lower() for token in ["cash", "portfolio", "position"])],
            "selected_or_bought_used": any(token in column.lower() for column in decision_columns for token in ["selected", "bought", "affordable"]),
            "current_pm_multiplier_used": any("pm_multiplier" in column.lower() or "current_pm" in column.lower() for column in decision_columns),
            "historical_predictions_regenerated": False,
            "strategy_backtest_executed": False,
            "profile_changed": False,
            "leakage_risk": "high" if blocking else "low",
            "blocking_issues": blocking,
        }

    def is_forbidden_column(self, column: str) -> bool:
        lowered = column.lower()
        return any(token in lowered for token in FORBIDDEN_TOKENS)

    def recommendation(self, comparison: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        best = self.best_rule(comparison)
        ready = bool(best) and leakage["leakage_risk"] == "low" and not leakage["blocking_issues"]
        return {
            "best_allocation_rule": best.get("rule"),
            "ready_for_phase11d": ready,
            "recommended_next_phase": "Phase 11-D Combined Backtest design with strict limited scope" if ready else "Return to Phase 11-B/C fixes",
            "reason": best.get("selection_reason"),
            "known_risks": [
                "This is allocation-quality simulation only, not a strategy backtest",
                "No sell logic, Exit AI, holding state, or cash carryover is modeled",
                "Future labels are used only for evaluation metrics",
            ],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "11-C",
            "prototype_only": True,
            "simulation_only": True,
            "strategy_backtest_executed": False,
            "profile_added": False,
            "profile_modified": False,
            "current_model_overwritten": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
            "period": {"start": START_DATE, "end": END_DATE},
            "model_dir": str(self.root / MODEL_DIR),
            "dataset_path": str(self.root / DATASET_PATH),
        }

    def simulation_conditions(self) -> dict[str, Any]:
        return {
            "initial_cash": self.options.initial_cash,
            "daily_buy_budget": self.options.daily_buy_budget,
            "max_positions": self.options.max_positions,
            "per_code_cap_rate": self.options.per_code_cap_rate,
            "round_lot": self.options.round_lot,
            "sell_logic_integrated": False,
            "exit_ai_integrated": False,
        }

    def dataset_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(data)),
            "unique_codes": int(data["code"].nunique()),
            "candidate_days": int(data["date"].nunique()),
            "date_range": {"min": data["date"].min().strftime("%Y-%m-%d"), "max": data["date"].max().strftime("%Y-%m-%d")},
        }

    def valuation_output_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        return {
            "opportunity_top_decile_proba_mean": _safe_float(data["opportunity_top_decile_proba"].mean()),
            "opportunity_top_decile_proba_p90": _safe_float(data["opportunity_top_decile_proba"].quantile(0.90)),
            "confidence_mean": _safe_float(data["confidence"].mean()),
            "opportunity_score_proba_rank_mean": _safe_float(data["opportunity_score_proba_rank"].mean()),
        }

    def save_outputs(self, report: dict[str, Any], simulation: pd.DataFrame) -> Phase11CPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        sim_path = None
        if self.options.save_simulation and not simulation.empty:
            sim_path = self.root / SIMULATION_PATH
            sim_path.parent.mkdir(parents=True, exist_ok=True)
            keep = [
                "rule",
                "date",
                "code",
                "opportunity_top_decile_proba",
                "confidence",
                "opportunity_score_proba_rank",
                "allocation_weight",
                "target_buy_amount",
                "target_lot_count",
                "allocation_bucket",
                *FUTURE_EVAL_COLUMNS,
            ]
            simulation[[column for column in keep if column in simulation.columns]].to_parquet(sim_path, index=False)
        return Phase11CPaths(markdown=md_path, json=json_path, simulation=sim_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Phase 11-C Capital Allocation Engine Prototype",
                "",
                "## Scope",
                "",
                "- 2025-only lightweight allocation-quality simulation",
                "- no strategy backtest, no profile addition, no existing model overwrite",
                "- future labels are used only for evaluation metrics",
                "",
                "## Simulation Conditions",
                "",
                self._table([report["simulation_conditions"]], ["initial_cash", "daily_buy_budget", "max_positions", "per_code_cap_rate", "round_lot", "sell_logic_integrated", "exit_ai_integrated"]),
                "",
                "## Dataset Summary",
                "",
                self._table([report.get("dataset_summary", {})], ["rows", "unique_codes", "candidate_days", "date_range"]),
                "",
                "## Leakage Checklist",
                "",
                self._table([report["leakage_checklist"]], ["future_columns_used_as_features", "backtest_columns_used", "trade_result_columns_used", "cash_or_portfolio_columns_used_as_model_features", "selected_or_bought_used", "current_pm_multiplier_used", "historical_predictions_regenerated", "strategy_backtest_executed", "profile_changed", "leakage_risk", "blocking_issues"]),
                "",
                "## Allocation Coverage",
                "",
                self._table(report.get("allocation_coverage", []), ["rule", "candidate_days", "allocated_days", "allocated_rows", "zero_allocation_rows", "average_allocated_candidates_per_day"]),
                "",
                "## Capital Usage Proxy",
                "",
                self._table(report.get("capital_usage_proxy", []), ["rule", "average_daily_budget_usage_rate", "median_daily_budget_usage_rate", "days_below_60pct_budget_usage", "days_above_80pct_budget_usage"]),
                "",
                "## Rule Comparison",
                "",
                self._table(report.get("rule_comparison", []), ["rule", "allocated_candidates_per_day", "budget_usage_rate", "allocated_rows", "weighted_future_return_20d", "weighted_future_max_return_20d", "weighted_future_max_drawdown_20d", "weighted_opportunity_value_20d", "weighted_opportunity_top_decile_rate"]),
                "",
                "## Quality By Bucket",
                "",
                self._table(report.get("quality_by_bucket", []), ["rule", "allocation_bucket", "count", "weighted_future_return_20d", "weighted_future_max_return_20d", "weighted_future_max_drawdown_20d", "weighted_opportunity_value_20d", "weighted_opportunity_top_decile_rate"]),
                "",
                "## Best Rule",
                "",
                self._table([report.get("best_rule", {})], ["rule", "weighted_opportunity_top_decile_rate", "weighted_opportunity_value_20d", "budget_usage_rate", "selection_reason"]),
                "",
                "## Recommendation",
                "",
                self._table([report["recommendation"]], ["best_allocation_rule", "ready_for_phase11d", "recommended_next_phase", "reason", "known_risks"]),
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
