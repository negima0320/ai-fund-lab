"""Phase 13-E2 Integrated Strategy Rule Tuning.

This is a 2025-only lightweight rule tuning prototype around the Phase13-E
break-even guard result. Entry and exit decisions do not use future columns;
future columns are reported only as post-trade BUY quality evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _safe_float
from ml.phase13a_horizon_reality_audit import ROOT
from ml.phase13e_integrated_strategy_prototype import (
    COST_RATE,
    DAILY_BUY_LIMIT,
    INITIAL_CASH,
    MAX_POSITIONS,
    PRIMARY_CANDIDATE_SET,
    ROUND_LOT,
    Phase13EIntegratedStrategyPrototype,
    Phase13EPaths,
    Position,
)


REPORT_STEM = "phase13e2_integrated_strategy_rule_tuning_2025"
PHASE13E_E3_REFERENCE_ANNUAL_RETURN = 0.15525364000000064
REQUIRED_REPORT_KEYS = [
    "recommended_strategy_variant",
    "recommended_reason",
    "annual_return_after_cost",
    "improvement_vs_phase13e_e3",
    "meets_annual_return_target",
    "meets_minimum_line",
    "ready_for_phase13f",
    "ready_for_broad_backtest",
    "ready_for_live_adoption",
    "recommended_next_phase",
    "leakage_risk",
    "blocking_issues",
]


@dataclass(frozen=True)
class VariantConfig:
    name: str
    candidate_top_n: int = 50
    entry_rank: str = "opportunity_downside_score"
    break_even_peak: float | None = 0.03
    break_even_floor: float = 0.00
    stop_loss: float | None = None
    fixed_hold_days: int = 20


VARIANTS = [
    VariantConfig("T0_baseline_e3_reproduction"),
    VariantConfig("T1_break_even_tighter_trigger", break_even_peak=0.04),
    VariantConfig("T2_break_even_looser_trigger", break_even_peak=0.02),
    VariantConfig("T3_break_even_small_profit_floor", break_even_peak=0.03, break_even_floor=0.01),
    VariantConfig("T4_stop_plus_break_even", break_even_peak=0.03, stop_loss=-0.08),
    VariantConfig("T5_stop_rework_plus_break_even", break_even_peak=0.03, stop_loss=-0.05),
    VariantConfig("T6_longer_max_hold_break_even", break_even_peak=0.03, fixed_hold_days=25),
    VariantConfig("T7_shorter_max_hold_break_even", break_even_peak=0.03, fixed_hold_days=15),
    VariantConfig("T8_candidate_strength_top100_e3", candidate_top_n=100),
    VariantConfig("T9_candidate_strength_rank_e3", entry_rank="candidate_strength"),
]


class Phase13E2IntegratedStrategyRuleTuning(Phase13EIntegratedStrategyPrototype):
    def __init__(self, root: Path | str = ROOT) -> None:
        super().__init__(root)
        self.active_config = VARIANTS[0]

    def run(self) -> Phase13EPaths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data, source_info = self.phase13d.phase13a.load_comparison_dataset()
        data = self.prepare_data(data)
        leakage = self.leakage_checklist()
        leakage["blocking_issues"] = self.blocking_issues(data)
        if len(VARIANTS) > 10:
            leakage["blocking_issues"].append("variant_count_exceeds_10")
        leakage["leakage_risk"] = "low" if not leakage["blocking_issues"] else "medium"
        variant_results = [self.simulate(data, config.name) for config in VARIANTS]
        recommendations = self.recommendations(variant_results, leakage)
        return {
            "metadata": self.metadata(),
            "input_artifact_summary": self.input_artifact_summary(data, source_info, leakage),
            "strategy_parameters": self.strategy_parameters(),
            "variant_definitions": [config.__dict__ for config in VARIANTS],
            "variant_count": len(VARIANTS),
            "variant_results": variant_results,
            "final_recommendation": recommendations,
            "leakage_checklist": leakage,
            **{key: recommendations.get(key) for key in REQUIRED_REPORT_KEYS},
        }

    def simulate(self, data: pd.DataFrame, variant: str) -> dict[str, Any]:
        self.active_config = self.config_for_variant(variant)
        return super().simulate(data, variant)

    def config_for_variant(self, name: str) -> VariantConfig:
        for config in VARIANTS:
            if config.name == name:
                return config
        raise ValueError(f"Unknown variant: {name}")

    def entry_candidates(self, frame: pd.DataFrame, open_codes: set[str]) -> pd.DataFrame:
        frame = frame.copy()
        frame["_strength_rank"] = frame.groupby("date")["candidate_strength"].rank(method="first", ascending=False)
        candidates = frame[(frame["_strength_rank"] <= self.active_config.candidate_top_n) & (~frame["code"].astype("string").isin(open_codes))].copy()
        primary_rank = self.active_config.entry_rank
        secondary_rank = "opportunity_downside_score" if primary_rank == "candidate_strength" else "candidate_strength"
        return candidates.sort_values([primary_rank, secondary_rank, "turnover_value", "code"], ascending=[False, False, False, True])

    def exit_reason(self, position: Position, current_price: float, variant: str) -> str | None:
        config = self.config_for_variant(variant)
        current_return = current_price / position.entry_price - 1.0
        if config.break_even_peak is not None and position.peak_return >= config.break_even_peak and current_return <= config.break_even_floor:
            return "break_even_exit"
        if config.stop_loss is not None and current_return <= config.stop_loss:
            return "stop_loss_exit"
        return "fixed_hold_exit" if position.holding_days >= config.fixed_hold_days else None

    def recommendations(self, results: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        valid = [row for row in results if row.get("annual_return_after_cost") is not None]
        if leakage["blocking_issues"] or not valid:
            return self.blocked_recommendation(leakage)
        best = max(valid, key=lambda row: self.score(row))
        t0 = next((row for row in valid if row["variant"] == "T0_baseline_e3_reproduction"), None)
        baseline = t0["annual_return_after_cost"] if t0 else PHASE13E_E3_REFERENCE_ANNUAL_RETURN
        improvement = _safe_float(best["annual_return_after_cost"] - baseline)
        meets_annual = best["annual_return_after_cost"] >= 0.50
        meets_minimum = (
            meets_annual
            and self.value(best, "PF") >= 2.0
            and self.value(best, "DD") >= -0.10
            and self.value(best, "capital_utilization") >= 0.60
            and self.value(best, "net_profit") > 0
        )
        blocking = list(leakage["blocking_issues"])
        if self.value(best, "DD") < -0.15:
            blocking.append("recommended_variant_dd_below_minus_15pct")
        if self.value(best, "PF") < 1.0:
            blocking.append("recommended_variant_pf_below_1")
        if self.value(best, "trade_count") < 10:
            blocking.append("recommended_variant_low_trade_count")
        if meets_minimum and not blocking:
            next_phase = "Phase13-F Strict OOS / Multi-Year Limited Check"
        elif improvement and improvement > 0.03:
            next_phase = "Phase13-E3 Entry/Exit Interaction Audit"
        else:
            next_phase = "Phase13-C Horizon-Aware Valuation Prototype"
        return {
            "recommended_strategy_variant": best["variant"],
            "recommended_reason": (
                f"Best annual_return_after_cost={best['annual_return_after_cost']:.4f}; "
                f"improvement_vs_T0={improvement:.4f}; PF={self.value(best, 'PF'):.4f}; "
                f"DD={self.value(best, 'DD'):.4f}; utilization={self.value(best, 'capital_utilization'):.4f}."
            ),
            "annual_return_after_cost": best["annual_return_after_cost"],
            "improvement_vs_phase13e_e3": improvement,
            "meets_annual_return_target": meets_annual,
            "meets_minimum_line": meets_minimum and not blocking,
            "ready_for_phase13f": meets_minimum and not blocking,
            "ready_for_broad_backtest": False,
            "ready_for_live_adoption": False,
            "recommended_next_phase": next_phase,
            "leakage_risk": "low" if not blocking else "medium",
            "blocking_issues": blocking,
        }

    def blocked_recommendation(self, leakage: dict[str, Any]) -> dict[str, Any]:
        return {
            "recommended_strategy_variant": None,
            "recommended_reason": "blocking issues or no variant results",
            "annual_return_after_cost": None,
            "improvement_vs_phase13e_e3": None,
            "meets_annual_return_target": False,
            "meets_minimum_line": False,
            "ready_for_phase13f": False,
            "ready_for_broad_backtest": False,
            "ready_for_live_adoption": False,
            "recommended_next_phase": "Phase13-C Horizon-Aware Valuation Prototype",
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def strategy_parameters(self) -> dict[str, Any]:
        params = super().strategy_parameters()
        params.update(
            {
                "phase13e_e3_reference_annual_return_after_cost": PHASE13E_E3_REFERENCE_ANNUAL_RETURN,
                "variant_count": len(VARIANTS),
                "variant_limit": 10,
            }
        )
        return params

    def leakage_checklist(self) -> dict[str, Any]:
        checklist = super().leakage_checklist()
        checklist["variant_count"] = len(VARIANTS)
        checklist["variant_count_within_limit"] = len(VARIANTS) <= 10
        return checklist

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "13-E2",
            "scope": "2025-only integrated strategy rule tuning",
            "new_model_trained": False,
            "full_backtest_executed": False,
            "strategy_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
            "ready_for_live_adoption": False,
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase13EPaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase13EPaths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Phase 13-E2 Integrated Strategy Rule Tuning",
                "",
                "## Input Artifact Summary",
                "",
                self.table([report["input_artifact_summary"]], ["row_count", "date_min", "date_max", "unique_code_count", "primary_candidate_set", "leakage_risk", "blocking_issues"]),
                "",
                "## Strategy Parameters",
                "",
                self.table([report["strategy_parameters"]], ["primary_candidate_set", "candidate_generation", "entry_rank", "initial_cash", "daily_buy_limit", "max_positions", "round_lot", "cost_rate", "variant_count", "variant_limit", "future_columns_used_for_entry_exit"]),
                "",
                "## Variant Definitions",
                "",
                self.table(report["variant_definitions"], ["name", "candidate_top_n", "entry_rank", "break_even_peak", "break_even_floor", "stop_loss", "fixed_hold_days"]),
                "",
                "## Variant Results",
                "",
                self.table(report["variant_results"], ["variant", "annual_return_after_cost", "net_profit", "final_assets", "PF", "DD", "win_rate", "capital_utilization", "trade_count", "average_holding_days", "median_holding_days", "winner_to_loser_count", "winner_to_loser_rate", "break_even_exit_count", "stop_loss_exit_count", "fixed_hold_exit_count", "avg_profit_decay_before_exit", "entry_top_decile_rate_20d", "entry_downside_bad_rate_20d", "entry_mean_future_return_20d"]),
                "",
                "## Final Recommendation",
                "",
                self.table([report["final_recommendation"]], REQUIRED_REPORT_KEYS),
                "",
                "## Leakage Checklist",
                "",
                self.table([report["leakage_checklist"]], ["future_columns_used_as_entry_or_exit_features", "future_columns_used_only_for_evaluation", "new_model_trained", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "strategy_backtest_executed", "historical_predictions_regenerated", "jquants_api_called", "openai_api_called", "cost_rate", "variant_count", "variant_count_within_limit", "leakage_risk", "blocking_issues"]),
                "",
            ]
        )
