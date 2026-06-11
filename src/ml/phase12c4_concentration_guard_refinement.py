"""Phase 12-C4 concentration guard refinement.

This 2025-only lightweight check keeps the Phase 12-C2 downside-squared
normalized allocation and B5_2 exit, then compares refined concentration
controls: capped redistribution, score-gap dynamic caps, and staged-buy
proxies. It does not run a full backtest, change profiles, overwrite models, or
regenerate historical predictions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase12c2_utilization_without_dd_explosion import FUTURE_EVAL_COLUMNS, VariantSpec
from ml.phase12c3_position_concentration_guard import Phase12C3PositionConcentrationGuard


REPORT_STEM = "phase12c4_concentration_guard_refinement_2025"


@dataclass(frozen=True)
class Phase12C4Paths:
    markdown: Path
    json: Path


C4_VARIANTS = [
    VariantSpec("C4_0_baseline_downside_squared", "downside_squared"),
    VariantSpec("C4_1_cap_40_redistribute", "cap_40_redistribute"),
    VariantSpec("C4_2_cap_35_redistribute", "cap_35_redistribute"),
    VariantSpec("C4_3_cap_30_redistribute", "cap_30_redistribute"),
    VariantSpec("C4_4_dynamic_cap_by_score_gap", "dynamic_cap_by_score_gap"),
    VariantSpec("C4_5_staged_buy_half_first", "staged_50"),
    VariantSpec("C4_6_staged_buy_70pct_first", "staged_70"),
]


class Phase12C4ConcentrationGuardRefinement(Phase12C3PositionConcentrationGuard):
    def run(self) -> Phase12C4Paths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data = self.load_frame()
        leakage = self.leakage_checklist()
        if leakage["blocking_issues"]:
            return {
                "metadata": self.metadata(),
                "conditions": self.conditions(),
                "dataset_summary": self.dataset_summary(data),
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], leakage),
            }

        variant_results = []
        concentration_rows = []
        for spec in C4_VARIANTS:
            trades, daily, snapshots = self.simulate(data, spec)
            metrics = self.metrics(spec.name, trades, daily)
            concentration = self.concentration_audit(snapshots)
            metrics.update(self.concentration_metrics(concentration))
            variant_results.append(metrics)
            concentration_rows.append({"variant": spec.name, **self.concentration_metrics(concentration)})

        comparison = self.variant_comparison(variant_results)
        return {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "variant_results": variant_results,
            "concentration_audit": concentration_rows,
            "variant_comparison": comparison,
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(variant_results, leakage),
        }

    def target_amounts(self, selected: pd.DataFrame, spec: VariantSpec, cash: float, positions: list[dict[str, Any]]) -> dict[Any, float]:
        if selected.empty:
            return {}
        available_budget = min(cash, self.options.daily_buy_budget)
        weights = self.downside_squared_weights(selected)
        total = float(weights.sum())
        if total <= 0:
            return {index: 0.0 for index in selected.index}
        normalized = weights / total
        amounts = {index: available_budget * float(weight) for index, weight in normalized.items()}

        if spec.allocation_mode == "cap_40_redistribute":
            amounts = self.redistribute_with_cap(amounts, available_budget, cap_rate=0.40)
        elif spec.allocation_mode == "cap_35_redistribute":
            amounts = self.redistribute_with_cap(amounts, available_budget, cap_rate=0.35)
        elif spec.allocation_mode == "cap_30_redistribute":
            amounts = self.redistribute_with_cap(amounts, available_budget, cap_rate=0.30)
        elif spec.allocation_mode == "dynamic_cap_by_score_gap":
            cap = self.dynamic_cap_rate(weights)
            amounts = self.redistribute_with_cap(amounts, available_budget, cap_rate=cap)
        elif spec.allocation_mode == "staged_50":
            amounts = {index: amount * 0.50 for index, amount in amounts.items()}
        elif spec.allocation_mode == "staged_70":
            amounts = {index: amount * 0.70 for index, amount in amounts.items()}
        return amounts

    def downside_squared_weights(self, selected: pd.DataFrame) -> pd.Series:
        weights = _numeric(selected["a3_3_allocation_weight"]).clip(lower=0)
        downside_penalty = (1.0 - _numeric(selected["downside_bad_proba"]).fillna(1.0)).clip(lower=0, upper=1) ** 2
        return weights * downside_penalty

    def redistribute_with_cap(self, amounts: dict[Any, float], target_total: float, *, cap_rate: float) -> dict[Any, float]:
        cap = self.options.initial_cash * cap_rate
        capped = {index: min(amount, cap) for index, amount in amounts.items()}
        for _ in range(8):
            remaining = target_total - sum(capped.values())
            if remaining <= 1e-6:
                break
            room = {index: max(0.0, cap - amount) for index, amount in capped.items()}
            room_total = sum(room.values())
            if room_total <= 1e-6:
                break
            for index, available_room in room.items():
                if available_room <= 0:
                    continue
                add = min(available_room, remaining * available_room / room_total)
                capped[index] += add
        return capped

    def dynamic_cap_rate(self, weights: pd.Series) -> float:
        ordered = weights.sort_values(ascending=False).reset_index(drop=True)
        if len(ordered) < 2:
            return 0.50
        gap = float(ordered.iloc[0] - ordered.iloc[1])
        if gap >= 0.20:
            return 0.50
        if gap >= 0.10:
            return 0.45
        return 0.35

    def variant_comparison(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        base = next((row for row in rows if row["strategy"] == "C4_0_baseline_downside_squared"), rows[0] if rows else {})
        minimum = [row["strategy"] for row in rows if self.minimum_passed(row)]
        ideal = [row["strategy"] for row in rows if self.ideal_passed(row)]
        best = self.best_variant(rows)
        return {
            "best_variant": best.get("strategy") if best else None,
            "best_variant_reason": self.best_variant_reason(best),
            "cap_redistribution_effect": self.group_effect(rows, base, ["C4_1", "C4_2", "C4_3"]),
            "dynamic_cap_effect": self.group_effect(rows, base, ["C4_4"]),
            "staged_buy_effect": self.group_effect(rows, base, ["C4_5", "C4_6"]),
            "utilization_vs_dd_summary": self.utilization_vs_dd_summary(rows),
            "variants_meeting_minimum_target": minimum,
            "variants_meeting_ideal_target": ideal,
            "ready_for_phase13": bool(minimum),
            "recommended_next_phase": "Phase13 limited OOS/year robustness check" if minimum else "Phase12-C5 portfolio-level risk gate",
        }

    def group_effect(self, rows: list[dict[str, Any]], base: dict[str, Any], prefixes: list[str]) -> dict[str, Any]:
        subset = [row for row in rows if any(row["strategy"].startswith(prefix) for prefix in prefixes)]
        if not subset:
            return {}
        best = self.best_variant(subset)
        return {
            "best_variant": best.get("strategy") if best else None,
            "net_profit_delta": _safe_float((_safe_float(best.get("net_profit")) or 0.0) - (_safe_float(base.get("net_profit")) or 0.0)) if best else None,
            "PF_delta": _safe_float((_safe_float(best.get("PF")) or 0.0) - (_safe_float(base.get("PF")) or 0.0)) if best else None,
            "DD_delta": _safe_float((_safe_float(best.get("DD")) or 0.0) - (_safe_float(base.get("DD")) or 0.0)) if best else None,
            "utilization_delta": _safe_float((_safe_float(best.get("capital_utilization")) or 0.0) - (_safe_float(base.get("capital_utilization")) or 0.0)) if best else None,
            "largest_position_weight_max_delta": _safe_float((_safe_float(best.get("largest_position_weight_max")) or 0.0) - (_safe_float(base.get("largest_position_weight_max")) or 0.0)) if best else None,
        }

    def recommendation(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"ready_for_phase13": False, "recommended_next_phase": "Fix Phase12-C4 leakage blockers"}
        comparison = self.variant_comparison(rows)
        return {
            "best_variant": comparison["best_variant"],
            "best_variant_reason": comparison["best_variant_reason"],
            "cap_redistribution_effect": comparison["cap_redistribution_effect"],
            "dynamic_cap_effect": comparison["dynamic_cap_effect"],
            "staged_buy_effect": comparison["staged_buy_effect"],
            "utilization_vs_dd_summary": comparison["utilization_vs_dd_summary"],
            "variants_meeting_minimum_target": comparison["variants_meeting_minimum_target"],
            "variants_meeting_ideal_target": comparison["variants_meeting_ideal_target"],
            "ready_for_phase13": comparison["ready_for_phase13"],
            "recommended_next_phase": comparison["recommended_next_phase"],
        }

    def conditions(self) -> dict[str, Any]:
        conditions = super().conditions()
        conditions.update(
            {
                "base_strategy": "C2c_normalized_downside_penalty_squared",
                "redistribute_to_uncapped": True,
                "staged_buy_mode": "proxy: same-day target amount scaled to 50% or 70%; no next-day add-on implemented",
                "variants": [spec.__dict__ for spec in C4_VARIANTS],
            }
        )
        return conditions

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update({"phase": "12-C4", "scope": "2025 concentration guard refinement only"})
        return metadata

    def save_outputs(self, report: dict[str, Any]) -> Phase12C4Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12C4Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-C4 Concentration Guard Refinement",
            "",
            "## Variant Results",
            "",
            self.table(
                report.get("variant_results", []),
                ["strategy", "net_profit", "PF", "DD", "win_rate", "final_assets", "capital_utilization", "average_holding_days", "cost_paid", "largest_position_weight_p90", "largest_position_weight_max", "top2_weight_mean", "top3_weight_mean"],
            ),
            "",
            "## Comparison",
            "",
            self.table(
                [report.get("variant_comparison", {})],
                ["best_variant", "best_variant_reason", "cap_redistribution_effect", "dynamic_cap_effect", "staged_buy_effect", "utilization_vs_dd_summary", "variants_meeting_minimum_target", "variants_meeting_ideal_target", "ready_for_phase13", "recommended_next_phase"],
            ),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_only_for_evaluation", "future_columns_used_as_features", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["best_variant", "cap_redistribution_effect", "dynamic_cap_effect", "staged_buy_effect", "utilization_vs_dd_summary", "variants_meeting_minimum_target", "variants_meeting_ideal_target", "ready_for_phase13", "recommended_next_phase"]),
            "",
        ]
        return "\n".join(lines)

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
