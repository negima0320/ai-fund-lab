"""Phase 12-C3 position concentration guard.

This is a 2025-only lightweight strategy check. It keeps the Phase 12-C2 best
normalized downside-squared allocation and B5_2 recalibrated exit, then adds a
small set of direct position concentration guards. It does not run a full
backtest, modify profiles, overwrite models, or regenerate historical
predictions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase12c2_utilization_without_dd_explosion import (
    FUTURE_EVAL_COLUMNS,
    ROOT,
    Phase12C2UtilizationWithoutDDExplosion,
    VariantSpec,
)


REPORT_STEM = "phase12c3_position_concentration_guard_2025"


@dataclass(frozen=True)
class Phase12C3Paths:
    markdown: Path
    json: Path


C3_VARIANTS = [
    VariantSpec("C3_0_baseline_downside_squared", "downside_squared"),
    VariantSpec("C3_1_per_name_cap_40pct", "per_name_cap_40pct"),
    VariantSpec("C3_2_per_name_cap_30pct", "per_name_cap_30pct"),
    VariantSpec("C3_3_per_name_cap_25pct", "per_name_cap_25pct"),
    VariantSpec("C3_4_top2_cap_60pct", "top2_cap_60pct"),
    VariantSpec("C3_5_per_name_30pct_and_top2_60pct", "per_name_30pct_top2_60pct"),
    VariantSpec("C3_6_concentration_scaled", "concentration_scaled"),
]


class Phase12C3PositionConcentrationGuard(Phase12C2UtilizationWithoutDDExplosion):
    def run(self) -> Phase12C3Paths:
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
        for spec in C3_VARIANTS:
            trades, daily, snapshots = self.simulate(data, spec)
            metrics = self.metrics(spec.name, trades, daily)
            concentration = self.concentration_audit(snapshots)
            metrics.update(self.concentration_metrics(concentration))
            variant_results.append(metrics)
            concentration_rows.append({"variant": spec.name, **self.concentration_metrics(concentration)})

        comparison = self.variant_comparison(variant_results)
        concentration_summary = self.concentration_reduction_summary(variant_results)
        return {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "variant_results": variant_results,
            "concentration_audit": concentration_rows,
            "variant_comparison": comparison,
            "concentration_reduction_summary": concentration_summary,
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(variant_results, leakage),
        }

    def target_amounts(self, selected: pd.DataFrame, spec: VariantSpec, cash: float, positions: list[dict[str, Any]]) -> dict[Any, float]:
        if selected.empty:
            return {}
        available_budget = min(cash, self.options.daily_buy_budget)
        weights = _numeric(selected["a3_3_allocation_weight"]).clip(lower=0)
        downside_penalty = (1.0 - _numeric(selected["downside_bad_proba"]).fillna(1.0)).clip(lower=0, upper=1) ** 2
        weights = weights * downside_penalty
        total = float(weights.sum())
        if total <= 0:
            return {index: 0.0 for index in selected.index}
        normalized = weights / total

        if spec.allocation_mode == "concentration_scaled":
            largest = float(normalized.max()) if not normalized.empty else 0.0
            if largest > 0.65:
                available_budget *= 0.50
            elif largest > 0.50:
                available_budget *= 0.70

        amounts = {index: available_budget * float(weight) for index, weight in normalized.items()}
        if spec.allocation_mode == "per_name_cap_40pct":
            amounts = self.apply_per_name_cap(amounts, 0.40)
        elif spec.allocation_mode == "per_name_cap_30pct":
            amounts = self.apply_per_name_cap(amounts, 0.30)
        elif spec.allocation_mode == "per_name_cap_25pct":
            amounts = self.apply_per_name_cap(amounts, 0.25)
        elif spec.allocation_mode == "top2_cap_60pct":
            amounts = self.apply_top_n_cap(amounts, n=2, cap_rate=0.60)
        elif spec.allocation_mode == "per_name_30pct_top2_60pct":
            amounts = self.apply_per_name_cap(amounts, 0.30)
            amounts = self.apply_top_n_cap(amounts, n=2, cap_rate=0.60)
        return amounts

    def apply_per_name_cap(self, amounts: dict[Any, float], cap_rate: float) -> dict[Any, float]:
        cap = self.options.initial_cash * cap_rate
        return {index: min(amount, cap) for index, amount in amounts.items()}

    def apply_top_n_cap(self, amounts: dict[Any, float], *, n: int, cap_rate: float) -> dict[Any, float]:
        cap = self.options.initial_cash * cap_rate
        ordered = sorted(amounts.items(), key=lambda item: item[1], reverse=True)
        top = ordered[:n]
        top_sum = sum(amount for _, amount in top)
        if top_sum <= cap or top_sum <= 0:
            return amounts
        scale = cap / top_sum
        capped = dict(amounts)
        for index, amount in top:
            capped[index] = amount * scale
        return capped

    def concentration_metrics(self, concentration: dict[str, Any]) -> dict[str, Any]:
        return {
            "largest_position_weight_mean": concentration.get("largest_position_weight_mean"),
            "largest_position_weight_p90": concentration.get("largest_position_weight_p90"),
            "largest_position_weight_max": concentration.get("largest_position_weight_max"),
            "top2_weight_mean": concentration.get("top2_weight_mean"),
            "top3_weight_mean": concentration.get("top3_weight_mean"),
        }

    def variant_comparison(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        minimum = [row["strategy"] for row in rows if self.minimum_passed(row)]
        ideal = [row["strategy"] for row in rows if self.ideal_passed(row)]
        best = self.best_variant(rows)
        return {
            "best_variant": best.get("strategy") if best else None,
            "best_variant_reason": self.best_variant_reason(best),
            "concentration_reduction_summary": self.concentration_reduction_summary(rows),
            "utilization_vs_dd_summary": self.utilization_vs_dd_summary(rows),
            "variants_meeting_minimum_target": minimum,
            "variants_meeting_ideal_target": ideal,
            "ready_for_phase13": bool(minimum),
            "recommended_next_phase": "Phase13 limited OOS/year robustness check" if minimum else "Phase12-C4 concentration guard refinement",
        }

    def best_variant(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None

        def key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
            pass_bonus = 10**7 if self.minimum_passed(row) else 0.0
            pf = _safe_float(row.get("PF")) or 0.0
            dd = _safe_float(row.get("DD")) or -1.0
            utilization = _safe_float(row.get("capital_utilization")) or 0.0
            profit = _safe_float(row.get("net_profit")) or -10**18
            largest = _safe_float(row.get("largest_position_weight_max")) or 1.0
            return (pass_bonus, dd, pf, utilization, profit - largest * 10_000)

        return max(rows, key=key)

    def concentration_reduction_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        base = next((row for row in rows if row["strategy"] == "C3_0_baseline_downside_squared"), rows[0])
        best_concentration = min(rows, key=lambda row: _safe_float(row.get("largest_position_weight_max")) or 1.0)
        return {
            "baseline_largest_position_weight_max": base.get("largest_position_weight_max"),
            "baseline_top2_weight_mean": base.get("top2_weight_mean"),
            "best_concentration_variant": best_concentration.get("strategy"),
            "best_largest_position_weight_max": best_concentration.get("largest_position_weight_max"),
            "best_top2_weight_mean": best_concentration.get("top2_weight_mean"),
        }

    def utilization_vs_dd_summary(self, rows: list[dict[str, Any]]) -> str:
        passing_util = [row for row in rows if (_safe_float(row.get("capital_utilization")) or 0.0) >= 0.50]
        dd_safe = [row for row in passing_util if (_safe_float(row.get("DD")) or -1.0) >= -0.12]
        if dd_safe:
            names = ", ".join(row["strategy"] for row in dd_safe)
            return f"Variants maintaining utilization >= 0.50 and DD >= -12%: {names}."
        return "No concentration guard maintained utilization >= 0.50 while reducing DD to the -12% minimum line."

    def recommendation(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"ready_for_phase13": False, "recommended_next_phase": "Fix Phase12-C3 leakage blockers"}
        comparison = self.variant_comparison(rows)
        return {
            "best_variant": comparison["best_variant"],
            "best_variant_reason": comparison["best_variant_reason"],
            "concentration_reduction_summary": comparison["concentration_reduction_summary"],
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
                "redistribute_to_uncapped": False,
                "variants": [spec.__dict__ for spec in C3_VARIANTS],
            }
        )
        return conditions

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update({"phase": "12-C3", "scope": "2025 position concentration guard check only"})
        return metadata

    def save_outputs(self, report: dict[str, Any]) -> Phase12C3Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12C3Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-C3 Position Concentration Guard",
            "",
            "## Variant Results",
            "",
            self.table(
                report.get("variant_results", []),
                [
                    "strategy",
                    "net_profit",
                    "PF",
                    "DD",
                    "win_rate",
                    "final_assets",
                    "capital_utilization",
                    "average_holding_days",
                    "cost_paid",
                    "largest_position_weight_mean",
                    "largest_position_weight_p90",
                    "largest_position_weight_max",
                    "top2_weight_mean",
                    "top3_weight_mean",
                ],
            ),
            "",
            "## Comparison",
            "",
            self.table(
                [report.get("variant_comparison", {})],
                [
                    "best_variant",
                    "best_variant_reason",
                    "concentration_reduction_summary",
                    "utilization_vs_dd_summary",
                    "variants_meeting_minimum_target",
                    "variants_meeting_ideal_target",
                    "ready_for_phase13",
                    "recommended_next_phase",
                ],
            ),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_only_for_evaluation", "future_columns_used_as_features", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["best_variant", "concentration_reduction_summary", "utilization_vs_dd_summary", "variants_meeting_minimum_target", "variants_meeting_ideal_target", "ready_for_phase13", "recommended_next_phase"]),
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
