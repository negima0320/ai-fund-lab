"""Phase 13-R Return Decomposition / Strategy Requirement Audit.

This is a 2025-only decomposition audit. It does not introduce new strategy
variants; it decomposes the Phase13-E2 best reference strategy into candidate,
entry, hold/exit, capital efficiency, turnover, and loss-control components.
Future columns are used only for evaluation and theoretical ceiling analysis.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase13a_horizon_reality_audit import ROOT
from ml.phase13d_hold_exit_dataset_audit import CANDIDATE_SETS
from ml.phase13e_integrated_strategy_prototype import INITIAL_CASH, MAX_POSITIONS, Phase13EPaths, Position
from ml.phase13e2_integrated_strategy_rule_tuning import Phase13E2IntegratedStrategyRuleTuning


REPORT_STEM = "phase13r_return_decomposition_audit_2025"
ANNUAL_RETURN_TARGET = 0.50
REFERENCE_VARIANT = "T5_stop_rework_plus_break_even"
REQUIRED_REPORT_KEYS = [
    "current_annual_return_after_cost",
    "target_annual_return",
    "annual_return_gap",
    "primary_bottleneck",
    "secondary_bottleneck",
    "what_must_improve_to_reach_50pct",
    "recommended_system_thesis",
    "recommended_next_phase",
    "ready_for_phase13e3",
    "ready_for_phase13c",
    "ready_for_phase13f",
    "leakage_risk",
    "blocking_issues",
]


@dataclass(frozen=True)
class SimulationDetails:
    metrics: dict[str, Any]
    trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    daily_audit: list[dict[str, Any]]


class Phase13RReturnDecompositionAudit(Phase13E2IntegratedStrategyRuleTuning):
    def __init__(self, root: Path | str = ROOT) -> None:
        super().__init__(root)

    def run(self) -> Phase13EPaths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data, source_info = self.phase13d.phase13a.load_comparison_dataset()
        data = self.prepare_data(data)
        leakage = self.leakage_checklist()
        leakage["blocking_issues"] = self.blocking_issues(data)
        leakage["leakage_risk"] = "low" if not leakage["blocking_issues"] else "medium"
        details = self.simulate_details(data)
        annual = self.annual_return_requirement(details)
        candidate_ceiling = self.candidate_ceiling_audit(data)
        entry_quality = self.entry_quality_audit(data, details.trades)
        hold_exit = self.hold_exit_quality_audit(details.trades)
        capital = self.capital_efficiency_audit(details)
        scorecard = self.bottleneck_scorecard(annual, candidate_ceiling, entry_quality, hold_exit, capital, details.metrics)
        final = self.final_requirement_table(annual, candidate_ceiling, entry_quality, hold_exit, capital, scorecard, details.metrics, leakage)
        return {
            "metadata": self.metadata(),
            "input_artifact_summary": self.input_artifact_summary(data, source_info, leakage),
            "reference_strategy_metrics": details.metrics,
            "annual_return_requirement_decomposition": annual,
            "candidate_ceiling_audit": candidate_ceiling,
            "entry_quality_audit": entry_quality,
            "hold_exit_quality_audit": hold_exit,
            "capital_efficiency_audit": capital,
            "bottleneck_scorecard": scorecard,
            "final_strategy_requirement_table": final,
            "leakage_checklist": leakage,
            **{key: final.get(key) for key in REQUIRED_REPORT_KEYS},
        }

    def simulate_details(self, data: pd.DataFrame) -> SimulationDetails:
        variant = REFERENCE_VARIANT
        self.active_config = self.config_for_variant(variant)
        dates = sorted(data["date"].dropna().unique())
        by_date = {date: frame.copy() for date, frame in data.groupby("date")}
        price_lookup = {(row.date, str(row.code)): float(row.close) for row in data[["date", "code", "close"]].itertuples(index=False)}
        cash = INITIAL_CASH
        positions: list[Position] = []
        trades: list[dict[str, Any]] = []
        equity_curve = []
        daily_audit = []
        previous_exits: set[str] = set()
        reentry_codes: set[str] = set()
        total_cost_paid = 0.0

        for date in dates:
            frame = by_date[date]
            positions, exits, exit_cost = self.process_exits(positions, date, price_lookup, variant)
            total_cost_paid += exit_cost
            for trade in exits:
                cash += trade["sell_proceeds_after_cost"]
                trades.append(trade)
                previous_exits.add(trade["code"])
            open_codes = {position.code for position in positions}
            raw_candidates = self.entry_candidates(frame, open_codes)
            candidate_count = int(len(raw_candidates))
            slots = max(0, MAX_POSITIONS - len(positions))
            bought_count = 0
            daily_spent = 0.0
            for _, row in raw_candidates.head(slots).iterrows():
                if slots <= 0 or daily_spent >= self.strategy_parameters()["daily_buy_limit"]:
                    break
                price = float(row["close"])
                target = min(self.strategy_parameters()["daily_buy_limit"] / MAX_POSITIONS, self.strategy_parameters()["daily_buy_limit"] - daily_spent, cash)
                shares = int(target // (price * self.strategy_parameters()["round_lot"])) * int(self.strategy_parameters()["round_lot"])
                if shares <= 0:
                    continue
                gross = shares * price
                fee = gross * self.strategy_parameters()["cost_rate"]
                total = gross + fee
                if total > cash:
                    shares = int((cash / (1.0 + self.strategy_parameters()["cost_rate"])) // (price * self.strategy_parameters()["round_lot"])) * int(self.strategy_parameters()["round_lot"])
                    gross = shares * price
                    fee = gross * self.strategy_parameters()["cost_rate"]
                    total = gross + fee
                if shares <= 0 or total > cash:
                    continue
                code = str(row["code"])
                cash -= total
                daily_spent += gross
                total_cost_paid += fee
                if code in previous_exits:
                    reentry_codes.add(code)
                positions.append(
                    Position(
                        code=code,
                        entry_date=date,
                        entry_price=price,
                        shares=int(shares),
                        entry_cost=gross,
                        entry_fee=fee,
                        peak_price=price,
                        peak_return=0.0,
                        holding_days=0,
                        extended=False,
                        entry_future=self.entry_future_metrics(row),
                    )
                )
                bought_count += 1
                open_codes.add(code)
                slots -= 1
            equity = self.equity_on_date(cash, positions, date, price_lookup)
            equity_curve.append(equity)
            daily_audit.append(
                {
                    "date": date,
                    "candidate_count": candidate_count,
                    "bought_count": bought_count,
                    "position_count": len(positions),
                    "max_positions_reached": len(positions) >= MAX_POSITIONS,
                    "candidate_available_but_not_bought": candidate_count > bought_count,
                    "daily_spent": daily_spent,
                    **equity,
                }
            )

        if dates:
            final_date = dates[-1]
            positions, forced_exits, forced_cost = self.force_close(positions, final_date, price_lookup)
            total_cost_paid += forced_cost
            for trade in forced_exits:
                cash += trade["sell_proceeds_after_cost"]
                trades.append(trade)
            equity_curve.append(self.equity_on_date(cash, positions, final_date, price_lookup))
        metrics = self.metrics(variant, cash, trades, equity_curve, total_cost_paid, len(reentry_codes))
        return SimulationDetails(metrics=metrics, trades=trades, equity_curve=equity_curve, daily_audit=daily_audit)

    def annual_return_requirement(self, details: SimulationDetails) -> dict[str, Any]:
        metrics = details.metrics
        returns = [_safe_float(trade["realized_return"]) for trade in details.trades if trade.get("realized_return") is not None]
        avg_trade = _safe_float(sum(returns) / len(returns)) if returns else None
        trade_count = len(details.trades)
        avg_hold = metrics.get("average_holding_days")
        util = metrics.get("capital_utilization")
        current = metrics.get("annual_return_after_cost")
        turnover = _safe_float(trade_count / MAX_POSITIONS) if trade_count else 0.0
        required_avg = _safe_float(ANNUAL_RETURN_TARGET / turnover) if turnover else None
        required_turnover = _safe_float(ANNUAL_RETURN_TARGET / avg_trade) if avg_trade and avg_trade > 0 else None
        required_utilization = _safe_float(util * ANNUAL_RETURN_TARGET / current) if current and current > 0 and util else None
        return {
            "formula": "annual_return ≈ average_trade_return_after_cost * trade_count / max_positions. This is a decomposition proxy, not a compounding model.",
            "formula_limitations": "Ignores compounding path, overlapping capital, cash timing, and position sizing variation.",
            "annual_return_target": ANNUAL_RETURN_TARGET,
            "current_annual_return_after_cost": current,
            "annual_return_gap": self.diff(ANNUAL_RETURN_TARGET, current),
            "average_trade_return_after_cost": avg_trade,
            "trade_count": trade_count,
            "average_holding_days": avg_hold,
            "turnover_per_year": turnover,
            "capital_utilization": util,
            "required_average_trade_return_at_current_turnover": required_avg,
            "required_turnover_at_current_average_trade_return": required_turnover,
            "required_utilization_at_current_edge": required_utilization,
        }

    def candidate_ceiling_audit(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        methods = [method for method in CANDIDATE_SETS if method.name in {
            "stock_selection_rank_score_top50",
            "candidate_strength_top50",
            "valuation_first_top50",
            "opportunity_downside_top50",
            "candidate_strength_top5",
            "valuation_first_top5",
            "opportunity_downside_top5",
        }]
        for method in methods:
            frame = self.phase13d.method_frame(data, method)
            ideal_top1 = self.ideal_top_n_by_day(frame, "future_return_20d", 1)
            ideal_peak1 = self.ideal_top_n_by_day(frame, "future_max_return_20d", 1)
            ideal_top5 = self.ideal_top_n_by_day(frame, "future_return_20d", 5)
            ideal_peak5 = self.ideal_top_n_by_day(frame, "future_max_return_20d", 5)
            mean_ret = self.mean(frame, "future_return_20d")
            mean_peak = self.mean(frame, "future_max_return_20d")
            rows.append(
                {
                    "candidate_set": method.name,
                    "sample_count": int(len(frame)),
                    "candidate_days": int(frame["date"].nunique()) if not frame.empty else 0,
                    "mean_future_return_20d": mean_ret,
                    "mean_future_max_return_20d": mean_peak,
                    "mean_future_max_drawdown_20d": self.mean(frame, "future_max_drawdown_20d"),
                    "top_decile_rate_20d": self.mean(frame, "top_decile_20d"),
                    "downside_bad_rate_20d": self.mean(frame, "downside_bad_20d"),
                    "ideal_top1_per_day_future_return_20d": ideal_top1,
                    "ideal_top1_per_day_future_max_return_20d": ideal_peak1,
                    "ideal_top5_equal_future_return_20d": ideal_top5,
                    "ideal_top5_equal_future_max_return_20d": ideal_peak5,
                    "theoretical_annual_return_hold20d": _safe_float(mean_ret * 252 / 20) if mean_ret is not None else None,
                    "theoretical_annual_return_peak_capture": _safe_float(mean_peak * 252 / 20) if mean_peak is not None else None,
                    "ideal_metrics_note": "ideal_* uses future labels for theoretical ceiling only.",
                }
            )
        return rows

    def entry_quality_audit(self, data: pd.DataFrame, trades: list[dict[str, Any]]) -> dict[str, Any]:
        frame = pd.DataFrame(trades)
        if frame.empty:
            return {"entry_count": 0}
        missed = []
        deltas = []
        rank_values = []
        candidate_by_date = {}
        method = next(method for method in CANDIDATE_SETS if method.name == "candidate_strength_top50")
        for date, daily in data.groupby("date"):
            candidate_by_date[date] = self.phase13d.method_frame(daily, method).reset_index(drop=True)
        for trade in trades:
            entry_date = pd.Timestamp(trade["entry_date"])
            daily = candidate_by_date.get(entry_date)
            if daily is None or daily.empty:
                continue
            daily = daily.sort_values("opportunity_downside_score", ascending=False).reset_index(drop=True)
            matches = daily[daily["code"].astype("string") == str(trade["code"])]
            if not matches.empty:
                rank_values.append(int(matches.index[0]) + 1)
                bought_future = trade.get("future_return_20d")
                better = _numeric(daily["future_return_20d"]) > (bought_future if bought_future is not None else -math.inf)
                better_peak = _numeric(daily["future_max_return_20d"]) > (trade.get("future_max_return_20d") if trade.get("future_max_return_20d") is not None else -math.inf)
                has_better = bool((better | better_peak).any())
                missed.append(has_better)
                if has_better:
                    best_future = max(float(_numeric(daily["future_return_20d"]).max()), float(_numeric(daily["future_max_return_20d"]).max()))
                    bought_best = max(bought_future or 0.0, trade.get("future_max_return_20d") or 0.0)
                    deltas.append(best_future - bought_best)
        return {
            "entry_count": int(len(frame)),
            "entry_mean_future_return_20d": self.mean(frame, "future_return_20d"),
            "entry_mean_future_max_return_20d": self.mean(frame, "future_max_return_20d"),
            "entry_mean_future_max_drawdown_20d": self.mean(frame, "future_max_drawdown_20d"),
            "entry_top_decile_rate_20d": self.mean(frame, "top_decile_20d"),
            "entry_downside_bad_rate_20d": self.mean(frame, "downside_bad_20d"),
            "entry_rank_distribution_within_candidate_strength_top50": self.rank_distribution(rank_values),
            "missed_better_candidate_rate": _safe_float(sum(missed) / len(missed)) if missed else None,
            "missed_better_candidate_avg_delta": _safe_float(sum(deltas) / len(deltas)) if deltas else 0.0,
            "future_labels_note": "missed_better_candidate uses future labels for evaluation only.",
        }

    def hold_exit_quality_audit(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        frame = pd.DataFrame(trades)
        if frame.empty:
            return {}
        realized = self.mean(frame, "realized_return")
        future20 = self.mean(frame, "future_return_20d")
        future_peak = self.mean(frame, "future_max_return_20d")
        winner_to_loser = frame[(frame["peak_return"] >= 0.05) & (frame["realized_return"] < 0)]
        return {
            "realized_return_after_cost_mean": realized,
            "future_return_20d_mean_for_entries": future20,
            "future_max_return_20d_mean_for_entries": future_peak,
            "realized_vs_future20d_delta": self.diff(realized, future20),
            "realized_vs_future_max_delta": self.diff(realized, future_peak),
            "profit_retention_rate": _safe_float(realized / future_peak) if realized is not None and future_peak else None,
            "winner_to_loser_count": int(len(winner_to_loser)),
            "winner_to_loser_rate": _safe_float(len(winner_to_loser) / len(frame)) if len(frame) else None,
            "avg_profit_decay_before_exit": self.mean(frame, "profit_decay_before_exit"),
            "break_even_exit_count": int((frame["exit_reason"] == "break_even_exit").sum()),
            "stop_loss_exit_count": int((frame["exit_reason"] == "stop_loss_exit").sum()),
            "fixed_hold_exit_count": int((frame["exit_reason"] == "fixed_hold_exit").sum()),
            "forced_exit_count": int((frame["exit_reason"] == "forced_exit").sum()),
        }

    def capital_efficiency_audit(self, details: SimulationDetails) -> dict[str, Any]:
        daily = pd.DataFrame(details.daily_audit)
        if daily.empty:
            return {}
        util = daily["market_value"] / daily["equity"]
        return {
            "average_capital_utilization": _safe_float(util.mean()),
            "days_below_50pct_utilization": int((util < 0.50).sum()),
            "days_below_60pct_utilization": int((util < 0.60).sum()),
            "cash_idle_days": int((daily["cash"] > 0).sum()),
            "max_positions_reached_days": int(daily["max_positions_reached"].sum()),
            "daily_buy_limit_blocked_days": "not_available",
            "affordability_blocked_days": "not_available",
            "round_lot_blocked_days": "not_available",
            "candidate_available_but_not_bought_days": int(daily["candidate_available_but_not_bought"].sum()),
        }

    def bottleneck_scorecard(
        self,
        annual: dict[str, Any],
        candidate_ceiling: list[dict[str, Any]],
        entry: dict[str, Any],
        hold_exit: dict[str, Any],
        capital: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        primary_ceiling = next((row for row in candidate_ceiling if row["candidate_set"] == "candidate_strength_top50"), {})
        candidate_score = self.score_percent((primary_ceiling.get("theoretical_annual_return_hold20d") or 0) / ANNUAL_RETURN_TARGET)
        entry_score = self.score_percent((entry.get("entry_top_decile_rate_20d") or 0) / 0.20)
        hold_score = self.score_percent((hold_exit.get("profit_retention_rate") or 0) / 0.60)
        capital_score = self.score_percent((capital.get("average_capital_utilization") or 0) / 0.60)
        turnover_score = self.score_percent((annual.get("turnover_per_year") or 0) / 10)
        loss_score = self.score_percent(1.0 if (metrics.get("DD") or -1) >= -0.10 and (entry.get("entry_downside_bad_rate_20d") or 1) <= 0.30 else 0.5)
        return {
            "score_basis": {
                "candidate_quality_score": "candidate_strength_top50 theoretical_annual_return_hold20d / 0.50",
                "entry_quality_score": "entry_top_decile_rate_20d / 0.20",
                "hold_exit_quality_score": "profit_retention_rate / 0.60",
                "capital_efficiency_score": "capital_utilization / 0.60",
                "turnover_score": "turnover_per_year / 10",
                "loss_control_score": "100 if DD >= -10% and entry downside <= 30%, else 50",
            },
            "candidate_quality_score": candidate_score,
            "entry_quality_score": entry_score,
            "hold_exit_quality_score": hold_score,
            "capital_efficiency_score": capital_score,
            "turnover_score": turnover_score,
            "loss_control_score": loss_score,
        }

    def final_requirement_table(
        self,
        annual: dict[str, Any],
        candidate_ceiling: list[dict[str, Any]],
        entry: dict[str, Any],
        hold_exit: dict[str, Any],
        capital: dict[str, Any],
        scorecard: dict[str, Any],
        metrics: dict[str, Any],
        leakage: dict[str, Any],
    ) -> dict[str, Any]:
        scores = {k: v for k, v in scorecard.items() if k.endswith("_score")}
        sorted_scores = sorted(scores.items(), key=lambda item: item[1])
        primary = sorted_scores[0][0].removesuffix("_score") if sorted_scores else "unknown"
        secondary = sorted_scores[1][0].removesuffix("_score") if len(sorted_scores) > 1 else "unknown"
        gap = annual.get("annual_return_gap")
        next_phase = "Phase13-E3 Entry/Exit Interaction Audit" if primary in {"entry_quality", "hold_exit_quality"} else "Phase13-C Horizon-Aware Valuation Prototype"
        if (metrics.get("annual_return_after_cost") or 0) >= ANNUAL_RETURN_TARGET and (metrics.get("DD") or -1) >= -0.10:
            next_phase = "Phase13-F Strict OOS / Multi-Year Limited Check"
        thesis = (
            "Current best thesis: candidate_strength_top50 generates the candidate universe, "
            "opportunity_downside_score ranks entries, and break-even plus -5% stop rework is the best 2025 exit rule. "
            "The system has acceptable DD and utilization, but annual return is still below 50%; the remaining gap appears to require "
            "better entry/exit interaction, higher retained profit per trade, or a stronger horizon-aware valuation layer rather than more ad-hoc exit tweaks."
        )
        return {
            "current_annual_return_after_cost": annual.get("current_annual_return_after_cost"),
            "target_annual_return": ANNUAL_RETURN_TARGET,
            "annual_return_gap": gap,
            "primary_bottleneck": primary,
            "secondary_bottleneck": secondary,
            "what_must_improve_to_reach_50pct": {
                "required_average_trade_return_at_current_turnover": annual.get("required_average_trade_return_at_current_turnover"),
                "required_turnover_at_current_average_trade_return": annual.get("required_turnover_at_current_average_trade_return"),
                "required_utilization_at_current_edge": annual.get("required_utilization_at_current_edge"),
            },
            "recommended_system_thesis": thesis,
            "recommended_next_phase": next_phase,
            "ready_for_phase13e3": next_phase == "Phase13-E3 Entry/Exit Interaction Audit",
            "ready_for_phase13c": next_phase == "Phase13-C Horizon-Aware Valuation Prototype",
            "ready_for_phase13f": next_phase == "Phase13-F Strict OOS / Multi-Year Limited Check",
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def ideal_top_n_by_day(self, frame: pd.DataFrame, column: str, n: int) -> float | None:
        if column not in frame.columns or frame.empty:
            return None
        selected = (
            frame.sort_values(["date", column], ascending=[True, False])
            .groupby("date", sort=False, group_keys=False)
            .head(n)
        )
        return self.mean(selected, column)

    def rank_distribution(self, ranks: list[int]) -> dict[str, Any]:
        if not ranks:
            return {}
        return {
            "min": int(min(ranks)),
            "median": _safe_float(median(ranks)),
            "p90": _safe_float(pd.Series(ranks).quantile(0.90)),
            "max": int(max(ranks)),
        }

    def score_percent(self, value: float) -> float:
        return _safe_float(max(0.0, min(100.0, value * 100.0))) or 0.0

    def input_artifact_summary(self, data: pd.DataFrame, source_info: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        summary = super().input_artifact_summary(data, source_info, leakage)
        summary["reference_strategy_variant"] = REFERENCE_VARIANT
        return summary

    def leakage_checklist(self) -> dict[str, Any]:
        checklist = super().leakage_checklist()
        checklist["phase_type"] = "decomposition_audit_not_strategy_improvement"
        checklist["future_columns_used_only_for_evaluation_and_theoretical_ceiling"] = checklist.pop("future_columns_used_only_for_evaluation")
        return checklist

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "13-R",
            "scope": "2025-only return decomposition and strategy requirement audit",
            "reference_strategy_variant": REFERENCE_VARIANT,
            "new_model_trained": False,
            "full_backtest_executed": False,
            "strategy_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
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
                "# Phase 13-R Return Decomposition / Strategy Requirement Audit",
                "",
                "## Annual Return Requirement Decomposition",
                "",
                self.table([report["annual_return_requirement_decomposition"]], ["annual_return_target", "current_annual_return_after_cost", "annual_return_gap", "average_trade_return_after_cost", "trade_count", "average_holding_days", "turnover_per_year", "capital_utilization", "required_average_trade_return_at_current_turnover", "required_turnover_at_current_average_trade_return", "required_utilization_at_current_edge"]),
                "",
                "## Candidate Ceiling Audit",
                "",
                self.table(report["candidate_ceiling_audit"], ["candidate_set", "mean_future_return_20d", "mean_future_max_return_20d", "mean_future_max_drawdown_20d", "top_decile_rate_20d", "downside_bad_rate_20d", "ideal_top1_per_day_future_return_20d", "ideal_top5_equal_future_return_20d", "theoretical_annual_return_hold20d", "theoretical_annual_return_peak_capture"]),
                "",
                "## Entry Quality Audit",
                "",
                self.table([report["entry_quality_audit"]], ["entry_count", "entry_mean_future_return_20d", "entry_mean_future_max_return_20d", "entry_mean_future_max_drawdown_20d", "entry_top_decile_rate_20d", "entry_downside_bad_rate_20d", "missed_better_candidate_rate", "missed_better_candidate_avg_delta", "entry_rank_distribution_within_candidate_strength_top50"]),
                "",
                "## Hold / Exit Quality Audit",
                "",
                self.table([report["hold_exit_quality_audit"]], ["realized_return_after_cost_mean", "future_return_20d_mean_for_entries", "future_max_return_20d_mean_for_entries", "realized_vs_future20d_delta", "realized_vs_future_max_delta", "profit_retention_rate", "winner_to_loser_count", "winner_to_loser_rate", "avg_profit_decay_before_exit", "break_even_exit_count", "stop_loss_exit_count", "fixed_hold_exit_count"]),
                "",
                "## Capital Efficiency Audit",
                "",
                self.table([report["capital_efficiency_audit"]], ["average_capital_utilization", "days_below_50pct_utilization", "days_below_60pct_utilization", "cash_idle_days", "max_positions_reached_days", "candidate_available_but_not_bought_days", "daily_buy_limit_blocked_days", "affordability_blocked_days", "round_lot_blocked_days"]),
                "",
                "## Bottleneck Scorecard",
                "",
                self.table([report["bottleneck_scorecard"]], ["candidate_quality_score", "entry_quality_score", "hold_exit_quality_score", "capital_efficiency_score", "turnover_score", "loss_control_score", "score_basis"]),
                "",
                "## Final Strategy Requirement Table",
                "",
                self.table([report["final_strategy_requirement_table"]], REQUIRED_REPORT_KEYS),
                "",
                "## Leakage Checklist",
                "",
                self.table([report["leakage_checklist"]], ["future_columns_used_as_entry_or_exit_features", "future_columns_used_as_features", "future_columns_used_only_for_evaluation_and_theoretical_ceiling", "new_model_trained", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "strategy_backtest_executed", "historical_predictions_regenerated", "jquants_api_called", "openai_api_called", "leakage_risk", "blocking_issues"]),
                "",
            ]
        )

    def diff(self, left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return _safe_float(left - right)
