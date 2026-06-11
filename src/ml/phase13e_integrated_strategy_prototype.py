"""Phase 13-E Integrated Strategy Prototype.

This is a 2025-only lightweight strategy prototype. Entry and exit decisions
use only prediction-time scores, current/entry prices, holding days, cash, and
position state. Future columns are used only for post-run BUY quality
evaluation.
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
from ml.phase13d_hold_exit_dataset_audit import Phase13DHoldExitDatasetAudit


REPORT_STEM = "phase13e_integrated_strategy_prototype_2025"
INITIAL_CASH = 1_000_000.0
DAILY_BUY_LIMIT = 900_000.0
MAX_POSITIONS = 5
ROUND_LOT = 100
COST_RATE = 0.002
PRIMARY_CANDIDATE_SET = "candidate_strength_top50"
VARIANTS = [
    "E0_baseline_candidate_strength_top50_fixed_20d",
    "E1_candidate_strength_top50_stop_loss_only",
    "E2_candidate_strength_top50_profit_protection",
    "E3_candidate_strength_top50_break_even",
    "E4_candidate_strength_top50_combined_conservative",
    "E5_candidate_strength_top50_combined_aggressive",
]
REQUIRED_REPORT_KEYS = [
    "recommended_strategy_variant",
    "recommended_reason",
    "annual_return_after_cost",
    "meets_annual_return_target",
    "meets_minimum_line",
    "ready_for_phase13f",
    "ready_for_broad_backtest",
    "ready_for_live_adoption",
    "recommended_next_phase",
    "leakage_risk",
    "blocking_issues",
]


@dataclass
class Position:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    entry_cost: float
    entry_fee: float
    peak_price: float
    peak_return: float
    holding_days: int
    extended: bool
    entry_future: dict[str, float | None]


@dataclass(frozen=True)
class Phase13EPaths:
    markdown: Path
    json: Path


class Phase13EIntegratedStrategyPrototype:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)
        self.phase13d = Phase13DHoldExitDatasetAudit(root)

    def run(self) -> Phase13EPaths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data, source_info = self.phase13d.phase13a.load_comparison_dataset()
        data = self.prepare_data(data)
        leakage = self.leakage_checklist()
        leakage["blocking_issues"] = self.blocking_issues(data)
        leakage["leakage_risk"] = "low" if not leakage["blocking_issues"] else "medium"
        variant_results = [self.simulate(data, variant) for variant in VARIANTS]
        recommendations = self.recommendations(variant_results, leakage)
        return {
            "metadata": self.metadata(),
            "input_artifact_summary": self.input_artifact_summary(data, source_info, leakage),
            "strategy_parameters": self.strategy_parameters(),
            "variant_results": variant_results,
            "final_recommendation": recommendations,
            "leakage_checklist": leakage,
            **{key: recommendations.get(key) for key in REQUIRED_REPORT_KEYS},
        }

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for column in result.columns:
            if column not in {"date", "code"}:
                result[column] = _numeric(result[column])
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result["code"] = result["code"].astype("string")
        return result.dropna(subset=["date", "code", "close", "candidate_strength", "opportunity_downside_score"]).sort_values(["date", "code"]).reset_index(drop=True)

    def simulate(self, data: pd.DataFrame, variant: str) -> dict[str, Any]:
        dates = sorted(data["date"].dropna().unique())
        by_date = {date: frame.copy() for date, frame in data.groupby("date")}
        price_lookup = {(row.date, str(row.code)): float(row.close) for row in data[["date", "code", "close"]].itertuples(index=False)}
        cash = INITIAL_CASH
        positions: list[Position] = []
        trades: list[dict[str, Any]] = []
        equity_curve = []
        reentry_codes: set[str] = set()
        previous_exits: set[str] = set()
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
            candidates = self.entry_candidates(frame, open_codes)
            slots = max(0, MAX_POSITIONS - len(positions))
            daily_spent = 0.0
            for _, row in candidates.head(slots).iterrows():
                if slots <= 0 or daily_spent >= DAILY_BUY_LIMIT:
                    break
                price = float(row["close"])
                target = min(DAILY_BUY_LIMIT / MAX_POSITIONS, DAILY_BUY_LIMIT - daily_spent, cash)
                shares = int(target // (price * ROUND_LOT)) * ROUND_LOT
                if shares <= 0:
                    continue
                gross = shares * price
                fee = gross * COST_RATE
                total = gross + fee
                if total > cash:
                    shares = int((cash / (1.0 + COST_RATE)) // (price * ROUND_LOT)) * ROUND_LOT
                    gross = shares * price
                    fee = gross * COST_RATE
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
                open_codes.add(code)
                slots -= 1
            equity_curve.append(self.equity_on_date(cash, positions, date, price_lookup))

        if dates:
            final_date = dates[-1]
            positions, forced_exits, forced_cost = self.force_close(positions, final_date, price_lookup)
            total_cost_paid += forced_cost
            for trade in forced_exits:
                cash += trade["sell_proceeds_after_cost"]
                trades.append(trade)
            equity_curve.append(self.equity_on_date(cash, positions, final_date, price_lookup))

        return self.metrics(variant, cash, trades, equity_curve, total_cost_paid, len(reentry_codes))

    def process_exits(
        self,
        positions: list[Position],
        date: pd.Timestamp,
        price_lookup: dict[tuple[pd.Timestamp, str], float],
        variant: str,
    ) -> tuple[list[Position], list[dict[str, Any]], float]:
        kept = []
        exits = []
        total_fee = 0.0
        for position in positions:
            price = price_lookup.get((date, position.code))
            if price is None:
                kept.append(position)
                continue
            position.holding_days += 1
            if price > position.peak_price:
                position.peak_price = price
            position.peak_return = max(position.peak_return, price / position.entry_price - 1.0)
            reason = self.exit_reason(position, price, variant)
            if reason:
                trade = self.close_trade(position, date, price, reason)
                total_fee += trade["sell_fee"]
                exits.append(trade)
            else:
                kept.append(position)
        return kept, exits, total_fee

    def exit_reason(self, position: Position, current_price: float, variant: str) -> str | None:
        current_return = current_price / position.entry_price - 1.0
        if variant == "E0_baseline_candidate_strength_top50_fixed_20d":
            return "fixed_hold_exit" if position.holding_days >= 20 else None
        if variant == "E1_candidate_strength_top50_stop_loss_only":
            if current_return <= -0.08:
                return "stop_loss_exit"
            return "fixed_hold_exit" if position.holding_days >= 20 else None
        if variant == "E2_candidate_strength_top50_profit_protection":
            if position.peak_return >= 0.05 and current_return <= 0.03:
                return "profit_protection_exit"
            return "fixed_hold_exit" if position.holding_days >= 20 else None
        if variant == "E3_candidate_strength_top50_break_even":
            if position.peak_return >= 0.03 and current_return <= 0.00:
                return "break_even_exit"
            return "fixed_hold_exit" if position.holding_days >= 20 else None
        if variant == "E4_candidate_strength_top50_combined_conservative":
            if position.peak_return >= 0.05 and current_return <= 0.03:
                return "profit_protection_exit"
            if position.peak_return >= 0.03 and current_return <= 0.00:
                return "break_even_exit"
            if current_return <= -0.08:
                return "stop_loss_exit"
            return "fixed_hold_exit" if position.holding_days >= 20 else None
        if variant == "E5_candidate_strength_top50_combined_aggressive":
            if position.peak_return >= 0.05 and current_return <= 0.05:
                return "profit_protection_exit"
            if position.peak_return >= 0.03 and current_return <= 0.00:
                return "break_even_exit"
            if current_return <= -0.05:
                return "stop_loss_exit"
            if position.holding_days >= 20 and position.peak_return >= 0.10 and current_return >= 0.05:
                position.extended = True
                return None
            if position.holding_days >= 30 and position.extended:
                return "extended_hold_exit"
            if position.holding_days >= 20 and not position.extended:
                return "fixed_hold_exit"
            return None
        raise ValueError(f"Unknown variant: {variant}")

    def close_trade(self, position: Position, exit_date: pd.Timestamp, price: float, reason: str) -> dict[str, Any]:
        gross = position.shares * price
        sell_fee = gross * COST_RATE
        proceeds = gross - sell_fee
        total_cost = position.entry_cost + position.entry_fee
        pnl = proceeds - total_cost
        realized_return = pnl / total_cost if total_cost else 0.0
        return {
            "code": position.code,
            "entry_date": position.entry_date,
            "exit_date": exit_date,
            "entry_price": position.entry_price,
            "exit_price": price,
            "shares": position.shares,
            "holding_days": position.holding_days,
            "exit_reason": reason,
            "entry_total_cost": total_cost,
            "sell_proceeds_after_cost": proceeds,
            "sell_fee": sell_fee,
            "pnl": pnl,
            "realized_return": realized_return,
            "peak_return": position.peak_return,
            "profit_decay_before_exit": max(0.0, position.peak_return - (price / position.entry_price - 1.0)),
            **position.entry_future,
        }

    def force_close(
        self,
        positions: list[Position],
        final_date: pd.Timestamp,
        price_lookup: dict[tuple[pd.Timestamp, str], float],
    ) -> tuple[list[Position], list[dict[str, Any]], float]:
        exits = []
        total_fee = 0.0
        for position in positions:
            price = price_lookup.get((final_date, position.code), position.entry_price)
            trade = self.close_trade(position, final_date, price, "forced_exit")
            total_fee += trade["sell_fee"]
            exits.append(trade)
        return [], exits, total_fee

    def entry_candidates(self, frame: pd.DataFrame, open_codes: set[str]) -> pd.DataFrame:
        frame = frame.copy()
        frame["_strength_rank"] = frame.groupby("date")["candidate_strength"].rank(method="first", ascending=False)
        candidates = frame[(frame["_strength_rank"] <= 50) & (~frame["code"].astype("string").isin(open_codes))].copy()
        return candidates.sort_values(["opportunity_downside_score", "candidate_strength", "turnover_value", "code"], ascending=[False, False, False, True])

    def entry_future_metrics(self, row: pd.Series) -> dict[str, float | None]:
        return {
            "future_return_20d": self.safe_row_float(row, "future_return_20d"),
            "future_max_drawdown_20d": self.safe_row_float(row, "future_max_drawdown_20d"),
            "future_max_return_20d": self.safe_row_float(row, "future_max_return_20d"),
            "top_decile_20d": self.safe_row_float(row, "top_decile_20d"),
            "downside_bad_20d": self.safe_row_float(row, "downside_bad_20d"),
        }

    def metrics(
        self,
        variant: str,
        final_cash: float,
        trades: list[dict[str, Any]],
        equity_curve: list[dict[str, Any]],
        total_cost_paid: float,
        same_code_reentry_count: int,
    ) -> dict[str, Any]:
        final_assets = final_cash
        net_profit = final_assets - INITIAL_CASH
        returns = [trade["realized_return"] for trade in trades]
        pnls = [trade["pnl"] for trade in trades]
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        hold_days = [trade["holding_days"] for trade in trades]
        winner_to_loser = [trade for trade in trades if trade["peak_return"] >= 0.05 and trade["realized_return"] < 0]
        annual_return = final_assets / INITIAL_CASH - 1.0
        exit_counts = self.exit_counts(trades)
        return {
            "variant": variant,
            "annual_return": _safe_float(annual_return),
            "annual_return_after_cost": _safe_float(annual_return),
            "net_profit": _safe_float(net_profit),
            "final_assets": _safe_float(final_assets),
            "PF": _safe_float(gross_profit / gross_loss) if gross_loss else None,
            "DD": self.max_drawdown(equity_curve),
            "win_rate": _safe_float(sum(1 for value in returns if value > 0) / len(returns)) if returns else None,
            "capital_utilization": self.capital_utilization(equity_curve),
            "trade_count": len(trades),
            "average_holding_days": _safe_float(sum(hold_days) / len(hold_days)) if hold_days else None,
            "median_holding_days": _safe_float(median(hold_days)) if hold_days else None,
            "same_code_reentry_count": same_code_reentry_count,
            "winner_to_loser_count": len(winner_to_loser),
            "winner_to_loser_rate": _safe_float(len(winner_to_loser) / len(trades)) if trades else None,
            "profit_protection_exit_count": exit_counts.get("profit_protection_exit", 0),
            "break_even_exit_count": exit_counts.get("break_even_exit", 0),
            "stop_loss_exit_count": exit_counts.get("stop_loss_exit", 0),
            "fixed_hold_exit_count": exit_counts.get("fixed_hold_exit", 0),
            "extended_hold_exit_count": exit_counts.get("extended_hold_exit", 0),
            "forced_exit_count": exit_counts.get("forced_exit", 0),
            "avg_profit_decay_before_exit": _safe_float(sum(trade["profit_decay_before_exit"] for trade in trades) / len(trades)) if trades else None,
            "cost_paid": _safe_float(total_cost_paid),
            **self.entry_quality(trades),
        }

    def exit_counts(self, trades: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trade in trades:
            counts[trade["exit_reason"]] = counts.get(trade["exit_reason"], 0) + 1
        return counts

    def equity_on_date(
        self,
        cash: float,
        positions: list[Position],
        date: pd.Timestamp,
        price_lookup: dict[tuple[pd.Timestamp, str], float],
    ) -> dict[str, Any]:
        market_value = 0.0
        for position in positions:
            price = price_lookup.get((date, position.code), position.entry_price)
            market_value += position.shares * price
        equity = cash + market_value
        return {"date": date, "equity": equity, "market_value": market_value, "cash": cash}

    def max_drawdown(self, equity_curve: list[dict[str, Any]]) -> float | None:
        if not equity_curve:
            return None
        peak = -math.inf
        max_dd = 0.0
        for row in equity_curve:
            equity = row["equity"]
            peak = max(peak, equity)
            if peak > 0:
                max_dd = min(max_dd, equity / peak - 1.0)
        return _safe_float(max_dd)

    def capital_utilization(self, equity_curve: list[dict[str, Any]]) -> float | None:
        rates = [row["market_value"] / row["equity"] for row in equity_curve if row["equity"] > 0]
        return _safe_float(sum(rates) / len(rates)) if rates else None

    def entry_quality(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        frame = pd.DataFrame(trades)
        if frame.empty:
            return {
                "entry_top_decile_rate_20d": None,
                "entry_downside_bad_rate_20d": None,
                "entry_mean_future_return_20d": None,
                "entry_mean_future_max_drawdown_20d": None,
            }
        return {
            "entry_top_decile_rate_20d": self.mean(frame, "top_decile_20d"),
            "entry_downside_bad_rate_20d": self.mean(frame, "downside_bad_20d"),
            "entry_mean_future_return_20d": self.mean(frame, "future_return_20d"),
            "entry_mean_future_max_drawdown_20d": self.mean(frame, "future_max_drawdown_20d"),
        }

    def recommendations(self, results: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        valid = [row for row in results if row.get("annual_return_after_cost") is not None]
        if leakage["blocking_issues"] or not valid:
            return self.blocked_recommendation(leakage)
        best = max(valid, key=lambda row: self.score(row))
        meets_annual = self.value(best, "annual_return_after_cost") >= 0.50
        meets_minimum = (
            meets_annual
            and self.value(best, "PF") >= 2.0
            and self.value(best, "DD") >= -0.10
            and self.value(best, "capital_utilization") >= 0.60
            and self.value(best, "net_profit") > 0
        )
        blocking = list(leakage["blocking_issues"])
        if self.value(best, "DD") < -0.25:
            blocking.append("recommended_variant_extreme_dd")
        if self.value(best, "trade_count") < 10:
            blocking.append("recommended_variant_low_trade_count")
        next_phase = "Phase13-F Strict OOS / Multi-Year Limited Check" if meets_minimum and not blocking else "Phase13-E2 Rule Tuning"
        return {
            "recommended_strategy_variant": best["variant"],
            "recommended_reason": (
                f"Best annual_return_after_cost={self.value(best, 'annual_return_after_cost'):.4f}, "
                f"PF={self.value(best, 'PF'):.4f}, DD={self.value(best, 'DD'):.4f}, "
                f"capital_utilization={self.value(best, 'capital_utilization'):.4f}."
            ),
            "annual_return_after_cost": best["annual_return_after_cost"],
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
            "meets_annual_return_target": False,
            "meets_minimum_line": False,
            "ready_for_phase13f": False,
            "ready_for_broad_backtest": False,
            "ready_for_live_adoption": False,
            "recommended_next_phase": "Phase13-C Horizon-Aware Valuation Prototype",
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def score(self, row: dict[str, Any]) -> float:
        dd_penalty = abs(min(0.0, self.value(row, "DD"))) * 0.25
        return (
            self.value(row, "annual_return_after_cost") * 0.50
            + self.value(row, "PF") * 0.08
            + self.value(row, "capital_utilization") * 0.10
            - dd_penalty
            - self.value(row, "winner_to_loser_rate") * 0.10
        )

    def strategy_parameters(self) -> dict[str, Any]:
        return {
            "primary_candidate_set": PRIMARY_CANDIDATE_SET,
            "candidate_generation": "candidate_strength_top50",
            "entry_rank": "opportunity_downside_score desc",
            "initial_cash": INITIAL_CASH,
            "daily_buy_limit": DAILY_BUY_LIMIT,
            "max_positions": MAX_POSITIONS,
            "round_lot": ROUND_LOT,
            "cost_rate": COST_RATE,
            "entry_price": "same-day close from existing 2025 artifact",
            "future_columns_used_for_entry_exit": [],
            "future_columns_used_only_for_evaluation": ["future_return_20d", "future_max_return_20d", "future_max_drawdown_20d", "top_decile_20d", "downside_bad_20d"],
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
            "primary_candidate_set": PRIMARY_CANDIDATE_SET,
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def blocking_issues(self, data: pd.DataFrame) -> list[str]:
        issues = []
        for column in ["date", "code", "close", "candidate_strength", "opportunity_downside_score"]:
            if column not in data.columns:
                issues.append(f"missing_required_entry_column:{column}")
        if data.empty:
            issues.append("empty_2025_dataset")
        return issues

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_as_entry_or_exit_features": [],
            "future_columns_used_only_for_evaluation": self.phase13d.phase13a.expected_future_columns(),
            "future_columns_used_as_features": [],
            "backtest_columns_used_as_features": [],
            "trade_result_columns_used_as_features": [],
            "new_model_trained": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "strategy_backtest_executed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
            "cost_rate": COST_RATE,
            "period": {"start": "2025-01-01", "end": "2025-12-31"},
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "13-E",
            "scope": "2025-only lightweight integrated strategy prototype",
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
                "# Phase 13-E Integrated Strategy Prototype",
                "",
                "## Input Artifact Summary",
                "",
                self.table([report["input_artifact_summary"]], ["row_count", "date_min", "date_max", "unique_code_count", "primary_candidate_set", "leakage_risk", "blocking_issues"]),
                "",
                "## Strategy Parameters",
                "",
                self.table([report["strategy_parameters"]], ["primary_candidate_set", "candidate_generation", "entry_rank", "initial_cash", "daily_buy_limit", "max_positions", "round_lot", "cost_rate", "entry_price", "future_columns_used_for_entry_exit", "future_columns_used_only_for_evaluation"]),
                "",
                "## Variant Results",
                "",
                self.table(report["variant_results"], ["variant", "annual_return_after_cost", "net_profit", "final_assets", "PF", "DD", "win_rate", "capital_utilization", "trade_count", "average_holding_days", "median_holding_days", "winner_to_loser_count", "winner_to_loser_rate", "profit_protection_exit_count", "break_even_exit_count", "stop_loss_exit_count", "fixed_hold_exit_count", "extended_hold_exit_count", "avg_profit_decay_before_exit", "entry_top_decile_rate_20d", "entry_downside_bad_rate_20d"]),
                "",
                "## Final Recommendation",
                "",
                self.table([report["final_recommendation"]], REQUIRED_REPORT_KEYS),
                "",
                "## Leakage Checklist",
                "",
                self.table([report["leakage_checklist"]], ["future_columns_used_as_entry_or_exit_features", "future_columns_used_only_for_evaluation", "new_model_trained", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "strategy_backtest_executed", "historical_predictions_regenerated", "jquants_api_called", "openai_api_called", "cost_rate", "leakage_risk", "blocking_issues"]),
                "",
            ]
        )

    def mean(self, frame: pd.DataFrame, column: str) -> float | None:
        if column not in frame.columns:
            return None
        values = _numeric(frame[column]).dropna()
        return _safe_float(values.mean()) if not values.empty else None

    def safe_row_float(self, row: pd.Series, column: str) -> float | None:
        if column not in row or pd.isna(row[column]):
            return None
        return _safe_float(row[column])

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
