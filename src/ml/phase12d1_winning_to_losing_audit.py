"""Phase 12-D1 winning trades turned into losers audit.

This is a 2025-only audit for the C2c normalized downside-squared allocation
with B5_2 recalibrated exit. It inspects the realized trade path to quantify
how often trades had meaningful unrealized gains before ending as losses. It
does not run a full backtest, change profiles, overwrite models, or regenerate
historical predictions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase12c2_utilization_without_dd_explosion import FUTURE_EVAL_COLUMNS, Phase12C2UtilizationWithoutDDExplosion, VariantSpec


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase12d1_winning_to_losing_audit_2025"
TARGET_SPEC = VariantSpec("D1_C2c_normalized_downside_squared_B5_2_exit", "downside_squared")


@dataclass(frozen=True)
class Phase12D1Paths:
    markdown: Path
    json: Path


class Phase12D1WinningToLosingAudit(Phase12C2UtilizationWithoutDDExplosion):
    def run(self) -> Phase12D1Paths:
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
                "recommendation": self.recommendation(pd.DataFrame(), {}, leakage),
            }

        trades, daily = self.simulate_target(data)
        audit_5 = self.winning_to_losing_audit(trades, threshold=0.05)
        audit_10 = self.winning_to_losing_audit(trades, threshold=0.10)
        top_loss = self.top_loss_trades(trades)
        distribution = self.profit_decay_distribution(trades)
        exit_timing = self.exit_timing_audit(trades)
        leakage_source = self.main_profit_leakage_source(trades, audit_5, exit_timing)
        return {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "strategy_metrics": self.metrics(TARGET_SPEC.name, trades, daily),
            "audit_5pct": audit_5,
            "audit_10pct": audit_10,
            "top_loss_trades": top_loss,
            "profit_decay_distribution": distribution,
            "exit_timing_audit": exit_timing,
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(trades, leakage_source, leakage),
        }

    def simulate_target(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cash = self.options.initial_cash
        positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        daily_rows: list[dict[str, Any]] = []
        dates = list(pd.Series(data["date"].dropna().unique()).sort_values())
        by_date = {date: group.set_index("code", drop=False) for date, group in data.groupby("date", sort=True)}

        for current_date in dates:
            current = by_date[current_date]
            current_rank_frame = current.reset_index(drop=True)
            still_open = []
            for position in positions:
                current_row = current.loc[position["code"]] if position["code"] in current.index else None
                if current_row is not None:
                    close = float(current_row["close"])
                    position["last_close"] = close
                    if close > float(position["peak_price"]):
                        position["peak_price"] = close
                        position["peak_date"] = current_date
                reason = self.exit_reason(position, current_date, current_row)
                if reason:
                    exit_close = float(current_row["close"]) if current_row is not None else float(position["last_close"])
                    trade = self.close_position(position, current_date, exit_close, reason, TARGET_SPEC.name)
                    cash += trade["exit_cash_flow"]
                    trades.append(trade)
                else:
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            ranked = current_rank_frame.sort_values(["opportunity_proba", "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions)
            ranked = ranked[_numeric(ranked["a3_3_allocation_weight"]) > 0].copy()
            selected = ranked.head(slots) if slots else ranked.iloc[0:0]
            target_amounts = self.target_amounts(selected, TARGET_SPEC, cash, positions)
            bought_today = 0
            for index, row in selected.iterrows():
                target_amount = target_amounts.get(index, 0.0)
                lot_cost = float(row["close"]) * self.options.round_lot
                lots = int(target_amount // (lot_cost * (1.0 + self.options.cost_rate))) if lot_cost > 0 else 0
                buy_amount = lots * lot_cost
                buy_cost = buy_amount * self.options.cost_rate
                cash_out = buy_amount + buy_cost
                if lots <= 0 or cash_out > cash:
                    continue
                cash -= cash_out
                bought_today += 1
                positions.append(
                    {
                        "entry_date": current_date,
                        "due_date": current_date + pd.offsets.BDay(self.options.holding_days),
                        "strategy": TARGET_SPEC.name,
                        "code": str(row["code"]),
                        "buy_amount": buy_amount,
                        "buy_cost": buy_cost,
                        "target_buy_amount": target_amount,
                        "lot_count": lots,
                        "entry_close": float(row["close"]),
                        "last_close": float(row["close"]),
                        "peak_price": float(row["close"]),
                        "peak_date": current_date,
                        "entry_opportunity_top_decile_proba": _safe_float(row.get("opportunity_top_decile_proba")),
                        "entry_downside_bad_proba": _safe_float(row.get("downside_bad_proba")),
                        "entry_downside_rank_percentile": _safe_float(row.get("downside_rank_percentile")),
                        "allocation_weight": _safe_float(row.get("a3_3_allocation_weight")),
                        "normalized_weight": _safe_float(target_amount / self.options.daily_buy_budget) if self.options.daily_buy_budget else None,
                        **{column: _safe_float(row.get(column)) for column in FUTURE_EVAL_COLUMNS},
                    }
                )

            marked_value = sum(self.position_value(position) for position in positions)
            total_assets = cash + marked_value
            daily_rows.append(
                {
                    "strategy": TARGET_SPEC.name,
                    "date": current_date,
                    "cash": cash,
                    "open_position_count": len(positions),
                    "bought_today": bought_today,
                    "marked_position_value": marked_value,
                    "total_assets": total_assets,
                    "capital_utilization": marked_value / self.options.initial_cash if self.options.initial_cash else None,
                }
            )

        if dates:
            last_date = dates[-1]
            for position in positions:
                trade = self.close_position(position, last_date, float(position["last_close"]), "forced_end_of_period", TARGET_SPEC.name)
                cash += trade["exit_cash_flow"]
                trades.append(trade)
            if daily_rows:
                daily_rows[-1]["total_assets"] = cash
                daily_rows[-1]["marked_position_value"] = 0.0
                daily_rows[-1]["capital_utilization"] = 0.0
        return pd.DataFrame(trades), pd.DataFrame(daily_rows)

    def close_position(self, position: dict[str, Any], exit_date: pd.Timestamp, exit_close: float, reason: str, strategy: str) -> dict[str, Any]:
        row = super().close_position(position, exit_date, exit_close, reason, strategy)
        peak_return = float(position["peak_price"]) / float(position["entry_close"]) - 1.0
        final_return_before_cost = exit_close / float(position["entry_close"]) - 1.0
        profit_decay = peak_return - final_return_before_cost
        row.update(
            {
                "entry_price": float(position["entry_close"]),
                "peak_price": float(position["peak_price"]),
                "peak_date": position["peak_date"],
                "exit_price": exit_close,
                "peak_return": peak_return,
                "final_return_before_cost": final_return_before_cost,
                "profit_decay": profit_decay,
                "peak_profit_amount": float(position["buy_amount"]) * peak_return,
                "profit_decay_amount": float(position["buy_amount"]) * profit_decay,
            }
        )
        return row

    def winning_to_losing_audit(self, trades: pd.DataFrame, *, threshold: float) -> dict[str, Any]:
        if trades.empty:
            return self.empty_winning_to_losing(threshold)
        frame = trades.copy()
        frame["realized_return"] = _numeric(frame["realized_return"])
        frame["peak_return"] = _numeric(frame["peak_return"])
        frame["profit_decay"] = _numeric(frame["profit_decay"])
        selected = frame[(frame["realized_return"] < 0) & (frame["peak_return"] >= threshold)].copy()
        return {
            "threshold": threshold,
            "count": int(len(selected)),
            "avg_peak_profit": _safe_float(selected["peak_return"].mean()) if not selected.empty else 0.0,
            "avg_final_return": _safe_float(selected["realized_return"].mean()) if not selected.empty else 0.0,
            "avg_profit_decay": _safe_float(selected["profit_decay"].mean()) if not selected.empty else 0.0,
            "winning_trades_turned_losers_count": int(len(selected)),
            "winning_trades_turned_losers_profit_loss": _safe_float(selected["realized_profit"].sum()) if not selected.empty else 0.0,
            "estimated_recoverable_profit": _safe_float(selected["profit_decay_amount"].sum()) if not selected.empty else 0.0,
            "estimated_break_even_recovery": _safe_float(abs(selected["realized_profit"].sum())) if not selected.empty else 0.0,
        }

    def empty_winning_to_losing(self, threshold: float) -> dict[str, Any]:
        return {
            "threshold": threshold,
            "count": 0,
            "avg_peak_profit": 0.0,
            "avg_final_return": 0.0,
            "avg_profit_decay": 0.0,
            "winning_trades_turned_losers_count": 0,
            "winning_trades_turned_losers_profit_loss": 0.0,
            "estimated_recoverable_profit": 0.0,
            "estimated_break_even_recovery": 0.0,
        }

    def top_loss_trades(self, trades: pd.DataFrame, limit: int = 30) -> list[dict[str, Any]]:
        if trades.empty:
            return []
        frame = trades.copy()
        frame["realized_profit"] = _numeric(frame["realized_profit"])
        frame = frame.sort_values("realized_profit").head(limit)
        return self.records(
            frame,
            [
                "entry_date",
                "exit_date",
                "code",
                "entry_price",
                "peak_price",
                "exit_price",
                "peak_return",
                "realized_return",
                "final_return_before_cost",
                "profit_decay",
                "realized_profit",
                "profit_decay_amount",
                "exit_reason",
            ],
        )

    def profit_decay_distribution(self, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty:
            return {}
        peak = _numeric(trades["peak_return"])
        final = _numeric(trades["realized_return"])
        decay = _numeric(trades["profit_decay"])
        return {
            "trade_count": int(len(trades)),
            "peak_return_mean": _safe_float(peak.mean()),
            "peak_return_p50": _safe_float(peak.quantile(0.50)),
            "peak_return_p90": _safe_float(peak.quantile(0.90)),
            "final_return_mean": _safe_float(final.mean()),
            "final_return_p50": _safe_float(final.quantile(0.50)),
            "final_return_p10": _safe_float(final.quantile(0.10)),
            "profit_decay_mean": _safe_float(decay.mean()),
            "profit_decay_p50": _safe_float(decay.quantile(0.50)),
            "profit_decay_p90": _safe_float(decay.quantile(0.90)),
            "peak_vs_final_corr": _safe_float(pd.concat([peak, final], axis=1).corr().iloc[0, 1]) if len(trades) > 1 else None,
        }

    def exit_timing_audit(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty:
            return []
        frame = trades.copy()
        rows = []
        for reason, group in frame.groupby("exit_reason", sort=True):
            rows.append(
                {
                    "exit_reason": reason,
                    "count": int(len(group)),
                    "avg_peak_profit": _safe_float(_numeric(group["peak_return"]).mean()),
                    "avg_final_return": _safe_float(_numeric(group["realized_return"]).mean()),
                    "avg_profit_decay": _safe_float(_numeric(group["profit_decay"]).mean()),
                    "profit_decay_amount_sum": _safe_float(_numeric(group["profit_decay_amount"]).sum()),
                    "realized_profit_sum": _safe_float(_numeric(group["realized_profit"]).sum()),
                }
            )
        return rows

    def main_profit_leakage_source(self, trades: pd.DataFrame, audit_5: dict[str, Any], exit_timing: list[dict[str, Any]]) -> dict[str, Any]:
        frame = trades.copy()
        if not frame.empty:
            frame["realized_return"] = _numeric(frame["realized_return"])
            frame["peak_return"] = _numeric(frame["peak_return"])
            winners_lost = frame[(frame["realized_return"] < 0) & (frame["peak_return"] >= 0.05)].copy()
        else:
            winners_lost = pd.DataFrame()
        if not winners_lost.empty:
            by_reason = []
            for reason, group in winners_lost.groupby("exit_reason", sort=True):
                by_reason.append(
                    {
                        "exit_reason": reason,
                        "count": int(len(group)),
                        "profit_decay_amount_sum": _safe_float(_numeric(group["profit_decay_amount"]).sum()),
                        "realized_profit_sum": _safe_float(_numeric(group["realized_profit"]).sum()),
                    }
                )
            biggest_wtl = max(by_reason, key=lambda row: _safe_float(row.get("profit_decay_amount_sum")) or 0.0)
        else:
            by_reason = []
            biggest_wtl = {}
        biggest_overall = max(exit_timing, key=lambda row: _safe_float(row.get("profit_decay_amount_sum")) or 0.0) if exit_timing else {}
        return {
            "main_profit_leakage_source": biggest_wtl.get("exit_reason") or biggest_overall.get("exit_reason"),
            "winning_to_losing_conversion_detected": audit_5.get("winning_trades_turned_losers_count", 0) > 0,
            "estimated_recoverable_profit": audit_5.get("estimated_recoverable_profit", 0.0),
            "winning_to_losing_exit_reason_breakdown": by_reason,
            "dominant_exit_reason_profit_decay_overall": biggest_overall,
        }

    def recommendation(self, trades: pd.DataFrame, leakage_source: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"recommended_next_phase": "Fix Phase12-D1 leakage blockers"}
        detected = bool(leakage_source.get("winning_to_losing_conversion_detected"))
        source = leakage_source.get("main_profit_leakage_source")
        if detected and source == "stop_loss":
            improvement = "Phase12-D2 profit_protection_exit or break_even_guard"
        elif detected:
            improvement = "Phase12-D2 trailing_profit_lock"
        else:
            improvement = "Phase12-D2 late_exit_decay_guard"
        return {
            "main_profit_leakage_source": source,
            "winning_to_losing_conversion_detected": detected,
            "estimated_recoverable_profit": leakage_source.get("estimated_recoverable_profit"),
            "recommended_exit_improvement": improvement,
            "recommended_next_phase": improvement.split()[0] if improvement else "Phase12-D2",
        }

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update({"phase": "12-D1", "scope": "2025 winning-to-losing trade path audit only"})
        return metadata

    def conditions(self) -> dict[str, Any]:
        conditions = super().conditions()
        conditions.update(
            {
                "target_strategy": TARGET_SPEC.name,
                "audit_only": True,
                "strategy_backtest_variant_count": 1,
                "winning_thresholds": [0.05, 0.10],
            }
        )
        return conditions

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_only_for_audit": FUTURE_EVAL_COLUMNS,
            "future_columns_used_as_features": [],
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase12D1Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12D1Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-D1 Winning Trades Turned Into Losers Audit",
            "",
            "## Strategy Metrics",
            "",
            self.table([report.get("strategy_metrics", {})], ["strategy", "net_profit", "PF", "DD", "win_rate", "total_trades", "final_assets", "capital_utilization", "average_holding_days", "cost_paid", "exit_reason_counts"]),
            "",
            "## Winning To Losing",
            "",
            self.table([report.get("audit_5pct", {}), report.get("audit_10pct", {})], ["threshold", "count", "avg_peak_profit", "avg_final_return", "avg_profit_decay", "winning_trades_turned_losers_profit_loss", "estimated_recoverable_profit", "estimated_break_even_recovery"]),
            "",
            "## Profit Decay Distribution",
            "",
            self.table([report.get("profit_decay_distribution", {})], ["trade_count", "peak_return_mean", "peak_return_p50", "peak_return_p90", "final_return_mean", "final_return_p50", "final_return_p10", "profit_decay_mean", "profit_decay_p50", "profit_decay_p90", "peak_vs_final_corr"]),
            "",
            "## Exit Timing Audit",
            "",
            self.table(report.get("exit_timing_audit", []), ["exit_reason", "count", "avg_peak_profit", "avg_final_return", "avg_profit_decay", "profit_decay_amount_sum", "realized_profit_sum"]),
            "",
            "## Top Loss Trades",
            "",
            self.table(report.get("top_loss_trades", []), ["entry_date", "exit_date", "code", "entry_price", "peak_price", "exit_price", "peak_return", "realized_return", "profit_decay", "realized_profit", "profit_decay_amount", "exit_reason"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_only_for_audit", "future_columns_used_as_features", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["main_profit_leakage_source", "winning_to_losing_conversion_detected", "estimated_recoverable_profit", "recommended_exit_improvement", "recommended_next_phase"]),
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
