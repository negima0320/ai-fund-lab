"""Phase 12-C2 utilization without DD explosion.

This module audits why the Phase 12-C normalized dynamic allocation improves
profit/utilization but worsens drawdown. It is limited to 2025, uses existing
Phase 12 artifacts, and does not run a full backtest, modify profiles,
overwrite models, or regenerate historical predictions.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase12a_dynamic_capital_allocation import EVAL_COLUMNS
from ml.phase12b_limited_allocation_strategy_check import ARTIFACT_PATH, END_DATE, ROUND_LOT, START_DATE
from ml.phase12c_dynamic_allocation_recalibrated_exit import Phase12COptions


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase12c2_utilization_without_dd_explosion_2025"
FUTURE_EVAL_COLUMNS = EVAL_COLUMNS


@dataclass(frozen=True)
class VariantSpec:
    name: str
    allocation_mode: str


@dataclass(frozen=True)
class Phase12C2Paths:
    markdown: Path
    json: Path


VARIANTS = [
    VariantSpec("C2_base_dynamic_normalized_B5_2_exit", "normalized"),
    VariantSpec("C2a_normalized_cap_20pct", "cap_20pct"),
    VariantSpec("C2b_normalized_cap_15pct", "cap_15pct"),
    VariantSpec("C2c_normalized_downside_penalty_squared", "downside_squared"),
    VariantSpec("C2d_normalized_top_weight_cap_30pct", "top_weight_cap_30pct"),
    VariantSpec("C2e_normalized_cash_reserve_80pct", "cash_reserve_80pct"),
]


class Phase12C2UtilizationWithoutDDExplosion:
    def __init__(self, root: Path | str = ROOT, *, options: Phase12COptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase12COptions()

    def run(self) -> Phase12C2Paths:
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
                "recommendation": self.recommendation([], {}, {}, {}, leakage),
            }

        variant_results = []
        simulations: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        for spec in VARIANTS:
            trades, daily, snapshots = self.simulate(data, spec)
            simulations[spec.name] = (trades, daily, snapshots)
            variant_results.append(self.metrics(spec.name, trades, daily))

        base_trades, base_daily, base_snapshots = simulations[VARIANTS[0].name]
        dd_audit = self.dd_attribution_audit(base_trades, base_daily, base_snapshots)
        concentration = self.concentration_audit(base_snapshots)
        downside_exposure = self.downside_exposure_audit(base_trades)
        comparison = self.variant_comparison(variant_results)
        return {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "dd_attribution_audit": dd_audit,
            "concentration_audit": concentration,
            "downside_exposure_audit": downside_exposure,
            "variant_results": variant_results,
            "variant_comparison": comparison,
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(variant_results, dd_audit, concentration, downside_exposure, leakage),
        }

    def load_frame(self) -> pd.DataFrame:
        data = pd.read_parquet(self.root / ARTIFACT_PATH)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        data = data[(data["date"] >= START_DATE) & (data["date"] <= END_DATE)].copy()
        columns = [
            "date",
            "code",
            "close",
            "turnover_value",
            "opportunity_proba",
            "downside_bad_proba",
            "opportunity_rank_percentile",
            "downside_rank_percentile",
            "confidence",
            *FUTURE_EVAL_COLUMNS,
        ]
        available = [column for column in columns if column in data.columns]
        data = data[available].drop_duplicates(["date", "code"])
        for column in data.columns:
            if column not in {"date", "code"}:
                data[column] = _numeric(data[column])
        data["opportunity_top_decile_proba"] = data["opportunity_proba"]
        data["opportunity_score_proba_rank"] = data["opportunity_rank_percentile"]
        data["a3_3_allocation_weight"] = self.a3_3_weight(data["downside_rank_percentile"])
        return data.dropna(subset=["date", "code", "close", "opportunity_proba", "downside_rank_percentile"]).sort_values(["date", "code"]).reset_index(drop=True)

    def a3_3_weight(self, downside_rank: pd.Series) -> pd.Series:
        rank = _numeric(downside_rank)
        return rank.map(lambda value: 1.0 if value <= 0.40 else 0.6 if value <= 0.70 else 0.3 if value <= 0.85 else 0.0)

    def simulate(self, data: pd.DataFrame, spec: VariantSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cash = self.options.initial_cash
        positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        daily_rows: list[dict[str, Any]] = []
        snapshot_rows: list[dict[str, Any]] = []
        dates = list(pd.Series(data["date"].dropna().unique()).sort_values())
        by_date = {date: group.set_index("code", drop=False) for date, group in data.groupby("date", sort=True)}

        for current_date in dates:
            current = by_date[current_date]
            current_rank_frame = current.reset_index(drop=True)
            still_open = []
            for position in positions:
                current_row = current.loc[position["code"]] if position["code"] in current.index else None
                if current_row is not None:
                    position["last_close"] = float(current_row["close"])
                reason = self.exit_reason(position, current_date, current_row)
                if reason:
                    exit_close = float(current_row["close"]) if current_row is not None else float(position["last_close"])
                    trade = self.close_position(position, current_date, exit_close, reason, spec.name)
                    cash += trade["exit_cash_flow"]
                    trades.append(trade)
                else:
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            ranked = current_rank_frame.sort_values(["opportunity_proba", "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions)
            ranked = ranked[_numeric(ranked["a3_3_allocation_weight"]) > 0].copy()
            selected = ranked.head(slots) if slots else ranked.iloc[0:0]
            target_amounts = self.target_amounts(selected, spec, cash, positions)
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
                        "strategy": spec.name,
                        "code": str(row["code"]),
                        "buy_amount": buy_amount,
                        "buy_cost": buy_cost,
                        "target_buy_amount": target_amount,
                        "lot_count": lots,
                        "entry_close": float(row["close"]),
                        "last_close": float(row["close"]),
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
                    "strategy": spec.name,
                    "date": current_date,
                    "cash": cash,
                    "open_position_count": len(positions),
                    "bought_today": bought_today,
                    "marked_position_value": marked_value,
                    "total_assets": total_assets,
                    "capital_utilization": marked_value / self.options.initial_cash if self.options.initial_cash else None,
                }
            )
            snapshot_rows.extend(self.position_snapshots(spec.name, current_date, positions, total_assets, marked_value))

        if dates:
            last_date = dates[-1]
            for position in positions:
                trade = self.close_position(position, last_date, float(position["last_close"]), "forced_end_of_period", position.get("strategy", spec.name))
                cash += trade["exit_cash_flow"]
                trades.append(trade)
            if daily_rows:
                daily_rows[-1]["total_assets"] = cash
                daily_rows[-1]["marked_position_value"] = 0.0
                daily_rows[-1]["capital_utilization"] = 0.0
        return pd.DataFrame(trades), pd.DataFrame(daily_rows), pd.DataFrame(snapshot_rows)

    def target_amounts(self, selected: pd.DataFrame, spec: VariantSpec, cash: float, positions: list[dict[str, Any]]) -> dict[Any, float]:
        if selected.empty:
            return {}
        if spec.allocation_mode == "cash_reserve_80pct":
            open_value = sum(self.position_value(position) for position in positions)
            utilization_room = max(0.0, self.options.initial_cash * 0.80 - open_value)
            available_budget = min(cash, self.options.daily_buy_budget, utilization_room)
        else:
            available_budget = min(cash, self.options.daily_buy_budget)
        weights = _numeric(selected["a3_3_allocation_weight"]).clip(lower=0)
        if spec.allocation_mode == "downside_squared":
            penalty = (1.0 - _numeric(selected["downside_bad_proba"]).fillna(1.0)).clip(lower=0, upper=1) ** 2
            weights = weights * penalty
        total = float(weights.sum())
        if total <= 0 or available_budget <= 0:
            return {index: 0.0 for index in selected.index}
        normalized = weights / total
        if spec.allocation_mode == "top_weight_cap_30pct":
            normalized = normalized.clip(upper=0.30)
        amounts = {index: available_budget * float(weight) for index, weight in normalized.items()}
        if spec.allocation_mode == "cap_20pct":
            cap = self.options.initial_cash * 0.20
            amounts = {index: min(amount, cap) for index, amount in amounts.items()}
        elif spec.allocation_mode == "cap_15pct":
            cap = self.options.initial_cash * 0.15
            amounts = {index: min(amount, cap) for index, amount in amounts.items()}
        return amounts

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None) -> str | None:
        if current_row is not None:
            observed_return = float(current_row["close"]) / float(position["entry_close"]) - 1.0
            if observed_return <= self.options.stop_loss_rate:
                return "stop_loss"
            current_proba = _safe_float(current_row.get("opportunity_top_decile_proba"))
            current_rank = _safe_float(current_row.get("opportunity_score_proba_rank"))
            entry_proba = _safe_float(position.get("entry_opportunity_top_decile_proba"))
            if current_rank is not None and current_rank < self.options.opportunity_rank_floor:
                return "opportunity_exit"
            if current_proba is not None and entry_proba is not None and current_proba <= entry_proba - self.options.opportunity_drop_threshold:
                return "opportunity_exit"
        if current_date >= position["due_date"]:
            return "time_exit_20d"
        return None

    def close_position(self, position: dict[str, Any], exit_date: pd.Timestamp, exit_close: float, reason: str, strategy: str) -> dict[str, Any]:
        exit_amount = float(position["lot_count"]) * self.options.round_lot * exit_close
        sell_cost = exit_amount * self.options.cost_rate
        exit_cash_flow = exit_amount - sell_cost
        total_cost = float(position["buy_cost"]) + sell_cost
        profit = exit_cash_flow - float(position["buy_amount"]) - float(position["buy_cost"])
        holding_days = len(pd.bdate_range(position["entry_date"], exit_date)) - 1
        return {
            "strategy": strategy,
            "entry_date": position["entry_date"],
            "exit_date": exit_date,
            "code": position["code"],
            "buy_amount": position["buy_amount"],
            "exit_amount": exit_amount,
            "exit_cash_flow": exit_cash_flow,
            "realized_profit": profit,
            "realized_return": profit / float(position["buy_amount"]) if position["buy_amount"] else None,
            "holding_days": holding_days,
            "exit_reason": reason,
            "cost_paid": total_cost,
            "allocation_weight": position.get("allocation_weight"),
            "normalized_weight": position.get("normalized_weight"),
            "entry_downside_bad_proba": position.get("entry_downside_bad_proba"),
            "entry_downside_rank_percentile": position.get("entry_downside_rank_percentile"),
            **{column: position.get(column) for column in FUTURE_EVAL_COLUMNS},
        }

    def position_value(self, position: dict[str, Any]) -> float:
        return float(position["lot_count"]) * self.options.round_lot * float(position["last_close"])

    def position_snapshots(self, strategy: str, current_date: pd.Timestamp, positions: list[dict[str, Any]], total_assets: float, marked_value: float) -> list[dict[str, Any]]:
        rows = []
        for position in positions:
            value = self.position_value(position)
            rows.append(
                {
                    "strategy": strategy,
                    "date": current_date,
                    "code": position["code"],
                    "position_value": value,
                    "portfolio_weight": value / total_assets if total_assets else None,
                    "position_weight": value / marked_value if marked_value else None,
                    "entry_downside_bad_proba": position.get("entry_downside_bad_proba"),
                    "entry_downside_rank_percentile": position.get("entry_downside_rank_percentile"),
                    "allocation_weight": position.get("allocation_weight"),
                    "normalized_weight": position.get("normalized_weight"),
                    "unrealized_return": float(position["last_close"]) / float(position["entry_close"]) - 1.0 if position.get("entry_close") else None,
                }
            )
        return rows

    def metrics(self, strategy: str, trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
        profits = _numeric(trades["realized_profit"]) if not trades.empty else pd.Series(dtype=float)
        gross_profit = float(profits[profits > 0].sum()) if not profits.empty else 0.0
        gross_loss = abs(float(profits[profits < 0].sum())) if not profits.empty else 0.0
        equity = _numeric(daily["total_assets"]) if not daily.empty else pd.Series([self.options.initial_cash])
        drawdown = equity / equity.cummax() - 1.0
        return {
            "strategy": strategy,
            "net_profit": _safe_float(profits.sum()) if not profits.empty else 0.0,
            "PF": _safe_float(gross_profit / gross_loss) if gross_loss else (None if gross_profit == 0 else float("inf")),
            "DD": _safe_float(drawdown.min()) if not drawdown.empty else 0.0,
            "win_rate": _safe_float((profits > 0).mean()) if not profits.empty else None,
            "total_trades": int(len(trades)),
            "final_assets": _safe_float(self.options.initial_cash + profits.sum()) if not profits.empty else self.options.initial_cash,
            "capital_utilization": _safe_float(_numeric(daily["capital_utilization"]).mean()) if not daily.empty else None,
            "average_holding_days": _safe_float(_numeric(trades["holding_days"]).mean()) if not trades.empty else None,
            "cost_paid": _safe_float(_numeric(trades["cost_paid"]).sum()) if "cost_paid" in trades.columns else 0.0,
            "exit_reason_counts": dict(Counter(trades["exit_reason"])) if "exit_reason" in trades.columns else {},
        }

    def dd_attribution_audit(self, trades: pd.DataFrame, daily: pd.DataFrame, snapshots: pd.DataFrame) -> dict[str, Any]:
        losses = trades.copy()
        if losses.empty:
            return {"top20_loss_trades": [], "loss_contribution_pct_top20": None, "largest_drawdown_periods": [], "max_concurrent_positions": 0}
        losses["realized_profit"] = _numeric(losses["realized_profit"])
        loss_trades = losses[losses["realized_profit"] < 0].sort_values("realized_profit").head(20).copy()
        total_loss = abs(float(losses.loc[losses["realized_profit"] < 0, "realized_profit"].sum()))
        top_loss = abs(float(loss_trades["realized_profit"].sum()))
        equity = daily.copy()
        if not equity.empty:
            equity["drawdown"] = _numeric(equity["total_assets"]) / _numeric(equity["total_assets"]).cummax() - 1.0
            dd_rows = equity.sort_values("drawdown").head(10)
            largest_dd = [
                {
                    "date": row["date"].date().isoformat(),
                    "total_assets": _safe_float(row["total_assets"]),
                    "drawdown": _safe_float(row["drawdown"]),
                    "open_position_count": int(row["open_position_count"]),
                    "capital_utilization": _safe_float(row["capital_utilization"]),
                }
                for _, row in dd_rows.iterrows()
            ]
        else:
            largest_dd = []
        return {
            "top20_loss_trades": self.records(loss_trades, ["entry_date", "exit_date", "code", "buy_amount", "realized_profit", "realized_return", "exit_reason", "entry_downside_bad_proba", "allocation_weight", "normalized_weight"]),
            "loss_contribution_pct_top20": _safe_float(top_loss / total_loss) if total_loss else None,
            "largest_drawdown_periods": largest_dd,
            "max_concurrent_positions": int(_numeric(daily["open_position_count"]).max()) if "open_position_count" in daily.columns and not daily.empty else 0,
            "max_daily_position_count_from_snapshots": int(snapshots.groupby("date")["code"].count().max()) if not snapshots.empty else 0,
        }

    def concentration_audit(self, snapshots: pd.DataFrame) -> dict[str, Any]:
        if snapshots.empty:
            return {}
        daily = []
        for date, group in snapshots.groupby("date", sort=True):
            weights = _numeric(group["portfolio_weight"]).sort_values(ascending=False).reset_index(drop=True)
            position_weights = _numeric(group["position_weight"]).sort_values(ascending=False).reset_index(drop=True)
            daily.append(
                {
                    "date": pd.Timestamp(date).date().isoformat(),
                    "largest_position_weight": _safe_float(weights.iloc[0]) if len(weights) else None,
                    "top2_weight": _safe_float(weights.head(2).sum()),
                    "top3_weight": _safe_float(weights.head(3).sum()),
                    "largest_within_invested_weight": _safe_float(position_weights.iloc[0]) if len(position_weights) else None,
                }
            )
        daily_df = pd.DataFrame(daily)
        return {
            "daily_concentration_top10": daily_df.sort_values("largest_position_weight", ascending=False).head(10).to_dict("records"),
            "largest_position_weight_mean": _safe_float(_numeric(daily_df["largest_position_weight"]).mean()),
            "largest_position_weight_p90": _safe_float(_numeric(daily_df["largest_position_weight"]).quantile(0.90)),
            "largest_position_weight_max": _safe_float(_numeric(daily_df["largest_position_weight"]).max()),
            "top2_weight_mean": _safe_float(_numeric(daily_df["top2_weight"]).mean()),
            "top3_weight_mean": _safe_float(_numeric(daily_df["top3_weight"]).mean()),
            "largest_within_invested_weight_mean": _safe_float(_numeric(daily_df["largest_within_invested_weight"]).mean()),
        }

    def downside_exposure_audit(self, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty:
            return {}
        frame = trades.copy()
        frame["entry_downside_bad_proba"] = _numeric(frame["entry_downside_bad_proba"])
        frame["realized_return"] = _numeric(frame["realized_return"])
        frame["buy_amount"] = _numeric(frame["buy_amount"])
        frame["high_downside"] = frame["entry_downside_bad_proba"] >= 0.40
        grouped = []
        for flag, group in frame.groupby("high_downside", dropna=False):
            grouped.append(
                {
                    "bucket": "downside_proba_ge_0.40" if flag else "downside_proba_lt_0.40",
                    "trade_count": int(len(group)),
                    "buy_amount_sum": _safe_float(group["buy_amount"].sum()),
                    "avg_buy_amount": _safe_float(group["buy_amount"].mean()),
                    "avg_realized_return": _safe_float(group["realized_return"].mean()),
                    "loss_rate": _safe_float((group["realized_return"] < 0).mean()),
                    "avg_allocation_weight": _safe_float(_numeric(group["allocation_weight"]).mean()),
                    "avg_normalized_weight": _safe_float(_numeric(group["normalized_weight"]).mean()),
                }
            )
        corr = frame[["entry_downside_bad_proba", "buy_amount", "realized_return"]].corr(numeric_only=True)
        return {
            "by_downside_bucket": grouped,
            "downside_proba_buy_amount_corr": _safe_float(corr.loc["entry_downside_bad_proba", "buy_amount"]) if "entry_downside_bad_proba" in corr.index and "buy_amount" in corr.columns else None,
            "downside_proba_realized_return_corr": _safe_float(corr.loc["entry_downside_bad_proba", "realized_return"]) if "entry_downside_bad_proba" in corr.index and "realized_return" in corr.columns else None,
            "top20_high_downside_allocations": self.records(frame.sort_values(["entry_downside_bad_proba", "buy_amount"], ascending=[False, False]).head(20), ["entry_date", "exit_date", "code", "buy_amount", "entry_downside_bad_proba", "allocation_weight", "normalized_weight", "realized_return", "exit_reason"]),
        }

    def variant_comparison(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        minimum = [row["strategy"] for row in rows if self.minimum_passed(row)]
        ideal = [row["strategy"] for row in rows if self.ideal_passed(row)]
        best = self.best_variant(rows)
        return {
            "best_variant": best.get("strategy") if best else None,
            "best_variant_reason": self.best_variant_reason(best),
            "variants_meeting_minimum_target": minimum,
            "variants_meeting_ideal_target": ideal,
            "ready_for_phase13": bool(minimum),
            "recommended_next_phase": "Phase13 limited OOS/year robustness check" if minimum else "Phase12-C3 DD guard refinement",
        }

    def minimum_passed(self, row: dict[str, Any]) -> bool:
        return (
            (_safe_float(row.get("PF")) or 0.0) >= 1.8
            and (_safe_float(row.get("DD")) or -1.0) >= -0.12
            and (_safe_float(row.get("capital_utilization")) or 0.0) >= 0.50
        )

    def ideal_passed(self, row: dict[str, Any]) -> bool:
        return (
            (_safe_float(row.get("PF")) or 0.0) >= 2.0
            and (_safe_float(row.get("DD")) or -1.0) >= -0.10
            and (_safe_float(row.get("capital_utilization")) or 0.0) >= 0.70
        )

    def best_variant(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None

        def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
            pass_bonus = 10**7 if self.minimum_passed(row) else 0.0
            pf = _safe_float(row.get("PF")) or 0.0
            dd = _safe_float(row.get("DD")) or -1.0
            profit = _safe_float(row.get("net_profit")) or -10**18
            utilization = _safe_float(row.get("capital_utilization")) or 0.0
            return (pass_bonus + pf * 100_000 + profit, dd, utilization, profit)

        return max(rows, key=key)

    def best_variant_reason(self, row: dict[str, Any] | None) -> str:
        if not row:
            return "No variants were evaluated."
        return (
            f"{row['strategy']} net_profit={row.get('net_profit'):.0f}, PF={row.get('PF'):.4f}, "
            f"DD={row.get('DD'):.4f}, utilization={row.get('capital_utilization'):.4f}."
        )

    def infer_main_dd_cause(self, dd_audit: dict[str, Any], concentration: dict[str, Any], downside: dict[str, Any]) -> str:
        largest = _safe_float(concentration.get("largest_position_weight_max")) or 0.0
        top2 = _safe_float(concentration.get("top2_weight_mean")) or 0.0
        high_bucket = next((row for row in downside.get("by_downside_bucket", []) if row.get("bucket") == "downside_proba_ge_0.40"), {})
        high_loss_rate = _safe_float(high_bucket.get("loss_rate")) or 0.0
        high_buy = _safe_float(high_bucket.get("avg_buy_amount")) or 0.0
        if largest >= 0.35:
            return "single_name_concentration"
        if top2 >= 0.55:
            return "multi_position_concentration"
        if high_loss_rate >= 0.50 and high_buy >= self.options.initial_cash * 0.15:
            return "oversized_high_downside_exposure"
        if dd_audit.get("max_concurrent_positions", 0) >= 4:
            return "multiple_concurrent_losers"
        return "mixed_dd_cause"

    def recommendation(
        self,
        rows: list[dict[str, Any]],
        dd_audit: dict[str, Any],
        concentration: dict[str, Any],
        downside: dict[str, Any],
        leakage: dict[str, Any],
    ) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"ready_for_phase13": False, "recommended_next_phase": "Fix Phase12-C2 leakage blockers"}
        comparison = self.variant_comparison(rows)
        return {
            "main_dd_cause": self.infer_main_dd_cause(dd_audit, concentration, downside),
            "best_variant": comparison["best_variant"],
            "best_variant_reason": comparison["best_variant_reason"],
            "utilization_vs_dd_summary": "Normalized allocation can preserve utilization, but only variants meeting PF/DD/utilization minimum should advance.",
            "ready_for_phase13": comparison["ready_for_phase13"],
            "recommended_next_phase": comparison["recommended_next_phase"],
        }

    def dataset_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(data)),
            "unique_codes": int(data["code"].nunique()) if not data.empty else 0,
            "candidate_days": int(data["date"].nunique()) if not data.empty else 0,
            "date_range": {
                "min": data["date"].min().date().isoformat() if not data.empty else None,
                "max": data["date"].max().date().isoformat() if not data.empty else None,
            },
            "source_artifact": str(self.root / ARTIFACT_PATH),
        }

    def conditions(self) -> dict[str, Any]:
        return {
            "period": {"start": START_DATE, "end": END_DATE},
            "base_strategy": "C2_dynamic_normalized_B5_2_exit",
            "opportunity_drop_threshold": self.options.opportunity_drop_threshold,
            "opportunity_rank_floor": self.options.opportunity_rank_floor,
            "stop_loss": self.options.stop_loss_rate,
            "daily_buy_budget": self.options.daily_buy_budget,
            "max_positions": self.options.max_positions,
            "cost_rate": self.options.cost_rate,
            "variants": [spec.__dict__ for spec in VARIANTS],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "12-C2",
            "scope": "2025 utilization/DD attribution and small variant check only",
            "full_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_added": False,
            "profile_modified": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
        }

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_only_for_evaluation": FUTURE_EVAL_COLUMNS,
            "future_columns_used_as_features": [],
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def records(self, frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
        rows = []
        for _, row in frame.iterrows():
            item = {}
            for column in columns:
                value = row.get(column)
                if isinstance(value, pd.Timestamp):
                    item[column] = value.date().isoformat()
                elif isinstance(value, float):
                    item[column] = _safe_float(value)
                else:
                    item[column] = value
            rows.append(item)
        return rows

    def save_outputs(self, report: dict[str, Any]) -> Phase12C2Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12C2Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-C2 Utilization Without DD Explosion",
            "",
            "## Variant Results",
            "",
            self.table(report.get("variant_results", []), ["strategy", "net_profit", "PF", "DD", "win_rate", "final_assets", "capital_utilization", "average_holding_days", "cost_paid", "exit_reason_counts"]),
            "",
            "## DD Attribution",
            "",
            self.table([report.get("dd_attribution_audit", {})], ["loss_contribution_pct_top20", "max_concurrent_positions", "max_daily_position_count_from_snapshots"]),
            "",
            "## Concentration Audit",
            "",
            self.table([report.get("concentration_audit", {})], ["largest_position_weight_mean", "largest_position_weight_p90", "largest_position_weight_max", "top2_weight_mean", "top3_weight_mean", "largest_within_invested_weight_mean"]),
            "",
            "## Downside Exposure",
            "",
            self.table(report.get("downside_exposure_audit", {}).get("by_downside_bucket", []), ["bucket", "trade_count", "buy_amount_sum", "avg_buy_amount", "avg_realized_return", "loss_rate", "avg_allocation_weight", "avg_normalized_weight"]),
            "",
            "## Comparison",
            "",
            self.table([report.get("variant_comparison", {})], ["best_variant", "best_variant_reason", "variants_meeting_minimum_target", "variants_meeting_ideal_target", "ready_for_phase13", "recommended_next_phase"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_only_for_evaluation", "future_columns_used_as_features", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["main_dd_cause", "best_variant", "utilization_vs_dd_summary", "ready_for_phase13", "recommended_next_phase"]),
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
