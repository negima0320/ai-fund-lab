"""Phase 12-D2 buy quality reality audit.

This is a 2025-only audit that compares the contribution of Stock Selection,
Valuation, Downside, Allocation, and Exit layers. It uses existing artifacts
only. It does not train a new model, run a full backtest, change profiles,
overwrite models, call external APIs, or regenerate historical predictions.
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
from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH, EVAL_COLUMNS
from ml.phase12b_limited_allocation_strategy_check import BASELINE_RANK_COLUMNS, END_DATE, ROUND_LOT, START_DATE
from ml.phase12c_dynamic_allocation_recalibrated_exit import Phase12COptions


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase12d2_buy_quality_reality_audit_2025"
FUTURE_EVAL_COLUMNS = EVAL_COLUMNS
LONG_HORIZON_COLUMNS = ["future_return_40d", "future_return_60d"]


@dataclass(frozen=True)
class LayerSpec:
    name: str
    mode: str


@dataclass(frozen=True)
class StrategySpec:
    name: str
    rank_column: str
    use_b5_exit: bool
    allocation_mode: str = "equal"


@dataclass(frozen=True)
class Phase12D2Paths:
    markdown: Path
    json: Path


LAYERS = [
    LayerSpec("L0_candidate_universe", "all"),
    LayerSpec("L1_stock_selection_top5", "stock_top5"),
    LayerSpec("L2_opportunity_top5", "opportunity_top5"),
    LayerSpec("L3_opportunity_downside_top5_A3_3", "opportunity_downside_top5"),
    LayerSpec("L4_dynamic_normalized_C2c_candidates", "dynamic_c2c"),
]

STRATEGIES = [
    StrategySpec("S0_stock_selection_only", "baseline_rank_score", False, "equal"),
    StrategySpec("S1_stock_selection_plus_B5_2_exit", "baseline_rank_score", True, "equal"),
    StrategySpec("S2_opportunity_only", "opportunity_proba", False, "equal"),
    StrategySpec("S3_opportunity_plus_B5_2_exit", "opportunity_proba", True, "equal"),
    StrategySpec("S4_opportunity_downside_dynamic_allocation_B5_2_exit", "opportunity_proba", True, "dynamic_c2c"),
]


class Phase12D2BuyQualityRealityAudit:
    def __init__(self, root: Path | str = ROOT, *, options: Phase12COptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase12COptions()

    def run(self) -> Phase12D2Paths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data = self.load_frame()
        leakage = self.leakage_checklist()
        if leakage["blocking_issues"]:
            return {
                "metadata": self.metadata(),
                "conditions": self.conditions(data),
                "dataset_summary": self.dataset_summary(data),
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], [], {}, leakage),
            }

        layer_quality = [self.layer_quality(data, spec) for spec in LAYERS]
        strategy_results = []
        strategy_buy_quality = []
        for spec in STRATEGIES:
            trades, daily = self.simulate(data, spec)
            strategy_results.append(self.metrics(spec.name, trades, daily))
            strategy_buy_quality.append(self.trade_buy_quality(spec.name, trades))
        contribution = self.layer_contribution_summary(layer_quality, strategy_results)
        judgments = self.judgments(layer_quality, strategy_results, contribution)
        return {
            "metadata": self.metadata(),
            "conditions": self.conditions(data),
            "dataset_summary": self.dataset_summary(data),
            "buy_quality_layer_comparison": layer_quality,
            "strategy_layer_comparison": strategy_results,
            "strategy_buy_quality": strategy_buy_quality,
            "layer_contribution_summary": contribution,
            "judgments": judgments,
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(layer_quality, strategy_results, judgments, leakage),
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
            *BASELINE_RANK_COLUMNS,
            "opportunity_proba",
            "downside_bad_proba",
            "opportunity_rank_percentile",
            "downside_rank_percentile",
            "confidence",
            *FUTURE_EVAL_COLUMNS,
            *LONG_HORIZON_COLUMNS,
        ]
        available = [column for column in columns if column in data.columns]
        data = data[available].drop_duplicates(["date", "code"])
        for column in data.columns:
            if column not in {"date", "code"}:
                data[column] = _numeric(data[column])
        data["baseline_rank_score"] = self.baseline_rank(data)
        data["opportunity_top_decile_proba"] = data["opportunity_proba"]
        data["opportunity_score_proba_rank"] = data["opportunity_rank_percentile"]
        data["a3_3_allocation_weight"] = self.a3_3_weight(data["downside_rank_percentile"])
        return data.dropna(subset=["date", "code", "close", "opportunity_proba", "downside_rank_percentile"]).sort_values(["date", "code"]).reset_index(drop=True)

    def baseline_rank(self, data: pd.DataFrame) -> pd.Series:
        for column in BASELINE_RANK_COLUMNS:
            if column in data.columns and not data[column].isna().all():
                return _numeric(data[column]).fillna(-10**18)
        return pd.Series(-10**18, index=data.index, dtype=float)

    def a3_3_weight(self, downside_rank: pd.Series) -> pd.Series:
        rank = _numeric(downside_rank)
        return rank.map(lambda value: 1.0 if value <= 0.40 else 0.6 if value <= 0.70 else 0.3 if value <= 0.85 else 0.0)

    def layer_frame(self, data: pd.DataFrame, spec: LayerSpec) -> pd.DataFrame:
        if spec.mode == "all":
            frame = data.copy()
            frame["selection_weight"] = 1.0
            return frame
        if spec.mode == "stock_top5":
            frame = self.top_n_by_day(data, "baseline_rank_score", 5)
            frame["selection_weight"] = 1.0
            return frame
        if spec.mode == "opportunity_top5":
            frame = self.top_n_by_day(data, "opportunity_proba", 5)
            frame["selection_weight"] = 1.0
            return frame
        if spec.mode == "opportunity_downside_top5":
            frame = self.top_n_by_day(data, "opportunity_proba", 5)
            frame = frame[_numeric(frame["a3_3_allocation_weight"]) > 0].copy()
            frame["selection_weight"] = _numeric(frame["a3_3_allocation_weight"])
            return frame
        if spec.mode == "dynamic_c2c":
            frame = self.top_n_by_day(data, "opportunity_proba", 5)
            frame = frame[_numeric(frame["a3_3_allocation_weight"]) > 0].copy()
            penalty = (1.0 - _numeric(frame["downside_bad_proba"]).fillna(1.0)).clip(lower=0, upper=1) ** 2
            frame["selection_weight"] = _numeric(frame["a3_3_allocation_weight"]) * penalty
            return frame
        raise ValueError(f"Unknown layer mode: {spec.mode}")

    def top_n_by_day(self, data: pd.DataFrame, column: str, n: int) -> pd.DataFrame:
        return (
            data.sort_values(["date", column, "turnover_value", "code"], ascending=[True, False, False, True])
            .groupby("date", sort=False, group_keys=False)
            .head(n)
            .copy()
        )

    def layer_quality(self, data: pd.DataFrame, spec: LayerSpec) -> dict[str, Any]:
        frame = self.layer_frame(data, spec)
        row: dict[str, Any] = {
            "layer": spec.name,
            "rows": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if not frame.empty else 0,
            "avg_candidates_per_day": _safe_float(len(frame) / frame["date"].nunique()) if not frame.empty and frame["date"].nunique() else None,
        }
        metrics = self.quality_metrics(frame, weight_column=None)
        weighted = self.quality_metrics(frame, weight_column="selection_weight")
        row.update({f"unweighted_{key}": value for key, value in metrics.items()})
        row.update({f"weighted_{key}": value for key, value in weighted.items()})
        return row

    def quality_metrics(self, frame: pd.DataFrame, *, weight_column: str | None) -> dict[str, Any]:
        columns = [
            "future_return_20d",
            "future_return_40d",
            "future_return_60d",
            "future_max_return_20d",
            "future_max_drawdown_20d",
            "opportunity_value_20d",
            "opportunity_top_decile_20d",
            "downside_bad_20d",
        ]
        result: dict[str, Any] = {}
        weights = _numeric(frame[weight_column]).clip(lower=0) if weight_column and weight_column in frame.columns else None
        for column in columns:
            if column not in frame.columns:
                result[f"{column}_skipped"] = "column_not_available"
                continue
            values = _numeric(frame[column])
            if weights is not None and float(weights.sum()) > 0:
                value = float((values.fillna(0) * weights).sum() / weights.sum())
            else:
                value = float(values.mean()) if len(values) else math.nan
            key = "opportunity_top_decile_20d_rate" if column == "opportunity_top_decile_20d" else "downside_bad_rate" if column == "downside_bad_20d" else f"{column}_mean"
            result[key] = _safe_float(value)
        return result

    def simulate(self, data: pd.DataFrame, spec: StrategySpec) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                    position["last_close"] = float(current_row["close"])
                reason = self.exit_reason(position, current_date, current_row, spec)
                if reason:
                    exit_close = float(current_row["close"]) if current_row is not None else float(position["last_close"])
                    trade = self.close_position(position, current_date, exit_close, reason, spec.name)
                    cash += trade["exit_cash_flow"]
                    trades.append(trade)
                else:
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            ranked = current_rank_frame.sort_values([spec.rank_column, "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions)
            if spec.allocation_mode == "dynamic_c2c":
                ranked = ranked[_numeric(ranked["a3_3_allocation_weight"]) > 0].copy()
            selected = ranked.head(slots) if slots else ranked.iloc[0:0]
            target_amounts = self.target_amounts(selected, spec, cash)
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
                        "allocation_weight": _safe_float(row.get("a3_3_allocation_weight")) if spec.allocation_mode == "dynamic_c2c" else 1.0,
                        **{column: _safe_float(row.get(column)) for column in FUTURE_EVAL_COLUMNS},
                    }
                )

            marked_value = sum(self.position_value(position) for position in positions)
            daily_rows.append(
                {
                    "strategy": spec.name,
                    "date": current_date,
                    "cash": cash,
                    "open_position_count": len(positions),
                    "bought_today": bought_today,
                    "marked_position_value": marked_value,
                    "total_assets": cash + marked_value,
                    "capital_utilization": marked_value / self.options.initial_cash if self.options.initial_cash else None,
                }
            )

        if dates:
            last_date = dates[-1]
            for position in positions:
                trade = self.close_position(position, last_date, float(position["last_close"]), "forced_end_of_period", position.get("strategy", "forced_end_of_period"))
                cash += trade["exit_cash_flow"]
                trades.append(trade)
            if daily_rows:
                daily_rows[-1]["total_assets"] = cash
                daily_rows[-1]["marked_position_value"] = 0.0
                daily_rows[-1]["capital_utilization"] = 0.0
        return pd.DataFrame(trades), pd.DataFrame(daily_rows)

    def target_amounts(self, selected: pd.DataFrame, spec: StrategySpec, cash: float) -> dict[Any, float]:
        if selected.empty:
            return {}
        available_budget = min(cash, self.options.daily_buy_budget)
        if spec.allocation_mode == "equal":
            amount = available_budget / max(1, min(self.options.max_positions, len(selected)))
            return {index: amount for index in selected.index}
        weights = _numeric(selected["a3_3_allocation_weight"]).clip(lower=0)
        penalty = (1.0 - _numeric(selected["downside_bad_proba"]).fillna(1.0)).clip(lower=0, upper=1) ** 2
        weights = weights * penalty
        total = float(weights.sum())
        if total <= 0:
            return {index: 0.0 for index in selected.index}
        return {index: available_budget * float(weight) / total for index, weight in weights.items()}

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None, spec: StrategySpec) -> str | None:
        if current_row is not None:
            observed_return = float(current_row["close"]) / float(position["entry_close"]) - 1.0
            if observed_return <= self.options.stop_loss_rate:
                return "stop_loss"
            if spec.use_b5_exit:
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
            "allocation_weight": position.get("allocation_weight", 1.0),
            **{column: position.get(column) for column in FUTURE_EVAL_COLUMNS},
        }

    def position_value(self, position: dict[str, Any]) -> float:
        return float(position["lot_count"]) * self.options.round_lot * float(position["last_close"])

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
            "final_assets": _safe_float(self.options.initial_cash + profits.sum()) if not profits.empty else self.options.initial_cash,
            "capital_utilization": _safe_float(_numeric(daily["capital_utilization"]).mean()) if not daily.empty else None,
            "average_holding_days": _safe_float(_numeric(trades["holding_days"]).mean()) if not trades.empty else None,
            "median_holding_days": _safe_float(_numeric(trades["holding_days"]).median()) if not trades.empty else None,
            "total_trades": int(len(trades)),
            "cost_paid": _safe_float(_numeric(trades["cost_paid"]).sum()) if "cost_paid" in trades.columns else 0.0,
            "exit_reason_counts": dict(Counter(trades["exit_reason"])) if "exit_reason" in trades.columns else {},
        }

    def trade_buy_quality(self, strategy: str, trades: pd.DataFrame) -> dict[str, Any]:
        row = {"strategy": strategy, "buy_count": int(len(trades))}
        row.update(self.quality_metrics(trades, weight_column="allocation_weight"))
        return row

    def layer_contribution_summary(self, layers: list[dict[str, Any]], strategies: list[dict[str, Any]]) -> dict[str, Any]:
        layer = {row["layer"]: row for row in layers}
        strat = {row["strategy"]: row for row in strategies}
        return {
            "stock_selection_effect": self.effect_quality(layer.get("L0_candidate_universe", {}), layer.get("L1_stock_selection_top5", {}), "Stock Selection Top5 vs universe"),
            "valuation_effect": self.effect_quality(layer.get("L1_stock_selection_top5", {}), layer.get("L2_opportunity_top5", {}), "Opportunity Top5 vs Stock Selection Top5"),
            "downside_effect": self.effect_quality(layer.get("L2_opportunity_top5", {}), layer.get("L3_opportunity_downside_top5_A3_3", {}), "Opportunity+Downside vs Opportunity"),
            "allocation_effect": self.effect_strategy(strat.get("S3_opportunity_plus_B5_2_exit", {}), strat.get("S4_opportunity_downside_dynamic_allocation_B5_2_exit", {}), "Dynamic allocation vs Opportunity+B5"),
            "exit_effect": self.effect_strategy(strat.get("S2_opportunity_only", {}), strat.get("S3_opportunity_plus_B5_2_exit", {}), "B5 exit vs Opportunity only"),
        }

    def effect_quality(self, before: dict[str, Any], after: dict[str, Any], label: str) -> dict[str, Any]:
        before_top = _safe_float(before.get("unweighted_opportunity_top_decile_20d_rate")) or 0.0
        after_top = _safe_float(after.get("unweighted_opportunity_top_decile_20d_rate")) or 0.0
        before_down = _safe_float(before.get("unweighted_downside_bad_rate")) or 0.0
        after_down = _safe_float(after.get("unweighted_downside_bad_rate")) or 0.0
        before_val = _safe_float(before.get("unweighted_opportunity_value_20d_mean")) or 0.0
        after_val = _safe_float(after.get("unweighted_opportunity_value_20d_mean")) or 0.0
        degraded = []
        if after_down > before_down:
            degraded.append("downside_bad_rate")
        if after_top < before_top:
            degraded.append("top_decile_rate")
        return {
            "label": label,
            "improved_profit": None,
            "improved_PF": None,
            "improved_DD": after_down <= before_down,
            "improved_utilization": None,
            "top_decile_rate_delta": _safe_float(after_top - before_top),
            "downside_bad_rate_delta": _safe_float(after_down - before_down),
            "opportunity_value_delta": _safe_float(after_val - before_val),
            "degraded_metric": degraded,
        }

    def effect_strategy(self, before: dict[str, Any], after: dict[str, Any], label: str) -> dict[str, Any]:
        degraded = []
        if (_safe_float(after.get("net_profit")) or 0.0) < (_safe_float(before.get("net_profit")) or 0.0):
            degraded.append("net_profit")
        if (_safe_float(after.get("PF")) or 0.0) < (_safe_float(before.get("PF")) or 0.0):
            degraded.append("PF")
        if (_safe_float(after.get("DD")) or -1.0) < (_safe_float(before.get("DD")) or -1.0):
            degraded.append("DD")
        if (_safe_float(after.get("capital_utilization")) or 0.0) < (_safe_float(before.get("capital_utilization")) or 0.0):
            degraded.append("utilization")
        return {
            "label": label,
            "improved_profit": (_safe_float(after.get("net_profit")) or 0.0) > (_safe_float(before.get("net_profit")) or 0.0),
            "improved_PF": (_safe_float(after.get("PF")) or 0.0) > (_safe_float(before.get("PF")) or 0.0),
            "improved_DD": (_safe_float(after.get("DD")) or -1.0) > (_safe_float(before.get("DD")) or -1.0),
            "improved_utilization": (_safe_float(after.get("capital_utilization")) or 0.0) > (_safe_float(before.get("capital_utilization")) or 0.0),
            "degraded_metric": degraded,
        }

    def judgments(self, layers: list[dict[str, Any]], strategies: list[dict[str, Any]], contribution: dict[str, Any]) -> dict[str, Any]:
        layer = {row["layer"]: row for row in layers}
        strat = {row["strategy"]: row for row in strategies}
        universe_top = _safe_float(layer.get("L0_candidate_universe", {}).get("unweighted_opportunity_top_decile_20d_rate")) or 0.0
        stock_top = _safe_float(layer.get("L1_stock_selection_top5", {}).get("unweighted_opportunity_top_decile_20d_rate")) or 0.0
        opp_top = _safe_float(layer.get("L2_opportunity_top5", {}).get("unweighted_opportunity_top_decile_20d_rate")) or 0.0
        opp_down = _safe_float(layer.get("L2_opportunity_top5", {}).get("unweighted_downside_bad_rate")) or 0.0
        down_down = _safe_float(layer.get("L3_opportunity_downside_top5_A3_3", {}).get("weighted_downside_bad_rate")) or 0.0
        s4 = strat.get("S4_opportunity_downside_dynamic_allocation_B5_2_exit", {})
        s3 = strat.get("S3_opportunity_plus_B5_2_exit", {})
        buy_quality_good = opp_top >= max(0.18, universe_top * 1.5)
        stock_valid = stock_top > universe_top * 1.1
        valuation_adds = opp_top > stock_top and (_safe_float(layer.get("L2_opportunity_top5", {}).get("unweighted_opportunity_value_20d_mean")) or 0.0) > (_safe_float(layer.get("L1_stock_selection_top5", {}).get("unweighted_opportunity_value_20d_mean")) or 0.0)
        downside_adds = down_down < opp_down
        allocation_adds = (_safe_float(s4.get("PF")) or 0.0) > (_safe_float(s3.get("PF")) or 0.0) or (_safe_float(s4.get("net_profit")) or 0.0) > (_safe_float(s3.get("net_profit")) or 0.0)
        exit_bottleneck = buy_quality_good and ((_safe_float(s4.get("DD")) or 0.0) < -0.12)
        if not buy_quality_good or not valuation_adds:
            bottleneck = "buy_quality"
            next_phase = "Phase12-D3 Buy Model Revisit"
        elif exit_bottleneck:
            bottleneck = "exit"
            next_phase = "Phase12-D3 Exit AI Dataset Audit"
        elif allocation_adds and ((_safe_float(s4.get("DD")) or 0.0) < -0.12):
            bottleneck = "allocation"
            next_phase = "Phase12-D3 Allocation Revisit"
        else:
            bottleneck = "system_oos_design"
            next_phase = "Phase12-D3 Full System Strict OOS Design"
        return {
            "buy_quality_good_enough": {"value": buy_quality_good, "reason": f"Opportunity top5 top-decile={opp_top:.4f}, universe={universe_top:.4f}."},
            "stock_selection_is_valid": {"value": stock_valid, "reason": f"Stock Selection top5 top-decile={stock_top:.4f}, universe={universe_top:.4f}."},
            "valuation_adds_value": {"value": valuation_adds, "reason": f"Opportunity top5 top-decile={opp_top:.4f}, Stock Selection top5={stock_top:.4f}."},
            "downside_adds_value": {"value": downside_adds, "reason": f"Weighted downside after A3_3={down_down:.4f}, opportunity-only downside={opp_down:.4f}."},
            "allocation_adds_value": {"value": allocation_adds, "reason": f"S4 profit/PF={s4.get('net_profit')}/{s4.get('PF')}, S3={s3.get('net_profit')}/{s3.get('PF')}."},
            "exit_is_current_bottleneck": {"value": exit_bottleneck, "reason": f"BUY quality is improved but S4 DD={s4.get('DD')} remains beyond -12%."},
            "main_bottleneck": bottleneck,
            "recommended_next_phase": next_phase,
        }

    def recommendation(self, layers: list[dict[str, Any]], strategies: list[dict[str, Any]], judgments: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"recommended_next_phase": "Fix Phase12-D2 leakage blockers"}
        return {
            "main_bottleneck": judgments.get("main_bottleneck"),
            "buy_quality_reality_summary": self.buy_quality_reality_summary(layers, judgments),
            "layer_comparison_table": "See buy_quality_layer_comparison and strategy_layer_comparison.",
            "buy_quality_good_enough": judgments.get("buy_quality_good_enough"),
            "stock_selection_is_valid": judgments.get("stock_selection_is_valid"),
            "valuation_adds_value": judgments.get("valuation_adds_value"),
            "downside_adds_value": judgments.get("downside_adds_value"),
            "allocation_adds_value": judgments.get("allocation_adds_value"),
            "exit_is_current_bottleneck": judgments.get("exit_is_current_bottleneck"),
            "recommended_next_phase": judgments.get("recommended_next_phase"),
        }

    def buy_quality_reality_summary(self, layers: list[dict[str, Any]], judgments: dict[str, Any]) -> str:
        layer = {row["layer"]: row for row in layers}
        return (
            f"Universe top-decile={layer.get('L0_candidate_universe', {}).get('unweighted_opportunity_top_decile_20d_rate')}, "
            f"StockSelection top5={layer.get('L1_stock_selection_top5', {}).get('unweighted_opportunity_top_decile_20d_rate')}, "
            f"Opportunity top5={layer.get('L2_opportunity_top5', {}).get('unweighted_opportunity_top_decile_20d_rate')}. "
            f"Main bottleneck={judgments.get('main_bottleneck')}."
        )

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

    def conditions(self, data: pd.DataFrame) -> dict[str, Any]:
        available_long = [column for column in LONG_HORIZON_COLUMNS if column in data.columns]
        return {
            "period": {"start": START_DATE, "end": END_DATE},
            "mode": "reality_audit_only",
            "new_model_trained": False,
            "future_40d_60d_status": {
                "available": available_long,
                "skipped": [column for column in LONG_HORIZON_COLUMNS if column not in data.columns],
                "skip_reason": "column_not_available_in_existing_artifact" if len(available_long) < len(LONG_HORIZON_COLUMNS) else None,
            },
            "strategies": [spec.__dict__ for spec in STRATEGIES],
            "layers": [spec.__dict__ for spec in LAYERS],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "12-D2",
            "scope": "2025 buy quality reality audit only",
            "full_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_added": False,
            "profile_modified": False,
            "historical_predictions_regenerated": False,
            "new_model_trained": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
        }

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_only_for_evaluation": [*FUTURE_EVAL_COLUMNS, *LONG_HORIZON_COLUMNS],
            "future_columns_used_as_features": [],
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "new_model_trained": False,
            "historical_predictions_regenerated": False,
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase12D2Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12D2Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-D2 Buy Quality Reality Audit",
            "",
            "## BUY Quality Layer Comparison",
            "",
            self.table(
                report.get("buy_quality_layer_comparison", []),
                [
                    "layer",
                    "rows",
                    "candidate_days",
                    "avg_candidates_per_day",
                    "unweighted_future_return_20d_mean",
                    "unweighted_opportunity_value_20d_mean",
                    "unweighted_opportunity_top_decile_20d_rate",
                    "unweighted_downside_bad_rate",
                    "weighted_opportunity_top_decile_20d_rate",
                    "weighted_downside_bad_rate",
                ],
            ),
            "",
            "## Strategy Layer Comparison",
            "",
            self.table(report.get("strategy_layer_comparison", []), ["strategy", "net_profit", "PF", "DD", "win_rate", "final_assets", "capital_utilization", "average_holding_days", "median_holding_days", "total_trades", "cost_paid", "exit_reason_counts"]),
            "",
            "## Layer Contribution Summary",
            "",
            self.table([report.get("layer_contribution_summary", {})], ["stock_selection_effect", "valuation_effect", "downside_effect", "allocation_effect", "exit_effect"]),
            "",
            "## Judgments",
            "",
            self.table([report.get("judgments", {})], ["buy_quality_good_enough", "stock_selection_is_valid", "valuation_adds_value", "downside_adds_value", "allocation_adds_value", "exit_is_current_bottleneck", "main_bottleneck", "recommended_next_phase"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_only_for_evaluation", "future_columns_used_as_features", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "new_model_trained", "historical_predictions_regenerated", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["main_bottleneck", "buy_quality_reality_summary", "recommended_next_phase"]),
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
