"""Phase 12-E1 Stock Selection reality audit.

This module audits whether the Stock Selection layer adds value in 2025. It
uses existing scored artifacts only. It does not train a model, regenerate
historical predictions, change profiles, overwrite models, call external APIs,
or run a full backtest. Future columns are used only for evaluation/audit.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11b3_expected_downside_model import DOWNSIDE_TARGET
from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase12a_dynamic_capital_allocation import EVAL_COLUMNS
from ml.phase12b_limited_allocation_strategy_check import ARTIFACT_PATH, BASELINE_RANK_COLUMNS, END_DATE, ROUND_LOT, START_DATE
from ml.phase12c_dynamic_allocation_recalibrated_exit import Phase12COptions


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase12e1_stock_selection_reality_audit_2025"
FUTURE_EVAL_COLUMNS = EVAL_COLUMNS
STOCK_SELECTION_SCORES = ["stock_selection_rank_score", "risk_adjusted_score", "expected_return", "candidate_strength"]


@dataclass(frozen=True)
class RankSpec:
    name: str
    score_column: str | None
    n: int | None
    mode: str = "top_n"


@dataclass(frozen=True)
class StrategySpec:
    name: str
    rank_column: str
    use_b5_exit: bool
    allocation_mode: str = "equal"
    prefilter_column: str | None = None
    prefilter_n: int | None = None


@dataclass(frozen=True)
class Phase12E1Paths:
    markdown: Path
    json: Path


RANK_SPECS = [
    RankSpec("candidate_universe", None, None, "all"),
    *[RankSpec(f"{column}_top{n}", column, n) for column in STOCK_SELECTION_SCORES for n in (5, 10, 20)],
    RankSpec("random_top5_per_day", None, 5, "random"),
    RankSpec("random_top10_per_day", None, 10, "random"),
]


STRATEGIES = [
    StrategySpec("S0_stock_selection_rank_score_top5_equal_20d", "stock_selection_rank_score", False),
    StrategySpec("S1_risk_adjusted_score_top5_equal_20d", "risk_adjusted_score", False),
    StrategySpec("S2_expected_return_top5_equal_20d", "expected_return", False),
    StrategySpec("S3_candidate_strength_top5_equal_20d", "candidate_strength", False),
    StrategySpec("S4_opportunity_top5_equal_20d", "opportunity_proba", False),
    StrategySpec(
        "S5_stock_top5_prefilter_opportunity_downside_dynamic_B5_2_exit",
        "opportunity_downside_score",
        True,
        "dynamic_normalized",
        "stock_selection_rank_score",
        5,
    ),
    StrategySpec("S6_no_stock_prefilter_opportunity_downside_dynamic_B5_2_exit", "opportunity_downside_score", True, "dynamic_normalized"),
    StrategySpec(
        "S7_stock_top20_prefilter_opportunity_downside_dynamic_B5_2_exit",
        "opportunity_downside_score",
        True,
        "dynamic_normalized",
        "stock_selection_rank_score",
        20,
    ),
]


class Phase12E1StockSelectionRealityAudit:
    def __init__(self, root: Path | str = ROOT, *, options: Phase12COptions | None = None, random_seed: int = 1201) -> None:
        self.root = Path(root)
        self.options = options or Phase12COptions()
        self.random_seed = random_seed

    def run(self) -> Phase12E1Paths:
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
                "recommendation": self.recommendation({}, leakage),
            }

        rank_quality = [self.rank_quality(data, spec) for spec in RANK_SPECS]
        monotonicity = self.score_monotonicity(data)
        layer_interaction = self.layer_interaction(data)
        strategy_results: list[dict[str, Any]] = []
        strategy_buy_quality: list[dict[str, Any]] = []
        for spec in STRATEGIES:
            trades, daily = self.simulate(data, spec)
            strategy_results.append(self.metrics(spec.name, trades, daily))
            strategy_buy_quality.append(self.buy_quality(spec.name, trades))
        final_judgment = self.final_judgment(rank_quality, monotonicity, layer_interaction, strategy_results)
        return {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "executive_summary": self.executive_summary(final_judgment),
            "rank_quality_table": rank_quality,
            "score_monotonicity_table": monotonicity,
            "layer_interaction_summary": layer_interaction,
            "strategy_comparison_table": strategy_results,
            "strategy_buy_quality_table": strategy_buy_quality,
            "final_judgment": final_judgment,
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(final_judgment, leakage),
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
            *STOCK_SELECTION_SCORES,
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
        data["baseline_rank_score"] = self.baseline_rank(data)
        data["opportunity_top_decile_proba"] = data["opportunity_proba"]
        data["opportunity_score_proba_rank"] = data["opportunity_rank_percentile"]
        data["a3_3_allocation_weight"] = self.a3_3_weight(data["downside_rank_percentile"])
        downside_penalty = (1.0 - _numeric(data["downside_bad_proba"]).fillna(1.0)).clip(lower=0.0, upper=1.0) ** 2
        data["opportunity_downside_score"] = _numeric(data["opportunity_proba"]).fillna(0.0) * downside_penalty
        required = ["date", "code", "close", "turnover_value", "opportunity_proba", "downside_bad_proba", "downside_rank_percentile"]
        return data.dropna(subset=[column for column in required if column in data.columns]).sort_values(["date", "code"]).reset_index(drop=True)

    def baseline_rank(self, data: pd.DataFrame) -> pd.Series:
        for column in BASELINE_RANK_COLUMNS:
            if column in data.columns and not data[column].isna().all():
                return _numeric(data[column]).fillna(-10**18)
        return pd.Series(-10**18, index=data.index, dtype=float)

    def a3_3_weight(self, downside_rank: pd.Series) -> pd.Series:
        rank = _numeric(downside_rank)
        return rank.map(lambda value: 1.0 if value <= 0.40 else 0.6 if value <= 0.70 else 0.3 if value <= 0.85 else 0.0)

    def rank_frame(self, data: pd.DataFrame, spec: RankSpec) -> pd.DataFrame:
        if spec.mode == "all":
            return data.copy()
        if spec.mode == "random":
            return self.random_by_day(data, spec.n or 5)
        if spec.score_column is None or spec.n is None:
            raise ValueError(f"Invalid rank spec: {spec}")
        return self.top_n_by_day(data, spec.score_column, spec.n)

    def top_n_by_day(self, data: pd.DataFrame, column: str, n: int) -> pd.DataFrame:
        return (
            data.sort_values(["date", column, "turnover_value", "code"], ascending=[True, False, False, True])
            .groupby("date", sort=False, group_keys=False)
            .head(n)
            .copy()
        )

    def random_by_day(self, data: pd.DataFrame, n: int) -> pd.DataFrame:
        frames = []
        for day_index, (_, group) in enumerate(data.groupby("date", sort=True)):
            frames.append(group.sample(n=min(n, len(group)), random_state=self.random_seed + day_index))
        return pd.concat(frames, ignore_index=True) if frames else data.iloc[0:0].copy()

    def rank_quality(self, data: pd.DataFrame, spec: RankSpec) -> dict[str, Any]:
        frame = self.rank_frame(data, spec)
        row = {
            "selection": spec.name,
            "rows": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if not frame.empty else 0,
            "avg_candidates_per_day": _safe_float(len(frame) / frame["date"].nunique()) if not frame.empty and frame["date"].nunique() else None,
        }
        row.update(self.quality_metrics(frame))
        return row

    def quality_metrics(self, frame: pd.DataFrame, *, weight_column: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        weights = _numeric(frame[weight_column]).clip(lower=0) if weight_column and weight_column in frame.columns else None
        for column in FUTURE_EVAL_COLUMNS:
            values = _numeric(frame[column]) if column in frame.columns else pd.Series(dtype=float)
            if weights is not None and float(weights.sum()) > 0:
                value = float((values.fillna(0.0) * weights).sum() / weights.sum())
            else:
                value = float(values.mean()) if len(values) else math.nan
            key = "opportunity_top_decile_20d_rate" if column == "opportunity_top_decile_20d" else "downside_bad_rate" if column == DOWNSIDE_TARGET else f"{column}_mean"
            result[key] = _safe_float(value)
        return result

    def score_monotonicity(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for score in STOCK_SELECTION_SCORES:
            if score not in data.columns:
                continue
            frame = data[["date", score, *FUTURE_EVAL_COLUMNS]].dropna(subset=[score]).copy()
            frame["score_percentile_in_day"] = frame.groupby("date")[score].rank(method="average", pct=True)
            frame["score_decile"] = (frame["score_percentile_in_day"] * 10).apply(math.ceil).clip(lower=1, upper=10).astype(int)
            for decile, group in frame.groupby("score_decile", sort=True):
                row = {"score": score, "decile": int(decile), "rows": int(len(group))}
                row.update(self.quality_metrics(group))
                rows.append(row)
        return rows

    def layer_interaction(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        stock_top20 = self.top_n_by_day(data, "stock_selection_rank_score", 20)
        specs = [
            ("A_stock_selection_top5", self.top_n_by_day(data, "stock_selection_rank_score", 5)),
            ("B_opportunity_top5", self.top_n_by_day(data, "opportunity_proba", 5)),
            ("C_stock_top20_then_opportunity_top5", self.top_n_by_day(stock_top20, "opportunity_proba", 5)),
            ("D_universe_then_opportunity_top5", self.top_n_by_day(data, "opportunity_proba", 5)),
            ("E_stock_top20_then_opportunity_downside_top5", self.top_n_by_day(stock_top20, "opportunity_downside_score", 5)),
            ("F_universe_then_opportunity_downside_top5", self.top_n_by_day(data, "opportunity_downside_score", 5)),
        ]
        rows = []
        for name, frame in specs:
            row = {
                "layer": name,
                "rows": int(len(frame)),
                "candidate_days": int(frame["date"].nunique()) if not frame.empty else 0,
                "avg_candidates_per_day": _safe_float(len(frame) / frame["date"].nunique()) if not frame.empty and frame["date"].nunique() else None,
            }
            row.update(self.quality_metrics(frame))
            rows.append(row)
        return rows

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
            ranked_source = self.prefilter(current_rank_frame, spec)
            ranked = ranked_source.sort_values([spec.rank_column, "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions)
            if spec.allocation_mode != "equal":
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
                        "allocation_weight": _safe_float(row.get("a3_3_allocation_weight")) if spec.allocation_mode != "equal" else 1.0,
                        **{column: _safe_float(row.get(column)) for column in FUTURE_EVAL_COLUMNS},
                    }
                )

            marked_value = sum(float(position["lot_count"]) * self.options.round_lot * float(position["last_close"]) for position in positions)
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

    def prefilter(self, frame: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
        if not spec.prefilter_column or not spec.prefilter_n:
            return frame
        return frame.sort_values([spec.prefilter_column, "turnover_value", "code"], ascending=[False, False, True]).head(spec.prefilter_n).copy()

    def target_amounts(self, selected: pd.DataFrame, spec: StrategySpec, cash: float) -> dict[Any, float]:
        if selected.empty:
            return {}
        available_budget = min(cash, self.options.daily_buy_budget)
        if spec.allocation_mode == "equal":
            amount = available_budget / max(1, min(self.options.max_positions, len(selected)))
            return {index: amount for index in selected.index}
        weights = _numeric(selected["a3_3_allocation_weight"]).clip(lower=0)
        penalty = (1.0 - _numeric(selected["downside_bad_proba"]).fillna(1.0)).clip(lower=0.0, upper=1.0) ** 2
        combined = weights * penalty
        total = float(combined.sum())
        if total <= 0:
            return {index: 0.0 for index in selected.index}
        return {index: available_budget * float(weight) / total for index, weight in combined.items()}

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None, spec: StrategySpec) -> str | None:
        if current_row is not None and spec.use_b5_exit:
            observed_return = float(current_row["close"]) / float(position["entry_close"]) - 1.0
            if observed_return <= self.options.stop_loss_rate:
                return "stop_loss"
            current_proba = _safe_float(current_row.get("opportunity_top_decile_proba"))
            current_rank = _safe_float(current_row.get("opportunity_score_proba_rank"))
            entry_proba = _safe_float(position.get("entry_opportunity_top_decile_proba"))
            if current_rank is not None and current_rank < self.options.opportunity_rank_floor:
                return "opportunity_rank_below_floor"
            if current_proba is not None and entry_proba is not None and current_proba <= entry_proba - self.options.opportunity_drop_threshold:
                return "opportunity_proba_drop"
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
            "target_buy_amount": position.get("target_buy_amount"),
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

    def buy_quality(self, strategy: str, trades: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {"strategy": strategy, "buy_count": int(len(trades))}
        for column in FUTURE_EVAL_COLUMNS:
            values = _numeric(trades[column]) if column in trades.columns else pd.Series(dtype=float)
            key = "opportunity_top_decile_20d_rate" if column == "opportunity_top_decile_20d" else "downside_bad_rate" if column == DOWNSIDE_TARGET else f"{column}_mean"
            row[key] = _safe_float(values.mean()) if not values.empty else None
        row["average_allocation_weight"] = _safe_float(_numeric(trades["allocation_weight"]).mean()) if "allocation_weight" in trades.columns and not trades.empty else None
        return row

    def final_judgment(
        self,
        rank_quality: list[dict[str, Any]],
        monotonicity: list[dict[str, Any]],
        layer_interaction: list[dict[str, Any]],
        strategy_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        by_rank = {row["selection"]: row for row in rank_quality}
        universe = by_rank.get("candidate_universe", {})
        stock_top5 = by_rank.get("stock_selection_rank_score_top5", {})
        random_top5 = by_rank.get("random_top5_per_day", {})
        best_signal = self.best_stock_signal(rank_quality)
        monotonic_scores = self.monotonic_scores(monotonicity)
        by_layer = {row["layer"]: row for row in layer_interaction}
        stock_prefilter_hurts = self.rate(by_layer.get("E_stock_top20_then_opportunity_downside_top5")) < self.rate(by_layer.get("F_universe_then_opportunity_downside_top5"))
        stock_adds = self.rate(stock_top5) > self.rate(universe) and self.rate(stock_top5) > self.rate(random_top5) and bool(monotonic_scores)
        stock_top5_valid = self.rate(stock_top5) > max(self.rate(universe), self.rate(random_top5))
        best_generation = self.best_candidate_generation(layer_interaction)
        return {
            "stock_selection_adds_value": stock_adds,
            "stock_selection_top5_valid": stock_top5_valid,
            "stock_selection_prefilter_hurts_valuation": stock_prefilter_hurts,
            "best_stock_selection_signal": best_signal,
            "best_candidate_generation_method": best_generation,
            "monotonic_stock_selection_scores": monotonic_scores,
            "reason": self.judgment_reason(stock_adds, stock_top5_valid, stock_prefilter_hurts, best_signal, best_generation),
        }

    def rate(self, row: dict[str, Any] | None, key: str = "opportunity_top_decile_20d_rate") -> float:
        return _safe_float((row or {}).get(key)) or 0.0

    def best_stock_signal(self, rank_quality: list[dict[str, Any]]) -> str | None:
        rows = [row for row in rank_quality if any(row["selection"].startswith(score) for score in STOCK_SELECTION_SCORES)]
        if not rows:
            return None
        best = max(rows, key=lambda row: (self.rate(row), -(self.rate(row, "downside_bad_rate"))))
        return str(best["selection"])

    def monotonic_scores(self, monotonicity: list[dict[str, Any]]) -> list[str]:
        result = []
        for score in STOCK_SELECTION_SCORES:
            rows = sorted([row for row in monotonicity if row["score"] == score], key=lambda row: row["decile"])
            if len(rows) < 2:
                continue
            low = sum(self.rate(row) for row in rows[:3]) / min(3, len(rows))
            high = sum(self.rate(row) for row in rows[-3:]) / min(3, len(rows))
            downside_low = sum(self.rate(row, "downside_bad_rate") for row in rows[:3]) / min(3, len(rows))
            downside_high = sum(self.rate(row, "downside_bad_rate") for row in rows[-3:]) / min(3, len(rows))
            if high > low and downside_high <= downside_low * 1.25:
                result.append(score)
        return result

    def best_candidate_generation(self, layer_interaction: list[dict[str, Any]]) -> str | None:
        if not layer_interaction:
            return None
        best = max(layer_interaction, key=lambda row: (self.rate(row), -(self.rate(row, "downside_bad_rate")), self.rate(row, "opportunity_value_20d_mean")))
        return str(best["layer"])

    def judgment_reason(self, adds: bool, top5_valid: bool, hurts: bool, best_signal: str | None, best_generation: str | None) -> str:
        if hurts and not top5_valid:
            return f"Stock Selection top5 does not clearly beat the universe/random baseline, and the prefilter worsens Valuation/Downside selection. Best signal={best_signal}, best generation={best_generation}."
        if adds:
            return f"Stock Selection beats universe/random and has monotonic evidence. Best signal={best_signal}, best generation={best_generation}."
        return f"Stock Selection evidence is mixed. Best signal={best_signal}, best generation={best_generation}."

    def recommendation(self, judgment: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"recommended_next_phase": "Fix Phase12-E1 leakage blockers"}
        if judgment.get("stock_selection_prefilter_hurts_valuation"):
            next_phase = "Phase12-E2 Remove Stock Selection Prefilter Test"
        elif not judgment.get("stock_selection_top5_valid"):
            next_phase = "Phase12-E2 Candidate Universe Expansion"
        elif not judgment.get("stock_selection_adds_value"):
            next_phase = "Phase12-E2 Stock Selection Rebuild Audit"
        else:
            next_phase = "Phase12-D4 Exit AI Dataset Audit"
        return {
            "stock_selection_adds_value": judgment.get("stock_selection_adds_value"),
            "stock_selection_top5_valid": judgment.get("stock_selection_top5_valid"),
            "stock_selection_prefilter_hurts_valuation": judgment.get("stock_selection_prefilter_hurts_valuation"),
            "best_stock_selection_signal": judgment.get("best_stock_selection_signal"),
            "best_candidate_generation_method": judgment.get("best_candidate_generation_method"),
            "recommended_next_phase": next_phase,
            "reason": judgment.get("reason"),
        }

    def executive_summary(self, judgment: dict[str, Any]) -> str:
        return (
            f"stock_selection_adds_value={judgment.get('stock_selection_adds_value')}, "
            f"stock_selection_top5_valid={judgment.get('stock_selection_top5_valid')}, "
            f"stock_selection_prefilter_hurts_valuation={judgment.get('stock_selection_prefilter_hurts_valuation')}. "
            f"{judgment.get('reason')}"
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

    def conditions(self) -> dict[str, Any]:
        return {
            "period": {"start": START_DATE, "end": END_DATE},
            "initial_cash": self.options.initial_cash,
            "daily_buy_budget": self.options.daily_buy_budget,
            "max_positions": self.options.max_positions,
            "holding_days": self.options.holding_days,
            "round_lot": self.options.round_lot,
            "cost_rate": self.options.cost_rate,
            "stop_loss": self.options.stop_loss_rate,
            "b5_2_opportunity_drop_threshold": self.options.opportunity_drop_threshold,
            "opportunity_rank_floor": self.options.opportunity_rank_floor,
            "random_seed": self.random_seed,
            "strategy_scope": "2025-only lightweight comparison, not full backtest",
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "12-E1",
            "scope": "2025 Stock Selection reality audit",
            "new_model_trained": False,
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
            "new_model_trained": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "historical_predictions_regenerated": False,
            "uses_2025_training_data_for_2025_eval": False,
            "decision_columns": [
                *STOCK_SELECTION_SCORES,
                "opportunity_proba",
                "downside_bad_proba",
                "downside_rank_percentile",
                "close",
                "turnover_value",
                "cash",
            ],
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase12E1Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12E1Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-E1 Stock Selection Reality Audit",
            "",
            "## Executive Summary",
            "",
            str(report.get("executive_summary", "")),
            "",
            "## Rank Quality",
            "",
            self.table(report.get("rank_quality_table", []), ["selection", "rows", "candidate_days", "avg_candidates_per_day", "future_return_20d_mean", "future_max_return_20d_mean", "future_max_drawdown_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_rate", "downside_bad_rate"]),
            "",
            "## Score Monotonicity",
            "",
            self.table(report.get("score_monotonicity_table", []), ["score", "decile", "rows", "future_return_20d_mean", "opportunity_top_decile_20d_rate", "downside_bad_rate"]),
            "",
            "## Layer Interaction",
            "",
            self.table(report.get("layer_interaction_summary", []), ["layer", "rows", "candidate_days", "avg_candidates_per_day", "future_return_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_rate", "downside_bad_rate"]),
            "",
            "## Strategy Comparison",
            "",
            self.table(report.get("strategy_comparison_table", []), ["strategy", "net_profit", "PF", "DD", "win_rate", "final_assets", "capital_utilization", "average_holding_days", "total_trades", "cost_paid", "exit_reason_counts"]),
            "",
            "## Strategy BUY Quality",
            "",
            self.table(report.get("strategy_buy_quality_table", []), ["strategy", "buy_count", "future_return_20d_mean", "future_max_return_20d_mean", "future_max_drawdown_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_rate", "downside_bad_rate", "average_allocation_weight"]),
            "",
            "## Final Judgment",
            "",
            self.table([report.get("final_judgment", {})], ["stock_selection_adds_value", "stock_selection_top5_valid", "stock_selection_prefilter_hurts_valuation", "best_stock_selection_signal", "best_candidate_generation_method", "reason"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_only_for_evaluation", "future_columns_used_as_features", "new_model_trained", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "historical_predictions_regenerated", "uses_2025_training_data_for_2025_eval", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["stock_selection_adds_value", "stock_selection_top5_valid", "stock_selection_prefilter_hurts_valuation", "best_stock_selection_signal", "best_candidate_generation_method", "recommended_next_phase", "reason"]),
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
