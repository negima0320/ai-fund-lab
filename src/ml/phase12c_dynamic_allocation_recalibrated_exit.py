"""Phase 12-C dynamic allocation plus recalibrated exit.

This is a 2025-only lightweight integration check. It combines the Phase 12-A3
dynamic allocation weights with the Phase 12-B5 recalibrated Opportunity Exit
threshold. It does not run a full backtest, change profiles, overwrite models,
or regenerate historical predictions.
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
from ml.phase12b2_allocation_execution_adjustment import Phase12B2Options
from ml.phase12b_limited_allocation_strategy_check import ARTIFACT_PATH, BASELINE_RANK_COLUMNS, END_DATE, ROUND_LOT, START_DATE


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase12c_dynamic_allocation_recalibrated_exit_2025"
FUTURE_EVAL_COLUMNS = EVAL_COLUMNS


@dataclass(frozen=True)
class Phase12COptions(Phase12B2Options):
    opportunity_drop_threshold: float = 0.30
    opportunity_rank_floor: float = 0.50


@dataclass(frozen=True)
class StrategySpec:
    name: str
    rank_column: str
    allocation_mode: str
    use_recalibrated_exit: bool


@dataclass(frozen=True)
class Phase12CPaths:
    markdown: Path
    json: Path


STRATEGIES = [
    StrategySpec("C0_baseline_equal_allocation", rank_column="baseline_rank_score", allocation_mode="equal", use_recalibrated_exit=False),
    StrategySpec("C1_dynamic_raw_B5_2_exit", rank_column="opportunity_proba", allocation_mode="dynamic_raw", use_recalibrated_exit=True),
    StrategySpec("C2_dynamic_normalized_B5_2_exit", rank_column="opportunity_proba", allocation_mode="dynamic_normalized", use_recalibrated_exit=True),
    StrategySpec("C3_partial_normalized_30_B5_2_exit", rank_column="opportunity_proba", allocation_mode="partial_30", use_recalibrated_exit=True),
    StrategySpec("C4_partial_normalized_50_B5_2_exit", rank_column="opportunity_proba", allocation_mode="partial_50", use_recalibrated_exit=True),
]


class Phase12CDynamicAllocationRecalibratedExit:
    def __init__(self, root: Path | str = ROOT, *, options: Phase12COptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase12COptions()

    def run(self) -> Phase12CPaths:
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

        strategy_results = []
        buy_quality = []
        for spec in STRATEGIES:
            trades, daily = self.simulate(data, spec)
            strategy_results.append(self.metrics(spec.name, trades, daily))
            buy_quality.append(self.buy_quality(spec.name, trades))
        comparison = self.strategy_comparison(strategy_results)
        return {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "strategy_results": strategy_results,
            "buy_quality": buy_quality,
            "strategy_comparison": comparison,
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(strategy_results, leakage),
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
            if spec.allocation_mode != "equal":
                ranked = ranked.copy()
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

    def target_amounts(self, selected: pd.DataFrame, spec: StrategySpec, cash: float) -> dict[Any, float]:
        if selected.empty:
            return {}
        available_budget = min(cash, self.options.daily_buy_budget)
        if spec.allocation_mode == "equal":
            amount = available_budget / max(1, min(self.options.max_positions, len(selected)))
            return {index: amount for index in selected.index}
        raw = self.raw_amounts(selected, available_budget)
        normalized = self.normalized_amounts(selected, available_budget)
        if spec.allocation_mode == "dynamic_raw":
            return raw
        if spec.allocation_mode == "dynamic_normalized":
            return normalized
        if spec.allocation_mode == "partial_30":
            return self.blend_amounts(raw, normalized, normalized_share=0.30)
        if spec.allocation_mode == "partial_50":
            return self.blend_amounts(raw, normalized, normalized_share=0.50)
        raise ValueError(f"Unknown allocation mode: {spec.allocation_mode}")

    def raw_amounts(self, selected: pd.DataFrame, available_budget: float) -> dict[Any, float]:
        weights = _numeric(selected["a3_3_allocation_weight"]).clip(lower=0)
        return {index: available_budget * float(weight) / self.options.max_positions for index, weight in weights.items()}

    def normalized_amounts(self, selected: pd.DataFrame, available_budget: float) -> dict[Any, float]:
        weights = _numeric(selected["a3_3_allocation_weight"]).clip(lower=0)
        total = float(weights.sum())
        if total <= 0:
            return {index: 0.0 for index in selected.index}
        return {index: available_budget * float(weight) / total for index, weight in weights.items()}

    def blend_amounts(self, raw: dict[Any, float], normalized: dict[Any, float], *, normalized_share: float) -> dict[Any, float]:
        raw_share = 1.0 - normalized_share
        return {index: raw_share * raw.get(index, 0.0) + normalized_share * normalized.get(index, 0.0) for index in raw}

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None, spec: StrategySpec) -> str | None:
        if current_row is not None:
            observed_return = float(current_row["close"]) / float(position["entry_close"]) - 1.0
            if observed_return <= self.options.stop_loss_rate:
                return "stop_loss"
            if spec.use_recalibrated_exit:
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

    def metrics(self, strategy: str, trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
        profits = _numeric(trades["realized_profit"]) if not trades.empty else pd.Series(dtype=float)
        gross_profit = float(profits[profits > 0].sum()) if not profits.empty else 0.0
        gross_loss = abs(float(profits[profits < 0].sum())) if not profits.empty else 0.0
        equity = _numeric(daily["total_assets"]) if not daily.empty else pd.Series([self.options.initial_cash])
        drawdown = equity / equity.cummax() - 1.0
        reentries = self.reentry_counts(trades)
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
            "median_holding_days": _safe_float(_numeric(trades["holding_days"]).median()) if not trades.empty else None,
            "same_code_reentry_count": reentries["same_code_reentry_count"],
            "reentry_within_5_days_count": reentries["reentry_within_5_days_count"],
            "cost_paid": _safe_float(_numeric(trades["cost_paid"]).sum()) if "cost_paid" in trades.columns else 0.0,
            "exit_reason_counts": dict(Counter(trades["exit_reason"])) if "exit_reason" in trades.columns else {},
        }

    def buy_quality(self, strategy: str, trades: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {"strategy": strategy, "buy_count": int(len(trades))}
        for column in FUTURE_EVAL_COLUMNS:
            values = _numeric(trades[column]) if column in trades.columns else pd.Series(dtype=float)
            key = "opportunity_top_decile_20d_rate" if column == "opportunity_top_decile_20d" else f"{column}_mean"
            row[key] = _safe_float(values.mean()) if not values.empty else None
        return row

    def reentry_counts(self, trades: pd.DataFrame) -> dict[str, int]:
        if trades.empty:
            return {"same_code_reentry_count": 0, "reentry_within_5_days_count": 0}
        same_code_reentry = 0
        reentry_5d = 0
        for _, group in trades.sort_values(["code", "entry_date"]).groupby("code", sort=False):
            previous_exit = None
            for _, row in group.iterrows():
                entry = pd.Timestamp(row["entry_date"])
                if previous_exit is not None:
                    same_code_reentry += 1
                    if len(pd.bdate_range(previous_exit, entry)) - 1 <= 5:
                        reentry_5d += 1
                previous_exit = pd.Timestamp(row["exit_date"])
        return {"same_code_reentry_count": same_code_reentry, "reentry_within_5_days_count": reentry_5d}

    def strategy_comparison(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_strategy = {row["strategy"]: row for row in rows}
        baseline = by_strategy.get("C0_baseline_equal_allocation", {})
        best = self.best_strategy(rows)
        minimum = [row["strategy"] for row in rows if self.minimum_passed(row)]
        ideal = [row["strategy"] for row in rows if self.ideal_passed(row)]
        best_dynamic = self.best_strategy([row for row in rows if row["strategy"] != "C0_baseline_equal_allocation"])
        return {
            "best_strategy": best.get("strategy") if best else None,
            "best_strategy_reason": self.best_strategy_reason(best),
            "best_dynamic_strategy": best_dynamic.get("strategy") if best_dynamic else None,
            "capital_utilization_improved": bool(best_dynamic and (_safe_float(best_dynamic.get("capital_utilization")) or 0.0) > (_safe_float(baseline.get("capital_utilization")) or 10**9)),
            "pf_improved": bool(best_dynamic and (_safe_float(best_dynamic.get("PF")) or 0.0) > (_safe_float(baseline.get("PF")) or 10**9)),
            "dd_improved": bool(best_dynamic and (_safe_float(best_dynamic.get("DD")) or -1.0) > (_safe_float(baseline.get("DD")) or 1.0)),
            "strategies_meeting_minimum_target": minimum,
            "strategies_meeting_ideal_target": ideal,
            "ready_for_phase13": bool(minimum),
            "recommended_next_phase": "Phase13 limited OOS/year robustness check" if minimum else "Phase12-C2 allocation utilization refinement",
        }

    def minimum_passed(self, row: dict[str, Any]) -> bool:
        return (
            (_safe_float(row.get("PF")) or 0.0) >= 1.8
            and (_safe_float(row.get("DD")) or -1.0) >= -0.10
            and (_safe_float(row.get("net_profit")) or 0.0) > 0
            and (_safe_float(row.get("capital_utilization")) or 0.0) >= 0.20
        )

    def ideal_passed(self, row: dict[str, Any]) -> bool:
        return (
            (_safe_float(row.get("PF")) or 0.0) >= 2.0
            and (_safe_float(row.get("DD")) or -1.0) >= -0.08
            and (_safe_float(row.get("capital_utilization")) or 0.0) >= 0.30
        )

    def best_strategy(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None

        def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
            profit = _safe_float(row.get("net_profit")) or -10**18
            pf = _safe_float(row.get("PF")) or 0.0
            dd = _safe_float(row.get("DD")) or -1.0
            utilization = _safe_float(row.get("capital_utilization")) or 0.0
            pass_bonus = 10**7 if self.minimum_passed(row) else 0.0
            return (pass_bonus + profit, pf, dd, utilization)

        return max(rows, key=sort_key)

    def best_strategy_reason(self, row: dict[str, Any] | None) -> str:
        if not row:
            return "No strategies were evaluated."
        return (
            f"{row['strategy']} net_profit={row.get('net_profit'):.0f}, PF={row.get('PF'):.4f}, "
            f"DD={row.get('DD'):.4f}, capital_utilization={row.get('capital_utilization'):.4f}."
        )

    def recommendation(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"ready_for_phase13": False, "recommended_next_phase": "Fix Phase12-C leakage blockers"}
        comparison = self.strategy_comparison(rows)
        return {
            "best_strategy": comparison["best_strategy"],
            "best_strategy_reason": comparison["best_strategy_reason"],
            "capital_utilization_improved": comparison["capital_utilization_improved"],
            "pf_improved": comparison["pf_improved"],
            "dd_improved": comparison["dd_improved"],
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
            "exit": "B5_2_proba_drop_larger",
            "opportunity_drop_threshold": self.options.opportunity_drop_threshold,
            "opportunity_rank_floor": self.options.opportunity_rank_floor,
            "stop_loss": self.options.stop_loss_rate,
            "daily_buy_budget": self.options.daily_buy_budget,
            "max_positions": self.options.max_positions,
            "holding_days": self.options.holding_days,
            "cost_rate": self.options.cost_rate,
            "strategies": [spec.__dict__ for spec in STRATEGIES],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "12-C",
            "scope": "2025 dynamic allocation + recalibrated exit integration only",
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

    def save_outputs(self, report: dict[str, Any]) -> Phase12CPaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12CPaths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-C Dynamic Allocation + Recalibrated Exit",
            "",
            "## Strategy Results",
            "",
            self.table(report.get("strategy_results", []), ["strategy", "net_profit", "PF", "DD", "win_rate", "final_assets", "capital_utilization", "average_holding_days", "median_holding_days", "same_code_reentry_count", "reentry_within_5_days_count", "cost_paid", "exit_reason_counts"]),
            "",
            "## BUY Quality",
            "",
            self.table(report.get("buy_quality", []), ["strategy", "buy_count", "future_return_20d_mean", "future_max_return_20d_mean", "future_max_drawdown_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_rate"]),
            "",
            "## Comparison",
            "",
            self.table([report.get("strategy_comparison", {})], ["best_strategy", "best_strategy_reason", "best_dynamic_strategy", "capital_utilization_improved", "pf_improved", "dd_improved", "strategies_meeting_minimum_target", "strategies_meeting_ideal_target", "ready_for_phase13", "recommended_next_phase"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_only_for_evaluation", "future_columns_used_as_features", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["best_strategy", "capital_utilization_improved", "pf_improved", "dd_improved", "ready_for_phase13", "recommended_next_phase"]),
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
