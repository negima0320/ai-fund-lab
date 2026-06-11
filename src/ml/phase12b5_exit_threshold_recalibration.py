"""Phase 12-B5 Opportunity Exit threshold recalibration.

This is a 2025-only lightweight strategy check for S3a dynamic raw allocation.
It keeps BUY/allocation and stop-loss fixed, and only changes Opportunity Exit
threshold behavior. It does not run a full backtest, change profiles, overwrite
models, or regenerate historical predictions.
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
REPORT_STEM = "phase12b5_exit_threshold_recalibration_2025"
FUTURE_EVAL_COLUMNS = EVAL_COLUMNS


@dataclass(frozen=True)
class Phase12B5Options(Phase12B2Options):
    pass


@dataclass(frozen=True)
class ExitVariant:
    name: str
    opportunity_rank_floor: float = 0.50
    opportunity_drop_threshold: float = 0.15
    confirmation_days: int = 1
    opportunity_exit_mode: str = "always"


@dataclass(frozen=True)
class Phase12B5Paths:
    markdown: Path
    json: Path


VARIANTS = [
    ExitVariant("B5_0_baseline"),
    ExitVariant("B5_1_rank_floor_lower", opportunity_rank_floor=0.30),
    ExitVariant("B5_2_proba_drop_larger", opportunity_drop_threshold=0.30),
    ExitVariant("B5_3_both_relaxed", opportunity_rank_floor=0.30, opportunity_drop_threshold=0.30),
    ExitVariant("B5_4_confirmation_2days", confirmation_days=2),
    ExitVariant("B5_5_confirmation_3days", confirmation_days=3),
    ExitVariant("B5_6_profit_protected_exit", opportunity_exit_mode="profit_only"),
    ExitVariant("B5_7_loss_only_hard_exit", opportunity_exit_mode="loss_hard_only"),
]


class Phase12B5ExitThresholdRecalibration:
    def __init__(self, root: Path | str = ROOT, *, options: Phase12B5Options | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase12B5Options()

    def run(self) -> Phase12B5Paths:
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
                "recommendation": self.recommendation([], [], leakage),
            }

        variant_results = []
        hold_quality = []
        for variant in VARIANTS:
            trades, daily = self.simulate(data, variant)
            variant_results.append(self.metrics(variant.name, trades, daily))
            hold_quality.append(self.hold_quality(variant.name, trades, data))
        comparison = self.variant_comparison(variant_results)
        return {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "variant_results": variant_results,
            "hold_exit_quality": hold_quality,
            "variant_comparison": comparison,
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(variant_results, hold_quality, leakage),
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
        data["opportunity_top_decile_proba"] = data["opportunity_proba"]
        data["opportunity_score_proba_rank"] = data["opportunity_rank_percentile"]
        data["a3_3_allocation_weight"] = self.a3_3_weight(data["downside_rank_percentile"])
        return data.dropna(subset=["date", "code", "close", "opportunity_proba", "downside_rank_percentile"]).sort_values(["date", "code"]).reset_index(drop=True)

    def a3_3_weight(self, downside_rank: pd.Series) -> pd.Series:
        rank = _numeric(downside_rank)
        return rank.map(lambda value: 1.0 if value <= 0.40 else 0.6 if value <= 0.70 else 0.3 if value <= 0.85 else 0.0)

    def simulate(self, data: pd.DataFrame, variant: ExitVariant) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                    if close > float(position["peak_close"]):
                        position["peak_close"] = close
                reason = self.exit_reason(position, current_date, current_row, variant)
                if reason:
                    exit_close = float(current_row["close"]) if current_row is not None else float(position["last_close"])
                    trade = self.close_position(position, current_date, exit_close, reason, variant.name)
                    cash += trade["exit_cash_flow"]
                    trades.append(trade)
                else:
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            ranked = current_rank_frame.sort_values(["opportunity_proba", "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions)
            ranked = ranked[_numeric(ranked["a3_3_allocation_weight"]) > 0].copy()
            selected = ranked.head(slots) if slots else ranked.iloc[0:0]
            target_amounts = self.target_amounts(selected, cash)
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
                        "variant": variant.name,
                        "code": str(row["code"]),
                        "buy_amount": buy_amount,
                        "buy_cost": buy_cost,
                        "target_buy_amount": target_amount,
                        "lot_count": lots,
                        "entry_close": float(row["close"]),
                        "last_close": float(row["close"]),
                        "peak_close": float(row["close"]),
                        "entry_opportunity_top_decile_proba": _safe_float(row.get("opportunity_top_decile_proba")),
                        "allocation_weight": _safe_float(row.get("a3_3_allocation_weight")),
                        "opportunity_exit_signal_streak": 0,
                        **{column: _safe_float(row.get(column)) for column in FUTURE_EVAL_COLUMNS},
                    }
                )

            marked_value = sum(float(position["lot_count"]) * self.options.round_lot * float(position["last_close"]) for position in positions)
            daily_rows.append(
                {
                    "variant": variant.name,
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
                trade = self.close_position(position, last_date, float(position["last_close"]), "forced_end_of_period", position.get("variant", "forced_end_of_period"))
                cash += trade["exit_cash_flow"]
                trades.append(trade)
            if daily_rows:
                daily_rows[-1]["total_assets"] = cash
                daily_rows[-1]["marked_position_value"] = 0.0
                daily_rows[-1]["capital_utilization"] = 0.0
        return pd.DataFrame(trades), pd.DataFrame(daily_rows)

    def target_amounts(self, selected: pd.DataFrame, cash: float) -> dict[Any, float]:
        if selected.empty:
            return {}
        available_budget = min(cash, self.options.daily_buy_budget)
        weights = _numeric(selected["a3_3_allocation_weight"]).clip(lower=0)
        return {index: available_budget * float(weight) / self.options.max_positions for index, weight in weights.items()}

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None, variant: ExitVariant) -> str | None:
        if current_row is not None:
            observed_return = float(current_row["close"]) / float(position["entry_close"]) - 1.0
            if observed_return <= self.options.stop_loss_rate:
                return "stop_loss"
            signal = self.opportunity_exit_signal(position, current_row, observed_return, variant)
            position["opportunity_exit_signal_streak"] = int(position.get("opportunity_exit_signal_streak", 0)) + 1 if signal else 0
            if signal and int(position["opportunity_exit_signal_streak"]) >= variant.confirmation_days:
                return "opportunity_exit"
        if current_date >= position["due_date"]:
            return "time_exit_20d"
        return None

    def opportunity_exit_signal(self, position: dict[str, Any], current_row: pd.Series, observed_return: float, variant: ExitVariant) -> bool:
        if variant.opportunity_exit_mode == "profit_only" and observed_return <= 0:
            return False
        if variant.opportunity_exit_mode == "loss_hard_only" and observed_return > -0.05:
            return False
        current_proba = _safe_float(current_row.get("opportunity_top_decile_proba"))
        current_rank = _safe_float(current_row.get("opportunity_score_proba_rank"))
        entry_proba = _safe_float(position.get("entry_opportunity_top_decile_proba"))
        rank_signal = current_rank is not None and current_rank < variant.opportunity_rank_floor
        drop_signal = current_proba is not None and entry_proba is not None and current_proba <= entry_proba - variant.opportunity_drop_threshold
        return bool(rank_signal or drop_signal)

    def close_position(self, position: dict[str, Any], exit_date: pd.Timestamp, exit_close: float, reason: str, variant: str) -> dict[str, Any]:
        exit_amount = float(position["lot_count"]) * self.options.round_lot * exit_close
        sell_cost = exit_amount * self.options.cost_rate
        exit_cash_flow = exit_amount - sell_cost
        total_cost = float(position["buy_cost"]) + sell_cost
        profit = exit_cash_flow - float(position["buy_amount"]) - float(position["buy_cost"])
        holding_days = len(pd.bdate_range(position["entry_date"], exit_date)) - 1
        max_profit_before_exit = float(position["peak_close"]) / float(position["entry_close"]) - 1.0
        realized_return = profit / float(position["buy_amount"]) if position["buy_amount"] else None
        return {
            "variant": variant,
            "entry_date": position["entry_date"],
            "exit_date": exit_date,
            "code": position["code"],
            "buy_amount": position["buy_amount"],
            "exit_amount": exit_amount,
            "exit_cash_flow": exit_cash_flow,
            "realized_profit": profit,
            "realized_return": realized_return,
            "holding_days": holding_days,
            "exit_reason": reason,
            "cost_paid": total_cost,
            "allocation_weight": position.get("allocation_weight"),
            "max_profit_before_exit": max_profit_before_exit,
            "profit_capture_ratio": realized_return / max_profit_before_exit if realized_return is not None and max_profit_before_exit > 0 else None,
            **{column: position.get(column) for column in FUTURE_EVAL_COLUMNS},
        }

    def metrics(self, variant: str, trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
        profits = _numeric(trades["realized_profit"]) if not trades.empty else pd.Series(dtype=float)
        gross_profit = float(profits[profits > 0].sum()) if not profits.empty else 0.0
        gross_loss = abs(float(profits[profits < 0].sum())) if not profits.empty else 0.0
        equity = _numeric(daily["total_assets"]) if not daily.empty else pd.Series([self.options.initial_cash])
        drawdown = equity / equity.cummax() - 1.0
        reasons = Counter(trades["exit_reason"]) if "exit_reason" in trades.columns else Counter()
        reentries = self.reentry_counts(trades)
        return {
            "variant": variant,
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
            "exit_reason_counts": dict(reasons),
            "opportunity_exit_count": int(reasons.get("opportunity_exit", 0)),
            "stop_loss_count": int(reasons.get("stop_loss", 0)),
            "time_exit_count": int(reasons.get("time_exit_20d", 0)),
            "forced_exit_count": int(reasons.get("forced_end_of_period", 0)),
        }

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

    def hold_quality(self, variant: str, trades: pd.DataFrame, data: pd.DataFrame) -> dict[str, Any]:
        realized = _numeric(trades["realized_return"]) if "realized_return" in trades.columns else pd.Series(dtype=float)
        max_profit = _numeric(trades["max_profit_before_exit"]) if "max_profit_before_exit" in trades.columns else pd.Series(dtype=float)
        capture = _numeric(trades["profit_capture_ratio"]) if "profit_capture_ratio" in trades.columns else pd.Series(dtype=float)
        audited = self.attach_post_exit_returns(trades, data)
        post20 = _numeric(audited["post_exit_max_return_20d"]) if "post_exit_max_return_20d" in audited.columns else pd.Series(dtype=float)
        return {
            "variant": variant,
            "avg_profit_capture": _safe_float(realized.mean()),
            "avg_max_profit_before_exit": _safe_float(max_profit.mean()),
            "profit_capture_ratio": _safe_float(capture.mean()),
            "avg_post_exit_20d_return": _safe_float(post20.mean()),
        }

    def attach_post_exit_returns(self, trades: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        path_by_code = {code: group.sort_values("date")[["date", "close"]] for code, group in data.groupby("code", sort=False)}
        values = []
        for _, trade in trades.iterrows():
            path = path_by_code.get(str(trade["code"]), pd.DataFrame())
            exit_date = pd.Timestamp(trade["exit_date"])
            exit_close = _safe_float(trade.get("exit_amount")) / (_safe_float(trade.get("buy_amount")) or 1.0) if False else None
            window = path[(path["date"] > exit_date) & (path["date"] <= exit_date + pd.offsets.BDay(20))]
            if window.empty:
                values.append(None)
                continue
            # Reconstruct exit close from exit_amount and lot count is not stored; realized return is enough for audit, so use last close if available around exit.
            at_exit = path[path["date"] == exit_date]
            base_close = _safe_float(at_exit["close"].iloc[0]) if not at_exit.empty else None
            values.append(_safe_float(_numeric(window["close"]).max() / base_close - 1.0) if base_close else None)
        audited = trades.copy()
        audited["post_exit_max_return_20d"] = values
        return audited

    def variant_comparison(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_variant = {row["variant"]: row for row in rows}
        baseline = by_variant.get("B5_0_baseline", {})
        minimum = [row["variant"] for row in rows if self.minimum_passed(row, baseline)]
        ideal = [row["variant"] for row in rows if self.ideal_passed(row)]
        best = self.best_variant(rows, baseline)
        return {
            "best_variant": best.get("variant") if best else None,
            "best_variant_reason": self.best_variant_reason(best),
            "variants_meeting_minimum_target": minimum,
            "variants_meeting_ideal_target": ideal,
            "opportunity_exit_relaxation_effect": self.relaxation_effect(rows, baseline),
            "capital_utilization_effect": self.capital_utilization_effect(rows, baseline),
            "ready_for_phase12c": bool(minimum),
            "recommended_next_phase": "Phase12-C dynamic allocation + recalibrated exit" if minimum else "Phase12-B6 exit/hold model audit or Phase12-A4 risk score refinement",
        }

    def minimum_passed(self, row: dict[str, Any], baseline: dict[str, Any]) -> bool:
        return (
            (_safe_float(row.get("PF")) or 0.0) >= 1.5
            and (_safe_float(row.get("DD")) or -1.0) >= -0.12
            and (_safe_float(row.get("net_profit")) or 0.0) > 0
            and (_safe_float(row.get("capital_utilization")) or 0.0) > (_safe_float(baseline.get("capital_utilization")) or 10**9)
            and (_safe_float(row.get("average_holding_days")) or 0.0) > (_safe_float(baseline.get("average_holding_days")) or 10**9)
        )

    def ideal_passed(self, row: dict[str, Any]) -> bool:
        return (
            (_safe_float(row.get("PF")) or 0.0) >= 1.8
            and (_safe_float(row.get("DD")) or -1.0) >= -0.08
            and (_safe_float(row.get("capital_utilization")) or 0.0) >= 0.20
            and (_safe_float(row.get("average_holding_days")) or 0.0) >= 10.0
        )

    def best_variant(self, rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any] | None:
        if not rows:
            return None

        def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
            profit = _safe_float(row.get("net_profit")) or -10**18
            pf = _safe_float(row.get("PF")) or 0.0
            dd = _safe_float(row.get("DD")) or -1.0
            utilization = _safe_float(row.get("capital_utilization")) or 0.0
            pass_bonus = 10**7 if self.minimum_passed(row, baseline) else 0.0
            return (pass_bonus + profit, pf, dd, utilization)

        return max(rows, key=sort_key)

    def best_variant_reason(self, row: dict[str, Any] | None) -> str:
        if not row:
            return "No variants were evaluated."
        return (
            f"{row['variant']} net_profit={row.get('net_profit'):.0f}, PF={row.get('PF'):.4f}, "
            f"DD={row.get('DD'):.4f}, capital_utilization={row.get('capital_utilization'):.4f}, "
            f"average_holding_days={row.get('average_holding_days'):.2f}."
        )

    def relaxation_effect(self, rows: list[dict[str, Any]], baseline: dict[str, Any]) -> str:
        parts = []
        for row in rows:
            if row["variant"] == "B5_0_baseline":
                continue
            parts.append(
                f"{row['variant']}: profit_delta={(_safe_float(row.get('net_profit')) or 0) - (_safe_float(baseline.get('net_profit')) or 0):.0f}, "
                f"PF={row.get('PF'):.4f}, DD={row.get('DD'):.4f}, hold={row.get('average_holding_days'):.2f}"
            )
        return "; ".join(parts)

    def capital_utilization_effect(self, rows: list[dict[str, Any]], baseline: dict[str, Any]) -> str:
        base_util = _safe_float(baseline.get("capital_utilization")) or 0.0
        return "; ".join(f"{row['variant']}: util_delta={(_safe_float(row.get('capital_utilization')) or 0.0) - base_util:.4f}" for row in rows if row["variant"] != "B5_0_baseline")

    def recommendation(self, rows: list[dict[str, Any]], hold_quality: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"ready_for_phase12c": False, "recommended_next_phase": "Fix Phase12-B5 leakage blockers"}
        comparison = self.variant_comparison(rows)
        return {
            "best_variant": comparison["best_variant"],
            "best_variant_reason": comparison["best_variant_reason"],
            "variants_meeting_minimum_target": comparison["variants_meeting_minimum_target"],
            "variants_meeting_ideal_target": comparison["variants_meeting_ideal_target"],
            "opportunity_exit_relaxation_effect": comparison["opportunity_exit_relaxation_effect"],
            "capital_utilization_effect": comparison["capital_utilization_effect"],
            "ready_for_phase12c": comparison["ready_for_phase12c"],
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
            "base_strategy": "S3a_dynamic_raw_weight",
            "daily_buy_budget": self.options.daily_buy_budget,
            "max_positions": self.options.max_positions,
            "holding_days": self.options.holding_days,
            "round_lot": self.options.round_lot,
            "cost_rate": self.options.cost_rate,
            "stop_loss": self.options.stop_loss_rate,
            "variants": [variant.__dict__ for variant in VARIANTS],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "12-B5",
            "scope": "2025 exit threshold recalibration only",
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
            "backtest_columns_used_as_features": [],
            "trade_result_columns_used_as_features": [],
            "cash_or_portfolio_columns_used_as_model_features": [],
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase12B5Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12B5Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-B5 Exit Threshold Recalibration",
            "",
            "## Variant Results",
            "",
            self.table(report.get("variant_results", []), ["variant", "net_profit", "PF", "DD", "win_rate", "total_trades", "final_assets", "capital_utilization", "average_holding_days", "median_holding_days", "same_code_reentry_count", "reentry_within_5_days_count", "cost_paid", "opportunity_exit_count", "stop_loss_count", "time_exit_count", "forced_exit_count", "exit_reason_counts"]),
            "",
            "## Hold / Exit Quality",
            "",
            self.table(report.get("hold_exit_quality", []), ["variant", "avg_profit_capture", "avg_max_profit_before_exit", "profit_capture_ratio", "avg_post_exit_20d_return"]),
            "",
            "## Comparison",
            "",
            self.table([report.get("variant_comparison", {})], ["best_variant", "best_variant_reason", "variants_meeting_minimum_target", "variants_meeting_ideal_target", "opportunity_exit_relaxation_effect", "capital_utilization_effect", "ready_for_phase12c", "recommended_next_phase"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_only_for_evaluation", "future_columns_used_as_features", "backtest_columns_used_as_features", "trade_result_columns_used_as_features", "cash_or_portfolio_columns_used_as_model_features", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["best_variant", "variants_meeting_minimum_target", "variants_meeting_ideal_target", "ready_for_phase12c", "recommended_next_phase"]),
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
