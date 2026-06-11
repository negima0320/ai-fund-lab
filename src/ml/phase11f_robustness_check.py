"""Phase 11-F limited robustness check for the Phase 11-E E4 candidate.

The check stays in 2025 only and reuses Phase 11-A/C artifacts. It tests
transaction cost, opportunity-exit threshold sensitivity, and overtrading risk
without changing profiles, overwriting models, regenerating historical
predictions, or using future columns for trade decisions.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import (
    DATASET_PATH,
    FUTURE_EVAL_COLUMNS,
    ROOT,
    SIMULATION_PATH,
    START_DATE,
    END_DATE,
    _numeric,
    _safe_float,
)


REPORT_STEM = "phase11f_robustness_check_2025"


@dataclass(frozen=True)
class Phase11FOptions:
    initial_cash: float = 1_000_000.0
    daily_buy_budget: float = 900_000.0
    max_positions: int = 5
    round_lot: int = 100
    holding_days: int = 20
    stop_loss_rate: float = -0.08


@dataclass(frozen=True)
class ThresholdProfile:
    name: str
    opportunity_drop_threshold: float
    opportunity_rank_floor: float


@dataclass(frozen=True)
class Phase11FPaths:
    markdown: Path
    json: Path


COST_RATES = [0.0, 0.001, 0.002, 0.003]
THRESHOLD_PROFILES = [
    ThresholdProfile("loose", opportunity_drop_threshold=0.25, opportunity_rank_floor=0.40),
    ThresholdProfile("baseline", opportunity_drop_threshold=0.15, opportunity_rank_floor=0.50),
    ThresholdProfile("strict", opportunity_drop_threshold=0.08, opportunity_rank_floor=0.60),
]


class Phase11FRobustnessCheck:
    def __init__(self, root: Path | str = ROOT, *, options: Phase11FOptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase11FOptions()

    def run(self) -> Phase11FPaths:
        report = self.build_report()
        return self.save_report(report)

    def build_report(self) -> dict[str, Any]:
        data = self.load_frame()
        leakage = self.leakage_checklist()
        if leakage["blocking_issues"]:
            return {
                "metadata": self.metadata(),
                "conditions": self.conditions(),
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], [], leakage),
            }

        cost_rows = []
        baseline_trades = pd.DataFrame()
        baseline_daily = pd.DataFrame()
        for cost_rate in COST_RATES:
            trades, daily = self.simulate(
                data,
                variant=f"cost_{cost_rate:.3f}",
                cost_rate=cost_rate,
                threshold=THRESHOLD_PROFILES[1],
            )
            if cost_rate == 0.0:
                baseline_trades = trades
                baseline_daily = daily
            cost_rows.append(self.metrics(f"cost_{cost_rate:.3f}", trades, daily, cost_rate=cost_rate))

        threshold_rows = []
        for threshold in THRESHOLD_PROFILES:
            trades, daily = self.simulate(data, variant=f"threshold_{threshold.name}", cost_rate=0.0, threshold=threshold)
            threshold_rows.append(self.metrics(f"threshold_{threshold.name}", trades, daily, threshold=threshold))

        overtrading = self.overtrading_check(baseline_trades, baseline_daily)
        report = {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "cost_sensitivity": cost_rows,
            "threshold_sensitivity": threshold_rows,
            "overtrading_check": overtrading,
            "combined_robustness_summary": self.combined_summary(cost_rows, threshold_rows, overtrading),
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(cost_rows, threshold_rows, leakage),
        }
        return report

    def load_frame(self) -> pd.DataFrame:
        simulation = pd.read_parquet(self.root / SIMULATION_PATH)
        simulation["date"] = pd.to_datetime(simulation["date"], errors="coerce")
        simulation["code"] = simulation["code"].astype("string")
        simulation = simulation[
            (simulation["rule"] == "equal_weight_top5")
            & (simulation["date"] >= pd.Timestamp(START_DATE))
            & (simulation["date"] <= pd.Timestamp(END_DATE))
        ].copy()
        keep = ["date", "code", "opportunity_top_decile_proba", "opportunity_score_proba_rank", *FUTURE_EVAL_COLUMNS]
        dataset = pd.read_parquet(self.root / DATASET_PATH, columns=["date", "code", "close", "turnover_value"])
        dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
        dataset["code"] = dataset["code"].astype("string")
        dataset = dataset[
            (dataset["date"] >= pd.Timestamp(START_DATE))
            & (dataset["date"] <= pd.Timestamp(END_DATE))
        ].drop_duplicates(["date", "code"], keep="last")
        data = simulation[keep].merge(dataset, on=["date", "code"], how="left", validate="many_to_one")
        for column in ["close", "turnover_value", "opportunity_top_decile_proba", "opportunity_score_proba_rank", *FUTURE_EVAL_COLUMNS]:
            data[column] = _numeric(data[column])
        return data.dropna(subset=["date", "code", "close"]).sort_values(["date", "code"]).reset_index(drop=True)

    def simulate(self, data: pd.DataFrame, *, variant: str, cost_rate: float, threshold: ThresholdProfile) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                reason = self.exit_reason(position, current_date, current_row, threshold)
                if reason:
                    exit_close = float(current_row["close"]) if current_row is not None else float(position["last_close"])
                    trade = self.close_position(position, current_date, exit_close, reason, variant, cost_rate)
                    cash += trade["exit_cash_flow"]
                    trades.append(trade)
                else:
                    if current_row is not None:
                        position["last_close"] = float(current_row["close"])
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            ranked = current_rank_frame.sort_values(["opportunity_top_decile_proba", "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions)
            selected = ranked.head(slots) if slots else ranked.iloc[0:0]
            raw_amount = min(cash, self.options.daily_buy_budget) / max(1, min(self.options.max_positions, len(ranked)))
            bought_today = 0
            for _, row in selected.iterrows():
                lot_cost = float(row["close"]) * self.options.round_lot
                lots = int(raw_amount // (lot_cost * (1.0 + cost_rate))) if lot_cost > 0 else 0
                buy_amount = lots * lot_cost
                buy_cost = buy_amount * cost_rate
                cash_out = buy_amount + buy_cost
                if lots <= 0 or cash_out > cash:
                    continue
                cash -= cash_out
                bought_today += 1
                positions.append(
                    {
                        "entry_date": current_date,
                        "due_date": current_date + pd.offsets.BDay(self.options.holding_days),
                        "code": str(row["code"]),
                        "buy_amount": buy_amount,
                        "buy_cost": buy_cost,
                        "lot_count": lots,
                        "entry_close": float(row["close"]),
                        "last_close": float(row["close"]),
                        "entry_opportunity_top_decile_proba": _safe_float(row.get("opportunity_top_decile_proba")),
                        "entry_opportunity_score_proba_rank": _safe_float(row.get("opportunity_score_proba_rank")),
                        **{column: _safe_float(row.get(column)) for column in FUTURE_EVAL_COLUMNS},
                    }
                )

            marked_value = sum(float(position["lot_count"]) * self.options.round_lot * float(position["last_close"]) for position in positions)
            daily_rows.append(
                {
                    "variant": variant,
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
                trade = self.close_position(position, last_date, float(position["last_close"]), "forced_end_of_period", variant, cost_rate)
                cash += trade["exit_cash_flow"]
                trades.append(trade)
            if daily_rows:
                daily_rows[-1]["total_assets"] = cash
                daily_rows[-1]["marked_position_value"] = 0.0
                daily_rows[-1]["capital_utilization"] = 0.0
        return pd.DataFrame(trades), pd.DataFrame(daily_rows)

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None, threshold: ThresholdProfile) -> str | None:
        if current_row is not None:
            observed_return = float(current_row["close"]) / float(position["entry_close"]) - 1.0
            if observed_return <= self.options.stop_loss_rate:
                return "stop_loss"
            current_proba = _safe_float(current_row.get("opportunity_top_decile_proba"))
            current_rank = _safe_float(current_row.get("opportunity_score_proba_rank"))
            entry_proba = _safe_float(position.get("entry_opportunity_top_decile_proba"))
            if current_rank is not None and current_rank < threshold.opportunity_rank_floor:
                return "opportunity_rank_below_floor"
            if current_proba is not None and entry_proba is not None and current_proba <= entry_proba - threshold.opportunity_drop_threshold:
                return "opportunity_proba_drop"
        if current_date >= position["due_date"]:
            return "time_exit_20d"
        return None

    def close_position(self, position: dict[str, Any], exit_date: pd.Timestamp, exit_close: float, reason: str, variant: str, cost_rate: float) -> dict[str, Any]:
        exit_amount = float(position["lot_count"]) * self.options.round_lot * exit_close
        sell_cost = exit_amount * cost_rate
        exit_cash_flow = exit_amount - sell_cost
        total_cost = float(position["buy_cost"]) + sell_cost
        profit = exit_cash_flow - float(position["buy_amount"]) - float(position["buy_cost"])
        holding_days = len(pd.bdate_range(position["entry_date"], exit_date)) - 1
        return {
            "variant": variant,
            "entry_date": position["entry_date"],
            "exit_date": exit_date,
            "code": position["code"],
            "buy_amount": position["buy_amount"],
            "exit_amount": exit_amount,
            "exit_cash_flow": exit_cash_flow,
            "entry_close": position["entry_close"],
            "exit_close": exit_close,
            "realized_profit": profit,
            "realized_return": profit / float(position["buy_amount"]) if position["buy_amount"] else None,
            "holding_days": holding_days,
            "exit_reason": reason,
            "cost_paid": total_cost,
            **{column: position.get(column) for column in FUTURE_EVAL_COLUMNS},
        }

    def metrics(self, name: str, trades: pd.DataFrame, daily: pd.DataFrame, *, cost_rate: float | None = None, threshold: ThresholdProfile | None = None) -> dict[str, Any]:
        profits = _numeric(trades["realized_profit"]) if not trades.empty else pd.Series(dtype=float)
        gross_profit = float(profits[profits > 0].sum()) if not profits.empty else 0.0
        gross_loss = abs(float(profits[profits < 0].sum())) if not profits.empty else 0.0
        equity = _numeric(daily["total_assets"]) if not daily.empty else pd.Series([self.options.initial_cash])
        drawdown = equity / equity.cummax() - 1.0
        row = {
            "variant": name,
            "net_profit": _safe_float(profits.sum()) if not profits.empty else 0.0,
            "PF": _safe_float(gross_profit / gross_loss) if gross_loss else (None if gross_profit == 0 else float("inf")),
            "DD": _safe_float(drawdown.min()) if not drawdown.empty else 0.0,
            "win_rate": _safe_float((profits > 0).mean()) if not profits.empty else None,
            "total_trades": int(len(trades)),
            "final_assets": _safe_float(self.options.initial_cash + profits.sum()) if not profits.empty else self.options.initial_cash,
            "capital_utilization": _safe_float(_numeric(daily["capital_utilization"]).mean()) if not daily.empty else None,
            "average_holding_days": _safe_float(_numeric(trades["holding_days"]).mean()) if not trades.empty else None,
            "median_holding_days": _safe_float(_numeric(trades["holding_days"]).median()) if not trades.empty else None,
            "cost_paid": _safe_float(_numeric(trades["cost_paid"]).sum()) if "cost_paid" in trades.columns else 0.0,
            "exit_reason_counts": dict(Counter(trades["exit_reason"])) if "exit_reason" in trades.columns else {},
        }
        if cost_rate is not None:
            row["cost_rate"] = cost_rate
        if threshold is not None:
            row["threshold_profile"] = threshold.name
            row["opportunity_drop_threshold"] = threshold.opportunity_drop_threshold
            row["opportunity_rank_floor"] = threshold.opportunity_rank_floor
        return row

    def overtrading_check(self, trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
        if trades.empty:
            return {"monthly": [], "same_code_reentry_count": 0, "reentry_within_5_days_count": 0}
        trades = trades.copy()
        trades["entry_month"] = pd.to_datetime(trades["entry_date"]).dt.to_period("M").astype(str)
        monthly = []
        for month, group in trades.groupby("entry_month", sort=True):
            profits = _numeric(group["realized_profit"])
            gross_profit = float(profits[profits > 0].sum())
            gross_loss = abs(float(profits[profits < 0].sum()))
            month_daily = daily[pd.to_datetime(daily["date"]).dt.to_period("M").astype(str) == month]
            equity = _numeric(month_daily["total_assets"]) if not month_daily.empty else pd.Series(dtype=float)
            dd = (equity / equity.cummax() - 1.0).min() if not equity.empty else None
            monthly.append(
                {
                    "month": month,
                    "monthly_trade_count": int(len(group)),
                    "monthly_net_profit": _safe_float(profits.sum()),
                    "monthly_pf": _safe_float(gross_profit / gross_loss) if gross_loss else (None if gross_profit == 0 else float("inf")),
                    "monthly_dd": _safe_float(dd),
                }
            )
        ordered = trades.sort_values(["code", "entry_date"])
        same_code_reentry = 0
        reentry_5d = 0
        for _, group in ordered.groupby("code", sort=False):
            previous_exit = None
            for _, row in group.iterrows():
                entry = pd.Timestamp(row["entry_date"])
                if previous_exit is not None:
                    same_code_reentry += 1
                    if len(pd.bdate_range(previous_exit, entry)) - 1 <= 5:
                        reentry_5d += 1
                previous_exit = pd.Timestamp(row["exit_date"])
        return {
            "monthly": monthly,
            "average_holding_days": _safe_float(_numeric(trades["holding_days"]).mean()),
            "median_holding_days": _safe_float(_numeric(trades["holding_days"]).median()),
            "trade_count_by_exit_reason": dict(Counter(trades["exit_reason"])),
            "same_code_reentry_count": same_code_reentry,
            "reentry_within_5_days_count": reentry_5d,
        }

    def combined_summary(self, cost_rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]], overtrading: dict[str, Any]) -> dict[str, Any]:
        baseline = next(row for row in cost_rows if row.get("cost_rate") == 0.0)
        cost_02 = next(row for row in cost_rows if row.get("cost_rate") == 0.002)
        checks = {
            "baseline_pf_at_least_2": (_safe_float(baseline.get("PF")) or 0.0) >= 2.0,
            "baseline_dd_within_10pct": (_safe_float(baseline.get("DD")) or -1.0) >= -0.10,
            "baseline_net_profit_at_least_300k": (_safe_float(baseline.get("net_profit")) or 0.0) >= 300_000,
            "average_holding_days_at_least_5": (_safe_float(baseline.get("average_holding_days")) or 0.0) >= 5.0,
            "cost_02_pf_at_least_1_8": (_safe_float(cost_02.get("PF")) or 0.0) >= 1.8,
            "cost_02_dd_within_10pct": (_safe_float(cost_02.get("DD")) or -1.0) >= -0.10,
        }
        return {
            **checks,
            "all_passed": all(checks.values()),
            "same_code_reentry_count": overtrading.get("same_code_reentry_count"),
            "reentry_within_5_days_count": overtrading.get("reentry_within_5_days_count"),
            "threshold_profiles_tested": [row["threshold_profile"] for row in threshold_rows],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "11-F",
            "limited_2025_only": True,
            "full_backtest_executed": False,
            "profile_added": False,
            "profile_modified": False,
            "current_model_overwritten": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
            "simulation_path": str(self.root / SIMULATION_PATH),
            "dataset_path": str(self.root / DATASET_PATH),
        }

    def conditions(self) -> dict[str, Any]:
        return {
            "period": {"start": START_DATE, "end": END_DATE},
            "base_strategy": "E4_stop_loss_8pct_plus_opportunity",
            "daily_buy_budget": self.options.daily_buy_budget,
            "max_positions": self.options.max_positions,
            "holding_days": self.options.holding_days,
            "rank_column": "opportunity_top_decile_proba",
            "candidate_threshold": "top5",
            "stop_loss": self.options.stop_loss_rate,
            "cost_rates": COST_RATES,
            "threshold_profiles": [profile.__dict__ for profile in THRESHOLD_PROFILES],
        }

    def dataset_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(data)),
            "unique_codes": int(data["code"].nunique()),
            "candidate_days": int(data["date"].nunique()),
            "date_range": {
                "min": data["date"].min().date().isoformat() if not data.empty else None,
                "max": data["date"].max().date().isoformat() if not data.empty else None,
            },
        }

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_as_features": [],
            "future_columns_used_only_for_evaluation": FUTURE_EVAL_COLUMNS,
            "backtest_columns_used_as_features": [],
            "trade_result_columns_used_as_features": [],
            "cash_or_portfolio_columns_used_as_model_features": [],
            "selected_or_bought_used_as_features": False,
            "current_pm_multiplier_used": False,
            "historical_predictions_regenerated": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "decision_columns": ["opportunity_top_decile_proba", "opportunity_score_proba_rank", "close", "turnover_value"],
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def recommendation(self, cost_rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"robustness_passed": False, "recommended_next_phase": "Fix Phase11-F leakage blockers"}
        summary = self.combined_summary(cost_rows, threshold_rows, {"same_code_reentry_count": None, "reentry_within_5_days_count": None}) if cost_rows and threshold_rows else {"all_passed": False}
        return {
            "robustness_passed": bool(summary.get("all_passed")),
            "recommended_next_phase": "Phase11-G limited out-of-sample year check" if summary.get("all_passed") else "Phase11-E/F threshold retuning",
            "reason": "2025-only research pass requires PF >= 2.0, DD >= -10%, net_profit >= 300k, average holding >= 5d, and cost 0.2% resilience.",
        }

    def save_report(self, report: dict[str, Any]) -> Phase11FPaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase11FPaths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 11-F Limited Robustness Check 2025",
            "",
            "## Scope",
            "",
            "- 2025 only",
            "- base strategy: E4 stop-loss 8% + Opportunity Exit",
            "- tests cost sensitivity, opportunity threshold sensitivity, and overtrading",
            "- no full-period backtest, no profile change, no model overwrite",
            "",
            "## Conditions",
            "",
            self.table([report["conditions"]], ["base_strategy", "daily_buy_budget", "max_positions", "holding_days", "stop_loss"]),
            "",
            "## Cost Sensitivity",
            "",
            self.table(report.get("cost_sensitivity", []), ["variant", "cost_rate", "net_profit", "PF", "DD", "win_rate", "total_trades", "average_holding_days", "cost_paid"]),
            "",
            "## Opportunity Exit Threshold Sensitivity",
            "",
            self.table(report.get("threshold_sensitivity", []), ["variant", "threshold_profile", "opportunity_drop_threshold", "opportunity_rank_floor", "net_profit", "PF", "DD", "win_rate", "total_trades", "average_holding_days", "exit_reason_counts"]),
            "",
            "## Monthly Overtrading Check",
            "",
            self.table(report.get("overtrading_check", {}).get("monthly", []), ["month", "monthly_trade_count", "monthly_net_profit", "monthly_pf", "monthly_dd"]),
            "",
            "## Overtrading Summary",
            "",
            self.table([report.get("overtrading_check", {})], ["average_holding_days", "median_holding_days", "trade_count_by_exit_reason", "same_code_reentry_count", "reentry_within_5_days_count"]),
            "",
            "## Combined Robustness Summary",
            "",
            self.table([report.get("combined_robustness_summary", {})], ["baseline_pf_at_least_2", "baseline_dd_within_10pct", "baseline_net_profit_at_least_300k", "average_holding_days_at_least_5", "cost_02_pf_at_least_1_8", "cost_02_dd_within_10pct", "all_passed"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_evaluation", "backtest_columns_used_as_features", "trade_result_columns_used_as_features", "cash_or_portfolio_columns_used_as_model_features", "selected_or_bought_used_as_features", "current_pm_multiplier_used", "historical_predictions_regenerated", "profile_changed", "full_backtest_executed", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["robustness_passed", "recommended_next_phase", "reason"]),
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
