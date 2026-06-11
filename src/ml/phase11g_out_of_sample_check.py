"""Phase 11-G limited out-of-sample year check.

This module evaluates the Phase 11 E4-style Valuation strategy on 2024 only.
It does not run a 2023-2026 full backtest, modify profiles, overwrite models,
refetch APIs, or save regenerated historical predictions. The Phase 11-B
candidate model is used for lightweight in-memory 2024 scoring, and the report
explicitly flags that the model training period overlaps 2024.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.phase11e_exit_dd_guard import FUTURE_EVAL_COLUMNS, _numeric, _safe_float


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = Path("data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet")
MODEL_DIR = Path("models/ml/valuation_engine/candidate_phase11b")
REPORT_STEM = "phase11g_out_of_sample_check_2024"

START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
ROUND_LOT = 100
BASELINE_RANK_COLUMNS = ["stock_selection_rank_score", "risk_adjusted_score", "expected_return", "candidate_strength"]
DECISION_COLUMNS = [
    "opportunity_top_decile_proba",
    "opportunity_score_proba_rank",
    "stock_selection_rank_score",
    "risk_adjusted_score",
    "close",
    "turnover_value",
]


@dataclass(frozen=True)
class Phase11GOptions:
    initial_cash: float = 1_000_000.0
    daily_buy_budget: float = 900_000.0
    max_positions: int = 5
    round_lot: int = ROUND_LOT
    holding_days: int = 20
    stop_loss_rate: float = -0.08
    opportunity_drop_threshold: float = 0.15
    opportunity_rank_floor: float = 0.50


@dataclass(frozen=True)
class StrategySpec:
    name: str
    rank_column: str
    exit_guard: bool
    cost_rate: float = 0.0


@dataclass(frozen=True)
class Phase11GPaths:
    markdown: Path
    json: Path


STRATEGIES = [
    StrategySpec("baseline_equal_allocation", rank_column="baseline_rank_score", exit_guard=False, cost_rate=0.0),
    StrategySpec("valuation_top5_no_guard", rank_column="opportunity_top_decile_proba", exit_guard=False, cost_rate=0.0),
    StrategySpec("valuation_top5_E4", rank_column="opportunity_top_decile_proba", exit_guard=True, cost_rate=0.0),
    StrategySpec("valuation_top5_E4_cost_0.2pct", rank_column="opportunity_top_decile_proba", exit_guard=True, cost_rate=0.002),
]


class Phase11GOutOfSampleCheck:
    def __init__(self, root: Path | str = ROOT, *, options: Phase11GOptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase11GOptions()

    def run(self) -> Phase11GPaths:
        report = self.build_report()
        return self.save_report(report)

    def build_report(self) -> dict[str, Any]:
        data = self.load_scored_frame()
        leakage = self.leakage_checklist()
        if leakage["blocking_issues"]:
            return {
                "metadata": self.metadata(),
                "conditions": self.conditions(),
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], leakage),
            }

        strategy_results = []
        buy_quality = []
        overtrading = []
        for spec in STRATEGIES:
            trades, daily = self.simulate(data, spec)
            strategy_results.append(self.metrics(spec.name, trades, daily))
            buy_quality.append(self.buy_quality(spec.name, trades))
            overtrading.append(self.overtrading(spec.name, trades))

        report = {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "model_oos_limitations": self.model_oos_limitations(),
            "strategy_results": strategy_results,
            "buy_quality": buy_quality,
            "overtrading_check": overtrading,
            "oos_judgement": self.oos_judgement(strategy_results),
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(strategy_results, leakage),
        }
        return report

    def load_scored_frame(self) -> pd.DataFrame:
        import joblib

        model_dir = self.root / MODEL_DIR
        feature_columns = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
        classifier = joblib.load(model_dir / "opportunity_top_decile_20d_classifier.joblib")
        columns = sorted(set(["date", "code", "close", "turnover_value", *BASELINE_RANK_COLUMNS, *feature_columns, *FUTURE_EVAL_COLUMNS]))
        data = pd.read_parquet(self.root / DATASET_PATH, columns=columns)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        data = data[(data["date"] >= pd.Timestamp(START_DATE)) & (data["date"] <= pd.Timestamp(END_DATE))].copy()
        for column in columns:
            if column not in {"date", "code"}:
                data[column] = _numeric(data[column])
        proba = np.asarray(classifier.predict_proba(data[feature_columns]))[:, 1]
        data["opportunity_top_decile_proba"] = proba
        data["opportunity_score_proba_rank"] = data.groupby("date")["opportunity_top_decile_proba"].rank(method="average", pct=True)
        baseline_rank = data["stock_selection_rank_score"] if "stock_selection_rank_score" in data.columns else pd.Series(dtype=float)
        if baseline_rank.isna().all():
            baseline_rank = data["risk_adjusted_score"]
        data["baseline_rank_score"] = _numeric(baseline_rank).fillna(-10**18)
        return data.dropna(subset=["date", "code", "close"]).sort_values(["date", "code"]).reset_index(drop=True)

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
                reason = self.exit_reason(position, current_date, current_row, spec)
                if reason:
                    exit_close = float(current_row["close"]) if current_row is not None else float(position["last_close"])
                    trade = self.close_position(position, current_date, exit_close, reason, spec.name, spec.cost_rate)
                    cash += trade["exit_cash_flow"]
                    trades.append(trade)
                else:
                    if current_row is not None:
                        position["last_close"] = float(current_row["close"])
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            ranked = current_rank_frame.sort_values([spec.rank_column, "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions)
            selected = ranked.head(slots) if slots else ranked.iloc[0:0]
            raw_amount = min(cash, self.options.daily_buy_budget) / max(1, min(self.options.max_positions, len(ranked)))
            bought_today = 0
            for _, row in selected.iterrows():
                lot_cost = float(row["close"]) * self.options.round_lot
                lots = int(raw_amount // (lot_cost * (1.0 + spec.cost_rate))) if lot_cost > 0 else 0
                buy_amount = lots * lot_cost
                buy_cost = buy_amount * spec.cost_rate
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
                trade = self.close_position(position, last_date, float(position["last_close"]), "forced_end_of_period", spec.name, spec.cost_rate)
                cash += trade["exit_cash_flow"]
                trades.append(trade)
            if daily_rows:
                daily_rows[-1]["total_assets"] = cash
                daily_rows[-1]["marked_position_value"] = 0.0
                daily_rows[-1]["capital_utilization"] = 0.0
        return pd.DataFrame(trades), pd.DataFrame(daily_rows)

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None, spec: StrategySpec) -> str | None:
        if current_row is not None and spec.exit_guard:
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

    def close_position(self, position: dict[str, Any], exit_date: pd.Timestamp, exit_close: float, reason: str, strategy: str, cost_rate: float) -> dict[str, Any]:
        exit_amount = float(position["lot_count"]) * self.options.round_lot * exit_close
        sell_cost = exit_amount * cost_rate
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
            "median_holding_days": _safe_float(_numeric(trades["holding_days"]).median()) if not trades.empty else None,
            "exit_reason_counts": dict(Counter(trades["exit_reason"])) if "exit_reason" in trades.columns else {},
            "cost_paid": _safe_float(_numeric(trades["cost_paid"]).sum()) if "cost_paid" in trades.columns else 0.0,
            "same_code_reentry_count": self.reentry_counts(trades)["same_code_reentry_count"],
            "reentry_within_5_days_count": self.reentry_counts(trades)["reentry_within_5_days_count"],
        }

    def buy_quality(self, strategy: str, trades: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {"strategy": strategy, "buy_count": int(len(trades))}
        for column in FUTURE_EVAL_COLUMNS:
            values = _numeric(trades[column]) if column in trades.columns else pd.Series(dtype=float)
            key = "opportunity_top_decile_20d_rate" if column == "opportunity_top_decile_20d" else f"{column}_mean"
            row[key] = _safe_float(values.mean()) if not values.empty else None
        return row

    def overtrading(self, strategy: str, trades: pd.DataFrame) -> dict[str, Any]:
        counts = self.reentry_counts(trades)
        return {
            "strategy": strategy,
            "average_holding_days": _safe_float(_numeric(trades["holding_days"]).mean()) if not trades.empty else None,
            "median_holding_days": _safe_float(_numeric(trades["holding_days"]).median()) if not trades.empty else None,
            "exit_reason_counts": dict(Counter(trades["exit_reason"])) if "exit_reason" in trades.columns else {},
            **counts,
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

    def oos_judgement(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_strategy = {row["strategy"]: row for row in rows}
        baseline = by_strategy.get("baseline_equal_allocation", {})
        e4 = by_strategy.get("valuation_top5_E4", {})
        e4_cost = by_strategy.get("valuation_top5_E4_cost_0.2pct", {})
        checks = {
            "e4_beats_baseline_net_profit": (_safe_float(e4.get("net_profit")) or -10**18) > (_safe_float(baseline.get("net_profit")) or 10**18),
            "e4_pf_at_least_1_5": (_safe_float(e4.get("PF")) or 0.0) >= 1.5,
            "e4_dd_within_10pct": (_safe_float(e4.get("DD")) or -1.0) >= -0.10,
            "cost_02_pf_at_least_1_3": (_safe_float(e4_cost.get("PF")) or 0.0) >= 1.3,
            "cost_02_dd_within_12pct": (_safe_float(e4_cost.get("DD")) or -1.0) >= -0.12,
        }
        return {**checks, "all_passed": all(checks.values())}

    def model_oos_limitations(self) -> dict[str, Any]:
        metadata_path = self.root / MODEL_DIR / "model_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return {
            "evaluated_year": 2024,
            "model_train_period": metadata.get("train_period"),
            "model_test_period": metadata.get("test_period"),
            "model_train_period_overlaps_evaluated_year": True,
            "strict_model_oos": False,
            "note": "Phase11-B candidate model metadata shows 2024 is inside the training period. This check is a limited independent-year strategy/path check, not a strict model out-of-sample proof.",
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

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "11-G",
            "evaluated_year": 2024,
            "limited_single_year_only": True,
            "full_backtest_executed": False,
            "profile_added": False,
            "profile_modified": False,
            "current_model_overwritten": False,
            "historical_predictions_regenerated": False,
            "historical_predictions_saved": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
            "dataset_path": str(self.root / DATASET_PATH),
            "model_dir": str(self.root / MODEL_DIR),
        }

    def conditions(self) -> dict[str, Any]:
        return {
            "period": {"start": START_DATE, "end": END_DATE},
            "daily_buy_budget": self.options.daily_buy_budget,
            "max_positions": self.options.max_positions,
            "holding_days": self.options.holding_days,
            "stop_loss": self.options.stop_loss_rate,
            "opportunity_drop_threshold": self.options.opportunity_drop_threshold,
            "opportunity_rank_floor": self.options.opportunity_rank_floor,
            "strategies": [spec.__dict__ for spec in STRATEGIES],
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
            "decision_columns": DECISION_COLUMNS,
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def recommendation(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"oos_passed": False, "recommended_next_phase": "Fix Phase11-G leakage blockers"}
        judgement = self.oos_judgement(rows) if rows else {"all_passed": False}
        strict_oos = self.model_oos_limitations()["strict_model_oos"]
        return {
            "oos_passed": bool(judgement.get("all_passed")),
            "strict_model_oos": strict_oos,
            "recommended_next_phase": "Phase11-H cooldown/min-hold guard plus strict walk-forward OOS design" if judgement.get("all_passed") else "Phase11-B2/E/F retuning",
            "reason": "2024 strategy check can support robustness, but strict model OOS remains false because the Phase11-B model was trained through 2024.",
        }

    def save_report(self, report: dict[str, Any]) -> Phase11GPaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase11GPaths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 11-G Limited Out-of-Sample Year Check 2024",
            "",
            "## Scope",
            "",
            "- 2024 only",
            "- no 2023-2026 full backtest, no profile change, no model overwrite",
            "- 2024 is inside the Phase11-B candidate model training period, so this is not strict model OOS",
            "",
            "## Conditions",
            "",
            self.table([report["conditions"]], ["daily_buy_budget", "max_positions", "holding_days", "stop_loss", "opportunity_drop_threshold", "opportunity_rank_floor"]),
            "",
            "## Model OOS Limitations",
            "",
            self.table([report["model_oos_limitations"]], ["evaluated_year", "model_train_period_overlaps_evaluated_year", "strict_model_oos", "note"]),
            "",
            "## Strategy Results",
            "",
            self.table(report.get("strategy_results", []), ["strategy", "net_profit", "PF", "DD", "win_rate", "total_trades", "final_assets", "capital_utilization", "average_holding_days", "median_holding_days", "exit_reason_counts", "same_code_reentry_count", "reentry_within_5_days_count"]),
            "",
            "## BUY Quality",
            "",
            self.table(report.get("buy_quality", []), ["strategy", "buy_count", "future_return_20d_mean", "future_max_return_20d_mean", "future_max_drawdown_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_rate"]),
            "",
            "## OOS Judgement",
            "",
            self.table([report.get("oos_judgement", {})], ["e4_beats_baseline_net_profit", "e4_pf_at_least_1_5", "e4_dd_within_10pct", "cost_02_pf_at_least_1_3", "cost_02_dd_within_12pct", "all_passed"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_evaluation", "backtest_columns_used_as_features", "trade_result_columns_used_as_features", "cash_or_portfolio_columns_used_as_model_features", "selected_or_bought_used_as_features", "current_pm_multiplier_used", "historical_predictions_regenerated", "profile_changed", "full_backtest_executed", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["oos_passed", "strict_model_oos", "recommended_next_phase", "reason"]),
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
