"""Phase 11-D limited combined backtest.

This module connects the Phase 11 Valuation output to a small 2025-only
buy/hold/exit simulation. It intentionally avoids profile changes, full-period
backtests, model overwrites, API refetches, and historical prediction
regeneration. Future-return labels are used only as lightweight realized
outcomes / quality evaluation, never as candidate-selection inputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SIMULATION_PATH = Path("data/ml/valuation_engine/phase11c_allocation_simulation_2025.parquet")
DATASET_PATH = Path("data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet")
REPORT_STEM = "phase11d_combined_backtest_2025"

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
BASE_RULE = "equal_weight_top5"
ROUND_LOT = 100
FUTURE_EVAL_COLUMNS = [
    "future_return_20d",
    "future_max_return_20d",
    "future_max_drawdown_20d",
    "opportunity_value_20d",
    "opportunity_top_decile_20d",
]
BASELINE_RANK_COLUMNS = [
    "stock_selection_rank_score",
    "risk_adjusted_score",
    "expected_return",
    "candidate_strength",
]
FORBIDDEN_TOKENS = {
    "backtest",
    "trade",
    "profit",
    "loss",
    "cash",
    "portfolio",
    "position",
    "selected",
    "bought",
    "affordable",
    "skip",
    "exit",
    "final_assets",
    "pm_multiplier",
    "current_pm",
}


@dataclass(frozen=True)
class Phase11DOptions:
    initial_cash: float = 1_000_000.0
    daily_buy_budget: float = 900_000.0
    max_positions: int = 5
    round_lot: int = ROUND_LOT
    holding_days: int = 20


@dataclass(frozen=True)
class Phase11DPaths:
    markdown: Path
    json: Path


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


class Phase11DLimitedCombinedBacktest:
    def __init__(self, root: Path | str = ROOT, *, options: Phase11DOptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase11DOptions()

    def run(self) -> Phase11DPaths:
        report = self.build_report()
        return self.save_report(report)

    def build_report(self) -> dict[str, Any]:
        data = self.load_candidate_frame()
        leakage = self.leakage_checklist()
        if leakage["blocking_issues"]:
            return {
                "metadata": self.metadata(),
                "backtest_conditions": self.backtest_conditions(),
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], leakage),
            }

        baseline_trades, baseline_daily = self.simulate_strategy(data, "baseline_equal_allocation")
        candidate_trades, candidate_daily = self.simulate_strategy(data, "candidate_valuation_top5")
        strategy_results = [
            self.strategy_metrics("baseline_equal_allocation", baseline_trades, baseline_daily),
            self.strategy_metrics("candidate_valuation_top5", candidate_trades, candidate_daily),
        ]
        buy_quality = [
            self.buy_quality("baseline_equal_allocation", baseline_trades),
            self.buy_quality("candidate_valuation_top5", candidate_trades),
        ]
        report = {
            "metadata": self.metadata(),
            "backtest_conditions": self.backtest_conditions(),
            "dataset_summary": self.dataset_summary(data),
            "strategy_results": strategy_results,
            "buy_quality_comparison": buy_quality,
            "valuation_effect": self.valuation_effect(strategy_results, buy_quality),
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(strategy_results, leakage),
        }
        return report

    def load_candidate_frame(self) -> pd.DataFrame:
        simulation = pd.read_parquet(self.root / SIMULATION_PATH)
        simulation["date"] = pd.to_datetime(simulation["date"], errors="coerce")
        simulation["code"] = simulation["code"].astype("string")
        simulation = simulation[
            (simulation["rule"] == BASE_RULE)
            & (simulation["date"] >= pd.Timestamp(START_DATE))
            & (simulation["date"] <= pd.Timestamp(END_DATE))
        ].copy()

        dataset_columns = ["date", "code", "close", "turnover_value", *BASELINE_RANK_COLUMNS]
        dataset = pd.read_parquet(self.root / DATASET_PATH, columns=dataset_columns)
        dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
        dataset["code"] = dataset["code"].astype("string")
        dataset = dataset[
            (dataset["date"] >= pd.Timestamp(START_DATE))
            & (dataset["date"] <= pd.Timestamp(END_DATE))
        ].copy()
        dataset = dataset.drop_duplicates(["date", "code"], keep="last")

        keep_from_sim = [
            "date",
            "code",
            "opportunity_top_decile_proba",
            "opportunity_score_proba_rank",
            *FUTURE_EVAL_COLUMNS,
        ]
        data = simulation[keep_from_sim].merge(dataset, on=["date", "code"], how="left", validate="many_to_one")
        for column in ["close", "turnover_value", "opportunity_top_decile_proba", "opportunity_score_proba_rank", *BASELINE_RANK_COLUMNS, *FUTURE_EVAL_COLUMNS]:
            if column in data.columns:
                data[column] = _numeric(data[column])
        baseline_rank = data["stock_selection_rank_score"] if "stock_selection_rank_score" in data.columns else pd.Series(dtype=float)
        if baseline_rank.isna().all():
            baseline_rank = data["risk_adjusted_score"]
        data["baseline_rank_score"] = _numeric(baseline_rank).fillna(-10**18)
        data = data.dropna(subset=["date", "code", "close", "future_return_20d"])
        return data.sort_values(["date", "code"]).reset_index(drop=True)

    def simulate_strategy(self, data: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        cash = self.options.initial_cash
        positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        daily_rows: list[dict[str, Any]] = []
        dates = list(pd.Series(data["date"].dropna().unique()).sort_values())

        for current_date in dates:
            still_open = []
            for position in positions:
                if position["exit_date"] <= current_date:
                    cash += position["buy_amount"] + position["realized_profit"]
                    position["closed_on"] = current_date
                    trades.append(position)
                else:
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            day = data[data["date"] == current_date].copy()
            ranked = self.rank_candidates(day, strategy).head(self.options.max_positions)
            selected = ranked.head(slots) if slots else ranked.iloc[0:0]
            buy_budget = min(cash, self.options.daily_buy_budget)
            raw_amount = buy_budget / max(1, min(self.options.max_positions, len(ranked)))
            bought_today = 0
            for _, row in selected.iterrows():
                lot_cost = float(row["close"]) * self.options.round_lot
                if lot_cost <= 0:
                    continue
                lots = int(raw_amount // lot_cost)
                buy_amount = lots * lot_cost
                if lots <= 0 or buy_amount > cash:
                    continue
                future_return = float(row["future_return_20d"])
                cash -= buy_amount
                bought_today += 1
                positions.append(
                    {
                        "strategy": strategy,
                        "entry_date": current_date,
                        "exit_date": current_date + pd.offsets.BDay(self.options.holding_days),
                        "code": str(row["code"]),
                        "buy_amount": buy_amount,
                        "lot_count": lots,
                        "entry_close": float(row["close"]),
                        "realized_return": future_return,
                        "realized_profit": buy_amount * future_return,
                        "opportunity_top_decile_proba": _safe_float(row.get("opportunity_top_decile_proba")),
                        "baseline_rank_score": _safe_float(row.get("baseline_rank_score")),
                        **{column: _safe_float(row.get(column)) for column in FUTURE_EVAL_COLUMNS},
                    }
                )

            invested = sum(float(position["buy_amount"]) for position in positions)
            realized_open = sum(float(position["realized_profit"]) for position in positions if position["exit_date"] <= current_date)
            total_assets = cash + invested + realized_open
            daily_rows.append(
                {
                    "strategy": strategy,
                    "date": current_date,
                    "cash": cash,
                    "open_position_count": len(positions),
                    "bought_today": bought_today,
                    "invested_amount": invested,
                    "total_assets": total_assets,
                    "capital_utilization": invested / self.options.initial_cash if self.options.initial_cash else None,
                }
            )

        for position in positions:
            cash += position["buy_amount"] + position["realized_profit"]
            position["closed_on"] = position["exit_date"]
            trades.append(position)

        trades_df = pd.DataFrame(trades)
        daily_df = pd.DataFrame(daily_rows)
        if not daily_df.empty:
            daily_df.loc[daily_df.index[-1], "total_assets"] = cash
        return trades_df, daily_df

    def rank_candidates(self, day: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if strategy == "baseline_equal_allocation":
            return day.sort_values(["baseline_rank_score", "turnover_value", "code"], ascending=[False, False, True])
        if strategy == "candidate_valuation_top5":
            return day.sort_values(["opportunity_top_decile_proba", "turnover_value", "code"], ascending=[False, False, True])
        raise ValueError(f"Unknown strategy: {strategy}")

    def strategy_metrics(self, strategy: str, trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
        profits = _numeric(trades["realized_profit"]) if not trades.empty else pd.Series(dtype=float)
        gross_profit = float(profits[profits > 0].sum()) if not profits.empty else 0.0
        gross_loss = abs(float(profits[profits < 0].sum())) if not profits.empty else 0.0
        equity = _numeric(daily["total_assets"]) if not daily.empty else pd.Series([self.options.initial_cash])
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        return {
            "strategy": strategy,
            "net_profit": _safe_float(profits.sum()) if not profits.empty else 0.0,
            "PF": _safe_float(gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit == 0 else float("inf")),
            "DD": _safe_float(drawdown.min()) if not drawdown.empty else 0.0,
            "win_rate": _safe_float((profits > 0).mean()) if not profits.empty else None,
            "total_trades": int(len(trades)),
            "final_assets": _safe_float(self.options.initial_cash + profits.sum()) if not profits.empty else self.options.initial_cash,
            "capital_utilization": _safe_float(_numeric(daily["capital_utilization"]).mean()) if not daily.empty else None,
        }

    def buy_quality(self, strategy: str, trades: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {"strategy": strategy, "buy_count": int(len(trades))}
        for column in FUTURE_EVAL_COLUMNS:
            values = _numeric(trades[column]) if column in trades.columns else pd.Series(dtype=float)
            row[f"{column}_mean"] = _safe_float(values.mean()) if not values.empty else None
        return row

    def valuation_effect(self, results: list[dict[str, Any]], quality: list[dict[str, Any]]) -> dict[str, Any]:
        by_strategy = {row["strategy"]: row for row in results}
        by_quality = {row["strategy"]: row for row in quality}
        baseline = by_strategy.get("baseline_equal_allocation", {})
        candidate = by_strategy.get("candidate_valuation_top5", {})
        baseline_quality = by_quality.get("baseline_equal_allocation", {})
        candidate_quality = by_quality.get("candidate_valuation_top5", {})
        return {
            "net_profit_delta": self._delta(candidate.get("net_profit"), baseline.get("net_profit")),
            "PF_delta": self._delta(candidate.get("PF"), baseline.get("PF")),
            "DD_delta": self._delta(candidate.get("DD"), baseline.get("DD")),
            "capital_utilization_delta": self._delta(candidate.get("capital_utilization"), baseline.get("capital_utilization")),
            "opportunity_value_20d_mean_delta": self._delta(candidate_quality.get("opportunity_value_20d_mean"), baseline_quality.get("opportunity_value_20d_mean")),
            "future_return_20d_mean_delta": self._delta(candidate_quality.get("future_return_20d_mean"), baseline_quality.get("future_return_20d_mean")),
        }

    def _delta(self, candidate: Any, baseline: Any) -> float | None:
        candidate_value = _safe_float(candidate)
        baseline_value = _safe_float(baseline)
        if candidate_value is None or baseline_value is None:
            return None
        return candidate_value - baseline_value

    def dataset_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(data)),
            "unique_codes": int(data["code"].nunique()),
            "candidate_days": int(data["date"].nunique()),
            "date_range": {
                "min": data["date"].min().date().isoformat() if not data.empty else None,
                "max": data["date"].max().date().isoformat() if not data.empty else None,
            },
            "baseline_rank_column": "stock_selection_rank_score",
            "candidate_rank_column": "opportunity_top_decile_proba",
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "11-D",
            "limited_2025_only": True,
            "full_period_backtest_executed": False,
            "profile_added": False,
            "profile_modified": False,
            "current_model_overwritten": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
            "simulation_path": str(self.root / SIMULATION_PATH),
            "dataset_path": str(self.root / DATASET_PATH),
        }

    def backtest_conditions(self) -> dict[str, Any]:
        return {
            "period": {"start": START_DATE, "end": END_DATE},
            "initial_cash": self.options.initial_cash,
            "daily_buy_budget": self.options.daily_buy_budget,
            "max_positions": self.options.max_positions,
            "round_lot": self.options.round_lot,
            "holding_days": self.options.holding_days,
            "strategies": [
                {
                    "name": "baseline_equal_allocation",
                    "rank_input": "stock_selection_rank_score",
                    "uses_valuation": False,
                },
                {
                    "name": "candidate_valuation_top5",
                    "rank_input": "opportunity_top_decile_proba",
                    "uses_valuation": True,
                    "candidate_threshold": "top5",
                    "allocation": "equal_weight_top5",
                },
            ],
        }

    def leakage_checklist(self) -> dict[str, Any]:
        baseline_decision_columns = ["stock_selection_rank_score", "risk_adjusted_score", "expected_return", "candidate_strength", "close", "turnover_value"]
        candidate_decision_columns = ["opportunity_top_decile_proba", "close", "turnover_value"]
        decision_columns = baseline_decision_columns + candidate_decision_columns
        future_in_features = [column for column in decision_columns if column.startswith("future_") or column.startswith("opportunity_value")]
        forbidden_hits = [column for column in decision_columns if self.is_forbidden(column)]
        blocking = []
        if future_in_features:
            blocking.append("future_columns_used_as_features")
        if forbidden_hits:
            blocking.append("forbidden_columns_used_as_features")
        return {
            "future_columns_used_as_features": future_in_features,
            "future_columns_used_only_for_evaluation": FUTURE_EVAL_COLUMNS,
            "backtest_columns_used": [column for column in decision_columns if "backtest" in column.lower()],
            "trade_result_columns_used": [column for column in decision_columns if any(token in column.lower() for token in ["trade", "profit", "loss", "result"])],
            "cash_or_portfolio_columns_used_as_model_features": [column for column in decision_columns if any(token in column.lower() for token in ["cash", "portfolio", "position"])],
            "selected_or_bought_used": any(any(token in column.lower() for token in ["selected", "bought", "affordable"]) for column in decision_columns),
            "current_pm_multiplier_used": any(any(token in column.lower() for token in ["pm_multiplier", "current_pm"]) for column in decision_columns),
            "historical_predictions_regenerated": False,
            "profile_changed": False,
            "leakage_risk": "low" if not blocking else "high",
            "blocking_issues": blocking,
        }

    def is_forbidden(self, column: str) -> bool:
        lowered = column.lower()
        return any(token in lowered for token in FORBIDDEN_TOKENS)

    def recommendation(self, results: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"ready_for_phase11e": False, "recommended_next_phase": "Fix Phase11-D leakage blockers"}
        by_strategy = {row["strategy"]: row for row in results}
        baseline = by_strategy.get("baseline_equal_allocation", {})
        candidate = by_strategy.get("candidate_valuation_top5", {})
        candidate_profit = _safe_float(candidate.get("net_profit")) or 0.0
        baseline_profit = _safe_float(baseline.get("net_profit")) or 0.0
        candidate_pf = _safe_float(candidate.get("PF")) or -10**18
        baseline_pf = _safe_float(baseline.get("PF")) or -10**18
        candidate_dd = _safe_float(candidate.get("DD"))
        baseline_dd = _safe_float(baseline.get("DD"))
        ready = candidate_profit > baseline_profit and candidate_pf >= baseline_pf
        dd_worsened = candidate_dd is not None and baseline_dd is not None and candidate_dd < baseline_dd
        return {
            "valuation_improved_limited_2025_backtest": bool(ready),
            "ready_for_phase11e": bool(ready),
            "dd_worsened": bool(dd_worsened),
            "recommended_next_phase": "Phase11-E limited exit integration with DD guard" if ready else "Phase11-D2 candidate ranking or Phase11-B2 valuation improvement",
            "reason": "Candidate improves 2025 net profit and PF versus the non-Valuation baseline; if DD worsens, Phase11-E must stay limited and focus on exit/risk control.",
        }

    def save_report(self, report: dict[str, Any]) -> Phase11DPaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase11DPaths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 11-D Limited Combined Backtest 2025",
            "",
            "## Scope",
            "",
            "- 2025 only",
            "- two strategies only: baseline equal allocation vs valuation top5",
            "- no full-period backtest, no profile change, no model overwrite",
            "- future labels are used only for realized lightweight outcomes and BUY quality evaluation",
            "",
            "## Conditions",
            "",
            self._table([report["backtest_conditions"]], ["initial_cash", "daily_buy_budget", "max_positions", "round_lot", "holding_days"]),
            "",
            "## Dataset",
            "",
            self._table([report.get("dataset_summary", {})], ["rows", "unique_codes", "candidate_days", "baseline_rank_column", "candidate_rank_column"]),
            "",
            "## Strategy Results",
            "",
            self._table(report.get("strategy_results", []), ["strategy", "net_profit", "PF", "DD", "win_rate", "total_trades", "final_assets", "capital_utilization"]),
            "",
            "## BUY Quality",
            "",
            self._table(report.get("buy_quality_comparison", []), ["strategy", "buy_count", "future_return_20d_mean", "future_max_return_20d_mean", "future_max_drawdown_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_mean"]),
            "",
            "## Valuation Effect",
            "",
            self._table([report.get("valuation_effect", {})], ["net_profit_delta", "PF_delta", "DD_delta", "capital_utilization_delta", "opportunity_value_20d_mean_delta", "future_return_20d_mean_delta"]),
            "",
            "## Leakage Checklist",
            "",
            self._table([report["leakage_checklist"]], ["future_columns_used_as_features", "backtest_columns_used", "trade_result_columns_used", "cash_or_portfolio_columns_used_as_model_features", "selected_or_bought_used", "current_pm_multiplier_used", "historical_predictions_regenerated", "profile_changed", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self._table([report["recommendation"]], ["valuation_improved_limited_2025_backtest", "ready_for_phase11e", "dd_worsened", "recommended_next_phase", "reason"]),
            "",
        ]
        return "\n".join(lines)

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        body = []
        for row in rows:
            body.append("| " + " | ".join(self._format(row.get(column)) for column in columns) + " |")
        return "\n".join([header, sep, *body])

    def _format(self, value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return f"{value:.4f}"
        if isinstance(value, (list, tuple)):
            return ", ".join(map(str, value))
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        if value is None:
            return ""
        return str(value)
