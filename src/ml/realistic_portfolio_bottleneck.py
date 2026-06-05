from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import ML_FEATURES_ROOT, ML_LABELS_ROOT, ML_REPORTS_ROOT
from ml.realistic_portfolio import (
    MLRealisticPortfolioSimulator,
    RANKING_COLUMNS,
    RealisticPortfolioConfig,
)


class MLRealisticPortfolioBottleneckAnalyzer:
    """Analyze which constraints block ML-ranked candidates in report-only portfolios."""

    def __init__(
        self,
        simulator: MLRealisticPortfolioSimulator | None = None,
        report_root: str | Path = ML_REPORTS_ROOT,
    ) -> None:
        self.simulator = simulator or MLRealisticPortfolioSimulator()
        self.report_root = Path(report_root)

    def analyze(
        self,
        start_date: str,
        end_date: str,
        baseline_config: RealisticPortfolioConfig,
    ) -> dict[str, Any]:
        configs = self._grid_configs(baseline_config)
        candidates = self.simulator._load_candidates(start_date, end_date, top_n=baseline_config.top_n)
        prices = self.simulator._load_prices(start_date, end_date)
        price_by_code = {
            str(code): group.sort_values("date").reset_index(drop=True)
            for code, group in prices.groupby("code", dropna=False)
        } if not prices.empty else {}

        grid_result = self.simulator.simulate_configs(start_date, end_date, configs)
        baseline_decisions = self._simulate_decisions(candidates, price_by_code, baseline_config)
        decision_df = pd.DataFrame(baseline_decisions)
        return {
            "period": {"start_date": start_date, "end_date": end_date},
            "baseline_config": asdict(baseline_config),
            "baseline_summary": self._baseline_summary(grid_result, baseline_config),
            "monthly_rejections": self._monthly_rejections(decision_df),
            "bought_vs_rejected": self._bought_vs_rejected(decision_df),
            "grid_summary": self._sorted_grid_summary(grid_result),
            "best_grid": self._best_grid(grid_result),
            "constraint_grid_slices": self._constraint_slices(grid_result),
            "candidate_rows": baseline_decisions,
        }

    def save_report(self, result: dict[str, Any]) -> Path:
        path = self.report_root / "realistic_portfolio_bottleneck_5y_enriched.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_markdown(result), encoding="utf-8")
        return path

    def save_json(self, result: dict[str, Any]) -> Path:
        path = self.report_root / "realistic_portfolio_bottleneck_5y_enriched.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in result.items() if key != "candidate_rows"}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def save_candidates_csv(self, result: dict[str, Any]) -> Path:
        path = self.report_root / "realistic_portfolio_bought_vs_rejected_5y_enriched.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.get("candidate_rows", [])).to_csv(path, index=False)
        return path

    def format_markdown(self, result: dict[str, Any]) -> str:
        period = result["period"]
        return "\n".join(
            [
                "# Realistic Portfolio Bottleneck Analysis",
                "",
                f"- period: {period['start_date']} to {period['end_date']}",
                "- note: report-only; existing trading logic is unchanged",
                "",
                "## Baseline",
                "",
                self._table([result["baseline_summary"]], [
                    "config_id",
                    "final_assets",
                    "total_return",
                    "win_rate",
                    "profit_factor",
                    "max_drawdown",
                    "total_trades",
                    "rejected_by_cash",
                    "rejected_by_max_positions",
                    "rejected_by_duplicate",
                    "rejected_by_liquidity",
                ]),
                "",
                "## Monthly Rejections",
                "",
                self._table(result["monthly_rejections"], [
                    "month",
                    "bought",
                    "rejected_by_cash",
                    "rejected_by_max_positions",
                    "rejected_by_duplicate",
                    "rejected_by_liquidity",
                    "rejected_by_no_exit",
                ]),
                "",
                "## Bought vs Rejected",
                "",
                self._table(result["bought_vs_rejected"], [
                    "status",
                    "reason",
                    "count",
                    "avg_future_10d_return",
                    "median_future_10d_return",
                    "bad_entry_rate",
                    "avg_realistic_return",
                    "total_realistic_profit",
                ]),
                "",
                "## Grid Summary Top 20",
                "",
                self._table(result["grid_summary"][:20], [
                    "config_id",
                    "ranking",
                    "position_size",
                    "max_positions",
                    "exit_rule",
                    "min_turnover_value",
                    "final_assets",
                    "total_return",
                    "win_rate",
                    "profit_factor",
                    "max_drawdown",
                    "total_trades",
                    "rejected_by_cash",
                    "rejected_by_max_positions",
                    "rejected_by_duplicate",
                    "rejected_by_liquidity",
                ]),
                "",
                "## Constraint Slices",
                "",
                self._table(result["constraint_grid_slices"], [
                    "dimension",
                    "value",
                    "configs",
                    "best_total_return",
                    "mean_total_return",
                    "best_profit_factor",
                    "mean_profit_factor",
                    "mean_total_trades",
                    "mean_rejected_by_liquidity",
                ]),
                "",
            ]
        )

    def _grid_configs(self, baseline: RealisticPortfolioConfig) -> list[RealisticPortfolioConfig]:
        configs = []
        for ranking in ["risk_adjusted_return", "expected_return_10d"]:
            for max_positions in [5, 10, 20]:
                for position_size in [50_000, 100_000, 200_000]:
                    for min_turnover in [0, 50_000_000, 100_000_000]:
                        for exit_rule in ["close_5d", "close_10d", "close_20d"]:
                            configs.append(
                                RealisticPortfolioConfig(
                                    ranking=ranking,
                                    top_n=baseline.top_n,
                                    initial_cash=baseline.initial_cash,
                                    position_size=position_size,
                                    max_positions=max_positions,
                                    exit_rule=exit_rule,
                                    fee_rate=baseline.fee_rate,
                                    slippage_rate=baseline.slippage_rate,
                                    min_turnover_value=min_turnover,
                                )
                            )
        return configs

    def _simulate_decisions(
        self,
        candidates: pd.DataFrame,
        price_by_code: dict[str, pd.DataFrame],
        config: RealisticPortfolioConfig,
    ) -> list[dict[str, Any]]:
        cash = float(config.initial_cash)
        positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        data = candidates[candidates["ranking"].eq(config.ranking)].sort_values(["date", "rank"])
        for date, daily in data.groupby("date", sort=True):
            current_date = pd.Timestamp(date)
            cash = self.simulator._close_due_positions(current_date, positions, cash, trades)
            for row in daily.head(config.top_n).to_dict("records"):
                decision = self._candidate_record(row, config)
                turnover = row.get("turnover_value")
                code = str(row["code"])
                if pd.isna(turnover) or float(turnover) < config.min_turnover_value:
                    decision.update({"status": "rejected", "reason": "liquidity"})
                    rows.append(decision)
                    continue
                if any(position["code"] == code for position in positions):
                    decision.update({"status": "rejected", "reason": "duplicate"})
                    rows.append(decision)
                    continue
                if len(positions) >= config.max_positions:
                    decision.update({"status": "rejected", "reason": "max_positions"})
                    rows.append(decision)
                    continue
                if cash < config.position_size * (1 + config.fee_rate):
                    decision.update({"status": "rejected", "reason": "cash"})
                    rows.append(decision)
                    continue
                setup = self.simulator._build_trade_setup(row, price_by_code.get(code), config)
                if setup is None:
                    decision.update({"status": "rejected", "reason": "no_exit"})
                    rows.append(decision)
                    continue
                cash -= setup["cost"]
                positions.append(setup)
                decision.update(
                    {
                        "status": "bought",
                        "reason": "bought",
                        "entry_date": self.simulator._json_value(setup["entry_date"]),
                        "exit_date": self.simulator._json_value(setup["exit_date"]),
                        "realistic_return": setup["return"],
                        "realistic_profit": setup["profit"],
                    }
                )
                rows.append(decision)
        return rows

    def _candidate_record(self, row: dict[str, Any], config: RealisticPortfolioConfig) -> dict[str, Any]:
        score_column = RANKING_COLUMNS.get(config.ranking)
        return {
            "config_id": config.config_id,
            "ranking": config.ranking,
            "signal_date": self.simulator._json_value(row.get("date")),
            "month": pd.Timestamp(row.get("date")).strftime("%Y-%m") if not pd.isna(row.get("date")) else None,
            "code": str(row.get("code")),
            "rank": int(row.get("rank")) if not pd.isna(row.get("rank")) else None,
            "ranking_score": self._number(row.get(score_column)) if score_column else None,
            "expected_return_10d": self._number(row.get("expected_return_10d")),
            "bad_entry_probability_10d": self._number(row.get("bad_entry_probability_10d")),
            "future_10d_return": self._number(row.get("future_10d_return")),
            "bad_entry_10d": self._bool_value(row.get("bad_entry_10d")),
            "turnover_value": self._number(row.get("turnover_value")),
            "position_size": config.position_size,
            "max_positions": config.max_positions,
            "exit_rule": config.exit_rule,
            "min_turnover_value": config.min_turnover_value,
        }

    def _baseline_summary(self, grid_result: dict[str, Any], config: RealisticPortfolioConfig) -> dict[str, Any]:
        for row in grid_result["summary"]:
            if row["config_id"] == config.config_id:
                return row
        return {}

    def _monthly_rejections(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        rows = []
        for month, group in df.groupby("month", dropna=False):
            rows.append(
                {
                    "month": str(month),
                    "bought": int(group["status"].eq("bought").sum()),
                    "rejected_by_cash": int(group["reason"].eq("cash").sum()),
                    "rejected_by_max_positions": int(group["reason"].eq("max_positions").sum()),
                    "rejected_by_duplicate": int(group["reason"].eq("duplicate").sum()),
                    "rejected_by_liquidity": int(group["reason"].eq("liquidity").sum()),
                    "rejected_by_no_exit": int(group["reason"].eq("no_exit").sum()),
                }
            )
        return rows

    def _bought_vs_rejected(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        rows = []
        for keys, group in df.groupby(["status", "reason"], dropna=False):
            status, reason = keys
            future = pd.to_numeric(group.get("future_10d_return"), errors="coerce")
            realistic_return = pd.to_numeric(group.get("realistic_return"), errors="coerce")
            realistic_profit = pd.to_numeric(group.get("realistic_profit"), errors="coerce")
            bad = group.get("bad_entry_10d")
            rows.append(
                {
                    "status": str(status),
                    "reason": str(reason),
                    "count": int(len(group)),
                    "avg_future_10d_return": self._mean(future),
                    "median_future_10d_return": self._median(future),
                    "bad_entry_rate": self._mean(pd.Series(bad).astype("boolean")) if bad is not None else None,
                    "avg_realistic_return": self._mean(realistic_return),
                    "total_realistic_profit": self._sum(realistic_profit),
                }
            )
        return sorted(rows, key=lambda row: (row["status"] != "bought", row["reason"]))

    def _sorted_grid_summary(self, grid_result: dict[str, Any]) -> list[dict[str, Any]]:
        rows = [self._with_grid_params(row) for row in grid_result["summary"]]
        return sorted(rows, key=lambda row: (row.get("total_return") is None, -(row.get("total_return") or -999)))

    def _best_grid(self, grid_result: dict[str, Any]) -> dict[str, Any] | None:
        rows = self._sorted_grid_summary(grid_result)
        return rows[0] if rows else None

    def _constraint_slices(self, grid_result: dict[str, Any]) -> list[dict[str, Any]]:
        df = pd.DataFrame([self._with_grid_params(row) for row in grid_result["summary"]])
        if df.empty:
            return []
        rows = []
        for dimension in ["ranking", "position_size", "max_positions", "exit_rule", "min_turnover_value"]:
            for value, group in df.groupby(dimension, dropna=False):
                rows.append(
                    {
                        "dimension": dimension,
                        "value": str(value),
                        "configs": int(len(group)),
                        "best_total_return": self._max(group["total_return"]),
                        "mean_total_return": self._mean(group["total_return"]),
                        "best_profit_factor": self._max(group["profit_factor"]),
                        "mean_profit_factor": self._mean(group["profit_factor"]),
                        "mean_total_trades": self._mean(group["total_trades"]),
                        "mean_rejected_by_liquidity": self._mean(group["rejected_by_liquidity"]),
                    }
                )
        return rows

    def _with_grid_params(self, row: dict[str, Any]) -> dict[str, Any]:
        output = dict(row)
        parts = str(row.get("config_id", "")).split("_")
        # config_id keeps all parameters except position_size, so retain it from simulation rows when absent.
        output.setdefault("position_size", None)
        return output

    def _number(self, value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _bool_value(self, value: Any) -> bool | None:
        if value is None or pd.isna(value):
            return None
        return bool(value)

    def _mean(self, values: Any) -> float | None:
        value = pd.Series(values).mean()
        return None if pd.isna(value) else float(value)

    def _median(self, values: Any) -> float | None:
        value = pd.Series(values).median()
        return None if pd.isna(value) else float(value)

    def _sum(self, values: Any) -> float | None:
        series = pd.Series(values).dropna()
        return None if series.empty else float(series.sum())

    def _max(self, values: Any) -> float | None:
        value = pd.Series(values).max()
        return None if pd.isna(value) else float(value)

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(self._fmt(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, separator, *body])

    def _fmt(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)
