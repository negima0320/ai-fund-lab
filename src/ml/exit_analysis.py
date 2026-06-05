from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ml.config import JQUANTS_CACHE_ROOT, ML_DATA_ROOT, ML_REPORTS_ROOT
from ml.data_loader import JQuantsDataLoader


REEVALUATION_HORIZONS = [5, 10, 15]
PREDICTION_COLUMNS = [
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "risk_adjusted_score",
]
TAX_RATE = 0.20315


@dataclass(frozen=True)
class ExitAnalysisPaths:
    markdown: Path
    json: Path
    trades_csv: Path


class MLExitAnalyzer:
    """Post-trade exit diagnostics using walk-forward predictions only."""

    def __init__(
        self,
        root: str | Path = ".",
        profile: str = "rookie_dealer_02_v2_66_ml_ranked",
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        predictions_root: str | Path | None = None,
        cache_root: str | Path | None = None,
        report_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.profile = profile
        self.start_date = start_date
        self.end_date = end_date
        self.period_key = f"{start_date}_to_{end_date}"
        self.predictions_root = Path(predictions_root) if predictions_root else self._rooted(ML_DATA_ROOT) / "walk_forward_predictions"
        self.cache_root = Path(cache_root) if cache_root else self._rooted(JQUANTS_CACHE_ROOT)
        self.report_root = Path(report_root) if report_root else self._rooted(ML_REPORTS_ROOT)

    def _rooted(self, path: Path) -> Path:
        root = self.root.resolve()
        try:
            return root / path.resolve().relative_to(root)
        except ValueError:
            return path

    def build(self) -> dict[str, Any]:
        trades = self._load_trades()
        prices = self._load_prices(trades)
        enriched = self._enrich_trades(trades, prices)
        rule_results = self._simulate_rules(enriched)
        loss_reduction = self._loss_reduction(enriched, rule_results)
        return {
            "profile": self.profile,
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "trade_count": int(len(enriched)),
            "reevaluation_horizons": REEVALUATION_HORIZONS,
            "note": "Reevaluation horizons use the backtest holding-days convention: entry date is day 1. Rules only trigger when the position is still open on the reevaluation date; v2_66 max holding is 5 business days.",
            "baseline": self._baseline_metrics(enriched),
            "reevaluation_coverage": self._reevaluation_coverage(enriched),
            "rules": rule_results,
            "loss_reduction": loss_reduction,
            "trades": enriched.to_dict("records"),
        }

    def save(self, result: dict[str, Any]) -> ExitAnalysisPaths:
        self.report_root.mkdir(parents=True, exist_ok=True)
        stem = "ml_exit_analysis_v2_66_2023-01_to_2026-05"
        markdown = self.report_root / f"{stem}.md"
        json_path = self.report_root / f"{stem}.json"
        trades_csv = self.report_root / f"{stem.replace('ml_exit_analysis', 'ml_exit_analysis_trades')}.csv"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        pd.DataFrame(result["trades"]).to_csv(trades_csv, index=False)
        return ExitAnalysisPaths(markdown, json_path, trades_csv)

    def format_markdown(self, result: dict[str, Any]) -> str:
        best = max(result["rules"], key=lambda row: row.get("profit_delta", -10**18)) if result["rules"] else {}
        worst = min(result["rules"], key=lambda row: row.get("profit_delta", 10**18)) if result["rules"] else {}
        lines = [
            "# ML Exit Analysis v2_66",
            "",
            f"- profile: `{self.profile}`",
            f"- period: `{self.start_date}` to `{self.end_date}`",
            "- source: existing backtest trades, walk-forward predictions, and local J-Quants price cache",
            "- note: no trading logic change, no backtest rerun, no API refetch",
            "- reevaluation dates follow the backtest holding-days convention: entry date is day 1",
            "- reevaluation rule trigger: only when the position is still open on the reevaluation date",
            "",
            "## Baseline",
            "",
            self._table([result["baseline"]], ["rule", "total_profit", "win_rate", "profit_factor", "max_drawdown", "average_holding_days", "trade_count"]),
            "",
            "## Reevaluation Coverage",
            "",
            self._table(
                result["reevaluation_coverage"],
                ["horizon", "position_open_count", "price_available_count", "prediction_available_count", "risk_adjusted_below_zero_open_count", "bad_entry_high_open_count"],
            ),
            "",
            "## Exit Rule Simulation",
            "",
            self._table(
                result["rules"],
                [
                    "rule",
                    "total_profit",
                    "profit_delta",
                    "win_rate",
                    "profit_factor",
                    "max_drawdown",
                    "average_holding_days",
                    "exit_changed_count",
                    "improved_trade_count",
                    "worsened_trade_count",
                ],
            ),
            "",
            "## Loss Reduction",
            "",
            self._table(
                result["loss_reduction"],
                ["rule", "losing_trades_improved_count", "profit_giveup_count", "worst20_profit_delta"],
            ),
            "",
            "## Summary",
            "",
            f"- best_rule_by_profit_delta: `{best.get('rule')}` ({best.get('profit_delta')})",
            f"- worst_rule_by_profit_delta: `{worst.get('rule')}` ({worst.get('profit_delta')})",
            "- 10d and 15d rules have limited opportunity under the current v2_66 max holding window.",
            "- This is an exit-research report only; do not wire these rules into trading without a separate forward-safe design.",
            "",
        ]
        return "\n".join(lines)

    def _load_trades(self) -> pd.DataFrame:
        path = self.root / "logs" / "backtests" / self.profile / self.period_key / "trades.csv"
        df = pd.read_csv(path)
        if "action" in df.columns:
            df = df[df["action"].astype(str).eq("SELL")].copy()
        for column in ["signal_date", "entry_date", "exit_date"]:
            df[column] = pd.to_datetime(df[column], errors="coerce")
        for column in ["entry_price", "exit_price", "shares", "net_profit", "net_profit_rate", "holding_days"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["code"] = df["code"].astype(str)
        df = df.dropna(subset=["entry_date", "exit_date", "entry_price", "exit_price", "shares"]).reset_index(drop=True)
        df["actual_exit_return"] = df["net_profit"] / (df["entry_price"] * df["shares"])
        return df

    def _load_prices(self, trades: pd.DataFrame) -> pd.DataFrame:
        start = trades["entry_date"].min().strftime("%Y-%m-%d")
        end = (trades["exit_date"].max() + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        loader = JQuantsDataLoader(self.cache_root)
        prices = loader.load_prices(start, end)
        if prices.empty:
            return prices
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices["code"] = prices["code"].astype(str)
        return prices.sort_values(["date", "code"]).reset_index(drop=True)

    def _enrich_trades(self, trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        trading_dates = sorted(prices["date"].dropna().unique()) if not prices.empty else []
        date_index = {pd.Timestamp(value): index for index, value in enumerate(trading_dates)}
        price_by_key = {
            (str(row.code), pd.Timestamp(row.date)): row
            for row in prices.itertuples(index=False)
        } if not prices.empty else {}
        rows: list[dict[str, Any]] = []
        for trade in trades.to_dict("records"):
            row = dict(trade)
            entry_date = pd.Timestamp(row["entry_date"])
            entry_index = date_index.get(entry_date)
            for horizon in REEVALUATION_HORIZONS:
                prefix = f"reval_{horizon}d"
                reval_date = trading_dates[entry_index + horizon - 1] if entry_index is not None and entry_index + horizon - 1 < len(trading_dates) else None
                row[f"{prefix}_date"] = pd.Timestamp(reval_date).strftime("%Y-%m-%d") if reval_date is not None else None
                market = price_by_key.get((str(row["code"]), pd.Timestamp(reval_date))) if reval_date is not None else None
                reval_close = float(getattr(market, "close")) if market is not None and pd.notna(getattr(market, "close")) else None
                row[f"exit_{horizon}d_price"] = reval_close
                row[f"exit_{horizon}d_return"] = self._net_return(row["entry_price"], reval_close, row["shares"]) if reval_close is not None else None
                is_open = reval_date is not None and pd.Timestamp(reval_date) <= pd.Timestamp(row["exit_date"])
                row[f"{prefix}_position_open"] = bool(is_open)
                prediction = self._prediction_for(str(row["code"]), row[f"{prefix}_date"])
                for column in PREDICTION_COLUMNS:
                    row[f"{prefix}_{column}"] = prediction.get(column) if prediction else None
            rows.append(row)
        return pd.DataFrame(rows)

    def _prediction_for(self, code: str, date_text: str | None) -> dict[str, Any] | None:
        if not date_text:
            return None
        path = self.predictions_root / f"predictions_{date_text}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if df.empty or "code" not in df.columns:
            return None
        df["code"] = df["code"].astype(str)
        match = df[df["code"].eq(code)]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        if {"expected_return_10d", "bad_entry_probability_10d"}.issubset(row):
            row["risk_adjusted_score"] = self._to_float(row.get("expected_return_10d")) - 0.5 * self._to_float(row.get("bad_entry_probability_10d"))
        return {column: self._to_float(row.get(column)) for column in PREDICTION_COLUMNS}

    def _simulate_rules(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        rules: list[tuple[str, int, Callable[[pd.Series], bool]]] = [
            ("Exit Rule A: 5d risk_adjusted_score < 0", 5, lambda row: self._lt(row, "reval_5d_risk_adjusted_score", 0)),
            ("Exit Rule B: 10d risk_adjusted_score < 0", 10, lambda row: self._lt(row, "reval_10d_risk_adjusted_score", 0)),
            ("Exit Rule C: 5d bad_entry_probability_10d >= 0.70", 5, lambda row: self._gte(row, "reval_5d_bad_entry_probability_10d", 0.70)),
            ("Exit Rule D: 10d bad_entry_probability_10d >= 0.70", 10, lambda row: self._gte(row, "reval_10d_bad_entry_probability_10d", 0.70)),
            ("Exit Rule E: 10d expected_return_10d < 0", 10, lambda row: self._lt(row, "reval_10d_expected_return_10d", 0)),
            (
                "Exit Rule F: 10d risk_adjusted_score < 0 or bad_entry_probability_10d >= 0.70",
                10,
                lambda row: self._lt(row, "reval_10d_risk_adjusted_score", 0) or self._gte(row, "reval_10d_bad_entry_probability_10d", 0.70),
            ),
        ]
        results = []
        for name, horizon, predicate in rules:
            simulated = self._apply_rule(trades, name, horizon, predicate)
            metrics = self._metrics(simulated, profit_col="simulated_profit", holding_col="simulated_holding_days")
            baseline_profit = float(trades["net_profit"].sum())
            metrics.update(
                {
                    "rule": name,
                    "profit_delta": float(metrics["total_profit"] - baseline_profit),
                    "exit_changed_count": int(simulated["exit_changed"].sum()),
                    "improved_trade_count": int((simulated["simulated_profit"] > simulated["net_profit"]).sum()),
                    "worsened_trade_count": int((simulated["simulated_profit"] < simulated["net_profit"]).sum()),
                }
            )
            results.append(metrics)
        return results

    def _reevaluation_coverage(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for horizon in REEVALUATION_HORIZONS:
            open_mask = trades[f"reval_{horizon}d_position_open"].astype(bool)
            rows.append(
                {
                    "horizon": f"{horizon}d",
                    "position_open_count": int(open_mask.sum()),
                    "price_available_count": int(trades[f"exit_{horizon}d_price"].notna().sum()),
                    "prediction_available_count": int(trades[f"reval_{horizon}d_risk_adjusted_score"].notna().sum()),
                    "risk_adjusted_below_zero_open_count": int((open_mask & (trades[f"reval_{horizon}d_risk_adjusted_score"] < 0)).sum()),
                    "bad_entry_high_open_count": int((open_mask & (trades[f"reval_{horizon}d_bad_entry_probability_10d"] >= 0.70)).sum()),
                }
            )
        return rows

    def _apply_rule(self, trades: pd.DataFrame, rule_name: str, horizon: int, predicate: Callable[[pd.Series], bool]) -> pd.DataFrame:
        rows = []
        for _, row in trades.iterrows():
            output = row.copy()
            can_trigger = bool(row.get(f"reval_{horizon}d_position_open")) and pd.notna(row.get(f"exit_{horizon}d_price"))
            triggered = bool(can_trigger and predicate(row))
            if triggered:
                output["simulated_exit_date"] = row.get(f"reval_{horizon}d_date")
                output["simulated_exit_price"] = row.get(f"exit_{horizon}d_price")
                output["simulated_profit"] = self._net_profit(row["entry_price"], row.get(f"exit_{horizon}d_price"), row["shares"])
                output["simulated_return"] = output["simulated_profit"] / (row["entry_price"] * row["shares"])
                output["simulated_holding_days"] = horizon
                output["exit_changed"] = True
                output["triggered_rule"] = rule_name
            else:
                output["simulated_exit_date"] = row["exit_date"].strftime("%Y-%m-%d")
                output["simulated_exit_price"] = row["exit_price"]
                output["simulated_profit"] = row["net_profit"]
                output["simulated_return"] = row["actual_exit_return"]
                output["simulated_holding_days"] = row["holding_days"]
                output["exit_changed"] = False
                output["triggered_rule"] = ""
            rows.append(output)
        return pd.DataFrame(rows)

    def _loss_reduction(self, trades: pd.DataFrame, rule_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        actual = trades.copy()
        worst20_indices = set(actual.sort_values("net_profit").head(20).index)
        for rule in rule_results:
            name = rule["rule"]
            horizon = 5 if "5d" in name else 10
            predicate = self._predicate_for_rule_name(name)
            simulated = self._apply_rule(trades, name, horizon, predicate)
            losing = simulated[simulated["net_profit"] < 0]
            rows.append(
                {
                    "rule": name,
                    "losing_trades_improved_count": int((losing["simulated_profit"] > losing["net_profit"]).sum()),
                    "profit_giveup_count": int(((simulated["net_profit"] > 0) & (simulated["simulated_profit"] < simulated["net_profit"])).sum()),
                    "worst20_profit_delta": float((simulated.loc[list(worst20_indices), "simulated_profit"] - simulated.loc[list(worst20_indices), "net_profit"]).sum()) if worst20_indices else 0.0,
                }
            )
        return rows

    def _predicate_for_rule_name(self, name: str) -> Callable[[pd.Series], bool]:
        if name.startswith("Exit Rule A"):
            return lambda row: self._lt(row, "reval_5d_risk_adjusted_score", 0)
        if name.startswith("Exit Rule B"):
            return lambda row: self._lt(row, "reval_10d_risk_adjusted_score", 0)
        if name.startswith("Exit Rule C"):
            return lambda row: self._gte(row, "reval_5d_bad_entry_probability_10d", 0.70)
        if name.startswith("Exit Rule D"):
            return lambda row: self._gte(row, "reval_10d_bad_entry_probability_10d", 0.70)
        if name.startswith("Exit Rule E"):
            return lambda row: self._lt(row, "reval_10d_expected_return_10d", 0)
        return lambda row: self._lt(row, "reval_10d_risk_adjusted_score", 0) or self._gte(row, "reval_10d_bad_entry_probability_10d", 0.70)

    def _baseline_metrics(self, trades: pd.DataFrame) -> dict[str, Any]:
        metrics = self._metrics(trades, profit_col="net_profit", holding_col="holding_days")
        metrics["rule"] = "Baseline actual exit"
        return metrics

    def _metrics(self, trades: pd.DataFrame, profit_col: str, holding_col: str) -> dict[str, Any]:
        profits = pd.to_numeric(trades[profit_col], errors="coerce").fillna(0.0)
        wins = profits > 0
        gross_profit = float(profits[wins].sum())
        gross_loss = float(-profits[profits < 0].sum())
        equity = 1_000_000 + profits.cumsum()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return {
            "rule": "",
            "total_profit": float(profits.sum()),
            "win_rate": float(wins.mean()) if len(wins) else None,
            "profit_factor": gross_profit / gross_loss if gross_loss else None,
            "max_drawdown": float(drawdown.min()) if len(drawdown) else None,
            "average_holding_days": float(pd.to_numeric(trades[holding_col], errors="coerce").mean()),
            "trade_count": int(len(trades)),
        }

    def _net_profit(self, entry_price: Any, exit_price: Any, shares: Any) -> float:
        entry = self._to_float(entry_price)
        exit_value = self._to_float(exit_price)
        share_count = self._to_float(shares)
        gross = (exit_value - entry) * share_count
        tax = gross * TAX_RATE if gross > 0 else 0.0
        return float(gross - tax)

    def _net_return(self, entry_price: Any, exit_price: Any, shares: Any) -> float:
        entry_value = self._to_float(entry_price) * self._to_float(shares)
        return self._net_profit(entry_price, exit_price, shares) / entry_value if entry_value else 0.0

    def _lt(self, row: pd.Series, column: str, threshold: float) -> bool:
        value = self._nullable_float(row.get(column))
        return value is not None and value < threshold

    def _gte(self, row: pd.Series, column: str, threshold: float) -> bool:
        value = self._nullable_float(row.get(column))
        return value is not None and value >= threshold

    def _nullable_float(self, value: Any) -> float | None:
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(self._format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
