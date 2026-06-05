from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import ML_LABELS_ROOT, ML_PREDICTIONS_ROOT, ML_REPORTS_ROOT
from ml.data_loader import JQuantsDataLoader
from ml.ranking_analysis import RANKING_SPECS


EXIT_RULES = [
    "close_20d",
    "close_10d",
    "take_profit_10pct_or_close_20d",
    "stop_loss_5pct_or_close_20d",
    "take_profit_10pct_stop_loss_5pct_or_close_20d",
]


class MLPaperPortfolioSimulator:
    """Simulate report-only paper portfolios from all-stock ML rankings."""

    def __init__(
        self,
        predictions_root: str | Path = ML_PREDICTIONS_ROOT,
        labels_root: str | Path = ML_LABELS_ROOT,
        report_root: str | Path = ML_REPORTS_ROOT,
        data_loader: JQuantsDataLoader | None = None,
    ) -> None:
        self.predictions_root = Path(predictions_root)
        self.labels_root = Path(labels_root)
        self.report_root = Path(report_root)
        self.data_loader = data_loader or JQuantsDataLoader()

    def simulate(self, start_date: str, end_date: str, top_n: int = 10) -> dict[str, Any]:
        ranked = self._ranked_candidates(start_date, end_date, top_n=top_n)
        prices = self._load_prices(start_date, end_date)
        trades = self._simulate_trades(ranked, prices)
        summary = self._summary_by_ranking_exit(trades)
        monthly = self._monthly_summary(trades)
        return {
            "period": {"start_date": start_date, "end_date": end_date},
            "top_n": int(top_n),
            "ranking_exit_summary": summary,
            "monthly_return": monthly,
            "paper_trades": self._records(trades),
        }

    def save_report(self, result: dict[str, Any]) -> Path:
        period = result["period"]
        path = self.report_root / f"ml_paper_portfolio_{period['start_date']}_to_{period['end_date']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_markdown(result), encoding="utf-8")
        return path

    def save_json(self, result: dict[str, Any]) -> Path:
        period = result["period"]
        path = self.report_root / f"ml_paper_portfolio_{period['start_date']}_to_{period['end_date']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def save_trades_csv(self, result: dict[str, Any]) -> Path:
        period = result["period"]
        path = self.report_root / f"ml_paper_trades_{period['start_date']}_to_{period['end_date']}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.get("paper_trades", [])).to_csv(path, index=False)
        return path

    def format_markdown(self, result: dict[str, Any]) -> str:
        period = result["period"]
        lines = [
            "# ML Paper Portfolio Simulation",
            "",
            f"- period: {period['start_date']} to {period['end_date']}",
            f"- top_n: {result['top_n']}",
            "- note: report-only simulation; existing trading logic is unchanged",
            "",
            "## Ranking x Exit Rule Summary",
            "",
            self._table(
                result["ranking_exit_summary"],
                [
                    "ranking",
                    "exit_rule",
                    "total_trades",
                    "win_rate",
                    "average_return",
                    "median_return",
                    "total_return_sum",
                    "profit_factor",
                    "max_drawdown",
                    "best_trade_return",
                    "worst_trade_return",
                    "bad_entry_10d_rate",
                ],
            ),
            "",
            "## Monthly Return",
            "",
            self._table(
                result["monthly_return"],
                ["ranking", "exit_rule", "month", "total_trades", "average_return", "monthly_return_sum", "win_rate"],
            ),
            "",
        ]
        return "\n".join(lines)

    def _ranked_candidates(self, start_date: str, end_date: str, top_n: int) -> pd.DataFrame:
        frames = []
        for date_text in self._date_texts(start_date, end_date):
            prediction_path = self.predictions_root / f"predictions_{date_text}.parquet"
            label_path = self.labels_root / f"labels_{date_text}.parquet"
            if not prediction_path.exists() or not label_path.exists():
                continue
            predictions = pd.read_parquet(prediction_path)
            labels = pd.read_parquet(label_path)
            if predictions.empty or labels.empty:
                continue
            joined = self._join_predictions_labels(predictions, labels)
            for ranking, score_column in RANKING_SPECS.items():
                if score_column not in joined.columns:
                    continue
                top = joined.dropna(subset=[score_column]).sort_values(score_column, ascending=False).head(top_n).copy()
                if top.empty:
                    continue
                top["ranking"] = ranking
                top["rank"] = range(1, len(top) + 1)
                frames.append(top)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _join_predictions_labels(self, predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        left = predictions.copy()
        right = labels.copy()
        left["date"] = pd.to_datetime(left["date"], errors="coerce")
        right["date"] = pd.to_datetime(right["date"], errors="coerce")
        left["code"] = left["code"].astype("string")
        right["code"] = right["code"].astype("string")
        return left.merge(right, on=["date", "code"], how="inner")

    def _load_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        end = (pd.Timestamp(end_date) + pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        prices = self.data_loader.load_prices(start_date, end)
        if prices.empty:
            return prices
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices["code"] = prices["code"].astype("string")
        for column in ["open", "high", "low", "close"]:
            prices[column] = pd.to_numeric(prices[column], errors="coerce")
        return prices.dropna(subset=["date", "code", "open", "high", "low", "close"]).sort_values(["code", "date"])

    def _simulate_trades(self, ranked: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        if ranked.empty or prices.empty:
            return pd.DataFrame()
        price_by_code = {
            str(code): group.sort_values("date").reset_index(drop=True)
            for code, group in prices.groupby("code", dropna=False)
        }
        rows = []
        for row in ranked.to_dict("records"):
            signal_date = pd.Timestamp(row["date"]).normalize()
            code = str(row["code"])
            code_prices = price_by_code.get(code)
            if code_prices is None:
                continue
            entry_idx = self._first_index_after(code_prices, signal_date)
            if entry_idx is None:
                continue
            if entry_idx + 19 >= len(code_prices):
                continue
            entry = code_prices.iloc[entry_idx]
            entry_price = float(entry["open"])
            if entry_price <= 0:
                continue
            for exit_rule in EXIT_RULES:
                trade = self._simulate_exit(code_prices, entry_idx, entry_price, exit_rule)
                if not trade:
                    continue
                rows.append(
                    {
                        "ranking": row["ranking"],
                        "rank": row["rank"],
                        "signal_date": signal_date,
                        "code": code,
                        "entry_date": entry["date"],
                        "entry_price": entry_price,
                        "exit_rule": exit_rule,
                        "exit_date": trade["exit_date"],
                        "exit_price": trade["exit_price"],
                        "return": trade["return"],
                        "exit_reason": trade["exit_reason"],
                        "expected_max_return_20d": row.get("expected_max_return_20d"),
                        "swing_success_probability_20d": row.get("swing_success_probability_20d"),
                        "ml_score": row.get("ml_score"),
                        "expected_return_10d": row.get("expected_return_10d"),
                        "future_10d_return": row.get("future_10d_return"),
                        "future_max_return_20d": row.get("future_max_return_20d"),
                        "future_swing_success_20d": row.get("future_swing_success_20d"),
                        "bad_entry_10d": row.get("bad_entry_10d"),
                    }
                )
        return pd.DataFrame(rows)

    def _simulate_exit(self, prices: pd.DataFrame, entry_idx: int, entry_price: float, exit_rule: str) -> dict[str, Any] | None:
        exit_10 = prices.iloc[entry_idx + 9]
        exit_20 = prices.iloc[entry_idx + 19]
        window = prices.iloc[entry_idx : entry_idx + 20]
        take_profit_price = entry_price * 1.10
        stop_loss_price = entry_price * 0.95

        if exit_rule == "close_10d":
            return self._exit_result(exit_10, float(exit_10["close"]), entry_price, "close_10d")
        if exit_rule == "close_20d":
            return self._exit_result(exit_20, float(exit_20["close"]), entry_price, "close_20d")
        if exit_rule == "take_profit_10pct_or_close_20d":
            take_profit = window[window["high"] >= take_profit_price]
            if not take_profit.empty:
                hit = take_profit.iloc[0]
                return self._exit_result(hit, take_profit_price, entry_price, "take_profit_10pct")
            return self._exit_result(exit_20, float(exit_20["close"]), entry_price, "close_20d")
        if exit_rule == "stop_loss_5pct_or_close_20d":
            stop_loss = window[window["low"] <= stop_loss_price]
            if not stop_loss.empty:
                hit = stop_loss.iloc[0]
                return self._exit_result(hit, stop_loss_price, entry_price, "stop_loss_5pct")
            return self._exit_result(exit_20, float(exit_20["close"]), entry_price, "close_20d")
        if exit_rule == "take_profit_10pct_stop_loss_5pct_or_close_20d":
            for _, day in window.iterrows():
                if day["low"] <= stop_loss_price:
                    return self._exit_result(day, stop_loss_price, entry_price, "stop_loss_5pct")
                if day["high"] >= take_profit_price:
                    return self._exit_result(day, take_profit_price, entry_price, "take_profit_10pct")
            return self._exit_result(exit_20, float(exit_20["close"]), entry_price, "close_20d")
        return None

    def _exit_result(self, row: pd.Series, exit_price: float, entry_price: float, reason: str) -> dict[str, Any]:
        return {
            "exit_date": row["date"],
            "exit_price": float(exit_price),
            "return": float(exit_price / entry_price - 1),
            "exit_reason": reason,
        }

    def _first_index_after(self, prices: pd.DataFrame, signal_date: pd.Timestamp) -> int | None:
        matches = prices.index[prices["date"] > signal_date].tolist()
        return int(matches[0]) if matches else None

    def _summary_by_ranking_exit(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty:
            return []
        rows = []
        for (ranking, exit_rule), group in trades.groupby(["ranking", "exit_rule"], dropna=False):
            item = self._performance(group)
            item["ranking"] = str(ranking)
            item["exit_rule"] = str(exit_rule)
            rows.append(item)
        return sorted(rows, key=lambda row: (row["ranking"], row["exit_rule"]))

    def _monthly_summary(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty:
            return []
        data = trades.copy()
        data["month"] = pd.to_datetime(data["signal_date"], errors="coerce").dt.strftime("%Y-%m")
        rows = []
        for (ranking, exit_rule, month), group in data.groupby(["ranking", "exit_rule", "month"], dropna=False):
            rows.append(
                {
                    "ranking": str(ranking),
                    "exit_rule": str(exit_rule),
                    "month": str(month),
                    "total_trades": int(len(group)),
                    "average_return": self._mean(group["return"]),
                    "monthly_return_sum": self._daily_portfolio_return_sum(group),
                    "win_rate": self._mean(group["return"] > 0),
                }
            )
        return sorted(rows, key=lambda row: (row["ranking"], row["exit_rule"], row["month"]))

    def _performance(self, group: pd.DataFrame) -> dict[str, Any]:
        returns = pd.to_numeric(group["return"], errors="coerce").dropna()
        best = group.loc[group["return"].idxmax()] if not group.empty else None
        worst = group.loc[group["return"].idxmin()] if not group.empty else None
        return {
            "total_trades": int(len(group)),
            "win_rate": self._mean(returns > 0),
            "average_return": self._mean(returns),
            "median_return": self._median(returns),
            "total_return_sum": float(returns.sum()) if not returns.empty else None,
            "profit_factor": self._profit_factor(returns),
            "max_drawdown": self._max_drawdown(group),
            "best_trade_return": self._json_value(best["return"]) if best is not None else None,
            "best_trade_code": self._json_value(best["code"]) if best is not None else None,
            "best_trade_signal_date": self._json_value(best["signal_date"]) if best is not None else None,
            "worst_trade_return": self._json_value(worst["return"]) if worst is not None else None,
            "worst_trade_code": self._json_value(worst["code"]) if worst is not None else None,
            "worst_trade_signal_date": self._json_value(worst["signal_date"]) if worst is not None else None,
            "bad_entry_10d_rate": self._mean(group["bad_entry_10d"]) if "bad_entry_10d" in group.columns else None,
        }

    def _daily_portfolio_returns(self, trades: pd.DataFrame) -> pd.Series:
        if trades.empty:
            return pd.Series(dtype=float)
        data = trades.copy()
        data["signal_date"] = pd.to_datetime(data["signal_date"], errors="coerce")
        return data.groupby("signal_date")["return"].mean().sort_index()

    def _daily_portfolio_return_sum(self, trades: pd.DataFrame) -> float | None:
        daily = self._daily_portfolio_returns(trades)
        return float(daily.sum()) if not daily.empty else None

    def _max_drawdown(self, trades: pd.DataFrame) -> float | None:
        daily = self._daily_portfolio_returns(trades)
        if daily.empty:
            return None
        cumulative = daily.cumsum()
        drawdown = cumulative - cumulative.cummax()
        return float(drawdown.min())

    def _profit_factor(self, returns: pd.Series) -> float | None:
        gains = returns[returns > 0].sum()
        losses = returns[returns < 0].sum()
        if losses == 0:
            return None
        return float(gains / abs(losses))

    def _date_texts(self, start_date: str, end_date: str) -> list[str]:
        return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]

    def _records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        return [
            {key: self._json_value(value) for key, value in row.items()}
            for row in df.to_dict("records")
        ]

    def _mean(self, values: Any) -> float | None:
        value = pd.Series(values).mean()
        return None if pd.isna(value) else float(value)

    def _median(self, values: Any) -> float | None:
        value = pd.Series(values).median()
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

    def _json_value(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if isinstance(value, (bool, int, float, str)):
            return value
        return str(value)
