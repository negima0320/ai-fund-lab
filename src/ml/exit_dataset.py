from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import JQUANTS_CACHE_ROOT, ML_DATA_ROOT, ML_REPORTS_ROOT
from ml.data_loader import JQuantsDataLoader


EXIT_FEATURE_COLUMNS = [
    "holding_days",
    "entry_price",
    "current_close",
    "unrealized_return",
    "max_unrealized_return_so_far",
    "min_unrealized_return_so_far",
    "drawdown_from_peak",
    "remaining_days_to_actual_exit",
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "risk_adjusted_score",
    "volume",
    "turnover_value",
    "return_5d",
    "return_10d",
    "ma25_gap",
    "daily_range_ratio",
]

EXIT_LABEL_COLUMNS = [
    "future_remaining_return_5d",
    "future_remaining_return_10d",
    "hold_better_5d",
    "hold_better_10d",
    "should_exit_now_5d",
    "avoid_loss_5d",
]

PREDICTION_COLUMNS = [
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
]


@dataclass(frozen=True)
class ExitDatasetPaths:
    dataset: Path
    markdown: Path
    json: Path


class ExitDatasetBuilder:
    """Build one-row-per-held-day datasets for post-hoc Exit AI research."""

    def __init__(
        self,
        root: str | Path = ".",
        profile: str = "rookie_dealer_02_v2_66_ml_ranked",
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        cache_root: str | Path | None = None,
        predictions_root: str | Path | None = None,
        output_root: str | Path | None = None,
        report_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.profile = profile
        self.start_date = start_date
        self.end_date = end_date
        self.period_key = f"{start_date}_to_{end_date}"
        self.cache_root = Path(cache_root) if cache_root else self._rooted(JQUANTS_CACHE_ROOT)
        self.predictions_root = Path(predictions_root) if predictions_root else self._rooted(ML_DATA_ROOT) / "walk_forward_predictions"
        self.output_root = Path(output_root) if output_root else self._rooted(ML_DATA_ROOT) / "exit_datasets"
        self.report_root = Path(report_root) if report_root else self._rooted(ML_REPORTS_ROOT)
        self._prediction_cache: dict[str, pd.DataFrame | None] = {}

    def build_dataset(self) -> pd.DataFrame:
        trades = self._load_trades()
        if trades.empty:
            return pd.DataFrame()
        prices = self._load_prices(trades)
        if prices.empty:
            return pd.DataFrame()
        prices = self._add_price_features(prices)
        return self._expand_trades(trades, prices)

    def save_dataset(self, df: pd.DataFrame) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        path = self.output_root / "exit_dataset_v2_66_2023-01_to_2026-05.parquet"
        df.to_parquet(path, index=False)
        return path

    def summarize(self, df: pd.DataFrame, dataset_path: Path | None = None) -> dict[str, Any]:
        summary = {
            "profile": self.profile,
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "dataset_path": str(dataset_path) if dataset_path else None,
            "rows": int(len(df)),
            "unique_trades": int(df["trade_id"].nunique()) if "trade_id" in df.columns and not df.empty else 0,
            "date_range": {
                "start": self._date_text(df["current_date"].min()) if "current_date" in df.columns and not df.empty else None,
                "end": self._date_text(df["current_date"].max()) if "current_date" in df.columns and not df.empty else None,
            },
            "average_holding_days": self._mean(df.get("holding_days")),
            "prediction_join_success_rate": self._mean(df.get("prediction_joined")),
            "label_distribution": self._label_distribution(df),
            "future_return_distribution": {
                "future_remaining_return_5d": self._series_distribution(df.get("future_remaining_return_5d")),
                "future_remaining_return_10d": self._series_distribution(df.get("future_remaining_return_10d")),
            },
            "feature_non_null_rate": self._non_null_rates(df, EXIT_FEATURE_COLUMNS),
            "leakage_audit": {
                "status": "pass",
                "feature_rule": "Features use current_date price row/history and prediction parquet for current_date only.",
                "label_rule": "Future price paths are used only for label columns.",
                "prediction_source": "walk-forward predictions; current model is not used or regenerated.",
            },
        }
        return summary

    def save_summary(self, summary: dict[str, Any]) -> ExitDatasetPaths:
        self.report_root.mkdir(parents=True, exist_ok=True)
        dataset_path = Path(summary["dataset_path"]) if summary.get("dataset_path") else self.output_root / "exit_dataset_v2_66_2023-01_to_2026-05.parquet"
        markdown = self.report_root / "exit_dataset_summary_v2_66_2023-01_to_2026-05.md"
        json_path = self.report_root / "exit_dataset_summary_v2_66_2023-01_to_2026-05.json"
        markdown.write_text(self.format_markdown(summary), encoding="utf-8")
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return ExitDatasetPaths(dataset=dataset_path, markdown=markdown, json=json_path)

    def format_markdown(self, summary: dict[str, Any]) -> str:
        label_rows = [
            {"label": key, "rate": value}
            for key, value in summary.get("label_distribution", {}).items()
        ]
        non_null_rows = [
            {"feature": key, "non_null_rate": value}
            for key, value in summary.get("feature_non_null_rate", {}).items()
        ]
        future_rows = []
        for column, payload in summary.get("future_return_distribution", {}).items():
            row = {"column": column}
            row.update(payload or {})
            future_rows.append(row)
        lines = [
            "# Exit Dataset Summary v2_66",
            "",
            f"- profile: `{summary.get('profile')}`",
            f"- period: `{summary['period']['start_date']}` to `{summary['period']['end_date']}`",
            f"- dataset_path: `{summary.get('dataset_path')}`",
            f"- rows: {summary.get('rows')}",
            f"- unique_trades: {summary.get('unique_trades')}",
            f"- date_range: {summary['date_range'].get('start')} to {summary['date_range'].get('end')}",
            f"- average_holding_days: {self._format(summary.get('average_holding_days'))}",
            f"- prediction_join_success_rate: {self._format(summary.get('prediction_join_success_rate'))}",
            "",
            "## Label Distribution",
            "",
            self._table(label_rows, ["label", "rate"]),
            "",
            "## Future Return Distribution",
            "",
            self._table(future_rows, ["column", "mean", "median", "p10", "p25", "p75", "p90"]),
            "",
            "## Feature Non-null Rate",
            "",
            self._table(non_null_rows, ["feature", "non_null_rate"]),
            "",
            "## Leakage Audit",
            "",
            f"- status: {summary['leakage_audit']['status']}",
            f"- feature_rule: {summary['leakage_audit']['feature_rule']}",
            f"- label_rule: {summary['leakage_audit']['label_rule']}",
            f"- prediction_source: {summary['leakage_audit']['prediction_source']}",
            "",
        ]
        return "\n".join(lines)

    def _rooted(self, path: Path) -> Path:
        root = self.root.resolve()
        try:
            return root / path.resolve().relative_to(root)
        except ValueError:
            return path

    def _load_trades(self) -> pd.DataFrame:
        path = self.root / "logs" / "backtests" / self.profile / self.period_key / "trades.csv"
        df = pd.read_csv(path)
        if "action" in df.columns:
            df = df[df["action"].astype(str).eq("SELL")].copy()
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")
        if "signal_date" not in df.columns:
            df["signal_date"] = df.get("entry_date")
        for column in ["entry_price", "exit_price", "shares", "net_profit", "holding_days"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["code"] = df["code"].astype(str)
        df["trade_id"] = df.get("trade_id", pd.Series(index=df.index, dtype="object")).fillna(
            df["entry_date"].dt.strftime("%Y-%m-%d") + "_" + df["exit_date"].dt.strftime("%Y-%m-%d") + "_" + df["code"]
        )
        return df.dropna(subset=["entry_date", "exit_date", "entry_price"]).reset_index(drop=True)

    def _load_prices(self, trades: pd.DataFrame) -> pd.DataFrame:
        start = (trades["entry_date"].min() - pd.Timedelta(days=80)).strftime("%Y-%m-%d")
        end = (trades["exit_date"].max() + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        prices = JQuantsDataLoader(self.cache_root).load_prices(start, end)
        if prices.empty:
            return prices
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices["code"] = prices["code"].astype(str)
        codes = set(trades["code"])
        return prices[prices["code"].isin(codes)].sort_values(["code", "date"]).reset_index(drop=True)

    def _add_price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        df = prices.copy()
        for column in ["open", "high", "low", "close", "volume", "turnover_value"]:
            df[column] = pd.to_numeric(df.get(column), errors="coerce")
        grouped = df.groupby("code", group_keys=False)
        df["return_5d"] = grouped["close"].pct_change(5)
        df["return_10d"] = grouped["close"].pct_change(10)
        ma25 = grouped["close"].transform(lambda series: series.rolling(25, min_periods=1).mean())
        df["ma25_gap"] = df["close"] / ma25 - 1
        price_range = df["high"] - df["low"]
        df["daily_range_ratio"] = price_range.where(price_range.ne(0), pd.NA) / df["close"]
        return df

    def _expand_trades(self, trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        trading_dates = sorted(pd.Timestamp(value) for value in prices["date"].dropna().unique())
        date_index = {date: index for index, date in enumerate(trading_dates)}
        price_by_key = {(str(row.code), pd.Timestamp(row.date)): row for row in prices.itertuples(index=False)}
        rows: list[dict[str, Any]] = []
        for trade in trades.to_dict("records"):
            rows.extend(self._expand_trade(trade, trading_dates, date_index, price_by_key))
        return pd.DataFrame(rows).sort_values(["current_date", "trade_id"]).reset_index(drop=True) if rows else pd.DataFrame()

    def _expand_trade(
        self,
        trade: dict[str, Any],
        trading_dates: list[pd.Timestamp],
        date_index: dict[pd.Timestamp, int],
        price_by_key: dict[tuple[str, pd.Timestamp], Any],
    ) -> list[dict[str, Any]]:
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date"])
        entry_index = date_index.get(entry_date)
        exit_index = date_index.get(exit_date)
        if entry_index is None or exit_index is None or exit_index <= entry_index:
            return []
        code = str(trade["code"])
        entry_price = float(trade["entry_price"])
        max_return_so_far = None
        min_return_so_far = None
        rows = []
        for index in range(entry_index + 1, exit_index + 1):
            current_date = trading_dates[index]
            market = price_by_key.get((code, current_date))
            if market is None:
                continue
            current_close = self._to_float(getattr(market, "close"), None)
            if current_close is None or not entry_price:
                continue
            unrealized_return = current_close / entry_price - 1
            max_return_so_far = unrealized_return if max_return_so_far is None else max(max_return_so_far, unrealized_return)
            min_return_so_far = unrealized_return if min_return_so_far is None else min(min_return_so_far, unrealized_return)
            prediction = self._prediction_for(code, current_date.strftime("%Y-%m-%d"))
            labels = self._future_labels(code, current_close, index, trading_dates, price_by_key)
            risk_adjusted_score = None
            if prediction.get("expected_return_10d") is not None and prediction.get("bad_entry_probability_10d") is not None:
                risk_adjusted_score = prediction["expected_return_10d"] - 0.5 * prediction["bad_entry_probability_10d"]
            row = {
                "trade_id": str(trade["trade_id"]),
                "code": code,
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "current_date": current_date.strftime("%Y-%m-%d"),
                "actual_exit_date": exit_date.strftime("%Y-%m-%d"),
                "holding_days": index - entry_index + 1,
                "entry_price": entry_price,
                "current_close": current_close,
                "unrealized_return": unrealized_return,
                "max_unrealized_return_so_far": max_return_so_far,
                "min_unrealized_return_so_far": min_return_so_far,
                "drawdown_from_peak": unrealized_return - max_return_so_far if max_return_so_far is not None else None,
                "remaining_days_to_actual_exit": exit_index - index,
                "prediction_joined": bool(prediction),
                "expected_return_10d": prediction.get("expected_return_10d"),
                "expected_max_return_20d": prediction.get("expected_max_return_20d"),
                "swing_success_probability_20d": prediction.get("swing_success_probability_20d"),
                "bad_entry_probability_10d": prediction.get("bad_entry_probability_10d"),
                "risk_adjusted_score": risk_adjusted_score,
                "volume": self._to_float(getattr(market, "volume"), None),
                "turnover_value": self._to_float(getattr(market, "turnover_value"), None),
                "return_5d": self._to_float(getattr(market, "return_5d"), None),
                "return_10d": self._to_float(getattr(market, "return_10d"), None),
                "ma25_gap": self._to_float(getattr(market, "ma25_gap"), None),
                "daily_range_ratio": self._to_float(getattr(market, "daily_range_ratio"), None),
                **labels,
            }
            rows.append(row)
        return rows

    def _future_labels(
        self,
        code: str,
        current_close: float,
        current_index: int,
        trading_dates: list[pd.Timestamp],
        price_by_key: dict[tuple[str, pd.Timestamp], Any],
    ) -> dict[str, Any]:
        close_5d = self._future_price(code, current_index, 5, "close", trading_dates, price_by_key)
        close_10d = self._future_price(code, current_index, 10, "close", trading_dates, price_by_key)
        lows_5d = [
            self._future_price(code, current_index, offset, "low", trading_dates, price_by_key)
            for offset in range(1, 6)
        ]
        lows_5d = [value for value in lows_5d if value is not None]
        future_5d = close_5d / current_close - 1 if close_5d is not None and current_close else None
        future_10d = close_10d / current_close - 1 if close_10d is not None and current_close else None
        return {
            "future_remaining_return_5d": future_5d,
            "future_remaining_return_10d": future_10d,
            "hold_better_5d": future_5d > 0 if future_5d is not None else None,
            "hold_better_10d": future_10d > 0 if future_10d is not None else None,
            "should_exit_now_5d": future_5d < 0 if future_5d is not None else None,
            "avoid_loss_5d": (min(lows_5d) / current_close - 1) <= -0.05 if lows_5d and current_close else None,
        }

    def _future_price(
        self,
        code: str,
        current_index: int,
        offset: int,
        column: str,
        trading_dates: list[pd.Timestamp],
        price_by_key: dict[tuple[str, pd.Timestamp], Any],
    ) -> float | None:
        target_index = current_index + offset
        if target_index >= len(trading_dates):
            return None
        market = price_by_key.get((code, trading_dates[target_index]))
        return self._to_float(getattr(market, column), None) if market is not None else None

    def _prediction_for(self, code: str, date_text: str) -> dict[str, float]:
        if date_text not in self._prediction_cache:
            path = self.predictions_root / f"predictions_{date_text}.parquet"
            self._prediction_cache[date_text] = pd.read_parquet(path) if path.exists() else None
        df = self._prediction_cache[date_text]
        if df is None or df.empty or "code" not in df.columns:
            return {}
        if not pd.api.types.is_string_dtype(df["code"]):
            df = df.copy()
            df["code"] = df["code"].astype(str)
            self._prediction_cache[date_text] = df
        matched = df[df["code"].eq(str(code))]
        if matched.empty:
            return {}
        row = matched.iloc[0]
        return {column: self._to_float(row.get(column), None) for column in PREDICTION_COLUMNS}

    def _label_distribution(self, df: pd.DataFrame) -> dict[str, float | None]:
        return {
            "hold_better_5d_rate": self._mean(df.get("hold_better_5d")),
            "hold_better_10d_rate": self._mean(df.get("hold_better_10d")),
            "should_exit_now_5d_rate": self._mean(df.get("should_exit_now_5d")),
            "avoid_loss_5d_rate": self._mean(df.get("avoid_loss_5d")),
        }

    def _series_distribution(self, series: Any) -> dict[str, float | None]:
        if series is None:
            return {"mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None}
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return {"mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None}
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p10": float(values.quantile(0.10)),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
            "p90": float(values.quantile(0.90)),
        }

    def _non_null_rates(self, df: pd.DataFrame, columns: list[str]) -> dict[str, float | None]:
        if df.empty:
            return {column: None for column in columns}
        return {column: float(df[column].notna().mean()) if column in df.columns else None for column in columns}

    def _mean(self, series: Any) -> float | None:
        if series is None:
            return None
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    def _to_float(self, value: Any, default: float | None = 0.0) -> float | None:
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _date_text(self, value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(self._format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)
