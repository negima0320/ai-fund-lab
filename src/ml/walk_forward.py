from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import (
    LABEL_LOOKAHEAD_DAYS,
    ML_DATA_ROOT,
    ML_FEATURES_ROOT,
    ML_LABELS_ROOT,
    ML_MODELS_ROOT,
    ML_REPORTS_ROOT,
)
from ml.data_loader import JQuantsDataLoader
from ml.dataset_builder import DatasetBuilder
from ml.model_trainer import ModelTrainer
from ml.predictor import Predictor


@dataclass(frozen=True)
class WalkForwardFold:
    month: str
    train_start: str
    requested_train_end: str
    effective_train_end: str
    test_start: str
    test_end: str


class MLWalkForwardRunner:
    """Run expanding-window walk-forward training, prediction, and paper evaluation."""

    def __init__(
        self,
        feature_root: str | Path = ML_FEATURES_ROOT,
        label_root: str | Path = ML_LABELS_ROOT,
        prediction_root: str | Path = ML_DATA_ROOT / "walk_forward_predictions",
        model_root: str | Path = ML_MODELS_ROOT / "walk_forward",
        report_root: str | Path = ML_REPORTS_ROOT,
        data_loader: JQuantsDataLoader | None = None,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.label_root = Path(label_root)
        self.prediction_root = Path(prediction_root)
        self.model_root = Path(model_root)
        self.report_root = Path(report_root)
        self.data_loader = data_loader or JQuantsDataLoader()

    def run(
        self,
        train_start: str,
        test_start: str,
        test_end: str,
        ranking: str = "expected_return_10d",
        top_n: int = 10,
        exit_rule: str = "close_10d",
    ) -> dict[str, Any]:
        folds = self._folds(train_start, test_start, test_end)
        fold_rows: list[dict[str, Any]] = []
        all_trades: list[dict[str, Any]] = []
        warnings: list[str] = []

        for fold in folds:
            dataset = DatasetBuilder(self.feature_root, self.label_root).build_dataset(
                fold.train_start,
                fold.effective_train_end,
            )
            if dataset.empty:
                warnings.append(f"{fold.month}: skipped because train dataset is empty")
                fold_rows.append(self._empty_fold_summary(fold))
                continue

            train_df, valid_df = self._split_train_valid(dataset)
            trainer = ModelTrainer(
                archive_root=self.model_root / "archive",
                current_root=self.model_root / "current" / fold.month,
                timestamp=f"walk_forward_{fold.month.replace('-', '')}",
            )
            trained = trainer.train_all(train_df, valid_df)
            trainer.save_models(trained["models"], trained["metrics"])

            predicted_dates = self._predict_month(fold, trainer.current_root)
            trades = self._simulate_month_portfolio(fold, ranking, top_n, exit_rule)
            all_trades.extend(trades)
            summary = self._summarize_trades(trades)
            fold_rows.append(
                {
                    "month": fold.month,
                    "train_start": fold.train_start,
                    "requested_train_end": fold.requested_train_end,
                    "effective_train_end": fold.effective_train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "train_rows": int(len(train_df)),
                    "valid_rows": int(len(valid_df)),
                    "predicted_dates": len(predicted_dates),
                    **summary,
                }
            )

        result = {
            "period": {"test_start": test_start, "test_end": test_end},
            "train_start": train_start,
            "ranking": ranking,
            "top_n": int(top_n),
            "exit_rule": exit_rule,
            "folds": fold_rows,
            "overall": self._summarize_trades(all_trades),
            "trades": all_trades,
            "warnings": warnings,
        }
        return result

    def save_report(self, result: dict[str, Any]) -> Path:
        path = self.report_root / f"walk_forward_{self._period_slug(result)}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_markdown(result), encoding="utf-8")
        return path

    def save_json(self, result: dict[str, Any]) -> Path:
        path = self.report_root / f"walk_forward_{self._period_slug(result)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def format_markdown(self, result: dict[str, Any]) -> str:
        period = result["period"]
        return "\n".join(
            [
                "# ML Walk-Forward Evaluation",
                "",
                f"- test_period: {period['test_start']} to {period['test_end']}",
                f"- train_start: {result['train_start']}",
                f"- ranking: {result['ranking']}_top{result['top_n']}",
                f"- exit_rule: {result['exit_rule']}",
                "- method: expanding-window monthly retrain",
                "- label safety: train rows are capped to labels that are 20 business days old by train-end",
                "- note: report-only; existing trading logic is unchanged",
                "",
                "## Overall",
                "",
                self._table([result["overall"]], ["total_trades", "win_rate", "average_return", "total_return", "profit_factor", "max_drawdown"]),
                "",
                "## Monthly Folds",
                "",
                self._table(
                    result["folds"],
                    [
                        "month",
                        "requested_train_end",
                        "effective_train_end",
                        "train_rows",
                        "valid_rows",
                        "predicted_dates",
                        "total_trades",
                        "win_rate",
                        "monthly_return",
                        "profit_factor",
                        "max_drawdown",
                    ],
                ),
                "",
                "## Warnings",
                "",
                "\n".join(f"- {warning}" for warning in result.get("warnings", [])) or "_No warnings._",
                "",
            ]
        )

    def _folds(self, train_start: str, test_start: str, test_end: str) -> list[WalkForwardFold]:
        folds = []
        for month_start in pd.date_range(test_start, test_end, freq="MS"):
            month_end = min(month_start + pd.offsets.MonthEnd(0), pd.Timestamp(test_end))
            requested_train_end = month_start - pd.Timedelta(days=1)
            effective_train_end = self._effective_train_end(requested_train_end)
            folds.append(
                WalkForwardFold(
                    month=month_start.strftime("%Y-%m"),
                    train_start=train_start,
                    requested_train_end=requested_train_end.strftime("%Y-%m-%d"),
                    effective_train_end=effective_train_end.strftime("%Y-%m-%d"),
                    test_start=month_start.strftime("%Y-%m-%d"),
                    test_end=month_end.strftime("%Y-%m-%d"),
                )
            )
        return folds

    def _effective_train_end(self, requested_train_end: pd.Timestamp) -> pd.Timestamp:
        prices = self.data_loader.load_prices(
            (requested_train_end - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
            requested_train_end.strftime("%Y-%m-%d"),
        )
        if prices.empty:
            return requested_train_end - pd.Timedelta(days=35)
        dates = sorted(pd.to_datetime(prices["date"], errors="coerce").dropna().unique())
        dates = [pd.Timestamp(date) for date in dates if pd.Timestamp(date) <= requested_train_end]
        if len(dates) <= LABEL_LOOKAHEAD_DAYS:
            return requested_train_end - pd.Timedelta(days=35)
        return dates[-LABEL_LOOKAHEAD_DAYS - 1]

    def _split_train_valid(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = dataset.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        unique_dates = sorted(data["date"].dropna().unique())
        if len(unique_dates) < 5:
            return data.reset_index(drop=True), data.reset_index(drop=True)
        valid_count = max(1, min(20, len(unique_dates) // 5))
        valid_start = pd.Timestamp(unique_dates[-valid_count])
        train = data[data["date"] < valid_start].copy()
        valid = data[data["date"] >= valid_start].copy()
        if train.empty or valid.empty:
            return data.reset_index(drop=True), data.reset_index(drop=True)
        return train.reset_index(drop=True), valid.reset_index(drop=True)

    def _predict_month(self, fold: WalkForwardFold, model_root: Path) -> list[str]:
        predictor = Predictor(
            feature_root=self.feature_root,
            model_root=model_root,
            prediction_root=self.prediction_root,
        )
        predictor.load_current_models()
        predicted_dates = []
        for date_text in self._date_texts(fold.test_start, fold.test_end):
            if not (self.feature_root / f"features_{date_text}.parquet").exists():
                continue
            predictions = predictor.predict_daily(date_text)
            if predictions.empty:
                continue
            predictor.save_predictions(predictions, date_text)
            predicted_dates.append(date_text)
        return predicted_dates

    def _simulate_month_portfolio(
        self,
        fold: WalkForwardFold,
        ranking: str,
        top_n: int,
        exit_rule: str,
    ) -> list[dict[str, Any]]:
        if ranking != "expected_return_10d":
            raise ValueError("Phase 20 currently supports expected_return_10d only")
        if exit_rule != "close_10d":
            raise ValueError("Phase 20 currently supports close_10d only")

        prices = self._load_prices(fold.test_start, fold.test_end)
        if prices.empty:
            return []
        price_by_code = {
            str(code): group.sort_values("date").reset_index(drop=True)
            for code, group in prices.groupby("code", dropna=False)
        }
        trades = []
        for date_text in self._date_texts(fold.test_start, fold.test_end):
            prediction_path = self.prediction_root / f"predictions_{date_text}.parquet"
            if not prediction_path.exists():
                continue
            predictions = pd.read_parquet(prediction_path)
            if predictions.empty or "expected_return_10d" not in predictions.columns:
                continue
            predictions = predictions.dropna(subset=["expected_return_10d"]).copy()
            predictions["code"] = predictions["code"].astype("string")
            top = predictions.sort_values("expected_return_10d", ascending=False).head(top_n)
            for rank, row in enumerate(top.to_dict("records"), start=1):
                trade = self._simulate_close_10d_trade(date_text, row, rank, price_by_code)
                if trade:
                    trade["month"] = fold.month
                    trades.append(trade)
        return trades

    def _simulate_close_10d_trade(
        self,
        signal_date: str,
        row: dict[str, Any],
        rank: int,
        price_by_code: dict[str, pd.DataFrame],
    ) -> dict[str, Any] | None:
        code = str(row["code"])
        prices = price_by_code.get(code)
        if prices is None or prices.empty:
            return None
        entry_idx = self._first_index_after(prices, pd.Timestamp(signal_date))
        if entry_idx is None or entry_idx + 9 >= len(prices):
            return None
        entry = prices.iloc[entry_idx]
        exit_row = prices.iloc[entry_idx + 9]
        entry_price = float(entry["open"])
        exit_price = float(exit_row["close"])
        if entry_price <= 0:
            return None
        trade_return = exit_price / entry_price - 1
        return {
            "month": "",
            "signal_date": signal_date,
            "code": code,
            "rank": int(rank),
            "entry_date": self._json_value(entry["date"]),
            "exit_date": self._json_value(exit_row["date"]),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": float(trade_return),
            "expected_return_10d": self._json_value(row.get("expected_return_10d")),
        }

    def _load_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        end = (pd.Timestamp(end_date) + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        prices = self.data_loader.load_prices(start_date, end)
        if prices.empty:
            return prices
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices["code"] = prices["code"].astype("string")
        for column in ["open", "close"]:
            prices[column] = pd.to_numeric(prices[column], errors="coerce")
        return prices.dropna(subset=["date", "code", "open", "close"]).sort_values(["code", "date"])

    def _summarize_trades(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        df = pd.DataFrame(trades)
        if df.empty:
            return {
                "total_trades": 0,
                "win_rate": None,
                "average_return": None,
                "monthly_return": 0.0,
                "total_return": 0.0,
                "profit_factor": None,
                "max_drawdown": None,
            }
        returns = pd.to_numeric(df["return"], errors="coerce").dropna()
        return {
            "total_trades": int(len(returns)),
            "win_rate": float((returns > 0).mean()),
            "average_return": float(returns.mean()),
            "monthly_return": float(returns.sum()),
            "total_return": float(returns.sum()),
            "profit_factor": self._profit_factor(returns),
            "max_drawdown": self._max_drawdown(returns),
        }

    def _empty_fold_summary(self, fold: WalkForwardFold) -> dict[str, Any]:
        return {
            "month": fold.month,
            "train_start": fold.train_start,
            "requested_train_end": fold.requested_train_end,
            "effective_train_end": fold.effective_train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "train_rows": 0,
            "valid_rows": 0,
            "predicted_dates": 0,
            **self._summarize_trades([]),
        }

    def _profit_factor(self, returns: pd.Series) -> float | None:
        gains = returns[returns > 0].sum()
        losses = returns[returns < 0].sum()
        if losses == 0:
            return None
        return float(gains / abs(losses))

    def _max_drawdown(self, returns: pd.Series) -> float | None:
        if returns.empty:
            return None
        equity = 1 + returns.cumsum() / 10
        running_max = equity.cummax()
        return float((equity / running_max - 1).min())

    def _first_index_after(self, prices: pd.DataFrame, signal_date: pd.Timestamp) -> int | None:
        matches = prices.index[prices["date"] > signal_date].tolist()
        return int(matches[0]) if matches else None

    def _date_texts(self, start_date: str, end_date: str) -> list[str]:
        return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]

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

    def _period_slug(self, result: dict[str, Any]) -> str:
        period = result["period"]
        start = pd.Timestamp(period["test_start"]).strftime("%Y-%m")
        end = pd.Timestamp(period["test_end"]).strftime("%Y-%m")
        return f"{start}_to_{end}"

    def _json_value(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if isinstance(value, (bool, int, float, str)):
            return value
        return str(value)
