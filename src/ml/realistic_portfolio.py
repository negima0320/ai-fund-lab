from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import ML_FEATURES_ROOT, ML_LABELS_ROOT, ML_PREDICTIONS_ROOT, ML_REPORTS_ROOT
from ml.data_loader import JQuantsDataLoader


RANKING_COLUMNS = {
    "expected_return_10d": "expected_return_10d",
    "expected_max_return_20d": "expected_max_return_20d",
    "ml_score": "ml_score",
}

GRID_RANKINGS = ["expected_return_10d", "expected_max_return_20d", "ml_score"]
GRID_MAX_POSITIONS = [5, 10]
GRID_EXIT_RULES = ["close_10d", "close_20d"]
GRID_MIN_TURNOVER_VALUES = [50_000_000, 100_000_000]


@dataclass(frozen=True)
class RealisticPortfolioConfig:
    ranking: str
    top_n: int
    initial_cash: float
    position_size: float
    max_positions: int
    exit_rule: str
    fee_rate: float
    slippage_rate: float
    min_turnover_value: float

    @property
    def config_id(self) -> str:
        return (
            f"{self.ranking}_top{self.top_n}_"
            f"pos{self.max_positions}_{self.exit_rule}_"
            f"turnover{int(self.min_turnover_value)}"
        )


class MLRealisticPortfolioSimulator:
    """Simulate realistic report-only portfolios from ML ranking candidates."""

    def __init__(
        self,
        predictions_root: str | Path = ML_PREDICTIONS_ROOT,
        labels_root: str | Path = ML_LABELS_ROOT,
        features_root: str | Path = ML_FEATURES_ROOT,
        report_root: str | Path = ML_REPORTS_ROOT,
        data_loader: JQuantsDataLoader | None = None,
    ) -> None:
        self.predictions_root = Path(predictions_root)
        self.labels_root = Path(labels_root)
        self.features_root = Path(features_root)
        self.report_root = Path(report_root)
        self.data_loader = data_loader or JQuantsDataLoader()

    def simulate_grid(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 10,
        initial_cash: float = 1_000_000,
        position_size: float = 100_000,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.001,
    ) -> dict[str, Any]:
        configs = [
            RealisticPortfolioConfig(
                ranking=ranking,
                top_n=top_n,
                initial_cash=initial_cash,
                position_size=position_size,
                max_positions=max_positions,
                exit_rule=exit_rule,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                min_turnover_value=min_turnover,
            )
            for ranking in GRID_RANKINGS
            for max_positions in GRID_MAX_POSITIONS
            for exit_rule in GRID_EXIT_RULES
            for min_turnover in GRID_MIN_TURNOVER_VALUES
        ]
        return self.simulate_configs(start_date, end_date, configs)

    def simulate_one(self, start_date: str, end_date: str, config: RealisticPortfolioConfig) -> dict[str, Any]:
        return self.simulate_configs(start_date, end_date, [config])

    def simulate_configs(self, start_date: str, end_date: str, configs: list[RealisticPortfolioConfig]) -> dict[str, Any]:
        candidates = self._load_candidates(start_date, end_date, top_n=max(config.top_n for config in configs))
        prices = self._load_prices(start_date, end_date)
        price_by_code = {
            str(code): group.sort_values("date").reset_index(drop=True)
            for code, group in prices.groupby("code", dropna=False)
        } if not prices.empty else {}

        summaries = []
        monthly_rows = []
        trade_rows = []
        for config in configs:
            summary, monthly, trades = self._simulate_config(candidates, price_by_code, config)
            summaries.append(summary)
            monthly_rows.extend(monthly)
            trade_rows.extend(trades)
        return {
            "period": {"start_date": start_date, "end_date": end_date},
            "summary": summaries,
            "monthly_summary": monthly_rows,
            "trades": trade_rows,
        }

    def save_report(self, result: dict[str, Any]) -> Path:
        period = result["period"]
        path = self.report_root / f"ml_realistic_portfolio_{period['start_date']}_to_{period['end_date']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_markdown(result), encoding="utf-8")
        return path

    def save_json(self, result: dict[str, Any]) -> Path:
        period = result["period"]
        path = self.report_root / f"ml_realistic_portfolio_{period['start_date']}_to_{period['end_date']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def save_trades_csv(self, result: dict[str, Any]) -> Path:
        period = result["period"]
        path = self.report_root / f"ml_realistic_trades_{period['start_date']}_to_{period['end_date']}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.get("trades", [])).to_csv(path, index=False)
        return path

    def format_markdown(self, result: dict[str, Any]) -> str:
        period = result["period"]
        return "\n".join(
            [
                "# ML Realistic Portfolio Simulation",
                "",
                f"- period: {period['start_date']} to {period['end_date']}",
                "- note: report-only; existing trading logic is unchanged",
                "",
                "## Grid Summary",
                "",
                self._table(
                    result["summary"],
                    [
                        "config_id",
                        "ranking",
                        "max_positions",
                        "exit_rule",
                        "min_turnover_value",
                        "final_assets",
                        "total_profit",
                        "total_return",
                        "win_rate",
                        "profit_factor",
                        "max_drawdown",
                        "total_trades",
                        "average_holding_days",
                        "rejected_by_cash",
                        "rejected_by_max_positions",
                        "rejected_by_duplicate",
                        "rejected_by_liquidity",
                    ],
                ),
                "",
                "## Monthly Summary",
                "",
                self._table(
                    result["monthly_summary"],
                    ["config_id", "month", "monthly_return", "monthly_win_rate", "monthly_trade_count"],
                ),
                "",
            ]
        )

    def _load_candidates(self, start_date: str, end_date: str, top_n: int) -> pd.DataFrame:
        frames = []
        for date_text in self._date_texts(start_date, end_date):
            prediction_path = self.predictions_root / f"predictions_{date_text}.parquet"
            label_path = self.labels_root / f"labels_{date_text}.parquet"
            feature_path = self.features_root / f"features_{date_text}.parquet"
            if not prediction_path.exists() or not label_path.exists():
                continue
            predictions = pd.read_parquet(prediction_path)
            labels = pd.read_parquet(label_path)
            if predictions.empty or labels.empty:
                continue
            data = self._normalize_keys(predictions).merge(self._normalize_keys(labels), on=["date", "code"], how="inner")
            if feature_path.exists():
                features = self._normalize_keys(pd.read_parquet(feature_path))
                feature_columns = [column for column in ["date", "code", "turnover_value", "volume", "close"] if column in features.columns]
                data = data.merge(features[feature_columns], on=["date", "code"], how="left")
            for column in [*RANKING_COLUMNS.values(), "turnover_value"]:
                if column in data.columns:
                    data[column] = pd.to_numeric(data[column], errors="coerce")
            for ranking, score_column in RANKING_COLUMNS.items():
                if score_column not in data.columns:
                    continue
                top = data.dropna(subset=[score_column]).sort_values(score_column, ascending=False).head(top_n).copy()
                if top.empty:
                    continue
                top["ranking"] = ranking
                top["rank"] = range(1, len(top) + 1)
                frames.append(top.dropna(axis=1, how="all"))
        if not frames:
            return pd.DataFrame()
        output = pd.concat(frames, ignore_index=True)
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
        output["code"] = output["code"].astype("string")
        return output.sort_values(["date", "ranking", "rank"]).reset_index(drop=True)

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
        return prices.dropna(subset=["date", "code", "open", "close"]).sort_values(["code", "date"])

    def _simulate_config(
        self,
        candidates: pd.DataFrame,
        price_by_code: dict[str, pd.DataFrame],
        config: RealisticPortfolioConfig,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        cash = float(config.initial_cash)
        positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        equity_rows: list[tuple[pd.Timestamp, float]] = []
        rejections = {"cash": 0, "max_positions": 0, "duplicate": 0, "liquidity": 0, "no_exit": 0}
        data = candidates[candidates["ranking"].eq(config.ranking)].sort_values(["date", "rank"])

        for date, daily in data.groupby("date", sort=True):
            current_date = pd.Timestamp(date)
            cash = self._close_due_positions(current_date, positions, cash, trades)
            for row in daily.head(config.top_n).to_dict("records"):
                turnover = row.get("turnover_value")
                if pd.isna(turnover) or float(turnover) < config.min_turnover_value:
                    rejections["liquidity"] += 1
                    continue
                code = str(row["code"])
                if any(position["code"] == code for position in positions):
                    rejections["duplicate"] += 1
                    continue
                if len(positions) >= config.max_positions:
                    rejections["max_positions"] += 1
                    continue
                if cash < config.position_size * (1 + config.fee_rate):
                    rejections["cash"] += 1
                    continue
                setup = self._build_trade_setup(row, price_by_code.get(code), config)
                if setup is None:
                    rejections["no_exit"] += 1
                    continue
                cash -= setup["cost"]
                positions.append(setup)
            equity_rows.append((current_date, cash + sum(position["cost"] for position in positions)))

        if not data.empty:
            final_date = data["date"].max() + pd.Timedelta(days=120)
            cash = self._close_due_positions(final_date, positions, cash, trades)
            equity_rows.append((final_date, cash))

        summary = self._summary(config, trades, equity_rows, rejections, cash)
        monthly = self._monthly_summary(config, trades)
        return summary, monthly, trades

    def _build_trade_setup(
        self,
        candidate: dict[str, Any],
        prices: pd.DataFrame | None,
        config: RealisticPortfolioConfig,
    ) -> dict[str, Any] | None:
        if prices is None or prices.empty:
            return None
        signal_date = pd.Timestamp(candidate["date"]).normalize()
        entry_idx = self._first_index_after(prices, signal_date)
        if entry_idx is None:
            return None
        offset = 9 if config.exit_rule == "close_10d" else 19
        if entry_idx + offset >= len(prices):
            return None
        entry = prices.iloc[entry_idx]
        exit_row = prices.iloc[entry_idx + offset]
        raw_buy = float(entry["open"])
        raw_sell = float(exit_row["close"])
        buy_price = raw_buy * (1 + config.slippage_rate)
        sell_price = raw_sell * (1 - config.slippage_rate)
        shares = config.position_size / buy_price
        buy_amount = shares * buy_price
        buy_fee = buy_amount * config.fee_rate
        sell_amount = shares * sell_price
        sell_fee = sell_amount * config.fee_rate
        cost = buy_amount + buy_fee
        proceeds = sell_amount - sell_fee
        profit = proceeds - cost
        return {
            "config_id": config.config_id,
            "ranking": config.ranking,
            "exit_rule": config.exit_rule,
            "signal_date": signal_date,
            "code": str(candidate["code"]),
            "rank": int(candidate["rank"]),
            "entry_date": entry["date"],
            "exit_date": exit_row["date"],
            "holding_days": int(offset + 1),
            "buy_price": buy_price,
            "sell_price": sell_price,
            "shares": shares,
            "cost": cost,
            "proceeds": proceeds,
            "profit": profit,
            "return": profit / cost,
            "turnover_value": candidate.get("turnover_value"),
        }

    def _close_due_positions(
        self,
        current_date: pd.Timestamp,
        positions: list[dict[str, Any]],
        cash: float,
        trades: list[dict[str, Any]],
    ) -> float:
        remaining = []
        for position in positions:
            if pd.Timestamp(position["exit_date"]) <= current_date:
                cash += float(position["proceeds"])
                trades.append({key: self._json_value(value) for key, value in position.items()})
            else:
                remaining.append(position)
        positions[:] = remaining
        return cash

    def _summary(
        self,
        config: RealisticPortfolioConfig,
        trades: list[dict[str, Any]],
        equity_rows: list[tuple[pd.Timestamp, float]],
        rejections: dict[str, int],
        final_cash: float,
    ) -> dict[str, Any]:
        df = pd.DataFrame(trades)
        profits = pd.to_numeric(df["profit"], errors="coerce") if not df.empty else pd.Series(dtype=float)
        returns = pd.to_numeric(df["return"], errors="coerce") if not df.empty else pd.Series(dtype=float)
        final_assets = float(final_cash)
        return {
            "config_id": config.config_id,
            "ranking": config.ranking,
            "top_n": config.top_n,
            "max_positions": config.max_positions,
            "exit_rule": config.exit_rule,
            "min_turnover_value": config.min_turnover_value,
            "final_assets": final_assets,
            "total_profit": final_assets - config.initial_cash,
            "total_return": final_assets / config.initial_cash - 1,
            "win_rate": self._mean(returns > 0),
            "profit_factor": self._profit_factor(profits),
            "max_drawdown": self._max_drawdown(equity_rows),
            "total_trades": int(len(df)),
            "average_holding_days": self._mean(df["holding_days"]) if not df.empty else None,
            "rejected_by_cash": rejections["cash"],
            "rejected_by_max_positions": rejections["max_positions"],
            "rejected_by_duplicate": rejections["duplicate"],
            "rejected_by_liquidity": rejections["liquidity"],
            "rejected_by_no_exit": rejections["no_exit"],
            "average_position_size": self._mean(df["cost"]) if not df.empty else None,
        }

    def _monthly_summary(self, config: RealisticPortfolioConfig, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        df = pd.DataFrame(trades)
        if df.empty:
            return []
        df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
        df["month"] = df["exit_date"].dt.strftime("%Y-%m")
        rows = []
        for month, group in df.groupby("month", dropna=False):
            rows.append(
                {
                    "config_id": config.config_id,
                    "month": str(month),
                    "monthly_return": float(group["profit"].sum() / config.initial_cash),
                    "monthly_win_rate": self._mean(group["return"] > 0),
                    "monthly_trade_count": int(len(group)),
                }
            )
        return rows

    def _normalize_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        return data

    def _first_index_after(self, prices: pd.DataFrame, signal_date: pd.Timestamp) -> int | None:
        matches = prices.index[prices["date"] > signal_date].tolist()
        return int(matches[0]) if matches else None

    def _profit_factor(self, profits: pd.Series) -> float | None:
        gains = profits[profits > 0].sum()
        losses = profits[profits < 0].sum()
        if losses == 0:
            return None
        return float(gains / abs(losses))

    def _max_drawdown(self, equity_rows: list[tuple[pd.Timestamp, float]]) -> float | None:
        if not equity_rows:
            return None
        equity = pd.Series(
            [value for _, value in sorted(equity_rows, key=lambda row: row[0])],
            dtype=float,
        )
        running_max = equity.cummax()
        drawdown = equity / running_max - 1
        return float(drawdown.min())

    def _date_texts(self, start_date: str, end_date: str) -> list[str]:
        return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]

    def _mean(self, values: Any) -> float | None:
        value = pd.Series(values).mean()
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
