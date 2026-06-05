from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import JQUANTS_CACHE_ROOT, ML_DATA_ROOT, ML_LABELS_ROOT, ML_REPORTS_ROOT
from ml.data_loader import JQuantsDataLoader


@dataclass(frozen=True)
class RankingStrategy:
    name: str
    bad_entry_threshold: float | None = None
    sector_cap: int | None = None
    code_month_cap: int | None = None


DEFAULT_STRATEGIES = [
    RankingStrategy("expected_return_10d"),
    RankingStrategy("risk_adjusted_return"),
    RankingStrategy("return_upside_combo"),
    RankingStrategy("swing_combo"),
    RankingStrategy("expected_return_10d_bad_entry_lt_0_60", bad_entry_threshold=0.60),
    RankingStrategy("expected_return_10d_bad_entry_lt_0_70", bad_entry_threshold=0.70),
    RankingStrategy("expected_return_10d_sector_cap_3", sector_cap=3),
    RankingStrategy("expected_return_10d_bad_entry_lt_0_70_sector_cap_3", bad_entry_threshold=0.70, sector_cap=3),
]


class WalkForwardRankingComparator:
    """Compare bad-entry-aware walk-forward ranking strategies without retraining."""

    def __init__(
        self,
        prediction_root: str | Path = ML_DATA_ROOT / "walk_forward_predictions",
        label_root: str | Path = ML_LABELS_ROOT,
        report_root: str | Path = ML_REPORTS_ROOT,
        cache_root: str | Path = JQUANTS_CACHE_ROOT,
    ) -> None:
        self.prediction_root = Path(prediction_root)
        self.label_root = Path(label_root)
        self.report_root = Path(report_root)
        self.data_loader = JQuantsDataLoader(cache_root)

    def compare(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 10,
        exit_rule: str = "close_10d",
        strategies: list[RankingStrategy] | None = None,
    ) -> dict[str, Any]:
        if exit_rule != "close_10d":
            raise ValueError("Phase 22 currently supports close_10d only")
        specs = strategies or DEFAULT_STRATEGIES
        universe = self._load_prediction_label_universe(start_date, end_date)
        prices = self._load_prices(start_date, end_date)
        price_by_code = {
            str(code): group.sort_values("date").reset_index(drop=True)
            for code, group in prices.groupby("code", dropna=False)
        } if not prices.empty else {}

        summary_rows = []
        monthly_rows = []
        trade_rows = []
        for spec in specs:
            trades = self._simulate_strategy(universe, price_by_code, spec, top_n)
            trade_rows.extend(trades)
            summary_rows.append({"strategy": spec.name, **self._summary(trades)})
            monthly_rows.extend(self._monthly_summary(spec.name, trades))

        result = {
            "period": {"start_date": start_date, "end_date": end_date},
            "top_n": int(top_n),
            "exit_rule": exit_rule,
            "summary": summary_rows,
            "monthly_summary": monthly_rows,
            "trades": trade_rows,
        }
        return result

    def save_report(self, result: dict[str, Any]) -> Path:
        period = result["period"]
        path = self.report_root / f"walk_forward_ranking_compare_{self._month_slug(period['start_date'])}_to_{self._month_slug(period['end_date'])}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_markdown(result), encoding="utf-8")
        return path

    def save_json(self, result: dict[str, Any]) -> Path:
        period = result["period"]
        path = self.report_root / f"walk_forward_ranking_compare_{self._month_slug(period['start_date'])}_to_{self._month_slug(period['end_date'])}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def format_markdown(self, result: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Walk-Forward Ranking Comparison",
                "",
                f"- period: {result['period']['start_date']} to {result['period']['end_date']}",
                f"- top_n: {result['top_n']}",
                f"- exit_rule: {result['exit_rule']}",
                "- note: no retraining; existing walk-forward predictions are reused",
                "",
                "## Overall Summary",
                "",
                self._table(result["summary"]),
                "",
                "## Monthly Summary",
                "",
                self._table(result["monthly_summary"]),
                "",
            ]
        )

    def _load_prediction_label_universe(self, start_date: str, end_date: str) -> pd.DataFrame:
        frames = []
        info = self._listed_info(end_date)
        for date_text in self._date_texts(start_date, end_date):
            prediction_path = self.prediction_root / f"predictions_{date_text}.parquet"
            label_path = self.label_root / f"labels_{date_text}.parquet"
            if not prediction_path.exists() or not label_path.exists():
                continue
            predictions = pd.read_parquet(prediction_path)
            labels = pd.read_parquet(label_path)
            if predictions.empty or labels.empty:
                continue
            joined = self._normalize(predictions).merge(self._normalize(labels), on=["date", "code"], how="inner")
            if not info.empty:
                joined = joined.merge(info, on="code", how="left")
            joined["month"] = joined["date"].dt.strftime("%Y-%m")
            frames.append(joined.dropna(axis=1, how="all"))
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        for column in [
            "expected_return_10d",
            "upside_probability_10d",
            "bad_entry_probability_10d",
            "expected_max_return_20d",
            "swing_success_probability_20d",
            "future_10d_return",
        ]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        return data

    def _simulate_strategy(
        self,
        universe: pd.DataFrame,
        price_by_code: dict[str, pd.DataFrame],
        spec: RankingStrategy,
        top_n: int,
    ) -> list[dict[str, Any]]:
        if universe.empty:
            return []
        data = universe.copy()
        data["ranking_score"] = self._score(data, spec.name)
        data = data.dropna(subset=["ranking_score"])
        if spec.bad_entry_threshold is not None and "bad_entry_probability_10d" in data.columns:
            data = data[data["bad_entry_probability_10d"] < spec.bad_entry_threshold]

        trades = []
        monthly_code_counts: dict[tuple[str, str], int] = {}
        for date, daily in data.groupby("date", sort=True):
            selected = []
            sector_counts: dict[str, int] = {}
            for row in daily.sort_values("ranking_score", ascending=False).to_dict("records"):
                month = str(row.get("month"))
                code = str(row["code"])
                sector = str(row.get("sector_name")) if pd.notna(row.get("sector_name")) else ""
                if spec.sector_cap is not None and sector:
                    if sector_counts.get(sector, 0) >= spec.sector_cap:
                        continue
                if spec.code_month_cap is not None:
                    key = (month, code)
                    if monthly_code_counts.get(key, 0) >= spec.code_month_cap:
                        continue
                selected.append(row)
                if sector:
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                if spec.code_month_cap is not None:
                    monthly_code_counts[(month, code)] = monthly_code_counts.get((month, code), 0) + 1
                if len(selected) >= top_n:
                    break
            for rank, row in enumerate(selected, start=1):
                trade = self._simulate_close_10d_trade(row, rank, price_by_code)
                if trade:
                    trade["strategy"] = spec.name
                    trades.append(trade)
        return trades

    def _score(self, data: pd.DataFrame, name: str) -> pd.Series:
        if name.startswith("expected_return_10d"):
            return data["expected_return_10d"]
        if name == "risk_adjusted_return":
            return data["expected_return_10d"] - 0.5 * data["bad_entry_probability_10d"]
        if name == "risk_adjusted_return_strong":
            return data["expected_return_10d"] - data["bad_entry_probability_10d"]
        if name == "swing_risk_adjusted":
            return data["expected_max_return_20d"] - 0.5 * data["bad_entry_probability_10d"]
        if name == "return_upside_combo":
            return data["expected_return_10d"] + 0.5 * data["upside_probability_10d"] - 0.5 * data["bad_entry_probability_10d"]
        if name == "swing_combo":
            return data["expected_return_10d"] + 0.5 * data["swing_success_probability_20d"] - 0.5 * data["bad_entry_probability_10d"]
        raise ValueError(f"unknown ranking strategy: {name}")

    def _simulate_close_10d_trade(self, row: dict[str, Any], rank: int, price_by_code: dict[str, pd.DataFrame]) -> dict[str, Any] | None:
        code = str(row["code"])
        prices = price_by_code.get(code)
        if prices is None or prices.empty:
            return None
        signal_date = pd.Timestamp(row["date"]).normalize()
        entry_idx = self._first_index_after(prices, signal_date)
        if entry_idx is None or entry_idx + 9 >= len(prices):
            return None
        entry = prices.iloc[entry_idx]
        exit_row = prices.iloc[entry_idx + 9]
        entry_price = float(entry["open"])
        exit_price = float(exit_row["close"])
        if entry_price <= 0:
            return None
        return {
            "month": str(row.get("month")),
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "code": code,
            "rank": int(rank),
            "entry_date": self._json_value(entry["date"]),
            "exit_date": self._json_value(exit_row["date"]),
            "return": float(exit_price / entry_price - 1),
            "ranking_score": self._json_value(row.get("ranking_score")),
            "expected_return_10d": self._json_value(row.get("expected_return_10d")),
            "bad_entry_probability_10d": self._json_value(row.get("bad_entry_probability_10d")),
            "expected_max_return_20d": self._json_value(row.get("expected_max_return_20d")),
            "swing_success_probability_20d": self._json_value(row.get("swing_success_probability_20d")),
            "future_10d_return": self._json_value(row.get("future_10d_return")),
            "bad_entry_10d": self._json_value(row.get("bad_entry_10d")),
            "sector_name": self._json_value(row.get("sector_name")),
            "market": self._json_value(row.get("market")),
        }

    def _summary(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        df = pd.DataFrame(trades)
        if df.empty:
            return {
                "total_trades": 0,
                "total_return": 0.0,
                "win_rate": None,
                "profit_factor": None,
                "max_drawdown": None,
                "bad_entry_rate": None,
                "unique_codes": 0,
                "top_sector": "",
                "may_return": 0.0,
                "may_win_rate": None,
                "may_profit_factor": None,
                "may_bad_entry_rate": None,
            }
        returns = pd.to_numeric(df["return"], errors="coerce").dropna()
        may = df[df["month"].eq("2026-05")]
        may_returns = pd.to_numeric(may["return"], errors="coerce").dropna()
        return {
            "total_trades": int(len(returns)),
            "total_return": float(returns.sum()),
            "win_rate": float((returns > 0).mean()),
            "profit_factor": self._profit_factor(returns),
            "max_drawdown": self._max_drawdown(returns),
            "bad_entry_rate": self._rate(df, "bad_entry_10d"),
            "unique_codes": int(df["code"].nunique()),
            "top_sector": self._top_counts(df, "sector_name", 3) if "sector_name" in df.columns else "",
            "may_return": float(may_returns.sum()) if not may_returns.empty else 0.0,
            "may_win_rate": float((may_returns > 0).mean()) if not may_returns.empty else None,
            "may_profit_factor": self._profit_factor(may_returns),
            "may_bad_entry_rate": self._rate(may, "bad_entry_10d") if not may.empty else None,
        }

    def _monthly_summary(self, strategy: str, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        df = pd.DataFrame(trades)
        if df.empty:
            return []
        rows = []
        for month, group in df.groupby("month", dropna=False):
            returns = pd.to_numeric(group["return"], errors="coerce").dropna()
            rows.append(
                {
                    "strategy": strategy,
                    "month": str(month),
                    "trade_count": int(len(returns)),
                    "monthly_return": float(returns.sum()),
                    "win_rate": float((returns > 0).mean()) if not returns.empty else None,
                    "profit_factor": self._profit_factor(returns),
                    "bad_entry_rate": self._rate(group, "bad_entry_10d"),
                    "unique_codes": int(group["code"].nunique()),
                    "top_sector": self._top_counts(group, "sector_name", 3) if "sector_name" in group.columns else "",
                }
            )
        return rows

    def _listed_info(self, end_date: str) -> pd.DataFrame:
        info = self.data_loader.load_listed_info(end_date)
        if info.empty:
            info = self.data_loader.load_listed_info(None)
        if info.empty:
            return pd.DataFrame()
        data = info.copy()
        data["code"] = data["code"].astype("string")
        columns = ["code"]
        if "S33Nm" in data.columns:
            data["sector_name"] = data["S33Nm"]
            columns.append("sector_name")
        elif "S17Nm" in data.columns:
            data["sector_name"] = data["S17Nm"]
            columns.append("sector_name")
        if "MktNm" in data.columns:
            data["market"] = data["MktNm"]
            columns.append("market")
        return data[columns].drop_duplicates("code")

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

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        return data

    def _first_index_after(self, prices: pd.DataFrame, signal_date: pd.Timestamp) -> int | None:
        matches = prices.index[prices["date"] > signal_date].tolist()
        return int(matches[0]) if matches else None

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

    def _rate(self, df: pd.DataFrame, column: str) -> float | None:
        if df.empty or column not in df.columns:
            return None
        value = df[column].astype("boolean").mean()
        return None if pd.isna(value) else float(value)

    def _top_counts(self, df: pd.DataFrame, column: str, limit: int) -> str:
        if column not in df.columns:
            return ""
        counts = df[column].dropna().astype(str).value_counts().head(limit)
        return ", ".join(f"{name}:{count}" for name, count in counts.items())

    def _date_texts(self, start_date: str, end_date: str) -> list[str]:
        return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]

    def _month_slug(self, date_text: str) -> str:
        return pd.Timestamp(date_text).strftime("%Y-%m")

    def _table(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "_No rows._"
        columns = list(rows[0].keys())
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
