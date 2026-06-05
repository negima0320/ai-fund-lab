from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import JQUANTS_CACHE_ROOT, ML_DATA_ROOT, ML_LABELS_ROOT, ML_REPORTS_ROOT
from ml.data_loader import JQuantsDataLoader


PREDICTION_DIAGNOSTIC_COLUMNS = [
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "ml_score",
]

LABEL_DIAGNOSTIC_COLUMNS = [
    "future_10d_return",
    "future_max_return_20d",
    "future_swing_success_20d",
    "bad_entry_10d",
]


class WalkForwardDiagnosticsAnalyzer:
    """Analyze monthly walk-forward prediction, label, and top-N trade behavior."""

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

    def analyze(self, walk_forward_json: str | Path, start_date: str, end_date: str) -> dict[str, Any]:
        wf = json.loads(Path(walk_forward_json).read_text(encoding="utf-8"))
        predictions_labels = self._load_predictions_labels(start_date, end_date)
        trades = self._enrich_trades(pd.DataFrame(wf.get("trades", [])), predictions_labels, end_date)
        topix = self._topix_monthly_proxy(start_date, end_date)

        months = [month.strftime("%Y-%m") for month in pd.date_range(start_date, end_date, freq="MS")]
        monthly_prediction_distribution = []
        monthly_label_distribution = []
        monthly_top10_summary = []
        monthly_concentration = []

        for month in months:
            month_joined = predictions_labels[predictions_labels["month"].eq(month)].copy()
            month_trades = trades[trades["month"].eq(month)].copy() if not trades.empty else pd.DataFrame()
            monthly_prediction_distribution.append({"month": month, **self._distribution(month_joined, PREDICTION_DIAGNOSTIC_COLUMNS)})
            monthly_label_distribution.append({"month": month, **self._distribution(month_joined, LABEL_DIAGNOSTIC_COLUMNS)})
            monthly_top10_summary.append({"month": month, **self._top10_summary(month_trades)})
            monthly_concentration.append({"month": month, **self._concentration(month_trades)})

        may_losers = trades[trades["month"].eq("2026-05")].sort_values("return").head(20) if not trades.empty else pd.DataFrame()
        result = {
            "period": {"start_date": start_date, "end_date": end_date},
            "source": str(walk_forward_json),
            "monthly_prediction_distribution": monthly_prediction_distribution,
            "monthly_label_distribution": monthly_label_distribution,
            "monthly_top10_details": self._trade_detail_records(trades),
            "monthly_top10_summary": monthly_top10_summary,
            "monthly_concentration": monthly_concentration,
            "monthly_topix_proxy": topix,
            "losing_trades_2026_05": self._records(may_losers),
            "diagnosis": self._diagnosis(monthly_prediction_distribution, monthly_label_distribution, monthly_top10_summary, monthly_concentration, topix),
        }
        return result

    def save_report(self, result: dict[str, Any]) -> Path:
        period = result["period"]
        path = self.report_root / f"walk_forward_diagnostics_{self._month_slug(period['start_date'])}_to_{self._month_slug(period['end_date'])}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_markdown(result), encoding="utf-8")
        return path

    def save_json(self, result: dict[str, Any]) -> Path:
        period = result["period"]
        path = self.report_root / f"walk_forward_diagnostics_{self._month_slug(period['start_date'])}_to_{self._month_slug(period['end_date'])}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def save_losing_trades_csv(self, result: dict[str, Any]) -> Path:
        path = self.report_root / "walk_forward_losing_trades_2026-05.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.get("losing_trades_2026_05", [])).to_csv(path, index=False)
        return path

    def format_markdown(self, result: dict[str, Any]) -> str:
        period = result["period"]
        return "\n".join(
            [
                "# Walk-Forward Diagnostics",
                "",
                f"- period: {period['start_date']} to {period['end_date']}",
                f"- source: {result['source']}",
                "- note: diagnostics only; no retraining or trading logic changes",
                "",
                "## Diagnosis",
                "",
                "\n".join(f"- {item}" for item in result["diagnosis"]) or "_No diagnosis._",
                "",
                "## Monthly Prediction Distribution",
                "",
                self._table(result["monthly_prediction_distribution"]),
                "",
                "## Monthly Label Distribution",
                "",
                self._table(result["monthly_label_distribution"]),
                "",
                "## Monthly Top10 Trade Summary",
                "",
                self._table(result["monthly_top10_summary"]),
                "",
                "## Monthly Top10 Details",
                "",
                self._table(result["monthly_top10_details"]),
                "",
                "## Monthly Concentration",
                "",
                self._table(result["monthly_concentration"]),
                "",
                "## TOPIX Proxy",
                "",
                self._table(result["monthly_topix_proxy"]),
                "",
                "## 2026-05 Losing Trades Top 20",
                "",
                self._table(result["losing_trades_2026_05"]),
                "",
            ]
        )

    def _load_predictions_labels(self, start_date: str, end_date: str) -> pd.DataFrame:
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
            joined["month"] = joined["date"].dt.strftime("%Y-%m")
            if not info.empty:
                joined = joined.merge(info, on="code", how="left")
            frames.append(joined.dropna(axis=1, how="all"))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _enrich_trades(self, trades: pd.DataFrame, predictions_labels: pd.DataFrame, end_date: str) -> pd.DataFrame:
        if trades.empty:
            return trades
        data = trades.copy()
        data["date"] = pd.to_datetime(data["signal_date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        data["month"] = pd.to_datetime(data["signal_date"], errors="coerce").dt.strftime("%Y-%m")
        data["return"] = pd.to_numeric(data["return"], errors="coerce")
        enrich_columns = [
            "date",
            "code",
            *PREDICTION_DIAGNOSTIC_COLUMNS,
            *LABEL_DIAGNOSTIC_COLUMNS,
            "sector_name",
            "market",
        ]
        available = [column for column in enrich_columns if column in predictions_labels.columns]
        if available:
            data = data.merge(predictions_labels[available], on=["date", "code"], how="left", suffixes=("", "_joined"))
            for column in PREDICTION_DIAGNOSTIC_COLUMNS + LABEL_DIAGNOSTIC_COLUMNS:
                joined = f"{column}_joined"
                if joined in data.columns:
                    if column in data.columns:
                        data[column] = data[column].combine_first(data[joined])
                    else:
                        data[column] = data[joined]
                    data = data.drop(columns=[joined])
        return data

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

    def _distribution(self, df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
        output: dict[str, Any] = {"rows": int(len(df))}
        for column in columns:
            if column not in df.columns:
                continue
            series = df[column]
            if str(series.dtype) in ("boolean", "bool"):
                output[f"{column}_rate"] = self._json_value(series.astype("boolean").mean())
                continue
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                continue
            output[f"{column}_mean"] = float(numeric.mean())
            output[f"{column}_median"] = float(numeric.median())
            output[f"{column}_p10"] = float(numeric.quantile(0.10))
            output[f"{column}_p90"] = float(numeric.quantile(0.90))
        return output

    def _top10_summary(self, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty:
            return {"trade_count": 0}
        returns = pd.to_numeric(trades["return"], errors="coerce").dropna()
        output = {
            "trade_count": int(len(returns)),
            "win_rate": self._json_value((returns > 0).mean()),
            "return_mean": self._json_value(returns.mean()),
            "return_sum": self._json_value(returns.sum()),
            "profit_factor": self._profit_factor(returns),
            "bad_entry_rate": self._rate(trades, "bad_entry_10d"),
            "swing_success_rate": self._rate(trades, "future_swing_success_20d"),
        }
        for column in ["expected_return_10d", "bad_entry_probability_10d", "expected_max_return_20d", "swing_success_probability_20d"]:
            if column in trades.columns:
                output[f"{column}_mean"] = self._json_value(pd.to_numeric(trades[column], errors="coerce").mean())
        return output

    def _concentration(self, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty:
            return {"trade_count": 0, "unique_codes": 0}
        output: dict[str, Any] = {
            "trade_count": int(len(trades)),
            "unique_codes": int(trades["code"].nunique()),
            "top_repeated_codes": self._top_counts(trades, "code", limit=5),
        }
        if "sector_name" in trades.columns and trades["sector_name"].notna().any():
            output["top_sector"] = self._top_counts(trades, "sector_name", limit=3)
        if "market" in trades.columns and trades["market"].notna().any():
            output["top_market"] = self._top_counts(trades, "market", limit=3)
        return output

    def _topix_monthly_proxy(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        topix = self.data_loader.load_topix(start_date, end_date)
        if topix.empty:
            return []
        data = topix.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data = data.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
        data["topix_return_10d"] = data["close"] / data["close"].shift(10) - 1
        data["topix_return_20d"] = data["close"] / data["close"].shift(20) - 1
        data["month"] = data["date"].dt.strftime("%Y-%m")
        rows = []
        for month, group in data.groupby("month", dropna=False):
            rows.append(
                {
                    "month": str(month),
                    "topix_return_10d_mean": self._json_value(group["topix_return_10d"].mean()),
                    "topix_return_20d_mean": self._json_value(group["topix_return_20d"].mean()),
                    "topix_month_return": self._json_value(group["close"].iloc[-1] / group["close"].iloc[0] - 1),
                }
            )
        return rows

    def _diagnosis(
        self,
        prediction_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        top10_rows: list[dict[str, Any]],
        concentration_rows: list[dict[str, Any]],
        topix_rows: list[dict[str, Any]],
    ) -> list[str]:
        by_month = {row["month"]: row for row in top10_rows}
        labels = {row["month"]: row for row in label_rows}
        predictions = {row["month"]: row for row in prediction_rows}
        concentration = {row["month"]: row for row in concentration_rows}
        topix = {row["month"]: row for row in topix_rows}
        may = by_month.get("2026-05", {})
        may_label = labels.get("2026-05", {})
        may_pred = predictions.get("2026-05", {})
        may_conc = concentration.get("2026-05", {})
        may_topix = topix.get("2026-05", {})
        previous = [row for row in top10_rows if row.get("month") != "2026-05" and row.get("trade_count", 0)]
        previous_return_mean = pd.Series([row.get("return_mean") for row in previous], dtype=float).mean() if previous else None
        messages = []
        if may:
            messages.append(
                "2026-05 top10 trades had lower win rate and negative return sum "
                f"(win_rate={may.get('win_rate')}, return_sum={may.get('return_sum')})."
            )
        if previous_return_mean is not None and may.get("return_mean") is not None:
            messages.append(
                "2026-05 trade average return was materially below prior months "
                f"({may.get('return_mean'):.4f} vs prior-month mean {previous_return_mean:.4f})."
            )
        if may_label.get("future_10d_return_mean") is not None:
            messages.append(
                "All-stock realized labels in 2026-05 were weaker: "
                f"future_10d_return_mean={may_label.get('future_10d_return_mean'):.4f}, "
                f"bad_entry_10d_rate={may_label.get('bad_entry_10d_rate')}."
            )
        if may_pred.get("bad_entry_probability_10d_mean") is not None:
            messages.append(
                "2026-05 model risk scores were not enough to prevent weak top10 picks: "
                f"bad_entry_probability_mean={may_pred.get('bad_entry_probability_10d_mean'):.4f}."
            )
        if may_conc.get("unique_codes") is not None:
            messages.append(
                "2026-05 concentration check: "
                f"unique_codes={may_conc.get('unique_codes')}, top_repeated_codes={may_conc.get('top_repeated_codes')}."
            )
        if may_topix.get("topix_month_return") is not None:
            messages.append(
                "TOPIX proxy for 2026-05: "
                f"month_return={may_topix.get('topix_month_return'):.4f}, "
                f"avg_10d={may_topix.get('topix_return_10d_mean'):.4f}, "
                f"avg_20d={may_topix.get('topix_return_20d_mean'):.4f}."
            )
        return messages

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        return data

    def _records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        columns = [
            "code",
            "entry_date",
            "exit_date",
            "return",
            "expected_return_10d",
            "bad_entry_probability_10d",
            "expected_max_return_20d",
            "swing_success_probability_20d",
            "future_10d_return",
            "bad_entry_10d",
            "sector_name",
            "market",
        ]
        available = [column for column in columns if column in df.columns]
        return [{key: self._json_value(value) for key, value in row.items()} for row in df[available].to_dict("records")]

    def _trade_detail_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        data = df.copy()
        data["ranking_score"] = pd.to_numeric(data.get("expected_return_10d"), errors="coerce")
        columns = [
            "month",
            "signal_date",
            "code",
            "ranking_score",
            "expected_return_10d",
            "future_10d_return",
            "bad_entry_10d",
            "sector_name",
            "market",
            "return",
        ]
        available = [column for column in columns if column in data.columns]
        data = data.sort_values(["month", "signal_date", "ranking_score"], ascending=[True, True, False])
        return [{key: self._json_value(value) for key, value in row.items()} for row in data[available].to_dict("records")]

    def _top_counts(self, df: pd.DataFrame, column: str, limit: int) -> str:
        counts = df[column].dropna().astype(str).value_counts().head(limit)
        return ", ".join(f"{name}:{count}" for name, count in counts.items())

    def _profit_factor(self, returns: pd.Series) -> float | None:
        gains = returns[returns > 0].sum()
        losses = returns[returns < 0].sum()
        if losses == 0:
            return None
        return float(gains / abs(losses))

    def _rate(self, df: pd.DataFrame, column: str) -> float | None:
        if column not in df.columns:
            return None
        series = df[column].astype("boolean")
        value = series.mean()
        return None if pd.isna(value) else float(value)

    def _date_texts(self, start_date: str, end_date: str) -> list[str]:
        return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]

    def _month_slug(self, date_text: str) -> str:
        return pd.Timestamp(date_text).strftime("%Y-%m")

    def _table(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "_No rows._"
        columns = sorted({key for row in rows for key in row})
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
