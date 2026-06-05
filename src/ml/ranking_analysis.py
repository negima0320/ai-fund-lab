from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import ML_LABELS_ROOT, ML_PREDICTIONS_ROOT, ML_REPORTS_ROOT, ROOT


RANKING_SPECS = {
    "expected_max_return_20d_top10": "expected_max_return_20d",
    "swing_success_probability_20d_top10": "swing_success_probability_20d",
    "ml_score_top10": "ml_score",
    "expected_return_10d_top10": "expected_return_10d",
}


class MLRankingAnalyzer:
    """Analyze all-stock ML top-N rankings against realized labels."""

    def __init__(
        self,
        predictions_root: str | Path = ML_PREDICTIONS_ROOT,
        labels_root: str | Path = ML_LABELS_ROOT,
        report_root: str | Path = ML_REPORTS_ROOT,
        root: str | Path = ROOT,
    ) -> None:
        self.predictions_root = Path(predictions_root)
        self.labels_root = Path(labels_root)
        self.report_root = Path(report_root)
        self.root = Path(root)

    def analyze(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 10,
        profile: str | None = None,
    ) -> dict[str, Any]:
        joined_frames = []
        skipped_dates = []
        for date_text in self._date_texts(start_date, end_date):
            prediction_path = self.predictions_root / f"predictions_{date_text}.parquet"
            label_path = self.labels_root / f"labels_{date_text}.parquet"
            if not prediction_path.exists() or not label_path.exists():
                skipped_dates.append({"date": date_text, "reason": "prediction or label missing"})
                continue
            predictions = pd.read_parquet(prediction_path)
            labels = pd.read_parquet(label_path)
            if predictions.empty or labels.empty:
                skipped_dates.append({"date": date_text, "reason": "prediction or label empty"})
                continue
            joined = self._join_predictions_labels(predictions, labels)
            if joined.empty:
                skipped_dates.append({"date": date_text, "reason": "prediction/label join empty"})
                continue
            joined_frames.append(joined)

        joined_all = pd.concat(joined_frames, ignore_index=True) if joined_frames else pd.DataFrame()
        trade_keys, trade_count, trades_source = self._load_trade_keys(profile, start_date, end_date)
        ranking_details = self._ranking_details(joined_all, top_n=top_n, trade_keys=trade_keys)
        ranking_summary = self._ranking_summary(ranking_details)
        monthly_summary = self._monthly_summary(ranking_details)
        baseline_summary = self._summary(joined_all)
        overlap_summary = self._overlap_summary(ranking_details, trade_keys, trade_count)

        return {
            "period": {"start_date": start_date, "end_date": end_date},
            "top_n": int(top_n),
            "profile": profile,
            "trades_source": trades_source,
            "processed_dates": sorted(joined_all["date"].dt.strftime("%Y-%m-%d").unique().tolist()) if not joined_all.empty else [],
            "skipped_dates": skipped_dates,
            "baseline_all_stocks": baseline_summary,
            "ranking_summary": ranking_summary,
            "monthly_summary": monthly_summary,
            "overlap_summary": overlap_summary,
            "ranking_details": self._records(ranking_details),
        }

    def save_report(self, analysis: dict[str, Any]) -> Path:
        period = analysis["period"]
        path = self.report_root / f"ml_ranking_analysis_{period['start_date']}_to_{period['end_date']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_markdown(analysis), encoding="utf-8")
        return path

    def save_json(self, analysis: dict[str, Any]) -> Path:
        period = analysis["period"]
        path = self.report_root / f"ml_ranking_analysis_{period['start_date']}_to_{period['end_date']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def save_details_csv(self, analysis: dict[str, Any]) -> Path:
        period = analysis["period"]
        path = self.report_root / f"ml_ranking_details_{period['start_date']}_to_{period['end_date']}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(analysis.get("ranking_details", [])).to_csv(path, index=False)
        return path

    def format_markdown(self, analysis: dict[str, Any]) -> str:
        period = analysis["period"]
        lines = [
            "# ML Ranking Analysis",
            "",
            f"- period: {period['start_date']} to {period['end_date']}",
            f"- top_n: {analysis['top_n']}",
            f"- profile: {analysis.get('profile') or ''}",
            f"- trades_source: {analysis.get('trades_source') or ''}",
            f"- processed_dates: {len(analysis.get('processed_dates', []))}",
            f"- skipped_dates: {len(analysis.get('skipped_dates', []))}",
            "",
            "## Baseline All Stocks",
            "",
            self._table([analysis["baseline_all_stocks"]], self._summary_columns()),
            "",
            "## Ranking Comparison",
            "",
            self._table(analysis["ranking_summary"], ["ranking", *self._summary_columns(), "date_count"]),
            "",
            "## Existing Strategy Overlap",
            "",
            self._table(
                analysis["overlap_summary"],
                [
                    "ranking",
                    "ranked_rows",
                    "bought_count",
                    "not_bought_count",
                    "ranked_bought_rate",
                    "existing_trade_count",
                    "existing_trades_in_topn",
                    "existing_trade_topn_rate",
                ],
            ),
            "",
            "## Monthly Summary",
            "",
            self._table(
                analysis["monthly_summary"],
                ["ranking", "month", *self._summary_columns(), "date_count"],
            ),
            "",
        ]
        return "\n".join(lines)

    def _join_predictions_labels(self, predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        left = self._normalize_keys(predictions)
        right = self._normalize_keys(labels)
        data = left.merge(right, on=["date", "code"], how="inner")
        for column in [
            "expected_max_return_20d",
            "swing_success_probability_20d",
            "ml_score",
            "expected_return_10d",
            "future_10d_return",
            "future_max_return_20d",
            "bad_entry_probability_10d",
        ]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        for column in ["future_swing_success_20d", "bad_entry_10d"]:
            if column in data.columns:
                data[column] = data[column].astype("boolean")
        return data

    def _ranking_details(self, joined: pd.DataFrame, top_n: int, trade_keys: set[tuple[str, str]]) -> pd.DataFrame:
        if joined.empty:
            return pd.DataFrame()
        rows = []
        for date, group in joined.groupby("date", dropna=False):
            for ranking, score_column in RANKING_SPECS.items():
                if score_column not in group.columns:
                    continue
                ranked = group.dropna(subset=[score_column]).sort_values(score_column, ascending=False).head(top_n).copy()
                ranked["ranking"] = ranking
                ranked["rank"] = range(1, len(ranked) + 1)
                rows.append(ranked)
        if not rows:
            return pd.DataFrame()
        details = pd.concat(rows, ignore_index=True)
        details["date_text"] = details["date"].dt.strftime("%Y-%m-%d")
        details["bought_by_profile"] = [
            (date_text, str(code)) in trade_keys
            for date_text, code in zip(details["date_text"], details["code"].astype(str))
        ]
        details["month"] = details["date"].dt.strftime("%Y-%m")
        return details

    def _ranking_summary(self, details: pd.DataFrame) -> list[dict[str, Any]]:
        if details.empty:
            return []
        rows = []
        for ranking, group in details.groupby("ranking", dropna=False):
            item = self._summary(group)
            item["ranking"] = str(ranking)
            item["date_count"] = int(group["date"].nunique())
            rows.append(item)
        return sorted(rows, key=lambda row: row["ranking"])

    def _monthly_summary(self, details: pd.DataFrame) -> list[dict[str, Any]]:
        if details.empty:
            return []
        rows = []
        for (ranking, month), group in details.groupby(["ranking", "month"], dropna=False):
            item = self._summary(group)
            item["ranking"] = str(ranking)
            item["month"] = str(month)
            item["date_count"] = int(group["date"].nunique())
            rows.append(item)
        return sorted(rows, key=lambda row: (row["ranking"], row["month"]))

    def _overlap_summary(
        self,
        details: pd.DataFrame,
        trade_keys: set[tuple[str, str]],
        trade_count: int,
    ) -> list[dict[str, Any]]:
        if details.empty:
            return []
        rows = []
        for ranking, group in details.groupby("ranking", dropna=False):
            ranked_keys = set(zip(group["date_text"], group["code"].astype(str)))
            bought_count = int(group["bought_by_profile"].sum())
            unique_overlap = len(ranked_keys & trade_keys)
            rows.append(
                {
                    "ranking": str(ranking),
                    "ranked_rows": int(len(group)),
                    "bought_count": bought_count,
                    "not_bought_count": int(len(group) - bought_count),
                    "ranked_bought_rate": self._ratio(bought_count, len(group)),
                    "existing_trade_count": int(trade_count),
                    "existing_trades_in_topn": int(unique_overlap),
                    "existing_trade_topn_rate": self._ratio(unique_overlap, trade_count),
                }
            )
        return sorted(rows, key=lambda row: row["ranking"])

    def _summary(self, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {
                "count": 0,
                "future_10d_return_mean": None,
                "future_max_return_20d_mean": None,
                "future_swing_success_20d_rate": None,
                "bad_entry_10d_rate": None,
            }
        return {
            "count": int(len(df)),
            "future_10d_return_mean": self._mean(df, "future_10d_return"),
            "future_max_return_20d_mean": self._mean(df, "future_max_return_20d"),
            "future_swing_success_20d_rate": self._mean(df, "future_swing_success_20d"),
            "bad_entry_10d_rate": self._mean(df, "bad_entry_10d"),
        }

    def _load_trade_keys(self, profile: str | None, start_date: str, end_date: str) -> tuple[set[tuple[str, str]], int, str | None]:
        if not profile:
            return set(), 0, None
        trades_path = self._find_trades_csv(profile, start_date, end_date)
        rows = self._read_csv_rows(trades_path)
        keys = set()
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        for row in rows:
            if str(row.get("action", "")).upper() != "SELL":
                continue
            date_text = row.get("signal_date") or row.get("date") or row.get("entry_date")
            code = row.get("code")
            if not date_text or not code:
                continue
            date = pd.to_datetime(date_text, errors="coerce")
            if pd.isna(date) or date < start or date > end:
                continue
            keys.add((pd.Timestamp(date).strftime("%Y-%m-%d"), str(code)))
        return keys, len(keys), self._relative_path(trades_path)

    def _find_trades_csv(self, profile: str, start_date: str, end_date: str) -> Path:
        profile_root = self.root / "logs" / "backtests" / profile
        exact = profile_root / f"{start_date}_to_{end_date}" / "trades.csv"
        if exact.exists():
            return exact
        if not profile_root.exists():
            raise FileNotFoundError(f"backtest profile directory not found: {profile_root}")
        candidates = []
        target_start = pd.Timestamp(start_date)
        target_end = pd.Timestamp(end_date)
        for path in profile_root.glob("*_to_*/trades.csv"):
            period = path.parent.name.split("_to_")
            if len(period) != 2:
                continue
            try:
                period_start = pd.Timestamp(period[0])
                period_end = pd.Timestamp(period[1])
            except ValueError:
                continue
            if period_start <= target_start and target_end <= period_end:
                candidates.append((period_end - period_start, path))
        if candidates:
            return sorted(candidates, key=lambda item: item[0])[0][1]
        raise FileNotFoundError(f"trades.csv not found for {profile} covering {start_date} to {end_date}")

    def _normalize_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        return data

    def _records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        columns = [
            "ranking",
            "date_text",
            "rank",
            "code",
            "expected_max_return_20d",
            "swing_success_probability_20d",
            "ml_score",
            "expected_return_10d",
            "future_10d_return",
            "future_max_return_20d",
            "future_swing_success_20d",
            "bad_entry_10d",
            "bought_by_profile",
        ]
        available = [column for column in columns if column in df.columns]
        return [
            {key: self._json_value(value) for key, value in row.items()}
            for row in df[available].to_dict("records")
        ]

    def _date_texts(self, start_date: str, end_date: str) -> list[str]:
        return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]

    def _read_csv_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    def _mean(self, df: pd.DataFrame, column: str) -> float | None:
        if column not in df.columns:
            return None
        value = df[column].mean()
        return None if pd.isna(value) else float(value)

    def _ratio(self, numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def _summary_columns(self) -> list[str]:
        return [
            "count",
            "future_10d_return_mean",
            "future_max_return_20d_mean",
            "future_swing_success_20d_rate",
            "bad_entry_10d_rate",
        ]

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

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)
