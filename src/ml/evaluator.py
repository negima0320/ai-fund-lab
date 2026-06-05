from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import ML_LABELS_ROOT, ML_PREDICTIONS_ROOT, ML_REPORTS_ROOT


class PredictionEvaluator:
    """Evaluate one day of ML predictions against generated labels."""

    def __init__(
        self,
        predictions_root: str | Path = ML_PREDICTIONS_ROOT,
        labels_root: str | Path = ML_LABELS_ROOT,
        report_root: str | Path = ML_REPORTS_ROOT,
    ) -> None:
        self.predictions_root = Path(predictions_root)
        self.labels_root = Path(labels_root)
        self.report_root = Path(report_root)

    def evaluate_daily(self, target_date: str, top_n: int = 10) -> dict[str, Any]:
        predictions = self._read_predictions(target_date)
        labels = self._read_labels(target_date)
        joined = self.join_predictions_labels(predictions, labels)
        return self.evaluate_joined(joined, target_date, top_n=top_n)

    def join_predictions_labels(self, predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        left = self._normalize_keys(predictions)
        right = self._normalize_keys(labels)
        return left.merge(right, on=["date", "code"], how="inner")

    def evaluate_joined(self, df: pd.DataFrame, target_date: str, top_n: int = 10) -> dict[str, Any]:
        data = df.copy()
        for column in [
            "ml_score",
            "future_5d_return",
            "future_10d_return",
            "future_max_return_10d",
            "future_max_return_20d",
            "expected_return_10d",
            "expected_max_return_10d",
            "expected_max_return_20d",
            "bad_entry_probability_10d",
            "swing_success_probability_20d",
        ]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        for column in ["upside_10d", "bad_entry_10d", "future_swing_success_20d"]:
            if column in data.columns:
                data[column] = data[column].astype("boolean")

        top = data.sort_values("ml_score", ascending=False).head(top_n) if "ml_score" in data.columns else data.head(0)
        return {
            "target_date": target_date,
            "joined_rows": int(len(data)),
            "top_n": int(top_n),
            "top_n_summary": self._summary(top),
            "risk_label_summary": self._risk_label_summary(data),
            "bad_entry_probability_bands": self._bad_probability_band_summary(data),
            "swing_success_probability_bands": self._swing_probability_band_summary(data),
            "expected_vs_future_10d_corr": self._correlation(data, "expected_return_10d", "future_10d_return"),
            "expected_max_vs_future_max_10d_corr": self._correlation(data, "expected_max_return_10d", "future_max_return_10d"),
            "expected_max_vs_future_max_20d_corr": self._correlation(data, "expected_max_return_20d", "future_max_return_20d"),
            "swing_probability_vs_success_20d_corr": self._correlation(
                data,
                "swing_success_probability_20d",
                "future_swing_success_20d",
            ),
            "top_rows": self._top_rows(top),
        }

    def save_report(self, evaluation: dict[str, Any], target_date: str) -> Path:
        path = self.report_root / f"evaluation_{target_date}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_markdown(evaluation), encoding="utf-8")
        return path

    def format_markdown(self, evaluation: dict[str, Any]) -> str:
        lines = [
            f"# ML Prediction Evaluation {evaluation['target_date']}",
            "",
            "This is a one-day smoke evaluation. Do not treat it as model quality evidence.",
            "",
            "## Overview",
            "",
            f"- joined_rows: {evaluation['joined_rows']}",
            f"- top_n: {evaluation['top_n']}",
            f"- expected_vs_future_10d_corr: {self._fmt(evaluation['expected_vs_future_10d_corr'])}",
            f"- expected_max_vs_future_max_10d_corr: {self._fmt(evaluation['expected_max_vs_future_max_10d_corr'])}",
            f"- expected_max_vs_future_max_20d_corr: {self._fmt(evaluation['expected_max_vs_future_max_20d_corr'])}",
            f"- swing_probability_vs_success_20d_corr: {self._fmt(evaluation['swing_probability_vs_success_20d_corr'])}",
            "",
            "## Top N By ML Score",
            "",
            self._summary_table(evaluation["top_n_summary"]),
            "",
            "## Entry Risk Label Summary",
            "",
            self._group_table(evaluation["risk_label_summary"], "entry_risk_label"),
            "",
            "## Bad Entry Probability Bands",
            "",
            self._group_table(evaluation["bad_entry_probability_bands"], "band"),
            "",
            "## Swing Success Probability Bands",
            "",
            self._group_table(evaluation["swing_success_probability_bands"], "band"),
            "",
            "## Top Rows",
            "",
            self._top_rows_table(evaluation["top_rows"]),
            "",
        ]
        return "\n".join(lines)

    def _read_predictions(self, target_date: str) -> pd.DataFrame:
        return pd.read_parquet(self.predictions_root / f"predictions_{target_date}.parquet")

    def _read_labels(self, target_date: str) -> pd.DataFrame:
        return pd.read_parquet(self.labels_root / f"labels_{target_date}.parquet")

    def _normalize_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        return data

    def _summary(self, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {
                "count": 0,
                "future_5d_return_mean": None,
                "future_10d_return_mean": None,
                "upside_10d_rate": None,
                "bad_entry_10d_rate": None,
                "future_max_return_10d_mean": None,
                "future_max_return_20d_mean": None,
                "future_swing_success_20d_rate": None,
            }
        return {
            "count": int(len(df)),
            "future_5d_return_mean": self._mean(df, "future_5d_return"),
            "future_10d_return_mean": self._mean(df, "future_10d_return"),
            "upside_10d_rate": self._mean(df, "upside_10d"),
            "bad_entry_10d_rate": self._mean(df, "bad_entry_10d"),
            "future_max_return_10d_mean": self._mean(df, "future_max_return_10d"),
            "future_max_return_20d_mean": self._mean(df, "future_max_return_20d"),
            "future_swing_success_20d_rate": self._mean(df, "future_swing_success_20d"),
        }

    def _risk_label_summary(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty or "entry_risk_label" not in df.columns:
            return []
        rows = []
        for label, group in df.groupby("entry_risk_label", dropna=False):
            item = self._summary(group)
            item["entry_risk_label"] = str(label)
            rows.append(item)
        return rows

    def _bad_probability_band_summary(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty or "bad_entry_probability_10d" not in df.columns:
            return []
        bands = [
            ("0.0-0.25", 0.0, 0.25),
            ("0.25-0.40", 0.25, 0.40),
            ("0.40-1.0", 0.40, 1.0),
        ]
        rows = []
        for label, lower, upper in bands:
            if upper == 1.0:
                group = df[(df["bad_entry_probability_10d"] >= lower) & (df["bad_entry_probability_10d"] <= upper)]
            else:
                group = df[(df["bad_entry_probability_10d"] >= lower) & (df["bad_entry_probability_10d"] < upper)]
            item = self._summary(group)
            item["band"] = label
            rows.append(item)
        return rows

    def _swing_probability_band_summary(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty or "swing_success_probability_20d" not in df.columns:
            return []
        bands = [
            ("0.0-0.25", 0.0, 0.25),
            ("0.25-0.50", 0.25, 0.50),
            ("0.50-0.75", 0.50, 0.75),
            ("0.75-1.0", 0.75, 1.0),
        ]
        rows = []
        for label, lower, upper in bands:
            if upper == 1.0:
                group = df[(df["swing_success_probability_20d"] >= lower) & (df["swing_success_probability_20d"] <= upper)]
            else:
                group = df[(df["swing_success_probability_20d"] >= lower) & (df["swing_success_probability_20d"] < upper)]
            item = self._summary(group)
            item["band"] = label
            rows.append(item)
        return rows

    def _top_rows(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        columns = [
            "date",
            "code",
            "ml_score",
            "expected_return_10d",
            "future_10d_return",
            "expected_max_return_10d",
            "future_max_return_10d",
            "expected_max_return_20d",
            "future_max_return_20d",
            "swing_success_probability_20d",
            "future_swing_success_20d",
            "upside_probability_10d",
            "bad_entry_probability_10d",
            "entry_risk_label",
            "upside_10d",
            "bad_entry_10d",
        ]
        available = [column for column in columns if column in df.columns]
        rows = []
        for row in df[available].to_dict("records"):
            rows.append({key: self._json_value(value) for key, value in row.items()})
        return rows

    def _correlation(self, df: pd.DataFrame, left: str, right: str) -> float | None:
        if df.empty or left not in df.columns or right not in df.columns:
            return None
        pairs = df[[left, right]].copy()
        pairs[left] = pd.to_numeric(pairs[left], errors="coerce")
        pairs[right] = pd.to_numeric(pairs[right], errors="coerce")
        pairs = pairs.dropna()
        value = pairs[left].corr(pairs[right])
        return None if pd.isna(value) else float(value)

    def _mean(self, df: pd.DataFrame, column: str) -> float | None:
        if column not in df.columns:
            return None
        value = df[column].mean()
        return None if pd.isna(value) else float(value)

    def _summary_table(self, summary: dict[str, Any]) -> str:
        rows = [summary]
        return self._markdown_table(rows, self._summary_columns())

    def _group_table(self, rows: list[dict[str, Any]], group_column: str) -> str:
        columns = [group_column, *self._summary_columns()]
        return self._markdown_table(rows, columns) if rows else "_No rows._"

    def _summary_columns(self) -> list[str]:
        return [
            "count",
            "future_5d_return_mean",
            "future_10d_return_mean",
            "future_max_return_10d_mean",
            "future_max_return_20d_mean",
            "upside_10d_rate",
            "bad_entry_10d_rate",
            "future_swing_success_20d_rate",
        ]

    def _top_rows_table(self, rows: list[dict[str, Any]]) -> str:
        columns = [
            "code",
            "ml_score",
            "expected_return_10d",
            "future_10d_return",
            "expected_max_return_20d",
            "future_max_return_20d",
            "swing_success_probability_20d",
            "future_swing_success_20d",
            "upside_probability_10d",
            "bad_entry_probability_10d",
            "entry_risk_label",
            "upside_10d",
            "bad_entry_10d",
        ]
        return self._markdown_table(rows, columns) if rows else "_No rows._"

    def _markdown_table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = [
            "| " + " | ".join(self._fmt(row.get(column)) for column in columns) + " |"
            for row in rows
        ]
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
