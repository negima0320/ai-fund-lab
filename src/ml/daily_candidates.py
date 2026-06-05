from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import JQUANTS_CACHE_ROOT, ML_FEATURES_ROOT, ML_PREDICTIONS_ROOT, ML_REPORTS_ROOT
from ml.data_loader import JQuantsDataLoader


DAILY_CANDIDATE_COLUMNS = [
    "rank",
    "candidate_status",
    "date",
    "code",
    "name",
    "market",
    "sector_name",
    "close",
    "turnover_value",
    "risk_adjusted_score",
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "entry_risk_label",
    "ml_score",
    "reason",
]


class DailyAICandidateExporter:
    """Export report-only daily candidates selected by AI predictions."""

    def __init__(
        self,
        prediction_root: str | Path = ML_PREDICTIONS_ROOT,
        feature_root: str | Path = ML_FEATURES_ROOT,
        report_root: str | Path = ML_REPORTS_ROOT / "daily_candidates",
        cache_root: str | Path = JQUANTS_CACHE_ROOT,
    ) -> None:
        self.prediction_root = Path(prediction_root)
        self.feature_root = Path(feature_root)
        self.report_root = Path(report_root)
        self.data_loader = JQuantsDataLoader(cache_root)

    def build_candidates(
        self,
        target_date: str,
        top_n: int = 10,
        min_turnover_value: float = 50_000_000,
        max_bad_entry_probability: float | None = None,
    ) -> pd.DataFrame:
        predictions = self._read_required_parquet(self.prediction_root / f"predictions_{target_date}.parquet", "predictions")
        features = self._read_required_parquet(self.feature_root / f"features_{target_date}.parquet", "features")

        data = self._normalize(predictions).merge(
            self._normalize(features)[["date", "code", *self._feature_columns(features)]],
            on=["date", "code"],
            how="left",
        )
        info = self._listed_info(target_date)
        if not info.empty:
            data = data.merge(info, on="code", how="left")

        for column in [
            "close",
            "turnover_value",
            "expected_return_10d",
            "expected_max_return_20d",
            "swing_success_probability_20d",
            "bad_entry_probability_10d",
            "ml_score",
        ]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")

        required = ["expected_return_10d", "bad_entry_probability_10d", "turnover_value"]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"daily candidate input is missing required columns: {', '.join(missing)}")

        data["risk_adjusted_score"] = data["expected_return_10d"] - 0.5 * data["bad_entry_probability_10d"]

        candidates = data[data["turnover_value"] >= min_turnover_value].copy()
        if max_bad_entry_probability is not None:
            candidates = candidates[candidates["bad_entry_probability_10d"] < max_bad_entry_probability].copy()
        candidates = (
            candidates.dropna(subset=["risk_adjusted_score"])
            .sort_values("risk_adjusted_score", ascending=False)
            .head(top_n)
        )
        candidates = candidates.reset_index(drop=True)
        candidates["rank"] = range(1, len(candidates) + 1)
        candidates = self._add_previous_comparison(candidates, target_date)
        candidates["reason"] = candidates.apply(
            lambda row: (
                f"risk_adjusted_score={row['risk_adjusted_score']:.4f}が高く、"
                f"turnover_value={row['turnover_value']:.0f}が流動性条件を満たすため。"
            ),
            axis=1,
        )
        for column in ["name", "market", "sector_name", "entry_risk_label"]:
            if column not in candidates.columns:
                candidates[column] = pd.NA
        return candidates[[column for column in DAILY_CANDIDATE_COLUMNS if column in candidates.columns]]

    def save_csv(self, df: pd.DataFrame, target_date: str) -> Path:
        path = self.report_root / f"{target_date}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def save_markdown(self, df: pd.DataFrame, target_date: str) -> Path:
        path = self.report_root / f"{target_date}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_markdown(df, target_date), encoding="utf-8")
        return path

    def format_markdown(self, df: pd.DataFrame, target_date: str) -> str:
        comparison = self._comparison_summary(df, target_date)
        return "\n".join(
            [
                "# Daily AI Candidates",
                "",
                f"- date: {target_date}",
                "- ranking: risk_adjusted_return",
                "- score: expected_return_10d - 0.5 * bad_entry_probability_10d",
                "- liquidity: turnover_value >= 50,000,000",
                "- exit assumption: close_10d",
                "- note: report-only; no orders are placed",
                "",
                "## Previous Day Comparison",
                "",
                comparison,
                "",
                "## Candidates",
                "",
                self._table(df.to_dict("records"), DAILY_CANDIDATE_COLUMNS),
                "",
            ]
        )

    def _read_required_parquet(self, path: Path, label: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{label} parquet not found: {path}")
        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"{label} parquet is empty: {path}")
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        return data

    def _feature_columns(self, features: pd.DataFrame) -> list[str]:
        return [column for column in ["close", "turnover_value"] if column in features.columns]

    def _add_previous_comparison(self, candidates: pd.DataFrame, target_date: str) -> pd.DataFrame:
        output = candidates.copy()
        previous = self._previous_candidates(target_date)
        previous_codes = set(previous["code"].astype(str)) if not previous.empty and "code" in previous.columns else set()
        output["candidate_status"] = output["code"].astype(str).map(
            lambda code: "continued" if code in previous_codes else "new"
        )
        return output

    def _previous_candidates(self, target_date: str) -> pd.DataFrame:
        target = pd.Timestamp(target_date)
        candidates = []
        for path in self.report_root.glob("*.csv"):
            date_text = path.stem.replace("ai_candidates_", "")
            try:
                date = pd.Timestamp(date_text)
            except ValueError:
                continue
            if date < target:
                candidates.append((date, path))
        if not candidates:
            return pd.DataFrame()
        _, path = max(candidates, key=lambda item: item[0])
        try:
            df = pd.read_csv(path, dtype={"code": "string"})
        except Exception:
            return pd.DataFrame()
        return df

    def _comparison_summary(self, df: pd.DataFrame, target_date: str) -> str:
        previous = self._previous_candidates(target_date)
        current_codes = set(df["code"].astype(str)) if not df.empty and "code" in df.columns else set()
        previous_codes = set(previous["code"].astype(str)) if not previous.empty and "code" in previous.columns else set()
        new_codes = sorted(current_codes - previous_codes)
        continued_codes = sorted(current_codes & previous_codes)
        dropped_codes = sorted(previous_codes - current_codes)
        return "\n".join(
            [
                f"- 新規(new): {len(new_codes)}" + (f" ({', '.join(new_codes)})" if new_codes else ""),
                f"- 継続(continued): {len(continued_codes)}" + (f" ({', '.join(continued_codes)})" if continued_codes else ""),
                f"- 脱落(dropped): {len(dropped_codes)}" + (f" ({', '.join(dropped_codes)})" if dropped_codes else ""),
            ]
        )

    def _listed_info(self, target_date: str) -> pd.DataFrame:
        info = self.data_loader.load_listed_info(target_date)
        if info.empty:
            info = self.data_loader.load_listed_info(None)
        if info.empty:
            return pd.DataFrame()
        data = info.copy()
        data["code"] = data["code"].astype("string")
        columns = ["code"]
        if "CoName" in data.columns:
            data["name"] = data["CoName"]
            columns.append("name")
        if "MktNm" in data.columns:
            data["market"] = data["MktNm"]
            columns.append("market")
        if "S33Nm" in data.columns:
            data["sector_name"] = data["S33Nm"]
            columns.append("sector_name")
        elif "S17Nm" in data.columns:
            data["sector_name"] = data["S17Nm"]
            columns.append("sector_name")
        return data[columns].drop_duplicates("code")

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No candidates._"
        available = [column for column in columns if any(column in row for row in rows)]
        header = "| " + " | ".join(available) + " |"
        separator = "| " + " | ".join(["---"] * len(available)) + " |"
        body = ["| " + " | ".join(self._fmt(row.get(column)) for column in available) + " |" for row in rows]
        return "\n".join([header, separator, *body])

    def _fmt(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|")
