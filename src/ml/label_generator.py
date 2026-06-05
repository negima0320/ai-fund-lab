from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.config import (
    BAD_ENTRY_THRESHOLD,
    LABEL_COLUMNS,
    LABEL_LOOKAHEAD_DAYS,
    ML_LABELS_ROOT,
    SWING_SUCCESS_THRESHOLD,
    UPSIDE_THRESHOLD,
)
from ml.data_loader import JQuantsDataLoader


class LabelGenerator:
    """Generate supervised labels from cached J-Quants prices."""

    def __init__(
        self,
        data_loader: JQuantsDataLoader | None = None,
        label_root: str | Path = ML_LABELS_ROOT,
        lookahead_days: int = LABEL_LOOKAHEAD_DAYS,
        upside_threshold: float = UPSIDE_THRESHOLD,
        bad_entry_threshold: float = BAD_ENTRY_THRESHOLD,
        swing_success_threshold: float = SWING_SUCCESS_THRESHOLD,
    ) -> None:
        self.data_loader = data_loader or JQuantsDataLoader()
        self.label_root = Path(label_root)
        self.lookahead_days = lookahead_days
        self.upside_threshold = upside_threshold
        self.bad_entry_threshold = bad_entry_threshold
        self.swing_success_threshold = swing_success_threshold

    def generate_labels(self, target_date: str) -> pd.DataFrame:
        target = pd.Timestamp(target_date)
        end_date = target + pd.Timedelta(days=self.lookahead_days * 3)
        prices = self.data_loader.load_prices(target.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if prices.empty:
            return pd.DataFrame(columns=LABEL_COLUMNS)

        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        for column in ["open", "high", "low", "close"]:
            if column in prices.columns:
                prices[column] = pd.to_numeric(prices[column], errors="coerce")
        prices = prices.dropna(subset=["date", "code", "open", "high", "low", "close"])
        prices = prices.sort_values(["code", "date"]).reset_index(drop=True)
        if prices.empty:
            return pd.DataFrame(columns=LABEL_COLUMNS)

        business_days = self._business_days(target, end_date, prices)
        if len(business_days) < 11 or business_days[0] != target:
            return pd.DataFrame(columns=LABEL_COLUMNS)
        entry_date = business_days[1]
        future_5d_date = business_days[5]
        future_10d_date = business_days[10]
        window_10d_dates = set(business_days[1:11])
        has_20d_window = len(business_days) >= 21
        window_20d_dates = set(business_days[1:21]) if has_20d_window else set()

        entry = prices[prices["date"] == entry_date][["code", "open"]].rename(columns={"open": "entry_price"})
        future_5d = prices[prices["date"] == future_5d_date][["code", "close"]].rename(columns={"close": "future_5d_close"})
        future_10d = prices[prices["date"] == future_10d_date][["code", "close"]].rename(columns={"close": "future_10d_close"})
        window_10d = prices[prices["date"].isin(window_10d_dates)]
        extremes = window_10d.groupby("code", as_index=False).agg(
            max_high_10d=("high", "max"),
            min_low_10d=("low", "min"),
        )

        labels = entry.merge(future_5d, on="code", how="inner")
        labels = labels.merge(future_10d, on="code", how="inner")
        labels = labels.merge(extremes, on="code", how="inner")
        if has_20d_window:
            window_20d = prices[prices["date"].isin(window_20d_dates)]
            extremes_20d = window_20d.groupby("code", as_index=False).agg(max_high_20d=("high", "max"))
            labels = labels.merge(extremes_20d, on="code", how="left")
        else:
            labels["max_high_20d"] = pd.NA
        labels = labels.dropna(subset=["entry_price", "future_5d_close", "future_10d_close"])
        if labels.empty:
            return pd.DataFrame(columns=LABEL_COLUMNS)

        labels["date"] = target
        labels["future_5d_return"] = labels["future_5d_close"] / labels["entry_price"] - 1
        labels["future_10d_return"] = labels["future_10d_close"] / labels["entry_price"] - 1
        labels["upside_10d"] = labels["max_high_10d"] / labels["entry_price"] - 1 >= self.upside_threshold
        labels["bad_entry_10d"] = labels["min_low_10d"] / labels["entry_price"] - 1 <= self.bad_entry_threshold
        labels["future_max_return_10d"] = labels["max_high_10d"] / labels["entry_price"] - 1
        labels["future_max_return_20d"] = labels["max_high_20d"] / labels["entry_price"] - 1
        labels["future_swing_success_20d"] = (labels["future_max_return_20d"] >= self.swing_success_threshold).astype("boolean")
        labels.loc[labels["future_max_return_20d"].isna(), "future_swing_success_20d"] = pd.NA
        return labels[LABEL_COLUMNS].sort_values("code").reset_index(drop=True)

    def save_labels(self, df: pd.DataFrame, target_date: str) -> Path:
        path = self.label_root / f"labels_{target_date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def update_available_labels(self, as_of_date: str) -> list[Path]:
        as_of = pd.Timestamp(as_of_date)
        start_date = as_of - pd.Timedelta(days=self.lookahead_days * 3)
        prices = self.data_loader.load_prices(start_date.strftime("%Y-%m-%d"), as_of.strftime("%Y-%m-%d"))
        if prices.empty:
            return []
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        business_days = self._business_days(start_date, as_of, prices)
        eligible = [day for day in business_days if day <= as_of]
        if len(eligible) < 11:
            return []
        offset = min(self.lookahead_days, len(eligible) - 1)
        target = eligible[-(offset + 1)]
        labels = self.generate_labels(target.strftime("%Y-%m-%d"))
        return [self.save_labels(labels, target.strftime("%Y-%m-%d"))] if not labels.empty else []

    def _business_days(self, start_date: pd.Timestamp, end_date: pd.Timestamp, prices: pd.DataFrame) -> list[pd.Timestamp]:
        try:
            calendar = self.data_loader.load_trading_calendar(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        except Exception:
            calendar = pd.DataFrame()

        if not calendar.empty and {"date", "is_business_day"}.issubset(calendar.columns):
            calendar = calendar.copy()
            calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
            days = calendar[calendar["is_business_day"]]["date"].dropna().sort_values().drop_duplicates().tolist()
            if days:
                normalized = [pd.Timestamp(day).normalize() for day in days]
                if start_date.normalize() in normalized:
                    return normalized

        days = prices["date"].dropna().sort_values().drop_duplicates().tolist()
        return [pd.Timestamp(day).normalize() for day in days]
