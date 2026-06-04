from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.config import (
    FEATURE_COLUMNS,
    FEATURE_HISTORY_DAYS,
    ML_FEATURES_ROOT,
    MOVING_AVERAGE_WINDOWS,
    RETURN_WINDOWS,
    TURNOVER_RATIO_WINDOWS,
    VOLUME_RATIO_WINDOWS,
)
from ml.data_loader import JQuantsDataLoader


class FeatureBuilder:
    """Build daily ML features from cached J-Quants price data."""

    def __init__(
        self,
        data_loader: JQuantsDataLoader | None = None,
        feature_root: str | Path = ML_FEATURES_ROOT,
        history_days: int = FEATURE_HISTORY_DAYS,
    ) -> None:
        self.data_loader = data_loader or JQuantsDataLoader()
        self.feature_root = Path(feature_root)
        self.history_days = history_days

    def build_daily_features(self, target_date: str) -> pd.DataFrame:
        target = pd.Timestamp(target_date)
        history_start = (target - pd.Timedelta(days=self.history_days)).strftime("%Y-%m-%d")
        prices = self.data_loader.load_prices(history_start, target.strftime("%Y-%m-%d"))
        if prices.empty:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices = prices[prices["date"] <= target]
        prices = prices.dropna(subset=["date", "code", "close"])
        prices = prices.sort_values(["code", "date"]).reset_index(drop=True)
        if prices.empty:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        features = self._add_price_features(prices)
        daily = features[features["date"] == target].copy()
        daily = daily.dropna(subset=["date", "code", "close"])
        return self._order_feature_columns(daily).reset_index(drop=True)

    def save_daily_features(self, df: pd.DataFrame, target_date: str) -> Path:
        path = self.feature_root / f"features_{target_date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def _add_price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        df = prices.copy()
        for column in ["open", "high", "low", "close", "volume", "turnover_value"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        grouped = df.groupby("code", group_keys=False)
        for window in RETURN_WINDOWS:
            df[f"return_{window}d"] = grouped["close"].transform(lambda values, w=window: values / values.shift(w) - 1)

        for window in MOVING_AVERAGE_WINDOWS:
            ma_column = f"ma{window}"
            df[ma_column] = grouped["close"].transform(lambda values, w=window: values.rolling(w, min_periods=w).mean())
            df[f"ma{window}_gap"] = df["close"] / df[ma_column] - 1

        for window in [5, 25]:
            ma_column = f"ma{window}"
            df[f"ma{window}_slope"] = grouped[ma_column].transform(lambda values: values / values.shift(5) - 1)

        for window in VOLUME_RATIO_WINDOWS:
            mean_volume = grouped["volume"].transform(lambda values, w=window: values.rolling(w, min_periods=w).mean())
            df[f"volume_ratio_{window}d"] = self._safe_divide(df["volume"], mean_volume)

        for window in TURNOVER_RATIO_WINDOWS:
            mean_turnover = grouped["turnover_value"].transform(lambda values, w=window: values.rolling(w, min_periods=w).mean())
            df[f"turnover_ratio_{window}d"] = self._safe_divide(df["turnover_value"], mean_turnover)

        price_range = df["high"] - df["low"]
        previous_close = grouped["close"].shift(1)
        df["body_ratio"] = self._safe_divide((df["close"] - df["open"]).abs(), price_range).fillna(0)
        df["upper_shadow_ratio"] = self._safe_divide(df["high"] - df[["open", "close"]].max(axis=1), price_range).fillna(0)
        df["lower_shadow_ratio"] = self._safe_divide(df[["open", "close"]].min(axis=1) - df["low"], price_range).fillna(0)
        df["close_position"] = self._safe_divide(df["close"] - df["low"], price_range).fillna(0)
        df["gap_up_ratio"] = self._safe_divide(df["open"], previous_close) - 1
        df["daily_range_ratio"] = self._safe_divide(price_range, df["close"])
        return df

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        safe_denominator = denominator.where(denominator != 0)
        return numerator / safe_denominator

    def _order_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = [column for column in FEATURE_COLUMNS if column in df.columns]
        return df[columns]
