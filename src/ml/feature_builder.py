from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.config import (
    EARNINGS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FINANCIAL_FEATURE_COLUMNS,
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
        daily = self._add_financial_features(daily, target)
        daily = self._add_earnings_features(daily, target)
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

    def _add_financial_features(self, daily: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
        output = daily.copy()
        for column in FINANCIAL_FEATURE_COLUMNS:
            output[column] = pd.NA

        lookback_start = (target - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
        statements = self.data_loader.load_financial_statements(lookback_start, target.strftime("%Y-%m-%d"))
        if statements.empty or not {"date", "code"}.issubset(statements.columns):
            return output

        statements = statements.copy()
        statements["date"] = pd.to_datetime(statements["date"], errors="coerce")
        statements["code"] = statements["code"].astype("string")
        statements = statements[statements["date"] <= target]
        statements = statements.dropna(subset=["date", "code"])
        if statements.empty:
            return output

        if "DiscTime" in statements.columns:
            statements["_disc_time"] = statements["DiscTime"].astype("string").fillna("")
        else:
            statements["_disc_time"] = ""

        numeric_columns = ["Sales", "OP", "NP", "EPS", "BPS", "EqAR", "FEPS", "FSales", "FOP", "PayoutRatioAnn"]
        for column in numeric_columns:
            if column in statements.columns:
                statements[column] = pd.to_numeric(statements[column], errors="coerce")

        statements = statements.sort_values(["code", "date", "_disc_time"]).reset_index(drop=True)
        grouped = statements.groupby("code", group_keys=False)
        growth_pairs = {
            "Sales_growth": "Sales",
            "OP_growth": "OP",
            "NP_growth": "NP",
        }
        for feature, source in growth_pairs.items():
            if source in statements.columns:
                previous = grouped[source].shift(1)
                statements[feature] = self._safe_divide(statements[source], previous) - 1
        forecast_pairs = {
            "FEPS_growth": ("FEPS", "EPS"),
            "FSales_growth": ("FSales", "Sales"),
            "FOP_growth": ("FOP", "OP"),
        }
        for feature, (forecast, actual) in forecast_pairs.items():
            if forecast in statements.columns and actual in statements.columns:
                statements[feature] = self._safe_divide(statements[forecast], statements[actual]) - 1

        latest = statements.groupby("code", as_index=False).tail(1)
        columns = ["code", *[column for column in FINANCIAL_FEATURE_COLUMNS if column in latest.columns]]
        if len(columns) == 1:
            return output
        output["code"] = output["code"].astype("string")
        return output.merge(latest[columns], on="code", how="left", suffixes=("", "_financial")).pipe(
            self._coalesce_financial_columns
        )

    def _coalesce_financial_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        for column in FINANCIAL_FEATURE_COLUMNS:
            financial_column = f"{column}_financial"
            if financial_column in output.columns:
                output[column] = output[financial_column].combine_first(output[column])
                output = output.drop(columns=[financial_column])
        return output

    def _add_earnings_features(self, daily: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
        output = daily.copy()
        for column in EARNINGS_FEATURE_COLUMNS:
            output[column] = pd.NA

        calendar = self.data_loader.load_earnings_calendar(
            (target - pd.Timedelta(days=365)).strftime("%Y-%m-%d"),
            (target + pd.Timedelta(days=365)).strftime("%Y-%m-%d"),
        )
        if calendar.empty or not {"date", "code"}.issubset(calendar.columns):
            output["is_near_earnings"] = False
            return output

        calendar = calendar.copy()
        calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
        calendar["code"] = calendar["code"].astype("string")
        calendar = calendar.dropna(subset=["date", "code"])
        if calendar.empty:
            output["is_near_earnings"] = False
            return output

        target_codes = output[["code"]].copy()
        target_codes["code"] = target_codes["code"].astype("string")
        merged = target_codes.merge(calendar[["date", "code"]], on="code", how="left")
        merged["delta_days"] = (merged["date"] - target).dt.days

        future = (
            merged[merged["delta_days"] >= 0]
            .sort_values(["code", "delta_days"])
            .groupby("code", as_index=False)
            .first()[["code", "delta_days"]]
            .rename(columns={"delta_days": "days_to_earnings"})
        )
        past = (
            merged[merged["delta_days"] <= 0]
            .assign(days_after_earnings=lambda df: -df["delta_days"])
            .sort_values(["code", "days_after_earnings"])
            .groupby("code", as_index=False)
            .first()[["code", "days_after_earnings"]]
        )

        output = output.merge(future, on="code", how="left", suffixes=("", "_earnings"))
        output = output.merge(past, on="code", how="left", suffixes=("", "_earnings"))
        for column in ["days_to_earnings", "days_after_earnings"]:
            earnings_column = f"{column}_earnings"
            if earnings_column in output.columns:
                output[column] = output[earnings_column].combine_first(output[column])
                output = output.drop(columns=[earnings_column])
            output[column] = pd.to_numeric(output[column], errors="coerce")
        output["is_near_earnings"] = output["days_to_earnings"].abs().le(5).fillna(False)
        return output

    def _order_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for column in FINANCIAL_FEATURE_COLUMNS:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        for column in ["days_to_earnings", "days_after_earnings"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        if "is_near_earnings" in df.columns:
            df["is_near_earnings"] = df["is_near_earnings"].fillna(False).astype(bool)
        columns = [column for column in FEATURE_COLUMNS if column in df.columns]
        return df[columns]
