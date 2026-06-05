from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import (
    ADJUSTED_PRICE_COLUMNS,
    FINANCIAL_NUMERIC_COLUMNS,
    INVESTOR_TYPE_DATE_COLUMNS,
    INVESTOR_TYPE_NUMERIC_COLUMNS,
    JQUANTS_CACHE_DIRS,
    JQUANTS_CACHE_ROOT,
    PRICE_COLUMNS,
)


DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

COMMON_ALIASES = {
    "Date": "date",
    "Code": "code",
    "LocalCode": "code",
}

PRICE_ALIASES = {
    **COMMON_ALIASES,
    "O": "open",
    "Open": "open",
    "H": "high",
    "High": "high",
    "L": "low",
    "Low": "low",
    "C": "close",
    "Close": "close",
    "Vo": "volume",
    "Volume": "volume",
    "Va": "turnover_value",
    "TurnoverValue": "turnover_value",
    "AdjO": "adjusted_open",
    "AdjH": "adjusted_high",
    "AdjL": "adjusted_low",
    "AdjC": "adjusted_close",
    "AdjVo": "adjusted_volume",
}

TRADING_CALENDAR_ALIASES = {
    "Date": "date",
    "HolDiv": "holiday_division",
}

FINANCIAL_STATEMENT_ALIASES = {
    **COMMON_ALIASES,
    "DiscDate": "date",
}


class JQuantsDataLoader:
    """Read normalized pandas DataFrames from local J-Quants cache JSON files."""

    def __init__(self, cache_root: str | Path = JQUANTS_CACHE_ROOT) -> None:
        self.cache_root = Path(cache_root)

    def load_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load_endpoint("prices", aliases=PRICE_ALIASES)
        df = self._normalize_date_column(df, "date")
        df = self._normalize_code_column(df)
        df = self._normalize_numeric_columns(df, PRICE_COLUMNS[2:] + ADJUSTED_PRICE_COLUMNS)
        df = self._filter_date_range(df, "date", start_date, end_date)
        return self._order_columns(df, PRICE_COLUMNS + ADJUSTED_PRICE_COLUMNS)

    def load_listed_info(self, as_of_date: str | None = None) -> pd.DataFrame:
        paths = self._listed_info_paths(as_of_date)
        df = self._records_to_frame(self._read_records_from_paths(paths), COMMON_ALIASES)
        df = self._normalize_date_column(df, "date")
        df = self._normalize_code_column(df)
        if as_of_date is not None:
            df = self._filter_date_range(df, "date", None, as_of_date)
        return self._order_columns(df, ["date", "code"])

    def load_topix(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load_endpoint("topix_prices", aliases=PRICE_ALIASES)
        df = self._normalize_date_column(df, "date")
        df = self._normalize_numeric_columns(df, ["open", "high", "low", "close"])
        df = self._filter_date_range(df, "date", start_date, end_date)
        return self._order_columns(df, ["date", "open", "high", "low", "close"])

    def load_earnings_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load_endpoint("earnings_calendar", aliases=COMMON_ALIASES)
        df = self._normalize_date_column(df, "date")
        df = self._normalize_code_column(df)
        df = self._filter_date_range(df, "date", start_date, end_date)
        return self._order_columns(df, ["date", "code"])

    def load_trading_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load_endpoint("trading_calendar", aliases=TRADING_CALENDAR_ALIASES)
        df = self._normalize_date_column(df, "date")
        if "holiday_division" in df.columns:
            df["holiday_division"] = df["holiday_division"].astype("string")
            df["is_business_day"] = df["holiday_division"].eq("1")
        df = self._filter_date_range(df, "date", start_date, end_date)
        return self._order_columns(df, ["date", "holiday_division", "is_business_day"])

    def load_investor_types(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load_endpoint("investor_types")
        for column in INVESTOR_TYPE_DATE_COLUMNS:
            df = self._normalize_date_column(df, column)
        df = self._normalize_numeric_columns(df, INVESTOR_TYPE_NUMERIC_COLUMNS)
        filter_column = "date" if "date" in df.columns else "EnDate"
        df = self._filter_date_range(df, filter_column, start_date, end_date)
        sort_columns = [column for column in [filter_column, "Section"] if column in df.columns]
        return df.sort_values(sort_columns).reset_index(drop=True) if sort_columns else df.reset_index(drop=True)

    def load_financial_statements(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load_endpoint("financial_statements", aliases=FINANCIAL_STATEMENT_ALIASES)
        df = self._normalize_date_column(df, "date")
        df = self._normalize_code_column(df)
        df = self._normalize_numeric_columns(df, FINANCIAL_NUMERIC_COLUMNS)
        df = self._filter_date_range(df, "date", start_date, end_date)
        return self._order_columns(df, ["date", "code"])

    def _load_endpoint(self, endpoint: str, aliases: dict[str, str] | None = None) -> pd.DataFrame:
        directory = self.cache_root / JQUANTS_CACHE_DIRS[endpoint]
        return self._records_to_frame(self._read_records_from_paths(sorted(directory.glob("*.json"))), aliases or {})

    def _listed_info_paths(self, as_of_date: str | None) -> list[Path]:
        directory = self.cache_root / JQUANTS_CACHE_DIRS["listed_info"]
        paths = sorted(directory.glob("*.json"))
        if not paths:
            return []
        dated = [(path, self._date_from_filename(path)) for path in paths]
        dated_paths = [(path, value) for path, value in dated if value is not None]
        if not dated_paths:
            return paths
        if as_of_date is None:
            return [max(dated_paths, key=lambda item: item[1])[0]]
        as_of = pd.Timestamp(as_of_date)
        candidates = [(path, value) for path, value in dated_paths if value <= as_of]
        return [max(candidates, key=lambda item: item[1])[0]] if candidates else []

    def _read_records_from_paths(self, paths: Iterable[Path]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as file:
                records.extend(self._extract_records(json.load(file)))
        return records

    def _extract_records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("records", "data", "master", "equities", "listed", "info"):
            value = payload.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
        for value in payload.values():
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
        return []

    def _records_to_frame(self, records: list[dict[str, Any]], aliases: dict[str, str]) -> pd.DataFrame:
        return pd.DataFrame([self._normalize_record_keys(record, aliases) for record in records])

    def _normalize_record_keys(self, record: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in record.items():
            normalized_key = aliases.get(key, key)
            if normalized_key not in normalized or normalized[normalized_key] is None:
                normalized[normalized_key] = value
        return normalized

    def _normalize_date_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
        return df

    def _normalize_code_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if "code" in df.columns:
            df["code"] = df["code"].astype("string")
        return df

    def _normalize_numeric_columns(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for column in columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def _filter_date_range(self, df: pd.DataFrame, column: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
        if column not in df.columns:
            return df.reset_index(drop=True)
        filtered = df
        if start_date is not None:
            filtered = filtered[filtered[column] >= pd.Timestamp(start_date)]
        if end_date is not None:
            filtered = filtered[filtered[column] <= pd.Timestamp(end_date)]
        return filtered.sort_values(column).reset_index(drop=True)

    def _order_columns(self, df: pd.DataFrame, first_columns: list[str]) -> pd.DataFrame:
        ordered = [column for column in first_columns if column in df.columns]
        rest = [column for column in df.columns if column not in ordered]
        return df[ordered + rest]

    def _date_from_filename(self, path: Path) -> pd.Timestamp | None:
        match = DATE_RE.search(path.name)
        if not match:
            return None
        return pd.Timestamp(match.group(0))
