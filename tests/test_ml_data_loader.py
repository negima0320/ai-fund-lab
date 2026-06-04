from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.data_loader import JQuantsDataLoader


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_prices_normalizes_short_jquants_keys(tmp_path) -> None:
    _write_json(
        tmp_path / "prices" / "2026-01-05.json",
        {
            "records": [
                {
                    "Date": "2026-01-05",
                    "Code": 1001,
                    "O": "10",
                    "H": "12",
                    "L": "9",
                    "C": "11",
                    "Vo": "1000",
                    "Va": "11000",
                    "AdjO": "10.1",
                    "AdjH": "12.1",
                    "AdjL": "9.1",
                    "AdjC": "11.1",
                    "AdjVo": "1001",
                }
            ]
        },
    )

    df = JQuantsDataLoader(tmp_path).load_prices("2026-01-01", "2026-01-31")

    assert df.columns[:8].tolist() == ["date", "code", "open", "high", "low", "close", "volume", "turnover_value"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df.loc[0, "code"] == "1001"
    assert df.loc[0, "open"] == 10
    assert df.loc[0, "turnover_value"] == 11000
    assert df.loc[0, "adjusted_close"] == 11.1


def test_load_prices_accepts_standard_lowercase_keys(tmp_path) -> None:
    _write_json(
        tmp_path / "prices" / "2026-01-06.json",
        {
            "data": [
                {
                    "date": "2026-01-06",
                    "code": "1002",
                    "open": 20,
                    "high": 22,
                    "low": 19,
                    "close": 21,
                    "volume": 2000,
                    "turnover_value": 42000,
                }
            ]
        },
    )

    df = JQuantsDataLoader(tmp_path).load_prices("2026-01-01", "2026-01-31")

    assert df.loc[0, "date"] == pd.Timestamp("2026-01-06")
    assert df.loc[0, "code"] == "1002"
    assert df.loc[0, "close"] == 21


def test_load_trading_calendar_adds_business_day_flag(tmp_path) -> None:
    _write_json(
        tmp_path / "trading_calendar" / "2026-01-01_to_2026-01-03.json",
        {"records": [{"Date": "2026-01-01", "HolDiv": "0"}, {"Date": "2026-01-02", "HolDiv": "1"}]},
    )

    df = JQuantsDataLoader(tmp_path).load_trading_calendar("2026-01-01", "2026-01-03")

    assert df["holiday_division"].tolist() == ["0", "1"]
    assert df["is_business_day"].tolist() == [False, True]


def test_load_investor_types_normalizes_dates_and_numeric_columns(tmp_path) -> None:
    _write_json(
        tmp_path / "investor_types" / "2026-01-01_to_2026-01-31.json",
        {
            "records": [
                {
                    "PubDate": "2026-01-09",
                    "StDate": "2026-01-05",
                    "EnDate": "2026-01-09",
                    "Section": "TSEPrime",
                    "FrgnBal": "100",
                    "IndBal": "-50",
                    "BrkBal": "10",
                    "PropBal": "20",
                    "InvTrBal": "30",
                    "TrstBnkBal": "40",
                    "overseas_net_buy": 100,
                }
            ]
        },
    )

    df = JQuantsDataLoader(tmp_path).load_investor_types("2026-01-01", "2026-01-31")

    assert pd.api.types.is_datetime64_any_dtype(df["PubDate"])
    assert pd.api.types.is_numeric_dtype(df["FrgnBal"])
    assert df.loc[0, "Section"] == "TSEPrime"
    assert df.loc[0, "overseas_net_buy"] == 100


def test_load_listed_info_uses_latest_snapshot_at_as_of_date(tmp_path) -> None:
    _write_json(tmp_path / "listed_info" / "2026-01-01.json", {"records": [{"Date": "2026-01-01", "Code": "1001"}]})
    _write_json(tmp_path / "listed_info" / "2026-02-01.json", {"records": [{"Date": "2026-02-01", "Code": "1002"}]})

    df = JQuantsDataLoader(tmp_path).load_listed_info("2026-01-15")

    assert df["code"].tolist() == ["1001"]


def test_data_loader_does_not_call_api(monkeypatch, tmp_path) -> None:
    import data_provider

    calls = {"count": 0}

    def fail_api_call(*_args, **_kwargs):
        calls["count"] += 1
        raise AssertionError("DataLoader must not call J-Quants API")

    monkeypatch.setattr(data_provider.JQuantsDataProvider, "get_daily_prices", fail_api_call)
    _write_json(tmp_path / "prices" / "2026-01-05.json", {"records": [{"date": "2026-01-05", "code": "1001", "open": 1}]})

    df = JQuantsDataLoader(tmp_path).load_prices("2026-01-05", "2026-01-05")

    assert calls["count"] == 0
    assert len(df) == 1
