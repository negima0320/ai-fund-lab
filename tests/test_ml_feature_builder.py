from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.data_loader import JQuantsDataLoader
from ml.feature_builder import FeatureBuilder


def _write_price(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")


def _write_cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")


def _price_row(day: int, code: str = "1001", close: float | None = None, volume: float | None = None) -> dict:
    close_value = float(close if close is not None else day * 10)
    return {
        "Date": f"2026-01-{day:02d}",
        "Code": code,
        "O": close_value - 1,
        "H": close_value + 1,
        "L": close_value - 2,
        "C": close_value,
        "Vo": float(volume if volume is not None else day * 100),
        "Va": close_value * 1000,
    }


def _builder(tmp_path: Path) -> FeatureBuilder:
    return FeatureBuilder(data_loader=JQuantsDataLoader(tmp_path / "jquants"), feature_root=tmp_path / "data" / "ml" / "features")


def test_build_daily_features_returns_target_date_only(tmp_path) -> None:
    _write_price(tmp_path / "jquants" / "prices" / "2026-01-01.json", [_price_row(1)])
    _write_price(tmp_path / "jquants" / "prices" / "2026-01-02.json", [_price_row(2)])
    _write_price(tmp_path / "jquants" / "prices" / "2026-01-03.json", [_price_row(3)])

    df = _builder(tmp_path).build_daily_features("2026-01-03")

    assert df["date"].tolist() == [pd.Timestamp("2026-01-03")]
    assert df["code"].tolist() == ["1001"]
    assert df.loc[0, "return_1d"] == pytest.approx(0.5)


def test_build_daily_features_does_not_use_future_prices(tmp_path) -> None:
    _write_price(tmp_path / "jquants" / "prices" / "2026-01-01.json", [_price_row(1)])
    _write_price(tmp_path / "jquants" / "prices" / "2026-01-02.json", [_price_row(2)])
    _write_price(tmp_path / "jquants" / "prices" / "2026-01-03.json", [_price_row(3, close=9999)])

    df = _builder(tmp_path).build_daily_features("2026-01-02")

    assert len(df) == 1
    assert df.loc[0, "close"] == 20
    assert df.loc[0, "return_1d"] == pytest.approx(1.0)


def test_rolling_features_are_calculated_per_code(tmp_path) -> None:
    for day in range(1, 7):
        _write_price(
            tmp_path / "jquants" / "prices" / f"2026-01-{day:02d}.json",
            [_price_row(day, "1001"), _price_row(day, "2001", close=1000 + day)],
        )

    df = _builder(tmp_path).build_daily_features("2026-01-06")
    by_code = df.set_index("code")

    assert by_code.loc["1001", "ma5_gap"] == pytest.approx(60 / 40 - 1)
    assert by_code.loc["2001", "ma5_gap"] == pytest.approx(1006 / 1004 - 1)


def test_zero_range_candlestick_features_do_not_crash(tmp_path) -> None:
    rows = [_price_row(1)]
    rows.append({"Date": "2026-01-02", "Code": "1001", "O": 10, "H": 10, "L": 10, "C": 10, "Vo": 100, "Va": 1000})
    _write_price(tmp_path / "jquants" / "prices" / "2026-01-01.json", [rows[0]])
    _write_price(tmp_path / "jquants" / "prices" / "2026-01-02.json", [rows[1]])

    df = _builder(tmp_path).build_daily_features("2026-01-02")

    assert df.loc[0, "body_ratio"] == 0
    assert df.loc[0, "upper_shadow_ratio"] == 0
    assert df.loc[0, "lower_shadow_ratio"] == 0
    assert df.loc[0, "close_position"] == 0
    assert pd.notna(df.loc[0, "daily_range_ratio"])


def test_save_daily_features_writes_parquet_path(monkeypatch, tmp_path) -> None:
    builder = _builder(tmp_path)
    df = pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "code": ["1001"], "close": [10.0]})
    calls = {}

    def fake_to_parquet(_self: pd.DataFrame, path: Path, index: bool = False) -> None:
        calls["path"] = path
        calls["index"] = index
        path.write_text("parquet", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    path = builder.save_daily_features(df, "2026-01-02")

    assert path == tmp_path / "data" / "ml" / "features" / "features_2026-01-02.parquet"
    assert calls == {"path": path, "index": False}
    assert path.exists()


def test_financial_features_use_only_disclosed_information(tmp_path) -> None:
    _write_price(tmp_path / "jquants" / "prices" / "2026-01-10.json", [_price_row(10)])
    _write_cache(
        tmp_path / "jquants" / "financial_statements" / "2026-01-01_to_2026-01-20.json",
        [
            {
                "DiscDate": "2026-01-01",
                "DiscTime": "15:00:00",
                "Code": "1001",
                "Sales": "100",
                "OP": "10",
                "NP": "5",
                "EPS": "10",
                "BPS": "100",
                "EqAR": "0.50",
                "FEPS": "12",
                "FSales": "130",
                "FOP": "13",
                "PayoutRatioAnn": "0.30",
            },
            {
                "DiscDate": "2026-01-05",
                "DiscTime": "15:00:00",
                "Code": "1001",
                "Sales": "120",
                "OP": "15",
                "NP": "6",
                "EPS": "11",
                "BPS": "110",
                "EqAR": "0.55",
                "FEPS": "13.2",
                "FSales": "150",
                "FOP": "18",
                "PayoutRatioAnn": "0.35",
            },
            {
                "DiscDate": "2026-01-11",
                "DiscTime": "15:00:00",
                "Code": "1001",
                "Sales": "999",
                "OP": "999",
                "NP": "999",
                "EPS": "999",
            },
        ],
    )

    df = _builder(tmp_path).build_daily_features("2026-01-10")

    assert df.loc[0, "EPS"] == pytest.approx(11)
    assert df.loc[0, "BPS"] == pytest.approx(110)
    assert df.loc[0, "EqAR"] == pytest.approx(0.55)
    assert df.loc[0, "Sales_growth"] == pytest.approx(0.2)
    assert df.loc[0, "OP_growth"] == pytest.approx(0.5)
    assert df.loc[0, "NP_growth"] == pytest.approx(0.2)
    assert df.loc[0, "FEPS_growth"] == pytest.approx(0.2)
    assert df.loc[0, "FSales_growth"] == pytest.approx(0.25)
    assert df.loc[0, "FOP_growth"] == pytest.approx(0.2)
    assert df.loc[0, "PayoutRatioAnn"] == pytest.approx(0.35)


def test_earnings_calendar_features_are_added(tmp_path) -> None:
    _write_price(tmp_path / "jquants" / "prices" / "2026-01-10.json", [_price_row(10)])
    _write_cache(
        tmp_path / "jquants" / "earnings_calendar" / "2026-01-01_to_2026-01-20.json",
        [
            {"Date": "2026-01-07", "Code": "1001"},
            {"Date": "2026-01-14", "Code": "1001"},
        ],
    )

    df = _builder(tmp_path).build_daily_features("2026-01-10")

    assert df.loc[0, "days_to_earnings"] == 4
    assert df.loc[0, "days_after_earnings"] == 3
    assert bool(df.loc[0, "is_near_earnings"]) is True
