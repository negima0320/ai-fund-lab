from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.data_loader import JQuantsDataLoader
from ml.label_generator import LabelGenerator


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")


def _price_row(day: int, code: str = "1001", open_price: float | None = None, close: float | None = None, high: float | None = None, low: float | None = None) -> dict:
    open_value = float(open_price if open_price is not None else 100 + day)
    close_value = float(close if close is not None else open_value)
    high_value = float(high if high is not None else max(open_value, close_value) + 1)
    low_value = float(low if low is not None else min(open_value, close_value) - 1)
    return {
        "Date": f"2026-01-{day:02d}",
        "Code": code,
        "O": open_value,
        "H": high_value,
        "L": low_value,
        "C": close_value,
        "Vo": 1000,
        "Va": 100000,
    }


def _calendar_row(day: int, business: bool = True) -> dict:
    return {"Date": f"2026-01-{day:02d}", "HolDiv": "1" if business else "0"}


def _generator(tmp_path: Path) -> LabelGenerator:
    return LabelGenerator(data_loader=JQuantsDataLoader(tmp_path / "jquants"), label_root=tmp_path / "data" / "ml" / "labels")


def test_generate_labels_uses_next_business_day_open_as_entry(tmp_path) -> None:
    for day in range(1, 12):
        _write_json(tmp_path / "jquants" / "prices" / f"2026-01-{day:02d}.json", [_price_row(day, open_price=100 + day, close=100 + day)])
    _write_json(tmp_path / "jquants" / "trading_calendar" / "2026-01-01_to_2026-01-21.json", [_calendar_row(day) for day in range(1, 12)])

    df = _generator(tmp_path).generate_labels("2026-01-01")

    assert df.loc[0, "entry_price"] == 102


def test_generate_labels_calculates_future_returns(tmp_path) -> None:
    for day in range(1, 12):
        close = 110 if day == 6 else 120 if day == 11 else 100
        _write_json(tmp_path / "jquants" / "prices" / f"2026-01-{day:02d}.json", [_price_row(day, open_price=100, close=close, high=close, low=close)])
    _write_json(tmp_path / "jquants" / "trading_calendar" / "2026-01-01_to_2026-01-21.json", [_calendar_row(day) for day in range(1, 12)])

    df = _generator(tmp_path).generate_labels("2026-01-01")

    assert df.loc[0, "future_5d_return"] == pytest.approx(0.10)
    assert df.loc[0, "future_10d_return"] == pytest.approx(0.20)


def test_generate_labels_marks_upside_and_bad_entry_within_10_business_days(tmp_path) -> None:
    for day in range(1, 12):
        high = 106 if day == 4 else 101
        low = 94 if day == 8 else 99
        _write_json(tmp_path / "jquants" / "prices" / f"2026-01-{day:02d}.json", [_price_row(day, open_price=100, close=100, high=high, low=low)])
    _write_json(tmp_path / "jquants" / "trading_calendar" / "2026-01-01_to_2026-01-21.json", [_calendar_row(day) for day in range(1, 12)])

    df = _generator(tmp_path).generate_labels("2026-01-01")

    assert bool(df.loc[0, "upside_10d"]) is True
    assert bool(df.loc[0, "bad_entry_10d"]) is True


def test_generate_labels_falls_back_to_price_dates_when_calendar_missing(tmp_path) -> None:
    for day in range(1, 12):
        _write_json(tmp_path / "jquants" / "prices" / f"2026-01-{day:02d}.json", [_price_row(day, open_price=100, close=100 + day)])

    df = _generator(tmp_path).generate_labels("2026-01-01")

    assert len(df) == 1
    assert df.loc[0, "future_10d_return"] == pytest.approx(0.11)


def test_generate_labels_returns_empty_when_future_data_is_incomplete(tmp_path) -> None:
    for day in range(1, 6):
        _write_json(tmp_path / "jquants" / "prices" / f"2026-01-{day:02d}.json", [_price_row(day)])
    _write_json(tmp_path / "jquants" / "trading_calendar" / "2026-01-01_to_2026-01-05.json", [_calendar_row(day) for day in range(1, 6)])

    df = _generator(tmp_path).generate_labels("2026-01-01")

    assert df.empty


def test_generate_labels_calculates_per_code_and_excludes_missing_future_code(tmp_path) -> None:
    for day in range(1, 12):
        rows = [_price_row(day, "1001", open_price=100, close=100 + day)]
        if day != 11:
            rows.append(_price_row(day, "2001", open_price=200, close=200 + day))
        _write_json(tmp_path / "jquants" / "prices" / f"2026-01-{day:02d}.json", rows)
    _write_json(tmp_path / "jquants" / "trading_calendar" / "2026-01-01_to_2026-01-21.json", [_calendar_row(day) for day in range(1, 12)])

    df = _generator(tmp_path).generate_labels("2026-01-01")

    assert df["code"].tolist() == ["1001"]


def test_save_labels_writes_parquet_path(monkeypatch, tmp_path) -> None:
    generator = _generator(tmp_path)
    df = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "code": ["1001"], "entry_price": [100.0]})
    calls = {}

    def fake_to_parquet(_self: pd.DataFrame, path: Path, index: bool = False) -> None:
        calls["path"] = path
        calls["index"] = index
        path.write_text("parquet", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    path = generator.save_labels(df, "2026-01-01")

    assert path == tmp_path / "data" / "ml" / "labels" / "labels_2026-01-01.parquet"
    assert calls == {"path": path, "index": False}
    assert path.exists()
