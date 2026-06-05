from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.data_loader import JQuantsDataLoader
from ml.walk_forward import MLWalkForwardRunner, WalkForwardFold


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _price(date: pd.Timestamp, code: str, close: float = 100.0) -> dict:
    return {
        "Date": date.strftime("%Y-%m-%d"),
        "Code": code,
        "O": 100.0,
        "H": max(close, 101.0),
        "L": 99.0,
        "C": close,
        "Vo": 1000,
        "Va": 100000,
    }


def _runner(tmp_path: Path) -> MLWalkForwardRunner:
    return MLWalkForwardRunner(
        feature_root=tmp_path / "data" / "ml" / "features",
        label_root=tmp_path / "data" / "ml" / "labels",
        prediction_root=tmp_path / "data" / "ml" / "walk_forward_predictions",
        model_root=tmp_path / "models" / "ml" / "walk_forward",
        report_root=tmp_path / "reports" / "ml",
        data_loader=JQuantsDataLoader(tmp_path / "jquants"),
    )


def test_walk_forward_builds_monthly_folds_with_label_safety_lag(tmp_path) -> None:
    runner = _runner(tmp_path)
    for date in pd.date_range("2025-12-01", "2025-12-31", freq="D"):
        _write_json(tmp_path / "jquants" / "prices" / f"{date:%Y-%m-%d}.json", [_price(date, "1001")])

    folds = runner._folds("2025-06-01", "2026-01-01", "2026-02-28")

    assert [fold.month for fold in folds] == ["2026-01", "2026-02"]
    assert folds[0].requested_train_end == "2025-12-31"
    assert folds[0].effective_train_end == "2025-12-11"


def test_walk_forward_simulates_expected_return_top10_close_10d(tmp_path) -> None:
    runner = _runner(tmp_path)
    _write_parquet(
        runner.prediction_root / "predictions_2026-01-01.parquet",
        [
            {"date": pd.Timestamp("2026-01-01"), "code": "1001", "expected_return_10d": 0.9},
            {"date": pd.Timestamp("2026-01-01"), "code": "1002", "expected_return_10d": 0.1},
        ],
    )
    for offset, date in enumerate(pd.date_range("2026-01-01", periods=14, freq="D")):
        close_1001 = 110.0 if offset == 10 else 100.0
        close_1002 = 90.0 if offset == 10 else 100.0
        _write_json(
            tmp_path / "jquants" / "prices" / f"{date:%Y-%m-%d}.json",
            [_price(date, "1001", close=close_1001), _price(date, "1002", close=close_1002)],
        )
    fold = WalkForwardFold(
        month="2026-01",
        train_start="2025-06-01",
        requested_train_end="2025-12-31",
        effective_train_end="2025-12-11",
        test_start="2026-01-01",
        test_end="2026-01-31",
    )

    trades = runner._simulate_month_portfolio(fold, "expected_return_10d", top_n=1, exit_rule="close_10d")
    summary = runner._summarize_trades(trades)

    assert len(trades) == 1
    assert trades[0]["code"] == "1001"
    assert trades[0]["return"] == pytest.approx(0.10)
    assert summary["win_rate"] == pytest.approx(1.0)
    assert summary["total_return"] == pytest.approx(0.10)


def test_walk_forward_saves_reports(tmp_path) -> None:
    runner = _runner(tmp_path)
    result = {
        "period": {"test_start": "2026-01-01", "test_end": "2026-01-31"},
        "train_start": "2025-06-01",
        "ranking": "expected_return_10d",
        "top_n": 10,
        "exit_rule": "close_10d",
        "overall": runner._summarize_trades([]),
        "folds": [],
        "trades": [],
        "warnings": [],
    }

    md_path = runner.save_report(result)
    json_path = runner.save_json(result)

    assert md_path.exists()
    assert json_path.exists()
    assert "ML Walk-Forward Evaluation" in md_path.read_text(encoding="utf-8")
