from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.data_loader import JQuantsDataLoader
from ml.paper_portfolio import MLPaperPortfolioSimulator


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")


def _price(day: int, code: str, open_price: float = 100.0, high: float = 101.0, low: float = 99.0, close: float = 100.0) -> dict:
    return {
        "Date": f"2026-01-{day:02d}",
        "Code": code,
        "O": open_price,
        "H": high,
        "L": low,
        "C": close,
        "Vo": 1000,
        "Va": 100000,
    }


def _prediction(date: str, code: str, score: float) -> dict:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "expected_max_return_20d": score,
        "swing_success_probability_20d": score,
        "ml_score": score,
        "expected_return_10d": score,
    }


def _label(date: str, code: str, max20: float, swing: bool, bad: bool) -> dict:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "future_10d_return": 0.01,
        "future_max_return_20d": max20,
        "future_swing_success_20d": swing,
        "bad_entry_10d": bad,
    }


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _simulator(tmp_path: Path) -> MLPaperPortfolioSimulator:
    return MLPaperPortfolioSimulator(
        predictions_root=tmp_path / "data" / "ml" / "predictions",
        labels_root=tmp_path / "data" / "ml" / "labels",
        report_root=tmp_path / "reports" / "ml",
        data_loader=JQuantsDataLoader(tmp_path / "jquants"),
    )


def test_paper_portfolio_simulates_exit_rules(tmp_path) -> None:
    simulator = _simulator(tmp_path)
    _write_parquet(
        simulator.predictions_root / "predictions_2026-01-01.parquet",
        [_prediction("2026-01-01", "1001", 0.9)],
    )
    _write_parquet(
        simulator.labels_root / "labels_2026-01-01.parquet",
        [_label("2026-01-01", "1001", 0.20, True, False)],
    )
    for day in range(1, 22):
        high = 112.0 if day == 5 else 101.0
        low = 94.0 if day == 8 else 99.0
        close = 120.0 if day == 21 else 100.0
        _write_json(tmp_path / "jquants" / "prices" / f"2026-01-{day:02d}.json", [_price(day, "1001", high=high, low=low, close=close)])

    result = simulator.simulate("2026-01-01", "2026-01-01", top_n=1)

    rows = {
        (row["ranking"], row["exit_rule"]): row
        for row in result["ranking_exit_summary"]
    }
    assert rows[("expected_max_return_20d_top10", "close_20d")]["average_return"] == pytest.approx(0.20)
    assert rows[("expected_max_return_20d_top10", "take_profit_10pct_or_close_20d")]["average_return"] == pytest.approx(0.10)
    assert rows[("expected_max_return_20d_top10", "stop_loss_5pct_or_close_20d")]["average_return"] == pytest.approx(-0.05)
    assert rows[("expected_max_return_20d_top10", "take_profit_10pct_stop_loss_5pct_or_close_20d")]["average_return"] == pytest.approx(0.10)
    assert len(result["paper_trades"]) == 20


def test_paper_portfolio_saves_reports(tmp_path) -> None:
    simulator = _simulator(tmp_path)
    result = {
        "period": {"start_date": "2026-01-01", "end_date": "2026-01-31"},
        "top_n": 1,
        "ranking_exit_summary": [],
        "monthly_return": [],
        "paper_trades": [{"ranking": "ml_score_top10", "return": 0.1}],
    }

    md_path = simulator.save_report(result)
    json_path = simulator.save_json(result)
    csv_path = simulator.save_trades_csv(result)

    assert md_path.exists()
    assert json_path.exists()
    assert csv_path.exists()
    assert "ML Paper Portfolio Simulation" in md_path.read_text(encoding="utf-8")
