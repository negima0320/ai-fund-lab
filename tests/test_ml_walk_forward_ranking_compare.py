from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.data_loader import JQuantsDataLoader
from ml.walk_forward_ranking_compare import RankingStrategy, WalkForwardRankingComparator


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _prediction(date: str, code: str, expected: float, bad_prob: float) -> dict:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "expected_return_10d": expected,
        "upside_probability_10d": 0.5,
        "bad_entry_probability_10d": bad_prob,
        "expected_max_return_20d": expected + 0.1,
        "swing_success_probability_20d": 0.5,
        "ml_score": expected,
    }


def _label(date: str, code: str, bad: bool) -> dict:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "future_10d_return": 0.1,
        "bad_entry_10d": bad,
    }


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


def _comparator(tmp_path: Path) -> WalkForwardRankingComparator:
    return WalkForwardRankingComparator(
        prediction_root=tmp_path / "data" / "ml" / "walk_forward_predictions",
        label_root=tmp_path / "data" / "ml" / "labels",
        report_root=tmp_path / "reports" / "ml",
        cache_root=tmp_path / "jquants",
    )


def test_ranking_compare_filters_bad_entry_and_applies_sector_cap(tmp_path) -> None:
    comparator = _comparator(tmp_path)
    _write_parquet(
        comparator.prediction_root / "predictions_2026-01-01.parquet",
        [
            _prediction("2026-01-01", "1001", 0.9, 0.8),
            _prediction("2026-01-01", "1002", 0.8, 0.2),
            _prediction("2026-01-01", "1003", 0.7, 0.2),
        ],
    )
    _write_parquet(
        comparator.label_root / "labels_2026-01-01.parquet",
        [_label("2026-01-01", "1001", True), _label("2026-01-01", "1002", False), _label("2026-01-01", "1003", False)],
    )
    _write_json(
        tmp_path / "jquants" / "listed_info" / "2026-01-01.json",
        [
            {"Date": "2026-01-01", "Code": "1001", "S33Nm": "A", "MktNm": "Prime"},
            {"Date": "2026-01-01", "Code": "1002", "S33Nm": "A", "MktNm": "Prime"},
            {"Date": "2026-01-01", "Code": "1003", "S33Nm": "B", "MktNm": "Prime"},
        ],
    )
    for offset, date in enumerate(pd.date_range("2026-01-01", periods=12, freq="D")):
        close = 110.0 if offset == 10 else 100.0
        _write_json(
            tmp_path / "jquants" / "prices" / f"{date:%Y-%m-%d}.json",
            [_price(date, "1001", close), _price(date, "1002", close), _price(date, "1003", close)],
        )
    strategies = [
        RankingStrategy("expected_return_10d"),
        RankingStrategy("expected_return_10d_bad_entry_lt_0_70", bad_entry_threshold=0.70),
        RankingStrategy("expected_return_10d_sector_cap_3", sector_cap=1),
    ]

    result = comparator.compare("2026-01-01", "2026-01-31", top_n=2, strategies=strategies)
    by_strategy = {row["strategy"]: row for row in result["summary"]}

    assert by_strategy["expected_return_10d"]["total_trades"] == 2
    assert by_strategy["expected_return_10d_bad_entry_lt_0_70"]["bad_entry_rate"] == pytest.approx(0.0)
    assert by_strategy["expected_return_10d_sector_cap_3"]["total_trades"] == 2
    assert by_strategy["expected_return_10d_sector_cap_3"]["unique_codes"] == 2


def test_ranking_compare_saves_reports(tmp_path) -> None:
    comparator = _comparator(tmp_path)
    result = {
        "period": {"start_date": "2026-01-01", "end_date": "2026-05-31"},
        "top_n": 10,
        "exit_rule": "close_10d",
        "summary": [{"strategy": "expected_return_10d", "total_return": 1.0}],
        "monthly_summary": [],
        "trades": [],
    }

    md_path = comparator.save_report(result)
    json_path = comparator.save_json(result)

    assert md_path.exists()
    assert json_path.exists()
    assert "Walk-Forward Ranking Comparison" in md_path.read_text(encoding="utf-8")
