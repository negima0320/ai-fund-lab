from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from ml.ranking_analysis import MLRankingAnalyzer


def _prediction(date: str, code: str, max20: float, swing: float, ml_score: float, ret10: float) -> dict:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "expected_max_return_20d": max20,
        "swing_success_probability_20d": swing,
        "ml_score": ml_score,
        "expected_return_10d": ret10,
    }


def _label(date: str, code: str, future10: float, max20: float, swing: bool, bad: bool) -> dict:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "future_10d_return": future10,
        "future_max_return_20d": max20,
        "future_swing_success_20d": swing,
        "bad_entry_10d": bad,
    }


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_trades(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["action", "signal_date", "entry_date", "code"])
        writer.writeheader()
        writer.writerow({"action": "SELL", "signal_date": "2026-05-01", "entry_date": "2026-05-02", "code": "1001"})
        writer.writerow({"action": "SELL", "signal_date": "2026-05-02", "entry_date": "2026-05-03", "code": "2001"})


def _analyzer(tmp_path: Path) -> MLRankingAnalyzer:
    return MLRankingAnalyzer(
        predictions_root=tmp_path / "data" / "ml" / "predictions",
        labels_root=tmp_path / "data" / "ml" / "labels",
        report_root=tmp_path / "reports" / "ml",
        root=tmp_path,
    )


def test_ml_ranking_analyzer_summarizes_top_rankings_and_overlap(tmp_path) -> None:
    analyzer = _analyzer(tmp_path)
    _write_parquet(
        analyzer.predictions_root / "predictions_2026-05-01.parquet",
        [
            _prediction("2026-05-01", "1001", 0.30, 0.90, 5.0, 0.10),
            _prediction("2026-05-01", "1002", 0.10, 0.20, 1.0, 0.02),
        ],
    )
    _write_parquet(
        analyzer.labels_root / "labels_2026-05-01.parquet",
        [
            _label("2026-05-01", "1001", 0.12, 0.35, True, False),
            _label("2026-05-01", "1002", -0.01, 0.05, False, True),
        ],
    )
    _write_parquet(
        analyzer.predictions_root / "predictions_2026-05-02.parquet",
        [
            _prediction("2026-05-02", "2001", 0.25, 0.80, 4.0, 0.08),
            _prediction("2026-05-02", "2002", 0.05, 0.10, 0.0, -0.02),
        ],
    )
    _write_parquet(
        analyzer.labels_root / "labels_2026-05-02.parquet",
        [
            _label("2026-05-02", "2001", 0.09, 0.20, True, False),
            _label("2026-05-02", "2002", -0.03, 0.02, False, True),
        ],
    )
    _write_trades(tmp_path / "logs" / "backtests" / "profile1" / "2026-01-01_to_2026-12-31" / "trades.csv")

    analysis = analyzer.analyze("2026-05-01", "2026-05-02", top_n=1, profile="profile1")

    assert analysis["baseline_all_stocks"]["count"] == 4
    assert analysis["baseline_all_stocks"]["future_max_return_20d_mean"] == pytest.approx(0.155)
    summary = {row["ranking"]: row for row in analysis["ranking_summary"]}
    assert summary["expected_max_return_20d_top10"]["count"] == 2
    assert summary["expected_max_return_20d_top10"]["future_max_return_20d_mean"] == pytest.approx(0.275)
    assert summary["swing_success_probability_20d_top10"]["future_swing_success_20d_rate"] == pytest.approx(1.0)
    overlap = {row["ranking"]: row for row in analysis["overlap_summary"]}
    assert overlap["expected_max_return_20d_top10"]["bought_count"] == 2
    assert overlap["expected_max_return_20d_top10"]["existing_trade_topn_rate"] == pytest.approx(1.0)
    assert len(analysis["ranking_details"]) == 8


def test_ml_ranking_analyzer_saves_markdown_json_and_csv(tmp_path) -> None:
    analyzer = _analyzer(tmp_path)
    analysis = {
        "period": {"start_date": "2026-05-01", "end_date": "2026-05-02"},
        "top_n": 1,
        "profile": "profile1",
        "trades_source": "trades.csv",
        "processed_dates": ["2026-05-01"],
        "skipped_dates": [],
        "baseline_all_stocks": {
            "count": 1,
            "future_10d_return_mean": 0.1,
            "future_max_return_20d_mean": 0.2,
            "future_swing_success_20d_rate": 1.0,
            "bad_entry_10d_rate": 0.0,
        },
        "ranking_summary": [],
        "monthly_summary": [],
        "overlap_summary": [],
        "ranking_details": [{"ranking": "ml_score_top10", "code": "1001"}],
    }

    md_path = analyzer.save_report(analysis)
    json_path = analyzer.save_json(analysis)
    csv_path = analyzer.save_details_csv(analysis)

    assert md_path.exists()
    assert json_path.exists()
    assert csv_path.exists()
    assert "## Ranking Comparison" in md_path.read_text(encoding="utf-8")
