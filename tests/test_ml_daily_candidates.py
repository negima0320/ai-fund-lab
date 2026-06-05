from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.daily_candidates import DailyAICandidateExporter


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _prediction(code: str, expected: float, bad: float) -> dict:
    return {
        "date": pd.Timestamp("2026-05-15"),
        "code": code,
        "expected_return_10d": expected,
        "expected_max_return_20d": expected + 0.1,
        "swing_success_probability_20d": 0.6,
        "bad_entry_probability_10d": bad,
        "entry_risk_label": "danger" if bad >= 0.4 else "watch",
        "ml_score": expected * 100 - bad * 15,
    }


def _feature(code: str, turnover: float) -> dict:
    return {
        "date": pd.Timestamp("2026-05-15"),
        "code": code,
        "close": 100.0,
        "turnover_value": turnover,
    }


def _exporter(tmp_path: Path) -> DailyAICandidateExporter:
    return DailyAICandidateExporter(
        prediction_root=tmp_path / "data" / "ml" / "predictions",
        feature_root=tmp_path / "data" / "ml" / "features",
        report_root=tmp_path / "reports" / "ml" / "daily_candidates",
        cache_root=tmp_path / "jquants",
    )


def test_daily_candidates_filter_sort_and_save(tmp_path) -> None:
    exporter = _exporter(tmp_path)
    _write_parquet(
        exporter.prediction_root / "predictions_2026-05-15.parquet",
        [
            _prediction("1001", 0.10, 0.20),
            _prediction("1002", 0.20, 0.80),
            _prediction("1003", 0.08, 0.30),
            _prediction("1004", 0.12, 0.30),
        ],
    )
    _write_parquet(
        exporter.feature_root / "features_2026-05-15.parquet",
        [
            _feature("1001", 60_000_000),
            _feature("1002", 60_000_000),
            _feature("1003", 10_000_000),
            _feature("1004", 70_000_000),
        ],
    )
    _write_json(
        tmp_path / "jquants" / "listed_info" / "2026-05-15.json",
        [
            {"Date": "2026-05-15", "Code": "1001", "CoName": "Alpha", "MktNm": "Prime", "S33Nm": "Tech"},
            {"Date": "2026-05-15", "Code": "1004", "CoName": "Delta", "MktNm": "Prime", "S33Nm": "Machinery"},
        ],
    )

    candidates = exporter.build_candidates(
        "2026-05-15",
        top_n=10,
        min_turnover_value=50_000_000,
        max_bad_entry_probability=0.70,
    )
    csv_path = exporter.save_csv(candidates, "2026-05-15")
    md_path = exporter.save_markdown(candidates, "2026-05-15")

    assert candidates["code"].tolist() == ["1004", "1001"]
    assert candidates["rank"].tolist() == [1, 2]
    assert candidates.loc[0, "name"] == "Delta"
    assert csv_path.exists()
    assert md_path.exists()
    assert "Daily AI Candidates" in md_path.read_text(encoding="utf-8")
