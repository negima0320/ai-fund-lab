from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.evaluator import PredictionEvaluator


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-15"),
                "code": "1001",
                "expected_return_10d": 0.10,
                "expected_max_return_10d": 0.12,
                "expected_max_return_20d": 0.22,
                "swing_success_probability_20d": 0.80,
                "upside_probability_10d": 0.80,
                "bad_entry_probability_10d": 0.10,
                "entry_risk_label": "safe",
                "ml_score": 20.0,
            },
            {
                "date": pd.Timestamp("2026-05-15"),
                "code": "1002",
                "expected_return_10d": 0.05,
                "expected_max_return_10d": 0.08,
                "expected_max_return_20d": 0.12,
                "swing_success_probability_20d": 0.55,
                "upside_probability_10d": 0.60,
                "bad_entry_probability_10d": 0.30,
                "entry_risk_label": "watch",
                "ml_score": 10.0,
            },
            {
                "date": pd.Timestamp("2026-05-15"),
                "code": "1003",
                "expected_return_10d": -0.02,
                "expected_max_return_10d": 0.01,
                "expected_max_return_20d": 0.03,
                "swing_success_probability_20d": 0.20,
                "upside_probability_10d": 0.20,
                "bad_entry_probability_10d": 0.50,
                "entry_risk_label": "danger",
                "ml_score": -5.0,
            },
            {
                "date": pd.Timestamp("2026-05-15"),
                "code": "9999",
                "expected_return_10d": 0.50,
                "expected_max_return_10d": 0.60,
                "expected_max_return_20d": 0.70,
                "swing_success_probability_20d": 0.99,
                "upside_probability_10d": 0.99,
                "bad_entry_probability_10d": 0.01,
                "entry_risk_label": "safe",
                "ml_score": 99.0,
            },
        ]
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-05-15"), "code": "1001", "future_5d_return": 0.04, "future_10d_return": 0.12, "future_max_return_10d": 0.15, "future_max_return_20d": 0.25, "future_swing_success_20d": True, "upside_10d": True, "bad_entry_10d": False},
            {"date": pd.Timestamp("2026-05-15"), "code": "1002", "future_5d_return": 0.01, "future_10d_return": 0.03, "future_max_return_10d": 0.06, "future_max_return_20d": 0.09, "future_swing_success_20d": False, "upside_10d": False, "bad_entry_10d": False},
            {"date": pd.Timestamp("2026-05-15"), "code": "1003", "future_5d_return": -0.03, "future_10d_return": -0.05, "future_max_return_10d": 0.02, "future_max_return_20d": 0.04, "future_swing_success_20d": False, "upside_10d": False, "bad_entry_10d": True},
        ]
    )


def test_evaluator_inner_joins_predictions_and_labels() -> None:
    evaluator = PredictionEvaluator()

    joined = evaluator.join_predictions_labels(_predictions(), _labels())

    assert joined["code"].tolist() == ["1001", "1002", "1003"]


def test_evaluator_computes_top_risk_band_and_correlation() -> None:
    evaluator = PredictionEvaluator()
    joined = evaluator.join_predictions_labels(_predictions(), _labels())

    evaluation = evaluator.evaluate_joined(joined, "2026-05-15", top_n=2)

    assert evaluation["joined_rows"] == 3
    assert evaluation["top_n_summary"]["count"] == 2
    assert evaluation["top_n_summary"]["future_10d_return_mean"] == pytest.approx(0.075)
    assert evaluation["top_n_summary"]["upside_10d_rate"] == pytest.approx(0.5)
    assert evaluation["top_n_summary"]["bad_entry_10d_rate"] == pytest.approx(0.0)
    assert [row["entry_risk_label"] for row in evaluation["risk_label_summary"]] == ["danger", "safe", "watch"]
    assert [row["band"] for row in evaluation["bad_entry_probability_bands"]] == ["0.0-0.25", "0.25-0.40", "0.40-1.0"]
    assert [row["band"] for row in evaluation["swing_success_probability_bands"]] == ["0.0-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.0"]
    assert evaluation["expected_vs_future_10d_corr"] is not None
    assert evaluation["expected_max_vs_future_max_20d_corr"] is not None
    assert evaluation["swing_probability_vs_success_20d_corr"] is not None


def test_evaluator_saves_markdown_report(tmp_path) -> None:
    evaluator = PredictionEvaluator(report_root=tmp_path / "reports" / "ml")
    joined = evaluator.join_predictions_labels(_predictions(), _labels())
    evaluation = evaluator.evaluate_joined(joined, "2026-05-15", top_n=2)

    path = evaluator.save_report(evaluation, "2026-05-15")
    content = path.read_text(encoding="utf-8")

    assert path == tmp_path / "reports" / "ml" / "evaluation_2026-05-15.md"
    assert "# ML Prediction Evaluation 2026-05-15" in content
    assert "## Entry Risk Label Summary" in content
    assert "## Bad Entry Probability Bands" in content
    assert "## Swing Success Probability Bands" in content


def test_evaluate_daily_reads_prediction_and_label_parquet(monkeypatch, tmp_path) -> None:
    evaluator = PredictionEvaluator(predictions_root=tmp_path / "predictions", labels_root=tmp_path / "labels")

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        if "predictions" in path.parts:
            return _predictions()
        return _labels()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    evaluation = evaluator.evaluate_daily("2026-05-15", top_n=1)

    assert evaluation["joined_rows"] == 3
    assert evaluation["top_rows"][0]["code"] == "1001"
