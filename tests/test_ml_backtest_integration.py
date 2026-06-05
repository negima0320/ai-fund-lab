from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.backtest_integration import apply_ml_ranked_overlay, build_ml_standalone_scoring_log


def _write_prices(root: Path, date_text: str) -> None:
    path = root / "data" / "cache" / "jquants" / "prices" / f"{date_text}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "prices": [
                    {"Date": date_text, "Code": "1001", "O": 100, "H": 110, "L": 95, "C": 105, "Vo": 1000, "Va": 80_000_000},
                    {"Date": date_text, "Code": "1002", "O": 200, "H": 205, "L": 190, "C": 195, "Vo": 1000, "Va": 70_000_000},
                    {"Date": date_text, "Code": "1003", "O": 300, "H": 305, "L": 290, "C": 295, "Vo": 1000, "Va": 30_000_000},
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_predictions(root: Path, date_text: str) -> None:
    path = root / "data" / "ml" / "walk_forward_predictions" / f"predictions_{date_text}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": date_text,
                "code": "1001",
                "expected_return_10d": 0.10,
                "expected_max_return_20d": 0.20,
                "swing_success_probability_20d": 0.70,
                "bad_entry_probability_10d": 0.20,
            },
            {
                "date": date_text,
                "code": "1002",
                "expected_return_10d": 0.08,
                "expected_max_return_20d": 0.15,
                "swing_success_probability_20d": 0.60,
                "bad_entry_probability_10d": 0.80,
            },
            {
                "date": date_text,
                "code": "1003",
                "expected_return_10d": 0.30,
                "expected_max_return_20d": 0.40,
                "swing_success_probability_20d": 0.90,
                "bad_entry_probability_10d": 0.10,
            },
        ]
    ).to_parquet(path, index=False)


def test_apply_ml_ranked_overlay_prioritizes_risk_adjusted_score(tmp_path: Path) -> None:
    date_text = "2026-05-15"
    _write_predictions(tmp_path, date_text)
    scoring_log = {
        "scores": [
            {"code": "1001", "total_score": 50, "confidence": 0.9, "turnover_value": 80_000_000, "selected": True},
            {"code": "1002", "total_score": 80, "confidence": 0.9, "turnover_value": 70_000_000, "selected": True},
            {"code": "9999", "total_score": 90, "confidence": 0.9, "turnover_value": 90_000_000, "selected": True},
        ],
        "selected": [],
    }
    config = {
        "selection": {"max_selected": 2},
        "ml_backtest": {
            "enabled": True,
            "mode": "ranked",
            "prediction_root": "data/ml/walk_forward_predictions",
            "min_turnover_value": 50_000_000,
        },
    }

    result = apply_ml_ranked_overlay(scoring_log, date_text, config, tmp_path)

    assert [row["code"] for row in result["scores"][:2]] == ["1001", "1002"]
    assert [row["code"] for row in result["selected"]] == ["1001", "1002"]
    assert result["ml_backtest"]["joined_count"] == 2
    assert result["ml_backtest"]["missing_count"] == 1
    assert result["scores"][-1]["ml_prediction_found"] is False


def test_build_ml_standalone_scoring_log_uses_only_prediction_and_price_cache(tmp_path: Path) -> None:
    date_text = "2026-05-15"
    _write_prices(tmp_path, date_text)
    _write_predictions(tmp_path, date_text)
    config = {
        "profile_id": "ml_standalone",
        "selection": {"max_selected": 10},
        "ml_backtest": {
            "enabled": True,
            "mode": "standalone",
            "prediction_root": "data/ml/walk_forward_predictions",
            "min_turnover_value": 50_000_000,
            "top_n": 10,
        },
    }

    result = build_ml_standalone_scoring_log(date_text, config, tmp_path)

    assert result["selected_count"] == 2
    assert [row["code"] for row in result["selected"]] == ["1001", "1002"]
    assert result["selected"][0]["risk_adjusted_score"] == 0.0
    assert result["selected"][1]["risk_adjusted_score"] == -0.32
    assert all(row["selected"] for row in result["scores"])


def test_build_ml_standalone_scoring_log_skips_when_prediction_missing(tmp_path: Path) -> None:
    result = build_ml_standalone_scoring_log(
        "2026-05-15",
        {"ml_backtest": {"enabled": True, "mode": "standalone", "prediction_root": "data/ml/walk_forward_predictions"}},
        tmp_path,
    )

    assert result["selected_count"] == 0
    assert result["ml_backtest"]["warning"] == "missing_prediction"
