from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.exit_dataset import ExitDatasetBuilder


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trades(root: Path) -> None:
    directory = root / "logs" / "backtests" / "rookie_dealer_02_v2_66_ml_ranked" / "2023-01-01_to_2023-01-31"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "trade_id": "t1",
                "code": "1001",
                "entry_date": "2023-01-02",
                "exit_date": "2023-01-09",
                "entry_price": 100,
                "exit_price": 106,
                "shares": 100,
                "net_profit": 4781,
                "holding_days": 6,
            }
        ]
    ).to_csv(directory / "trades.csv", index=False)


def _write_prices(cache_root: Path) -> None:
    dates = pd.bdate_range("2022-12-01", "2023-01-31")
    for index, date_value in enumerate(dates):
        date_text = date_value.strftime("%Y-%m-%d")
        close = 90 + index
        if date_text == "2023-01-02":
            close = 100
        if date_text == "2023-01-03":
            close = 102
        if date_text == "2023-01-10":
            close = 108
        payload = {
            "records": [
                {
                    "date": date_text,
                    "code": "1001",
                    "open": close - 1,
                    "high": close + 2,
                    "low": close - 2,
                    "close": close,
                    "volume": 1000 + index,
                    "turnover_value": (1000 + index) * close,
                }
            ]
        }
        _write_json(cache_root / "prices" / f"{date_text}.json", payload)


def _builder(root: Path, cache_root: Path, predictions_root: Path, output_root: Path, report_root: Path) -> ExitDatasetBuilder:
    return ExitDatasetBuilder(
        root=root,
        profile="rookie_dealer_02_v2_66_ml_ranked",
        start_date="2023-01-01",
        end_date="2023-01-31",
        cache_root=cache_root,
        predictions_root=predictions_root,
        output_root=output_root,
        report_root=report_root,
    )


def test_exit_dataset_builds_held_day_rows_and_future_labels(tmp_path: Path) -> None:
    _write_trades(tmp_path)
    cache_root = tmp_path / "data" / "cache" / "jquants"
    predictions_root = tmp_path / "data" / "ml" / "walk_forward_predictions"
    _write_prices(cache_root)
    predictions_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2023-01-03",
                "code": "1001",
                "expected_return_10d": 0.2,
                "expected_max_return_20d": 0.4,
                "swing_success_probability_20d": 0.8,
                "bad_entry_probability_10d": 0.3,
            }
        ]
    ).to_parquet(predictions_root / "predictions_2023-01-03.parquet", index=False)

    df = _builder(tmp_path, cache_root, predictions_root, tmp_path / "out", tmp_path / "reports").build_dataset()

    assert df["current_date"].min() == "2023-01-03"
    assert df["current_date"].max() == "2023-01-09"
    first = df[df["current_date"].eq("2023-01-03")].iloc[0]
    assert first["holding_days"] == 2
    assert first["prediction_joined"] == True
    assert first["risk_adjusted_score"] == 0.2 - 0.5 * 0.3
    assert first["future_remaining_return_5d"] == (108 / 102) - 1
    assert first["hold_better_5d"] == True
    assert first["should_exit_now_5d"] == False


def test_exit_dataset_allows_missing_prediction_without_future_prediction_leak(tmp_path: Path) -> None:
    _write_trades(tmp_path)
    cache_root = tmp_path / "data" / "cache" / "jquants"
    predictions_root = tmp_path / "data" / "ml" / "walk_forward_predictions"
    _write_prices(cache_root)
    predictions_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2023-01-04",
                "code": "1001",
                "expected_return_10d": 0.9,
                "expected_max_return_20d": 0.9,
                "swing_success_probability_20d": 0.9,
                "bad_entry_probability_10d": 0.0,
            }
        ]
    ).to_parquet(predictions_root / "predictions_2023-01-04.parquet", index=False)

    df = _builder(tmp_path, cache_root, predictions_root, tmp_path / "out", tmp_path / "reports").build_dataset()
    first = df[df["current_date"].eq("2023-01-03")].iloc[0]
    second = df[df["current_date"].eq("2023-01-04")].iloc[0]

    assert first["prediction_joined"] == False
    assert pd.isna(first["expected_return_10d"])
    assert second["prediction_joined"] == True
    assert second["expected_return_10d"] == 0.9


def test_exit_dataset_saves_dataset_and_summary(tmp_path: Path) -> None:
    _write_trades(tmp_path)
    cache_root = tmp_path / "data" / "cache" / "jquants"
    predictions_root = tmp_path / "data" / "ml" / "walk_forward_predictions"
    output_root = tmp_path / "out"
    report_root = tmp_path / "reports"
    _write_prices(cache_root)
    builder = _builder(tmp_path, cache_root, predictions_root, output_root, report_root)
    df = builder.build_dataset()
    dataset_path = builder.save_dataset(df)
    summary = builder.summarize(df, dataset_path)
    paths = builder.save_summary(summary)

    assert dataset_path.exists()
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert summary["rows"] == len(df)
    assert summary["unique_trades"] == 1
