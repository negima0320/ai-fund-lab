from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.portfolio_manager_dataset import AUDIT_COLUMNS
from ml.portfolio_manager_dataset import CLEAN_FEATURE_COLUMNS
from ml.portfolio_manager_dataset import CLEAN_FORBIDDEN_FEATURE_COLUMNS
from ml.portfolio_manager_dataset import FEATURE_COLUMNS
from ml.portfolio_manager_dataset import PROFILE
from ml.portfolio_manager_dataset import CleanPortfolioManagerDatasetBuilder
from ml.portfolio_manager_dataset import PortfolioManagerDatasetBuilder


def _write_fake_inputs(root: Path, with_prediction_parquet: bool = True) -> None:
    period = "2023-01-01_to_2026-05-31"
    log_dir = root / "logs" / "backtests" / PROFILE / period
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "code": "1001",
                "candidate_rank": 1,
                "score_rank": 1,
                "risk_adjusted_score": -0.10,
                "expected_return_10d": 0.01,
                "bad_entry_probability_10d": 0.20,
                "cash_before": 800000,
                "cash_after": 700000,
                "daily_buy_limit_remaining_before": 900000,
                "daily_buy_limit_remaining_after": 800000,
                "max_positions_remaining_before": 9,
                "final_amount": 100000,
                "final_shares": 100,
                "decision": "BUY",
            },
            {
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "code": "1002",
                "candidate_rank": 2,
                "score_rank": 2,
                "risk_adjusted_score": -0.20,
                "expected_return_10d": 0.02,
                "bad_entry_probability_10d": 0.80,
                "cash_before": 700000,
                "cash_after": 700000,
                "daily_buy_limit_remaining_before": 800000,
                "daily_buy_limit_remaining_after": 800000,
                "max_positions_remaining_before": 8,
                "final_amount": 0,
                "final_shares": 0,
                "decision": "SKIP",
                "skip_reason": "test",
            },
        ]
    ).to_csv(log_dir / "purchase_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "exit_date": "2023-01-20",
                "code": "1001",
                "net_profit": 12000,
                "holding_days": 10,
                "exit_reason": "test_exit",
            }
        ]
    ).to_csv(log_dir / "trades.csv", index=False)

    feature_dir = root / "data" / "ml" / "features"
    label_dir = root / "data" / "ml" / "labels"
    prediction_dir = root / "data" / "ml" / "walk_forward_predictions"
    feature_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2023-01-04",
                "code": "1001",
                "return_1d": 0.01,
                "return_3d": 0.03,
                "return_5d": 0.05,
                "return_10d": 0.10,
                "return_20d": 0.20,
                "ma5_gap": 0.01,
                "ma10_gap": 0.02,
                "ma25_gap": 0.03,
                "ma75_gap": 0.04,
                "ma5_slope": 0.01,
                "ma25_slope": 0.02,
                "body_ratio": 0.4,
                "upper_shadow_ratio": 0.1,
                "lower_shadow_ratio": 0.2,
                "close_position": 0.8,
                "gap_up_ratio": 0.01,
                "daily_range_ratio": 0.05,
                "volume": 10000,
                "turnover_value": 100000000,
                "volume_ratio_5d": 1.1,
                "volume_ratio_20d": 1.2,
                "turnover_ratio_5d": 1.3,
                "turnover_ratio_20d": 1.4,
                "topix_return_5d": 0.01,
                "relative_return_5d": 0.04,
                "EPS": 100,
                "days_to_earnings": 5,
                "is_near_earnings": True,
            },
            {
                "date": "2023-01-04",
                "code": "1002",
                "return_1d": -0.01,
                "return_3d": -0.02,
                "return_5d": -0.03,
                "return_10d": -0.04,
                "return_20d": -0.05,
                "volume": 20000,
                "turnover_value": 50000000,
            },
        ]
    ).to_parquet(feature_dir / "features_2023-01-04.parquet", index=False)
    pd.DataFrame(
        [
            {"date": "2023-01-04", "code": "1001", "future_5d_return": 0.04, "future_10d_return": 0.12},
            {"date": "2023-01-04", "code": "1002", "future_5d_return": -0.02, "future_10d_return": -0.07},
        ]
    ).to_parquet(label_dir / "labels_2023-01-04.parquet", index=False)
    if with_prediction_parquet:
        pd.DataFrame(
            [
                {
                    "date": "2023-01-04",
                    "code": "1001",
                    "expected_return_10d": 0.20,
                    "expected_max_return_20d": 0.30,
                    "swing_success_probability_20d": 0.90,
                    "bad_entry_probability_10d": 0.10,
                },
                {
                    "date": "2023-01-04",
                    "code": "1002",
                    "expected_return_10d": 0.01,
                    "expected_max_return_20d": 0.05,
                    "swing_success_probability_20d": 0.20,
                    "bad_entry_probability_10d": 0.80,
                },
            ]
        ).to_parquet(prediction_dir / "predictions_2023-01-04.parquet", index=False)


def test_portfolio_manager_dataset_joins_prediction_features_and_labels(tmp_path: Path) -> None:
    _write_fake_inputs(tmp_path)
    builder = PortfolioManagerDatasetBuilder(root=tmp_path)

    dataset = builder.build_dataset()
    first = dataset[dataset["code"].eq("1001")].iloc[0]

    assert len(dataset) == 2
    assert first["expected_return_10d"] == 0.20
    assert first["risk_adjusted_score"] == 0.20 - 0.5 * 0.10
    assert first["expected_max_return_20d"] == 0.30
    assert first["realized_return"] == 0.12
    assert first["positive_trade"] == True
    assert first["prediction_source"] == "prediction_parquet"
    assert first["actual_net_profit"] == 12000


def test_portfolio_manager_dataset_percentiles_and_leakage_contract(tmp_path: Path) -> None:
    _write_fake_inputs(tmp_path)
    dataset = PortfolioManagerDatasetBuilder(root=tmp_path).build_dataset()
    high = dataset[dataset["code"].eq("1001")].iloc[0]
    low = dataset[dataset["code"].eq("1002")].iloc[0]

    assert high["risk_adjusted_score_percentile_in_day"] > low["risk_adjusted_score_percentile_in_day"]
    assert high["high_conviction_target"] == True
    assert low["avoid_target"] == True
    assert set(AUDIT_COLUMNS).isdisjoint(set(FEATURE_COLUMNS))
    assert "realized_return" not in FEATURE_COLUMNS
    assert "actual_net_profit" not in FEATURE_COLUMNS


def test_portfolio_manager_dataset_uses_audit_prediction_snapshot_fallback(tmp_path: Path) -> None:
    _write_fake_inputs(tmp_path, with_prediction_parquet=False)
    dataset = PortfolioManagerDatasetBuilder(root=tmp_path).build_dataset()
    first = dataset[dataset["code"].eq("1001")].iloc[0]

    assert first["expected_return_10d"] == 0.01
    assert first["risk_adjusted_score"] == -0.10
    assert first["prediction_source"] == "purchase_audit_prediction_snapshot"
    assert pd.isna(first["expected_max_return_20d"])


def test_portfolio_manager_dataset_saves_outputs(tmp_path: Path) -> None:
    _write_fake_inputs(tmp_path)
    builder = PortfolioManagerDatasetBuilder(root=tmp_path)
    dataset = builder.build_dataset()
    paths = builder.save(dataset)
    summary = builder.summary(dataset)

    assert paths.dataset.exists()
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert summary["rows"] == 2
    assert summary["feature_count"] == len(FEATURE_COLUMNS)
    assert summary["prediction_join_rate"] == 1.0


def test_clean_portfolio_manager_dataset_excludes_backtest_state_features(tmp_path: Path) -> None:
    _write_fake_inputs(tmp_path)
    builder = CleanPortfolioManagerDatasetBuilder(root=tmp_path)

    dataset = builder.build_dataset()
    summary = builder.summary(dataset)
    first = dataset[dataset["code"].eq("1001")].iloc[0]

    assert first["prediction_source"] == "prediction_parquet"
    assert first["expected_max_return_20d"] == 0.30
    assert first["swing_success_probability_20d"] == 0.90
    assert "expected_max_return_percentile_in_day" in dataset.columns
    assert "day_avg_swing_success_probability_20d" in dataset.columns
    assert set(CLEAN_FORBIDDEN_FEATURE_COLUMNS).isdisjoint(set(CLEAN_FEATURE_COLUMNS))
    assert summary["forbidden_columns_in_features"] == []
    assert summary["expected_max_return_20d_non_null_rate"] == 1.0
    assert summary["swing_success_probability_20d_non_null_rate"] == 1.0
