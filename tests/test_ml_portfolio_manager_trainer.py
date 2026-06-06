from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.portfolio_manager_trainer import PHASE3A_EXCLUDED_FEATURES
from ml.portfolio_manager_dataset import CLEAN_FEATURE_COLUMNS
from ml.portfolio_manager_dataset import CLEAN_FORBIDDEN_FEATURE_COLUMNS
from ml.portfolio_manager_trainer import PortfolioManagerTrainer


def _fake_dataset() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2023-01-02", periods=36)
    weight_buckets = ["strong", "normal", "weak"]
    cash_buckets = ["aggressive", "normal", "defensive"]
    for index, date_value in enumerate(dates):
        score = (index % 9 - 4) / 20
        expected = 0.02 + score / 2
        bad_entry = 0.2 + (index % 5) * 0.12
        realized = score + (0.03 if index % 3 == 0 else -0.01)
        rows.append(
            {
                "signal_date": date_value,
                "code": f"{1000 + index}",
                "expected_return_10d": expected,
                "expected_max_return_20d": pd.NA,
                "swing_success_probability_20d": pd.NA,
                "bad_entry_probability_10d": bad_entry,
                "risk_adjusted_score": expected - 0.5 * bad_entry,
                "return_1d": score,
                "return_3d": score * 1.1,
                "return_5d": score * 1.2,
                "volume": 10000 + index,
                "turnover_value": 100000000 + index,
                "rank_in_day": 1 + index % 3,
                "score_rank_in_day": 1 + index % 3,
                "risk_adjusted_score_percentile_in_day": (index % 10 + 1) / 10,
                "candidate_count_in_day": 3,
                "day_avg_risk_adjusted_score": score / 2,
                "current_capital_utilization": 0.2,
                "current_positions_count": 2,
                "cash_before_ratio": 0.8,
                "realized_return": realized,
                "positive_trade": realized > 0,
                "high_conviction_target": index % 3 == 0,
                "avoid_target": index % 3 == 2,
                "ideal_weight_bucket": weight_buckets[index % 3],
                "ideal_cash_reserve_bucket": cash_buckets[index % 3],
                "future_5d_return": realized / 2,
                "future_10d_return": realized,
                "decision": "BUY",
                "actual_net_profit": realized * 100000,
            }
        )
    return pd.DataFrame(rows)


def test_portfolio_manager_trainer_time_split_and_feature_exclusion() -> None:
    trainer = PortfolioManagerTrainer()
    df = _fake_dataset()

    train, test = trainer.split_by_time(df, train_end="2023-01-31", test_start="2023-02-01", test_end="2023-03-31")
    features = trainer.extract_feature_columns(train)

    assert train["signal_date"].max() <= pd.Timestamp("2023-01-31")
    assert test["signal_date"].min() >= pd.Timestamp("2023-02-01")
    assert "expected_return_10d" in features
    assert "risk_adjusted_score" in features
    assert "realized_return" not in features
    assert "decision" not in features
    assert all(column not in features for column in PHASE3A_EXCLUDED_FEATURES)


def test_portfolio_manager_trainer_trains_small_dataset_and_saves(tmp_path: Path) -> None:
    trainer = PortfolioManagerTrainer(model_root=tmp_path / "models", report_root=tmp_path / "reports")
    df = _fake_dataset()
    dataset_path = tmp_path / "portfolio.parquet"
    df.to_parquet(dataset_path, index=False)
    loaded = trainer.load_dataset(dataset_path)
    train, test = trainer.split_by_time(loaded, train_end="2023-01-31", test_start="2023-02-01", test_end="2023-03-31")

    result = trainer.train_all(train, test)
    metadata = trainer.build_metadata(dataset_path, train, test, result["feature_columns"])
    model_dir = trainer.save_models(result, metadata)
    report = trainer.build_report(dataset_path, train, test, result, metadata)
    paths = trainer.save_report(report)

    assert result["feature_columns"]
    assert "ideal_weight_bucket_classification" in result["models"]
    assert "realized_return_regression" in result["models"]
    assert (model_dir / "feature_columns.json").exists()
    assert (model_dir / "metrics.json").exists()
    assert (model_dir / "model_metadata.json").exists()
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert report["decile_analysis"]["predicted_realized_return"]


def test_portfolio_manager_trainer_clean_features_do_not_use_forbidden_columns() -> None:
    trainer = PortfolioManagerTrainer(feature_candidates=CLEAN_FEATURE_COLUMNS, excluded_features=[])
    df = _fake_dataset()
    df["cash_before"] = 999999
    df["actual_shares"] = 100
    df["expected_max_return_20d"] = 0.10
    df["swing_success_probability_20d"] = 0.50

    features = trainer.extract_feature_columns(df)

    assert "expected_max_return_20d" in features
    assert "swing_success_probability_20d" in features
    assert set(CLEAN_FORBIDDEN_FEATURE_COLUMNS).isdisjoint(set(features))
    assert "realized_return" not in features
