from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.exit_model_trainer import ExitModelTrainer


class FakeRegressor:
    def fit(self, x_train, y_train, eval_set=None):
        self.feature_columns = list(x_train.columns)
        self.mean = float(y_train.mean())
        return self

    def predict(self, x_valid):
        base = [self.mean for _ in range(len(x_valid))]
        if "unrealized_return" in x_valid.columns:
            return [value + float(extra) * 0.1 for value, extra in zip(base, x_valid["unrealized_return"])]
        return base


class FakeClassifier:
    def fit(self, x_train, y_train, eval_set=None):
        self.feature_columns = list(x_train.columns)
        self.base = min(0.9, max(0.1, float(y_train.mean())))
        return self

    def predict_proba(self, x_valid):
        values = []
        for _, row in x_valid.iterrows():
            probability = self.base + float(row.get("unrealized_return", 0.0))
            probability = min(0.95, max(0.05, probability))
            values.append([1 - probability, probability])
        return values


class FakeExitTrainer(ExitModelTrainer):
    def _make_regression_model(self):
        return FakeRegressor()

    def _make_classification_model(self):
        return FakeClassifier()

    def _dump_model(self, model, path: Path) -> None:
        path.write_text(",".join(model.feature_columns), encoding="utf-8")


def _dataset() -> pd.DataFrame:
    rows = []
    for index in range(20):
        current_date = pd.Timestamp("2025-12-20") + pd.Timedelta(days=index)
        future_5d = -0.02 + index * 0.003
        future_10d = -0.03 + index * 0.004
        rows.append(
            {
                "trade_id": f"t{index}",
                "code": f"10{index:02d}",
                "entry_date": "2025-12-01",
                "current_date": current_date,
                "actual_exit_date": "2026-01-31",
                "holding_days": index % 5 + 1,
                "entry_price": 100.0,
                "current_close": 100.0 + index,
                "unrealized_return": index / 100.0,
                "max_unrealized_return_so_far": index / 100.0,
                "min_unrealized_return_so_far": -0.01,
                "drawdown_from_peak": 0.0,
                "remaining_days_to_actual_exit": 10 - index,
                "expected_return_10d": 0.01 * index,
                "expected_max_return_20d": 0.02 * index,
                "swing_success_probability_20d": min(0.9, 0.1 + index / 30),
                "bad_entry_probability_10d": max(0.05, 0.8 - index / 30),
                "risk_adjusted_score": 0.01 * index - 0.5 * max(0.05, 0.8 - index / 30),
                "volume": 1000.0 + index,
                "turnover_value": 100000.0 + index,
                "return_5d": 0.001 * index,
                "return_10d": 0.002 * index,
                "ma25_gap": 0.003 * index,
                "daily_range_ratio": 0.01,
                "future_remaining_return_5d": future_5d,
                "future_remaining_return_10d": future_10d,
                "hold_better_5d": future_5d > 0,
                "hold_better_10d": future_10d > 0,
                "should_exit_now_5d": future_5d < 0,
                "avoid_loss_5d": index < 5,
            }
        )
    return pd.DataFrame(rows)


def test_exit_model_trainer_time_split_and_excludes_remaining_days(tmp_path: Path) -> None:
    trainer = FakeExitTrainer(model_root=tmp_path / "models", report_root=tmp_path / "reports")
    train, valid = trainer.split_by_time(_dataset(), train_end="2025-12-31", valid_start="2026-01-01", valid_end="2026-01-31")
    result = trainer.train_all(train, valid, include_remaining_days=False)

    assert len(train) == 12
    assert len(valid) == 8
    assert "remaining_days_to_actual_exit" not in result["feature_columns"]
    assert sorted(result["models"]) == [
        "avoid_loss_5d_classification",
        "future_remaining_return_10d_regression",
        "future_remaining_return_5d_regression",
        "hold_better_5d_classification",
        "should_exit_now_5d_classification",
    ]
    assert result["metrics"]["future_remaining_return_5d_regression"]["decile_analysis"]


def test_exit_model_trainer_can_compare_with_remaining_days(tmp_path: Path) -> None:
    trainer = FakeExitTrainer(model_root=tmp_path / "models", report_root=tmp_path / "reports")
    train, valid = trainer.split_by_time(_dataset(), train_end="2025-12-31", valid_start="2026-01-01", valid_end="2026-01-31")

    comparison = trainer.compare_feature_sets(train, valid)

    assert comparison["selected_feature_set"] == "without_remaining_days"
    assert "remaining_days_to_actual_exit" not in comparison["without_remaining_days"]["feature_columns"]
    assert "remaining_days_to_actual_exit" in comparison["with_remaining_days"]["feature_columns"]


def test_exit_model_trainer_saves_models_and_report(tmp_path: Path) -> None:
    trainer = FakeExitTrainer(model_root=tmp_path / "models", report_root=tmp_path / "reports")
    train, valid = trainer.split_by_time(_dataset(), train_end="2025-12-31", valid_start="2026-01-01", valid_end="2026-01-31")
    result = trainer.train_all(train, valid, include_remaining_days=False)
    metadata = trainer.build_metadata(train, valid, result["feature_columns"])

    model_dir = trainer.save_models(result, metadata)
    report = {
        "dataset_path": "dummy.parquet",
        "model_dir": str(model_dir),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "selected_feature_set": "without_remaining_days",
        "feature_count": len(result["feature_columns"]),
        "metrics": result["metrics"],
        "decile_analysis": {name: payload["decile_analysis"] for name, payload in result["metrics"].items()},
        "feature_set_comparison": trainer.compare_feature_sets(train, valid),
    }
    paths = trainer.save_report(report)

    assert (model_dir / "feature_columns.json").exists()
    assert (model_dir / "metrics.json").exists()
    assert (model_dir / "model_metadata.json").exists()
    assert len(list(model_dir.glob("*.joblib"))) == 5
    assert paths.markdown.exists()
    assert paths.json.exists()
