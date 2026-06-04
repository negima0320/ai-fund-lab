from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.model_trainer import ModelTrainer


class FakeRegressor:
    def fit(self, x_train, y_train, eval_set=None):
        self.feature_columns = list(x_train.columns)
        self.prediction = float(y_train.mean())
        return self

    def predict(self, x_valid):
        return [self.prediction for _ in range(len(x_valid))]


class FakeClassifier:
    def fit(self, x_train, y_train, eval_set=None):
        self.feature_columns = list(x_train.columns)
        self.probability = min(0.9, max(0.1, float(y_train.mean())))
        return self

    def predict_proba(self, x_valid):
        return [[1 - self.probability, self.probability] for _ in range(len(x_valid))]


class FakeTrainer(ModelTrainer):
    def _make_regression_model(self):
        return FakeRegressor()

    def _make_classification_model(self):
        return FakeClassifier()

    def _dump_model(self, model, path: Path) -> None:
        path.write_text(",".join(model.feature_columns), encoding="utf-8")


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-01"),
                "code": "1001",
                "close": 100.0,
                "volume": 1000.0,
                "turnover_value": 100000.0,
                "return_1d": 0.01,
                "ma5_gap": 0.02,
                "entry_price": 101.0,
                "future_5d_return": 0.04,
                "future_10d_return": 0.08,
                "upside_10d": True,
                "bad_entry_10d": False,
            },
            {
                "date": pd.Timestamp("2026-05-02"),
                "code": "1002",
                "close": 90.0,
                "volume": 2000.0,
                "turnover_value": 180000.0,
                "return_1d": -0.01,
                "ma5_gap": -0.02,
                "entry_price": 91.0,
                "future_5d_return": -0.03,
                "future_10d_return": -0.06,
                "upside_10d": False,
                "bad_entry_10d": True,
            },
        ]
    )


def test_train_all_trains_four_models_and_excludes_label_columns(tmp_path) -> None:
    trainer = FakeTrainer(archive_root=tmp_path / "archive", current_root=tmp_path / "current", timestamp="20260501_120000")
    result = trainer.train_all(_dataset(), _dataset())

    assert sorted(result["models"]) == [
        "bad_entry_10d_classification",
        "future_10d_return_regression",
        "future_5d_return_regression",
        "upside_10d_classification",
    ]
    assert "date" not in result["feature_columns"]
    assert "code" not in result["feature_columns"]
    assert "future_10d_return" not in result["feature_columns"]
    assert "upside_10d" not in result["feature_columns"]
    assert "close" in result["feature_columns"]


def test_save_models_writes_archive_and_current_files(tmp_path) -> None:
    trainer = FakeTrainer(archive_root=tmp_path / "archive", current_root=tmp_path / "current", timestamp="20260501_120000")
    result = trainer.train_all(_dataset(), _dataset())

    archive_dir = trainer.save_models(result["models"], result["metrics"])

    expected_files = {
        "future_5d_return_regression.joblib",
        "future_10d_return_regression.joblib",
        "upside_10d_classification.joblib",
        "bad_entry_10d_classification.joblib",
        "feature_columns.json",
        "metrics.json",
    }
    assert archive_dir == tmp_path / "archive" / "20260501_120000"
    assert {path.name for path in archive_dir.iterdir()} == expected_files
    assert {path.name for path in (tmp_path / "current").iterdir()} == expected_files


def test_metrics_include_regression_and_classification_scores(tmp_path) -> None:
    trainer = FakeTrainer(archive_root=tmp_path / "archive", current_root=tmp_path / "current")
    result = trainer.train_all(_dataset(), _dataset())

    assert set(result["metrics"]["future_5d_return_regression"]) == {"rmse", "mae"}
    assert set(result["metrics"]["upside_10d_classification"]) == {"auc", "accuracy", "precision", "recall"}


def test_category_columns_are_optional_and_converted_when_present(tmp_path) -> None:
    train = _dataset()
    valid = _dataset()
    train["market"] = ["Prime", "Standard"]
    valid["market"] = ["Prime", "Standard"]
    trainer = FakeTrainer(archive_root=tmp_path / "archive", current_root=tmp_path / "current")

    prepared_train, prepared_valid, feature_columns = trainer._prepare_frames(train, valid)

    assert str(prepared_train["market"].dtype) == "category"
    assert str(prepared_valid["market"].dtype) == "category"
    assert "market" in feature_columns


def test_load_dataset_reads_parquet(monkeypatch, tmp_path) -> None:
    expected = _dataset()

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        assert path == tmp_path / "train.parquet"
        return expected

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    loaded = ModelTrainer().load_dataset(tmp_path / "train.parquet")

    assert loaded is expected
