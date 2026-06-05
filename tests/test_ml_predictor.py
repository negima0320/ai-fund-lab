from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.predictor import Predictor


class FakeRegressionModel:
    def __init__(self, value: float) -> None:
        self.value = value
        self.seen_columns: list[str] = []
        self.seen_missing = None

    def predict(self, features: pd.DataFrame) -> list[float]:
        self.seen_columns = list(features.columns)
        self.seen_missing = bool(features["missing_feature"].isna().all()) if "missing_feature" in features.columns else None
        return [self.value for _ in range(len(features))]


class FakeProbabilityModel:
    def __init__(self, probabilities: list[float], use_proba: bool = True) -> None:
        self.probabilities = probabilities
        self.use_proba = use_proba
        self.seen_columns: list[str] = []

    def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
        if not self.use_proba:
            raise AttributeError("predict_proba disabled")
        self.seen_columns = list(features.columns)
        return [[1 - value, value] for value in self.probabilities[: len(features)]]

    def predict(self, features: pd.DataFrame) -> list[float]:
        self.seen_columns = list(features.columns)
        return self.probabilities[: len(features)]


class FakePredictor(Predictor):
    def __init__(self, *args, fake_models: dict, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fake_models = fake_models

    def _load_model(self, path: Path):
        return self.fake_models[path.name]


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-06-01"), "code": "1001", "close": 100.0, "volume": 1000, "extra_feature": 999},
            {"date": pd.Timestamp("2026-06-01"), "code": "1002", "close": 200.0, "volume": 2000, "extra_feature": 999},
            {"date": pd.Timestamp("2026-06-01"), "code": "1003", "close": 300.0, "volume": 3000, "extra_feature": 999},
        ]
    )


def _predictor(tmp_path: Path, fake_models: dict) -> FakePredictor:
    model_root = tmp_path / "models" / "ml" / "current"
    model_root.mkdir(parents=True)
    (model_root / "feature_columns.json").write_text(json.dumps(["close", "volume", "missing_feature"]), encoding="utf-8")
    for filename in fake_models:
        (model_root / filename).write_text("fake", encoding="utf-8")
    return FakePredictor(
        feature_root=tmp_path / "data" / "ml" / "features",
        model_root=model_root,
        prediction_root=tmp_path / "data" / "ml" / "predictions",
        fake_models=fake_models,
    )


def test_predict_daily_outputs_required_columns_and_score(monkeypatch, tmp_path) -> None:
    fake_models = {
        "future_5d_return_regression.joblib": FakeRegressionModel(0.03),
        "future_10d_return_regression.joblib": FakeRegressionModel(0.08),
        "future_max_return_10d_regression.joblib": FakeRegressionModel(0.12),
        "future_max_return_20d_regression.joblib": FakeRegressionModel(0.20),
        "upside_10d_classification.joblib": FakeProbabilityModel([0.7, 0.6, 0.5]),
        "bad_entry_10d_classification.joblib": FakeProbabilityModel([0.1, 0.3, 0.5]),
        "future_swing_success_20d_classification.joblib": FakeProbabilityModel([0.9, 0.8, 0.7]),
    }
    predictor = _predictor(tmp_path, fake_models)
    monkeypatch.setattr(pd, "read_parquet", lambda path: _features())

    df = predictor.predict_daily("2026-06-01")

    assert df.columns.tolist() == [
        "date",
        "code",
        "market",
        "sector_name",
        "scale_category",
        "margin_category",
        "credit_category",
        "expected_return_5d",
        "expected_return_10d",
        "upside_probability_10d",
        "bad_entry_probability_10d",
        "expected_max_return_10d",
        "expected_max_return_20d",
        "swing_success_probability_20d",
        "entry_risk_label",
        "ml_score",
    ]
    assert df["market"].isna().all()
    assert df["entry_risk_label"].tolist() == ["safe", "watch", "danger"]
    assert df.loc[0, "ml_score"] == pytest.approx(0.08 * 100 + 0.7 * 10 - 0.1 * 15)


def test_predict_daily_aligns_missing_and_extra_features(monkeypatch, tmp_path) -> None:
    regression = FakeRegressionModel(0.03)
    fake_models = {
        "future_5d_return_regression.joblib": regression,
        "future_10d_return_regression.joblib": FakeRegressionModel(0.08),
        "future_max_return_10d_regression.joblib": FakeRegressionModel(0.12),
        "future_max_return_20d_regression.joblib": FakeRegressionModel(0.20),
        "upside_10d_classification.joblib": FakeProbabilityModel([0.7, 0.6, 0.5]),
        "bad_entry_10d_classification.joblib": FakeProbabilityModel([0.1, 0.3, 0.5]),
        "future_swing_success_20d_classification.joblib": FakeProbabilityModel([0.9, 0.8, 0.7]),
    }
    predictor = _predictor(tmp_path, fake_models)
    monkeypatch.setattr(pd, "read_parquet", lambda path: _features())

    predictor.predict_daily("2026-06-01")

    assert regression.seen_columns == ["close", "volume", "missing_feature"]
    assert regression.seen_missing is True


def test_predict_probability_falls_back_to_predict(monkeypatch, tmp_path) -> None:
    fake_models = {
        "future_5d_return_regression.joblib": FakeRegressionModel(0.03),
        "future_10d_return_regression.joblib": FakeRegressionModel(0.08),
        "future_max_return_10d_regression.joblib": FakeRegressionModel(0.12),
        "future_max_return_20d_regression.joblib": FakeRegressionModel(0.20),
        "upside_10d_classification.joblib": FakeProbabilityModel([0.7, 0.6, 0.5]),
        "bad_entry_10d_classification.joblib": FakeProbabilityModel([0.2, 0.2, 0.2], use_proba=False),
        "future_swing_success_20d_classification.joblib": FakeProbabilityModel([0.9, 0.8, 0.7]),
    }
    predictor = _predictor(tmp_path, fake_models)
    monkeypatch.setattr(pd, "read_parquet", lambda path: _features())

    df = predictor.predict_daily("2026-06-01")

    assert df["bad_entry_probability_10d"].tolist() == [0.2, 0.2, 0.2]


def test_save_predictions_writes_parquet_path(monkeypatch, tmp_path) -> None:
    predictor = Predictor(prediction_root=tmp_path / "data" / "ml" / "predictions")
    df = pd.DataFrame({"date": [pd.Timestamp("2026-06-01")], "code": ["1001"]})
    calls = {}

    def fake_to_parquet(_self: pd.DataFrame, path: Path, index: bool = False) -> None:
        calls["path"] = path
        calls["index"] = index
        path.write_text("parquet", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    path = predictor.save_predictions(df, "2026-06-01")

    assert path == tmp_path / "data" / "ml" / "predictions" / "predictions_2026-06-01.parquet"
    assert calls == {"path": path, "index": False}
    assert path.exists()
