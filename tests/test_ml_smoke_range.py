from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from ml.evaluator import PredictionEvaluator
from ml.range_evaluator import RangeSmokeRunner


def _load_smoke_range_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ml" / "smoke_ml_range.py"
    spec = importlib.util.spec_from_file_location("smoke_ml_range", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeFeatureBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[str] = []

    def build_daily_features(self, target_date: str) -> pd.DataFrame:
        self.calls.append(target_date)
        if target_date == "2026-05-09":
            return pd.DataFrame(columns=["date", "code", "close", "volume"])
        return pd.DataFrame(
            {
                "date": [pd.Timestamp(target_date), pd.Timestamp(target_date)],
                "code": ["1001", "1002"],
                "close": [100.0, 200.0],
                "volume": [1000, 2000],
            }
        )

    def save_daily_features(self, df: pd.DataFrame, target_date: str) -> Path:
        path = self.root / "features" / f"features_{target_date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path


class FakeLabelGenerator:
    def __init__(self, root: Path) -> None:
        self.root = root

    def generate_labels(self, target_date: str) -> pd.DataFrame:
        if target_date == "2026-05-09":
            return pd.DataFrame(columns=["date", "code"])
        return pd.DataFrame(
            {
                "date": [pd.Timestamp(target_date), pd.Timestamp(target_date)],
                "code": ["1001", "1002"],
                "entry_price": [101.0, 202.0],
                "future_5d_return": [0.02, -0.01],
                "future_10d_return": [0.06, -0.03],
                "upside_10d": [True, False],
                "bad_entry_10d": [False, True],
            }
        )

    def save_labels(self, df: pd.DataFrame, target_date: str) -> Path:
        path = self.root / "labels" / f"labels_{target_date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path


class FakeDatasetBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root

    def build_dataset(self, start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-05-10", "2026-05-11", "2026-05-14"]),
                "code": ["1001", "1001", "1002"],
                "close": [100.0, 101.0, 202.0],
                "volume": [1000, 1100, 2100],
                "entry_price": [101.0, 102.0, 203.0],
                "future_5d_return": [0.01, 0.02, -0.01],
                "future_10d_return": [0.05, 0.06, -0.02],
                "upside_10d": [True, True, False],
                "bad_entry_10d": [False, False, True],
            }
        )

    def split_by_time(self, df: pd.DataFrame, train_end: str, valid_end: str):
        train_cutoff = pd.Timestamp(train_end)
        valid_cutoff = pd.Timestamp(valid_end)
        train = df[df["date"] <= train_cutoff].copy()
        valid = df[(df["date"] > train_cutoff) & (df["date"] <= valid_cutoff)].copy()
        test = df[df["date"] > valid_cutoff].copy()
        return train, valid, test

    def save_dataset(self, df: pd.DataFrame, name: str) -> Path:
        path = self.root / "datasets" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path


class FakeModelTrainer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def train_all(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
        return {"models": {"one": object()}, "metrics": {"warnings": ["fake trainer warning"]}, "feature_columns": ["close"]}

    def save_models(self, models: dict, metrics: dict) -> Path:
        path = self.root / "models" / "archive" / "fake"
        path.mkdir(parents=True, exist_ok=True)
        return path


class FakePredictor:
    def __init__(self, root: Path) -> None:
        self.root = root

    def predict_daily(self, target_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [pd.Timestamp(target_date), pd.Timestamp(target_date)],
                "code": ["1001", "1002"],
                "expected_return_5d": [0.01, -0.01],
                "expected_return_10d": [0.05, -0.02],
                "upside_probability_10d": [0.8, 0.2],
                "bad_entry_probability_10d": [0.1, 0.5],
                "entry_risk_label": ["safe", "danger"],
                "ml_score": [12.0, -7.0],
            }
        )

    def save_predictions(self, df: pd.DataFrame, target_date: str) -> Path:
        path = self.root / "predictions" / f"predictions_{target_date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path


def test_range_runner_processes_days_saves_report_and_summarizes(tmp_path) -> None:
    runner = RangeSmokeRunner(
        feature_builder=FakeFeatureBuilder(tmp_path),
        label_generator=FakeLabelGenerator(tmp_path),
        dataset_builder=FakeDatasetBuilder(tmp_path),
        model_trainer=FakeModelTrainer(tmp_path),
        predictor=FakePredictor(tmp_path),
        evaluator=PredictionEvaluator(report_root=tmp_path / "reports"),
        report_root=tmp_path / "reports",
    )

    result = runner.run("2026-05-09", "2026-05-11", "2026-05-10", "2026-05-11")

    assert result["processed_dates"] == ["2026-05-10", "2026-05-11"]
    assert result["skipped_dates"] == [{"date": "2026-05-09", "reason": "features empty, labels empty"}]
    assert result["features_total_rows"] == 4
    assert result["labels_total_rows"] == 4
    assert result["dataset_rows"] == 3
    assert result["train_rows"] == 1
    assert result["valid_rows"] == 1
    assert result["prediction_rows_total"] == 4
    assert result["joined_evaluation_rows_total"] == 4
    assert result["risk_bad_entry_rates"]["safe"] == 0.0
    assert result["range_report_path"].exists()
    assert "fake trainer warning" in result["warnings"]


def test_smoke_range_cli_formats_summary(tmp_path) -> None:
    module = _load_smoke_range_module()

    class FakeRunner:
        def run(self, start_date: str, end_date: str, train_end: str, valid_end: str, top_n: int = 10) -> dict:
            return {
                "start_date": start_date,
                "end_date": end_date,
                "train_end": train_end,
                "valid_end": valid_end,
                "processed_dates": ["2026-05-10"],
                "skipped_dates": [{"date": "2026-05-09", "reason": "labels empty"}],
                "features_total_rows": 2,
                "labels_total_rows": 2,
                "dataset_rows": 2,
                "train_rows": 1,
                "valid_rows": 1,
                "test_rows": 0,
                "prediction_rows_total": 2,
                "joined_evaluation_rows_total": 2,
                "risk_bad_entry_rates": {"safe": 0.0, "danger": 1.0},
                "top_n_future_10d_return_mean": 0.03,
                "expected_vs_future_10d_corr": 0.5,
                "dataset_path": tmp_path / "dataset.parquet",
                "train_path": tmp_path / "train.parquet",
                "valid_path": tmp_path / "valid.parquet",
                "test_path": tmp_path / "test.parquet",
                "model_path": tmp_path / "models",
                "range_report_path": tmp_path / "report.md",
                "warnings": ["small smoke"],
            }

    result = module.run_smoke_range("2026-05-09", "2026-05-15", "2026-05-13", "2026-05-15", runner=FakeRunner())
    output = module.format_range_result(result)

    assert "processed_dates=2026-05-10" in output
    assert "skipped_dates=2026-05-09:labels empty" in output
    assert "dataset rows=2" in output
    assert "risk_bad_entry_rates=danger:1.000000,safe:0.000000" in output
    assert "warning=small smoke" in output
