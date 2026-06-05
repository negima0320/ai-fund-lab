from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.config import MODEL_FILENAMES
from ml.pipeline import DailyMLPipeline


class FakeFeatureBuilder:
    def __init__(self, calls: list[str], root: Path) -> None:
        self.calls = calls
        self.root = root

    def build_daily_features(self, target_date: str) -> pd.DataFrame:
        self.calls.append(f"build_features:{target_date}")
        return pd.DataFrame({"date": [pd.Timestamp(target_date)], "code": ["1001"]})

    def save_daily_features(self, df: pd.DataFrame, target_date: str) -> Path:
        self.calls.append(f"save_features:{target_date}")
        return self.root / f"features_{target_date}.parquet"


class FakePredictor:
    def __init__(self, calls: list[str], root: Path) -> None:
        self.calls = calls
        self.root = root

    def predict_daily(self, target_date: str) -> pd.DataFrame:
        self.calls.append(f"predict:{target_date}")
        return pd.DataFrame({"date": [pd.Timestamp(target_date)], "code": ["1001"]})

    def save_predictions(self, df: pd.DataFrame, target_date: str) -> Path:
        self.calls.append(f"save_predictions:{target_date}")
        return self.root / f"predictions_{target_date}.parquet"


class FakeLabelGenerator:
    def __init__(self, calls: list[str], root: Path) -> None:
        self.calls = calls
        self.root = root

    def update_available_labels(self, as_of_date: str) -> list[Path]:
        self.calls.append(f"update_labels:{as_of_date}")
        return [self.root / f"labels_{as_of_date}.parquet"]


class FakeCandidateExporter:
    def __init__(self, calls: list[str], root: Path) -> None:
        self.calls = calls
        self.root = root

    def build_candidates(
        self,
        target_date: str,
        top_n: int,
        min_turnover_value: float,
        max_bad_entry_probability: float | None,
    ) -> pd.DataFrame:
        self.calls.append(
            f"build_candidates:{target_date}:{top_n}:{int(min_turnover_value)}:{max_bad_entry_probability}"
        )
        return pd.DataFrame({"date": [pd.Timestamp(target_date)], "code": ["1001"]})

    def save_csv(self, df: pd.DataFrame, target_date: str) -> Path:
        self.calls.append(f"save_candidates_csv:{target_date}")
        return self.root / f"{target_date}.csv"

    def save_markdown(self, df: pd.DataFrame, target_date: str) -> Path:
        self.calls.append(f"save_candidates_md:{target_date}")
        return self.root / f"{target_date}.md"


def _touch_current_models(model_root: Path) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "feature_columns.json").write_text(
        '["topix_return_5d", "topix_return_10d", "topix_return_20d", '
        '"relative_return_10d", "EPS", "Sales_growth", "FEPS_growth"]',
        encoding="utf-8",
    )
    for filename in MODEL_FILENAMES.values():
        (model_root / filename).write_text("model", encoding="utf-8")


def _pipeline(tmp_path: Path, calls: list[str], model_root: Path) -> DailyMLPipeline:
    return DailyMLPipeline(
        feature_builder=FakeFeatureBuilder(calls, tmp_path / "features"),
        predictor=FakePredictor(calls, tmp_path / "predictions"),
        label_generator=FakeLabelGenerator(calls, tmp_path / "labels"),
        candidate_exporter=FakeCandidateExporter(calls, tmp_path / "candidates"),
        model_root=model_root,
    )


def test_daily_pipeline_runs_steps_in_order_when_models_exist(tmp_path) -> None:
    calls: list[str] = []
    model_root = tmp_path / "models" / "ml" / "current"
    _touch_current_models(model_root)

    result = _pipeline(tmp_path, calls, model_root).run_daily_pipeline("2026-06-01")

    assert calls == [
        "build_features:2026-06-01",
        "save_features:2026-06-01",
        "predict:2026-06-01",
        "save_predictions:2026-06-01",
        "build_candidates:2026-06-01:10:50000000:None",
        "save_candidates_csv:2026-06-01",
        "save_candidates_md:2026-06-01",
        "update_labels:2026-06-01",
    ]
    assert result["features_path"] == tmp_path / "features" / "features_2026-06-01.parquet"
    assert result["predictions_path"] == tmp_path / "predictions" / "predictions_2026-06-01.parquet"
    assert result["candidate_csv_path"] == tmp_path / "candidates" / "2026-06-01.csv"
    assert result["candidate_md_path"] == tmp_path / "candidates" / "2026-06-01.md"
    assert result["labels_paths"] == [tmp_path / "labels" / "labels_2026-06-01.parquet"]
    assert result["warnings"] == []


def test_daily_pipeline_skips_prediction_when_models_are_missing(tmp_path) -> None:
    calls: list[str] = []
    model_root = tmp_path / "models" / "ml" / "current"

    result = _pipeline(tmp_path, calls, model_root).run_daily_pipeline("2026-06-01")

    assert calls == [
        "build_features:2026-06-01",
        "save_features:2026-06-01",
        "update_labels:2026-06-01",
    ]
    assert result["predictions_path"] is None
    assert result["candidate_csv_path"] is None
    assert result["candidate_md_path"] is None
    assert result["labels_paths"] == [tmp_path / "labels" / "labels_2026-06-01.parquet"]
    assert result["warnings"] == [
        "current ML models are missing; skipped prediction",
        "prediction was skipped; skipped AI candidate export",
    ]


def test_daily_pipeline_skips_prediction_when_any_model_file_is_missing(tmp_path) -> None:
    calls: list[str] = []
    model_root = tmp_path / "models" / "ml" / "current"
    model_root.mkdir(parents=True)
    (model_root / "feature_columns.json").write_text("[]", encoding="utf-8")

    result = _pipeline(tmp_path, calls, model_root).run_daily_pipeline("2026-06-01")

    assert "predict:2026-06-01" not in calls
    assert result["predictions_path"] is None
    assert result["warnings"]


def test_daily_pipeline_can_disable_candidate_export(tmp_path) -> None:
    calls: list[str] = []
    model_root = tmp_path / "models" / "ml" / "current"
    _touch_current_models(model_root)

    result = _pipeline(tmp_path, calls, model_root).run_daily_pipeline(
        "2026-06-01",
        export_candidates=False,
    )

    assert "build_candidates:2026-06-01:10:50000000:None" not in calls
    assert result["candidate_csv_path"] is None
    assert result["candidate_md_path"] is None


def test_daily_pipeline_passes_candidate_export_options(tmp_path) -> None:
    calls: list[str] = []
    model_root = tmp_path / "models" / "ml" / "current"
    _touch_current_models(model_root)

    result = _pipeline(tmp_path, calls, model_root).run_daily_pipeline(
        "2026-06-01",
        candidate_top_n=5,
        min_turnover_value=100_000_000,
        max_bad_entry_probability=0.6,
    )

    assert "build_candidates:2026-06-01:5:100000000:0.6" in calls
    assert result["candidate_csv_path"] == tmp_path / "candidates" / "2026-06-01.csv"


def test_daily_pipeline_warns_when_current_model_is_not_enriched_v2(tmp_path) -> None:
    calls: list[str] = []
    model_root = tmp_path / "models" / "ml" / "current"
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "feature_columns.json").write_text('["close"]', encoding="utf-8")
    for filename in MODEL_FILENAMES.values():
        (model_root / filename).write_text("model", encoding="utf-8")

    result = _pipeline(tmp_path, calls, model_root).run_daily_pipeline("2026-06-01")

    assert any("does not look like enriched_v2" in warning for warning in result["warnings"])
    assert "predict:2026-06-01" in calls
