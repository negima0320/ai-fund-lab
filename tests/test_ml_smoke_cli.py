from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ml" / "smoke_ml_pipeline.py"
    spec = importlib.util.spec_from_file_location("smoke_ml_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakePipeline:
    def __init__(self, root: Path, predictions: bool = False) -> None:
        self.root = root
        self.predictions = predictions

    def run_daily_pipeline(self, target_date: str) -> dict:
        feature_path = self.root / "features" / f"features_{target_date}.parquet"
        label_path = self.root / "labels" / f"labels_{target_date}.parquet"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        feature_path.write_text("feature", encoding="utf-8")
        label_path.write_text("label", encoding="utf-8")
        prediction_path = None
        warnings = ["current ML models are missing; skipped prediction"]
        if self.predictions:
            prediction_path = self.root / "predictions" / f"predictions_{target_date}.parquet"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text("prediction", encoding="utf-8")
            warnings = []
        return {
            "features_path": feature_path,
            "predictions_path": prediction_path,
            "labels_paths": [label_path],
            "warnings": warnings,
        }


class FakeLabelGenerator:
    def __init__(self, rows: int = 2) -> None:
        self.rows = rows
        self.calls: list[str] = []

    def generate_labels(self, target_date: str) -> pd.DataFrame:
        self.calls.append(target_date)
        if self.rows == 0:
            return pd.DataFrame(columns=["date", "code", "entry_price"])
        return pd.DataFrame(
            {
                "date": [target_date] * self.rows,
                "code": [str(index) for index in range(self.rows)],
                "entry_price": [100.0] * self.rows,
            }
        )


def test_smoke_cli_formats_pipeline_summary_with_prediction_skip(tmp_path) -> None:
    module = _load_smoke_module()

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        if "features" in path.parts:
            return pd.DataFrame({"date": ["2026-05-29"], "code": ["1001"], "close": [100]})
        return pd.DataFrame({"date": ["2026-05-19"], "code": ["1001"]})

    label_generator = FakeLabelGenerator(rows=2)
    result = module.run_smoke(
        "2026-05-29",
        pipeline=FakePipeline(tmp_path),
        label_generator=label_generator,
        read_parquet=fake_read_parquet,
    )
    output = module.format_smoke_result(result)

    assert "target_date=2026-05-29" in output
    assert "features rows=1 columns=3" in output
    assert "updated_labels path=" in output
    assert "rows=1 columns=2" in output
    assert "target_date_label_check rows=2 columns=3" in output
    assert "predictions_path=skipped" in output
    assert "warning=current ML models are missing; skipped prediction" in output
    assert label_generator.calls == ["2026-05-29"]


def test_smoke_cli_formats_prediction_summary_when_present(tmp_path) -> None:
    module = _load_smoke_module()

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        if "predictions" in path.parts:
            return pd.DataFrame({"date": ["2026-05-29"], "code": ["1001"], "ml_score": [10.0]})
        return pd.DataFrame({"date": ["2026-05-29"], "code": ["1001"]})

    result = module.run_smoke(
        "2026-05-29",
        pipeline=FakePipeline(tmp_path, predictions=True),
        label_generator=FakeLabelGenerator(rows=1),
        read_parquet=fake_read_parquet,
    )
    output = module.format_smoke_result(result)

    assert "predictions_path=" in output
    assert "predictions_path=skipped" not in output
    assert "predictions rows=1 columns=3" in output
    assert result["warnings"] == []


def test_smoke_cli_target_date_label_check_empty_does_not_crash(tmp_path) -> None:
    module = _load_smoke_module()

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        return pd.DataFrame({"date": ["2026-05-29"], "code": ["1001"]})

    result = module.run_smoke(
        "2026-05-29",
        pipeline=FakePipeline(tmp_path),
        label_generator=FakeLabelGenerator(rows=0),
        read_parquet=fake_read_parquet,
    )
    output = module.format_smoke_result(result)

    assert "target_date_label_check rows=0 columns=0" in output


def test_smoke_cli_target_date_label_check_does_not_save_parquet(tmp_path) -> None:
    module = _load_smoke_module()

    class NoSaveLabelGenerator(FakeLabelGenerator):
        def save_labels(self, *_args, **_kwargs):
            raise AssertionError("smoke target_date_label_check must not save labels")

    result = module.run_smoke(
        "2026-05-29",
        pipeline=FakePipeline(tmp_path),
        label_generator=NoSaveLabelGenerator(rows=1),
        read_parquet=lambda path: pd.DataFrame({"date": ["2026-05-29"], "code": ["1001"]}),
    )

    assert result["target_date_label_check"] == {"rows": 1, "columns": 3}


def test_smoke_cli_reports_pipeline_failure_without_raising(tmp_path) -> None:
    module = _load_smoke_module()

    class BrokenPipeline:
        def run_daily_pipeline(self, target_date: str) -> dict:
            raise RuntimeError("missing cache")

    result = module.run_smoke("2026-05-29", pipeline=BrokenPipeline())

    assert result["target_date"] == "2026-05-29"
    assert result["features_path"] is None
    assert result["warnings"] == ["pipeline failed: missing cache"]
