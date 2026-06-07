from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.phase5c_exit_ai_v2_dataset_builder import BuildOptions, Phase5CExitAIV2DatasetBuilder


def _write_base_dataset(root: Path, *, forbidden: bool = False) -> None:
    dataset_dir = root / "data" / "ml" / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2021-06-01", periods=30)
    rows = []
    for code in ["11110", "22220"]:
        for index, date in enumerate(dates):
            row = {
                "date": date.strftime("%Y-%m-%d"),
                "code": code,
                "close": 100 + index,
                "volume": 1000 + index,
                "return_5d": 0.01,
                "ma25_gap": 0.2,
                "scale_category": "mid",
                "future_5d_return": 0.03,
                "bad_entry_10d": False,
                "mostly_missing": None,
            }
            if index < 1:
                row["mostly_missing"] = 1.0
            if forbidden:
                row["selected_count_in_day"] = 2
            rows.append(row)
    pd.DataFrame(rows).to_parquet(dataset_dir / "ml_dataset.parquet", index=False)


def test_phase5c_dry_run_builds_api_only_labels_without_writing_dataset(tmp_path: Path) -> None:
    _write_base_dataset(tmp_path)

    builder = Phase5CExitAIV2DatasetBuilder(tmp_path)
    result = builder.build(BuildOptions(dry_run=True))

    assert result["metadata"]["dataset_written"] is False
    assert result["metadata"]["model_retraining_executed"] is False
    assert result["row_counts"]["rows_before_filter"] == 60
    assert result["row_counts"]["rows_after_horizon_filter"] == 20
    assert result["row_counts"]["final_feature_count"] >= 4
    assert "future_5d_return" not in result["schema"]["feature_columns"]
    assert "bad_entry_10d" not in result["schema"]["feature_columns"]
    assert "mostly_missing" not in result["schema"]["feature_columns"]
    assert result["leakage_audit"]["leakage_risk"] == "low"
    assert result["label_distribution"]["overall"]["future_return_5d_mean"] is not None
    assert not (tmp_path / "data/ml/exit_ai_v2/exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet").exists()


def test_phase5c_sample_mode_writes_sample_dataset_and_report(tmp_path: Path) -> None:
    _write_base_dataset(tmp_path)

    builder = Phase5CExitAIV2DatasetBuilder(tmp_path)
    result = builder.build(BuildOptions(dry_run=False, sample_rows=5))
    paths = builder.save_report(result)

    assert paths.dataset is not None
    assert paths.dataset.exists()
    sample = pd.read_parquet(paths.dataset)
    assert len(sample) == 5
    assert {"code", "as_of_date", "future_return_5d", "exit_quality_score", "split"}.issubset(sample.columns)
    assert paths.markdown.exists()
    assert paths.json.exists()


def test_phase5c_blocks_forbidden_feature_columns(tmp_path: Path) -> None:
    _write_base_dataset(tmp_path, forbidden=True)

    result = Phase5CExitAIV2DatasetBuilder(tmp_path).build(BuildOptions(dry_run=True))

    assert "selected_count_in_day" in result["leakage_audit"]["forbidden_columns_found"]
    assert result["leakage_audit"]["leakage_risk"] == "high"
    assert result["metadata"]["dataset_written"] is False
    assert result["recommended_next_phase"] == "Retraining deferred"
