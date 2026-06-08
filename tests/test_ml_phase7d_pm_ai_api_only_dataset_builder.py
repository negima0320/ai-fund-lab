from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.phase7d_pm_ai_api_only_dataset_builder import BuildOptions, Phase7DPMAIAPIOnlyDatasetBuilder


def _write_base_dataset(root: Path, *, selected_count: bool = False) -> None:
    dataset_dir = root / "data/ml/datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2021-06-01", periods=40)
    rows = []
    for code in ["11110", "22220"]:
        for index, date in enumerate(dates):
            row = {
                "date": date.strftime("%Y-%m-%d"),
                "code": code,
                "close": 100.0 + index,
                "volume": 1000 + index,
                "turnover_value": 100000 + index,
                "return_1d": 0.01,
                "return_5d": 0.02,
                "ma5_gap": 0.1,
                "ma25_gap": 0.2,
                "close_position": 0.7,
                "body_ratio": 0.5,
                "topix_return_5d": 0.01,
                "relative_return_5d": 0.01,
                "EPS": 10.0,
                "days_to_earnings": None,
                "PayoutRatioAnn": None,
                "risk_adjusted_score": 0.2,
                "expected_return_10d": 0.03,
                "candidate_count_in_day": 20,
                "max_positions_remaining_before": 3,
                "future_5d_return": 0.01,
                "future_10d_return": 0.02,
            }
            if selected_count:
                row["selected_count_in_day"] = 5
            rows.append(row)
    pd.DataFrame(rows).to_parquet(dataset_dir / "ml_dataset.parquet", index=False)


def test_phase7d_dry_run_builds_report_without_writing_dataset(tmp_path: Path) -> None:
    _write_base_dataset(tmp_path)

    builder = Phase7DPMAIAPIOnlyDatasetBuilder(tmp_path)
    result = builder.build(BuildOptions(dry_run=True))

    assert result["metadata"]["dataset_written"] is False
    assert result["row_counts"]["total_rows"] == 80
    assert result["row_counts"]["final_rows"] > 0
    assert result["row_counts"]["final_feature_count"] > 0
    assert "future_5d_return" not in result["schema"]["feature_columns"]
    assert "candidate_count_in_day" not in result["schema"]["feature_columns"]
    assert "max_positions_remaining_before" not in result["schema"]["feature_columns"]
    assert "days_to_earnings" not in result["schema"]["feature_columns"]
    assert "PayoutRatioAnn" not in result["schema"]["feature_columns"]
    assert {row["feature"] for row in result["dropped_high_missing_features"]} == {"days_to_earnings", "PayoutRatioAnn"}
    assert result["leakage_audit"]["leakage_risk"] == "low"
    assert not (tmp_path / "data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05_sample.parquet").exists()


def test_phase7d_sample_mode_writes_sample_dataset(tmp_path: Path) -> None:
    _write_base_dataset(tmp_path)

    builder = Phase7DPMAIAPIOnlyDatasetBuilder(tmp_path)
    result = builder.build(BuildOptions(dry_run=False, sample_rows=7))
    paths = builder.save_report(result)

    assert paths.dataset is not None
    assert paths.dataset.exists()
    sample = pd.read_parquet(paths.dataset)
    assert len(sample) == 7
    assert {"as_of_date", "code", "split", "future_5d_return", "high_conviction_target", "avoid_target"}.issubset(sample.columns)
    assert "candidate_count_in_day" not in sample.columns
    assert "max_positions_remaining_before" not in sample.columns


def test_phase7d_detects_selected_count_in_day_as_blocking(tmp_path: Path) -> None:
    _write_base_dataset(tmp_path, selected_count=True)

    result = Phase7DPMAIAPIOnlyDatasetBuilder(tmp_path).build(BuildOptions(dry_run=True))

    assert result["leakage_audit"]["selected_count_in_day_found"] is True
    assert result["leakage_audit"]["leakage_risk"] == "high"
    assert result["leakage_audit"]["ready_for_phase7e"] is False


def test_phase7d_saves_report(tmp_path: Path) -> None:
    _write_base_dataset(tmp_path)
    builder = Phase7DPMAIAPIOnlyDatasetBuilder(tmp_path)

    paths = builder.save_report(builder.build(BuildOptions(dry_run=True)))

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 7-D" in paths.markdown.read_text(encoding="utf-8")
