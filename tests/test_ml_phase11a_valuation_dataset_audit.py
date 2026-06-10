from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase11a_valuation_dataset_audit import Phase11AValuationDatasetAudit


def _write_fixture(root: Path) -> None:
    feature_dir = root / "data/ml/features"
    pred_dir = root / "data/ml/walk_forward_predictions"
    raw_dir = root / "data/raw"
    feature_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "date": "2023-01-04",
                "code": "11110",
                "close": 100,
                "volume": 1000,
                "turnover_value": 100000,
                "return_1d": 0.01,
                "return_3d": 0.02,
                "return_5d": 0.03,
                "return_10d": 0.04,
                "ma5_gap": 0.01,
                "ma25_gap": 0.02,
                "ma75_gap": 0.03,
                "ma25_slope": 0.001,
                "volume_ratio_5d": 1.5,
                "EPS": 10,
                "BPS": 100,
                "Sales_growth": 0.2,
                "OP_growth": 0.1,
                "EqAR": 0.5,
                "topix_return_5d": 0.01,
                "selected_count_in_day": 99,
                "pm_multiplier": 1.3,
                "future_leaky_feature": 0.7,
            },
            {
                "date": "2023-01-04",
                "code": "22220",
                "close": 200,
                "volume": 2000,
                "turnover_value": 200000,
                "return_1d": -0.01,
                "return_3d": -0.02,
                "return_5d": -0.03,
                "return_10d": -0.04,
                "ma5_gap": -0.01,
                "ma25_gap": -0.02,
                "ma75_gap": -0.03,
                "ma25_slope": -0.001,
                "volume_ratio_5d": 0.8,
                "EPS": 5,
                "BPS": 80,
                "Sales_growth": -0.1,
                "OP_growth": -0.2,
                "EqAR": 0.4,
                "topix_return_5d": 0.01,
            },
        ]
    ).to_parquet(feature_dir / "features_2023-01-04.parquet", index=False)

    pd.DataFrame(
        [
            {
                "date": "2023-01-04",
                "code": "11110",
                "expected_return_10d": 0.08,
                "bad_entry_probability_10d": 0.1,
                "expected_max_return_20d": 0.2,
                "swing_success_probability_20d": 0.7,
                "ml_score": 0.9,
            },
            {
                "date": "2023-01-04",
                "code": "22220",
                "expected_return_10d": 0.01,
                "bad_entry_probability_10d": 0.4,
                "expected_max_return_20d": 0.05,
                "swing_success_probability_20d": 0.2,
                "ml_score": 0.2,
            },
        ]
    ).to_parquet(pred_dir / "predictions_2023-01-04.parquet", index=False)

    for offset in range(0, 22):
        date = pd.Timestamp("2023-01-04") + pd.Timedelta(days=offset)
        rows = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "code": "11110",
                "open": 100 + offset,
                "high": 101 + offset * 2,
                "low": 99 + offset,
                "close": 100 + offset * 1.5,
                "volume": 1000,
                "turnover_value": 100000,
            },
            {
                "date": date.strftime("%Y-%m-%d"),
                "code": "22220",
                "open": 200 - offset,
                "high": 201 - offset,
                "low": 190 - offset * 2,
                "close": 200 - offset * 1.5,
                "volume": 2000,
                "turnover_value": 200000,
            },
        ]
        (raw_dir / f"prices_{date.strftime('%Y-%m-%d')}.json").write_text(
            json.dumps({"prices": rows}),
            encoding="utf-8",
        )


def test_phase11a_builds_labels_and_excludes_leaky_features(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    audit = Phase11AValuationDatasetAudit(tmp_path, start_date="2023-01-01", end_date="2023-01-31", save_dataset=False)
    dataset, _info = audit.build_dataset()
    features = audit.feature_columns(dataset)
    report = audit.build_report()

    assert "future_return_20d" in dataset.columns
    assert "future_max_return_20d" in dataset.columns
    assert "future_max_drawdown_20d" in dataset.columns
    assert "opportunity_value_20d" in dataset.columns
    assert "opportunity_top_decile_20d" in dataset.columns
    assert all(not column.startswith("future_") for column in features)
    assert "selected_count_in_day" not in features
    assert "pm_multiplier" not in features
    assert "future_leaky_feature" not in features
    assert report["leakage_checklist"]["selected_count_in_day_used"] is False
    assert report["leakage_checklist"]["current_pm_multiplier_used"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"


def test_phase11a_report_json_is_generated(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    audit = Phase11AValuationDatasetAudit(tmp_path, start_date="2023-01-01", end_date="2023-01-31", save_dataset=True)
    paths = audit.run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.dataset is not None and paths.dataset.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "11-A"
    assert loaded["metadata"]["historical_predictions_regenerated"] is False
    assert loaded["recommendation"]["recommended_primary_label"] == "opportunity_value_20d"
