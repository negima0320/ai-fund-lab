from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_v3_dataset_builder import (
    CONDITIONAL_RELATIVE_FEATURES,
    FORBIDDEN_TOKENS,
    LABEL_COLUMNS,
    PMAIV3BuildOptions,
    PMAIV3CleanDatasetBuilder,
)


PROFILE = "rookie_dealer_02_v2_82_cap38"


def _base_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = {
        "11110": [100, 102, 105, 104, 106, 108, 107, 109, 110, 111, 112, 113],
        "22220": [100, 99, 98, 101, 100, 103, 102, 101, 104, 105, 106, 107],
        "33330": [100, 101, 99, 98, 97, 96, 98, 99, 100, 101, 102, 103],
        "44440": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    }
    dates = pd.bdate_range("2023-01-02", periods=12)
    for code, closes in specs.items():
        for idx, date in enumerate(dates):
            close = float(closes[idx])
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "code": code,
                    "close": close,
                    "volume": 100000 + idx,
                    "turnover_value": 80_000_000 if code != "44440" else 10_000_000,
                    "return_1d": 0.01,
                    "return_3d": 0.02,
                    "return_5d": 0.03,
                    "return_10d": 0.04,
                    "return_20d": 0.05,
                    "ma5_gap": 0.01,
                    "ma10_gap": 0.01,
                    "ma25_gap": 0.02,
                    "ma75_gap": 0.03,
                    "ma5_slope": 0.01,
                    "ma25_slope": 0.01,
                    "volume_ratio_5d": 1.2,
                    "volume_ratio_20d": 1.3,
                    "turnover_ratio_5d": 1.1,
                    "turnover_ratio_20d": 1.2,
                    "body_ratio": 0.5,
                    "upper_shadow_ratio": 0.1,
                    "lower_shadow_ratio": 0.1,
                    "gap_up_ratio": 0.01,
                    "daily_range_ratio": 0.03,
                    "EPS": 100.0,
                    "BPS": 1000.0,
                    "EqAR": 0.5,
                    "Sales_growth": 0.1,
                    "OP_growth": 0.1,
                    "NP_growth": 0.1,
                    "FEPS_growth": 0.1,
                    "FSales_growth": 0.1,
                    "FOP_growth": 0.1,
                    "PayoutRatioAnn": 0.3,
                    "days_to_earnings": 10,
                    "days_after_earnings": 20,
                    "is_near_earnings": False,
                    "topix_return_5d": 0.01 + idx * 0.001,
                    "topix_return_10d": 0.02,
                    "topix_return_20d": 0.04,
                    "relative_return_5d": 0.02,
                    "relative_return_10d": 0.02,
                    "relative_return_20d": 0.02,
                    "future_5d_return": (float(closes[min(idx + 5, len(closes) - 1)]) / close) - 1.0,
                    "future_10d_return": (float(closes[min(idx + 10, len(closes) - 1)]) / close) - 1.0,
                    "future_max_return_10d": max(closes[idx : min(idx + 11, len(closes))]) / close - 1.0,
                }
            )
    return rows


def _write_fixture(root: Path) -> None:
    base_path = root / "data/ml/datasets/ml_dataset.parquet"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_base_rows()).to_parquet(base_path, index=False)

    pred_dir = root / "data/ml/walk_forward_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "code": "11110",
                "expected_return_5d": 0.04,
                "expected_return_10d": 0.08,
                "expected_max_return_20d": 0.12,
                "swing_success_probability_20d": 0.7,
                "bad_entry_probability_10d": 0.1,
                "ml_score": 0.9,
            },
            {
                "date": "2023-01-02",
                "code": "22220",
                "expected_return_5d": 0.02,
                "expected_return_10d": 0.04,
                "expected_max_return_20d": 0.08,
                "swing_success_probability_20d": 0.5,
                "bad_entry_probability_10d": 0.2,
                "ml_score": 0.7,
            },
            {
                "date": "2023-01-02",
                "code": "33330",
                "expected_return_5d": 0.01,
                "expected_return_10d": 0.03,
                "expected_max_return_20d": 0.05,
                "swing_success_probability_20d": 0.4,
                "bad_entry_probability_10d": 0.3,
                "ml_score": 0.5,
            },
            {
                "date": "2023-01-02",
                "code": "44440",
                "expected_return_5d": 0.20,
                "expected_return_10d": 0.30,
                "expected_max_return_20d": 0.40,
                "swing_success_probability_20d": 0.9,
                "bad_entry_probability_10d": 0.01,
                "ml_score": 0.99,
            },
        ]
    ).to_parquet(pred_dir / "predictions_2023-01-02.parquet", index=False)

    for path in [
        root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean",
        root / "models/ml/exit/current_v2_66",
        root / "config/profiles",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    (root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json").write_text(
        json.dumps({"model": "current_pm"}),
        encoding="utf-8",
    )
    (root / "models/ml/exit/current_v2_66/model_metadata.json").write_text(
        json.dumps({"model": "current_exit"}),
        encoding="utf-8",
    )
    (root / f"config/profiles/{PROFILE}.yaml").write_text(
        f"profile_id: {PROFILE}\nportfolio_manager_ai_sizing:\n  enabled: true\n",
        encoding="utf-8",
    )


def test_phase9b_dataset_builder_generates_clean_dataset(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    builder = PMAIV3CleanDatasetBuilder(tmp_path)
    report = builder.build(PMAIV3BuildOptions(start_date="2023-01-01", end_date="2023-01-31", top_n=2))
    dataset = report["_dataset"]

    assert report["metadata"]["training_executed"] is False
    assert report["metadata"]["backtest_executed"] is False
    assert report["dataset_summary"]["row_count"] == 2
    assert report["dataset_summary"]["date_min"] == "2023-01-02"
    assert report["dataset_summary"]["code_count"] == 2
    assert set(LABEL_COLUMNS).issubset(set(report["label_columns"]))
    assert set(CONDITIONAL_RELATIVE_FEATURES).issubset(set(report["conditional_feature_columns"]))
    assert report["leakage_audit"]["forbidden_feature_count"] == 0
    assert report["leakage_audit"]["leakage_risk"] == "low"
    assert "future_10d_return" not in report["feature_columns"]
    assert "close_position" not in report["feature_columns"]
    assert set(LABEL_COLUMNS).isdisjoint(set(report["feature_columns"]))

    by_code = {row["code"]: row for row in dataset.to_dict("records")}
    assert "44440" not in by_code
    assert by_code["11110"]["rank_in_day"] == 1.0
    assert by_code["11110"]["candidate_count_in_day"] == 2
    assert by_code["11110"]["top_decile_future_utility_in_day"] is True


def test_phase9b_forbidden_tokens_do_not_enter_features(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    report = PMAIV3CleanDatasetBuilder(tmp_path).build(
        PMAIV3BuildOptions(start_date="2023-01-01", end_date="2023-01-31", top_n=3)
    )

    for feature in report["feature_columns"]:
        lower = feature.lower()
        assert not any(token in lower for token in FORBIDDEN_TOKENS), feature
        assert "future" not in lower
        assert "label" not in lower
        assert "target" not in lower

    conditional = {row["category"]: row for row in report["feature_classification"]}["conditional_relative"]
    assert conditional["count"] == len(CONDITIONAL_RELATIVE_FEATURES)
    assert set(conditional["columns"]) == set(CONDITIONAL_RELATIVE_FEATURES)


def test_phase9b_saves_outputs_without_overwriting_current_artifacts(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    pm_file = tmp_path / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json"
    exit_file = tmp_path / "models/ml/exit/current_v2_66/model_metadata.json"
    profile_file = tmp_path / f"config/profiles/{PROFILE}.yaml"
    before = {
        "pm": pm_file.read_text(encoding="utf-8"),
        "exit": exit_file.read_text(encoding="utf-8"),
        "profile": profile_file.read_text(encoding="utf-8"),
    }

    builder = PMAIV3CleanDatasetBuilder(tmp_path)
    report = builder.build(PMAIV3BuildOptions(start_date="2023-01-01", end_date="2023-01-31", top_n=2))
    paths = builder.save_report(report)

    assert paths.dataset.exists()
    assert paths.market_regime.exists()
    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "9-B"
    assert loaded["current_artifact_safety"]["current_pm_ai_overwritten"] is False
    assert loaded["current_artifact_safety"]["current_exit_ai_overwritten"] is False
    assert loaded["current_artifact_safety"]["v2_82_profile_overwritten"] is False

    assert pm_file.read_text(encoding="utf-8") == before["pm"]
    assert exit_file.read_text(encoding="utf-8") == before["exit"]
    assert profile_file.read_text(encoding="utf-8") == before["profile"]

