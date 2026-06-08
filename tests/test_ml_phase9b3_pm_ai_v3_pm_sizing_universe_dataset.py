from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_v3_dataset_builder import PMAIV3BuildOptions
from ml.portfolio_manager_v3_pm_sizing_universe_builder import PMAIV3PMSizingUniverseDatasetBuilder


def _base_row(date: str, code: str, close: float, rank: int) -> dict[str, object]:
    return {
        "date": date,
        "code": code,
        "topix_return_5d": 0.01,
        "topix_return_10d": 0.02,
        "topix_return_20d": 0.03,
        "close": close,
        "return_1d": 0.01,
        "return_3d": 0.02,
        "return_5d": 0.03,
        "return_10d": 0.04,
        "return_20d": 0.05,
        "ma5_gap": 0.01,
        "ma10_gap": 0.01,
        "ma25_gap": 0.01,
        "ma75_gap": 0.01,
        "ma5_slope": 0.01,
        "ma25_slope": 0.01,
        "body_ratio": 0.5,
        "upper_shadow_ratio": 0.1,
        "lower_shadow_ratio": 0.1,
        "gap_up_ratio": 0.01,
        "daily_range_ratio": 0.02,
        "volume": 100000 + rank,
        "turnover_value": 100000000 + rank,
        "volume_ratio_5d": 1.2,
        "volume_ratio_20d": 1.1,
        "turnover_ratio_5d": 1.2,
        "turnover_ratio_20d": 1.1,
        "EPS": 100,
        "BPS": 1000,
        "EqAR": 0.5,
        "Sales_growth": 0.1,
        "OP_growth": 0.1,
        "NP_growth": 0.1,
        "future_5d_return": 0.05 - rank * 0.005,
        "future_10d_return": 0.08 - rank * 0.01,
        "future_max_return_10d": 0.1 - rank * 0.005,
    }


def _write_fixture(root: Path) -> None:
    base_dir = root / "data/ml/datasets"
    base_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for date in ["2026-01-05", "2026-01-06"]:
        for rank, code in enumerate(["10010", "20020", "30030", "40040", "50050"], start=1):
            rows.append(_base_row(date, code, 100 + rank, rank))
    pd.DataFrame(rows).to_parquet(base_dir / "ml_dataset.parquet", index=False)

    pred_dir = root / "data/ml/walk_forward_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for date in ["2026-01-05", "2026-01-06"]:
        pd.DataFrame(
            [
                {
                    "date": date,
                    "code": code,
                    "expected_return_5d": 0.03 - rank * 0.001,
                    "expected_return_10d": 0.05 - rank * 0.002,
                    "expected_max_return_10d": 0.07 - rank * 0.002,
                    "expected_max_return_20d": 0.09 - rank * 0.002,
                    "swing_success_probability_20d": 0.6,
                    "upside_probability_10d": 0.55,
                    "bad_entry_probability_10d": 0.1 + rank * 0.01,
                    "ml_score": 0.5,
                }
                for rank, code in enumerate(["10010", "20020", "30030", "40040", "50050"], start=1)
            ]
        ).to_parquet(pred_dir / f"predictions_{date}.parquet", index=False)

    bt_dir = root / "logs/backtests/fixture_profile/2023-01-01_to_2026-05-31"
    bt_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"signal_date": "2026-01-05", "entry_date": "2026-01-06", "code": "10010", "pm_model_version": "pm_ai_v3_phase9d", "pm_missing_reason": "pm_v3_feature_row_missing"},
            {"signal_date": "2026-01-05", "entry_date": "2026-01-06", "code": "50050", "pm_model_version": "pm_ai_v3_phase9d", "pm_missing_reason": "pm_v3_feature_row_missing"},
        ]
    ).to_csv(bt_dir / "purchase_audit.csv", index=False)


def test_phase9b3_builder_creates_pm_sizing_universe_and_coverage_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    pm_file = tmp_path / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json"
    exit_file = tmp_path / "models/ml/exit/current_v2_66/model_metadata.json"
    profile_file = tmp_path / "config/profiles/rookie_dealer_02_v2_82_cap38.yaml"
    pm_file.parent.mkdir(parents=True, exist_ok=True)
    exit_file.parent.mkdir(parents=True, exist_ok=True)
    profile_file.parent.mkdir(parents=True, exist_ok=True)
    pm_file.write_text("pm", encoding="utf-8")
    exit_file.write_text("exit", encoding="utf-8")
    profile_file.write_text("profile_id: rookie_dealer_02_v2_82_cap38\n", encoding="utf-8")
    before = (pm_file.read_text(encoding="utf-8"), exit_file.read_text(encoding="utf-8"), profile_file.read_text(encoding="utf-8"))

    builder = PMAIV3PMSizingUniverseDatasetBuilder(tmp_path, profiles={"fixture": "fixture_profile"})
    report = builder.build(PMAIV3BuildOptions(start_date="2026-01-01", end_date="2026-01-31", top_n=999, min_turnover_value=0, write_outputs=True))
    paths = builder.save_report(report)

    dataset = report["_dataset"]
    assert paths.dataset.exists()
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert len(dataset) == 10
    assert dataset.groupby("prediction_date")["code"].count().max() == 5
    assert report["universe_design"]["uses_top10_fixed"] is False
    assert report["coverage_against_phase9f_pm_sizing_keys"]["pm_sizing_key_coverage"] == 1.0
    assert report["leakage_audit"]["forbidden_feature_count"] == 0
    assert report["leakage_audit"]["label_columns_in_features"] == []
    assert "future_10d_return" not in report["feature_columns"]
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["strategy_backtest_executed"] is False
    assert pm_file.read_text(encoding="utf-8") == before[0]
    assert exit_file.read_text(encoding="utf-8") == before[1]
    assert profile_file.read_text(encoding="utf-8") == before[2]
