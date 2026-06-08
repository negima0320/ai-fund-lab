from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase7c_pm_ai_api_only_dataset_design import (
    Phase7CPMAIAPIOnlyDatasetDesignAudit,
    classify_rebuild_column,
    is_candidate_list_feature,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_fixture(root: Path) -> None:
    model_dir = root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean"
    _write_json(
        model_dir / "model_metadata.json",
        {
            "model_profile": "portfolio_manager_v2_73_phase3b_clean",
            "feature_columns": ["close_position", "return_5d", "risk_adjusted_score", "candidate_count_in_day"],
        },
    )

    pm_path = root / "data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet"
    stock_path = root / "data/ml/datasets/ml_dataset.parquet"
    exit_path = root / "data/ml/exit_ai_v2/exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet"
    pm_path.parent.mkdir(parents=True, exist_ok=True)
    stock_path.parent.mkdir(parents=True, exist_ok=True)
    exit_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "signal_date": "2023-01-04",
                "code": "11110",
                "close_position": 0.8,
                "return_5d": 0.02,
                "risk_adjusted_score": 0.3,
                "candidate_count_in_day": 20,
                "rank_in_day": 1,
                "max_positions_remaining_before": 4,
                "actual_net_profit": 1000,
                "future_10d_return": 0.05,
                "realized_return": 0.05,
            }
        ]
    ).to_parquet(pm_path)

    pd.DataFrame(
        [
            {
                "date": "2021-06-01",
                "code": "11110",
                "close_position": 0.7,
                "return_5d": 0.01,
                "topix_return_5d": 0.01,
                "EPS": 10.0,
                "risk_adjusted_score": 0.2,
                "expected_return_10d": 0.03,
                "future_5d_return": 0.01,
                "future_10d_return": 0.02,
                "future_max_return_20d": 0.05,
            },
            {
                "date": "2026-05-29",
                "code": "22220",
                "close_position": 0.2,
                "return_5d": -0.01,
                "topix_return_5d": -0.01,
                "EPS": 12.0,
                "risk_adjusted_score": 0.1,
                "expected_return_10d": 0.01,
                "future_5d_return": -0.01,
                "future_10d_return": 0.0,
                "future_max_return_20d": 0.02,
            },
        ]
    ).to_parquet(stock_path)

    pd.DataFrame(
        [
            {"as_of_date": "2021-06-01", "code": "11110", "exit_quality_score": 0.1},
            {"as_of_date": "2026-03-30", "code": "22220", "exit_quality_score": -0.1},
        ]
    ).to_parquet(exit_path)


def test_phase7c_classifies_candidate_list_features_as_forbidden_for_rebuild() -> None:
    assert is_candidate_list_feature("candidate_count_in_day") is True
    assert is_candidate_list_feature("expected_return_gap_to_best") is True
    assert classify_rebuild_column("close_position") == "api_price_feature"
    assert classify_rebuild_column("candidate_count_in_day") == "forbidden_candidate_list_dependent"
    assert classify_rebuild_column("max_positions_remaining_before") == "backtest_position_state"
    assert classify_rebuild_column("risk_adjusted_score") == "stock_selection_walk_forward_prediction"


def test_phase7c_designs_api_only_pm_dataset_and_requires_candidate_removal(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = Phase7CPMAIAPIOnlyDatasetDesignAudit(tmp_path).build_report()

    verdict = result["final_judgement"]
    features = result["feature_set_design"]
    assert verdict["api_only_pm_dataset_feasible"] is True
    assert verdict["candidate_feature_removal_required"] is True
    assert verdict["ready_for_phase7d"] is True
    assert "candidate_count_in_day" in features["candidate_list_features_removed"]
    assert "max_positions_remaining_before" not in features["recommended_features"]
    assert "max_positions_remaining_before" in features["forbidden_columns_to_remove"]
    assert "risk_adjusted_score" in features["recommended_features"]


def test_phase7c_label_design_recommends_future_return_labels(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = Phase7CPMAIAPIOnlyDatasetDesignAudit(tmp_path).build_report()

    labels = result["label_design"]
    assert "future_5d_return" in labels["recommended_label_set"]
    assert "risk_adjusted_future_return" in labels["recommended_label_set"]
    assert all(row["api_only"] for row in labels["labels"])


def test_phase7c_saves_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = Phase7CPMAIAPIOnlyDatasetDesignAudit(tmp_path)

    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 7-C" in paths.markdown.read_text(encoding="utf-8")
