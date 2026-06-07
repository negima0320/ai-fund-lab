from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.phase5b_exit_ai_v2_dataset_design import Phase5BExitAIV2DatasetDesignAudit


def _write_fixture_tree(root: Path, *, forbidden_base_column: bool = False) -> None:
    dataset_dir = root / "data" / "ml" / "datasets"
    exit_dir = root / "data" / "ml" / "exit_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    exit_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    dates = pd.bdate_range("2021-06-01", periods=30)
    for code in ["11110", "22220"]:
        for index, date in enumerate(dates):
            row = {
                "date": date.strftime("%Y-%m-%d"),
                "code": code,
                "close": 100 + index,
                "volume": 100_000 + index,
                "return_5d": 0.01,
                "volatility_20d": 0.2,
                "risk_adjusted_score": 0.5,
                "eps": 12.3,
                "future_5d_return": 0.02,
                "bad_entry_10d": False,
            }
            if forbidden_base_column:
                row["selected_count_in_day"] = 3
            rows.append(row)
    pd.DataFrame(rows).to_parquet(dataset_dir / "ml_dataset.parquet")

    pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "code": "11110",
                "actual_exit_date": "2023-01-10",
                "remaining_days_to_actual_exit": 2,
                "holding_days": 5,
                "exit_reason": "exit_ai",
                "realized_return": 0.03,
            }
        ]
    ).to_parquet(exit_dir / "exit_dataset_v2_66_2023-01_to_2026-05.parquet")


def test_phase5b_design_uses_api_only_base_and_blocks_existing_exit_dataset(tmp_path: Path) -> None:
    _write_fixture_tree(tmp_path)

    result = Phase5BExitAIV2DatasetDesignAudit(tmp_path).build_report()

    assert result["metadata"]["audit_only"] is True
    assert result["metadata"]["dataset_generated"] is False
    assert result["metadata"]["api_only_design"] is True
    assert "trades.csv as teacher labels" in result["data_policy"]["forbidden_sources"]
    assert result["dataset_schema_design"]["row_definition"].startswith("one code/as_of_date")
    assert "selected_count_in_day" not in result["dataset_schema_design"]["safe_feature_columns"]
    assert "future_5d_return" not in result["dataset_schema_design"]["safe_feature_columns"]
    assert "future_5d_return" in result["dataset_schema_design"]["future_label_source_columns_excluded_from_features"]
    assert result["label_design"]["recommended_label"] == "exit_quality_score"
    assert result["sample_generation_feasibility"]["candidate_rows_before_label_horizon_filtering"] == 60
    assert result["sample_generation_feasibility"]["rows_after_20d_label_available"] == 20
    assert result["leakage_audit"]["blocking_issues"] == []
    assert "bad_entry_10d" in result["leakage_audit"]["future_label_source_columns_excluded_from_features"]
    assert result["existing_exit_dataset_comparison"]["existing_dataset_retraining_allowed"] is False
    assert "trade_id" in result["existing_exit_dataset_comparison"]["existing_forbidden_columns_found"]
    assert result["existing_exit_dataset_comparison"]["new_dataset_design_retraining_allowed"] is True
    assert result["recommended_next_phase"] == "Phase 5-C Exit AI v2 Dataset Builder"


def test_phase5b_leakage_audit_blocks_forbidden_base_columns(tmp_path: Path) -> None:
    _write_fixture_tree(tmp_path, forbidden_base_column=True)

    result = Phase5BExitAIV2DatasetDesignAudit(tmp_path).build_report()

    assert "selected_count_in_day" in result["leakage_audit"]["forbidden_columns_found"]
    assert result["leakage_audit"]["leakage_risk"] == "high"
    assert result["existing_exit_dataset_comparison"]["new_dataset_design_retraining_allowed"] is False
    assert result["recommended_next_phase"] == "Retraining deferred"


def test_phase5b_saves_markdown_and_json(tmp_path: Path) -> None:
    _write_fixture_tree(tmp_path)

    audit = Phase5BExitAIV2DatasetDesignAudit(tmp_path)
    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 5-B" in paths.markdown.read_text(encoding="utf-8")
