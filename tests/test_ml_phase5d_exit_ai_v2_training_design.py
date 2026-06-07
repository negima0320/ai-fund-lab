from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.phase5d_exit_ai_v2_training_design import Phase5DExitAIV2TrainingDesignAudit


def _write_dataset(root: Path, *, forbidden: bool = False) -> None:
    path = root / "data" / "ml" / "exit_ai_v2" / "exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(30):
        if index < 12:
            split = "train"
            date = pd.Timestamp("2021-06-01") + pd.offsets.BDay(index)
        elif index < 20:
            split = "validation"
            date = pd.Timestamp("2024-01-04") + pd.offsets.BDay(index - 12)
        else:
            split = "test"
            date = pd.Timestamp("2025-01-06") + pd.offsets.BDay(index - 20)
        row = {
            "code": f"{1000 + index}",
            "as_of_date": date.strftime("%Y-%m-%d"),
            "split": split,
            "close": 100 + index,
            "volume": 10000 + index,
            "return_5d": 0.01,
            "ma25_gap": 0.2,
            "BPS": None if index < 20 else 100.0,
            "FOP_growth": None if index < 13 else 0.1,
            "future_return_3d": 0.01,
            "future_return_5d": -0.04 if index % 5 == 0 else 0.02,
            "future_return_10d": 0.03,
            "future_return_20d": 0.04,
            "avoid_loss_5d": index % 5 == 0,
            "miss_profit_5d": index % 7 == 0,
            "exit_quality_score": 0.04 if index % 5 == 0 else -0.02,
            "exit_quality_score_risk_adjusted": 0.05 if index % 5 == 0 else -0.01,
            "future_max_drawdown_5d": -0.05,
            "future_max_drawdown_10d": -0.06,
            "future_max_return_5d": 0.03,
            "future_max_return_10d": 0.05,
        }
        if forbidden:
            row["selected_count_in_day"] = 3
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase5d_builds_training_design_without_training(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    result = Phase5DExitAIV2TrainingDesignAudit(tmp_path).build_report()

    assert result["metadata"]["training_design_only"] is True
    assert result["metadata"]["model_training_executed"] is False
    assert result["recommended_task"] == "ranking-style exit_quality_score top decile"
    assert result["recommended_feature_set"] == "feature_set_drop_missing_30pct"
    assert result["feature_set_design"]["feature_set_all_41"]["feature_count"] >= 5
    assert "BPS" in result["feature_set_design"]["feature_set_drop_missing_30pct"]["dropped_features"]
    assert result["imputation_design"]["leakage_safe_imputation"] is True
    assert result["fair_comparison_policy"]["changed_component"] == "Exit AI only"
    assert result["leakage_audit"]["leakage_risk"] == "low"
    assert result["recommended_next_phase"] == "Phase 5-E Exit AI v2 Trainer Prototype"


def test_phase5d_blocks_forbidden_dataset_columns(tmp_path: Path) -> None:
    _write_dataset(tmp_path, forbidden=True)

    result = Phase5DExitAIV2TrainingDesignAudit(tmp_path).build_report()

    assert "selected_count_in_day" in result["leakage_audit"]["forbidden_columns_found"]
    assert result["leakage_audit"]["selected_count_in_day_found"] is True
    assert result["leakage_audit"]["leakage_risk"] == "high"
    assert result["recommended_next_phase"] == "Retraining deferred"


def test_phase5d_saves_report(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    audit = Phase5DExitAIV2TrainingDesignAudit(tmp_path)
    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 5-D" in paths.markdown.read_text(encoding="utf-8")
