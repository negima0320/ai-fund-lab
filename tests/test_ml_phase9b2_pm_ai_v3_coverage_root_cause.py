from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_v3_coverage_root_cause_audit import (
    PMAIV3CoverageRootCauseAudit,
    normalize_code,
    normalize_date,
    previous_business_day,
)


FEATURES = ["expected_return_10d", "risk_adjusted_score", "rank_in_day"]


def _write_fixture(root: Path) -> None:
    data_dir = root / "data/ml/portfolio_manager_v3"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"prediction_date": "2026-01-05", "code": "10010", "data_source": "walk_forward_predictions", "relative_feature_timing": "pre_cash", "candidate_count_in_day": 2, "rank_in_day": 1},
            {"prediction_date": "2026-01-05", "code": "20020", "data_source": "walk_forward_predictions", "relative_feature_timing": "pre_cash", "candidate_count_in_day": 2, "rank_in_day": 2},
            {"prediction_date": "2026-01-06", "code": "30030", "data_source": "walk_forward_predictions", "relative_feature_timing": "pre_cash", "candidate_count_in_day": 1, "rank_in_day": 1},
        ]
    ).to_parquet(data_dir / "portfolio_manager_v3_dataset_2023-01_to_2026-05.parquet", index=False)

    model_dir = root / "models/ml/portfolio_manager_v3/candidate_phase9d"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "feature_columns.json").write_text(json.dumps(FEATURES), encoding="utf-8")

    profile = "fixture_profile"
    bt_dir = root / "logs/backtests" / profile / "2023-01-01_to_2026-05-31"
    bt_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"signal_date": "2026-01-05", "entry_date": "2026-01-06", "code": 10010, "decision": "BUY", "candidate_source": "selected", "pm_model_version": "pm_ai_v3_phase9d", "pm_status": "ok", "pm_feature_found": True, "pm_missing_reason": ""},
            {"signal_date": "2026-01-05", "entry_date": "2026-01-06", "code": 99990, "decision": "BUY", "candidate_source": "selected", "pm_model_version": "pm_ai_v3_phase9d", "pm_status": "missing", "pm_feature_found": False, "pm_missing_reason": "pm_v3_feature_row_missing"},
        ]
    ).to_csv(bt_dir / "purchase_audit.csv", index=False)


def test_normalize_date_and_code_helpers() -> None:
    assert normalize_code(1001) == "1001"
    assert normalize_code("1001.0") == "1001"
    assert normalize_code("001") == "0001"
    assert normalize_date("2026/01/05") == "2026-01-05"
    assert previous_business_day("2026-01-06") == "2026-01-05"


def test_phase9b2_coverage_matrix_and_report_generation(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    current_pm = tmp_path / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json"
    current_exit = tmp_path / "models/ml/exit/current_v2_66/model_metadata.json"
    v282_profile = tmp_path / "config/profiles/rookie_dealer_02_v2_82_cap38.yaml"
    current_pm.parent.mkdir(parents=True, exist_ok=True)
    current_exit.parent.mkdir(parents=True, exist_ok=True)
    v282_profile.parent.mkdir(parents=True, exist_ok=True)
    current_pm.write_text("pm", encoding="utf-8")
    current_exit.write_text("exit", encoding="utf-8")
    v282_profile.write_text("profile_id: rookie_dealer_02_v2_82_cap38\n", encoding="utf-8")
    before = (current_pm.read_text(encoding="utf-8"), current_exit.read_text(encoding="utf-8"), v282_profile.read_text(encoding="utf-8"))

    audit = PMAIV3CoverageRootCauseAudit(
        tmp_path,
        profiles={"fixture": "fixture_profile"},
    )
    report = audit.build_report()
    paths = audit.save_report(report)

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    matrix = {row["key"]: row for row in loaded["coverage_matrix"]}
    assert matrix["prediction_date+code"]["matched_rows"] == 1
    assert matrix["prediction_date+code"]["coverage_rate"] == 0.5
    assert loaded["leakage_audit"]["forbidden_feature_count"] == 0
    assert loaded["leakage_audit"]["leakage_risk"] == "low"
    assert loaded["metadata"]["strategy_backtest_executed"] is False
    assert current_pm.read_text(encoding="utf-8") == before[0]
    assert current_exit.read_text(encoding="utf-8") == before[1]
    assert v282_profile.read_text(encoding="utf-8") == before[2]
