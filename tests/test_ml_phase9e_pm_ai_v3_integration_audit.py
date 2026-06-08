from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from ml.portfolio_manager_v3_integration_audit import (
    MAPPING_NAMES,
    PMAIV3CandidateIntegrationAudit,
    PREDICTION_COLUMNS,
)


PROFILE = "rookie_dealer_02_v2_82_cap38"
FEATURES = ["expected_return_10d", "risk_adjusted_score", "bad_entry_probability_10d", "rank_in_day"]


def _write_fixture(root: Path) -> None:
    data_dir = root / "data/ml/portfolio_manager_v3"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_start, days in [("2023-01-04", 8), ("2025-01-06", 6), ("2026-01-05", 6)]:
        for day_idx, date in enumerate(pd.bdate_range(split_start, periods=days)):
            for rank, code in enumerate(["11110", "22220", "33330", "44440", "55550"], start=1):
                utility = 0.05 - rank * 0.015 + (day_idx % 3) * 0.002
                rows.append(
                    {
                        "prediction_date": date.strftime("%Y-%m-%d"),
                        "code": code,
                        "expected_return_10d": 0.04 - rank * 0.005,
                        "risk_adjusted_score": 0.03 - rank * 0.004,
                        "bad_entry_probability_10d": 0.1 + rank * 0.05,
                        "rank_in_day": float(rank),
                        "future_10d_return": utility + 0.01,
                        "downside_penalized_return_10d": utility,
                        "relative_future_utility_percentile_in_day": 1.0 - ((rank - 1) / 4.0),
                        "top_decile_future_utility_in_day": rank == 1,
                        "bottom_decile_future_utility_in_day": rank == 5,
                        "max_adverse_excursion_10d": -0.01 * rank,
                    }
                )
    pd.DataFrame(rows).to_parquet(data_dir / "portfolio_manager_v3_dataset_2023-01_to_2026-05.parquet", index=False)

    model_dir = root / "models/ml/portfolio_manager_v3/candidate_phase9d"
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    x = frame[FEATURES]
    y_reg = frame["relative_future_utility_percentile_in_day"]
    y_down = frame["downside_penalized_return_10d"]
    y_cls = frame["top_decile_future_utility_in_day"].astype(int)
    rank_model = DummyRegressor(strategy="mean").fit(x, y_reg)
    down_model = DummyRegressor(strategy="mean").fit(x, y_down)
    cls_model = DummyClassifier(strategy="prior").fit(x, y_cls)
    joblib.dump(rank_model, model_dir / "model_a_candidate_ranking_regressor.joblib")
    joblib.dump(down_model, model_dir / "model_b_downside_utility_regressor.joblib")
    joblib.dump(cls_model, model_dir / "model_c_top_utility_classifier.joblib")
    (model_dir / "feature_columns.json").write_text(json.dumps(FEATURES), encoding="utf-8")
    (model_dir / "training_metadata.json").write_text(json.dumps({"phase": "9-D"}), encoding="utf-8")

    purchase = root / "reports/final/v2_82_cap38/core_2023-01_to_2026-05"
    purchase.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"signal_date": "2026-01-05", "code": "11110", "pm_multiplier": 1.3, "decision": "BUY"},
            {"signal_date": "2026-01-05", "code": "55550", "pm_multiplier": 0.8, "decision": "BUY"},
        ]
    ).to_csv(purchase / "purchase_audit.csv", index=False)

    for path in [
        root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean",
        root / "models/ml/exit/current_v2_66",
        root / "config/profiles",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    (root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json").write_text("pm", encoding="utf-8")
    (root / "models/ml/exit/current_v2_66/model_metadata.json").write_text("exit", encoding="utf-8")
    (root / f"config/profiles/{PROFILE}.yaml").write_text(f"profile_id: {PROFILE}\n", encoding="utf-8")


def test_phase9e_integration_audit_builds_predictions_and_mappings(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    report = PMAIV3CandidateIntegrationAudit(tmp_path).build_report()

    assert report["metadata"]["audit_only"] is True
    assert report["metadata"]["backtest_executed"] is False
    assert set(PREDICTION_COLUMNS) == set(report["prediction_columns"])
    assert set(MAPPING_NAMES) == set(report["mapping_candidates"])
    assert report["leakage_guard"]["forbidden_feature_count"] == 0
    assert report["leakage_guard"]["leakage_risk"] == "low"
    assert report["current_pm_comparison"]["summary"]["joined_rows"] >= 1


def test_phase9e_saves_report_without_overwriting_current_artifacts(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    pm_file = tmp_path / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json"
    exit_file = tmp_path / "models/ml/exit/current_v2_66/model_metadata.json"
    profile_file = tmp_path / f"config/profiles/{PROFILE}.yaml"
    before = {
        "pm": pm_file.read_text(encoding="utf-8"),
        "exit": exit_file.read_text(encoding="utf-8"),
        "profile": profile_file.read_text(encoding="utf-8"),
    }

    audit = PMAIV3CandidateIntegrationAudit(tmp_path)
    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "9-E"
    assert loaded["metadata"]["current_pm_ai_overwritten"] is False
    assert loaded["metadata"]["current_exit_ai_overwritten"] is False
    assert loaded["metadata"]["v2_82_profile_overwritten"] is False
    assert pm_file.read_text(encoding="utf-8") == before["pm"]
    assert exit_file.read_text(encoding="utf-8") == before["exit"]
    assert profile_file.read_text(encoding="utf-8") == before["profile"]

