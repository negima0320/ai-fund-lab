from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ml.portfolio_manager_v3_integration_audit_pm_sizing_universe import (
    MAPPING_CONFIGS,
    PMAIV3IntegrationAuditPMSizingUniverse,
    PREDICTION_COLUMNS,
)


FEATURES = ["rank_in_day", "percentile_in_day", "risk_adjusted_score"]
PROFILES = {
    "a": "rookie_dealer_02_v2_93_pm_ai_v3_candidate",
    "b": "rookie_dealer_02_v2_93b_pm_ai_v3_candidate_conservative",
    "c": "rookie_dealer_02_v2_93c_pm_ai_v3_candidate_half_pm130",
}


def _write_fixture(root: Path) -> None:
    data_dir = root / "data/ml/portfolio_manager_v3"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    codes = [f"{10000 + idx * 10}" for idx in range(1, 31)]
    for year in [2023, 2024, 2025, 2026]:
        for date in pd.bdate_range(f"{year}-01-04", periods=24):
            for rank, code in enumerate(codes, start=1):
                pct = 1.0 - (rank - 1) / (len(codes) - 1)
                rows.append(
                    {
                        "prediction_date": date.strftime("%Y-%m-%d"),
                        "code": code,
                        "rank_in_day": float(rank),
                        "percentile_in_day": pct,
                        "risk_adjusted_score": pct,
                        "future_10d_return": 0.04 * pct,
                        "downside_penalized_return_10d": 0.08 * pct - 0.02 * (rank - 1),
                        "relative_future_utility_percentile_in_day": pct,
                        "top_decile_future_utility_in_day": rank <= 3,
                        "bottom_decile_future_utility_in_day": rank >= 28,
                        "max_adverse_excursion_10d": -0.02 * (rank - 1),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(data_dir / "portfolio_manager_v3_dataset_pm_sizing_universe_2023-01_to_2026-05.parquet", index=False)

    model_dir = root / "models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe"
    model_dir.mkdir(parents=True, exist_ok=True)
    x = frame[FEATURES]
    joblib.dump(HistGradientBoostingRegressor(max_iter=8, random_state=1).fit(x, frame["percentile_in_day"]), model_dir / "model_a_candidate_ranking_regressor.joblib")
    joblib.dump(HistGradientBoostingRegressor(max_iter=8, random_state=2).fit(x, frame["downside_penalized_return_10d"]), model_dir / "model_b_downside_utility_regressor.joblib")
    joblib.dump(HistGradientBoostingClassifier(max_iter=8, random_state=3).fit(x, (frame["rank_in_day"] <= 3).astype(int)), model_dir / "model_c_top_utility_classifier.joblib")
    (model_dir / "feature_columns.json").write_text(json.dumps(FEATURES), encoding="utf-8")

    report_dir = root / "reports/ml"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase9d3_mapping_threshold_audit_2023-01_to_2026-05.json").write_text(
        json.dumps({"recommended_threshold_config": {"config_id": "e_139", "classifier_gate_threshold": 0.8, "rank_threshold": 0.75, "downside_threshold": 0.8}}),
        encoding="utf-8",
    )

    for profile in PROFILES.values():
        path = root / "logs/backtests" / profile / "2023-01-01_to_2026-05-31"
        path.mkdir(parents=True, exist_ok=True)
        targets = frame.groupby("prediction_date").head(3).head(120).copy()
        targets["signal_date"] = targets["prediction_date"]
        targets["pm_model_version"] = "pm_ai_v3_phase9d_mapping_a_rank_score_only"
        targets["pm_status"] = "missing"
        targets["pm_missing_reason"] = "pm_v3_feature_row_missing"
        targets[["signal_date", "code", "pm_model_version", "pm_status", "pm_missing_reason"]].to_csv(path / "purchase_audit.csv", index=False)


def test_phase9e2_pm_sizing_universe_integration_audit(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    current_pm = tmp_path / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json"
    current_exit = tmp_path / "models/ml/exit/current_v2_66/model_metadata.json"
    v282 = tmp_path / "config/profiles/rookie_dealer_02_v2_82_cap38.yaml"
    current_pm.parent.mkdir(parents=True, exist_ok=True)
    current_exit.parent.mkdir(parents=True, exist_ok=True)
    v282.parent.mkdir(parents=True, exist_ok=True)
    current_pm.write_text("pm", encoding="utf-8")
    current_exit.write_text("exit", encoding="utf-8")
    v282.write_text("profile_id: rookie_dealer_02_v2_82_cap38\n", encoding="utf-8")
    before = (current_pm.read_text(encoding="utf-8"), current_exit.read_text(encoding="utf-8"), v282.read_text(encoding="utf-8"))

    audit = PMAIV3IntegrationAuditPMSizingUniverse(tmp_path)
    report = audit.build_report()
    paths = audit.save_report(report)

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert report["metadata"]["training_executed"] is False
    assert report["metadata"]["strategy_backtest_executed"] is False
    assert report["metadata"]["old_candidate_phase9d_model_used"] is False
    assert report["metadata"]["old_top10_fixed_dataset_used"] is False
    assert set(PREDICTION_COLUMNS).issubset(report["prediction_summary"])
    assert {"e_139_classifier_gate_recommended", "e_140_classifier_gate_more_pm130", "e_120_classifier_gate_wider"}.issubset(
        {row["mapping"] for row in report["mapping_candidates"]}
    )
    assert set(MAPPING_CONFIGS).issubset({row["mapping"] for row in report["mapping_quality_overall"]})
    assert report["coverage_audit"]["coverage_rate"] >= 0.95
    assert report["leakage_checklist"]["forbidden_feature_count"] == 0
    assert report["leakage_checklist"]["label_columns_in_features"] == []
    assert report["leakage_checklist"]["future_columns_in_features"] == []
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["best_mapping"]["phase9f2_backtest_worth_testing"] is True
    assert current_pm.read_text(encoding="utf-8") == before[0]
    assert current_exit.read_text(encoding="utf-8") == before[1]
    assert v282.read_text(encoding="utf-8") == before[2]
