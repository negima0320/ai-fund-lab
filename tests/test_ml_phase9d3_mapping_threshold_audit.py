from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ml.portfolio_manager_v3_mapping_threshold_audit import PMAIV3MappingThresholdAudit


FEATURES = ["rank_in_day", "percentile_in_day", "risk_adjusted_score"]


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
                        "downside_penalized_return_10d": 0.05 * pct - 0.02 * (rank - 1),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(data_dir / "portfolio_manager_v3_dataset_pm_sizing_universe_2023-01_to_2026-05.parquet", index=False)
    model_dir = root / "models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe"
    model_dir.mkdir(parents=True, exist_ok=True)
    x = frame[FEATURES]
    joblib.dump(HistGradientBoostingRegressor(max_iter=8, random_state=1).fit(x, frame["percentile_in_day"]), model_dir / "model_a_candidate_ranking_regressor.joblib")
    joblib.dump(HistGradientBoostingRegressor(max_iter=8, random_state=2).fit(x, frame["downside_penalized_return_10d"]), model_dir / "model_b_downside_utility_regressor.joblib")
    joblib.dump(HistGradientBoostingClassifier(max_iter=8, random_state=3).fit(x, (frame["rank_in_day"] == 1).astype(int)), model_dir / "model_c_top_utility_classifier.joblib")
    (model_dir / "feature_columns.json").write_text(json.dumps(FEATURES), encoding="utf-8")


def test_phase9d3_threshold_audit_builds_grid_and_report(tmp_path: Path) -> None:
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

    audit = PMAIV3MappingThresholdAudit(tmp_path)
    report = audit.build_report()
    paths = audit.save_report(report)

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert report["metadata"]["training_executed"] is False
    assert report["metadata"]["strategy_backtest_executed"] is False
    assert report["threshold_grid"]["candidate_count"] == 180
    assert report["top10_configs"]
    assert report["recommended_threshold_config"]["pm130_2026_count"] > 0
    assert report["leakage_checklist"]["forbidden_feature_count"] == 0
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["phase9e2_integration_audit_worth_testing"] is True
    assert current_pm.read_text(encoding="utf-8") == before[0]
    assert current_exit.read_text(encoding="utf-8") == before[1]
    assert v282.read_text(encoding="utf-8") == before[2]
