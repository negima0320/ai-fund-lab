from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ml.portfolio_manager_v3_mapping_stability_audit import PMAIV3MappingStabilityAudit


FEATURES = ["rank_in_day", "percentile_in_day", "risk_adjusted_score", "topix_volatility"]


def _write_fixture(root: Path) -> None:
    data_dir = root / "data/ml/portfolio_manager_v3"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in [2023, 2024, 2025, 2026]:
        for day_idx, date in enumerate(pd.bdate_range(f"{year}-01-04", periods=8)):
            for rank, code in enumerate(["11110", "22220", "33330", "44440", "55550"], start=1):
                pct = 1.0 - (rank - 1) / 4.0
                downside = 0.04 * pct - 0.025 * (rank - 1)
                rows.append(
                    {
                        "prediction_date": date.strftime("%Y-%m-%d"),
                        "code": code,
                        "market_regime_key": ["attack", "neutral", "defensive"][day_idx % 3],
                        "rank_in_day": float(rank),
                        "percentile_in_day": pct,
                        "risk_adjusted_score": pct,
                        "topix_volatility": 0.01 * (day_idx % 3),
                        "downside_penalized_return_10d": downside,
                        "relative_future_utility_percentile_in_day": pct,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(data_dir / "portfolio_manager_v3_dataset_pm_sizing_universe_2023-01_to_2026-05.parquet", index=False)

    model_dir = root / "models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe"
    model_dir.mkdir(parents=True, exist_ok=True)
    x = frame[FEATURES]
    rank_model = HistGradientBoostingRegressor(max_iter=8, random_state=1).fit(x, frame["relative_future_utility_percentile_in_day"])
    down_model = HistGradientBoostingRegressor(max_iter=8, random_state=2).fit(x, frame["downside_penalized_return_10d"])
    top_model = HistGradientBoostingClassifier(max_iter=8, random_state=3).fit(x, (frame["rank_in_day"] == 1).astype(int))
    joblib.dump(rank_model, model_dir / "model_a_candidate_ranking_regressor.joblib")
    joblib.dump(down_model, model_dir / "model_b_downside_utility_regressor.joblib")
    joblib.dump(top_model, model_dir / "model_c_top_utility_classifier.joblib")
    (model_dir / "feature_columns.json").write_text(json.dumps(FEATURES), encoding="utf-8")


def test_mapping_stability_audit_builds_group_reports_and_saves(tmp_path: Path) -> None:
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

    audit = PMAIV3MappingStabilityAudit(tmp_path)
    report = audit.build_report()
    paths = audit.save_report(report)

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert report["metadata"]["training_executed"] is False
    assert report["metadata"]["strategy_backtest_executed"] is False
    assert {row["group"] for row in report["yearly_results"] if row["mapping"] == "mapping_d_conservative_high_conviction"} == {"2023", "2024", "2025", "2026"}
    assert report["market_regime_results"]
    assert report["rolling_results"]
    assert len(report["stability_score"]) == 5
    assert report["leakage_checklist"]["forbidden_feature_count"] == 0
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert "best_mapping_by_stability" in loaded["conclusion"]
    assert current_pm.read_text(encoding="utf-8") == before[0]
    assert current_exit.read_text(encoding="utf-8") == before[1]
    assert v282.read_text(encoding="utf-8") == before[2]
