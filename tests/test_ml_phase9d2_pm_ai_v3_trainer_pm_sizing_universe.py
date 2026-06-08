from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_v3_pm_sizing_universe_trainer import (
    DATASET_PATH,
    MODEL_DIR,
    PMAIV3PMSizingUniverseTrainer,
)
from ml.portfolio_manager_v3_trainer import Phase9DTrainOptions


PROFILE = "rookie_dealer_02_v2_82_cap38"


def _write_dataset(root: Path, *, forbidden: bool = False) -> None:
    path = root / DATASET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for start, days in [("2023-01-04", 30), ("2025-01-06", 20), ("2026-01-05", 20)]:
        for day_idx, date in enumerate(pd.bdate_range(start, periods=days)):
            for rank, code in enumerate(["11110", "22220", "33330", "44440", "55550", "66660"], start=1):
                quality = 1.0 - (rank - 1) / 5.0 + (day_idx % 4) * 0.02
                utility = 0.05 * quality - 0.02 * (rank - 1)
                row = {
                    "prediction_date": date.strftime("%Y-%m-%d"),
                    "code": code,
                    "market_date": date.strftime("%Y-%m-%d"),
                    "market_regime_key": "neutral",
                    "expected_return_5d": 0.02 * quality,
                    "expected_return_10d": 0.04 * quality,
                    "expected_max_return_10d": 0.05 * quality,
                    "expected_max_return_20d": 0.06 * quality,
                    "swing_success_probability_20d": 0.3 + quality * 0.4,
                    "upside_probability_10d": 0.3 + quality * 0.3,
                    "bad_entry_probability_10d": 0.1 + rank * 0.05,
                    "ml_score": quality,
                    "risk_adjusted_score": 0.04 * quality - 0.5 * (0.1 + rank * 0.05),
                    "stock_selection_rank_score": quality,
                    "topix_return_5d": 0.01,
                    "topix_return_10d": 0.02,
                    "topix_return_20d": 0.03,
                    "topix_return_1d_proxy": 0.001,
                    "topix_ma_distance": 0.02,
                    "topix_volatility": 0.01,
                    "market_attack_score_prototype": 0.02,
                    "close": 100 + rank,
                    "return_1d": 0.001 * rank,
                    "return_3d": 0.002 * rank,
                    "return_5d": 0.003 * rank,
                    "return_10d": 0.004 * rank,
                    "return_20d": 0.005 * rank,
                    "ma5_gap": 0.01 * quality,
                    "ma10_gap": 0.01 * quality,
                    "ma25_gap": 0.02 * quality,
                    "ma75_gap": 0.02 * quality,
                    "ma5_slope": 0.01,
                    "ma25_slope": 0.01,
                    "body_ratio": 0.5,
                    "volume": 100000 + rank,
                    "turnover_value": 100_000_000 + rank,
                    "volume_ratio_5d": 1.0 + quality,
                    "EPS": 100.0 + rank,
                    "EqAR": 0.5,
                    "BPS": None,
                    "days_to_earnings": None,
                    "days_after_earnings": None,
                    "candidate_count_in_day": 6,
                    "rank_in_day": float(rank),
                    "percentile_in_day": 1.0 - ((rank - 1) / 5),
                    "gap_to_best": 0.01 * (rank - 1),
                    "candidate_strength": quality - 0.5,
                    "future_5d_return": utility * 0.7,
                    "future_10d_return": utility,
                    "max_favorable_excursion_10d": utility + 0.02,
                    "max_adverse_excursion_10d": -0.01 * rank,
                    "downside_penalized_return_10d": utility - 0.01 * rank,
                    "risk_adjusted_future_return_10d": utility - 0.01 * rank,
                    "relative_future_utility_rank_in_day": float(rank),
                    "relative_future_utility_percentile_in_day": 1.0 - ((rank - 1) / 5),
                    "top_decile_future_utility_in_day": rank == 1,
                    "bottom_decile_future_utility_in_day": rank == 6,
                    "data_source": "fixture",
                    "relative_feature_timing": "computed_before_cash",
                }
                if forbidden:
                    row["cash_leak_feature"] = 1.0
                rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_artifacts(root: Path) -> dict[str, str]:
    for path in [
        root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean",
        root / "models/ml/exit/current_v2_66",
        root / "config/profiles",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    pm = root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json"
    exit_model = root / "models/ml/exit/current_v2_66/model_metadata.json"
    profile = root / f"config/profiles/{PROFILE}.yaml"
    pm.write_text(json.dumps({"model": "current_pm"}), encoding="utf-8")
    exit_model.write_text(json.dumps({"model": "current_exit"}), encoding="utf-8")
    profile.write_text(f"profile_id: {PROFILE}\n", encoding="utf-8")
    return {"pm": pm.read_text(encoding="utf-8"), "exit": exit_model.read_text(encoding="utf-8"), "profile": profile.read_text(encoding="utf-8")}


def test_phase9d2_trainer_runs_on_pm_sizing_universe_and_saves_candidate_models(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    before = _write_artifacts(tmp_path)

    trainer = PMAIV3PMSizingUniverseTrainer(tmp_path, max_iter=8)
    report = trainer.run(Phase9DTrainOptions(save_models=True, include_market_comparison=True))
    paths = trainer.save_report(report)

    assert report["metadata"]["phase"] == "9-D2"
    assert report["metadata"]["strategy_backtest_executed"] is False
    assert report["input_paths"]["dataset"].endswith(str(DATASET_PATH))
    assert report["model_output"]["model_dir"].endswith(str(MODEL_DIR))
    assert report["feature_plan"]["forbidden_feature_count"] == 0
    assert report["leakage_checklist"]["label_columns_in_features"] == []
    assert report["leakage_checklist"]["future_columns_in_features"] == []
    assert report["split"]["split_overlap"] is False
    assert (paths.model_dir / "model_a_candidate_ranking_regressor.joblib").exists()
    assert (paths.model_dir / "model_b_downside_utility_regressor.joblib").exists()
    assert (paths.model_dir / "model_c_top_utility_classifier.joblib").exists()
    assert (paths.model_dir / "metrics.json").exists()
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert (tmp_path / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json").read_text(encoding="utf-8") == before["pm"]
    assert (tmp_path / "models/ml/exit/current_v2_66/model_metadata.json").read_text(encoding="utf-8") == before["exit"]
    assert (tmp_path / f"config/profiles/{PROFILE}.yaml").read_text(encoding="utf-8") == before["profile"]


def test_phase9d2_drops_forbidden_and_high_missing_features(tmp_path: Path) -> None:
    _write_dataset(tmp_path, forbidden=True)
    _write_artifacts(tmp_path)

    report = PMAIV3PMSizingUniverseTrainer(tmp_path, max_iter=4).run(Phase9DTrainOptions(save_models=False, include_market_comparison=False))

    assert "cash_leak_feature" in report["feature_plan"]["dropped_features"]
    assert "BPS" in report["feature_plan"]["dropped_features"]
    assert report["leakage_checklist"]["forbidden_feature_count"] == 0
    assert report["leakage_checklist"]["leakage_risk"] == "low"
