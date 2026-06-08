from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_v3_trainer import MODEL_DIR, PMAIV3TrainerPrototype, Phase9DTrainOptions


PROFILE = "rookie_dealer_02_v2_82_cap38"


def _write_dataset(root: Path, *, forbidden: bool = False) -> None:
    path = root / "data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_2023-01_to_2026-05.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = [
        ("2023-01-04", 36),
        ("2025-01-06", 24),
        ("2026-01-05", 24),
    ]
    for start, days in specs:
        for day_idx, date in enumerate(pd.bdate_range(start, periods=days)):
            for rank, code in enumerate(["11110", "22220", "33330", "44440", "55550"], start=1):
                quality = 1.0 - (rank - 1) / 4.0 + (day_idx % 5) * 0.02
                utility = 0.04 * quality - 0.025 * (rank - 1)
                row = {
                    "prediction_date": date.strftime("%Y-%m-%d"),
                    "code": code,
                    "market_date": date.strftime("%Y-%m-%d"),
                    "market_regime_key": "neutral",
                    "expected_return_5d": 0.02 * quality,
                    "expected_return_10d": 0.04 * quality,
                    "expected_max_return_20d": 0.06 * quality,
                    "swing_success_probability_20d": 0.2 + quality * 0.5,
                    "upside_probability_10d": 0.2 + quality * 0.4,
                    "bad_entry_probability_10d": 0.1 + rank * 0.08,
                    "ml_score": quality,
                    "risk_adjusted_score": 0.04 * quality - 0.5 * (0.1 + rank * 0.08),
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
                    "ma5_gap": 0.01 * quality,
                    "ma25_gap": 0.02 * quality,
                    "volume": 100000 + rank,
                    "turnover_value": 100_000_000 + rank,
                    "volume_ratio_5d": 1.0 + quality,
                    "EPS": 100.0 + rank,
                    "EqAR": 0.5,
                    "days_to_earnings": None,
                    "days_after_earnings": None,
                    "candidate_count_in_day": 5,
                    "rank_in_day": float(rank),
                    "percentile_in_day": 1.0 - ((rank - 1) / 4),
                    "gap_to_best": 0.01 * (rank - 1),
                    "candidate_strength": quality - 0.5,
                    "future_5d_return": utility * 0.7,
                    "future_10d_return": utility,
                    "max_favorable_excursion_10d": utility + 0.02,
                    "max_adverse_excursion_10d": -0.01 * rank,
                    "downside_penalized_return_10d": utility - 0.01 * rank,
                    "risk_adjusted_future_return_10d": utility - 0.01 * rank,
                    "relative_future_utility_rank_in_day": float(rank),
                    "relative_future_utility_percentile_in_day": 1.0 - ((rank - 1) / 4),
                    "top_decile_future_utility_in_day": rank == 1,
                    "bottom_decile_future_utility_in_day": rank == 5,
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
    return {
        "pm": pm.read_text(encoding="utf-8"),
        "exit": exit_model.read_text(encoding="utf-8"),
        "profile": profile.read_text(encoding="utf-8"),
    }


def test_phase9d_trainer_runs_and_saves_candidate_models(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _write_artifacts(tmp_path)

    trainer = PMAIV3TrainerPrototype(tmp_path)
    report = trainer.run(Phase9DTrainOptions(save_models=True))
    paths = trainer.save_report(report)

    assert report["metadata"]["training_executed"] is True
    assert report["metadata"]["backtest_executed"] is False
    assert report["feature_plan"]["forbidden_feature_count"] == 0
    assert report["feature_plan"]["label_columns_in_features"] == []
    assert report["model_output"]["model_dir"].endswith(str(MODEL_DIR))
    assert (paths.model_dir / "model_a_candidate_ranking_regressor.joblib").exists()
    assert (paths.model_dir / "model_b_downside_utility_regressor.joblib").exists()
    assert (paths.model_dir / "model_c_top_utility_classifier.joblib").exists()
    assert (paths.model_dir / "feature_columns.json").exists()
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert report["split"]["split_overlap"] is False
    assert set(report["metrics"]) == {
        "model_a_candidate_ranking_regressor",
        "model_b_downside_utility_regressor",
        "model_c_top_utility_classifier",
    }


def test_phase9d_blocks_forbidden_feature(tmp_path: Path) -> None:
    _write_dataset(tmp_path, forbidden=True)
    _write_artifacts(tmp_path)

    report = PMAIV3TrainerPrototype(tmp_path).run(Phase9DTrainOptions(save_models=True))

    assert report["metadata"]["training_executed"] is True
    assert report["feature_plan"]["forbidden_feature_count"] == 0
    assert "cash_leak_feature" in report["feature_plan"]["dropped_features"]
    assert (tmp_path / MODEL_DIR / "model_a_candidate_ranking_regressor.joblib").exists()


def test_phase9d_does_not_overwrite_current_artifacts(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    before = _write_artifacts(tmp_path)

    trainer = PMAIV3TrainerPrototype(tmp_path)
    report = trainer.run()
    trainer.save_report(report)

    assert (tmp_path / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json").read_text(encoding="utf-8") == before["pm"]
    assert (tmp_path / "models/ml/exit/current_v2_66/model_metadata.json").read_text(encoding="utf-8") == before["exit"]
    assert (tmp_path / f"config/profiles/{PROFILE}.yaml").read_text(encoding="utf-8") == before["profile"]
    assert report["metadata"]["current_pm_ai_overwritten"] is False
    assert report["metadata"]["current_exit_ai_overwritten"] is False
    assert report["metadata"]["v2_82_profile_overwritten"] is False
