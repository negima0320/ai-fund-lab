from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.phase5e_exit_ai_v2_trainer import Phase5EExitAIV2TrainerPrototype, TrainOptions


def _write_dataset(root: Path, *, forbidden: bool = False) -> None:
    path = root / "data" / "ml" / "exit_ai_v2" / "exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, start, n in [("train", "2021-06-01", 80), ("validation", "2024-01-04", 40), ("test", "2025-01-06", 40)]:
        for index, date in enumerate(pd.bdate_range(start, periods=n)):
            exit_quality = 0.08 if index % 8 == 0 else -0.01 + (index % 5) * 0.002
            row = {
                "code": f"{1000 + index}",
                "as_of_date": date.strftime("%Y-%m-%d"),
                "split": split,
                "close": 100 + index,
                "volume": 10000 + index,
                "return_1d": 0.001,
                "return_5d": 0.01,
                "ma25_gap": 0.2,
                "EPS": None if index % 6 == 0 else 10.0,
                "BPS": None,
                "OP_growth": None,
                "FEPS_growth": None,
                "FSales_growth": None,
                "FOP_growth": None,
                "future_return_3d": 0.01,
                "future_return_5d": -exit_quality,
                "future_return_10d": 0.02,
                "future_return_20d": 0.03,
                "avoid_loss_5d": exit_quality > 0.03,
                "miss_profit_5d": False,
                "exit_quality_score": exit_quality,
                "exit_quality_score_risk_adjusted": exit_quality + 0.01,
                "future_max_drawdown_5d": -0.04,
                "future_max_drawdown_10d": -0.05,
                "future_max_return_5d": 0.03,
                "future_max_return_10d": 0.04,
            }
            if forbidden:
                row["selected_count_in_day"] = 2
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase5e_dry_run_checks_leakage_without_training(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    report = Phase5EExitAIV2TrainerPrototype(tmp_path).run(TrainOptions(dry_run=True))

    assert report["training_status"]["trained"] is False
    assert report["metadata"]["current_model_overwritten"] is False
    assert report["dataset"]["feature_count"] == 6
    assert "BPS" not in report["feature_policy"]["feature_columns"]
    assert report["leakage_check"]["leakage_risk"] == "low"


def test_phase5e_sample_training_uses_train_threshold_and_skips_model_save(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    trainer = Phase5EExitAIV2TrainerPrototype(tmp_path)
    report = trainer.run(TrainOptions(dry_run=False, sample_rows=90))
    paths = trainer.save_report(report)

    assert report["training_status"]["trained"] is True
    assert report["training_status"]["model_saved"] is False
    assert report["target"]["threshold_fit_scope"] == "train split only"
    assert report["metrics"]["validation"]["precision_at_top10pct"] >= 0.0
    assert report["metrics"]["test"]["recall_at_top10pct"] >= 0.0
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.model_dir is None
    assert not (tmp_path / "models/ml/exit_ai_v2/candidate_v2_api_only").exists()


def test_phase5e_blocks_forbidden_columns(tmp_path: Path) -> None:
    _write_dataset(tmp_path, forbidden=True)

    report = Phase5EExitAIV2TrainerPrototype(tmp_path).run(TrainOptions(dry_run=False, sample_rows=90))

    assert report["training_status"]["trained"] is False
    assert report["leakage_check"]["selected_count_in_day_found"] is True
    assert report["leakage_check"]["leakage_risk"] == "high"
