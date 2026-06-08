from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.phase7e_pm_ai_api_only_trainer import Phase7EPMAIAPIOnlyTrainerPrototype, TrainOptions


def _write_dataset(root: Path, *, forbidden: bool = False) -> None:
    path = root / "data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = [("train", "2021-06-01", 90), ("validation", "2024-01-04", 45), ("test", "2025-01-06", 45)]
    for split, start, count in specs:
        for index, date in enumerate(pd.bdate_range(start, periods=count)):
            high = index % 7 == 0
            avoid = index % 11 == 0
            row = {
                "as_of_date": date.strftime("%Y-%m-%d"),
                "code": f"{1000 + index}",
                "split": split,
                "close": 100.0 + index,
                "volume": 10000 + index,
                "return_1d": 0.001 * (index % 5),
                "return_5d": 0.01,
                "ma25_gap": 0.2,
                "EPS": None if index % 9 == 0 else 10.0,
                "risk_adjusted_score": 0.5 if high else 0.1,
                "expected_return_10d": 0.04 if high else 0.01,
                "future_5d_return": -0.04 if avoid else 0.01,
                "future_10d_return": 0.08 if high else (-0.04 if avoid else 0.01),
                "risk_adjusted_future_return": 0.05 if high else -0.02 if avoid else 0.0,
                "high_conviction_target": high,
                "avoid_target": avoid,
            }
            if forbidden:
                row["candidate_count_in_day"] = 20
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase7e_dry_run_checks_schema_without_training(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    report = Phase7EPMAIAPIOnlyTrainerPrototype(tmp_path).run(TrainOptions(dry_run=True))

    assert report["training_status"]["trained"] is False
    assert report["metadata"]["current_model_overwritten"] is False
    assert report["dataset"]["feature_count"] >= 5
    assert "future_10d_return" not in report["feature_policy"]["feature_columns"]
    assert report["leakage_check"]["leakage_risk"] == "low"


def test_phase7e_sample_training_trains_two_classifiers_without_saving(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    trainer = Phase7EPMAIAPIOnlyTrainerPrototype(tmp_path)
    report = trainer.run(TrainOptions(dry_run=False, sample_rows=120))
    paths = trainer.save_report(report)

    assert report["training_status"]["trained"] is True
    assert report["training_status"]["model_saved"] is False
    assert set(report["metrics"].keys()) == {"high_conviction_target", "avoid_target"}
    assert report["metrics"]["high_conviction_target"]["validation"]["precision_at_top10pct"] >= 0.0
    assert report["metrics"]["avoid_target"]["test"]["recall_at_top10pct"] >= 0.0
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.model_dir is None
    assert not (tmp_path / "models/ml/portfolio_manager/current_v2_73_phase3b_clean").exists()


def test_phase7e_blocks_candidate_list_dependent_feature(tmp_path: Path) -> None:
    _write_dataset(tmp_path, forbidden=True)

    report = Phase7EPMAIAPIOnlyTrainerPrototype(tmp_path).run(TrainOptions(dry_run=False, sample_rows=120))

    assert report["training_status"]["trained"] is False
    assert report["leakage_check"]["candidate_list_dependent_columns_in_features"] == ["candidate_count_in_day"]
    assert report["leakage_check"]["leakage_risk"] == "high"


def test_phase7e_saves_dry_run_report(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    trainer = Phase7EPMAIAPIOnlyTrainerPrototype(tmp_path)

    paths = trainer.save_report(trainer.run(TrainOptions(dry_run=True)))

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 7-E" in paths.markdown.read_text(encoding="utf-8")

