from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_v3_dataset_audit import PMAIV3DatasetQualityAudit


PROFILE = "rookie_dealer_02_v2_82_cap38"


def _write_fixture(root: Path) -> None:
    out = root / "data/ml/portfolio_manager_v3"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for day in ["2023-01-02", "2023-01-03"]:
        for idx, code in enumerate(["11110", "22220", "33330"], start=1):
            utility = 0.05 - idx * 0.02
            rows.append(
                {
                    "prediction_date": day,
                    "code": code,
                    "market_date": day,
                    "market_regime_key": "attack" if day == "2023-01-02" else "defensive",
                    "expected_return_10d": 0.08 - idx * 0.02,
                    "bad_entry_probability_10d": 0.1 * idx,
                    "risk_adjusted_score": 0.08 - idx * 0.03,
                    "candidate_count_in_day": 3,
                    "rank_in_day": float(idx),
                    "percentile_in_day": 1.0 - ((idx - 1) / 2),
                    "gap_to_best": 0.01 * (idx - 1),
                    "candidate_strength": 0.02 - idx * 0.01,
                    "topix_return_5d": 0.02 if day == "2023-01-02" else -0.02,
                    "market_attack_score_prototype": 0.05 if day == "2023-01-02" else -0.05,
                    "turnover_value": 100_000_000,
                    "future_5d_return": utility,
                    "future_10d_return": utility + 0.01,
                    "max_favorable_excursion_10d": utility + 0.02,
                    "max_adverse_excursion_10d": -0.01 * idx,
                    "downside_penalized_return_10d": utility,
                    "risk_adjusted_future_return_10d": utility,
                    "relative_future_utility_rank_in_day": float(idx),
                    "relative_future_utility_percentile_in_day": 1.0 - ((idx - 1) / 2),
                    "top_decile_future_utility_in_day": idx == 1,
                    "bottom_decile_future_utility_in_day": idx == 3,
                    "data_source": "fixture",
                    "relative_feature_timing": "computed_before_cash",
                }
            )
    pd.DataFrame(rows).to_parquet(out / "portfolio_manager_v3_dataset_2023-01_to_2026-05.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "market_regime_class_prototype": "attack",
                "market_attack_score_prototype": 0.05,
            },
            {
                "date": "2023-01-03",
                "market_regime_class_prototype": "defensive",
                "market_attack_score_prototype": -0.05,
            },
        ]
    ).to_parquet(out / "pm_v3_market_regime_daily_2023-01_to_2026-05.parquet", index=False)

    for path in [
        root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean",
        root / "models/ml/exit/current_v2_66",
        root / "config/profiles",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    (root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json").write_text(
        json.dumps({"model": "current_pm"}),
        encoding="utf-8",
    )
    (root / "models/ml/exit/current_v2_66/model_metadata.json").write_text(
        json.dumps({"model": "current_exit"}),
        encoding="utf-8",
    )
    (root / f"config/profiles/{PROFILE}.yaml").write_text(
        f"profile_id: {PROFILE}\n",
        encoding="utf-8",
    )


def test_phase9c_builds_dataset_quality_audit(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    report = PMAIV3DatasetQualityAudit(tmp_path).build_report()

    assert report["metadata"]["audit_only"] is True
    assert report["metadata"]["training_executed"] is False
    assert report["metadata"]["backtest_executed"] is False
    summary = report["basic_quality"]["summary"]
    assert summary["row_count"] == 6
    assert summary["duplicate_key_count"] == 0
    assert summary["feature_count"] > 0
    assert summary["label_count"] == 10
    assert report["same_day_ranking_audit"]["rank_bounds_valid"] is True
    assert report["leakage_audit"]["forbidden_feature_count"] == 0
    assert report["leakage_audit"]["label_columns_in_features"] == []
    assert report["verdict"]["dataset_is_trainable"] is True
    assert report["verdict"]["recommended_layer2_label"] == "relative_future_utility_percentile_in_day"


def test_phase9c_flags_forbidden_feature_columns(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    dataset_path = tmp_path / "data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_2023-01_to_2026-05.parquet"
    df = pd.read_parquet(dataset_path)
    df["cash_leak_feature"] = 1.0
    df.to_parquet(dataset_path, index=False)

    report = PMAIV3DatasetQualityAudit(tmp_path).build_report()

    assert report["leakage_audit"]["forbidden_feature_count"] == 1
    assert "cash_leak_feature" in report["leakage_audit"]["forbidden_feature_columns"]
    assert report["verdict"]["dataset_is_trainable"] is False
    assert report["verdict"]["next_phase_recommendation"] == "Phase 9-B2: Dataset Builder Fix"


def test_phase9c_saves_report_without_overwriting_current_artifacts(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    pm_file = tmp_path / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json"
    exit_file = tmp_path / "models/ml/exit/current_v2_66/model_metadata.json"
    profile_file = tmp_path / f"config/profiles/{PROFILE}.yaml"
    before = {
        "pm": pm_file.read_text(encoding="utf-8"),
        "exit": exit_file.read_text(encoding="utf-8"),
        "profile": profile_file.read_text(encoding="utf-8"),
    }

    audit = PMAIV3DatasetQualityAudit(tmp_path)
    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "9-C"
    assert loaded["metadata"]["current_pm_ai_overwritten"] is False
    assert loaded["metadata"]["current_exit_ai_overwritten"] is False
    assert loaded["metadata"]["v2_82_profile_overwritten"] is False

    assert pm_file.read_text(encoding="utf-8") == before["pm"]
    assert exit_file.read_text(encoding="utf-8") == before["exit"]
    assert profile_file.read_text(encoding="utf-8") == before["profile"]

