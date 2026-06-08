from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase9a_pm_ai_rearchitecture_audit import (
    CONDITIONAL_RELATIVE_FEATURES,
    FORBIDDEN_TOKENS,
    Phase9APMAIRearchitectureAudit,
)


PROFILE = "rookie_dealer_02_v2_82_cap38"


def _write_fixture(root: Path) -> None:
    run = root / "reports/final/v2_82_cap38/core_2023-01_to_2026-05"
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "code": "11110",
                "decision": "BUY",
                "candidate_source": "selected",
                "candidate_rank": 1,
                "score_rank": 1,
                "expected_return_10d": 0.08,
                "risk_adjusted_score": 0.70,
                "bad_entry_probability_10d": 0.10,
                "pm_multiplier": 1.30,
                "market_regime": "attack",
            },
            {
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "code": "22220",
                "decision": "BUY",
                "candidate_source": "selected",
                "candidate_rank": 2,
                "score_rank": 2,
                "expected_return_10d": 0.03,
                "risk_adjusted_score": 0.20,
                "bad_entry_probability_10d": 0.30,
                "pm_multiplier": 1.00,
                "market_regime": "attack",
            },
            {
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "code": "33330",
                "decision": "SKIP",
                "candidate_source": "selected",
                "candidate_rank": 3,
                "score_rank": 3,
                "expected_return_10d": -0.01,
                "risk_adjusted_score": -0.10,
                "bad_entry_probability_10d": 0.55,
                "pm_multiplier": 0.80,
                "market_regime": "attack",
                "skip_reason": "selected_but_not_affordable",
            },
        ]
    ).to_csv(run / "purchase_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "exit_date": "2023-01-10",
                "code": "11110",
                "pm_multiplier": 1.30,
                "net_profit": 12000,
            },
            {
                "action": "SELL",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "exit_date": "2023-01-10",
                "code": "22220",
                "pm_multiplier": 1.00,
                "net_profit": -3000,
            },
        ]
    ).to_csv(run / "trades.csv", index=False)

    dataset = root / "data/ml/portfolio_manager"
    dataset.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "signal_date": "2023-01-04",
                "code": "11110",
                "return_5d": 0.04,
                "daily_range_ratio": 0.02,
                "volume": 100000,
                "turnover_value": 50000000,
                "volume_ratio_5d": 1.5,
                "EPS": 100.0,
                "EqAR": 0.55,
                "days_to_earnings": 14,
                "is_near_earnings": False,
            },
            {
                "signal_date": "2023-01-04",
                "code": "22220",
                "return_5d": 0.01,
                "daily_range_ratio": 0.03,
                "volume": 80000,
                "turnover_value": 30000000,
                "volume_ratio_5d": 1.1,
                "EPS": 50.0,
                "EqAR": 0.40,
                "days_to_earnings": 3,
                "is_near_earnings": True,
            },
        ]
    ).to_parquet(dataset / "portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet", index=False)

    (root / "data/ml/walk_forward_predictions").mkdir(parents=True, exist_ok=True)
    (root / "data/ml/walk_forward_predictions/predictions_2023-01-04.parquet").write_bytes(b"fixture")

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
        f"profile_id: {PROFILE}\nportfolio_manager_ai_sizing:\n  enabled: true\n",
        encoding="utf-8",
    )


def test_phase9a_builds_pm_ai_rearchitecture_audit(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    report = Phase9APMAIRearchitectureAudit(tmp_path).build_report()

    assert report["metadata"]["phase"] == "9-A"
    assert report["metadata"]["audit_only"] is True
    assert report["metadata"]["training_executed"] is False
    assert report["metadata"]["backtest_executed"] is False
    assert report["metadata"]["current_pm_ai_overwritten"] is False
    assert report["metadata"]["current_exit_ai_overwritten"] is False
    assert report["metadata"]["v2_82_profile_overwritten"] is False
    assert report["current_pm_ai_role"]["summary"]["buy_rows_with_pm"] == 2

    pm_rows = {
        row["pm_multiplier"]: row
        for row in report["current_pm_ai_role"]["pm_multiplier_feature_summary"]
    }
    assert pm_rows["1.3"]["avg_expected_return_10d"] == 0.08
    assert pm_rows["1.3"]["avg_candidate_count_in_day"] == 3.0
    assert pm_rows["1.3"]["avg_percentile_in_day"] == 1.0

    assert report["verdict"]["keep_current_v282"] is True
    assert report["verdict"]["replace_current_pm_now"] is False


def test_phase9a_feature_classification_blocks_forbidden_candidates(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    report = Phase9APMAIRearchitectureAudit(tmp_path).build_report()
    allowed = report["feature_classification"]["allowed"]["features"]
    conditional = report["feature_classification"]["conditional"]["features"]
    forbidden = report["feature_classification"]["forbidden"]["features"]

    assert "candidate_count_in_day" in conditional
    assert "rank_in_day" in conditional
    assert "percentile_in_day" in conditional
    assert "gap_to_best" in conditional
    assert "candidate_strength" in conditional
    assert set(CONDITIONAL_RELATIVE_FEATURES).issubset(set(conditional))

    for feature in allowed:
        lowered = feature.lower()
        assert not any(token in lowered for token in FORBIDDEN_TOKENS), feature

    forbidden_text = ",".join(forbidden)
    for token in ["selected", "bought", "affordable", "cash", "profit", "backtest", "result", "exit", "skip", "final_assets"]:
        assert token in forbidden_text

    checklist = report["leakage_risk_checklist"]
    assert checklist["forbidden_feature_candidate_count"] == 0
    assert checklist["conditional_relative_features_classified"] is True
    assert checklist["overall_leakage_risk"] == "low_for_phase9a_design"


def test_phase9a_saves_markdown_and_json_without_overwriting_current_artifacts(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    pm_file = tmp_path / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json"
    exit_file = tmp_path / "models/ml/exit/current_v2_66/model_metadata.json"
    profile_file = tmp_path / f"config/profiles/{PROFILE}.yaml"
    before = {
        "pm": pm_file.read_text(encoding="utf-8"),
        "exit": exit_file.read_text(encoding="utf-8"),
        "profile": profile_file.read_text(encoding="utf-8"),
    }

    audit = Phase9APMAIRearchitectureAudit(tmp_path)
    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "9-A"
    assert loaded["leakage_risk_checklist"]["current_artifacts_overwritten"] is False
    assert "Phase 9-A" in paths.markdown.read_text(encoding="utf-8")

    assert pm_file.read_text(encoding="utf-8") == before["pm"]
    assert exit_file.read_text(encoding="utf-8") == before["exit"]
    assert profile_file.read_text(encoding="utf-8") == before["profile"]

