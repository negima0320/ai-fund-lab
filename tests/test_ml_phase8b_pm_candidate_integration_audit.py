from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.phase8b_pm_candidate_integration_audit import Phase8BPMCandidateIntegrationAudit


PROFILE = "rookie_dealer_02_v2_82_cap38"
PERIOD = "2023-01-01_to_2026-05-31"


class _FixedProbaModel:
    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = float(positive_probability)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), self.positive_probability)
        return np.column_stack([1.0 - positive, positive])


def _write_fixture(root: Path) -> None:
    core = root / "reports" / "final" / "v2_82_cap38" / "core_2023-01_to_2026-05"
    core.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "exit_date": "2023-01-10",
                "code": "11110",
                "net_profit": 10_000,
                "pm_multiplier": 0.80,
                "pm_score": -0.10,
                "pm_high_conviction_proba": 0.40,
                "pm_avoid_proba": 0.50,
            },
            {
                "action": "SELL",
                "signal_date": "2023-01-05",
                "entry_date": "2023-01-06",
                "exit_date": "2023-01-11",
                "code": "22220",
                "net_profit": -5_000,
                "pm_multiplier": 1.30,
                "pm_score": 0.50,
                "pm_high_conviction_proba": 0.80,
                "pm_avoid_proba": 0.30,
            },
        ]
    ).to_csv(core / "trades.csv", index=False)

    current = root / "models" / "ml" / "portfolio_manager" / "current_v2_73_phase3b_clean"
    candidate = root / "models" / "ml" / "portfolio_manager" / "candidate_v2_api_only"
    current.mkdir(parents=True, exist_ok=True)
    candidate.mkdir(parents=True, exist_ok=True)
    (current / "feature_columns.json").write_text(
        json.dumps(["close", "volume_ratio_5d", "candidate_count_in_day", "rank_in_day"]),
        encoding="utf-8",
    )
    (candidate / "feature_columns.json").write_text(json.dumps(["close", "volume_ratio_5d"]), encoding="utf-8")
    (candidate / "preprocess.json").write_text(
        json.dumps(
            {
                "feature_columns": ["close", "volume_ratio_5d"],
                "numeric_columns": ["close", "volume_ratio_5d"],
                "categorical_columns": [],
                "medians": {"close": 100.0, "volume_ratio_5d": 1.0},
            }
        ),
        encoding="utf-8",
    )
    joblib.dump(_FixedProbaModel(0.80), candidate / "high_conviction_target_classifier.joblib")
    joblib.dump(_FixedProbaModel(0.20), candidate / "avoid_target_classifier.joblib")

    dataset_path = root / "data" / "ml" / "portfolio_manager_api_only"
    dataset_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"as_of_date": "2023-01-04", "code": "11110", "close": 100.0, "volume_ratio_5d": 2.0},
            {"as_of_date": "2023-01-05", "code": "22220", "close": 200.0, "volume_ratio_5d": 3.0},
        ]
    ).to_parquet(dataset_path / "pm_ai_api_only_dataset_2021-06_to_2026-05.parquet", index=False)


def test_phase8b_scores_candidate_and_compares_pm_decisions(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    report = Phase8BPMCandidateIntegrationAudit(tmp_path).build_report()

    assert report["metadata"]["audit_only"] is True
    assert report["metadata"]["backtest_executed"] is False
    assert report["coverage"]["prediction_available"] == 2
    assert report["agreement"]["agreement_rate"] == 0.5
    assert report["candidate_changes"]["promoted_trades"] == 1
    assert report["candidate_changes"]["pm080_removed_trades"] == 1
    assert report["feature_diff"]["candidate_list_dependent_removed_count"] == 2
    assert report["trust_verdict"]["candidate_pm_safe"] is True


def test_phase8b_saves_reports(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = Phase8BPMCandidateIntegrationAudit(tmp_path)

    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 8-B" in paths.markdown.read_text(encoding="utf-8")
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["final_verdict"]["next_phase_recommended"] in {
        "Phase 8-C PM Candidate Backtest",
        "Stay current PM",
        "Rebuild PM candidate",
    }

