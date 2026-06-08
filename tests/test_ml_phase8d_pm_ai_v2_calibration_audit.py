from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase8d_pm_ai_v2_calibration_audit import Phase8DPMCalibrationAudit, assign_calibration


def _sample_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_high_conviction_proba": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "candidate_avoid_proba": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "candidate_pm_score": [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "current_pm_multiplier": [0.8, 0.8, 1.0, 1.0, 1.0, 1.15, 1.15, 1.3, 1.3, 1.3],
            "net_profit": [-10, -5, 1, 2, 3, 4, 5, 10, 20, 30],
        }
    )


def test_assign_calibration_rules_create_pm130_and_pm080() -> None:
    frame = _sample_scores()

    rule_c = assign_calibration(frame, "Rule C")
    rule_e = assign_calibration(frame, "Rule E")

    assert (rule_c == 1.30).sum() >= 1
    assert (rule_c == 0.80).sum() >= 2
    assert set(rule_e.unique()).issubset({0.8, 1.0, 1.15, 1.3})


def test_phase8d_rule_result_estimates_profit() -> None:
    audit = Phase8DPMCalibrationAudit(Path("."))
    frame = _sample_scores()

    row = audit._rule_result(frame, "Rule C", base_v282=100.0, base_v290=50.0)

    assert row["rule"] == "Rule C"
    assert row["pm130_count"] >= 1
    assert row["pm080_count"] >= 1
    assert row["estimated_profit_delta_vs_v2_90"] is not None
    assert 0 <= row["current_pm130_recall"] <= 1


def test_phase8d_save_report_writes_markdown_and_json(tmp_path: Path) -> None:
    audit = Phase8DPMCalibrationAudit(tmp_path)
    report = {
        "score_distribution": [{"source": "x", "metric": "candidate_pm_score", "min": 0.0, "p10": 0.1, "p25": 0.2, "p50": 0.5, "p75": 0.7, "p90": 0.9, "p95": 0.95, "max": 1.0}],
        "current_thresholds": {"pm130": "x", "pm115": "x", "pm100": "x", "pm080": "x", "threshold_too_strict": True},
        "calibration_candidates": [{"rule": "Rule C", "estimated_profit": 1, "estimated_profit_delta_vs_v2_90": 1, "estimated_profit_delta_vs_v2_82": -1, "estimated_pf": 2, "estimated_capital_utilization_direction": "up", "estimated_dd_direction": "limited", "pm130_count": 1, "pm130_profit_approximation": 1, "pm080_count": 1, "pm080_profit_approximation": 1, "current_pm130_recall": 0.5, "current_pm130_precision": 0.5}],
        "pm130_reproducibility": [{"rule": "Rule C", "current_pm130_recall_by_rule": 0.5, "current_pm130_precision_by_rule": 0.5, "recovered_pm130_profit": 1, "missed_pm130_profit": 1}],
        "pm080_overuse_check": [{"rule": "Rule C", "pm080_count_by_rule": 1, "pm080_profit_by_rule": 1, "pm080_overuse_risk": False}],
        "verdict": {"calibration_rule_recommended": "Rule C", "pm_ai_v2_calibration_feasible": True, "pm_ai_v2_needs_retraining": False, "pm_ai_v2_needs_label_redesign": False, "ready_for_phase8e_backtest": True, "reason": "ok"},
    }

    paths = audit.save_report(report)

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 8-D" in paths.markdown.read_text(encoding="utf-8")
    assert json.loads(paths.json.read_text(encoding="utf-8"))["verdict"]["ready_for_phase8e_backtest"] is True

