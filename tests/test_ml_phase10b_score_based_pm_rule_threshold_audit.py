from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_score_based_rule import (
    ALLOWED_SCORE_FEATURES,
    apply_score_based_pm_rule,
    feature_leakage_audit,
    multiplier_for_score,
)
from ml.portfolio_manager_score_based_threshold_audit import OPT_PROFILES, PERIOD, ScoreBasedPMThresholdAudit


def _candidates() -> list[dict]:
    return [
        {"code": "1", "risk_adjusted_score": 0.9, "expected_return_10d": 0.9, "stock_selection_rank_score": 0.9, "candidate_strength": 90},
        {"code": "2", "risk_adjusted_score": 0.7, "expected_return_10d": 0.7, "stock_selection_rank_score": 0.7, "candidate_strength": 70},
        {"code": "3", "risk_adjusted_score": 0.5, "expected_return_10d": 0.5, "stock_selection_rank_score": 0.5, "candidate_strength": 50},
        {"code": "4", "risk_adjusted_score": 0.3, "expected_return_10d": 0.3, "stock_selection_rank_score": 0.3, "candidate_strength": 30},
        {"code": "5", "risk_adjusted_score": 0.1, "expected_return_10d": 0.1, "stock_selection_rank_score": 0.1, "candidate_strength": 10},
    ]


def _write_profile_logs(root: Path, label: str, profile: str, profit: float, pf: float, dd: float) -> None:
    base = root / "logs/backtests" / profile / PERIOD
    base.mkdir(parents=True, exist_ok=True)
    (base / "backtest_summary.json").write_text(
        json.dumps(
            {
                "initial_capital": 1_000_000,
                "final_assets": 1_000_000 + profit,
                "net_cumulative_profit": profit,
                "profit_factor": pf,
                "max_drawdown": dd,
                "win_rate": 0.52,
                "closed_trades_count": 2,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"date": "2026-01-05", "positions_value": 500_000, "total_assets": 1_000_000}]).to_csv(base / "summary.csv", index=False)
    pd.DataFrame(
        [
            {"action": "SELL", "signal_date": "2026-01-05", "exit_date": "2026-01-20", "code": "1", "pm_multiplier": 1.3, "net_profit": profit * 0.8, "holding_days": 5},
            {"action": "SELL", "signal_date": "2026-01-06", "exit_date": "2026-01-21", "code": "5", "pm_multiplier": 0.6, "net_profit": profit * 0.2, "holding_days": 4},
        ]
    ).to_csv(base / "trades.csv", index=False)
    pd.DataFrame(
        [
            {"decision": "BUY", "signal_date": "2026-01-05", "code": "1", "pm_multiplier": 1.3, "pm_rule_score": 1.0, "pm_rule_bucket": "PM1.30"},
            {"decision": "BUY", "signal_date": "2026-01-06", "code": "5", "pm_multiplier": 0.6, "pm_rule_score": 0.0, "pm_rule_bucket": "PM0.60"},
        ]
    ).to_csv(base / "purchase_audit.csv", index=False)


def _write_fixture(root: Path) -> None:
    base_profiles = {
        "v2_82_reference": ("rookie_dealer_02_v2_82_cap38", 300_000, 2.0, -0.05),
        "v2_95_pm_disabled": ("rookie_dealer_02_v2_95_pm_disabled_equal_weight_cap38", 100_000, 1.4, -0.10),
        "v2_96c_rule_c": ("rookie_dealer_02_v2_96c_score_based_pm_rule_c", 120_000, 1.45, -0.09),
    }
    for label, (profile, profit, pf, dd) in base_profiles.items():
        _write_profile_logs(root, label, profile, profit, pf, dd)
    for index, (label, profile) in enumerate(OPT_PROFILES.items(), start=1):
        _write_profile_logs(root, label, profile, 130_000 + index * 1_000, 1.50 + index * 0.01, -0.08)


def test_threshold_variants_map_scores_to_expected_multipliers() -> None:
    assert multiplier_for_score("score_based_rule_c", 0.96, threshold_variant="conservative_high") == 1.3
    assert multiplier_for_score("score_based_rule_c", 0.90, threshold_variant="conservative_high") == 1.15
    assert multiplier_for_score("score_based_rule_c", 0.05, threshold_variant="no_060") == 0.8
    assert multiplier_for_score("score_based_rule_c", 0.80, threshold_variant="no_115") == 1.0
    assert multiplier_for_score("score_based_rule_c", 0.20, threshold_variant="inverted_low_check") == 0.8
    assert multiplier_for_score("score_based_rule_c", 0.05, threshold_variant="inverted_low_check") == 1.0


def test_weight_variant_and_score_logging_fields_are_present() -> None:
    rows = apply_score_based_pm_rule(
        _candidates(),
        {
            "rule": "score_based_rule_c",
            "score_based_pm_threshold_variant": "original",
            "score_based_pm_weight_variant": "strength_heavy",
        },
    )

    assert rows[0]["pm_rule_score"] == rows[0]["score_based_pm_score"]
    assert rows[0]["pm_rule_weight_variant"] == "strength_heavy"
    assert rows[0]["pm_rule_bucket"] == "PM1.30"
    assert rows[-1]["pm_rule_candidate_strength_percentile"] == 0.0


def test_feature_leakage_guard_still_limits_to_four_features() -> None:
    audit = feature_leakage_audit(sorted(ALLOWED_SCORE_FEATURES))

    assert audit["forbidden_feature_count"] == 0
    assert audit["disallowed_feature_columns"] == []
    assert audit["leakage_risk"] == "low"


def test_phase10b_report_generates(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = ScoreBasedPMThresholdAudit(tmp_path)
    report = audit.build_report()
    paths = audit.save_report(report)

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["leakage_checklist"]["forbidden_feature_count"] == 0
    assert loaded["leakage_checklist"]["pm_ai_model_used"] is False
    assert loaded["leakage_checklist"]["pm_ai_v3_model_used"] is False
    assert loaded["score_distribution_by_profile"]["v2_96c_rule_c"][0]["score_count"] == 2
    assert loaded["best_candidate"]["summary"]["label"].startswith("v2_97")
