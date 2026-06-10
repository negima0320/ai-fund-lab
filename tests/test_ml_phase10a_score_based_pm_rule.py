from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import paper_trade
from ml.portfolio_manager_score_based_rule import (
    ALLOWED_SCORE_FEATURES,
    apply_score_based_pm_rule,
    feature_leakage_audit,
)
from ml.portfolio_manager_score_based_rule_audit import PERIOD, RULE_PROFILES, ScoreBasedPMRuleAudit
from profile_loader import load_profile


def _candidates() -> list[dict]:
    return [
        {"code": "10001", "risk_adjusted_score": 0.90, "expected_return_10d": 0.08, "stock_selection_rank_score": 0.95, "candidate_strength": 90},
        {"code": "10002", "risk_adjusted_score": 0.75, "expected_return_10d": 0.07, "stock_selection_rank_score": 0.70, "candidate_strength": 75},
        {"code": "10003", "risk_adjusted_score": 0.50, "expected_return_10d": 0.05, "stock_selection_rank_score": 0.55, "candidate_strength": 55},
        {"code": "10004", "risk_adjusted_score": 0.25, "expected_return_10d": 0.03, "stock_selection_rank_score": 0.30, "candidate_strength": 30},
        {"code": "10005", "risk_adjusted_score": 0.10, "expected_return_10d": 0.01, "stock_selection_rank_score": 0.05, "candidate_strength": 10},
    ]


def _config(rule: str) -> dict:
    return {
        "portfolio_manager_ai_sizing": {
            "enabled": True,
            "rule": rule,
            "model_dir": "",
            "dataset_path": "",
            "low_score_skip_enabled": False,
            "buy_ordering_mode": "default",
        },
        "trading": {"use_round_lot": True, "round_lot_size": 100},
        "capital_utilization_policy": {"buy_lot_size": 100},
    }


def _write_fixture(root: Path) -> None:
    labels = {"v2_82_reference": "rookie_dealer_02_v2_82_cap38", "v2_95_pm_disabled": "rookie_dealer_02_v2_95_pm_disabled_equal_weight_cap38", **RULE_PROFILES}
    for label, profile in labels.items():
        base = root / "logs/backtests" / profile / PERIOD
        base.mkdir(parents=True, exist_ok=True)
        is_rule = label.startswith("v2_96")
        is_reference = label == "v2_82_reference"
        profit = 300_000 if is_reference else 120_000 if is_rule else 100_000
        pf = 2.0 if is_reference else 1.6 if is_rule else 1.4
        dd = -0.05 if is_reference else -0.06 if is_rule else -0.07
        summary = {
            "initial_capital": 1_000_000,
            "final_assets": 1_000_000 + profit,
            "net_cumulative_profit": profit,
            "profit_factor": pf,
            "max_drawdown": dd,
            "win_rate": 0.55 if is_reference else 0.51 if is_rule else 0.49,
            "closed_trades_count": 3,
        }
        (base / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        pd.DataFrame([{"date": "2026-01-05", "positions_value": 500_000, "total_assets": 1_000_000}]).to_csv(base / "summary.csv", index=False)
        pd.DataFrame(
            [
                {"action": "SELL", "signal_date": "2026-01-05", "exit_date": "2026-01-20", "code": "10001", "pm_multiplier": 1.3 if is_rule else 1.0, "net_profit": 60_000, "holding_days": 5},
                {"action": "SELL", "signal_date": "2026-01-06", "exit_date": "2026-01-21", "code": "10005", "pm_multiplier": 0.6 if is_rule else 1.0, "net_profit": -20_000, "holding_days": 4},
            ]
        ).to_csv(base / "trades.csv", index=False)
        pd.DataFrame(
            [
                {"decision": "BUY", "signal_date": "2026-01-05", "code": "10001", "pm_multiplier": 1.3 if is_rule else 1.0, "score_based_pm_score": 1.0 if is_rule else ""},
                {"decision": "BUY", "signal_date": "2026-01-06", "code": "10005", "pm_multiplier": 0.6 if is_rule else 1.0, "score_based_pm_score": 0.0 if is_rule else ""},
            ]
        ).to_csv(base / "purchase_audit.csv", index=False)


def test_rule_a_b_c_assign_expected_multipliers() -> None:
    rule_a = apply_score_based_pm_rule(_candidates(), "score_based_rule_a")
    rule_b = apply_score_based_pm_rule(_candidates(), "score_based_rule_b")
    rule_c = apply_score_based_pm_rule(_candidates(), "score_based_rule_c")

    assert [row["pm_multiplier"] for row in rule_a] == [1.3, 1.15, 1.0, 0.8, 0.6]
    assert [row["pm_multiplier"] for row in rule_b] == [1.3, 1.15, 1.0, 0.8, 0.6]
    assert [row["pm_multiplier"] for row in rule_c] == [1.3, 1.15, 1.0, 0.8, 0.6]
    assert all(row["pm_multiplier_source"] == "score_based_pm_rule" for row in rule_c)


def test_score_based_rule_features_are_limited_and_forbidden_free() -> None:
    audit = feature_leakage_audit(sorted(ALLOWED_SCORE_FEATURES))

    assert set(audit["feature_columns"]) == ALLOWED_SCORE_FEATURES
    assert audit["forbidden_feature_count"] == 0
    assert audit["disallowed_feature_columns"] == []
    assert audit["leakage_risk"] == "low"


def test_score_based_pm_does_not_call_pm_advisor(monkeypatch) -> None:
    config = _config("score_based_rule_a")
    selected = paper_trade._apply_score_based_pm_rule_to_candidates(_candidates(), config)
    item = selected[0]

    def fail_advisor(_config):
        raise AssertionError("PM advisor must not be read for score-based PM rules")

    monkeypatch.setattr(paper_trade, "_portfolio_manager_sizing_advisor", fail_advisor)
    shares, fields = paper_trade._apply_portfolio_manager_sizing(
        item=item,
        trade_date="2026-01-05",
        shares=100,
        entry_price=1000.0,
        cash=500_000.0,
        config=config,
    )

    assert shares == 100
    assert fields["pm_multiplier"] == 1.3
    assert fields["pm_ai_enabled"] is False
    assert fields["pm_model_path"] == ""
    assert fields["pm_multiplier_source"] == "score_based_pm_rule"


def test_v296_profiles_load_aliases_and_do_not_overwrite_v282() -> None:
    a = load_profile("rookie_dealer_02_v2_96")
    dot = load_profile("rookie_dealer_02_v2.96")
    b = load_profile("rookie_dealer_02_v2_96b")
    c = load_profile("rookie_dealer_02_v2_96c")

    assert a["profile_id"] == "rookie_dealer_02_v2_96_score_based_pm_rule_a"
    assert dot["profile_id"] == a["profile_id"]
    assert b["portfolio_manager_ai_sizing"]["rule"] == "score_based_rule_b"
    assert c["portfolio_manager_ai_sizing"]["rule"] == "score_based_rule_c"
    for profile in [a, b, c]:
        policy = profile["portfolio_manager_ai_sizing"]
        assert policy["model_dir"] == ""
        assert policy["dataset_path"] == ""
        assert paper_trade._portfolio_manager_pm_aware_ordering_enabled(profile) is False
    assert load_profile("rookie_dealer_02_v2_82")["profile_id"] == "rookie_dealer_02_v2_82_cap38"


def test_phase10a_report_generates(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    manifest = {
        "dry_run_reported": True,
        "deleted": True,
        "deleted_path_count": 2,
        "bytes_before": 100,
        "bytes_after": 0,
        "bytes_saved": 100,
        "paths": ["data/ml/portfolio_manager_v3", "models/ml/portfolio_manager_v3"],
    }
    manifest_path = tmp_path / "reports/ml/phase10a_phase9_cleanup_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    audit = ScoreBasedPMRuleAudit(tmp_path)
    report = audit.build_report()
    paths = audit.save_report(report)

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["leakage_checklist"]["forbidden_feature_count"] == 0
    assert loaded["leakage_checklist"]["pm_ai_model_used"] is False
    assert loaded["leakage_checklist"]["pm_ai_v3_model_used"] is False
    assert loaded["adoption"]["candidate_gate_passed"] is True
    assert loaded["phase9_cleanup"]["deleted"] is True
