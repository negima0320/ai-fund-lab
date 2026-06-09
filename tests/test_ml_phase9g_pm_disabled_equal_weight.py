from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import paper_trade
from ml.portfolio_manager_pm_disabled_equal_weight_audit import (
    BASELINE_PROFILE,
    PERIOD,
    PM_DISABLED_PROFILE,
    PMDisabledEqualWeightAudit,
)
from profile_loader import load_profile


def _disabled_config() -> dict:
    return {
        "portfolio_manager_ai_sizing": {
            "enabled": True,
            "rule": "disabled_equal_weight",
            "low_score_skip_enabled": False,
            "buy_ordering_mode": "default",
            "fallback_to_next_affordable_selected": True,
            "fallback_min_pm_multiplier": 1.0,
            "per_code_exposure_cap_enabled": True,
            "per_code_exposure_cap_rate": 0.38,
        },
        "trading": {"use_round_lot": True, "round_lot_size": 100},
        "capital_utilization_policy": {"buy_lot_size": 100},
    }


def _write_backtest_fixture(root: Path) -> None:
    for label, profile in {"v2_82_cap38": BASELINE_PROFILE, "v2_95": PM_DISABLED_PROFILE}.items():
        base = root / "logs/backtests" / profile / PERIOD
        base.mkdir(parents=True, exist_ok=True)
        is_disabled = label == "v2_95"
        summary = {
            "initial_capital": 1_000_000,
            "final_assets": 1_300_000 if not is_disabled else 1_220_000,
            "net_cumulative_profit": 300_000 if not is_disabled else 220_000,
            "profit_factor": 2.0 if not is_disabled else 1.4,
            "max_drawdown": -0.05 if not is_disabled else -0.07,
            "win_rate": 0.55 if not is_disabled else 0.50,
            "closed_trades_count": 3,
        }
        (base / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        pd.DataFrame(
            [
                {"date": "2026-01-05", "positions_value": 600_000, "total_assets": 1_000_000},
                {"date": "2026-01-06", "positions_value": 700_000, "total_assets": 1_100_000},
            ]
        ).to_csv(base / "summary.csv", index=False)
        multiplier = 1.0 if is_disabled else 1.3
        pd.DataFrame(
            [
                {"action": "SELL", "signal_date": "2026-01-05", "exit_date": "2026-01-20", "code": "10001", "pm_multiplier": multiplier, "net_profit": 100_000 if not is_disabled else 50_000, "holding_days": 5},
                {"action": "SELL", "signal_date": "2026-01-06", "exit_date": "2026-01-21", "code": "10002", "pm_multiplier": 1.0, "net_profit": -20_000, "holding_days": 4},
            ]
        ).to_csv(base / "trades.csv", index=False)
        if is_disabled:
            audit_rows = [
                {"decision": "BUY", "signal_date": "2026-01-05", "code": "10001", "pm_multiplier": 1.0, "pm_status": "disabled", "pm_model_version": "disabled", "pm_missing_reason": "pm_disabled", "pm_warning": "pm_disabled", "selection_source": "regular", "per_code_exposure_cap_applied": False},
                {"decision": "BUY", "signal_date": "2026-01-06", "code": "10002", "pm_multiplier": 1.0, "pm_status": "disabled", "pm_model_version": "disabled", "pm_missing_reason": "pm_disabled", "pm_warning": "pm_disabled", "selection_source": "affordable_fallback_buy", "per_code_exposure_cap_applied": True},
                {"decision": "SKIP", "signal_date": "2026-01-07", "code": "10003", "skip_reason": "selected_but_not_affordable"},
            ]
        else:
            audit_rows = [
                {"decision": "BUY", "signal_date": "2026-01-05", "code": "10001", "pm_multiplier": 1.3, "pm_status": "ok", "pm_model_version": "current"},
                {"decision": "BUY", "signal_date": "2026-01-06", "code": "10002", "pm_multiplier": 1.0, "pm_status": "ok", "pm_model_version": "current"},
            ]
        pd.DataFrame(audit_rows).to_csv(base / "purchase_audit.csv", index=False)


def test_pm_disabled_rule_does_not_read_pm_advisor(monkeypatch) -> None:
    config = _disabled_config()
    item = {"signal_date": "2026-01-05", "code": "10001"}

    def fail_advisor(_config):
        raise AssertionError("PM advisor must not be read when PM is disabled")

    monkeypatch.setattr(paper_trade, "_portfolio_manager_sizing_advisor", fail_advisor)
    shares, fields = paper_trade._apply_portfolio_manager_sizing(
        item=item,
        trade_date="2026-01-05",
        shares=200,
        entry_price=1000.0,
        cash=500_000.0,
        config=config,
    )

    assert shares == 200
    assert fields["pm_multiplier"] == 1.0
    assert fields["pm_status"] == "disabled"
    assert fields["pm_model_version"] == "disabled"
    assert fields["pm_missing_reason"] == "pm_disabled"
    assert fields["pm_multiplier_source"] == "pm_disabled_equal_weight"
    assert paper_trade._portfolio_manager_pm_aware_ordering_enabled(config) is False


def test_pm_disabled_decision_fields_do_not_call_current_or_v3_pm(monkeypatch) -> None:
    config = _disabled_config()
    item = {"signal_date": "2026-01-05", "code": "10001"}

    def fail_advisor(_config):
        raise AssertionError("PM advisor must not be read for decision fields")

    monkeypatch.setattr(paper_trade, "_portfolio_manager_sizing_advisor", fail_advisor)
    fields = paper_trade._ensure_portfolio_manager_decision_fields(item, config)

    assert fields["pm_status"] == "disabled"
    assert fields["pm_multiplier"] == 1.0
    assert item["pm_model_version"] == "disabled"


def test_v295_profile_loads_aliases_and_disables_pm_ordering() -> None:
    profile = load_profile(PM_DISABLED_PROFILE)
    underscore_alias = load_profile("rookie_dealer_02_v2_95")
    dot_alias = load_profile("rookie_dealer_02_v2.95")

    assert profile["profile_id"] == PM_DISABLED_PROFILE
    assert underscore_alias["profile_id"] == PM_DISABLED_PROFILE
    assert dot_alias["profile_id"] == PM_DISABLED_PROFILE
    pm_policy = profile["portfolio_manager_ai_sizing"]
    assert pm_policy["rule"] == "disabled_equal_weight"
    assert pm_policy["model_dir"] == ""
    assert pm_policy["dataset_path"] == ""
    assert pm_policy["low_score_skip_enabled"] is False
    assert pm_policy["buy_ordering_mode"] == "default"
    assert paper_trade._portfolio_manager_pm_aware_ordering_enabled(profile) is False


def test_phase9g_report_generates_and_verifies_pm_disabled(tmp_path: Path) -> None:
    _write_backtest_fixture(tmp_path)
    audit = PMDisabledEqualWeightAudit(tmp_path)
    report = audit.build_report()
    paths = audit.save_report(report)

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    correctness = loaded["pm_disabled_correctness"]
    assert correctness["pm_disabled_correct"] is True
    assert correctness["non_pm100_buy_count"] == 0
    assert correctness["pm_model_lookup_used"] is False
    assert correctness["pm_ai_v3_lookup_used"] is False
    assert loaded["ordering_method"]["pm_aware_ordering_used"] is False
    assert loaded["leakage_checklist"]["leakage_risk"] == "low"
    assert loaded["metadata"]["v2_82_profile_overwritten"] is False
