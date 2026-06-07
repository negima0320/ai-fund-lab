from __future__ import annotations

import importlib.util
from pathlib import Path

from profile_loader import load_profile

import paper_trade


def _config(days: int = 5) -> dict:
    return {
        "portfolio_manager_ai_sizing": {
            "enabled": True,
            "high_pm_min_hold_enabled": True,
            "high_pm_min_hold_days": days,
            "high_pm_min_hold_min_multiplier": 1.15,
        }
    }


def test_v2_79_profiles_load_with_high_pm_min_hold() -> None:
    five = load_profile("rookie_dealer_02_v2_79_high_pm_min_hold_5d")
    seven = load_profile("rookie_dealer_02_v2_79_high_pm_min_hold_7d")

    assert five["portfolio_manager_ai_sizing"]["high_pm_min_hold_days"] == 5
    assert seven["portfolio_manager_ai_sizing"]["high_pm_min_hold_days"] == 7
    assert five["portfolio_manager_ai_sizing"]["selected_count_in_day_forbidden"] is True


def test_v2_78_profile_is_not_changed() -> None:
    config = load_profile("rookie_dealer_02_v2_78_pm_aware_order_fallback_w025")

    assert config["profile_id"] == "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
    assert config["portfolio_manager_ai_sizing"].get("high_pm_min_hold_enabled") is None


def test_high_pm_min_hold_blocks_exit_ai_only() -> None:
    plan = {
        "exit_reason": "Exit AI avoid_loss_5d",
        "exit_price": 100.0,
        "intended_exit_price": 100.0,
        "execute_now": True,
        "exit_ai_triggered": True,
    }
    updated, fields = paper_trade._apply_high_pm_min_hold_exit_guard(
        plan,
        {"entry_date": "2026-03-02", "pm_multiplier": 1.15},
        _config(5),
        "2026-03-04",
        2,
    )

    assert updated["exit_reason"] == ""
    assert updated["exit_ai_triggered"] is False
    assert fields["high_pm_min_hold_blocked_exit"] is True
    assert fields["high_pm_min_hold_blocked_exit_count"] == 1
    assert fields["high_pm_min_hold_release_date"] == "2026-03-06"


def test_high_pm_min_hold_does_not_apply_to_low_pm() -> None:
    plan = {"exit_reason": "Exit AI avoid_loss_5d", "exit_ai_triggered": True, "exit_price": 100.0}

    updated, fields = paper_trade._apply_high_pm_min_hold_exit_guard(
        plan,
        {"entry_date": "2026-03-02", "pm_multiplier": 1.0},
        _config(5),
        "2026-03-04",
        2,
    )

    assert updated["exit_reason"] == "Exit AI avoid_loss_5d"
    assert fields["high_pm_min_hold_applied"] is False
    assert fields["high_pm_min_hold_blocked_exit"] is False


def test_high_pm_min_hold_does_not_block_hard_stop_like_exit() -> None:
    plan = {"exit_reason": "stop_loss", "exit_ai_triggered": False, "exit_price": 90.0}

    updated, fields = paper_trade._apply_high_pm_min_hold_exit_guard(
        plan,
        {"entry_date": "2026-03-02", "pm_multiplier": 1.3},
        _config(7),
        "2026-03-04",
        2,
    )

    assert updated["exit_reason"] == "stop_loss"
    assert fields["high_pm_min_hold_applied"] is True
    assert fields["high_pm_min_hold_blocked_exit"] is False


def test_phase4c_report_handles_missing_logs(tmp_path: Path) -> None:
    script = Path("scripts/ml/report_portfolio_manager_phase4c_high_pm_min_hold.py")
    spec = importlib.util.spec_from_file_location("phase4c_report", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.build_report(tmp_path)
    markdown, json_path = module.save_report(result, tmp_path)

    assert result["constraints"]["selected_count_in_day_used"] is False
    assert {row["status"] for row in result["comparison"]} == {"missing_backtest_logs"}
    assert markdown.exists()
    assert json_path.exists()
