from __future__ import annotations

from pathlib import Path

from profile_loader import load_profile


def test_v282_profile_loads_with_cap38() -> None:
    profile = load_profile("rookie_dealer_02_v2_82_cap38")

    assert profile["profile_id"] == "rookie_dealer_02_v2_82_cap38"
    assert profile["portfolio_manager_ai_sizing"]["per_code_exposure_cap_enabled"] is True
    assert profile["portfolio_manager_ai_sizing"]["per_code_exposure_cap_rate"] == 0.38


def test_v278_profile_remains_cap30() -> None:
    profile = load_profile("rookie_dealer_02_v2_78_pm_aware_order_fallback_w025")

    assert profile["profile_id"] == "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
    assert profile["portfolio_manager_ai_sizing"]["per_code_exposure_cap_rate"] == 0.30


def test_v282_aliases_load_same_profile() -> None:
    profile = load_profile("rookie_dealer_02_v2_82_cap38")
    underscore_alias = load_profile("rookie_dealer_02_v2_82")
    dot_alias = load_profile("rookie_dealer_02_v2.82")

    assert underscore_alias["profile_id"] == profile["profile_id"]
    assert dot_alias["profile_id"] == profile["profile_id"]
    assert underscore_alias["portfolio_manager_ai_sizing"]["per_code_exposure_cap_rate"] == 0.38
    assert dot_alias["portfolio_manager_ai_sizing"]["per_code_exposure_cap_rate"] == 0.38


def test_v282_keeps_current_models_unchanged() -> None:
    v278 = load_profile("rookie_dealer_02_v2_78_pm_aware_order_fallback_w025")
    v282 = load_profile("rookie_dealer_02_v2_82_cap38")

    assert v282["ml_exit_ai"]["model_dir"] == v278["ml_exit_ai"]["model_dir"] == "models/ml/exit/current_v2_66"
    assert v282["portfolio_manager_ai_sizing"]["model_dir"] == v278["portfolio_manager_ai_sizing"]["model_dir"]


def test_v282_profile_does_not_mix_bear_booster_or_exit_ai_v2() -> None:
    profile = load_profile("rookie_dealer_02_v2_82_cap38")

    assert "bear_pm_booster" not in profile or not profile.get("bear_pm_booster", {}).get("bear_pm_booster_enabled", False)
    assert profile["ml_exit_ai"]["model_dir"] == "models/ml/exit/current_v2_66"
    assert "exit_ai_v2" not in profile["ml_exit_ai"]["model_dir"]


def test_phase6g_report_script_exists() -> None:
    assert Path("scripts/ml/report_portfolio_manager_phase6g_cap38.py").exists()
