from __future__ import annotations

from profile_loader import load_profile


def test_rookie_dealer_config_loads(config: dict) -> None:
    assert config["dealer"]["id"] == "rookie_dealer_01"
    assert config["profile_id"] == "rookie_dealer_01"


def test_required_config_keys_exist(config: dict) -> None:
    assert config["portfolio"]["initial_cash"] > 0
    for key in [
        "data_provider",
        "trading",
        "selection",
        "costs",
        "database",
        "news",
        "ai_commentary",
        "ai_decision",
        "safety",
        "broker",
        "tachibana",
    ]:
        assert key in config


def test_rookie_dealer_02_profile_uses_intraday_stop() -> None:
    profile = load_profile("rookie_dealer_02")

    assert profile["profile_id"] == "rookie_dealer_02"
    assert profile["profile_name"] == "新人ディーラー2号"
    assert profile["execution"]["stop_loss_execution"] == "intraday_stop"
    assert "max_rsi_for_new_position" not in profile["selection"]


def test_rookie_dealer_02_v2_profile_uses_rsi_filter() -> None:
    profile = load_profile("rookie_dealer_02_v2")

    assert profile["profile_id"] == "rookie_dealer_02_v2"
    assert profile["profile_name"] == "新人ディーラー2号 v2"
    assert profile["execution"]["stop_loss_execution"] == "intraday_stop"
    assert profile["selection"]["max_rsi_for_new_position"] == 65
    assert profile["selection"]["reject_overheated_rsi"] is True


def test_rookie_dealer_02_v3_profile_uses_score_and_volume_filters() -> None:
    profile = load_profile("rookie_dealer_02_v3")

    assert profile["profile_id"] == "rookie_dealer_02_v3"
    assert profile["profile_name"] == "新人ディーラー2号 v3"
    assert profile["execution"]["stop_loss_execution"] == "intraday_stop"
    assert profile["selection"]["min_score"] == 74
    assert profile["selection"]["max_rsi_for_new_position"] == 65
    assert profile["selection"]["reject_overheated_rsi"] is True
    assert profile["volume_filter"]["enabled"] is True
    assert profile["volume_filter"]["min_volume_ratio"] == 3.0
    assert profile["broker"]["provider"] == "paper"
    assert profile["broker"]["live_trading_enabled"] is False


def test_rookie_dealer_03_profile_uses_fast_take_profit() -> None:
    profile = load_profile("rookie_dealer_03")

    assert profile["profile_id"] == "rookie_dealer_03"
    assert profile["profile_name"] == "新人ディーラー3号"
    assert profile["execution"]["stop_loss_execution"] == "intraday_stop"
    assert profile["trading"]["take_profit_rate"] == 0.03
    assert profile["risk"]["take_profit_pct"] == 0.03
    assert profile["trading"]["max_holding_days"] == 3
    assert profile["risk"]["max_holding_business_days"] == 3
    assert profile["broker"]["provider"] == "paper"
    assert profile["broker"]["live_trading_enabled"] is False
