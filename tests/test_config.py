from __future__ import annotations

from profile_loader import load_profile


def test_rookie_dealer_config_loads(config: dict) -> None:
    assert config["dealer"]["id"] == "rookie_dealer_02_v2_1"
    assert config["profile_id"] == "rookie_dealer_02_v2_1"
    assert config["data_provider"] == "jquants"
    assert config["broker"]["provider"] == "paper"
    assert config["_value_sources"]["profile"] == "config"


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


def test_rookie_dealer_02_v2_1_profile_uses_volume_filter() -> None:
    profile = load_profile("rookie_dealer_02_v2_1")

    assert profile["profile_id"] == "rookie_dealer_02_v2_1"
    assert profile["profile_name"] == "新人ディーラー2号 v2.1"
    assert profile["execution"]["stop_loss_execution"] == "intraday_stop"
    assert profile["selection"]["min_score"] == 45
    assert profile["selection"]["fallback_min_score"] == 40
    assert profile["selection"]["top_pick_min_score"] == 40
    assert profile["scoring"]["total_score_formula"] == "technical_plus_relative_strength_market_penalty"
    assert profile["selection"]["max_rsi_for_new_position"] == 65
    assert profile["selection"]["reject_overheated_rsi"] is True
    assert profile["volume_filter"]["enabled"] is True
    assert profile["volume_filter"]["min_volume_ratio"] == 2.0
    assert profile["broker"]["provider"] == "paper"
    assert profile["broker"]["live_trading_enabled"] is False


def test_rookie_dealer_02_v2_dot_1_alias_loads() -> None:
    profile = load_profile("rookie_dealer_02_v2.1")

    assert profile["profile_id"] == "rookie_dealer_02_v2_1"
    assert profile["volume_filter"]["enabled"] is True
    assert profile["volume_filter"]["min_volume_ratio"] == 2.0


def test_rookie_dealer_02_v2_volume_hot_guard_profiles_load() -> None:
    expected = {
        "rookie_dealer_02_v2_12": 5.0,
        "rookie_dealer_02_v2_13": 4.0,
        "rookie_dealer_02_v2_14": 3.5,
    }

    for profile_id, max_volume_ratio in expected.items():
        profile = load_profile(profile_id)

        assert profile["profile_id"] == profile_id
        assert profile["volume_filter"]["enabled"] is True
        assert profile["volume_filter"]["min_volume_ratio"] == 2.0
        assert profile["volume_filter"]["max_volume_ratio"] == max_volume_ratio
        assert "出来高過熱ガード" in profile["description"]
        assert profile["selection"]["min_score"] == 45
        assert profile["broker"]["provider"] == "paper"


def test_rookie_dealer_02_v2_dot_12_alias_loads() -> None:
    profile = load_profile("rookie_dealer_02_v2.12")

    assert profile["profile_id"] == "rookie_dealer_02_v2_12"
    assert profile["volume_filter"]["max_volume_ratio"] == 5.0


def test_rookie_dealer_02_v2_15_profile_uses_rsi_volume_hot_zone_filter() -> None:
    profile = load_profile("rookie_dealer_02_v2_15")

    assert profile["profile_id"] == "rookie_dealer_02_v2_15"
    assert profile["scoring"]["use_relative_strength_score"] is True
    assert profile["volume_filter"]["min_volume_ratio"] == 2.0
    assert profile["rsi_volume_hot_zone_filter"] == {
        "enabled": True,
        "min_rsi": 60,
        "min_volume_ratio": 3,
        "max_volume_ratio": 5,
        "reason": "rsi_volume_hot_zone",
    }


def test_rookie_dealer_02_v2_dot_15_alias_loads() -> None:
    profile = load_profile("rookie_dealer_02_v2.15")

    assert profile["profile_id"] == "rookie_dealer_02_v2_15"
    assert profile["rsi_volume_hot_zone_filter"]["enabled"] is True


def test_rookie_dealer_02_v2_2_profile_relaxes_risk_off_filter() -> None:
    profile = load_profile("rookie_dealer_02_v2_2")

    assert profile["profile_id"] == "rookie_dealer_02_v2_2"
    assert profile["profile_name"] == "新人ディーラー2号 v2.2"
    assert profile["execution"]["stop_loss_execution"] == "intraday_stop"
    assert profile["selection"]["min_score"] == 45
    assert profile["selection"]["max_rsi_for_new_position"] == 65
    assert profile["selection"]["reject_overheated_rsi"] is True
    assert profile["volume_filter"]["enabled"] is True
    assert profile["volume_filter"]["min_volume_ratio"] == 2.0
    assert profile["market_filter"]["risk_off_buy_policy"] == "relaxed"
    assert profile["market_filter"]["risk_off_max_buy_orders"] == 2
    assert profile["market_filter"]["risk_off_min_score"] == 50
    assert profile["broker"]["provider"] == "paper"
    assert profile["broker"]["live_trading_enabled"] is False


def test_rookie_dealer_02_v2_dot_2_alias_loads() -> None:
    profile = load_profile("rookie_dealer_02_v2.2")

    assert profile["profile_id"] == "rookie_dealer_02_v2_2"
    assert profile["market_filter"]["risk_off_buy_policy"] == "relaxed"
    assert profile["market_filter"]["risk_off_max_buy_orders"] == 2


def test_rookie_dealer_02_v2_3_profile_raises_min_score_only() -> None:
    base = load_profile("rookie_dealer_02_v2_1")
    profile = load_profile("rookie_dealer_02_v2_3")

    assert profile["profile_id"] == "rookie_dealer_02_v2_3"
    assert profile["profile_name"] == "新人ディーラー2号 v2.3"
    assert profile["selection"]["min_score"] == 47
    assert profile["selection"]["fallback_min_score"] == 47
    assert profile["selection"]["top_pick_min_score"] == 47
    assert profile["selection"]["max_rsi_for_new_position"] == 65
    assert profile["selection"]["reject_overheated_rsi"] is True
    assert profile["volume_filter"]["enabled"] is True
    assert profile["volume_filter"]["min_volume_ratio"] == 2.0
    assert profile["execution"]["stop_loss_execution"] == "intraday_stop"
    assert profile["broker"]["provider"] == "paper"
    assert profile["broker"]["live_trading_enabled"] is False
    assert base["selection"]["min_score"] == 45
    assert base["selection"]["fallback_min_score"] == 40
    assert base["selection"]["top_pick_min_score"] == 40


def test_rookie_dealer_02_v2_dot_3_alias_loads() -> None:
    profile = load_profile("rookie_dealer_02_v2.3")

    assert profile["profile_id"] == "rookie_dealer_02_v2_3"
    assert profile["selection"]["min_score"] == 47


def test_rookie_dealer_02_v2_4_profile_uses_conditional_selection() -> None:
    base = load_profile("rookie_dealer_02_v2_1")
    profile = load_profile("rookie_dealer_02_v2_4")
    conditional = profile["selection"]["conditional_selection"]

    assert profile["profile_id"] == "rookie_dealer_02_v2_4"
    assert profile["profile_name"] == "新人ディーラー2号 v2.4"
    assert profile["selection"]["min_score"] == 45
    assert profile["selection"]["fallback_min_score"] == 40
    assert profile["selection"]["top_pick_min_score"] == 40
    assert conditional["enabled"] is True
    assert conditional["low_score_range"] == {"min": 40, "max": 44}
    assert conditional["allow_if"]["min_volume_ratio"] == 3.0
    assert conditional["allow_if"]["required_candlestick_signals"] == ["volume_confirmed_breakout"]
    assert conditional["allow_if"]["min_rsi"] == 50
    assert conditional["allow_if"]["max_rsi"] == 65
    assert conditional["allow_if"]["allowed_market_regimes"] == ["risk_on", "neutral"]
    assert "conditional_selection" not in base["selection"]


def test_rookie_dealer_02_v2_dot_4_alias_loads() -> None:
    profile = load_profile("rookie_dealer_02_v2.4")

    assert profile["profile_id"] == "rookie_dealer_02_v2_4"
    assert profile["selection"]["conditional_selection"]["enabled"] is True


def test_rookie_dealer_02_v2_6_profile_uses_relative_strength_score() -> None:
    base = load_profile("rookie_dealer_02_v2_1")
    profile = load_profile("rookie_dealer_02_v2_6")

    assert profile["profile_id"] == "rookie_dealer_02_v2_6"
    assert profile["profile_name"] == "新人ディーラー2号 v2.6"
    assert profile["features"]["relative_strength"] is True
    assert profile["scoring"]["use_relative_strength_score"] is True
    assert profile["scoring"]["total_score_formula"] == "technical_plus_relative_strength_market_penalty"
    assert profile["scoring"]["relative_strength_score_weight"] == 10
    assert profile["selection"]["min_score"] == 45
    assert profile["selection"]["fallback_min_score"] == 40
    assert profile["selection"]["top_pick_min_score"] == 40
    assert base["features"].get("relative_strength") is None
    assert "scoring" not in base or not base["scoring"].get("use_relative_strength_score")


def test_rookie_dealer_02_v2_dot_6_alias_loads() -> None:
    profile = load_profile("rookie_dealer_02_v2.6")

    assert profile["profile_id"] == "rookie_dealer_02_v2_6"
    assert profile["features"]["relative_strength"] is True


def test_rookie_dealer_02_v2_11_profile_uses_investor_context_filter() -> None:
    profile = load_profile("rookie_dealer_02_v2_11")

    assert profile["profile_id"] == "rookie_dealer_02_v2_11"
    assert profile["features"]["investor_context"] is True
    assert profile["scoring"].get("use_investor_context_score") is False
    assert profile["investor_context_filter"]["enabled"] is True
    assert profile["investor_context_filter"]["reason"] == "investor_context_negative"


def test_rookie_dealer_02_v3_profile_uses_score_and_volume_filters() -> None:
    profile = load_profile("rookie_dealer_02_v3")

    assert profile["profile_id"] == "rookie_dealer_02_v3"
    assert profile["profile_name"] == "新人ディーラー2号 v3"
    assert profile["execution"]["stop_loss_execution"] == "intraday_stop"
    assert profile["selection"]["min_score"] == 49
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
