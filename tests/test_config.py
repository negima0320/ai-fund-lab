from __future__ import annotations

from profile_loader import load_profile
from scoring import _apply_selection_rules, _selection_config


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


def test_rookie_dealer_02_v2_capital_exposure_profiles_load() -> None:
    expected = {
        "rookie_dealer_02_v2_20": 0.7,
        "rookie_dealer_02_v2_21": 0.8,
        "rookie_dealer_02_v2_22": 1.0,
    }

    for profile_id, max_position_value_rate in expected.items():
        profile = load_profile(profile_id)

        assert profile["profile_id"] == profile_id
        assert profile["scoring"]["use_relative_strength_score"] is True
        assert profile["selection"]["min_score"] == 45
        assert profile["selection"]["max_rsi_for_new_position"] == 65
        assert profile["disable_single_order_amount_limit"] is True
        assert profile["capital_utilization_policy"]["enabled"] is True
        assert profile["capital_utilization_policy"]["target_exposure"] == 0.9
        assert profile["capital_utilization_policy"]["max_position_value_rate"] == max_position_value_rate
        assert profile["capital_utilization_policy"]["buy_as_much_as_possible"] is True
        assert profile["trading"]["round_lot_size"] == 100
        assert profile["safety"]["max_single_order_amount"] == 500000


def test_rookie_dealer_02_v2_dot_20_alias_loads() -> None:
    profile = load_profile("rookie_dealer_02_v2.20")

    assert profile["profile_id"] == "rookie_dealer_02_v2_20"
    assert profile["capital_utilization_policy"]["max_position_value_rate"] == 0.7


def test_rookie_dealer_02_v2_affordability_profiles_load() -> None:
    expected = {
        "rookie_dealer_02_v2_23": 500000,
        "rookie_dealer_02_v2_24": 400000,
        "rookie_dealer_02_v2_25": 300000,
    }

    for profile_id, preferred_round_lot_amount in expected.items():
        profile = load_profile(profile_id)

        assert profile["profile_id"] == profile_id
        assert profile["scoring"]["use_relative_strength_score"] is True
        assert profile["disable_single_order_amount_limit"] is True
        assert profile["capital_utilization_policy"]["enabled"] is True
        assert profile["capital_utilization_policy"]["target_exposure"] == 0.9
        assert profile["capital_utilization_policy"]["max_position_value_rate"] == 0.5
        assert profile["affordability_filter"]["enabled"] is True
        assert profile["affordability_filter"]["preferred_round_lot_amount"] == preferred_round_lot_amount
        assert profile["affordability_filter"]["penalty_points"] == 3
        assert profile["affordability_filter"]["reason"] == "price_band_penalty"


def test_rookie_dealer_02_v2_dot_23_alias_loads() -> None:
    profile = load_profile("rookie_dealer_02_v2.23")

    assert profile["profile_id"] == "rookie_dealer_02_v2_23"
    assert profile["affordability_filter"]["preferred_round_lot_amount"] == 500000


def test_rookie_dealer_02_v2_allocation_strategy_profiles_load() -> None:
    expected = {
        "rookie_dealer_02_v2_26": "relaxed_pending_target_exposure",
        "rookie_dealer_02_v2_27": "same_day_equal_budget",
        "rookie_dealer_02_v2_28": "round_lot_priority_near_score",
    }

    for profile_id, allocation_strategy in expected.items():
        profile = load_profile(profile_id)

        assert profile["profile_id"] == profile_id
        assert profile["scoring"]["use_relative_strength_score"] is True
        assert profile["disable_single_order_amount_limit"] is True
        assert profile["capital_utilization_policy"]["enabled"] is True
        assert profile["capital_utilization_policy"]["target_exposure"] == 0.9
        assert profile["capital_utilization_policy"]["max_position_value_rate"] == 0.5
        assert profile["capital_utilization_policy"]["allocation_strategy"] == allocation_strategy
        assert profile["capital_utilization_policy"]["buy_as_much_as_possible"] is True


def test_rookie_dealer_02_v2_dot_26_alias_loads() -> None:
    profile = load_profile("rookie_dealer_02_v2.26")

    assert profile["profile_id"] == "rookie_dealer_02_v2_26"
    assert profile["capital_utilization_policy"]["allocation_strategy"] == "relaxed_pending_target_exposure"


def test_rookie_dealer_02_v2_v26_capital_relaxation_profiles_load() -> None:
    expected = {
        "rookie_dealer_02_v2_29": (0.6, 0.9),
        "rookie_dealer_02_v2_30": (0.7, 0.9),
        "rookie_dealer_02_v2_32": (0.7, 0.95),
    }

    for profile_id, (max_position_value_rate, target_exposure) in expected.items():
        profile = load_profile(profile_id)

        assert profile["profile_id"] == profile_id
        assert profile["scoring"]["use_relative_strength_score"] is True
        assert profile["disable_single_order_amount_limit"] is True
        assert profile["capital_utilization_policy"]["enabled"] is True
        assert profile["capital_utilization_policy"]["allocation_strategy"] == "relaxed_pending_target_exposure"
        assert profile["capital_utilization_policy"]["max_position_value_rate"] == max_position_value_rate
        assert profile["capital_utilization_policy"]["target_exposure"] == target_exposure
        assert profile["capital_utilization_policy"]["buy_as_much_as_possible"] is True


def test_rookie_dealer_02_v2_dot_29_alias_loads() -> None:
    profile = load_profile("rookie_dealer_02_v2.29")

    assert profile["profile_id"] == "rookie_dealer_02_v2_29"
    assert profile["capital_utilization_policy"]["max_position_value_rate"] == 0.6


def test_rookie_dealer_02_v2_dynamic_exposure_profiles_load() -> None:
    expected = {
        "rookie_dealer_02_v2_35": {"strong_bull": 1.0, "bull": 0.90, "range": 0.70, "bear": 0.40, "strong_bear": 0.0},
    }

    for profile_id, target_exposure_by_regime in expected.items():
        profile = load_profile(profile_id)

        assert profile["profile_id"] == profile_id
        assert profile["disable_single_order_amount_limit"] is True
        assert profile["capital_utilization_policy"]["allocation_strategy"] == "relaxed_pending_target_exposure"
        assert profile["dynamic_exposure"]["enabled"] is True
        assert profile["dynamic_exposure"]["target_exposure_by_regime"] == target_exposure_by_regime


def test_rookie_dealer_02_v2_38_holding_revaluation_profile_loads() -> None:
    profile_38 = load_profile("rookie_dealer_02_v2_38")

    assert profile_38["profile_id"] == "rookie_dealer_02_v2_38"
    assert profile_38["holding_revaluation"]["enabled"] is True
    assert profile_38["holding_revaluation"]["hold_extension_max_days"] == 10
    assert profile_38["holding_revaluation"]["early_exit_on_signal_lost"] is True
    assert profile_38["holding_revaluation"]["score_drop_exit_threshold"] == 10
    assert profile_38["capital_utilization_policy"]["allocation_strategy"] == "relaxed_pending_target_exposure"
    assert load_profile("rookie_dealer_02_v2.38")["profile_id"] == "rookie_dealer_02_v2_38"


def test_rookie_dealer_02_v2_39_hold_extension_only_profile_loads() -> None:
    profile_39 = load_profile("rookie_dealer_02_v2_39")

    assert profile_39["profile_id"] == "rookie_dealer_02_v2_39"
    assert profile_39["holding_revaluation"]["enabled"] is True
    assert profile_39["holding_revaluation"]["hold_reselection_enabled"] is True
    assert profile_39["holding_revaluation"]["hold_extension_max_days"] == 10
    assert profile_39["holding_revaluation"]["early_exit_on_signal_lost"] is False
    assert "score_drop_exit_threshold" not in profile_39["holding_revaluation"]
    assert profile_39["capital_utilization_policy"]["allocation_strategy"] == "relaxed_pending_target_exposure"
    assert load_profile("rookie_dealer_02_v2.39")["profile_id"] == "rookie_dealer_02_v2_39"


def test_rookie_dealer_02_v2_52_to_54_holding_revaluation_profiles_load() -> None:
    profile_52 = load_profile("rookie_dealer_02_v2_52")
    profile_53 = load_profile("rookie_dealer_02_v2_53")
    profile_54 = load_profile("rookie_dealer_02_v2_54")

    assert profile_52["holding_revaluation"]["signal_lost_exit_min_consecutive_days"] == 2
    assert profile_52["holding_revaluation"]["early_exit_on_signal_lost"] is True
    assert profile_53["holding_revaluation"]["suppress_signal_lost_exit_when_unrealized_profit"] is True
    assert profile_54["holding_revaluation"]["hold_reselection_enabled"] is True
    assert profile_54["holding_revaluation"]["hold_extension_require_confirmation"] is True
    assert profile_54["holding_revaluation"]["hold_extension_max_count"] == 1
    assert load_profile("rookie_dealer_02_v2.52")["profile_id"] == "rookie_dealer_02_v2_52"
    assert load_profile("rookie_dealer_02_v2.53")["profile_id"] == "rookie_dealer_02_v2_53"
    assert load_profile("rookie_dealer_02_v2.54")["profile_id"] == "rookie_dealer_02_v2_54"


def test_rookie_dealer_02_v2_55_max_holding_days_only_profile_loads() -> None:
    profile_55 = load_profile("rookie_dealer_02_v2_55")

    assert profile_55["profile_id"] == "rookie_dealer_02_v2_55"
    assert profile_55["trading"]["max_holding_days"] == 7
    assert profile_55["capital_utilization_policy"]["allocation_strategy"] == "relaxed_pending_target_exposure"
    assert "holding_revaluation" not in profile_55
    assert load_profile("rookie_dealer_02_v2.55")["profile_id"] == "rookie_dealer_02_v2_55"


def test_rookie_dealer_02_v2_56_conditional_hold_extension_profile_loads() -> None:
    profile_56 = load_profile("rookie_dealer_02_v2_56")

    assert profile_56["profile_id"] == "rookie_dealer_02_v2_56"
    assert profile_56["trading"]["max_holding_days"] == 5
    assert profile_56["conditional_hold_extension"]["enabled"] is True
    assert profile_56["conditional_hold_extension"]["min_unrealized_profit_rate"] == 0.03
    assert profile_56["conditional_hold_extension"]["max_holding_days"] == 7
    assert "holding_revaluation" not in profile_56
    assert load_profile("rookie_dealer_02_v2.56")["profile_id"] == "rookie_dealer_02_v2_56"


def test_rookie_dealer_02_v2_57_trend_continuation_hold_extension_profile_loads() -> None:
    profile_57 = load_profile("rookie_dealer_02_v2_57")

    assert profile_57["profile_id"] == "rookie_dealer_02_v2_57"
    assert profile_57["trading"]["max_holding_days"] == 5
    assert profile_57["conditional_hold_extension"]["enabled"] is True
    assert profile_57["conditional_hold_extension"]["require_trend_continuation"] is True
    assert profile_57["conditional_hold_extension"]["min_unrealized_profit_rate"] == 0.015
    assert profile_57["conditional_hold_extension"]["min_relative_strength_score"] == 60
    assert profile_57["conditional_hold_extension"]["max_holding_days"] == 7
    assert "holding_revaluation" not in profile_57
    assert load_profile("rookie_dealer_02_v2.57")["profile_id"] == "rookie_dealer_02_v2_57"


def test_rookie_dealer_02_v2_58_ma25_uptrend_hold_extension_profile_loads() -> None:
    profile_58 = load_profile("rookie_dealer_02_v2_58")

    assert profile_58["profile_id"] == "rookie_dealer_02_v2_58"
    assert profile_58["trading"]["max_holding_days"] == 5
    assert profile_58["conditional_hold_extension"]["enabled"] is True
    assert profile_58["conditional_hold_extension"]["require_trend_continuation"] is True
    assert profile_58["conditional_hold_extension"]["skip_ma5_condition"] is True
    assert profile_58["conditional_hold_extension"]["require_ma25_uptrend"] is True
    assert profile_58["conditional_hold_extension"]["min_unrealized_profit_rate"] == 0.03
    assert profile_58["conditional_hold_extension"]["min_relative_strength_score"] == 60
    assert profile_58["conditional_hold_extension"]["profit_reject_reason"] == "profit_below_threshold"
    assert profile_58["conditional_hold_extension"]["relative_strength_reject_reason"] == "relative_strength_below_threshold"
    assert profile_58["conditional_hold_extension"]["max_holding_days"] == 7
    assert "holding_revaluation" not in profile_58
    assert load_profile("rookie_dealer_02_v2.58")["profile_id"] == "rookie_dealer_02_v2_58"


def test_rookie_dealer_02_v2_59_indicator_enriched_hold_extension_profile_loads() -> None:
    profile_59 = load_profile("rookie_dealer_02_v2_59")

    assert profile_59["profile_id"] == "rookie_dealer_02_v2_59"
    assert profile_59["trading"]["max_holding_days"] == 5
    assert profile_59["conditional_hold_extension"] == load_profile("rookie_dealer_02_v2_58")["conditional_hold_extension"]
    assert profile_59["conditional_hold_extension"]["require_ma25_uptrend"] is True
    assert profile_59["conditional_hold_extension"]["min_unrealized_profit_rate"] == 0.03
    assert profile_59["conditional_hold_extension"]["min_relative_strength_score"] == 60
    assert "holding_revaluation" not in profile_59
    assert load_profile("rookie_dealer_02_v2.59")["profile_id"] == "rookie_dealer_02_v2_59"


def test_rookie_dealer_02_v2_61_report_enhanced_hold_extension_profile_loads() -> None:
    profile_61 = load_profile("rookie_dealer_02_v2_61")

    assert profile_61["profile_id"] == "rookie_dealer_02_v2_61"
    assert profile_61["trading"]["max_holding_days"] == 5
    assert profile_61["conditional_hold_extension"] == load_profile("rookie_dealer_02_v2_60")["conditional_hold_extension"]
    assert profile_61["conditional_hold_extension"]["min_relative_strength_score"] == 5
    assert profile_61["conditional_hold_extension"]["minimum_relative_strength_score"] == 5
    assert "holding_revaluation" not in profile_61
    assert load_profile("rookie_dealer_02_v2.61")["profile_id"] == "rookie_dealer_02_v2_61"


def test_rookie_dealer_02_v2_62_extension_exit_guard_profile_loads() -> None:
    profile_62 = load_profile("rookie_dealer_02_v2_62")

    assert profile_62["profile_id"] == "rookie_dealer_02_v2_62"
    assert profile_62["trading"]["max_holding_days"] == 5
    conditional = profile_62["conditional_hold_extension"]
    assert conditional["enabled"] is True
    assert conditional["min_relative_strength_score"] == 5
    assert conditional["minimum_relative_strength_score"] == 5
    guard = conditional["extension_exit_guard"]
    assert guard["enabled"] is True
    assert guard["max_profit_pullback_points"] == 0.02
    assert guard["min_remaining_profit_rate"] == 0.01
    assert guard["exit_reason"] == "延長後失速撤退"
    assert profile_62["capital_utilization_policy"] == load_profile("rookie_dealer_02_v2_61")["capital_utilization_policy"]
    assert profile_62["risk_margin"] == load_profile("rookie_dealer_02_v2_61")["risk_margin"]
    assert "holding_revaluation" not in profile_62
    assert load_profile("rookie_dealer_02_v2.62")["profile_id"] == "rookie_dealer_02_v2_62"


def test_rookie_dealer_02_v2_63_risk_off_entry_filter_profile_loads() -> None:
    profile_63 = load_profile("rookie_dealer_02_v2_63")

    assert profile_63["profile_id"] == "rookie_dealer_02_v2_63"
    assert profile_63["trading"] == load_profile("rookie_dealer_02_v2_26")["trading"]
    assert profile_63["scoring"] == load_profile("rookie_dealer_02_v2_26")["scoring"]
    assert profile_63["capital_utilization_policy"] == load_profile("rookie_dealer_02_v2_26")["capital_utilization_policy"]
    assert profile_63["market_filter"]["risk_off_max_buy_orders"] == 0
    assert profile_63["market_filter"]["risk_off_disable_top_pick"] is True
    assert load_profile("rookie_dealer_02_v2.63")["profile_id"] == "rookie_dealer_02_v2_63"


def test_rookie_dealer_02_v2_64_risk_off_relative_strength_filter_profile_loads() -> None:
    profile_64 = load_profile("rookie_dealer_02_v2_64")

    assert profile_64["profile_id"] == "rookie_dealer_02_v2_64"
    assert profile_64["trading"] == load_profile("rookie_dealer_02_v2_26")["trading"]
    assert profile_64["scoring"] == load_profile("rookie_dealer_02_v2_26")["scoring"]
    assert profile_64["capital_utilization_policy"] == load_profile("rookie_dealer_02_v2_26")["capital_utilization_policy"]
    assert profile_64["market_filter"]["risk_off_min_score"] == 50
    assert profile_64["market_filter"]["risk_off_relative_strength_min_score"] == 10
    assert profile_64["market_filter"]["risk_off_max_buy_orders"] == load_profile("rookie_dealer_02_v2_26")["market_filter"]["risk_off_max_buy_orders"]
    assert load_profile("rookie_dealer_02_v2.64")["profile_id"] == "rookie_dealer_02_v2_64"


def test_rookie_dealer_02_v2_65_score_upper_filter_profile_loads() -> None:
    profile_65 = load_profile("rookie_dealer_02_v2_65")

    assert profile_65["profile_id"] == "rookie_dealer_02_v2_65"
    assert profile_65["trading"] == load_profile("rookie_dealer_02_v2_26")["trading"]
    assert profile_65["scoring"] == load_profile("rookie_dealer_02_v2_26")["scoring"]
    assert profile_65["capital_utilization_policy"] == load_profile("rookie_dealer_02_v2_26")["capital_utilization_policy"]
    assert profile_65["score_upper_filter"]["enabled"] is True
    assert profile_65["score_upper_filter"]["max_entry_score"] == 55
    assert profile_65["score_upper_filter"]["rejected_reason"] == "score_upper_filter"
    assert load_profile("rookie_dealer_02_v2.65")["profile_id"] == "rookie_dealer_02_v2_65"


def test_rookie_dealer_02_v2_40_affordable_fallback_quality_profile_loads() -> None:
    profile_40 = load_profile("rookie_dealer_02_v2_40")

    assert profile_40["affordable_fallback_buy"]["max_rank_in_day"] == 20
    assert load_profile("rookie_dealer_02_v2.40")["profile_id"] == "rookie_dealer_02_v2_40"


def test_rookie_dealer_02_v2_41_to_42_market_section_expansion_profiles_load() -> None:
    profile_41 = load_profile("rookie_dealer_02_v2_41")
    profile_42 = load_profile("rookie_dealer_02_v2_42")

    assert profile_41["profile_id"] == "rookie_dealer_02_v2_41"
    assert profile_41["market_filter"]["allowed_sections"] == ["TSEPrime", "TSEStandard"]
    assert profile_42["profile_id"] == "rookie_dealer_02_v2_42"
    assert profile_42["market_filter"]["allowed_sections"] == ["TSEPrime", "TSEStandard", "TSEGrowth"]
    assert profile_41["capital_utilization_policy"]["allocation_strategy"] == "relaxed_pending_target_exposure"
    assert profile_42["scoring"]["use_relative_strength_score"] is True
    assert load_profile("rookie_dealer_02_v2.41")["profile_id"] == "rookie_dealer_02_v2_41"
    assert load_profile("rookie_dealer_02_v2.42")["profile_id"] == "rookie_dealer_02_v2_42"


def test_rookie_dealer_02_v2_43_to_47_standard_screening_profiles_load() -> None:
    profile_43 = load_profile("rookie_dealer_02_v2_43")
    profile_44 = load_profile("rookie_dealer_02_v2_44")
    profile_45 = load_profile("rookie_dealer_02_v2_45")
    profile_46 = load_profile("rookie_dealer_02_v2_46")
    profile_47 = load_profile("rookie_dealer_02_v2_47")

    for profile in [profile_43, profile_44, profile_45, profile_46, profile_47]:
        assert profile["market_filter"]["allowed_sections"] == ["TSEPrime", "TSEStandard"]
        assert "TSEStandard" in profile["screening"]["market_overrides"]
        assert "TSEPrime" not in profile["screening"]["market_overrides"]
        assert profile["capital_utilization_policy"]["allocation_strategy"] == "relaxed_pending_target_exposure"

    assert profile_43["screening"]["market_overrides"]["TSEStandard"]["min_turnover_value"] == 300000000
    assert profile_44["screening"]["market_overrides"]["TSEStandard"]["min_volume_ratio"] == 1.2
    assert profile_45["screening"]["market_overrides"]["TSEStandard"]["require_close_above_ma5"] is False
    assert profile_46["screening"]["market_overrides"]["TSEStandard"]["require_ma5_above_ma25"] is False
    assert profile_47["screening"]["market_overrides"]["TSEStandard"]["rsi_max"] == 75
    assert load_profile("rookie_dealer_02_v2.47")["profile_id"] == "rookie_dealer_02_v2_47"


def test_rookie_dealer_02_v2_48_to_50_standard_min_score_profiles_load() -> None:
    expected = {
        "rookie_dealer_02_v2_48": 35,
        "rookie_dealer_02_v2_49": 30,
        "rookie_dealer_02_v2_50": 25,
    }

    for profile_id, standard_min_score in expected.items():
        profile = load_profile(profile_id)

        assert profile["profile_id"] == profile_id
        assert profile["market_filter"]["allowed_sections"] == ["TSEPrime", "TSEStandard"]
        assert profile["selection"]["min_score"] == 45
        assert profile["selection"]["market_min_score_overrides"] == {"TSEStandard": standard_min_score}
        assert "TSEStandard" in profile["screening"]["market_overrides"]
        assert profile["capital_utilization_policy"]["allocation_strategy"] == "relaxed_pending_target_exposure"

    assert load_profile("rookie_dealer_02_v2.48")["profile_id"] == "rookie_dealer_02_v2_48"
    assert load_profile("rookie_dealer_02_v2.49")["profile_id"] == "rookie_dealer_02_v2_49"
    assert load_profile("rookie_dealer_02_v2.50")["profile_id"] == "rookie_dealer_02_v2_50"


def test_standard_market_min_score_override_selects_only_standard() -> None:
    profile = load_profile("rookie_dealer_02_v2_48")
    scoring_rows = [
        {
            "code": "2001",
            "market_section": "TSEStandard",
            "section": "TSEStandard",
            "listing_market": "TSEStandard",
            "total_score": 37,
            "confidence": 0.8,
            "selected": False,
        },
        {
            "code": "1001",
            "market_section": "TSEPrime",
            "section": "TSEPrime",
            "listing_market": "TSEPrime",
            "total_score": 37,
            "confidence": 0.8,
            "selected": False,
        },
    ]

    _apply_selection_rules(
        scoring_rows,
        _selection_config(profile),
        {
            "enabled": False,
            "risk_off_buy_policy": "conservative",
            "risk_off_max_buy_orders": 1,
            "risk_off_min_score": 50,
            "risk_off_disable_top_pick": True,
            "allowed_sections": {"TSEPrime", "TSEStandard"},
            "allow_unknown_market": False,
        },
        "neutral",
    )

    assert scoring_rows[0]["selected"] is True
    assert scoring_rows[0]["effective_min_score"] == 35
    assert scoring_rows[0]["market_min_score_override_applied"] is True
    assert scoring_rows[1]["selected"] is False
    assert scoring_rows[1]["effective_min_score"] == 45


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
