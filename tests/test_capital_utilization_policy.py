from __future__ import annotations

from copy import deepcopy

from paper_trade import (
    _available_buy_budget,
    _configured_daily_buy_limit,
    _daily_buy_limit_remaining,
    _scale_buy_to_daily_limit,
    execute_real_data_paper_trade,
    initial_live_paper_state,
)


def _candidate(code: str = "10010", price: float = 3000.0) -> dict:
    return {
        "code": code,
        "name": f"Test {code}",
        "sector_name": "機械",
        "date": "2026-01-05",
        "entry_date": "2026-01-06",
        "entry_price": price,
        "close": price,
        "selected": True,
        "total_score": 55.0,
        "technical_score": 50.0,
        "confidence": 0.9,
        "reason": "test selected",
        "selected_reason": "test selected",
        "selection_reason": "test selected",
        "market_section": "TSEPrime",
        "section": "TSEPrime",
        "listing_market": "TSEPrime",
        "rsi": 60.0,
        "volume_ratio": 2.5,
    }


def _enable_policy(
    config: dict,
    *,
    max_position_value_rate: float = 1.0,
    min_cash_buffer: int = 50000,
    disable_single_order_amount_limit: bool = True,
    allocation_strategy: str | None = None,
) -> None:
    config["portfolio"]["initial_cash"] = 1_000_000
    config["portfolio"]["max_positions"] = 10
    config["portfolio"]["max_allocation_per_symbol"] = 0.20
    config["selection"]["fallback_min_score"] = 40
    config["selection"]["min_confidence"] = 0.7
    config["trading"]["use_round_lot"] = True
    config["trading"]["round_lot_size"] = 100
    config["safety"]["max_single_order_amount"] = 200000
    config["safety"]["max_daily_buy_amount"] = 900000
    config["disable_single_order_amount_limit"] = disable_single_order_amount_limit
    config.setdefault("execution", {})["use_next_day_open_execution"] = False
    config["capital_utilization_policy"] = {
        "enabled": True,
        "target_exposure": 0.9,
        "min_cash_buffer": min_cash_buffer,
        "max_position_value_rate": max_position_value_rate,
        "buy_as_much_as_possible": True,
        "buy_lot_size": 100,
    }
    if allocation_strategy:
        config["capital_utilization_policy"]["allocation_strategy"] = allocation_strategy


def test_capital_policy_buys_maximum_round_lots(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0)
    state = initial_live_paper_state(config_copy)

    new_state, _summary, trades = execute_real_data_paper_trade([_candidate(price=3000)], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert len(buys) == 1
    assert buys[0]["shares"] == 300
    assert buys[0]["amount"] == 900000
    assert buys[0]["allocation_reason"] == "capital_utilization_policy"
    assert new_state["cash"] == 100000


def test_scaled_buy_reduces_order_to_daily_buy_limit(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0)
    config_copy["scaled_buy"] = {"enabled": True}

    shares, fields = _scale_buy_to_daily_limit(
        shares=20200,
        price=47,
        config=config_copy,
        today_orders=[],
    )

    assert shares == 19100
    assert fields == {
        "scaled_buy_triggered": True,
        "original_planned_shares": 20200,
        "scaled_shares": 19100,
        "original_amount": 949400,
        "scaled_amount": 897700,
        "scale_reason": "daily_buy_limit",
        "daily_buy_limit_type": "fixed",
        "daily_buy_limit_ratio": None,
        "daily_buy_limit_applied": 900000,
    }


def test_scaled_buy_uses_fixed_limit_override(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0)
    config_copy["scaled_buy"] = {"enabled": True, "daily_buy_limit": 500000}

    shares, fields = _scale_buy_to_daily_limit(
        shares=600,
        price=1000,
        config=config_copy,
        today_orders=[],
    )

    assert _configured_daily_buy_limit(config_copy, total_assets=2_000_000) == 500000
    assert shares == 500
    assert fields["daily_buy_limit_type"] == "fixed"
    assert fields["daily_buy_limit_applied"] == 500000


def test_scaled_buy_uses_asset_ratio_limit(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0)
    config_copy["scaled_buy"] = {"enabled": True, "limit_mode": "asset_ratio", "daily_buy_limit_ratio": 0.5}

    shares, fields = _scale_buy_to_daily_limit(
        shares=1200,
        price=1000,
        config=config_copy,
        today_orders=[],
        total_assets=1_500_000,
    )

    assert _configured_daily_buy_limit(config_copy, total_assets=1_500_000) == 750000
    assert shares == 700
    assert fields["daily_buy_limit_type"] == "asset_ratio"
    assert fields["daily_buy_limit_ratio"] == 0.5
    assert fields["daily_buy_limit_applied"] == 750000


def test_scaled_buy_unlimited_keeps_risk_controls_but_no_daily_cap(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5)
    config_copy["scaled_buy"] = {"enabled": True, "limit_mode": "unlimited"}
    state = initial_live_paper_state(config_copy)

    _new_state, _summary, trades = execute_real_data_paper_trade([_candidate(price=3000)], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert _configured_daily_buy_limit(config_copy, total_assets=1_000_000) == 0
    assert _daily_buy_limit_remaining(config_copy, [], total_assets=1_000_000) == float("inf")
    assert buys[0]["shares"] == 100


def test_scaled_buy_allows_buy_that_would_exceed_daily_limit(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0, min_cash_buffer=0)
    config_copy["capital_utilization_policy"]["target_exposure"] = 0.95
    config_copy["scaled_buy"] = {"enabled": True}
    state = initial_live_paper_state(config_copy)

    new_state, _summary, trades = execute_real_data_paper_trade([_candidate(price=47)], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert len(buys) == 1
    assert buys[0]["shares"] == 19100
    assert buys[0]["amount"] == 897700
    assert buys[0]["scaled_buy_triggered"] is True
    assert buys[0]["original_planned_shares"] == 20200
    assert buys[0]["scaled_shares"] == 19100
    assert buys[0]["scale_reason"] == "daily_buy_limit"
    assert not [trade for trade in trades if trade.get("order_status") == "REJECTED"]
    assert new_state["positions"][0]["shares"] == 19100
    assert new_state["positions"][0]["scaled_buy_triggered"] is True


def _enable_ai_purchase_policy(config: dict) -> None:
    _enable_policy(config, max_position_value_rate=1.0, min_cash_buffer=0, allocation_strategy="relaxed_pending_target_exposure")
    config["capital_utilization_policy"]["target_exposure"] = 1.0
    config["scaled_buy"] = {"enabled": True}
    config["ai_purchase_policy"] = {
        "enabled": True,
        "daily_buy_limit": 900000,
        "max_position_amount_ratio": 0.30,
        "max_position_amount_abs": 900000,
        "rank_position_ratio_tiers": [
            {"max_rank": 1, "ratio": 0.30},
            {"max_rank": 3, "ratio": 0.20},
            {"max_rank": None, "ratio": 0.10},
        ],
    }


def test_ai_purchase_policy_uses_rank_based_allocation_and_audit(config_copy: dict) -> None:
    _enable_ai_purchase_policy(config_copy)
    state = initial_live_paper_state(config_copy)
    first = _candidate("10010", 1000)
    first["daily_score_rank"] = 1
    first["risk_adjusted_score"] = 0.30
    second = _candidate("10020", 1000)
    second["daily_score_rank"] = 2
    second["risk_adjusted_score"] = 0.20

    new_state, _summary, trades = execute_real_data_paper_trade([first, second], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    audits = [trade for trade in trades if trade.get("action") == "PURCHASE_AUDIT"]
    assert [buy["code"] for buy in buys] == ["10010", "10020"]
    assert [buy["amount"] for buy in buys] == [300000, 200000]
    assert [audit["decision"] for audit in audits] == ["BUY", "BUY"]
    assert audits[0]["daily_buy_limit_remaining_before"] == 900000
    assert audits[0]["daily_buy_limit_remaining_after"] == 600000
    assert audits[1]["daily_buy_limit_remaining_before"] == 600000
    assert audits[1]["daily_buy_limit_remaining_after"] == 400000
    assert new_state["cash"] == 500000


def test_ai_purchase_policy_skips_unaffordable_top_and_continues_next_candidate(config_copy: dict) -> None:
    _enable_ai_purchase_policy(config_copy)
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 4000)
    expensive["daily_score_rank"] = 4
    expensive["risk_adjusted_score"] = 0.40
    affordable = _candidate("90020", 1000)
    affordable["daily_score_rank"] = 5
    affordable["risk_adjusted_score"] = 0.30

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive, affordable], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    audits = [trade for trade in trades if trade.get("action") == "PURCHASE_AUDIT"]
    assert [buy["code"] for buy in buys] == ["90020"]
    assert audits[0]["code"] == "90010"
    assert audits[0]["decision"] == "SKIP"
    assert audits[0]["skip_reason"] == "round_lot_unaffordable"
    assert audits[1]["code"] == "90020"
    assert audits[1]["decision"] == "BUY"


def test_ai_purchase_policy_continues_after_round_lot_skip(config_copy: dict) -> None:
    _enable_ai_purchase_policy(config_copy)
    state = initial_live_paper_state(config_copy)
    first = _candidate("10010", 1000)
    first["daily_score_rank"] = 1
    first["risk_adjusted_score"] = 0.30
    second = _candidate("10020", 6500)
    second["daily_score_rank"] = 2
    second["risk_adjusted_score"] = 0.20
    third = _candidate("10030", 1000)
    third["daily_score_rank"] = 3
    third["risk_adjusted_score"] = 0.10

    _new_state, _summary, trades = execute_real_data_paper_trade([first, second, third], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    audits = [trade for trade in trades if trade.get("action") == "PURCHASE_AUDIT"]
    assert [buy["code"] for buy in buys] == ["10010", "10030"]
    second_audit = [audit for audit in audits if audit["code"] == "10020"][0]
    assert second_audit["decision"] == "SKIP"
    assert second_audit["skip_reason"] == "round_lot_unaffordable"


def test_purchase_audit_scaled_buy_continue_preserves_v71_sizing(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0, min_cash_buffer=0, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["capital_utilization_policy"]["target_exposure"] = 1.0
    config_copy["safety"]["max_daily_buy_amount"] = 500000
    config_copy["scaled_buy"] = {"enabled": True}
    config_copy["purchase_audit"] = {"enabled": True}
    state = initial_live_paper_state(config_copy)
    first = _candidate("70010", 2000)
    first["total_score"] = 60
    second = _candidate("70020", 1000)
    second["total_score"] = 59

    new_state, _summary, trades = execute_real_data_paper_trade([first, second], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    audits = [trade for trade in trades if trade.get("action") == "PURCHASE_AUDIT"]
    assert [buy["code"] for buy in buys] == ["70010", "70020"]
    assert [buy["shares"] for buy in buys] == [200, 100]
    assert [buy["amount"] for buy in buys] == [400000, 100000]
    assert [audit["decision"] for audit in audits] == ["SCALED_BUY", "SCALED_BUY"]
    assert audits[0]["daily_buy_limit_remaining_before"] == 500000
    assert audits[0]["daily_buy_limit_remaining_after"] == 100000
    assert audits[0]["daily_buy_limit_type"] == "fixed"
    assert audits[0]["daily_buy_limit_applied"] == 500000
    assert audits[1]["daily_buy_limit_remaining_before"] == 100000
    assert audits[1]["daily_buy_limit_remaining_after"] == 0
    assert new_state["cash"] == 500000


def test_capital_policy_buys_stock_that_legacy_200k_limit_cannot_buy(config_copy: dict) -> None:
    config_copy["portfolio"]["initial_cash"] = 1_000_000
    config_copy["portfolio"]["max_allocation_per_symbol"] = 0.20
    config_copy["selection"]["fallback_min_score"] = 40
    config_copy["selection"]["min_confidence"] = 0.7
    config_copy["trading"]["use_round_lot"] = True
    config_copy["trading"]["round_lot_size"] = 100
    config_copy.setdefault("execution", {})["use_next_day_open_execution"] = False

    legacy_state = initial_live_paper_state(config_copy)
    _new_state, _summary, legacy_trades = execute_real_data_paper_trade([_candidate(price=3000)], legacy_state, config_copy, "2026-01-06")
    assert [trade for trade in legacy_trades if trade.get("action") == "SKIP_BUY"][0]["skipped_reason"] == "round_lot_unaffordable"

    _enable_policy(config_copy, max_position_value_rate=1.0)
    policy_state = initial_live_paper_state(config_copy)
    _new_state, _summary, policy_trades = execute_real_data_paper_trade([_candidate(price=3000)], policy_state, config_copy, "2026-01-06")
    buys = [trade for trade in policy_trades if trade.get("action") == "BUY"]
    assert buys and buys[0]["shares"] == 300


def test_capital_policy_respects_min_cash_buffer(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0, min_cash_buffer=50_000)
    state = initial_live_paper_state(config_copy)
    state["cash"] = 40_000
    state["total_assets"] = 40_000

    _new_state, _summary, trades = execute_real_data_paper_trade([_candidate(price=3000)], state, config_copy, "2026-01-06")

    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert len(skipped) == 1
    assert skipped[0]["skipped_reason"] == "insufficient_available_cash"


def test_capital_policy_does_not_exceed_max_position_value_rate(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5)
    state = initial_live_paper_state(config_copy)

    _new_state, _summary, trades = execute_real_data_paper_trade([_candidate(price=3000)], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert len(buys) == 1
    assert buys[0]["shares"] == 100
    assert buys[0]["amount"] <= 500000


def test_capital_policy_does_not_exceed_max_positions(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0)
    config_copy["portfolio"]["max_positions"] = 1
    state = initial_live_paper_state(config_copy)
    state["positions"] = [
        {
            "code": "HELD",
            "name": "Held",
            "entry_price": 1000,
            "current_price": 1000,
            "shares": 100,
            "market_value": 100000,
            "holding_days": 1,
        }
    ]

    _new_state, _summary, trades = execute_real_data_paper_trade([_candidate(price=3000)], state, config_copy, "2026-01-06")

    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert not buys
    assert len(skipped) == 1
    assert skipped[0]["skipped_reason"] == "max_positions_limit"


def test_v2_19_style_budget_subtracts_pending_from_target_exposure(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0)
    state = initial_live_paper_state(config_copy)
    state["positions"] = [
        {
            "code": "HELD",
            "name": "Held",
            "entry_price": 8000,
            "current_price": 8000,
            "shares": 100,
            "market_value": 800000,
        }
    ]
    state["total_assets"] = 1_000_000

    allocation, reason = _available_buy_budget(1_000_000, 1_000_000, state, config_copy, pending_buy_amount=100_000)

    assert allocation == 0
    assert reason == "target_exposure_limit"


def test_v2_26_budget_keeps_pending_on_cash_side_only(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0, allocation_strategy="relaxed_pending_target_exposure")
    state = initial_live_paper_state(config_copy)
    state["positions"] = [
        {
            "code": "HELD",
            "name": "Held",
            "entry_price": 8000,
            "current_price": 8000,
            "shares": 100,
            "market_value": 800000,
        }
    ]
    state["total_assets"] = 1_000_000

    allocation, reason = _available_buy_budget(1_000_000, 1_000_000, state, config_copy, pending_buy_amount=100_000)

    assert allocation == 100_000
    assert reason == "capital_utilization_policy"


def test_v2_27_same_day_equal_budget_splits_buy_capacity(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0, allocation_strategy="same_day_equal_budget")
    state = initial_live_paper_state(config_copy)
    candidates = [_candidate("10010", 3000), _candidate("10020", 3000), _candidate("10030", 3000)]

    _new_state, _summary, trades = execute_real_data_paper_trade(candidates, state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert len(buys) == 3
    assert [buy["shares"] for buy in buys] == [100, 100, 100]
    assert {buy["allocation_reason"] for buy in buys} == {"same_day_allocation_budget"}


def test_v2_28_round_lot_priority_wins_when_scores_are_close(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="round_lot_priority_near_score")
    config_copy["capital_utilization_policy"]["round_lot_priority_score_tolerance"] = 3
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 5000)
    expensive["total_score"] = 60
    cheap = _candidate("90020", 1000)
    cheap["total_score"] = 58

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive, cheap], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert buys
    assert buys[0]["code"] == "90020"
    assert buys[0]["allocation_strategy"] == "round_lot_priority_near_score"


def test_dynamic_exposure_overrides_target_exposure_for_strong_bull(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["dynamic_exposure"] = {
        "enabled": True,
        "target_exposure_by_regime": {"strong_bull": 0.95},
    }
    state = initial_live_paper_state(config_copy)
    state["positions"] = [
        {
            "code": "HELD",
            "name": "Held",
            "entry_price": 8000,
            "current_price": 8000,
            "shares": 100,
            "market_value": 800000,
        }
    ]
    state["total_assets"] = 1_000_000

    allocation, reason = _available_buy_budget(
        1_000_000,
        1_000_000,
        state,
        config_copy,
        market_context={"advance_ratio": 0.75, "market_average_change_rate": 0.01, "market_regime": "risk_on"},
    )

    assert allocation == 150_000
    assert reason == "capital_utilization_policy"


def test_dynamic_exposure_strong_bear_zero_prevents_new_buy(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=1.0, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["dynamic_exposure"] = {
        "enabled": True,
        "target_exposure_by_regime": {"strong_bear": 0.0},
    }
    state = initial_live_paper_state(config_copy)
    candidate = _candidate(price=1000)
    candidate["advance_ratio"] = 0.2
    candidate["market_average_change_rate"] = -0.01
    candidate["market_regime"] = "risk_off"
    candidate["dynamic_exposure_regime"] = "strong_bear"
    candidate["dynamic_exposure_source_date"] = "2026-01-05"
    candidate["dynamic_exposure_source_date_mode"] = "previous_trading_day"
    candidate["dynamic_exposure_source_lag_days"] = 1
    candidate["dynamic_exposure_same_day_context_used"] = False

    _new_state, _summary, trades = execute_real_data_paper_trade([candidate], state, config_copy, "2026-01-06")

    assert not [trade for trade in trades if trade.get("action") == "BUY"]
    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert len(skipped) == 1
    assert skipped[0]["skipped_reason"] == "target_exposure_limit"
    assert skipped[0]["dynamic_exposure_regime"] == "strong_bear"
    assert skipped[0]["dynamic_target_exposure"] == 0.0
    assert skipped[0]["dynamic_exposure_source_date"] == "2026-01-05"
    assert skipped[0]["dynamic_exposure_same_day_context_used"] is False


def test_affordable_fallback_buys_affordable_candidate_when_selected_is_unaffordable(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {"enabled": True}
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    expensive["total_score"] = 60
    fallback = _candidate("90020", 3000)
    fallback["selected"] = False
    fallback["total_score"] = 55

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive, fallback], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert len(buys) == 1
    assert buys[0]["code"] == "90020"
    assert buys[0]["affordable_fallback_buy_selected"] is True
    assert buys[0]["affordable_fallback_original_code"] == "90010"
    assert skipped[0]["code"] == "90010"
    assert skipped[0]["affordable_fallback_attempted"] is True
    assert skipped[0]["affordable_fallback_replaced_by_code"] == "90020"


def test_affordable_fallback_surplus_buys_unselected_high_rank_candidate(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.3, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {
        "enabled": True,
        "surplus_after_selection": True,
        "replace_unaffordable_selected": False,
        "min_total_score": 45,
        "max_rank_in_day": 20,
    }
    state = initial_live_paper_state(config_copy)
    regular = _candidate("90010", 3000)
    regular["total_score"] = 60
    regular["rank"] = 1
    fallback = _candidate("90020", 2000)
    fallback["selected"] = False
    fallback["total_score"] = 50
    fallback["rank"] = 2

    new_state, _summary, trades = execute_real_data_paper_trade([regular, fallback], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert [trade["code"] for trade in buys] == ["90010", "90020"]
    assert buys[1]["affordable_fallback_buy_selected"] is True
    assert buys[1]["affordable_fallback_reason"] == "surplus_available_cash"
    assert buys[1]["affordable_fallback_original_code"] == ""
    assert buys[1]["rank"] == 2
    assert new_state["cash"] < 1_000_000


def test_affordable_fallback_does_not_relabel_same_day_regular_selected_candidate(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {"enabled": True}
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    expensive["total_score"] = 60
    regular_selected = _candidate("90020", 3000)
    regular_selected["selected"] = True
    regular_selected["total_score"] = 55

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive, regular_selected], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert len(buys) == 1
    assert buys[0]["code"] == "90020"
    assert buys[0].get("affordable_fallback_buy_selected") is False
    assert skipped[0]["code"] == "90010"
    assert skipped[0]["affordable_fallback_attempted"] is True
    assert skipped[0]["affordable_fallback_no_candidate"] is True


def test_affordable_fallback_uses_non_selected_candidate_instead_of_regular_selected_candidate(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {"enabled": True}
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    expensive["total_score"] = 60
    regular_selected = _candidate("90020", 3000)
    regular_selected["selected"] = True
    regular_selected["total_score"] = 55
    fallback = _candidate("90030", 2500)
    fallback["selected"] = False
    fallback["total_score"] = 54

    _new_state, _summary, trades = execute_real_data_paper_trade(
        [expensive, regular_selected, fallback],
        state,
        config_copy,
        "2026-01-06",
    )

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert buys[0]["code"] == "90030"
    assert buys[0]["affordable_fallback_buy_selected"] is True
    assert buys[0]["affordable_fallback_original_code"] == "90010"


def test_affordable_fallback_trade_is_not_same_entry_code_as_baseline_regular_buy(config_copy: dict) -> None:
    base_config = deepcopy(config_copy)
    target_config = deepcopy(config_copy)
    _enable_policy(base_config, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    _enable_policy(target_config, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    target_config["affordable_fallback_buy"] = {"enabled": True}
    expensive = _candidate("90010", 6000)
    expensive["total_score"] = 60
    regular_selected = _candidate("90020", 3000)
    regular_selected["selected"] = True
    regular_selected["total_score"] = 55
    fallback = _candidate("90030", 2500)
    fallback["selected"] = False
    fallback["total_score"] = 54

    _base_state, _base_summary, base_trades = execute_real_data_paper_trade(
        [deepcopy(expensive), deepcopy(regular_selected), deepcopy(fallback)],
        initial_live_paper_state(base_config),
        base_config,
        "2026-01-06",
    )
    _target_state, _target_summary, target_trades = execute_real_data_paper_trade(
        [deepcopy(expensive), deepcopy(regular_selected), deepcopy(fallback)],
        initial_live_paper_state(target_config),
        target_config,
        "2026-01-06",
    )

    base_buy_keys = {
        (trade.get("entry_date"), trade.get("code"))
        for trade in base_trades
        if trade.get("action") == "BUY"
    }
    fallback_buys = [
        trade
        for trade in target_trades
        if trade.get("action") == "BUY" and trade.get("affordable_fallback_buy_selected")
    ]
    assert fallback_buys
    assert all((trade.get("entry_date"), trade.get("code")) not in base_buy_keys for trade in fallback_buys)


def test_affordable_fallback_min_total_score_50_rejects_score_49(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {"enabled": True, "min_total_score": 50}
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    fallback = _candidate("90020", 3000)
    fallback["selected"] = False
    fallback["total_score"] = 49

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive, fallback], state, config_copy, "2026-01-06")

    assert not [trade for trade in trades if trade.get("action") == "BUY"]
    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert skipped[0]["affordable_fallback_no_candidate"] is True
    assert skipped[0]["fallback_score_below_min_count"] == 1


def test_affordable_fallback_min_total_score_55_rejects_score_54(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {"enabled": True, "min_total_score": 55}
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    fallback = _candidate("90020", 3000)
    fallback["selected"] = False
    fallback["total_score"] = 54

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive, fallback], state, config_copy, "2026-01-06")

    assert not [trade for trade in trades if trade.get("action") == "BUY"]
    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert skipped[0]["affordable_fallback_no_candidate"] is True
    assert skipped[0]["fallback_score_below_min_count"] == 1


def test_affordable_fallback_max_rank_in_day_rejects_rank_21(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {"enabled": True, "max_rank_in_day": 20}
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    fallback = _candidate("90020", 3000)
    fallback["selected"] = False
    fallback["total_score"] = 55
    fallback["rank"] = 21

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive, fallback], state, config_copy, "2026-01-06")

    assert not [trade for trade in trades if trade.get("action") == "BUY"]
    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert skipped[0]["affordable_fallback_no_candidate"] is True
    assert skipped[0]["fallback_rank_out_of_range_count"] == 1


def test_affordable_fallback_min_risk_adjusted_score_filters_weak_candidate(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {"enabled": True, "min_risk_adjusted_score": 0.05}
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    weak = _candidate("90020", 3000)
    weak["selected"] = False
    weak["expected_return_10d"] = 0.06
    weak["bad_entry_probability_10d"] = 0.04
    strong = _candidate("90030", 2500)
    strong["selected"] = False
    strong["expected_return_10d"] = 0.08
    strong["bad_entry_probability_10d"] = 0.04

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive, weak, strong], state, config_copy, "2026-01-06")

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert buys[0]["code"] == "90030"
    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert skipped[0]["fallback_risk_adjusted_below_min_count"] == 1


def test_affordable_fallback_expected_return_and_bad_entry_filters_candidate(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {
        "enabled": True,
        "min_expected_return_10d": 0.02,
        "max_bad_entry_probability_10d": 0.70,
    }
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    high_bad_entry = _candidate("90020", 3000)
    high_bad_entry["selected"] = False
    high_bad_entry["expected_return_10d"] = 0.03
    high_bad_entry["bad_entry_probability_10d"] = 0.80
    low_expected = _candidate("90030", 2500)
    low_expected["selected"] = False
    low_expected["expected_return_10d"] = 0.01
    low_expected["bad_entry_probability_10d"] = 0.50
    good = _candidate("90040", 2000)
    good["selected"] = False
    good["expected_return_10d"] = 0.03
    good["bad_entry_probability_10d"] = 0.60

    _new_state, _summary, trades = execute_real_data_paper_trade(
        [expensive, high_bad_entry, low_expected, good],
        state,
        config_copy,
        "2026-01-06",
    )

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert buys[0]["code"] == "90040"
    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert skipped[0]["fallback_expected_return_below_min_count"] == 1
    assert skipped[0]["fallback_bad_entry_above_max_count"] == 1


def test_affordable_fallback_max_buys_per_day_limits_surplus_fallback(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.2, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {
        "enabled": True,
        "surplus_after_selection": True,
        "replace_unaffordable_selected": False,
        "min_total_score": 45,
        "max_fallback_buys_per_day": 1,
    }
    state = initial_live_paper_state(config_copy)
    regular = _candidate("90010", 2000)
    regular["total_score"] = 60
    fallback1 = _candidate("90020", 1500)
    fallback1["selected"] = False
    fallback1["total_score"] = 55
    fallback1["rank"] = 2
    fallback2 = _candidate("90030", 1000)
    fallback2["selected"] = False
    fallback2["total_score"] = 54
    fallback2["rank"] = 3

    _new_state, _summary, trades = execute_real_data_paper_trade(
        [regular, fallback1, fallback2],
        state,
        config_copy,
        "2026-01-06",
    )

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    fallback_buys = [trade for trade in buys if trade.get("affordable_fallback_buy_selected")]
    assert len(fallback_buys) == 1
    assert fallback_buys[0]["code"] == "90020"


def test_affordable_fallback_skips_when_no_affordable_candidate(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {"enabled": True}
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    expensive["total_score"] = 60

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive], state, config_copy, "2026-01-06")

    assert not [trade for trade in trades if trade.get("action") == "BUY"]
    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert skipped[0]["code"] == "90010"
    assert skipped[0]["affordable_fallback_no_candidate"] is True


def test_affordable_fallback_ignores_market_filter_excluded_candidate(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {"enabled": True}
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    fallback = _candidate("90020", 3000)
    fallback["selected"] = False
    fallback["market_section"] = "Unknown"
    fallback["section"] = "Unknown"
    fallback["listing_market"] = "Unknown"

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive, fallback], state, config_copy, "2026-01-06")

    assert not [trade for trade in trades if trade.get("action") == "BUY"]
    assert [trade for trade in trades if trade.get("action") == "SKIP_BUY"][0]["affordable_fallback_no_candidate"] is True


def test_affordable_fallback_ignores_held_candidate(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {"enabled": True}
    state = initial_live_paper_state(config_copy)
    state["positions"] = [
        {
            "code": "90020",
            "name": "Held",
            "entry_price": 3000,
            "current_price": 3000,
            "shares": 100,
            "market_value": 300000,
            "holding_days": 1,
        }
    ]
    expensive = _candidate("90010", 6000)
    fallback = _candidate("90020", 3000)
    fallback["selected"] = False

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive, fallback], state, config_copy, "2026-01-06")

    assert not [trade for trade in trades if trade.get("action") == "BUY"]
    assert [trade for trade in trades if trade.get("action") == "SKIP_BUY"][0]["affordable_fallback_no_candidate"] is True


def test_affordable_fallback_prioritizes_score_then_cheaper_round_lot(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    config_copy["affordable_fallback_buy"] = {"enabled": True}
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    high_score = _candidate("90020", 4000)
    high_score["selected"] = False
    high_score["total_score"] = 56
    cheap_same_score = _candidate("90030", 2000)
    cheap_same_score["selected"] = False
    cheap_same_score["total_score"] = 56
    lower_score = _candidate("90040", 1000)
    lower_score["selected"] = False
    lower_score["total_score"] = 55

    _new_state, _summary, trades = execute_real_data_paper_trade(
        [expensive, high_score, cheap_same_score, lower_score],
        state,
        config_copy,
        "2026-01-06",
    )

    buys = [trade for trade in trades if trade.get("action") == "BUY"]
    assert buys[0]["code"] == "90030"


def test_affordable_fallback_disabled_keeps_v2_26_behavior(config_copy: dict) -> None:
    _enable_policy(config_copy, max_position_value_rate=0.5, allocation_strategy="relaxed_pending_target_exposure")
    state = initial_live_paper_state(config_copy)
    expensive = _candidate("90010", 6000)
    fallback = _candidate("90020", 3000)
    fallback["selected"] = False

    _new_state, _summary, trades = execute_real_data_paper_trade([expensive, fallback], state, config_copy, "2026-01-06")

    assert not [trade for trade in trades if trade.get("action") == "BUY"]
    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert skipped[0].get("affordable_fallback_attempted") is False
