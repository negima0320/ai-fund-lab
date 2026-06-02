from __future__ import annotations

from copy import deepcopy

from paper_trade import _available_buy_budget, execute_real_data_paper_trade, initial_live_paper_state


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
