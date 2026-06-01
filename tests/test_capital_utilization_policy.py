from __future__ import annotations

from paper_trade import execute_real_data_paper_trade, initial_live_paper_state


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
