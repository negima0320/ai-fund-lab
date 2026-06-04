from __future__ import annotations

from copy import deepcopy

from paper_trade import execute_real_data_paper_trade


def _state_with_position(*, score: float = 55.0, holding_days: int = 1, current_price: float = 1000.0, extra_position: dict | None = None) -> dict:
    position = {
        "code": "1001",
        "name": "Hold Test",
        "entry_date": "2026-03-01",
        "entry_price": 1000.0,
        "current_price": current_price,
        "shares": 100,
        "market_value": current_price * 100,
        "buy_commission": 0.0,
        "holding_days": holding_days,
        "score": score,
        "entry_score": score,
        "reason": "test",
    }
    if extra_position:
        position.update(extra_position)
    return {
        "cash": 900000.0,
        "positions": [position],
        "closed_trades": [],
        "pending_orders": [],
        "total_assets": 1000000.0,
        "cumulative_profit": 0.0,
        "asset_history": [1000000.0],
    }


def _candidate(*, selected: bool, score: float = 55.0, close: float = 1000.0, low: float | None = None, rank: int = 1) -> dict:
    return {
        "code": "1001",
        "name": "Hold Test",
        "date": "2026-03-02",
        "entry_date": "2026-03-02",
        "entry_price": close,
        "open": close,
        "high": close,
        "low": close if low is None else low,
        "close": close,
        "selected": selected,
        "total_score": score,
        "technical_score": score,
        "confidence": 0.9,
        "rank": rank,
        "daily_score_rank": rank,
        "reason": "test",
        "selected_reason": "test",
        "selection_reason": "test",
        "market_section": "TSEPrime",
        "section": "TSEPrime",
        "listing_market": "TSEPrime",
    }


def _enable_holding_revaluation(config: dict) -> dict:
    config = deepcopy(config)
    config.setdefault("execution", {})["use_next_day_open_execution"] = False
    config.setdefault("execution", {})["stop_loss_execution"] = "intraday_stop"
    config["risk"]["take_profit_pct"] = 0.06
    config["risk"]["stop_loss_pct"] = -0.03
    config["risk"]["max_holding_business_days"] = 5
    config["selection"]["fallback_min_score"] = 40
    config["selection"]["min_confidence"] = 0.7
    config["holding_revaluation"] = {
        "enabled": True,
        "hold_reselection_enabled": True,
        "hold_extension_max_days": 10,
        "early_exit_on_signal_lost": True,
        "score_drop_exit_threshold": 10,
        "stop_loss_always_priority": True,
    }
    return config


def _enable_hold_extension_only(config: dict) -> dict:
    config = _enable_holding_revaluation(config)
    config["holding_revaluation"] = {
        "enabled": True,
        "hold_reselection_enabled": True,
        "hold_extension_max_days": 10,
        "early_exit_on_signal_lost": False,
        "stop_loss_always_priority": True,
    }
    return config


def _enable_delayed_signal_lost_exit(config: dict, *, suppress_profit: bool = False, extension: bool = False) -> dict:
    config = _enable_hold_extension_only(config)
    config["holding_revaluation"].update(
        {
            "hold_reselection_enabled": extension,
            "early_exit_on_signal_lost": True,
            "signal_lost_exit_min_consecutive_days": 2,
            "signal_lost_exit_unrealized_loss_rate_threshold": -0.02,
            "suppress_signal_lost_exit_when_unrealized_profit": suppress_profit,
        }
    )
    if extension:
        config["holding_revaluation"].update(
            {
                "hold_extension_max_days": 10,
                "hold_extension_max_count": 1,
                "hold_extension_require_confirmation": True,
                "hold_extension_min_unrealized_profit_rate": 0.0,
                "hold_extension_max_score_drop": 3,
                "hold_extension_max_rank": 10,
            }
        )
    return config


def _enable_conditional_hold_extension(config: dict) -> dict:
    config = deepcopy(config)
    config.setdefault("execution", {})["use_next_day_open_execution"] = False
    config.setdefault("execution", {})["stop_loss_execution"] = "intraday_stop"
    config["risk"]["take_profit_pct"] = 0.06
    config["risk"]["stop_loss_pct"] = -0.03
    config["risk"]["max_holding_business_days"] = 5
    config["selection"]["fallback_min_score"] = 40
    config["selection"]["min_confidence"] = 0.7
    config.pop("holding_revaluation", None)
    config["conditional_hold_extension"] = {
        "enabled": True,
        "min_unrealized_profit_rate": 0.03,
        "max_holding_days": 7,
        "max_extension_count": 1,
    }
    return config


def _enable_extension_exit_guard(config: dict) -> dict:
    config = _enable_conditional_hold_extension(config)
    config["conditional_hold_extension"]["extension_exit_guard"] = {
        "enabled": True,
        "max_profit_pullback_points": 0.02,
        "min_remaining_profit_rate": 0.01,
        "exit_reason": "延長後失速撤退",
    }
    return config


def _enable_trend_conditional_hold_extension(config: dict) -> dict:
    config = _enable_conditional_hold_extension(config)
    config["conditional_hold_extension"] = {
        "enabled": True,
        "require_trend_continuation": True,
        "min_unrealized_profit_rate": 0.015,
        "min_relative_strength_score": 60,
        "max_holding_days": 7,
        "max_extension_count": 1,
    }
    return config


def _enable_ma25_uptrend_conditional_hold_extension(config: dict) -> dict:
    config = _enable_conditional_hold_extension(config)
    config["conditional_hold_extension"] = {
        "enabled": True,
        "require_trend_continuation": True,
        "skip_ma5_condition": True,
        "require_ma25_uptrend": True,
        "min_unrealized_profit_rate": 0.03,
        "min_relative_strength_score": 60,
        "profit_reject_reason": "profit_below_threshold",
        "relative_strength_reject_reason": "relative_strength_below_threshold",
        "max_holding_days": 7,
        "max_extension_count": 1,
    }
    return config


def _trend_candidate(**overrides: object) -> dict:
    row = _candidate(selected=False, close=1020.0)
    row.update(
        {
            "ma5": 1010.0,
            "ma25": 1005.0,
            "previous_ma25": 1000.0,
            "relative_strength_score": 65.0,
        }
    )
    row.update(overrides)
    return row


def test_reselected_holding_extends_max_holding_exit(config_copy: dict) -> None:
    config = _enable_holding_revaluation(config_copy)
    state = _state_with_position(holding_days=4)

    new_state, _summary, trades = execute_real_data_paper_trade([_candidate(selected=True, score=58.0)], state, config, "2026-03-02")

    assert [trade for trade in trades if trade.get("action") == "SELL"] == []
    assert new_state["positions"][0]["holding_signal_status"] == "reselected"
    assert new_state["positions"][0]["holding_effective_max_days"] == 10
    assert new_state["positions"][0]["holding_extended"] is True


def test_hold_extension_only_does_not_exit_on_signal_lost(config_copy: dict) -> None:
    config = _enable_hold_extension_only(config_copy)
    state = _state_with_position(holding_days=1)

    _new_state, _summary, trades = execute_real_data_paper_trade([], state, config, "2026-03-02")

    assert [trade for trade in trades if trade.get("action") == "SELL"] == []


def test_hold_extension_only_does_not_exit_on_score_drop(config_copy: dict) -> None:
    config = _enable_hold_extension_only(config_copy)
    state = _state_with_position(score=60.0, holding_days=1)

    _new_state, _summary, trades = execute_real_data_paper_trade([_candidate(selected=False, score=40.0)], state, config, "2026-03-02")

    assert [trade for trade in trades if trade.get("action") == "SELL"] == []


def test_delayed_signal_lost_exits_only_after_two_consecutive_days(config_copy: dict) -> None:
    config = _enable_delayed_signal_lost_exit(config_copy)
    state = _state_with_position(holding_days=1)

    new_state, _summary, trades = execute_real_data_paper_trade([], state, config, "2026-03-02")

    assert [trade for trade in trades if trade.get("action") == "SELL"] == []
    assert new_state["positions"][0]["holding_signal_lost_streak"] == 1
    assert new_state["positions"][0]["holding_signal_lost_exit_avoided"] is True

    _new_state, _summary, trades = execute_real_data_paper_trade([], new_state, config, "2026-03-03")

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "シグナル消失"
    assert sell["holding_signal_lost_streak"] == 2


def test_delayed_signal_lost_can_exit_on_first_day_when_loss_is_large(config_copy: dict) -> None:
    config = _enable_delayed_signal_lost_exit(config_copy)
    state = _state_with_position(holding_days=1, current_price=975.0)

    _new_state, _summary, trades = execute_real_data_paper_trade([], state, config, "2026-03-02")

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "シグナル消失"
    assert sell["holding_unrealized_profit_rate"] == -0.025


def test_signal_lost_exit_is_suppressed_while_profitable(config_copy: dict) -> None:
    config = _enable_delayed_signal_lost_exit(config_copy, suppress_profit=True)
    state = _state_with_position(holding_days=2, current_price=1050.0, extra_position={"holding_signal_lost_streak": 1})

    new_state, _summary, trades = execute_real_data_paper_trade([], state, config, "2026-03-03")

    assert [trade for trade in trades if trade.get("action") == "SELL"] == []
    assert new_state["positions"][0]["holding_signal_lost_streak"] == 2
    assert new_state["positions"][0]["holding_signal_lost_exit_avoided"] is True


def test_conditional_reselection_extends_only_when_quality_condition_matches(config_copy: dict) -> None:
    config = _enable_delayed_signal_lost_exit(config_copy, suppress_profit=True, extension=True)
    state = _state_with_position(score=55.0, holding_days=4)

    new_state, _summary, trades = execute_real_data_paper_trade([_candidate(selected=True, score=54.0, close=1010.0, rank=5)], state, config, "2026-03-02")

    assert [trade for trade in trades if trade.get("action") == "SELL"] == []
    assert new_state["positions"][0]["holding_signal_status"] == "reselected"
    assert new_state["positions"][0]["holding_extension_eligible"] is True
    assert new_state["positions"][0]["holding_effective_max_days"] == 10
    assert new_state["positions"][0]["holding_extended"] is True


def test_conditional_reselection_does_not_extend_when_quality_condition_fails(config_copy: dict) -> None:
    config = _enable_delayed_signal_lost_exit(config_copy, suppress_profit=True, extension=True)
    state = _state_with_position(score=55.0, holding_days=4)

    _new_state, _summary, trades = execute_real_data_paper_trade([_candidate(selected=True, score=45.0, close=990.0, rank=30)], state, config, "2026-03-02")

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "最大保有期間到達"
    assert sell["holding_extension_eligible"] is False


def test_signal_lost_can_trigger_early_exit(config_copy: dict) -> None:
    config = _enable_holding_revaluation(config_copy)
    state = _state_with_position(holding_days=1)

    _new_state, _summary, trades = execute_real_data_paper_trade([], state, config, "2026-03-02")

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "シグナル消失"
    assert sell["holding_signal_status"] == "signal_lost"


def test_score_drop_can_trigger_early_exit(config_copy: dict) -> None:
    config = _enable_holding_revaluation(config_copy)
    state = _state_with_position(score=60.0, holding_days=1)

    _new_state, _summary, trades = execute_real_data_paper_trade([_candidate(selected=False, score=45.0)], state, config, "2026-03-02")

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "スコア低下"
    assert sell["holding_signal_status"] == "score_deteriorated"
    assert sell["holding_score_drop"] == 15.0


def test_conditional_hold_extension_extends_only_profitable_max_holding_position(config_copy: dict) -> None:
    config = _enable_conditional_hold_extension(config_copy)
    state = _state_with_position(holding_days=4)

    new_state, _summary, trades = execute_real_data_paper_trade([_candidate(selected=False, close=1030.0)], state, config, "2026-03-02")

    assert [trade for trade in trades if trade.get("action") == "SELL"] == []
    position = new_state["positions"][0]
    assert position["holding_effective_max_days"] == 7
    assert position["conditional_hold_extension_applied"] is True
    assert position["conditional_hold_extension_count"] == 1
    assert position["conditional_hold_extension_trigger_profit_rate"] == 0.03


def test_conditional_hold_extension_does_not_extend_when_profit_is_below_threshold(config_copy: dict) -> None:
    config = _enable_conditional_hold_extension(config_copy)
    state = _state_with_position(holding_days=4)

    _new_state, _summary, trades = execute_real_data_paper_trade([_candidate(selected=False, close=1029.0)], state, config, "2026-03-02")

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "最大保有期間到達"
    assert sell["conditional_hold_extension_applied"] is False


def test_conditional_hold_extension_keeps_take_profit_priority(config_copy: dict) -> None:
    config = _enable_conditional_hold_extension(config_copy)
    state = _state_with_position(holding_days=4)

    _new_state, _summary, trades = execute_real_data_paper_trade([_candidate(selected=False, close=1060.0)], state, config, "2026-03-02")

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "利確"
    assert sell["conditional_hold_extension_applied"] is True


def test_extension_exit_guard_sells_after_profit_pullback(config_copy: dict) -> None:
    config = _enable_extension_exit_guard(config_copy)
    state = _state_with_position(
        holding_days=5,
        extra_position={
            "conditional_hold_extension_applied": True,
            "conditional_hold_extension_count": 1,
            "conditional_hold_extension_trigger_profit_rate": 0.05,
            "extension_profit_rate": 0.05,
        },
    )

    _new_state, _summary, trades = execute_real_data_paper_trade(
        [_candidate(selected=False, close=1029.0)],
        state,
        config,
        "2026-03-02",
    )

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "延長後失速撤退"
    assert sell["extension_profit_rate"] == 0.05
    assert sell["extension_exit_guard_triggered"] is True
    assert sell["extension_exit_guard_reason"] == "profit_pullback_exceeded"


def test_extension_exit_guard_sells_when_remaining_profit_is_too_low(config_copy: dict) -> None:
    config = _enable_extension_exit_guard(config_copy)
    state = _state_with_position(
        holding_days=5,
        extra_position={
            "conditional_hold_extension_applied": True,
            "conditional_hold_extension_count": 1,
            "conditional_hold_extension_trigger_profit_rate": 0.03,
            "extension_profit_rate": 0.03,
        },
    )

    _new_state, _summary, trades = execute_real_data_paper_trade(
        [_candidate(selected=False, close=1005.0)],
        state,
        config,
        "2026-03-02",
    )

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "延長後失速撤退"
    assert sell["extension_exit_guard_triggered"] is True
    assert sell["extension_exit_guard_reason"] == "profit_pullback_exceeded+remaining_profit_below_min"


def test_trend_conditional_hold_extension_requires_momentum_conditions(config_copy: dict) -> None:
    config = _enable_trend_conditional_hold_extension(config_copy)
    state = _state_with_position(holding_days=4)

    new_state, _summary, trades = execute_real_data_paper_trade([_trend_candidate()], state, config, "2026-03-02")

    assert [trade for trade in trades if trade.get("action") == "SELL"] == []
    position = new_state["positions"][0]
    assert position["holding_effective_max_days"] == 7
    assert position["conditional_hold_extension_applied"] is True
    assert position["conditional_hold_extension_reason"] == "trend_continuation_profit>=0.0150_rs>=60.0"


def test_trend_conditional_hold_extension_rejects_below_ma5(config_copy: dict) -> None:
    config = _enable_trend_conditional_hold_extension(config_copy)
    state = _state_with_position(holding_days=4)

    _new_state, _summary, trades = execute_real_data_paper_trade([_trend_candidate(ma5=1030.0)], state, config, "2026-03-02")

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "最大保有期間到達"
    assert sell["conditional_hold_extension_rejected"] is True
    assert "below_ma5" in sell["conditional_hold_extension_rejected_reason"]


def test_trend_conditional_hold_extension_rejects_low_relative_strength(config_copy: dict) -> None:
    config = _enable_trend_conditional_hold_extension(config_copy)
    state = _state_with_position(holding_days=4)

    _new_state, _summary, trades = execute_real_data_paper_trade([_trend_candidate(relative_strength_score=59.0)], state, config, "2026-03-02")

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "最大保有期間到達"
    assert sell["conditional_hold_extension_rejected"] is True
    assert "low_relative_strength" in sell["conditional_hold_extension_rejected_reason"]


def test_ma25_uptrend_conditional_hold_extension_extends_when_trend_continues(config_copy: dict) -> None:
    config = _enable_ma25_uptrend_conditional_hold_extension(config_copy)
    state = _state_with_position(holding_days=4)

    new_state, _summary, trades = execute_real_data_paper_trade(
        [_trend_candidate(close=1035.0, ma5=1045.0, ma25=1010.0, previous_ma25=1000.0)],
        state,
        config,
        "2026-03-02",
    )

    assert [trade for trade in trades if trade.get("action") == "SELL"] == []
    position = new_state["positions"][0]
    assert position["holding_effective_max_days"] == 7
    assert position["conditional_hold_extension_applied"] is True
    assert position["conditional_hold_extension_reason"] == "trend_continuation_profit>=0.0300_rs>=60.0_ma25_uptrend"


def test_ma25_uptrend_conditional_hold_extension_rejects_flat_ma25(config_copy: dict) -> None:
    config = _enable_ma25_uptrend_conditional_hold_extension(config_copy)
    state = _state_with_position(holding_days=4)

    _new_state, _summary, trades = execute_real_data_paper_trade(
        [_trend_candidate(close=1035.0, ma25=1000.0, previous_ma25=1000.0)],
        state,
        config,
        "2026-03-02",
    )

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "最大保有期間到達"
    assert sell["conditional_hold_extension_rejected"] is True
    assert "ma25_not_uptrend" in sell["conditional_hold_extension_rejected_reason"]


def test_ma25_uptrend_conditional_hold_extension_uses_profile_profit_reject_reason(config_copy: dict) -> None:
    config = _enable_ma25_uptrend_conditional_hold_extension(config_copy)
    state = _state_with_position(holding_days=4)

    _new_state, _summary, trades = execute_real_data_paper_trade(
        [_trend_candidate(close=1029.0, ma25=1010.0, previous_ma25=1000.0)],
        state,
        config,
        "2026-03-02",
    )

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "最大保有期間到達"
    assert sell["conditional_hold_extension_rejected"] is True
    assert "profit_below_threshold" in sell["conditional_hold_extension_rejected_reason"]


def test_ma25_uptrend_conditional_hold_extension_uses_profile_relative_strength_reject_reason(config_copy: dict) -> None:
    config = _enable_ma25_uptrend_conditional_hold_extension(config_copy)
    state = _state_with_position(holding_days=4)

    _new_state, _summary, trades = execute_real_data_paper_trade(
        [_trend_candidate(close=1035.0, ma25=1010.0, previous_ma25=1000.0, relative_strength_score=59.0)],
        state,
        config,
        "2026-03-02",
    )

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "最大保有期間到達"
    assert sell["conditional_hold_extension_rejected"] is True
    assert "relative_strength_below_threshold" in sell["conditional_hold_extension_rejected_reason"]


def test_stop_loss_keeps_priority_over_reselected_extension(config_copy: dict) -> None:
    config = _enable_holding_revaluation(config_copy)
    state = _state_with_position(holding_days=1)

    _new_state, _summary, trades = execute_real_data_paper_trade(
        [_candidate(selected=True, score=58.0, close=950.0, low=960.0)],
        state,
        config,
        "2026-03-02",
    )

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["exit_reason"] == "損切り"
    assert sell["holding_signal_status"] == "reselected"
