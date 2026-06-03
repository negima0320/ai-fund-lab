from __future__ import annotations

from copy import deepcopy

from paper_trade import execute_real_data_paper_trade


def _state_with_position(*, score: float = 55.0, holding_days: int = 1) -> dict:
    return {
        "cash": 900000.0,
        "positions": [
            {
                "code": "1001",
                "name": "Hold Test",
                "entry_date": "2026-03-01",
                "entry_price": 1000.0,
                "current_price": 1000.0,
                "shares": 100,
                "market_value": 100000.0,
                "buy_commission": 0.0,
                "holding_days": holding_days,
                "score": score,
                "entry_score": score,
                "reason": "test",
            }
        ],
        "closed_trades": [],
        "pending_orders": [],
        "total_assets": 1000000.0,
        "cumulative_profit": 0.0,
        "asset_history": [1000000.0],
    }


def _candidate(*, selected: bool, score: float = 55.0, close: float = 1000.0, low: float | None = None) -> dict:
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


def test_reselected_holding_extends_max_holding_exit(config_copy: dict) -> None:
    config = _enable_holding_revaluation(config_copy)
    state = _state_with_position(holding_days=4)

    new_state, _summary, trades = execute_real_data_paper_trade([_candidate(selected=True, score=58.0)], state, config, "2026-03-02")

    assert [trade for trade in trades if trade.get("action") == "SELL"] == []
    assert new_state["positions"][0]["holding_signal_status"] == "reselected"
    assert new_state["positions"][0]["holding_effective_max_days"] == 10
    assert new_state["positions"][0]["holding_extended"] is True


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
