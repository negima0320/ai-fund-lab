from __future__ import annotations

from paper_trade import execute_real_data_paper_trade


def _state_with_position() -> dict:
    return {
        "cash": 900000.0,
        "positions": [
            {
                "code": "1001",
                "name": "Stop Test",
                "entry_date": "2026-03-01",
                "entry_price": 1000.0,
                "current_price": 1000.0,
                "shares": 100,
                "market_value": 100000.0,
                "buy_commission": 0.0,
                "holding_days": 1,
                "reason": "test",
            }
        ],
        "closed_trades": [],
        "pending_orders": [],
        "total_assets": 1000000.0,
        "cumulative_profit": 0.0,
        "asset_history": [1000000.0],
    }


def test_intraday_stop_records_loss_near_stop_loss_rate(config_copy: dict) -> None:
    config_copy["execution"]["use_next_day_open_execution"] = True
    config_copy["execution"]["stop_loss_execution"] = "intraday_stop"
    scored_candidates = [
        {"code": "1001", "open": 995.0, "high": 1000.0, "low": 960.0, "close": 950.0, "selected": False},
    ]

    state, _summary, trades = execute_real_data_paper_trade(scored_candidates, _state_with_position(), config_copy, "2026-03-02")

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["order_status"] == "FILLED"
    assert sell["exit_reason"] == "損切り"
    assert sell["exit_price"] == 970.0
    assert sell["gross_profit_rate"] == -0.03
    assert sell["stop_loss_trigger_price"] == 970.0
    assert sell["stop_loss_triggered_date"] == "2026-03-02"
    assert sell["stop_loss_slippage_rate"] == 0.0
    assert state["positions"] == []


def test_next_day_open_can_exceed_stop_loss_rate_on_gap_down(config_copy: dict) -> None:
    config_copy["execution"]["use_next_day_open_execution"] = True
    config_copy["execution"]["stop_loss_execution"] = "next_day_open"
    day1_candidates = [
        {"code": "1001", "open": 995.0, "high": 1000.0, "low": 950.0, "close": 960.0, "selected": False},
    ]
    state, _summary, trades = execute_real_data_paper_trade(day1_candidates, _state_with_position(), config_copy, "2026-03-02")
    pending_sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert pending_sell["order_status"] == "PENDING"

    day2_candidates = [
        {"code": "1001", "open": 900.0, "high": 920.0, "low": 890.0, "close": 910.0, "selected": False},
    ]
    state, _summary, trades = execute_real_data_paper_trade(day2_candidates, state, config_copy, "2026-03-03")

    sell = next(trade for trade in trades if trade.get("action") == "SELL")
    assert sell["order_status"] == "FILLED"
    assert sell["exit_reason"] == "損切り"
    assert sell["exit_price"] == 900.0
    assert sell["gross_profit_rate"] == -0.1
    assert sell["gap_slippage_rate"] < 0
    assert sell["stop_loss_slippage_rate"] < 0
