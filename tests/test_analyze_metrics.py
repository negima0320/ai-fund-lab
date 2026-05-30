from __future__ import annotations

from db import _trade_analysis


def test_trade_analysis_includes_extended_metrics(config_copy: dict) -> None:
    rows = [
        {
            "action": "SELL",
            "trade_id": "win",
            "code": "1001",
            "name": "Winner",
            "profit": 10000,
            "profit_rate": 0.1,
            "gross_profit": 10000,
            "total_commission": 100,
            "result": "WIN",
            "exit_reason": "利確",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "loss",
            "code": "1002",
            "name": "Loser",
            "profit": -4000,
            "profit_rate": -0.04,
            "gross_profit": -4000,
            "total_commission": 100,
            "result": "LOSS",
            "exit_reason": "損切り",
            "stop_loss_slippage_rate": -0.01,
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "pending",
            "code": "1003",
            "name": "Pending",
            "profit": 999999,
            "profit_rate": 9.99,
            "gross_profit": 999999,
            "result": "WIN",
            "exit_reason": "利確",
            "order_status": "PENDING",
        },
    ]

    analysis = _trade_analysis(config_copy, rows)

    assert analysis["win_count"] == 1
    assert analysis["loss_count"] == 1
    assert analysis["total_trades"] == 2
    assert analysis["average_win_profit_rate"] == 0.1
    assert analysis["average_loss_profit_rate"] == -0.04
    assert analysis["profit_ratio"] == 2.5
    assert analysis["expectancy"] == 0.03
    assert analysis["worst_loss_profit_rate"] == -0.04
    assert analysis["stop_loss_slippage_average"] == -0.01
    assert analysis["stop_loss_slippage_max"] == -0.01
    assert analysis["loss_over_stop_count"] == 1
    assert analysis["loss_over_stop_rate"] == 1.0
    assert analysis["best_trade"]["code"] == "1001"
    assert analysis["worst_trade"]["code"] == "1002"
    assert analysis["exit_reason_analysis"] == [
        {"exit_reason": "利確", "count": 1, "average_profit_rate": 0.1},
        {"exit_reason": "損切り", "count": 1, "average_profit_rate": -0.04},
    ]
