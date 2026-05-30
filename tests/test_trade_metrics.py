from __future__ import annotations

from trade_metrics import profit_factor_metrics


def test_profit_factor_metrics_use_closed_sell_trades_only() -> None:
    rows = [
        {"action": "SELL", "result": "WIN", "order_status": "FILLED", "gross_profit": 10000},
        {"action": "SELL", "result": "LOSS", "order_status": "FILLED", "gross_profit": -4000},
        {"action": "BUY", "result": "", "order_status": "FILLED", "gross_profit": 0},
        {"action": "SELL", "result": "WIN", "order_status": "PENDING", "gross_profit": 999999},
        {"action": "SKIP_BUY", "result": "", "order_status": "REJECTED", "gross_profit": 0},
        {"action": "NO_BUY", "result": "", "order_status": "FILLED", "gross_profit": 0},
    ]

    metrics = profit_factor_metrics(rows)

    assert metrics["total_trades"] == 2
    assert metrics["closed_trade_count"] == 2
    assert metrics["win_count"] == 1
    assert metrics["loss_count"] == 1
    assert metrics["win_rate"] == 0.5
    assert metrics["profit_factor"] == 2.5
    assert metrics["excluded_order_event_count"] == 3
