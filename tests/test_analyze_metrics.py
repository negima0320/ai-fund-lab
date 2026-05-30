from __future__ import annotations

from db import _portfolio_analysis, _trade_analysis


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
    assert analysis["closed_trade_count"] == 2
    assert analysis["excluded_order_event_count"] == 1
    assert analysis["total_trades"] == 2
    assert analysis["gross_profit_total"] == 10000
    assert analysis["realized_profit_total"] == 6000
    assert analysis["gross_loss_total"] == -4000
    assert analysis["profit_factor"] == 2.5
    assert analysis["average_win_profit_rate"] == 0.1
    assert analysis["average_loss_profit_rate"] == -0.04
    assert analysis["average_holding_days"] is None
    assert analysis["largest_win"] == 10000
    assert analysis["largest_loss"] == -4000
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


def test_portfolio_analysis_reconciles_realized_and_unrealized_profit(config_copy: dict) -> None:
    rows = [
        {
            "date": "2026-03-05",
            "cash": 500000,
            "positions_value": 533337,
            "total_assets": 1033337,
            "max_drawdown": 0,
            "gross_cumulative_profit": -3976,
            "net_cumulative_profit": -3976,
            "total_commission": 0,
            "estimated_tax_total": 0,
            "open_positions_count": 3,
            "closed_trades_count": 8,
        }
    ]

    analysis = _portfolio_analysis(config_copy, rows)

    assert analysis["initial_capital"] == 1000000
    assert analysis["latest_total_assets"] == 1033337
    assert analysis["realized_profit"] == -3976
    assert analysis["unrealized_profit"] == 37313
    assert analysis["reconciled_assets"] == 1033337
    assert analysis["reconciliation_difference"] == 0
    assert analysis["reconciliation_ok"] is True
