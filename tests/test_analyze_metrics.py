from __future__ import annotations

from db import _monthly_performance_analysis, _portfolio_analysis, _trade_analysis, _yearly_performance_analysis
from main import render_analysis_markdown


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
        {
            "action": "BUY",
            "trade_id": "buy",
            "code": "1004",
            "name": "Open",
            "profit": 0,
            "profit_rate": 0,
            "gross_profit": 0,
            "result": "",
            "order_status": "FILLED",
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
    assert analysis["sold_before_take_profit_rate"] == 0.5
    assert analysis["exit_reason_analysis"] == [
        {"exit_reason": "利確", "count": 1, "win_rate": 1.0, "average_profit_rate": 0.1, "total_profit": 10000.0},
        {"exit_reason": "損切り", "count": 1, "win_rate": 0.0, "average_profit_rate": -0.04, "total_profit": -4000.0},
        {"exit_reason": "最大保有期間到達", "count": 0, "win_rate": None, "average_profit_rate": None, "total_profit": 0.0},
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


def test_yearly_and_monthly_performance_analysis() -> None:
    portfolio_rows = [
        {"date": "2025-12-31", "max_drawdown": -0.05},
        {"date": "2026-01-31", "max_drawdown": -0.02},
        {"date": "2026-02-28", "max_drawdown": -0.08},
    ]
    trade_rows = [
        {
            "action": "SELL",
            "entry_date": "2025-12-20",
            "exit_date": "2025-12-25",
            "profit": 10000,
            "gross_profit": 10000,
            "result": "WIN",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "entry_date": "2026-01-10",
            "exit_date": "2026-01-15",
            "profit": -4000,
            "gross_profit": -4000,
            "result": "LOSS",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "entry_date": "2026-02-10",
            "exit_date": "2026-02-15",
            "profit": 8000,
            "gross_profit": 8000,
            "result": "WIN",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "entry_date": "2026-02-10",
            "exit_date": "2026-02-15",
            "profit": 999999,
            "gross_profit": 999999,
            "result": "WIN",
            "order_status": "PENDING",
        },
    ]

    yearly = _yearly_performance_analysis(portfolio_rows, trade_rows)
    monthly = _monthly_performance_analysis(trade_rows)

    assert yearly == [
        {"year": "2025", "profit": 10000.0, "win_rate": 1.0, "profit_factor": None, "max_drawdown": -0.05, "trades": 1},
        {"year": "2026", "profit": 4000.0, "win_rate": 0.5, "profit_factor": 2.0, "max_drawdown": -0.08, "trades": 2},
    ]
    assert monthly == [
        {"month": "2025-12", "profit": 10000.0, "trades": 1, "win_rate": 1.0},
        {"month": "2026-01", "profit": -4000.0, "trades": 1, "win_rate": 0.0},
        {"month": "2026-02", "profit": 8000.0, "trades": 1, "win_rate": 1.0},
    ]


def test_analysis_markdown_includes_yearly_and_monthly_sections() -> None:
    analysis = {
        "current_profile_id": "rookie_dealer_01",
        "current_profile_name": "新人ディーラー1号",
        "current_config_version": "test",
        "generated_at": "2026-03-06T00:00:00",
        "portfolio_analysis": {
            "initial_capital": 1000000,
            "latest_total_assets": 1010000,
            "cumulative_profit": 10000,
            "cumulative_profit_rate": 0.01,
            "gross_cumulative_profit": 10000,
            "net_cumulative_profit": 8000,
            "estimated_tax_total": 2000,
            "total_commission": 0,
            "max_drawdown": -0.02,
            "operation_days": 2,
            "realized_profit": 8000,
            "unrealized_profit": 2000,
            "gross_profit_total": 12000,
            "gross_loss_total": -2000,
            "cash": 800000,
            "positions_value": 210000,
            "reconciled_assets": 1010000,
            "reconciliation_difference": 0,
            "reconciliation_ok": True,
            "open_positions_count": 1,
            "closed_trades_count": 2,
        },
        "trade_analysis": {
            "total_trades": 2,
            "winning_trades": 1,
            "losing_trades": 1,
            "win_count": 1,
            "loss_count": 1,
            "win_rate": 0.5,
            "gross_profit_total": 12000,
            "gross_loss_total": -2000,
            "profit_factor": 6.0,
            "closed_trade_count": 2,
            "excluded_order_event_count": 0,
            "profit_ratio": 2.0,
            "expectancy": 0.02,
            "average_profit_rate": 0.04,
            "average_loss_rate": -0.02,
            "average_win_profit_rate": 0.04,
            "average_loss_profit_rate": -0.02,
            "average_holding_days": 3,
            "largest_win": 12000,
            "largest_loss": -2000,
            "worst_loss_profit_rate": -0.02,
            "best_trade": None,
            "worst_trade": None,
            "take_profit_count": 1,
            "stop_loss_count": 1,
            "stop_loss_slippage_average": 0,
            "stop_loss_slippage_max": 0,
            "loss_over_stop_count": 0,
            "loss_over_stop_rate": 0,
            "max_holding_exit_count": 0,
            "average_slippage": 0,
            "max_slippage": 0,
            "gap_up_count": 0,
            "gap_down_count": 0,
            "sold_before_take_profit_rate": 0.5,
            "exit_reason_analysis": [
                {"exit_reason": "利確", "count": 1, "win_rate": 1.0, "average_profit_rate": 0.04, "total_profit": 12000},
                {"exit_reason": "損切り", "count": 1, "win_rate": 0.0, "average_profit_rate": -0.02, "total_profit": -2000},
                {"exit_reason": "最大保有期間到達", "count": 0, "win_rate": None, "average_profit_rate": None, "total_profit": 0},
            ],
        },
        "score_analysis": {
            "selected_count": 0,
            "selected_average_score": None,
            "rejected_average_score": None,
            "score_bands": {"90_or_more": 0, "80s": 0, "70s": 0, "60s": 0, "under_60": 0},
        },
        "reflection_analysis": {
            "reflection_count": 0,
            "win_common_good_points": [],
            "loss_common_bad_points": [],
            "suggestions": [],
        },
        "config_version_analysis": [],
        "sector_win_rate_analysis": [],
        "profile_analysis": [],
        "yearly_performance": [
            {"year": "2026", "profit": 10000, "win_rate": 0.5, "profit_factor": 6.0, "max_drawdown": -0.02}
        ],
        "monthly_performance": [
            {"month": "2026-03", "profit": 10000, "trades": 2, "win_rate": 0.5}
        ],
    }

    markdown = render_analysis_markdown(analysis)

    assert "## Yearly Performance" in markdown
    assert "### 2026" in markdown
    assert "- profit_factor: 6.00" in markdown
    assert "## Monthly Performance" in markdown
    assert "### 2026-03" in markdown
    assert "- trades: 2" in markdown
    assert "利確到達前に売った取引の割合: 50.00%" in markdown
    assert "利確: 件数 1件, 勝率 100.00%, 平均利益率 4.00%, 合計利益 12,000円" in markdown
    assert "最大保有期間到達: 件数 0件" in markdown
