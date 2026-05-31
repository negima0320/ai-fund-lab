from __future__ import annotations

from report import generate_daily_report


def test_daily_report_includes_paper_trading_operation_sections(config_copy: dict) -> None:
    summary = {
        "day_number": 12,
        "date": "2026-03-06",
        "market_context": {},
        "total_assets": 1_020_000,
        "day_change": 5_000,
        "day_change_pct": 0.0049,
        "cumulative_pnl": 20_000,
        "gross_cumulative_profit": 25_000,
        "estimated_tax_total": 1_000,
        "net_cumulative_profit": 24_000,
        "total_commission": 0,
        "win_rate": 0.5,
        "max_drawdown": -0.02,
        "max_drawdown_note": "日次total_assets履歴から計算",
        "walk_forward_validation": {
            "periods": [
                {
                    "start_date": "2025-03-01",
                    "end_date": "2025-06-30",
                    "net_cumulative_profit": 10_000,
                    "win_rate": 0.6,
                    "profit_factor": 1.8,
                    "max_drawdown": -0.03,
                    "total_trades": 10,
                    "expectancy": 0.012,
                }
            ],
            "stable_periods": [
                {
                    "start_date": "2025-03-01",
                    "end_date": "2025-06-30",
                    "net_cumulative_profit": 10_000,
                    "win_rate": 0.6,
                    "profit_factor": 1.8,
                    "max_drawdown": -0.03,
                    "total_trades": 10,
                    "expectancy": 0.012,
                }
            ],
            "weak_periods": [],
            "overfit_risk": {
                "risk_level": "low",
                "stable_period_count": 1,
                "weak_period_count": 0,
                "reason": "安定期間が確認できる",
            },
        },
    }
    paper_trade_log = {
        "date": "2026-03-06",
        "orders": [
            {
                "action": "BUY",
                "code": "1001",
                "name": "Buy Co",
                "quantity": 100,
                "price": 1_000,
            }
        ],
        "pending_orders": [],
        "executed_orders": [],
        "skipped_buys": [],
        "closed_trades": [
            {
                "trade_id": "T-1",
                "action": "SELL",
                "code": "1002",
                "name": "Sell Co",
                "result": "WIN",
                "entry_date": "2026-03-02",
                "exit_date": "2026-03-06",
                "holding_days": 4,
                "shares": 100,
                "profit": 5_000,
                "profit_rate": 0.05,
                "gross_profit": 5_000,
                "net_profit": 4_000,
                "net_profit_rate": 0.04,
                "estimated_tax": 1_000,
                "exit_reason": "利確",
            }
        ],
        "all_closed_trades": [
            {"exit_date": "2026-03-03", "net_profit": 4_000},
            {"exit_date": "2026-03-04", "net_profit": -2_000},
            {"exit_date": "2026-02-27", "net_profit": 1_000},
        ],
        "positions": [
            {
                "code": "1003",
                "name": "Hold A",
                "quantity": 100,
                "market_value": 60_000,
                "unrealized_pnl": 3_000,
                "sector_name": "機械",
            },
            {
                "code": "1004",
                "name": "Hold B",
                "quantity": 100,
                "market_value": 40_000,
                "unrealized_pnl": -1_000,
                "sector_name": "小売業",
            },
        ],
        "safety_events": [],
    }

    markdown = generate_daily_report(summary, paper_trade_log, {"decisions": []}, config_copy)

    assert "## Daily Report" in markdown
    assert "### 今日買った銘柄" in markdown
    assert "1001 Buy Co" in markdown
    assert "### 今日売った銘柄" in markdown
    assert "1002 Sell Co" in markdown
    assert "### 含み損益" in markdown
    assert "2,000円" in markdown
    assert "### セクター比率" in markdown
    assert "機械: 60.00%" in markdown
    assert "### 勝率 / PF" in markdown
    assert "- PF: 2.50" in markdown
    assert "## Weekly Report" in markdown
    assert "### 今週の成績" in markdown
    assert "### 月初来成績" in markdown
    assert "### 年初来成績" in markdown
    assert "## Walk Forward Validation" in markdown
    assert "### Stable Periods" in markdown
    assert "### Weak Periods" in markdown
    assert "- risk_level: low" in markdown
