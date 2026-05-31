from __future__ import annotations

from db import (
    _market_regime_performance_analysis,
    _monthly_performance_analysis,
    _portfolio_analysis,
    _trade_analysis,
    _walk_forward_validation,
    _yearly_performance_analysis,
)
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
            "holding_days": 2,
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
            "holding_days": 2,
            "stop_loss_slippage_rate": -0.01,
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "max_holding_win",
            "code": "1005",
            "name": "Max Holding Winner",
            "profit": 3000,
            "profit_rate": 0.02,
            "gross_profit": 3000,
            "total_commission": 100,
            "result": "WIN",
            "exit_reason": "最大保有期間到達",
            "holding_days": 5,
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "other_loss",
            "code": "1006",
            "name": "Other Loss",
            "profit": -1000,
            "profit_rate": -0.01,
            "gross_profit": -1000,
            "total_commission": 100,
            "result": "LOSS",
            "exit_reason": "手動売却",
            "holding_days": 3,
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

    assert analysis["win_count"] == 2
    assert analysis["loss_count"] == 2
    assert analysis["closed_trade_count"] == 4
    assert analysis["excluded_order_event_count"] == 1
    assert analysis["total_trades"] == 4
    assert analysis["gross_profit_total"] == 13000
    assert analysis["realized_profit_total"] == 8000
    assert analysis["gross_loss_total"] == -5000
    assert analysis["profit_factor"] == 2.6
    assert analysis["average_win_profit_rate"] == 0.06
    assert analysis["average_loss_profit_rate"] == -0.025
    assert analysis["average_holding_days"] == 3.0
    assert analysis["largest_win"] == 10000
    assert analysis["largest_loss"] == -4000
    assert analysis["profit_ratio"] == 2.4
    assert analysis["expectancy"] == 0.0175
    assert analysis["worst_loss_profit_rate"] == -0.04
    assert analysis["stop_loss_slippage_average"] == -0.01
    assert analysis["stop_loss_slippage_max"] == -0.01
    assert analysis["loss_over_stop_count"] == 1
    assert analysis["loss_over_stop_rate"] == 0.5
    assert analysis["best_trade"]["code"] == "1001"
    assert analysis["worst_trade"]["code"] == "1002"
    assert analysis["sold_before_take_profit_rate"] == 0.75
    assert analysis["exit_reason_analysis"] == [
        {"exit_reason": "利確", "count": 1, "win_rate": 1.0, "avg_profit": 10000.0, "avg_profit_rate": 0.1, "average_profit_rate": 0.1, "total_profit": 10000.0, "avg_holding_days": 2.0},
        {"exit_reason": "損切り", "count": 1, "win_rate": 0.0, "avg_profit": -4000.0, "avg_profit_rate": -0.04, "average_profit_rate": -0.04, "total_profit": -4000.0, "avg_holding_days": 2.0},
        {"exit_reason": "最大保有期間到達", "count": 1, "win_rate": 1.0, "avg_profit": 3000.0, "avg_profit_rate": 0.02, "average_profit_rate": 0.02, "total_profit": 3000.0, "avg_holding_days": 5.0},
        {"exit_reason": "その他", "count": 1, "win_rate": 0.0, "avg_profit": -1000.0, "avg_profit_rate": -0.01, "average_profit_rate": -0.01, "total_profit": -1000.0, "avg_holding_days": 3.0},
    ]
    assert analysis["exit_efficiency"] == {
        "take_profit_count": 1,
        "stop_loss_count": 1,
        "max_holding_count": 1,
        "max_holding_profit_count": 1,
        "max_holding_loss_count": 0,
    }
    assert analysis["holding_period_analysis"] == [
        {"holding_days": 2, "count": 2, "win_rate": 0.5, "avg_profit_rate": 0.03, "total_profit": 6000.0},
        {"holding_days": 3, "count": 1, "win_rate": 0.0, "avg_profit_rate": -0.01, "total_profit": -1000.0},
        {"holding_days": 5, "count": 1, "win_rate": 1.0, "avg_profit_rate": 0.02, "total_profit": 3000.0},
    ]
    assert [item["suggestion"] for item in analysis["candidate_exit_improvements"]] == [
        "max_holding_days を延ばす候補",
        "stop_loss は機能",
    ]


def test_holding_period_optimization_ranks_six_day_candidate(config_copy: dict) -> None:
    rows = [
        {
            "action": "SELL",
            "trade_id": "hold4",
            "code": "2004",
            "profit": -1000,
            "profit_rate": -0.01,
            "gross_profit": -1000,
            "result": "LOSS",
            "holding_days": 4,
            "exit_date": "2026-03-04",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "hold5",
            "code": "2005",
            "profit": 1000,
            "profit_rate": 0.01,
            "gross_profit": 1000,
            "result": "WIN",
            "holding_days": 5,
            "exit_date": "2026-03-05",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "hold6",
            "code": "2006",
            "profit": 10000,
            "profit_rate": 0.1,
            "gross_profit": 10000,
            "result": "WIN",
            "holding_days": 6,
            "exit_date": "2026-03-06",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "hold7",
            "code": "2007",
            "profit": -5000,
            "profit_rate": -0.05,
            "gross_profit": -5000,
            "result": "LOSS",
            "holding_days": 7,
            "exit_date": "2026-03-07",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "hold8",
            "code": "2008",
            "profit": -1000,
            "profit_rate": -0.01,
            "gross_profit": -1000,
            "result": "LOSS",
            "holding_days": 8,
            "exit_date": "2026-03-08",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "hold10",
            "code": "2010",
            "profit": -1000,
            "profit_rate": -0.01,
            "gross_profit": -1000,
            "result": "LOSS",
            "holding_days": 10,
            "exit_date": "2026-03-10",
            "order_status": "FILLED",
        },
    ]

    analysis = _trade_analysis(config_copy, rows)
    optimization = analysis["holding_period_optimization"]
    ranking = optimization["estimated_profit_ranking"]

    assert [item["max_holding_days"] for item in ranking] == [6, 7, 8, 10, 5, 4]
    assert ranking[0] == {
        "max_holding_days": 6,
        "sample_count": 3,
        "estimated_profit": 10000.0,
        "estimated_profit_factor": 11.0,
        "estimated_win_rate": 0.6667,
        "estimated_drawdown": -0.001,
    }
    assert optimization["calculation_details"] == {
        "current_profit_formula": "sum(profit for all closed trades)",
        "simulated_profit_formula": "sum(profit for closed trades with holding_days <= max_holding_days)",
        "lift_vs_current_formula": "simulated_profit - current_profit",
        "current_profit": 3000.0,
        "simulated_profit": 10000.0,
        "profit_difference": 7000.0,
        "lift_vs_current": 7000.0,
        "base_trade_count": 6,
        "simulated_trade_count": 3,
    }
    assert optimization["candidate_holding_days"] == [
        {
            "recommended_max_holding_days": 6,
            "reason": "max_holding_days=6 が推定利益 10,000円で、実績利益 3,000円を上回ります。",
            "estimated_profit": 10000.0,
            "estimated_profit_lift_vs_current": 7000.0,
            "estimated_profit_factor": 11.0,
            "estimated_win_rate": 0.6667,
            "estimated_drawdown": -0.001,
        }
    ]


def test_holding_period_optimization_does_not_recommend_when_actual_profit_is_higher(config_copy: dict) -> None:
    rows = [
        {
            "action": "SELL",
            "trade_id": "hold4",
            "code": "3004",
            "profit": -1000,
            "profit_rate": -0.01,
            "gross_profit": -1000,
            "result": "LOSS",
            "holding_days": 4,
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "hold6",
            "code": "3006",
            "profit": 10000,
            "profit_rate": 0.1,
            "gross_profit": 10000,
            "result": "WIN",
            "holding_days": 6,
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "hold12",
            "code": "3012",
            "profit": 5000,
            "profit_rate": 0.05,
            "gross_profit": 5000,
            "result": "WIN",
            "holding_days": 12,
            "order_status": "FILLED",
        },
    ]

    analysis = _trade_analysis(config_copy, rows)
    optimization = analysis["holding_period_optimization"]

    assert optimization["estimated_profit_ranking"][0]["max_holding_days"] == 6
    assert optimization["estimated_profit_ranking"][0]["estimated_profit"] == 9000.0
    assert optimization["calculation_details"]["current_profit"] == 14000.0
    assert optimization["calculation_details"]["simulated_profit"] == 9000.0
    assert optimization["calculation_details"]["profit_difference"] == -5000.0
    assert optimization["calculation_details"]["lift_vs_current"] == -5000.0
    assert optimization["calculation_details"]["base_trade_count"] == 3
    assert optimization["calculation_details"]["simulated_trade_count"] == 2
    assert optimization["candidate_holding_days"] == []


def test_trade_replay_analysis_outputs_top_profit_and_loss_paths(config_copy: dict) -> None:
    rows = [
        {
            "action": "SELL",
            "trade_id": "replay_win",
            "code": "4001",
            "name": "Replay Winner",
            "entry_date": "2026-04-01",
            "entry_price": 100,
            "holding_days": 6,
            "profit": 10000,
            "profit_rate": 0.1,
            "gross_profit": 10000,
            "result": "WIN",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "replay_loss",
            "code": "4002",
            "name": "Replay Loser",
            "entry_date": "2026-04-01",
            "entry_price": 100,
            "holding_days": 5,
            "profit": -5000,
            "profit_rate": -0.05,
            "gross_profit": -5000,
            "result": "LOSS",
            "order_status": "FILLED",
        },
    ]
    prices_by_code = {
        "4001": [
            {"date": "2026-04-01", "close": 100},
            *[
                {"date": f"2026-04-{day:02d}", "close": 100 + day - 1}
                for day in range(2, 12)
            ],
        ],
        "4002": [
            {"date": "2026-04-01", "close": 100},
            *[
                {"date": f"2026-04-{day:02d}", "close": 100 - day + 1}
                for day in range(2, 12)
            ],
        ],
    }

    analysis = _trade_analysis(config_copy, rows, prices_by_code)
    replay = analysis["trade_replay_analysis"]

    assert replay["top_profit_trades"][0]["code"] == "4001"
    assert replay["top_profit_trades"][0]["entry_return_rate"] == 0.0
    assert replay["top_profit_trades"][0]["day_returns"][0] == {
        "day": 1,
        "close": 101.0,
        "return_rate": 0.01,
    }
    assert replay["top_profit_trades"][0]["day_returns"][9] == {
        "day": 10,
        "close": 110.0,
        "return_rate": 0.1,
    }
    assert replay["top_loss_trades"][0]["code"] == "4002"
    assert replay["top_loss_trades"][0]["day_returns"][0] == {
        "day": 1,
        "close": 99.0,
        "return_rate": -0.01,
    }
    assert replay["top_loss_trades"][0]["day_returns"][9] == {
        "day": 10,
        "close": 90.0,
        "return_rate": -0.1,
    }
    assert replay["winner_average_replay"][1] == {"label": "day1", "return_rate": 0.01, "count": 1}
    assert replay["loser_average_replay"][1] == {"label": "day1", "return_rate": -0.01, "count": 1}


def test_stop_loss_recovery_analysis_compares_recovered_and_unrecovered(config_copy: dict) -> None:
    rows = [
        {
            "action": "SELL",
            "trade_id": "stop_recovered",
            "code": "5001",
            "name": "Recovered Stop",
            "entry_date": "2026-05-01",
            "entry_price": 100,
            "holding_days": 2,
            "profit": -3000,
            "profit_rate": -0.03,
            "gross_profit": -3000,
            "result": "LOSS",
            "exit_reason": "損切り",
            "rsi": 55,
            "volume_ratio": 3.2,
            "market_regime": "neutral",
            "sector_name": "機械",
            "candlestick_signals": '["volume_confirmed_breakout"]',
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "stop_unrecovered",
            "code": "5002",
            "name": "Unrecovered Stop",
            "entry_date": "2026-05-01",
            "entry_price": 100,
            "holding_days": 2,
            "profit": -4000,
            "profit_rate": -0.04,
            "gross_profit": -4000,
            "result": "LOSS",
            "exit_reason": "損切り",
            "rsi": 72,
            "volume_ratio": 1.2,
            "market_regime": "risk_off",
            "sector_name": "小売業",
            "candlestick_signals": '["upper_shadow"]',
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "trade_id": "take_profit_ignored",
            "code": "5003",
            "name": "Ignored Winner",
            "entry_date": "2026-05-01",
            "entry_price": 100,
            "holding_days": 2,
            "profit": 6000,
            "profit_rate": 0.06,
            "gross_profit": 6000,
            "result": "WIN",
            "exit_reason": "利確",
            "order_status": "FILLED",
        },
    ]
    prices_by_code = {
        "5001": [
            {"date": "2026-05-01", "close": 100},
            {"date": "2026-05-02", "close": 99},
            {"date": "2026-05-03", "close": 98},
            {"date": "2026-05-04", "close": 99},
            {"date": "2026-05-05", "close": 100},
            {"date": "2026-05-06", "close": 101},
            {"date": "2026-05-07", "close": 102},
            {"date": "2026-05-08", "close": 103},
            {"date": "2026-05-09", "close": 104},
            {"date": "2026-05-10", "close": 105},
            {"date": "2026-05-11", "close": 106},
        ],
        "5002": [
            {"date": "2026-05-01", "close": 100},
            {"date": "2026-05-02", "close": 97},
            {"date": "2026-05-03", "close": 96},
            {"date": "2026-05-04", "close": 95},
            {"date": "2026-05-05", "close": 96},
            {"date": "2026-05-06", "close": 96},
            {"date": "2026-05-07", "close": 95},
            {"date": "2026-05-08", "close": 95},
            {"date": "2026-05-09", "close": 94},
            {"date": "2026-05-10", "close": 94},
            {"date": "2026-05-11", "close": 94},
        ],
    }

    analysis = _trade_analysis(config_copy, rows, prices_by_code)
    recovery = analysis["stop_loss_recovery_analysis"]
    signals = {
        (item["feature"], item["value"]): item
        for item in recovery["recovery_signals"]
    }

    assert recovery["stop_loss_count"] == 2
    assert recovery["replay_count"] == 2
    assert recovery["day5_recovery_rate"] == 0.5
    assert recovery["day10_recovery_rate"] == 0.5
    assert recovery["recovery_winners"][0]["code"] == "5001"
    assert recovery["recovery_winners"][0]["day1_return"] == -0.01
    assert recovery["recovery_winners"][0]["day5_return"] == 0.01
    assert recovery["recovery_winners"][0]["day10_return"] == 0.06
    assert recovery["recovery_losers"][0]["code"] == "5002"
    assert signals[("candlestick_signal", "volume_confirmed_breakout")]["share_difference"] == 1.0
    assert signals[("market_regime", "risk_off")]["share_difference"] == -1.0
    assert recovery["candidate_dynamic_stop_rules"][0]["rule"] == "Day1 -2%以内なら保有継続候補"


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


def test_walk_forward_validation_splits_stable_and_weak_periods(config_copy: dict) -> None:
    portfolio_rows = [
        {"date": "2025-03-31", "max_drawdown": -0.01},
        {"date": "2025-07-31", "max_drawdown": -0.08},
        {"date": "2025-11-30", "max_drawdown": -0.02},
    ]
    trade_rows = [
        {
            "action": "SELL",
            "entry_date": "2025-03-10",
            "exit_date": "2025-03-14",
            "profit": 10000,
            "gross_profit": 10000,
            "profit_rate": 0.05,
            "result": "WIN",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "entry_date": "2025-07-10",
            "exit_date": "2025-07-14",
            "profit": -6000,
            "gross_profit": -6000,
            "profit_rate": -0.03,
            "result": "LOSS",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "entry_date": "2025-11-10",
            "exit_date": "2025-11-14",
            "profit": 8000,
            "gross_profit": 8000,
            "profit_rate": 0.04,
            "result": "WIN",
            "order_status": "FILLED",
        },
        {
            "action": "SELL",
            "entry_date": "2026-01-10",
            "exit_date": "2026-01-14",
            "profit": -2000,
            "gross_profit": -2000,
            "profit_rate": -0.01,
            "result": "LOSS",
            "order_status": "FILLED",
        },
    ]

    validation = _walk_forward_validation(config_copy, portfolio_rows, trade_rows)

    assert validation["periods"] == [
        {
            "start_date": "2025-03-01",
            "end_date": "2025-06-30",
            "net_cumulative_profit": 10000.0,
            "win_rate": 1.0,
            "profit_factor": None,
            "max_drawdown": -0.01,
            "total_trades": 1,
            "expectancy": 0.05,
        },
        {
            "start_date": "2025-07-01",
            "end_date": "2025-10-31",
            "net_cumulative_profit": -6000.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": -0.08,
            "total_trades": 1,
            "expectancy": -0.03,
        },
        {
            "start_date": "2025-11-01",
            "end_date": "2026-03-06",
            "net_cumulative_profit": 6000.0,
            "win_rate": 0.5,
            "profit_factor": 4.0,
            "max_drawdown": -0.02,
            "total_trades": 2,
            "expectancy": 0.015,
        },
    ]
    assert [item["start_date"] for item in validation["stable_periods"]] == ["2025-03-01", "2025-11-01"]
    assert [item["start_date"] for item in validation["weak_periods"]] == ["2025-07-01"]
    assert validation["overfit_risk"] == {
        "risk_level": "moderate",
        "stable_period_count": 2,
        "weak_period_count": 1,
        "reason": "安定期間と弱い期間が混在しています。",
    }


def test_market_regime_performance_identifies_best_worst_and_filters(config_copy: dict) -> None:
    rows = [
        {
            "action": "SELL",
            "entry_date": "2026-01-05",
            "exit_date": "2026-01-08",
            "profit": 10000,
            "gross_profit": 10000,
            "profit_rate": 0.05,
            "result": "WIN",
            "order_status": "FILLED",
            "market_regime": "risk_on",
        },
        {
            "action": "SELL",
            "entry_date": "2026-01-10",
            "exit_date": "2026-01-14",
            "profit": -2000,
            "gross_profit": -2000,
            "profit_rate": -0.01,
            "result": "LOSS",
            "order_status": "FILLED",
            "market_regime": "risk_on",
        },
        {
            "action": "SELL",
            "entry_date": "2026-01-15",
            "exit_date": "2026-01-19",
            "profit": 3000,
            "gross_profit": 3000,
            "profit_rate": 0.02,
            "result": "WIN",
            "order_status": "FILLED",
            "market_regime": "neutral",
        },
        {
            "action": "SELL",
            "entry_date": "2026-01-20",
            "exit_date": "2026-01-23",
            "profit": -6000,
            "gross_profit": -6000,
            "profit_rate": -0.03,
            "result": "LOSS",
            "order_status": "FILLED",
            "market_regime": "risk_off",
        },
        {
            "action": "BUY",
            "entry_date": "2026-01-24",
            "profit": 999999,
            "gross_profit": 999999,
            "profit_rate": 9.99,
            "result": "WIN",
            "order_status": "FILLED",
            "market_regime": "risk_off",
        },
    ]

    analysis = _market_regime_performance_analysis(config_copy, rows)
    by_regime = {item["market_regime"]: item for item in analysis["regimes"]}

    assert by_regime["risk_on"] == {
        "market_regime": "risk_on",
        "profit": 8000.0,
        "win_rate": 0.5,
        "profit_factor": 5.0,
        "expectancy": 0.02,
        "max_drawdown": -0.002,
        "trade_count": 2,
    }
    assert by_regime["neutral"]["profit"] == 3000.0
    assert by_regime["neutral"]["profit_factor"] is None
    assert by_regime["risk_off"]["profit"] == -6000.0
    assert by_regime["risk_off"]["profit_factor"] == 0.0
    assert analysis["best_regime"]["market_regime"] == "risk_on"
    assert analysis["worst_regime"]["market_regime"] == "risk_off"
    assert [
        item["rule"] for item in analysis["candidate_regime_filters"]
    ] == [
        "market_regime = risk_on は採用維持",
        "market_regime = risk_off は買付抑制候補",
        "risk_off の新規買付制限を維持または強化",
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
                {"exit_reason": "利確", "count": 1, "win_rate": 1.0, "avg_profit": 12000, "avg_profit_rate": 0.04, "average_profit_rate": 0.04, "total_profit": 12000, "avg_holding_days": 3},
                {"exit_reason": "損切り", "count": 1, "win_rate": 0.0, "avg_profit": -2000, "avg_profit_rate": -0.02, "average_profit_rate": -0.02, "total_profit": -2000, "avg_holding_days": 3},
                {"exit_reason": "最大保有期間到達", "count": 0, "win_rate": None, "avg_profit": None, "avg_profit_rate": None, "average_profit_rate": None, "total_profit": 0, "avg_holding_days": None},
                {"exit_reason": "その他", "count": 0, "win_rate": None, "avg_profit": None, "avg_profit_rate": None, "average_profit_rate": None, "total_profit": 0, "avg_holding_days": None},
            ],
            "exit_efficiency": {
                "take_profit_count": 1,
                "stop_loss_count": 1,
                "max_holding_count": 0,
                "max_holding_profit_count": 0,
                "max_holding_loss_count": 0,
            },
            "holding_period_analysis": [
                {"holding_days": 3, "count": 2, "win_rate": 0.5, "avg_profit_rate": 0.01, "total_profit": 10000},
            ],
            "holding_period_optimization": {
                "current_max_holding_days": 5,
                "current_profit": 10000,
                "calculation_details": {
                    "current_profit_formula": "sum(profit for all closed trades)",
                    "simulated_profit_formula": "sum(profit for closed trades with holding_days <= max_holding_days)",
                    "lift_vs_current_formula": "simulated_profit - current_profit",
                    "current_profit": 10000,
                    "simulated_profit": 15000,
                    "profit_difference": 5000,
                    "lift_vs_current": 5000,
                    "base_trade_count": 2,
                    "simulated_trade_count": 3,
                },
                "estimated_profit_ranking": [
                    {"max_holding_days": 6, "sample_count": 3, "estimated_profit": 15000, "estimated_profit_factor": 4.0, "estimated_win_rate": 0.6667, "estimated_drawdown": -0.01},
                    {"max_holding_days": 5, "sample_count": 2, "estimated_profit": 10000, "estimated_profit_factor": 6.0, "estimated_win_rate": 0.5, "estimated_drawdown": -0.02},
                ],
                "candidate_holding_days": [
                    {"recommended_max_holding_days": 6, "estimated_profit": 15000, "estimated_profit_lift_vs_current": 5000, "estimated_profit_factor": 4.0, "estimated_win_rate": 0.6667, "estimated_drawdown": -0.01, "reason": "max_holding_days=6 が推定利益 15,000円で最上位です。"},
                ],
            },
            "candidate_exit_improvements": [
                {"suggestion": "stop_loss は機能", "reason": "損切り 1件の平均損失率が -2.00% で、設定値 -3.00% 付近です。", "current_value": -0.03},
            ],
            "trade_replay_analysis": {
                "top_profit_trades": [
                    {
                        "entry_date": "2026-03-01",
                        "code": "9001",
                        "holding_days": 6,
                        "profit": 12000,
                        "profit_rate": 0.12,
                        "entry_return_rate": 0.0,
                        "day_returns": [
                            {"day": 1, "return_rate": 0.012},
                            {"day": 2, "return_rate": 0.034},
                        ],
                    }
                ],
                "top_loss_trades": [
                    {
                        "entry_date": "2026-03-01",
                        "code": "9002",
                        "holding_days": 5,
                        "profit": -4000,
                        "profit_rate": -0.04,
                        "entry_return_rate": 0.0,
                        "day_returns": [
                            {"day": 1, "return_rate": -0.01},
                            {"day": 2, "return_rate": -0.03},
                        ],
                    }
                ],
                "winner_average_replay": [
                    {"label": "entry", "return_rate": 0.0, "count": 1},
                    {"label": "day1", "return_rate": 0.012, "count": 1},
                ],
                "loser_average_replay": [
                    {"label": "entry", "return_rate": 0.0, "count": 1},
                    {"label": "day1", "return_rate": -0.01, "count": 1},
                ],
            },
            "stop_loss_recovery_analysis": {
                "stop_loss_count": 2,
                "replay_count": 2,
                "day5_recovery_rate": 0.5,
                "day10_recovery_rate": 0.5,
                "recovery_winners": [
                    {
                        "entry_date": "2026-03-01",
                        "code": "9101",
                        "name": "Recovered",
                        "holding_days": 2,
                        "day1_return": -0.01,
                        "day5_return": 0.01,
                        "day10_return": 0.05,
                        "rsi": 55,
                        "volume_ratio": 3.2,
                        "market_regime": "neutral",
                        "sector": "機械",
                        "candlestick_signals": ["volume_confirmed_breakout"],
                    }
                ],
                "recovery_losers": [
                    {
                        "entry_date": "2026-03-01",
                        "code": "9102",
                        "name": "Unrecovered",
                        "holding_days": 2,
                        "day1_return": -0.03,
                        "day5_return": -0.04,
                        "day10_return": -0.05,
                        "rsi": 72,
                        "volume_ratio": 1.2,
                        "market_regime": "risk_off",
                        "sector": "小売業",
                        "candlestick_signals": ["upper_shadow"],
                    }
                ],
                "recovery_signals": [
                    {"feature": "candlestick_signal", "value": "volume_confirmed_breakout", "winner_share": 1.0, "winner_count": 1, "loser_share": 0.0, "loser_count": 0, "share_difference": 1.0},
                ],
                "candidate_dynamic_stop_rules": [
                    {"rule": "Day1 -2%以内なら保有継続候補", "winner_share": 1.0, "winner_count": 1, "loser_share": 0.0, "loser_count": 0, "reason": "損切り後に回復した銘柄はDay1の下落が浅い傾向があります。"},
                ],
            },
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
        "walk_forward_validation": {
            "periods": [
                {"start_date": "2025-03-01", "end_date": "2025-06-30", "net_cumulative_profit": 10000, "win_rate": 1.0, "profit_factor": None, "max_drawdown": -0.01, "total_trades": 1, "expectancy": 0.05},
                {"start_date": "2025-07-01", "end_date": "2025-10-31", "net_cumulative_profit": -6000, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": -0.08, "total_trades": 1, "expectancy": -0.03},
            ],
            "stable_periods": [
                {"start_date": "2025-03-01", "end_date": "2025-06-30", "net_cumulative_profit": 10000, "win_rate": 1.0, "profit_factor": None, "max_drawdown": -0.01, "total_trades": 1, "expectancy": 0.05},
            ],
            "weak_periods": [
                {"start_date": "2025-07-01", "end_date": "2025-10-31", "net_cumulative_profit": -6000, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": -0.08, "total_trades": 1, "expectancy": -0.03},
            ],
            "overfit_risk": {"risk_level": "moderate", "stable_period_count": 1, "weak_period_count": 1, "reason": "安定期間と弱い期間が混在しています。"},
        },
        "market_regime_performance": {
            "regimes": [
                {"market_regime": "risk_on", "profit": 10000, "win_rate": 1.0, "profit_factor": None, "expectancy": 0.05, "max_drawdown": 0.0, "trade_count": 1},
                {"market_regime": "neutral", "profit": 2000, "win_rate": 0.5, "profit_factor": 2.0, "expectancy": 0.01, "max_drawdown": -0.01, "trade_count": 2},
                {"market_regime": "risk_off", "profit": -6000, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": -0.03, "max_drawdown": -0.02, "trade_count": 1},
            ],
            "best_regime": {"market_regime": "risk_on", "profit": 10000, "win_rate": 1.0, "profit_factor": None, "expectancy": 0.05, "max_drawdown": 0.0, "trade_count": 1},
            "worst_regime": {"market_regime": "risk_off", "profit": -6000, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": -0.03, "max_drawdown": -0.02, "trade_count": 1},
            "candidate_regime_filters": [
                {"rule": "market_regime = risk_on は採用維持", "reason": "risk_on が最も強い相場です。"},
                {"rule": "market_regime = risk_off は買付抑制候補", "reason": "risk_off が最も弱い相場です。"},
            ],
        },
    }

    markdown = render_analysis_markdown(analysis)

    assert "## Yearly Performance" in markdown
    assert "### 2026" in markdown
    assert "- profit_factor: 6.00" in markdown
    assert "## Monthly Performance" in markdown
    assert "### 2026-03" in markdown
    assert "- trades: 2" in markdown
    assert "利確到達前に売った取引の割合: 50.00%" in markdown
    assert "## Exit Reason Analysis" in markdown
    assert "利確: 件数 1件, 勝率 100.00%, 平均利益 12,000円, 平均利益率 4.00%, 合計利益 12,000円, 平均保有日数 3.00" in markdown
    assert "最大保有期間到達: 件数 0件" in markdown
    assert "## Exit Efficiency" in markdown
    assert "利確到達件数: 1" in markdown
    assert "## Holding Period Analysis" in markdown
    assert "3日: count 2" in markdown
    assert "## Holding Period Optimization" in markdown
    assert "calculation_details" in markdown
    assert "current_profit_formula: sum(profit for all closed trades)" in markdown
    assert "simulated_profit_formula: sum(profit for closed trades with holding_days <= max_holding_days)" in markdown
    assert "lift_vs_current_formula: simulated_profit - current_profit" in markdown
    assert "current_profit: 10,000円" in markdown
    assert "simulated_profit: 15,000円" in markdown
    assert "profit_difference: 5,000円" in markdown
    assert "base_trade_count: 2" in markdown
    assert "simulated_trade_count: 3" in markdown
    assert "推定利益ランキング" in markdown
    assert "max_holding_days=6: 推定利益 15,000円, 推定PF 4.00, 推定勝率 66.67%, 推定DD -1.00%, sample_count 3" in markdown
    assert "recommended_max_holding_days: 6" in markdown
    assert "## Candidate Exit Improvements" in markdown
    assert "stop_loss は機能" in markdown
    assert "## Trade Replay Analysis" in markdown
    assert "TOP10利益トレード" in markdown
    assert "entry_date 2026-03-01, code 9001, holding_days 6" in markdown
    assert "Day1: 1.20%" in markdown
    assert "TOP10損失トレード" in markdown
    assert "entry_date 2026-03-01, code 9002, holding_days 5" in markdown
    assert "Day1: -1.00%" in markdown
    assert "勝ち組平均推移" in markdown
    assert "負け組平均推移" in markdown
    assert "## Stop Loss Recovery Analysis" in markdown
    assert "Day5回復率: 50.00%" in markdown
    assert "Day10回復率: 50.00%" in markdown
    assert "Recovery Winners" in markdown
    assert "2026-03-01 9101 Recovered" in markdown
    assert "Recovery Losers" in markdown
    assert "2026-03-01 9102 Unrecovered" in markdown
    assert "Recovery Signals" in markdown
    assert "candlestick_signal=volume_confirmed_breakout" in markdown
    assert "Candidate Dynamic Stop Rules" in markdown
    assert "Day1 -2%以内なら保有継続候補" in markdown
    assert "## Walk Forward Validation" in markdown
    assert "2025-03-01 to 2025-06-30: net_cumulative_profit 10,000円" in markdown
    assert "## Stable Periods" in markdown
    assert "## Weak Periods" in markdown
    assert "## Overfit Risk" in markdown
    assert "risk_level: moderate" in markdown
    assert "## Market Regime Performance Analysis" in markdown
    assert "risk_on: profit 10,000円, win_rate 100.00%, PF N/A, expectancy 5.00%, DD 0.00%, trade_count 1" in markdown
    assert "risk_off: profit -6,000円, win_rate 0.00%, PF 0.00, expectancy -3.00%, DD -2.00%, trade_count 1" in markdown
    assert "## Best Regime" in markdown
    assert "## Worst Regime" in markdown
    assert "## Candidate Regime Filters" in markdown
    assert "market_regime = risk_off は買付抑制候補" in markdown
