from __future__ import annotations

from db import analyze_operation_data, get_database_path, initialize_database, save_portfolio_snapshot, save_trades
from main import _profile_compare_row, build_profile_ranking


def test_analyze_and_compare_profiles_profit_factor_match(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
    save_portfolio_snapshot(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-06",
            "day": 1,
            "cash": 1000000,
            "positions_value": 0,
            "total_assets": 1006000,
            "daily_profit": 6000,
            "cumulative_profit": 6000,
            "cumulative_profit_rate": 0.006,
            "win_rate": 0.5,
            "max_drawdown": 0,
            "open_positions_count": 0,
            "closed_trades_count": 2,
            "gross_cumulative_profit": 6000,
            "net_cumulative_profit": 6000,
            "total_commission": 0,
            "estimated_tax_total": 0,
        },
    )
    save_trades(
        config_copy,
        tmp_path,
        "2026-03-06",
        [
            {
                "trade_id": "win",
                "action": "SELL",
                "code": "1001",
                "name": "Winner",
                "entry_date": "2026-03-01",
                "exit_date": "2026-03-06",
                "profit": 10000,
                "profit_rate": 0.1,
                "gross_profit": 10000,
                "holding_days": 2,
                "result": "WIN",
                "order_status": "FILLED",
            },
            {
                "trade_id": "loss",
                "action": "SELL",
                "code": "1002",
                "name": "Loser",
                "entry_date": "2026-03-01",
                "exit_date": "2026-03-06",
                "profit": -4000,
                "profit_rate": -0.04,
                "gross_profit": -4000,
                "holding_days": 4,
                "result": "LOSS",
                "order_status": "FILLED",
            },
            {
                "trade_id": "pending",
                "action": "SELL",
                "code": "1003",
                "name": "Pending",
                "entry_date": "2026-03-01",
                "exit_date": "2026-03-06",
                "profit": 999999,
                "profit_rate": 9.99,
                "gross_profit": 999999,
                "result": "WIN",
                "order_status": "PENDING",
            },
        ],
    )

    analyze_pf = analyze_operation_data(config_copy, tmp_path)["trade_analysis"]["profit_factor"]
    compare_pf = _profile_compare_row(
        config_copy,
        get_database_path(config_copy, tmp_path),
        "2026-03-01",
        "2026-03-06",
    )["profit_factor"]

    assert analyze_pf == compare_pf == 2.5

    compare_row = _profile_compare_row(
        config_copy,
        get_database_path(config_copy, tmp_path),
        "2026-03-01",
        "2026-03-06",
    )
    assert compare_row["average_win_profit_rate"] == 0.1
    assert compare_row["average_loss_profit_rate"] == -0.04
    assert compare_row["average_holding_days"] == 3.0
    assert compare_row["expectancy"] == 0.03


def test_profile_ranking_uses_profit_factor_drawdown_profit_and_expectancy() -> None:
    ranking = build_profile_ranking(
        [
            {
                "profile_id": "rookie_dealer_01",
                "net_cumulative_profit": 1000,
                "profit_factor": 1.1,
                "max_drawdown": -0.08,
                "expectancy": 0.002,
            },
            {
                "profile_id": "rookie_dealer_02",
                "net_cumulative_profit": 6000,
                "profit_factor": 1.8,
                "max_drawdown": -0.03,
                "expectancy": 0.01,
            },
            {
                "profile_id": "rookie_dealer_03",
                "net_cumulative_profit": 3500,
                "profit_factor": 1.4,
                "max_drawdown": -0.05,
                "expectancy": 0.006,
            },
        ]
    )

    assert [row["profile_id"] for row in ranking] == [
        "rookie_dealer_02",
        "rookie_dealer_03",
        "rookie_dealer_01",
    ]
    assert ranking[0]["rank"] == 1
    assert ranking[0]["score"] > ranking[1]["score"] > ranking[2]["score"]
