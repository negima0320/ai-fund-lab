from __future__ import annotations

from db import initialize_database, save_scoring_results, save_screening_results, save_trades
from feature_analysis import build_feature_analysis, render_feature_analysis_markdown


def test_feature_analysis_groups_closed_trade_results(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
    save_screening_results(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-01",
            "candidates": [
                {
                    "code": "1001",
                    "name": "Winner",
                    "sector_name": "情報・通信",
                    "rsi": 55,
                    "volume_ratio": 2.2,
                    "candlestick_signals": ["bullish_candle"],
                },
                {
                    "code": "1002",
                    "name": "Loser",
                    "sector_name": "機械",
                    "rsi": 72,
                    "volume_ratio": 0.8,
                    "candlestick_signals": ["long_upper_shadow_warning"],
                },
            ],
        },
    )
    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-01",
            "scores": [
                {
                    "code": "1001",
                    "name": "Winner",
                    "sector_name": "情報・通信",
                    "rank": 1,
                    "total_score": 82,
                    "selected": True,
                    "market_regime": "risk_on",
                    "candlestick_signals": ["bullish_candle"],
                },
                {
                    "code": "1002",
                    "name": "Loser",
                    "sector_name": "機械",
                    "rank": 2,
                    "total_score": 68,
                    "selected": True,
                    "market_regime": "risk_off",
                    "candlestick_signals": ["long_upper_shadow_warning"],
                },
            ],
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
                "profit": -3000,
                "profit_rate": -0.03,
                "gross_profit": -3000,
                "result": "LOSS",
                "order_status": "FILLED",
            },
        ],
    )

    analysis = build_feature_analysis(config_copy, tmp_path)

    assert analysis["closed_trade_count"] == 2
    assert analysis["rsi"] == [
        {
            "bucket": "rsi_50_64",
            "count": 1,
            "win_count": 1,
            "loss_count": 0,
            "win_rate": 1.0,
            "average_profit": 10000.0,
            "average_profit_rate": 0.1,
            "total_profit": 10000.0,
        },
        {
            "bucket": "rsi_70_plus",
            "count": 1,
            "win_count": 0,
            "loss_count": 1,
            "win_rate": 0.0,
            "average_profit": -3000.0,
            "average_profit_rate": -0.03,
            "total_profit": -3000.0,
        },
    ]
    assert analysis["market_regime"][0]["bucket"] == "risk_off"
    assert analysis["market_regime"][1]["bucket"] == "risk_on"
    assert {item["bucket"] for item in analysis["candlestick_signal"]} == {
        "bullish_candle",
        "long_upper_shadow_warning",
    }
    assert "RSI別勝率" in render_feature_analysis_markdown(analysis)
