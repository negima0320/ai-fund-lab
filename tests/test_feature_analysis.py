from __future__ import annotations

from db import initialize_database, save_scoring_results, save_trades
from feature_analysis import build_feature_analysis, render_feature_analysis_markdown
from paper_trade import execute_real_data_paper_trade, initial_live_paper_state


def test_feature_analysis_groups_closed_trade_results(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
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
                "rsi": 55,
                "volume_ratio": 2.2,
                "total_score": 82,
                "technical_score": 42,
                "news_score": 10,
                "financial_score": 10,
                "market_regime": "risk_on",
                "sector_name": "情報・通信",
                "candlestick_signals": ["bullish_candle"],
                "selected_reason": "強い形状",
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
                "rsi": 72,
                "volume_ratio": 0.8,
                "total_score": 68,
                "technical_score": 35,
                "news_score": 8,
                "financial_score": 10,
                "market_regime": "risk_off",
                "sector_name": "機械",
                "candlestick_signals": ["long_upper_shadow_warning"],
                "selected_reason": "警戒あり",
            },
        ],
    )
    config_copy["selection"]["max_rsi_for_new_position"] = 65
    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-01",
            "scores": [
                {
                    "code": "1003",
                    "name": "Rejected",
                    "rank": 1,
                    "total_score": 76,
                    "selected": False,
                    "rejected_reason": "RSI過熱のため新規買付見送り",
                }
            ],
        },
    )

    analysis = build_feature_analysis(config_copy, tmp_path)

    assert analysis["closed_trade_count"] == 2
    assert analysis["rsi_filter_rejected_count"] == 1
    assert analysis["rsi_filter_rejected_avg_score"] == 76
    assert analysis["rsi_filter_threshold"] == 65
    rsi_by_bucket = {item["bucket"]: item for item in analysis["rsi"]}
    assert list(rsi_by_bucket) == ["0-30", "30-40", "40-50", "50-60", "60-70", "70+"]
    assert rsi_by_bucket["50-60"]["win_rate"] == 1.0
    assert rsi_by_bucket["50-60"]["average_profit_rate"] == 0.1
    assert rsi_by_bucket["70+"]["win_rate"] == 0.0
    assert rsi_by_bucket["70+"]["average_profit_rate"] == -0.03
    market_by_bucket = {item["bucket"]: item for item in analysis["market_regime"]}
    assert list(market_by_bucket) == ["risk_on", "neutral", "risk_off"]
    assert market_by_bucket["risk_on"]["win_rate"] == 1.0
    assert market_by_bucket["risk_off"]["win_rate"] == 0.0
    assert {item["bucket"] for item in analysis["candlestick_signal"]} == {
        "bullish_candle",
        "long_upper_shadow_warning",
    }
    assert "RSI別勝率" in render_feature_analysis_markdown(analysis)


def test_sell_trade_inherits_buy_time_features(config_copy: dict) -> None:
    config_copy.setdefault("execution", {})["use_next_day_open_execution"] = False
    state = initial_live_paper_state(config_copy)
    day1_candidate = {
        "code": "1001",
        "name": "Feature Stock",
        "sector_name": "情報・通信",
        "close": 1000,
        "selected": True,
        "total_score": 82,
        "technical_score": 44,
        "news_score": 8,
        "financial_score": 10,
        "confidence": 0.9,
        "reason": "test buy",
        "rsi": 56,
        "volume_ratio": 2.4,
        "market_regime": "risk_on",
        "candlestick_signals": ["bullish_candle"],
    }
    state, _summary, trades = execute_real_data_paper_trade([day1_candidate], state, config_copy, "2026-03-02")
    assert any(trade["action"] == "BUY" for trade in trades)

    day2_candidate = {
        **day1_candidate,
        "close": 1080,
        "selected": False,
        "confidence": 0.0,
    }
    state, _summary, trades = execute_real_data_paper_trade([day2_candidate], state, config_copy, "2026-03-03")
    sell = next(trade for trade in trades if trade["action"] == "SELL")

    assert sell["rsi"] == 56
    assert sell["volume_ratio"] == 2.4
    assert sell["total_score"] == 82
    assert sell["technical_score"] == 44
    assert sell["news_score"] == 8
    assert sell["financial_score"] == 10
    assert sell["market_regime"] == "risk_on"
    assert sell["candlestick_signals"] == ["bullish_candle"]
    assert sell["selected_reason"] == "test buy"
