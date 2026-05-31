from __future__ import annotations

from db import initialize_database, save_market_context, save_scoring_results, save_trades
from feature_analysis import build_feature_analysis, render_feature_analysis_markdown
from paper_trade import execute_real_data_paper_trade, initial_live_paper_state


def test_feature_analysis_groups_closed_trade_results(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    config_copy.setdefault("features", {})["relative_strength"] = True
    config_copy.setdefault("scoring", {})["use_relative_strength_score"] = True
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
                "stock_return_5d": 0.08,
                "stock_return_10d": 0.12,
                "stock_return_20d": 0.18,
                "benchmark_return_5d": 0.01,
                "benchmark_return_10d": 0.02,
                "benchmark_return_20d": 0.03,
                "relative_strength_5d": 0.07,
                "relative_strength_10d": 0.10,
                "relative_strength_20d": 0.15,
                "relative_strength_score": 10,
                "total_score": 52,
                "technical_score": 42,
                "ma_score": 12,
                "rsi_score": 10,
                "volume_score": 8,
                "candlestick_score": 12,
                "market_context_score": 0,
                "sector_score": 0,
                "penalty_score": 0,
                "score_components": {
                    "ma_score": 12,
                    "rsi_score": 10,
                    "volume_score": 8,
                    "candlestick_score": 12,
                    "market_context_score": 0,
                    "relative_strength_score": 10,
                    "sector_score": 0,
                    "penalty_score": 0,
                    "component_total": 52,
                    "total_score": 52,
                    "matches_total_score": True,
                },
                "score_components_total": 52,
                "score_components_match": True,
                "market_regime": "risk_on",
                "advance_ratio": 0.62,
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
                "total_score": 40,
                "technical_score": 41,
                "ma_score": 16,
                "rsi_score": 6,
                "volume_score": 4,
                "candlestick_score": 15,
                "market_context_score": 0,
                "sector_score": 0,
                "penalty_score": -1,
                "score_components": {
                    "ma_score": 16,
                    "rsi_score": 6,
                    "volume_score": 4,
                    "candlestick_score": 15,
                    "market_context_score": 0,
                    "sector_score": 0,
                    "penalty_score": -1,
                    "component_total": 40,
                    "total_score": 40,
                    "matches_total_score": True,
                },
                "score_components_total": 40,
                "score_components_match": True,
                "market_regime": "risk_off",
                "advance_ratio": 0.28,
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
                    "total_score": 41,
                    "ma_score": 12,
                    "rsi_score": 10,
                    "volume_score": 8,
                    "candlestick_score": 11,
                    "market_context_score": 0,
                    "sector_score": 0,
                    "penalty_score": 0,
                    "score_components": {
                        "ma_score": 12,
                        "rsi_score": 10,
                        "volume_score": 8,
                        "candlestick_score": 11,
                        "market_context_score": 0,
                        "sector_score": 0,
                        "penalty_score": 0,
                        "component_total": 41,
                        "total_score": 41,
                        "matches_total_score": True,
                    },
                    "score_components_total": 41,
                    "score_components_match": True,
                    "selected": False,
                    "rejected_reason": "RSI過熱のため新規買付見送り",
                }
            ],
        },
    )

    analysis = build_feature_analysis(config_copy, tmp_path)

    assert analysis["closed_trade_count"] == 2
    assert analysis["missing_feature_counts"]["market_regime"] == 0
    assert analysis["missing_feature_counts"]["advance_ratio"] == 0
    assert analysis["rsi_filter_rejected_count"] == 1
    assert analysis["rsi_filter_rejected_avg_score"] == 41
    assert analysis["rsi_filter_threshold"] == 65
    rsi_by_bucket = {item["bucket"]: item for item in analysis["rsi"]}
    assert list(rsi_by_bucket) == ["0-30", "30-40", "40-50", "50-60", "60-70", "70+"]
    assert rsi_by_bucket["50-60"]["win_rate"] == 1.0
    assert rsi_by_bucket["50-60"]["average_profit_rate"] == 0.1
    assert rsi_by_bucket["70+"]["win_rate"] == 0.0
    assert rsi_by_bucket["70+"]["average_profit_rate"] == -0.03
    score_detail_by_bucket = {item["bucket"]: item for item in analysis["score_detail"]}
    assert list(score_detail_by_bucket) == ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-71", "72-73", "74-75", "76-79", "80+"]
    assert score_detail_by_bucket["40-44"]["count"] == 1
    assert score_detail_by_bucket["40-44"]["win_rate"] == 0.0
    assert score_detail_by_bucket["50-54"]["count"] == 1
    assert score_detail_by_bucket["50-54"]["win_rate"] == 1.0
    contribution = analysis["score_contribution"]
    assert contribution["selected_score_averages"]["technical_score"] == 41.5
    assert contribution["selected_score_averages"]["relative_strength_score"] == 10.0
    technical_by_bucket = {item["bucket"]: item for item in contribution["technical_score"]}
    assert technical_by_bucket["40-50"]["count"] == 2
    assert technical_by_bucket["40-50"]["win_rate"] == 0.5
    component_analysis = analysis["score_component_analysis"]
    assert component_analysis["score_components_validation"]["missing_score_components_count"] == 0
    assert component_analysis["score_components_validation"]["total_score_mismatch_count"] == 0
    assert component_analysis["score_components_validation"]["selected_scoring_rows_count"] == 0
    assert component_analysis["score_components_validation"]["rejected_scoring_rows_count"] == 1
    rsi_score_by_bucket = {item["bucket"]: item for item in component_analysis["rsi_score"]}
    assert rsi_score_by_bucket["10-15"]["count"] == 1
    assert rsi_score_by_bucket["5-8"]["count"] == 1
    volume_score_by_bucket = {item["bucket"]: item for item in component_analysis["volume_score"]}
    assert volume_score_by_bucket["8-10"]["win_rate"] == 1.0
    penalty_by_bucket = {item["bucket"]: item for item in component_analysis["penalty_score"]}
    assert penalty_by_bucket["<0"]["count"] == 1
    formula_audit = analysis["score_formula_audit"]
    assert formula_audit["total_score_mismatch_count"] == 0
    assert formula_audit["expected_score_range"] == "0-60"
    assert any("RSI is used in rsi_score" in item["message"] for item in formula_audit["duplicated_signal_warnings"])
    assert any("candlestick signals" in item["message"] for item in formula_audit["duplicated_signal_warnings"])
    stats_by_component = {item["component"]: item for item in formula_audit["component_stats"]}
    assert stats_by_component["relative_strength_score"]["max"] == 10
    effective_audit = analysis["score_effective_range_audit"]
    assert effective_audit["theoretical_max_score"] == 60.0
    assert effective_audit["observed_max_score"] == 41
    assert effective_audit["effective_max_score"] < effective_audit["theoretical_max_score"]
    market_by_bucket = {item["bucket"]: item for item in analysis["market_regime"]}
    assert list(market_by_bucket) == ["risk_on", "neutral", "risk_off"]
    assert market_by_bucket["risk_on"]["win_rate"] == 1.0
    assert market_by_bucket["risk_off"]["win_rate"] == 0.0
    assert {item["bucket"] for item in analysis["candlestick_signal"]} == {
        "bullish_candle",
        "long_upper_shadow_warning",
    }
    assert "RSI別勝率" in render_feature_analysis_markdown(analysis)
    assert "score詳細分析" in render_feature_analysis_markdown(analysis)
    assert "Score Contribution Analysis" in render_feature_analysis_markdown(analysis)
    assert "Score Component Analysis" in render_feature_analysis_markdown(analysis)
    assert "Score Formula Audit" in render_feature_analysis_markdown(analysis)
    assert "Score Effective Range Audit" in render_feature_analysis_markdown(analysis)
    assert "Duplicated Signal Warning" in render_feature_analysis_markdown(analysis)
    assert "Relative Strength Analysis" in render_feature_analysis_markdown(analysis)
    assert "relative_strength_5d帯別" in render_feature_analysis_markdown(analysis)
    assert "technical_score average: 41.50" in render_feature_analysis_markdown(analysis)


def test_feature_activation_audit_marks_enabled_feature_inactive_in_practice(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    config_copy.setdefault("features", {})["financial_context"] = True
    config_copy.setdefault("scoring", {})["use_financial_score"] = True
    initialize_database(config_copy, tmp_path)
    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-01",
            "scores": [
                {
                    "code": "1001",
                    "name": "No Financial Trigger",
                    "rank": 1,
                    "total_score": 45,
                    "technical_score": 45,
                    "selected": True,
                }
            ],
        },
    )

    analysis = build_feature_analysis(config_copy, tmp_path)
    audit = analysis["feature_activation_audit"]

    assert audit["features"]["financial_context"]["enabled"] is True
    assert audit["features"]["financial_context"]["non_zero_score_count"] == 0
    assert audit["features"]["financial_context"]["status"] == "inactive_in_practice"
    assert "financial_context" in audit["inactive_in_practice"]
    markdown = render_feature_analysis_markdown(analysis)
    assert "## Feature Activation Audit" in markdown
    assert "inactive_in_practice" in markdown


def test_score_effective_range_audit_marks_inactive_components(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-01",
            "scores": [
                {
                    "code": "1001",
                    "name": "Selected",
                    "rank": 1,
                    "total_score": 49,
                    "technical_score": 49,
                    "market_context_score": 0,
                    "penalty_score": 0,
                    "rsi_score": 10,
                    "volume_score": 8,
                    "candlestick_score": 12,
                    "sector_score": 0,
                    "score_components": {
                        "ma_score": 19,
                        "rsi_score": 10,
                        "volume_score": 8,
                        "candlestick_score": 12,
                        "sector_score": 0,
                        "market_context_score": 0,
                        "relative_strength_score": 0,
                        "penalty_score": 0,
                        "component_total": 49,
                        "total_score": 49,
                        "matches_total_score": True,
                    },
                    "score_components_total": 49,
                    "score_components_match": True,
                    "selected": True,
                },
                {
                    "code": "1002",
                    "name": "Rejected",
                    "rank": 2,
                    "total_score": 45,
                    "technical_score": 45,
                    "market_context_score": 0,
                    "penalty_score": 0,
                    "rsi_score": 8,
                    "volume_score": 7,
                    "candlestick_score": 10,
                    "sector_score": 0,
                    "score_components": {
                        "ma_score": 20,
                        "rsi_score": 8,
                        "volume_score": 7,
                        "candlestick_score": 10,
                        "sector_score": 0,
                        "market_context_score": 0,
                        "relative_strength_score": 0,
                        "penalty_score": 0,
                        "component_total": 45,
                        "total_score": 45,
                        "matches_total_score": True,
                    },
                    "score_components_total": 45,
                    "score_components_match": True,
                    "selected": False,
                },
            ],
        },
    )

    analysis = build_feature_analysis(config_copy, tmp_path)
    audit = analysis["score_effective_range_audit"]
    components = {item["component"]: item for item in audit["components"]}

    assert audit["theoretical_max_score"] == 50.0
    assert audit["observed_max_score"] == 49
    assert audit["observed_max_score"] < audit["theoretical_max_score"]
    assert "news" + "_score" not in components
    assert "financial_score" not in components
    assert components["market_context_score"]["status"] == "inactive"
    assert components["penalty_score"]["status"] == "inactive"


def test_sell_trade_inherits_buy_time_features(config_copy: dict) -> None:
    config_copy.setdefault("execution", {})["use_next_day_open_execution"] = False
    state = initial_live_paper_state(config_copy)
    day1_candidate = {
        "code": "1001",
        "name": "Feature Stock",
        "sector_name": "情報・通信",
        "close": 1000,
        "selected": True,
        "total_score": 44,
        "technical_score": 44,
        "ma_score": 12,
        "rsi_score": 10,
        "volume_score": 8,
        "candlestick_score": 14,
        "market_context_score": 0,
        "sector_score": 0,
        "penalty_score": 0,
        "score_components": {
            "ma_score": 12,
            "rsi_score": 10,
            "volume_score": 8,
            "candlestick_score": 14,
            "market_context_score": 0,
            "sector_score": 0,
            "penalty_score": 0,
            "component_total": 44,
            "total_score": 44,
            "matches_total_score": True,
        },
        "score_components_total": 44,
        "score_components_match": True,
        "confidence": 0.9,
        "reason": "test buy",
        "rsi": 56,
        "volume_ratio": 2.4,
        "market_regime": "risk_on",
        "advance_ratio": 0.62,
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
    assert sell["total_score"] == 44
    assert sell["technical_score"] == 44
    assert "news" + "_score" not in sell
    assert "financial_score" not in sell
    assert sell["rsi_score"] == 10
    assert sell["volume_score"] == 8
    assert sell["candlestick_score"] == 14
    assert sell["score_components"]["matches_total_score"] is True
    assert sell["market_regime"] == "risk_on"
    assert sell["advance_ratio"] == 0.62
    assert sell["candlestick_signals"] == ["bullish_candle"]
    assert sell["selected_reason"] == "test buy"


def test_feature_analysis_fills_market_regime_from_entry_market_context(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
    save_market_context(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-01",
            "provider": "test",
            "market_regime": "risk_on",
            "advance_ratio": 0.71,
        },
    )
    save_trades(
        config_copy,
        tmp_path,
        "2026-03-06",
        [
            {
                "trade_id": "closed-without-trade-market",
                "action": "SELL",
                "code": "1001",
                "name": "Fallback",
                "entry_date": "2026-03-01",
                "exit_date": "2026-03-06",
                "profit": 5000,
                "profit_rate": 0.05,
                "gross_profit": 5000,
                "result": "WIN",
                "order_status": "FILLED",
                "rsi": 55,
                "volume_ratio": 1.5,
                "total_score": 74,
            }
        ],
    )

    analysis = build_feature_analysis(config_copy, tmp_path)

    market_by_bucket = {item["bucket"]: item for item in analysis["market_regime"]}
    assert market_by_bucket["risk_on"]["count"] == 1
    assert analysis["missing_feature_counts"]["market_regime"] == 0
    assert analysis["missing_feature_counts"]["advance_ratio"] == 0
    assert analysis["records_used"][0]["market_regime"] == "risk_on"
    assert analysis["records_used"][0]["advance_ratio"] == 0.71


def test_feature_analysis_uses_unknown_only_without_trade_or_context_market(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
    save_trades(
        config_copy,
        tmp_path,
        "2026-03-06",
        [
            {
                "trade_id": "closed-without-market",
                "action": "SELL",
                "code": "1001",
                "name": "Unknown",
                "entry_date": "2026-03-01",
                "exit_date": "2026-03-06",
                "profit": -1000,
                "profit_rate": -0.01,
                "gross_profit": -1000,
                "result": "LOSS",
                "order_status": "FILLED",
                "rsi": 55,
                "volume_ratio": 1.5,
                "total_score": 74,
            }
        ],
    )

    analysis = build_feature_analysis(config_copy, tmp_path)

    market_by_bucket = {item["bucket"]: item for item in analysis["market_regime"]}
    assert market_by_bucket["unknown"]["count"] == 1
    assert analysis["missing_feature_counts"]["market_regime"] == 1
