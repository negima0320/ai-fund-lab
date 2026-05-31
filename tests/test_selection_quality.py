from __future__ import annotations

import json

from db import initialize_database, save_scoring_results, save_screening_results
from selection_quality import build_selection_quality_analysis, render_selection_quality_markdown


def test_selection_quality_compares_selected_and_rejected_future_returns(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    base_date = "2026-01-01"
    future_dates = [f"2026-01-{day:02d}" for day in range(2, 12)]
    prices_by_code = {
        "1001": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "1002": [104, 108, 112, 116, 120, 122, 124, 126, 128, 130],
        "1003": [98, 96, 94, 92, 90, 88, 86, 84, 82, 80],
        "1004": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "1005": [99, 98, 97, 96, 95, 94, 93, 92, 91, 90],
    }
    for index, target_date in enumerate(future_dates):
        payload = {
            "prices": [
                {"code": code, "date": target_date, "close": closes[index]}
                for code, closes in prices_by_code.items()
            ]
        }
        (raw_dir / f"prices_{target_date}.json").write_text(json.dumps(payload), encoding="utf-8")

    candidates = [
        {"code": "1001", "name": "Selected Up", "close": 100, "rsi": 55, "volume_ratio": 2.5, "sector_name": "Tech"},
        {"code": "1002", "name": "Missed Winner", "close": 100, "rsi": 62, "volume_ratio": 1.5, "sector_name": "Retail"},
        {"code": "1003", "name": "False Positive", "close": 100, "rsi": 45, "volume_ratio": 3.5, "sector_name": "Tech"},
        {"code": "1004", "name": "Screen Only", "close": 100, "rsi": 50, "volume_ratio": 1.0, "sector_name": "Bank"},
        {"code": "1005", "name": "Correct Reject", "close": 100, "rsi": 68, "volume_ratio": 2.5, "sector_name": "Retail"},
    ]
    save_screening_results(
        config_copy,
        tmp_path,
        {"date": base_date, "candidates": candidates},
    )
    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": base_date,
            "scores": [
                {
                    "code": "1001",
                    "name": "Selected Up",
                    "rank": 1,
                    "total_score": 80,
                    "relative_strength_score": 10,
                    "relative_strength_5d": 0.07,
                    "relative_strength_10d": 0.09,
                    "relative_strength_20d": 0.12,
                    "selected": True,
                    "market_regime": "risk_on",
                    "sector_name": "Tech",
                    "candlestick_signals": ["bullish_candle"],
                },
                {
                    "code": "1002",
                    "name": "Missed Winner",
                    "rank": 2,
                    "total_score": 78,
                    "relative_strength_score": 7,
                    "relative_strength_5d": 0.04,
                    "relative_strength_10d": 0.06,
                    "relative_strength_20d": 0.01,
                    "selected": False,
                    "rejected_reason": "上限超過",
                    "market_regime": "neutral",
                    "sector_name": "Retail",
                    "candlestick_signals": ["upper_shadow"],
                },
                {
                    "code": "1003",
                    "name": "False Positive",
                    "rank": 3,
                    "total_score": 76,
                    "relative_strength_score": 0,
                    "relative_strength_5d": -0.02,
                    "relative_strength_10d": -0.04,
                    "relative_strength_20d": -0.06,
                    "selected": True,
                    "market_regime": "risk_on",
                    "sector_name": "Tech",
                    "candlestick_signals": ["bullish_candle"],
                },
                {
                    "code": "1005",
                    "name": "Correct Reject",
                    "rank": 4,
                    "total_score": 60,
                    "relative_strength_score": 0,
                    "relative_strength_5d": -0.03,
                    "relative_strength_10d": -0.02,
                    "relative_strength_20d": -0.01,
                    "selected": False,
                    "rejected_reason": "出来高倍率不足のため新規買付見送り",
                    "market_regime": "risk_off",
                    "sector_name": "Retail",
                    "candlestick_signals": ["upper_shadow"],
                },
            ],
        },
    )

    analysis = build_selection_quality_analysis(config_copy, tmp_path)

    assert analysis["screen_candidate_count"] == 5
    assert analysis["score_candidate_count"] == 4
    assert analysis["selected_count"] == 2
    assert analysis["rejected_count"] == 2
    assert analysis["selected"]["average_return_5d"] == -0.025
    assert analysis["rejected"]["average_return_5d"] == 0.075
    assert analysis["selected"]["average_return_10d"] == -0.05
    assert analysis["rejected"]["average_return_10d"] == 0.1
    assert analysis["selection_lift"]["return_5d"] == -0.1
    assert analysis["selection_lift"]["return_10d"] == -0.15
    lift_optimization = analysis["selection_lift_optimization_analysis"]
    averages_by_feature = {item["feature"]: item for item in lift_optimization["feature_averages"]}
    assert averages_by_feature["RSI"]["selected_average"] == 50
    assert averages_by_feature["RSI"]["rejected_average"] == 65
    assert averages_by_feature["RSI"]["difference"] == -15
    assert averages_by_feature["volume_ratio"]["selected_average"] == 3
    assert averages_by_feature["volume_ratio"]["rejected_average"] == 2
    assert averages_by_feature["volume_ratio"]["difference"] == 1
    assert averages_by_feature["total_score"]["selected_average"] == 78
    assert averages_by_feature["total_score"]["rejected_average"] == 69
    assert averages_by_feature["total_score"]["difference"] == 9
    assert averages_by_feature["relative_strength_score"]["selected_average"] == 5
    assert averages_by_feature["relative_strength_score"]["rejected_average"] == 3.5
    assert averages_by_feature["relative_strength_score"]["difference"] == 1.5
    selected_heavy_by_feature_value = {
        (item["feature"], item["value"]): item
        for item in lift_optimization["selected_heavy_features"]
    }
    rejected_heavy_by_feature_value = {
        (item["feature"], item["value"]): item
        for item in lift_optimization["rejected_heavy_features"]
    }
    assert selected_heavy_by_feature_value[("sector", "Tech")]["share_difference"] == 1
    assert rejected_heavy_by_feature_value[("sector", "Retail")]["share_difference"] == -1
    category_by_feature_value = {
        (item["feature"], item["value"]): item
        for item in lift_optimization["category_share_differences"]
    }
    assert category_by_feature_value[("market_regime", "risk_on")]["share_difference"] == 1
    assert category_by_feature_value[("candlestick_signal", "upper_shadow")]["share_difference"] == -1
    deep_analysis = analysis["selection_lift_deep_analysis"]
    volume_bucket = {
        item["value"]: item
        for item in deep_analysis["features"]["volume_ratio"]
    }
    assert volume_bucket["2-3"]["selected_average_future_return_10d"] == 0.1
    assert volume_bucket["2-3"]["selected_win_rate"] == 1.0
    assert volume_bucket["2-3"]["selected_count"] == 1
    assert volume_bucket["2-3"]["rejected_average_future_return_10d"] == -0.1
    assert volume_bucket["2-3"]["rejected_win_rate"] == 0.0
    assert volume_bucket["2-3"]["rejected_count"] == 1
    assert volume_bucket["2-3"]["return_lift_10d"] == 0.2
    assert volume_bucket["2-3"]["sample_note"] == "参考扱い: selected/rejectedのいずれかが5件未満"
    positive_by_feature_value = {
        (item["feature"], item["value"]): item
        for item in deep_analysis["positive_lift_features"]
    }
    assert positive_by_feature_value[("volume_ratio", "2-3")]["return_lift_10d"] == 0.2
    negative_by_feature_value = {
        (item["feature"], item["value"]): item
        for item in deep_analysis["negative_lift_features"]
    }
    assert negative_by_feature_value[("total_score", "75-80")]["return_lift_10d"] == -0.5
    assert deep_analysis["candidate_new_rules"][0]["rule"] == "volume_ratio >= 2 and volume_ratio < 3"
    assert deep_analysis["candidate_new_rules"][0]["confidence"] == "reference"
    assert analysis["top_missed_opportunities"][0]["code"] == "1002"
    assert analysis["top_false_positives"][0]["code"] == "1003"
    reason_by_name = {item["rejected_reason"]: item for item in analysis["rejected_reason_analysis"]}
    assert reason_by_name["上限超過"]["count"] == 1
    assert reason_by_name["上限超過"]["avg_future_return_10d"] == 0.3
    assert reason_by_name["上限超過"]["median_future_return_10d"] == 0.3
    assert reason_by_name["上限超過"]["top_10d_return"] == 0.3
    assert reason_by_name["上限超過"]["bottom_10d_return"] == 0.3
    assert reason_by_name["上限超過"]["positive_rate_10d"] == 1.0
    assert reason_by_name["上限超過"]["filter_effectiveness"] == "harmful"
    assert reason_by_name["出来高倍率不足のため新規買付見送り"]["avg_future_return_10d"] == -0.1
    assert reason_by_name["出来高倍率不足のため新規買付見送り"]["positive_rate_10d"] == 0.0
    assert reason_by_name["出来高倍率不足のため新規買付見送り"]["filter_effectiveness"] == "effective"
    assert analysis["missed_opportunity_by_reason"] == [
        {"rejected_reason": "上限超過", "count": 1, "top_10d_return": 0.3}
    ]

    markdown = render_selection_quality_markdown(analysis)
    assert "selected平均5日リターン" in markdown
    assert "Rejected Reason Analysis" in markdown
    assert "Selection Lift Optimization Analysis" in markdown
    assert "selected平均" in markdown
    assert "Selected側だけに多い特徴" in markdown
    assert "Rejected側だけに多い特徴" in markdown
    assert "relative_strength_score" in markdown
    assert "Selection Lift Deep Analysis" in markdown
    assert "Positive Lift Features" in markdown
    assert "Negative Lift Features" in markdown
    assert "Candidate New Rules" in markdown
    assert "参考扱い" in markdown
    assert "Missed Opportunity by Reason" in markdown
    assert "False Rejection Check" in markdown
    assert "Top Missed Opportunities" in markdown
    assert "Top False Positives" in markdown


def test_sector_lift_analysis_suggests_positive_and_negative_sector_filters(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    base_date = "2026-02-01"
    future_dates = [f"2026-02-{day:02d}" for day in range(2, 12)]
    prices_by_code = {
        "2001": [102, 104, 106, 108, 110, 112, 114, 116, 118, 120],
        "2002": [100, 101, 101, 102, 102, 103, 103, 104, 104, 105],
        "2003": [99, 98, 98, 97, 97, 96, 96, 95, 95, 95],
        "2004": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    }
    for index, target_date in enumerate(future_dates):
        payload = {
            "prices": [
                {"code": code, "date": target_date, "close": closes[index]}
                for code, closes in prices_by_code.items()
            ]
        }
        (raw_dir / f"prices_{target_date}.json").write_text(json.dumps(payload), encoding="utf-8")

    save_screening_results(
        config_copy,
        tmp_path,
        {
            "date": base_date,
            "candidates": [
                {"code": "2001", "name": "Selected Machinery", "close": 100, "sector_name": "機械"},
                {"code": "2002", "name": "Rejected Machinery", "close": 100, "sector_name": "機械"},
                {"code": "2003", "name": "Selected Retail", "close": 100, "sector_name": "小売業"},
                {"code": "2004", "name": "Rejected Retail", "close": 100, "sector_name": "小売業"},
            ],
        },
    )
    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": base_date,
            "scores": [
                {"code": "2001", "name": "Selected Machinery", "rank": 1, "total_score": 82, "selected": True, "sector_name": "機械"},
                {"code": "2002", "name": "Rejected Machinery", "rank": 2, "total_score": 78, "selected": False, "sector_name": "機械", "rejected_reason": "上限超過"},
                {"code": "2003", "name": "Selected Retail", "rank": 3, "total_score": 76, "selected": True, "sector_name": "小売業"},
                {"code": "2004", "name": "Rejected Retail", "rank": 4, "total_score": 72, "selected": False, "sector_name": "小売業", "rejected_reason": "上限超過"},
            ],
        },
    )

    analysis = build_selection_quality_analysis(config_copy, tmp_path)
    sector_lift = analysis["sector_lift_analysis"]
    sector_by_name = {item["sector"]: item for item in sector_lift["sectors"]}

    assert sector_by_name["機械"]["selected_avg10d"] == 0.2
    assert sector_by_name["機械"]["rejected_avg10d"] == 0.05
    assert sector_by_name["機械"]["lift"] == 0.15
    assert sector_by_name["機械"]["selected_count"] == 1
    assert sector_by_name["機械"]["rejected_count"] == 1
    assert sector_by_name["機械"]["confidence"] == "reference"
    assert sector_by_name["機械"]["sample_note"] == "参考扱い: selected/rejectedのいずれかが10件未満"
    assert sector_by_name["小売業"]["lift"] == -0.15

    assert sector_lift["positive_sector_filters"][0]["filter"] == "sector = 機械"
    assert sector_lift["positive_sector_filters"][0]["confidence"] == "reference"
    assert sector_lift["negative_sector_filters"][0]["filter"] == "sector = 小売業"
    assert sector_lift["negative_sector_filters"][0]["confidence"] == "reference"

    markdown = render_selection_quality_markdown(analysis)
    assert "Sector Lift Analysis" in markdown
    assert "Positive Sector Filters" in markdown
    assert "Negative Sector Filters" in markdown
    assert "sector = 機械" in markdown
    assert "参考扱い" in markdown


def test_low_score_deep_analysis_finds_rescue_rule_candidates(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    base_date = "2026-03-01"
    future_dates = [f"2026-03-{day:02d}" for day in range(2, 12)]
    prices_by_code = {
        "3001": [102, 104, 106, 108, 110, 112, 114, 116, 118, 120],
        "3002": [101, 103, 105, 107, 109, 111, 112, 113, 114, 115],
        "3003": [99, 98, 97, 96, 95, 94, 93, 92, 91, 90],
        "3004": [103, 106, 109, 112, 115, 118, 121, 124, 127, 130],
    }
    for index, target_date in enumerate(future_dates):
        payload = {
            "prices": [
                {"code": code, "date": target_date, "close": closes[index]}
                for code, closes in prices_by_code.items()
            ]
        }
        (raw_dir / f"prices_{target_date}.json").write_text(json.dumps(payload), encoding="utf-8")

    save_screening_results(
        config_copy,
        tmp_path,
        {
            "date": base_date,
            "candidates": [
                {"code": "3001", "name": "Low Score Breakout A", "close": 100, "rsi": 55, "volume_ratio": 3.2, "sector_name": "機械"},
                {"code": "3002", "name": "Low Score Breakout B", "close": 100, "rsi": 58, "volume_ratio": 3.5, "sector_name": "機械"},
                {"code": "3003", "name": "Low Score Weak", "close": 100, "rsi": 72, "volume_ratio": 1.5, "sector_name": "小売業"},
                {"code": "3004", "name": "High Score Outside", "close": 100, "rsi": 54, "volume_ratio": 3.8, "sector_name": "機械"},
            ],
        },
    )
    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": base_date,
            "scores": [
                {
                    "code": "3001",
                    "name": "Low Score Breakout A",
                    "rank": 1,
                    "total_score": 43,
                    "selected": True,
                    "market_regime": "neutral",
                    "sector_name": "機械",
                    "candlestick_signals": ["volume_confirmed_breakout"],
                },
                {
                    "code": "3002",
                    "name": "Low Score Breakout B",
                    "rank": 2,
                    "total_score": 41,
                    "selected": False,
                    "market_regime": "risk_on",
                    "sector_name": "機械",
                    "candlestick_signals": ["volume_confirmed_breakout", "long_lower_shadow_support"],
                    "rejected_reason": "低スコア例外採用枠超過",
                },
                {
                    "code": "3003",
                    "name": "Low Score Weak",
                    "rank": 3,
                    "total_score": 42,
                    "selected": False,
                    "market_regime": "risk_off",
                    "sector_name": "小売業",
                    "candlestick_signals": ["upper_shadow"],
                    "rejected_reason": "低スコア例外条件不足",
                },
                {
                    "code": "3004",
                    "name": "High Score Outside",
                    "rank": 4,
                    "total_score": 47,
                    "selected": True,
                    "market_regime": "neutral",
                    "sector_name": "機械",
                    "candlestick_signals": ["volume_confirmed_breakout"],
                },
            ],
        },
    )

    analysis = build_selection_quality_analysis(config_copy, tmp_path)
    low_score = analysis["low_score_deep_analysis"]

    assert low_score["score_range"] == {"min": 40, "max": 44}
    assert low_score["low_score_count"] == 3
    assert low_score["winner_count"] == 2
    assert low_score["loser_count"] == 1
    assert [item["code"] for item in low_score["winners"]] == ["3001", "3002"]
    assert [item["code"] for item in low_score["losers"]] == ["3003"]
    separation_by_feature_value = {
        (item["feature"], item["value"]): item
        for item in low_score["feature_separation"]
    }
    assert separation_by_feature_value[("volume_ratio", "3+")] == {
        "feature": "volume_ratio",
        "value": "3+",
        "winner_count": 2,
        "loser_count": 0,
        "winner_share": 1.0,
        "loser_share": 0.0,
        "separation": 1.0,
    }
    assert separation_by_feature_value[("volume_confirmed_breakout", "yes")]["separation"] == 1.0
    assert separation_by_feature_value[("long_lower_shadow_support", "yes")]["winner_count"] == 1
    assert all(item["code"] != "3004" for item in low_score["winners"] + low_score["losers"])
    assert low_score["candidate_rescue_rules"][0]["rule"] == (
        "total_score 40-44 でも volume_ratio >= 3 かつ volume_confirmed_breakout なら採用候補"
    )

    markdown = render_selection_quality_markdown(analysis)
    assert "Low Score Deep Analysis" in markdown
    assert "Winners in 40-44" in markdown
    assert "Losers in 40-44" in markdown
    assert "Features with highest separation" in markdown
    assert "Candidate Rescue Rules" in markdown
