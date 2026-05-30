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
    assert "Selection Lift Deep Analysis" in markdown
    assert "Positive Lift Features" in markdown
    assert "Negative Lift Features" in markdown
    assert "Candidate New Rules" in markdown
    assert "参考扱い" in markdown
    assert "Missed Opportunity by Reason" in markdown
    assert "False Rejection Check" in markdown
    assert "Top Missed Opportunities" in markdown
    assert "Top False Positives" in markdown
