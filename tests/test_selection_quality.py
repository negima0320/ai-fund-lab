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
        {"code": "1001", "name": "Selected Up", "close": 100},
        {"code": "1002", "name": "Missed Winner", "close": 100},
        {"code": "1003", "name": "False Positive", "close": 100},
        {"code": "1004", "name": "Screen Only", "close": 100},
        {"code": "1005", "name": "Correct Reject", "close": 100},
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
                {"code": "1001", "name": "Selected Up", "rank": 1, "total_score": 80, "selected": True},
                {"code": "1002", "name": "Missed Winner", "rank": 2, "total_score": 78, "selected": False, "rejected_reason": "上限超過"},
                {"code": "1003", "name": "False Positive", "rank": 3, "total_score": 76, "selected": True},
                {"code": "1005", "name": "Correct Reject", "rank": 4, "total_score": 60, "selected": False, "rejected_reason": "出来高倍率不足のため新規買付見送り"},
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
    assert "Missed Opportunity by Reason" in markdown
    assert "False Rejection Check" in markdown
    assert "Top Missed Opportunities" in markdown
    assert "Top False Positives" in markdown
