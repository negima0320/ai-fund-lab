from __future__ import annotations

import json
import sqlite3
from copy import deepcopy

from db import get_database_path, initialize_database, save_market_context, save_scoring_results, save_trades
from feature_analysis import (
    _backtest_integrity_audits,
    _capital_utilization_audit,
    _compounding_capital_flow_audit,
    _market_section_performance_audit,
    _market_section_performance_audit_lines,
    _merge_scoring_rows_with_processed_scores,
    _monthly_performance_audit,
    _price_band_affordability_audit,
    _standard_ranking_input_audit,
    _standard_scoring_funnel_audit,
    build_feature_analysis,
    render_feature_analysis_markdown,
)
from paper_trade import execute_real_data_paper_trade, initial_live_paper_state
from profile_loader import load_profile


def test_capital_utilization_audit_reports_target_exposure_blockers(config_copy: dict, tmp_path) -> None:
    profile_id = config_copy["profile_id"]
    config_copy["capital_utilization_policy"] = {
        "enabled": True,
        "target_exposure": 0.9,
        "min_cash_buffer": 50000,
        "buy_lot_size": 100,
        "max_position_value_rate": 0.5,
        "buy_as_much_as_possible": True,
    }
    log_dir = tmp_path / "logs" / "backtests" / profile_id / "2026-01-01_to_2026-01-03"
    log_dir.mkdir(parents=True)
    (log_dir / "summary.csv").write_text(
        "date,total_assets,cash,positions_value,open_positions_count\n"
        "2026-01-01,1000000,800000,200000,1\n"
        "2026-01-02,1000000,550000,450000,2\n"
        "2026-01-03,1000000,100000,900000,3\n",
        encoding="utf-8",
    )
    backtest_summary = {
        "all_trades": [
            {"action": "BUY", "entry_date": "2026-01-02", "amount": 450000, "shares": 100, "entry_price": 4500, "allocation_reason": "capital_utilization_policy"},
            {"action": "SKIP_BUY", "entry_date": "2026-01-02", "skipped_reason": "target_exposure_limit"},
            {"action": "SKIP_BUY", "entry_date": "2026-01-03", "skipped_reason": "insufficient_available_cash"},
            {"action": "SKIP_BUY", "entry_date": "2026-01-03", "skipped_reason": "round_lot_unaffordable"},
            {"action": "NO_BUY", "entry_date": "2026-01-01", "reason": "no candidates"},
        ]
    }

    audit = _capital_utilization_audit(
        tmp_path,
        profile_id,
        "2026-01-01",
        "2026-01-03",
        backtest_summary,
        config_copy,
        [
            {"selected": True, "close": 2500},
            {"selected": True, "round_lot_amount": 450000},
            {"selected": False, "close": 9000},
        ],
    )

    assert audit["target_exposure_gap_average"] == 0.3833
    assert audit["target_exposure"] == 0.9
    assert audit["max_position_value_rate"] == 0.5
    assert audit["average_round_lot_amount"] == 350000
    assert audit["median_round_lot_amount"] == 350000
    assert audit["affordable_under_300k_count"] == 1
    assert audit["affordable_under_400k_count"] == 1
    assert audit["affordable_under_500k_count"] == 2
    assert audit["selected_round_lot_amount_breakdown"]["<=300k"] == 1
    assert audit["selected_round_lot_amount_breakdown"]["400k-500k"] == 1
    assert audit["bought_round_lot_amount_breakdown"]["400k-500k"] == 1
    assert audit["target_exposure_blocked_reason_breakdown"]["target_exposure_limit"] == 1
    assert audit["affordable_selected_count"] == 2
    assert audit["unaffordable_selected_count"] == 2
    assert audit["cash_after_buy_average"] == 550000
    assert audit["min_cash_buffer_hit_count"] == 1
    assert audit["no_candidate_days"] == 1
    assert audit["no_affordable_candidate_days"] == 1


def test_monthly_performance_audit_summarizes_returns_and_streaks(tmp_path) -> None:
    profile_id = "monthly_profile"
    log_dir = tmp_path / "logs" / "backtests" / profile_id / "2026-01-01_to_2026-03-31"
    log_dir.mkdir(parents=True)
    (log_dir / "summary.csv").write_text(
        "date,total_assets,max_drawdown\n"
        "2026-01-05,1000000,0\n"
        "2026-01-30,1100000,-0.02\n"
        "2026-02-02,1100000,-0.01\n"
        "2026-02-27,1045000,-0.06\n"
        "2026-03-02,1045000,-0.03\n"
        "2026-03-31,1045000,-0.03\n",
        encoding="utf-8",
    )
    (log_dir / "trades.csv").write_text(
        "action,entry_date,exit_date,gross_profit,profit,result\n"
        "SELL,2026-01-10,2026-01-20,50000,40000,WIN\n"
        "SELL,2026-01-21,2026-01-28,-10000,-10000,LOSS\n"
        "SELL,2026-02-10,2026-02-18,-20000,-20000,LOSS\n",
        encoding="utf-8",
    )
    backtest_summary = {
        "all_trades": [
            {"action": "BUY", "entry_date": "2026-01-10"},
            {"action": "SELL", "exit_date": "2026-01-20"},
            {"action": "BUY", "entry_date": "2026-02-10"},
            {"action": "SELL", "exit_date": "2026-02-18"},
        ]
    }

    audit = _monthly_performance_audit(tmp_path, profile_id, "2026-01-01", "2026-03-31", backtest_summary)

    summary = audit["summary"]
    assert summary["total_months"] == 3
    assert summary["winning_months"] == 1
    assert summary["losing_months"] == 1
    assert summary["flat_months"] == 1
    assert summary["monthly_win_rate"] == 0.3333
    assert summary["best_month"] == "2026-01"
    assert summary["worst_month"] == "2026-02"
    assert summary["max_consecutive_winning_months"] == 1
    assert summary["max_consecutive_losing_months"] == 1
    assert audit["months"][0]["trade_count"] == 2
    assert audit["months"][0]["buy_trade_count"] == 1
    assert audit["months"][0]["sell_trade_count"] == 1
    assert audit["months"][0]["profit_factor"] == 5.0
    rendered = render_feature_analysis_markdown(
        {
            "profile_id": profile_id,
            "profile_name": profile_id,
            "closed_trade_count": 0,
            "monthly_performance_audit": audit,
        }
    )
    assert "## Monthly Performance Audit" in rendered
    assert "| 2026-01 |" in rendered


def test_market_section_performance_audit_summarizes_market_sections(config_copy: dict) -> None:
    scoring_rows = [
        {"date": "2026-01-05", "code": "1001", "market_section": "TSEPrime", "selected": True, "close": 1000, "total_score": 52, "rank": 1},
        {"date": "2026-01-05", "code": "1002", "name": "Standard Winner", "market_section": "TSEStandard", "selected": True, "close": 2000, "total_score": 51, "rank": 2},
        {"date": "2026-01-05", "code": "1004", "name": "Standard Below", "market_section": "TSEStandard", "selected": False, "close": 1500, "total_score": 44, "rank": 9},
        {"code": "1003", "market_section": "TSEGrowth", "selected": False, "close": 3000},
    ]
    backtest_summary = {
        "all_trades": [
            {"action": "BUY", "code": "1001", "market_section": "TSEPrime", "entry_price": 1000, "shares": 100},
            {"action": "BUY", "code": "1002", "market_section": "TSEStandard", "entry_price": 2000, "shares": 100},
            {"action": "SELL", "code": "1001", "market_section": "TSEPrime", "gross_profit": 10000, "profit": 10000, "result": "WIN", "exit_date": "2026-01-10"},
            {"action": "SELL", "code": "1002", "market_section": "TSEStandard", "gross_profit": -5000, "profit": -5000, "result": "LOSS", "exit_date": "2026-01-10"},
        ]
    }

    audit = _market_section_performance_audit(
        backtest_summary,
        config_copy,
        scoring_rows,
        market_filter_audit={
            "candidate_market_breakdown_before_filter": {"Prime": 1, "Standard": 4, "Growth": 1, "Unknown": 0},
            "candidate_market_breakdown_after_filter": {"Prime": 1, "Standard": 3, "Growth": 1, "Unknown": 0},
            "candidate_market_breakdown_after_screening": {"Prime": 1, "Standard": 2, "Growth": 1, "Unknown": 0},
            "excluded_market_breakdown": {"Prime": 0, "Standard": 1, "Growth": 0, "Unknown": 0},
            "screening_excluded_market_breakdown": {"Prime": 0, "Standard": 1, "Growth": 0, "Unknown": 0},
        },
    )
    lines = _market_section_performance_audit_lines(audit)

    assert audit["candidate_count_by_market"]["Prime"] == 1
    assert audit["candidate_count_by_market"]["Standard"] == 2
    assert audit["candidate_count_by_market"]["Growth"] == 1
    assert audit["selected_count_by_market"]["Growth"] == 0
    assert audit["buy_trade_count_by_market"]["Prime"] == 1
    assert audit["gross_profit_by_market"]["Prime"] == 10000
    assert audit["gross_profit_by_market"]["Standard"] == -5000
    assert audit["average_round_lot_amount_by_market"]["Standard"] == 175000
    standard = audit["standard_funnel_audit"]
    assert standard["raw"] == 4
    assert standard["after_market_filter"] == 3
    assert standard["after_screening"] == 2
    assert standard["after_scoring"] == 2
    assert standard["score_assigned"] == 2
    assert standard["above_min_score"] == 1
    assert standard["selected"] == 1
    assert standard["top_20_standard_by_total_score"][0]["code"] == "1002"
    rendered = render_feature_analysis_markdown(
        {
            "profile_id": "p",
            "profile_name": "p",
            "closed_trade_count": 0,
            "market_section_performance_audit": audit,
        }
    )
    assert "## Standard Funnel Audit" in rendered
    assert "- above_min_score: 1" in rendered
    assert "| 2026-01-05 | 1002 | Standard Winner | Standard | 51.00 | 2.00 | true |" in rendered
    assert any("Standard" in line for line in lines)


def test_save_scoring_results_persists_standard_market_section(config_copy: dict, tmp_path) -> None:
    config_copy["profile_id"] = "rookie_dealer_02_v2_47"
    initialize_database(config_copy, tmp_path)
    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": "2026-01-05",
            "scores": [
                {
                    "code": "2001",
                    "name": "Standard Scored",
                    "section": "TSEStandard",
                    "market_section": "TSEStandard",
                    "listing_market": "TSEStandard",
                    "rank": 3,
                    "total_score": 48,
                    "technical_score": 44,
                    "confidence": 0.8,
                    "selected": False,
                    "reason": "通常基準45点には届かないため落選",
                }
            ],
        },
    )

    with sqlite3.connect(get_database_path(config_copy, tmp_path)) as connection:
        row = connection.execute(
            "SELECT section, market_section, listing_market FROM scoring_results WHERE profile_id = ? AND code = ?",
            ("rookie_dealer_02_v2_47", "2001"),
        ).fetchone()

    assert row == ("TSEStandard", "TSEStandard", "TSEStandard")


def test_processed_scored_candidates_fill_standard_scoring_audit(config_copy: dict, tmp_path) -> None:
    profile_id = "rookie_dealer_02_v2_47"
    processed_dir = tmp_path / "data" / "processed" / profile_id
    processed_dir.mkdir(parents=True)
    (processed_dir / "scored_candidates_2026-01-05.json").write_text(
        json.dumps(
            {
                "date": "2026-01-05",
                "scores": [
                    {
                        "date": "2026-01-05",
                        "code": "60470",
                        "name": "Gunosy",
                        "market_section": "TSEStandard",
                        "section": "TSEStandard",
                        "listing_market": "TSEStandard",
                        "rank": 49,
                        "total_score": 23.0,
                        "selected": False,
                        "rejected_reason": "below_min_score",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    merged = _merge_scoring_rows_with_processed_scores(
        [
            {
                "date": "2026-01-05",
                "code": "60470",
                "name": "Gunosy",
                "market_section": None,
                "rank": 49,
                "total_score": 23.0,
                "selected": False,
            }
        ],
        tmp_path,
        profile_id,
        "2026-01-01",
        "2026-01-31",
    )

    audit = _market_section_performance_audit(
        {"all_trades": []},
        config_copy,
        merged,
        market_filter_audit={
            "candidate_market_breakdown_before_filter": {"Standard": 1},
            "candidate_market_breakdown_after_filter": {"Standard": 1},
            "candidate_market_breakdown_after_screening": {"Standard": 1},
        },
    )

    assert len(merged) == 1
    assert merged[0]["market_section"] == "TSEStandard"
    assert audit["scored_candidate_audit"]["scored_count_by_market"]["Standard"] == 1
    assert audit["standard_funnel_audit"]["after_scoring"] == 1
    assert audit["standard_funnel_audit"]["score_assigned"] == 1
    assert audit["standard_funnel_audit"]["top_20_standard_by_total_score"][0]["code"] == "60470"


def test_standard_funnel_uses_market_specific_min_score(config_copy: dict) -> None:
    config_copy["selection"]["min_score"] = 45
    config_copy["selection"]["market_min_score_overrides"] = {"TSEStandard": 35}
    audit = _market_section_performance_audit(
        {"all_trades": []},
        config_copy,
        [
            {
                "date": "2026-01-05",
                "code": "2001",
                "name": "Standard Candidate",
                "market_section": "TSEStandard",
                "total_score": 37,
                "selected": True,
            }
        ],
        market_filter_audit={
            "candidate_market_breakdown_before_filter": {"Standard": 1},
            "candidate_market_breakdown_after_filter": {"Standard": 1},
            "candidate_market_breakdown_after_screening": {"Standard": 1},
        },
    )

    assert audit["standard_funnel_audit"]["min_score"] == 35
    assert audit["standard_funnel_audit"]["above_min_score"] == 1
    assert audit["standard_funnel_audit"]["selected"] == 1


def test_standard_selection_audit_reports_scored_but_not_selected_reason(config_copy: dict) -> None:
    config_copy["selection"]["min_score"] = 45
    config_copy["selection"]["max_selected"] = 10
    config_copy["selection"]["market_min_score_overrides"] = {"TSEStandard": 35}
    config_copy.setdefault("market_filter", {})["allowed_sections"] = ["TSEPrime", "TSEStandard"]
    audit = _market_section_performance_audit(
        {"all_trades": []},
        config_copy,
        [
            {
                "date": "2026-01-05",
                "code": "2001",
                "name": "Standard Above Min",
                "market_section": "TSEStandard",
                "total_score": 37,
                "rank": 12,
                "selected": False,
                "rejected_reason": "上位候補だが最大採用数を超えたため落選",
            },
            {
                "date": "2026-01-05",
                "code": "2002",
                "name": "Standard Outside Quota",
                "market_section": "TSEStandard",
                "total_score": 36,
                "rank": 13,
                "selected": False,
                "rejected_reason": "outside_standard_quota",
            },
            {
                "date": "2026-01-05",
                "code": "2003",
                "name": "Standard Below Min",
                "market_section": "TSEStandard",
                "total_score": 32,
                "rank": 14,
                "selected": False,
                "rejected_reason": "通常基準35点には届かないため落選",
            },
        ],
        market_filter_audit={
            "candidate_market_breakdown_before_filter": {"Standard": 3},
            "candidate_market_breakdown_after_filter": {"Standard": 3},
            "candidate_market_breakdown_after_screening": {"Standard": 3},
        },
    )

    selection_audit = audit["standard_selection_audit"]
    assert audit["standard_funnel_audit"]["above_min_score"] == 2
    assert audit["standard_funnel_audit"]["selected"] == 0
    assert selection_audit["standard_above_min_score_count"] == 2
    assert selection_audit["standard_selected_count"] == 0
    assert selection_audit["above_min_not_selected_count"] == 2
    assert selection_audit["selection_exclusion_reason_counts"]["outside_selection_rank"] == 1
    assert selection_audit["selection_exclusion_reason_counts"]["outside_standard_quota"] == 1
    assert selection_audit["selection_exclusion_reason_counts"]["below_market_min_score"] == 1
    rendered = render_feature_analysis_markdown(
        {
            "profile_id": "p",
            "profile_name": "p",
            "closed_trade_count": 0,
            "market_section_performance_audit": audit,
        }
    )
    assert "## Standard Selection Audit" in rendered
    assert "| 2026-01-05 | 2001 | Standard Above Min | Standard | 37.00 | 12.00 | 35.00 | false | outside_selection_rank |" in rendered


def test_standard_scoring_funnel_audit_reports_candidates_missing_from_scored_output(
    config_copy: dict,
    tmp_path,
) -> None:
    profile_id = "rookie_dealer_02_v2_47"
    profile_dir = tmp_path / "data" / "processed" / profile_id
    profile_dir.mkdir(parents=True)
    (profile_dir / "candidates_2026-01-05.json").write_text(
        json.dumps(
            {
                "date": "2026-01-05",
                "candidates": [
                    {
                        "code": "1001",
                        "name": "Prime Scored",
                        "market_section": "TSEPrime",
                        "close": 1000,
                        "volume_ratio": 2.5,
                        "rsi": 55,
                    },
                    {
                        "code": "1002",
                        "name": "Standard Omitted",
                        "market_section": "TSEStandard",
                        "close": 1200,
                        "volume_ratio": 2.8,
                        "rsi": 58,
                    },
                    {
                        "code": "1003",
                        "name": "Standard Missing Indicator",
                        "market_section": "TSEStandard",
                        "volume_ratio": 3.1,
                        "rsi": 62,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (profile_dir / "scored_candidates_2026-01-05.json").write_text(
        json.dumps(
            {
                "date": "2026-01-05",
                "storage_mode": "compact",
                "storage_omitted_score_count": 2,
                "scores": [
                    {
                        "code": "1001",
                        "name": "Prime Scored",
                        "market_section": "TSEPrime",
                        "total_score": 52,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit = _standard_scoring_funnel_audit(tmp_path, profile_id, "2026-01-01", "2026-01-31", config_copy)
    rendered = render_feature_analysis_markdown(
        {
            "profile_id": profile_id,
            "profile_name": profile_id,
            "closed_trade_count": 0,
            "standard_scoring_funnel_audit": audit,
        }
    )

    assert audit["after_screening_count"] == 2
    assert audit["after_scoring_count"] == 0
    assert audit["after_scoring_input_filter_count"] == 2
    assert audit["scoring_input_exclusion_reasons"]["Standard"]["storage_pruned_rejected_score"] == 2
    assert audit["standard_scoring_excluded_samples"][0]["code"] == "1002"
    assert "## Standard Scoring Funnel Audit" in rendered
    assert "| Standard | storage_pruned_rejected_score | 2 |" in rendered
    assert "| 2026-01-05 | 1002 | Standard Omitted | Standard | storage_pruned_rejected_score |" in rendered


def test_standard_scoring_funnel_audit_reports_missing_scored_payload(tmp_path, config_copy: dict) -> None:
    profile_id = "rookie_dealer_02_v2_47"
    profile_dir = tmp_path / "data" / "processed" / profile_id
    profile_dir.mkdir(parents=True)
    (profile_dir / "candidates_2026-01-06.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "code": "2001",
                        "name": "Standard No Score File",
                        "market_section": "TSEStandard",
                        "close": 900,
                        "volume_ratio": 2.2,
                        "rsi": 50,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit = _standard_scoring_funnel_audit(tmp_path, profile_id, "2026-01-01", "2026-01-31", config_copy)

    assert audit["after_screening_count"] == 1
    assert audit["after_scoring_count"] == 0
    assert audit["scored_payload_missing_count"] == 1
    assert audit["scoring_input_exclusion_reasons"]["Standard"]["scored_candidates_missing"] == 1


def test_standard_ranking_input_audit_breaks_down_not_in_ranking_universe(
    tmp_path,
    config_copy: dict,
) -> None:
    profile_id = "rookie_dealer_02_v2_47"
    profile_dir = tmp_path / "data" / "processed" / profile_id
    profile_dir.mkdir(parents=True)
    config_copy["relative_strength"] = {"enabled": False}
    (profile_dir / "candidates_2026-01-05.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "date": "2026-01-05",
                        "code": "3001",
                        "name": "Standard Ready",
                        "market_section": "TSEStandard",
                        "close": 1000,
                        "volume": 200000,
                        "turnover_value": 200000000,
                        "volume_ratio": 2.5,
                        "ma5": 990,
                        "ma25": 970,
                        "rsi": 55,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (profile_dir / "scored_candidates_2026-01-05.json").write_text(
        json.dumps({"candidate_count": 10, "scores": []}, ensure_ascii=False),
        encoding="utf-8",
    )

    audit = _standard_ranking_input_audit(tmp_path, profile_id, "2026-01-01", "2026-01-31", config_copy)
    markdown = render_feature_analysis_markdown(
        {
            "profile_id": profile_id,
            "profile_name": profile_id,
            "closed_trade_count": 0,
            "standard_ranking_input_audit": audit,
        }
    )

    assert audit["after_screening_count"] == 1
    assert audit["after_scoring_count"] == 0
    assert audit["exclusion_reasons"]["Standard"]["not_in_ranking_universe"] == 1
    row = audit["standard_ranking_input_excluded_rows"][0]
    assert row["trading_value"] == 200000000
    assert row["ranking_exclusion_reason"] == "not_in_ranking_universe"
    assert "## Standard Ranking Input Audit" in markdown
    assert "| Standard | not_in_ranking_universe | 1 |" in markdown
    assert "| 2026-01-05 | 3001 | Standard Ready | Standard | 1000.00 | 200000.00 | 200000000.00 |" in markdown


def test_standard_ranking_input_audit_reports_missing_indicator(tmp_path, config_copy: dict) -> None:
    profile_id = "rookie_dealer_02_v2_47"
    profile_dir = tmp_path / "data" / "processed" / profile_id
    profile_dir.mkdir(parents=True)
    (profile_dir / "candidates_2026-01-06.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "date": "2026-01-06",
                        "code": "3002",
                        "name": "Standard Missing MA",
                        "market_section": "TSEStandard",
                        "close": 1200,
                        "volume": 100000,
                        "volume_ratio": 2.1,
                        "rsi": 52,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (profile_dir / "scored_candidates_2026-01-06.json").write_text(
        json.dumps({"candidate_count": 1, "scores": []}, ensure_ascii=False),
        encoding="utf-8",
    )

    audit = _standard_ranking_input_audit(tmp_path, profile_id, "2026-01-01", "2026-01-31", config_copy)

    assert audit["exclusion_reasons"]["Standard"]["missing_indicator"] == 1
    assert audit["standard_ranking_input_excluded_rows"][0]["ranking_exclusion_reason"] == "missing_indicator"


def test_market_section_performance_audit_fills_unknown_from_master(config_copy: dict, tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "listed_stocks_jquants.json").write_text(
        json.dumps(
            {
                "stocks": [
                    {"code": "2001", "MktNm": "スタンダード"},
                    {"code": "3001", "Mkt": "0113"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scoring_rows = [
        {"code": "2001", "selected": True, "close": 2000},
        {"code": "3001", "selected": False, "close": 3000},
    ]
    backtest_summary = {
        "all_trades": [
            {"action": "BUY", "code": "2001", "entry_price": 2000, "shares": 100},
            {"action": "SELL", "code": "2001", "gross_profit": 5000, "profit": 5000, "result": "WIN"},
        ]
    }

    audit = _market_section_performance_audit(
        backtest_summary,
        config_copy,
        scoring_rows,
        root=tmp_path,
        market_filter_audit={
            "allowed_sections": ["TSEPrime", "TSEStandard", "TSEGrowth"],
            "candidate_market_breakdown_before_filter": {"Prime": 0, "Standard": 1, "Growth": 1, "Unknown": 0},
            "candidate_market_breakdown_after_filter": {"Prime": 0, "Standard": 1, "Growth": 1, "Unknown": 0},
            "excluded_market_breakdown": {"Prime": 0, "Standard": 0, "Growth": 0, "Unknown": 0},
            "unknown_market_count": 0,
            "market_section_lookup_source": {"master": 2, "row": 0, "unknown": 0},
            "daily_breakdown": [
                {
                    "date": "2026-01-05",
                    "raw_candidate_count_by_market": {"Prime": 0, "Standard": 1, "Growth": 1, "Unknown": 0},
                    "after_market_filter_candidate_count_by_market": {"Prime": 0, "Standard": 1, "Growth": 1, "Unknown": 0},
                    "excluded_count_by_market": {"Prime": 0, "Standard": 0, "Growth": 0, "Unknown": 0},
                    "market_section_lookup_source": {"master": 2},
                }
            ],
        },
    )

    assert audit["market_section_master_lookup_count"] == 2
    assert audit["candidate_count_by_market"]["Standard"] == 1
    assert audit["candidate_count_by_market"]["Growth"] == 1
    assert audit["candidate_count_by_market"]["Unknown"] == 0
    assert audit["buy_trade_count_by_market"]["Standard"] == 1
    assert audit["candidate_universe_audit"]["raw_candidate_count_by_market"]["Standard"] == 1
    assert audit["scored_candidate_audit"]["scored_count_by_market"]["Growth"] == 1
    assert audit["selected_candidate_audit"]["selected_count_by_market"]["Standard"] == 1
    assert audit["trade_market_audit"]["buy_trade_count_by_market"]["Standard"] == 1


def test_capital_utilization_audit_reports_round_lot_amount_by_market(config_copy: dict, tmp_path) -> None:
    profile_id = config_copy["profile_id"]
    log_dir = tmp_path / "logs" / "backtests" / profile_id / "2026-01-01_to_2026-01-02"
    log_dir.mkdir(parents=True)
    (log_dir / "summary.csv").write_text(
        "date,total_assets,cash,positions_value,open_positions_count\n"
        "2026-01-01,1000000,1000000,0,0\n"
        "2026-01-02,1000000,700000,300000,2\n",
        encoding="utf-8",
    )
    scoring_rows = [
        {"code": "1001", "market_section": "TSEPrime", "selected": True, "close": 1000},
        {"code": "1002", "market_section": "TSEStandard", "selected": True, "close": 2000},
    ]
    backtest_summary = {
        "all_trades": [
            {"action": "BUY", "entry_date": "2026-01-02", "code": "1001", "market_section": "TSEPrime", "entry_price": 1000, "shares": 100, "amount": 100000},
            {"action": "BUY", "entry_date": "2026-01-02", "code": "1002", "market_section": "TSEStandard", "entry_price": 2000, "shares": 100, "amount": 200000},
        ]
    }

    audit = _capital_utilization_audit(
        tmp_path,
        profile_id,
        "2026-01-01",
        "2026-01-02",
        backtest_summary,
        config_copy,
        scoring_rows,
    )

    assert audit["average_round_lot_amount_by_market"]["Prime"] == 100000
    assert audit["average_round_lot_amount_by_market"]["Standard"] == 200000
    assert audit["bought_round_lot_amount_by_market"]["Prime"] == 100000
    assert audit["bought_round_lot_amount_by_market"]["Standard"] == 200000


def test_price_band_affordability_audit_reports_penalized_rows(config_copy: dict) -> None:
    config_copy["capital_utilization_policy"] = {"buy_lot_size": 100}
    config_copy["affordability_filter"] = {
        "enabled": True,
        "preferred_round_lot_amount": 400000,
        "penalty_points": 3,
        "reason": "price_band_penalty",
    }
    scoring_rows = [
        {
            "date": "2026-01-02",
            "code": "1001",
            "name": "Affordable",
            "close": 3500,
            "selected": True,
            "price_band_penalty": 0,
            "total_score": 50,
        },
        {
            "date": "2026-01-02",
            "code": "1002",
            "name": "Expensive",
            "close": 5000,
            "selected": False,
            "price_band_penalty": 3,
            "total_score": 47,
        },
    ]
    backtest_summary = {"all_trades": [{"action": "BUY", "entry_price": 3500, "shares": 100}]}

    audit = _price_band_affordability_audit(config_copy, scoring_rows, backtest_summary)

    assert audit["enabled"] is True
    assert audit["preferred_round_lot_amount"] == 400000
    assert audit["penalty_points"] == 3
    assert audit["scored_count"] == 2
    assert audit["selected_count"] == 1
    assert audit["penalized_count"] == 1
    assert audit["average_round_lot_amount"] == 425000
    assert audit["selected_average_round_lot_amount"] == 350000
    assert audit["bought_average_round_lot_amount"] == 350000
    assert audit["scored_round_lot_amount_breakdown"]["300k-400k"] == 1
    assert audit["scored_round_lot_amount_breakdown"]["400k-500k"] == 1
    assert audit["sample_penalized_rows"][0]["code"] == "1002"
    assert audit["sample_penalized_rows"][0]["round_lot_amount"] == 500000


def test_compounding_capital_flow_audit_tracks_profit_reinvestment(config_copy: dict, tmp_path) -> None:
    profile_id = config_copy["profile_id"]
    config_copy["initial_capital"] = 1000000
    log_dir = tmp_path / "logs" / "backtests" / profile_id / "2026-01-01_to_2026-01-03"
    log_dir.mkdir(parents=True)
    (log_dir / "summary.csv").write_text(
        "date,total_assets,cash,positions_value,open_positions_count\n"
        "2026-01-01,1000000,700000,300000,1\n"
        "2026-01-02,1030000,630000,400000,1\n"
        "2026-01-03,1010000,710000,300000,1\n",
        encoding="utf-8",
    )
    backtest_summary = {
        "initial_capital": 1000000,
        "final_assets": 1010000,
        "net_cumulative_profit": 10000,
        "all_trades": [
            {"action": "BUY", "entry_date": "2026-01-01", "code": "1001", "amount": 300000},
            {"action": "SELL", "exit_date": "2026-01-02", "code": "1001", "amount": 330000, "profit": 30000},
            {"action": "BUY", "entry_date": "2026-01-02", "code": "1002", "amount": 400000},
            {"action": "SELL", "exit_date": "2026-01-03", "code": "1002", "amount": 380000, "profit": -20000},
            {"action": "BUY", "entry_date": "2026-01-03", "code": "1003", "amount": 300000},
        ],
    }

    audit = _compounding_capital_flow_audit(
        tmp_path,
        profile_id,
        "2026-01-01",
        "2026-01-03",
        backtest_summary,
        config_copy,
    )

    assert audit["capital_flow_status"] == "OK"
    assert audit["realized_profit_total"] == 10000
    assert audit["cash_end"] == 710000
    assert audit["total_buy_amount"] == 1000000
    assert audit["total_sell_amount"] == 710000
    assert audit["profit_reinvested_check"]["status"] == "OK"
    assert audit["profit_reinvested_check"]["subsequent_buy_count"] == 1
    assert audit["sell_cash_flow_issue_count"] == 0


def test_compounding_capital_flow_audit_tracks_loss_reduced_cash(config_copy: dict, tmp_path) -> None:
    profile_id = config_copy["profile_id"]
    config_copy["initial_capital"] = 1000000
    log_dir = tmp_path / "logs" / "backtests" / profile_id / "2026-01-01_to_2026-01-02"
    log_dir.mkdir(parents=True)
    (log_dir / "summary.csv").write_text(
        "date,total_assets,cash,positions_value,open_positions_count\n"
        "2026-01-01,1000000,600000,400000,1\n"
        "2026-01-02,980000,680000,300000,1\n",
        encoding="utf-8",
    )
    backtest_summary = {
        "initial_capital": 1000000,
        "final_assets": 980000,
        "net_cumulative_profit": -20000,
        "all_trades": [
            {"action": "BUY", "entry_date": "2026-01-01", "code": "1001", "amount": 400000},
            {"action": "SELL", "exit_date": "2026-01-02", "code": "1001", "amount": 380000, "profit": -20000},
            {"action": "BUY", "entry_date": "2026-01-02", "code": "1002", "amount": 300000},
        ],
    }

    audit = _compounding_capital_flow_audit(
        tmp_path,
        profile_id,
        "2026-01-01",
        "2026-01-02",
        backtest_summary,
        config_copy,
    )

    assert audit["capital_flow_status"] == "OK"
    assert audit["realized_profit_total"] == -20000
    assert audit["cash_end"] == 680000
    assert audit["profit_reinvested_check"]["status"] == "N/A"
    assert audit["sell_cash_flow_issue_count"] == 0


def test_compounding_capital_flow_audit_documents_net_profit_definition_without_warning(config_copy: dict, tmp_path) -> None:
    profile_id = config_copy["profile_id"]
    log_dir = tmp_path / "logs" / "backtests" / profile_id / "2026-01-01_to_2026-01-01"
    log_dir.mkdir(parents=True)
    (log_dir / "summary.csv").write_text(
        "date,total_assets,cash,positions_value,open_positions_count\n"
        "2026-01-01,1000000,1000000,0,0\n",
        encoding="utf-8",
    )
    backtest_summary = {
        "initial_capital": 1000000,
        "final_assets": 1000000,
        "net_cumulative_profit": 50000,
        "all_trades": [],
    }

    audit = _compounding_capital_flow_audit(
        tmp_path,
        profile_id,
        "2026-01-01",
        "2026-01-01",
        backtest_summary,
        config_copy,
    )

    assert audit["capital_flow_status"] == "OK"
    assert audit["final_assets_profit_match"] is True
    assert audit["asset_reconciliation"]["final_assets_minus_initial_plus_net_cumulative_profit"] == -50000
    assert "Period-level net P/L" in audit["net_cumulative_profit_definition"]


def test_compounding_capital_flow_audit_warns_on_asset_reconciliation_mismatch(config_copy: dict, tmp_path) -> None:
    profile_id = config_copy["profile_id"]
    log_dir = tmp_path / "logs" / "backtests" / profile_id / "2026-01-01_to_2026-01-01"
    log_dir.mkdir(parents=True)
    (log_dir / "summary.csv").write_text(
        "date,total_assets,cash,positions_value,open_positions_count\n"
        "2026-01-01,1000000,1000000,0,0\n",
        encoding="utf-8",
    )
    backtest_summary = {
        "initial_capital": 1000000,
        "final_assets": 1000000,
        "net_cumulative_profit": 50000,
        "realized_profit_total": 50000,
        "unrealized_profit_total": 0,
        "all_trades": [{"action": "SELL", "exit_date": "2026-01-01", "code": "1001", "amount": 100000, "profit": 50000}],
    }

    audit = _compounding_capital_flow_audit(
        tmp_path,
        profile_id,
        "2026-01-01",
        "2026-01-01",
        backtest_summary,
        config_copy,
    )

    assert audit["capital_flow_status"] == "WARNING"
    assert audit["final_assets_profit_match"] is False
    assert "realized_profit_total + unrealized_profit_total" in audit["capital_flow_warning_reason"]


def test_feature_analysis_repairs_stale_trade_without_selected_audit(config_copy: dict, tmp_path) -> None:
    profile_id = config_copy["profile_id"]
    log_dir = tmp_path / "logs" / "backtests" / profile_id / "2026-01-01_to_2026-03-06"
    log_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed" / profile_id
    processed_dir.mkdir(parents=True)
    (log_dir / "backtest_summary.json").write_text(
        json.dumps(
            {
                "all_trades": [
                    {
                        "action": "BUY",
                        "order_status": "FILLED",
                        "signal_date": "2026-01-05",
                        "entry_date": "2026-01-06",
                        "code": "67230",
                    }
                ],
                "backtest_result_integrity_audit": {
                    "result_integrity_status": "WARNING",
                    "trade_without_selected_count": 1,
                    "trade_without_selected_sample": ["2026-01-05|67230"],
                    "warnings": ["buy trade exists without selected candidate in the run period"],
                    "errors": [],
                    "integrity_warning_count": 1,
                    "integrity_error_count": 0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (processed_dir / "scored_candidates_2026-01-05.json").write_text(
        json.dumps(
            {
                "scores": [{"date": "2026-01-05", "code": "67230", "selected": True}],
                "selected": [{"date": "2026-01-05", "code": "67230", "selected": True}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audits = _backtest_integrity_audits(tmp_path, profile_id, "2026-01-01", "2026-03-06")
    result = audits["backtest_result_integrity_audit"]

    assert result["trade_without_selected_count"] == 0
    assert result["trade_without_selected_sample"] == []
    assert result["trade_without_selected_debug_sample"] == []
    assert result["result_integrity_status"] == "OK"
    assert result["warnings"] == []


def test_feature_analysis_treats_affordable_fallback_buy_as_selected(config_copy: dict, tmp_path) -> None:
    profile_id = config_copy["profile_id"]
    log_dir = tmp_path / "logs" / "backtests" / profile_id / "2026-01-01_to_2026-03-06"
    log_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed" / profile_id
    processed_dir.mkdir(parents=True)
    (log_dir / "backtest_summary.json").write_text(
        json.dumps(
            {
                "all_trades": [
                    {
                        "action": "BUY",
                        "order_status": "FILLED",
                        "signal_date": "2026-01-05",
                        "entry_date": "2026-01-06",
                        "code": "1002",
                        "name": "Fallback Buy",
                        "market_section": "Prime",
                        "total_score": 52,
                        "affordable_fallback_buy_selected": True,
                        "affordable_fallback_reason": "surplus_available_cash",
                    }
                ],
                "backtest_result_integrity_audit": {
                    "result_integrity_status": "WARNING",
                    "trade_without_selected_count": 1,
                    "trade_without_selected_sample": ["2026-01-05|1002"],
                    "warnings": ["buy trade exists without selected candidate in the run period"],
                    "errors": [],
                    "integrity_warning_count": 1,
                    "integrity_error_count": 0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (processed_dir / "scored_candidates_2026-01-05.json").write_text(
        json.dumps(
            {
                "scores": [{"date": "2026-01-05", "code": "1002", "selected": False, "market_section": "Prime"}],
                "selected": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audits = _backtest_integrity_audits(tmp_path, profile_id, "2026-01-01", "2026-03-06")
    result = audits["backtest_result_integrity_audit"]

    assert result["trade_without_selected_count"] == 0
    assert result["trade_without_selected_sample"] == []
    assert result["trade_without_selected_debug_sample"] == []
    assert result["result_integrity_status"] == "OK"
    assert result["warnings"] == []


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
    assert "API Field Usage Audit" in render_feature_analysis_markdown(analysis)
    assert analysis["api_field_usage_audit"]["adjusted_price_usage"]["ma_basis"].startswith("close")
    assert "Backtest Result Integrity Audit" in render_feature_analysis_markdown(analysis)
    assert "Market Filter Audit" in render_feature_analysis_markdown(analysis)
    assert "Score Integrity Audit" in render_feature_analysis_markdown(analysis)
    assert "Duplicated Signal Warning" in render_feature_analysis_markdown(analysis)
    assert "Relative Strength Analysis" in render_feature_analysis_markdown(analysis)
    assert "Relative Strength Effect Analysis" in render_feature_analysis_markdown(analysis)
    assert "relative_strength_5d帯別" in render_feature_analysis_markdown(analysis)
    assert "Relative Strength Debug" in render_feature_analysis_markdown(analysis)
    assert "technical_score average: 41.50" in render_feature_analysis_markdown(analysis)


def test_relative_strength_effect_analysis_uses_closed_trade_profit_and_baseline(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    config_copy["profile_id"] = "rookie_dealer_02_v2_6"
    config_copy["profile_name"] = "RS Test"
    config_copy.setdefault("features", {})["relative_strength"] = True
    config_copy.setdefault("scoring", {})["use_relative_strength_score"] = True
    initialize_database(config_copy, tmp_path)

    save_trades(
        config_copy,
        tmp_path,
        "2026-03-06",
        [
            {
                "trade_id": "target-new",
                "action": "SELL",
                "code": "1001",
                "name": "New RS",
                "signal_date": "2026-03-01",
                "entry_date": "2026-03-02",
                "exit_date": "2026-03-06",
                "profit": 12000,
                "net_profit": 10000,
                "profit_rate": 0.1,
                "net_profit_rate": 0.08,
                "result": "WIN",
                "order_status": "FILLED",
                "relative_strength_score": 10,
                "relative_strength_5d": 0.04,
                "relative_strength_10d": 0.06,
                "relative_strength_20d": 0.09,
                "score_components": {"relative_strength_score": 10},
            },
            {
                "trade_id": "target-common",
                "action": "SELL",
                "code": "1003",
                "name": "Common",
                "signal_date": "2026-03-01",
                "entry_date": "2026-03-02",
                "exit_date": "2026-03-06",
                "profit": -2000,
                "net_profit": -3000,
                "profit_rate": -0.02,
                "net_profit_rate": -0.03,
                "result": "LOSS",
                "order_status": "FILLED",
                "relative_strength_score": 0,
                "score_components": {"relative_strength_score": 0},
            },
        ],
    )
    baseline_config = deepcopy(config_copy)
    baseline_config["profile_id"] = "rookie_dealer_02_v2_1"
    baseline_config["profile_name"] = "Baseline"
    save_trades(
        baseline_config,
        tmp_path,
        "2026-03-06",
        [
            {
                "trade_id": "base-removed",
                "action": "SELL",
                "code": "1002",
                "name": "Removed",
                "signal_date": "2026-03-01",
                "entry_date": "2026-03-02",
                "exit_date": "2026-03-06",
                "profit": 5000,
                "net_profit": 4000,
                "profit_rate": 0.05,
                "net_profit_rate": 0.04,
                "result": "WIN",
                "order_status": "FILLED",
            },
            {
                "trade_id": "base-common",
                "action": "SELL",
                "code": "1003",
                "name": "Common",
                "signal_date": "2026-03-01",
                "entry_date": "2026-03-02",
                "exit_date": "2026-03-06",
                "profit": -2000,
                "net_profit": -3000,
                "profit_rate": -0.02,
                "net_profit_rate": -0.03,
                "result": "LOSS",
                "order_status": "FILLED",
            },
        ],
    )

    analysis = build_feature_analysis(config_copy, tmp_path)
    effect = analysis["relative_strength_effect_analysis"]
    buckets = {row["bucket"]: row for row in effect["buckets"]}
    comparison = effect["selected_vs_baseline"]

    assert buckets["relative_strength_score 10"]["total_profit"] == 10000
    assert buckets["relative_strength_score = 0"]["total_profit"] == -3000
    assert effect["top_selected_trades"][0]["code"] == "1001"
    assert comparison["newly_selected_count"] == 1
    assert comparison["removed_count"] == 1
    assert comparison["newly_selected_profit"] == 10000
    assert comparison["removed_profit_if_kept"] == 4000
    assert comparison["net_selection_effect_profit"] == 6000

    markdown = render_feature_analysis_markdown(analysis)
    assert "## Top Relative Strength Selected Trades" in markdown
    assert "## Relative Strength Selected vs Baseline" in markdown


def test_relative_strength_debug_outputs_distribution_and_benchmark_warning(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    config_copy.setdefault("features", {})["relative_strength"] = True
    config_copy.setdefault("scoring", {})["use_relative_strength_score"] = True
    initialize_database(config_copy, tmp_path)
    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-01",
            "scores": [
                {
                    "code": "1001",
                    "name": "Topix Strong",
                    "rank": 1,
                    "selected": True,
                    "benchmark_source": "topix",
                    "relative_strength_5d": 0.04,
                    "relative_strength_10d": 0.06,
                    "relative_strength_20d": 0.09,
                    "relative_strength_score": 10,
                    "topix_records_loaded": 35,
                    "topix_api_calls": 1,
                    "topix_cache_path": str(tmp_path / "data/cache/jquants/topix_prices/test.json"),
                    "relative_strength_feature_enabled": True,
                    "relative_strength_scoring_enabled": True,
                    "relative_strength_benchmark_provider_called": True,
                    "relative_strength_cache_exists": True,
                    "relative_strength_calculated": True,
                    "score_components": {"relative_strength_score": 10},
                },
                {
                    "code": "1002",
                    "name": "Prime Modest",
                    "rank": 2,
                    "selected": False,
                    "benchmark_source": "prime_average",
                    "relative_strength_5d": 0.01,
                    "relative_strength_10d": 0.02,
                    "relative_strength_20d": 0.03,
                    "relative_strength_score": 3,
                    "score_components": {"relative_strength_score": 3},
                },
                {
                    "code": "1003",
                    "name": "Missing",
                    "rank": 3,
                    "selected": False,
                },
            ],
        },
    )

    analysis = build_feature_analysis(config_copy, tmp_path)
    debug = analysis["relative_strength_debug"]
    pipeline = analysis["relative_strength_pipeline"]

    assert debug["candidate_count"] == 3
    assert debug["topix_records_loaded"] == 35
    assert debug["topix_api_calls"] == 1
    assert debug["rs_data_available_count"] == 2
    assert debug["rs_data_missing_count"] == 1
    assert debug["relative_strength_score_distribution"]["1-3"] == 1
    assert debug["relative_strength_score_distribution"]["7-10"] == 1
    assert debug["benchmark_source_distribution"]["topix"] == 1
    assert debug["benchmark_source_distribution"]["prime_average"] == 1
    assert debug["top_20_relative_strength_score"][0]["code"] == "1001"
    assert pipeline["feature_enabled"] is True
    assert pipeline["scoring_enabled"] is True
    assert pipeline["benchmark_provider_called"] is True
    assert pipeline["records_loaded"] == 35
    assert pipeline["benchmark_source"] == "topix"
    assert pipeline["rs_calculated"] is True
    markdown = render_feature_analysis_markdown(analysis)
    assert "## Relative Strength Debug" in markdown
    assert "### Relative Strength Pipeline" in markdown
    assert "- records loaded: 35" in markdown
    assert "### benchmark_source" in markdown
    assert "Top 20 relative_strength_score" in markdown


def test_investor_context_debug_shows_top_candidates_and_effect(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    config_copy.setdefault("features", {})["investor_context"] = True
    config_copy.setdefault("scoring", {})["use_investor_context_score"] = True
    initialize_database(config_copy, tmp_path)
    save_trades(
        config_copy,
        tmp_path,
        "2026-03-06",
        [
            {
                "trade_id": "high-win",
                "action": "SELL",
                "code": "1001",
                "name": "High Winner",
                "signal_date": "2026-03-01",
                "entry_date": "2026-03-02",
                "exit_date": "2026-03-06",
                "profit": 10000,
                "profit_rate": 0.1,
                "gross_profit": 10000,
                "result": "WIN",
                "order_status": "FILLED",
                "investor_context_score": 5,
                "overseas_net_buy_4w_sum": 1000,
                "overseas_net_buy_4w_trend": "improving",
                "score_components": {"investor_context_score": 5, "component_total": 50, "matches_total_score": True},
            },
            {
                "trade_id": "low-loss",
                "action": "SELL",
                "code": "1002",
                "name": "Low Loser",
                "signal_date": "2026-03-01",
                "entry_date": "2026-03-02",
                "exit_date": "2026-03-06",
                "profit": -5000,
                "profit_rate": -0.05,
                "gross_profit": -5000,
                "result": "LOSS",
                "order_status": "FILLED",
                "investor_context_score": 0,
                "overseas_net_buy_4w_sum": -100,
                "overseas_net_buy_4w_trend": "worsening",
                "score_components": {"investor_context_score": 0, "component_total": 45, "matches_total_score": True},
            },
        ],
    )

    full = build_feature_analysis(config_copy, tmp_path)
    analysis = full["investor_context_analysis"]
    top = analysis["top_candidates"][0]
    effect_by_bucket = {item["bucket"]: item for item in analysis["effect_analysis"]}

    assert top["code"] == "1001"
    assert top["investor_context_score"] == 5
    assert effect_by_bucket["investor_context_score >= 4"]["count"] == 1
    assert effect_by_bucket["investor_context_score >= 4"]["win_rate"] == 1.0
    assert effect_by_bucket["investor_context_score <= 0"]["count"] == 1
    assert effect_by_bucket["investor_context_score <= 0"]["profit_factor"] == 0.0
    markdown = render_feature_analysis_markdown(full)
    assert "### Top Investor Context Candidates" in markdown
    assert "| 2026-03-01 | 1001 | 5.00 | 1,000.00 | improving | true | WIN | 10,000円 |" in markdown
    assert "### Investor Context Effect Analysis" in markdown


def test_investor_context_filter_analysis_reports_rejections(tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_11")
    config["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config, tmp_path)
    save_scoring_results(
        config,
        tmp_path,
        {
            "date": "2026-03-01",
            "scores": [
                {
                    "code": "1001",
                    "name": "Rejected",
                    "rank": 1,
                    "selected": False,
                    "rejected_reason": "investor_context_negative",
                    "investor_context_score": -2,
                    "total_score": 48,
                    "technical_score": 48,
                    "confidence": 0.8,
                    "score_components": {
                        "technical_score": 48,
                        "investor_context_score": 0,
                        "component_total": 48,
                        "total_score": 48,
                        "matches_total_score": True,
                    },
                }
            ],
        },
    )
    save_trades(
        config,
        tmp_path,
        "2026-03-06",
        [
            {
                "trade_id": "accepted",
                "action": "SELL",
                "code": "1002",
                "name": "Accepted",
                "signal_date": "2026-03-01",
                "entry_date": "2026-03-02",
                "exit_date": "2026-03-06",
                "profit": 7000,
                "profit_rate": 0.07,
                "gross_profit": 7000,
                "result": "WIN",
                "order_status": "FILLED",
                "investor_context_score": 1,
            }
        ],
    )

    analysis = build_feature_analysis(config, tmp_path)
    investor_filter = analysis["investor_context_filter"]
    markdown = render_feature_analysis_markdown(analysis)

    assert investor_filter["rejected_count"] == 1
    assert investor_filter["rejected_codes"] == ["1001"]
    assert investor_filter["accepted_profit"] == 7000
    assert "## Investor Context Filter" in markdown
    assert "- rejected_codes: 1001" in markdown


def test_relative_strength_debug_warns_when_all_scores_are_zero(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    config_copy.setdefault("features", {})["relative_strength"] = True
    config_copy.setdefault("scoring", {})["use_relative_strength_score"] = True
    initialize_database(config_copy, tmp_path)
    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-01",
            "scores": [
                {
                    "code": "1001",
                    "rank": 1,
                    "selected": True,
                    "benchmark_source": "topix",
                    "relative_strength_5d": 0.01,
                    "relative_strength_10d": 0.02,
                    "relative_strength_20d": 0.03,
                    "relative_strength_score": 0,
                    "score_components": {"relative_strength_score": 0},
                }
            ],
        },
    )

    debug = build_feature_analysis(config_copy, tmp_path)["relative_strength_debug"]

    assert "relative_strength_score is zero for all candidates" in debug["warnings"]


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

    assert audit["features"]["financial_context"]["data_enabled"] is True
    assert audit["features"]["financial_context"]["scoring_enabled"] is True
    assert audit["features"]["financial_context"]["non_zero_score_count"] == 0
    assert audit["features"]["financial_context"]["status"] == "inactive_in_practice"
    assert "financial_context" in audit["inactive_in_practice"]
    markdown = render_feature_analysis_markdown(analysis)
    assert "## Feature Activation Audit" in markdown
    assert "inactive_in_practice" in markdown


def test_feature_activation_audit_marks_v2_9_financial_context_data_only(tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_9")
    config["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config, tmp_path)

    audit = build_feature_analysis(config, tmp_path)["feature_activation_audit"]
    financial = audit["features"]["financial_context"]

    assert financial["data_enabled"] is True
    assert financial["scoring_enabled"] is False
    assert financial["status"] == "data_only"
    assert "financial_context" in audit["data_only"]


def test_feature_activation_audit_marks_v2_7_earnings_filter_inactive_in_practice(tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_7")
    config["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config, tmp_path)

    audit = build_feature_analysis(config, tmp_path)["feature_activation_audit"]
    earnings = audit["features"]["earnings_filter"]

    assert earnings["data_enabled"] is True
    assert earnings["scoring_enabled"] == "N/A"
    assert earnings["rejected_count"] == 0
    assert earnings["status"] == "inactive_in_practice"


def test_earnings_filter_debug_reports_records_and_rejections(tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_7")
    config["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config, tmp_path)
    save_scoring_results(
        config,
        tmp_path,
        {
            "date": "2026-03-05",
            "scores": [
                {
                    "code": "1001",
                    "name": "Blocked",
                    "rank": 1,
                    "selected": False,
                    "earnings_calendar_records_count": 3,
                    "earnings_filter_checked": True,
                    "earnings_filter_blocked": True,
                    "earnings_filter_reason": "決算予定日前後のため新規買付見送り",
                    "earnings_announcement_date": "2026-03-06",
                    "earnings_info_found": True,
                    "earnings_candidate_date": "2026-03-06",
                    "earnings_days_until_earnings": 1,
                    "earnings_pipeline_feature_enabled": True,
                    "earnings_pipeline_fetch_start": "2026-03-01",
                    "earnings_pipeline_fetch_end": "2026-03-20",
                    "earnings_pipeline_cache_path": "data/cache/jquants/earnings_calendar/2026-03-05.json",
                    "earnings_pipeline_cache_exists": True,
                    "earnings_pipeline_cache_records": 3,
                    "earnings_pipeline_cache_loaded": True,
                    "earnings_pipeline_index_built": True,
                    "earnings_pipeline_candidate_matching_called": True,
                    "earnings_pipeline_records_loaded": 3,
                    "earnings_pipeline_matched_candidates": 1,
                    "earnings_pipeline_rejected_candidates": 1,
                },
                {
                    "code": "1002",
                    "name": "Unknown",
                    "rank": 2,
                    "selected": True,
                    "earnings_calendar_records_count": 3,
                    "earnings_filter_checked": True,
                    "earnings_filter_blocked": False,
                    "earnings_info_found": False,
                },
            ],
        },
    )

    analysis = build_feature_analysis(config, tmp_path)
    debug = analysis["earnings_filter_debug"]
    markdown = render_feature_analysis_markdown(analysis)

    assert debug["earnings_calendar_records"] == 3
    assert debug["candidate_count"] == 2
    assert debug["earnings_info_found_count"] == 1
    assert debug["earnings_info_missing_count"] == 1
    assert debug["earnings_filter_candidate_count"] == 1
    assert debug["earnings_filter_rejected_count"] == 1
    assert debug["earnings_filter_applied_count"] == 2
    assert debug["unknown_earnings_count"] == 1
    assert debug["status"] == "active"
    assert debug["days_to_earnings_distribution"]["-2 to +2"] == 1
    assert debug["days_to_earnings_distribution"]["unknown"] == 1
    assert debug["nearest_earnings_candidates"][0]["company_name"] == "Blocked"
    assert debug["nearest_earnings_candidates"][0]["action"] == "rejected"
    assert debug["top_rejected_candidates"][0]["code"] == "1001"
    pipeline = analysis["earnings_pipeline"]
    assert pipeline["feature_enabled"] is True
    assert pipeline["fetch_start"] == "2026-03-01"
    assert pipeline["fetch_end"] == "2026-03-20"
    assert pipeline["cache_exists"] is True
    assert pipeline["cache_records"] == 3
    assert pipeline["cache_loaded"] is True
    assert pipeline["index_built"] is True
    assert pipeline["candidate_matching_called"] is True
    assert pipeline["earnings_records_loaded"] == 3
    assert pipeline["matched_candidates"] == 1
    assert pipeline["rejected_candidates"] == 1
    assert "## Earnings Filter Debug" in markdown
    assert "## Earnings Pipeline" in markdown
    assert "cache exists: true" in markdown
    assert "days_to_earnings distribution" in markdown
    assert "nearest earnings candidates" in markdown
    assert "Top rejected candidates" in markdown


def test_earnings_filter_debug_statuses(tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_7")
    config["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config, tmp_path)

    unavailable = build_feature_analysis(config, tmp_path)["earnings_filter_debug"]
    assert unavailable["status"] == "earnings_data_unavailable"

    save_scoring_results(
        config,
        tmp_path,
        {
            "date": "2026-03-05",
            "scores": [
                {
                    "code": "1002",
                    "name": "No Match",
                    "rank": 1,
                    "selected": True,
                    "earnings_calendar_records_count": 3,
                    "earnings_filter_checked": True,
                    "earnings_filter_blocked": False,
                    "earnings_info_found": False,
                }
            ],
        },
    )
    no_match = build_feature_analysis(config, tmp_path)["earnings_filter_debug"]
    assert no_match["status"] == "no_candidate_match"

    save_scoring_results(
        config,
        tmp_path,
        {
            "date": "2026-03-06",
            "scores": [
                {
                    "code": "1003",
                    "name": "Far Earnings",
                    "rank": 1,
                    "selected": True,
                    "earnings_calendar_records_count": 3,
                    "earnings_filter_checked": True,
                    "earnings_filter_blocked": False,
                    "earnings_info_found": True,
                    "earnings_candidate_date": "2026-03-20",
                    "earnings_days_until_earnings": 14,
                }
            ],
        },
    )
    inactive = build_feature_analysis(config, tmp_path)["earnings_filter_debug"]
    assert inactive["status"] == "inactive_in_practice"
    assert inactive["days_to_earnings_distribution"]["+8 to +14"] == 1


def test_feature_activation_audit_shows_v2_6_relative_strength_layers(tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_6")
    config["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config, tmp_path)

    audit = build_feature_analysis(config, tmp_path)["feature_activation_audit"]
    relative_strength = audit["features"]["relative_strength"]

    assert relative_strength["data_enabled"] is True
    assert relative_strength["scoring_enabled"] is True
    assert relative_strength["runtime_active"] is False
    assert relative_strength["status"] == "inactive_in_practice"


def test_feature_activation_audit_marks_registry_profile_mismatch(tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_9")
    config["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    config["features"]["financial_context"] = False
    registry_dir = tmp_path / "config"
    registry_dir.mkdir()
    (registry_dir / "profile_registry.yaml").write_text(
        "profiles:\n"
        "  rookie_dealer_02_v2_9:\n"
        "    features:\n"
        "      financial_context: true\n",
        encoding="utf-8",
    )
    initialize_database(config, tmp_path)

    audit = build_feature_analysis(config, tmp_path)["feature_activation_audit"]

    assert audit["features"]["financial_context"]["status"] == "config_mismatch"
    assert "financial_context" in audit["config_mismatch"]


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
        "section": "TSEPrime",
        "market_section": "TSEPrime",
        "listing_market": "TSEPrime",
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
