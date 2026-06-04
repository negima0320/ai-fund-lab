from __future__ import annotations

import csv
import json
from copy import deepcopy

import main
from db import analyze_operation_data, get_database_path, initialize_database, save_portfolio_snapshot, save_scoring_results, save_trades
from main import (
    _compare_profiles_output_stem,
    _effective_config_differences,
    _profile_compare_row,
    build_profile_diff_analysis,
    build_profile_diff_analyses,
    build_profile_ranking,
    enrich_candidates_with_position_prices,
    render_compare_profiles_markdown,
    write_trades_csv,
    write_backtest_summary,
)
from profile_loader import load_profile


def test_compare_profiles_output_stem_keeps_two_profile_name_readable() -> None:
    stem = _compare_profiles_output_stem(
        "2026-01-05",
        "2026-03-06",
        ["rookie_dealer_02_v2_26", "rookie_dealer_02_v2_38"],
    )

    assert stem == "compare_2026-01-05_to_2026-03-06_rookie_dealer_02_v2_26_vs_rookie_dealer_02_v2_38"
    assert len(f"{stem}.json") < 200


def test_compare_profiles_output_stem_shortens_many_profiles_with_hash() -> None:
    profile_ids = [f"rookie_dealer_02_v2_{index}" for index in range(1, 25)]

    stem = _compare_profiles_output_stem("2026-01-05", "2026-03-06", profile_ids)

    assert stem.startswith("compare_2026-01-05_to_2026-03-06_24profiles_")
    assert len(stem.rsplit("_", 1)[-1]) == 10
    assert len(f"{stem}.json") < 200
    assert "rookie_dealer_02_v2_1_vs" not in stem


def test_effective_config_differences_include_conditional_hold_extension() -> None:
    base = load_profile("rookie_dealer_02_v2_26")
    target = load_profile("rookie_dealer_02_v2_58")

    differences = _effective_config_differences(base, target)

    keys = {item["key"] for item in differences}
    assert "conditional_hold_extension" in keys


def test_enrich_candidates_with_position_prices_copies_hold_extension_indicator_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "indicators_2026-02-06.json").write_text(
        json.dumps(
            {
                "indicators": [
                    {
                        "code": "18780",
                        "name": "大東建託",
                        "close": 3385.0,
                        "volume": 100000,
                        "ma5": 3200.0,
                        "ma25": 3089.36,
                        "previous_ma25": 3074.16,
                        "rsi": 61.2,
                        "volume_ratio": 2.5,
                        "relative_strength_score": 7,
                        "relative_strength_5d": 0.04,
                        "relative_strength_10d": 0.05,
                        "relative_strength_20d": 0.08,
                        "stock_return_5d": 0.06,
                        "stock_return_10d": 0.09,
                        "stock_return_20d": 0.12,
                        "benchmark_return_5d": 0.02,
                        "benchmark_return_10d": 0.04,
                        "benchmark_return_20d": 0.04,
                        "benchmark_source": "topix",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )
    state = {"positions": [{"code": "18780"}]}

    enriched = enrich_candidates_with_position_prices([], state, "2026-02-06")

    assert len(enriched) == 1
    row = enriched[0]
    assert row["previous_ma25"] == 3074.16
    assert row["relative_strength_score"] == 7
    assert row["relative_strength_5d"] == 0.04
    assert row["relative_strength_10d"] == 0.05
    assert row["relative_strength_20d"] == 0.08
    assert row["stock_return_5d"] == 0.06
    assert row["stock_return_10d"] == 0.09
    assert row["stock_return_20d"] == 0.12
    assert row["benchmark_return_5d"] == 0.02
    assert row["benchmark_return_10d"] == 0.04
    assert row["benchmark_return_20d"] == 0.04
    assert row["benchmark_source"] == "topix"


def test_run_compare_profiles_writes_readable_two_profile_outputs(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "ai_fund_lab.sqlite3"
    db_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "load_profile", lambda profile_id: {"profile_id": profile_id})
    monkeypatch.setattr(main, "get_database_path", lambda _config, _root: db_path)
    monkeypatch.setattr(
        main,
        "_profile_compare_row",
        lambda config, *_args: {
            "profile_id": config["profile_id"],
            "profile_name": config["profile_id"],
            "final_assets": 1000000,
            "total_trades": 0,
        },
    )
    monkeypatch.setattr(main, "build_profile_ranking", lambda rows: [])
    monkeypatch.setattr(main, "build_profile_diff_analyses", lambda *_args: [])

    markdown_path, json_path = main.run_compare_profiles(
        ["rookie_dealer_02_v2_26", "rookie_dealer_02_v2_38"],
        "2026-01-05",
        "2026-03-06",
    )

    assert markdown_path.exists()
    assert json_path.exists()
    assert "rookie_dealer_02_v2_26_vs_rookie_dealer_02_v2_38" in json_path.name
    assert "compared_profiles" in json_path.read_text(encoding="utf-8")


def test_run_compare_profiles_writes_short_many_profile_outputs(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "ai_fund_lab.sqlite3"
    db_path.write_text("", encoding="utf-8")
    profile_ids = [f"rookie_dealer_02_v2_{index}" for index in range(1, 25)]
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "load_profile", lambda profile_id: {"profile_id": profile_id})
    monkeypatch.setattr(main, "get_database_path", lambda _config, _root: db_path)
    monkeypatch.setattr(
        main,
        "_profile_compare_row",
        lambda config, *_args: {
            "profile_id": config["profile_id"],
            "profile_name": config["profile_id"],
            "final_assets": 1000000,
            "total_trades": 0,
        },
    )
    monkeypatch.setattr(main, "build_profile_ranking", lambda rows: [])
    monkeypatch.setattr(main, "build_profile_diff_analyses", lambda *_args: [])

    markdown_path, json_path = main.run_compare_profiles(profile_ids, "2026-01-05", "2026-03-06")

    assert markdown_path.exists()
    assert json_path.exists()
    assert "_24profiles_" in json_path.name
    assert len(json_path.name) < 200
    assert all(profile_id in json_path.read_text(encoding="utf-8") for profile_id in profile_ids)


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
                "total_score": 76,
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
                "total_score": 72,
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
            {
                "trade_id": "buy",
                "action": "BUY",
                "code": "1004",
                "name": "Buy Only",
                "entry_date": "2026-03-06",
                "profit": 0,
                "profit_rate": 0,
                "gross_profit": 0,
                "result": "",
                "order_status": "FILLED",
            },
            {
                "trade_id": "skip",
                "action": "SKIP_BUY",
                "code": "1005",
                "name": "Skip",
                "entry_date": "2026-03-06",
                "profit": 0,
                "profit_rate": 0,
                "gross_profit": 0,
                "result": "",
                "order_status": "REJECTED",
            },
        ],
    )

    analyze_trades = analyze_operation_data(config_copy, tmp_path)["trade_analysis"]
    compare_row = _profile_compare_row(
        config_copy,
        get_database_path(config_copy, tmp_path),
        "2026-03-01",
        "2026-03-06",
    )
    compare_pf = compare_row["profit_factor"]

    assert analyze_trades["profit_factor"] == compare_pf == 2.5
    assert analyze_trades["total_trades"] == compare_row["total_trades"] == 2
    assert analyze_trades["closed_trade_count"] == compare_row["closed_trade_count"] == 2
    assert analyze_trades["win_count"] == compare_row["win_count"] == 1
    assert analyze_trades["loss_count"] == compare_row["loss_count"] == 1
    assert analyze_trades["win_rate"] == compare_row["win_rate"] == 0.5
    assert analyze_trades["excluded_order_event_count"] == compare_row["excluded_order_event_count"] == 0

    assert compare_row["average_win_profit_rate"] == 0.1
    assert compare_row["average_loss_profit_rate"] == -0.04
    assert compare_row["average_holding_days"] == 3.0
    assert compare_row["expectancy"] == 0.03
    score_detail_by_bucket = {item["bucket"]: item for item in compare_row["score_detail"]}
    assert score_detail_by_bucket["72-73"]["count"] == 1
    assert score_detail_by_bucket["72-73"]["win_rate"] == 0.0
    assert score_detail_by_bucket["76-79"]["count"] == 1
    assert score_detail_by_bucket["76-79"]["win_rate"] == 1.0
    markdown = render_compare_profiles_markdown(
        {
            "start_date": "2026-03-01",
            "end_date": "2026-03-06",
            "profiles": [compare_row],
            "ranking": [],
        }
    )
    assert "## Score Detail" in markdown
    assert "| 72-73 | 1 | 0.00% | -4.00% | -4,000円 |" in markdown
    assert "| 76-79 | 1 | 100.00% | 10.00% | 10,000円 |" in markdown


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


def test_backtest_summary_uses_closed_trade_metrics(config_copy: dict, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    backtest_dir = tmp_path / "logs" / "backtests" / "test"
    backtest_dir.mkdir(parents=True)
    closed_trades = [
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
            "exit_reason": "利確",
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
            "result": "LOSS",
            "order_status": "FILLED",
            "exit_reason": "損切り",
        },
    ]
    all_trades = [
        *closed_trades,
        {
            "trade_id": "buy",
            "action": "BUY",
            "code": "1003",
            "name": "Buy",
            "entry_date": "2026-03-06",
            "profit": 0,
            "gross_profit": 0,
            "result": "",
            "order_status": "FILLED",
        },
        {
            "trade_id": "pending",
            "action": "SELL",
            "code": "1004",
            "name": "Pending",
            "entry_date": "2026-03-01",
            "exit_date": "2026-03-06",
            "profit": 999999,
            "gross_profit": 999999,
            "result": "WIN",
            "order_status": "PENDING",
        },
    ]
    state = {"total_assets": 1006000, "closed_trades": closed_trades}

    summary = write_backtest_summary(
        "2026-03-01_to_2026-03-06",
        "2026-03-01",
        "2026-03-06",
        config_copy,
        state,
        [
            {
                "date": "2026-03-06",
                "day": 1,
                "cash": 1006000,
                "positions_value": 0,
                "total_assets": 1006000,
                "daily_profit": 6000,
                "cumulative_profit": 6000,
                "cumulative_profit_rate": 0.006,
                "win_rate": 0.5,
                "max_drawdown": 0,
                "open_positions_count": 0,
                "closed_trades_count": 2,
            }
        ],
        all_trades,
        backtest_dir,
    )

    assert summary["total_trades"] == 2
    assert summary["closed_trade_count"] == 2
    assert summary["win_count"] == 1
    assert summary["loss_count"] == 1
    assert summary["excluded_order_event_count"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["profit_factor"] == 2.5


def test_analyze_filters_portfolio_and_trades_by_profile(config_copy: dict, tmp_path) -> None:
    profile_a = deepcopy(config_copy)
    profile_b = deepcopy(config_copy)
    db_path = str(tmp_path / "ai_fund_lab.sqlite3")
    profile_a["database"]["path"] = db_path
    profile_b["database"]["path"] = db_path
    profile_a["profile_id"] = "profile_a"
    profile_a["profile_name"] = "Profile A"
    profile_b["profile_id"] = "profile_b"
    profile_b["profile_name"] = "Profile B"
    initialize_database(profile_a, tmp_path)

    save_portfolio_snapshot(
        profile_a,
        tmp_path,
        {
            "date": "2026-03-06",
            "cash": 1000000,
            "positions_value": 0,
            "total_assets": 1010000,
            "cumulative_profit": 10000,
            "gross_cumulative_profit": 10000,
            "net_cumulative_profit": 10000,
            "total_commission": 0,
            "estimated_tax_total": 0,
            "max_drawdown": -0.01,
        },
    )
    save_portfolio_snapshot(
        profile_b,
        tmp_path,
        {
            "date": "2026-03-06",
            "cash": 1000000,
            "positions_value": 0,
            "total_assets": 1120000,
            "cumulative_profit": 120000,
            "gross_cumulative_profit": 120000,
            "net_cumulative_profit": 120000,
            "total_commission": 0,
            "estimated_tax_total": 0,
            "max_drawdown": -0.02,
        },
    )
    save_trades(
        profile_a,
        tmp_path,
        "2026-03-06",
        [
            {
                "trade_id": "a-win",
                "action": "SELL",
                "code": "1001",
                "name": "A Winner",
                "entry_date": "2026-03-01",
                "exit_date": "2026-03-06",
                "profit": 10000,
                "profit_rate": 0.1,
                "gross_profit": 10000,
                "result": "WIN",
                "order_status": "FILLED",
            }
        ],
    )
    save_trades(
        profile_b,
        tmp_path,
        "2026-03-06",
        [
            {
                "trade_id": "b-loss",
                "action": "SELL",
                "code": "2001",
                "name": "B Loser",
                "entry_date": "2026-03-01",
                "exit_date": "2026-03-06",
                "profit": -5000,
                "profit_rate": -0.05,
                "gross_profit": -5000,
                "result": "LOSS",
                "order_status": "FILLED",
            }
        ],
    )

    analysis_a = analyze_operation_data(profile_a, tmp_path)
    analysis_b = analyze_operation_data(profile_b, tmp_path)

    assert analysis_a["current_profile_id"] == "profile_a"
    assert analysis_b["current_profile_id"] == "profile_b"
    assert analysis_a["portfolio_analysis"]["latest_total_assets"] == 1010000
    assert analysis_b["portfolio_analysis"]["latest_total_assets"] == 1120000
    assert analysis_a["trade_analysis"]["total_trades"] == 1
    assert analysis_b["trade_analysis"]["total_trades"] == 1
    assert analysis_a["trade_analysis"]["win_count"] == 1
    assert analysis_a["trade_analysis"]["loss_count"] == 0
    assert analysis_b["trade_analysis"]["win_count"] == 0
    assert analysis_b["trade_analysis"]["loss_count"] == 1


def test_profile_diff_analysis_compares_v2_1_and_v2_2(tmp_path) -> None:
    profile_21 = load_profile("rookie_dealer_02_v2_1")
    profile_22 = load_profile("rookie_dealer_02_v2_2")
    db_path = str(tmp_path / "ai_fund_lab.sqlite3")
    profile_21["database"]["path"] = db_path
    profile_22["database"]["path"] = db_path
    initialize_database(profile_21, tmp_path)
    save_scoring_results(
        profile_21,
        tmp_path,
        {
            "date": "2026-03-06",
            "scores": [
                {
                    "code": "1001",
                    "name": "Shared",
                    "rank": 1,
                    "total_score": 80,
                    "selected": True,
                    "market_regime": "risk_off",
                },
                {
                    "code": "1002",
                    "name": "Blocked",
                    "rank": 2,
                    "total_score": 76,
                    "selected": False,
                    "market_regime": "risk_off",
                    "rejected_reason": "risk_offのため買付抑制",
                },
                {
                    "code": "1003",
                    "name": "Removed",
                    "rank": 3,
                    "total_score": 74,
                    "selected": True,
                    "market_regime": "neutral",
                },
            ],
        },
    )
    save_scoring_results(
        profile_22,
        tmp_path,
        {
            "date": "2026-03-06",
            "scores": [
                {
                    "code": "1001",
                    "name": "Shared",
                    "rank": 1,
                    "total_score": 80,
                    "selected": True,
                    "market_regime": "risk_off",
                },
                {
                    "code": "1002",
                    "name": "Newly Selected",
                    "rank": 2,
                    "total_score": 76,
                    "selected": True,
                    "market_regime": "risk_off",
                },
                {
                    "code": "1003",
                    "name": "Removed",
                    "rank": 3,
                    "total_score": 74,
                    "selected": False,
                    "market_regime": "neutral",
                    "rejected_reason": "investor_context_negative",
                },
            ],
        },
    )

    analysis = build_profile_diff_analysis(
        [profile_21, profile_22],
        get_database_path(profile_21, tmp_path),
        "2026-03-01",
        "2026-03-31",
    )

    assert analysis is not None
    assert analysis["base_profile_id"] == "rookie_dealer_02_v2_1"
    assert analysis["target_profile_id"] == "rookie_dealer_02_v2_2"
    assert analysis["base_selected_count"] == 2
    assert analysis["target_selected_count"] == 2
    assert analysis["base_risk_off_candidate_count"] == 2
    assert analysis["target_risk_off_candidate_count"] == 2
    assert analysis["base_risk_off_rejected_count"] == 1
    assert analysis["target_risk_off_rejected_count"] == 0
    assert analysis["newly_selected_count"] == 1
    assert analysis["removed_count"] == 1
    assert analysis["investor_filter_rejected_count"] == 1
    assert analysis["newly_selected"][0]["code"] == "1002"
    assert analysis["removed"][0]["code"] == "1003"
    assert analysis["no_practical_effect"] is False
    assert {
        "key": "market_filter.risk_off_buy_policy",
        "base": "conservative",
        "target": "relaxed",
    } in analysis["effective_config_differences"]
    markdown = render_compare_profiles_markdown(
        {
            "start_date": "2026-03-01",
            "end_date": "2026-03-31",
            "profiles": [],
            "ranking": [],
            "profile_diff_analysis": analysis,
        }
    )
    assert "## Profile Diff Analysis" in markdown
    assert "- newly selected by target: 1" in markdown
    assert "- investor_filter_rejected_count: 1" in markdown
    assert "2026-03-06 1002 Newly Selected" in markdown


def test_profile_diff_analysis_marks_no_practical_effect(tmp_path) -> None:
    profile_21 = load_profile("rookie_dealer_02_v2_1")
    profile_22 = load_profile("rookie_dealer_02_v2_2")
    db_path = str(tmp_path / "ai_fund_lab.sqlite3")
    profile_21["database"]["path"] = db_path
    profile_22["database"]["path"] = db_path
    initialize_database(profile_21, tmp_path)
    for profile in [profile_21, profile_22]:
        save_scoring_results(
            profile,
            tmp_path,
            {
                "date": "2026-03-06",
                "scores": [
                    {
                        "code": "1001",
                        "name": "Same",
                        "rank": 1,
                        "total_score": 80,
                        "selected": True,
                        "market_regime": "risk_off",
                    }
                ],
            },
        )

    analysis = build_profile_diff_analysis(
        [profile_21, profile_22],
        get_database_path(profile_21, tmp_path),
        "2026-03-01",
        "2026-03-31",
    )

    assert analysis is not None
    assert analysis["newly_selected_count"] == 0
    assert analysis["removed_count"] == 0
    assert analysis["no_practical_effect"] is True
    markdown = render_compare_profiles_markdown(
        {
            "start_date": "2026-03-01",
            "end_date": "2026-03-31",
            "profiles": [],
            "ranking": [],
            "profile_diff_analysis": analysis,
        }
    )
    assert "No practical effect" in markdown


def test_profile_diff_analysis_counts_affordable_fallback_buy_as_selection_diff(tmp_path) -> None:
    profile_26 = load_profile("rookie_dealer_02_v2_26")
    profile_40 = load_profile("rookie_dealer_02_v2_40")
    db_path = str(tmp_path / "ai_fund_lab.sqlite3")
    profile_26["database"]["path"] = db_path
    profile_40["database"]["path"] = db_path
    initialize_database(profile_26, tmp_path)
    scores = [
        {
            "code": "1001",
            "name": "Expensive Selected",
            "rank": 1,
            "total_score": 80,
            "selected": True,
        },
        {
            "code": "1002",
            "name": "Affordable Fallback",
            "rank": 2,
            "total_score": 77,
            "selected": False,
        },
    ]
    for profile in [profile_26, profile_40]:
        save_scoring_results(profile, tmp_path, {"date": "2026-03-06", "scores": deepcopy(scores)})
    save_trades(
        profile_40,
        tmp_path,
        "2026-03-07",
        [
            {
                "action": "BUY",
                "status": "FILLED",
                "code": "1002",
                "name": "Affordable Fallback",
                "signal_date": "2026-03-06",
                "entry_date": "2026-03-07",
                "entry_price": 2500,
                "shares": 100,
                "amount": 250000,
                "total_score": 77,
                "reason": "affordable_fallback_buy",
                "affordable_fallback_buy_selected": True,
                "affordable_fallback_original_code": "1001",
                "affordable_fallback_original_name": "Expensive Selected",
                "affordable_fallback_reason": "selected_but_not_affordable",
                "affordable_fallback_round_lot_amount": 250000,
            }
        ],
    )

    analysis = build_profile_diff_analysis(
        [profile_26, profile_40],
        get_database_path(profile_26, tmp_path),
        "2026-03-01",
        "2026-03-31",
    )

    assert analysis is not None
    assert analysis["base_profile_id"] == "rookie_dealer_02_v2_26"
    assert analysis["target_profile_id"] == "rookie_dealer_02_v2_40"
    assert analysis["newly_selected_count"] == 1
    assert analysis["selection_diff_count"] == 1
    assert analysis["no_practical_effect"] is False
    assert analysis["practical_effect"] == "selection_effect"
    assert analysis["newly_selected"][0]["code"] == "1002"
    assert analysis["newly_selected"][0]["selection_source"] == "affordable_fallback_buy"
    assert analysis["newly_selected"][0]["affordable_fallback_original_code"] == "1001"
    assert analysis["outcome_diff_count"] == 1
    assert analysis["trade_outcome_diff"]["new_buy_trade_count"] == 1
    assert analysis["trade_outcome_diff"]["new_buy_trades"][0]["affordable_fallback_buy_selected"] is True


def test_profile_diff_analysis_falls_back_to_scoring_logs_when_target_db_rows_are_skipped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    profile_21 = load_profile("rookie_dealer_02_v2_1")
    profile_22 = load_profile("rookie_dealer_02_v2_2")
    db_path = str(tmp_path / "ai_fund_lab.sqlite3")
    profile_21["database"]["path"] = db_path
    profile_22["database"]["path"] = db_path
    initialize_database(profile_21, tmp_path)
    save_scoring_results(
        profile_21,
        tmp_path,
        {
            "date": "2026-03-06",
            "scores": [
                {
                    "code": "1001",
                    "name": "Same",
                    "rank": 1,
                    "total_score": 80,
                    "selected": True,
                }
            ],
        },
    )
    scoring_dir = tmp_path / "logs" / "backtests" / "rookie_dealer_02_v2_2" / "2026-03-01_to_2026-03-31"
    scoring_dir.mkdir(parents=True)
    main.write_json(
        scoring_dir / "scoring_2026-03-06.json",
        {
            "date": "2026-03-06",
            "profile_id": "rookie_dealer_02_v2_2",
            "scores": [
                {
                    "code": "1001",
                    "name": "Same",
                    "rank": 1,
                    "total_score": 80,
                    "selected": True,
                }
            ],
        },
    )

    analysis = build_profile_diff_analysis(
        [profile_21, profile_22],
        get_database_path(profile_21, tmp_path),
        "2026-03-01",
        "2026-03-31",
    )

    assert analysis is not None
    assert analysis["base_selected_count"] == 1
    assert analysis["target_selected_count"] == 1
    assert analysis["selection_diff_count"] == 0
    assert analysis["practical_effect"] == "no_practical_effect"


def test_profile_diff_analysis_uses_backtest_summary_fallback_trades_when_db_is_stale(tmp_path, monkeypatch) -> None:
    profile_26 = load_profile("rookie_dealer_02_v2_26")
    profile_40 = load_profile("rookie_dealer_02_v2_40")
    db_path = str(tmp_path / "ai_fund_lab.sqlite3")
    profile_26["database"]["path"] = db_path
    profile_40["database"]["path"] = db_path
    initialize_database(profile_26, tmp_path)
    for profile in [profile_26, profile_40]:
        save_scoring_results(
            profile,
            tmp_path,
            {
                "date": "2026-03-06",
                "scores": [
                    {"code": "1001", "name": "Expensive Selected", "rank": 1, "total_score": 80, "selected": True},
                    {"code": "1002", "name": "Affordable Fallback", "rank": 2, "total_score": 77, "selected": False},
                ],
            },
    )
    monkeypatch.setattr(main, "ROOT", tmp_path)
    summary_path = tmp_path / "logs" / "backtests" / "rookie_dealer_02_v2_40" / "2026-03-01_to_2026-03-31" / "backtest_summary.json"
    main.write_json(
        summary_path,
        {
            "all_trades": [
                {
                    "action": "BUY",
                    "code": "1002",
                    "name": "Affordable Fallback",
                    "signal_date": "2026-03-06",
                    "entry_date": "2026-03-07",
                    "amount": 250000,
                    "shares": 100,
                    "total_score": 77,
                    "reason": "affordable_fallback_buy",
                    "affordable_fallback_buy_selected": True,
                    "affordable_fallback_original_code": "1001",
                    "affordable_fallback_reason": "selected_but_not_affordable",
                }
            ]
        },
    )

    analysis = build_profile_diff_analysis(
        [profile_26, profile_40],
        get_database_path(profile_26, tmp_path),
        "2026-03-01",
        "2026-03-31",
    )

    assert analysis is not None
    assert analysis["selection_diff_count"] == 1
    assert analysis["outcome_diff_count"] == 1
    assert analysis["no_practical_effect"] is False
    assert analysis["newly_selected"][0]["code"] == "1002"
    assert analysis["trade_outcome_diff"]["new_buy_trade_count"] == 1


def test_write_trades_csv_includes_affordable_fallback_flags(tmp_path) -> None:
    path = tmp_path / "trades.csv"
    write_trades_csv(
        path,
        [
            {
                "action": "BUY",
                "code": "1002",
                "name": "Affordable Fallback",
                "signal_date": "2026-03-06",
                "entry_date": "2026-03-07",
                "amount": 250000,
                "affordable_fallback_buy_selected": True,
                "affordable_fallback_original_code": "1001",
                "affordable_fallback_original_name": "Expensive Selected",
                "affordable_fallback_reason": "selected_but_not_affordable",
                "affordable_fallback_round_lot_amount": 250000,
            }
        ],
    )

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["affordable_fallback_buy_selected"] == "True"
    assert rows[0]["affordable_fallback_original_code"] == "1001"
    assert rows[0]["affordable_fallback_reason"] == "selected_but_not_affordable"


def test_profile_diff_analysis_detects_trade_outcome_diff_without_entry_diff(tmp_path) -> None:
    profile_21 = load_profile("rookie_dealer_02_v2_1")
    profile_22 = load_profile("rookie_dealer_02_v2_2")
    db_path = str(tmp_path / "ai_fund_lab.sqlite3")
    profile_21["database"]["path"] = db_path
    profile_22["database"]["path"] = db_path
    initialize_database(profile_21, tmp_path)
    for profile in [profile_21, profile_22]:
        save_scoring_results(
            profile,
            tmp_path,
            {
                "date": "2026-03-06",
                "scores": [
                    {
                        "code": "1001",
                        "name": "Same Entry",
                        "rank": 1,
                        "total_score": 80,
                        "selected": True,
                    }
                ],
            },
        )
    save_trades(
        profile_21,
        tmp_path,
        "2026-03-10",
        [
            {
                "action": "SELL",
                "status": "FILLED",
                "code": "1001",
                "name": "Same Entry",
                "entry_date": "2026-03-06",
                "exit_date": "2026-03-10",
                "holding_days": 3,
                "profit": 1000,
                "profit_rate": 0.01,
                "exit_reason": "利確",
                "result": "WIN",
            }
        ],
    )
    save_trades(
        profile_22,
        tmp_path,
        "2026-03-11",
        [
            {
                "action": "SELL",
                "status": "FILLED",
                "code": "1001",
                "name": "Same Entry",
                "entry_date": "2026-03-06",
                "exit_date": "2026-03-11",
                "holding_days": 4,
                "profit": 2000,
                "profit_rate": 0.02,
                "exit_reason": "最大保有期間到達",
                "result": "WIN",
            }
        ],
    )

    analysis = build_profile_diff_analysis(
        [profile_21, profile_22],
        get_database_path(profile_21, tmp_path),
        "2026-03-01",
        "2026-03-31",
    )

    assert analysis is not None
    assert analysis["selection_diff_count"] == 0
    assert analysis["outcome_diff_count"] == 1
    assert analysis["practical_effect"] == "execution_or_exit_effect"
    assert analysis["no_practical_effect"] is False
    assert analysis["trade_outcome_diff"]["same_entry_different_profit_count"] == 1


def test_profile_diff_analysis_reports_conditional_hold_extension_profit_diff(tmp_path, monkeypatch) -> None:
    profile_26 = load_profile("rookie_dealer_02_v2_26")
    profile_61 = load_profile("rookie_dealer_02_v2_61")
    db_path = str(tmp_path / "ai_fund_lab.sqlite3")
    profile_26["database"]["path"] = db_path
    profile_61["database"]["path"] = db_path
    monkeypatch.setattr(main, "ROOT", tmp_path)
    initialize_database(profile_26, tmp_path)
    for profile in [profile_26, profile_61]:
        save_scoring_results(
            profile,
            tmp_path,
            {
                "date": "2026-03-06",
                "scores": [
                    {
                        "code": "1001",
                        "name": "Extension Target",
                        "rank": 1,
                        "total_score": 80,
                        "selected": True,
                    }
                ],
            },
        )
    save_trades(
        profile_26,
        tmp_path,
        "2026-03-10",
        [
            {
                "action": "SELL",
                "status": "FILLED",
                "code": "1001",
                "name": "Extension Target",
                "entry_date": "2026-03-06",
                "exit_date": "2026-03-10",
                "holding_days": 5,
                "profit": 10000,
                "profit_rate": 0.04,
                "exit_reason": "最大保有期間到達",
                "result": "WIN",
            }
        ],
    )
    save_trades(
        profile_61,
        tmp_path,
        "2026-03-12",
        [
            {
                "action": "SELL",
                "status": "FILLED",
                "code": "1001",
                "name": "Extension Target",
                "entry_date": "2026-03-06",
                "exit_date": "2026-03-12",
                "holding_days": 7,
                "profit": 7000,
                "profit_rate": 0.028,
                "exit_reason": "最大保有期間到達",
                "result": "WIN",
            }
        ],
    )
    summary_path = tmp_path / "logs" / "backtests" / "rookie_dealer_02_v2_61" / "2026-03-01_to_2026-03-31" / "backtest_summary.json"
    main.write_json(
        summary_path,
        {
            "all_trades": [
                {
                    "action": "SELL",
                    "status": "FILLED",
                    "code": "1001",
                    "name": "Extension Target",
                    "entry_date": "2026-03-06",
                    "exit_date": "2026-03-12",
                    "holding_days": 7,
                    "profit": 7000,
                    "profit_rate": 0.028,
                    "exit_reason": "最大保有期間到達",
                    "conditional_hold_extension_applied": True,
                    "conditional_hold_extension_reason": "trend_continuation_profit>=0.0300_rs>=5.0_ma25_uptrend",
                    "conditional_hold_extension_trigger_profit_rate": 0.035,
                    "conditional_hold_extension_close_at_max_holding": 2420,
                    "conditional_hold_extension_ma25_at_max_holding": 2170.74,
                    "conditional_hold_extension_previous_ma25_at_max_holding": 2151.2,
                    "conditional_hold_extension_relative_strength_score_at_max_holding": 7,
                    "extension_profit_rate": 0.035,
                    "extension_exit_guard_triggered": True,
                    "extension_exit_guard_reason": "profit_pullback_exceeded",
                }
            ]
        },
    )

    analysis = build_profile_diff_analysis(
        [profile_26, profile_61],
        get_database_path(profile_26, tmp_path),
        "2026-03-01",
        "2026-03-31",
    )

    assert analysis is not None
    assert analysis["conditional_hold_extension_applied_count"] == 1
    assert analysis["conditional_hold_extension_loss_count"] == 1
    assert analysis["conditional_hold_extension_profit_diff_total"] == -3000
    assert analysis["conditional_hold_extension_profit_diff_average"] == -3000
    assert analysis["extension_exit_guard_count"] == 1
    assert analysis["extension_exit_guard_profit_diff_total"] == -3000
    assert analysis["extension_exit_guard_reasons"] == {"profit_pullback_exceeded": 1}
    detail = analysis["conditional_hold_extension_applied_detail"][0]
    assert detail["original_max_holding_exit_date"] == "2026-03-10"
    assert detail["actual_exit_date"] == "2026-03-12"
    assert detail["base_profit"] == 10000
    assert detail["target_profit"] == 7000
    assert detail["extension_profit_diff"] == -3000
    assert detail["relative_strength_score"] == 7
    assert detail["extension_exit_guard_triggered"] is True
    assert detail["extension_exit_guard_reason"] == "profit_pullback_exceeded"

    markdown = render_compare_profiles_markdown(
        {
            "start_date": "2026-03-01",
            "end_date": "2026-03-31",
            "profiles": [],
            "profile_diff_analysis": analysis,
        }
    )
    assert "## Conditional Hold Extension Applied Detail" in markdown
    assert "extension_exit_guard_count: 1" in markdown
    assert "Extension Target" in markdown
    assert "-3,000" in markdown


def test_profile_diff_analyses_include_all_targets(tmp_path) -> None:
    profile_21 = load_profile("rookie_dealer_02_v2_1")
    profile_22 = load_profile("rookie_dealer_02_v2_2")
    profile_24 = load_profile("rookie_dealer_02_v2_4")
    db_path = str(tmp_path / "ai_fund_lab.sqlite3")
    for profile in [profile_21, profile_22, profile_24]:
        profile["database"]["path"] = db_path
    initialize_database(profile_21, tmp_path)
    save_scoring_results(
        profile_21,
        tmp_path,
        {"date": "2026-03-06", "scores": [{"code": "1001", "name": "Base", "rank": 1, "selected": True}]},
    )
    save_scoring_results(
        profile_22,
        tmp_path,
        {"date": "2026-03-06", "scores": [{"code": "1002", "name": "Target A", "rank": 1, "selected": True}]},
    )
    save_scoring_results(
        profile_24,
        tmp_path,
        {"date": "2026-03-06", "scores": [{"code": "1003", "name": "Target B", "rank": 1, "selected": True}]},
    )

    analyses = build_profile_diff_analyses(
        [profile_21, profile_22, profile_24],
        get_database_path(profile_21, tmp_path),
        "2026-03-01",
        "2026-03-31",
    )

    assert [item["target_profile_id"] for item in analyses] == ["rookie_dealer_02_v2_2", "rookie_dealer_02_v2_4"]
    assert [item["newly_selected_count"] for item in analyses] == [1, 1]
