from __future__ import annotations

from copy import deepcopy

import main
from db import analyze_operation_data, get_database_path, initialize_database, save_portfolio_snapshot, save_scoring_results, save_trades
from main import _profile_compare_row, build_profile_diff_analysis, build_profile_ranking, render_compare_profiles_markdown, write_backtest_summary
from profile_loader import load_profile


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
                    "rejected_reason": "上位候補だが最大採用数を超えたため落選",
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
    assert "- newly selected by v2.2: 1" in markdown
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
