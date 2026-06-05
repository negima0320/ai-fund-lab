from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from ml.backtest_ml_analysis import BacktestMLAnalyzer


def _write_trades(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "action",
        "code",
        "signal_date",
        "entry_date",
        "exit_date",
        "gross_profit",
        "gross_profit_rate",
        "net_profit",
        "net_profit_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-11"),
                "code": "1001",
                "expected_return_5d": 0.01,
                "expected_return_10d": 0.04,
                "expected_max_return_10d": 0.08,
                "expected_max_return_20d": 0.16,
                "swing_success_probability_20d": 0.75,
                "upside_probability_10d": 0.7,
                "bad_entry_probability_10d": 0.1,
                "entry_risk_label": "safe",
                "ml_score": 12.0,
            },
            {
                "date": pd.Timestamp("2026-05-11"),
                "code": "1002",
                "expected_return_5d": -0.01,
                "expected_return_10d": -0.03,
                "expected_max_return_10d": 0.02,
                "expected_max_return_20d": 0.04,
                "swing_success_probability_20d": 0.20,
                "upside_probability_10d": 0.2,
                "bad_entry_probability_10d": 0.5,
                "entry_risk_label": "danger",
                "ml_score": -8.0,
            },
            {
                "date": pd.Timestamp("2026-05-11"),
                "code": "1003",
                "expected_return_5d": 0.0,
                "expected_return_10d": 0.01,
                "expected_max_return_10d": 0.04,
                "expected_max_return_20d": 0.08,
                "swing_success_probability_20d": 0.45,
                "upside_probability_10d": 0.45,
                "bad_entry_probability_10d": 0.3,
                "entry_risk_label": "watch",
                "ml_score": 2.0,
            },
        ]
    ).to_parquet(path, index=False)


def test_backtest_ml_analyzer_joins_trades_with_predictions(tmp_path) -> None:
    trades_path = tmp_path / "logs" / "backtests" / "profile1" / "2026-05-01_to_2026-05-31" / "trades.csv"
    _write_trades(
        trades_path,
        [
            {"action": "SELL", "code": "1001", "signal_date": "2026-05-11", "entry_date": "2026-05-12", "gross_profit": 1000, "gross_profit_rate": 0.1, "net_profit": 800, "net_profit_rate": 0.08},
            {"action": "SELL", "code": "1002", "signal_date": "2026-05-11", "entry_date": "2026-05-12", "gross_profit": -500, "gross_profit_rate": -0.05, "net_profit": -500, "net_profit_rate": -0.05},
            {"action": "SELL", "code": "1003", "signal_date": "2026-05-11", "entry_date": "2026-05-12", "gross_profit": -100, "gross_profit_rate": -0.01, "net_profit": -100, "net_profit_rate": -0.01},
            {"action": "SELL", "code": "9999", "signal_date": "2026-05-11", "entry_date": "2026-05-12", "gross_profit": 200, "gross_profit_rate": 0.02, "net_profit": 160, "net_profit_rate": 0.016},
        ],
    )
    _write_predictions(tmp_path / "data" / "ml" / "predictions" / "predictions_2026-05-11.parquet")
    analyzer = BacktestMLAnalyzer(root=tmp_path, predictions_root=tmp_path / "data" / "ml" / "predictions", report_root=tmp_path / "reports" / "ml")

    analysis = analyzer.analyze_profile("profile1", "2026-05-11", "2026-05-11", top_n=1)

    assert analysis["source"]["trades_csv"] == "logs/backtests/profile1/2026-05-01_to_2026-05-31/trades.csv"
    assert analysis["join_summary"]["trade_rows"] == 4
    assert analysis["join_summary"]["joined_count"] == 3
    assert analysis["join_summary"]["missing_count"] == 1
    risk = {row["entry_risk_label"]: row for row in analysis["risk_label_performance"]}
    assert risk["safe"]["net_profit_total"] == pytest.approx(800)
    assert risk["danger"]["net_profit_total"] == pytest.approx(-500)
    assert analysis["bad_entry_probability_bands"][0]["count"] == 1
    assert analysis["bad_entry_probability_bands"][2]["count"] == 1
    score_bands = {(row["entry_risk_label"], row["ml_score_band"]): row for row in analysis["risk_label_ml_score_bands"]}
    assert score_bands[("safe", ">= 10")]["count"] == 1
    assert score_bands[("danger", "< 0")]["count"] == 1
    matrix = {
        (row["bad_entry_probability_band"], row["expected_return_10d_band"]): row
        for row in analysis["bad_probability_expected_return_matrix"]
    }
    assert matrix[("0 to 0.25", "0.03 to 0.10")]["net_profit_total"] == pytest.approx(800)
    assert matrix[("0.40 to 0.70", "< 0")]["net_profit_total"] == pytest.approx(-500)
    danger_expected = {row["expected_return_10d_band"]: row for row in analysis["danger_expected_return_comparison"]}
    assert danger_expected["< 0"]["count"] == 1
    danger_upside = {row["upside_probability_10d_band"]: row for row in analysis["danger_upside_probability_comparison"]}
    assert danger_upside["< 0.40"]["count"] == 1
    filters = {row["filter_id"]: row for row in analysis["virtual_filter_simulation"]}
    assert filters["A"]["removed_trade_count"] == 1
    assert filters["A"]["kept_net_profit_total"] == pytest.approx(700)
    assert filters["A"]["profit_delta"] == pytest.approx(500)
    assert filters["C"]["removed_trade_count"] == 1
    assert filters["D"]["removed_trade_count"] == 1
    assert filters["E"]["removed_trade_count"] == 1
    assert filters["F"]["removed_trade_count"] == 1
    sizing = {row["sizing_id"]: row for row in analysis["virtual_position_sizing_simulation"]}
    assert sizing["A"]["adjusted_net_profit_total"] == pytest.approx(450)
    assert sizing["A"]["profit_delta"] == pytest.approx(250)
    assert sizing["A"]["average_position_multiplier"] == pytest.approx(5 / 6)
    assert sizing["A"]["weighted_win_rate"] == pytest.approx(0.4)
    assert sizing["E"]["adjusted_net_profit_total"] == pytest.approx(610)
    assert analysis["trade_details"][0]["code"] == "1001"
    assert "expected_return_10d" in analysis["trade_details"][0]
    win_loss = {row["win_loss"]: row for row in analysis["win_loss_analysis"]["ml_average_by_result"]}
    assert win_loss["win"]["expected_return_10d_mean"] == pytest.approx(0.04)
    assert win_loss["win"]["expected_max_return_20d_mean"] == pytest.approx(0.16)
    assert win_loss["loss"]["bad_entry_probability_10d_mean"] == pytest.approx(0.4)
    assert win_loss["loss"]["swing_success_probability_20d_mean"] == pytest.approx(0.325)
    danger_diff = {row["bucket"]: row for row in analysis["win_loss_analysis"]["danger_win_loss_difference"]}
    assert danger_diff["loss"]["count"] == 1
    assert analysis["win_loss_analysis"]["watch_distribution"][0]["count"] == 1
    assert analysis["win_loss_analysis"]["watch_trade_details"][0]["code"] == "1003"
    assert len(analysis["ml_trade_details_csv"]) == 4
    assert "expected_max_return_20d" in analysis["ml_trade_details_csv"][0]


def test_backtest_ml_analyzer_keeps_rows_when_predictions_missing(tmp_path) -> None:
    trades_path = tmp_path / "logs" / "backtests" / "profile1" / "2026-05-01_to_2026-05-31" / "trades.csv"
    _write_trades(
        trades_path,
        [{"action": "SELL", "code": "1001", "signal_date": "2026-05-11", "entry_date": "2026-05-12", "gross_profit": 1000, "gross_profit_rate": 0.1, "net_profit": 800, "net_profit_rate": 0.08}],
    )
    analyzer = BacktestMLAnalyzer(root=tmp_path, predictions_root=tmp_path / "data" / "ml" / "predictions", report_root=tmp_path / "reports" / "ml")

    analysis = analyzer.analyze_profile("profile1", "2026-05-11", "2026-05-11")

    assert analysis["join_summary"]["trade_rows"] == 1
    assert analysis["join_summary"]["joined_count"] == 0
    assert analysis["join_summary"]["missing_count"] == 1
    assert analysis["risk_label_performance"] == []
    assert analysis["warnings"] == ["prediction missing for 2026-05-11: data/ml/predictions/predictions_2026-05-11.parquet"]


def test_backtest_ml_analyzer_saves_markdown_and_json(tmp_path) -> None:
    analyzer = BacktestMLAnalyzer(root=tmp_path, report_root=tmp_path / "reports" / "ml")
    analysis = {
        "profile": "profile1",
        "period": {"start_date": "2026-05-11", "end_date": "2026-05-11"},
        "source": {"trades_csv": "trades.csv", "join_key": "signal_date + code", "note": "report-only ML join; trading logic is unchanged"},
        "join_summary": {"trade_rows": 0, "joined_count": 0, "missing_count": 0, "join_rate": None},
        "risk_label_performance": [],
        "bad_entry_probability_bands": [],
        "ml_score_top_bottom": [],
        "risk_label_ml_score_bands": [],
        "bad_probability_expected_return_matrix": [],
        "danger_expected_return_comparison": [],
        "danger_upside_probability_comparison": [],
        "virtual_filter_simulation": [],
        "virtual_position_sizing_simulation": [],
        "trade_details": [],
        "win_loss_analysis": {},
        "ml_trade_details_csv": [],
        "warnings": [],
    }

    md_path = analyzer.save_report(analysis)
    json_path = analyzer.save_json(analysis)
    win_loss_md_path = analyzer.save_win_loss_report(analysis)
    win_loss_json_path = analyzer.save_win_loss_json(analysis)
    csv_path = analyzer.save_ml_trades_csv(analysis)

    assert md_path.exists()
    assert json_path.exists()
    assert win_loss_md_path.exists()
    assert win_loss_json_path.exists()
    assert csv_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "trading logic is unchanged" in content
    assert "## Risk Label x ML Score Band" in content
    assert "## Virtual ML Filter Simulation" in content
    assert "## Virtual ML Position Sizing Simulation" in content
    assert "## Trade Details With ML Predictions" in content
    assert "## Win vs Loss ML Averages" in win_loss_md_path.read_text(encoding="utf-8")
