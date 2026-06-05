from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.exit_analysis import MLExitAnalyzer


def _write_trade_log(root: Path, net_profit: float = -2000.0) -> None:
    directory = root / "logs" / "backtests" / "rookie_dealer_02_v2_66_ml_ranked" / "2023-01-01_to_2023-01-31"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "signal_date": "2023-01-02",
                "entry_date": "2023-01-02",
                "exit_date": "2023-01-09",
                "code": "1001",
                "entry_price": 100,
                "exit_price": 80,
                "shares": 100,
                "net_profit": net_profit,
                "net_profit_rate": -0.2,
                "holding_days": 6,
            }
        ]
    ).to_csv(directory / "trades.csv", index=False)


def _write_price_cache(cache_root: Path) -> None:
    closes = {
        "2023-01-02": 100,
        "2023-01-03": 99,
        "2023-01-04": 97,
        "2023-01-05": 95,
        "2023-01-06": 90,
        "2023-01-09": 80,
    }
    for date_text, close in closes.items():
        path = cache_root / "prices" / f"{date_text}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "date": date_text,
                            "code": "1001",
                            "open": close,
                            "high": close,
                            "low": close,
                            "close": close,
                            "volume": 1000,
                            "turnover_value": 100000,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )


def _analyzer(root: Path, cache_root: Path, predictions_root: Path, report_root: Path) -> MLExitAnalyzer:
    return MLExitAnalyzer(
        root=root,
        profile="rookie_dealer_02_v2_66_ml_ranked",
        start_date="2023-01-01",
        end_date="2023-01-31",
        cache_root=cache_root,
        predictions_root=predictions_root,
        report_root=report_root,
    )


def test_exit_analysis_simulates_5d_risk_adjusted_exit(tmp_path: Path) -> None:
    _write_trade_log(tmp_path)
    cache_root = tmp_path / "data" / "cache" / "jquants"
    predictions_root = tmp_path / "data" / "ml" / "walk_forward_predictions"
    report_root = tmp_path / "reports" / "ml"
    _write_price_cache(cache_root)
    predictions_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2023-01-06",
                "code": "1001",
                "expected_return_10d": -0.1,
                "expected_max_return_20d": 0.02,
                "swing_success_probability_20d": 0.2,
                "bad_entry_probability_10d": 0.3,
            }
        ]
    ).to_parquet(predictions_root / "predictions_2023-01-06.parquet", index=False)

    result = _analyzer(tmp_path, cache_root, predictions_root, report_root).build()

    rule_a = next(row for row in result["rules"] if row["rule"].startswith("Exit Rule A"))
    assert rule_a["exit_changed_count"] == 1
    assert rule_a["improved_trade_count"] == 1
    assert rule_a["total_profit"] > result["baseline"]["total_profit"]


def test_exit_analysis_missing_prediction_does_not_trigger_bad_entry_rule(tmp_path: Path) -> None:
    _write_trade_log(tmp_path)
    cache_root = tmp_path / "data" / "cache" / "jquants"
    predictions_root = tmp_path / "data" / "ml" / "walk_forward_predictions"
    report_root = tmp_path / "reports" / "ml"
    _write_price_cache(cache_root)

    result = _analyzer(tmp_path, cache_root, predictions_root, report_root).build()

    rule_c = next(row for row in result["rules"] if row["rule"].startswith("Exit Rule C"))
    assert rule_c["exit_changed_count"] == 0
    assert rule_c["total_profit"] == result["baseline"]["total_profit"]


def test_exit_analysis_saves_reports(tmp_path: Path) -> None:
    _write_trade_log(tmp_path)
    cache_root = tmp_path / "data" / "cache" / "jquants"
    predictions_root = tmp_path / "data" / "ml" / "walk_forward_predictions"
    report_root = tmp_path / "reports" / "ml"
    _write_price_cache(cache_root)
    analyzer = _analyzer(tmp_path, cache_root, predictions_root, report_root)
    result = analyzer.build()

    paths = analyzer.save(result)

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.trades_csv.exists()
