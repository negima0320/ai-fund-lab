from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.backtest_diagnostics import MLBacktestDiagnostics


def _write_backtest(root: Path, profile: str, rows: list[dict], all_trades: list[dict] | None = None) -> None:
    period = "2023-01-01_to_2023-01-31"
    directory = root / "logs" / "backtests" / profile / period
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "trades.csv", index=False)
    summary = {
        "final_assets": 1_000_000 + sum(row.get("net_profit", 0) for row in rows),
        "net_cumulative_profit": sum(row.get("net_profit", 0) for row in rows),
        "win_rate": sum(1 for row in rows if row.get("net_profit", 0) > 0) / len(rows),
        "profit_factor": 2.0,
        "max_drawdown": -0.1,
        "total_trades": len(rows),
        "average_holding_days": 5,
        "all_trades": all_trades if all_trades is not None else rows,
    }
    (directory / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_backtest_diagnostics_builds_reports(tmp_path: Path) -> None:
    profiles = ["rookie_dealer_02_v2_65", "rookie_dealer_02_v2_66_ml_ranked", "rookie_dealer_02_v2_67_ml_standalone"]
    base_rows = [
        {"action": "SELL", "code": "1001", "signal_date": "2023-01-04", "exit_date": "2023-01-10", "net_profit": -1000, "holding_days": 5},
        {"action": "SELL", "code": "1002", "signal_date": "2023-01-05", "exit_date": "2023-01-11", "net_profit": 2000, "holding_days": 5},
    ]
    ranked_rows = [
        {"action": "SELL", "code": "1001", "signal_date": "2023-01-04", "exit_date": "2023-01-10", "net_profit": 3000, "holding_days": 5},
        {"action": "SELL", "code": "1003", "signal_date": "2023-01-05", "exit_date": "2023-01-11", "net_profit": 4000, "holding_days": 5},
    ]
    standalone_rows = [
        {"action": "SELL", "code": "1004", "signal_date": "2023-01-04", "exit_date": "2023-01-25", "net_profit": 500, "holding_days": 20},
    ]
    _write_backtest(tmp_path, profiles[0], base_rows)
    _write_backtest(
        tmp_path,
        profiles[1],
        ranked_rows,
        all_trades=[
            {"action": "BUY", "code": "1001", "signal_date": "2023-01-04", "risk_adjusted_score": 0.1},
            *ranked_rows,
        ],
    )
    _write_backtest(
        tmp_path,
        profiles[2],
        standalone_rows,
        all_trades=[
            {"action": "SKIP_BUY", "code": "1005", "skipped_reason": "max_positions_limit"},
            {"action": "BUY", "code": "1004", "signal_date": "2023-01-04", "risk_adjusted_score": 0.2},
            *standalone_rows,
        ],
    )
    prediction_path = tmp_path / "data" / "ml" / "walk_forward_predictions" / "predictions_2023-01-04.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"code": "1001", "expected_return_10d": 0.2, "bad_entry_probability_10d": 0.1}]).to_parquet(prediction_path, index=False)
    scored_path = tmp_path / "data" / "processed" / profiles[1] / "scored_candidates_2023-01-04.json"
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    scored_path.write_text(json.dumps({"scores": [{"code": "1001", "selected": True}, {"code": "9999", "selected": False}]}), encoding="utf-8")

    diagnostics = MLBacktestDiagnostics(root=tmp_path, profiles=profiles, start_date="2023-01-01", end_date="2023-01-31")
    result = diagnostics.build()
    paths = diagnostics.save(result)

    assert result["monthly_diff_v66_vs_v65"]["improved_months"] == 1
    assert result["ml_ranked_analysis"]["candidate_prediction_coverage"]["missing_prediction_count"] == 1
    assert result["standalone_analysis"]["skip_counts"]["max_positions_limit"] == 1
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.monthly_csv.exists()
    assert paths.code_csv.exists()
