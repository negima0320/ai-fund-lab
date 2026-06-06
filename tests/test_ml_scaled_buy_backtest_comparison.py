from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.scaled_buy_backtest_comparison import ScaledBuyBacktestComparison


PERIOD = "2023-01-01_to_2026-05-31"


def _write_backtest(root: Path, profile: str, trades: list[dict], all_trades: list[dict] | None = None) -> None:
    directory = root / "logs" / "backtests" / profile / PERIOD
    directory.mkdir(parents=True)
    pd.DataFrame(trades).to_csv(directory / "trades.csv", index=False)
    payload = {
        "final_assets": 1_100_000,
        "net_cumulative_profit": sum(float(row.get("net_profit", 0) or 0) for row in trades),
        "win_rate": 0.5,
        "profit_factor": 1.2,
        "max_drawdown": -0.1,
        "all_trades": all_trades or [],
    }
    (directory / "backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    pd.DataFrame(
        [
            {"date": "2026-03-31", "total_assets": 1_100_000, "net_cumulative_profit": payload["net_cumulative_profit"]},
        ]
    ).to_csv(directory / "summary.csv", index=False)


def test_scaled_buy_backtest_comparison_reports_scaled_67400(tmp_path: Path) -> None:
    profiles = [
        "rookie_dealer_02_v2_66_ml_ranked",
        "rookie_dealer_02_v2_68_ml_ranked_exit_ai_050",
        "rookie_dealer_02_v2_70_ml_ranked_exit_ai_060",
        "rookie_dealer_02_v2_71_ml_ranked_exit_ai_050_scaled_buy",
    ]
    base_trade = {
        "action": "SELL",
        "code": "10010",
        "entry_date": "2026-03-01",
        "exit_date": "2026-03-11",
        "shares": 100,
        "net_profit": 1000,
        "holding_days": 5,
    }
    for profile in profiles[:-1]:
        _write_backtest(tmp_path, profile, [base_trade])
    scaled_trade = {
        "action": "SELL",
        "code": "67400",
        "entry_date": "2026-03-09",
        "exit_date": "2026-03-19",
        "shares": 19100,
        "net_profit": 400000,
        "net_profit_rate": 0.45,
        "holding_days": 5,
        "exit_reason": "最大保有期間到達",
        "scaled_buy_triggered": True,
        "original_planned_shares": 20200,
        "scaled_shares": 19100,
        "original_amount": 949400,
        "scaled_amount": 897700,
        "scale_reason": "daily_buy_limit",
    }
    _write_backtest(tmp_path, profiles[-1], [scaled_trade])

    comparison = ScaledBuyBacktestComparison(root=tmp_path, profiles=profiles)
    result = comparison.build()
    paths = comparison.save(result)

    assert result["scaled_buy_summary"]["scaled_buy_trigger_count"] == 1
    assert result["scaled_buy_summary"]["scaled_buy_profit"] == 400000
    assert result["focus_67400"]["bought"] is True
    assert result["focus_67400"]["shares"] == 19100
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.scaled_buy_trades_csv.exists()
