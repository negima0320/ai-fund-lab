from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.position_sizing_phase1 import PositionSizingPhase1Simulation


PROFILE = "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue"


def _write_fake_backtest(root: Path) -> None:
    out = root / "logs" / "backtests" / PROFILE / "2023-01-01_to_2026-05-31"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "trade_id": "t1_sell",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "exit_date": "2023-01-10",
                "code": "1001",
                "name": "A",
                "net_profit": 100.0,
                "gross_profit": 100.0,
                "net_profit_rate": 0.10,
            },
            {
                "action": "SELL",
                "trade_id": "t2_sell",
                "signal_date": "2023-01-05",
                "entry_date": "2023-01-06",
                "exit_date": "2023-01-11",
                "code": "1002",
                "name": "B",
                "net_profit": -50.0,
                "gross_profit": -50.0,
                "net_profit_rate": -0.05,
            },
            {
                "action": "SELL",
                "trade_id": "t3_sell",
                "signal_date": "2023-01-06",
                "entry_date": "2023-01-10",
                "exit_date": "2023-01-12",
                "code": "67400",
                "name": "C",
                "net_profit": 200.0,
                "gross_profit": 200.0,
                "net_profit_rate": 0.20,
            },
        ]
    ).to_csv(out / "trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "decision": "BUY",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "code": "1001",
                "risk_adjusted_score": 0.12,
                "expected_return_10d": 0.06,
                "bad_entry_probability_10d": 0.20,
            },
            {
                "decision": "BUY",
                "signal_date": "2023-01-05",
                "entry_date": "2023-01-06",
                "code": "1002",
                "risk_adjusted_score": -0.05,
                "expected_return_10d": 0.005,
                "bad_entry_probability_10d": 0.90,
            },
            {
                "decision": "BUY",
                "signal_date": "2023-01-06",
                "entry_date": "2023-01-10",
                "code": "67400",
                "risk_adjusted_score": 0.02,
                "expected_return_10d": 0.03,
                "bad_entry_probability_10d": 0.60,
            },
        ]
    ).to_csv(out / "purchase_audit.csv", index=False)


def test_position_sizing_phase1_builds_rules_and_saves(tmp_path: Path) -> None:
    _write_fake_backtest(tmp_path)
    simulation = PositionSizingPhase1Simulation(root=tmp_path, profiles=[PROFILE])

    result = simulation.build()
    paths = simulation.save(result)

    assert result["join_summary"][0]["ml_join_rate"] == 1.0
    rows = {(row["profile"], row["sizing_rule"]): row for row in result["summary"]}
    assert rows[(PROFILE, "baseline")]["adjusted_net_profit"] == 250.0
    assert rows[(PROFILE, "score_simple_boost")]["adjusted_net_profit"] == 325.0
    assert rows[(PROFILE, "score_simple_boost")]["profit_delta"] == 75.0
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.trades_csv.exists()
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["best_by_net_profit"]["sizing_rule"] != "baseline"
