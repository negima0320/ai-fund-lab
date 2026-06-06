from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase1 import PROFILE
from ml.portfolio_manager_phase1 import PortfolioManagerPhase1Simulation


def _write_fake_logs(root: Path) -> None:
    out = root / "logs" / "backtests" / PROFILE / "2023-01-01_to_2026-05-31"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "trade_id": "t1_sell",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "exit_date": "2023-01-20",
                "code": "1001",
                "net_profit": 100.0,
                "net_profit_rate": 0.1,
                "holding_days": 10,
                "exit_reason": "test",
            },
            {
                "action": "SELL",
                "trade_id": "t2_sell",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "exit_date": "2023-01-20",
                "code": "1002",
                "net_profit": 50.0,
                "net_profit_rate": 0.05,
                "holding_days": 10,
                "exit_reason": "test",
            },
            {
                "action": "SELL",
                "trade_id": "t3_sell",
                "signal_date": "2023-01-05",
                "entry_date": "2023-01-06",
                "exit_date": "2023-01-23",
                "code": "1003",
                "net_profit": -20.0,
                "net_profit_rate": -0.02,
                "holding_days": 10,
                "exit_reason": "test",
            },
        ]
    ).to_csv(out / "trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "code": "1001",
                "name": "A",
                "decision": "BUY",
                "final_amount": 100000.0,
                "risk_adjusted_score": 0.10,
                "expected_return_10d": 0.05,
                "expected_max_return_20d": 0.15,
                "swing_success_probability_20d": 0.60,
                "bad_entry_probability_10d": 0.20,
            },
            {
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "code": "1002",
                "name": "B",
                "decision": "BUY",
                "final_amount": 100000.0,
                "risk_adjusted_score": 0.02,
                "expected_return_10d": 0.02,
                "expected_max_return_20d": 0.08,
                "swing_success_probability_20d": 0.30,
                "bad_entry_probability_10d": 0.80,
            },
            {
                "signal_date": "2023-01-05",
                "entry_date": "2023-01-06",
                "code": "1003",
                "name": "C",
                "decision": "BUY",
                "final_amount": 100000.0,
                "risk_adjusted_score": -0.30,
                "expected_return_10d": -0.01,
                "expected_max_return_20d": 0.03,
                "swing_success_probability_20d": 0.10,
                "bad_entry_probability_10d": 0.60,
            },
            {
                "signal_date": "2023-01-05",
                "entry_date": "2023-01-06",
                "code": "9999",
                "name": "Skip",
                "decision": "SKIP",
                "final_amount": 0.0,
                "risk_adjusted_score": -0.40,
                "expected_return_10d": -0.02,
                "expected_max_return_20d": 0.02,
                "swing_success_probability_20d": 0.05,
                "bad_entry_probability_10d": 0.70,
            },
        ]
    ).to_csv(out / "purchase_audit.csv", index=False)


def test_portfolio_manager_builds_candidate_set_and_weights(tmp_path: Path) -> None:
    _write_fake_logs(tmp_path)
    simulation = PortfolioManagerPhase1Simulation(root=tmp_path)
    candidates = simulation.build_candidate_set()

    assert len(candidates) == 4
    group = candidates[candidates["signal_date"].eq(pd.Timestamp("2023-01-04"))]
    risk_weights = simulation.weights_for_rule(group, "risk_adjusted_weight")
    bad_weights = simulation.weights_for_rule(group, "bad_entry_defensive_weight")

    assert round(float(risk_weights.sum()), 6) == 1.0
    assert risk_weights.loc[group.index[group["code"].eq("1001")][0]] > risk_weights.loc[group.index[group["code"].eq("1002")][0]]
    assert bad_weights.loc[group.index[group["code"].eq("1001")][0]] > bad_weights.loc[group.index[group["code"].eq("1002")][0]]
    assert simulation._cash_reserve_rate("cash_reserve_dynamic", -0.10) == 0.10
    assert simulation._cash_reserve_rate("cash_reserve_dynamic", -0.30) == 0.50


def test_portfolio_manager_simulation_saves_outputs(tmp_path: Path) -> None:
    _write_fake_logs(tmp_path)
    simulation = PortfolioManagerPhase1Simulation(root=tmp_path)

    result = simulation.build()
    paths = simulation.save(result)

    assert result["candidate_rows"] == 4
    assert result["closed_trade_rows"] == 3
    assert {row["portfolio_rule"] for row in result["summary"]} >= {"baseline", "risk_adjusted_weight", "cash_reserve_dynamic"}
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.daily_allocations_csv.exists()
    assert paths.trade_allocations_csv.exists()
