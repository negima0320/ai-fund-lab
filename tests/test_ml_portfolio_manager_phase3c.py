from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase3c import PortfolioManagerPhase3CLightBacktest


def _evaluator(tmp_path: Path) -> PortfolioManagerPhase3CLightBacktest:
    return PortfolioManagerPhase3CLightBacktest(root=tmp_path)


def _fake_trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": pd.Timestamp("2023-01-04"),
                "entry_date": pd.Timestamp("2023-01-05"),
                "exit_date": pd.Timestamp("2023-01-20"),
                "code": "1001",
                "actual_buy_amount": 700000.0,
                "actual_net_profit": 70000.0,
                "actual_shares": 100,
                "decision": "BUY",
                "pm_high_conviction_probability": 0.80,
                "pm_avoid_probability": 0.10,
            },
            {
                "signal_date": pd.Timestamp("2023-01-04"),
                "entry_date": pd.Timestamp("2023-01-05"),
                "exit_date": pd.Timestamp("2023-01-20"),
                "code": "1002",
                "actual_buy_amount": 400000.0,
                "actual_net_profit": -20000.0,
                "actual_shares": 100,
                "decision": "BUY",
                "pm_high_conviction_probability": 0.50,
                "pm_avoid_probability": 0.80,
            },
            {
                "signal_date": pd.Timestamp("2023-02-01"),
                "entry_date": pd.Timestamp("2023-02-02"),
                "exit_date": pd.Timestamp("2023-02-15"),
                "code": "1003",
                "actual_buy_amount": 200000.0,
                "actual_net_profit": 10000.0,
                "actual_shares": 100,
                "decision": "SCALED_BUY",
                "pm_high_conviction_probability": pd.NA,
                "pm_avoid_probability": pd.NA,
            },
        ]
    )


def test_phase3c_multiplier_rules() -> None:
    evaluator = PortfolioManagerPhase3CLightBacktest()

    assert evaluator.multiplier_for_rule(0.80, 0.20, "phase3c_01_high_only") == 1.30
    assert evaluator.multiplier_for_rule(0.50, 0.80, "phase3c_02_avoid_only") == 0.60
    assert evaluator.multiplier_for_rule(0.80, 0.20, "phase3c_03_high_minus_avoid") == 1.30
    assert evaluator.multiplier_for_rule(0.40, 0.80, "phase3c_04_avoid_strong") == 0.50
    assert evaluator.multiplier_for_rule(0.80, 0.20, "phase3c_05_high_strong") == 1.35
    assert evaluator.multiplier_for_rule(pd.NA, pd.NA, "phase3c_03_high_minus_avoid") == 1.0


def test_phase3c_daily_buy_limit_scales_down(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    trades = _fake_trades()

    simulated = evaluator.simulate_rule(trades, "phase3c_01_high_only", "test")
    day = simulated[simulated["signal_date"].eq(pd.Timestamp("2023-01-04"))]

    assert round(float(day["pm_planned_amount"].sum()), 2) == 1_230_000.0
    assert round(float(day["pm_final_amount"].sum()), 2) == 900_000.0
    assert (day["pm_daily_limit_scale"] < 1.0).all()


def test_phase3c_summary_and_report_generation(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    trades = _fake_trades()
    result = {
        "profile": evaluator.profile,
        "period": {"start_date": evaluator.start_date, "end_date": evaluator.end_date},
        "dataset_path": str(evaluator.dataset_path),
        "model_dir": str(evaluator.model_dir),
        "leakage_check": ["PASS"],
        "summary": [
            evaluator._summary("baseline", "baseline", evaluator.simulate_rule(trades, "baseline", "baseline")),
            evaluator._summary("phase3c_01_high_only", "test", evaluator.simulate_rule(trades, "phase3c_01_high_only", "test")),
        ],
        "best_by_profit_factor": {},
        "best_by_net_profit": {},
        "best_by_drawdown": {},
        "recommendation": ["ok"],
        "trades": evaluator.simulate_rule(trades, "baseline", "baseline").to_dict(orient="records"),
    }

    paths = evaluator.save(result)

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.trades_csv.exists()
    assert "Strategy Comparison" in paths.markdown.read_text(encoding="utf-8")
