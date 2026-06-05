from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.exit_avoid_loss_simulation import ExitAvoidLossSimulator


class FakeAvoidLossModel:
    def predict_proba(self, features: pd.DataFrame):
        probabilities = []
        for _, row in features.iterrows():
            probability = 0.7 if float(row["holding_days"]) == 2 else 0.2
            probabilities.append([1 - probability, probability])
        return probabilities


class FakeSimulator(ExitAvoidLossSimulator):
    def _load_model(self, path: Path):
        return FakeAvoidLossModel()


FEATURE_COLUMNS = [
    "holding_days",
    "entry_price",
    "current_close",
    "unrealized_return",
    "max_unrealized_return_so_far",
    "min_unrealized_return_so_far",
    "drawdown_from_peak",
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "risk_adjusted_score",
    "volume",
    "turnover_value",
    "return_5d",
    "return_10d",
    "ma25_gap",
    "daily_range_ratio",
]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset_path = tmp_path / "dataset.parquet"
    model_dir = tmp_path / "model"
    trades_path = tmp_path / "trades.csv"
    report_root = tmp_path / "reports"
    model_dir.mkdir(parents=True)
    (model_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS), encoding="utf-8")
    rows = []
    for trade_id, actual_exit, actual_profit, avoid_loss in [
        ("t1", 80.0, -2000.0, True),
        ("t2", 120.0, 1593.7, False),
    ]:
        for holding_days, close in [(2, 95.0), (3, actual_exit)]:
            row = {
                "trade_id": trade_id,
                "code": "1001",
                "entry_date": "2023-01-02",
                "current_date": f"2023-01-0{holding_days + 1}",
                "actual_exit_date": "2023-01-04",
                "holding_days": holding_days,
                "entry_price": 100.0,
                "current_close": close,
                "unrealized_return": close / 100.0 - 1,
                "max_unrealized_return_so_far": max(0.0, close / 100.0 - 1),
                "min_unrealized_return_so_far": min(0.0, close / 100.0 - 1),
                "drawdown_from_peak": 0.0,
                "remaining_days_to_actual_exit": 3 - holding_days,
                "prediction_joined": True,
                "expected_return_10d": 0.1,
                "expected_max_return_20d": 0.2,
                "swing_success_probability_20d": 0.3,
                "bad_entry_probability_10d": 0.4,
                "risk_adjusted_score": -0.1,
                "volume": 1000,
                "turnover_value": 100000,
                "return_5d": 0.01,
                "return_10d": 0.02,
                "ma25_gap": 0.03,
                "daily_range_ratio": 0.04,
                "avoid_loss_5d": avoid_loss,
            }
            rows.append(row)
    pd.DataFrame(rows).to_parquet(dataset_path, index=False)
    pd.DataFrame(
        [
            {"action": "SELL", "trade_id": "t1", "code": "1001", "entry_date": "2023-01-02", "exit_date": "2023-01-04", "entry_price": 100.0, "exit_price": 80.0, "shares": 100, "net_profit": actual_profit, "holding_days": 3}
            for actual_profit in [-2000.0]
        ]
        + [
            {"action": "SELL", "trade_id": "t2", "code": "1001", "entry_date": "2023-01-02", "exit_date": "2023-01-04", "entry_price": 100.0, "exit_price": 120.0, "shares": 100, "net_profit": 1593.7, "holding_days": 3}
        ]
    ).to_csv(trades_path, index=False)
    return dataset_path, model_dir, trades_path, report_root


def test_avoid_loss_simulation_uses_threshold_and_actual_trade_baseline(tmp_path: Path) -> None:
    dataset_path, model_dir, trades_path, report_root = _write_inputs(tmp_path)
    simulator = FakeSimulator(root=tmp_path, dataset_path=dataset_path, model_dir=model_dir, trades_path=trades_path, report_root=report_root)

    result = simulator.build(thresholds=[0.5, 0.8])

    assert result["baseline"]["total_profit"] == pytest.approx(-406.3)
    low = result["results"][0]
    high = result["results"][1]
    assert low["threshold"] == 0.5
    assert low["exit_changed_count"] == 2
    assert low["improved_trade_count"] == 1
    assert low["worsened_trade_count"] == 1
    assert low["precision"] == 0.5
    assert low["recall"] == 1.0
    assert high["exit_changed_count"] == 0
    assert high["profit_delta"] == 0


def test_avoid_loss_simulation_saves_reports(tmp_path: Path) -> None:
    dataset_path, model_dir, trades_path, report_root = _write_inputs(tmp_path)
    simulator = FakeSimulator(root=tmp_path, dataset_path=dataset_path, model_dir=model_dir, trades_path=trades_path, report_root=report_root)
    result = simulator.build(thresholds=[0.5])

    paths = simulator.save(result)

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.trades_csv.exists()
