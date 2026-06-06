from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ml.backtest_exit_ai import apply_exit_ai_to_plan, get_exit_ai_advisor, update_unrealized_extrema


class FakeAvoidLossModel:
    def predict_proba(self, frame: pd.DataFrame):
        probability = float(frame["bad_entry_probability_10d"].iloc[0])
        return [[1.0 - probability, probability]]


def test_exit_ai_uses_current_date_prediction_and_triggers(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    feature_columns = [
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
    (model_dir / "feature_columns.json").write_text(json.dumps(feature_columns), encoding="utf-8")
    joblib.dump(FakeAvoidLossModel(), model_dir / "avoid_loss_5d_classification.joblib")

    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2026-05-15",
                "code": "1234",
                "expected_return_10d": 0.02,
                "expected_max_return_20d": 0.08,
                "swing_success_probability_20d": 0.55,
                "bad_entry_probability_10d": 0.61,
            }
        ]
    ).to_parquet(predictions_root / "predictions_2026-05-15.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-05-16",
                "code": "1234",
                "expected_return_10d": 0.02,
                "expected_max_return_20d": 0.08,
                "swing_success_probability_20d": 0.55,
                "bad_entry_probability_10d": 0.10,
            }
        ]
    ).to_parquet(predictions_root / "predictions_2026-05-16.parquet", index=False)

    config = {
        "ml_exit_ai": {
            "enabled": True,
            "model_dir": str(model_dir),
            "prediction_root": str(predictions_root),
            "threshold": 0.60,
        }
    }
    advisor = get_exit_ai_advisor(config, root=tmp_path)
    plan = {
        "exit_reason": "",
        "exit_price": None,
        "mark_profit_rate": 0.03,
        "intended_exit_price": None,
    }
    updated = apply_exit_ai_to_plan(
        plan,
        advisor,
        position={"code": "1234", "entry_price": 100.0, "max_unrealized_return_so_far": 0.01, "min_unrealized_return_so_far": -0.02},
        market={"close": 103.0, "high": 105.0, "low": 101.0, "volume": 1000, "turnover_value": 100000000, "ma25": 100.0},
        trade_date="2026-05-15",
        current_price=103.0,
        holding_days=3,
    )

    assert updated["exit_reason"] == "Exit AI avoid_loss_5d"
    assert updated["exit_ai_triggered"] is True
    assert updated["exit_ai_probability"] == 0.61
    assert updated["exit_price"] == 103.0


def test_exit_ai_does_not_override_existing_exit_reason(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "feature_columns.json").write_text(json.dumps(["bad_entry_probability_10d"]), encoding="utf-8")
    joblib.dump(FakeAvoidLossModel(), model_dir / "avoid_loss_5d_classification.joblib")
    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    pd.DataFrame([{"code": "1234", "bad_entry_probability_10d": 0.90}]).to_parquet(
        predictions_root / "predictions_2026-05-15.parquet",
        index=False,
    )
    advisor = get_exit_ai_advisor(
        {"ml_exit_ai": {"enabled": True, "model_dir": str(model_dir), "prediction_root": str(predictions_root), "threshold": 0.5}},
        root=tmp_path,
    )

    updated = apply_exit_ai_to_plan(
        {"exit_reason": "損切り", "exit_price": 97.0, "mark_profit_rate": -0.03, "intended_exit_price": 97.0},
        advisor,
        position={"code": "1234", "entry_price": 100.0},
        market={},
        trade_date="2026-05-15",
        current_price=99.0,
        holding_days=2,
    )

    assert updated["exit_reason"] == "損切り"
    assert updated["exit_price"] == 97.0
    assert updated["exit_ai_signal"] is True
    assert updated["exit_ai_triggered"] is False


def test_update_unrealized_extrema() -> None:
    assert update_unrealized_extrema({}, 0.1) == (0.1, 0.1)
    assert update_unrealized_extrema({"max_unrealized_return_so_far": 0.2, "min_unrealized_return_so_far": -0.1}, 0.05) == (0.2, -0.1)
