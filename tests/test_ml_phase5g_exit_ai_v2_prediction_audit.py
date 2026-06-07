from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ml.phase5g_exit_ai_v2_prediction_audit import Phase5GExitAIV2PredictionAudit


FEATURES = ["close", "volume", "return_5d", "entry_price"]


def _write_fixture(root: Path) -> None:
    log_dir = root / "logs/backtests/rookie_dealer_02_v2_78_pm_aware_order_fallback_w025/2023-01-01_to_2026-05-31"
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "code": "11110",
                "entry_date": "2023-01-04",
                "exit_date": "2023-01-10",
                "exit_reason": "Exit AI avoid_loss_5d",
                "exit_ai_triggered": True,
                "net_profit": -1000,
                "net_profit_rate": -0.02,
                "pm_score": 0.3,
                "pm_multiplier": 1.15,
            },
            {
                "trade_id": "t2",
                "code": "22220",
                "entry_date": "2023-01-04",
                "exit_date": "2023-01-11",
                "exit_reason": "利確",
                "exit_ai_triggered": False,
                "net_profit": 2000,
                "net_profit_rate": 0.03,
                "pm_score": -0.1,
                "pm_multiplier": 0.8,
            },
        ]
    ).to_csv(log_dir / "trades.csv", index=False)

    dataset_path = root / "data/ml/exit_ai_v2/exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "code": "11110",
            "as_of_date": "2023-01-10",
            "close": 100,
            "volume": 1000,
            "return_5d": -0.01,
            "entry_price": 102,
            "future_return_5d": 0.04,
            "future_return_10d": 0.02,
            "future_return_20d": -0.01,
            "exit_quality_score": -0.04,
        },
        {
            "code": "22220",
            "as_of_date": "2023-01-11",
            "close": 200,
            "volume": 2000,
            "return_5d": 0.02,
            "entry_price": 190,
            "future_return_5d": -0.05,
            "future_return_10d": -0.04,
            "future_return_20d": -0.03,
            "exit_quality_score": 0.05,
        },
    ]
    pd.DataFrame(rows).to_parquet(dataset_path, index=False)

    model_dir = root / "models/ml/exit_ai_v2/candidate_v2_api_only"
    model_dir.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(
        {
            "close": [100, 200, 110, 220],
            "volume": [1000, 2000, 1100, 2200],
            "return_5d": [-0.01, 0.02, -0.02, 0.03],
            "entry_price": [102, 190, 111, 210],
        }
    )
    y = [0, 1, 0, 1]
    model = HistGradientBoostingClassifier(max_iter=5, random_state=42).fit(train, y)
    joblib.dump(model, model_dir / "exit_quality_top_decile_classifier.joblib")
    metadata = {
        "feature_columns": FEATURES,
        "train_top_decile_threshold": 0.046,
        "current_model_overwrite_forbidden": True,
    }
    preprocess = {
        "feature_columns": FEATURES,
        "numeric_columns": FEATURES,
        "categorical_columns": [],
        "medians": {column: float(train[column].median()) for column in FEATURES},
        "modes": {},
        "missing_indicator_columns": [],
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (model_dir / "preprocess.json").write_text(json.dumps(preprocess), encoding="utf-8")
    (model_dir / "feature_columns.json").write_text(json.dumps(FEATURES), encoding="utf-8")
    (root / "models/ml/exit/current_v2_66").mkdir(parents=True, exist_ok=True)


def test_phase5g_scores_existing_v278_logs_without_overwriting_current_model(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = Phase5GExitAIV2PredictionAudit(tmp_path).build_report()

    assert result["metadata"]["audit_only"] is True
    assert result["metadata"]["full_backtest_executed"] is False
    assert result["prediction_summary"]["prediction_available_count"] == 2
    assert result["prediction_summary"]["top_decile_count"] == 1
    assert result["existing_exit_ai_comparison"]["disagreement_count"] >= 1
    assert result["high_pm_early_exit_audit"]["high_pm_exit_count"] == 1
    assert result["leakage_integrity_audit"]["current_model_not_overwritten"] is True
    assert result["leakage_integrity_audit"]["feature_schema_matches_training_metadata"] is True


def test_phase5g_saves_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    audit = Phase5GExitAIV2PredictionAudit(tmp_path)
    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 5-G" in paths.markdown.read_text(encoding="utf-8")
