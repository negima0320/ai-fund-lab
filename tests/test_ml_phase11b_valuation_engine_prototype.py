from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase11b_valuation_engine_prototype import Phase11BOptions, Phase11BValuationEnginePrototype


def _write_dataset(root: Path) -> None:
    path = root / "data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(80):
        year = 2023 if idx < 30 else 2024 if idx < 55 else 2025
        date = pd.Timestamp(f"{year}-01-04") + pd.Timedelta(days=idx % 20)
        quality = (idx % 10) / 9.0
        rows.append(
            {
                "date": date,
                "code": f"{10000 + idx}",
                "return_5d": quality - 0.5,
                "ma25_gap": quality / 10,
                "volume_ratio_5d": 1.0 + quality,
                "EPS": 10 + quality,
                "is_near_earnings": idx % 2 == 0,
                "risk_adjusted_score": quality,
                "expected_return": quality / 10,
                "stock_selection_rank_score": quality * 2,
                "candidate_strength": quality * 3,
                "risk_adjusted_score_percentile_in_day": quality,
                "selected_count_in_day": 99,
                "pm_multiplier": 1.3,
                "cash_leak": 1000,
                "future_return_20d": quality / 5,
                "future_max_return_20d": quality / 4,
                "future_max_drawdown_20d": -0.1 + quality / 20,
                "opportunity_value_20d": quality / 4 - abs(-0.1 + quality / 20),
                "opportunity_top_decile_20d": 1 if quality >= 0.9 else 0,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase11b_feature_policy_excludes_future_and_forbidden_columns(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    trainer = Phase11BValuationEnginePrototype(
        tmp_path,
        options=Phase11BOptions(max_train_rows=40, max_iter=4, save_model=False),
    )
    dataset = trainer.load_dataset()
    features = trainer.extract_feature_columns(dataset)
    leakage = trainer.leakage_checklist(features)

    assert "future_return_20d" not in features
    assert "future_max_return_20d" not in features
    assert "future_max_drawdown_20d" not in features
    assert "opportunity_value_20d" not in features
    assert "opportunity_top_decile_20d" not in features
    assert "selected_count_in_day" not in features
    assert "pm_multiplier" not in features
    assert "cash_leak" not in features
    assert leakage["leakage_risk"] == "low"
    assert leakage["blocking_issues"] == []


def test_phase11b_trains_saves_report_and_candidate_model(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    trainer = Phase11BValuationEnginePrototype(
        tmp_path,
        options=Phase11BOptions(max_train_rows=50, max_iter=4, save_model=True),
    )
    paths = trainer.run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.model_dir is not None
    assert (paths.model_dir / "opportunity_value_20d_regressor.joblib").exists()
    assert (paths.model_dir / "opportunity_top_decile_20d_classifier.joblib").exists()
    assert (paths.model_dir / "feature_columns.json").exists()
    features = json.loads((paths.model_dir / "feature_columns.json").read_text(encoding="utf-8"))
    assert "selected_count_in_day" not in features
    assert "pm_multiplier" not in features
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "11-B"
    assert loaded["metadata"]["backtest_executed"] is False
    assert loaded["leakage_checklist"]["leakage_risk"] == "low"
    assert loaded["recommendation"]["ready_for_phase11c"] is True
