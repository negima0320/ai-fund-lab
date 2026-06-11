from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ml.phase11g_out_of_sample_check import Phase11GOutOfSampleCheck


class _FakeClassifier:
    def predict_proba(self, x):
        values = pd.to_numeric(x["risk_adjusted_score"], errors="coerce").fillna(0.0).clip(0.01, 0.99)
        return [[1.0 - value, value] for value in values]


def _write_fixture(root: Path) -> None:
    dataset_path = root / "data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet"
    model_dir = root / "models/ml/valuation_engine/candidate_phase11b"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for day_index, date in enumerate(pd.bdate_range("2024-01-04", periods=35)):
        for rank in range(8):
            quality = (7 - rank) / 7
            code = f"{10000 + rank}"
            close = 100 + rank * 4
            if rank == 0 and day_index >= 5:
                close = 88
            if rank == 1 and day_index >= 4:
                close = 114
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": float(close),
                    "turnover_value": 1_000_000 + rank,
                    "stock_selection_rank_score": rank / 7,
                    "risk_adjusted_score": quality,
                    "expected_return": quality / 10,
                    "candidate_strength": quality,
                    "future_return_20d": 0.04 - rank * 0.005,
                    "future_max_return_20d": 0.12 - rank * 0.01,
                    "future_max_drawdown_20d": -0.08,
                    "opportunity_value_20d": 0.03 - rank * 0.004,
                    "opportunity_top_decile_20d": 1 if rank < 2 else 0,
                }
            )
    pd.DataFrame(rows).to_parquet(dataset_path, index=False)
    (model_dir / "feature_columns.json").write_text(json.dumps(["risk_adjusted_score"]), encoding="utf-8")
    (model_dir / "model_metadata.json").write_text(
        json.dumps({"train_period": {"start": "2023-01-04", "end": "2024-12-31"}, "test_period": {"start": "2025-01-01", "end": "2025-12-31"}}),
        encoding="utf-8",
    )
    joblib.dump(_FakeClassifier(), model_dir / "opportunity_top_decile_20d_classifier.joblib")


def test_phase11g_builds_2024_check_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    report = Phase11GOutOfSampleCheck(tmp_path).build_report()

    assert report["metadata"]["evaluated_year"] == 2024
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert {row["strategy"] for row in report["strategy_results"]} == {
        "baseline_equal_allocation",
        "valuation_top5_no_guard",
        "valuation_top5_E4",
        "valuation_top5_E4_cost_0.2pct",
    }
    assert report["model_oos_limitations"]["strict_model_oos"] is False
    assert "all_passed" in report["oos_judgement"]


def test_phase11g_saves_reports(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    paths = Phase11GOutOfSampleCheck(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "11-G"
    assert loaded["leakage_checklist"]["future_columns_used_as_features"] == []
