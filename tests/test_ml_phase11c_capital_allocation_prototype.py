from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ml.phase11c_capital_allocation_prototype import Phase11COptions, Phase11CCapitalAllocationPrototype


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
    for day in range(3):
        date = pd.Timestamp("2025-01-07") + pd.Timedelta(days=day)
        for rank in range(8):
            quality = (7 - rank) / 7
            rows.append(
                {
                    "date": date,
                    "code": f"{10000 + day * 10 + rank}",
                    "close": 100 + rank * 10,
                    "turnover_value": 1_000_000 + rank,
                    "risk_adjusted_score": quality,
                    "future_return_20d": quality / 10,
                    "future_max_return_20d": quality / 5,
                    "future_max_drawdown_20d": -0.10 + quality / 20,
                    "opportunity_value_20d": quality / 5 - abs(-0.10 + quality / 20),
                    "opportunity_top_decile_20d": 1 if quality > 0.85 else 0,
                }
            )
    pd.DataFrame(rows).to_parquet(dataset_path, index=False)
    (model_dir / "feature_columns.json").write_text(json.dumps(["risk_adjusted_score"]), encoding="utf-8")
    joblib.dump(_FakeClassifier(), model_dir / "opportunity_top_decile_20d_classifier.joblib")


def test_phase11c_leakage_checklist_is_low_risk(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    runner = Phase11CCapitalAllocationPrototype(tmp_path, options=Phase11COptions(save_simulation=False))
    report, simulation = runner.build_report()

    assert not simulation.empty
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert report["leakage_checklist"]["strategy_backtest_executed"] is False
    assert report["recommendation"]["ready_for_phase11d"] is True


def test_phase11c_saves_reports_and_simulation(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    runner = Phase11CCapitalAllocationPrototype(tmp_path, options=Phase11COptions(save_simulation=True))
    paths = runner.run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.simulation is not None and paths.simulation.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "11-C"
    assert loaded["metadata"]["strategy_backtest_executed"] is False
    assert {row["rule"] for row in loaded["rule_comparison"]} == {
        "equal_weight_top5",
        "proba_rank_weighted",
        "proba_confidence_weighted",
        "conservative_top_only",
    }
