from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase11i_strict_oos import Phase11IOptions, Phase11IStrictOOS


def _write_fixture(root: Path) -> None:
    dataset_path = root / "data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for year in [2023, 2024, 2025]:
        for day_index, date in enumerate(pd.bdate_range(f"{year}-01-04", periods=35)):
            for rank in range(10):
                quality = (9 - rank) / 9
                code = f"{10000 + rank}"
                close = 100 + rank * 3
                if rank == 0 and day_index >= 4:
                    close = 88
                if rank == 1 and day_index >= 3:
                    close = 116
                rows.append(
                    {
                        "date": date,
                        "code": code,
                        "close": float(close),
                        "turnover_value": 1_000_000 + rank,
                        "risk_adjusted_score": quality,
                        "expected_return": quality / 10,
                        "candidate_strength": quality,
                        "stock_selection_rank_score": rank / 9,
                        "future_return_20d": 0.05 - rank * 0.004,
                        "future_max_return_20d": 0.12 - rank * 0.008,
                        "future_max_drawdown_20d": -0.08,
                        "opportunity_value_20d": 0.04 - rank * 0.004,
                        "opportunity_top_decile_20d": 1 if rank == 0 else 0,
                    }
                )
    pd.DataFrame(rows).to_parquet(dataset_path, index=False)


def test_phase11i_trains_research_model_and_reports_strict_oos(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    runner = Phase11IStrictOOS(tmp_path, options=Phase11IOptions(max_train_rows=500, max_iter=20))
    report, model = runner.build_report_and_model()

    assert model
    assert report["split"]["strict_model_oos"] is True
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert report["leakage_checklist"]["existing_model_overwritten"] is False
    assert {row["strategy"] for row in report["strategy_results"]} == {
        "baseline_equal_allocation",
        "strict_oos_valuation_top5_no_guard",
        "strict_oos_E4",
        "strict_oos_H2_cooldown_10d",
        "strict_oos_H3_min_hold_3d",
    }
    assert "AUC" in report["test_model_quality"]


def test_phase11i_saves_reports_and_research_model(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    paths = Phase11IStrictOOS(tmp_path, options=Phase11IOptions(max_train_rows=500, max_iter=20)).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.model_dir is not None
    assert (paths.model_dir / "opportunity_top_decile_20d_classifier.joblib").exists()
    assert (paths.model_dir / "feature_columns.json").exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "11-I"
    assert loaded["leakage_checklist"]["strict_model_oos"] is True
