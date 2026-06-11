from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase11b2_strict_oos_failure_diagnosis import Phase11B2StrictOOSFailureDiagnosis
from ml.phase11i_strict_oos import Phase11IOptions, Phase11IStrictOOS


def _write_fixture(root: Path) -> None:
    dataset_path = root / "data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for year in [2023, 2024, 2025]:
        for day_index, date in enumerate(pd.bdate_range(f"{year}-01-04", periods=35)):
            for rank in range(10):
                quality = (9 - rank) / 9
                risky_top = rank == 0 and year == 2025
                rows.append(
                    {
                        "date": date,
                        "code": f"{10000 + rank}",
                        "close": float(100 + rank * 4 + (20 if risky_top and day_index > 4 else 0)),
                        "volume": 100_000 + rank,
                        "turnover_value": 1_000_000 + rank * 1000,
                        "return_1d": quality / 100,
                        "return_3d": quality / 90,
                        "return_5d": quality / 80,
                        "return_10d": quality / 70,
                        "return_20d": quality / 60,
                        "ma5_gap": quality / 50,
                        "ma10_gap": quality / 45,
                        "ma25_gap": quality / 40,
                        "ma75_gap": quality / 8 if risky_top else quality / 60,
                        "ma5_slope": quality / 30,
                        "ma25_slope": quality / 35,
                        "volume_ratio_5d": 1 + quality,
                        "volume_ratio_20d": 1 + quality / 2,
                        "turnover_ratio_5d": 1 + quality,
                        "turnover_ratio_20d": 1 + quality / 2,
                        "body_ratio": 0.2 + quality / 10,
                        "upper_shadow_ratio": 0.1,
                        "lower_shadow_ratio": 0.1,
                        "gap_up_ratio": quality / 100,
                        "daily_range_ratio": 0.03 + (0.20 if risky_top else quality / 100),
                        "EPS": 100 + rank,
                        "BPS": 1000 + rank,
                        "EqAR": 0.4,
                        "Sales_growth": quality / 5,
                        "OP_growth": quality / 6,
                        "NP_growth": quality / 7,
                        "FEPS_growth": quality / 8,
                        "FSales_growth": quality / 9,
                        "FOP_growth": quality / 10,
                        "PayoutRatioAnn": 0.3,
                        "topix_return_5d": 0.01,
                        "topix_return_10d": 0.02,
                        "topix_return_20d": 0.03,
                        "relative_return_5d": quality / 20,
                        "relative_return_10d": quality / 15,
                        "relative_return_20d": quality / 4 if risky_top else quality / 25,
                        "risk_adjusted_score": quality,
                        "expected_return": quality / 10,
                        "candidate_strength": quality,
                        "stock_selection_rank_score": quality,
                        "risk_adjusted_score_rank_in_day": rank + 1,
                        "risk_adjusted_score_percentile_in_day": quality,
                        "risk_adjusted_score_gap_to_best": 1 - quality,
                        "expected_return_rank_in_day": rank + 1,
                        "expected_return_percentile_in_day": quality,
                        "expected_return_gap_to_best": 1 - quality,
                        "stock_selection_rank_score_rank_in_day": rank + 1,
                        "stock_selection_rank_score_percentile_in_day": quality,
                        "stock_selection_rank_score_gap_to_best": 1 - quality,
                        "candidate_strength_rank_in_day": rank + 1,
                        "candidate_strength_percentile_in_day": quality,
                        "candidate_strength_gap_to_best": 1 - quality,
                        "future_return_20d": -0.12 if risky_top else 0.05 - rank * 0.004,
                        "future_max_return_20d": 0.18 - rank * 0.006,
                        "future_max_drawdown_20d": -0.18 if risky_top else -0.04 - rank * 0.004,
                        "opportunity_value_20d": -0.02 if risky_top else 0.05 - rank * 0.004,
                        "opportunity_top_decile_20d": 1 if rank == 0 and not risky_top else 0,
                    }
                )
    pd.DataFrame(rows).to_parquet(dataset_path, index=False)


def _prepare_phase11i_model(root: Path) -> None:
    Phase11IStrictOOS(root, options=Phase11IOptions(max_train_rows=500, max_iter=20)).run()


def test_phase11b2_builds_diagnosis_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _prepare_phase11i_model(tmp_path)

    report = Phase11B2StrictOOSFailureDiagnosis(tmp_path).build_report()

    assert report["metadata"]["phase"] == "11-B2"
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert report["prediction_quality_by_decile"]
    assert {row["candidate_set"] for row in report["top_candidate_diagnostics"]} == {
        "baseline_top5",
        "strict_oos_valuation_top5",
    }
    assert "main_failure_reason" in report["diagnosis_summary"]


def test_phase11b2_saves_markdown_and_json(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _prepare_phase11i_model(tmp_path)

    paths = Phase11B2StrictOOSFailureDiagnosis(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "11-B2"
    assert loaded["leakage_checklist"]["historical_predictions_regenerated"] is False
