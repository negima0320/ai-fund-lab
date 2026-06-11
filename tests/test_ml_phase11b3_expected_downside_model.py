from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase11b3_expected_downside_model import Phase11B3ExpectedDownsideModel, Phase11B3Options


def _write_fixture(root: Path) -> None:
    dataset_path = root / "data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for year in [2023, 2024, 2025]:
        for date in pd.bdate_range(f"{year}-01-04", periods=40):
            for rank in range(12):
                opportunity = (11 - rank) / 11
                high_downside = rank in {0, 1, 10, 11}
                rows.append(
                    {
                        "date": date,
                        "code": f"{10000 + rank}",
                        "close": float(100 + rank * 5),
                        "volume": 100_000 + rank,
                        "turnover_value": 1_000_000 + rank * 10_000,
                        "return_1d": opportunity / 100,
                        "return_3d": opportunity / 90,
                        "return_5d": opportunity / 80,
                        "return_10d": opportunity / 70,
                        "return_20d": opportunity / 60,
                        "ma5_gap": opportunity / 50,
                        "ma10_gap": opportunity / 45,
                        "ma25_gap": opportunity / 40,
                        "ma75_gap": 0.20 if high_downside else opportunity / 60,
                        "ma5_slope": opportunity / 30,
                        "ma25_slope": opportunity / 35,
                        "volume_ratio_5d": 1 + opportunity,
                        "volume_ratio_20d": 1 + opportunity / 2,
                        "turnover_ratio_5d": 1 + opportunity,
                        "turnover_ratio_20d": 1 + opportunity / 2,
                        "body_ratio": 0.2 + opportunity / 10,
                        "upper_shadow_ratio": 0.1,
                        "lower_shadow_ratio": 0.1,
                        "gap_up_ratio": opportunity / 100,
                        "daily_range_ratio": 0.08 if high_downside else 0.02,
                        "EPS": 100 + rank,
                        "BPS": 1000 + rank,
                        "EqAR": 0.4,
                        "Sales_growth": opportunity / 5,
                        "OP_growth": opportunity / 6,
                        "NP_growth": opportunity / 7,
                        "FEPS_growth": opportunity / 8,
                        "FSales_growth": opportunity / 9,
                        "FOP_growth": opportunity / 10,
                        "PayoutRatioAnn": 0.3,
                        "topix_return_5d": 0.01,
                        "topix_return_10d": 0.02,
                        "topix_return_20d": 0.03,
                        "relative_return_5d": opportunity / 20,
                        "relative_return_10d": opportunity / 15,
                        "relative_return_20d": 0.25 if high_downside else opportunity / 25,
                        "risk_adjusted_score": opportunity,
                        "expected_return": opportunity / 10,
                        "candidate_strength": opportunity,
                        "stock_selection_rank_score": opportunity,
                        "risk_adjusted_score_rank_in_day": rank + 1,
                        "risk_adjusted_score_percentile_in_day": opportunity,
                        "risk_adjusted_score_gap_to_best": 1 - opportunity,
                        "expected_return_rank_in_day": rank + 1,
                        "expected_return_percentile_in_day": opportunity,
                        "expected_return_gap_to_best": 1 - opportunity,
                        "stock_selection_rank_score_rank_in_day": rank + 1,
                        "stock_selection_rank_score_percentile_in_day": opportunity,
                        "stock_selection_rank_score_gap_to_best": 1 - opportunity,
                        "candidate_strength_rank_in_day": rank + 1,
                        "candidate_strength_percentile_in_day": opportunity,
                        "candidate_strength_gap_to_best": 1 - opportunity,
                        "future_return_20d": 0.06 - rank * 0.004 - (0.04 if high_downside else 0.0),
                        "future_max_return_20d": 0.15 - rank * 0.005,
                        "future_max_drawdown_20d": -0.14 if high_downside else -0.04 - rank * 0.002,
                        "opportunity_value_20d": 0.06 - rank * 0.004,
                        "opportunity_top_decile_20d": 1 if rank in {0, 1} else 0,
                    }
                )
    pd.DataFrame(rows).to_parquet(dataset_path, index=False)


def test_phase11b3_trains_downside_model_and_audits_buy_quality(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    report, models = Phase11B3ExpectedDownsideModel(tmp_path, options=Phase11B3Options(max_train_rows=600, max_iter=20)).build_report_and_models()

    assert models
    assert report["split"]["strict_model_oos"] is True
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert report["downside_model_quality"]["test"]["target"] == "downside_bad_20d"
    assert {row["candidate_set"] for row in report["combined_ranking_audit"]} == {
        "opportunity_only_top5",
        "score_v1_top5",
        "score_v2_top5",
        "score_v3_top5",
    }


def test_phase11b3_saves_reports_and_research_models(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    paths = Phase11B3ExpectedDownsideModel(tmp_path, options=Phase11B3Options(max_train_rows=600, max_iter=20)).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.model_dir is not None
    assert (paths.model_dir / "opportunity_top_decile_20d_classifier.joblib").exists()
    assert (paths.model_dir / "downside_bad_20d_classifier.joblib").exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "11-B3"
    assert loaded["leakage_checklist"]["strict_model_oos"] is True
