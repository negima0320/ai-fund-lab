from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase13c2_horizon_aware_valuation_dataset_audit import REQUIRED_REPORT_KEYS, Phase13C2HorizonAwareValuationDatasetAudit


def _write_artifacts(root: Path) -> None:
    artifact_path = root / ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    label_dir = root / "data/ml/labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    labels_by_date: dict[str, list[dict[str, object]]] = {}
    for date in pd.bdate_range("2025-01-07", periods=30):
        date_text = date.date().isoformat()
        labels_by_date[date_text] = []
        for rank in range(60):
            code = f"{10000 + rank}"
            strong = rank < 8
            risky = 8 <= rank < 14
            future_return = 0.07 if strong else -0.03 if risky else 0.01
            future_peak = 0.14 if strong else 0.09 if risky else 0.04
            future_dd = -0.03 if strong else -0.12 if risky else -0.04
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": 100.0 + rank,
                    "turnover_value": 1_000_000 + rank,
                    "stock_selection_rank_score": (59 - rank) / 59,
                    "risk_adjusted_score": (59 - rank) / 59,
                    "expected_return": (59 - rank) / 590,
                    "candidate_strength": 1.0 - rank / 100,
                    "opportunity_proba": 0.90 if strong else 0.80 if risky else 0.30,
                    "downside_bad_proba": 0.15 if strong else 0.70 if risky else 0.25,
                    "opportunity_rank_percentile": 1.0 - rank / 100,
                    "downside_rank_percentile": 0.15 if strong else 0.80 if risky else 0.30,
                    "future_return_20d": future_return,
                    "future_max_return_20d": future_peak,
                    "future_max_drawdown_20d": future_dd,
                    "opportunity_value_20d": future_return + future_peak,
                    "opportunity_top_decile_20d": 1 if strong else 0,
                    "downside_bad_20d": 1 if future_dd <= -0.10 else 0,
                }
            )
            labels_by_date[date_text].append(
                {
                    "date": date,
                    "code": code,
                    "future_5d_return": 0.02 if strong else -0.01,
                    "future_10d_return": 0.04 if strong else -0.02 if risky else 0.0,
                    "bad_entry_10d": 1 if risky else 0,
                    "future_max_return_10d": 0.09 if strong else 0.03,
                    "future_max_return_20d": future_peak,
                }
            )
    pd.DataFrame(rows).to_parquet(artifact_path, index=False)
    for date_text, rows_for_date in labels_by_date.items():
        pd.DataFrame(rows_for_date).to_parquet(label_dir / f"labels_{date_text}.parquet", index=False)


def test_phase13c2_builds_required_report(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    report = Phase13C2HorizonAwareValuationDatasetAudit(tmp_path).build_report()

    assert report["metadata"]["phase"] == "13-C2"
    assert report["metadata"]["new_model_trained"] is False
    assert report["metadata"]["strategy_backtest_executed"] is False
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["input_summary"]["date_min"].startswith("2025")
    assert report["input_summary"]["date_max"].startswith("2025")
    assert report["target_definitions"]
    assert report["target_distribution_audit"]
    assert report["existing_score_detectability_audit"]
    assert report["strict_oos_design_feasibility"]["strict_oos_possible"] is True
    assert report["recommended_training_target"]
    assert "dataset_ready_for_training" in report
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert report["leakage_checklist"]["existing_model_overwritten"] is False
    assert report["leakage_checklist"]["profile_changed"] is False
    assert report["leakage_checklist"]["strategy_backtest_executed"] is False
    assert report["leakage_checklist"]["full_backtest_executed"] is False
    assert "leakage_risk" in report
    assert "blocking_issues" in report
    for key in REQUIRED_REPORT_KEYS:
        assert key in report


def test_phase13c2_saves_json_report(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    paths = Phase13C2HorizonAwareValuationDatasetAudit(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "13-C2"
    assert loaded["leakage_checklist"]["historical_predictions_regenerated"] is False
    assert loaded["leakage_checklist"]["future_columns_used_as_features"] == []
