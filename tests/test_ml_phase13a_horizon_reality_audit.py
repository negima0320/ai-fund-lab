from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase13a_horizon_reality_audit import REQUIRED_REPORT_KEYS, Phase13AHorizonRealityAudit


def _write_artifacts(root: Path) -> None:
    artifact_path = root / ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    label_dir = root / "data/ml/labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    label_rows_by_date: dict[str, list[dict[str, object]]] = {}
    dates = pd.bdate_range("2025-01-07", periods=25)
    for date in dates:
        date_text = date.date().isoformat()
        label_rows_by_date[date_text] = []
        for rank in range(12):
            quality = (11 - rank) / 11
            opportunity = 0.9 if rank in {0, 1} else quality * 0.5
            downside = 0.15 if rank == 0 else 0.55 if rank == 1 else 0.25
            rows.append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "close": 100.0 + rank,
                    "turnover_value": 1_000_000 + rank,
                    "stock_selection_rank_score": quality,
                    "risk_adjusted_score": quality - downside * 0.1,
                    "expected_return": quality / 10,
                    "candidate_strength": 0.8 if rank == 0 else quality * 0.4,
                    "opportunity_proba": opportunity,
                    "downside_bad_proba": downside,
                    "opportunity_rank_percentile": opportunity,
                    "downside_rank_percentile": downside,
                    "future_return_20d": 0.08 if rank == 0 else -0.03 if rank == 1 else 0.01,
                    "future_max_return_20d": 0.15 if rank == 0 else 0.08 if rank == 1 else 0.03,
                    "future_max_drawdown_20d": -0.03 if rank == 0 else -0.14 if rank == 1 else -0.05,
                    "opportunity_value_20d": 0.10 if rank == 0 else -0.02 if rank == 1 else 0.01,
                    "opportunity_top_decile_20d": 1 if rank == 0 else 0,
                    "downside_bad_20d": 1 if rank == 1 else 0,
                }
            )
            label_rows_by_date[date_text].append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "future_5d_return": 0.02 if rank == 0 else -0.01,
                    "future_10d_return": 0.04 if rank == 0 else -0.02 if rank == 1 else 0.0,
                    "bad_entry_10d": 1 if rank == 1 else 0,
                    "future_max_return_10d": 0.10 if rank == 0 else 0.02,
                    "future_max_return_20d": 0.15 if rank == 0 else 0.05,
                    "future_swing_success_20d": 1 if rank == 0 else 0,
                }
            )
    pd.DataFrame(rows).to_parquet(artifact_path, index=False)
    for date_text, label_rows in label_rows_by_date.items():
        pd.DataFrame(label_rows).to_parquet(label_dir / f"labels_{date_text}.parquet", index=False)


def test_phase13a_builds_required_report(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    report = Phase13AHorizonRealityAudit(tmp_path).build_report()

    assert report["metadata"]["phase"] == "13-A"
    assert report["metadata"]["new_model_trained"] is False
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["input_artifact_summary"]["date_min"].startswith("2025")
    assert report["input_artifact_summary"]["date_max"].startswith("2025")
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert report["leakage_checklist"]["existing_model_overwritten"] is False
    assert report["leakage_checklist"]["historical_predictions_regenerated"] is False
    assert report["leakage_risk"] == "low"
    assert "blocking_issues" in report
    for key in REQUIRED_REPORT_KEYS:
        assert key in report
    assert report["horizon_score_quality"]
    assert report["score_monotonicity"]
    assert report["stock_selection_vs_candidate_strength_vs_valuation"]


def test_phase13a_saves_json_report(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    paths = Phase13AHorizonRealityAudit(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "13-A"
    assert loaded["leakage_checklist"]["profile_changed"] is False
    assert loaded["leakage_checklist"]["future_columns_used_as_features"] == []
