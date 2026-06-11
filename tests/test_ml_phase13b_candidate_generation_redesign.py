from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase13b_candidate_generation_redesign import REQUIRED_REPORT_KEYS, Phase13BCandidateGenerationRedesign


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
        for rank in range(15):
            opportunity = 0.90 if rank == 0 else 0.80 if rank == 1 else (14 - rank) / 25
            downside = 0.10 if rank == 0 else 0.55 if rank == 1 else 0.25
            strength = 0.85 if rank == 0 else (14 - rank) / 14
            rows.append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "close": 100.0 + rank,
                    "turnover_value": 1_000_000 + rank,
                    "stock_selection_rank_score": (14 - rank) / 14,
                    "risk_adjusted_score": (14 - rank) / 14 - downside * 0.1,
                    "expected_return": (14 - rank) / 140,
                    "candidate_strength": strength,
                    "opportunity_proba": opportunity,
                    "downside_bad_proba": downside,
                    "opportunity_rank_percentile": opportunity,
                    "downside_rank_percentile": downside,
                    "future_return_20d": 0.08 if rank == 0 else -0.03 if rank == 1 else 0.01,
                    "future_max_return_20d": 0.14 if rank == 0 else 0.08,
                    "future_max_drawdown_20d": -0.03 if rank == 0 else -0.14 if rank == 1 else -0.05,
                    "opportunity_value_20d": 0.10 if rank == 0 else -0.02 if rank == 1 else 0.01,
                    "opportunity_top_decile_20d": 1 if rank == 0 else 0,
                    "downside_bad_20d": 1 if rank == 1 else 0,
                }
            )
            labels_by_date[date_text].append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "future_5d_return": 0.02 if rank == 0 else -0.01,
                    "future_10d_return": 0.04 if rank == 0 else -0.02 if rank == 1 else 0.0,
                    "bad_entry_10d": 1 if rank == 1 else 0,
                    "future_max_return_10d": 0.09 if rank == 0 else 0.03,
                    "future_max_return_20d": 0.14 if rank == 0 else 0.05,
                }
            )
    pd.DataFrame(rows).to_parquet(artifact_path, index=False)
    for date_text, rows_for_date in labels_by_date.items():
        pd.DataFrame(rows_for_date).to_parquet(label_dir / f"labels_{date_text}.parquet", index=False)


def test_phase13b_builds_required_report(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    report = Phase13BCandidateGenerationRedesign(tmp_path).build_report()

    assert report["metadata"]["phase"] == "13-B"
    assert report["metadata"]["new_model_trained"] is False
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["input_artifact_summary"]["date_min"].startswith("2025")
    assert report["input_artifact_summary"]["date_max"].startswith("2025")
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert report["leakage_checklist"]["existing_model_overwritten"] is False
    assert report["leakage_checklist"]["profile_changed"] is False
    assert report["leakage_checklist"]["full_backtest_executed"] is False
    assert report["leakage_risk"] == "low"
    assert "blocking_issues" in report
    for key in REQUIRED_REPORT_KEYS:
        assert key in report
    methods = {row["method"] for row in report["candidate_generation_results"]}
    assert "valuation_first_top50" in methods
    assert "opportunity_downside_top100" in methods


def test_phase13b_saves_json_report(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    paths = Phase13BCandidateGenerationRedesign(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "13-B"
    assert loaded["leakage_checklist"]["historical_predictions_regenerated"] is False
