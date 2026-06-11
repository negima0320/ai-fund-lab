from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase13e_integrated_strategy_prototype import COST_RATE, REQUIRED_REPORT_KEYS, Phase13EIntegratedStrategyPrototype


def _write_artifacts(root: Path) -> None:
    artifact_path = root / ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    label_dir = root / "data/ml/labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    labels_by_date: dict[str, list[dict[str, object]]] = {}
    dates = list(pd.bdate_range("2025-01-07", periods=45))
    for day_index, date in enumerate(dates):
        date_text = date.date().isoformat()
        labels_by_date[date_text] = []
        for rank in range(20):
            code = f"{10000 + rank}"
            base = 100.0 + rank
            drift = 0.004 * day_index if rank < 5 else -0.002 * day_index if 5 <= rank < 10 else 0.001 * day_index
            drawdown = -0.08 if day_index > 10 and 5 <= rank < 10 else 0.0
            close = base * (1.0 + drift + drawdown)
            future_return = 0.08 if rank < 5 else -0.04 if 5 <= rank < 10 else 0.01
            future_max = 0.14 if rank < 5 else 0.07 if 5 <= rank < 10 else 0.04
            future_dd = -0.03 if rank < 5 else -0.12 if 5 <= rank < 10 else -0.04
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": close,
                    "turnover_value": 1_000_000 + rank,
                    "stock_selection_rank_score": (19 - rank) / 19,
                    "risk_adjusted_score": (19 - rank) / 19,
                    "expected_return": (19 - rank) / 190,
                    "candidate_strength": 1.0 - rank / 100,
                    "opportunity_proba": 0.90 if rank < 5 else 0.40,
                    "downside_bad_proba": 0.20 if rank < 5 else 0.70 if 5 <= rank < 10 else 0.30,
                    "opportunity_rank_percentile": 1.0 - rank / 100,
                    "downside_rank_percentile": 0.20 if rank < 5 else 0.80 if 5 <= rank < 10 else 0.30,
                    "future_return_20d": future_return,
                    "future_max_return_20d": future_max,
                    "future_max_drawdown_20d": future_dd,
                    "opportunity_value_20d": future_return + future_max,
                    "opportunity_top_decile_20d": 1 if rank < 5 else 0,
                    "downside_bad_20d": 1 if future_dd <= -0.10 else 0,
                }
            )
            labels_by_date[date_text].append(
                {
                    "date": date,
                    "code": code,
                    "future_5d_return": 0.02 if rank < 5 else -0.01,
                    "future_10d_return": 0.04 if rank < 5 else -0.02 if 5 <= rank < 10 else 0.0,
                    "bad_entry_10d": 1 if 5 <= rank < 10 else 0,
                    "future_max_return_10d": 0.09 if rank < 5 else 0.03,
                    "future_max_return_20d": future_max,
                }
            )
    pd.DataFrame(rows).to_parquet(artifact_path, index=False)
    for date_text, rows_for_date in labels_by_date.items():
        pd.DataFrame(rows_for_date).to_parquet(label_dir / f"labels_{date_text}.parquet", index=False)


def test_phase13e_builds_required_report(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    report = Phase13EIntegratedStrategyPrototype(tmp_path).build_report()

    assert report["metadata"]["phase"] == "13-E"
    assert report["metadata"]["new_model_trained"] is False
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["metadata"]["ready_for_live_adoption"] is False
    assert report["input_artifact_summary"]["date_min"].startswith("2025")
    assert report["input_artifact_summary"]["date_max"].startswith("2025")
    assert report["input_artifact_summary"]["primary_candidate_set"] == "candidate_strength_top50"
    assert report["strategy_parameters"]["cost_rate"] == COST_RATE
    assert report["strategy_parameters"]["future_columns_used_for_entry_exit"] == []
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert report["leakage_checklist"]["future_columns_used_as_entry_or_exit_features"] == []
    assert report["leakage_checklist"]["existing_model_overwritten"] is False
    assert report["leakage_checklist"]["profile_changed"] is False
    assert report["leakage_checklist"]["full_backtest_executed"] is False
    assert report["ready_for_live_adoption"] is False
    assert report["leakage_risk"] in {"low", "medium"}
    assert "blocking_issues" in report
    assert report["annual_return_after_cost"] is not None
    for key in REQUIRED_REPORT_KEYS:
        assert key in report
    variants = {row["variant"] for row in report["variant_results"]}
    assert "E5_candidate_strength_top50_combined_aggressive" in variants


def test_phase13e_saves_json_report(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    paths = Phase13EIntegratedStrategyPrototype(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "13-E"
    assert loaded["leakage_checklist"]["historical_predictions_regenerated"] is False
    assert loaded["leakage_checklist"]["future_columns_used_as_features"] == []
    assert loaded["ready_for_live_adoption"] is False
