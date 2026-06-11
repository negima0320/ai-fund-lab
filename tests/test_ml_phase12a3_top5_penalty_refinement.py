from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase12a3_top5_penalty_refinement import Phase12A3Top5PenaltyRefinement


EXPECTED_RULES = {
    "A2_baseline_penalty_rank_medium",
    "A3_1_rank_medium_plus",
    "A3_2_rank_medium_stronger_tail",
    "A3_3_rank_medium_floor_zero",
    "A3_4_hybrid_rank_and_proba",
    "A3_5_hybrid_rank_and_proba_strict",
}


def _write_artifact(root: Path) -> None:
    path = root / ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for date in pd.bdate_range("2025-01-07", periods=8):
        for rank in range(12):
            opportunity = (11 - rank) / 11
            top5 = rank < 5
            downside_rank = 0.95 if rank in {0, 1} else 0.20 + rank / 20
            downside_proba = 0.58 if rank == 0 else 0.46 if rank == 1 else 0.12 + rank / 100
            rows.append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "close": float(100 + rank),
                    "turnover_value": 1_000_000 + rank,
                    "opportunity_proba": 0.05 + opportunity * 0.5,
                    "downside_bad_proba": downside_proba,
                    "opportunity_rank_percentile": (rank + 1) / 12,
                    "downside_rank_percentile": downside_rank,
                    "confidence": 0.5,
                    "future_return_20d": 0.05 - rank * 0.002,
                    "future_max_return_20d": 0.15 - rank * 0.003,
                    "future_max_drawdown_20d": -0.13 if rank in {0, 1} else -0.04,
                    "opportunity_value_20d": 0.06 - rank * 0.002,
                    "opportunity_top_decile_20d": 1 if top5 else 0,
                    "downside_bad_20d": 1 if rank in {0, 1} else 0,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase12a3_builds_top5_penalty_report(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    report = Phase12A3Top5PenaltyRefinement(tmp_path).build_report()

    assert report["metadata"]["phase"] == "12-A3"
    assert report["metadata"]["strategy_backtest_executed"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert {row["allocation_rule"] for row in report["rule_quality"]} == EXPECTED_RULES
    assert all(row["candidate_universe"] == "opportunity_top5" for row in report["rule_quality"])
    assert all(row["average_allocated_candidates_per_day"] == 5.0 for row in report["rule_quality"])
    assert "best_allocation_rule" in report["summary"]


def test_phase12a3_saves_reports(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    paths = Phase12A3Top5PenaltyRefinement(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "12-A3"
    assert loaded["leakage_checklist"]["strategy_backtest_executed"] is False
    assert loaded["leakage_checklist"]["existing_model_overwritten"] is False
