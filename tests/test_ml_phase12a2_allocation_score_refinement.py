from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase12a2_allocation_score_refinement import Phase12A2AllocationScoreRefinement


def _write_artifact(root: Path) -> None:
    path = root / ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for date in pd.bdate_range("2025-01-07", periods=8):
        for rank in range(30):
            opportunity = (29 - rank) / 29
            downside = rank in {0, 1, 25, 26, 27, 28, 29}
            rows.append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "close": float(100 + rank),
                    "turnover_value": 1_000_000 + rank,
                    "opportunity_proba": 0.05 + opportunity * 0.5,
                    "downside_bad_proba": 0.55 if downside else 0.10 + rank / 300,
                    "opportunity_rank_percentile": (rank + 1) / 30,
                    "downside_rank_percentile": 0.95 if downside else 0.25,
                    "confidence": 0.5,
                    "future_return_20d": 0.04 - rank * 0.001,
                    "future_max_return_20d": 0.12 - rank * 0.002,
                    "future_max_drawdown_20d": -0.14 if downside else -0.04,
                    "opportunity_value_20d": 0.05 - rank * 0.001,
                    "opportunity_top_decile_20d": 1 if rank <= 5 else 0,
                    "downside_bad_20d": 1 if downside else 0,
                    "allocation_rule": "fixture",
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase12a2_builds_refinement_report(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    report = Phase12A2AllocationScoreRefinement(tmp_path).build_report()

    assert report["metadata"]["phase"] == "12-A2"
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert len(report["rule_quality"]) == 25
    assert "best_allocation_rule" in report["summary"]
    assert any(row["candidate_universe"] == "opportunity_top5" for row in report["rule_quality"])


def test_phase12a2_saves_reports(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    paths = Phase12A2AllocationScoreRefinement(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "12-A2"
    assert loaded["leakage_checklist"]["strategy_backtest_executed"] is False
