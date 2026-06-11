from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase12b3_exit_hold_audit import Phase12B3ExitHoldAudit


def _write_artifact(root: Path) -> None:
    path = root / ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for day_index, date in enumerate(pd.bdate_range("2025-01-07", periods=45)):
        for rank in range(8):
            quality = (7 - rank) / 7
            close = 100 + rank * 4
            if rank == 0:
                close = 100 + min(day_index, 12) * 2
            if rank == 1 and day_index >= 5:
                close = 90
            if rank == 2 and day_index >= 8:
                close = 116
            rows.append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "close": float(close),
                    "turnover_value": 1_000_000 + rank,
                    "stock_selection_rank_score": quality,
                    "risk_adjusted_score": quality,
                    "expected_return": quality / 10,
                    "candidate_strength": quality,
                    "opportunity_proba": 0.05 + quality * 0.5,
                    "downside_bad_proba": 0.40 if rank < 2 else 0.10,
                    "opportunity_rank_percentile": (rank + 1) / 8,
                    "downside_rank_percentile": 0.95 if rank == 1 else 0.30 + rank / 10,
                    "confidence": 0.5,
                    "future_return_20d": 0.05 - rank * 0.004,
                    "future_max_return_20d": 0.12 - rank * 0.006,
                    "future_max_drawdown_20d": -0.12 if rank == 1 else -0.04,
                    "opportunity_value_20d": 0.06 - rank * 0.004,
                    "opportunity_top_decile_20d": 1 if rank < 2 else 0,
                    "downside_bad_20d": 1 if rank == 1 else 0,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase12b3_builds_exit_hold_audit(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    report = Phase12B3ExitHoldAudit(tmp_path).build_report()

    assert report["metadata"]["phase"] == "12-B3"
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert {row["strategy"] for row in report["trade_summary"]} == {"S2_opportunity_top5_E4", "S3a_dynamic_raw_weight"}
    assert "main_exit_problem" in report["recommendation"]
    assert report["early_exit_audit"]
    assert report["exit_reason_ranking"]


def test_phase12b3_saves_reports(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    paths = Phase12B3ExitHoldAudit(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "12-B3"
    assert loaded["leakage_checklist"]["existing_model_overwritten"] is False
