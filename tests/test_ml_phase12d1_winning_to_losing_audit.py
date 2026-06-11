from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase12d1_winning_to_losing_audit import Phase12D1WinningToLosingAudit


def _write_artifact(root: Path) -> None:
    path = root / ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for day_index, date in enumerate(pd.bdate_range("2025-01-07", periods=45)):
        for rank in range(8):
            quality = (7 - rank) / 7
            close = 100.0 + rank * 5
            if rank == 0:
                close = 100.0 + min(day_index, 6) * 2.5
                if day_index > 6:
                    close = 115.0 - (day_index - 6) * 3.0
            if rank == 1:
                close = 105.0 + min(day_index, 4)
                if day_index > 10:
                    close = 108.0 - (day_index - 10) * 1.0
            downside_rank = 0.30 if rank == 0 else 0.60 if rank == 1 else 0.78 if rank == 2 else 0.90
            rows.append(
                {
                    "date": date,
                    "code": f"{10000 + rank}",
                    "close": close,
                    "turnover_value": 1_000_000 + rank,
                    "opportunity_proba": 0.08 + quality * 0.50,
                    "downside_bad_proba": 0.20,
                    "opportunity_rank_percentile": quality,
                    "downside_rank_percentile": downside_rank,
                    "confidence": 0.6,
                    "future_return_20d": 0.05 - rank * 0.005,
                    "future_max_return_20d": 0.12 - rank * 0.006,
                    "future_max_drawdown_20d": -0.12 if rank == 0 else -0.04,
                    "opportunity_value_20d": 0.06 - rank * 0.004,
                    "opportunity_top_decile_20d": 1 if rank < 2 else 0,
                    "downside_bad_20d": 1 if rank == 0 else 0,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_phase12d1_detects_winning_to_losing_trades(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    report = Phase12D1WinningToLosingAudit(tmp_path).build_report()

    assert report["metadata"]["phase"] == "12-D1"
    assert report["metadata"]["full_backtest_executed"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert "peak_return_mean" in report["profit_decay_distribution"]
    assert report["audit_5pct"]["winning_trades_turned_losers_count"] >= 1
    assert report["recommendation"]["winning_to_losing_conversion_detected"] is True


def test_phase12d1_saves_reports(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    paths = Phase12D1WinningToLosingAudit(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "12-D1"
    assert loaded["leakage_checklist"]["existing_model_overwritten"] is False
