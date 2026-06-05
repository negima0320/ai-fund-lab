from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.walk_forward_diagnostics import WalkForwardDiagnosticsAnalyzer


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _analyzer(tmp_path: Path) -> WalkForwardDiagnosticsAnalyzer:
    return WalkForwardDiagnosticsAnalyzer(
        prediction_root=tmp_path / "data" / "ml" / "walk_forward_predictions",
        label_root=tmp_path / "data" / "ml" / "labels",
        report_root=tmp_path / "reports" / "ml",
        cache_root=tmp_path / "jquants",
    )


def test_walk_forward_diagnostics_enriches_losing_trades_and_saves_outputs(tmp_path) -> None:
    analyzer = _analyzer(tmp_path)
    wf_path = tmp_path / "reports" / "ml" / "walk_forward_2026-05.json"
    wf_path.parent.mkdir(parents=True, exist_ok=True)
    wf_path.write_text(
        json.dumps(
            {
                "trades": [
                    {
                        "month": "2026-05",
                        "signal_date": "2026-05-01",
                        "code": "1001",
                        "entry_date": "2026-05-02",
                        "exit_date": "2026-05-15",
                        "return": -0.2,
                        "expected_return_10d": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_parquet(
        analyzer.prediction_root / "predictions_2026-05-01.parquet",
        [
            {
                "date": pd.Timestamp("2026-05-01"),
                "code": "1001",
                "expected_return_10d": 0.1,
                "expected_max_return_20d": 0.2,
                "swing_success_probability_20d": 0.8,
                "bad_entry_probability_10d": 0.7,
                "ml_score": 5.0,
            }
        ],
    )
    _write_parquet(
        analyzer.label_root / "labels_2026-05-01.parquet",
        [
            {
                "date": pd.Timestamp("2026-05-01"),
                "code": "1001",
                "future_10d_return": -0.2,
                "future_max_return_20d": 0.05,
                "future_swing_success_20d": False,
                "bad_entry_10d": True,
            }
        ],
    )

    result = analyzer.analyze(wf_path, "2026-05-01", "2026-05-31")
    md_path = analyzer.save_report(result)
    json_path = analyzer.save_json(result)
    csv_path = analyzer.save_losing_trades_csv(result)

    assert result["monthly_top10_summary"][0]["trade_count"] == 1
    assert result["monthly_top10_summary"][0]["bad_entry_rate"] == 1.0
    assert result["losing_trades_2026_05"][0]["bad_entry_probability_10d"] == 0.7
    assert md_path.exists()
    assert json_path.exists()
    assert csv_path.exists()
