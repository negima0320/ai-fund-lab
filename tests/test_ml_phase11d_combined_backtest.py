from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase11d_combined_backtest import Phase11DOptions, Phase11DLimitedCombinedBacktest


def _write_fixture(root: Path) -> None:
    simulation_path = root / "data/ml/valuation_engine/phase11c_allocation_simulation_2025.parquet"
    dataset_path = root / "data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet"
    simulation_path.parent.mkdir(parents=True, exist_ok=True)

    simulation_rows = []
    dataset_rows = []
    for day in range(6):
        date = pd.Timestamp("2025-01-07") + pd.Timedelta(days=day)
        for rank in range(8):
            code = f"{10000 + rank}"
            valuation_quality = (7 - rank) / 7
            baseline_quality = rank / 7
            future_return = valuation_quality / 10 - 0.02
            simulation_rows.append(
                {
                    "rule": "equal_weight_top5",
                    "date": date,
                    "code": code,
                    "opportunity_top_decile_proba": valuation_quality,
                    "opportunity_score_proba_rank": valuation_quality,
                    "future_return_20d": future_return,
                    "future_max_return_20d": valuation_quality / 5,
                    "future_max_drawdown_20d": -0.10 + valuation_quality / 20,
                    "opportunity_value_20d": future_return - abs(-0.10 + valuation_quality / 20),
                    "opportunity_top_decile_20d": 1 if valuation_quality > 0.85 else 0,
                }
            )
            dataset_rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": 100.0 + rank,
                    "turnover_value": 1_000_000 + rank,
                    "stock_selection_rank_score": baseline_quality,
                    "risk_adjusted_score": baseline_quality,
                    "expected_return": baseline_quality / 10,
                    "candidate_strength": baseline_quality,
                }
            )

    pd.DataFrame(simulation_rows).to_parquet(simulation_path, index=False)
    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)


def test_phase11d_limited_backtest_compares_baseline_and_candidate(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    runner = Phase11DLimitedCombinedBacktest(tmp_path, options=Phase11DOptions(initial_cash=1_000_000, daily_buy_budget=900_000, max_positions=5))
    report = runner.build_report()

    assert report["metadata"]["limited_2025_only"] is True
    assert report["metadata"]["full_period_backtest_executed"] is False
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["blocking_issues"] == []
    assert {row["strategy"] for row in report["strategy_results"]} == {
        "baseline_equal_allocation",
        "candidate_valuation_top5",
    }
    assert report["valuation_effect"]["opportunity_value_20d_mean_delta"] > 0


def test_phase11d_saves_markdown_and_json(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    runner = Phase11DLimitedCombinedBacktest(tmp_path)
    paths = runner.run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "11-D"
    assert loaded["backtest_conditions"]["period"] == {"start": "2025-01-01", "end": "2025-12-31"}
    assert loaded["leakage_checklist"]["future_columns_used_as_features"] == []
