from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase11c2_budget_usage_constraint_audit import Phase11C2BudgetUsageConstraintAudit


def _write_fixture(root: Path) -> None:
    sim_path = root / "data/ml/valuation_engine/phase11c_allocation_simulation_2025.parquet"
    dataset_path = root / "data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet"
    sim_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    dataset_rows = []
    for day in range(3):
        date = pd.Timestamp("2025-01-07") + pd.Timedelta(days=day)
        for rank in range(6):
            code = f"{10000 + day * 10 + rank}"
            close = 1000 + rank * 300
            rank_pct = 1.0 - rank / 10
            amount = 100_000 if rank < 2 else 0
            rows.append(
                {
                    "rule": "equal_weight_top5",
                    "date": date,
                    "code": code,
                    "opportunity_top_decile_proba": 0.9 - rank * 0.05,
                    "confidence": 0.8,
                    "opportunity_score_proba_rank": rank_pct,
                    "allocation_weight": 1.0 if rank < 5 else 0.0,
                    "target_buy_amount": amount,
                    "target_lot_count": 1 if amount else 0,
                    "allocation_bucket": "top5" if rank < 5 else "zero",
                    "future_return_20d": rank_pct / 10,
                    "future_max_return_20d": rank_pct / 5,
                    "future_max_drawdown_20d": -0.1,
                    "opportunity_value_20d": rank_pct / 5 - 0.1,
                    "opportunity_top_decile_20d": 1 if rank == 0 else 0,
                }
            )
            dataset_rows.append({"date": date, "code": code, "close": close, "turnover_value": 1_000_000})
    pd.DataFrame(rows).to_parquet(sim_path, index=False)
    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)


def test_phase11c2_builds_budget_constraint_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = Phase11C2BudgetUsageConstraintAudit(tmp_path)
    report = audit.build_report()

    assert report["metadata"]["phase"] == "11-C2"
    assert report["leakage_checklist"]["leakage_risk"] == "low"
    assert report["leakage_checklist"]["future_columns_used_as_features"] == []
    assert report["leakage_checklist"]["future_columns_used_only_for_evaluation"]
    assert report["daily_budget_usage_breakdown"]
    assert report["budget_sensitivity"]
    assert report["candidate_threshold_sensitivity"]
    assert "main_budget_bottleneck" in report["recommendation"]


def test_phase11c2_saves_markdown_and_json(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    paths = Phase11C2BudgetUsageConstraintAudit(tmp_path).run()

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["strategy_backtest_executed"] is False
    assert loaded["leakage_checklist"]["profile_changed"] is False
