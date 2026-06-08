from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase8f_pm_ai_label_redesign_audit import Phase8FPMAILabelRedesignAudit


def _write_fixture(root: Path) -> None:
    trades_path = root / "reports/final/v2_82_cap38/core_2023-01_to_2026-05/trades.csv"
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "code": "11110",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "net_profit": 1000,
                "pm_multiplier": 1.30,
                "holding_days": 4,
                "volume_ratio": 2.2,
                "expected_return_10d": 0.08,
                "risk_adjusted_score": 0.7,
                "sector_name": "Info",
            },
            {
                "action": "SELL",
                "code": "22220",
                "signal_date": "2023-01-04",
                "entry_date": "2023-01-05",
                "net_profit": -200,
                "pm_multiplier": 0.80,
                "holding_days": 3,
                "volume_ratio": 1.1,
                "expected_return_10d": 0.01,
                "risk_adjusted_score": 0.2,
                "sector_name": "Retail",
            },
            {
                "action": "SELL",
                "code": "33330",
                "signal_date": "2023-01-05",
                "entry_date": "2023-01-06",
                "net_profit": 500,
                "pm_multiplier": 1.00,
                "holding_days": 5,
                "volume_ratio": 1.5,
                "expected_return_10d": 0.03,
                "risk_adjusted_score": 0.4,
                "sector_name": "Service",
            },
        ]
    ).to_csv(trades_path, index=False)

    dataset_path = root / "data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05.parquet"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "as_of_date": "2023-01-04",
                "code": "11110",
                "future_5d_return": 0.05,
                "future_10d_return": 0.10,
                "risk_adjusted_future_return": 0.08,
                "high_conviction_target": 1,
                "avoid_target": 0,
                "volume_ratio_5d": 2.0,
            },
            {
                "as_of_date": "2023-01-04",
                "code": "22220",
                "future_5d_return": -0.02,
                "future_10d_return": -0.04,
                "risk_adjusted_future_return": -0.08,
                "high_conviction_target": 0,
                "avoid_target": 1,
                "volume_ratio_5d": 1.0,
            },
            {
                "as_of_date": "2023-01-05",
                "code": "33330",
                "future_5d_return": 0.01,
                "future_10d_return": 0.02,
                "risk_adjusted_future_return": 0.01,
                "high_conviction_target": 0,
                "avoid_target": 0,
                "volume_ratio_5d": 1.5,
            },
        ]
    ).to_parquet(dataset_path, index=False)

    for model_dir in [
        root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean",
        root / "models/ml/portfolio_manager/candidate_v2_api_only",
    ]:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model_metadata.json").write_text(json.dumps({"model_profile": model_dir.name}), encoding="utf-8")


def test_phase8f_builds_label_redesign_audit(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    audit = Phase8FPMAILabelRedesignAudit(tmp_path)
    report = audit.build_report()

    assert report["metadata"]["audit_only"] is True
    assert report["pm_ai_true_objective"]["pm_ai_true_objective"] == "capital_allocation_for_position_sizing"
    assert report["dataset_inventory"]["merged_trade_rows"] == 3
    labels = {row["label"]: row for row in report["current_label_audit"]}
    assert labels["future_10d_return"]["available"] is True
    assert labels["future_10d_return"]["correlation_to_trade_profit"] is not None
    assert report["pm130_group_analysis"]["summary"]["trade_count"] == 1
    assert report["pm080_group_analysis"]["summary"]["trade_count"] == 1
    assert report["pm130_mimic_feasibility"]["api_only_feasible"] is False
    assert "recommended_label_design" in report["verdict"]


def test_phase8f_saves_markdown_and_json(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = Phase8FPMAILabelRedesignAudit(tmp_path)
    report = audit.build_report()
    paths = audit.save_report(report)

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "8-F"
