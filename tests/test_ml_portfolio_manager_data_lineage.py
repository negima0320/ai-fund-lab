from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_data_lineage import PortfolioManagerDataLineageAudit
from ml.portfolio_manager_dataset import CLEAN_FEATURE_COLUMNS
from ml.portfolio_manager_dataset import LABEL_COLUMNS


def _write_dataset_and_features(root: Path, feature_columns: list[str]) -> tuple[Path, Path]:
    dataset_path = root / "dataset.parquet"
    feature_path = root / "feature_columns.json"
    row = {
        "signal_date": "2023-01-04",
        "code": "1001",
        "expected_return_10d": 0.1,
        "bad_entry_probability_10d": 0.2,
        "risk_adjusted_score": 0.0,
        "close": 100.0,
        "return_1d": 0.01,
        "candidate_count_in_day": 3,
        "realized_return": 0.1,
        "positive_trade": True,
        "high_conviction_target": True,
        "avoid_target": False,
        "ideal_weight_bucket": "strong",
        "ideal_cash_reserve_bucket": "aggressive",
        "future_5d_return": 0.05,
        "future_10d_return": 0.1,
        "decision": "BUY",
        "actual_net_profit": 10000,
        "cash_before": 1000000,
    }
    for column in CLEAN_FEATURE_COLUMNS:
        row.setdefault(column, 0.0)
    pd.DataFrame([row]).to_parquet(dataset_path, index=False)
    feature_path.write_text(json.dumps(feature_columns), encoding="utf-8")
    return dataset_path, feature_path


def test_data_lineage_passes_for_clean_features(tmp_path: Path) -> None:
    features = list(CLEAN_FEATURE_COLUMNS)
    dataset_path, feature_path = _write_dataset_and_features(tmp_path, features)
    audit = PortfolioManagerDataLineageAudit(root=tmp_path, dataset_path=dataset_path, feature_columns_path=feature_path)

    result = audit.run()

    assert result["result"] == "PASS"
    assert result["forbidden_feature_hits"] == []
    assert result["label_feature_hits"] == []
    assert result["unknown_or_audit_feature_hits"] == []


def test_data_lineage_fails_for_actual_or_label_features(tmp_path: Path) -> None:
    features = ["expected_return_10d", "actual_net_profit", LABEL_COLUMNS[0]]
    dataset_path, feature_path = _write_dataset_and_features(tmp_path, features)
    audit = PortfolioManagerDataLineageAudit(root=tmp_path, dataset_path=dataset_path, feature_columns_path=feature_path)

    result = audit.run()

    assert result["result"] == "FAIL"
    assert "actual_net_profit" in result["forbidden_feature_hits"]
    assert LABEL_COLUMNS[0] in result["label_feature_hits"]


def test_data_lineage_report_is_written(tmp_path: Path) -> None:
    dataset_path, feature_path = _write_dataset_and_features(tmp_path, list(CLEAN_FEATURE_COLUMNS))
    report_path = tmp_path / "report.md"
    audit = PortfolioManagerDataLineageAudit(
        root=tmp_path,
        dataset_path=dataset_path,
        feature_columns_path=feature_path,
        report_path=report_path,
    )
    result = audit.run()
    paths = audit.save(result)

    assert paths.markdown.exists()
    assert "Conclusion" in paths.markdown.read_text(encoding="utf-8")
