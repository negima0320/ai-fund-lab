#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_v3_dataset_builder import PMAIV3CleanDatasetBuilder  # noqa: E402


def main() -> None:
    builder = PMAIV3CleanDatasetBuilder(ROOT)
    report = builder.build()
    paths = builder.save_report(report)
    summary = report["dataset_summary"]
    leakage = report["leakage_audit"]
    print(f"dataset={paths.dataset}")
    print(f"market_regime={paths.market_regime}")
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"row_count={summary.get('row_count')}")
    print(f"date_min={summary.get('date_min')}")
    print(f"date_max={summary.get('date_max')}")
    print(f"code_count={summary.get('code_count')}")
    print(f"feature_count={len(report.get('feature_columns', []))}")
    print(f"label_count={len(report.get('label_columns', []))}")
    print(f"forbidden_feature_count={leakage.get('forbidden_feature_count')}")
    print(f"leakage_risk={leakage.get('leakage_risk')}")


if __name__ == "__main__":
    main()

