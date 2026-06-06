#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_data_lineage import PortfolioManagerDataLineageAudit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Portfolio Manager AI data lineage and feature cleanliness.")
    parser.add_argument("--dataset", default="data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet")
    parser.add_argument("--feature-columns", default="models/ml/portfolio_manager/current_v2_73_phase3b_clean/feature_columns.json")
    parser.add_argument("--report", default="reports/ml/portfolio_manager_data_lineage_audit_2023-01_to_2026-05.md")
    args = parser.parse_args()

    audit = PortfolioManagerDataLineageAudit(
        root=ROOT,
        dataset_path=args.dataset,
        feature_columns_path=args.feature_columns,
        report_path=args.report,
    )
    result = audit.run()
    paths = audit.save(result)
    print(f"result={result['result']}")
    print(f"report={paths.markdown}")
    print(f"feature_count={result['feature_count']}")
    print(f"forbidden_feature_hits={result['forbidden_feature_hits']}")
    print(f"label_feature_hits={result['label_feature_hits']}")
    print(f"unknown_or_audit_feature_hits={result['unknown_or_audit_feature_hits']}")


if __name__ == "__main__":
    main()
