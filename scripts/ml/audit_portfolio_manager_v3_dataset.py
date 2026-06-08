#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_v3_dataset_audit import PMAIV3DatasetQualityAudit  # noqa: E402


def main() -> None:
    audit = PMAIV3DatasetQualityAudit(ROOT)
    report = audit.build_report()
    paths = audit.save_report(report)
    summary = report["basic_quality"]["summary"]
    verdict = report["verdict"]
    leakage = report["leakage_audit"]
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"dataset_is_trainable={verdict.get('dataset_is_trainable')}")
    print(f"row_count={summary.get('row_count')}")
    print(f"date_min={summary.get('date_min')}")
    print(f"date_max={summary.get('date_max')}")
    print(f"code_count={summary.get('code_count')}")
    print(f"feature_count={summary.get('feature_count')}")
    print(f"label_count={summary.get('label_count')}")
    print(f"leakage_risk={leakage.get('leakage_risk')}")
    print(f"next_phase={verdict.get('next_phase_recommendation')}")


if __name__ == "__main__":
    main()

