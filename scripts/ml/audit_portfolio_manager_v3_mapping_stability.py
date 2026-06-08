#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_v3_mapping_stability_audit import PMAIV3MappingStabilityAudit


def main() -> None:
    audit = PMAIV3MappingStabilityAudit(ROOT)
    report = audit.build_report()
    paths = audit.save_report(report)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"best_mapping_by_stability={report['conclusion']['best_mapping_by_stability']}")
    print(f"mapping_d_is_stable={report['conclusion']['mapping_d_is_stable']}")
    print(f"leakage_risk={report['leakage_checklist']['leakage_risk']}")


if __name__ == "__main__":
    main()
