#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_v3_mapping_threshold_audit import PMAIV3MappingThresholdAudit


def main() -> None:
    audit = PMAIV3MappingThresholdAudit(ROOT)
    report = audit.build_report()
    paths = audit.save_report(report)
    rec = report.get("recommended_threshold_config", {})
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"threshold_candidates={report['threshold_grid']['candidate_count']}")
    print(f"recommended_config={rec.get('config_id')}")
    print(f"phase9e2_worth_testing={report['phase9e2_integration_audit_worth_testing']}")
    print(f"leakage_risk={report['leakage_checklist']['leakage_risk']}")


if __name__ == "__main__":
    main()
