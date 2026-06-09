#!/usr/bin/env python3
"""Run Phase 9-E2 PM AI v3 integration audit on PM sizing universe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_v3_integration_audit_pm_sizing_universe import PMAIV3IntegrationAuditPMSizingUniverse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    audit = PMAIV3IntegrationAuditPMSizingUniverse(Path(args.root))
    report = audit.build_report()
    paths = audit.save_report(report)
    best = report["best_mapping"]
    coverage = report["coverage_audit"]
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"coverage_rate={coverage.get('coverage_rate')}")
    print(f"best_mapping={best.get('best_mapping_name')}")
    print(f"confidence={best.get('confidence_level')}")
    print(f"phase9f2_worth_testing={best.get('phase9f2_backtest_worth_testing')}")
    print(f"leakage_risk={report['leakage_checklist']['leakage_risk']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

