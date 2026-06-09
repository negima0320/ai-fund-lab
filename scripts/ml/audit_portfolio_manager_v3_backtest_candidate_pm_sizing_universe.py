#!/usr/bin/env python3
"""Run Phase 9-F2 PM AI v3 backtest candidate audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_v3_backtest_candidate_audit_pm_sizing_universe import PMAIV3BacktestCandidateAuditPMSizingUniverse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    audit = PMAIV3BacktestCandidateAuditPMSizingUniverse(Path(args.root))
    report = audit.build_report()
    paths = audit.save_report(report)
    adoption = report["adoption_gate"]
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"best_candidate={adoption.get('best_candidate_label')}")
    print(f"adoption={adoption.get('adoption_recommendation')}")
    print(f"leakage_risk={report['leakage_checklist']['leakage_risk']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

