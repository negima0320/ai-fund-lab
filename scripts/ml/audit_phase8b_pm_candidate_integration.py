#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase8b_pm_candidate_integration_audit import Phase8BPMCandidateIntegrationAudit  # noqa: E402


def main() -> int:
    audit = Phase8BPMCandidateIntegrationAudit(ROOT)
    report = audit.build_report()
    paths = audit.save_report(report)
    trust = report.get("trust_verdict", {})
    verdict = report.get("final_verdict", {})
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"candidate_pm_safe={trust.get('candidate_pm_safe')}")
    print(f"candidate_pm_better_than_current={trust.get('candidate_pm_better_than_current')}")
    print(f"candidate_pm_worth_backtesting={trust.get('candidate_pm_worth_backtesting')}")
    print(f"next_phase_recommended={verdict.get('next_phase_recommended')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

