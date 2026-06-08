#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.phase9a_pm_ai_rearchitecture_audit import Phase9APMAIRearchitectureAudit  # noqa: E402


def main() -> None:
    audit = Phase9APMAIRearchitectureAudit(ROOT)
    report = audit.build_report()
    paths = audit.save_report(report)
    verdict = report["verdict"]
    leakage = report["leakage_risk_checklist"]
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"keep_current_v282={verdict.get('keep_current_v282')}")
    print(f"phase9b_ready={verdict.get('phase9b_ready')}")
    print(f"overall_leakage_risk={leakage.get('overall_leakage_risk')}")


if __name__ == "__main__":
    main()

