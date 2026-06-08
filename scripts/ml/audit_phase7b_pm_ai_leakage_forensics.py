#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase7b_pm_ai_leakage_forensics import Phase7BPMLeakageForensics  # noqa: E402


def main() -> int:
    audit = Phase7BPMLeakageForensics(ROOT)
    result = audit.build_report()
    paths = audit.save_report(result)
    safety = result.get("v2_82_safety_impact_assessment", {})
    judgement = result.get("retraining_judgement", {})
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"current_pm_model_operational_risk={safety.get('current_pm_model_operational_risk')}")
    print(f"v282_result_trust_level={safety.get('v282_result_trust_level')}")
    print(f"next_phase_recommended={judgement.get('next_phase_recommended')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

