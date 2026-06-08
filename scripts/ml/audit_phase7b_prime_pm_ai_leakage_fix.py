#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase7b_prime_pm_ai_leakage_fix import Phase7BPrimePMLeakageFixAudit  # noqa: E402


def main() -> int:
    audit = Phase7BPrimePMLeakageFixAudit(ROOT)
    result = audit.build_report()
    paths = audit.save_report(result)
    judgement = result.get("final_judgement", {})
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"feature_leakage_confirmed={judgement.get('feature_leakage_confirmed')}")
    print(f"feature_leakage_suspected={judgement.get('feature_leakage_suspected')}")
    print(f"v282_result_trust_level={judgement.get('v282_result_trust_level')}")
    print(f"next_phase_recommended={judgement.get('next_phase_recommended')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

