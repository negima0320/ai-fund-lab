#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase8a_system_understanding_audit import Phase8ASystemUnderstandingAudit  # noqa: E402


def main() -> int:
    audit = Phase8ASystemUnderstandingAudit(ROOT)
    result = audit.build_report()
    paths = audit.save_report(result)
    verdict = result.get("final_verdict", {})
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"why_v282_wins={verdict.get('why_v282_wins')}")
    print(f"main_alpha_source={verdict.get('main_alpha_source')}")
    print(f"next_phase_recommended={verdict.get('next_phase_recommended')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

