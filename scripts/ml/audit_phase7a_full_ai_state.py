#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase7a_full_ai_state_audit import Phase7AFullAIStateAudit  # noqa: E402


def main() -> int:
    audit = Phase7AFullAIStateAudit(ROOT)
    result = audit.build_report()
    paths = audit.save_report(result)
    verdict = result.get("final_verdict", {})
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"recommended_first_retraining_target={verdict.get('recommended_first_retraining_target')}")
    print(f"next_phase_recommended={verdict.get('next_phase_recommended')}")
    print(f"retraining_should_start_now={verdict.get('retraining_should_start_now')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

