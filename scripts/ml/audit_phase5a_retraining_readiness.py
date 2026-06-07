#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase5a_retraining_readiness import Phase5ARetrainingReadinessAudit  # noqa: E402


def main() -> int:
    audit = Phase5ARetrainingReadinessAudit(ROOT)
    result = audit.build_report()
    paths = audit.save_report(result)
    priority = result.get("retraining_priority", {}).get("recommended_order", [])
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"recommended_next_phase={result.get('recommended_next_phase')}")
    print(f"recommended_order={','.join(priority)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

