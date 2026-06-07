#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_phase3j_affordability import PortfolioManagerPhase3JAffordabilityAudit


def main() -> int:
    audit = PortfolioManagerPhase3JAffordabilityAudit(root=ROOT)
    result = audit.build()
    paths = audit.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"selected_but_not_affordable_count={result['selected_but_not_affordable_count']}")
    for row in result["reason_summary"]:
        print(f"reason={row['dominant_blocking_reason']} count={row['count']}")
    for row in result["improvement_candidates"]:
        print(f"candidate={row['candidate']} priority={row['priority']} signal={row['signal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
