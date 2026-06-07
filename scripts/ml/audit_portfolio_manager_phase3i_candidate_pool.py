#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_phase3i_candidate_pool import PortfolioManagerPhase3ICandidatePoolAudit


def main() -> int:
    audit = PortfolioManagerPhase3ICandidatePoolAudit(root=ROOT)
    result = audit.build()
    paths = audit.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    for row in result["comparison"]:
        print(
            f"{row['variant']}: max_selected={row['max_selected']} "
            f"net_profit={row['net_profit']} pf={row['profit_factor']} "
            f"dd={row['max_drawdown']} avg_util={row['average_capital_utilization']} "
            f"no_candidates={row['no_candidates_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
