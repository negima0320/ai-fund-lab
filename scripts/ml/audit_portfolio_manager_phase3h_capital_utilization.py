#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_phase3h_capital_utilization import PortfolioManagerPhase3HCapitalUtilizationAudit


def main() -> int:
    audit = PortfolioManagerPhase3HCapitalUtilizationAudit(root=ROOT)
    result = audit.build()
    paths = audit.save(result)
    target = result["capital_utilization_distribution"]["v2_77_cap_030"]
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(
        "v2_77_cap_030: "
        f"avg_util={target.get('average_capital_utilization')} "
        f"median_util={target.get('median_capital_utilization')} "
        f"days_below_50pct={target.get('days_below_50pct')}"
    )
    for row in result["bottleneck_flags"]:
        print(f"{row['flag']}={row['value']} detail={row['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
