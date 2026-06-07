#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_phase4b_high_pm_min_hold import PortfolioManagerPhase4BHighPMMinHoldAudit


def main() -> int:
    audit = PortfolioManagerPhase4BHighPMMinHoldAudit(root=ROOT)
    result = audit.build()
    paths = audit.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    for row in result["pm_multiplier_exit_quality"]:
        print(
            f"pm_multiplier={row['pm_multiplier']} "
            f"trades={row['trade_count']} "
            f"avg_holding={row['average_holding_days']} "
            f"avg_post_20d={row['average_post_exit_return_20d']} "
            f"early_exit_rate={row['early_exit_rate']:.4f} "
            f"good_exit_rate={row['good_exit_rate']:.4f}"
        )
    for row in result["minimum_hold_simulation"]:
        print(
            f"min_hold={row['minimum_hold_days']}d "
            f"changed={row['changed_trade_count']} "
            f"actual_profit={row['actual_net_profit']:.2f} "
            f"virtual_profit={row['virtual_net_profit']:.2f} "
            f"delta={row['profit_delta']:.2f} "
            f"virtual_pf={row['virtual_profit_factor']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
