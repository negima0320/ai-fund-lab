#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_phase1 import PortfolioManagerPhase1Simulation
from ml.portfolio_manager_phase1 import PROFILE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-trade portfolio manager allocation simulation.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--profile", default=PROFILE)
    args = parser.parse_args()

    simulation = PortfolioManagerPhase1Simulation(
        root=ROOT,
        profile=args.profile,
        start_date=args.start,
        end_date=args.end,
    )
    result = simulation.build()
    paths = simulation.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"daily_allocations_csv={paths.daily_allocations_csv}")
    print(f"trade_allocations_csv={paths.trade_allocations_csv}")
    for row in result["summary"]:
        print(
            f"{row['portfolio_rule']}: adjusted_net_profit={row.get('adjusted_net_profit')} "
            f"delta={row.get('profit_delta')} pf={row.get('profit_factor')} "
            f"dd={row.get('max_drawdown')} reserve={row.get('average_cash_reserve_rate')} "
            f"67400={row.get('focus_67400_contribution')}"
        )


if __name__ == "__main__":
    main()
