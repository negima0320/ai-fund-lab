#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.position_sizing_phase1 import DEFAULT_PROFILES
from ml.position_sizing_phase2 import PositionSizingPhase2SoftRules


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-trade soft AI position sizing simulation.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES)
    args = parser.parse_args()

    simulation = PositionSizingPhase2SoftRules(
        root=ROOT,
        profiles=list(args.profiles),
        start_date=args.start,
        end_date=args.end,
    )
    result = simulation.build()
    paths = simulation.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"summary_csv={paths.summary_csv}")
    for row in result["summary"]:
        print(
            f"{row['profile']} {row['sizing_rule']}: "
            f"adjusted_net_profit={row.get('adjusted_net_profit')} "
            f"delta={row.get('profit_delta')} pf={row.get('profit_factor')} "
            f"dd={row.get('max_drawdown')} 67400={row.get('focus_67400_contribution')}"
        )


if __name__ == "__main__":
    main()
