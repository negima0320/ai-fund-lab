#!/usr/bin/env python3
"""Run the Phase 11-C Capital Allocation Engine prototype."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ml.phase11c_capital_allocation_prototype import Phase11COptions, Phase11CCapitalAllocationPrototype


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 11-C capital allocation prototype")
    parser.add_argument("--root", default=str(ROOT), help="Repository root")
    parser.add_argument("--daily-buy-budget", type=float, default=300_000.0)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--per-code-cap-rate", type=float, default=0.38)
    parser.add_argument("--no-save-simulation", action="store_true")
    args = parser.parse_args()
    options = Phase11COptions(
        initial_cash=args.initial_cash,
        daily_buy_budget=args.daily_buy_budget,
        max_positions=args.max_positions,
        per_code_cap_rate=args.per_code_cap_rate,
        save_simulation=not args.no_save_simulation,
    )
    runner = Phase11CCapitalAllocationPrototype(Path(args.root), options=options)
    paths = runner.run()
    print(paths.markdown)
    print(paths.json)
    if paths.simulation is not None:
        print(paths.simulation)


if __name__ == "__main__":
    main()
