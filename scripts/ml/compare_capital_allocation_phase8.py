#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.capital_allocation_phase8 import CapitalAllocationPhase8FallbackFilter
from ml.capital_allocation_phase8 import PRIORITY_CONDITIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare phase8 filtered affordable-fallback profiles.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--run-backtests", action="store_true", help="Run missing backtests with cached data and no API fetch.")
    parser.add_argument(
        "--conditions",
        nargs="*",
        help="Condition names to include. Default: priority 3 conditions.",
    )
    args = parser.parse_args()

    conditions = PRIORITY_CONDITIONS
    if args.conditions:
        requested = set(args.conditions)
        conditions = [condition for condition in PRIORITY_CONDITIONS if condition["condition"] in requested]

    runner = CapitalAllocationPhase8FallbackFilter(
        root=ROOT,
        start_date=args.start,
        end_date=args.end,
        conditions=conditions,
    )
    profiles = runner.ensure_profiles()
    print("profiles:")
    for profile in profiles:
        print(f"- {profile}")
    if args.run_backtests:
        for row in runner.run_missing_backtests():
            print(f"{row['profile']}: {row['status']}")
            if row["status"] == "failed":
                raise SystemExit(row.get("returncode") or 1)
    result = runner.build()
    paths = runner.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"summary_csv={paths.summary_csv}")
    for row in result["summary"]:
        print(
            f"{row['condition']}: net_profit={row.get('net_profit')} "
            f"pf={row.get('profit_factor')} dd={row.get('max_drawdown')} "
            f"utilization={row.get('capital_utilization')} "
            f"fallback_buys={row.get('fallback_buy_count')} "
            f"fallback_pf={row.get('fallback_profit_factor')}"
        )


if __name__ == "__main__":
    main()
