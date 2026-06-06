#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.capital_allocation_phase7 import DEFAULT_PROFILES, CapitalAllocationPhase7AffordableFallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare phase7 affordable-fallback capital allocation profile.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES)
    parser.add_argument("--run-missing", action="store_true", help="Run missing backtests with cached data and no API fetch.")
    args = parser.parse_args()

    comparison = CapitalAllocationPhase7AffordableFallback(
        root=ROOT,
        profiles=list(args.profiles),
        start_date=args.start,
        end_date=args.end,
    )
    if args.run_missing:
        for row in comparison.run_missing_backtests():
            print(f"{row['profile']}: {row['status']}")
            if row["status"] == "failed":
                raise SystemExit(row.get("returncode") or 1)
    result = comparison.build()
    paths = comparison.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"fallback_trades_csv={paths.fallback_trades_csv}")
    for row in result["summary"]:
        print(
            f"{row['profile']}: final_assets={row.get('final_assets')} "
            f"net_profit={row.get('net_profit')} pf={row.get('profit_factor')} "
            f"dd={row.get('max_drawdown')} utilization={row.get('capital_utilization')} "
            f"fallback_buys={row.get('fallback_buy_count')}"
        )


if __name__ == "__main__":
    main()
