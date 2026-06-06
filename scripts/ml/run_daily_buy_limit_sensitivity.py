#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.daily_buy_limit_sensitivity import DailyBuyLimitSensitivity
from ml.daily_buy_limit_sensitivity import CONDITIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/report daily_buy_limit sensitivity for v2_73.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--run-backtests", action="store_true")
    parser.add_argument("--conditions", nargs="*", help="Condition names to include, e.g. fixed_500000 asset_ratio_050")
    args = parser.parse_args()

    conditions = CONDITIONS
    if args.conditions:
        requested = set(args.conditions)
        conditions = [condition for condition in CONDITIONS if condition["condition"] in requested]
    runner = DailyBuyLimitSensitivity(root=ROOT, start_date=args.start, end_date=args.end, conditions=conditions)
    profiles = runner.ensure_profiles()
    print("profiles:")
    for profile in profiles:
        print(f"- {profile}")
    if args.run_backtests:
        statuses = runner.run_backtests(profiles)
        for row in statuses:
            print(f"{row['profile']}: {row['status']}")
    result = runner.build(profiles)
    paths = runner.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"summary_csv={paths.summary_csv}")
    for row in result["summary"]:
        print(
            f"{row['condition']}: net_profit={row.get('net_profit')} "
            f"pf={row.get('profit_factor')} dd={row.get('max_drawdown')} "
            f"trades={row.get('total_trades')} scaled={row.get('scaled_buy_count')}"
        )


if __name__ == "__main__":
    main()
