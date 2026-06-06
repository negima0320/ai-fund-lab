#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.scaled_buy_audit import PERIOD, PROFILE, ScaledBuyAudit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v2_71 scaled-buy concentration and side effects.")
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--period", default=PERIOD)
    parser.add_argument("--daily-buy-limit", type=float, default=900_000)
    args = parser.parse_args()

    audit = ScaledBuyAudit(root=ROOT, profile=args.profile, period_key=args.period, daily_buy_limit=args.daily_buy_limit)
    result = audit.build()
    paths = audit.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"trades_csv={paths.trades_csv}")
    stats = result["scaled_buy_stats"]
    concentration = result["concentration"]["summary"]
    print(
        "scaled_buy: "
        f"count={stats.get('count')} total_profit={stats.get('total_profit')} "
        f"win_rate={stats.get('win_rate')} pf={stats.get('profit_factor')}"
    )
    print(
        "concentration: "
        f"67400_rate={concentration.get('67400_contribution_rate')} "
        f"top3_rate={concentration.get('top3_code_contribution_rate')} "
        f"top5_rate={concentration.get('top5_code_contribution_rate')}"
    )
    for row in result["profile_comparison"]:
        print(
            f"{row['case']}: net_profit={row.get('net_profit')} "
            f"delta_vs_v2_68={row.get('profit_delta_vs_v2_68')} trades={row.get('trade_count')}"
        )


if __name__ == "__main__":
    main()
