#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.scaled_buy_backtest_comparison import DEFAULT_PROFILES, ScaledBuyBacktestComparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scaled-buy derived backtest profile.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES)
    args = parser.parse_args()

    comparison = ScaledBuyBacktestComparison(root=ROOT, profiles=list(args.profiles), start_date=args.start, end_date=args.end)
    result = comparison.build()
    paths = comparison.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"scaled_buy_trades_csv={paths.scaled_buy_trades_csv}")
    for row in result["summary"]:
        print(
            f"{row['profile']}: final_assets={row.get('final_assets')} "
            f"net_profit={row.get('net_profit')} win_rate={row.get('win_rate')} "
            f"pf={row.get('profit_factor')} dd={row.get('max_drawdown')} "
            f"trades={row.get('total_trades')}"
        )
    scaled = result["scaled_buy_summary"]
    print(
        "scaled_buy: "
        f"trigger_count={scaled.get('scaled_buy_trigger_count')} "
        f"profit={scaled.get('scaled_buy_profit')}"
    )
    focus = result["focus_67400"]
    print(
        "67400: "
        f"bought={focus.get('bought')} shares={focus.get('shares')} "
        f"amount={focus.get('amount')} net_profit={focus.get('net_profit')}"
    )


if __name__ == "__main__":
    main()
