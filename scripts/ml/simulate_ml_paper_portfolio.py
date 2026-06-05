#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.paper_portfolio import MLPaperPortfolioSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate report-only paper portfolios from ML top-N rankings.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--top-n", type=int, default=10, help="Daily ranking size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulator = MLPaperPortfolioSimulator()
    result = simulator.simulate(args.start, args.end, top_n=args.top_n)
    md_path = simulator.save_report(result)
    json_path = simulator.save_json(result)
    csv_path = simulator.save_trades_csv(result)
    print(f"saved markdown report to {md_path}")
    print(f"saved json report to {json_path}")
    print(f"saved paper trades csv to {csv_path}")
    for row in result["ranking_exit_summary"]:
        print(
            "ranking_exit="
            f"{row['ranking']} {row['exit_rule']} "
            f"trades={row['total_trades']} "
            f"win_rate={row['win_rate']} "
            f"average_return={row['average_return']} "
            f"total_return_sum={row['total_return_sum']} "
            f"profit_factor={row['profit_factor']} "
            f"max_drawdown={row['max_drawdown']}"
        )


if __name__ == "__main__":
    main()
