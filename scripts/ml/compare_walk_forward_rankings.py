#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.walk_forward_ranking_compare import WalkForwardRankingComparator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare bad-entry-aware walk-forward ranking strategies.")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--exit-rule", default="close_10d", choices=["close_10d"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparator = WalkForwardRankingComparator()
    result = comparator.compare(args.start, args.end, top_n=args.top_n, exit_rule=args.exit_rule)
    md_path = comparator.save_report(result)
    json_path = comparator.save_json(result)
    print(f"saved markdown report to {md_path}")
    print(f"saved json report to {json_path}")
    for row in result["summary"]:
        print(
            f"strategy={row['strategy']} total_return={row['total_return']} "
            f"win_rate={row['win_rate']} profit_factor={row['profit_factor']} "
            f"max_drawdown={row['max_drawdown']} bad_entry_rate={row['bad_entry_rate']} "
            f"may_return={row['may_return']} may_win_rate={row['may_win_rate']} "
            f"may_pf={row['may_profit_factor']} may_bad_entry_rate={row['may_bad_entry_rate']}"
        )


if __name__ == "__main__":
    main()
