#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.walk_forward import MLWalkForwardRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanding-window ML walk-forward evaluation.")
    parser.add_argument("--train-start", default="2025-06-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-05-31")
    parser.add_argument("--ranking", default="expected_return_10d", choices=["expected_return_10d"])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--exit-rule", default="close_10d", choices=["close_10d"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = MLWalkForwardRunner()
    result = runner.run(
        train_start=args.train_start,
        test_start=args.test_start,
        test_end=args.test_end,
        ranking=args.ranking,
        top_n=args.top_n,
        exit_rule=args.exit_rule,
    )
    md_path = runner.save_report(result)
    json_path = runner.save_json(result)
    print(f"saved markdown report to {md_path}")
    print(f"saved json report to {json_path}")
    overall = result["overall"]
    print(
        "overall "
        f"total_trades={overall['total_trades']} "
        f"win_rate={overall['win_rate']} "
        f"total_return={overall['total_return']} "
        f"profit_factor={overall['profit_factor']} "
        f"max_drawdown={overall['max_drawdown']}"
    )
    for row in result["folds"]:
        print(
            f"month={row['month']} "
            f"train_rows={row['train_rows']} valid_rows={row['valid_rows']} "
            f"predicted_dates={row['predicted_dates']} trades={row['total_trades']} "
            f"monthly_return={row['monthly_return']} win_rate={row['win_rate']} "
            f"profit_factor={row['profit_factor']} max_drawdown={row['max_drawdown']}"
        )
    for warning in result.get("warnings", []):
        print(f"warning={warning}")


if __name__ == "__main__":
    main()
