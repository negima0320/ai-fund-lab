#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.evaluator import PredictionEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one day of ML predictions against labels.")
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top ml_score rows to summarize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = PredictionEvaluator()
    evaluation = evaluator.evaluate_daily(args.date, top_n=args.top_n)
    path = evaluator.save_report(evaluation, args.date)
    print(f"saved evaluation report to {path}")
    print(f"joined_rows={evaluation['joined_rows']}")
    print(f"top_n={evaluation['top_n']}")
    print(f"expected_vs_future_10d_corr={evaluation['expected_vs_future_10d_corr']}")


if __name__ == "__main__":
    main()
