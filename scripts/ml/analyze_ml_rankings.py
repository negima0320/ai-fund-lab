#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.ranking_analysis import MLRankingAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze all-stock ML ranking performance against labels.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--top-n", type=int, default=10, help="Daily ranking size.")
    parser.add_argument("--profile", help="Optional backtest profile for overlap analysis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = MLRankingAnalyzer()
    analysis = analyzer.analyze(args.start, args.end, top_n=args.top_n, profile=args.profile)
    md_path = analyzer.save_report(analysis)
    json_path = analyzer.save_json(analysis)
    csv_path = analyzer.save_details_csv(analysis)

    print(f"saved markdown report to {md_path}")
    print(f"saved json report to {json_path}")
    print(f"saved ranking details csv to {csv_path}")
    print(f"processed_dates={len(analysis['processed_dates'])}")
    print(f"skipped_dates={len(analysis['skipped_dates'])}")
    baseline = analysis["baseline_all_stocks"]
    print(
        "baseline "
        f"count={baseline['count']} "
        f"future_max_return_20d_mean={baseline['future_max_return_20d_mean']} "
        f"swing_success_rate={baseline['future_swing_success_20d_rate']} "
        f"bad_entry_rate={baseline['bad_entry_10d_rate']}"
    )
    for row in analysis["ranking_summary"]:
        print(
            "ranking="
            f"{row['ranking']} count={row['count']} "
            f"future_max_return_20d_mean={row['future_max_return_20d_mean']} "
            f"swing_success_rate={row['future_swing_success_20d_rate']} "
            f"bad_entry_rate={row['bad_entry_10d_rate']} "
            f"date_count={row['date_count']}"
        )
    for row in analysis["overlap_summary"]:
        print(
            "overlap="
            f"{row['ranking']} bought_count={row['bought_count']} "
            f"not_bought_count={row['not_bought_count']} "
            f"ranked_bought_rate={row['ranked_bought_rate']} "
            f"existing_trade_topn_rate={row['existing_trade_topn_rate']}"
        )


if __name__ == "__main__":
    main()
