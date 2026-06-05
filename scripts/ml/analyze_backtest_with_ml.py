#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.backtest_ml_analysis import BacktestMLAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join existing backtest trades with ML predictions for report-only analysis.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", help="Backtest profile id under logs/backtests.")
    source.add_argument("--trades-csv", help="Direct path to an existing trades.csv.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top/bottom ml_score trades to summarize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = BacktestMLAnalyzer()
    if args.profile:
        analysis = analyzer.analyze_profile(args.profile, args.start, args.end, top_n=args.top_n)
    else:
        analysis = analyzer.analyze_trades_csv(args.trades_csv, args.start, args.end, top_n=args.top_n)
    md_path = analyzer.save_report(analysis)
    json_path = analyzer.save_json(analysis)
    win_loss_md_path = analyzer.save_win_loss_report(analysis)
    win_loss_json_path = analyzer.save_win_loss_json(analysis)
    ml_trades_csv_path = analyzer.save_ml_trades_csv(analysis)

    join = analysis["join_summary"]
    print(f"saved markdown report to {md_path}")
    print(f"saved json report to {json_path}")
    print(f"saved win/loss markdown report to {win_loss_md_path}")
    print(f"saved win/loss json report to {win_loss_json_path}")
    print(f"saved ML trades csv to {ml_trades_csv_path}")
    print(f"trades_csv={analysis['source']['trades_csv']}")
    print(f"trade_rows={join['trade_rows']}")
    print(f"joined_count={join['joined_count']}")
    print(f"missing_count={join['missing_count']}")
    print(f"join_rate={join['join_rate']}")
    for row in analysis["risk_label_performance"]:
        print(
            "risk_label="
            f"{row['entry_risk_label']} count={row['count']} "
            f"win_rate={row['win_rate']} net_profit_total={row['net_profit_total']} "
            f"net_profit_rate_mean={row['net_profit_rate_mean']}"
        )
    win_loss = analysis.get("win_loss_analysis", {})
    for row in win_loss.get("ml_average_by_result", []):
        print(
            "win_loss="
            f"{row['win_loss']} count={row['count']} "
            f"expected_return_10d_mean={row['expected_return_10d_mean']} "
            f"upside_probability_10d_mean={row['upside_probability_10d_mean']} "
            f"bad_entry_probability_10d_mean={row['bad_entry_probability_10d_mean']} "
            f"ml_score_mean={row['ml_score_mean']} "
            f"net_profit_total={row['net_profit_total']}"
        )
    for row in win_loss.get("danger_win_loss_difference", []):
        print(
            "danger_bucket="
            f"{row['bucket']} count={row['count']} "
            f"expected_return_10d_mean={row['expected_return_10d_mean']} "
            f"upside_probability_10d_mean={row['upside_probability_10d_mean']} "
            f"bad_entry_probability_10d_mean={row['bad_entry_probability_10d_mean']} "
            f"ml_score_mean={row['ml_score_mean']} "
            f"net_profit_rate_mean={row['net_profit_rate_mean']}"
        )
    for warning in analysis["warnings"]:
        print(f"warning={warning}")


if __name__ == "__main__":
    main()
