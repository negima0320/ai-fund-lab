#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.walk_forward_diagnostics import WalkForwardDiagnosticsAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze walk-forward losing months and diagnostics.")
    parser.add_argument("--walk-forward-json", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = WalkForwardDiagnosticsAnalyzer()
    result = analyzer.analyze(args.walk_forward_json, args.start, args.end)
    md_path = analyzer.save_report(result)
    json_path = analyzer.save_json(result)
    csv_path = analyzer.save_losing_trades_csv(result)
    print(f"saved markdown report to {md_path}")
    print(f"saved json report to {json_path}")
    print(f"saved 2026-05 losing trades csv to {csv_path}")
    for item in result["diagnosis"]:
        print(f"diagnosis={item}")
    for row in result["monthly_top10_summary"]:
        print(
            f"month={row['month']} trades={row.get('trade_count')} "
            f"return_sum={row.get('return_sum')} win_rate={row.get('win_rate')} "
            f"bad_entry_rate={row.get('bad_entry_rate')} profit_factor={row.get('profit_factor')}"
        )


if __name__ == "__main__":
    main()
