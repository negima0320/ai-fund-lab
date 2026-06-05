#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.exit_analysis import MLExitAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze report-only ML exit rules from existing backtest trades.")
    parser.add_argument("--profile", default="rookie_dealer_02_v2_66_ml_ranked")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    args = parser.parse_args()

    analyzer = MLExitAnalyzer(root=ROOT, profile=args.profile, start_date=args.start, end_date=args.end)
    result = analyzer.build()
    paths = analyzer.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"trades_csv={paths.trades_csv}")
    for row in result["rules"]:
        print(
            f"{row['rule']}: total_profit={row['total_profit']:.2f} "
            f"delta={row['profit_delta']:.2f} changed={row['exit_changed_count']} "
            f"pf={row['profit_factor']}"
        )


if __name__ == "__main__":
    main()
