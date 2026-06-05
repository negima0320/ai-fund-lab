#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.backtest_diagnostics import DEFAULT_PROFILES, MLBacktestDiagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ML backtest integration diagnostics from existing logs.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES)
    args = parser.parse_args()

    diagnostics = MLBacktestDiagnostics(root=ROOT, profiles=list(args.profiles), start_date=args.start, end_date=args.end)
    result = diagnostics.build()
    paths = diagnostics.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"monthly_csv={paths.monthly_csv}")
    print(f"code_csv={paths.code_csv}")
    for row in result["summary"]:
        print(
            f"{row['profile']}: final_assets={row.get('final_assets')} "
            f"net_profit={row.get('net_profit')} win_rate={row.get('win_rate')} "
            f"pf={row.get('profit_factor')} dd={row.get('max_drawdown')}"
        )


if __name__ == "__main__":
    main()
