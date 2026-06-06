#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_phase3e import PortfolioManagerPhase3EReporter


def main() -> None:
    reporter = PortfolioManagerPhase3EReporter(root=ROOT)
    result = reporter.build()
    paths = reporter.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    summaries = {row["profile"]: row for row in result["summaries"]}
    for profile, row in summaries.items():
        print(
            f"{profile}: net_profit={row['net_profit']:.2f} "
            f"pf={row['profit_factor']:.4f} dd={row['max_drawdown']:.4f} "
            f"trades={row['total_trades']}"
        )
    print(f"pm_low_score_skip_count={result['pm_low_score_skip_count']}")


if __name__ == "__main__":
    main()
