#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_phase3g import PortfolioManagerPhase3GReporter


def main() -> None:
    reporter = PortfolioManagerPhase3GReporter(root=ROOT)
    result = reporter.build()
    paths = reporter.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    for row in result["summary_comparison"]:
        print(
            f"{row['profile']}: net_profit={row['net_profit']:.2f} "
            f"pf={row['profit_factor']:.4f} dd={row['max_drawdown']:.4f} "
            f"win_rate={row['win_rate']:.4f} trades={row['total_trades']}"
        )
    for row in result["cap_rate_sweep"]:
        print(
            f"cap_rate={row['cap_rate']}: net_profit={row['net_profit']:.2f} "
            f"pf={row['profit_factor']:.4f} dd={row['max_drawdown']:.4f} "
            f"cap_reductions={row['per_code_exposure_cap_reduction_count']} "
            f"cap_skips={row['per_code_exposure_cap_skip_count']}"
        )


if __name__ == "__main__":
    main()
