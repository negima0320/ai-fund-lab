#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_phase3f_drawdown import PortfolioManagerPhase3FDrawdownAudit


def main() -> None:
    audit = PortfolioManagerPhase3FDrawdownAudit(root=ROOT)
    result = audit.build()
    paths = audit.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    for key, row in result["drawdown_windows"].items():
        print(
            f"{key}: dd={row['max_drawdown']:.4f} "
            f"start={row['start_date']} trough={row['trough_date']} "
            f"recovery={row['recovery_date']} amount={row['drawdown_amount']:.2f}"
        )
    same = result["same_period_comparison"]
    print(
        "same_period: "
        f"{same['period_start']}..{same['period_end']} "
        f"phase3d_profit={same['phase3d_period_profit']:.2f} "
        f"phase3e_profit={same['phase3e_period_profit']:.2f} "
        f"delta={same['delta']:.2f}"
    )


if __name__ == "__main__":
    main()
