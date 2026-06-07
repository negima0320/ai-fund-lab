#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_phase4a_exit_quality import PortfolioManagerPhase4AExitQualityAudit


def main() -> int:
    audit = PortfolioManagerPhase4AExitQualityAudit(root=ROOT)
    result = audit.build()
    paths = audit.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    for row in result["profile_summary"]:
        print(
            f"{row['profile']}: "
            f"trades={row['trade_count']} "
            f"profit={row['realized_net_profit']:.2f} "
            f"avg_post_10d={row.get('average_post_exit_return_10d')} "
            f"avg_post_20d={row.get('average_post_exit_return_20d')} "
            f"early_exit_rate={row['early_exit_rate']:.4f} "
            f"good_exit_rate={row['good_exit_rate']:.4f} "
            f"loss_cut_success_rate={row['loss_cut_success_rate']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
