#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_phase3d_detail_audit import PortfolioManagerPhase3DDetailAudit


def main() -> int:
    auditor = PortfolioManagerPhase3DDetailAudit(root=ROOT)
    result = auditor.build()
    paths = auditor.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    summary = result["phase3d_summary"]
    print(
        "phase3d "
        f"net_profit={summary['net_profit']} "
        f"pf={summary['profit_factor']} "
        f"dd={summary['max_drawdown']} "
        f"trades={summary['total_trades']}"
    )
    focus = result["focus_code_dependency"]
    print(
        "67400 "
        f"profit={focus['phase3d_profit']} "
        f"contribution={focus['phase3d_contribution_rate']}"
    )
    passed = [row for row in result["promotion_judgement"] if row["passed"]]
    total = len(result["promotion_judgement"])
    print(f"promotion_checks={len(passed)}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

