#!/usr/bin/env python3
"""Generate the Phase 11-C2 budget usage constraint audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ml.phase11c2_budget_usage_constraint_audit import Phase11C2BudgetUsageConstraintAudit, Phase11C2Options


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 11-C2 budget usage constraint audit")
    parser.add_argument("--root", default=str(ROOT), help="Repository root")
    parser.add_argument("--daily-buy-budget", type=float, default=300_000.0)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--per-code-cap-rate", type=float, default=0.38)
    args = parser.parse_args()
    audit = Phase11C2BudgetUsageConstraintAudit(
        Path(args.root),
        options=Phase11C2Options(
            daily_buy_budget=args.daily_buy_budget,
            initial_cash=args.initial_cash,
            per_code_cap_rate=args.per_code_cap_rate,
        ),
    )
    paths = audit.run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
