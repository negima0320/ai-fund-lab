#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.capital_allocation_audit import CapitalAllocationAudit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit capital allocation changes after Exit AI backtest exits.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--period-key", default="2023-01-01_to_2026-05-31")
    parser.add_argument("--focus-month", default="2026-03")
    parser.add_argument("--focus-code", default="67400")
    args = parser.parse_args()

    audit = CapitalAllocationAudit(
        root=args.root,
        period_key=args.period_key,
        focus_month=args.focus_month,
        focus_code=args.focus_code,
    )
    result = audit.build()
    paths = audit.save(result)
    focus = result["focus_trade"]
    month = result["month_trade_diff"]
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"focus_code={focus['code']}")
    print(f"base_net_profit={focus['base_net_profit']}")
    print(f"exit_profile_same_signal_trade={focus['exit_profile_same_signal_trade']}")
    print(f"capital_blocked_reason={focus['capital_blocked_reason']}")
    print(f"base_only_profit_{args.focus_month}={month['base_only_profit']:.2f}")
    print(f"exit_only_profit_{args.focus_month}={month['exit_only_profit']:.2f}")


if __name__ == "__main__":
    main()
