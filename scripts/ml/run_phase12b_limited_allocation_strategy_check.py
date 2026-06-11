#!/usr/bin/env python3
"""Run Phase 12-B limited allocation strategy check."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12b_limited_allocation_strategy_check import Phase12BLimitedAllocationStrategyCheck  # noqa: E402


def main() -> None:
    paths = Phase12BLimitedAllocationStrategyCheck(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
