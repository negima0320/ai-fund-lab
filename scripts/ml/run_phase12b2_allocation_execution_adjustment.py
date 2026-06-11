#!/usr/bin/env python3
"""Run Phase 12-B2 allocation execution adjustment."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12b2_allocation_execution_adjustment import Phase12B2AllocationExecutionAdjustment  # noqa: E402


def main() -> None:
    paths = Phase12B2AllocationExecutionAdjustment(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
