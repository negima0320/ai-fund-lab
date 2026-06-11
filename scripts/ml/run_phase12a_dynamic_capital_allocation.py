#!/usr/bin/env python3
"""Run Phase 12-A dynamic capital allocation quality audit."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12a_dynamic_capital_allocation import Phase12ADynamicCapitalAllocation  # noqa: E402


def main() -> None:
    paths = Phase12ADynamicCapitalAllocation(ROOT).run()
    print(paths.markdown)
    print(paths.json)
    if paths.artifact:
        print(paths.artifact)


if __name__ == "__main__":
    main()
