#!/usr/bin/env python3
"""Run Phase 12-B5 exit threshold recalibration."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12b5_exit_threshold_recalibration import Phase12B5ExitThresholdRecalibration  # noqa: E402


def main() -> None:
    paths = Phase12B5ExitThresholdRecalibration(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
