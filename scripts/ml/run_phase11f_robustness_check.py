#!/usr/bin/env python3
"""Run Phase 11-F limited robustness check for 2025 only."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase11f_robustness_check import Phase11FRobustnessCheck  # noqa: E402


def main() -> None:
    paths = Phase11FRobustnessCheck(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
