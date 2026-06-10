#!/usr/bin/env python3
"""Run Phase 11-E limited exit / DD guard experiment for 2025 only."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase11e_exit_dd_guard import Phase11EExitDDGuard  # noqa: E402


def main() -> None:
    paths = Phase11EExitDDGuard(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
