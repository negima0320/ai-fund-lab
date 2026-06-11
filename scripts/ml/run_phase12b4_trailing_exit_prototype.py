#!/usr/bin/env python3
"""Run Phase 12-B4 trailing exit prototype."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12b4_trailing_exit_prototype import Phase12B4TrailingExitPrototype  # noqa: E402


def main() -> None:
    paths = Phase12B4TrailingExitPrototype(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
