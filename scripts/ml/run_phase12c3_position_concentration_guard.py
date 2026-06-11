#!/usr/bin/env python3
"""Run Phase 12-C3 position concentration guard check."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12c3_position_concentration_guard import Phase12C3PositionConcentrationGuard  # noqa: E402


def main() -> None:
    paths = Phase12C3PositionConcentrationGuard(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
