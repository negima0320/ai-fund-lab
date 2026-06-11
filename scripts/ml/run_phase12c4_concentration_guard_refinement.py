#!/usr/bin/env python3
"""Run Phase 12-C4 concentration guard refinement."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12c4_concentration_guard_refinement import Phase12C4ConcentrationGuardRefinement  # noqa: E402


def main() -> None:
    paths = Phase12C4ConcentrationGuardRefinement(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
