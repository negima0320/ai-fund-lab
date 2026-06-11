#!/usr/bin/env python3
"""Run Phase 13-D2 Hold / Exit Label Refinement."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase13d2_hold_exit_label_refinement import Phase13D2HoldExitLabelRefinement  # noqa: E402


def main() -> None:
    paths = Phase13D2HoldExitLabelRefinement(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
