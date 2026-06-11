#!/usr/bin/env python3
"""Run Phase 12-A3 top5 penalty refinement."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12a3_top5_penalty_refinement import Phase12A3Top5PenaltyRefinement  # noqa: E402


def main() -> None:
    paths = Phase12A3Top5PenaltyRefinement(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
