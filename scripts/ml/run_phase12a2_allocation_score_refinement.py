#!/usr/bin/env python3
"""Run Phase 12-A2 allocation score refinement."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12a2_allocation_score_refinement import Phase12A2AllocationScoreRefinement  # noqa: E402


def main() -> None:
    paths = Phase12A2AllocationScoreRefinement(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
