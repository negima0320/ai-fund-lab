#!/usr/bin/env python3
"""Run Phase 11-B3 expected downside model prototype."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase11b3_expected_downside_model import Phase11B3ExpectedDownsideModel  # noqa: E402


def main() -> None:
    paths = Phase11B3ExpectedDownsideModel(ROOT).run()
    print(paths.markdown)
    print(paths.json)
    if paths.model_dir:
        print(paths.model_dir)


if __name__ == "__main__":
    main()
