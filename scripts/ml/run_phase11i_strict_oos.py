#!/usr/bin/env python3
"""Run Phase 11-I strict walk-forward OOS prototype."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase11i_strict_oos import Phase11IStrictOOS  # noqa: E402


def main() -> None:
    paths = Phase11IStrictOOS(ROOT).run()
    print(paths.markdown)
    print(paths.json)
    if paths.model_dir:
        print(paths.model_dir)


if __name__ == "__main__":
    main()
