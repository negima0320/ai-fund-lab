#!/usr/bin/env python3
"""Run Phase 11-G limited 2024 out-of-sample year check."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase11g_out_of_sample_check import Phase11GOutOfSampleCheck  # noqa: E402


def main() -> None:
    paths = Phase11GOutOfSampleCheck(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
