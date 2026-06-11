#!/usr/bin/env python3
"""Run Phase 11-B2 strict OOS failure diagnosis."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase11b2_strict_oos_failure_diagnosis import Phase11B2StrictOOSFailureDiagnosis  # noqa: E402


def main() -> None:
    paths = Phase11B2StrictOOSFailureDiagnosis(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
