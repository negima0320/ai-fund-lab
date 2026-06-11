#!/usr/bin/env python3
"""Run Phase 12-B3 exit / hold decision audit."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12b3_exit_hold_audit import Phase12B3ExitHoldAudit  # noqa: E402


def main() -> None:
    paths = Phase12B3ExitHoldAudit(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
