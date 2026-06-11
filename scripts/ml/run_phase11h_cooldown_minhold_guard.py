#!/usr/bin/env python3
"""Run Phase 11-H cooldown / minimum holding guard check."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase11h_cooldown_minhold_guard import Phase11HCooldownMinHoldGuard  # noqa: E402


def main() -> None:
    paths = Phase11HCooldownMinHoldGuard(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
