#!/usr/bin/env python3
"""Run Phase 13-D3 Hold / Exit Rule Prototype."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase13d3_hold_exit_rule_prototype import Phase13D3HoldExitRulePrototype  # noqa: E402


def main() -> None:
    paths = Phase13D3HoldExitRulePrototype(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
