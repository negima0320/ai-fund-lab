#!/usr/bin/env python3
"""Run Phase 13-E2 Integrated Strategy Rule Tuning."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase13e2_integrated_strategy_rule_tuning import Phase13E2IntegratedStrategyRuleTuning  # noqa: E402


def main() -> None:
    paths = Phase13E2IntegratedStrategyRuleTuning(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
