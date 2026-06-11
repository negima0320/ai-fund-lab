#!/usr/bin/env python3
"""Run Phase 12-D2 buy quality reality audit."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12d2_buy_quality_reality_audit import Phase12D2BuyQualityRealityAudit  # noqa: E402


def main() -> None:
    paths = Phase12D2BuyQualityRealityAudit(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
