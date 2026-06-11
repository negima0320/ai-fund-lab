#!/usr/bin/env python3
"""Run Phase 12-E2 Stock Selection architecture audit."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase12e2_stock_selection_architecture_audit import Phase12E2StockSelectionArchitectureAudit  # noqa: E402


def main() -> None:
    paths = Phase12E2StockSelectionArchitectureAudit(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
