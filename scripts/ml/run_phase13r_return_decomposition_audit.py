#!/usr/bin/env python3
"""Run Phase 13-R Return Decomposition / Strategy Requirement Audit."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase13r_return_decomposition_audit import Phase13RReturnDecompositionAudit  # noqa: E402


def main() -> None:
    paths = Phase13RReturnDecompositionAudit(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
