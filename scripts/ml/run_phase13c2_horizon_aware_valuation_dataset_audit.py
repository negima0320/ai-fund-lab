#!/usr/bin/env python3
"""Run Phase 13-C2 Horizon-Aware Valuation Training Dataset Audit."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase13c2_horizon_aware_valuation_dataset_audit import Phase13C2HorizonAwareValuationDatasetAudit  # noqa: E402


def main() -> None:
    paths = Phase13C2HorizonAwareValuationDatasetAudit(ROOT).run()
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
