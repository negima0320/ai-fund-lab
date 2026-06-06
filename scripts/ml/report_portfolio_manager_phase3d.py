#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_phase3d import PortfolioManagerPhase3DReporter


def main() -> None:
    reporter = PortfolioManagerPhase3DReporter(root=ROOT)
    result = reporter.build()
    paths = reporter.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    for row in result["comparison"]:
        print(f"{row['metric']}: baseline={row['baseline']} phase3d={row['phase3d']} delta={row['delta']}")


if __name__ == "__main__":
    main()
