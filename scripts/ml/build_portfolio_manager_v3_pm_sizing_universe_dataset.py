#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_v3_pm_sizing_universe_builder import PMAIV3PMSizingUniverseDatasetBuilder


def main() -> None:
    builder = PMAIV3PMSizingUniverseDatasetBuilder(ROOT)
    report = builder.build()
    paths = builder.save_report(report)
    print(f"dataset={paths.dataset}")
    print(f"market_regime={paths.market_regime}")
    print(f"markdown={paths.markdown.relative_to(ROOT)}")
    print(f"json={paths.json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
