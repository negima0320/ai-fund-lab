#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase6d_bear_booster_design_audit import Phase6DBearBoosterDesignAudit


def main() -> None:
    audit = Phase6DBearBoosterDesignAudit(ROOT)
    paths = audit.save_report(audit.build_report())
    print(f"generated markdown={paths.markdown}")
    print(f"generated json={paths.json}")


if __name__ == "__main__":
    main()
