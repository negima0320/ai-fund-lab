#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase6c_bear_alpha_booster_audit import Phase6CBearAlphaBoosterAudit


def main() -> None:
    audit = Phase6CBearAlphaBoosterAudit(ROOT)
    paths = audit.save_report(audit.build_report())
    print(f"generated markdown={paths.markdown}")
    print(f"generated json={paths.json}")


if __name__ == "__main__":
    main()
