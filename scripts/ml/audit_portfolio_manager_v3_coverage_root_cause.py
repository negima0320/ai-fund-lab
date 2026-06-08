#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_v3_coverage_root_cause_audit import PMAIV3CoverageRootCauseAudit


def main() -> None:
    audit = PMAIV3CoverageRootCauseAudit(ROOT)
    paths = audit.save_report(audit.build_report())
    print(f"markdown={paths.markdown.relative_to(ROOT)}")
    print(f"json={paths.json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
