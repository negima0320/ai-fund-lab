#!/usr/bin/env python3
"""Generate the Phase 9-G PM disabled equal-weight audit report."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_pm_disabled_equal_weight_audit import PMDisabledEqualWeightAudit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 9-G PM disabled equal-weight audit report")
    parser.add_argument("--root", default=str(ROOT), help="Repository root")
    args = parser.parse_args()
    audit = PMDisabledEqualWeightAudit(Path(args.root))
    paths = audit.save_report(audit.build_report())
    print(paths.markdown)
    print(paths.json)


if __name__ == "__main__":
    main()
