#!/usr/bin/env python3
"""Generate the Phase 11-A Valuation Engine dataset audit report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ml.phase11a_valuation_dataset_audit import Phase11AValuationDatasetAudit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 11-A valuation dataset audit report")
    parser.add_argument("--root", default=str(ROOT), help="Repository root")
    parser.add_argument("--start", default="2023-01-01", help="Start date")
    parser.add_argument("--end", default="2026-05-31", help="End date")
    parser.add_argument("--no-save-dataset", action="store_true", help="Do not save the intermediate valuation dataset")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Ignore the cached Phase 11-A dataset and rebuild it")
    args = parser.parse_args()

    audit = Phase11AValuationDatasetAudit(
        Path(args.root),
        start_date=args.start,
        end_date=args.end,
        save_dataset=not args.no_save_dataset,
        use_cached_dataset=not args.rebuild_dataset,
    )
    paths = audit.run()
    print(paths.markdown)
    print(paths.json)
    if paths.dataset is not None:
        print(paths.dataset)


if __name__ == "__main__":
    main()
