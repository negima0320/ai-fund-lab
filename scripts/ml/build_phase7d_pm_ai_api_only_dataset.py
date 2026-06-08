#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase7d_pm_ai_api_only_dataset_builder import BuildOptions, Phase7DPMAIAPIOnlyDatasetBuilder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or audit the PM AI API-only dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Generate report only. This is the safe default.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Write a sample dataset with the first N rows.")
    parser.add_argument("--write-full", action="store_true", help="Write the full dataset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_full and args.sample_rows:
        raise SystemExit("--write-full and --sample-rows cannot be used together")
    options = BuildOptions(
        dry_run=args.dry_run or (not args.sample_rows and not args.write_full),
        sample_rows=args.sample_rows,
        write_full=args.write_full,
    )
    builder = Phase7DPMAIAPIOnlyDatasetBuilder(ROOT)
    result = builder.build(options)
    paths = builder.save_report(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"dataset={paths.dataset}")
    print(f"dataset_written={result['metadata']['dataset_written']}")
    print(f"final_rows={result['row_counts']['final_rows']}")
    print(f"final_feature_count={result['row_counts']['final_feature_count']}")
    print(f"leakage_risk={result['leakage_audit']['leakage_risk']}")
    print(f"ready_for_phase7e={result['leakage_audit']['ready_for_phase7e']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

