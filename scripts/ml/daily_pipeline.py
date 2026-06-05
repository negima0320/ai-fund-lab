#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.pipeline import run_daily_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lightweight daily ML pipeline.")
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--top-n", type=int, default=10, help="Daily AI candidate count.")
    parser.add_argument("--min-turnover-value", type=float, default=50_000_000)
    parser.add_argument("--max-bad-entry-probability", type=float, default=None)
    parser.add_argument("--no-export-candidates", action="store_true", help="Skip daily AI candidate CSV/Markdown export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_daily_pipeline(
        args.date,
        export_candidates=not args.no_export_candidates,
        candidate_top_n=args.top_n,
        min_turnover_value=args.min_turnover_value,
        max_bad_entry_probability=args.max_bad_entry_probability,
    )
    print(f"features_path={result['features_path']}")
    print(f"predictions_path={result['predictions_path']}")
    print(f"candidate_csv_path={result['candidate_csv_path']}")
    print(f"candidate_md_path={result['candidate_md_path']}")
    print(f"labels_paths={result['labels_paths']}")
    for warning in result["warnings"]:
        print(f"warning={warning}")


if __name__ == "__main__":
    main()
