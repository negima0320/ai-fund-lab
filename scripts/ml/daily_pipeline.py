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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_daily_pipeline(args.date)
    print(f"features_path={result['features_path']}")
    print(f"predictions_path={result['predictions_path']}")
    print(f"labels_paths={result['labels_paths']}")
    for warning in result["warnings"]:
        print(f"warning={warning}")


if __name__ == "__main__":
    main()
