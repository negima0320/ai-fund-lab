#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.feature_builder import FeatureBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily ML features from cached J-Quants prices.")
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = FeatureBuilder()
    features = builder.build_daily_features(args.date)
    path = builder.save_daily_features(features, args.date)
    print(f"saved {len(features)} rows to {path}")


if __name__ == "__main__":
    main()
