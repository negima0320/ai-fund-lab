#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.dataset_builder import DatasetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML dataset from saved feature and label parquet files.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--train-end", required=True, help="Last train date in YYYY-MM-DD format.")
    parser.add_argument("--valid-end", required=True, help="Last validation date in YYYY-MM-DD format.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = DatasetBuilder()
    dataset = builder.build_dataset(args.start, args.end)
    train, valid, test = builder.split_by_time(dataset, args.train_end, args.valid_end)
    dataset_path = builder.save_dataset(dataset, "ml_dataset")
    train_path = builder.save_dataset(train, "train")
    valid_path = builder.save_dataset(valid, "valid")
    test_path = builder.save_dataset(test, "test")
    print(f"saved dataset rows={len(dataset)} to {dataset_path}")
    print(f"saved train rows={len(train)} to {train_path}")
    print(f"saved valid rows={len(valid)} to {valid_path}")
    print(f"saved test rows={len(test)} to {test_path}")


if __name__ == "__main__":
    main()
