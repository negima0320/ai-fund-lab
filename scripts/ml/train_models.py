#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.model_trainer import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train initial LightGBM ML models.")
    parser.add_argument("--train", required=True, help="Path to train parquet.")
    parser.add_argument("--valid", required=True, help="Path to validation parquet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = ModelTrainer()
    train_df = trainer.load_dataset(args.train)
    valid_df = trainer.load_dataset(args.valid)
    result = trainer.train_all(train_df, valid_df)
    path = trainer.save_models(result["models"], result["metrics"])
    print(f"saved {len(result['models'])} models to {path}")


if __name__ == "__main__":
    main()
