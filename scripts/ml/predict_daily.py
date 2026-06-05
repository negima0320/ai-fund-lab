#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.predictor import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily ML predictions from current models.")
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--model-dir", default=None, help="Model directory to use. Defaults to models/ml/current.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = Predictor(model_root=args.model_dir) if args.model_dir else Predictor()
    predictions = predictor.predict_daily(args.date)
    path = predictor.save_predictions(predictions, args.date)
    print(f"saved {len(predictions)} rows to {path}")


if __name__ == "__main__":
    main()
