#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.label_generator import LabelGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ML labels from cached J-Quants prices.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Target date in YYYY-MM-DD format.")
    group.add_argument("--as-of", help="Generate labels available as of this date.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = LabelGenerator()
    if args.date:
        labels = generator.generate_labels(args.date)
        path = generator.save_labels(labels, args.date)
        print(f"saved {len(labels)} rows to {path}")
        return

    paths = generator.update_available_labels(args.as_of)
    print(f"saved {len(paths)} label file(s)")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
