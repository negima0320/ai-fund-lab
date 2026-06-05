#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.feature_builder import FeatureBuilder
from ml.label_generator import LabelGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML features and labels for a local cached date range.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--features-only", action="store_true", help="Only build features.")
    parser.add_argument("--labels-only", action="store_true", help="Only build labels.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.features_only and args.labels_only:
        print("error: --features-only and --labels-only cannot be used together", file=sys.stderr)
        return 2

    builder = FeatureBuilder()
    labeler = LabelGenerator()
    rows = []
    for date_text in _date_texts(args.start, args.end):
        row: dict[str, Any] = {"date": date_text, "features_rows": 0, "labels_rows": 0, "warnings": []}
        if not args.labels_only:
            try:
                features = builder.build_daily_features(date_text)
                builder.save_daily_features(features, date_text)
                row["features_rows"] = int(len(features))
            except Exception as exc:
                row["warnings"].append(f"features failed: {exc}")
        if not args.features_only:
            try:
                labels = labeler.generate_labels(date_text)
                if not labels.empty:
                    labeler.save_labels(labels, date_text)
                row["labels_rows"] = int(len(labels))
            except Exception as exc:
                row["warnings"].append(f"labels failed: {exc}")
        rows.append(row)
        status = "ok" if not row["warnings"] else "warning"
        print(
            f"date={date_text} status={status} "
            f"features_rows={row['features_rows']} labels_rows={row['labels_rows']}"
        )
        for warning in row["warnings"]:
            print(f"warning={date_text}: {warning}")

    features_total = sum(int(row["features_rows"]) for row in rows)
    labels_total = sum(int(row["labels_rows"]) for row in rows)
    warning_count = sum(len(row["warnings"]) for row in rows)
    print(
        f"summary dates={len(rows)} features_rows={features_total} "
        f"labels_rows={labels_total} warnings={warning_count}"
    )
    return 0 if warning_count == 0 else 1


def _date_texts(start_date: str, end_date: str) -> list[str]:
    return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]


if __name__ == "__main__":
    raise SystemExit(main())
