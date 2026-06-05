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

from ml.config import FEATURE_HISTORY_DAYS, ML_FEATURES_ROOT
from ml.feature_builder import FeatureBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add enriched J-Quants features to existing daily feature parquet files.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builder = FeatureBuilder()
    results = []
    for date_text in date_texts(args.start, args.end):
        result = enrich_one(builder, date_text)
        results.append(result)
        print(
            f"date={date_text} status={result['status']} "
            f"rows={result['rows']} columns={result['columns']}"
        )
        if result.get("warning"):
            print(f"warning={date_text}: {result['warning']}")
    processed = sum(1 for row in results if row["status"] == "written")
    skipped = sum(1 for row in results if row["status"] == "skipped")
    print(f"summary dates={len(results)} written={processed} skipped={skipped}")
    return 0


def enrich_one(builder: FeatureBuilder, date_text: str) -> dict[str, Any]:
    path = ML_FEATURES_ROOT / f"features_{date_text}.parquet"
    if not path.exists():
        return {"date": date_text, "status": "skipped", "rows": 0, "columns": 0, "warning": "feature parquet missing"}
    features = pd.read_parquet(path)
    if features.empty:
        return {"date": date_text, "status": "skipped", "rows": 0, "columns": len(features.columns), "warning": "feature parquet empty"}
    target = pd.Timestamp(date_text)
    history_start = (target - pd.Timedelta(days=FEATURE_HISTORY_DAYS)).strftime("%Y-%m-%d")
    enriched = features.copy()
    enriched["date"] = pd.to_datetime(enriched["date"], errors="coerce")
    enriched["code"] = enriched["code"].astype("string")
    enriched = builder._add_financial_features(enriched, target)
    enriched = builder._add_earnings_features(enriched, target)
    enriched = builder._add_listed_info_features(enriched, target)
    enriched = builder._add_topix_features(enriched, history_start, target)
    enriched = builder._order_feature_columns(enriched).reset_index(drop=True)
    enriched.to_parquet(path, index=False)
    return {"date": date_text, "status": "written", "rows": int(len(enriched)), "columns": int(len(enriched.columns))}


def date_texts(start_date: str, end_date: str) -> list[str]:
    return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]


if __name__ == "__main__":
    raise SystemExit(main())
