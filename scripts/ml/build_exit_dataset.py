#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.exit_dataset import ExitDatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an Exit AI held-day dataset from existing backtest trades.")
    parser.add_argument("--profile", default="rookie_dealer_02_v2_66_ml_ranked")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    args = parser.parse_args()

    builder = ExitDatasetBuilder(root=ROOT, profile=args.profile, start_date=args.start, end_date=args.end)
    df = builder.build_dataset()
    dataset_path = builder.save_dataset(df)
    summary = builder.summarize(df, dataset_path)
    paths = builder.save_summary(summary)

    print(f"dataset={paths.dataset}")
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"rows={summary['rows']}")
    print(f"unique_trades={summary['unique_trades']}")
    print(f"prediction_join_success_rate={summary['prediction_join_success_rate']}")
    for key, value in summary["label_distribution"].items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
