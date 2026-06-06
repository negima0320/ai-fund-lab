#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_dataset import PROFILE
from ml.portfolio_manager_dataset import CleanPortfolioManagerDatasetBuilder
from ml.portfolio_manager_dataset import PortfolioManagerDatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Portfolio Manager AI training dataset.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--clean", action="store_true", help="Build Phase 3-B clean dataset with no backtest-state features.")
    args = parser.parse_args()

    builder_class = CleanPortfolioManagerDatasetBuilder if args.clean else PortfolioManagerDatasetBuilder
    builder = builder_class(
        root=ROOT,
        profile=args.profile,
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
    )
    dataset = builder.build_dataset()
    paths = builder.save(dataset)
    summary = builder.summary(dataset)

    print(f"dataset_path={paths.dataset}")
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"rows={summary['rows']}")
    print(f"unique_dates={summary['unique_dates']}")
    print(f"unique_codes={summary['unique_codes']}")
    print(f"feature_count={summary['feature_count']}")
    print(f"prediction_join_rate={summary['prediction_join_rate']}")
    print(f"positive_trade_rate={summary['label_distribution']['positive_trade_rate']}")
    print(f"avoid_target_rate={summary['label_distribution']['avoid_target_rate']}")
    print("quality_assessment:")
    for item in summary["quality_assessment"]:
        print(f"- {item}")


if __name__ == "__main__":
    main()
