#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.exit_avoid_loss_simulation import ExitAvoidLossSimulator, THRESHOLDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc avoid_loss_5d threshold simulation for Exit AI.")
    parser.add_argument("--dataset", default="data/ml/exit_datasets/exit_dataset_v2_66_2023-01_to_2026-05.parquet")
    parser.add_argument("--model-dir", default="models/ml/exit/current_v2_66")
    parser.add_argument("--trades", default="logs/backtests/rookie_dealer_02_v2_66_ml_ranked/2023-01-01_to_2026-05-31/trades.csv")
    parser.add_argument("--thresholds", nargs="*", type=float, default=THRESHOLDS)
    args = parser.parse_args()

    simulator = ExitAvoidLossSimulator(
        root=ROOT,
        dataset_path=ROOT / args.dataset,
        model_dir=ROOT / args.model_dir,
        trades_path=ROOT / args.trades,
    )
    result = simulator.build(thresholds=args.thresholds)
    paths = simulator.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"trades_csv={paths.trades_csv}")
    print(f"baseline_total_profit={result['baseline']['total_profit']}")
    for row in result["results"]:
        print(
            f"threshold={row['threshold']:.2f} total_profit={row['total_profit']:.2f} "
            f"delta={row['profit_delta']:.2f} changed={row['exit_changed_count']} "
            f"pf={row['profit_factor']} precision={row['precision']:.4f} recall={row['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
