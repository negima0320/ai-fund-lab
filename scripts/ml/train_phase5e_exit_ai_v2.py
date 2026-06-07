#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase5e_exit_ai_v2_trainer import Phase5EExitAIV2TrainerPrototype, TrainOptions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exit AI v2 trainer prototype.")
    parser.add_argument("--dry-run", action="store_true", help="Run leakage/data checks only. This is the safe default.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Train on a split-preserving sample of N rows.")
    parser.add_argument("--train-full", action="store_true", help="Train on the full Phase 5-C dataset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.train_full and args.sample_rows:
        raise SystemExit("--train-full and --sample-rows cannot be used together")
    options = TrainOptions(
        dry_run=args.dry_run or (not args.sample_rows and not args.train_full),
        sample_rows=args.sample_rows,
        train_full=args.train_full,
    )
    trainer = Phase5EExitAIV2TrainerPrototype(ROOT)
    report = trainer.run(options)
    paths = trainer.save_report(report)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"model_dir={paths.model_dir}")
    print(f"trained={report['training_status'].get('trained')}")
    print(f"model_saved={report['training_status'].get('model_saved')}")
    print(f"rows_used={report['dataset']['rows_used']}")
    print(f"feature_count={report['dataset']['feature_count']}")
    print(f"leakage_risk={report['leakage_check']['leakage_risk']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
