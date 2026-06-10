#!/usr/bin/env python3
"""Train the Phase 11-B Valuation Engine prototype."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ml.phase11b_valuation_engine_prototype import Phase11BOptions, Phase11BValuationEnginePrototype


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 11-B valuation engine prototype")
    parser.add_argument("--root", default=str(ROOT), help="Repository root")
    parser.add_argument("--max-train-rows", type=int, default=250_000, help="Deterministic train sample size")
    parser.add_argument("--max-test-rows", type=int, default=None, help="Optional deterministic test sample size")
    parser.add_argument("--max-iter", type=int, default=80, help="HistGradientBoosting max_iter")
    parser.add_argument("--learning-rate", type=float, default=0.06, help="HistGradientBoosting learning rate")
    parser.add_argument("--no-save-model", action="store_true", help="Do not save candidate model artifacts")
    args = parser.parse_args()

    options = Phase11BOptions(
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        save_model=not args.no_save_model,
    )
    trainer = Phase11BValuationEnginePrototype(Path(args.root), options=options)
    paths = trainer.run()
    print(paths.markdown)
    print(paths.json)
    if paths.model_dir is not None:
        print(paths.model_dir)


if __name__ == "__main__":
    main()
