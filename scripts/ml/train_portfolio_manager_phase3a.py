#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_dataset import CLEAN_FEATURE_COLUMNS
from ml.portfolio_manager_trainer import PortfolioManagerTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight Portfolio Manager AI Phase 3-A models.")
    parser.add_argument(
        "--dataset",
        default="data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_2023-01_to_2026-05.parquet",
    )
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-05-31")
    parser.add_argument("--model-dir", default="models/ml/portfolio_manager/current_v2_73_phase3a")
    parser.add_argument("--clean", action="store_true", help="Train Phase 3-B clean models.")
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    model_dir = "models/ml/portfolio_manager/current_v2_73_phase3b_clean" if args.clean and args.model_dir == "models/ml/portfolio_manager/current_v2_73_phase3a" else args.model_dir
    trainer = PortfolioManagerTrainer(
        model_root=ROOT / model_dir,
        report_root=ROOT / "reports" / "ml",
        feature_candidates=CLEAN_FEATURE_COLUMNS if args.clean else None,
        excluded_features=[] if args.clean else None,
        report_name="portfolio_manager_phase3b_clean_training_2023-01_to_2026-05" if args.clean else "portfolio_manager_phase3a_training_2023-01_to_2026-05",
        model_profile="portfolio_manager_v2_73_phase3b_clean" if args.clean else "portfolio_manager_v2_73_phase3a",
    )
    dataset = trainer.load_dataset(dataset_path)
    train, test = trainer.split_by_time(dataset, train_end=args.train_end, test_start=args.test_start, test_end=args.test_end)
    result = trainer.train_all(train, test)
    metadata = trainer.build_metadata(dataset_path, train, test, result["feature_columns"])
    model_dir = trainer.save_models(result, metadata)
    report = trainer.build_report(dataset_path, train, test, result, metadata)
    paths = trainer.save_report(report)

    print(f"model_dir={model_dir}")
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"train_rows={len(train)}")
    print(f"test_rows={len(test)}")
    print(f"feature_count={len(result['feature_columns'])}")
    for model_name, metrics in result["metrics"].items():
        visible = {
            key: metrics.get(key)
            for key in ["rmse", "mae", "correlation", "auc", "accuracy", "precision", "recall"]
            if key in metrics
        }
        print(f"{model_name}: {visible}")


if __name__ == "__main__":
    main()
