#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.exit_model_trainer import ExitModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Exit AI models from held-day exit dataset.")
    parser.add_argument("--dataset", default="data/ml/exit_datasets/exit_dataset_v2_66_2023-01_to_2026-05.parquet")
    parser.add_argument("--model-dir", default="models/ml/exit/current_v2_66")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--valid-start", default="2026-01-01")
    parser.add_argument("--valid-end", default="2026-05-31")
    args = parser.parse_args()

    trainer = ExitModelTrainer(model_root=ROOT / args.model_dir)
    df = trainer.load_dataset(ROOT / args.dataset)
    train_df, valid_df = trainer.split_by_time(df, train_end=args.train_end, valid_start=args.valid_start, valid_end=args.valid_end)
    selected = trainer.train_all(train_df, valid_df, include_remaining_days=False)
    comparison = trainer.compare_feature_sets(train_df, valid_df)
    metadata = trainer.build_metadata(train_df, valid_df, selected["feature_columns"])
    model_dir = trainer.save_models(selected, metadata)
    report = {
        "dataset_path": str((ROOT / args.dataset).resolve()),
        "model_dir": str(model_dir.resolve()),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "selected_feature_set": "without_remaining_days",
        "feature_count": len(selected["feature_columns"]),
        "feature_columns": selected["feature_columns"],
        "metrics": selected["metrics"],
        "decile_analysis": {name: payload.get("decile_analysis", []) for name, payload in selected["metrics"].items()},
        "feature_set_comparison": comparison,
        "metadata": metadata,
    }
    paths = trainer.save_report(report)
    print(f"model_dir={paths.model_dir}")
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"train_rows={len(train_df)}")
    print(f"valid_rows={len(valid_df)}")
    print(f"feature_count={len(selected['feature_columns'])}")
    for model_name, metrics in selected["metrics"].items():
        visible = {key: metrics.get(key) for key in ["rmse", "mae", "correlation", "auc", "accuracy", "precision", "recall"] if key in metrics}
        print(f"{model_name}={visible}")


if __name__ == "__main__":
    main()
