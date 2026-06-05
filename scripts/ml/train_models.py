#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.config import (
    EARNINGS_FEATURE_COLUMNS,
    FINANCIAL_FEATURE_COLUMNS,
    LABEL_HORIZONS,
    LISTED_INFO_FEATURE_COLUMNS,
    ML_MODEL_ARCHIVE_ROOT,
    ML_MODEL_CURRENT_ROOT,
    TECHNICAL_FEATURE_COLUMNS,
    TOPIX_FEATURE_COLUMNS,
)
from ml.model_trainer import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train initial LightGBM ML models.")
    parser.add_argument("--train", required=True, help="Path to train parquet.")
    parser.add_argument("--valid", required=True, help="Path to validation parquet.")
    parser.add_argument("--output-dir", default=None, help="Directory to write the current model files.")
    parser.add_argument("--model-profile", default="default")
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = ModelTrainer(
        archive_root=ML_MODEL_ARCHIVE_ROOT,
        current_root=Path(args.output_dir) if args.output_dir else ML_MODEL_CURRENT_ROOT,
    )
    train_df = trainer.load_dataset(args.train)
    valid_df = trainer.load_dataset(args.valid)
    result = trainer.train_all(train_df, valid_df)
    metadata = {
        "model_profile": args.model_profile,
        "train_start": args.train_start or _date_min(train_df),
        "train_end": args.train_end or _date_max(train_df),
        "feature_count": len(result["feature_columns"]),
        "feature_groups": _feature_groups(result["feature_columns"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label_horizons": sorted({*LABEL_HORIZONS, 20}),
        "leakage_guard": "effective_train_end uses only confirmed labels",
    }
    path = trainer.save_models(result["models"], result["metrics"], metadata=metadata)
    print(f"saved {len(result['models'])} models to {path}")
    print(f"current_model_dir={trainer.current_root}")
    print(f"feature_count={len(result['feature_columns'])}")


def _date_min(df) -> str | None:
    if "date" not in df.columns or df.empty:
        return None
    return str(df["date"].min().date())


def _date_max(df) -> str | None:
    if "date" not in df.columns or df.empty:
        return None
    return str(df["date"].max().date())


def _feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    features = set(feature_columns)
    grouped = {
        "technical": [column for column in TECHNICAL_FEATURE_COLUMNS if column in features],
        "financial": [column for column in FINANCIAL_FEATURE_COLUMNS if column in features],
        "earnings": [column for column in EARNINGS_FEATURE_COLUMNS if column in features],
        "listed_info": [column for column in LISTED_INFO_FEATURE_COLUMNS if column in features],
        "topix": [column for column in TOPIX_FEATURE_COLUMNS if column in features],
    }
    grouped_features = {column for columns in grouped.values() for column in columns}
    grouped["other"] = [column for column in feature_columns if column not in grouped_features]
    return grouped


if __name__ == "__main__":
    main()
