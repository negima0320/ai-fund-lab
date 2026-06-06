#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.predictor import Predictor
from ml.portfolio_manager_dataset import PROFILE


def _candidate_dates(profile: str, period: str, start: str, end: str) -> list[str]:
    path = ROOT / "logs" / "backtests" / profile / period / "purchase_audit.csv"
    if not path.exists():
        raise FileNotFoundError(f"purchase_audit.csv not found: {path}")
    df = pd.read_csv(path, usecols=["signal_date"])
    dates = pd.to_datetime(df["signal_date"], errors="coerce").dropna()
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    return sorted(dates.dt.strftime("%Y-%m-%d").unique())


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore walk-forward prediction parquet from fold model archive.")
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--period", default="2023-01-01_to_2026-05-31")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--archive-root", default="models/ml/walk_forward/archive")
    parser.add_argument("--feature-root", default="data/ml/features")
    parser.add_argument("--prediction-root", default="data/ml/walk_forward_predictions")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dates = _candidate_dates(args.profile, args.period, args.start, args.end)
    archive_root = ROOT / args.archive_root
    feature_root = ROOT / args.feature_root
    prediction_root = ROOT / args.prediction_root
    missing_models = []
    missing_features = []
    existing = 0
    planned = 0
    written = 0

    for date_text in dates:
        month = date_text[:7].replace("-", "")
        model_root = archive_root / f"walk_forward_{month}"
        feature_path = feature_root / f"features_{date_text}.parquet"
        prediction_path = prediction_root / f"predictions_{date_text}.parquet"
        if not model_root.exists():
            missing_models.append(str(model_root))
            continue
        if not feature_path.exists():
            missing_features.append(str(feature_path))
            continue
        if prediction_path.exists() and not args.overwrite:
            existing += 1
            continue
        planned += 1
        if args.dry_run:
            continue
        predictor = Predictor(feature_root=feature_root, model_root=model_root, prediction_root=prediction_root)
        predictions = predictor.predict_daily(date_text)
        predictor.save_predictions(predictions, date_text)
        written += 1

    print(f"candidate_dates={len(dates)}")
    print(f"planned={planned}")
    print(f"existing_skipped={existing}")
    print(f"written={written}")
    print(f"missing_models={len(set(missing_models))}")
    print(f"missing_features={len(set(missing_features))}")
    if missing_models:
        print("missing_model_examples=" + ",".join(sorted(set(missing_models))[:5]))
    if missing_features:
        print("missing_feature_examples=" + ",".join(sorted(set(missing_features))[:5]))
    if args.dry_run:
        print("dry_run=true")


if __name__ == "__main__":
    main()
