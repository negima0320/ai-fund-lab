#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.pipeline import DailyMLPipeline
from ml.label_generator import LabelGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight ML smoke test for one target date.")
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format.")
    return parser.parse_args()


def run_smoke(
    target_date: str,
    pipeline: DailyMLPipeline | None = None,
    label_generator: LabelGenerator | None = None,
    read_parquet: Callable[[Path], pd.DataFrame] = pd.read_parquet,
) -> dict[str, Any]:
    runner = pipeline or DailyMLPipeline()
    checker = label_generator or LabelGenerator()
    warnings: list[str] = []
    try:
        result = runner.run_daily_pipeline(target_date)
    except Exception as exc:
        return {
            "target_date": target_date,
            "features_path": None,
            "features_summary": {"rows": None, "columns": None},
            "updated_labels_paths": [],
            "updated_labels_summaries": [],
            "target_date_label_check": {"rows": None, "columns": None},
            "predictions_path": None,
            "predictions_summary": {"rows": None, "columns": None},
            "warnings": [f"pipeline failed: {exc}"],
        }

    warnings.extend(result.get("warnings", []))
    features_path = _optional_path(result.get("features_path"))
    predictions_path = _optional_path(result.get("predictions_path"))
    updated_labels_paths = [_optional_path(path) for path in result.get("labels_paths", [])]

    features_summary = _summarize_parquet(features_path, read_parquet, warnings, "features")
    predictions_summary = _summarize_parquet(predictions_path, read_parquet, warnings, "predictions")
    updated_labels_summaries = [
        {"path": path, **_summarize_parquet(path, read_parquet, warnings, "labels")}
        for path in updated_labels_paths
    ]
    target_date_label_check = _check_target_date_labels(checker, target_date, warnings)

    return {
        "target_date": target_date,
        "features_path": features_path,
        "features_summary": features_summary,
        "updated_labels_paths": updated_labels_paths,
        "updated_labels_summaries": updated_labels_summaries,
        "target_date_label_check": target_date_label_check,
        "predictions_path": predictions_path,
        "predictions_summary": predictions_summary,
        "warnings": warnings,
    }


def format_smoke_result(result: dict[str, Any]) -> str:
    lines = [
        f"target_date={result['target_date']}",
        f"features_path={result['features_path']}",
        f"features rows={result['features_summary']['rows']} columns={result['features_summary']['columns']}",
        f"updated_labels_paths={result['updated_labels_paths']}",
    ]
    if result["updated_labels_summaries"]:
        for summary in result["updated_labels_summaries"]:
            lines.append(f"updated_labels path={summary['path']} rows={summary['rows']} columns={summary['columns']}")
    else:
        lines.append("updated_labels rows=0 columns=0")
    lines.append(
        f"target_date_label_check rows={result['target_date_label_check']['rows']} "
        f"columns={result['target_date_label_check']['columns']}"
    )

    if result["predictions_path"] is None:
        lines.append("predictions_path=skipped")
    else:
        lines.append(f"predictions_path={result['predictions_path']}")
        lines.append(
            f"predictions rows={result['predictions_summary']['rows']} "
            f"columns={result['predictions_summary']['columns']}"
        )

    for warning in result["warnings"]:
        lines.append(f"warning={warning}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    print(format_smoke_result(run_smoke(args.date)))
    return 0


def _summarize_parquet(
    path: Path | None,
    read_parquet: Callable[[Path], pd.DataFrame],
    warnings: list[str],
    label: str,
) -> dict[str, int | None]:
    if path is None:
        return {"rows": None, "columns": None}
    if not path.exists():
        warnings.append(f"{label} file is missing: {path}")
        return {"rows": None, "columns": None}
    try:
        frame = read_parquet(path)
    except Exception as exc:
        warnings.append(f"failed to read {label} parquet {path}: {exc}")
        return {"rows": None, "columns": None}
    return {"rows": int(len(frame)), "columns": int(len(frame.columns))}


def _check_target_date_labels(
    label_generator: LabelGenerator,
    target_date: str,
    warnings: list[str],
) -> dict[str, int | None]:
    try:
        labels = label_generator.generate_labels(target_date)
    except Exception as exc:
        warnings.append(f"target_date_label_check failed: {exc}")
        return {"rows": None, "columns": None}
    if labels.empty:
        return {"rows": 0, "columns": 0}
    return {"rows": int(len(labels)), "columns": int(len(labels.columns))}


def _optional_path(value: Any) -> Path | None:
    return Path(value) if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
