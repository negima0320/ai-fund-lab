#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.config import ML_DATA_ROOT, ML_MODELS_ROOT, ML_PREDICTIONS_ROOT, ML_REPORTS_ROOT, MODEL_FILENAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit walk-forward model/prediction lineage.")
    parser.add_argument(
        "--walk-forward-json",
        default=str(ML_REPORTS_ROOT / "walk_forward_5y_enriched_v2_2023-01_to_2026-05.json"),
    )
    parser.add_argument("--prediction-root", default=str(ML_DATA_ROOT / "walk_forward_predictions"))
    parser.add_argument("--walk-forward-model-root", default=str(ML_MODELS_ROOT / "walk_forward"))
    parser.add_argument("--current-model-root", default=str(ML_MODELS_ROOT / "current"))
    parser.add_argument("--output", default=str(ML_REPORTS_ROOT / "walk_forward_model_audit.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        walk_forward_json=Path(args.walk_forward_json),
        prediction_root=Path(args.prediction_root),
        walk_forward_model_root=Path(args.walk_forward_model_root),
        current_model_root=Path(args.current_model_root),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_markdown(report), encoding="utf-8")
    print(f"saved markdown report to {output}")
    print(f"folds={len(report['fold_rows'])}")
    print(f"all_fold_model_dirs_exist={report['summary']['all_fold_model_dirs_exist']}")
    print(f"all_predictions_older_than_report={report['summary']['all_predictions_older_than_report']}")
    print(f"prediction_root_is_walk_forward={report['summary']['prediction_root_is_walk_forward']}")
    print(f"current_model_prediction_evidence={report['summary']['current_model_prediction_evidence']}")


def build_report(
    walk_forward_json: Path,
    prediction_root: Path,
    walk_forward_model_root: Path,
    current_model_root: Path,
) -> dict[str, Any]:
    result = json.loads(walk_forward_json.read_text(encoding="utf-8"))
    report_mtime = _mtime(walk_forward_json)
    current_model_mtime = _max_mtime(list(current_model_root.glob("*"))) if current_model_root.exists() else None
    fold_rows = []
    for fold in result.get("folds", []):
        month = fold["month"]
        model_id = f"walk_forward_{month.replace('-', '')}"
        fold_current_dir = walk_forward_model_root / "current" / month
        fold_archive_dir = walk_forward_model_root / "archive" / model_id
        model_files = [fold_current_dir / filename for filename in MODEL_FILENAMES.values()]
        model_files.append(fold_current_dir / "feature_columns.json")
        prediction_files = _prediction_files_for_fold(prediction_root, fold["test_start"], fold["test_end"])
        prediction_mtimes = [_mtime(path) for path in prediction_files]
        model_mtimes = [_mtime(path) for path in model_files if path.exists()]
        feature_count = _feature_count(fold_current_dir / "feature_columns.json")
        metrics_path = fold_current_dir / "metrics.json"
        archive_metrics_path = fold_archive_dir / "metrics.json"
        fold_rows.append(
            {
                "month": month,
                "model_id": model_id,
                "fold_model_root": str(fold_current_dir),
                "archive_model_root": str(fold_archive_dir),
                "fold_model_dir_exists": fold_current_dir.exists(),
                "archive_model_dir_exists": fold_archive_dir.exists(),
                "all_model_files_exist": all(path.exists() for path in model_files),
                "feature_count": feature_count,
                "train_start": fold.get("train_start"),
                "requested_train_end": fold.get("requested_train_end"),
                "effective_train_end": fold.get("effective_train_end"),
                "test_start": fold.get("test_start"),
                "test_end": fold.get("test_end"),
                "predicted_dates_in_report": fold.get("predicted_dates"),
                "prediction_file_count": len(prediction_files),
                "prediction_created_first": _fmt_dt(min(prediction_mtimes) if prediction_mtimes else None),
                "prediction_created_last": _fmt_dt(max(prediction_mtimes) if prediction_mtimes else None),
                "model_file_created_first": _fmt_dt(min(model_mtimes) if model_mtimes else None),
                "model_file_created_last": _fmt_dt(max(model_mtimes) if model_mtimes else None),
                "metrics_file": str(metrics_path) if metrics_path.exists() else "",
                "archive_metrics_file": str(archive_metrics_path) if archive_metrics_path.exists() else "",
                "prediction_files_after_walk_forward_report": _count_after(prediction_mtimes, report_mtime),
                "uses_fold_model_by_code_path": True,
            }
        )

    may_2026 = next((row for row in fold_rows if row["month"] == "2026-05"), None)
    all_prediction_mtimes = [
        _mtime(path)
        for row in result.get("folds", [])
        for path in _prediction_files_for_fold(prediction_root, row["test_start"], row["test_end"])
    ]
    summary = {
        "walk_forward_json": str(walk_forward_json),
        "walk_forward_report_mtime": _fmt_dt(report_mtime),
        "prediction_root": str(prediction_root),
        "prediction_root_is_walk_forward": prediction_root.name == "walk_forward_predictions",
        "walk_forward_model_root": str(walk_forward_model_root),
        "current_model_root": str(current_model_root),
        "daily_current_prediction_root": str(ML_PREDICTIONS_ROOT),
        "current_model_mtime": _fmt_dt(current_model_mtime),
        "fold_count": len(fold_rows),
        "all_fold_model_dirs_exist": all(row["fold_model_dir_exists"] for row in fold_rows),
        "all_archive_model_dirs_exist": all(row["archive_model_dir_exists"] for row in fold_rows),
        "all_model_files_exist": all(row["all_model_files_exist"] for row in fold_rows),
        "all_predictions_older_than_report": all(
            mtime is not None and report_mtime is not None and mtime <= report_mtime
            for mtime in all_prediction_mtimes
        ),
        "prediction_root_is_daily_current_root": prediction_root.resolve() == ML_PREDICTIONS_ROOT.resolve(),
        "fold_specific_model_used_by_code_path": True,
        "current_model_used_by_walk_forward_code_path": False,
        "current_model_prediction_evidence": (
            "no direct evidence: walk-forward prediction files do not store model_id metadata; "
            "code path, prediction root separation, and file mtimes indicate fold-specific model roots were used"
        ),
        "may_2026_model_id": may_2026["model_id"] if may_2026 else None,
        "may_2026_prediction_file_count": may_2026["prediction_file_count"] if may_2026 else None,
        "may_2026_prediction_created_first": may_2026["prediction_created_first"] if may_2026 else None,
        "may_2026_prediction_created_last": may_2026["prediction_created_last"] if may_2026 else None,
    }
    return {"summary": summary, "fold_rows": fold_rows}


def _prediction_files_for_fold(prediction_root: Path, start_date: str, end_date: str) -> list[Path]:
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    paths = []
    for path in prediction_root.glob("predictions_*.parquet"):
        date_text = path.stem.replace("predictions_", "")
        try:
            date = datetime.fromisoformat(date_text).date()
        except ValueError:
            continue
        if start <= date <= end:
            paths.append(path)
    return sorted(paths)


def _feature_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return len(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def _max_mtime(paths: list[Path]) -> datetime | None:
    mtimes = [_mtime(path) for path in paths]
    mtimes = [mtime for mtime in mtimes if mtime is not None]
    return max(mtimes) if mtimes else None


def _count_after(mtimes: list[datetime | None], reference: datetime | None) -> int:
    if reference is None:
        return 0
    return sum(1 for mtime in mtimes if mtime is not None and mtime > reference)


def _fmt_dt(value: datetime | None) -> str:
    return value.isoformat(sep=" ", timespec="seconds") if value is not None else ""


def format_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = report["fold_rows"]
    return "\n".join(
        [
            "# Walk-Forward Model Audit",
            "",
            "## Summary",
            "",
            f"- walk_forward_json: `{summary['walk_forward_json']}`",
            f"- walk_forward_report_mtime: `{summary['walk_forward_report_mtime']}`",
            f"- prediction_root: `{summary['prediction_root']}`",
            f"- daily_current_prediction_root: `{summary['daily_current_prediction_root']}`",
            f"- walk_forward_model_root: `{summary['walk_forward_model_root']}`",
            f"- current_model_root: `{summary['current_model_root']}`",
            f"- current_model_mtime: `{summary['current_model_mtime']}`",
            f"- fold_count: {summary['fold_count']}",
            f"- all_fold_model_dirs_exist: {summary['all_fold_model_dirs_exist']}",
            f"- all_archive_model_dirs_exist: {summary['all_archive_model_dirs_exist']}",
            f"- all_model_files_exist: {summary['all_model_files_exist']}",
            f"- prediction_root_is_walk_forward: {summary['prediction_root_is_walk_forward']}",
            f"- prediction_root_is_daily_current_root: {summary['prediction_root_is_daily_current_root']}",
            f"- all_predictions_older_than_report: {summary['all_predictions_older_than_report']}",
            f"- fold_specific_model_used_by_code_path: {summary['fold_specific_model_used_by_code_path']}",
            f"- current_model_used_by_walk_forward_code_path: {summary['current_model_used_by_walk_forward_code_path']}",
            f"- current_model_prediction_evidence: {summary['current_model_prediction_evidence']}",
            "",
            "## Key Checks",
            "",
            "- Walk-forward code path trains each fold with `ModelTrainer(current_root=models/ml/walk_forward/current/YYYY-MM)`.",
            "- Prediction code path calls `Predictor(model_root=trainer.current_root, prediction_root=data/ml/walk_forward_predictions)` for each fold.",
            "- Therefore walk-forward predictions are generated into `data/ml/walk_forward_predictions`, not `data/ml/predictions`.",
            "- Prediction parquet files do not currently embed `model_id`; this audit verifies lineage from code path, fold directories, and file modification times.",
            f"- 2026-05 model_id: `{summary['may_2026_model_id']}`",
            f"- 2026-05 prediction files: {summary['may_2026_prediction_file_count']}",
            f"- 2026-05 prediction created first/last: `{summary['may_2026_prediction_created_first']}` / `{summary['may_2026_prediction_created_last']}`",
            "",
            "## Monthly Fold Lineage",
            "",
            _table(
                rows,
                [
                    "month",
                    "model_id",
                    "effective_train_end",
                    "test_start",
                    "test_end",
                    "feature_count",
                    "prediction_file_count",
                    "prediction_created_first",
                    "prediction_created_last",
                    "fold_model_dir_exists",
                    "all_model_files_exist",
                    "prediction_files_after_walk_forward_report",
                ],
            ),
            "",
        ]
    )


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(_fmt_cell(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _fmt_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
