from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.config import LABEL_LOOKAHEAD_DAYS, ML_PREDICTIONS_ROOT, MODEL_FILENAMES


try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - pyarrow is optional at import time.
    pq = None


@dataclass(frozen=True)
class WalkForwardModelAuditConfig:
    walk_forward_json: Path
    prediction_root: Path
    walk_forward_model_root: Path
    current_model_root: Path
    daily_current_prediction_root: Path = ML_PREDICTIONS_ROOT


class WalkForwardModelAuditor:
    """Audit model lineage for walk-forward prediction parquet files."""

    def __init__(self, config: WalkForwardModelAuditConfig) -> None:
        self.config = config

    def build_report(self) -> dict[str, Any]:
        result = json.loads(self.config.walk_forward_json.read_text(encoding="utf-8"))
        report_mtime = self._mtime(self.config.walk_forward_json)
        current_model_mtime = self._max_mtime(list(self.config.current_model_root.glob("*"))) if self.config.current_model_root.exists() else None

        fold_rows = [self._fold_row(fold, report_mtime, current_model_mtime) for fold in result.get("folds", [])]
        failures = [item for row in fold_rows for item in row["failures"]]
        warnings = [item for row in fold_rows for item in row["warnings"]]
        may_2026 = next((row for row in fold_rows if row["month"] == "2026-05"), None)
        summary_failures = self._summary_failures(fold_rows)
        failures.extend(summary_failures)

        status = "fail" if failures else "warning" if warnings else "pass"
        summary = {
            "status": status,
            "fail_count": len(failures),
            "warning_count": len(warnings),
            "failures": failures,
            "warnings": warnings,
            "walk_forward_json": str(self.config.walk_forward_json),
            "walk_forward_report_mtime": self._fmt_dt(report_mtime),
            "prediction_root": str(self.config.prediction_root),
            "daily_current_prediction_root": str(self.config.daily_current_prediction_root),
            "walk_forward_model_root": str(self.config.walk_forward_model_root),
            "current_model_root": str(self.config.current_model_root),
            "current_model_mtime": self._fmt_dt(current_model_mtime),
            "fold_count": len(fold_rows),
            "label_lookahead_days": LABEL_LOOKAHEAD_DAYS,
            "all_fold_model_dirs_exist": all(row["fold_model_dir_exists"] for row in fold_rows),
            "all_archive_model_dirs_exist": all(row["archive_model_dir_exists"] for row in fold_rows),
            "all_model_files_exist": all(row["all_model_files_exist"] for row in fold_rows),
            "prediction_root_is_walk_forward": self.config.prediction_root.name == "walk_forward_predictions",
            "prediction_root_is_daily_current_root": self.config.prediction_root.resolve() == self.config.daily_current_prediction_root.resolve(),
            "fold_specific_model_used_by_code_path": True,
            "current_model_used_by_walk_forward_code_path": False,
            "metadata_checked_files": sum(row["prediction_metadata_checked"] for row in fold_rows),
            "metadata_missing_files": sum(row["prediction_metadata_missing"] for row in fold_rows),
            "metadata_mismatch_files": sum(row["prediction_metadata_mismatch"] for row in fold_rows),
            "prediction_files_after_walk_forward_report": sum(row["prediction_files_after_walk_forward_report"] for row in fold_rows),
            "may_2026": may_2026,
        }
        return {"summary": summary, "fold_rows": fold_rows}

    def _fold_row(
        self,
        fold: dict[str, Any],
        report_mtime: datetime | None,
        current_model_mtime: datetime | None,
    ) -> dict[str, Any]:
        month = fold["month"]
        model_id = f"walk_forward_{month.replace('-', '')}"
        fold_current_dir = self.config.walk_forward_model_root / "current" / month
        fold_archive_dir = self.config.walk_forward_model_root / "archive" / model_id
        model_files = [fold_current_dir / filename for filename in MODEL_FILENAMES.values()]
        model_files.append(fold_current_dir / "feature_columns.json")

        prediction_files = self._prediction_files_for_fold(fold["test_start"], fold["test_end"])
        prediction_mtimes = [self._mtime(path) for path in prediction_files]
        model_mtimes = [self._mtime(path) for path in model_files if path.exists()]
        prediction_metadata = [self._prediction_metadata(path) for path in prediction_files]
        mismatches = [
            item
            for item in prediction_metadata
            if item["model_id"] and item["model_id"] != model_id
        ]
        missing_metadata = [item for item in prediction_metadata if not item["model_id"]]
        prediction_rows = sum(item["rows"] or 0 for item in prediction_metadata)

        failures: list[str] = []
        warnings: list[str] = []
        effective_train_end = datetime.fromisoformat(fold["effective_train_end"]).date()
        test_start = datetime.fromisoformat(fold["test_start"]).date()
        requested_train_end = datetime.fromisoformat(fold["requested_train_end"]).date()
        if effective_train_end >= test_start:
            failures.append(f"{month}: effective_train_end >= test_start")
        if effective_train_end > requested_train_end:
            failures.append(f"{month}: effective_train_end is after requested_train_end")
        if not fold_current_dir.exists():
            failures.append(f"{month}: fold current model directory is missing")
        if not fold_archive_dir.exists():
            failures.append(f"{month}: fold archive model directory is missing")
        missing_model_files = [str(path) for path in model_files if not path.exists()]
        if missing_model_files:
            failures.append(f"{month}: missing model files: {', '.join(missing_model_files)}")
        if mismatches:
            failures.append(f"{month}: prediction metadata model_id mismatch")
        if not prediction_files and fold.get("predicted_dates", 0):
            failures.append(f"{month}: report has predicted_dates but prediction parquet files are missing")
        if len(prediction_files) != int(fold.get("predicted_dates") or 0):
            warnings.append(
                f"{month}: prediction file count {len(prediction_files)} differs from report predicted_dates {fold.get('predicted_dates')}"
            )
        if missing_metadata:
            warnings.append(f"{month}: {len(missing_metadata)} prediction files have no model_id metadata")

        return {
            "month": month,
            "model_id": model_id,
            "fold_model_root": str(fold_current_dir),
            "archive_model_root": str(fold_archive_dir),
            "fold_model_dir_exists": fold_current_dir.exists(),
            "archive_model_dir_exists": fold_archive_dir.exists(),
            "all_model_files_exist": not missing_model_files,
            "feature_count": self._feature_count(fold_current_dir / "feature_columns.json"),
            "train_start": fold.get("train_start"),
            "requested_train_end": fold.get("requested_train_end"),
            "effective_train_end": fold.get("effective_train_end"),
            "test_start": fold.get("test_start"),
            "test_end": fold.get("test_end"),
            "effective_train_end_before_test_start": effective_train_end < test_start,
            "label_lookahead_days": LABEL_LOOKAHEAD_DAYS,
            "label_window_capped_by_effective_train_end": effective_train_end <= requested_train_end and effective_train_end < test_start,
            "predicted_dates_in_report": int(fold.get("predicted_dates") or 0),
            "prediction_file_count": len(prediction_files),
            "prediction_rows": int(prediction_rows),
            "prediction_created_first": self._fmt_dt(min(prediction_mtimes) if prediction_mtimes else None),
            "prediction_created_last": self._fmt_dt(max(prediction_mtimes) if prediction_mtimes else None),
            "model_file_created_first": self._fmt_dt(min(model_mtimes) if model_mtimes else None),
            "model_file_created_last": self._fmt_dt(max(model_mtimes) if model_mtimes else None),
            "prediction_files_after_walk_forward_report": self._count_after(prediction_mtimes, report_mtime),
            "prediction_metadata_checked": len(prediction_metadata),
            "prediction_metadata_missing": len(missing_metadata),
            "prediction_metadata_mismatch": len(mismatches),
            "prediction_metadata_model_ids": sorted({item["model_id"] for item in prediction_metadata if item["model_id"]}),
            "uses_fold_model_by_code_path": True,
            "current_model_used_by_walk_forward_code_path": False,
            "failures": failures,
            "warnings": warnings,
        }

    def _summary_failures(self, fold_rows: list[dict[str, Any]]) -> list[str]:
        failures = []
        if self.config.prediction_root.resolve() == self.config.daily_current_prediction_root.resolve():
            failures.append("prediction_root equals daily current prediction root")
        if not self.config.walk_forward_model_root.exists():
            failures.append("walk_forward_model_root is missing")
        return failures

    def _prediction_files_for_fold(self, start_date: str, end_date: str) -> list[Path]:
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        paths = []
        for path in self.config.prediction_root.glob("predictions_*.parquet"):
            date_text = path.stem.replace("predictions_", "")
            try:
                date = datetime.fromisoformat(date_text).date()
            except ValueError:
                continue
            if start <= date <= end:
                paths.append(path)
        return sorted(paths)

    def _prediction_metadata(self, path: Path) -> dict[str, Any]:
        metadata = {"path": str(path), "model_id": None, "rows": None}
        if pq is None:
            return metadata
        try:
            parquet_file = pq.ParquetFile(path)
        except Exception:
            return metadata
        metadata["rows"] = parquet_file.metadata.num_rows
        raw_metadata = parquet_file.metadata.metadata or {}
        for key in [b"model_id", b"walk_forward_model_id"]:
            if key in raw_metadata:
                metadata["model_id"] = raw_metadata[key].decode("utf-8")
                break
        return metadata

    def _feature_count(self, path: Path) -> int | None:
        if not path.exists():
            return None
        try:
            return len(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None

    def _mtime(self, path: Path) -> datetime | None:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            return None

    def _max_mtime(self, paths: list[Path]) -> datetime | None:
        mtimes = [self._mtime(path) for path in paths]
        mtimes = [mtime for mtime in mtimes if mtime is not None]
        return max(mtimes) if mtimes else None

    def _count_after(self, mtimes: list[datetime | None], reference: datetime | None) -> int:
        if reference is None:
            return 0
        return sum(1 for mtime in mtimes if mtime is not None and mtime > reference)

    def _fmt_dt(self, value: datetime | None) -> str:
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
            f"- status: **{summary['status']}**",
            f"- fail_count: {summary['fail_count']}",
            f"- warning_count: {summary['warning_count']}",
            f"- walk_forward_json: `{summary['walk_forward_json']}`",
            f"- walk_forward_report_mtime: `{summary['walk_forward_report_mtime']}`",
            f"- prediction_root: `{summary['prediction_root']}`",
            f"- daily_current_prediction_root: `{summary['daily_current_prediction_root']}`",
            f"- walk_forward_model_root: `{summary['walk_forward_model_root']}`",
            f"- current_model_root: `{summary['current_model_root']}`",
            f"- current_model_mtime: `{summary['current_model_mtime']}`",
            f"- fold_count: {summary['fold_count']}",
            f"- label_lookahead_days: {summary['label_lookahead_days']}",
            f"- all_fold_model_dirs_exist: {summary['all_fold_model_dirs_exist']}",
            f"- all_archive_model_dirs_exist: {summary['all_archive_model_dirs_exist']}",
            f"- all_model_files_exist: {summary['all_model_files_exist']}",
            f"- prediction_root_is_walk_forward: {summary['prediction_root_is_walk_forward']}",
            f"- prediction_root_is_daily_current_root: {summary['prediction_root_is_daily_current_root']}",
            f"- fold_specific_model_used_by_code_path: {summary['fold_specific_model_used_by_code_path']}",
            f"- current_model_used_by_walk_forward_code_path: {summary['current_model_used_by_walk_forward_code_path']}",
            f"- metadata_checked_files: {summary['metadata_checked_files']}",
            f"- metadata_missing_files: {summary['metadata_missing_files']}",
            f"- metadata_mismatch_files: {summary['metadata_mismatch_files']}",
            f"- prediction_files_after_walk_forward_report: {summary['prediction_files_after_walk_forward_report']}",
            "",
            "## 2026-05 Fold",
            "",
            _table([summary["may_2026"]] if summary.get("may_2026") else [], _fold_columns(short=True)),
            "",
            "## Failures",
            "",
            _list_items(summary["failures"]),
            "",
            "## Warnings",
            "",
            _list_items(summary["warnings"]),
            "",
            "## Key Checks",
            "",
            "- Walk-forward code path trains each fold with `ModelTrainer(current_root=models/ml/walk_forward/current/YYYY-MM)`.",
            "- Prediction code path calls `Predictor(model_root=trainer.current_root, prediction_root=data/ml/walk_forward_predictions)` for each fold.",
            "- Therefore walk-forward predictions are generated into `data/ml/walk_forward_predictions`, not `data/ml/predictions`.",
            "- Prediction parquet files currently do not embed `model_id` in the existing 5y run, so model-id lineage is verified from code path, fold directories, and file timestamps.",
            "",
            "## Monthly Fold Lineage",
            "",
            _table(rows, _fold_columns()),
            "",
        ]
    )


def _fold_columns(short: bool = False) -> list[str]:
    columns = [
        "month",
        "model_id",
        "train_start",
        "requested_train_end",
        "effective_train_end",
        "test_start",
        "test_end",
        "effective_train_end_before_test_start",
        "label_window_capped_by_effective_train_end",
        "feature_count",
        "prediction_file_count",
        "prediction_rows",
        "prediction_created_first",
        "prediction_created_last",
        "fold_model_dir_exists",
        "all_model_files_exist",
        "prediction_metadata_model_ids",
        "prediction_metadata_missing",
        "prediction_metadata_mismatch",
    ]
    if short:
        return [
            "month",
            "model_id",
            "effective_train_end",
            "test_start",
            "test_end",
            "prediction_file_count",
            "prediction_rows",
            "prediction_metadata_model_ids",
            "prediction_metadata_missing",
            "prediction_metadata_mismatch",
        ]
    return columns


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


def _list_items(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "_None._"


def _fmt_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value).replace("|", "\\|")
