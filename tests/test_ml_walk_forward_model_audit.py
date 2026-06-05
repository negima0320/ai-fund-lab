from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.config import MODEL_FILENAMES
from ml.walk_forward_model_audit import (
    WalkForwardModelAuditConfig,
    WalkForwardModelAuditor,
    format_markdown,
)


def _write_walk_forward_json(path: Path, effective_train_end: str = "2026-04-01") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "month": "2026-05",
                        "train_start": "2021-06-01",
                        "requested_train_end": "2026-04-30",
                        "effective_train_end": effective_train_end,
                        "test_start": "2026-05-01",
                        "test_end": "2026-05-31",
                        "predicted_dates": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_fold_model(root: Path, month: str = "2026-05") -> None:
    model_root = root / "current" / month
    archive_root = root / "archive" / f"walk_forward_{month.replace('-', '')}"
    model_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    (model_root / "feature_columns.json").write_text('["close"]', encoding="utf-8")
    (model_root / "metrics.json").write_text("{}", encoding="utf-8")
    (archive_root / "metrics.json").write_text("{}", encoding="utf-8")
    for filename in MODEL_FILENAMES.values():
        (model_root / filename).write_text("model", encoding="utf-8")


def _write_prediction(path: Path, model_id: str | None = "walk_forward_202605") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        pd.DataFrame({"date": [pd.Timestamp("2026-05-01")], "code": ["1001"], "expected_return_10d": [0.1]})
    )
    if model_id is not None:
        table = table.replace_schema_metadata({b"model_id": model_id.encode("utf-8")})
    pq.write_table(table, path)


def _audit(tmp_path: Path, prediction_root: Path | None = None) -> dict:
    wf_json = tmp_path / "reports" / "walk_forward.json"
    model_root = tmp_path / "models" / "ml" / "walk_forward"
    current_root = tmp_path / "models" / "ml" / "current"
    pred_root = prediction_root or tmp_path / "data" / "ml" / "walk_forward_predictions"
    current_root.mkdir(parents=True, exist_ok=True)
    _write_walk_forward_json(wf_json)
    _write_fold_model(model_root)
    _write_prediction(pred_root / "predictions_2026-05-01.parquet")
    return WalkForwardModelAuditor(
        WalkForwardModelAuditConfig(
            walk_forward_json=wf_json,
            prediction_root=pred_root,
            walk_forward_model_root=model_root,
            current_model_root=current_root,
            daily_current_prediction_root=tmp_path / "data" / "ml" / "predictions",
        )
    ).build_report()


def test_walk_forward_model_audit_passes_with_matching_metadata(tmp_path) -> None:
    report = _audit(tmp_path)

    assert report["summary"]["status"] == "pass"
    assert report["summary"]["fold_count"] == 1
    assert report["summary"]["fail_count"] == 0
    assert report["summary"]["warning_count"] == 0
    assert report["summary"]["may_2026"]["model_id"] == "walk_forward_202605"
    assert report["summary"]["may_2026"]["prediction_rows"] == 1
    assert report["summary"]["current_model_used_by_walk_forward_code_path"] is False
    markdown = format_markdown(report)
    assert "walk_forward_202605" in markdown
    assert "status: **pass**" in markdown


def test_walk_forward_model_audit_fails_when_effective_train_end_reaches_test_start(tmp_path) -> None:
    wf_json = tmp_path / "reports" / "walk_forward.json"
    model_root = tmp_path / "models" / "ml" / "walk_forward"
    pred_root = tmp_path / "data" / "ml" / "walk_forward_predictions"
    _write_walk_forward_json(wf_json, effective_train_end="2026-05-01")
    _write_fold_model(model_root)
    _write_prediction(pred_root / "predictions_2026-05-01.parquet")

    report = WalkForwardModelAuditor(
        WalkForwardModelAuditConfig(
            walk_forward_json=wf_json,
            prediction_root=pred_root,
            walk_forward_model_root=model_root,
            current_model_root=tmp_path / "models" / "ml" / "current",
            daily_current_prediction_root=tmp_path / "data" / "ml" / "predictions",
        )
    ).build_report()

    assert report["summary"]["status"] == "fail"
    assert any("effective_train_end >= test_start" in item for item in report["summary"]["failures"])


def test_walk_forward_model_audit_fails_on_model_id_mismatch(tmp_path) -> None:
    wf_json = tmp_path / "reports" / "walk_forward.json"
    model_root = tmp_path / "models" / "ml" / "walk_forward"
    pred_root = tmp_path / "data" / "ml" / "walk_forward_predictions"
    _write_walk_forward_json(wf_json)
    _write_fold_model(model_root)
    _write_prediction(pred_root / "predictions_2026-05-01.parquet", model_id="walk_forward_202604")

    report = WalkForwardModelAuditor(
        WalkForwardModelAuditConfig(
            walk_forward_json=wf_json,
            prediction_root=pred_root,
            walk_forward_model_root=model_root,
            current_model_root=tmp_path / "models" / "ml" / "current",
            daily_current_prediction_root=tmp_path / "data" / "ml" / "predictions",
        )
    ).build_report()

    assert report["summary"]["status"] == "fail"
    assert report["summary"]["metadata_mismatch_files"] == 1


def test_walk_forward_model_audit_warns_when_metadata_is_missing(tmp_path) -> None:
    wf_json = tmp_path / "reports" / "walk_forward.json"
    model_root = tmp_path / "models" / "ml" / "walk_forward"
    pred_root = tmp_path / "data" / "ml" / "walk_forward_predictions"
    _write_walk_forward_json(wf_json)
    _write_fold_model(model_root)
    _write_prediction(pred_root / "predictions_2026-05-01.parquet", model_id=None)

    report = WalkForwardModelAuditor(
        WalkForwardModelAuditConfig(
            walk_forward_json=wf_json,
            prediction_root=pred_root,
            walk_forward_model_root=model_root,
            current_model_root=tmp_path / "models" / "ml" / "current",
            daily_current_prediction_root=tmp_path / "data" / "ml" / "predictions",
        )
    ).build_report()

    assert report["summary"]["status"] == "warning"
    assert report["summary"]["metadata_missing_files"] == 1


def test_walk_forward_model_audit_fails_when_prediction_root_is_current_root(tmp_path) -> None:
    daily_root = tmp_path / "data" / "ml" / "predictions"
    report = _audit(tmp_path, prediction_root=daily_root)

    assert report["summary"]["status"] == "fail"
    assert any("prediction_root equals daily current prediction root" in item for item in report["summary"]["failures"])
