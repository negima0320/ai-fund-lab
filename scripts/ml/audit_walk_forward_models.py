#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.config import ML_DATA_ROOT, ML_MODELS_ROOT, ML_REPORTS_ROOT
from ml.walk_forward_model_audit import (
    WalkForwardModelAuditConfig,
    WalkForwardModelAuditor,
    format_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit walk-forward model/prediction lineage.")
    parser.add_argument(
        "--walk-forward-json",
        default=str(ML_REPORTS_ROOT / "walk_forward_5y_enriched_v2_2023-01_to_2026-05.json"),
    )
    parser.add_argument("--prediction-root", default=str(ML_DATA_ROOT / "walk_forward_predictions"))
    parser.add_argument("--walk-forward-model-root", default=str(ML_MODELS_ROOT / "walk_forward"))
    parser.add_argument("--current-model-root", default=str(ML_MODELS_ROOT / "current"))
    parser.add_argument("--output-md", default=str(ML_REPORTS_ROOT / "walk_forward_model_audit_5y_enriched_v2.md"))
    parser.add_argument("--output-json", default=str(ML_REPORTS_ROOT / "walk_forward_model_audit_5y_enriched_v2.json"))
    parser.add_argument("--output", default=None, help="Backward-compatible alias for --output-md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_md = Path(args.output or args.output_md)
    output_json = Path(args.output_json)
    report = WalkForwardModelAuditor(
        WalkForwardModelAuditConfig(
            walk_forward_json=Path(args.walk_forward_json),
            prediction_root=Path(args.prediction_root),
            walk_forward_model_root=Path(args.walk_forward_model_root),
            current_model_root=Path(args.current_model_root),
        )
    ).build_report()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(format_markdown(report), encoding="utf-8")
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = report["summary"]
    may_2026 = summary.get("may_2026") or {}
    print(f"saved markdown report to {output_md}")
    print(f"saved json report to {output_json}")
    print(f"audit_result={summary['status']}")
    print(f"folds={summary['fold_count']}")
    print(f"fail_count={summary['fail_count']}")
    print(f"warning_count={summary['warning_count']}")
    print(f"may_2026_model_id={may_2026.get('model_id')}")
    print(f"may_2026_train_end={may_2026.get('effective_train_end')}")
    print(f"may_2026_prediction_rows={may_2026.get('prediction_rows')}")
    print(f"current_model_used_by_walk_forward_code_path={summary['current_model_used_by_walk_forward_code_path']}")


if __name__ == "__main__":
    main()
