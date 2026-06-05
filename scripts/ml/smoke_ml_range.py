#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.range_evaluator import RangeSmokeRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short multi-day ML smoke loop.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--train-end", required=True, help="Last train date in YYYY-MM-DD format.")
    parser.add_argument("--valid-end", required=True, help="Last validation date in YYYY-MM-DD format.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top ml_score rows to summarize.")
    return parser.parse_args()


def run_smoke_range(
    start_date: str,
    end_date: str,
    train_end: str,
    valid_end: str,
    top_n: int = 10,
    runner: RangeSmokeRunner | None = None,
) -> dict[str, Any]:
    runner = runner or RangeSmokeRunner()
    return runner.run(start_date, end_date, train_end, valid_end, top_n=top_n)


def format_range_result(result: dict[str, Any]) -> str:
    lines = [
        f"start_date={result['start_date']}",
        f"end_date={result['end_date']}",
        f"train_end={result['train_end']}",
        f"valid_end={result['valid_end']}",
        f"processed_dates={_format_list(result['processed_dates'])}",
        f"skipped_dates={_format_skipped(result['skipped_dates'])}",
        f"features_total_rows={result['features_total_rows']}",
        f"labels_total_rows={result['labels_total_rows']}",
        f"dataset rows={result['dataset_rows']}",
        f"train rows={result['train_rows']}",
        f"valid rows={result['valid_rows']}",
        f"test rows={result['test_rows']}",
        f"prediction_rows_total={result['prediction_rows_total']}",
        f"joined_evaluation_rows_total={result['joined_evaluation_rows_total']}",
        f"risk_bad_entry_rates={_format_rates(result['risk_bad_entry_rates'])}",
        f"top10_future_10d_return_mean={_fmt(result['top_n_future_10d_return_mean'])}",
        f"expected_vs_future_10d_corr={_fmt(result['expected_vs_future_10d_corr'])}",
        f"dataset_path={result['dataset_path']}",
        f"train_path={result['train_path']}",
        f"valid_path={result['valid_path']}",
        f"test_path={result['test_path']}",
        f"model_path={result['model_path'] or 'skipped'}",
        f"range_report_path={result['range_report_path']}",
    ]
    for warning in result["warnings"]:
        lines.append(f"warning={warning}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = run_smoke_range(args.start, args.end, args.train_end, args.valid_end, top_n=args.top_n)
    print(format_range_result(result))


def _format_list(values: list[str]) -> str:
    return ",".join(values) if values else "none"


def _format_skipped(values: list[dict[str, str]]) -> str:
    if not values:
        return "none"
    return "; ".join(f"{item['date']}:{item['reason']}" for item in values)


def _format_rates(values: dict[str, float | None]) -> str:
    if not values:
        return "none"
    return ",".join(f"{key}:{_fmt(value)}" for key, value in sorted(values.items()))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
