#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.config import (
    CATEGORICAL_FEATURE_COLUMNS,
    EARNINGS_FEATURE_COLUMNS,
    FINANCIAL_FEATURE_COLUMNS,
    ML_FEATURES_ROOT,
    ML_REPORTS_ROOT,
    TOPIX_FEATURE_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report enriched ML feature cache and non-null coverage.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--model-root", default="models/ml/walk_forward/current/2026-05")
    parser.add_argument("--output-prefix", default="enriched_feature_coverage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "period": {"start": args.start, "end": args.end},
        "cache": {
            "financial_statements": cache_summary(ROOT / "data/cache/jquants/financial_statements", "DiscDate", "Code"),
            "listed_info": cache_summary(ROOT / "data/cache/jquants/listed_info", "Date", "Code"),
            "topix_prices": cache_summary(ROOT / "data/cache/jquants/topix_prices", "date", None),
            "earnings_calendar": cache_summary(ROOT / "data/cache/jquants/earnings_calendar", "Date", "Code"),
        },
        "feature_coverage": feature_coverage(args.start, args.end),
        "feature_importance_top50": feature_importance(Path(args.model_root), 50),
    }
    md_path, json_path = save_reports(result, args.output_prefix)
    print(f"saved markdown report to {md_path}")
    print(f"saved json report to {json_path}")
    coverage = result["feature_coverage"]
    print(f"feature_files={coverage['files']} rows={coverage['rows']} columns={coverage['columns']}")
    for group, stats in coverage["groups"].items():
        print(f"{group}_mean_non_null_rate={stats['mean_non_null_rate']}")


def cache_summary(root: Path, date_key: str, code_key: str | None) -> dict[str, Any]:
    files = sorted(root.glob("*.json")) if root.exists() else []
    records_total = 0
    dates = []
    codes = set()
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records = payload.get("records", []) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            continue
        records_total += len(records)
        for record in records:
            if not isinstance(record, dict):
                continue
            date_value = record.get(date_key) or record.get("date") or record.get("Date")
            if date_value:
                dates.append(normalize_date(str(date_value)))
            if code_key:
                code_value = record.get(code_key) or record.get("code") or record.get("Code")
                if code_value:
                    codes.add(str(code_value))
    dates = sorted(value for value in dates if value)
    return {
        "files": len(files),
        "records": records_total,
        "date_min": dates[0] if dates else "",
        "date_max": dates[-1] if dates else "",
        "codes": len(codes) if code_key else None,
    }


def feature_coverage(start_date: str, end_date: str) -> dict[str, Any]:
    frames = []
    for date_text in date_texts(start_date, end_date):
        path = ML_FEATURES_ROOT / f"features_{date_text}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return {"files": 0, "rows": 0, "columns": 0, "groups": {}}
    data = pd.concat(frames, ignore_index=True)
    groups = {
        "financial": FINANCIAL_FEATURE_COLUMNS,
        "listed_info": CATEGORICAL_FEATURE_COLUMNS,
        "topix": TOPIX_FEATURE_COLUMNS,
        "earnings": EARNINGS_FEATURE_COLUMNS,
    }
    return {
        "files": len(frames),
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "groups": {name: coverage_for_columns(data, columns) for name, columns in groups.items()},
    }


def coverage_for_columns(data: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    rates = {}
    for column in columns:
        if column in data.columns:
            rates[column] = float(data[column].notna().mean())
        else:
            rates[column] = 0.0
    return {
        "mean_non_null_rate": float(sum(rates.values()) / len(rates)) if rates else 0.0,
        "columns": rates,
    }


def feature_importance(model_root: Path, top_n: int) -> list[dict[str, Any]]:
    path = model_root / "future_10d_return_regression.joblib"
    columns_path = model_root / "feature_columns.json"
    if not path.exists() or not columns_path.exists():
        return []
    try:
        import joblib
    except ModuleNotFoundError:
        return []
    model = joblib.load(path)
    columns = json.loads(columns_path.read_text(encoding="utf-8"))
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    rows = [
        {"rank": rank, "feature": feature, "importance": int(importance)}
        for rank, (feature, importance) in enumerate(
            sorted(zip(columns, importances), key=lambda item: item[1], reverse=True)[:top_n],
            start=1,
        )
    ]
    return rows


def save_reports(result: dict[str, Any], prefix: str) -> tuple[Path, Path]:
    ML_REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    md_path = ML_REPORTS_ROOT / f"{prefix}_{result['period']['start']}_to_{result['period']['end']}.md"
    json_path = ML_REPORTS_ROOT / f"{prefix}_{result['period']['start']}_to_{result['period']['end']}.json"
    md_path.write_text(format_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return md_path, json_path


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Enriched Feature Coverage",
        "",
        f"- period: {result['period']['start']} to {result['period']['end']}",
        "",
        "## Cache Summary",
        "",
        table(result["cache"].values(), ["files", "records", "date_min", "date_max", "codes"], names=list(result["cache"].keys())),
        "",
        "## Feature Coverage",
        "",
    ]
    for group, stats in result["feature_coverage"].get("groups", {}).items():
        lines.extend([f"### {group}", "", table(stats["columns"].items(), ["column", "non_null_rate"]), ""])
    lines.extend(["## Feature Importance Top 50", "", table(result["feature_importance_top50"], ["rank", "feature", "importance"]), ""])
    return "\n".join(lines)


def table(rows: Any, columns: list[str], names: list[str] | None = None) -> str:
    materialized = list(rows)
    if not materialized:
        return "_No rows._"
    header = "| " + " | ".join((["name"] if names else []) + columns) + " |"
    separator = "| " + " | ".join(["---"] * (len(columns) + (1 if names else 0))) + " |"
    body = []
    for index, row in enumerate(materialized):
        if isinstance(row, tuple) and len(row) == 2:
            values = [row[0], row[1]]
        elif isinstance(row, dict):
            values = [row.get(column, "") for column in columns]
        else:
            values = ["" for _ in columns]
        if names:
            values = [names[index], *values]
        body.append("| " + " | ".join(fmt(value) for value in values) + " |")
    return "\n".join([header, separator, *body])


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def normalize_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value[:10]


def date_texts(start_date: str, end_date: str) -> list[str]:
    return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]


if __name__ == "__main__":
    main()
