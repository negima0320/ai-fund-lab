#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw"
CACHE_ROOT = ROOT / "data" / "cache" / "jquants" / "prices"
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync raw price snapshots to ML J-Quants cache format.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Sync one date in YYYY-MM-DD format.")
    group.add_argument("--start", help="Start date for a small inclusive range.")
    parser.add_argument("--end", help="End date for a small inclusive range. Required with --start.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned syncs without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start and not args.end:
        print("error: --end is required when --start is used", file=sys.stderr)
        return 2
    dates = [args.date] if args.date else _date_range(args.start, args.end)
    summary = sync_dates(dates, dry_run=args.dry_run, overwrite=args.overwrite)
    print(format_summary(summary))
    return 0


def sync_dates(
    dates: list[str],
    raw_root: Path = RAW_ROOT,
    cache_root: Path = CACHE_ROOT,
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    results = []
    for date_text in dates:
        raw_path = raw_root / f"prices_{date_text}.json"
        cache_path = cache_root / f"{date_text}.json"
        result = sync_one(raw_path, cache_path, date_text, dry_run=dry_run, overwrite=overwrite)
        results.append(result)
    return results


def sync_one(raw_path: Path, cache_path: Path, date_text: str, dry_run: bool = False, overwrite: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "date": date_text,
        "raw_path": raw_path,
        "cache_path": cache_path,
        "records": 0,
        "written": False,
        "skipped": False,
        "warnings": [],
    }
    if not raw_path.exists():
        result["skipped"] = True
        result["warnings"].append(f"raw file is missing: {raw_path}")
        return result
    if cache_path.exists() and not overwrite:
        result["skipped"] = True
        result["warnings"].append(f"cache file already exists: {cache_path}")
        return result

    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["skipped"] = True
        result["warnings"].append(f"failed to read raw json: {exc}")
        return result

    prices = _extract_prices(payload)
    if prices is None:
        result["skipped"] = True
        result["warnings"].append(f"raw prices array not found: {raw_path}")
        return result

    records = [_normalize_price(record, date_text) for record in prices if isinstance(record, dict)]
    records = [record for record in records if record is not None]
    result["records"] = len(records)
    if not records:
        result["warnings"].append(f"raw prices array is empty or invalid: {raw_path}")

    if dry_run:
        return result

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["written"] = True
    return result


def format_summary(results: list[dict[str, Any]]) -> str:
    lines = []
    for item in results:
        status = "written" if item["written"] else "skipped" if item["skipped"] else "planned"
        lines.append(
            f"date={item['date']} status={status} records={item['records']} "
            f"raw={item['raw_path']} cache={item['cache_path']}"
        )
        for warning in item["warnings"]:
            lines.append(f"warning={warning}")
    total_written = sum(1 for item in results if item["written"])
    total_records = sum(int(item["records"]) for item in results)
    lines.append(f"summary files={len(results)} written={total_written} records={total_records}")
    return "\n".join(lines)


def _extract_prices(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("prices"), list):
        return payload["prices"]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    return None


def _normalize_price(record: dict[str, Any], fallback_date: str) -> dict[str, Any] | None:
    code = _first(record, ["code", "Code", "LocalCode"])
    date_value = _first(record, ["date", "Date"]) or fallback_date
    close = _number(_first(record, ["close", "C", "Close"]))
    if code is None or date_value is None or close is None:
        return None
    return {
        "date": _normalize_date(str(date_value)),
        "code": str(code),
        "open": _number(_first(record, ["open", "O", "Open"])),
        "high": _number(_first(record, ["high", "H", "High"])),
        "low": _number(_first(record, ["low", "L", "Low"])),
        "close": close,
        "volume": _number(_first(record, ["volume", "Vo", "Volume"])),
        "turnover_value": _number(_first(record, ["turnover_value", "Va", "TurnoverValue"])),
    }


def _first(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    match = DATE_RE.search(value)
    return match.group(0) if match else value


def _date_range(start_date: str, end_date: str) -> list[str]:
    return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]


if __name__ == "__main__":
    raise SystemExit(main())
