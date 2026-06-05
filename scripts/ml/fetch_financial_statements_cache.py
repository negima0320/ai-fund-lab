#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_provider import JQuantsDataProvider
from jquants_plan import resolve_jquants_plan


CACHE_ROOT = ROOT / "data" / "cache"
FINANCIAL_CACHE_ROOT = CACHE_ROOT / "jquants" / "financial_statements"


@dataclass(frozen=True)
class FetchTarget:
    start_date: date
    end_date: date
    cache_path: Path
    exists: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch /fins/summary into data/cache/jquants/financial_statements.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--dry-run", action="store_true", help="Print fetch plan without calling J-Quants API.")
    parser.add_argument("--overwrite", action="store_true", help="Refresh files that already exist. Default is skip existing cache.")
    parser.add_argument(
        "--include-weekends",
        action="store_true",
        help="Also request Saturday/Sunday. Default requests weekdays only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        print("error: --end must be on or after --start", file=sys.stderr)
        return 2

    targets = build_targets(start, end, include_weekends=args.include_weekends)
    print(format_plan(targets, overwrite=args.overwrite, dry_run=args.dry_run))
    if args.dry_run:
        return 0

    result = fetch_targets(targets, overwrite=args.overwrite)
    print(format_fetch_result(result))
    print(format_cache_summary(summarize_cache(FINANCIAL_CACHE_ROOT)))
    return 0 if not result["failed"] else 1


def build_targets(start: date, end: date, include_weekends: bool = False) -> list[FetchTarget]:
    dates = pd.date_range(start=start, end=end, freq="D")
    targets = []
    for value in dates:
        day = value.date()
        if not include_weekends and day.weekday() >= 5:
            continue
        cache_path = FINANCIAL_CACHE_ROOT / f"{day.isoformat()}_to_{day.isoformat()}.json"
        targets.append(FetchTarget(day, day, cache_path, cache_path.exists()))
    return targets


def fetch_targets(targets: list[FetchTarget], overwrite: bool = False) -> dict[str, Any]:
    resolution = resolve_jquants_plan(config_root=ROOT)
    for warning in resolution.warnings:
        print(f"warning={warning}")
    print(
        "jquants_plan="
        f"{resolution.plan} source={resolution.source} requests_per_minute={resolution.requests_per_minute}"
    )

    provider = JQuantsDataProvider(
        ROOT / ".env",
        plan=resolution.plan,
        requests_per_minute=resolution.requests_per_minute,
        parallel_fetch=False,
        max_parallel_requests=1,
    )
    service = provider.service(CACHE_ROOT)

    fetched = []
    skipped = []
    failed = []
    records_total = 0
    for index, target in enumerate(targets, start=1):
        if target.exists and not overwrite:
            skipped.append({"date": target.start_date.isoformat(), "reason": "cache_exists", "cache_path": str(target.cache_path)})
            continue
        try:
            payload = service.fetch_financial_statements_cached(
                start_date=target.start_date,
                end_date=target.end_date,
                force_refresh=overwrite,
            )
        except Exception as exc:
            failed.append({"date": target.start_date.isoformat(), "error": str(exc)})
            print(f"[{index}/{len(targets)}] date={target.start_date} status=failed error={exc}")
            continue
        records = payload.get("records", [])
        record_count = len(records) if isinstance(records, list) else 0
        records_total += record_count
        status = "saved" if payload.get("saved") else "empty" if not record_count else "ok"
        fetched.append(
            {
                "date": target.start_date.isoformat(),
                "records": record_count,
                "status": status,
                "cache_path": payload.get("cache_path", str(target.cache_path)),
                "warning": payload.get("warning", ""),
                "reason": payload.get("reason", ""),
            }
        )
        print(f"[{index}/{len(targets)}] date={target.start_date} status={status} records={record_count}")
    return {"fetched": fetched, "skipped": skipped, "failed": failed, "records_total": records_total}


def summarize_cache(cache_root: Path = FINANCIAL_CACHE_ROOT) -> dict[str, Any]:
    files = sorted(cache_root.glob("*.json")) if cache_root.exists() else []
    total_records = 0
    disc_dates = []
    codes = set()
    failed_files = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failed_files.append({"path": str(path), "error": str(exc)})
            continue
        records = payload.get("records", [])
        if not isinstance(records, list):
            continue
        total_records += len(records)
        for record in records:
            if not isinstance(record, dict):
                continue
            disc_date = record.get("DiscDate") or record.get("DisclosedDate") or record.get("date") or record.get("Date")
            if disc_date:
                disc_dates.append(_normalize_date(str(disc_date)))
            code = record.get("Code") or record.get("code") or record.get("LocalCode")
            if code:
                codes.add(str(code))
    disc_dates = sorted(value for value in disc_dates if value)
    return {
        "files": len(files),
        "records": total_records,
        "disc_date_min": disc_dates[0] if disc_dates else "",
        "disc_date_max": disc_dates[-1] if disc_dates else "",
        "codes": len(codes),
        "failed_files": failed_files,
    }


def format_plan(targets: list[FetchTarget], overwrite: bool, dry_run: bool) -> str:
    existing = sum(1 for target in targets if target.exists)
    planned = sum(1 for target in targets if overwrite or not target.exists)
    skipped = len(targets) - planned
    first = targets[0].start_date.isoformat() if targets else ""
    last = targets[-1].start_date.isoformat() if targets else ""
    return "\n".join(
        [
            f"dry_run={str(dry_run).lower()}",
            f"target_range={first}_to_{last}",
            "granularity=daily_weekdays",
            f"target_files={len(targets)}",
            f"existing_files={existing}",
            f"planned_fetches={planned}",
            f"skipped_existing={skipped}",
            f"overwrite={str(overwrite).lower()}",
            f"cache_root={FINANCIAL_CACHE_ROOT}",
        ]
    )


def format_fetch_result(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"fetch_summary fetched={len(result['fetched'])} skipped={len(result['skipped'])} "
            f"failed={len(result['failed'])} records={result['records_total']}",
            *[f"failed date={item['date']} error={item['error']}" for item in result["failed"]],
        ]
    )


def format_cache_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"financial_cache_files={summary['files']}",
        f"financial_cache_records={summary['records']}",
        f"financial_cache_disc_date_range={summary['disc_date_min']}_to_{summary['disc_date_max']}",
        f"financial_cache_code_count={summary['codes']}",
    ]
    for item in summary["failed_files"]:
        lines.append(f"warning=failed_to_read_cache path={item['path']} error={item['error']}")
    return "\n".join(lines)


def _normalize_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value[:10]


if __name__ == "__main__":
    raise SystemExit(main())
