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
TOPIX_CACHE_ROOT = CACHE_ROOT / "jquants" / "topix_prices"


@dataclass(frozen=True)
class FetchTarget:
    start_date: date
    end_date: date
    cache_path: Path
    exists: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch TOPIX daily bars into data/cache/jquants/topix_prices.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        print("error: --end must be on or after --start", file=sys.stderr)
        return 2
    targets = build_monthly_targets(start, end)
    print(format_plan(targets, args.overwrite, args.dry_run))
    if args.dry_run:
        return 0
    result = fetch_targets(targets, overwrite=args.overwrite)
    print(format_fetch_result(result))
    print(format_cache_summary(summarize_cache()))
    return 0 if not result["failed"] else 1


def build_monthly_targets(start: date, end: date) -> list[FetchTarget]:
    targets = []
    for month_start in pd.date_range(start=start, end=end, freq="MS"):
        target_start = max(month_start.date(), start)
        target_end = min((month_start + pd.offsets.MonthEnd(0)).date(), end)
        cache_path = TOPIX_CACHE_ROOT / f"{target_start.isoformat()}_to_{target_end.isoformat()}.json"
        targets.append(FetchTarget(target_start, target_end, cache_path, cache_path.exists()))
    if start.day != 1:
        first_month = pd.Timestamp(start).replace(day=1)
        if not targets or targets[0].start_date != start:
            target_end = min((first_month + pd.offsets.MonthEnd(0)).date(), end)
            cache_path = TOPIX_CACHE_ROOT / f"{start.isoformat()}_to_{target_end.isoformat()}.json"
            targets.insert(0, FetchTarget(start, target_end, cache_path, cache_path.exists()))
    return sorted({(t.start_date, t.end_date): t for t in targets}.values(), key=lambda t: t.start_date)


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
            skipped.append({"range": _range_text(target), "reason": "cache_exists", "cache_path": str(target.cache_path)})
            continue
        try:
            payload = service.fetch_topix_prices_cached(
                start_date=target.start_date,
                end_date=target.end_date,
                force_refresh=overwrite,
            )
        except Exception as exc:
            failed.append({"range": _range_text(target), "error": str(exc)})
            print(f"[{index}/{len(targets)}] range={_range_text(target)} status=failed error={exc}")
            continue
        records = payload.get("records", [])
        record_count = len(records) if isinstance(records, list) else 0
        records_total += record_count
        status = "saved" if payload.get("saved") else "empty" if not record_count else "ok"
        fetched.append(
            {
                "range": _range_text(target),
                "records": record_count,
                "status": status,
                "cache_path": payload.get("cache_path", str(target.cache_path)),
                "warning": payload.get("warning", ""),
                "reason": payload.get("reason", ""),
            }
        )
        print(f"[{index}/{len(targets)}] range={_range_text(target)} status={status} records={record_count}")
    return {"fetched": fetched, "skipped": skipped, "failed": failed, "records_total": records_total}


def summarize_cache(cache_root: Path = TOPIX_CACHE_ROOT) -> dict[str, Any]:
    files = sorted(cache_root.glob("*.json")) if cache_root.exists() else []
    total_records = 0
    dates = []
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
            if isinstance(record, dict) and (record.get("date") or record.get("Date")):
                dates.append(str(record.get("date") or record.get("Date"))[:10])
    dates = sorted(value for value in dates if value)
    return {
        "files": len(files),
        "records": total_records,
        "date_min": dates[0] if dates else "",
        "date_max": dates[-1] if dates else "",
        "failed_files": failed_files,
    }


def format_plan(targets: list[FetchTarget], overwrite: bool, dry_run: bool) -> str:
    existing = sum(1 for target in targets if target.exists)
    planned = sum(1 for target in targets if overwrite or not target.exists)
    return "\n".join(
        [
            f"dry_run={str(dry_run).lower()}",
            f"target_range={targets[0].start_date if targets else ''}_to_{targets[-1].end_date if targets else ''}",
            "granularity=monthly",
            f"target_files={len(targets)}",
            f"existing_files={existing}",
            f"planned_fetches={planned}",
            f"skipped_existing={len(targets) - planned}",
            f"overwrite={str(overwrite).lower()}",
            f"cache_root={TOPIX_CACHE_ROOT}",
        ]
    )


def format_fetch_result(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"fetch_summary fetched={len(result['fetched'])} skipped={len(result['skipped'])} "
            f"failed={len(result['failed'])} records={result['records_total']}",
            *[f"failed range={item['range']} error={item['error']}" for item in result["failed"]],
        ]
    )


def format_cache_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"topix_cache_files={summary['files']}",
        f"topix_cache_records={summary['records']}",
        f"topix_cache_date_range={summary['date_min']}_to_{summary['date_max']}",
    ]
    for item in summary["failed_files"]:
        lines.append(f"warning=failed_to_read_cache path={item['path']} error={item['error']}")
    return "\n".join(lines)


def _range_text(target: FetchTarget) -> str:
    return f"{target.start_date.isoformat()}_to_{target.end_date.isoformat()}"


if __name__ == "__main__":
    raise SystemExit(main())
