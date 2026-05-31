"""Earnings calendar filtering helpers."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any


EARNINGS_FILTER_REJECTED_REASON = "決算予定日前後のため新規買付見送り"


def normalize_earnings_calendar_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        announcement_date = _format_date(_first(record, ["Date", "date", "announcement_date"]))
        code = str(_first(record, ["Code", "code", "LocalCode"]) or "").strip()
        if not announcement_date or not code:
            continue
        normalized.append(
            {
                "Date": announcement_date,
                "Code": code,
                "CoName": _first(record, ["CoName", "CompanyName", "name"]) or "",
                "FY": _first(record, ["FY", "fiscal_year"]) or "",
                "SectorNm": _first(record, ["SectorNm", "SectorName", "sector_name"]) or "",
                "FQ": _first(record, ["FQ", "quarter"]) or "",
                "Section": _first(record, ["Section", "section", "market"]) or "",
            }
        )
    return normalized


def earnings_filter_result(
    candidate: dict[str, Any],
    target_date_text: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    filter_config = config.get("earnings_filter", {})
    if not bool(filter_config.get("enabled", False)):
        return _result(False, False, None, "")

    records = config.get("_earnings_calendar_records")
    if records is None:
        if bool(filter_config.get("fail_open", True)):
            return _result(True, False, None, "決算予定データ未取得のためfail_open")
        return _result(True, True, None, EARNINGS_FILTER_REJECTED_REASON)

    target_date = date.fromisoformat(target_date_text)
    code = str(candidate.get("code") or "").strip()
    before_days = int(filter_config.get("block_before_business_days", 0))
    after_days = int(filter_config.get("block_after_business_days", 0))
    block_on_day = bool(filter_config.get("block_on_earnings_day", True))

    matched_date: str | None = None
    matched_days_until: int | None = None
    for record in records:
        if str(record.get("Code") or "").strip() != code:
            continue
        earnings_date_text = _format_date(str(record.get("Date") or ""))
        if not earnings_date_text:
            continue
        earnings_date = date.fromisoformat(earnings_date_text)
        days_until = (earnings_date - target_date).days
        if matched_date is None or abs(days_until) < abs(matched_days_until or 999999):
            matched_date = earnings_date_text
            matched_days_until = days_until
        blocked_dates = set()
        if block_on_day:
            blocked_dates.add(earnings_date)
        blocked_dates.update(_business_days_before(earnings_date, before_days))
        blocked_dates.update(_business_days_after(earnings_date, after_days))
        if target_date in blocked_dates:
            return _result(
                True,
                True,
                earnings_date_text,
                EARNINGS_FILTER_REJECTED_REASON,
                info_found=True,
                candidate_earnings_date=earnings_date_text,
                days_until_earnings=days_until,
            )
    return _result(
        True,
        False,
        None,
        "",
        info_found=matched_date is not None,
        candidate_earnings_date=matched_date,
        days_until_earnings=matched_days_until,
    )


def earnings_counts(records: list[dict[str, Any]], target_date: date) -> dict[str, int]:
    next_day = _add_business_days(target_date, 1)
    return {
        "today": sum(1 for record in records if _format_date(str(record.get("Date") or "")) == target_date.isoformat()),
        "next_business_day": sum(1 for record in records if _format_date(str(record.get("Date") or "")) == next_day.isoformat()),
    }


def _result(
    checked: bool,
    blocked: bool,
    earnings_date: str | None,
    reason: str,
    info_found: bool = False,
    candidate_earnings_date: str | None = None,
    days_until_earnings: int | None = None,
) -> dict[str, Any]:
    return {
        "checked": checked,
        "blocked": blocked,
        "earnings_date": earnings_date,
        "reason": reason,
        "info_found": info_found,
        "candidate_earnings_date": candidate_earnings_date or earnings_date,
        "days_until_earnings": days_until_earnings,
    }


def _business_days_before(value: date, count: int) -> list[date]:
    days = []
    current = value
    while len(days) < count:
        current -= timedelta(days=1)
        if current.weekday() < 5:
            days.append(current)
    return days


def _business_days_after(value: date, count: int) -> list[date]:
    days = []
    current = value
    while len(days) < count:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days.append(current)
    return days


def _add_business_days(value: date, count: int) -> date:
    current = value
    added = 0
    while added < count:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def _first(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _format_date(value: str) -> str:
    value = str(value or "").strip()
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value
