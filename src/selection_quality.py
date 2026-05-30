"""Selection quality analysis for screened, scored, selected, and rejected stocks."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from db import get_database_path


def build_selection_quality_analysis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    db_path = get_database_path(config, root)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    profile_id = _profile_id(config)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        screening_rows = _rows(
            connection,
            """
            SELECT *
            FROM screening_results
            WHERE profile_id = ?
            ORDER BY date, id
            """,
            (profile_id,),
        )
        scoring_rows = _rows(
            connection,
            """
            SELECT *
            FROM scoring_results
            WHERE profile_id = ?
            ORDER BY date, rank, id
            """,
            (profile_id,),
        )

    prices_by_code = _load_price_history(root)
    screening_by_key = {(row.get("date"), row.get("code")): row for row in screening_rows}
    screen_records = [
        _quality_record(row, row, prices_by_code, source_stage="screen")
        for row in screening_rows
    ]
    score_records = [
        _quality_record(row, screening_by_key.get((row.get("date"), row.get("code")), {}), prices_by_code, source_stage="score")
        for row in scoring_rows
    ]
    selected_records = [record for record in score_records if record.get("selected")]
    rejected_records = [record for record in score_records if not record.get("selected")]

    return {
        "profile_id": profile_id,
        "profile_name": _profile_name(config),
        "screen_candidate_count": len(screen_records),
        "score_candidate_count": len(score_records),
        "selected_count": len(selected_records),
        "rejected_count": len(rejected_records),
        "screen_candidates": _return_summary(screen_records),
        "score_candidates": _return_summary(score_records),
        "selected": _return_summary(selected_records),
        "rejected": _return_summary(rejected_records),
        "selection_lift": {
            "return_5d": _lift(selected_records, rejected_records, "return_5d"),
            "return_10d": _lift(selected_records, rejected_records, "return_10d"),
        },
        "top_missed_opportunities": _top_records(
            rejected_records,
            key="return_10d",
            reverse=True,
            limit=10,
        ),
        "top_false_positives": _top_records(
            selected_records,
            key="return_10d",
            reverse=False,
            limit=10,
        ),
    }


def render_selection_quality_markdown(analysis: dict[str, Any]) -> str:
    selected = analysis.get("selected", {})
    rejected = analysis.get("rejected", {})
    lift = analysis.get("selection_lift", {})
    lines = [
        "# Selection Quality Analysis",
        "",
        f"- profile_id: {analysis.get('profile_id')}",
        f"- profile_name: {analysis.get('profile_name')}",
        f"- screen_candidate_count: {analysis.get('screen_candidate_count')}",
        f"- score_candidate_count: {analysis.get('score_candidate_count')}",
        f"- selected_count: {analysis.get('selected_count')}",
        f"- rejected_count: {analysis.get('rejected_count')}",
        "",
        "## Future Return Summary",
        "",
        f"- selected平均5日リターン: {_format_percent(selected.get('average_return_5d'))}",
        f"- rejected平均5日リターン: {_format_percent(rejected.get('average_return_5d'))}",
        f"- selected平均10日リターン: {_format_percent(selected.get('average_return_10d'))}",
        f"- rejected平均10日リターン: {_format_percent(rejected.get('average_return_10d'))}",
        f"- Selection Lift 5d: {_format_percent(lift.get('return_5d'))}",
        f"- Selection Lift 10d: {_format_percent(lift.get('return_10d'))}",
        "",
        "## Stage Comparison",
        "",
        *_stage_lines(analysis),
        "",
        "## Top Missed Opportunities",
        "",
        *_record_lines(analysis.get("top_missed_opportunities", [])),
        "",
        "## Top False Positives",
        "",
        *_record_lines(analysis.get("top_false_positives", [])),
    ]
    return "\n".join(lines)


def _quality_record(
    row: dict[str, Any],
    screening: dict[str, Any],
    prices_by_code: dict[str, list[dict[str, Any]]],
    source_stage: str,
) -> dict[str, Any]:
    code = str(row.get("code") or "")
    base_date = row.get("date")
    base_close = _number(screening.get("close")) or _price_on_date(prices_by_code.get(code, []), base_date)
    price_5d = _future_price(prices_by_code.get(code, []), base_date, 5)
    price_10d = _future_price(prices_by_code.get(code, []), base_date, 10)
    return {
        "source_stage": source_stage,
        "date": base_date,
        "code": code,
        "name": row.get("name") or screening.get("name"),
        "selected": bool(row.get("selected")) if source_stage == "score" else None,
        "rank": _int(row.get("rank")),
        "total_score": _number(row.get("total_score")),
        "rejected_reason": row.get("rejected_reason"),
        "base_close": base_close,
        "price_5d": price_5d,
        "price_10d": price_10d,
        "return_5d": _return_rate(base_close, price_5d),
        "return_10d": _return_rate(base_close, price_10d),
    }


def _return_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    returns_5d = _valid_numbers(record.get("return_5d") for record in records)
    returns_10d = _valid_numbers(record.get("return_10d") for record in records)
    return {
        "count": len(records),
        "return_5d_count": len(returns_5d),
        "return_10d_count": len(returns_10d),
        "average_return_5d": _average(returns_5d),
        "average_return_10d": _average(returns_10d),
    }


def _lift(selected: list[dict[str, Any]], rejected: list[dict[str, Any]], key: str) -> float | None:
    selected_average = _average(_valid_numbers(record.get(key) for record in selected))
    rejected_average = _average(_valid_numbers(record.get(key) for record in rejected))
    if selected_average is None or rejected_average is None:
        return None
    return round(selected_average - rejected_average, 6)


def _top_records(records: list[dict[str, Any]], key: str, reverse: bool, limit: int) -> list[dict[str, Any]]:
    ranked = [record for record in records if _number(record.get(key)) is not None]
    ranked.sort(key=lambda record: float(record.get(key) or 0.0), reverse=reverse)
    return [
        {
            "date": record.get("date"),
            "code": record.get("code"),
            "name": record.get("name"),
            "rank": record.get("rank"),
            "total_score": record.get("total_score"),
            "rejected_reason": record.get("rejected_reason"),
            "return_5d": record.get("return_5d"),
            "return_10d": record.get("return_10d"),
        }
        for record in ranked[:limit]
    ]


def _load_price_history(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_dir = root / "data" / "raw"
    for path in sorted(raw_dir.glob("prices_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("prices", []):
            code = str(row.get("code") or row.get("Code") or row.get("LocalCode") or "")
            row_date = row.get("date") or row.get("Date")
            close = _number(row.get("close") or row.get("Close") or row.get("AdjustmentClose"))
            if code and row_date and close is not None:
                result[code].append({"date": str(row_date), "close": close})
    for code in result:
        result[code] = sorted(result[code], key=lambda row: row.get("date") or "")
    return result


def _future_price(prices: list[dict[str, Any]], base_date: str | None, offset: int) -> float | None:
    if not base_date:
        return None
    future = [row for row in prices if row.get("date") and row.get("date") > base_date]
    if len(future) < offset:
        return None
    return _number(future[offset - 1].get("close"))


def _price_on_date(prices: list[dict[str, Any]], target_date: str | None) -> float | None:
    if not target_date:
        return None
    for row in prices:
        if row.get("date") == target_date:
            return _number(row.get("close"))
    return None


def _return_rate(base: float | None, future: float | None) -> float | None:
    if base is None or future is None or base == 0:
        return None
    return round((future - base) / base, 6)


def _stage_lines(analysis: dict[str, Any]) -> list[str]:
    labels = [
        ("screen_candidates", "screen候補"),
        ("score_candidates", "score候補"),
        ("selected", "selected銘柄"),
        ("rejected", "rejected銘柄"),
    ]
    lines = []
    for key, label in labels:
        summary = analysis.get(key, {})
        lines.append(
            f"- {label}: {summary.get('count', 0)}件, "
            f"平均5日リターン {_format_percent(summary.get('average_return_5d'))}, "
            f"平均10日リターン {_format_percent(summary.get('average_return_10d'))}"
        )
    return lines


def _record_lines(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return ["- データなし"]
    return [
        (
            f"- {record.get('date')} {record.get('code')} {record.get('name')}: "
            f"score {_format_number(record.get('total_score'))}, "
            f"5d {_format_percent(record.get('return_5d'))}, "
            f"10d {_format_percent(record.get('return_10d'))}, "
            f"reason {record.get('rejected_reason') or 'N/A'}"
        )
        for record in records
    ]


def _rows(connection: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, params)]


def _valid_numbers(values: Any) -> list[float]:
    result = []
    for value in values:
        number = _number(value)
        if number is not None:
            result.append(number)
    return result


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _format_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2%}"


def _format_number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}"


def _profile_id(config: dict[str, Any]) -> str:
    return str(config.get("profile_id") or config.get("dealer", {}).get("id") or "rookie_dealer_01")


def _profile_name(config: dict[str, Any]) -> str:
    return str(config.get("profile_name") or config.get("dealer", {}).get("name") or _profile_id(config))
