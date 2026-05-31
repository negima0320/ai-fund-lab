"""Benchmark return provider for Relative Strength."""

from __future__ import annotations

from statistics import median
from typing import Any


WINDOWS = [5, 10, 20]


def build_relative_strength_benchmark(
    price_rows: list[dict[str, Any]],
    target_date: str,
    topix_prices: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    topix_returns = _returns_from_rows(topix_prices or [], target_date)
    if _has_any_return(topix_returns):
        return {"benchmark_source": "topix", "benchmark_returns": topix_returns}

    average_returns = average_market_returns(price_rows, target_date)
    if _has_any_return(average_returns):
        return {"benchmark_source": "prime_average", "benchmark_returns": average_returns}

    median_returns = median_candidate_returns(price_rows, target_date)
    if _has_any_return(median_returns):
        return {"benchmark_source": "candidate_median", "benchmark_returns": median_returns}

    return {"benchmark_source": "unavailable", "benchmark_returns": {horizon: None for horizon in WINDOWS}}


def average_market_returns(price_rows: list[dict[str, Any]], target_date: str) -> dict[int, float | None]:
    values_by_horizon = _candidate_return_values(price_rows, target_date)
    return {
        horizon: (sum(values) / len(values) if values else None)
        for horizon, values in values_by_horizon.items()
    }


def median_candidate_returns(price_rows: list[dict[str, Any]], target_date: str) -> dict[int, float | None]:
    values_by_horizon = _candidate_return_values(price_rows, target_date)
    return {
        horizon: (float(median(values)) if values else None)
        for horizon, values in values_by_horizon.items()
    }


def _candidate_return_values(price_rows: list[dict[str, Any]], target_date: str) -> dict[int, list[float]]:
    by_code: dict[str, list[dict[str, Any]]] = {}
    for row in price_rows:
        code = str(row.get("code") or row.get("Code") or row.get("LocalCode") or "")
        if code:
            by_code.setdefault(code, []).append(row)
    values_by_horizon: dict[int, list[float]] = {horizon: [] for horizon in WINDOWS}
    for rows in by_code.values():
        returns = _returns_from_rows(rows, target_date)
        for horizon, value in returns.items():
            if value is not None:
                values_by_horizon[horizon].append(value)
    return values_by_horizon


def _returns_from_rows(rows: list[dict[str, Any]], target_date: str) -> dict[int, float | None]:
    normalized = sorted((_normalize_row(row) for row in rows), key=lambda item: item["date"])
    target_index = next((index for index, row in enumerate(normalized) if row["date"] == target_date), None)
    if target_index is None:
        return {horizon: None for horizon in WINDOWS}
    history = normalized[: target_index + 1]
    return {horizon: _return_over_horizon(history, horizon) for horizon in WINDOWS}


def _return_over_horizon(rows: list[dict[str, Any]], horizon: int) -> float | None:
    if len(rows) <= horizon:
        return None
    latest = _float_or_none(rows[-1].get("close"))
    base = _float_or_none(rows[-horizon - 1].get("close"))
    if latest is None or base in {None, 0}:
        return None
    return (latest - base) / base


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": _format_date(str(row.get("date") or row.get("Date") or "")),
        "close": row.get("close") or row.get("Close") or row.get("C"),
    }


def _format_date(value: str) -> str:
    value = str(value or "").strip()
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_any_return(returns: dict[int, float | None]) -> bool:
    return any(value is not None for value in returns.values())
