"""Technical indicator calculation for J-Quants daily prices."""

from __future__ import annotations

from typing import Any, Callable, Optional

from benchmark_provider import average_market_returns
from candlestick import calculate_candlestick_indicators, detect_candlestick_signals
from technical_indicators import TechnicalIndicatorDependencyError, calculate_indicators_with_pandas_ta


_BENCHMARK_RETURN_CACHE: dict[tuple[str, int, tuple[tuple[Any, ...], ...]], dict[int, float | None]] = {}


def calculate_indicators(
    price_rows: list[dict[str, Any]],
    stock_names: dict[str, str],
    target_date: str,
    stock_sectors: dict[str, str] | None = None,
    stock_sections: dict[str, str] | None = None,
    indicator_mode: str = "full",
    progress_callback: Callable[[int, int, str], None] | None = None,
    enable_relative_strength: bool = False,
    benchmark_returns: dict[int, float | None] | None = None,
    benchmark_source: str = "prime_average",
) -> tuple[list[dict[str, Any]], int]:
    indicator_mode = indicator_mode if indicator_mode in {"full", "fast", "minimal"} else "full"
    stock_sectors = stock_sectors or {}
    stock_sections = stock_sections or {}
    by_code: dict[str, list[dict[str, Any]]] = {}
    for row in price_rows:
        if row.get("close") is None or row.get("volume") is None:
            continue
        by_code.setdefault(row["code"], []).append(row)

    indicators = []
    excluded_count = 0
    total_codes = len(by_code)
    if enable_relative_strength:
        benchmark_returns = benchmark_returns if benchmark_returns is not None else _cached_benchmark_returns(by_code, target_date)
        if not benchmark_source:
            benchmark_source = "prime_average"
    else:
        benchmark_returns = {}
        benchmark_source = "unavailable"
    for index, (code, rows) in enumerate(by_code.items(), start=1):
        if progress_callback and (index == 1 or index % 100 == 0 or index == total_codes):
            progress_callback(index, total_codes, code)
        rows.sort(key=lambda item: item["date"])
        target_index = _find_target_index(rows, target_date)
        if target_index is None:
            continue

        history = rows[: target_index + 1]
        required_history = _required_history_length(indicator_mode)
        if len(history) < required_history:
            excluded_count += 1
            continue

        try:
            if indicator_mode == "full":
                calculated = calculate_indicators_with_pandas_ta(_to_dataframe(history))
                target = calculated.iloc[-1].to_dict()
                previous = calculated.iloc[-2].to_dict()
            else:
                target, previous = _calculate_lightweight_target(history, indicator_mode)
        except Exception as exc:
            raise RuntimeError(f"Indicator calculation failed for code={code}") from exc

        base = {
            "code": code,
            "name": stock_names.get(code, ""),
            "sector_name": stock_sectors.get(code, ""),
            "section": stock_sections.get(code, "Unknown"),
            "market_section": stock_sections.get(code, "Unknown"),
            "listing_market": stock_sections.get(code, "Unknown"),
            "date": target["date"],
            "open": target.get("open"),
            "high": target.get("high"),
            "low": target.get("low"),
            "close": target["close"],
            "volume": target["volume"],
            "ma5": _round_optional(target.get("ma5"), 2),
            "ma25": _round_optional(target.get("ma25"), 2),
            "previous_close": _round_optional(previous.get("close"), 2),
            "previous_ma5": _round_optional(previous.get("ma5"), 2),
            "previous_ma25": _round_optional(previous.get("ma25"), 2),
            "rsi": _round_optional(target.get("rsi"), 2),
            "macd": _round_optional(target.get("macd"), 4),
            "macd_signal": _round_optional(target.get("macd_signal"), 4),
            "macd_hist": _round_optional(target.get("macd_hist"), 4),
            "bb_upper": _round_optional(target.get("bb_upper"), 2),
            "bb_middle": _round_optional(target.get("bb_middle"), 2),
            "bb_lower": _round_optional(target.get("bb_lower"), 2),
            "bb_position": _round_optional(target.get("bb_position"), 4),
            "atr": _round_optional(target.get("atr"), 4),
            "volume_ratio": _round_optional(target.get("volume_ratio"), 4),
            "turnover_value": round(float(target["close"]) * float(target["volume"]), 2),
            "five_day_volatility": _round_optional(target.get("five_day_volatility"), 4),
            "five_day_change_rate": _round_optional(target.get("five_day_change_rate"), 4),
            **(_relative_strength_fields(history, benchmark_returns, benchmark_source) if enable_relative_strength else _empty_relative_strength_fields()),
            "candlestick_score": None,
            "trend_score": None,
            "volume_score": None,
            "rsi_score": None,
            "candle_type": target.get("candle_type"),
            "candle_body_rate": _round_optional(target.get("candle_body_rate"), 4),
            "upper_shadow_rate": _round_optional(target.get("upper_shadow_rate"), 4),
            "lower_shadow_rate": _round_optional(target.get("lower_shadow_rate"), 4),
            "close_position_in_range": _round_optional(target.get("close_position_in_range"), 4),
            "gap_rate": _round_optional(target.get("gap_rate"), 4),
            "candlestick_signals": target.get("candlestick_signals", []),
        }

        indicators.append(base)

    indicators.sort(key=lambda item: item["turnover_value"], reverse=True)
    return indicators, excluded_count


def _required_history_length(indicator_mode: str) -> int:
    if indicator_mode == "minimal":
        return 25
    if indicator_mode == "fast":
        return 25
    return 35


def _find_target_index(rows: list[dict[str, Any]], target_date: str) -> Optional[int]:
    for index, row in enumerate(rows):
        if row["date"] == target_date:
            return index
    return None


def _to_dataframe(rows: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise TechnicalIndicatorDependencyError(
            "pandas がインストールされていません。`pip install -r requirements.txt` を実行してください。"
        ) from exc

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _calculate_lightweight_target(history: list[dict[str, Any]], indicator_mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [_normalize_numeric_row(row) for row in history]
    enriched = []
    for index, row in enumerate(rows):
        previous = enriched[index - 1] if index > 0 else None
        closes = [item["close"] for item in rows[: index + 1]]
        volumes = [item["volume"] for item in rows[: index + 1]]
        ma5 = _sma(closes, 5)
        ma25 = _sma(closes, 25)
        result = {
            **row,
            "ma5": ma5,
            "ma25": ma25,
            "rsi": _rsi(closes, 14),
            "volume_ratio": (volumes[-1] / volumes[-2]) if len(volumes) >= 2 and volumes[-2] else None,
            "macd": None,
            "macd_signal": None,
            "macd_hist": None,
            "bb_upper": None,
            "bb_middle": None,
            "bb_lower": None,
            "bb_position": None,
            "atr": None,
            "five_day_volatility": None,
            "five_day_change_rate": None,
        }
        if indicator_mode == "fast":
            result["five_day_volatility"] = _five_day_volatility(closes)
            result["five_day_change_rate"] = _five_day_change_rate(closes)
            candle = calculate_candlestick_indicators(row, previous)
            result.update(candle)
            result["candlestick_signals"] = detect_candlestick_signals(result)
        else:
            result.update(
                {
                    "candle_type": None,
                    "candle_body_rate": None,
                    "upper_shadow_rate": None,
                    "lower_shadow_rate": None,
                    "close_position_in_range": None,
                    "gap_rate": None,
                    "candlestick_signals": [],
                }
            )
        enriched.append(result)
    return enriched[-1], enriched[-2]


def _normalize_numeric_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "open": _float_or_none(row.get("open")),
        "high": _float_or_none(row.get("high")),
        "low": _float_or_none(row.get("low")),
        "close": _float_or_none(row.get("close")),
        "volume": _float_or_none(row.get("volume")),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sma(values: list[float | None], length: int) -> float | None:
    valid = [value for value in values[-length:] if value is not None]
    if len(valid) < length:
        return None
    return sum(valid) / length


def _rsi(values: list[float | None], length: int = 14) -> float | None:
    valid = [value for value in values if value is not None]
    if len(valid) <= length:
        return None
    changes = [valid[index] - valid[index - 1] for index in range(1, len(valid))]
    recent = changes[-length:]
    gains = [change for change in recent if change > 0]
    losses = [-change for change in recent if change < 0]
    average_gain = sum(gains) / length
    average_loss = sum(losses) / length
    if average_loss == 0:
        return 100.0
    rs = average_gain / average_loss
    return 100 - (100 / (1 + rs))


def _five_day_volatility(values: list[float | None]) -> float | None:
    valid = [value for value in values[-5:] if value is not None]
    latest = values[-1]
    if len(valid) < 5 or not latest:
        return None
    return (max(valid) - min(valid)) / latest


def _five_day_change_rate(values: list[float | None]) -> float | None:
    if len(values) < 5 or values[-5] in {None, 0} or values[-1] is None:
        return None
    return (values[-1] - values[-5]) / values[-5]


def _benchmark_returns(by_code: dict[str, list[dict[str, Any]]], target_date: str) -> dict[int, float | None]:
    rows = [row for values in by_code.values() for row in values]
    return average_market_returns(rows, target_date)


def _cached_benchmark_returns(by_code: dict[str, list[dict[str, Any]]], target_date: str) -> dict[int, float | None]:
    key = _benchmark_cache_key(by_code, target_date)
    if key not in _BENCHMARK_RETURN_CACHE:
        _BENCHMARK_RETURN_CACHE[key] = _benchmark_returns(by_code, target_date)
    return _BENCHMARK_RETURN_CACHE[key]


def _benchmark_cache_key(
    by_code: dict[str, list[dict[str, Any]]],
    target_date: str,
) -> tuple[str, int, tuple[tuple[Any, ...], ...]]:
    close_points: list[tuple[Any, ...]] = []
    for code, rows in by_code.items():
        sorted_rows = sorted(rows, key=lambda item: item["date"])
        target_index = _find_target_index(sorted_rows, target_date)
        closes = []
        for horizon in [0, 5, 10, 20]:
            index = target_index - horizon if target_index is not None else None
            closes.append(_float_or_none(sorted_rows[index].get("close")) if index is not None and index >= 0 else None)
        close_points.append((str(code), *closes))
    return (target_date, len(by_code), tuple(sorted(close_points)))


def _empty_relative_strength_fields() -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for horizon in [5, 10, 20]:
        fields[f"stock_return_{horizon}d"] = None
        fields[f"benchmark_return_{horizon}d"] = None
        fields[f"relative_strength_{horizon}d"] = None
    fields["benchmark_source"] = "unavailable"
    fields["relative_strength_score"] = 0
    return fields


def _relative_strength_fields(
    history: list[dict[str, Any]],
    benchmark_returns: dict[int, float | None],
    benchmark_source: str,
) -> dict[str, Any]:
    normalized = [_normalize_numeric_row(row) for row in history]
    fields: dict[str, Any] = {}
    relative_strengths: dict[int, float | None] = {}
    for horizon in [5, 10, 20]:
        stock_return = _return_over_horizon(normalized, horizon)
        benchmark_return = benchmark_returns.get(horizon)
        relative_strength = None
        if stock_return is not None and benchmark_return is not None:
            relative_strength = stock_return - benchmark_return
        fields[f"stock_return_{horizon}d"] = _round_optional(stock_return, 4)
        fields[f"benchmark_return_{horizon}d"] = _round_optional(benchmark_return, 4)
        fields[f"relative_strength_{horizon}d"] = _round_optional(relative_strength, 4)
        relative_strengths[horizon] = relative_strength
    fields["benchmark_source"] = benchmark_source
    fields["relative_strength_score"] = _relative_strength_score(relative_strengths)
    return fields


def _return_over_horizon(rows: list[dict[str, Any]], horizon: int) -> float | None:
    if len(rows) <= horizon:
        return None
    latest = _float_or_none(rows[-1].get("close"))
    base = _float_or_none(rows[-horizon - 1].get("close"))
    if latest is None or base in {None, 0}:
        return None
    return (latest - base) / base


def _relative_strength_score(relative_strengths: dict[int, float | None]) -> int:
    score = 0
    if (relative_strengths.get(5) or 0) > 0.03:
        score += 3
    if (relative_strengths.get(10) or 0) > 0.05:
        score += 4
    if (relative_strengths.get(20) or 0) > 0.08:
        score += 3
    return max(0, min(10, score))


def _round_optional(value: Any, digits: int) -> Any:
    if value is None:
        return None
    try:
        if value != value:
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
