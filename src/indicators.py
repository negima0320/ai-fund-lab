"""Technical indicator calculation for J-Quants daily prices."""

from __future__ import annotations

from typing import Any, Optional

from technical_indicators import TechnicalIndicatorDependencyError, calculate_indicators_with_pandas_ta


def calculate_indicators(
    price_rows: list[dict[str, Any]],
    stock_names: dict[str, str],
    target_date: str,
    stock_sectors: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    stock_sectors = stock_sectors or {}
    by_code: dict[str, list[dict[str, Any]]] = {}
    for row in price_rows:
        if row.get("close") is None or row.get("volume") is None:
            continue
        by_code.setdefault(row["code"], []).append(row)

    indicators = []
    excluded_count = 0
    for code, rows in by_code.items():
        rows.sort(key=lambda item: item["date"])
        target_index = _find_target_index(rows, target_date)
        if target_index is None:
            continue

        history = rows[: target_index + 1]
        if len(history) < 35:
            excluded_count += 1
            continue

        calculated = calculate_indicators_with_pandas_ta(_to_dataframe(history))
        target = calculated.iloc[-1].to_dict()
        previous = calculated.iloc[-2].to_dict()
        base = {
            "code": code,
            "name": stock_names.get(code, ""),
            "sector_name": stock_sectors.get(code, ""),
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


def _round_optional(value: Any, digits: int) -> Any:
    if value is None:
        return None
    try:
        if value != value:
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
