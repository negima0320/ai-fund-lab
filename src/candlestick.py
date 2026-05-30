"""Candlestick metrics and short-term technical signals."""

from __future__ import annotations

from typing import Any


DEFAULT_SIGNAL_THRESHOLDS = {
    "doji_body_rate": 0.002,
    "large_body_rate": 0.018,
    "short_upper_shadow_rate": 0.008,
    "long_upper_shadow_rate": 0.025,
    "long_lower_shadow_rate": 0.025,
    "near_high_position": 0.70,
    "very_near_high_position": 0.80,
    "volume_breakout_ratio": 1.8,
    "overheated_rsi": 70.0,
    "rapid_rise_5d": 0.08,
}


def calculate_candlestick_indicators(
    target: dict[str, Any],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    open_price = _to_float(target.get("open"))
    high = _to_float(target.get("high"))
    low = _to_float(target.get("low"))
    close = _to_float(target.get("close"))
    previous_close = _to_float(previous.get("close")) if previous else None

    if open_price is None or high is None or low is None or close is None or close <= 0:
        return {
            "candle_type": "unknown",
            "candle_body_rate": None,
            "upper_shadow_rate": None,
            "lower_shadow_rate": None,
            "close_position_in_range": None,
            "gap_rate": None,
        }

    body = abs(close - open_price)
    upper_shadow = max(0.0, high - max(open_price, close))
    lower_shadow = max(0.0, min(open_price, close) - low)
    range_width = high - low
    body_rate = body / close

    if body_rate <= DEFAULT_SIGNAL_THRESHOLDS["doji_body_rate"]:
        candle_type = "doji"
    elif close > open_price:
        candle_type = "bullish"
    else:
        candle_type = "bearish"

    gap_rate = None
    if previous_close and previous_close > 0:
        gap_rate = round((open_price - previous_close) / previous_close, 4)

    return {
        "candle_type": candle_type,
        "candle_body_rate": round(body_rate, 4),
        "upper_shadow_rate": round(upper_shadow / close, 4),
        "lower_shadow_rate": round(lower_shadow / close, 4),
        "close_position_in_range": round((close - low) / range_width, 4) if range_width > 0 else 0.5,
        "gap_rate": gap_rate,
    }


def detect_candlestick_signals(
    item: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    thresholds = {**DEFAULT_SIGNAL_THRESHOLDS, **(thresholds or {})}
    candle_type = item.get("candle_type")
    body_rate = _to_float(item.get("candle_body_rate")) or 0.0
    upper_shadow_rate = _to_float(item.get("upper_shadow_rate")) or 0.0
    lower_shadow_rate = _to_float(item.get("lower_shadow_rate")) or 0.0
    close_position = _to_float(item.get("close_position_in_range"))
    close = _to_float(item.get("close"))
    ma5 = _to_float(item.get("ma5"))
    ma25 = _to_float(item.get("ma25"))
    previous_close = _to_float(item.get("previous_close"))
    previous_ma5 = _to_float(item.get("previous_ma5"))
    volume_ratio = _to_float(item.get("volume_ratio")) or 0.0
    rsi = _to_float(item.get("rsi")) or 0.0
    five_day_change_rate = _to_float(item.get("five_day_change_rate")) or 0.0

    signals = []
    near_high = close_position is not None and close_position >= thresholds["near_high_position"]
    very_near_high = close_position is not None and close_position >= thresholds["very_near_high_position"]
    long_upper = upper_shadow_rate >= thresholds["long_upper_shadow_rate"] and close_position is not None and close_position < 0.70
    long_lower = lower_shadow_rate >= thresholds["long_lower_shadow_rate"] and close_position is not None and close_position > 0.45

    if candle_type == "bullish" and near_high:
        signals.append("bullish_candle")
    if (
        candle_type == "bullish"
        and body_rate >= thresholds["large_body_rate"]
        and upper_shadow_rate <= thresholds["short_upper_shadow_rate"]
        and very_near_high
    ):
        signals.append("strong_bullish_candle")
    if long_upper:
        signals.append("long_upper_shadow_warning")
    if long_lower:
        signals.append("long_lower_shadow_support")
    if close is not None and ma5 is not None and close > ma5 and (previous_close is None or previous_ma5 is None or previous_close <= previous_ma5):
        signals.append("ma_reclaim")
    if close is not None and ma5 is not None and ma25 is not None and close > ma5 > ma25:
        signals.append("ma_trend_alignment")
    if volume_ratio >= thresholds["volume_breakout_ratio"] and "ma_trend_alignment" in signals and "bullish_candle" in signals:
        signals.append("volume_confirmed_breakout")
    if rsi >= thresholds["overheated_rsi"] and long_upper and five_day_change_rate >= thresholds["rapid_rise_5d"]:
        signals.append("overheated_warning")
    return signals


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
        if numeric != numeric:
            return None
        return numeric
    except (TypeError, ValueError):
        return None
