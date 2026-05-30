from __future__ import annotations

from candlestick import calculate_candlestick_indicators, detect_candlestick_signals


def test_bullish_candlestick_metrics_and_signals() -> None:
    metrics = calculate_candlestick_indicators(
        {"open": 100, "high": 122.8, "low": 98, "close": 122},
        {"close": 99},
    )
    item = {
        **metrics,
        "close": 122,
        "ma5": 115,
        "ma25": 108,
        "previous_close": 99,
        "previous_ma5": 101,
        "volume_ratio": 2.2,
        "rsi": 60,
        "five_day_change_rate": 0.06,
    }

    signals = detect_candlestick_signals(item)

    assert metrics["candle_type"] == "bullish"
    assert metrics["close_position_in_range"] >= 0.9
    assert "bullish_candle" in signals
    assert "strong_bullish_candle" in signals
    assert "volume_confirmed_breakout" in signals


def test_overheated_upper_shadow_warning() -> None:
    metrics = calculate_candlestick_indicators(
        {"open": 110, "high": 135, "low": 108, "close": 116},
        {"close": 104},
    )
    item = {
        **metrics,
        "close": 116,
        "ma5": 112,
        "ma25": 106,
        "previous_close": 104,
        "previous_ma5": 105,
        "volume_ratio": 2.0,
        "rsi": 76,
        "five_day_change_rate": 0.10,
    }

    signals = detect_candlestick_signals(item)

    assert "long_upper_shadow_warning" in signals
    assert "overheated_warning" in signals
