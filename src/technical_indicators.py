"""Technical indicator calculation backed by pandas-ta."""

from __future__ import annotations

from typing import Any

from candlestick import calculate_candlestick_indicators, detect_candlestick_signals


class TechnicalIndicatorDependencyError(RuntimeError):
    """Raised when pandas-ta or pandas is unavailable."""


def calculate_indicators_with_pandas_ta(df: Any) -> Any:
    _ensure_dependencies()
    result = df.copy()
    result["ma5"] = calculate_sma(result, 5)
    result["ma25"] = calculate_sma(result, 25)
    result["rsi"] = calculate_rsi(result, 14)

    macd = calculate_macd(result)
    result["macd"] = macd["macd"]
    result["macd_signal"] = macd["macd_signal"]
    result["macd_hist"] = macd["macd_hist"]

    bbands = calculate_bbands(result)
    result["bb_upper"] = bbands["bb_upper"]
    result["bb_middle"] = bbands["bb_middle"]
    result["bb_lower"] = bbands["bb_lower"]
    result["bb_position"] = _bb_position(result["close"], result["bb_lower"], result["bb_upper"])

    result["atr"] = calculate_atr(result)
    result["volume_ratio"] = result["volume"] / result["volume"].shift(1)
    result["five_day_volatility"] = (result["close"].rolling(5).max() - result["close"].rolling(5).min()) / result["close"]
    result["five_day_change_rate"] = (result["close"] - result["close"].shift(4)) / result["close"].shift(4)
    return calculate_candlestick_features(result)


def calculate_sma(df: Any, length: int) -> Any:
    ta = _pandas_ta()
    return ta.sma(df["close"], length=length)


def calculate_rsi(df: Any, length: int = 14) -> Any:
    ta = _pandas_ta()
    return ta.rsi(df["close"], length=length)


def calculate_macd(df: Any) -> dict[str, Any]:
    ta = _pandas_ta()
    macd = ta.macd(df["close"])
    if macd is None or macd.empty:
        return {"macd": None, "macd_signal": None, "macd_hist": None}
    return {
        "macd": _column_by_prefix(macd, "MACD_"),
        "macd_signal": _column_by_prefix(macd, "MACDs_"),
        "macd_hist": _column_by_prefix(macd, "MACDh_"),
    }


def calculate_bbands(df: Any) -> dict[str, Any]:
    ta = _pandas_ta()
    bbands = ta.bbands(df["close"], length=20, std=2)
    if bbands is None or bbands.empty:
        return {"bb_upper": None, "bb_middle": None, "bb_lower": None}
    return {
        "bb_upper": _column_by_prefix(bbands, "BBU_"),
        "bb_middle": _column_by_prefix(bbands, "BBM_"),
        "bb_lower": _column_by_prefix(bbands, "BBL_"),
    }


def calculate_atr(df: Any) -> Any:
    ta = _pandas_ta()
    return ta.atr(high=df["high"], low=df["low"], close=df["close"], length=14)


def calculate_candlestick_features(df: Any) -> Any:
    result = df.copy()
    features = []
    rows = result.to_dict("records")
    for index, row in enumerate(rows):
        previous = rows[index - 1] if index > 0 else None
        features.append(calculate_candlestick_indicators(row, previous))
    for key in [
        "candle_type",
        "candle_body_rate",
        "upper_shadow_rate",
        "lower_shadow_rate",
        "close_position_in_range",
        "gap_rate",
    ]:
        result[key] = [item.get(key) for item in features]

    signal_rows = result.to_dict("records")
    result["candlestick_signals"] = [detect_candlestick_signals(row) for row in signal_rows]
    return result


def _bb_position(close: Any, lower: Any, upper: Any) -> Any:
    if lower is None or upper is None:
        return None
    width = upper - lower
    return (close - lower) / width.where(width != 0)


def _column_by_prefix(df: Any, prefix: str) -> Any:
    for column in df.columns:
        if str(column).startswith(prefix):
            return df[column]
    return None


def _ensure_dependencies() -> None:
    _pandas()
    _pandas_ta()


def _pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise TechnicalIndicatorDependencyError(
            "pandas がインストールされていません。`pip install -r requirements.txt` を実行してください。"
        ) from exc
    return pd


def _pandas_ta() -> Any:
    try:
        import pandas_ta as ta
    except ModuleNotFoundError as exc:
        raise TechnicalIndicatorDependencyError(
            "pandas-ta がインストールされていません。`pip install -r requirements.txt` を実行してください。"
        ) from exc
    return ta
