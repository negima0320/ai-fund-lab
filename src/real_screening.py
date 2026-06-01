"""Screening real J-Quants indicator data into candidate stocks."""

from __future__ import annotations

from typing import Any


STRICT_CONDITIONS = {
    "min_turnover_value": 500_000_000,
    "min_volume_ratio": 1.5,
    "rsi_min": 40,
    "rsi_max": 70,
    "max_five_day_volatility": 0.12,
}

FALLBACK_CONDITIONS = {
    "min_turnover_value": 300_000_000,
    "min_volume_ratio": 1.2,
    "rsi_min": 35,
    "rsi_max": 75,
    "max_five_day_volatility": 0.16,
}


def screen_candidates(indicators: list[dict[str, Any]], target_count: int = 50) -> dict[str, Any]:
    strict_passed = []
    excluded_reasons: dict[str, int] = {}
    for item in indicators:
        reasons = _exclude_reasons(item, STRICT_CONDITIONS)
        if not reasons:
            strict_passed.append(_candidate(item, fallback=False, pass_reason="strict条件を通過"))
        else:
            _count_reasons(excluded_reasons, reasons)

    candidates = _rank_candidates(strict_passed)[:target_count]
    fallback_used = len(candidates) < target_count
    fallback_passed_count = 0

    if fallback_used:
        existing_codes = {item["code"] for item in candidates}
        fallback_candidates = []
        for item in indicators:
            if item["code"] in existing_codes:
                continue
            reasons = _exclude_reasons(item, FALLBACK_CONDITIONS)
            if not reasons:
                fallback_candidates.append(_candidate(item, fallback=True, pass_reason="fallback条件を通過"))
        fallback_passed_count = len(fallback_candidates)
        candidates.extend(_rank_candidates(fallback_candidates)[: target_count - len(candidates)])

    return {
        "conditions": {
            "strict": STRICT_CONDITIONS,
            "fallback": FALLBACK_CONDITIONS,
            "ranking": [
                "volume_ratioが高い",
                "turnover_valueが大きい",
                "ma5とma25の乖離が適度",
                "RSIが過熱しすぎていない",
            ],
        },
        "strict_passed_count": len(strict_passed),
        "fallback_used": fallback_used,
        "fallback_passed_count": fallback_passed_count,
        "candidates": candidates,
        "excluded_summary": excluded_reasons,
    }


def _candidate(item: dict[str, Any], fallback: bool, pass_reason: str) -> dict[str, Any]:
    spread = (float(item["ma5"]) - float(item["ma25"])) / float(item["ma25"])
    return {
        "code": item["code"],
        "name": item["name"],
        "sector_name": item.get("sector_name", ""),
        "section": item.get("section", "Unknown"),
        "market_section": item.get("market_section", item.get("section", "Unknown")),
        "listing_market": item.get("listing_market", item.get("section", "Unknown")),
        "date": item["date"],
        "open": item.get("open"),
        "high": item.get("high"),
        "low": item.get("low"),
        "close": item["close"],
        "volume": item["volume"],
        "ma5": item["ma5"],
        "ma25": item["ma25"],
        "previous_close": item.get("previous_close"),
        "previous_ma5": item.get("previous_ma5"),
        "previous_ma25": item.get("previous_ma25"),
        "rsi": item["rsi"],
        "macd": item.get("macd"),
        "macd_signal": item.get("macd_signal"),
        "macd_hist": item.get("macd_hist"),
        "bb_upper": item.get("bb_upper"),
        "bb_middle": item.get("bb_middle"),
        "bb_lower": item.get("bb_lower"),
        "bb_position": item.get("bb_position"),
        "atr": item.get("atr"),
        "volume_ratio": item["volume_ratio"],
        "turnover_value": item["turnover_value"],
        "five_day_volatility": item["five_day_volatility"],
        "five_day_change_rate": item.get("five_day_change_rate"),
        "stock_return_5d": item.get("stock_return_5d"),
        "stock_return_10d": item.get("stock_return_10d"),
        "stock_return_20d": item.get("stock_return_20d"),
        "benchmark_source": item.get("benchmark_source"),
        "benchmark_return_5d": item.get("benchmark_return_5d"),
        "benchmark_return_10d": item.get("benchmark_return_10d"),
        "benchmark_return_20d": item.get("benchmark_return_20d"),
        "relative_strength_5d": item.get("relative_strength_5d"),
        "relative_strength_10d": item.get("relative_strength_10d"),
        "relative_strength_20d": item.get("relative_strength_20d"),
        "relative_strength_score": item.get("relative_strength_score"),
        "topix_records_loaded": item.get("topix_records_loaded"),
        "topix_api_calls": item.get("topix_api_calls"),
        "topix_cache_path": item.get("topix_cache_path"),
        "relative_strength_feature_enabled": item.get("relative_strength_feature_enabled"),
        "relative_strength_scoring_enabled": item.get("relative_strength_scoring_enabled"),
        "relative_strength_benchmark_provider_called": item.get("relative_strength_benchmark_provider_called"),
        "relative_strength_cache_exists": item.get("relative_strength_cache_exists"),
        "relative_strength_calculated": item.get("relative_strength_calculated"),
        "candle_type": item.get("candle_type"),
        "candle_body_rate": item.get("candle_body_rate"),
        "upper_shadow_rate": item.get("upper_shadow_rate"),
        "lower_shadow_rate": item.get("lower_shadow_rate"),
        "close_position_in_range": item.get("close_position_in_range"),
        "gap_rate": item.get("gap_rate"),
        "candlestick_signals": item.get("candlestick_signals", []),
        "candlestick_score": item.get("candlestick_score"),
        "trend_score": item.get("trend_score"),
        "volume_score": item.get("volume_score"),
        "rsi_score": item.get("rsi_score"),
        "sector_momentum_score": item.get("sector_momentum_score"),
        "sector_rank": item.get("sector_rank"),
        "sector_comment": item.get("sector_comment", ""),
        "ma_spread": round(spread, 4),
        "fallback": fallback,
        "pass_reason": pass_reason,
    }


def _exclude_reasons(item: dict[str, Any], conditions: dict[str, float]) -> list[str]:
    reasons = []
    if float(item["turnover_value"]) < conditions["min_turnover_value"]:
        reasons.append("売買代金不足")
    if item["volume_ratio"] is None or float(item["volume_ratio"]) < conditions["min_volume_ratio"]:
        reasons.append("出来高前日比不足")
    if float(item["close"]) <= float(item["ma5"]):
        reasons.append("終値が5日移動平均以下")
    if float(item["ma5"]) <= float(item["ma25"]):
        reasons.append("5日移動平均が25日移動平均以下")
    if not (conditions["rsi_min"] <= float(item["rsi"]) <= conditions["rsi_max"]):
        reasons.append("RSI範囲外")
    if float(item["five_day_volatility"]) > conditions["max_five_day_volatility"]:
        reasons.append("直近5営業日の値動きが大きすぎる")
    return reasons


def _rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            float(item["volume_ratio"] or 0),
            float(item["turnover_value"]),
            float(item.get("sector_momentum_score") or 50),
            -abs(float(item["ma_spread"]) - 0.03),
            -max(float(item["rsi"]) - 65, 0),
        ),
        reverse=True,
    )


def _count_reasons(summary: dict[str, int], reasons: list[str]) -> None:
    for reason in reasons:
        summary[reason] = summary.get(reason, 0) + 1
