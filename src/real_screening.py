"""Screening real J-Quants indicator data into candidate stocks."""

from __future__ import annotations

from typing import Any

from market_sections import market_section_counts, market_section_from_row


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


def screen_candidates(
    indicators: list[dict[str, Any]],
    target_count: int = 50,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strict_passed = []
    excluded_reasons: dict[str, int] = {}
    for item in indicators:
        reasons = _exclude_reasons(item, _conditions_for_item(item, STRICT_CONDITIONS, config))
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
            reasons = _exclude_reasons(item, _conditions_for_item(item, FALLBACK_CONDITIONS, config))
            if not reasons:
                fallback_candidates.append(_candidate(item, fallback=True, pass_reason="fallback条件を通過"))
        fallback_passed_count = len(fallback_candidates)
        candidates.extend(_rank_candidates(fallback_candidates)[: target_count - len(candidates)])

    return {
        "conditions": {
            "strict": STRICT_CONDITIONS,
            "fallback": FALLBACK_CONDITIONS,
            "market_overrides": _screening_market_overrides(config),
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


def screening_market_rejection_audit(
    indicators: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    target_count: int = 50,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize screening-stage drops without changing candidate selection."""
    candidate_codes = {str(item.get("code")) for item in candidates if isinstance(item, dict)}
    strict_passed = []
    fallback_passed = []
    reason_by_market: dict[str, dict[str, int]] = _empty_market_reason_counts()
    date_by_market: dict[str, dict[str, int]] = _empty_market_reason_counts()
    sample: list[dict[str, Any]] = []

    for item in indicators:
        if not isinstance(item, dict):
            continue
        strict_conditions = _conditions_for_item(item, STRICT_CONDITIONS, config)
        strict_reasons = _safe_exclude_reasons(item, strict_conditions)
        if not strict_reasons:
            strict_passed.append(_candidate(item, fallback=False, pass_reason="strict条件を通過"))
            continue
        fallback_conditions = _conditions_for_item(item, FALLBACK_CONDITIONS, config)
        fallback_reasons = _safe_exclude_reasons(item, fallback_conditions)
        if not fallback_reasons:
            fallback_passed.append(_candidate(item, fallback=True, pass_reason="fallback条件を通過"))
        if str(item.get("code")) in candidate_codes:
            continue
        market = _market_label(item)
        date_text = str(item.get("date") or "")
        reason_keys = _screening_reason_keys(strict_reasons)
        for reason in reason_keys:
            reason_by_market[market][reason] = int(reason_by_market[market].get(reason, 0) or 0) + 1
        if date_text:
            date_by_market[market][date_text] = int(date_by_market[market].get(date_text, 0) or 0) + 1
        if market in {"Standard", "Growth"} and len(sample) < 50:
            sample.append(
                {
                    "date": item.get("date"),
                    "code": item.get("code"),
                    "name": item.get("name"),
                    "market_section": market_section_from_row(item),
                    "filter_result": "market_filter_allowed_but_screening_excluded",
                    "reject_reason": ";".join(reason_keys),
                }
            )

    strict_ranked = _rank_candidates(strict_passed)
    fallback_ranked = _rank_candidates(fallback_passed)
    ranked_codes = {str(item.get("code")) for item in strict_ranked[:target_count]}
    if len(ranked_codes) < target_count:
        ranked_codes.update(str(item.get("code")) for item in fallback_ranked[: target_count - len(ranked_codes)])
    ranking_drop_by_market = {market: 0 for market in _market_labels()}
    for item in [*strict_ranked, *fallback_ranked]:
        code = str(item.get("code"))
        if code in candidate_codes or code in ranked_codes:
            continue
        market = _market_label(item)
        ranking_drop_by_market[market] = int(ranking_drop_by_market.get(market, 0) or 0) + 1
        reason_by_market[market]["ranking_drop"] = int(reason_by_market[market].get("ranking_drop", 0) or 0) + 1
        date_text = str(item.get("date") or "")
        if date_text:
            date_by_market[market][date_text] = int(date_by_market[market].get(date_text, 0) or 0) + 1
        if market in {"Standard", "Growth"} and len(sample) < 50:
            sample.append(
                {
                    "date": item.get("date"),
                    "code": item.get("code"),
                    "name": item.get("name"),
                    "market_section": market_section_from_row(item),
                    "filter_result": "market_filter_allowed_but_screening_excluded",
                    "reject_reason": "ranking_drop",
                }
            )

    return {
        "input_count_by_market": market_section_counts(indicators),
        "screening_candidate_count_by_market": market_section_counts(candidates),
        "screening_excluded_reason_by_market": reason_by_market,
        "screening_excluded_date_by_market": date_by_market,
        "screening_ranking_drop_by_market": ranking_drop_by_market,
        "representative_sample": sample,
        "market_overrides": _screening_market_overrides(config),
    }


def _empty_market_reason_counts() -> dict[str, dict[str, int]]:
    return {market: {} for market in _market_labels()}


def _market_labels() -> list[str]:
    return ["Prime", "Standard", "Growth", "Unknown"]


def _market_label(item: dict[str, Any]) -> str:
    section = market_section_from_row(item)
    return {
        "TSEPrime": "Prime",
        "TSEStandard": "Standard",
        "TSEGrowth": "Growth",
    }.get(section, "Unknown")


def _safe_exclude_reasons(item: dict[str, Any], conditions: dict[str, float]) -> list[str]:
    try:
        return _exclude_reasons(item, conditions)
    except (KeyError, TypeError, ValueError):
        return ["missing_required_price_or_indicator"]


def _conditions_for_item(
    item: dict[str, Any],
    base_conditions: dict[str, Any],
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    conditions = dict(base_conditions)
    overrides = _market_override_for_item(item, config)
    if not overrides:
        return conditions
    for key, value in overrides.items():
        if key in {"strict", "fallback"} and isinstance(value, dict):
            continue
        conditions[key] = value
    stage_key = "fallback" if base_conditions is FALLBACK_CONDITIONS else "strict"
    stage_overrides = overrides.get(stage_key)
    if isinstance(stage_overrides, dict):
        conditions.update(stage_overrides)
    return conditions


def _market_override_for_item(item: dict[str, Any], config: dict[str, Any] | None) -> dict[str, Any]:
    overrides = _screening_market_overrides(config)
    if not overrides:
        return {}
    section = market_section_from_row(item)
    market = _market_label(item)
    for key in [section, market]:
        value = overrides.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _screening_market_overrides(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    screening = config.get("screening", {})
    if not isinstance(screening, dict):
        return {}
    overrides = screening.get("market_overrides", {})
    return overrides if isinstance(overrides, dict) else {}


def _screening_reason_keys(reasons: list[str]) -> list[str]:
    keys = [_screening_reason_key(reason) for reason in reasons]
    return keys or ["unknown"]


def _screening_reason_key(reason: str) -> str:
    mapping = {
        "売買代金不足": "trading_value_low",
        "出来高前日比不足": "volume_ratio_low",
        "終値が5日移動平均以下": "close_below_ma5",
        "5日移動平均が25日移動平均以下": "ma5_below_ma25",
        "RSI範囲外": "rsi_out_of_range",
        "直近5営業日の値動きが大きすぎる": "volatility_too_high",
        "missing_required_price_or_indicator": "missing_required_price_or_indicator",
    }
    return mapping.get(str(reason), "unknown")


def _bool_condition(conditions: dict[str, Any], key: str, default: bool = True) -> bool:
    value = conditions.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)


def _candidate(item: dict[str, Any], fallback: bool, pass_reason: str) -> dict[str, Any]:
    spread = (float(item["ma5"]) - float(item["ma25"])) / float(item["ma25"])
    return {
        "code": item["code"],
        "name": item["name"],
        "sector_name": item.get("sector_name", ""),
        "section": item.get("section") or "Unknown",
        "market_section": item.get("market_section") or item.get("section") or "Unknown",
        "listing_market": item.get("listing_market") or item.get("section") or "Unknown",
        "date": item["date"],
        "open": item.get("open"),
        "high": item.get("high"),
        "low": item.get("low"),
        "close": item["close"],
        "volume": item["volume"],
        "adjusted_open": item.get("adjusted_open"),
        "adjusted_high": item.get("adjusted_high"),
        "adjusted_low": item.get("adjusted_low"),
        "adjusted_close": item.get("adjusted_close"),
        "adjusted_volume": item.get("adjusted_volume"),
        "adjusted_price_usage": item.get("adjusted_price_usage"),
        "limit_up_flag": item.get("limit_up_flag"),
        "limit_down_flag": item.get("limit_down_flag"),
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
        "direct_turnover_value": item.get("direct_turnover_value"),
        "direct_turnover_value_source": item.get("direct_turnover_value_source"),
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
        "sector17_code": item.get("sector17_code"),
        "sector17_name": item.get("sector17_name"),
        "sector33_code": item.get("sector33_code"),
        "sector33_name": item.get("sector33_name"),
        "scale_category": item.get("scale_category"),
        "margin_type": item.get("margin_type"),
        "product_category": item.get("product_category"),
        "ma_spread": round(spread, 4),
        "fallback": fallback,
        "pass_reason": pass_reason,
    }


def _exclude_reasons(item: dict[str, Any], conditions: dict[str, Any]) -> list[str]:
    reasons = []
    if float(item["turnover_value"]) < conditions["min_turnover_value"]:
        reasons.append("売買代金不足")
    if item["volume_ratio"] is None or float(item["volume_ratio"]) < conditions["min_volume_ratio"]:
        reasons.append("出来高前日比不足")
    if _bool_condition(conditions, "require_close_above_ma5", True) and float(item["close"]) <= float(item["ma5"]):
        reasons.append("終値が5日移動平均以下")
    if _bool_condition(conditions, "require_ma5_above_ma25", True) and float(item["ma5"]) <= float(item["ma25"]):
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
