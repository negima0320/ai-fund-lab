"""Shared market regime helpers."""

from __future__ import annotations

from datetime import date
from typing import Any


REGIME_ORDER = ["strong_bull", "bull", "range", "bear", "strong_bear", "unknown"]


def _number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_market_regime(
    advance_ratio: Any = None,
    average_change_rate: Any = None,
    fallback_market_regime: Any = None,
) -> str:
    """Classify the market regime using entry-date market breadth fields."""

    advance = _number(advance_ratio)
    change = _number(average_change_rate)
    base = str(fallback_market_regime or "")
    if advance is None and change is None:
        return {"risk_on": "bull", "neutral": "range", "risk_off": "bear"}.get(base, "unknown")

    advance = advance if advance is not None else 0.5
    change = change if change is not None else 0.0
    if advance >= 0.70 and change >= 0.008:
        return "strong_bull"
    if advance >= 0.58 and change >= 0.002:
        return "bull"
    if advance <= 0.30 and change <= -0.008:
        return "strong_bear"
    if advance <= 0.42 and change <= -0.002:
        return "bear"
    return "range"


def classify_market_context(context: dict[str, Any] | None) -> str:
    context = context or {}
    return classify_market_regime(
        context.get("advance_ratio"),
        context.get("average_change_rate"),
        context.get("market_regime"),
    )


def effective_market_context_for_signal(
    signal_date: str,
    contexts_by_date: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return the latest context strictly before signal_date."""

    available_dates = sorted(day for day in contexts_by_date if day < signal_date)
    if not available_dates:
        return {
            "market_context": {},
            "source_date": "",
            "regime": "unknown",
            "lag_days": None,
            "fallback_used": True,
            "same_day_used": False,
        }
    source_date = available_dates[-1]
    context = contexts_by_date.get(source_date, {})
    lag_days = _date_lag_days(source_date, signal_date)
    return {
        "market_context": context,
        "source_date": source_date,
        "regime": classify_market_context(context),
        "lag_days": lag_days,
        "fallback_used": lag_days is None or lag_days > 1,
        "same_day_used": source_date >= signal_date,
    }


def dynamic_exposure_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("dynamic_exposure")
    return policy if isinstance(policy, dict) else {}


def dynamic_exposure_target(
    config: dict[str, Any],
    regime: str,
    default_target_exposure: Any,
) -> tuple[float | None, bool]:
    policy = dynamic_exposure_policy(config)
    if not bool(policy.get("enabled", False)):
        return _number(default_target_exposure), False
    targets = policy.get("target_exposure_by_regime")
    if not isinstance(targets, dict) or regime not in targets:
        return _number(default_target_exposure), False
    return _number(targets.get(regime)), True


def classification_definition() -> dict[str, Any]:
    return {
        "primary_source": "data/processed/market_context_YYYY-MM-DD.json",
        "fields_used": ["advance_ratio", "average_change_rate", "market_regime fallback"],
        "strong_bull": "advance_ratio >= 0.70 and average_change_rate >= 0.008",
        "bull": "advance_ratio >= 0.58 and average_change_rate >= 0.002",
        "strong_bear": "advance_ratio <= 0.30 and average_change_rate <= -0.008",
        "bear": "advance_ratio <= 0.42 and average_change_rate <= -0.002",
        "range": "all other market_context rows",
        "fallback": "market_context欠損時のみ risk_on→bull, neutral→range, risk_off→bear",
    }


def _date_lag_days(source_date: str, signal_date: str) -> int | None:
    try:
        return (date.fromisoformat(signal_date) - date.fromisoformat(source_date)).days
    except ValueError:
        return None
