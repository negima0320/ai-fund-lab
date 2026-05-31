"""J-Quants contract plan capability definitions."""

from __future__ import annotations

from typing import Any


VALID_JQUANTS_PLANS = {"free", "light"}

JQUANTS_PLAN_CAPABILITIES: dict[str, set[str]] = {
    "free": {
        "listed_info",
        "prices",
        "financial_statements",
        "earnings_calendar",
        "trading_calendar",
    },
    "light": {
        "listed_info",
        "prices",
        "financial_statements",
        "earnings_calendar",
        "trading_calendar",
        "topix_prices",
        "investor_breakdown",
        "investor_types",
    },
}

DISPLAY_CAPABILITIES = [
    "listed_info",
    "prices",
    "financial_statements",
    "earnings_calendar",
    "trading_calendar",
    "topix_prices",
    "investor_breakdown",
    "investor_types",
]

PROFILE_REQUIRED_CAPABILITIES: dict[str, set[str]] = {
    "rookie_dealer_02_v2_1": {"listed_info", "prices"},
    "rookie_dealer_02_v2_6": {"listed_info", "prices", "topix_prices"},
    "rookie_dealer_02_v2_8": {"listed_info", "prices", "investor_types"},
    "rookie_dealer_02_v2_9": {"listed_info", "prices", "financial_statements"},
    "rookie_dealer_02_v2_10": {"listed_info", "prices", "earnings_calendar"},
}

FALLBACKABLE_PROFILE_CAPABILITIES: dict[str, dict[str, str]] = {
    "rookie_dealer_02_v2_6": {
        "topix_prices": "fallback to prime market average or candidate median benchmark",
    },
    "rookie_dealer_02_v2_8": {
        "investor_types": "disable investor_context_score and continue with score 0",
    },
}


def normalize_jquants_plan(value: Any) -> str:
    plan = str(value or "free").strip().lower()
    if plan not in VALID_JQUANTS_PLANS:
        return "free"
    return plan


def normalize_profile_id(value: Any) -> str:
    profile_id = str(value or "").strip()
    aliases = {
        "rookie_dealer_02_v2.1": "rookie_dealer_02_v2_1",
        "rookie_dealer_02_v2.6": "rookie_dealer_02_v2_6",
        "rookie_dealer_02_v2.8": "rookie_dealer_02_v2_8",
        "rookie_dealer_02_v2.9": "rookie_dealer_02_v2_9",
        "rookie_dealer_02_v2.10": "rookie_dealer_02_v2_10",
    }
    return aliases.get(profile_id, profile_id)


def jquants_capabilities(plan: Any) -> set[str]:
    return set(JQUANTS_PLAN_CAPABILITIES[normalize_jquants_plan(plan)])


def jquants_has_capability(plan: Any, capability: str) -> bool:
    return capability in jquants_capabilities(plan)


def jquants_capability_status(plan: Any) -> dict[str, str]:
    capabilities = jquants_capabilities(plan)
    return {
        capability: "OK" if capability in capabilities else "disabled"
        for capability in DISPLAY_CAPABILITIES
    }


def profile_required_capabilities(profile_id: Any) -> set[str]:
    return set(PROFILE_REQUIRED_CAPABILITIES.get(normalize_profile_id(profile_id), {"listed_info", "prices"}))


def jquants_profile_compatibility(profile_id: Any, plan: Any) -> dict[str, Any]:
    normalized_profile_id = normalize_profile_id(profile_id)
    normalized_plan = normalize_jquants_plan(plan)
    required = profile_required_capabilities(normalized_profile_id)
    current = jquants_capabilities(normalized_plan)
    missing = sorted(required - current)
    fallback_rules = FALLBACKABLE_PROFILE_CAPABILITIES.get(normalized_profile_id, {})
    fallback_applied = [
        {
            "capability": capability,
            "policy": fallback_rules[capability],
        }
        for capability in missing
        if capability in fallback_rules
    ]
    unresolved_missing = [capability for capability in missing if capability not in fallback_rules]
    can_run = not unresolved_missing
    return {
        "profile_id": normalized_profile_id,
        "plan": normalized_plan,
        "profile_required_capabilities": sorted(required),
        "current_plan_capabilities": sorted(current),
        "missing_capabilities": missing,
        "fallback_applied": fallback_applied,
        "unresolved_missing_capabilities": unresolved_missing,
        "can_run_backtest": can_run,
        "can_run_live": can_run,
        "can_run_paper": can_run,
    }
