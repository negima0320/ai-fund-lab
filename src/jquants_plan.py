"""J-Quants contract plan capability definitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

try:  # pragma: no cover - PyYAML is part of the supported runtime.
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


VALID_JQUANTS_PLANS = {"free", "light"}

JQUANTS_PLAN_CAPABILITIES: dict[str, set[str]] = {
    "free": {
        "listed_info",
        "prices",
        "financial_statements",
        "earnings_calendar",
    },
    "light": {
        "listed_info",
        "prices",
        "financial_statements",
        "earnings_calendar",
        "trading_calendar",
        "topix_prices",
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
    "investor_types",
]

DEFAULT_PLAN_SETTINGS = {
    "free": {"requests_per_minute": 5, "parallel_fetch": False},
    "light": {"requests_per_minute": 60, "parallel_fetch": True, "max_parallel_requests": 4},
}


@dataclass(frozen=True)
class PlanResolution:
    plan: str
    source: str
    config_path: str | None
    capabilities: dict[str, str]
    requests_per_minute: int
    parallel_fetch: bool
    max_parallel_requests: int
    supported_date_ranges: dict[str, str]
    config: dict[str, Any]
    warnings: list[str]

PROFILE_REQUIRED_CAPABILITIES: dict[str, set[str]] = {
    "rookie_dealer_02_v2_1": {"listed_info", "prices"},
    "rookie_dealer_02_v2_6": {"listed_info", "prices", "topix_prices"},
    "rookie_dealer_02_v2_9": {"listed_info", "prices", "financial_statements"},
    "rookie_dealer_02_v2_10": {"listed_info", "prices", "earnings_calendar"},
}

FALLBACKABLE_PROFILE_CAPABILITIES: dict[str, dict[str, str]] = {
    "rookie_dealer_02_v2_6": {
        "topix_prices": "fallback to prime market average or candidate median benchmark",
    },
}


def normalize_jquants_plan(value: Any) -> str:
    plan = str(value or "free").strip().lower()
    if plan not in VALID_JQUANTS_PLANS:
        return "free"
    return plan


def resolve_jquants_plan(
    args: Any | None = None,
    config: dict[str, Any] | None = None,
    config_root: Path | str | None = None,
    provider_config: dict[str, Any] | None = None,
) -> PlanResolution:
    root = Path(config_root) if config_root is not None else None
    jquants_path = root / "config" / "jquants.yaml" if root is not None else None
    provider_path = root / "config" / "provider.yaml" if root is not None else None
    jquants_file = _load_yaml_file(jquants_path) if jquants_path is not None else {}
    provider_file = _load_yaml_file(provider_path) if provider_path is not None else {}
    provider_payload = provider_config if provider_config is not None else provider_file
    explicit_config = config or {}
    cli_plan = getattr(args, "jquants_plan", None) if args is not None else None

    jquants_section = _section(jquants_file, "jquants")
    explicit_section = _section(explicit_config, "jquants")
    provider_section = _section(provider_payload, "jquants")
    warnings: list[str] = []
    provider_plan = provider_section.get("plan")
    jquants_file_plan = jquants_section.get("plan")
    explicit_plan = explicit_section.get("plan")

    if jquants_file_plan is not None and provider_plan is not None:
        warnings.append("deprecated config/provider.yaml jquants.plan is ignored; use config/jquants.yaml")

    if cli_plan not in (None, ""):
        raw_plan = cli_plan
        source = "cli"
        config_path = None
    elif jquants_file_plan not in (None, ""):
        raw_plan = jquants_file_plan
        source = "config/jquants.yaml"
        config_path = str(jquants_path) if jquants_path is not None else None
    elif explicit_plan not in (None, ""):
        raw_plan = explicit_plan
        source = "config"
        config_path = None
    elif provider_plan not in (None, ""):
        raw_plan = provider_plan
        source = "config/provider.yaml"
        config_path = str(provider_path) if provider_path is not None else None
        warnings.append("deprecated config/provider.yaml jquants.plan is used; move it to config/jquants.yaml")
    else:
        raw_plan = "free"
        source = "default"
        config_path = None

    plan = normalize_jquants_plan(raw_plan)
    if str(raw_plan or "").strip().lower() not in VALID_JQUANTS_PLANS:
        warnings.append(f"unknown J-Quants plan `{raw_plan}`; falling back to free")

    merged = {}
    merged.update(provider_section)
    merged.update(explicit_section)
    merged.update(jquants_section)
    merged["plan"] = plan
    plan_settings = _plan_settings(merged, plan)
    requests_per_minute = int(plan_settings.get("requests_per_minute", merged.get("requests_per_minute", merged.get("rate_limit_per_minute", 5))))
    parallel_fetch = bool(plan_settings.get("parallel_fetch", False)) and plan == "light"
    max_parallel_requests = int(plan_settings.get("max_parallel_requests", merged.get("max_parallel_requests", 4)))
    merged["requests_per_minute"] = requests_per_minute
    merged["rate_limit_per_minute"] = requests_per_minute
    merged["parallel_fetch"] = parallel_fetch
    merged["max_parallel_requests"] = max_parallel_requests
    merged["capability_status"] = jquants_capability_status(plan)
    return PlanResolution(
        plan=plan,
        source=source,
        config_path=config_path,
        capabilities=jquants_capability_status(plan),
        requests_per_minute=requests_per_minute,
        parallel_fetch=parallel_fetch,
        max_parallel_requests=max_parallel_requests,
        supported_date_ranges=jquants_supported_date_ranges({"jquants": merged}),
        config=merged,
        warnings=warnings,
    )


def jquants_earliest_supported_date(config: dict[str, Any], endpoint: str = "prices") -> date | None:
    jquants = config.get("jquants", {}) if isinstance(config.get("jquants"), dict) else {}
    plan = normalize_jquants_plan(jquants.get("plan", "free"))
    payload = jquants.get("earliest_supported_date", {})
    value = None
    if isinstance(payload, dict):
        endpoint_payload = payload.get(endpoint)
        if isinstance(endpoint_payload, dict):
            value = endpoint_payload.get(plan)
        if value is None:
            value = payload.get(plan)
    elif isinstance(payload, str):
        value = payload
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def jquants_supported_date_ranges(config: dict[str, Any]) -> dict[str, str]:
    result = {}
    for endpoint in ["prices", "topix_prices", "investor_types", "earnings_calendar", "financial_statements"]:
        earliest = jquants_earliest_supported_date(config, endpoint) or jquants_earliest_supported_date(config, "prices")
        result[endpoint] = earliest.isoformat() if earliest else "unknown"
    return result


def _load_yaml_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        if yaml is not None:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            payload = {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    section = payload.get(key, payload)
    return section if isinstance(section, dict) else {}


def _plan_settings(jquants: dict[str, Any], plan: str) -> dict[str, Any]:
    plans = jquants.get("plans", {})
    if isinstance(plans, dict) and isinstance(plans.get(plan), dict):
        return dict(plans[plan])
    return DEFAULT_PLAN_SETTINGS.get(plan, DEFAULT_PLAN_SETTINGS["free"])


def normalize_profile_id(value: Any) -> str:
    profile_id = str(value or "").strip()
    aliases = {
        "rookie_dealer_02_v2.1": "rookie_dealer_02_v2_1",
        "rookie_dealer_02_v2.6": "rookie_dealer_02_v2_6",
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
