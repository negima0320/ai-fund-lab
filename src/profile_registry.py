"""Profile registry helpers for experiment management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config_version import load_config as load_yaml_config
from jquants_plan import jquants_profile_compatibility
from profile_loader import get_profile_path, load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE_REGISTRY_PATH = ROOT / "config" / "profile_registry.yaml"
VALID_ROLES = {"baseline", "experiment", "deprecated"}
VALID_REQUIRED_PLANS = {"free", "light"}


def load_profile_registry(path: Path = PROFILE_REGISTRY_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Profile registry not found: {path}")
    payload = load_yaml_config(path)
    if not isinstance(payload, dict):
        raise ValueError("profile_registry.yaml must be a mapping")
    return payload


def registry_profiles(registry: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    payload = registry or load_profile_registry()
    profiles = payload.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("profile_registry.yaml must contain a profiles mapping")
    return {str(profile_id): dict(value or {}) for profile_id, value in profiles.items()}


def list_profiles(registry: dict[str, Any] | None = None, include_deprecated: bool = True) -> list[dict[str, Any]]:
    rows = []
    for profile_id, item in registry_profiles(registry).items():
        role = str(item.get("role", ""))
        if role == "deprecated" and not include_deprecated:
            continue
        features = item.get("features", {}) if isinstance(item.get("features"), dict) else {}
        enabled_features = [feature for feature, enabled in features.items() if bool(enabled)]
        rows.append(
            {
                "profile_id": profile_id,
                "role": role,
                "required_plan": item.get("required_plan", ""),
                "enabled_features": enabled_features,
                "compare_to": item.get("compare_to"),
                "description": item.get("description", ""),
                "recommendation_status": item.get("recommendation_status", ""),
                "recommendation_note": item.get("recommendation_note", ""),
            }
        )
    return sorted(rows, key=lambda row: row["profile_id"])


def get_profile_info(profile_id: str, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    profiles = registry_profiles(registry)
    if profile_id not in profiles:
        raise KeyError(f"Profile not found in registry: {profile_id}")
    item = profiles[profile_id]
    config = load_profile(profile_id)
    yaml_path = Path(config.get("_profile_path") or get_profile_path(profile_id))
    features = item.get("features", {}) if isinstance(item.get("features"), dict) else {}
    enabled_features = [feature for feature, enabled in features.items() if bool(enabled)]
    compare_to = item.get("compare_to")
    backtest_command = (
        f"python src/main.py --mode backtest --profile {profile_id} "
        "--start-date YYYY-MM-DD --end-date YYYY-MM-DD --fast-analysis"
    )
    compare_command = (
        f"python src/main.py --mode compare-profiles --profiles {compare_to} {profile_id} "
        "--start-date YYYY-MM-DD --end-date YYYY-MM-DD"
        if compare_to
        else "N/A"
    )
    return {
        "profile_id": profile_id,
        "role": item.get("role", ""),
        "description": item.get("description", ""),
        "recommendation_status": item.get("recommendation_status", ""),
        "recommendation_note": item.get("recommendation_note", ""),
        "required_plan": item.get("required_plan", ""),
        "compare_to": compare_to,
        "enabled_features": enabled_features,
        "profile_yaml_path": str(yaml_path),
        "score_formula": config.get("scoring", {}).get("total_score_formula", "technical_score + market_context_score + penalty_score"),
        "required_capabilities": sorted(jquants_profile_compatibility(profile_id, item.get("required_plan", "free"))["profile_required_capabilities"]),
        "recommended_backtest_command": backtest_command,
        "recommended_compare_command": compare_command,
    }


def get_experiments(base_profile: str, registry: dict[str, Any] | None = None) -> list[str]:
    return [
        row["profile_id"]
        for row in list_profiles(registry, include_deprecated=False)
        if row["role"] == "experiment" and row.get("compare_to") == base_profile
    ]


def validate_registry(
    registry: dict[str, Any] | None = None,
    profiles_dir: Path | None = None,
) -> dict[str, Any]:
    payload = registry or load_profile_registry()
    profile_dir = profiles_dir or (ROOT / "config" / "profiles")
    checks: list[dict[str, str]] = []
    try:
        profiles = registry_profiles(payload)
    except ValueError as exc:
        checks.append({"status": "FAIL", "name": "registry_profiles_mapping", "message": str(exc)})
        return _validation_payload(checks)

    for profile_id, item in sorted(profiles.items()):
        _validate_registry_item(checks, profile_id, item, profile_dir, profiles)
    return _validation_payload(checks)


def _validate_registry_item(
    checks: list[dict[str, str]],
    profile_id: str,
    item: dict[str, Any],
    profiles_dir: Path,
    profiles: dict[str, dict[str, Any]],
) -> None:
    yaml_path = profiles_dir / f"{profile_id}.yaml"
    _add_check(checks, yaml_path.exists(), f"{profile_id}.yaml_exists", f"{yaml_path.name} exists", f"{yaml_path.name} is missing")
    if yaml_path.exists():
        try:
            payload = load_yaml_config(yaml_path)
        except Exception as exc:
            payload = {}
            _add_check(checks, False, f"{profile_id}.yaml_load", "", f"{yaml_path.name} cannot be loaded: {exc}")
        yaml_profile_id = str(payload.get("profile_id") or "")
        _add_check(
            checks,
            yaml_profile_id == profile_id,
            f"{profile_id}.profile_id_match",
            "profile_id matches registry key",
            f"profile_id mismatch: registry={profile_id}, yaml={yaml_profile_id or '<missing>'}",
        )

    role = item.get("role")
    _add_check(checks, role in VALID_ROLES, f"{profile_id}.role", f"role is {role}", f"invalid role: {role}")
    compare_to = item.get("compare_to")
    _add_check(
        checks,
        role != "experiment" or bool(compare_to),
        f"{profile_id}.compare_to",
        "compare_to is valid",
        "experiment profile must define compare_to",
    )
    if compare_to:
        _add_check(
            checks,
            str(compare_to) in profiles,
            f"{profile_id}.compare_to_exists",
            "compare_to exists",
            "compare_to profile is missing",
        )
    required_plan = item.get("required_plan")
    _add_check(
        checks,
        required_plan in VALID_REQUIRED_PLANS,
        f"{profile_id}.required_plan",
        f"required_plan is {required_plan}",
        f"invalid required_plan: {required_plan}",
    )
    features = item.get("features")
    if not isinstance(features, dict):
        _add_check(checks, False, f"{profile_id}.features", "", "features must be a mapping")
    else:
        non_bool = [name for name, enabled in features.items() if not isinstance(enabled, bool)]
        _add_check(
            checks,
            not non_bool,
            f"{profile_id}.features_bool",
            "all features are bool",
            f"features must be bool: {', '.join(non_bool)}",
        )


def _add_check(checks: list[dict[str, str]], ok: bool, name: str, ok_message: str, fail_message: str) -> None:
    checks.append({"status": "OK" if ok else "FAIL", "name": name, "message": ok_message if ok else fail_message})


def _validation_payload(checks: list[dict[str, str]]) -> dict[str, Any]:
    fail_count = sum(1 for item in checks if item["status"] == "FAIL")
    warn_count = sum(1 for item in checks if item["status"] == "WARN")
    return {
        "status": "FAILED" if fail_count else "OK_WITH_WARNINGS" if warn_count else "OK",
        "fail_count": fail_count,
        "warn_count": warn_count,
        "checks": checks,
    }
