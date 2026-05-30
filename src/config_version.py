"""Stable config version helpers.

The version is derived from parsed config/profile content. Comments,
whitespace, and key order do not affect the hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - minimal local environments.
    yaml = None


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        text = file.read()
    if yaml is not None:
        config = yaml.safe_load(text) or {}
    else:
        config = _load_simple_yaml(text)
    config["_config_version"] = get_config_version(config)
    return config


def normalize_config(config: dict[str, Any]) -> str:
    normalized = _normalize_value(config)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def calculate_config_hash(config: dict[str, Any]) -> str:
    normalized = normalize_config(config)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_config_version(config: dict[str, Any]) -> str:
    return f"cfg_{calculate_config_hash(config)[:7]}"


def config_version_from(config: dict[str, Any]) -> str:
    version = config.get("_config_version")
    if version:
        return str(version)
    return get_config_version(config)


def attach_config_version(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload["config_version"] = config_version_from(config)
    if config.get("profile_id"):
        payload["profile_id"] = config.get("profile_id")
        payload["profile_name"] = config.get("profile_name") or config.get("dealer", {}).get("name")
    return payload


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in sorted(value.items()) if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    return value


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small two-level YAML config used by this project."""
    config: dict[str, Any] = {}
    current_section = ""
    current_key = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and ":" in line and not line.endswith(":"):
            key, value = line.split(":", 1)
            config[key.strip()] = _parse_scalar(value.strip())
            current_section = ""
            current_key = ""
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1]
            config[current_section] = {}
            current_key = ""
            continue
        if current_section and line.startswith("  ") and not line.startswith("    ") and ":" in line:
            key, value = line.strip().split(":", 1)
            current_key = key.strip()
            stripped_value = value.strip()
            config[current_section][current_key] = [] if stripped_value == "" else _parse_scalar(stripped_value)
            continue
        if current_section and current_key and line.startswith("    - "):
            target = config[current_section].setdefault(current_key, [])
            if isinstance(target, list):
                target.append(_parse_scalar(line.strip()[2:].strip()))
    return config


def _parse_scalar(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    if value in {"", "null", "None"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
