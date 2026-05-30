"""Profile-aware configuration loading for AI fund personalities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from config_version import get_config_version, load_config as load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "config" / "rookie_dealer.yaml"
PROFILES_DIR = ROOT / "config" / "profiles"
DEFAULT_PROFILE_ID = "rookie_dealer_01"
PROFILE_ALIASES = {
    "rookie_dealer_02_v2.1": "rookie_dealer_02_v2_1",
}


def load_profile(profile_id: str | None = None) -> dict[str, Any]:
    profile_id = profile_id or DEFAULT_PROFILE_ID
    profile_id = PROFILE_ALIASES.get(profile_id, profile_id)
    base = load_yaml_config(BASE_CONFIG_PATH) if BASE_CONFIG_PATH.exists() else {}
    path = get_profile_path(profile_id)
    if path.exists():
        profile = load_yaml_config(path)
    elif profile_id in {"rookie_dealer", DEFAULT_PROFILE_ID} and BASE_CONFIG_PATH.exists():
        profile = {}
    else:
        raise FileNotFoundError(f"Profile not found: {path}")

    config = _deep_merge(base, profile)
    config = _adapt_profile_schema(config, profile_id)
    validate_profile(config)
    config["_profile_path"] = str(path if path.exists() else BASE_CONFIG_PATH)
    config["_config_version"] = get_config_version(config)
    return config


def list_profiles() -> list[dict[str, str]]:
    profiles = []
    if PROFILES_DIR.exists():
        for path in sorted(PROFILES_DIR.glob("*.yaml")):
            payload = load_yaml_config(path)
            profiles.append(
                {
                    "profile_id": str(payload.get("profile_id") or path.stem),
                    "profile_name": str(payload.get("profile_name") or payload.get("dealer", {}).get("name") or path.stem),
                    "path": str(path),
                }
            )
    if not any(item["profile_id"] == DEFAULT_PROFILE_ID for item in profiles) and BASE_CONFIG_PATH.exists():
        profiles.append({"profile_id": DEFAULT_PROFILE_ID, "profile_name": "新人ディーラー1号", "path": str(BASE_CONFIG_PATH)})
    return profiles


def validate_profile(profile: dict[str, Any]) -> None:
    required = ["profile_id", "profile_name", "dealer", "portfolio", "selection", "risk", "safety", "broker"]
    missing = [key for key in required if key not in profile or profile.get(key) is None or profile.get(key) == ""]
    if missing:
        raise ValueError(f"Profile missing required keys: {', '.join(missing)}")


def get_profile_path(profile_id: str) -> Path:
    return PROFILES_DIR / f"{profile_id}.yaml"


def _adapt_profile_schema(config: dict[str, Any], profile_id: str) -> dict[str, Any]:
    config = deepcopy(config)
    config.setdefault("profile_id", profile_id)
    config.setdefault("profile_name", config.get("dealer", {}).get("name", profile_id))
    config.setdefault("description", "")

    dealer = config.setdefault("dealer", {})
    dealer["id"] = config["profile_id"]
    dealer["name"] = config["profile_name"]
    if config.get("description"):
        dealer.setdefault("persona", config["description"])

    features = config.get("features", {})
    if features:
        config.setdefault("ai_decision", {})["enabled"] = bool(features.get("ai_decision", config.get("ai_decision", {}).get("enabled", False)))
        config.setdefault("news", {})["enabled"] = bool(features.get("news", config.get("news", {}).get("enabled", True)))
        commentary = config.setdefault("ai_commentary", {})
        if "openai_commentary" in features:
            commentary["provider"] = "openai" if features.get("openai_commentary") else "rule_based"
        config.setdefault("features", {}).setdefault("market_context", bool(features.get("market_context", True)))
        config.setdefault("features", {}).setdefault("sector_analysis", bool(features.get("sector_analysis", True)))
        config.setdefault("features", {}).setdefault("candlestick_analysis", bool(features.get("candlestick_analysis", True)))

    trading = config.get("trading", {})
    portfolio = config.setdefault("portfolio", {})
    risk = config.setdefault("risk", {})
    if "initial_capital" in config:
        portfolio["initial_cash"] = config["initial_capital"]
    if "max_positions" in trading:
        portfolio["max_positions"] = trading["max_positions"]
    if "position_size_ratio" in trading:
        portfolio["max_allocation_per_symbol"] = trading["position_size_ratio"]
    if "stop_loss_rate" in trading:
        risk["stop_loss_pct"] = trading["stop_loss_rate"]
    if "take_profit_rate" in trading:
        risk["take_profit_pct"] = trading["take_profit_rate"]
    if "max_holding_days" in trading:
        risk["max_holding_business_days"] = trading["max_holding_days"]

    risk_margin = config.get("risk_margin", {})
    safety = config.setdefault("safety", {})
    for src, dst in {
        "max_daily_buy_amount": "max_daily_buy_amount",
        "max_single_order_amount": "max_single_order_amount",
        "stop_trading_if_daily_loss_rate_exceeds": "stop_trading_if_daily_loss_rate_exceeds",
        "stop_trading_if_drawdown_exceeds": "stop_trading_if_drawdown_exceeds",
    }.items():
        if src in risk_margin:
            safety[dst] = risk_margin[src]
    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
