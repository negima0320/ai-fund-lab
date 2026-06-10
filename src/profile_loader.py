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
    "rookie_dealer_02_v2.2": "rookie_dealer_02_v2_2",
    "rookie_dealer_02_v2.3": "rookie_dealer_02_v2_3",
    "rookie_dealer_02_v2.4": "rookie_dealer_02_v2_4",
    "rookie_dealer_02_v2.6": "rookie_dealer_02_v2_6",
    "rookie_dealer_02_v2.7": "rookie_dealer_02_v2_7",
    "rookie_dealer_02_v2.9": "rookie_dealer_02_v2_9",
    "rookie_dealer_02_v2.10": "rookie_dealer_02_v2_10",
    "rookie_dealer_02_v2.12": "rookie_dealer_02_v2_12",
    "rookie_dealer_02_v2.13": "rookie_dealer_02_v2_13",
    "rookie_dealer_02_v2.14": "rookie_dealer_02_v2_14",
    "rookie_dealer_02_v2.15": "rookie_dealer_02_v2_15",
    "rookie_dealer_02_v2.16": "rookie_dealer_02_v2_16",
    "rookie_dealer_02_v2.17": "rookie_dealer_02_v2_17",
    "rookie_dealer_02_v2.18": "rookie_dealer_02_v2_18",
    "rookie_dealer_02_v2.19": "rookie_dealer_02_v2_19",
    "rookie_dealer_02_v2.20": "rookie_dealer_02_v2_20",
    "rookie_dealer_02_v2.21": "rookie_dealer_02_v2_21",
    "rookie_dealer_02_v2.22": "rookie_dealer_02_v2_22",
    "rookie_dealer_02_v2.23": "rookie_dealer_02_v2_23",
    "rookie_dealer_02_v2.24": "rookie_dealer_02_v2_24",
    "rookie_dealer_02_v2.25": "rookie_dealer_02_v2_25",
    "rookie_dealer_02_v2.26": "rookie_dealer_02_v2_26",
    "rookie_dealer_02_v2.27": "rookie_dealer_02_v2_27",
    "rookie_dealer_02_v2.28": "rookie_dealer_02_v2_28",
    "rookie_dealer_02_v2.29": "rookie_dealer_02_v2_29",
    "rookie_dealer_02_v2.30": "rookie_dealer_02_v2_30",
    "rookie_dealer_02_v2.32": "rookie_dealer_02_v2_32",
    "rookie_dealer_02_v2.35": "rookie_dealer_02_v2_35",
    "rookie_dealer_02_v2.38": "rookie_dealer_02_v2_38",
    "rookie_dealer_02_v2.39": "rookie_dealer_02_v2_39",
    "rookie_dealer_02_v2.40": "rookie_dealer_02_v2_40",
    "rookie_dealer_02_v2.41": "rookie_dealer_02_v2_41",
    "rookie_dealer_02_v2.42": "rookie_dealer_02_v2_42",
    "rookie_dealer_02_v2.43": "rookie_dealer_02_v2_43",
    "rookie_dealer_02_v2.44": "rookie_dealer_02_v2_44",
    "rookie_dealer_02_v2.45": "rookie_dealer_02_v2_45",
    "rookie_dealer_02_v2.46": "rookie_dealer_02_v2_46",
    "rookie_dealer_02_v2.47": "rookie_dealer_02_v2_47",
    "rookie_dealer_02_v2.48": "rookie_dealer_02_v2_48",
    "rookie_dealer_02_v2.49": "rookie_dealer_02_v2_49",
    "rookie_dealer_02_v2.50": "rookie_dealer_02_v2_50",
    "rookie_dealer_02_v2.52": "rookie_dealer_02_v2_52",
    "rookie_dealer_02_v2.53": "rookie_dealer_02_v2_53",
    "rookie_dealer_02_v2.54": "rookie_dealer_02_v2_54",
    "rookie_dealer_02_v2.55": "rookie_dealer_02_v2_55",
    "rookie_dealer_02_v2.56": "rookie_dealer_02_v2_56",
    "rookie_dealer_02_v2.57": "rookie_dealer_02_v2_57",
    "rookie_dealer_02_v2.58": "rookie_dealer_02_v2_58",
    "rookie_dealer_02_v2.59": "rookie_dealer_02_v2_59",
    "rookie_dealer_02_v2.60": "rookie_dealer_02_v2_60",
    "rookie_dealer_02_v2.61": "rookie_dealer_02_v2_61",
    "rookie_dealer_02_v2.62": "rookie_dealer_02_v2_62",
    "rookie_dealer_02_v2.63": "rookie_dealer_02_v2_63",
    "rookie_dealer_02_v2.64": "rookie_dealer_02_v2_64",
    "rookie_dealer_02_v2.65": "rookie_dealer_02_v2_65",
    "rookie_dealer_02_v2.66": "rookie_dealer_02_v2_66",
    "rookie_dealer_02_v2.67": "rookie_dealer_02_v2_67",
    "rookie_dealer_02_v2.68": "rookie_dealer_02_v2_68",
    "rookie_dealer_02_v2.69": "rookie_dealer_02_v2_69",
    "rookie_dealer_02_v2.71": "rookie_dealer_02_v2_71_ml_ranked_exit_ai_050_scaled_buy",
    "rookie_dealer_02_v2_71": "rookie_dealer_02_v2_71_ml_ranked_exit_ai_050_scaled_buy",
    "rookie_dealer_02_v2.72": "rookie_dealer_02_v2_72_ml_ranked_exit_ai_scaled_buy_v2",
    "rookie_dealer_02_v2_72": "rookie_dealer_02_v2_72_ml_ranked_exit_ai_scaled_buy_v2",
    "rookie_dealer_02_v2.73": "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue",
    "rookie_dealer_02_v2_73": "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue",
    "rookie_dealer_02_v2.75": "rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing",
    "rookie_dealer_02_v2_75": "rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing",
    "rookie_dealer_02_v2.76": "rookie_dealer_02_v2_76_pm_ai_low_score_skip",
    "rookie_dealer_02_v2_76": "rookie_dealer_02_v2_76_pm_ai_low_score_skip",
    "rookie_dealer_02_v2.77": "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap",
    "rookie_dealer_02_v2_77": "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap",
    "rookie_dealer_02_v2.78": "rookie_dealer_02_v2_78_pm_aware_order_fallback_w050",
    "rookie_dealer_02_v2_78": "rookie_dealer_02_v2_78_pm_aware_order_fallback_w050",
    "rookie_dealer_02_v2.79": "rookie_dealer_02_v2_79_high_pm_min_hold_5d",
    "rookie_dealer_02_v2_79": "rookie_dealer_02_v2_79_high_pm_min_hold_5d",
    "rookie_dealer_02_v2.80": "rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate",
    "rookie_dealer_02_v2_80": "rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate",
    "rookie_dealer_02_v2.81": "rookie_dealer_02_v2_81_bear_pm115_booster_50",
    "rookie_dealer_02_v2_81": "rookie_dealer_02_v2_81_bear_pm115_booster_50",
    "rookie_dealer_02_v2.82": "rookie_dealer_02_v2_82_cap38",
    "rookie_dealer_02_v2_82": "rookie_dealer_02_v2_82_cap38",
    "rookie_dealer_02_v2.90": "rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38",
    "rookie_dealer_02_v2_90": "rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38",
    "rookie_dealer_02_v2.91": "rookie_dealer_02_v2_91_pm_ai_v2_calibrated_rule_e_cap38",
    "rookie_dealer_02_v2_91": "rookie_dealer_02_v2_91_pm_ai_v2_calibrated_rule_e_cap38",
    "rookie_dealer_02_v2.92": "rookie_dealer_02_v2_92_relative_allocator_cap38",
    "rookie_dealer_02_v2_92": "rookie_dealer_02_v2_92_relative_allocator_cap38",
    "rookie_dealer_02_v2.95": "rookie_dealer_02_v2_95_pm_disabled_equal_weight_cap38",
    "rookie_dealer_02_v2_95": "rookie_dealer_02_v2_95_pm_disabled_equal_weight_cap38",
    "rookie_dealer_02_v2.96": "rookie_dealer_02_v2_96_score_based_pm_rule_a",
    "rookie_dealer_02_v2_96": "rookie_dealer_02_v2_96_score_based_pm_rule_a",
    "rookie_dealer_02_v2_96b": "rookie_dealer_02_v2_96b_score_based_pm_rule_b",
    "rookie_dealer_02_v2_96c": "rookie_dealer_02_v2_96c_score_based_pm_rule_c",
    "rookie_dealer_02_v2.97": "rookie_dealer_02_v2_97_score_based_pm_rule_c_opt1",
    "rookie_dealer_02_v2_97": "rookie_dealer_02_v2_97_score_based_pm_rule_c_opt1",
    "rookie_dealer_02_v2_97b": "rookie_dealer_02_v2_97b_score_based_pm_rule_c_opt2",
    "rookie_dealer_02_v2_97c": "rookie_dealer_02_v2_97c_score_based_pm_rule_c_opt3",
    "rookie_dealer_02_v2_97d": "rookie_dealer_02_v2_97d_score_based_pm_rule_c_opt4",
    "rookie_dealer_02_v2_97e": "rookie_dealer_02_v2_97e_score_based_pm_rule_c_opt5",
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
    capital_policy = config.get("capital_utilization_policy", {})
    if isinstance(capital_policy, dict) and capital_policy.get("buy_lot_size"):
        trading["round_lot_size"] = capital_policy["buy_lot_size"]

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
