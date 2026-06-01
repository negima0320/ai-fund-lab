"""Market section helpers shared by screening, scoring, and trading."""

from __future__ import annotations

from typing import Any


SECTION_ALIASES = {
    "prime": "TSEPrime",
    "tseprime": "TSEPrime",
    "tse prime": "TSEPrime",
    "東証プライム": "TSEPrime",
    "プライム": "TSEPrime",
    "プライム市場": "TSEPrime",
    "primemarket": "TSEPrime",
    "0111": "TSEPrime",
    "standard": "TSEStandard",
    "tsestandard": "TSEStandard",
    "tse standard": "TSEStandard",
    "東証スタンダード": "TSEStandard",
    "スタンダード": "TSEStandard",
    "スタンダード市場": "TSEStandard",
    "standardmarket": "TSEStandard",
    "0112": "TSEStandard",
    "growth": "TSEGrowth",
    "tsegrowth": "TSEGrowth",
    "tse growth": "TSEGrowth",
    "東証グロース": "TSEGrowth",
    "グロース": "TSEGrowth",
    "グロース市場": "TSEGrowth",
    "growthmarket": "TSEGrowth",
    "0113": "TSEGrowth",
}


SECTION_LABELS = {
    "TSEPrime": "Prime",
    "TSEStandard": "Standard",
    "TSEGrowth": "Growth",
}


def normalize_market_section(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    compact = text.replace("_", "").replace("-", "").replace(" ", "").lower()
    if text in {"TSEPrime", "TSEStandard", "TSEGrowth"}:
        return text
    direct = SECTION_ALIASES.get(compact) or SECTION_ALIASES.get(text.lower())
    if direct:
        return direct
    if "プライム" in text or "prime" in compact:
        return "TSEPrime"
    if "スタンダード" in text or "standard" in compact:
        return "TSEStandard"
    if "グロース" in text or "growth" in compact:
        return "TSEGrowth"
    return "Unknown"


def market_section_from_row(row: dict[str, Any]) -> str:
    for key in ("section", "market_section", "listing_market", "market", "MarketCodeName", "Section"):
        value = row.get(key)
        if value:
            section = normalize_market_section(value)
            if section != "Unknown":
                return section
    return "Unknown"


def allowed_market_sections(config: dict[str, Any]) -> set[str]:
    market_filter = config.get("market_filter", {}) if isinstance(config.get("market_filter"), dict) else {}
    explicit = market_filter.get("allowed_sections")
    if isinstance(explicit, list) and explicit:
        return {normalize_market_section(item) for item in explicit if normalize_market_section(item) != "Unknown"}
    allowed: set[str] = set()
    if market_filter.get("prime", True):
        allowed.add("TSEPrime")
    if market_filter.get("standard", False):
        allowed.add("TSEStandard")
    if market_filter.get("growth", False):
        allowed.add("TSEGrowth")
    return allowed or {"TSEPrime"}


def allow_unknown_market(config: dict[str, Any]) -> bool:
    market_filter = config.get("market_filter", {}) if isinstance(config.get("market_filter"), dict) else {}
    return bool(market_filter.get("allow_unknown_market", False))


def market_section_allowed(row: dict[str, Any], config: dict[str, Any]) -> bool:
    section = market_section_from_row(row)
    if section == "Unknown":
        return allow_unknown_market(config)
    return section in allowed_market_sections(config)


def market_section_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"Prime": 0, "Standard": 0, "Growth": 0, "Unknown": 0}
    for row in rows:
        label = SECTION_LABELS.get(market_section_from_row(row), "Unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def attach_market_section_fields(row: dict[str, Any], section: Any) -> dict[str, Any]:
    normalized = normalize_market_section(section)
    return {
        **row,
        "section": normalized,
        "market_section": normalized,
        "listing_market": normalized,
    }
