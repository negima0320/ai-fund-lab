"""Leakage-safe score-based Portfolio Manager rules.

These rules use only prediction-time Stock Selection scores. They do not read
PM models, PM AI v3 models, backtest outcomes, cash, portfolio state, or the
current PM multiplier.
"""

from __future__ import annotations

from typing import Any


ALLOWED_SCORE_FEATURES = {
    "risk_adjusted_score",
    "expected_return",
    "stock_selection_rank_score",
    "candidate_strength",
}

FORBIDDEN_TOKENS = {
    "selected",
    "bought",
    "affordable",
    "cash",
    "portfolio",
    "position",
    "profit",
    "loss",
    "pnl",
    "result",
    "backtest",
    "exit",
    "skip",
    "filled",
    "actual",
    "final_assets",
    "trade_result",
    "realized",
    "pm_multiplier",
    "current_pm",
}


RULE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "score_based_rule_a": {
        "name": "risk_adjusted_score_percentile",
        "features": ["risk_adjusted_score"],
        "description": "Same-day percentile of Stock Selection risk_adjusted_score.",
    },
    "score_based_rule_b": {
        "name": "stock_selection_rank_score_percentile",
        "features": ["stock_selection_rank_score"],
        "description": "Same-day percentile of Stock Selection rank score.",
    },
    "score_based_rule_c": {
        "name": "composite_score",
        "features": ["risk_adjusted_score", "expected_return", "stock_selection_rank_score", "candidate_strength"],
        "weights": {
            "risk_adjusted_score": 0.35,
            "expected_return": 0.30,
            "stock_selection_rank_score": 0.25,
            "candidate_strength": 0.10,
        },
        "description": "Weighted same-day percentile blend of four prediction-time Stock Selection scores.",
    },
}

THRESHOLD_VARIANTS: dict[str, dict[str, Any]] = {
    "original": {
        "thresholds": [(0.90, 1.30), (0.75, 1.15), (0.35, 1.00), (0.10, 0.80)],
        "default": 0.60,
    },
    "conservative_high": {
        "thresholds": [(0.95, 1.30), (0.85, 1.15), (0.40, 1.00), (0.15, 0.80)],
        "default": 0.60,
    },
    "no_060": {
        "thresholds": [(0.90, 1.30), (0.75, 1.15), (0.35, 1.00)],
        "default": 0.80,
    },
    "no_115": {
        "thresholds": [(0.90, 1.30), (0.35, 1.00), (0.10, 0.80)],
        "default": 0.60,
    },
    "binary_boost": {
        "thresholds": [(0.90, 1.30)],
        "default": 1.00,
    },
    "mild": {
        "thresholds": [(0.90, 1.15), (0.35, 1.00)],
        "default": 0.80,
    },
    "inverted_low_check": {
        "thresholds": [(0.90, 1.30), (0.75, 1.15), (0.35, 1.00), (0.10, 0.80)],
        "default": 1.00,
    },
}

WEIGHT_VARIANTS: dict[str, dict[str, float]] = {
    "original": {
        "risk_adjusted_score": 0.35,
        "expected_return": 0.30,
        "stock_selection_rank_score": 0.25,
        "candidate_strength": 0.10,
    },
    "risk_heavy": {
        "risk_adjusted_score": 0.50,
        "expected_return": 0.20,
        "stock_selection_rank_score": 0.20,
        "candidate_strength": 0.10,
    },
    "expected_heavy": {
        "risk_adjusted_score": 0.20,
        "expected_return": 0.50,
        "stock_selection_rank_score": 0.20,
        "candidate_strength": 0.10,
    },
    "rank_heavy": {
        "risk_adjusted_score": 0.20,
        "expected_return": 0.20,
        "stock_selection_rank_score": 0.50,
        "candidate_strength": 0.10,
    },
    "strength_heavy": {
        "risk_adjusted_score": 0.25,
        "expected_return": 0.25,
        "stock_selection_rank_score": 0.25,
        "candidate_strength": 0.25,
    },
    "no_strength": {
        "risk_adjusted_score": 0.40,
        "expected_return": 0.30,
        "stock_selection_rank_score": 0.30,
        "candidate_strength": 0.00,
    },
}


def normalize_rule_name(rule: str) -> str:
    rule = str(rule or "").strip().lower()
    aliases = {
        "rule_a": "score_based_rule_a",
        "rule_b": "score_based_rule_b",
        "rule_c": "score_based_rule_c",
        "risk_adjusted_score_percentile": "score_based_rule_a",
        "stock_selection_rank_score_percentile": "score_based_rule_b",
        "composite_score": "score_based_rule_c",
    }
    return aliases.get(rule, rule)


def is_score_based_rule(rule: str) -> bool:
    return normalize_rule_name(rule) in RULE_DEFINITIONS


def feature_leakage_audit(feature_names: list[str] | tuple[str, ...] | set[str]) -> dict[str, Any]:
    names = [str(name) for name in feature_names]
    forbidden = [name for name in names if any(token in name.lower() for token in FORBIDDEN_TOKENS)]
    disallowed = [name for name in names if name not in ALLOWED_SCORE_FEATURES]
    return {
        "feature_columns": names,
        "allowed_feature_count": len([name for name in names if name in ALLOWED_SCORE_FEATURES]),
        "forbidden_feature_columns": forbidden,
        "forbidden_feature_count": len(forbidden),
        "disallowed_feature_columns": disallowed,
        "leakage_risk": "low" if not forbidden and not disallowed else "high",
    }


def apply_score_based_pm_rule(candidates: list[dict[str, Any]], rule: str | dict[str, Any]) -> list[dict[str, Any]]:
    """Return copied candidates with score-based PM fields attached."""
    policy = rule if isinstance(rule, dict) else {"rule": rule}
    rule_name = normalize_rule_name(str(policy.get("rule") or ""))
    threshold_variant = str(policy.get("score_based_pm_threshold_variant") or policy.get("threshold_variant") or "original")
    weight_variant = str(policy.get("score_based_pm_weight_variant") or policy.get("weight_variant") or "original")
    if not candidates or rule_name not in RULE_DEFINITIONS:
        return [dict(item) for item in candidates]

    items = [dict(item) for item in candidates]
    percentiles = {
        "risk_adjusted_score": _percentiles(items, "risk_adjusted_score"),
        "expected_return": _percentiles(items, "expected_return"),
        "stock_selection_rank_score": _percentiles(items, "stock_selection_rank_score"),
        "candidate_strength": _percentiles(items, "candidate_strength"),
    }
    candidate_count = len(items)
    for item in items:
        code = str(item.get("code") or "")
        score = _score_for_rule(code, rule_name, percentiles, weight_variant=weight_variant)
        multiplier = multiplier_for_score(rule_name, score, threshold_variant=threshold_variant)
        rank = _rank_for_score(score, [_score_for_rule(str(row.get("code") or ""), rule_name, percentiles, weight_variant=weight_variant) for row in items])
        bucket = _bucket_for_multiplier(multiplier)
        item.update(
            {
                "score_based_pm_enabled": True,
                "score_based_pm_rule": rule_name,
                "score_based_pm_threshold_variant": threshold_variant,
                "score_based_pm_weight_variant": weight_variant,
                "score_based_pm_candidate_count": candidate_count,
                "score_based_pm_rank": rank,
                "score_based_pm_score": score,
                "score_based_pm_multiplier": multiplier,
                "pm_rule_score": score,
                "pm_rule_score_percentile": score,
                "pm_rule_source": rule_name,
                "pm_rule_threshold_variant": threshold_variant,
                "pm_rule_weight_variant": weight_variant,
                "pm_rule_bucket": bucket,
                "pm_rule_risk_adjusted_score_percentile": percentiles["risk_adjusted_score"].get(code),
                "pm_rule_expected_return_percentile": percentiles["expected_return"].get(code),
                "pm_rule_stock_selection_rank_score_percentile": percentiles["stock_selection_rank_score"].get(code),
                "pm_rule_candidate_strength_percentile": percentiles["candidate_strength"].get(code),
                "pm_ai_enabled": False,
                "pm_status": "ok",
                "pm_missing_reason": "",
                "pm_feature_count": len(RULE_DEFINITIONS[rule_name]["features"]),
                "pm_high_conviction_proba": None,
                "pm_avoid_proba": None,
                "pm_score": score,
                "pm_multiplier": multiplier,
                "pm_model_version": f"{rule_name}_v1",
                "pm_feature_found": score is not None,
                "pm_warning": "" if score is not None else "score_based_pm_score_missing",
                "pm_model_path": "",
                "pm_api_only_candidate_enabled": False,
                "pm_multiplier_source": "score_based_pm_rule",
            }
        )
    return items


def multiplier_for_score(rule: str, score: float | None, *, threshold_variant: str = "original") -> float:
    if score is None:
        return 1.0
    value = float(score)
    rule_name = normalize_rule_name(rule)
    if rule_name in {"score_based_rule_a", "score_based_rule_b"}:
        if value >= 0.90:
            return 1.30
        if value >= 0.75:
            return 1.15
        if value > 0.25:
            return 1.00
        if value > 0.10:
            return 0.80
        return 0.60
    variant = THRESHOLD_VARIANTS.get(threshold_variant, THRESHOLD_VARIANTS["original"])
    for threshold, multiplier in variant["thresholds"]:
        if value >= float(threshold):
            return float(multiplier)
    return float(variant["default"])


def _score_for_rule(code: str, rule: str, percentiles: dict[str, dict[str, float]], *, weight_variant: str = "original") -> float | None:
    if rule == "score_based_rule_a":
        return percentiles["risk_adjusted_score"].get(code)
    if rule == "score_based_rule_b":
        return percentiles["stock_selection_rank_score"].get(code)
    weights = WEIGHT_VARIANTS.get(weight_variant, WEIGHT_VARIANTS["original"])
    parts = [(percentiles[column].get(code), float(weight)) for column, weight in weights.items()]
    available = [(float(value), weight) for value, weight in parts if value is not None]
    if not available:
        return None
    weight_sum = sum(weight for _value, weight in available)
    return sum(value * weight for value, weight in available) / weight_sum if weight_sum else None


def _percentiles(items: list[dict[str, Any]], feature: str) -> dict[str, float]:
    values: list[tuple[str, float]] = []
    for item in items:
        value = _feature_value(item, feature)
        if value is not None:
            values.append((str(item.get("code") or ""), value))
    if not values:
        return {}
    ordered = sorted(values, key=lambda pair: (pair[1], pair[0]), reverse=True)
    count = len(ordered)
    return {
        code: 1.0 if count <= 1 else 1.0 - ((index - 1.0) / float(count - 1))
        for index, (code, _value) in enumerate(ordered, start=1)
    }


def _feature_value(item: dict[str, Any], feature: str) -> float | None:
    aliases = {
        "risk_adjusted_score": ("risk_adjusted_score", "ml_risk_adjusted_score"),
        "expected_return": ("expected_return", "expected_return_10d", "expected_return_5d"),
        "stock_selection_rank_score": ("stock_selection_rank_score", "rank_score", "daily_rank_score", "total_score", "score"),
        "candidate_strength": ("candidate_strength", "total_score", "score", "confidence"),
    }
    for key in aliases[feature]:
        value = item.get(key)
        try:
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _bucket_for_multiplier(multiplier: float) -> str:
    return f"PM{float(multiplier):.2f}"


def _rank_for_score(score: float | None, scores: list[float | None]) -> int | None:
    if score is None:
        return None
    present = sorted([float(value) for value in scores if value is not None], reverse=True)
    for index, value in enumerate(present, start=1):
        if value == float(score):
            return index
    return None
