from __future__ import annotations

from profile_loader import load_profile
from scoring import _apply_selection_rules, _selection_config, score_real_candidates


def candidate(
    code: str,
    *,
    volume_ratio: float,
    turnover_value: float,
    rsi: float,
    volatility: float,
) -> dict:
    return {
        "code": code,
        "name": f"Test{code}",
        "section": "TSEPrime",
        "market_section": "TSEPrime",
        "listing_market": "TSEPrime",
        "date": "2026-03-06",
        "close": 1200,
        "volume": 100000,
        "ma5": 1100,
        "ma25": 1000,
        "rsi": rsi,
        "volume_ratio": volume_ratio,
        "turnover_value": turnover_value,
        "five_day_volatility": volatility,
        "fallback": False,
    }


def scored_item(
    code: str,
    *,
    total_score: float,
    volume_ratio: float,
    rsi: float,
    candlestick_signals: list[str] | None = None,
    confidence: float = 0.8,
) -> dict:
    return {
        "code": code,
        "name": f"Test{code}",
        "section": "TSEPrime",
        "market_section": "TSEPrime",
        "listing_market": "TSEPrime",
        "total_score": total_score,
        "confidence": confidence,
        "volume_ratio": volume_ratio,
        "rsi": rsi,
        "candlestick_signals": candlestick_signals or [],
        "selected": False,
        "reason": "",
        "selection_reason": "",
        "selected_reason": "",
        "rejected_reason": "",
        "rsi_selection_excluded": False,
        "volume_filter_excluded": False,
        "market_filter_applied": False,
        "market_filter_reason": "",
    }


def conditional_selection_config(config_copy: dict) -> dict:
    config_copy["selection"] = {
        **config_copy["selection"],
        "min_score": 45,
        "fallback_min_score": 40,
        "top_pick_min_score": 40,
        "conditional_selection": {
            "enabled": True,
            "low_score_range": {"min": 40, "max": 44},
            "allow_if": {
                "min_volume_ratio": 3.0,
                "required_candlestick_signals": ["volume_confirmed_breakout"],
                "min_rsi": 50,
                "max_rsi": 65,
                "allowed_market_regimes": ["risk_on", "neutral"],
            },
        },
    }
    return _selection_config(config_copy)


def disabled_market_filter() -> dict:
    return {
        "enabled": False,
        "risk_off_buy_policy": "conservative",
        "risk_off_max_buy_orders": 1,
        "risk_off_min_score": 75,
        "risk_off_disable_top_pick": True,
    }


def test_scores_are_in_range(config_copy: dict) -> None:
    result = score_real_candidates(
        [candidate("1001", volume_ratio=3.0, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )
    item = result["scores"][0]
    assert 0 <= item["technical_score"] <= 50
    assert 0 <= item["total_score"] <= 100
    assert item["score_components"]["matches_total_score"] is True
    assert item["score_components"]["component_total"] == item["total_score"]
    assert item["ma_score"] == item["trend_score"]
    assert item["score_components"]["volume_score"] == item["volume_score"]
    assert item["score_components"]["rsi_score"] == item["rsi_score"]
    assert item["score_components"]["candlestick_score"] == item["candlestick_score"]


def test_v2_1_total_score_uses_only_evaluated_components() -> None:
    profile = load_profile("rookie_dealer_02_v2_1")
    result = score_real_candidates(
        [candidate("1001", volume_ratio=3.0, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02)],
        "2026-03-06",
        profile,
        "test",
    )
    item = result["scores"][0]
    expected_total = item["technical_score"] + item["market_context_score"] + item["penalty_score"]

    assert "news" + "_score" not in item
    assert "financial_score" not in item
    assert "base_score" not in item
    assert item["total_score"] == expected_total
    assert item["score_components"]["component_total"] == item["total_score"]
    assert item["score_components"]["matches_total_score"] is True


def test_v2_1_old_70_threshold_maps_to_new_45_threshold() -> None:
    profile = load_profile("rookie_dealer_02_v2_1")

    assert profile["selection"]["min_score"] == 45
    assert profile["selection"]["fallback_min_score"] == 40
    assert profile["selection"]["top_pick_min_score"] == 40


def test_v2_6_relative_strength_score_is_added_once() -> None:
    profile = load_profile("rookie_dealer_02_v2_6")
    item = score_real_candidates(
        [
            {
                **candidate("1001", volume_ratio=3.0, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02),
                "relative_strength_score": 10,
            }
        ],
        "2026-03-06",
        profile,
        "test",
    )["scores"][0]
    expected_total = (
        item["technical_score"]
        + item["relative_strength_score"]
        + item["market_context_score"]
        + item["penalty_score"]
    )

    assert item["relative_strength_score"] == 10
    assert "news" + "_score" not in item
    assert "financial_score" not in item
    assert item["total_score"] == expected_total
    assert item["score_components"]["component_total"] == item["total_score"]


def test_regular_selection_follows_config(config_copy: dict) -> None:
    result = score_real_candidates(
        [candidate("1001", volume_ratio=3.0, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )
    assert len(result["selected"]) == 1
    assert result["selected"][0]["total_score"] >= config_copy["selection"]["min_score"]


def test_top_pick_selected_when_below_regular_threshold(config_copy: dict) -> None:
    result = score_real_candidates(
        [
            candidate("1001", volume_ratio=3.0, turnover_value=1_250_000_000, rsi=57.5, volatility=0.08),
            candidate("1002", volume_ratio=2.0, turnover_value=800_000_000, rsi=50, volatility=0.10),
        ],
        "2026-03-06",
        config_copy,
        "test",
    )
    assert max(item["total_score"] for item in result["scores"]) < config_copy["selection"]["min_score"]
    assert len(result["selected"]) == 1
    assert result["selected"][0]["total_score"] >= config_copy["selection"]["top_pick_min_score"]


def test_no_selection_below_top_pick_threshold(config_copy: dict) -> None:
    result = score_real_candidates(
        [candidate("1001", volume_ratio=0.5, turnover_value=100_000_000, rsi=20, volatility=0.20)],
        "2026-03-06",
        config_copy,
        "test",
    )
    assert result["scores"][0]["total_score"] < config_copy["selection"]["top_pick_min_score"]
    assert len(result["selected"]) == 0


def test_conditional_selection_accepts_low_score_when_strong_conditions_match(config_copy: dict) -> None:
    scored = [
        scored_item(
            "1001",
            total_score=43,
            volume_ratio=3.2,
            rsi=55,
            candlestick_signals=["volume_confirmed_breakout"],
        )
    ]

    _apply_selection_rules(scored, conditional_selection_config(config_copy), disabled_market_filter(), "neutral")

    assert scored[0]["selected"] is True
    assert scored[0]["conditional_selection_checked"] is True
    assert scored[0]["conditional_selection_matched"] is True
    assert scored[0]["selection_reason"] == "conditional selected: 低スコア例外条件を満たしたため採用"


def test_conditional_selection_rejects_low_score_when_conditions_do_not_match(config_copy: dict) -> None:
    scored = [
        scored_item(
            "1001",
            total_score=43,
            volume_ratio=2.9,
            rsi=55,
            candlestick_signals=["bullish_candle"],
        )
    ]

    _apply_selection_rules(scored, conditional_selection_config(config_copy), disabled_market_filter(), "neutral")

    assert scored[0]["selected"] is False
    assert scored[0]["conditional_selection_checked"] is True
    assert scored[0]["conditional_selection_matched"] is False
    assert scored[0]["rejected_reason"].startswith("conditional rejected:")
    assert "volume_ratio" in scored[0]["rejected_reason"]
    assert "candlestick_signal" in scored[0]["rejected_reason"]


def test_conditional_selection_keeps_regular_selection_for_regular_score_or_more(config_copy: dict) -> None:
    scored = [
        scored_item(
            "1001",
            total_score=45,
            volume_ratio=2.1,
            rsi=55,
            candlestick_signals=["bullish_candle"],
        )
    ]

    _apply_selection_rules(scored, conditional_selection_config(config_copy), disabled_market_filter(), "neutral")

    assert scored[0]["selected"] is True
    assert scored[0]["conditional_selection_checked"] is False
    assert scored[0]["selection_reason"] == "スコア基準を満たしたため採用"


def test_conditional_selection_rejects_score_under_fallback(config_copy: dict) -> None:
    scored = [
        scored_item(
            "1001",
            total_score=39,
            volume_ratio=4.0,
            rsi=55,
            candlestick_signals=["volume_confirmed_breakout"],
        )
    ]

    _apply_selection_rules(scored, conditional_selection_config(config_copy), disabled_market_filter(), "neutral")

    assert scored[0]["selected"] is False
    assert scored[0]["conditional_selection_checked"] is False
    assert scored[0]["rejected_reason"] == "トップピック基準40点を下回るため落選"


def test_v2_1_style_selection_still_allows_top_pick_without_conditional_rules(config_copy: dict) -> None:
    scored = [
        scored_item(
            "1001",
            total_score=43,
            volume_ratio=2.1,
            rsi=55,
            candlestick_signals=["bullish_candle"],
        )
    ]
    selection_config = _selection_config(config_copy)

    _apply_selection_rules(scored, selection_config, disabled_market_filter(), "neutral")

    assert scored[0]["selected"] is True
    assert scored[0]["selection_reason"] == "通常基準45点には届かなかったが、ノートレード回避ルールにより最上位候補として採用"


def test_risk_off_blocks_low_score_candidate(config_copy: dict) -> None:
    result = score_real_candidates(
        [candidate("1001", volume_ratio=3.0, turnover_value=1_250_000_000, rsi=57.5, volatility=0.08)],
        "2026-03-06",
        config_copy,
        "test",
        market_context={"market_regime": "risk_off", "advance_ratio": 0.28},
    )

    assert len(result["selected"]) == 0
    assert result["scores"][0]["market_filter_applied"] is True
    assert result["scores"][0]["market_regime"] == "risk_off"
    assert result["scores"][0]["advance_ratio"] == 0.28
    assert result["scores"][0]["rejected_reason"] == "risk_offのため買付抑制"


def test_risk_off_disables_top_pick(config_copy: dict) -> None:
    result = score_real_candidates(
        [
            candidate("1001", volume_ratio=3.0, turnover_value=1_250_000_000, rsi=57.5, volatility=0.08),
            candidate("1002", volume_ratio=2.0, turnover_value=800_000_000, rsi=50, volatility=0.10),
        ],
        "2026-03-06",
        config_copy,
        "test",
        market_context={"market_regime": "risk_off"},
    )

    assert max(item["total_score"] for item in result["scores"]) >= config_copy["selection"]["top_pick_min_score"]
    assert max(item["total_score"] for item in result["scores"]) < config_copy["market_filter"]["risk_off_min_score"]
    assert len(result["selected"]) == 0
    assert all(item["rejected_reason"] == "risk_offのため買付抑制" for item in result["scores"])


def test_neutral_keeps_existing_top_pick_behavior(config_copy: dict) -> None:
    result = score_real_candidates(
        [
            candidate("1001", volume_ratio=3.0, turnover_value=1_250_000_000, rsi=57.5, volatility=0.08),
            candidate("1002", volume_ratio=2.0, turnover_value=800_000_000, rsi=50, volatility=0.10),
        ],
        "2026-03-06",
        config_copy,
        "test",
        market_context={"market_regime": "neutral"},
    )

    assert len(result["selected"]) == 1
    assert result["selected"][0]["market_filter_applied"] is False


def test_existing_profile_keeps_rsi_filter_disabled(config_copy: dict) -> None:
    config_copy = load_profile("rookie_dealer_01")
    result = score_real_candidates(
        [candidate("1001", volume_ratio=3.0, turnover_value=2_500_000_000, rsi=71, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert result["scores"][0]["rsi_selection_excluded"] is False
    assert result["scores"][0]["rsi_selection_penalty"] == 0
    assert len(result["selected"]) == 1


def test_rsi_above_configured_max_is_excluded_from_new_position(config_copy: dict) -> None:
    config_copy["selection"]["max_rsi_for_new_position"] = 65
    config_copy["selection"]["reject_overheated_rsi"] = True
    result = score_real_candidates(
        [candidate("1001", volume_ratio=3.0, turnover_value=2_500_000_000, rsi=65.1, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert len(result["selected"]) == 0
    assert result["scores"][0]["rsi_selection_excluded"] is True
    assert result["scores"][0]["rejected_reason"] == "RSI過熱のため新規買付見送り"


def test_rsi_at_configured_max_is_evaluated_normally(config_copy: dict) -> None:
    config_copy["selection"]["max_rsi_for_new_position"] = 65
    config_copy["selection"]["reject_overheated_rsi"] = True
    result = score_real_candidates(
        [candidate("1001", volume_ratio=3.0, turnover_value=2_500_000_000, rsi=65, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert result["scores"][0]["rsi_selection_excluded"] is False
    assert len(result["selected"]) == 1


def test_volume_filter_excludes_low_volume_ratio_candidate(config_copy: dict) -> None:
    config_copy["volume_filter"] = {"enabled": True, "min_volume_ratio": 3.0}
    result = score_real_candidates(
        [candidate("1001", volume_ratio=2.99, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert len(result["selected"]) == 0
    assert result["scores"][0]["volume_filter_excluded"] is True
    assert result["scores"][0]["volume_filter_threshold"] == 3.0
    assert result["scores"][0]["rejected_reason"] == "出来高倍率不足のため新規買付見送り"


def test_volume_filter_allows_candidate_at_threshold(config_copy: dict) -> None:
    config_copy["volume_filter"] = {"enabled": True, "min_volume_ratio": 3.0}
    result = score_real_candidates(
        [candidate("1001", volume_ratio=3.0, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert result["scores"][0]["volume_filter_excluded"] is False
    assert len(result["selected"]) == 1


def test_volume_filter_allows_candidate_when_max_volume_ratio_unset(config_copy: dict) -> None:
    config_copy["volume_filter"] = {"enabled": True, "min_volume_ratio": 2.0}
    result = score_real_candidates(
        [candidate("1001", volume_ratio=9.0, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert result["scores"][0]["volume_filter_excluded"] is False
    assert result["scores"][0]["volume_filter_max_threshold"] is None
    assert len(result["selected"]) == 1


def test_volume_filter_excludes_candidate_above_max_volume_ratio(config_copy: dict) -> None:
    config_copy["volume_filter"] = {"enabled": True, "min_volume_ratio": 2.0, "max_volume_ratio": 4.0}
    result = score_real_candidates(
        [candidate("1001", volume_ratio=4.01, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    score = result["scores"][0]
    assert len(result["selected"]) == 0
    assert score["volume_filter_excluded"] is True
    assert score["volume_filter_threshold"] == 2.0
    assert score["volume_filter_max_threshold"] == 4.0
    assert score["volume_filter_reason"] == "volume_ratio_above_max"
    assert score["rejected_reason"] == "volume_ratio_above_max"


def test_volume_filter_allows_candidate_at_max_volume_ratio(config_copy: dict) -> None:
    config_copy["volume_filter"] = {"enabled": True, "min_volume_ratio": 2.0, "max_volume_ratio": 4.0}
    result = score_real_candidates(
        [candidate("1001", volume_ratio=4.0, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert result["scores"][0]["volume_filter_excluded"] is False
    assert len(result["selected"]) == 1


def test_rsi_volume_hot_zone_filter_excludes_matching_candidate(config_copy: dict) -> None:
    config_copy["rsi_volume_hot_zone_filter"] = {
        "enabled": True,
        "min_rsi": 60,
        "min_volume_ratio": 3,
        "max_volume_ratio": 5,
        "reason": "rsi_volume_hot_zone",
    }
    result = score_real_candidates(
        [candidate("1001", volume_ratio=3.5, turnover_value=2_500_000_000, rsi=60, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    score = result["scores"][0]
    assert len(result["selected"]) == 0
    assert score["rsi_volume_hot_zone_excluded"] is True
    assert score["rsi_volume_hot_zone_reason"] == "rsi_volume_hot_zone"
    assert score["rejected_reason"] == "rsi_volume_hot_zone"


def test_rsi_volume_hot_zone_filter_uses_inclusive_volume_upper_bound(config_copy: dict) -> None:
    config_copy["rsi_volume_hot_zone_filter"] = {
        "enabled": True,
        "min_rsi": 60,
        "min_volume_ratio": 3,
        "max_volume_ratio": 5,
        "reason": "rsi_volume_hot_zone",
    }
    result = score_real_candidates(
        [candidate("1001", volume_ratio=5.0, turnover_value=2_500_000_000, rsi=64, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert result["scores"][0]["rsi_volume_hot_zone_excluded"] is True
    assert result["scores"][0]["rejected_reason"] == "rsi_volume_hot_zone"


def test_rsi_volume_hot_zone_filter_allows_non_matching_candidates(config_copy: dict) -> None:
    config_copy["rsi_volume_hot_zone_filter"] = {
        "enabled": True,
        "min_rsi": 60,
        "min_volume_ratio": 3,
        "max_volume_ratio": 5,
        "reason": "rsi_volume_hot_zone",
    }
    result = score_real_candidates(
        [
            candidate("1001", volume_ratio=2.99, turnover_value=2_500_000_000, rsi=64, volatility=0.02),
            candidate("1002", volume_ratio=5.01, turnover_value=2_500_000_000, rsi=64, volatility=0.02),
            candidate("1003", volume_ratio=3.5, turnover_value=2_500_000_000, rsi=59.99, volatility=0.02),
        ],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert [score["rsi_volume_hot_zone_excluded"] for score in result["scores"]] == [False, False, False]
    assert len(result["selected"]) == 3


def test_rsi_volume_hot_zone_filter_disabled_keeps_existing_profile_behavior(config_copy: dict) -> None:
    result = score_real_candidates(
        [candidate("1001", volume_ratio=3.5, turnover_value=2_500_000_000, rsi=60, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert result["scores"][0]["rsi_volume_hot_zone_excluded"] is False
    assert len(result["selected"]) == 1


def test_volume_filter_can_use_two_times_threshold(config_copy: dict) -> None:
    config_copy["volume_filter"] = {"enabled": True, "min_volume_ratio": 2.0}
    below = score_real_candidates(
        [candidate("1001", volume_ratio=1.99, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )
    at_threshold = score_real_candidates(
        [candidate("1001", volume_ratio=2.0, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert below["scores"][0]["volume_filter_excluded"] is True
    assert len(below["selected"]) == 0
    assert at_threshold["scores"][0]["volume_filter_excluded"] is False
    assert len(at_threshold["selected"]) == 1


def test_existing_profile_keeps_volume_filter_disabled(config_copy: dict) -> None:
    result = score_real_candidates(
        [candidate("1001", volume_ratio=2.0, turnover_value=2_500_000_000, rsi=57.5, volatility=0.02)],
        "2026-03-06",
        config_copy,
        "test",
    )

    assert result["scores"][0]["volume_filter_excluded"] is False
