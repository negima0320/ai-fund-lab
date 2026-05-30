from __future__ import annotations

from scoring import score_real_candidates


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
