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
        market_context={"market_regime": "risk_off"},
    )

    assert len(result["selected"]) == 0
    assert result["scores"][0]["market_filter_applied"] is True
    assert result["scores"][0]["market_regime"] == "risk_off"
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
