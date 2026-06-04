from __future__ import annotations

from real_screening import screen_candidates, screening_market_rejection_audit


def _indicator(code: str, market_section: str, **overrides):
    row = {
        "code": code,
        "name": code,
        "sector_name": "機械",
        "section": market_section,
        "market_section": market_section,
        "listing_market": market_section,
        "date": "2026-01-05",
        "open": 100,
        "high": 120,
        "low": 95,
        "close": 110,
        "volume": 1_000_000,
        "ma5": 105,
        "ma25": 100,
        "rsi": 55,
        "volume_ratio": 2.0,
        "turnover_value": 600_000_000,
        "five_day_volatility": 0.04,
    }
    row.update(overrides)
    return row


def test_screening_market_rejection_audit_uses_stable_reason_keys() -> None:
    indicators = [
        _indicator("1001", "TSEPrime", volume_ratio=5.0, turnover_value=2_000_000_000),
        _indicator(
            "2001",
            "TSEStandard",
            turnover_value=100_000_000,
            volume_ratio=0.8,
            close=95,
            ma5=100,
            ma25=105,
            rsi=80,
        ),
    ]
    candidates = screen_candidates(indicators, target_count=50)["candidates"]

    audit = screening_market_rejection_audit(indicators, candidates, target_count=50)
    reasons = audit["screening_excluded_reason_by_market"]["Standard"]

    assert audit["input_count_by_market"]["Standard"] == 1
    assert audit["screening_candidate_count_by_market"]["Standard"] == 0
    assert reasons["trading_value_low"] == 1
    assert reasons["volume_ratio_low"] == 1
    assert reasons["close_below_ma5"] == 1
    assert reasons["ma5_below_ma25"] == 1
    assert reasons["rsi_out_of_range"] == 1
    assert audit["representative_sample"][0]["reject_reason"] == (
        "trading_value_low;volume_ratio_low;close_below_ma5;ma5_below_ma25;rsi_out_of_range"
    )


def test_screening_market_rejection_audit_counts_ranking_drop_by_market() -> None:
    indicators = [
        _indicator("1001", "TSEPrime", volume_ratio=5.0, turnover_value=2_000_000_000),
        _indicator("2001", "TSEStandard", volume_ratio=2.0, turnover_value=600_000_000),
    ]
    candidates = screen_candidates(indicators, target_count=1)["candidates"]

    audit = screening_market_rejection_audit(indicators, candidates, target_count=1)

    assert audit["screening_candidate_count_by_market"]["Prime"] == 1
    assert audit["screening_candidate_count_by_market"]["Standard"] == 0
    assert audit["screening_excluded_reason_by_market"]["Standard"]["ranking_drop"] == 1
    assert audit["screening_ranking_drop_by_market"]["Standard"] == 1


def test_screen_candidates_applies_standard_only_market_override() -> None:
    indicators = [
        _indicator("1001", "TSEPrime", turnover_value=250_000_000, volume_ratio=2.0),
        _indicator("2001", "TSEStandard", turnover_value=250_000_000, volume_ratio=2.0),
    ]
    config = {
        "screening": {
            "market_overrides": {
                "TSEStandard": {
                    "min_turnover_value": 200_000_000,
                }
            }
        }
    }

    result = screen_candidates(indicators, target_count=50, config=config)
    codes = {item["code"] for item in result["candidates"]}

    assert "1001" not in codes
    assert "2001" in codes


def test_screen_candidates_can_relax_standard_ma_conditions_only() -> None:
    indicators = [
        _indicator("1001", "TSEPrime", close=95, ma5=100, ma25=105),
        _indicator("2001", "TSEStandard", close=95, ma5=100, ma25=105),
    ]
    config = {
        "screening": {
            "market_overrides": {
                "TSEStandard": {
                    "require_close_above_ma5": False,
                    "require_ma5_above_ma25": False,
                }
            }
        }
    }

    result = screen_candidates(indicators, target_count=50, config=config)
    codes = {item["code"] for item in result["candidates"]}
    audit = screening_market_rejection_audit(indicators, result["candidates"], target_count=50, config=config)

    assert "1001" not in codes
    assert "2001" in codes
    assert audit["screening_candidate_count_by_market"]["Standard"] == 1
    assert audit["screening_excluded_reason_by_market"]["Prime"]["close_below_ma5"] == 1


def test_breakout_rsi_54w_screening_requires_rsi_bullish_candle_and_high_breakout() -> None:
    config = {
        "screening": {
            "strategy": "breakout_rsi_54w",
            "breakout_rsi_54w": {
                "min_rsi": 80,
                "lookback_business_days": 270,
                "require_previous_bullish_candle": True,
                "require_54w_high_breakout": True,
            },
        }
    }
    indicators = [
        _indicator(
            "1001",
            "TSEPrime",
            open=100,
            close=112,
            high=115,
            rsi=81,
            previous_54w_high=114,
            is_54w_high_breakout=True,
        ),
        _indicator(
            "1002",
            "TSEPrime",
            open=100,
            close=112,
            high=115,
            rsi=80,
            previous_54w_high=114,
            is_54w_high_breakout=True,
        ),
        _indicator(
            "1003",
            "TSEPrime",
            open=112,
            close=100,
            high=116,
            rsi=82,
            previous_54w_high=114,
            is_54w_high_breakout=True,
        ),
        _indicator(
            "1004",
            "TSEPrime",
            open=100,
            close=112,
            high=113,
            rsi=82,
            previous_54w_high=114,
            is_54w_high_breakout=False,
        ),
    ]

    result = screen_candidates(indicators, target_count=50, config=config)
    audit = screening_market_rejection_audit(indicators, result["candidates"], target_count=50, config=config)

    assert [item["code"] for item in result["candidates"]] == ["1001"]
    reasons = audit["screening_excluded_reason_by_market"]["Prime"]
    assert reasons["rsi_breakout_low"] == 1
    assert reasons["previous_candle_not_bullish"] == 1
    assert reasons["not_54w_high_breakout"] == 1
    assert result["conditions"]["strategy"] == "breakout_rsi_54w"
