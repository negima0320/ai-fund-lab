from __future__ import annotations

from pathlib import Path

import main as main_module
from market_sections import allowed_market_sections, market_section_allowed, normalize_market_section
from paper_trade import execute_real_data_paper_trade, initial_live_paper_state
from scoring import score_real_candidates


def _candidate(code: str, section: str) -> dict:
    return {
        "code": code,
        "name": f"Test{code}",
        "sector_name": "Test",
        "section": section,
        "market_section": section,
        "listing_market": section,
        "date": "2026-03-06",
        "open": 1190,
        "high": 1220,
        "low": 1180,
        "close": 1200,
        "volume": 100000,
        "ma5": 1150,
        "ma25": 1100,
        "rsi": 58,
        "volume_ratio": 3.0,
        "turnover_value": 2_500_000_000,
        "five_day_volatility": 0.02,
        "fallback": False,
    }


def test_market_section_aliases_and_allowed_sections() -> None:
    assert normalize_market_section("プライム市場") == "TSEPrime"
    assert normalize_market_section("プライム（内国株式）") == "TSEPrime"
    assert normalize_market_section("Standard Market") == "TSEStandard"
    assert normalize_market_section("スタンダード（内国株式）") == "TSEStandard"
    assert normalize_market_section("グロース") == "TSEGrowth"
    assert normalize_market_section("グロース（内国株式）") == "TSEGrowth"

    config = {"market_filter": {"allowed_sections": ["TSEStandard", "TSEGrowth"]}}
    assert allowed_market_sections(config) == {"TSEStandard", "TSEGrowth"}
    assert market_section_allowed({"section": "TSEGrowth"}, config) is True
    assert market_section_allowed({"section": "TSEPrime"}, config) is False
    assert market_section_allowed({"section": "Unknown"}, config) is False


def test_market_section_falls_back_from_unknown_section_to_market_value() -> None:
    config = {"market_filter": {"prime": True, "standard": False, "growth": False, "allow_unknown_market": False}}

    assert market_section_allowed({"section": "Unknown", "market": "プライム（内国株式）"}, config) is True
    assert market_section_allowed({"section": "Unknown", "market": "グロース（内国株式）"}, config) is False


def test_prime_only_profile_does_not_select_growth(config_copy: dict) -> None:
    config_copy["market_filter"] = {"prime": True, "standard": False, "growth": False, "allow_unknown_market": False}
    config_copy["selection"]["min_score"] = 40
    config_copy["selection"]["fallback_min_score"] = 40
    config_copy["selection"]["top_pick_min_score"] = 40

    result = score_real_candidates([_candidate("1001", "TSEGrowth")], "2026-03-06", config_copy, "test")
    item = result["scores"][0]

    assert item["selected"] is False
    assert item["rejected_reason"] == "market_filter_excluded"
    assert result["market_filter"]["market_filter_excluded_count"] == 1


def test_growth_only_profile_does_not_select_prime(config_copy: dict) -> None:
    config_copy["market_filter"] = {"prime": False, "standard": False, "growth": True, "allow_unknown_market": False}
    config_copy["selection"]["min_score"] = 40
    config_copy["selection"]["fallback_min_score"] = 40
    config_copy["selection"]["top_pick_min_score"] = 40

    result = score_real_candidates([_candidate("1001", "TSEPrime")], "2026-03-06", config_copy, "test")
    item = result["scores"][0]

    assert item["selected"] is False
    assert item["rejected_reason"] == "market_filter_excluded"
    assert result["market_filter"]["market_filter_excluded_count"] == 1


def test_unknown_market_is_excluded_by_default(config_copy: dict) -> None:
    config_copy["market_filter"] = {"prime": True, "standard": False, "growth": False, "allow_unknown_market": False}
    config_copy["selection"]["min_score"] = 40
    config_copy["selection"]["fallback_min_score"] = 40
    config_copy["selection"]["top_pick_min_score"] = 40

    result = score_real_candidates([_candidate("1001", "Unknown")], "2026-03-06", config_copy, "test")
    item = result["scores"][0]

    assert item["selected"] is False
    assert item["rejected_reason"] == "market_filter_excluded"


def test_trade_precheck_blocks_disallowed_market_section(config_copy: dict) -> None:
    config_copy["market_filter"] = {"prime": True, "standard": False, "growth": False, "allow_unknown_market": False}
    config_copy["selection"]["fallback_min_score"] = 1
    config_copy["selection"]["min_confidence"] = 0.1
    selected = {
        **_candidate("1001", "TSEGrowth"),
        "selected": True,
        "total_score": 99,
        "confidence": 1.0,
        "reason": "test",
        "selection_reason": "test",
    }

    state, _summary, trades = execute_real_data_paper_trade(
        [selected],
        initial_live_paper_state(config_copy),
        config_copy,
        "2026-03-06",
    )

    assert state["positions"] == []
    skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    assert skipped
    assert skipped[0]["skipped_reason"] == "market_filter_excluded"


def test_screen_saves_empty_candidates_when_market_filter_excludes_all(tmp_path: Path, monkeypatch, config_copy: dict) -> None:
    config_copy["profile_id"] = "market_test"
    config_copy["profile_name"] = "market_test"
    config_copy["market_filter"] = {"prime": True, "standard": False, "growth": False, "allow_unknown_market": False}
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)

    shared_processed_dir = tmp_path / "data" / "processed"
    processed_dir = shared_processed_dir / "market_test"
    processed_dir.mkdir(parents=True)
    main_module.write_json(
        shared_processed_dir / "indicators_2026-03-06.json",
        {
            "date": "2026-03-06",
            "provider": "test",
            "indicators": [_candidate("1001", "TSEGrowth")],
        },
    )

    main_module.run_screen("jquants", "2026-03-06")
    payload = main_module.read_json(processed_dir / "candidates_2026-03-06.json")

    assert payload["candidates"] == []
    assert payload["market_coverage"]["market_filter_excluded_count"] == 1
