from __future__ import annotations

from datetime import date

import pytest

from data_provider import JQuantsDataProvider
from earnings_calendar import EARNINGS_FILTER_REJECTED_REASON
import main as main_module
from profile_loader import load_profile
from paper_trade import execute_real_data_paper_trade
from scoring import score_real_candidates


def test_earnings_calendar_can_be_fetched(monkeypatch) -> None:
    provider = _provider_without_init()
    monkeypatch.setattr(
        provider,
        "_get_paginated_records",
        lambda path, params: [{"Date": "20260306", "Code": "1001", "CoName": "Test", "FY": "2026", "SectorNm": "機械", "FQ": "3Q", "Section": "Prime"}],
    )

    records = provider.fetch_earnings_calendar()

    assert records[0]["Date"] == "2026-03-06"
    assert records[0]["Code"] == "1001"


def test_earnings_calendar_cache_is_used(tmp_path, monkeypatch) -> None:
    provider = _provider_without_init()
    payload = {"records": [{"Date": "2026-03-06", "Code": "1001"}]}
    cache_path = tmp_path / "jquants" / "earnings_calendar" / "2026-03-06.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    def fail_fetch(*args, **kwargs):
        raise AssertionError("cache should be used")

    monkeypatch.setattr(provider, "fetch_earnings_calendar", fail_fetch)

    result = provider.fetch_earnings_calendar_cached(tmp_path, target_date=date(2026, 3, 6))

    assert result["from_cache"] is True
    assert result["records"] == payload["records"]


def test_earnings_calendar_api_success_creates_cache_file(tmp_path, monkeypatch) -> None:
    provider = _provider_without_init()
    calls = {"count": 0}

    def fake_fetch(*_args, **_kwargs):
        calls["count"] += 1
        return [{"Date": "2026-03-06", "Code": "1001"}]

    monkeypatch.setattr(provider, "fetch_earnings_calendar", fake_fetch)

    payload = provider.fetch_earnings_calendar_cached(tmp_path, target_date=date(2026, 3, 6))
    cached = provider.fetch_earnings_calendar_cached(tmp_path, target_date=date(2026, 3, 6))

    assert calls["count"] == 1
    assert payload["saved"] is True
    assert cached["from_cache"] is True
    assert (tmp_path / "jquants" / "earnings_calendar" / "2026-03-06.json").exists()


def test_historical_backtest_does_not_fetch_current_earnings_calendar(monkeypatch, tmp_path) -> None:
    config = _earnings_config()
    monkeypatch.setattr(main_module, "ROOT", tmp_path)

    def fail_provider(*args, **kwargs):
        raise AssertionError("past backtest date must not fetch current earnings calendar")

    monkeypatch.setattr(main_module, "JQuantsDataProvider", fail_provider)

    payload = main_module._load_earnings_calendar_for_date(date(2000, 1, 3), config)

    assert payload["records"] == []
    assert payload["metadata"]["filter_available"] is False
    assert "future leak" in payload["metadata"]["warning"]


@pytest.mark.parametrize(
    ("target_date", "expected_blocked"),
    [
        ("2026-03-05", True),
        ("2026-03-06", True),
        ("2026-03-09", True),
        ("2026-03-10", False),
    ],
)
def test_earnings_filter_blocks_around_earnings_business_days(target_date: str, expected_blocked: bool) -> None:
    config = _earnings_config()
    result = score_real_candidates(
        [_candidate("1001", target_date)],
        target_date,
        config,
        "jquants",
    )["scores"][0]

    assert result["earnings_filter_blocked"] is expected_blocked
    assert result["selected"] is (not expected_blocked)
    if expected_blocked:
        assert result["rejected_reason"] == EARNINGS_FILTER_REJECTED_REASON


def test_non_earnings_candidate_is_not_filtered() -> None:
    config = _earnings_config()
    result = score_real_candidates([_candidate("1002", "2026-03-06")], "2026-03-06", config, "jquants")["scores"][0]

    assert result["selected"] is True
    assert result["earnings_filter_blocked"] is False


def test_sell_orders_are_not_filtered_by_earnings_filter() -> None:
    config = _earnings_config()
    config["execution"]["use_next_day_open_execution"] = False
    config["broker"]["provider"] = "paper"
    state = {
        "cash": 500_000,
        "positions": [
            {
                "code": "1001",
                "name": "Test 1001",
                "entry_date": "2026-03-05",
                "entry_price": 1000,
                "current_price": 1000,
                "shares": 100,
                "market_value": 100000,
                "buy_commission": 0,
                "holding_days": 1,
                "score": 45,
                "reason": "existing position",
                "earnings_filter_blocked": True,
                "earnings_announcement_date": "2026-03-06",
            }
        ],
        "pending_orders": [],
        "closed_trades": [],
        "current_day": 1,
    }

    _new_state, _summary, trades = execute_real_data_paper_trade(
        [{"code": "1001", "name": "Test 1001", "date": "2026-03-06", "close": 960, "low": 960, "high": 1000}],
        state,
        config,
        "2026-03-06",
    )

    assert any(trade.get("action") == "SELL" for trade in trades)


def test_earnings_filter_fail_open_continues_without_records() -> None:
    config = _earnings_config()
    config.pop("_earnings_calendar_records")

    result = score_real_candidates([_candidate("1001", "2026-03-06")], "2026-03-06", config, "jquants")["scores"][0]

    assert result["selected"] is True
    assert result["earnings_filter_checked"] is True
    assert result["earnings_filter_reason"] == "決算予定データ未取得のためfail_open"


def test_v2_1_earnings_filter_is_disabled() -> None:
    config = load_profile("rookie_dealer_02_v2_1")
    config["_earnings_calendar_records"] = [{"Date": "2026-03-06", "Code": "1001"}]

    result = score_real_candidates([_candidate("1001", "2026-03-06")], "2026-03-06", config, "jquants")["scores"][0]

    assert result["earnings_filter_checked"] is False
    assert result["earnings_filter_blocked"] is False


def test_v2_10_uses_earnings_filter_without_changing_v2_1() -> None:
    config = load_profile("rookie_dealer_02_v2_10")
    config["_earnings_calendar_records"] = [{"Date": "2026-03-06", "Code": "1001"}]
    config["market_filter"]["enabled"] = False

    blocked = score_real_candidates([_candidate("1001", "2026-03-06")], "2026-03-06", config, "jquants")["scores"][0]

    assert config["profile_id"] == "rookie_dealer_02_v2_10"
    assert config["earnings_filter"]["enabled"] is True
    assert blocked["earnings_filter_checked"] is True
    assert blocked["earnings_filter_blocked"] is True
    assert blocked["selected"] is False
    assert blocked["rejected_reason"] == EARNINGS_FILTER_REJECTED_REASON

    v2_1 = load_profile("rookie_dealer_02_v2_1")
    assert v2_1["earnings_filter"]["enabled"] is False


def _provider_without_init() -> JQuantsDataProvider:
    provider = JQuantsDataProvider.__new__(JQuantsDataProvider)
    provider.plan = "free"
    provider.capabilities = {"earnings_calendar"}
    return provider


def _earnings_config() -> dict:
    config = load_profile("rookie_dealer_02_v2_7")
    config["_earnings_calendar_records"] = [{"Date": "2026-03-06", "Code": "1001"}]
    config["market_filter"]["enabled"] = False
    config["selection"]["max_selected"] = 5
    return config


def _candidate(code: str, target_date: str) -> dict:
    return {
        "code": code,
        "name": f"Test {code}",
        "date": target_date,
        "close": 1000,
        "volume": 100000,
        "ma5": 1030,
        "ma25": 1000,
        "rsi": 55,
        "volume_ratio": 3.0,
        "turnover_value": 2_000_000_000,
        "five_day_volatility": 0.02,
        "fallback": False,
    }
