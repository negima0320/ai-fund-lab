from __future__ import annotations

from datetime import date

import main as main_module
from data_provider import JQuantsDataProvider
from investor_context import build_investor_context
from jquants_plan import jquants_capability_status
from profile_loader import load_profile
from scoring import score_real_candidates


def _provider_without_init(plan: str) -> JQuantsDataProvider:
    provider = JQuantsDataProvider.__new__(JQuantsDataProvider)
    provider.plan = plan
    provider.capabilities = {
        capability
        for capability, status in jquants_capability_status(plan).items()
        if status == "OK"
    }
    provider.fetch_stats = {"api_calls": 0, "cache_hits": 0, "cache_misses": 0, "total_fetch_time": 0.0, "rate_limit_wait_time": 0.0}
    return provider


def _candidate() -> dict:
    return {
        "code": "1001",
        "name": "Test1001",
        "date": "2026-03-06",
        "close": 1200,
        "volume": 100000,
        "ma5": 1100,
        "ma25": 1000,
        "rsi": 57.5,
        "volume_ratio": 3.0,
        "turnover_value": 2_500_000_000,
        "five_day_volatility": 0.02,
        "fallback": False,
    }


def _investor_records() -> list[dict]:
    return [
        {"Date": "2026-02-13", "overseas_net_buy": -100, "individual_net_buy": 80},
        {"Date": "2026-02-20", "overseas_net_buy": -50, "individual_net_buy": 40},
        {"Date": "2026-02-27", "overseas_net_buy": 120, "individual_net_buy": -60},
        {"Date": "2026-03-06", "overseas_net_buy": 180, "individual_net_buy": -90},
    ]


def test_light_plan_fetches_investor_types_api(monkeypatch) -> None:
    provider = _provider_without_init("light")
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_fetch(path: str, params: dict[str, str]):
        calls.append((path, params))
        return _investor_records()

    monkeypatch.setattr(provider, "_get_paginated_records", fake_fetch)

    rows = provider.fetch_investor_types(date(2026, 2, 1), date(2026, 3, 6))

    assert calls == [("/equities/investor-types", {"from": "2026-02-01", "to": "2026-03-06"})]
    assert rows[-1]["overseas_net_buy"] == 180


def test_free_plan_does_not_call_investor_types_api(monkeypatch, tmp_path) -> None:
    provider = _provider_without_init("free")

    def fail_fetch(*args, **kwargs):
        raise AssertionError("investor_types endpoint should not be called on free plan")

    monkeypatch.setattr(provider, "fetch_investor_types", fail_fetch)

    result = provider.fetch_investor_types_cached(tmp_path, date(2026, 2, 1), date(2026, 3, 6))

    assert result["available"] is False
    assert result["records"] == []


def test_investor_types_cache_fallback_on_api_failure(monkeypatch, tmp_path) -> None:
    provider = _provider_without_init("light")
    cache_path = tmp_path / "jquants" / "investor_types" / "2026-02-01_to_2026-03-06.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text('{"records":[{"date":"2026-03-06","overseas_net_buy":100}]}', encoding="utf-8")
    monkeypatch.setattr(provider, "fetch_investor_types", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    payload = provider.fetch_investor_types_cached(tmp_path, date(2026, 2, 1), date(2026, 3, 6), force_refresh=True)

    assert payload["fallback_used"] is True
    assert payload["records"][0]["overseas_net_buy"] == 100


def test_investor_types_api_success_creates_cache_file(monkeypatch, tmp_path) -> None:
    provider = _provider_without_init("light")
    calls = {"count": 0}

    def fake_fetch(*_args, **_kwargs):
        calls["count"] += 1
        return _investor_records()

    monkeypatch.setattr(provider, "fetch_investor_types", fake_fetch)

    payload = provider.fetch_investor_types_cached(tmp_path, date(2026, 2, 1), date(2026, 3, 6))
    cached = provider.fetch_investor_types_cached(tmp_path, date(2026, 2, 1), date(2026, 3, 6))

    assert calls["count"] == 1
    assert payload["saved"] is True
    assert cached["from_cache"] is True
    assert (tmp_path / "jquants" / "investor_types" / "2026-02-01_to_2026-03-06.json").exists()


def test_investor_context_expands_range_when_first_response_is_empty(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    calls: list[tuple[date, date]] = []

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_investor_types_cached(self, _cache_root, start_date, end_date, force_refresh=False):
            calls.append((start_date, end_date))
            records = [] if len(calls) == 1 else _investor_records()
            return {
                "records": records,
                "cache_path": str(tmp_path / f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"),
                "from_cache": False,
                "fallback_used": False,
                "warning": "" if records else "api_success_but_empty",
                "available": bool(records),
                "saved": True,
                "usable": bool(records),
                "api_status": "200",
                "reason": "" if records else "empty_response",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    payload = main_module._load_investor_context_for_date(date(2026, 5, 31), config)
    api_log = (tmp_path / "logs" / "jquants_api.log").read_text(encoding="utf-8")

    assert calls == [
        (date(2025, 11, 14), date(2026, 5, 15)),
        (date(2025, 5, 16), date(2026, 5, 15)),
    ]
    assert payload["metadata"]["investor_context_source"] == "investor_types"
    assert "reason=empty_response" in api_log
    assert "retry_range=2025-05-16_to_2026-05-15" in api_log


def test_investor_context_score_range_and_unavailable() -> None:
    context = build_investor_context(_investor_records(), "2026-03-06")
    unavailable = build_investor_context([], "2026-03-06")

    assert context["investor_context_source"] == "investor_types"
    assert context["overseas_net_buy_4w_trend"] == "improving"
    assert -3 <= context["investor_context_score"] <= 5
    assert context["investor_context_score"] == 5
    assert unavailable["investor_context_score"] == 0


def test_v2_1_ignores_investor_context_score() -> None:
    profile = load_profile("rookie_dealer_02_v2_1")
    profile["_investor_context"] = {**build_investor_context(_investor_records(), "2026-03-06"), "investor_context_score": 5}

    item = score_real_candidates([_candidate()], "2026-03-06", profile, "test")["scores"][0]

    assert item["investor_context_score"] == 0
    assert item["score_components"]["investor_context_score"] == 0


def test_v2_8_adds_investor_context_score_once() -> None:
    profile = load_profile("rookie_dealer_02_v2_8")
    profile["_investor_context"] = build_investor_context(_investor_records(), "2026-03-06")

    item = score_real_candidates([_candidate()], "2026-03-06", profile, "test")["scores"][0]
    expected_total = (
        item["technical_score"]
        + item["relative_strength_score"]
        + item["investor_context_score"]
        + item["market_context_score"]
        + item["penalty_score"]
    )

    assert item["investor_context_score"] == 5
    assert item["score_components"]["investor_context_score"] == 5
    assert item["total_score"] == expected_total
    assert item["score_components"]["matches_total_score"] is True


def test_fast_analysis_does_not_change_investor_context_decision() -> None:
    profile = load_profile("rookie_dealer_02_v2_8")
    profile["_investor_context"] = build_investor_context(_investor_records(), "2026-03-06")
    fast_profile = {**profile, "analysis": {**profile.get("analysis", {}), "fast_analysis": True}}

    normal = score_real_candidates([_candidate()], "2026-03-06", profile, "test")["scores"][0]
    fast = score_real_candidates([_candidate()], "2026-03-06", fast_profile, "test")["scores"][0]

    assert normal["selected"] == fast["selected"]
    assert normal["total_score"] == fast["total_score"]


def test_preflight_reports_investor_types_capability(monkeypatch, config_copy: dict) -> None:
    results: list[dict] = []
    config_copy["jquants"]["plan"] = "light"
    monkeypatch.setattr(main_module, "JQUANTS_PLAN_OVERRIDE", None)
    monkeypatch.setattr(main_module, "_load_jquants_config_file", lambda: {})
    main_module._apply_jquants_plan_settings(config_copy)

    main_module._check_jquants_plan_capabilities(results, config_copy)

    assert "J-Quants capability investor_types: OK" in [item["message"] for item in results]
