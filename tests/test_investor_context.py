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


def test_v2_investor_types_fields_are_normalized(monkeypatch) -> None:
    provider = _provider_without_init("light")

    def fake_fetch(*_args, **_kwargs):
        return [
            {
                "PubDate": "2026-03-12",
                "StDate": "2026-03-02",
                "EnDate": "2026-03-06",
                "Section": "TSEPrime",
                "FrgnSell": 1000,
                "FrgnBuy": 1250,
                "FrgnBal": 250,
                "IndSell": 700,
                "IndBuy": 600,
                "IndBal": -100,
                "PropSell": 200,
                "PropBuy": 240,
                "PropBal": 40,
            }
        ]

    monkeypatch.setattr(provider, "_get_paginated_records", fake_fetch)

    rows = provider.fetch_investor_types(date(2026, 3, 1), date(2026, 3, 13))

    assert rows == [
        {
            "PubDate": "2026-03-12",
            "StDate": "2026-03-02",
            "EnDate": "2026-03-06",
            "Section": "TSEPrime",
            "FrgnSell": 1000,
            "FrgnBuy": 1250,
            "FrgnBal": 250,
            "IndSell": 700,
            "IndBuy": 600,
            "IndBal": -100,
            "PropSell": 200,
            "PropBuy": 240,
            "PropBal": 40,
            "date": "2026-03-06",
            "overseas_net_buy": 250.0,
            "overseas_buy": 1250.0,
            "overseas_sell": 1000.0,
            "individual_net_buy": -100.0,
            "institution_net_buy": None,
            "trust_bank_net_buy": None,
            "proprietary_net_buy": 40.0,
        }
    ]


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


def test_investor_context_stops_after_three_empty_responses(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    calls: list[tuple[date, date]] = []

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_investor_types_cached(self, _cache_root, start_date, end_date, force_refresh=False):
            calls.append((start_date, end_date))
            return {
                "records": [],
                "cache_path": str(tmp_path / f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"),
                "from_cache": False,
                "fallback_used": False,
                "warning": "api_success_but_empty",
                "available": False,
                "saved": False,
                "usable": False,
                "api_status": "200",
                "reason": "empty_response",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    main_module._reset_jquants_api_session()
    payload = main_module._load_investor_context_for_date(date(2026, 5, 31), config)
    second = main_module._load_investor_context_for_date(date(2026, 5, 30), config)

    assert len(calls) == 3
    assert payload["metadata"]["available"] is False
    assert second["metadata"]["disabled_reason"] == "empty_response"
    assert len(calls) == 3


def test_investor_context_uses_preloaded_records_without_daily_api(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    main_module._reset_jquants_api_session()
    main_module._jquants_api_session().setdefault("payloads", {})["investor_types"] = {
        "records": _investor_records(),
        "metadata": {"available": True, "cache_path": "preloaded"},
    }

    class FailProvider:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("daily scoring should use preloaded investor_types records")

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FailProvider)

    payload = main_module._load_investor_context_for_date(date(2026, 3, 6), config)

    assert payload["metadata"]["from_preloaded"] is True
    assert payload["metadata"]["investor_context_source"] == "investor_types"


def test_investor_context_empty_preload_suppresses_daily_api(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    main_module._reset_jquants_api_session()
    main_module._jquants_api_session().setdefault("payloads", {})["investor_types"] = {
        "records": [],
        "metadata": {"available": False, "reason": "empty_response", "warning": "api_success_but_empty"},
    }

    class FailProvider:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("empty preload should suppress daily investor_types API calls")

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FailProvider)

    payload = main_module._load_investor_context_for_date(date(2026, 3, 6), config)

    assert payload["metadata"]["from_preloaded"] is True
    assert payload["metadata"]["available"] is False
    assert payload["metadata"]["disabled_reason"] == "empty_response"


def test_investor_types_period_preload_reused_by_daily_scoring(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    main_module._reset_jquants_api_session()
    calls: list[tuple[date, date]] = []

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {}

        def fetch_investor_types_cached(self, *_args, start_date, end_date, **_kwargs):
            calls.append((start_date, end_date))
            return {
                "records": _investor_records(),
                "cache_path": "",
                "from_cache": False,
                "saved": True,
                "available": True,
                "api_status": "200",
                "reason": "",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    main_module._preload_light_api_context(config, date(2026, 1, 5), date(2026, 3, 6))
    first = main_module._load_investor_context_for_date(date(2026, 2, 2), config)
    second = main_module._load_investor_context_for_date(date(2026, 3, 6), config)

    assert len(calls) == 1
    assert calls[0][0] <= date(2025, 7, 7)
    assert calls[0][1] <= date(2026, 2, 20)
    assert first["metadata"]["from_preloaded"] is True
    assert second["metadata"]["from_preloaded"] is True


def test_investor_types_period_preload_splits_long_range(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    main_module._reset_jquants_api_session()
    calls: list[tuple[date, date]] = []

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {}

        def fetch_investor_types_cached(self, *_args, start_date, end_date, **_kwargs):
            calls.append((start_date, end_date))
            return {
                "records": [{"Date": end_date.isoformat(), "overseas_net_buy": len(calls)}],
                "cache_path": f"{start_date}_to_{end_date}.json",
                "from_cache": False,
                "saved": True,
                "available": True,
                "api_status": "200",
                "reason": "",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    payload = main_module._load_investor_types_for_period(date(2021, 5, 1), date(2026, 5, 30), config)

    assert len(calls) > 1
    assert all((end - start).days <= 364 for start, end in calls)
    assert payload["metadata"]["chunks_total"] == len(calls)
    assert payload["metadata"]["chunks_success"] == len(calls)
    assert payload["metadata"]["records_loaded"] == len(calls)
    assert main_module._jquants_api_session()["investor_types_fetch_summary"]["investor_types_chunks_total"] == len(calls)


def test_investor_types_period_preload_clamps_to_endpoint_supported_start(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    config["jquants"]["earliest_supported_date"] = {
        "light": "2021-05-01",
        "investor_types": {"light": "2021-05-31"},
    }
    main_module._reset_jquants_api_session()
    calls: list[tuple[date, date]] = []

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {}

        def fetch_investor_types_cached(self, *_args, start_date, end_date, **_kwargs):
            calls.append((start_date, end_date))
            return {
                "records": [{"Date": end_date.isoformat(), "overseas_net_buy": 100}],
                "cache_path": f"{start_date}_to_{end_date}.json",
                "from_cache": False,
                "saved": True,
                "available": True,
                "api_status": "200",
                "reason": "",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    payload = main_module._load_investor_types_for_period(date(2021, 5, 1), date(2026, 5, 30), config)
    summary = main_module._jquants_api_session()["investor_types_fetch_summary"]

    assert calls[0][0] == date(2021, 5, 31)
    assert payload["requested_start_date"] == "2020-10-31"
    assert payload["clamped_start_date"] == "2021-05-31"
    assert payload["clamp_reason"] == "plan_supported_date"
    assert summary["investor_types_fetch_requested_start"] == "2020-10-31"
    assert summary["investor_types_fetch_clamped_start"] == "2021-05-31"


def test_investor_types_period_preload_continues_when_one_chunk_fails(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    main_module._reset_jquants_api_session()
    calls = {"count": 0}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {}

        def fetch_investor_types_cached(self, *_args, start_date, end_date, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return {
                    "records": [],
                    "cache_path": "",
                    "from_cache": False,
                    "saved": False,
                    "available": False,
                    "api_status": "bad_request",
                    "reason": "bad_request",
                }
            return {
                "records": [{"Date": end_date.isoformat(), "overseas_net_buy": 100}],
                "cache_path": f"{start_date}_to_{end_date}.json",
                "from_cache": False,
                "saved": True,
                "available": True,
                "api_status": "200",
                "reason": "",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    payload = main_module._load_investor_types_for_period(date(2021, 5, 1), date(2023, 5, 30), config)

    assert payload["metadata"]["available"] is True
    assert payload["metadata"]["chunks_failed"] == 1
    assert payload["metadata"]["chunks_success"] >= 1
    assert main_module._jquants_api_session()["disabled_features_reason"] == {}


def test_investor_types_period_preload_disables_only_when_all_chunks_fail(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    main_module._reset_jquants_api_session()

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {}

        def fetch_investor_types_cached(self, *_args, **_kwargs):
            return {
                "records": [],
                "cache_path": "",
                "from_cache": False,
                "saved": False,
                "available": False,
                "api_status": "bad_request",
                "reason": "bad_request",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    payload = main_module._load_investor_types_for_period(date(2021, 5, 1), date(2022, 5, 30), config)

    assert payload["metadata"]["available"] is False
    assert payload["metadata"]["chunks_success"] == 0
    assert main_module._jquants_api_session()["disabled_features_reason"]["investor_types"] == "bad_request"


def test_investor_types_rate_limit_retries_and_continues(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    config["jquants"]["retry_backoff_seconds"] = [0, 0, 0]
    main_module._reset_jquants_api_session()
    calls = {"count": 0}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {}

        def fetch_investor_types_cached(self, *_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return {
                    "records": [],
                    "cache_path": "",
                    "from_cache": False,
                    "saved": False,
                    "available": False,
                    "api_status": "rate_limit",
                    "reason": "rate_limit",
                    "retry_after": "0",
                }
            return {
                "records": _investor_records(),
                "cache_path": "",
                "from_cache": False,
                "saved": True,
                "available": True,
                "api_status": "200",
                "reason": "",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    payload = main_module._load_investor_context_for_date(date(2026, 3, 6), config)

    assert calls["count"] == 2
    assert payload["metadata"]["investor_context_source"] == "investor_types"
    assert main_module._jquants_api_session()["api_retry_count"]["investor_types"] == 1
    assert main_module._jquants_api_session()["api_retry_success_count"]["investor_types"] == 1


def test_investor_types_bad_request_does_not_retry(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    config["jquants"]["retry_backoff_seconds"] = [0, 0, 0]
    main_module._reset_jquants_api_session()
    calls = {"count": 0}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {}

        def fetch_investor_types_cached(self, *_args, **_kwargs):
            calls["count"] += 1
            return {
                "records": [],
                "cache_path": "",
                "from_cache": False,
                "saved": False,
                "available": False,
                "api_status": "bad_request",
                "reason": "bad_request",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    payload = main_module._load_investor_context_for_date(date(2026, 3, 6), config)

    assert calls["count"] == 1
    assert payload["metadata"]["available"] is False
    assert main_module._jquants_api_session().get("api_retry_count", {}) == {}


def test_investor_types_cache_hit_suppresses_api_call_limit_consumption(monkeypatch, tmp_path) -> None:
    config = load_profile("rookie_dealer_02_v2_8")
    config.setdefault("jquants", {})["plan"] = "light"
    main_module._reset_jquants_api_session()
    start_date, end_date = main_module._investor_types_fetch_ranges(date(2026, 3, 20))[0]
    cache_path = (
        tmp_path
        / "data"
        / "cache"
        / "jquants"
        / "investor_types"
        / f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"
    )
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text('{"records":[{"Date":"2026-03-06","overseas_net_buy":100}]}', encoding="utf-8")

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("cache hit should not call provider")

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    payload = main_module._load_investor_context_for_date(date(2026, 3, 20), config)

    assert payload["records"][0]["overseas_net_buy"] == 100
    assert main_module._jquants_api_session()["api_calls_by_endpoint"] == {}


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


def test_v2_11_uses_negative_investor_context_as_filter_not_score() -> None:
    profile = load_profile("rookie_dealer_02_v2_11")
    profile["_investor_context"] = {**build_investor_context([], "2026-03-06"), "investor_context_score": -2}

    item = score_real_candidates([_candidate()], "2026-03-06", profile, "test")["scores"][0]
    expected_total = item["technical_score"] + item["relative_strength_score"] + item["market_context_score"] + item["penalty_score"]

    assert item["investor_context_score"] == -2
    assert item["score_components"]["investor_context_score"] == 0
    assert item["total_score"] == expected_total
    assert item["selected"] is False
    assert item["rejected_reason"] == "investor_context_negative"


def test_v2_11_allows_non_negative_investor_context() -> None:
    profile = load_profile("rookie_dealer_02_v2_11")
    profile["_investor_context"] = {**build_investor_context([], "2026-03-06"), "investor_context_score": 0}

    item = score_real_candidates([_candidate()], "2026-03-06", profile, "test")["scores"][0]

    assert item["investor_context_score"] == 0
    assert item["rejected_reason"] != "investor_context_negative"


def test_investor_context_filter_rejections_are_kept_when_rejected_details_disabled() -> None:
    config = load_profile("rookie_dealer_02_v2_11")
    config["analysis"] = {"save_rejected_candidates": False}
    scores = [
        {"code": "1001", "selected": True},
        {"code": "1002", "selected": False, "rejected_reason": "investor_context_negative"},
        {"code": "1003", "selected": False, "rejected_reason": "通常基準45点には届かないため落選"},
    ]

    stored = main_module._scores_for_storage(scores, config)

    assert [item["code"] for item in stored] == ["1001", "1002"]


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
