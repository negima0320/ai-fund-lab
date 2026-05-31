from __future__ import annotations

from datetime import date

import main as main_module
from data_provider import JQuantsApiError, JQuantsDataProvider
from jquants_plan import jquants_capability_status


def _provider_without_init(plan: str = "light") -> JQuantsDataProvider:
    provider = JQuantsDataProvider.__new__(JQuantsDataProvider)
    provider.plan = plan
    provider.capabilities = {
        capability
        for capability, status in jquants_capability_status(plan).items()
        if status == "OK"
    }
    provider.fetch_stats = {"api_calls": 0, "cache_hits": 0, "cache_misses": 0, "total_fetch_time": 0.0, "rate_limit_wait_time": 0.0}
    return provider


def test_financial_statements_api_success_creates_cache_file(monkeypatch, tmp_path) -> None:
    provider = _provider_without_init("light")
    calls = {"count": 0}

    def fake_fetch(*_args, **_kwargs):
        calls["count"] += 1
        return [{"Date": "2026-03-06", "LocalCode": "1001", "OperatingProfit": 1000}]

    monkeypatch.setattr(provider, "fetch_financial_statements", fake_fetch)

    payload = provider.fetch_financial_statements_cached(tmp_path, date(2026, 1, 1), date(2026, 3, 6))
    cached = provider.fetch_financial_statements_cached(tmp_path, date(2026, 1, 1), date(2026, 3, 6))

    assert calls["count"] == 1
    assert payload["saved"] is True
    assert cached["from_cache"] is True
    assert cached["records"][0]["OperatingProfit"] == 1000
    assert (tmp_path / "jquants" / "financial_statements" / "2026-01-01_to_2026-03-06.json").exists()


def test_records_zero_cache_is_not_usable_or_cache_hit(monkeypatch, tmp_path) -> None:
    provider = _provider_without_init("light")
    cache_path = tmp_path / "jquants" / "investor_types" / "2026-02-01_to_2026-03-06.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text('{"records":[]}', encoding="utf-8")
    calls = {"count": 0}

    def fake_fetch(*_args, **_kwargs):
        calls["count"] += 1
        return [{"date": "2026-03-06", "overseas_net_buy": 100}]

    monkeypatch.setattr(provider, "fetch_investor_types", fake_fetch)

    payload = provider.fetch_investor_types_cached(tmp_path, date(2026, 2, 1), date(2026, 3, 6))

    assert calls["count"] == 1
    assert payload["from_cache"] is False
    assert payload["usable"] is True
    assert provider.fetch_stats["cache_hits"] == 0


def test_records_zero_api_response_is_not_saved_as_cache(monkeypatch, tmp_path) -> None:
    provider = _provider_without_init("light")
    monkeypatch.setattr(provider, "fetch_investor_types", lambda *_args, **_kwargs: [])

    payload = provider.fetch_investor_types_cached(tmp_path, date(2026, 2, 1), date(2026, 3, 6))
    retry = provider.fetch_investor_types_cached(tmp_path, date(2026, 2, 1), date(2026, 3, 6))

    assert payload["saved"] is False
    assert payload["usable"] is False
    assert payload["available"] is False
    assert payload["reason"] == "empty_response"
    assert retry["from_cache"] is False
    assert not (tmp_path / "jquants" / "investor_types" / "2026-02-01_to_2026-03-06.json").exists()
    assert (tmp_path / "jquants" / "empty_ranges.json").exists()


def test_topix_api_error_logs_http_status_and_body(monkeypatch, tmp_path) -> None:
    config = {"jquants": {"plan": "light", "requests_per_minute": 60, "parallel_fetch": True}}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_topix_prices_cached(self, *_args, **_kwargs):
            raise JQuantsApiError(
                "J-Quants API request failed with HTTP 403.",
                status_code=403,
                category="auth_or_plan_error",
                endpoint="/indices/topix",
                request_url="https://api.jquants.com/v2/indices/topix?from=2026-01-01&to=2026-01-26",
                request_params={"from": "2026-01-01", "to": "2026-01-26"},
                response_body='{"message":"forbidden"}',
            )

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    payload = main_module._load_topix_prices_for_period(date(2026, 1, 1), date(2026, 1, 26), config)
    log = (tmp_path / "logs" / "jquants_api.log").read_text(encoding="utf-8")

    assert payload["records"] == []
    assert "endpoint=topix_prices" in log
    assert "status=auth_or_plan_error" in log
    assert "http_status=403" in log
    assert "request_url=https://api.jquants.com/v2/indices/topix?from=2026-01-01&to=2026-01-26" in log
    assert "response_body=" in log


def test_topix_auth_error_disables_repeated_calls(monkeypatch, tmp_path) -> None:
    config = {"jquants": {"plan": "light", "requests_per_minute": 60, "parallel_fetch": True}}
    main_module._reset_jquants_api_session()
    calls = {"count": 0}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_topix_prices_cached(self, *_args, **_kwargs):
            calls["count"] += 1
            raise JQuantsApiError("forbidden", status_code=403, category="auth_or_plan_error")

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    first = main_module._load_topix_prices_for_period(date(2026, 1, 1), date(2026, 1, 26), config)
    second = main_module._load_topix_prices_for_period(date(2026, 1, 1), date(2026, 1, 26), config)

    assert calls["count"] == 1
    assert first["records"] == []
    assert second["reason"] == "auth_or_plan_error"
    assert main_module._jquants_api_session()["disabled_features_reason"]["topix_prices"] == "auth_or_plan_error"


def test_http_status_categories_are_distinct() -> None:
    assert main_module._api_error_status(JQuantsApiError("x", status_code=401, category="auth_or_plan_error")) == "auth_or_plan_error"
    assert main_module._api_error_status(JQuantsApiError("x", status_code=403, category="auth_or_plan_error")) == "auth_or_plan_error"
    assert main_module._api_error_status(JQuantsApiError("x", status_code=400, category="bad_request")) == "bad_request"
    assert main_module._api_error_status(JQuantsApiError("x", status_code=404, category="endpoint_not_found")) == "endpoint_not_found"
    assert main_module._provider_payload_status({"api_status": "200", "records": [], "reason": "empty_response"}) == "200"


def test_topix_smoke_uses_indices_topix_path_and_iso_params(monkeypatch) -> None:
    provider = _provider_without_init("light")
    calls = []

    def fake_fetch(path, params):
        calls.append((path, params))
        return [{"Date": "20260126", "Close": 102.0}]

    monkeypatch.setattr(provider, "_get_paginated_records", fake_fetch)

    rows = provider.fetch_topix_prices(date(2026, 1, 1), date(2026, 1, 26))

    assert calls == [("/indices/topix", {"from": "2026-01-01", "to": "2026-01-26"})]
    assert rows == [{"date": "2026-01-26", "open": None, "high": None, "low": None, "close": 102.0}]


def test_preflight_cache_status_reports_missing_cache_as_false(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)

    status = main_module._jquants_endpoint_cache_status("investor_types", date(2026, 5, 31))

    assert status["path"] == "data/cache/jquants/investor_types/2025-11-14_to_2026-05-15.json"
    assert status["exists"] is False
    assert status["records"] == 0
    assert status["usable"] is False


def test_jquants_api_summary_uses_cache_files_and_api_log(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    cache_path = tmp_path / "data" / "cache" / "jquants" / "topix_prices" / "2026-01-01_to_2026-01-26.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text('{"records":[{"date":"2026-01-26","close":102.0}]}', encoding="utf-8")
    log_path = tmp_path / "logs" / "jquants_api.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "timestamp=2026-05-31T10:00:00 endpoint=topix_prices plan=light cache_hit=false status=200 records=1 saved=true cache_path=data/cache/jquants/topix_prices/2026-01-01_to_2026-01-26.json\n",
        encoding="utf-8",
    )

    summary = main_module.build_jquants_api_summary()
    topix = next(row for row in summary["endpoints"] if row["endpoint"] == "topix_prices")

    assert topix["cache_files"] == 1
    assert topix["total_records"] == 1
    assert topix["latest_cache_date"] == "2026-01-26"
    assert topix["usable_cache_files"] == 1
    assert topix["empty_cache_files"] == 0
    assert topix["last_status"] == "200"
    assert topix["last_records"] == "1"
