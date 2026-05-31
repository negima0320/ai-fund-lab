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
                endpoint="/indices/bars/daily/topix",
                request_url="https://api.jquants.com/v2/indices/bars/daily/topix?from=2026-01-01&to=2026-01-26",
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
    assert "request_url=https://api.jquants.com/v2/indices/bars/daily/topix?from=2026-01-01&to=2026-01-26" in log
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


def test_topix_rate_limit_retries_and_continues_relative_strength(monkeypatch, tmp_path) -> None:
    config = {"jquants": {"plan": "light", "requests_per_minute": 60, "parallel_fetch": True, "retry_backoff_seconds": [0, 0, 0]}}
    main_module._reset_jquants_api_session()
    calls = {"count": 0}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {}

        def fetch_topix_prices_cached(self, *_args, **_kwargs):
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
                "records": [{"date": "2026-01-26", "close": 102.0}],
                "cache_path": "",
                "from_cache": False,
                "saved": True,
                "available": True,
                "api_status": "200",
                "reason": "",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    payload = main_module._load_topix_prices_for_period(date(2026, 1, 1), date(2026, 1, 26), config)

    assert calls["count"] == 2
    assert payload["records"]
    assert payload["retry_success"] is True
    assert main_module._jquants_api_session()["api_retry_count"]["topix_prices"] == 1
    assert main_module._jquants_api_session()["api_retry_success_count"]["topix_prices"] == 1
    assert "topix_prices" not in main_module._jquants_api_session()["disabled_features_reason"]


def test_topix_bad_request_does_not_retry(monkeypatch, tmp_path) -> None:
    config = {"jquants": {"plan": "light", "requests_per_minute": 60, "parallel_fetch": True, "retry_backoff_seconds": [0, 0, 0]}}
    main_module._reset_jquants_api_session()
    calls = {"count": 0}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {}

        def fetch_topix_prices_cached(self, *_args, **_kwargs):
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

    payload = main_module._load_topix_prices_for_period(date(2026, 1, 1), date(2026, 1, 26), config)

    assert calls["count"] == 1
    assert payload["reason"] == "bad_request"
    assert main_module._jquants_api_session().get("api_retry_count", {}) == {}


def test_topix_cache_hit_suppresses_api_call_limit_consumption(monkeypatch, tmp_path) -> None:
    config = {"jquants": {"plan": "light", "requests_per_minute": 60, "parallel_fetch": True}}
    main_module._reset_jquants_api_session()
    cache_path = tmp_path / "data" / "cache" / "jquants" / "topix_prices" / "2026-01-01_to_2026-01-26.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text('{"records":[{"date":"2026-01-26","close":102.0}]}', encoding="utf-8")

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_topix_prices_cached(self, *_args, **_kwargs):
            raise AssertionError("cache hit should not call provider")

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    payload = main_module._load_topix_prices_for_period(date(2026, 1, 1), date(2026, 1, 26), config)

    assert payload["from_cache"] is True
    assert payload["records"][0]["close"] == 102.0
    assert main_module._jquants_api_session()["api_calls_by_endpoint"] == {}


def test_http_status_categories_are_distinct() -> None:
    assert main_module._api_error_status(JQuantsApiError("x", status_code=401, category="auth_or_plan_error")) == "auth_or_plan_error"
    assert main_module._api_error_status(JQuantsApiError("x", status_code=403, category="auth_or_plan_error")) == "auth_or_plan_error"
    assert main_module._api_error_status(JQuantsApiError("x", status_code=400, category="bad_request")) == "bad_request"
    assert main_module._api_error_status(JQuantsApiError("x", status_code=404, category="endpoint_not_found")) == "endpoint_not_found"
    assert main_module._provider_payload_status({"api_status": "200", "records": [], "reason": "empty_response"}) == "200"


def test_topix_smoke_uses_indices_bars_daily_topix_path_and_iso_params(monkeypatch) -> None:
    provider = _provider_without_init("light")
    calls = []

    def fake_fetch(path, params):
        calls.append((path, params))
        return [{"Date": "20260126", "Close": 102.0}]

    monkeypatch.setattr(provider, "_get_paginated_records", fake_fetch)

    rows = provider.fetch_topix_prices(date(2026, 1, 1), date(2026, 1, 26))

    assert calls == [("/indices/bars/daily/topix", {"from": "2026-01-01", "to": "2026-01-26"})]
    assert rows == [{"date": "2026-01-26", "open": None, "high": None, "low": None, "close": 102.0}]


def test_financial_statements_uses_v2_fins_summary_endpoint(monkeypatch) -> None:
    provider = _provider_without_init("light")
    calls = []

    def fake_fetch(path, params):
        calls.append((path, params))
        return [{"Date": "2026-01-26", "LocalCode": "1001"}]

    monkeypatch.setattr(provider, "_get_paginated_records", fake_fetch)

    records = provider.fetch_financial_statements(date(2026, 1, 1), date(2026, 1, 26))

    assert records == [{"Date": "2026-01-26", "LocalCode": "1001"}]
    assert calls == [("/fins/summary", {"from": "2026-01-01", "to": "2026-01-26"})]


def test_investor_types_uses_v2_equities_investor_types_endpoint(monkeypatch) -> None:
    provider = _provider_without_init("light")
    calls = []

    def fake_fetch(path, params):
        calls.append((path, params))
        return [{"Date": "2026-01-26", "Section": "TSEPrime", "ForeignersBalance": 100}]

    monkeypatch.setattr(provider, "_get_paginated_records", fake_fetch)

    provider.fetch_investor_types(date(2026, 1, 1), date(2026, 1, 26))

    assert calls == [("/equities/investor-types", {"from": "2026-01-01", "to": "2026-01-26"})]


def test_topix_cached_api_error_preserves_endpoint_params_and_category(monkeypatch, tmp_path) -> None:
    provider = _provider_without_init("light")

    def fake_fetch(*_args, **_kwargs):
        raise JQuantsApiError(
            "J-Quants API request failed with HTTP 400.",
            status_code=400,
            category="bad_request",
            endpoint="/indices/bars/daily/topix",
            request_url="https://api.jquants.com/v2/indices/bars/daily/topix?from=2026-01-01&to=2026-01-26",
            request_params={"from": "2026-01-01", "to": "2026-01-26"},
            response_body='{"message":"bad request"}',
        )

    monkeypatch.setattr(provider, "fetch_topix_prices", fake_fetch)

    payload = provider.fetch_topix_prices_cached(tmp_path, date(2026, 1, 1), date(2026, 1, 26), force_refresh=True)

    assert payload["api_status"] == "bad_request"
    assert payload["reason"] == "bad_request"
    assert payload["http_status"] == 400
    assert payload["request_url"].endswith("/indices/bars/daily/topix?from=2026-01-01&to=2026-01-26")
    assert payload["request_params"] == {"from": "2026-01-01", "to": "2026-01-26"}
    assert "bad request" in payload["response_body"]


def test_topix_smoke_reports_bad_request_as_endpoint_params_issue(monkeypatch, tmp_path) -> None:
    config = {"jquants": {"plan": "light", "requests_per_minute": 60, "parallel_fetch": True}}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {
                "url": "https://api.jquants.com/v2/indices/bars/daily/topix?from=2026-05-15&to=2026-05-29",
                "params": {"from": "2026-05-15", "to": "2026-05-29"},
                "status_code": 400,
                "response_body": '{"message":"invalid params"}',
            }

        def fetch_topix_prices_cached(self, *_args, **_kwargs):
            return {
                "records": [],
                "cache_path": str(tmp_path / "data" / "cache" / "jquants" / "topix_prices" / "x.json"),
                "from_cache": False,
                "saved": False,
                "usable": False,
                "available": False,
                "api_status": "bad_request",
                "reason": "bad_request",
                "http_status": 400,
                "request_url": self.last_request_metadata["url"],
                "request_params": self.last_request_metadata["params"],
                "response_body": self.last_request_metadata["response_body"],
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    result = main_module.build_jquants_smoke_test("topix_prices", config)

    assert result["error_reason"] == "bad_request"
    assert result["status_code"] == 400
    assert result["url"].endswith("/indices/bars/daily/topix?from=2026-05-15&to=2026-05-29")
    assert result["params"] == {"from": "2026-05-15", "to": "2026-05-29"}
    assert result["error_reason"] != "auth_or_plan_error"


def test_topix_smoke_reports_empty_response(monkeypatch, tmp_path) -> None:
    config = {"jquants": {"plan": "light", "requests_per_minute": 60, "parallel_fetch": True}}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {
                "url": "https://api.jquants.com/v2/indices/bars/daily/topix?from=2026-05-15&to=2026-05-29",
                "params": {"from": "2026-05-15", "to": "2026-05-29"},
                "status_code": 200,
                "response_body": '{"data":[]}',
            }

        def fetch_topix_prices_cached(self, *_args, **_kwargs):
            return {
                "records": [],
                "cache_path": str(tmp_path / "data" / "cache" / "jquants" / "topix_prices" / "x.json"),
                "from_cache": False,
                "saved": False,
                "usable": False,
                "available": False,
                "api_status": "200",
                "reason": "empty_response",
                "http_status": 200,
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    result = main_module.build_jquants_smoke_test("topix_prices", config)

    assert result["error_reason"] == "empty_response"
    assert result["status_code"] == 200
    assert result["records"] == 0
    assert result["response_body_sample"] == '{"data":[]}'


def test_investor_types_smoke_reports_empty_response_body_sample(monkeypatch, tmp_path, capsys) -> None:
    config = {"jquants": {"plan": "light", "requests_per_minute": 60, "parallel_fetch": True}}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {
                "url": "https://api.jquants.com/v2/equities/investor-types?from=2025-05-30&to=2026-05-29",
                "params": {"from": "2025-05-30", "to": "2026-05-29"},
                "status_code": 200,
                "response_body": '{"data":[]}',
            }

        def fetch_investor_types_cached(self, *_args, **_kwargs):
            return {
                "records": [],
                "cache_path": str(tmp_path / "data" / "cache" / "jquants" / "investor_types" / "x.json"),
                "from_cache": False,
                "saved": False,
                "usable": False,
                "available": False,
                "api_status": "200",
                "reason": "empty_response",
                "http_status": 200,
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    result = main_module.build_jquants_smoke_test("investor_types", config)

    assert result["endpoint"] == "investor_types"
    assert result["url"].endswith("/equities/investor-types?from=2025-05-30&to=2026-05-29")
    assert result["params"] == {"from": "2025-05-30", "to": "2026-05-29"}
    assert result["status_code"] == 200
    assert result["records"] == 0
    assert result["first_record_keys"] == []
    assert result["cache_saved"] is False
    assert result["error_reason"] == "empty_response"
    assert result["response_body_sample"] == '{"data":[]}'

    main_module.run_jquants_smoke_test("investor_types")
    output = capsys.readouterr().out
    assert "endpoint: investor_types" in output
    assert "response_body_sample: {\"data\":[]}" in output


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
