from __future__ import annotations

from datetime import date

import main as main_module
from data_provider import JQuantsDataProvider
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


def test_preflight_cache_status_reports_missing_cache_as_false(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)

    status = main_module._jquants_endpoint_cache_status("investor_types", date(2026, 5, 31))

    assert status["path"] == "data/cache/jquants/investor_types/2026-04-16_to_2026-05-31.json"
    assert status["exists"] is False
    assert status["records"] == 0


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
    assert topix["last_api_status"] == "200"
