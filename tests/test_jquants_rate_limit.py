from __future__ import annotations

from datetime import date

import data_provider as data_provider_module
import main as main_module
from data_provider import JQuantsDataProvider, RateLimiter


def test_rate_limiter_does_not_exceed_requests_per_minute() -> None:
    current = {"value": 0.0}
    sleeps: list[float] = []

    def clock() -> float:
        return current["value"]

    def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        current["value"] += seconds

    limiter = RateLimiter(60, sleeper=sleeper, clock=clock)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert sleeps == [1.0, 1.0]
    assert limiter.acquire_count == 3
    assert limiter.total_wait_time == 2.0


def test_plan_settings_switch_rate_limit(config_copy: dict, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "JQUANTS_PLAN_OVERRIDE", None)
    monkeypatch.setattr(main_module, "_load_jquants_config_file", lambda: {
        "plan": "free",
        "plans": {
            "free": {"requests_per_minute": 5, "parallel_fetch": False},
            "light": {"requests_per_minute": 60, "parallel_fetch": True},
        },
    })
    config_copy["jquants"]["plan"] = "free"
    main_module._apply_jquants_plan_settings(config_copy)
    assert config_copy["jquants"]["requests_per_minute"] == 5
    assert config_copy["jquants"]["parallel_fetch"] is False

    monkeypatch.setattr(main_module, "JQUANTS_PLAN_OVERRIDE", "light")
    main_module._apply_jquants_plan_settings(config_copy)
    assert config_copy["jquants"]["requests_per_minute"] == 60
    assert config_copy["jquants"]["parallel_fetch"] is True


def test_parallel_fetch_is_light_only(config_copy: dict, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "_load_jquants_config_file", lambda: {
        "plans": {
            "free": {"requests_per_minute": 5, "parallel_fetch": True},
            "light": {"requests_per_minute": 60, "parallel_fetch": True},
        },
    })
    monkeypatch.setattr(main_module, "JQUANTS_PLAN_OVERRIDE", None)
    config_copy["jquants"]["plan"] = "free"
    main_module._apply_jquants_plan_settings(config_copy)
    assert main_module._jquants_parallel_fetch(config_copy) is False

    config_copy["jquants"]["plan"] = "light"
    main_module._apply_jquants_plan_settings(config_copy)
    assert main_module._jquants_parallel_fetch(config_copy) is True


def test_provider_requests_go_through_rate_limiter(monkeypatch) -> None:
    provider = JQuantsDataProvider.__new__(JQuantsDataProvider)
    provider.timeout_seconds = 20
    provider.rate_limiter = _FakeLimiter()
    provider.fetch_stats = {
        "api_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "total_fetch_time": 0.0,
        "rate_limit_wait_time": 0.0,
    }
    monkeypatch.setattr(provider, "_build_request", lambda path: path)
    monkeypatch.setattr(data_provider_module, "urlopen", lambda request, timeout: _FakeResponse('{"data": []}'))

    assert provider._get_json("/test") == {"data": []}

    assert provider.rate_limiter.calls == 1
    assert provider.fetch_stats["api_calls"] == 1
    assert provider.fetch_stats["rate_limit_wait_time"] == 0.25


def test_fetch_price_history_uses_parallel_path_for_light(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    provider = _FakePriceProvider()
    provider.parallel_fetch = True
    provider.max_parallel_requests = 2
    target_dates = [date(2026, 1, 5), date(2026, 1, 6)]

    rows = main_module.fetch_price_history(
        provider,
        target_dates[-1],
        {"1001"},
        lookback_business_days=2,
        rate_limit_per_minute=60,
        fetch_dates=target_dates,
    )

    assert len(rows) == 2
    assert sorted(provider.calls) == target_dates


class _FakeLimiter:
    def __init__(self) -> None:
        self.calls = 0

    def acquire(self) -> float:
        self.calls += 1
        return 0.25


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return self.body.encode("utf-8")


class _FakePriceProvider:
    def __init__(self) -> None:
        self.calls: list[date] = []
        self.fetch_stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_fetch_time": 0.0,
            "rate_limit_wait_time": 0.0,
        }

    def get_daily_prices(self, target_date: date) -> list[dict]:
        self.calls.append(target_date)
        return [{"Code": "1001", "Date": target_date.isoformat(), "Close": 1000}]
