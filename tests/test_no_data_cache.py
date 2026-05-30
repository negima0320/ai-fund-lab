from __future__ import annotations

from datetime import date

import main


class FakeProvider:
    def __init__(self, payload=None, error: Exception | None = None) -> None:
        self.payload = payload if payload is not None else []
        self.error = error
        self.calls: list[date] = []

    def get_daily_prices(self, target_date: date) -> list[dict]:
        self.calls.append(target_date)
        if self.error:
            raise self.error
        return self.payload


def test_no_data_cache_hit_skips_api(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    target = date(2026, 1, 1)
    main.save_no_data_day(target, reason="no_prime_rows", source="test")
    provider = FakeProvider(payload=[{"Code": "1001", "Date": "2026-01-01", "Close": 1000}])

    rows = main.fetch_price_history(
        provider,
        target,
        {"1001"},
        lookback_business_days=1,
        rate_limit_per_minute=60,
        fetch_dates=[target],
        verbose=True,
    )

    assert rows == []
    assert provider.calls == []


def test_prime_rows_zero_saves_no_data_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    target = date(2026, 1, 2)
    provider = FakeProvider(payload=[{"Code": "9999", "Date": "2026-01-02", "Close": 1000}])

    rows = main.fetch_price_history(
        provider,
        target,
        {"1001"},
        lookback_business_days=1,
        rate_limit_per_minute=60,
        fetch_dates=[target],
        verbose=True,
    )

    assert rows == []
    assert provider.calls == [target]
    entry = main.load_no_data_day(target)
    assert entry is not None
    assert entry["provider"] == "jquants"
    assert entry["reason"] == "no_prime_rows"
    assert entry["source"] == "fetch-period-prices"


def test_rate_limit_error_is_not_saved_to_no_data_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    target = date(2026, 1, 5)
    provider = FakeProvider(error=RuntimeError("J-Quants rate limit exceeded"))

    rows = main.fetch_price_history(
        provider,
        target,
        {"1001"},
        lookback_business_days=1,
        rate_limit_per_minute=60,
        fetch_dates=[target],
        continue_on_error=True,
        verbose=True,
    )

    assert rows == []
    assert provider.calls == [target]
    assert main.load_no_data_day(target) is None


def test_saved_no_data_cache_is_used_on_next_fetch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    target = date(2026, 1, 6)
    first_provider = FakeProvider(payload=[])
    second_provider = FakeProvider(payload=[{"Code": "1001", "Date": "2026-01-06", "Close": 1000}])

    main.fetch_price_history(
        first_provider,
        target,
        {"1001"},
        lookback_business_days=1,
        rate_limit_per_minute=60,
        fetch_dates=[target],
        verbose=True,
    )
    rows = main.fetch_price_history(
        second_provider,
        target,
        {"1001"},
        lookback_business_days=1,
        rate_limit_per_minute=60,
        fetch_dates=[target],
        verbose=True,
    )

    assert main.load_no_data_day(target) is not None
    assert first_provider.calls == [target]
    assert second_provider.calls == []
    assert rows == []
