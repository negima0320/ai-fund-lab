from __future__ import annotations

from datetime import date

import main


class FakeProvider:
    def __init__(self, payload=None, error: Exception | None = None, errors: list[Exception] | None = None) -> None:
        self.payload = payload if payload is not None else []
        self.error = error
        self.errors = list(errors or [])
        self.calls: list[date] = []

    def get_daily_prices(self, target_date: date) -> list[dict]:
        self.calls.append(target_date)
        if self.errors:
            raise self.errors.pop(0)
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
    monkeypatch.setattr(main.time, "sleep", lambda seconds: None)
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
    assert provider.calls == [target, target, target, target]
    assert main.load_no_data_day(target) is None
    assert main.load_unsupported_day(target) is None


def test_rate_limit_retries_then_succeeds(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    sleeps: list[int] = []
    monkeypatch.setattr(main.time, "sleep", sleeps.append)
    target = date(2026, 1, 7)
    provider = FakeProvider(
        payload=[{"Code": "1001", "Date": "2026-01-07", "Close": 1000}],
        errors=[
            RuntimeError("J-Quants rate limit exceeded"),
            RuntimeError("J-Quants API rate limit exceeded. Wait a while and retry."),
        ],
    )

    rows = main.fetch_price_history(
        provider,
        target,
        {"1001"},
        lookback_business_days=1,
        rate_limit_per_minute=60,
        fetch_dates=[target],
        verbose=True,
    )

    assert len(rows) == 1
    assert provider.calls == [target, target, target]
    assert sleeps == [12, 24]
    assert main.load_no_data_day(target) is None


def test_rate_limit_retries_are_exhausted(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    sleeps: list[int] = []
    monkeypatch.setattr(main.time, "sleep", sleeps.append)
    target = date(2026, 1, 8)
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
    assert provider.calls == [target, target, target, target]
    assert sleeps == [12, 24, 48]
    assert main.load_no_data_day(target) is None
    assert main.load_unsupported_day(target) is None


def test_http_400_is_saved_to_unsupported_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    target = date(2023, 11, 21)
    provider = FakeProvider(error=RuntimeError("J-Quants API request failed with HTTP 400."))

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
    entry = main.load_unsupported_day(target)
    assert entry is not None
    assert entry["provider"] == "jquants"
    assert entry["reason"] == "bad_request_or_out_of_range"
    assert entry["source"] == "fetch-period-prices"


def test_unsupported_cache_hit_skips_api(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    target = date(2023, 11, 22)
    main.save_unsupported_day(
        target,
        reason="http_400_bad_request_or_out_of_range",
        source="test",
    )
    provider = FakeProvider(payload=[{"Code": "1001", "Date": "2023-11-22", "Close": 1000}])

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


def test_consecutive_400_stops_early_and_saves_range(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    dates = [date(2021, 4, 20), date(2021, 4, 21), date(2021, 4, 22), date(2021, 4, 23)]
    provider = FakeProvider(error=RuntimeError("J-Quants API request failed with HTTP 400."))

    rows = main.fetch_price_history(
        provider,
        dates[-1],
        {"1001"},
        lookback_business_days=4,
        rate_limit_per_minute=60,
        fetch_dates=dates,
        continue_on_error=True,
        verbose=True,
    )

    assert rows == []
    assert provider.calls == dates[:3]
    cache = main.load_unsupported_days_cache()
    assert cache["prices"][0]["start"] == "2021-04-20"
    assert cache["prices"][0]["end"] == "2021-04-22"
    assert main.unsupported_days_cache_path() == tmp_path / "data" / "cache" / "jquants" / "unsupported_ranges.json"
    assert not (tmp_path / "data" / "raw" / "unsupported_days_jquants.json").exists()


def test_supported_date_after_unsupported_range_fetches_normally(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    main.save_unsupported_range(
        date(2021, 4, 20),
        date(2021, 4, 30),
        reason="bad_request_or_out_of_range",
        source="test",
    )
    target = date(2021, 5, 6)
    provider = FakeProvider(payload=[{"Code": "1001", "Date": "2021-05-06", "Close": 1000}])

    rows = main.fetch_price_history(
        provider,
        target,
        {"1001"},
        lookback_business_days=1,
        rate_limit_per_minute=60,
        fetch_dates=[target],
        verbose=True,
    )

    assert len(rows) == 1
    assert provider.calls == [target]


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
