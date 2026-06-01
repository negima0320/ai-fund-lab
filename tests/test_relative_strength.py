from __future__ import annotations

import json
from datetime import date

import main as main_module
import indicators as indicators_module
from benchmark_provider import build_relative_strength_benchmark
from data_provider import JQuantsDataProvider
from indicators import calculate_indicators
from profile_loader import load_profile
from real_screening import screen_candidates
from scoring import score_real_candidates


def test_relative_strength_uses_universe_benchmark_when_topix_is_missing() -> None:
    price_rows = []
    dates = [f"2026-01-{day:02d}" for day in range(1, 27)]
    for code, latest_close in [("1001", 120), ("1002", 100)]:
        for date in dates:
            price_rows.append(
                {
                    "code": code,
                    "date": date,
                    "open": 100,
                    "high": max(100, latest_close),
                    "low": 95,
                    "close": latest_close if date == "2026-01-26" else 100,
                    "volume": 1000,
                }
            )

    indicators, excluded = calculate_indicators(
        price_rows,
        {"1001": "Strong", "1002": "Flat"},
        "2026-01-26",
        indicator_mode="minimal",
        enable_relative_strength=True,
    )
    by_code = {item["code"]: item for item in indicators}

    assert excluded == 0
    assert by_code["1001"]["stock_return_5d"] == 0.2
    assert by_code["1001"]["stock_return_10d"] == 0.2
    assert by_code["1001"]["stock_return_20d"] == 0.2
    assert by_code["1001"]["benchmark_return_5d"] == 0.1
    assert by_code["1001"]["benchmark_return_10d"] == 0.1
    assert by_code["1001"]["benchmark_return_20d"] == 0.1
    assert by_code["1001"]["relative_strength_5d"] == 0.1
    assert by_code["1001"]["relative_strength_10d"] == 0.1
    assert by_code["1001"]["relative_strength_20d"] == 0.1
    assert by_code["1001"]["relative_strength_score"] == 10
    assert 0 <= by_code["1002"]["relative_strength_score"] <= 10


def test_light_plan_uses_topix_benchmark() -> None:
    price_rows = _relative_strength_price_rows()
    topix_rows = _topix_rows(latest_close=102)

    benchmark = build_relative_strength_benchmark(price_rows, "2026-01-26", topix_rows)
    indicators, _excluded = calculate_indicators(
        price_rows,
        {"1001": "Strong", "1002": "Flat"},
        "2026-01-26",
        indicator_mode="minimal",
        enable_relative_strength=True,
        benchmark_returns=benchmark["benchmark_returns"],
        benchmark_source=benchmark["benchmark_source"],
    )
    by_code = {item["code"]: item for item in indicators}

    assert benchmark["benchmark_source"] == "topix"
    assert by_code["1001"]["benchmark_source"] == "topix"
    assert by_code["1001"]["benchmark_return_5d"] == 0.02
    assert by_code["1001"]["relative_strength_5d"] == 0.18


def test_free_plan_topix_cache_does_not_call_api(monkeypatch, tmp_path) -> None:
    provider = JQuantsDataProvider.__new__(JQuantsDataProvider)
    provider.plan = "free"
    provider.capabilities = set()
    provider.fetch_stats = {"api_calls": 0, "cache_hits": 0, "cache_misses": 0, "total_fetch_time": 0.0, "rate_limit_wait_time": 0.0}

    def fail(*args, **kwargs):
        raise AssertionError("TOPIX API should not be called on free plan")

    monkeypatch.setattr(provider, "get_topix_prices", fail)

    result = provider.fetch_topix_prices_cached(tmp_path, date(2026, 1, 1), date(2026, 1, 26))

    assert result["available"] is False
    assert result["records"] == []


def test_topix_cache_and_api_failure_fallback(monkeypatch, tmp_path) -> None:
    provider = JQuantsDataProvider.__new__(JQuantsDataProvider)
    provider.plan = "light"
    provider.capabilities = {"topix_prices"}
    provider.fetch_stats = {"api_calls": 0, "cache_hits": 0, "cache_misses": 0, "total_fetch_time": 0.0, "rate_limit_wait_time": 0.0}
    cache_path = tmp_path / "jquants" / "topix_prices" / "2026-01-01_to_2026-01-26.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text('{"records":[{"date":"2026-01-26","close":102.0}]}', encoding="utf-8")

    monkeypatch.setattr(provider, "fetch_topix_prices", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    cached = provider.fetch_topix_prices_cached(tmp_path, date(2026, 1, 1), date(2026, 1, 26))
    fallback = provider.fetch_topix_prices_cached(tmp_path, date(2026, 1, 1), date(2026, 1, 26), force_refresh=True)

    assert cached["from_cache"] is True
    assert fallback["fallback_used"] is True
    assert fallback["records"][0]["close"] == 102.0


def test_topix_prices_api_success_creates_cache_file(monkeypatch, tmp_path) -> None:
    provider = JQuantsDataProvider.__new__(JQuantsDataProvider)
    provider.plan = "light"
    provider.capabilities = {"topix_prices"}
    provider.fetch_stats = {"api_calls": 0, "cache_hits": 0, "cache_misses": 0, "total_fetch_time": 0.0, "rate_limit_wait_time": 0.0}
    calls = {"count": 0}

    def fake_fetch(*_args, **_kwargs):
        calls["count"] += 1
        return [{"date": "2026-01-26", "close": 102.0}]

    monkeypatch.setattr(provider, "fetch_topix_prices", fake_fetch)

    payload = provider.fetch_topix_prices_cached(tmp_path, date(2026, 1, 1), date(2026, 1, 26))
    cached = provider.fetch_topix_prices_cached(tmp_path, date(2026, 1, 1), date(2026, 1, 26))

    assert calls["count"] == 1
    assert payload["saved"] is True
    assert cached["from_cache"] is True
    assert (tmp_path / "jquants" / "topix_prices" / "2026-01-01_to_2026-01-26.json").exists()


def test_v2_1_does_not_calculate_relative_strength(monkeypatch) -> None:
    price_rows = _relative_strength_price_rows()

    def fail_benchmark(*args, **kwargs):
        raise AssertionError("benchmark calculation should not run")

    monkeypatch.setattr(indicators_module, "_benchmark_returns", fail_benchmark)
    indicators, _excluded = calculate_indicators(
        price_rows,
        {"1001": "Strong", "1002": "Flat"},
        "2026-01-26",
        indicator_mode="minimal",
        enable_relative_strength=False,
    )

    assert indicators[0]["relative_strength_score"] == 0
    assert indicators[0]["relative_strength_5d"] is None


def test_v2_6_calculates_relative_strength(monkeypatch) -> None:
    price_rows = _relative_strength_price_rows()
    calls = {"count": 0}
    original = indicators_module._benchmark_returns
    indicators_module._BENCHMARK_RETURN_CACHE.clear()

    def count_benchmark(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(indicators_module, "_benchmark_returns", count_benchmark)
    indicators, _excluded = calculate_indicators(
        price_rows,
        {"1001": "Strong", "1002": "Flat"},
        "2026-01-26",
        indicator_mode="minimal",
        enable_relative_strength=True,
    )

    assert calls["count"] == 1
    assert max(item["relative_strength_score"] for item in indicators) > 0


def test_benchmark_returns_are_cached_by_date(monkeypatch) -> None:
    price_rows = _relative_strength_price_rows()
    calls = {"count": 0}
    original = indicators_module._benchmark_returns
    indicators_module._BENCHMARK_RETURN_CACHE.clear()

    def count_benchmark(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(indicators_module, "_benchmark_returns", count_benchmark)
    for _ in range(2):
        calculate_indicators(
            price_rows,
            {"1001": "Strong", "1002": "Flat"},
            "2026-01-26",
            indicator_mode="minimal",
            enable_relative_strength=True,
        )

    assert calls["count"] == 1


def test_relative_strength_score_is_used_only_by_v2_6() -> None:
    candidate = {
        "code": "1001",
        "name": "Strong",
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
        "relative_strength_5d": 0.04,
        "relative_strength_10d": 0.06,
        "relative_strength_20d": 0.09,
        "benchmark_return_5d": 0.01,
        "benchmark_return_10d": 0.02,
        "benchmark_return_20d": 0.03,
        "relative_strength_score": 10,
    }
    v2_1 = load_profile("rookie_dealer_02_v2_1")
    v2_6 = load_profile("rookie_dealer_02_v2_6")

    base = score_real_candidates([candidate], "2026-03-06", v2_1, "test")["scores"][0]
    enhanced = score_real_candidates([candidate], "2026-03-06", v2_6, "test")["scores"][0]

    assert base["relative_strength_score"] == 0.0
    assert enhanced["relative_strength_score"] == 10
    assert enhanced["total_score"] == base["total_score"] + 10
    assert enhanced["score_components"]["relative_strength_score"] == 10
    assert enhanced["score_components"]["matches_total_score"] is True
    assert enhanced["relative_strength_5d"] == 0.04
    assert enhanced["benchmark_return_20d"] == 0.03


def test_v2_6_indicator_route_fetches_topix_and_records_relative_strength(monkeypatch, tmp_path) -> None:
    main_module._reset_jquants_api_session()
    config = load_profile("rookie_dealer_02_v2_6")
    config.setdefault("jquants", {})["plan"] = "light"
    config["jquants"]["requests_per_minute"] = 60
    config["jquants"]["parallel_fetch"] = True
    config.setdefault("backtest", {})["indicator_mode"] = "minimal"

    target = date(2026, 1, 26)
    fetch_dates = main_module.previous_business_dates(target, 35)
    _write_prime_fixture(tmp_path)
    _write_price_history_fixture(tmp_path, fetch_dates)
    calls = {"count": 0}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_topix_prices_cached(self, *_args, **_kwargs):
            calls["count"] += 1
            return {
                "records": _topix_rows(latest_close=102, dates=[day.isoformat() for day in fetch_dates]),
                "cache_path": "",
                "from_cache": False,
                "fallback_used": False,
                "warning": "",
                "available": True,
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)
    monkeypatch.setattr(main_module, "BACKTEST_MODE_ACTIVE", True)

    main_module.run_calculate_indicators("jquants", target.isoformat())

    payload = main_module.read_json(tmp_path / "data" / "processed" / "rookie_dealer_02_v2_6" / "indicators_2026-01-26.json")
    by_code = {item["code"]: item for item in payload["indicators"]}
    api_log = (tmp_path / "logs" / "jquants_api.log").read_text(encoding="utf-8")

    assert calls["count"] == 1
    assert payload["benchmark_source"] == "topix"
    assert by_code["1001"]["benchmark_source"] == "topix"
    assert by_code["1001"]["stock_return_5d"] is not None
    assert by_code["1001"]["benchmark_return_5d"] == 0.02
    assert by_code["1001"]["relative_strength_5d"] == 0.18
    assert by_code["1001"]["relative_strength_score"] > 0
    assert "endpoint=topix_prices" in api_log
    assert "plan=light" in api_log
    assert "cache_hit=false" in api_log
    assert "status=200" in api_log


def test_screening_preserves_relative_strength_pipeline_fields() -> None:
    result = screen_candidates(
        [
            {
                "code": "1001",
                "name": "Strong",
                "sector_name": "機械",
                "date": "2026-01-26",
                "open": 100,
                "high": 121,
                "low": 99,
                "close": 120,
                "volume": 10_000_000,
                "ma5": 110,
                "ma25": 100,
                "previous_close": 110,
                "previous_ma5": 105,
                "previous_ma25": 99,
                "rsi": 55,
                "volume_ratio": 3.0,
                "turnover_value": 1_200_000_000,
                "five_day_volatility": 0.05,
                "stock_return_5d": 0.2,
                "stock_return_10d": 0.2,
                "stock_return_20d": 0.2,
                "benchmark_source": "topix",
                "benchmark_return_5d": 0.02,
                "benchmark_return_10d": 0.03,
                "benchmark_return_20d": 0.04,
                "relative_strength_5d": 0.18,
                "relative_strength_10d": 0.17,
                "relative_strength_20d": 0.16,
                "relative_strength_score": 10,
                "topix_records_loaded": 35,
                "topix_api_calls": 1,
                "topix_cache_path": "/tmp/topix.json",
                "relative_strength_feature_enabled": True,
                "relative_strength_scoring_enabled": True,
                "relative_strength_benchmark_provider_called": True,
                "relative_strength_cache_exists": True,
                "relative_strength_calculated": True,
            }
        ],
        target_count=1,
    )

    candidate = result["candidates"][0]

    assert candidate["benchmark_source"] == "topix"
    assert candidate["relative_strength_5d"] == 0.18
    assert candidate["relative_strength_score"] == 10
    assert candidate["topix_records_loaded"] == 35
    assert candidate["relative_strength_benchmark_provider_called"] is True


def test_stale_relative_strength_indicator_cache_is_recalculated(monkeypatch, tmp_path) -> None:
    main_module._reset_jquants_api_session()
    config = load_profile("rookie_dealer_02_v2_6")
    config.setdefault("jquants", {})["plan"] = "light"
    config.setdefault("backtest", {})["indicator_mode"] = "minimal"
    target = date(2026, 1, 26)
    fetch_dates = main_module.previous_business_dates(target, 35)
    _write_prime_fixture(tmp_path)
    _write_price_history_fixture(tmp_path, fetch_dates)
    stale_path = tmp_path / "data" / "processed" / "rookie_dealer_02_v2_6" / "indicators_2026-01-26.json"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text(
        '{"indicator_mode":"minimal","relative_strength_enabled":true,"benchmark_source":"topix","indicators":[{"code":"1001","benchmark_source":"topix","relative_strength_score":0}]}',
        encoding="utf-8",
    )
    calls = {"count": 0}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_topix_prices_cached(self, *_args, **_kwargs):
            calls["count"] += 1
            return {
                "records": _topix_rows(latest_close=102, dates=[day.isoformat() for day in fetch_dates]),
                "cache_path": "",
                "from_cache": False,
                "fallback_used": False,
                "warning": "",
                "available": True,
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)
    monkeypatch.setattr(main_module, "BACKTEST_MODE_ACTIVE", True)

    main_module.ensure_indicators("jquants", target.isoformat())

    refreshed = main_module.read_json(stale_path)
    assert calls["count"] == 1
    assert refreshed["indicators"][0]["relative_strength_5d"] is not None


def test_v2_6_indicator_cache_hit_still_checks_topix_cache(monkeypatch, tmp_path) -> None:
    main_module._reset_jquants_api_session()
    config = load_profile("rookie_dealer_02_v2_6")
    config.setdefault("jquants", {})["plan"] = "light"
    config.setdefault("backtest", {})["indicator_mode"] = "minimal"
    target = date(2026, 1, 26)
    profile_path = tmp_path / "data" / "processed" / "rookie_dealer_02_v2_6" / "indicators_2026-01-26.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(
            {
                "indicator_mode": "minimal",
                "relative_strength_enabled": True,
                "benchmark_source": "topix",
                "indicators": [
                    {
                        "code": "1001",
                        "benchmark_source": "topix",
                        "stock_return_5d": 0.05,
                        "stock_return_10d": 0.08,
                        "stock_return_20d": 0.12,
                        "benchmark_return_5d": 0.01,
                        "benchmark_return_10d": 0.02,
                        "benchmark_return_20d": 0.03,
                        "relative_strength_5d": 0.04,
                        "relative_strength_10d": 0.06,
                        "relative_strength_20d": 0.09,
                        "relative_strength_score": 10,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = {"count": 0}

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_topix_prices_cached(self, *_args, **_kwargs):
            calls["count"] += 1
            return {
                "records": _topix_rows(latest_close=102),
                "cache_path": str(tmp_path / "data" / "cache" / "jquants" / "topix_prices" / "x.json"),
                "from_cache": False,
                "fallback_used": False,
                "warning": "",
                "available": True,
                "saved": True,
                "usable": True,
                "api_status": "200",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)
    monkeypatch.setattr(main_module, "BACKTEST_MODE_ACTIVE", True)

    main_module.ensure_indicators("jquants", target.isoformat())

    assert calls["count"] == 1
    assert "endpoint=topix_prices" in (tmp_path / "logs" / "jquants_api.log").read_text(encoding="utf-8")


def test_light_topix_benchmark_payload_does_not_emit_fallback_warning(monkeypatch, capsys) -> None:
    config = load_profile("rookie_dealer_02_v2_6")
    config.setdefault("jquants", {})["plan"] = "light"
    price_rows = _relative_strength_price_rows()
    fetch_dates = main_module.previous_business_dates(date(2026, 1, 26), 35)
    monkeypatch.setattr(
        main_module,
        "_load_topix_prices_for_period",
        lambda *_args, **_kwargs: {
            "records": _topix_rows(latest_close=102),
            "from_cache": False,
            "fallback_used": False,
            "warning": "",
            "available": True,
        },
    )

    payload = main_module._relative_strength_benchmark_payload(price_rows, date(2026, 1, 26), fetch_dates, config)

    assert payload["benchmark_source"] == "topix"
    assert "fallback benchmark" not in capsys.readouterr().out


def test_backtest_preloaded_topix_is_reused_without_daily_api_calls(monkeypatch, tmp_path) -> None:
    main_module._reset_jquants_api_session()
    config = load_profile("rookie_dealer_02_v2_6")
    config.setdefault("jquants", {})["plan"] = "light"
    calls = {"count": 0}
    fetch_dates = main_module.previous_business_dates(date(2026, 1, 26), 35)

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.last_request_metadata = {}

        def fetch_topix_prices_cached(self, *_args, **_kwargs):
            calls["count"] += 1
            return {
                "records": _topix_rows(latest_close=102, dates=[day.isoformat() for day in fetch_dates]),
                "cache_path": str(tmp_path / "data" / "cache" / "jquants" / "topix_prices" / "x.json"),
                "from_cache": False,
                "fallback_used": False,
                "warning": "",
                "available": True,
                "saved": True,
                "usable": True,
                "api_status": "200",
            }

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", FakeProvider)

    main_module._preload_light_api_context(config, date(2026, 1, 1), date(2026, 1, 26))
    first = main_module._relative_strength_benchmark_payload(_relative_strength_price_rows(), date(2026, 1, 23), fetch_dates, config)
    second = main_module._relative_strength_benchmark_payload(_relative_strength_price_rows(), date(2026, 1, 26), fetch_dates, config)

    assert calls["count"] == 1
    assert first["benchmark_source"] == "topix"
    assert second["benchmark_source"] == "topix"
    assert second["topix_records_loaded"] > 0


def test_scoring_storage_keeps_relative_strength_log_fields() -> None:
    candidate = {
        "code": "1001",
        "name": "Strong",
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
        "benchmark_source": "topix",
        "stock_return_5d": 0.05,
        "stock_return_10d": 0.08,
        "stock_return_20d": 0.12,
        "benchmark_return_5d": 0.01,
        "benchmark_return_10d": 0.02,
        "benchmark_return_20d": 0.03,
        "relative_strength_5d": 0.04,
        "relative_strength_10d": 0.06,
        "relative_strength_20d": 0.09,
        "relative_strength_score": 10,
    }
    config = load_profile("rookie_dealer_02_v2_6")

    scoring_log = score_real_candidates([candidate], "2026-03-06", config, "test")
    stored = main_module._scoring_log_for_storage(scoring_log, config)["scores"][0]

    for field in [
        "benchmark_source",
        "stock_return_5d",
        "stock_return_10d",
        "stock_return_20d",
        "benchmark_return_5d",
        "benchmark_return_10d",
        "benchmark_return_20d",
        "relative_strength_5d",
        "relative_strength_10d",
        "relative_strength_20d",
        "relative_strength_score",
    ]:
        assert field in stored
    assert stored["benchmark_source"] == "topix"
    assert stored["score_components"]["relative_strength_score"] == 10


def test_fast_analysis_omits_rejected_candidate_storage() -> None:
    scoring_log = {
        "scores": [
            {"code": "1001", "selected": True},
            {"code": "1002", "selected": False},
        ],
        "selected": [{"code": "1001", "selected": True}],
    }
    config = {"analysis": {"save_rejected_candidates": False}}

    storage_log = main_module._scoring_log_for_storage(scoring_log, config)

    assert [item["code"] for item in storage_log["scores"]] == ["1001"]
    assert storage_log["rejected_candidate_detail_saved"] is False


def test_score_components_sum_matches_total_for_v2_1_and_v2_6() -> None:
    candidate = {
        "code": "1001",
        "name": "Formula",
        "date": "2026-03-06",
        "close": 1200,
        "volume": 100000,
        "ma5": 1100,
        "ma25": 1000,
        "rsi": 55,
        "volume_ratio": 3.0,
        "turnover_value": 2_500_000_000,
        "five_day_volatility": 0.02,
        "fallback": False,
        "relative_strength_score": 7,
        "relative_strength_5d": 0.04,
        "relative_strength_10d": 0.06,
        "relative_strength_20d": 0.01,
    }
    for profile_id in ["rookie_dealer_02_v2_1", "rookie_dealer_02_v2_6"]:
        profile = load_profile(profile_id)
        item = score_real_candidates([candidate], "2026-03-06", profile, "test")["scores"][0]
        components = item["score_components"]

        assert components["matches_total_score"] is True
        assert components["component_total"] == item["total_score"]


def test_total_score_stays_within_expected_range_for_v2_6() -> None:
    profile = load_profile("rookie_dealer_02_v2_6")
    candidate = {
        "code": "1001",
        "name": "Maxish",
        "date": "2026-03-06",
        "close": 1200,
        "volume": 300000,
        "ma5": 1165,
        "ma25": 1131,
        "rsi": 55,
        "volume_ratio": 5.0,
        "turnover_value": 5_000_000_000,
        "five_day_volatility": 0.02,
        "candle_body_rate": 0.08,
        "upper_shadow_rate": 0.01,
        "lower_shadow_rate": 0.02,
        "close_position_in_range": 0.95,
        "candlestick_signals": [
            "bullish_candle",
            "strong_bullish_candle",
            "long_lower_shadow_support",
            "ma_reclaim",
            "volume_confirmed_breakout",
        ],
        "fallback": False,
        "relative_strength_score": 10,
        "relative_strength_5d": 0.04,
        "relative_strength_10d": 0.06,
        "relative_strength_20d": 0.09,
    }
    item = score_real_candidates([candidate], "2026-03-06", profile, "test")["scores"][0]

    assert item["technical_score"] <= 50
    assert "news" + "_score" not in item
    assert "financial_score" not in item
    assert item["relative_strength_score"] <= 10
    assert 0 <= item["total_score"] <= 60


def test_backtest_does_not_run_analyze_only_processing(monkeypatch, config_copy, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    portfolio = {"total_assets": 1_000_000, "safety_events": [], "date": "2026-01-05"}
    state = main_module.initial_live_paper_state(config_copy)

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)
    monkeypatch.setattr(main_module, "ensure_price_history_for_backtest", lambda *_args: None)
    monkeypatch.setattr(main_module, "available_cached_price_dates", lambda *_args: [date(2026, 1, 2), date(2026, 1, 5)])
    monkeypatch.setattr(main_module, "ensure_indicators", lambda *_args: None)
    monkeypatch.setattr(main_module, "ensure_market_context", lambda *_args: None)
    monkeypatch.setattr(main_module, "run_screen", lambda *_args: None)
    monkeypatch.setattr(main_module, "load_cached_prime_prices", lambda *_args: [])
    monkeypatch.setattr(
        main_module,
        "score_for_date",
        lambda *_args: {"scores": [], "selected": [], "candidate_count": 0, "selected_count": 0},
    )
    monkeypatch.setattr(main_module, "execute_real_data_paper_trade", lambda *_args: (state, portfolio, []))
    monkeypatch.setattr(main_module, "attach_commentary", lambda *_args: None)
    monkeypatch.setattr(main_module, "write_backtest_reflections", lambda *_args: tmp_path / "reflections.json")
    monkeypatch.setattr(
        main_module,
        "write_backtest_daily_markdown",
        lambda *_args: (tmp_path / "report.md", tmp_path / "article.md"),
    )
    monkeypatch.setattr(main_module, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "save_portfolio_snapshot", lambda *_args: None)
    monkeypatch.setattr(main_module, "save_trades", lambda *_args: None)
    monkeypatch.setattr(main_module, "save_pending_orders", lambda *_args: None)
    monkeypatch.setattr(main_module, "save_safety_events", lambda *_args: None)
    monkeypatch.setattr(
        main_module,
        "write_backtest_summary",
        lambda *_args: {
            "final_assets": 1_000_000,
            "cumulative_profit": 0,
            "report_markdown_path": str(tmp_path / "summary.md"),
            "report_json_path": str(tmp_path / "summary.json"),
            "rule_based_90d_report_path": str(tmp_path / "rule.md"),
        },
    )

    def fail_analyze(*_args, **_kwargs):
        raise AssertionError("analyze-only processing should not run during backtest")

    monkeypatch.setattr(main_module, "analyze_operation_data", fail_analyze)

    main_module.run_backtest("jquants", "2026-01-02", "2026-01-05")


def test_fast_analysis_skips_heavy_daily_report_generation(monkeypatch, config_copy, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    config_copy.setdefault("analysis", {})["save_backtest_daily_reports"] = False
    portfolio = {"total_assets": 1_000_000, "safety_events": [], "date": "2026-01-05"}
    state = main_module.initial_live_paper_state(config_copy)

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)
    monkeypatch.setattr(main_module, "ensure_price_history_for_backtest", lambda *_args: None)
    monkeypatch.setattr(main_module, "available_cached_price_dates", lambda *_args: [date(2026, 1, 2), date(2026, 1, 5)])
    monkeypatch.setattr(main_module, "ensure_indicators", lambda *_args: None)
    monkeypatch.setattr(main_module, "ensure_market_context", lambda *_args: None)
    monkeypatch.setattr(main_module, "run_screen", lambda *_args: None)
    monkeypatch.setattr(main_module, "load_cached_prime_prices", lambda *_args: [])
    monkeypatch.setattr(
        main_module,
        "score_for_date",
        lambda *_args: {"scores": [], "selected": [], "candidate_count": 0, "selected_count": 0},
    )
    monkeypatch.setattr(main_module, "execute_real_data_paper_trade", lambda *_args: (state, portfolio, []))
    monkeypatch.setattr(main_module, "attach_commentary", lambda *_args: None)
    monkeypatch.setattr(main_module, "write_json", lambda path, payload, **_kwargs: path.parent.mkdir(parents=True, exist_ok=True) or path.write_text("{}", encoding="utf-8"))
    monkeypatch.setattr(main_module, "save_portfolio_snapshot", lambda *_args: None)
    monkeypatch.setattr(main_module, "save_trades", lambda *_args: None)
    monkeypatch.setattr(main_module, "save_pending_orders", lambda *_args: None)
    monkeypatch.setattr(main_module, "save_safety_events", lambda *_args: None)
    monkeypatch.setattr(
        main_module,
        "write_backtest_summary",
        lambda *_args: {
            "final_assets": 1_000_000,
            "cumulative_profit": 0,
            "report_markdown_path": str(tmp_path / "summary.md"),
            "report_json_path": str(tmp_path / "summary.json"),
            "rule_based_90d_report_path": str(tmp_path / "rule.md"),
        },
    )

    def fail_daily_report(*_args, **_kwargs):
        raise AssertionError("heavy daily report generation should not run in fast-analysis")

    monkeypatch.setattr(main_module, "write_backtest_daily_markdown", fail_daily_report)

    main_module.run_backtest("jquants", "2026-01-02", "2026-01-05")


def test_summary_only_skips_backtest_daily_reports_articles_and_reflections(monkeypatch, config_copy, tmp_path, capsys) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    config_copy.setdefault("reporting", {})["generate_daily_markdown_in_backtest"] = True
    config_copy.setdefault("reporting", {})["generate_articles_in_backtest"] = True
    config_copy.setdefault("backtest", {})["indicator_min_history_days"] = 1
    portfolio = {"total_assets": 1_000_000, "safety_events": [], "date": "2026-01-05"}
    state = main_module.initial_live_paper_state(config_copy)

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "SUMMARY_ONLY_ACTIVE", True)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)
    monkeypatch.setattr(main_module, "ensure_price_history_for_backtest", lambda *_args: None)
    monkeypatch.setattr(main_module, "available_cached_price_dates", lambda *_args: [date(2026, 1, 2), date(2026, 1, 5)])
    monkeypatch.setattr(main_module, "ensure_indicators", lambda *_args: None)
    monkeypatch.setattr(main_module, "ensure_market_context", lambda *_args: None)
    monkeypatch.setattr(main_module, "run_screen", lambda *_args: None)
    monkeypatch.setattr(main_module, "load_cached_prime_prices", lambda *_args: [])
    monkeypatch.setattr(
        main_module,
        "score_for_date",
        lambda *_args: {"scores": [], "selected": [], "candidate_count": 0, "selected_count": 0},
    )
    monkeypatch.setattr(main_module, "execute_real_data_paper_trade", lambda *_args: (state, portfolio, []))
    monkeypatch.setattr(main_module, "attach_commentary", lambda *_args: None)
    monkeypatch.setattr(main_module, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "save_portfolio_snapshot", lambda *_args: None)
    monkeypatch.setattr(main_module, "save_trades", lambda *_args: None)
    monkeypatch.setattr(main_module, "save_pending_orders", lambda *_args: None)
    monkeypatch.setattr(main_module, "save_safety_events", lambda *_args: None)
    monkeypatch.setattr(
        main_module,
        "write_backtest_summary",
        lambda *_args: {
            "final_assets": 1_000_000,
            "cumulative_profit": 0,
            "report_markdown_path": str(tmp_path / "summary.md"),
            "report_json_path": str(tmp_path / "summary.json"),
            "rule_based_90d_report_path": str(tmp_path / "rule.md"),
        },
    )

    def fail_generation(*_args, **_kwargs):
        raise AssertionError("summary-only should skip daily report/article/reflection generation")

    monkeypatch.setattr(main_module, "write_backtest_reflections", fail_generation)
    monkeypatch.setattr(main_module, "write_backtest_daily_markdown", fail_generation)
    monkeypatch.setattr(main_module, "write_backtest_report_markdown", fail_generation)
    monkeypatch.setattr(main_module, "write_backtest_article_markdown", fail_generation)

    main_module.run_backtest("jquants", "2026-01-02", "2026-01-05")

    output = capsys.readouterr().out
    assert "reports/articles skipped: summary_only=true" in output
    assert "daily_reports_skipped_count: 1" in output
    assert "articles_skipped_count: 1" in output
    assert "reflections_skipped_count: 1" in output


def test_quiet_backtest_suppresses_day_step_detail(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_module, "QUIET_ACTIVE", True)
    monkeypatch.setattr(main_module, "PROGRESS_INTERVAL", 50)
    monkeypatch.setattr(main_module, "BACKTEST_MODE_ACTIVE", True)

    result = main_module._run_backtest_day_step(
        2,
        100,
        "2026-01-06",
        "score",
        lambda: print("noisy scoring internals") or {"ok": True},
        lambda: ["scoring candidates: 10"],
    )

    output = capsys.readouterr().out
    assert result == {"ok": True}
    assert "noisy scoring internals" not in output
    assert "scoring candidates" not in output
    assert "2026-01-06 score start" not in output


def _relative_strength_price_rows() -> list[dict]:
    rows = []
    dates = [f"2026-01-{day:02d}" for day in range(1, 27)]
    for code, latest_close in [("1001", 120), ("1002", 100)]:
        for target_date in dates:
            rows.append(
                {
                    "code": code,
                    "date": target_date,
                    "open": 100,
                    "high": max(100, latest_close),
                    "low": 95,
                    "close": latest_close if target_date == "2026-01-26" else 100,
                    "volume": 1000,
                }
            )
    return rows


def _write_prime_fixture(root) -> None:
    path = root / "data" / "raw" / "prime_stocks_jquants.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"stocks":[{"code":"1001","name":"Strong","sector_name":"機械"},{"code":"1002","name":"Flat","sector_name":"情報通信"}]}',
        encoding="utf-8",
    )


def _write_price_history_fixture(root, dates: list[date]) -> None:
    for target_date in dates:
        date_text = target_date.isoformat()
        prices = []
        for code, latest_close in [("1001", 120), ("1002", 100)]:
            close = latest_close if target_date == dates[-1] else 100
            prices.append(
                {
                    "code": code,
                    "date": date_text,
                    "open": 100,
                    "high": max(100, latest_close),
                    "low": 95,
                    "close": close,
                    "volume": 1000,
                }
            )
        path = root / "data" / "raw" / f"prices_{date_text}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"provider": "jquants", "date": date_text, "prices": prices}),
            encoding="utf-8",
        )


def _topix_rows(latest_close: float, dates: list[str] | None = None) -> list[dict]:
    dates = dates or [f"2026-01-{day:02d}" for day in range(1, 27)]
    latest_date = dates[-1]
    return [
        {
            "date": date_text,
            "open": 100,
            "high": latest_close,
            "low": 99,
            "close": latest_close if date_text == latest_date else 100,
        }
        for date_text in dates
    ]
