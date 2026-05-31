from __future__ import annotations

from datetime import date

import main as main_module
import indicators as indicators_module
from benchmark_provider import build_relative_strength_benchmark
from data_provider import JQuantsDataProvider
from indicators import calculate_indicators
from profile_loader import load_profile
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
    portfolio = {"total_assets": 1_000_000, "safety_events": [], "date": "2026-01-02"}
    state = main_module.initial_live_paper_state(config_copy)

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)
    monkeypatch.setattr(main_module, "ensure_price_history_for_backtest", lambda *_args: None)
    monkeypatch.setattr(main_module, "available_cached_price_dates", lambda *_args: [date(2026, 1, 2)])
    monkeypatch.setattr(main_module, "ensure_indicators", lambda *_args: None)
    monkeypatch.setattr(main_module, "ensure_market_context", lambda *_args: None)
    monkeypatch.setattr(main_module, "run_screen", lambda *_args: None)
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

    main_module.run_backtest("jquants", "2026-01-02", "2026-01-02")


def test_fast_analysis_skips_heavy_daily_report_generation(monkeypatch, config_copy, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    config_copy.setdefault("analysis", {})["save_backtest_daily_reports"] = False
    portfolio = {"total_assets": 1_000_000, "safety_events": [], "date": "2026-01-02"}
    state = main_module.initial_live_paper_state(config_copy)

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)
    monkeypatch.setattr(main_module, "ensure_price_history_for_backtest", lambda *_args: None)
    monkeypatch.setattr(main_module, "available_cached_price_dates", lambda *_args: [date(2026, 1, 2)])
    monkeypatch.setattr(main_module, "ensure_indicators", lambda *_args: None)
    monkeypatch.setattr(main_module, "ensure_market_context", lambda *_args: None)
    monkeypatch.setattr(main_module, "run_screen", lambda *_args: None)
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

    main_module.run_backtest("jquants", "2026-01-02", "2026-01-02")


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


def _topix_rows(latest_close: float) -> list[dict]:
    dates = [f"2026-01-{day:02d}" for day in range(1, 27)]
    return [
        {
            "date": date_text,
            "open": 100,
            "high": latest_close,
            "low": 99,
            "close": latest_close if date_text == "2026-01-26" else 100,
        }
        for date_text in dates
    ]
