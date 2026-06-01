from __future__ import annotations

from datetime import date

import main as main_module


def _patch_minimal_backtest(monkeypatch, config: dict, tmp_path, trades: list[dict] | None = None) -> None:
    config["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    portfolio = {"total_assets": 1_000_000, "safety_events": [], "date": "2026-01-02"}
    state = main_module.initial_live_paper_state(config)
    trades = trades or []

    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "ensure_price_history_for_backtest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(main_module, "available_cached_price_dates", lambda *_args: [date(2026, 1, 2)])
    monkeypatch.setattr(main_module, "ensure_indicators", lambda *_args: None)
    monkeypatch.setattr(main_module, "ensure_market_context", lambda *_args: None)
    monkeypatch.setattr(main_module, "run_screen", lambda *_args: None)
    monkeypatch.setattr(
        main_module,
        "score_for_date",
        lambda *_args: {"scores": [], "selected": [], "candidate_count": 0, "selected_count": 0},
    )
    monkeypatch.setattr(main_module, "execute_real_data_paper_trade", lambda *_args: (state, portfolio, trades))
    monkeypatch.setattr(main_module, "attach_commentary", lambda *_args: None)
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


def test_backtest_does_not_generate_article_markdown_by_default(monkeypatch, config_copy, tmp_path) -> None:
    config_copy.setdefault("reporting", {})["generate_articles_in_backtest"] = False
    config_copy.setdefault("reporting", {})["generate_daily_markdown_in_backtest"] = False
    _patch_minimal_backtest(monkeypatch, config_copy, tmp_path)

    monkeypatch.setattr(
        main_module,
        "write_backtest_daily_markdown",
        lambda *_args: (_ for _ in ()).throw(AssertionError("daily markdown should not be generated")),
    )
    monkeypatch.setattr(
        main_module,
        "generate_note_article",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("article should not be generated")),
    )
    monkeypatch.setattr(
        main_module,
        "write_backtest_reflections",
        lambda *_args: (_ for _ in ()).throw(AssertionError("reflection detail should not be generated")),
    )

    main_module.run_backtest("jquants", "2026-01-02", "2026-01-02")

    assert not (tmp_path / "reports" / "articles" / "backtests").exists()


def test_indicator_fetch_start_is_before_trade_start(config_copy) -> None:
    config_copy.setdefault("backtest", {})["indicator_fetch_lookback_days"] = 180
    config_copy.setdefault("jquants", {})["earliest_supported_date"] = {"free": "2024-01-01"}

    trade_start = date(2024, 7, 22)
    fetch_start = main_module._indicator_fetch_start_date(trade_start, config_copy)

    assert fetch_start < trade_start


def test_backtest_fetches_prices_from_indicator_lookback(monkeypatch, config_copy, tmp_path) -> None:
    config_copy.setdefault("backtest", {})["indicator_fetch_lookback_days"] = 180
    config_copy.setdefault("backtest", {})["indicator_min_history_days"] = 2
    config_copy.setdefault("jquants", {})["earliest_supported_date"] = {"free": "2024-01-01"}
    _patch_minimal_backtest(monkeypatch, config_copy, tmp_path)
    captured: dict[str, object] = {"indicator_dates": []}

    def fake_ensure_price_history(
        _provider: str,
        fetch_start: date,
        fetch_end: date,
        price_fetch_min_start: date | None = None,
    ) -> dict[str, object]:
        captured["fetch_start"] = fetch_start
        captured["fetch_end"] = fetch_end
        captured["price_fetch_min_start"] = price_fetch_min_start
        return {
            "price_fetch_requested_start": fetch_start.isoformat(),
            "price_fetch_clamped_start": (price_fetch_min_start or fetch_start).isoformat(),
            "first_fetch_attempt_date": (price_fetch_min_start or fetch_start).isoformat(),
        }

    def fake_available_cached_price_dates(start: date, _end: date) -> list[date]:
        if start < date(2026, 1, 2):
            return [date(2025, 12, 30), date(2026, 1, 2)]
        return [date(2026, 1, 2)]

    monkeypatch.setattr(main_module, "ensure_price_history_for_backtest", fake_ensure_price_history)
    monkeypatch.setattr(main_module, "available_cached_price_dates", fake_available_cached_price_dates)
    monkeypatch.setattr(main_module, "ensure_indicators", lambda _provider, target_date: captured["indicator_dates"].append(target_date))

    main_module.run_backtest("jquants", "2026-01-02", "2026-01-02")

    assert captured["fetch_start"] < date(2026, 1, 2)
    assert captured["price_fetch_min_start"] == date(2026, 1, 2)
    assert captured["fetch_end"] == date(2026, 1, 2)
    assert captured["indicator_dates"] == ["2026-01-02"]


def test_price_history_fetch_does_not_short_circuit_on_processed_indicators(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    main_module.write_json(processed_dir / "indicators_2026-01-05.json", {"indicators": []})
    calls = {"list_stocks": 0}

    def fake_run_list_stocks(_provider_name: str) -> None:
        calls["list_stocks"] += 1
        raise SystemExit("stop after proving price cache was checked")

    monkeypatch.setattr(main_module, "run_list_stocks", fake_run_list_stocks)
    try:
        main_module.ensure_price_history_for_backtest("jquants", date(2026, 1, 5), date(2026, 1, 5))
    except SystemExit:
        pass
    else:
        raise AssertionError("price history fetch should not return only because processed indicators exist")

    assert calls["list_stocks"] == 1


def test_backtest_indicator_recalculation_fetches_missing_price_history(monkeypatch, config_copy, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)
    monkeypatch.setattr(
        main_module,
        "_allowed_stock_master_by_code",
        lambda _config: {"1001": {"name": "Test", "sector_name": "機械", "section": "TSEPrime"}},
    )
    calls = {"loads": 0, "fetch": None}

    def fake_load_cached_price_history(fetch_dates):
        calls["loads"] += 1
        if calls["loads"] == 1:
            return []
        return [{"date": "2026-01-05", "code": "1001", "close": 100, "volume": 1000}]

    def fake_ensure_price_history(_provider, fetch_start, fetch_end, price_fetch_min_start=None):
        calls["fetch"] = (fetch_start, fetch_end, price_fetch_min_start)
        return {}

    monkeypatch.setattr(main_module, "load_cached_price_history", fake_load_cached_price_history)
    monkeypatch.setattr(main_module, "ensure_price_history_for_backtest", fake_ensure_price_history)
    monkeypatch.setattr(
        main_module,
        "calculate_indicators",
        lambda *_args, **_kwargs: ([{"date": "2026-01-05", "code": "1001", "close": 100}], 0),
    )
    monkeypatch.setattr(main_module, "write_json", lambda path, payload: path.parent.mkdir(parents=True, exist_ok=True) or path.write_text("{}", encoding="utf-8"))

    previous = main_module.BACKTEST_MODE_ACTIVE
    main_module.BACKTEST_MODE_ACTIVE = True
    try:
        main_module.run_calculate_indicators("jquants", "2026-01-05")
    finally:
        main_module.BACKTEST_MODE_ACTIVE = previous

    assert calls["loads"] == 2
    assert calls["fetch"] is not None
    assert calls["fetch"][1] == date(2026, 1, 5)


def test_price_history_fetch_clamps_fetch_dates_before_enumeration(monkeypatch, config_copy, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    config_copy.setdefault("jquants", {})["plan"] = "light"
    config_copy.setdefault("jquants", {})["earliest_supported_date"] = {"light": "2021-05-01"}
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    main_module.write_json(raw_dir / "prime_stocks_jquants.json", {"stocks": [{"code": "1001"}]})
    captured: dict[str, object] = {}

    class FakeProvider:
        fetch_stats = {}

    def fake_fetch_price_history(_provider, _end_date, _codes, **kwargs):
        captured["fetch_dates"] = kwargs["fetch_dates"]
        return []

    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)
    monkeypatch.setattr(main_module, "load_cached_prime_prices", lambda _date: None)
    monkeypatch.setattr(main_module, "load_no_data_days_cache", lambda: {})
    monkeypatch.setattr(main_module, "load_unsupported_days_cache", lambda: {})
    monkeypatch.setattr(main_module, "no_data_cache_entry", lambda *_args: None)
    monkeypatch.setattr(main_module, "unsupported_cache_entry", lambda *_args: None)
    monkeypatch.setattr(main_module, "JQuantsDataProvider", lambda *_args, **_kwargs: FakeProvider())
    monkeypatch.setattr(main_module, "fetch_price_history", fake_fetch_price_history)
    monkeypatch.setattr(main_module, "_print_fetch_statistics", lambda _provider: None)
    monkeypatch.setattr(main_module, "available_cached_price_dates", lambda *_args: [])

    audit = main_module.ensure_price_history_for_backtest(
        "jquants",
        date(2021, 5, 1),
        date(2021, 6, 4),
        price_fetch_min_start=date(2021, 5, 31),
    )

    fetch_dates = captured["fetch_dates"]
    assert fetch_dates[0] == date(2021, 5, 31)
    assert audit["price_fetch_requested_start"] == "2021-05-01"
    assert audit["price_fetch_clamped_start"] == "2021-05-31"
    assert audit["first_fetch_attempt_date"] == "2021-05-31"


def test_backtest_skips_day_when_indicator_history_is_insufficient(monkeypatch, config_copy, tmp_path) -> None:
    config_copy.setdefault("backtest", {})["indicator_fetch_lookback_days"] = 180
    config_copy.setdefault("backtest", {})["indicator_min_history_days"] = 60
    _patch_minimal_backtest(monkeypatch, config_copy, tmp_path)
    calls = {"indicators": 0}

    monkeypatch.setattr(main_module, "available_cached_price_dates", lambda *_args: [date(2026, 1, 2)])
    monkeypatch.setattr(main_module, "ensure_indicators", lambda *_args: calls.__setitem__("indicators", calls["indicators"] + 1))

    main_module.run_backtest("jquants", "2026-01-02", "2026-01-02")

    assert calls["indicators"] == 0


def test_backtest_date_range_audit_detects_price_coverage_gap(monkeypatch, config_copy, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)

    audit = main_module.build_backtest_date_range_audit(
        config=config_copy,
        requested_start_date=date(2021, 5, 1),
        requested_end_date=date(2026, 5, 30),
        effective_trade_start_date=date(2021, 5, 1),
        effective_trade_end_date=date(2026, 5, 30),
        indicator_fetch_start_date=date(2021, 5, 1),
        price_history_dates=[date(2024, 9, 2), date(2026, 3, 6)],
        trading_dates=[date(2024, 9, 2), date(2026, 3, 6)],
        processed_dates=["2024-09-02", "2026-03-06"],
        skipped_days=[],
        all_trades=[{"entry_date": "2024-10-01", "exit_date": "2026-03-06"}],
    )

    assert audit["effective_trade_end_date"] == "2026-05-30"
    assert audit["last_price_date"] == "2026-03-06"
    assert audit["last_processed_day"] == "2026-03-06"
    assert audit["raw_price_last_date"] == "2026-03-06"
    assert audit["processed_last_date"] == "2026-03-06"
    assert audit["missing_processed_dates_count"] == 0
    assert audit["target_trading_days_source"] == "raw_price_cache"
    assert audit["data_coverage"]["prices"]["coverage_ok"] is False
    assert audit["backtest_execution_audit"]["status"] == "ERROR"
    assert audit["backtest_execution_audit"]["last_processed_day"] == "2026-03-06"
    assert audit["backtest_execution_audit"]["date_range_limited_reason"] == "processed_ends_before_expected_processing_end_date"
    assert audit["effective_range_warning"]


def test_cached_jquants_price_files_are_used_as_backtest_trading_days(monkeypatch, config_copy, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    raw_dir = tmp_path / "data" / "raw"
    cache_dir = tmp_path / "data" / "cache" / "jquants" / "prices"
    raw_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    main_module.write_json(raw_dir / "prime_stocks_jquants.json", {"stocks": [{"code": "1001"}]})
    main_module.write_json(
        cache_dir / "2026-05-29.json",
        {
            "records": [
                {"Date": "2026-05-29", "Code": "1001", "O": 100, "H": 110, "L": 95, "C": 105, "Vo": 1000},
                {"Date": "2026-05-29", "Code": "9999", "O": 200, "H": 210, "L": 195, "C": 205, "Vo": 2000},
            ]
        },
    )

    dates = main_module.available_cached_price_dates(date(2026, 5, 29), date(2026, 5, 29))
    rows = main_module.load_cached_prime_prices(date(2026, 5, 29))

    assert dates == [date(2026, 5, 29)]
    assert rows == [
        {
            "code": "1001",
            "date": "2026-05-29",
            "open": 100,
            "high": 110,
            "low": 95,
            "close": 105,
            "volume": 1000,
            "section": "TSEPrime",
            "market_section": "TSEPrime",
            "listing_market": "TSEPrime",
        }
    ]


def test_screen_writes_empty_candidates_when_indicators_are_empty(monkeypatch, config_copy, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)
    monkeypatch.setattr(main_module, "save_screening_results", lambda *_args: None)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    main_module.write_json(
        processed_dir / "indicators_2026-05-29.json",
        {"date": "2026-05-29", "skip_reason": "insufficient_history", "indicators": []},
    )

    main_module.run_screen("jquants", "2026-05-29")

    output = tmp_path / "data" / "processed" / main_module.profile_id_from(config_copy) / "candidates_2026-05-29.json"
    payload = main_module.read_json(output)
    assert payload["candidate_count"] == 0
    assert payload["candidates"] == []


def test_score_writes_empty_scored_candidates_when_candidates_are_empty(monkeypatch, config_copy, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)
    monkeypatch.setattr(main_module, "load_market_context_for_date", lambda *_args: {})
    monkeypatch.setattr(main_module, "run_ai_decision_if_enabled", lambda *_args: None)
    monkeypatch.setattr(main_module, "save_scoring_results", lambda *_args: None)
    monkeypatch.setattr(main_module, "write_daily_ai_dataset", lambda *_args: None)
    candidates_dir = tmp_path / "data" / "processed" / main_module.profile_id_from(config_copy)
    candidates_dir.mkdir(parents=True)
    main_module.write_json(
        candidates_dir / "candidates_2026-05-29.json",
        {"date": "2026-05-29", "candidate_count": 0, "candidates": []},
    )

    main_module.run_score("jquants", "2026-05-29")

    output = candidates_dir / "scored_candidates_2026-05-29.json"
    payload = main_module.read_json(output)
    assert payload["candidate_count"] == 0
    assert payload["scored_count"] == 0
    assert payload["selected_count"] == 0
    assert payload["scores"] == []


def test_processed_data_audit_reports_stage_gaps(monkeypatch, config_copy, tmp_path) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    profile_dir = tmp_path / "data" / "processed" / main_module.profile_id_from(config_copy)
    profile_dir.mkdir(parents=True)
    main_module.write_json(profile_dir / "indicators_2026-05-28.json", {"indicators": []})
    main_module.write_json(profile_dir / "indicators_2026-05-29.json", {"indicators": []})
    main_module.write_json(profile_dir / "candidates_2026-05-28.json", {"candidates": []})
    main_module.write_json(profile_dir / "scored_candidates_2026-05-28.json", {"scores": []})

    audit = main_module.build_processed_data_audit(
        config_copy,
        [date(2026, 5, 28), date(2026, 5, 29)],
    )

    assert audit["indicators_last_date"] == "2026-05-29"
    assert audit["candidates_last_date"] == "2026-05-28"
    assert audit["scored_candidates_last_date"] == "2026-05-28"
    assert audit["indicators_first_date"] == "2026-05-28"
    assert audit["candidates_first_date"] == "2026-05-28"
    assert audit["scored_candidates_first_date"] == "2026-05-28"
    assert audit["dates_with_indicators_but_no_candidates"] == ["2026-05-29"]


def test_backtest_execution_audit_is_ok_when_last_business_day_is_processed() -> None:
    audit = main_module.build_backtest_execution_audit(
        processed_dates=["2026-05-29"],
        skipped_days=[],
        trading_dates=[date(2026, 5, 29)],
        expected_trading_days=[date(2026, 5, 29)],
        processed_data_audit={
            "indicators_first_date": "2026-05-29",
            "indicators_last_date": "2026-05-29",
            "candidates_first_date": "2026-05-29",
            "candidates_last_date": "2026-05-29",
            "scored_candidates_first_date": "2026-05-29",
            "scored_candidates_last_date": "2026-05-29",
            "dates_with_indicators_but_no_candidates_count": 0,
            "dates_with_candidates_but_no_scored_count": 0,
        },
        all_trades=[{"entry_date": "2026-05-29"}],
        effective_trade_end_date=date(2026, 5, 30),
    )

    assert audit["status"] == "OK"
    assert audit["effective_end_date"] == "2026-05-30"
    assert audit["expected_processing_end_date"] == "2026-05-29"
    assert audit["date_range_limited_reason"] == "none"


def test_backtest_execution_audit_lines_show_error() -> None:
    lines = main_module._backtest_execution_audit_lines(
        {
            "status": "ERROR",
            "last_processed_day": "2026-03-06",
            "expected_processing_end_date": "2026-05-29",
            "date_range_limited_reason": "processed_ends_before_expected_processing_end_date",
        }
    )

    assert any(line.startswith("- ERROR:") for line in lines)


def test_backtest_summary_markdown_separates_requested_and_effective_period(config_copy) -> None:
    summary = {
        "start_date": "2021-05-01",
        "end_date": "2026-05-30",
        "date_resolution": {
            "requested_start_date": "2021-05-01",
            "requested_end_date": "2026-05-30",
            "effective_start_date": "2021-05-01",
            "effective_end_date": "2026-05-30",
            "start_date_source": "cli",
            "end_date_source": "cli",
        },
        "date_range_audit": {
            "requested_start_date": "2021-05-01",
            "requested_end_date": "2026-05-30",
            "effective_trade_start_date": "2021-05-01",
            "effective_trade_end_date": "2026-05-30",
            "indicator_fetch_start_date": "2021-05-01",
            "first_price_date": "2024-09-02",
            "last_price_date": "2026-03-06",
            "first_trading_day": "2024-09-02",
            "last_trading_day": "2026-03-06",
            "target_trading_days": 1000,
            "processed_days": 397,
            "skipped_days": 0,
            "last_processed_day": "2026-03-06",
            "first_trade_date": "2024-10-01",
            "last_trade_date": "2026-03-06",
            "data_coverage": {
                "prices": {
                    "requested_end_date": "2026-05-30",
                    "latest_available_price_date": "2026-03-06",
                    "coverage_ok": False,
                    "warning": "price data ends before requested_end_date",
                }
            },
            "effective_range_warning": "processed days end before requested_end_date",
            "hardcoded_date_audit": {"target": "2026-03-06", "match_count": 0, "matches_sample": []},
            "backtest_execution_audit": {
                "status": "ERROR",
                "first_processed_day": "2024-09-02",
                "last_processed_day": "2026-03-06",
                "processed_days": 397,
                "skipped_days": 0,
                "first_indicator_date": "2024-09-02",
                "last_indicator_date": "2026-03-06",
                "first_candidate_date": "2024-09-02",
                "last_candidate_date": "2026-03-06",
                "first_scored_candidate_date": "2024-09-02",
                "last_scored_candidate_date": "2026-03-06",
                "first_trade_date": "2024-10-01",
                "last_trade_date": "2026-03-06",
                "target_trading_days": 1000,
                "actual_trading_days": 397,
                "effective_end_date": "2026-05-30",
                "expected_processing_end_date": "2026-05-29",
                "date_range_limited_reason": "processed_ends_before_expected_processing_end_date",
            },
            "backtest_coverage_audit": {
                "requested_start_date": "2021-05-01",
                "requested_end_date": "2026-05-30",
                "first_price_date": "2024-09-02",
                "last_price_date": "2026-03-06",
                "first_indicator_date": "2024-09-02",
                "last_indicator_date": "2026-03-06",
                "first_candidate_date": "2024-09-02",
                "last_candidate_date": "2026-03-06",
                "first_trade_date": "2024-10-01",
                "last_trade_date": "2026-03-06",
                "candidate_days": 397,
                "trade_days": 100,
                "price_days": 397,
                "expected_business_days": 1000,
                "coverage_ratio": 0.397,
                "coverage_warning": "requested_start_date is earlier than first_price_date; historical price coverage incomplete",
                "coverage_warnings": [
                    "requested_start_date is earlier than first_price_date",
                    "historical price coverage incomplete",
                ],
            },
        },
        "initial_capital": 1_000_000,
        "final_assets": 1_000_000,
        "cumulative_profit": 0,
        "cumulative_profit_rate": 0,
        "gross_cumulative_profit": 0,
        "estimated_tax_total": 0,
        "total_commission": 0,
        "net_cumulative_profit": 0,
        "net_cumulative_profit_rate": 0,
        "win_rate": None,
        "profit_factor": None,
        "closed_trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "excluded_order_event_count": 0,
        "total_trades": 0,
        "max_drawdown": 0,
        "take_profit_count": 0,
        "stop_loss_count": 0,
        "max_holding_exit_count": 0,
        "no_trade_days": 0,
        "selected_count_total": 0,
        "dealer_comment": "",
        "best_trade": None,
        "worst_trade": None,
        "daily_asset_curve": [],
    }

    markdown = main_module.render_backtest_summary_markdown(summary, config_copy)

    assert "- requested_period: 2021-05-01 to 2026-05-30" in markdown
    assert "- effective_period: 2024-09-02 to 2026-03-06" in markdown
    assert "- prices.coverage_ok: false" in markdown
    assert "## Backtest Coverage Audit" in markdown
    assert "- first_price_date: 2024-09-02" in markdown
    assert "- first_candidate_date: 2024-09-02" in markdown
    assert "- coverage_ratio: 39.70%" in markdown
    assert "- coverage_warning: requested_start_date is earlier than first_price_date; historical price coverage incomplete" in markdown
    assert "## Backtest Execution Audit" in markdown
    assert "- status: ERROR" in markdown
    assert "- last_candidate_date: 2026-03-06" in markdown


def test_date_resolution_effective_start_uses_first_available_price() -> None:
    resolved = main_module._date_resolution_with_coverage(
        {
            "requested_start_date": "2021-05-01",
            "requested_end_date": "2026-05-30",
            "effective_start_date": "2021-05-01",
            "effective_end_date": "2026-05-30",
        },
        {
            "backtest_coverage_audit": {
                "first_price_date": "2023-12-14",
            }
        },
    )

    assert resolved["effective_start_date"] == "2023-12-14"
    assert resolved["effective_start_date_source"] == "price_coverage"


def test_backtest_integrity_audit_marks_realistic_entry_but_experimental_risks(config_copy: dict, tmp_path) -> None:
    config_copy.setdefault("backtest", {})["entry_timing"] = "next_business_day_open"
    trades = [
        {
            "action": "BUY",
            "order_status": "FILLED",
            "signal_date": "2026-01-05",
            "entry_date": "2026-01-06",
            "entry_price": 1030.0,
            "entry_price_source": "open",
            "signal_close_price": 1000.0,
            "entry_open_price": 1030.0,
            "entry_gap_rate": 0.03,
        }
    ]

    audit = main_module.build_backtest_integrity_audit(config_copy, trades, tmp_path, {})

    checks = audit["checks"]
    assert checks["same_day_execution"]["status"] == "OK"
    assert checks["signal_date_entry_date_separated"]["status"] == "OK"
    assert checks["entry_price_source"]["status"] == "OK"
    assert checks["same_day_cash_reuse"]["status"] == "WARN"
    assert checks["survivorship_bias_risk"]["status"] == "WARN"
    assert audit["evaluation_label"] == "experimental"


def test_backtest_integrity_audit_detects_same_day_close_as_warning(config_copy: dict, tmp_path) -> None:
    config_copy.setdefault("backtest", {})["entry_timing"] = "same_day_close"
    trades = [
        {
            "action": "BUY",
            "order_status": "FILLED",
            "signal_date": "2026-01-05",
            "entry_date": "2026-01-05",
            "entry_price": 1000.0,
            "entry_price_source": "close",
            "signal_close_price": 1000.0,
            "entry_open_price": 990.0,
            "entry_gap_rate": -0.01,
        }
    ]

    audit = main_module.build_backtest_integrity_audit(config_copy, trades, tmp_path, {})

    assert audit["checks"]["same_day_execution"]["status"] == "WARN"
    assert audit["checks"]["signal_date_entry_date_separated"]["status"] == "WARN"
    assert audit["evaluation_label"] == "experimental"


def test_backtest_integrity_audit_lines_include_required_items(config_copy: dict, tmp_path) -> None:
    audit = main_module.build_backtest_integrity_audit(config_copy, [], tmp_path, {})
    lines = main_module._backtest_integrity_audit_lines(audit)

    assert any(line.startswith("- same_day_execution:") for line in lines)
    assert any(line.startswith("- investor_context_pubdate_safe:") for line in lines)
    assert any(line.startswith("- survivorship_bias_risk:") for line in lines)
    assert any("experimental" in line for line in lines)


def test_execution_model_stats_are_rendered() -> None:
    model = {
        "signal_timing": "after_close",
        "entry_timing": "next_business_day_open",
        "entry_price_source": "open",
        "same_day_execution": False,
    }
    stats = main_module._backtest_execution_model_stats(
        [
            {
                "action": "BUY",
                "order_status": "FILLED",
                "signal_date": "2026-05-22",
                "entry_date": "2026-05-25",
                "entry_gap_rate": 0.0123,
            }
        ],
        model,
    )
    model.update(stats)
    lines = main_module._backtest_execution_model_lines(model)

    assert "- signal_date_entry_date_separated: true" in lines
    assert "- signal_entry_same_day_count: 0" in lines
    assert "- signal_entry_next_day_count: 1" in lines
    assert "- signal_entry_gap_average: 1.23%" in lines


def test_run_daily_report_writer_generates_daily_article(config_copy, monkeypatch, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)

    report_path, article_path = main_module.write_real_daily_markdown(
        "2026-03-06",
        {
            "portfolio_summary": {
                "total_assets": 1_000_000,
                "daily_profit": 0,
                "cumulative_profit": 0,
                "cumulative_profit_rate": 0,
                "gross_cumulative_profit": 0,
                "net_cumulative_profit": 0,
                "estimated_tax_total": 0,
                "total_commission": 0,
                "win_rate": None,
                "max_drawdown": 0,
                "safety_events": [],
                "date": "2026-03-06",
            },
            "state": main_module.initial_live_paper_state(config_copy),
            "trades": [],
            "safety_events": [],
        },
        {"scores": []},
    )

    assert report_path.exists()
    assert article_path.exists()
    assert "reports/articles/daily/2026/03" in str(article_path)


def test_publish_article_uses_date_partitioned_article_path(config_copy, monkeypatch, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config_copy)

    draft_path = main_module._daily_article_path(config_copy, "2026-03-06", main_module.profile_id_from(config_copy))
    main_module.write_text(draft_path, "# Draft\n\nbody")

    main_module.run_publish_article("2026-03-06", "https://note.com/example")

    published_path = main_module._published_article_path(config_copy, "2026-03-06", main_module.profile_id_from(config_copy))
    assert published_path.exists()
    assert not draft_path.exists()
