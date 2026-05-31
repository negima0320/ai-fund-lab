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

    def fake_ensure_price_history(_provider: str, fetch_start: date, fetch_end: date) -> None:
        captured["fetch_start"] = fetch_start
        captured["fetch_end"] = fetch_end

    def fake_available_cached_price_dates(start: date, _end: date) -> list[date]:
        if start < date(2026, 1, 2):
            return [date(2025, 12, 30), date(2026, 1, 2)]
        return [date(2026, 1, 2)]

    monkeypatch.setattr(main_module, "ensure_price_history_for_backtest", fake_ensure_price_history)
    monkeypatch.setattr(main_module, "available_cached_price_dates", fake_available_cached_price_dates)
    monkeypatch.setattr(main_module, "ensure_indicators", lambda _provider, target_date: captured["indicator_dates"].append(target_date))

    main_module.run_backtest("jquants", "2026-01-02", "2026-01-02")

    assert captured["fetch_start"] < date(2026, 1, 2)
    assert captured["fetch_end"] == date(2026, 1, 2)
    assert captured["indicator_dates"] == ["2026-01-02"]


def test_backtest_skips_day_when_indicator_history_is_insufficient(monkeypatch, config_copy, tmp_path) -> None:
    config_copy.setdefault("backtest", {})["indicator_fetch_lookback_days"] = 180
    config_copy.setdefault("backtest", {})["indicator_min_history_days"] = 60
    _patch_minimal_backtest(monkeypatch, config_copy, tmp_path)
    calls = {"indicators": 0}

    monkeypatch.setattr(main_module, "available_cached_price_dates", lambda *_args: [date(2026, 1, 2)])
    monkeypatch.setattr(main_module, "ensure_indicators", lambda *_args: calls.__setitem__("indicators", calls["indicators"] + 1))

    main_module.run_backtest("jquants", "2026-01-02", "2026-01-02")

    assert calls["indicators"] == 0


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
