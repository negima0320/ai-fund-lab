from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import main as main_module


def test_validate_config_passes_current_baseline() -> None:
    result = main_module.build_config_validation(
        "rookie_dealer_02_v2_1",
        runtime_settings={
            "profile_id": "rookie_dealer_02_v2_1",
            "provider": "jquants",
            "jquants_plan": "free",
            "broker_mode": "paper",
            "auto_order_enabled": False,
            "sources": {},
        },
    )

    assert result["fail_count"] == 0
    assert result["status"] in {"OK", "OK_WITH_WARNINGS"}


def test_validate_config_detects_stale_score_settings() -> None:
    profile_config = {
        "profile_id": "bad_profile",
        "selection": {"min_score": 45},
        "scoring": {
            "total_score_formula": "technical_score + base_score",
            "base_score": 25,
            "financial_score": 10,
        },
    }

    result = main_module.build_config_validation(
        "bad_profile",
        runtime_settings={
            "profile_id": "bad_profile",
            "provider": "jquants",
            "jquants_plan": "free",
            "broker_mode": "paper",
            "auto_order_enabled": False,
            "sources": {},
        },
        registry={"profiles": {}},
        operation_schedule={"safety": {"require_manual_approval": True, "forbid_live_auto_order": True}},
        profile_config=profile_config,
    )

    stale_check = next(item for item in result["checks"] if item["name"].endswith(".stale_score_components"))
    formula_check = next(item for item in result["checks"] if item["name"].endswith(".score_formula_terms"))
    assert stale_check["status"] == "FAIL"
    assert formula_check["status"] == "FAIL"


def test_validate_config_warns_for_light_profile_on_free_plan() -> None:
    result = main_module.build_config_validation(
        "rookie_dealer_02_v2_6",
        runtime_settings={
            "profile_id": "rookie_dealer_02_v2_6",
            "provider": "jquants",
            "jquants_plan": "free",
            "broker_mode": "paper",
            "auto_order_enabled": False,
            "sources": {"jquants_plan": "cli"},
        },
    )

    messages = [item["message"] for item in result["checks"] if item["status"] == "WARN"]
    assert any("requires light but current plan is free" in message for message in messages)
    assert any("missing capabilities have fallback" in message for message in messages)


def test_validate_config_warns_only_for_real_registry_profile_feature_mismatch() -> None:
    registry = {
        "profiles": {
            "example": {
                "role": "experiment",
                "required_plan": "free",
                "compare_to": "baseline",
                "features": {"financial_context": True},
            },
            "baseline": {
                "role": "baseline",
                "required_plan": "free",
                "features": {"financial_context": False},
            },
        }
    }
    profile_config = {
        "profile_id": "example",
        "profile_name": "Example",
        "features": {"financial_context": False},
        "selection": {"min_score": 45},
        "scoring": {"total_score_formula": "technical_score + market_context_score + penalty_score"},
    }

    result = main_module.build_config_validation(
        "example",
        runtime_settings={
            "profile_id": "example",
            "provider": "jquants",
            "jquants_plan": "free",
            "broker_mode": "paper",
            "auto_order_enabled": False,
            "sources": {},
        },
        registry=registry,
        operation_schedule={"safety": {"require_manual_approval": True, "forbid_live_auto_order": True}},
        profile_config=profile_config,
    )

    assert any(item["status"] == "WARN" and item["name"].endswith(".financial_context_registry_mismatch") for item in result["checks"])


def test_validate_config_accepts_data_only_feature() -> None:
    profile_config = {
        "profile_id": "example",
        "profile_name": "Example",
        "features": {"financial_context": True},
        "selection": {"min_score": 45},
        "scoring": {
            "total_score_formula": "technical_score + market_context_score + penalty_score",
            "use_financial_score": False,
        },
    }

    result = main_module.build_config_validation(
        "example",
        runtime_settings={
            "profile_id": "example",
            "provider": "jquants",
            "jquants_plan": "free",
            "broker_mode": "paper",
            "auto_order_enabled": False,
            "sources": {},
        },
        registry={"profiles": {}},
        operation_schedule={"safety": {"require_manual_approval": True, "forbid_live_auto_order": True}},
        profile_config=profile_config,
    )

    assert any(item["status"] == "OK" and item["name"].endswith(".financial_context_data_only") for item in result["checks"])
    assert not any("use_financial_score=true" in item["message"] for item in result["checks"])


def test_validate_config_strict_turns_warning_into_exit_code() -> None:
    result = main_module.build_config_validation(
        "rookie_dealer_02_v2_6",
        runtime_settings={
            "profile_id": "rookie_dealer_02_v2_6",
            "provider": "jquants",
            "jquants_plan": "free",
            "broker_mode": "paper",
            "auto_order_enabled": False,
            "sources": {"jquants_plan": "cli"},
        },
        strict=True,
    )

    assert result["warn_count"] > 0
    assert result["exit_code"] == 1


def test_validate_config_invalid_required_plan_is_error() -> None:
    registry = {
        "profiles": {
            "rookie_dealer_02_v2_1": {
                "role": "baseline",
                "required_plan": "premium",
                "features": {"relative_strength": False},
            }
        }
    }

    result = main_module.build_config_validation(
        "rookie_dealer_02_v2_1",
        runtime_settings={
            "profile_id": "rookie_dealer_02_v2_1",
            "provider": "jquants",
            "jquants_plan": "free",
            "broker_mode": "paper",
            "auto_order_enabled": False,
            "sources": {"jquants_plan": "config"},
        },
        registry=registry,
    )

    assert any(item["status"] == "FAIL" and item["name"].endswith(".required_plan") for item in result["checks"])


def test_validate_config_live_auto_order_is_error() -> None:
    result = main_module.build_config_validation(
        "rookie_dealer_02_v2_1",
        runtime_settings={
            "profile_id": "rookie_dealer_02_v2_1",
            "provider": "jquants",
            "jquants_plan": "free",
            "broker_mode": "tachibana_live",
            "auto_order_enabled": True,
            "sources": {"jquants_plan": "config"},
        },
        operation_schedule={
            "execution_policy": {"broker": "tachibana_live", "auto_order_enabled": True, "forbid_live_auto_order": True},
            "safety": {"require_manual_approval": True, "forbid_live_auto_order": True},
        },
    )

    live_auto = next(item for item in result["checks"] if item["name"] == "safety.live_auto_order")
    assert live_auto["status"] == "FAIL"


def test_clean_reports_is_dry_run_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    target = tmp_path / "reports" / "rookie_dealer_02_v2_1"
    target.mkdir(parents=True)
    report = target / "backtest_20260101.md"
    report.write_text("old", encoding="utf-8")

    targets = main_module.build_clean_targets("clean-reports", "rookie_dealer_02_v2_1", "jquants")
    result = main_module.execute_clean_targets(targets, yes=False)

    assert result["dry_run"] is True
    assert result["deleted_count"] == 0
    assert report.exists()


def test_clean_reports_deletes_only_with_yes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    target = tmp_path / "reports" / "rookie_dealer_02_v2_1"
    target.mkdir(parents=True)
    report = target / "backtest_20260101.md"
    report.write_text("old", encoding="utf-8")

    targets = main_module.build_clean_targets("clean-reports", "rookie_dealer_02_v2_1", "jquants")
    result = main_module.execute_clean_targets(targets, yes=True)

    assert result["dry_run"] is False
    assert result["deleted_count"] == 1
    assert not report.exists()


def test_clean_reports_keeps_latest_without_include_latest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    target = tmp_path / "reports" / "rookie_dealer_02_v2_1"
    target.mkdir(parents=True)
    (target / "analysis_latest.md").write_text("latest", encoding="utf-8")
    old = target / "backtest_20260101.md"
    old.write_text("old", encoding="utf-8")

    plan = main_module.build_clean_plan("clean-reports", "rookie_dealer_02_v2_1")

    assert [item["relative_path"] for item in plan["targets"]] == ["reports/rookie_dealer_02_v2_1/backtest_20260101.md"]
    assert plan["latest_kept"] == 1


def test_clean_older_than_days_filters_recent_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    target = tmp_path / "reports" / "experiments"
    target.mkdir(parents=True)
    old = target / "old.json"
    recent = target / "recent.json"
    old.write_text("old", encoding="utf-8")
    recent.write_text("recent", encoding="utf-8")
    old_time = (main_module.datetime.now() - main_module.timedelta(days=40)).timestamp()
    old.touch()
    recent.touch()
    import os

    os.utime(old, (old_time, old_time))

    plan = main_module.build_clean_plan("clean-experiments", older_than_days=30)

    assert [item["relative_path"] for item in plan["targets"]] == ["reports/experiments/old.json"]


def test_clean_does_not_delete_allowlist_outside_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    outside = tmp_path / "config" / "danger.txt"
    outside.parent.mkdir(parents=True)
    outside.write_text("do not delete", encoding="utf-8")

    result = main_module.execute_clean_targets([outside], yes=True)

    assert result["deleted_count"] == 0
    assert outside.exists()


def test_clean_skips_symlinks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    reports = tmp_path / "reports" / "articles"
    reports.mkdir(parents=True)
    target = tmp_path / "outside.txt"
    target.write_text("target", encoding="utf-8")
    link = reports / "link.md"
    link.symlink_to(target)

    plan = main_module.build_clean_plan("clean-articles")
    result = main_module.execute_clean_targets([link], yes=True)

    assert plan["file_count"] == 0
    assert result["deleted_count"] == 0
    assert target.exists()


def test_clean_articles_targets_only_articles(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    article = tmp_path / "reports" / "articles" / "daily" / "article.md"
    report = tmp_path / "reports" / "rookie_dealer_02_v2_1" / "backtest.md"
    article.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    article.write_text("article", encoding="utf-8")
    report.write_text("report", encoding="utf-8")

    plan = main_module.build_clean_plan("clean-articles")

    assert [item["relative_path"] for item in plan["targets"]] == ["reports/articles/daily/article.md"]


def test_clean_cache_targets_jquants_cache_kind(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    price_cache = tmp_path / "data" / "cache" / "jquants" / "prices" / "a.json"
    topix_cache = tmp_path / "data" / "cache" / "jquants" / "topix_prices" / "b.json"
    price_cache.parent.mkdir(parents=True)
    topix_cache.parent.mkdir(parents=True)
    price_cache.write_text("price", encoding="utf-8")
    topix_cache.write_text("topix", encoding="utf-8")

    plan = main_module.build_clean_plan("clean-cache", provider_name="jquants", cache_kind="prices")

    assert [item["relative_path"] for item in plan["targets"]] == ["data/cache/jquants/prices/a.json"]


def test_storage_audit_reports_processed_duplication(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    for profile in ["a", "b"]:
        path = tmp_path / "data" / "processed" / profile / "indicators_2026-01-05.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")

    audit = main_module.build_storage_audit()

    assert audit["cache_duplication"]["profile_indicator_files"] == 2
    assert audit["cache_duplication"]["duplicate_indicator_dates"] == 1


def test_cleanup_storage_is_dry_run_by_default_and_keeps_raw_prices(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    log = tmp_path / "logs" / "backtests" / "old.json"
    raw = tmp_path / "data" / "raw" / "prices_2026-01-05.json"
    log.parent.mkdir(parents=True)
    raw.parent.mkdir(parents=True)
    log.write_text("old", encoding="utf-8")
    raw.write_text("price", encoding="utf-8")
    old_time = (main_module.datetime.now() - main_module.timedelta(days=40)).timestamp()
    import os

    os.utime(log, (old_time, old_time))
    os.utime(raw, (old_time, old_time))

    plan = main_module.build_cleanup_storage_plan(include_logs=True, keep_days=30)
    result = main_module.execute_cleanup_storage_plan(plan, apply=False)

    assert [item["relative_path"] for item in plan["targets"]] == ["logs/backtests/old.json"]
    assert result["deleted_count"] == 0
    assert log.exists()
    assert raw.exists()


def test_cleanup_storage_apply_deletes_only_allowed_targets(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    log = tmp_path / "logs" / "scoring" / "old.json"
    config = tmp_path / "config" / "danger.json"
    log.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    log.write_text("old", encoding="utf-8")
    config.write_text("keep", encoding="utf-8")

    result = main_module.execute_cleanup_storage_plan(
        {"targets": [main_module._cleanup_target(log, 3, "test"), main_module._cleanup_target(config, 4, "test")]},
        apply=True,
    )

    assert result["deleted_count"] == 1
    assert not log.exists()
    assert config.exists()


def test_common_processed_cache_restores_candidate_with_target_profile_metadata(tmp_path, monkeypatch, config_copy) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    config_copy["profile_id"] = "target_profile"
    config_copy["profile_name"] = "Target"
    payload = {"profile_id": "source_profile", "profile_name": "Source", "candidates": [{"code": "1001"}]}

    main_module._save_common_processed_cache(config_copy, "candidates", "2026-01-05", payload)
    target = tmp_path / "data" / "processed" / "target_profile" / "candidates_2026-01-05.json"

    assert main_module._restore_common_processed_cache(config_copy, "candidates", "2026-01-05", target) is True
    restored = main_module.read_json(target)
    assert restored["profile_id"] == "target_profile"
    assert restored["profile_name"] == "Target"
    assert restored["candidates"] == [{"code": "1001"}]


def test_compact_processed_cache_hardlinks_duplicate_profile_cache(tmp_path, monkeypatch, config_copy) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_profile", lambda profile_id: {**config_copy, "profile_id": profile_id, "profile_name": profile_id})
    for profile in ["p1", "p2"]:
        path = tmp_path / "data" / "processed" / profile / "indicators_2026-01-05.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"indicators":[]}', encoding="utf-8")

    plan = main_module.build_compact_processed_cache_plan()
    result = main_module.execute_compact_processed_cache_plan(plan, apply=True)
    p1 = tmp_path / "data" / "processed" / "p1" / "indicators_2026-01-05.json"
    p2 = tmp_path / "data" / "processed" / "p2" / "indicators_2026-01-05.json"

    assert result["compacted_count"] == 2
    assert main_module._same_inode(p1, p2)
    assert (tmp_path / "data" / "processed" / "common").exists()


def test_inspect_cache_reports_required_and_removable_indicator_fields(tmp_path, monkeypatch, config_copy) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    config_copy["profile_id"] = "rookie_dealer_02_v2_1"
    path = tmp_path / "data" / "processed" / "rookie_dealer_02_v2_1" / "indicators_2026-03-06.json"
    path.parent.mkdir(parents=True)
    main_module.write_json(
        path,
        {
            "indicators": [
                {
                    "code": "1001",
                    "date": "2026-03-06",
                    "close": 100,
                    "ma5": 101,
                    "ma25": 99,
                    "rsi": 55,
                    "volume_ratio": 2,
                    "turnover_value": 1_000_000_000,
                    "five_day_volatility": 0.02,
                    "macd": 1.2,
                }
            ]
        },
    )

    inspection = main_module.inspect_cache("indicators", "2026-03-06", config_copy)

    assert inspection["row_count"] == 1
    assert inspection["column_count"] == 10
    assert "close" in inspection["required_fields"]
    assert "macd" in inspection["removable_fields"]
    assert inspection["largest_fields"][0]["field"] in {"turnover_value", "date", "code", "macd"}


def test_inspect_cache_keeps_relative_strength_fields_when_enabled(tmp_path, monkeypatch, config_copy) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    config_copy["profile_id"] = "rookie_dealer_02_v2_6"
    config_copy.setdefault("features", {})["relative_strength"] = True
    config_copy.setdefault("scoring", {})["use_relative_strength_score"] = True
    path = tmp_path / "data" / "processed" / "rookie_dealer_02_v2_6" / "indicators_2026-03-06.json"
    path.parent.mkdir(parents=True)
    main_module.write_json(path, {"indicators": [{"code": "1001", "relative_strength_score": 3, "macd": 1.2}]})

    inspection = main_module.inspect_cache("indicators", "2026-03-06", config_copy)

    assert "relative_strength_score" in inspection["required_fields"]
    assert "relative_strength_score" not in inspection["removable_fields"]
    assert "macd" in inspection["removable_fields"]


def test_compact_storage_keeps_selected_scores_and_prunes_debug_fields(config_copy) -> None:
    config_copy.setdefault("storage", {})["save_mode"] = "compact"
    scores = [
        {
            "code": "1001",
            "selected": True,
            "total_score": 45,
            "macd": 1.2,
            "score_components": {"technical_score": 45},
            "relative_strength_score": 3,
        },
        {"code": "1002", "selected": False, "total_score": 40, "macd": 0.1},
    ]

    stored = main_module._scores_for_storage(scores, config_copy)

    assert [row["code"] for row in stored] == ["1001"]
    assert stored[0]["total_score"] == 45
    assert "macd" not in stored[0]
    assert "score_components" not in stored[0]


def test_analysis_storage_keeps_rows_but_prunes_debug_fields(config_copy) -> None:
    config_copy.setdefault("storage", {})["save_mode"] = "analysis"
    config_copy.setdefault("analysis", {})["save_rejected_candidates"] = True
    scores = [
        {"code": "1001", "selected": True, "total_score": 45, "macd": 1.2, "relative_strength_score": 3},
        {"code": "1002", "selected": False, "total_score": 40, "macd": 0.1, "relative_strength_score": 0},
    ]

    stored = main_module._scores_for_storage(scores, config_copy)

    assert [row["code"] for row in stored] == ["1001", "1002"]
    assert "relative_strength_score" in stored[0]
    assert "macd" not in stored[0]


def test_compact_trade_storage_prunes_debug_fields(config_copy) -> None:
    config_copy.setdefault("storage", {})["save_mode"] = "compact"
    trades = [
        {
            "trade_id": "t1",
            "action": "BUY",
            "code": "1001",
            "entry_date": "2026-01-06",
            "profit": 100,
            "dealer_comment": "debug text",
            "score_components": {"technical_score": 45},
        }
    ]

    stored = main_module._trades_for_storage(trades, config_copy)

    assert stored[0]["trade_id"] == "t1"
    assert "dealer_comment" not in stored[0]
    assert "score_components" not in stored[0]


def test_performance_audit_reports_hot_files_and_load_samples(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    indicator = tmp_path / "data" / "processed" / "p1" / "indicators_2026-01-05.json"
    candidate = tmp_path / "data" / "processed" / "p1" / "candidates_2026-01-05.json"
    scored = tmp_path / "data" / "processed" / "p1" / "scored_candidates_2026-01-05.json"
    log = tmp_path / "logs" / "backtests" / "summary.json"
    report = tmp_path / "reports" / "experiments" / "summary.md"
    for path, key in [(indicator, "indicators"), (candidate, "candidates"), (scored, "scores")]:
        path.parent.mkdir(parents=True, exist_ok=True)
        main_module.write_json(path, {key: [{"code": "1001"}]})
    log.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    log.write_text('{"ok":true}', encoding="utf-8")
    report.write_text("ok", encoding="utf-8")

    audit = main_module.build_performance_audit()

    assert audit["processed_indicator_total_size"] > 0
    assert audit["processed_candidate_total_size"] > 0
    assert audit["scored_candidate_total_size"] > 0
    assert audit["logs_total_size"] > 0
    assert audit["reports_total_size"] > 0
    assert audit["largest_indicator_files"][0]["path"].endswith("indicators_2026-01-05.json")
    assert audit["runtime_cost_estimate"]["indicator_load_time_sample"]["sample_count"] == 1


def test_period_5y_sets_start_date_from_today(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "_today_jst", lambda: date(2026, 5, 31))
    args = SimpleNamespace(period="5y", start_date=None, end_date=None)

    main_module._apply_period_preset(args)

    assert args.start_date == "2021-05-31"
    assert args.end_date == "2026-05-31"


def test_period_preset_is_applied_by_parse_args(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "_today_jst", lambda: date(2026, 5, 31))
    monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--mode", "backtest", "--period", "5y"])

    args = main_module.parse_args()

    assert args.requested_start_date == "2021-05-31"
    assert args.start_date == "2021-05-31"
    assert args.end_date == "2026-05-31"
    assert args.start_date_source == "cli"
    assert args.end_date_source == "cli"


def test_summary_only_flag_is_parsed(monkeypatch) -> None:
    monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--mode", "run-experiments", "--summary-only"])

    args = main_module.parse_args()

    assert args.summary_only is True


def test_experiment_judgement_candidate_needs_review_and_rejected() -> None:
    base = {"net_cumulative_profit": 100, "profit_factor": 1.2, "max_drawdown": -0.1, "total_trades": 10}
    candidate = {"net_cumulative_profit": 130, "profit_factor": 1.18, "max_drawdown": -0.1, "total_trades": 8}
    low_trade_count = {"net_cumulative_profit": 150, "profit_factor": 1.4, "max_drawdown": -0.05, "total_trades": 4}
    worse = {"net_cumulative_profit": 80, "profit_factor": 1.1, "max_drawdown": -0.2, "total_trades": 8}

    assert main_module._experiment_judgement(base, candidate)["judgement"] == "candidate"
    assert main_module._experiment_judgement(base, low_trade_count)["judgement"] == "needs_review"
    assert main_module._experiment_judgement(base, worse)["judgement"] == "rejected"
