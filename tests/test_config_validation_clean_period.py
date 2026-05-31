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

    assert not hasattr(args, "requested_start_date")
    assert args.start_date == "2021-05-31"
    assert args.end_date == "2026-05-31"


def test_experiment_judgement_candidate_needs_review_and_rejected() -> None:
    base = {"net_cumulative_profit": 100, "profit_factor": 1.2, "max_drawdown": -0.1, "total_trades": 10}
    candidate = {"net_cumulative_profit": 130, "profit_factor": 1.18, "max_drawdown": -0.1, "total_trades": 8}
    low_trade_count = {"net_cumulative_profit": 150, "profit_factor": 1.4, "max_drawdown": -0.05, "total_trades": 4}
    worse = {"net_cumulative_profit": 80, "profit_factor": 1.1, "max_drawdown": -0.2, "total_trades": 8}

    assert main_module._experiment_judgement(base, candidate)["judgement"] == "candidate"
    assert main_module._experiment_judgement(base, low_trade_count)["judgement"] == "needs_review"
    assert main_module._experiment_judgement(base, worse)["judgement"] == "rejected"
