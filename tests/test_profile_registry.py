from __future__ import annotations

from datetime import date

import pytest

import main as main_module
import profile_registry


def test_profile_registry_can_be_loaded() -> None:
    registry = main_module.load_profile_registry()
    profiles = main_module.registry_profiles(registry)

    assert "rookie_dealer_02_v2_1" in profiles
    assert profiles["rookie_dealer_02_v2_1"]["role"] == "baseline"
    assert profiles["rookie_dealer_02_v2_6"]["compare_to"] == "rookie_dealer_02_v2_1"


def test_list_profiles_outputs_registry_rows(capsys) -> None:
    main_module.run_list_profiles()

    output = capsys.readouterr().out
    assert "profile_id | role | required_plan | enabled_features | compare_to | description" in output
    assert "rookie_dealer_02_v2_1 | baseline | free" in output
    assert "rookie_dealer_02_v2_6 | experiment | light | relative_strength | rookie_dealer_02_v2_1" in output


def test_profile_info_outputs_formula_and_compare_command(capsys) -> None:
    main_module.run_profile_info("rookie_dealer_02_v2_6")

    output = capsys.readouterr().out
    assert "profile_id: rookie_dealer_02_v2_6" in output
    assert "compare_to: rookie_dealer_02_v2_1" in output
    assert "profile yaml path:" in output
    assert "score formula: technical_plus_relative_strength_market_penalty" in output
    assert "required capabilities:" in output
    assert "topix_prices" in output
    assert "recommended backtest command:" in output
    assert "recommended compare command:" in output


def test_profile_info_unknown_profile_errors() -> None:
    with pytest.raises(SystemExit):
        main_module.build_profile_info("missing_profile", main_module.load_profile_registry())


def test_compare_experiments_builds_target_profile_list(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)

    main_module.run_compare_experiments("rookie_dealer_02_v2_1")

    summary_path = tmp_path / "reports" / "profile_comparisons" / "experiment_summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    assert "base_profile: rookie_dealer_02_v2_1" in summary
    assert "rookie_dealer_02_v2_6" in summary
    assert "rookie_dealer_02_v2_7" in summary
    assert "rookie_dealer_02_v2_8" in summary
    assert "rookie_dealer_02_v2_9" in summary
    assert "rookie_dealer_02_v2_10" in summary


def test_run_experiments_selects_registry_profiles() -> None:
    registry = main_module.load_profile_registry()

    profiles = main_module.select_experiment_profiles("rookie_dealer_02_v2_1", registry, None)

    assert profiles == [
        "rookie_dealer_02_v2_10",
        "rookie_dealer_02_v2_11",
        "rookie_dealer_02_v2_6",
        "rookie_dealer_02_v2_7",
        "rookie_dealer_02_v2_8",
        "rookie_dealer_02_v2_9",
        "rookie_dealer_03_growth",
        "rookie_dealer_03_standard_growth",
    ]


def test_run_experiments_profiles_option_filters_targets() -> None:
    registry = main_module.load_profile_registry()

    profiles = main_module.select_experiment_profiles(
        "rookie_dealer_02_v2_1",
        registry,
        ["rookie_dealer_02_v2_6", "rookie_dealer_02_v2_10"],
    )

    assert profiles == ["rookie_dealer_02_v2_10", "rookie_dealer_02_v2_6"]


def test_run_experiments_selects_v2_19_capital_exposure_profiles() -> None:
    registry = main_module.load_profile_registry()

    profiles = main_module.select_experiment_profiles("rookie_dealer_02_v2_19", registry, None)

    assert profiles == [
        "rookie_dealer_02_v2_20",
        "rookie_dealer_02_v2_21",
        "rookie_dealer_02_v2_22",
        "rookie_dealer_02_v2_23",
        "rookie_dealer_02_v2_24",
        "rookie_dealer_02_v2_25",
    ]


def test_affordability_filter_changes_scoring_reuse_signature() -> None:
    base = main_module.load_profile("rookie_dealer_02_v2_19")
    affordability = main_module.load_profile("rookie_dealer_02_v2_23")

    assert main_module._experiment_scoring_signature(base) != main_module._experiment_scoring_signature(affordability)


def test_run_experiments_skip_backtest_writes_summary(tmp_path, monkeypatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "run_backtest", lambda *args: calls.append(("backtest", args)))
    monkeypatch.setattr(main_module, "run_analyze", lambda *args: calls.append(("analyze", args)))
    monkeypatch.setattr(main_module, "run_compare_profiles", lambda *args: calls.append(("compare", args)))
    monkeypatch.setattr(main_module, "_ensure_experiment_db_rows", lambda *args: None)

    main_module.run_experiments(
        "rookie_dealer_02_v2_1",
        "2026-01-01",
        "2026-03-06",
        ["rookie_dealer_02_v2_6"],
        skip_backtest=True,
    )

    assert [call[0] for call in calls] == ["analyze", "analyze", "compare"]
    summary_path = tmp_path / "reports" / "experiments" / "2026-01-01_to_2026-03-06" / "rookie_dealer_02_v2_1" / "experiment_summary.md"
    json_path = tmp_path / "reports" / "experiments" / "2026-01-01_to_2026-03-06" / "rookie_dealer_02_v2_1" / "experiment_summary.json"
    profile_path = tmp_path / "reports" / "experiments" / "2026-01-01_to_2026-03-06" / "rookie_dealer_02_v2_1" / "profiles" / "rookie_dealer_02_v2_6.md"
    assert summary_path.exists()
    assert json_path.exists()
    assert profile_path.exists()
    assert "candidate" in summary_path.read_text(encoding="utf-8")


def test_experiment_candidate_judgement() -> None:
    base = {"net_cumulative_profit": 100, "profit_factor": 1.2, "max_drawdown": -0.1, "total_trades": 10}
    better = {"net_cumulative_profit": 150, "profit_factor": 1.2, "max_drawdown": -0.08, "total_trades": 8}
    worse_pf = {"net_cumulative_profit": 150, "profit_factor": 1.1, "max_drawdown": -0.08, "total_trades": 8}

    assert main_module._experiment_candidate(base, better) is True
    assert main_module._experiment_candidate(base, worse_pf) is False


def test_run_experiments_skip_analyze_skips_analyze_step(tmp_path, monkeypatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "run_backtest", lambda *args: calls.append(("backtest", args)))
    monkeypatch.setattr(main_module, "run_analyze", lambda *args: calls.append(("analyze", args)))
    monkeypatch.setattr(main_module, "run_compare_profiles", lambda *args: calls.append(("compare", args)))
    monkeypatch.setattr(
        main_module,
        "prepare_run_experiments_common_stages",
        lambda *_args: {
            "price_fetch_time": 0,
            "indicator_time": 0,
            "candidate_time": 0,
            "scoring_time": 0,
            "trade_time": 0,
            "reused_indicator_count": 0,
            "reused_candidate_count": 0,
        },
    )

    main_module.run_experiments(
        "rookie_dealer_02_v2_1",
        "2026-01-01",
        "2026-03-06",
        ["rookie_dealer_02_v2_6"],
        skip_backtest=False,
        skip_analyze=True,
    )

    assert "analyze" not in [call[0] for call in calls]
    assert [call[0] for call in calls].count("backtest") == 2
    assert [call[0] for call in calls].count("compare") == 1


def test_summary_only_still_runs_analyze_for_feature_audits(tmp_path, monkeypatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "SUMMARY_ONLY_ACTIVE", True)
    monkeypatch.setattr(main_module, "run_backtest", lambda *args: calls.append(("backtest", args)))
    monkeypatch.setattr(main_module, "run_analyze", lambda *args: calls.append(("analyze", args)))
    monkeypatch.setattr(main_module, "run_compare_profiles", lambda *args: calls.append(("compare", args)))
    monkeypatch.setattr(
        main_module,
        "prepare_run_experiments_common_stages",
        lambda *_args: {
            "shared_price_fetch_time": 0,
            "shared_indicator_time": 0,
            "shared_candidate_time": 0,
            "profile_scoring_time_by_profile": {},
            "profile_trade_time_by_profile": {},
            "reused_scoring_count": 0,
        },
    )

    main_module.run_experiments(
        "rookie_dealer_02_v2_1",
        "2026-01-01",
        "2026-03-06",
        ["rookie_dealer_02_v2_10"],
        skip_backtest=False,
        skip_analyze=False,
    )

    assert [call[0] for call in calls].count("analyze") == 2
    assert [call[0] for call in calls].count("backtest") == 2
    assert [call[0] for call in calls].count("compare") == 1


def test_scoring_reuse_sources_detect_identical_scoring_profiles() -> None:
    reuse = main_module._experiment_scoring_reuse_sources(
        ["rookie_dealer_02_v2_1", "rookie_dealer_02_v2_7", "rookie_dealer_02_v2_10"]
    )

    assert reuse.get("rookie_dealer_02_v2_10") == "rookie_dealer_02_v2_7"
    assert "rookie_dealer_02_v2_7" not in reuse


def test_prepare_run_experiments_common_stages_reuses_indicator_and_candidate_generation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "ensure_price_history_for_backtest", lambda *_args: None)
    monkeypatch.setattr(main_module, "available_cached_price_dates", lambda *_args: [date(2026, 1, 5), date(2026, 1, 6)])
    monkeypatch.setattr(main_module, "_preload_light_api_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "ensure_market_context", lambda *_args: {})
    calls = {"indicators": 0, "candidates": 0}

    def fake_ensure_indicators(_provider: str, target_date_text: str) -> None:
        calls["indicators"] += 1
        config = main_module.load_config(main_module.CONFIG_PATH)
        main_module.write_json(
            main_module.processed_profile_path(config, f"indicators_{target_date_text}.json"),
            {
                "date": target_date_text,
                "indicator_mode": main_module._backtest_indicator_mode(config),
                "relative_strength_enabled": main_module._relative_strength_enabled_for_indicators(config),
                "indicators": [{"code": "1001", "date": target_date_text}],
            },
        )

    def fake_ensure_screen(_provider: str, target_date_text: str) -> None:
        calls["candidates"] += 1
        config = main_module.load_config(main_module.CONFIG_PATH)
        profile_id = main_module.profile_id_from(config)
        payload = {
            "date": target_date_text,
            "profile_id": profile_id,
            "profile_name": main_module.profile_name_from(config),
            "config_version": main_module.config_version_from(config),
            "candidate_count": 1,
            "candidates": [{"code": "1001", "date": target_date_text}],
        }
        main_module.write_json(main_module.processed_profile_path(config, f"candidates_{target_date_text}.json"), payload)
        main_module.write_json(tmp_path / "logs" / "screening" / profile_id / f"screening_{target_date_text}.json", payload)

    monkeypatch.setattr(main_module, "ensure_indicators", fake_ensure_indicators)
    monkeypatch.setattr(main_module, "ensure_screen", fake_ensure_screen)

    report = main_module.prepare_run_experiments_common_stages(
        ["rookie_dealer_02_v2_1", "rookie_dealer_02_v2_7", "rookie_dealer_02_v2_9"],
        "2026-01-05",
        "2026-01-06",
    )

    assert calls == {"indicators": 2, "candidates": 2}
    assert report["reused_indicator_count"] == 4
    assert report["reused_candidate_count"] == 4
    for profile_id in ["rookie_dealer_02_v2_7", "rookie_dealer_02_v2_9"]:
        assert (tmp_path / "data" / "processed" / profile_id / "indicators_2026-01-05.json").exists()
        assert (tmp_path / "data" / "processed" / profile_id / "candidates_2026-01-05.json").exists()


def test_prepare_run_experiments_common_stages_can_skip_price_fetch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "SKIP_PRICE_FETCH_ACTIVE", True)
    monkeypatch.setattr(
        main_module,
        "ensure_price_history_for_backtest",
        lambda *_args: (_ for _ in ()).throw(AssertionError("price fetch should be skipped")),
    )
    monkeypatch.setattr(main_module, "available_cached_price_dates", lambda *_args: [date(2026, 1, 5)])
    monkeypatch.setattr(main_module, "_preload_light_api_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "ensure_indicators", lambda *_args: None)
    monkeypatch.setattr(main_module, "ensure_market_context", lambda *_args: {})
    monkeypatch.setattr(main_module, "ensure_screen", lambda *_args: None)
    monkeypatch.setattr(main_module, "_copy_common_indicator_stage", lambda *_args: True)
    monkeypatch.setattr(main_module, "_copy_common_candidate_stage", lambda *_args: True)

    report = main_module.prepare_run_experiments_common_stages(
        ["rookie_dealer_02_v2_1"],
        "2026-01-05",
        "2026-01-05",
    )

    assert report["price_fetch_skipped"] is True
    assert report["target_trading_days"] == 1


def test_scored_candidates_cache_reuses_same_profile_date_and_hash(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "ACTIVE_PROFILE_ID", "rookie_dealer_02_v2_1")
    monkeypatch.setattr(main_module, "save_scoring_results", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_module,
        "run_score",
        lambda *_args: (_ for _ in ()).throw(AssertionError("score should use cached payload")),
    )
    config = main_module.load_config(main_module.CONFIG_PATH)
    cache_bits = main_module._score_cache_payload(config, "2026-01-05")
    path = main_module.processed_profile_path(config, "scored_candidates_2026-01-05.json")
    main_module.write_json(
        path,
        {
            "date": "2026-01-05",
            "provider": "jquants",
            "profile_id": "rookie_dealer_02_v2_1",
            "profile_name": main_module.profile_name_from(config),
            "config_version": main_module.config_version_from(config),
            **cache_bits,
            "candidate_count": 1,
            "scored_count": 1,
            "selected_count": 1,
            "scores": [{"code": "1001", "selected": True, "total_score": 50}],
            "selected": [{"code": "1001", "selected": True, "total_score": 50}],
        },
    )

    payload = main_module.ensure_score("jquants", "2026-01-05")

    assert payload["selected_count"] == 1
    assert payload["scoring_cache_key"] == cache_bits["scoring_cache_key"]


def test_profile_phase_time_snapshot_accounts_for_profile_total(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "BACKTEST_PROFILE_TIMINGS",
        {
            "indicator": 2.0,
            "screening": 3.0,
            "scoring": 4.0,
            "trading": 5.0,
            "report": 1.0,
            "json_read": 0.5,
            "json_write": 0.5,
            "sqlite_access": 1.0,
        },
    )

    snapshot = main_module._profile_phase_time_snapshot(20.0)

    assert snapshot["accounting_delta"] == 0
    assert round(sum(snapshot["phases"].values()), 4) == 20.0
    assert snapshot["phases"]["misc"] == 3.0


def test_top_experiment_bottlenecks_are_sorted() -> None:
    bottlenecks = main_module._top_experiment_bottlenecks(
        {
            "profile_phase_time_by_profile": {
                "p1": {"scoring": 10, "trade": 2},
                "p2": {"json_read": 7, "timing_overlap_adjustment": -1},
            },
            "compare_profiles_time": 4,
        },
        limit=3,
    )

    assert bottlenecks == [
        {"profile": "p1", "phase": "scoring", "elapsed_seconds": 10.0},
        {"profile": "p2", "phase": "json_read", "elapsed_seconds": 7.0},
        {"profile": "__batch__", "phase": "compare_profiles", "elapsed_seconds": 4.0},
    ]


def test_json_read_audit_flags_period_out_of_range_reads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "BACKTEST_MODE_ACTIVE", True)
    monkeypatch.setattr(main_module, "BACKTEST_JSON_READ_PERIOD", ("2026-01-01", "2026-03-06"))
    monkeypatch.setattr(main_module, "BACKTEST_JSON_READ_AUDIT", {"file_count": 0, "out_of_range_count": 0, "out_of_range_sample": []})
    path = tmp_path / "data" / "processed" / "rookie_dealer_02_v2_1" / "scored_candidates_2026-05-29.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    main_module.read_json(path)

    audit = main_module.BACKTEST_JSON_READ_AUDIT
    assert audit["file_count"] == 1
    assert audit["date_min"] == "2026-05-29"
    assert audit["date_max"] == "2026-05-29"
    assert audit["out_of_range_count"] == 1
    assert audit["bytes"] == 2
    assert main_module._json_read_breakdown(audit)["scored_candidates"]["count"] == 1
    assert main_module._json_read_scope_breakdown(audit)["profile"]["count"] == 1
    assert main_module._top_json_read_files(audit)[0]["path"].endswith("scored_candidates_2026-05-29.json")


def test_ensure_indicators_prefers_common_cache_over_profile_cache(tmp_path, monkeypatch) -> None:
    config = main_module.load_profile("rookie_dealer_02_v2_1")
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "BACKTEST_MODE_ACTIVE", True)
    monkeypatch.setattr(
        main_module,
        "COMMON_CACHE_METRICS",
        {
            "cache_reused_from_common_count": 0,
            "profile_specific_cache_count": 0,
            "generated_cache_size": 0,
            "indicator_cache_source": {},
            "candidate_cache_source": {},
        },
    )
    target_date = "2026-03-06"
    profile_path = main_module.processed_profile_path(config, f"indicators_{target_date}.json")
    common_path = main_module._common_processed_cache_path(config, "indicators", target_date)
    profile_path.parent.mkdir(parents=True)
    common_path.parent.mkdir(parents=True)
    payload_base = {
        "date": target_date,
        "indicator_mode": main_module._backtest_indicator_mode(config),
        "relative_strength_enabled": False,
        "benchmark_source": "unavailable",
    }
    main_module.write_json(profile_path, {**payload_base, "indicators": [{"code": "profile"}]})
    main_module.write_json(common_path, {**payload_base, "indicators": [{"code": "common"}]})

    main_module.ensure_indicators("jquants", target_date)

    restored = main_module.read_json(profile_path)
    assert restored["indicators"][0]["code"] == "common"
    assert main_module.COMMON_CACHE_METRICS["cache_reused_from_common_count"] == 1
    assert main_module.COMMON_CACHE_METRICS["indicator_cache_source"]["common"] == 1


def test_experiment_verdict_no_practical_effect() -> None:
    base = {"net_cumulative_profit": 100, "profit_factor": 1.2, "max_drawdown": -0.1, "total_trades": 10}
    same = {"net_cumulative_profit": 100, "profit_factor": 1.2, "max_drawdown": -0.1, "total_trades": 10}
    diff = {"selection_diff_count": 0, "outcome_diff_count": 0}

    assert main_module._experiment_judgement(base, same, diff)["judgement"] == "no_practical_effect"


def test_experiment_verdict_not_no_practical_effect_when_metrics_change_with_outcome_diff() -> None:
    base = {"net_cumulative_profit": 100, "profit_factor": 1.2, "max_drawdown": -0.1, "total_trades": 10}
    changed = {"net_cumulative_profit": 120, "profit_factor": 1.2, "max_drawdown": -0.1, "total_trades": 10}
    diff = {"selection_diff_count": 0, "outcome_diff_count": 1}

    result = main_module._experiment_judgement(base, changed, diff)

    assert result["judgement"] != "no_practical_effect"
    assert "execution_or_exit_effect" in result["reasons"]


def test_experiment_summary_row_includes_feature_activation() -> None:
    row = {
        "profile_id": "rookie_dealer_02_v2_9",
        "role": "experiment",
        "description": "財務スコア検証",
        "required_plan": "free",
        "enabled_features": ["financial_context"],
        "total_trades": 10,
        "newly_selected_count": 0,
        "removed_count": 0,
        "selection_diff_count": 0,
        "outcome_diff_count": 0,
        "feature_data_enabled": {"financial_context": True},
        "feature_scoring_enabled": {"financial_context": False},
        "feature_trigger_count": {"financial_context": 0},
        "earnings_calendar_records": 3,
        "earnings_filter_rejected_count": 1,
        "earnings_filter_status": "active",
        "practical_effect": "no_practical_effect",
        "effect_reason": "selection_diff_count=0 and outcome_diff_count=0",
        "verdict": "no_practical_effect",
        "verdict_reason": "no_practical_effect",
    }

    rendered = main_module._experiment_summary_table_row(row)

    assert '{"financial_context":true}' in rendered
    assert '{"financial_context":false}' in rendered
    assert '{"financial_context":0}' in rendered
    assert "| 3 | 1 | active |" in rendered


def test_profile_compare_row_prefers_backtest_summary_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    profile_id = "rookie_dealer_02_v2_1"
    summary_dir = tmp_path / "logs" / "backtests" / profile_id / "2026-01-01_to_2026-03-06"
    summary_dir.mkdir(parents=True)
    main_module.write_json(
        summary_dir / "backtest_summary.json",
        {
            "total_trades": 0,
            "closed_trade_count": 0,
            "net_cumulative_profit": 0,
            "market_coverage": {
                "candidate_count": {"Prime": 1819, "Standard": 0, "Growth": 0, "Unknown": 0},
                "selected_count": {"Prime": 0, "Standard": 0, "Growth": 0, "Unknown": 0},
                "market_filter_excluded_count": 29,
            },
        },
    )

    row = main_module._profile_compare_row_with_backtest_summary(
        {
            "profile_id": profile_id,
            "total_trades": 1,
            "market_coverage": {
                "candidate_count": {"Prime": 0, "Standard": 0, "Growth": 0, "Unknown": 0},
                "selected_count": {"Prime": 0, "Standard": 0, "Growth": 0, "Unknown": 0},
                "market_filter_excluded_count": 1848,
            },
        },
        profile_id,
        "2026-01-01",
        "2026-03-06",
    )

    assert row["total_trades"] == 0
    assert row["market_coverage"]["market_filter_excluded_count"] == 29
    assert row["market_coverage"]["candidate_count"]["Prime"] == 1819
    assert row["summary_source"] == "backtest_summary"


def test_experiment_capability_warning_for_light_profile_on_free_plan() -> None:
    result = main_module.resolve_experiment_capabilities(["rookie_dealer_02_v2_6"], "free")

    assert [item["profile_id"] for item in result["runnable"]] == ["rookie_dealer_02_v2_6"]
    assert result["warnings"]


def test_registry_module_reads_profile_info() -> None:
    info = profile_registry.get_profile_info("rookie_dealer_02_v2_6")

    assert info["profile_id"] == "rookie_dealer_02_v2_6"
    assert info["required_plan"] == "light"
    assert info["compare_to"] == "rookie_dealer_02_v2_1"
    assert "relative_strength" in info["enabled_features"]


def test_registry_validation_detects_yaml_profile_id_mismatch(tmp_path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "example.yaml").write_text("profile_id: wrong\n", encoding="utf-8")
    registry = {
        "profiles": {
            "example": {
                "role": "baseline",
                "required_plan": "free",
                "compare_to": None,
                "features": {"relative_strength": False},
            }
        }
    }

    result = profile_registry.validate_registry(registry, profiles_dir)

    assert result["fail_count"] == 1
    mismatch = next(item for item in result["checks"] if item["name"] == "example.profile_id_match")
    assert mismatch["status"] == "FAIL"


def test_registry_validation_requires_compare_to_for_experiment(tmp_path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "experiment.yaml").write_text("profile_id: experiment\n", encoding="utf-8")
    registry = {
        "profiles": {
            "experiment": {
                "role": "experiment",
                "required_plan": "free",
                "features": {"relative_strength": False},
            }
        }
    }

    result = profile_registry.validate_registry(registry, profiles_dir)

    compare_to = next(item for item in result["checks"] if item["name"] == "experiment.compare_to")
    assert compare_to["status"] == "FAIL"


def test_registry_validation_detects_missing_compare_to_profile(tmp_path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "experiment.yaml").write_text("profile_id: experiment\n", encoding="utf-8")
    registry = {
        "profiles": {
            "experiment": {
                "role": "experiment",
                "required_plan": "free",
                "compare_to": "missing_base",
                "features": {"relative_strength": False},
            }
        }
    }

    result = profile_registry.validate_registry(registry, profiles_dir)

    compare_to = next(item for item in result["checks"] if item["name"] == "experiment.compare_to_exists")
    assert compare_to["status"] == "FAIL"


def test_deprecated_profile_is_excluded_from_experiments() -> None:
    registry = {
        "profiles": {
            "base": {"role": "baseline", "required_plan": "free", "features": {}},
            "active": {"role": "experiment", "required_plan": "free", "compare_to": "base", "features": {}},
            "old": {"role": "deprecated", "required_plan": "free", "compare_to": "base", "features": {}},
        }
    }

    assert profile_registry.get_experiments("base", registry) == ["active"]
