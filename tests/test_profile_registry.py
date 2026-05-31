from __future__ import annotations

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
        "rookie_dealer_02_v2_6",
        "rookie_dealer_02_v2_7",
        "rookie_dealer_02_v2_8",
        "rookie_dealer_02_v2_9",
    ]


def test_run_experiments_profiles_option_filters_targets() -> None:
    registry = main_module.load_profile_registry()

    profiles = main_module.select_experiment_profiles(
        "rookie_dealer_02_v2_1",
        registry,
        ["rookie_dealer_02_v2_6", "rookie_dealer_02_v2_10"],
    )

    assert profiles == ["rookie_dealer_02_v2_10", "rookie_dealer_02_v2_6"]


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
        "feature_active": {"financial_context": False},
        "feature_trigger_count": {"financial_context": 0},
        "practical_effect": "no_practical_effect",
        "effect_reason": "selection_diff_count=0 and outcome_diff_count=0",
        "verdict": "no_practical_effect",
        "verdict_reason": "no_practical_effect",
    }

    rendered = main_module._experiment_summary_table_row(row)

    assert '{"financial_context":false}' in rendered
    assert '{"financial_context":0}' in rendered


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
