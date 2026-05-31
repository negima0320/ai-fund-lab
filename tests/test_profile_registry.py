from __future__ import annotations

import pytest

import main as main_module


def test_profile_registry_can_be_loaded() -> None:
    registry = main_module.load_profile_registry()
    profiles = main_module.registry_profiles(registry)

    assert "rookie_dealer_02_v2_1" in profiles
    assert profiles["rookie_dealer_02_v2_1"]["role"] == "baseline"
    assert profiles["rookie_dealer_02_v2_6"]["compare_to"] == "rookie_dealer_02_v2_1"


def test_list_profiles_outputs_registry_rows(capsys) -> None:
    main_module.run_list_profiles()

    output = capsys.readouterr().out
    assert "profile_id | role | required_plan | enabled_features | description" in output
    assert "rookie_dealer_02_v2_1 | baseline | free" in output
    assert "rookie_dealer_02_v2_6 | experiment | light | relative_strength" in output


def test_profile_info_outputs_formula_and_compare_command(capsys) -> None:
    main_module.run_profile_info("rookie_dealer_02_v2_6")

    output = capsys.readouterr().out
    assert "profile_id: rookie_dealer_02_v2_6" in output
    assert "score formula: technical_plus_relative_strength_market_penalty" in output
    assert "required capabilities:" in output
    assert "topix_prices" in output
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
    assert "rookie_dealer_02_v2_8" in summary
    assert "rookie_dealer_02_v2_9" in summary
    assert "rookie_dealer_02_v2_10" in summary


def test_run_experiments_selects_registry_profiles() -> None:
    registry = main_module.load_profile_registry()

    profiles = main_module.select_experiment_profiles("rookie_dealer_02_v2_1", registry, None)

    assert profiles == [
        "rookie_dealer_02_v2_10",
        "rookie_dealer_02_v2_6",
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

    main_module.run_experiments(
        "rookie_dealer_02_v2_1",
        "2026-01-01",
        "2026-03-06",
        ["rookie_dealer_02_v2_6"],
        skip_backtest=True,
    )

    assert [call[0] for call in calls] == ["compare"]
    summary_path = tmp_path / "reports" / "experiments" / "2026-01-01_to_2026-03-06" / "experiment_summary.md"
    json_path = tmp_path / "reports" / "experiments" / "2026-01-01_to_2026-03-06" / "experiment_summary.json"
    profile_path = tmp_path / "reports" / "experiments" / "2026-01-01_to_2026-03-06" / "profiles" / "rookie_dealer_02_v2_6.md"
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
