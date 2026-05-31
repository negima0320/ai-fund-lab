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

    stale_check = next(item for item in result["checks"] if item["name"] == "stale_score_components")
    formula_check = next(item for item in result["checks"] if item["name"] == "score_formula_terms")
    assert stale_check["status"] == "FAIL"
    assert formula_check["status"] == "FAIL"


def test_clean_reports_is_dry_run_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    target = tmp_path / "reports" / "rookie_dealer_02_v2_1"
    target.mkdir(parents=True)

    targets = main_module.build_clean_targets("clean-reports", "rookie_dealer_02_v2_1", "jquants")
    result = main_module.execute_clean_targets(targets, yes=False)

    assert result["dry_run"] is True
    assert result["deleted_count"] == 0
    assert target.exists()


def test_clean_reports_deletes_only_with_yes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    target = tmp_path / "reports" / "rookie_dealer_02_v2_1"
    target.mkdir(parents=True)

    targets = main_module.build_clean_targets("clean-reports", "rookie_dealer_02_v2_1", "jquants")
    result = main_module.execute_clean_targets(targets, yes=True)

    assert result["dry_run"] is False
    assert result["deleted_count"] == 1
    assert not target.exists()


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
