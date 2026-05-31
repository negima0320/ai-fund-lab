from __future__ import annotations

from types import SimpleNamespace

import main as main_module


def _args(**overrides):
    values = {
        "provider": None,
        "profile": None,
        "jquants_plan": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runtime_settings_use_config_when_cli_is_omitted() -> None:
    provider_config = {
        "data_provider": "jquants",
        "jquants": {"plan": "light"},
        "broker": {"mode": "paper"},
        "profile": {"default": "rookie_dealer_02_v2_1"},
        "operation": {"auto_order_enabled": False},
    }

    settings = main_module.resolve_runtime_settings(_args(), provider_config)

    assert settings["profile_id"] == "rookie_dealer_02_v2_1"
    assert settings["provider"] == "jquants"
    assert settings["jquants_plan"] == "light"
    assert settings["broker_mode"] == "paper"
    assert settings["auto_order_enabled"] is False
    assert settings["sources"] == {
        "profile": "config",
        "provider": "config",
        "jquants_plan": "config",
        "broker": "config",
        "auto_order_enabled": "config",
    }


def test_runtime_settings_fall_back_to_defaults_without_config() -> None:
    settings = main_module.resolve_runtime_settings(_args(), {})

    assert settings["profile_id"] == "rookie_dealer_01"
    assert settings["provider"] == "jquants"
    assert settings["jquants_plan"] == "free"
    assert settings["broker_mode"] == "paper"
    assert settings["sources"]["profile"] == "default"
    assert settings["sources"]["provider"] == "default"
    assert settings["sources"]["jquants_plan"] == "default"


def test_cli_overrides_provider_config() -> None:
    provider_config = {
        "data_provider": "jquants",
        "jquants": {"plan": "free"},
        "profile": {"default": "rookie_dealer_02_v2_1"},
    }

    settings = main_module.resolve_runtime_settings(
        _args(provider="dummy", profile="rookie_dealer_03", jquants_plan="light"),
        provider_config,
    )

    assert settings["profile_id"] == "rookie_dealer_03"
    assert settings["provider"] == "dummy"
    assert settings["jquants_plan"] == "light"
    assert settings["sources"]["profile"] == "cli"
    assert settings["sources"]["provider"] == "cli"
    assert settings["sources"]["jquants_plan"] == "cli"


def test_runtime_sources_are_displayed_in_preflight_rows(config_copy: dict) -> None:
    config_copy["_value_sources"] = {
        "profile": "config",
        "provider": "config",
        "jquants_plan": "config",
        "broker": "config",
        "auto_order_enabled": "config",
    }
    config_copy["data_provider"] = "jquants"
    config_copy["jquants"]["plan"] = "light"
    config_copy["broker"]["mode"] = "paper"
    config_copy["broker"]["provider"] = "paper"
    config_copy.setdefault("operation", {})["auto_order_enabled"] = False
    results: list[dict] = []

    main_module._check_runtime_value_sources(results, config_copy)
    main_module._check_jquants_plan_capabilities(results, config_copy)

    messages = [item["message"] for item in results]
    assert "Profile: rookie_dealer_02_v2_1 (source=config)" in messages
    assert "Data Provider: jquants (source=config)" in messages
    assert "Broker: paper (source=config)" in messages
    assert "operation.auto_order_enabled: false (source=config)" in messages
    assert "J-Quants Plan: light (source=config)" in messages


def test_parse_args_uses_configured_backtest_dates_when_cli_omits_them(monkeypatch) -> None:
    monkeypatch.setattr(main_module.sys, "argv", ["main.py", "--mode", "backtest"])

    args = main_module.parse_args()

    assert args.provider is None
    assert args.profile is None
    assert args.start_date == "2026-01-05"
    assert args.end_date == "2026-03-06"
