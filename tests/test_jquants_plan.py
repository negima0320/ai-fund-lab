from __future__ import annotations

from datetime import date

import main as main_module
from data_provider import JQuantsDataProvider
from jquants_plan import jquants_capability_status, jquants_profile_compatibility, resolve_jquants_plan
from profile_loader import load_profile


def _provider_without_init(plan: str) -> JQuantsDataProvider:
    provider = JQuantsDataProvider.__new__(JQuantsDataProvider)
    provider.api_key = "test"
    provider.base_url = "https://api.jquants.com/v2"
    provider.default_headers = {"x-api-key": "test"}
    provider.timeout_seconds = 20
    provider.plan = plan
    provider.capabilities = {
        capability
        for capability, status in jquants_capability_status(plan).items()
        if status == "OK"
    }
    return provider


def test_free_plan_does_not_call_topix_prices(monkeypatch) -> None:
    provider = _provider_without_init("free")

    def fail_fetch(*args, **kwargs):
        raise AssertionError("topix endpoint should not be called on free plan")

    monkeypatch.setattr(provider, "_get_paginated_records", fail_fetch)

    assert provider.get_topix_prices(date(2026, 3, 6)) == []


def test_light_plan_can_call_topix_prices(monkeypatch) -> None:
    provider = _provider_without_init("light")
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_fetch(path: str, params: dict[str, str]):
        calls.append((path, params))
        return [{"date": "2026-03-06", "close": 2800}]

    monkeypatch.setattr(provider, "_get_paginated_records", fake_fetch)

    rows = provider.get_topix_prices(date(2026, 3, 1), date(2026, 3, 6))

    assert rows == [{"date": "2026-03-06", "close": 2800}]
    assert calls == [("/indices/bars/daily/topix", {"from": "2026-03-01", "to": "2026-03-06"})]


def test_cli_jquants_plan_overrides_config(monkeypatch, config_copy: dict) -> None:
    config_copy["jquants"]["plan"] = "free"
    monkeypatch.setattr(main_module, "JQUANTS_PLAN_OVERRIDE", "light")
    monkeypatch.setattr(main_module, "_load_jquants_config_file", lambda: {"plan": "free"})

    main_module._apply_jquants_plan_settings(config_copy)

    assert config_copy["jquants"]["plan"] == "light"
    assert config_copy["jquants"]["capability_status"]["topix_prices"] == "OK"


def test_free_plan_relative_strength_uses_prime_fallback(monkeypatch, config_copy: dict) -> None:
    config_copy["jquants"]["plan"] = "free"
    config_copy.setdefault("features", {})["relative_strength"] = True
    config_copy["features"]["topix_relative_strength"] = True
    config_copy.setdefault("scoring", {})["use_relative_strength_score"] = True
    monkeypatch.setattr(main_module, "JQUANTS_PLAN_OVERRIDE", "free")
    monkeypatch.setattr(main_module, "_load_jquants_config_file", lambda: {})

    main_module._apply_jquants_plan_settings(config_copy)

    assert main_module._relative_strength_enabled_for_indicators(config_copy) is True
    assert config_copy["features"]["topix_relative_strength"] is False
    assert config_copy["jquants"]["relative_strength_benchmark"] == "prime_market_average"


def test_preflight_displays_plan_and_capabilities(config_copy: dict) -> None:
    results: list[dict] = []
    config_copy["jquants"]["plan"] = "free"

    main_module._check_jquants_plan_capabilities(results, config_copy)

    messages = [item["message"] for item in results]
    assert "J-Quants Plan: free (source=config/jquants.yaml)" in messages
    assert "J-Quants capability prices: OK" in messages
    assert "J-Quants capability topix_prices: disabled" in messages
    assert "J-Quants capability investor_types: disabled" in messages
    assert "profile required capabilities: listed_info, prices" in messages
    assert "missing capabilities: none" in messages
    assert "fallback applied: none" in messages
    assert "can_run_backtest: true" in messages
    assert "can_run_live/paper: true" in messages
    assert any(message.startswith("prices earliest:") for message in messages)


def test_resolve_jquants_plan_prefers_jquants_yaml_over_provider_yaml(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "jquants.yaml").write_text(
        "jquants:\n"
        "  plan: light\n"
        "  plans:\n"
        "    light:\n"
        "      requests_per_minute: 60\n"
        "      parallel_fetch: true\n",
        encoding="utf-8",
    )
    (config_dir / "provider.yaml").write_text("jquants:\n  plan: free\n", encoding="utf-8")

    resolution = resolve_jquants_plan(config_root=tmp_path)

    assert resolution.plan == "light"
    assert resolution.source == "config/jquants.yaml"
    assert resolution.capabilities["topix_prices"] == "OK"
    assert resolution.requests_per_minute == 60
    assert resolution.parallel_fetch is True
    assert resolution.warnings


def test_jquants_earliest_supported_date_from_config() -> None:
    config = {
        "jquants": {
            "plan": "light",
            "earliest_supported_date": {
                "free": "2024-01-01",
                "light": "2021-05-01",
            },
        }
    }

    assert main_module.jquants_earliest_supported_date(config, "prices") == date(2021, 5, 1)


def test_free_light_profile_compatibility_matrix() -> None:
    expectations = {
        ("free", "rookie_dealer_02_v2_1"): ([], []),
        ("free", "rookie_dealer_02_v2_6"): (["topix_prices"], ["topix_prices"]),
        ("free", "rookie_dealer_02_v2_9"): ([], []),
        ("free", "rookie_dealer_02_v2_10"): ([], []),
        ("light", "rookie_dealer_02_v2_1"): ([], []),
        ("light", "rookie_dealer_02_v2_6"): ([], []),
        ("light", "rookie_dealer_02_v2_9"): ([], []),
        ("light", "rookie_dealer_02_v2_10"): ([], []),
    }
    for (plan, profile_id), (missing, fallback_capabilities) in expectations.items():
        compatibility = jquants_profile_compatibility(profile_id, plan)
        assert compatibility["missing_capabilities"] == missing
        assert [item["capability"] for item in compatibility["fallback_applied"]] == fallback_capabilities
        assert compatibility["can_run_backtest"] is True
        assert compatibility["can_run_paper"] is True


def test_all_matrix_profiles_load() -> None:
    for profile_id in [
        "rookie_dealer_02_v2_1",
        "rookie_dealer_02_v2_6",
        "rookie_dealer_02_v2_9",
        "rookie_dealer_02_v2_10",
    ]:
        assert load_profile(profile_id)["profile_id"] == profile_id


def test_preflight_warns_but_allows_free_v2_6_fallback() -> None:
    results: list[dict] = []
    config = load_profile("rookie_dealer_02_v2_6")
    config["jquants"]["plan"] = "free"

    main_module._check_jquants_plan_capabilities(results, config)

    messages = [item["message"] for item in results]
    assert "missing capabilities: topix_prices" in messages
    assert any(message.startswith("fallback applied: topix_prices ->") for message in messages)
    assert "can_run_backtest: true" in messages

