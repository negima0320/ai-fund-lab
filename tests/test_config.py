from __future__ import annotations

from profile_loader import load_profile


def test_rookie_dealer_config_loads(config: dict) -> None:
    assert config["dealer"]["id"] == "rookie_dealer_01"
    assert config["profile_id"] == "rookie_dealer_01"


def test_required_config_keys_exist(config: dict) -> None:
    assert config["portfolio"]["initial_cash"] > 0
    for key in [
        "data_provider",
        "trading",
        "selection",
        "costs",
        "database",
        "news",
        "ai_commentary",
        "ai_decision",
        "safety",
        "broker",
        "tachibana",
    ]:
        assert key in config


def test_rookie_dealer_02_profile_uses_intraday_stop() -> None:
    profile = load_profile("rookie_dealer_02")

    assert profile["profile_id"] == "rookie_dealer_02"
    assert profile["profile_name"] == "新人ディーラー2号"
    assert profile["execution"]["stop_loss_execution"] == "intraday_stop"
