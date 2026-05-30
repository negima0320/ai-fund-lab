from __future__ import annotations


def test_rookie_dealer_config_loads(config: dict) -> None:
    assert config["dealer"]["id"] == "rookie_dealer_001"


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
