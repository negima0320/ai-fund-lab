from __future__ import annotations

import pytest

from broker import KabuStationBrokerStub, LiveTradingDisabledError, PaperBroker, TachibanaDemoBrokerStub, TachibanaLiveBrokerStub, build_broker


def test_paper_broker_fills_order(config_copy: dict) -> None:
    broker = PaperBroker({"cash": 1000000, "positions": []}, config_copy)
    result = broker.place_buy_order({"code": "1001", "name": "Test", "amount": 100000, "entry_date": "2026-03-06"})
    assert result["order_status"] == "FILLED"
    assert result["broker_provider"] == "paper"
    assert result["live_trading"] is False


def test_kabu_station_stub_raises(config_copy: dict) -> None:
    with pytest.raises(LiveTradingDisabledError):
        KabuStationBrokerStub(config_copy)


def test_tachibana_demo_stub_raises(config_copy: dict) -> None:
    broker = TachibanaDemoBrokerStub(config_copy)
    assert broker.login()["status"] == "stub"
    assert broker.get_account_info()["broker"] == "tachibana_demo"
    with pytest.raises(LiveTradingDisabledError):
        broker.place_buy_order({"code": "1001", "amount": 100000})


def test_tachibana_live_stub_requires_safety_locks(config_copy: dict) -> None:
    config_copy["broker"]["provider"] = "tachibana_live"
    with pytest.raises(LiveTradingDisabledError):
        TachibanaLiveBrokerStub(config_copy)


def test_build_broker_paper_does_not_use_live_trading(config_copy: dict) -> None:
    broker = build_broker({"cash": 1000000, "positions": []}, config_copy)
    result = broker.place_sell_order({"code": "1001", "name": "Test", "amount": 100000, "exit_date": "2026-03-06"})
    assert result["broker_provider"] == "paper"
    assert result["live_trading"] is False
