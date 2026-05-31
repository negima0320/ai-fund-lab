from __future__ import annotations

import pytest

from broker import (
    KabuStationBrokerStub,
    LiveTradingDisabledError,
    PaperBroker,
    TachibanaDemoBroker,
    TachibanaDemoBrokerStub,
    TachibanaLiveBroker,
    TachibanaLiveBrokerStub,
    account_snapshot,
    build_broker,
    render_account_snapshot,
)


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
    assert broker.login()["status"] == "read_only"
    assert broker.get_account_info()["broker"] == "tachibana_demo"
    assert broker.get_positions() == []
    assert broker.get_orders() == []
    assert broker.get_executions() == []
    result = broker.place_buy_order({"code": "1001", "amount": 100000})
    assert result["order_status"] == "DEMO_ACCEPTED"
    assert result["broker_provider"] == "tachibana_demo"


def test_tachibana_live_stub_requires_safety_locks(config_copy: dict) -> None:
    config_copy["broker"]["provider"] = "tachibana_live"
    with pytest.raises(LiveTradingDisabledError):
        TachibanaLiveBrokerStub(config_copy)


def test_build_broker_paper_does_not_use_live_trading(config_copy: dict) -> None:
    broker = build_broker({"cash": 1000000, "positions": []}, config_copy)
    result = broker.place_sell_order({"code": "1001", "name": "Test", "amount": 100000, "exit_date": "2026-03-06"})
    assert result["broker_provider"] == "paper"
    assert result["live_trading"] is False


class FakeTachibanaReadOnlyClient:
    def get_account_balance(self) -> dict:
        return {"cash": 500000, "evaluation_amount": 250000}

    def get_positions(self) -> list[dict]:
        return [{"code": "1001", "name": "Read Only Position", "quantity": 100, "market_value": 250000}]

    def get_orders(self) -> list[dict]:
        return [{"order_id": "O-1", "code": "1001", "order_status": "ACCEPTED"}]

    def get_executions(self) -> list[dict]:
        return [{"execution_id": "E-1", "code": "1001", "quantity": 100}]


def test_tachibana_demo_read_only_snapshot(config_copy: dict) -> None:
    broker = TachibanaDemoBroker(config_copy, client=FakeTachibanaReadOnlyClient())

    snapshot = account_snapshot(broker)
    markdown = render_account_snapshot(snapshot)

    assert snapshot["broker_provider"] == "tachibana_demo"
    assert snapshot["cash"] == 500000
    assert snapshot["evaluation_amount"] == 250000
    assert snapshot["positions"][0]["code"] == "1001"
    assert snapshot["orders"][0]["order_id"] == "O-1"
    assert snapshot["today_executions"][0]["execution_id"] == "E-1"
    assert snapshot["read_only"] is True
    assert snapshot["order_submission_enabled"] is False
    assert "# Account Snapshot" in markdown
    assert "Cash: 500,000" in markdown
    assert "Evaluation Amount: 250,000" in markdown
    assert "## Positions" in markdown
    assert "## Today's Executions" in markdown
    result = broker.place_order({"action": "BUY", "code": "1001", "amount": 100000})
    assert result["order_status"] == "DEMO_ACCEPTED"


def test_build_broker_can_switch_to_tachibana_demo_read_only(config_copy: dict) -> None:
    config_copy["broker"]["provider"] = "tachibana_demo"
    broker = build_broker({}, config_copy)
    assert isinstance(broker, TachibanaDemoBroker)
    assert broker.get_account_balance()["cash"] == 0.0


def test_tachibana_live_read_only_can_be_created_when_environment_is_live(config_copy: dict) -> None:
    config_copy["tachibana"]["environment"] = "live"
    broker = TachibanaLiveBroker(config_copy, client=FakeTachibanaReadOnlyClient())
    assert broker.get_account_balance()["cash"] == 500000
    with pytest.raises(LiveTradingDisabledError):
        broker.place_sell_order({"code": "1001", "amount": 100000})
