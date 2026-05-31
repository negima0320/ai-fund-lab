from __future__ import annotations

import pytest

from broker import TachibanaDemoBroker
from demo_auto_order import DemoAutoOrderBlocked, execute_demo_auto_orders, validate_demo_auto_order


def _schedule() -> dict:
    return {
        "execution_policy": {
            "execution_mode": "auto_demo",
            "auto_order_enabled": True,
            "broker": "tachibana_demo",
        },
        "safety": {
            "forbid_live_auto_order": True,
        },
    }


def _config(config: dict, broker: str = "tachibana_demo", env: str = "demo") -> dict:
    config["broker"]["provider"] = broker
    config["tachibana"]["environment"] = env
    config["portfolio"]["max_positions"] = 2
    config["safety"]["max_daily_buy_amount"] = 300000
    config["safety"]["max_single_order_amount"] = 200000
    config["safety"]["max_orders_per_day"] = 3
    return config


def _orders() -> list[dict]:
    return [
        {
            "action": "BUY",
            "code": "1001",
            "name": "Demo Buy",
            "shares": 100,
            "estimated_price": 1000,
            "estimated_amount": 100000,
        }
    ]


class DemoClient:
    def __init__(self, cash: float = 500000, positions: list[dict] | None = None, orders: list[dict] | None = None) -> None:
        self.cash = cash
        self.positions = positions or []
        self.orders = orders or []

    def get_account_balance(self) -> dict:
        return {"cash": self.cash, "evaluation_amount": 0}

    def get_positions(self) -> list[dict]:
        return self.positions

    def get_orders(self) -> list[dict]:
        return self.orders

    def get_executions(self) -> list[dict]:
        return []


def test_demo_broker_allows_demo_auto_order(config_copy: dict) -> None:
    config = _config(config_copy)
    broker = TachibanaDemoBroker(config, client=DemoClient())

    result = execute_demo_auto_orders(config, _schedule(), _orders(), broker)

    assert result["status"] == "ordered"
    assert result["validation"]["allowed"] is True
    assert result["orders"][0]["order_status"] == "DEMO_ACCEPTED"
    assert result["orders"][0]["broker_provider"] == "tachibana_demo"
    assert result["orders"][0]["live_trading"] is False


def test_live_broker_is_always_blocked(config_copy: dict) -> None:
    config = _config(config_copy, broker="tachibana_live", env="live")

    with pytest.raises(DemoAutoOrderBlocked, match="env=live"):
        validate_demo_auto_order(config, _schedule(), _orders(), {"cash": 500000}, [])


def test_forbid_live_auto_order_blocks_live_broker(config_copy: dict) -> None:
    config = _config(config_copy, broker="tachibana_live", env="demo")

    with pytest.raises(DemoAutoOrderBlocked, match="broker=tachibana_live"):
        validate_demo_auto_order(config, _schedule(), _orders(), {"cash": 500000}, [])


def test_auto_order_disabled_blocks_orders(config_copy: dict) -> None:
    config = _config(config_copy)
    schedule = _schedule()
    schedule["execution_policy"]["auto_order_enabled"] = False

    with pytest.raises(DemoAutoOrderBlocked, match="auto_order_enabled=false"):
        validate_demo_auto_order(config, schedule, _orders(), {"cash": 500000}, [])


def test_cash_shortage_blocks_orders(config_copy: dict) -> None:
    config = _config(config_copy)

    with pytest.raises(DemoAutoOrderBlocked, match="cash不足"):
        validate_demo_auto_order(config, _schedule(), _orders(), {"cash": 99999}, [])


def test_existing_position_blocks_duplicate_buy(config_copy: dict) -> None:
    config = _config(config_copy)

    with pytest.raises(DemoAutoOrderBlocked, match="同一銘柄保有中"):
        validate_demo_auto_order(config, _schedule(), _orders(), {"cash": 500000}, [{"code": "1001"}])


def test_max_positions_blocks_orders(config_copy: dict) -> None:
    config = _config(config_copy)

    with pytest.raises(DemoAutoOrderBlocked, match="max_positions超過"):
        validate_demo_auto_order(config, _schedule(), _orders(), {"cash": 500000}, [{"code": "2001"}, {"code": "2002"}])


def test_order_limits_block_orders(config_copy: dict) -> None:
    config = _config(config_copy)
    config["safety"]["max_daily_buy_amount"] = 50000

    with pytest.raises(DemoAutoOrderBlocked, match="max_daily_buy_amount超過"):
        validate_demo_auto_order(config, _schedule(), _orders(), {"cash": 500000}, [])

    config["safety"]["max_daily_buy_amount"] = 300000
    config["safety"]["max_single_order_amount"] = 50000
    with pytest.raises(DemoAutoOrderBlocked, match="max_single_order_amount超過"):
        validate_demo_auto_order(config, _schedule(), _orders(), {"cash": 500000}, [])

    config["safety"]["max_single_order_amount"] = 200000
    with pytest.raises(DemoAutoOrderBlocked, match="当日注文上限超過"):
        validate_demo_auto_order(config, _schedule(), _orders(), {"cash": 500000}, [], [{"code": "1"}, {"code": "2"}, {"code": "3"}])
