"""Broker execution boundary.

Only PaperBroker is usable today. Tachibana and KabuStation stubs exist to make
the future live-trading boundary explicit without implementing live trading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tachibana_auth import build_login_payload, load_private_key, load_tachibana_auth_config


class LiveTradingDisabledError(RuntimeError):
    """Raised whenever a live trading path is attempted while disabled."""


class BaseBroker(ABC):
    @abstractmethod
    def place_buy_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def place_sell_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_cash(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        raise NotImplementedError


class PaperBroker(BaseBroker):
    provider = "paper"

    def __init__(self, state: dict[str, Any], config: dict[str, Any]) -> None:
        self.state = state
        self.config = config
        self.orders: dict[str, dict[str, Any]] = {}

    def place_buy_order(self, order: dict[str, Any]) -> dict[str, Any]:
        return self._fill_order(order, "BUY")

    def place_sell_order(self, order: dict[str, Any]) -> dict[str, Any]:
        return self._fill_order(order, "SELL")

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.state.get("positions", []))

    def get_cash(self) -> float:
        return float(self.state.get("cash", 0))

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return self.orders.get(order_id, {"order_id": order_id, "order_status": "UNKNOWN"})

    def _fill_order(self, order: dict[str, Any], action: str) -> dict[str, Any]:
        order_id = str(order.get("order_id") or order.get("trade_id") or self._next_order_id(action, order))
        result = {
            **order,
            "action": action,
            "order_id": order_id,
            "order_status": "FILLED",
            "broker_provider": self.provider,
            "live_trading": False,
            "safety_checked": True,
        }
        self.orders[order_id] = result
        return result

    def _next_order_id(self, action: str, order: dict[str, Any]) -> str:
        code = order.get("code", "UNKNOWN")
        date = order.get("entry_date") or order.get("exit_date") or order.get("date") or "NO_DATE"
        return f"PAPER-{date}-{code}-{action}-{len(self.orders) + 1:04d}"


class KabuStationBrokerStub(BaseBroker):
    provider = "kabu_station"
    disabled_message = "kabuステーションAPI接続は未実装です。現在はPaperBrokerのみ利用可能です。"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        broker = config.get("broker", {})
        safety = config.get("safety", {})
        kabu_station = config.get("kabu_station", {})
        self.kabu_station = kabu_station
        self.api_base_url = _clean_config_string(kabu_station.get("api_base_url", "http://localhost:18080/kabusapi"))
        self.api_password_env = _clean_config_string(kabu_station.get("api_password_env", "KABU_STATION_API_PASSWORD"))
        self.symbol_exchange = int(kabu_station.get("symbol_exchange", 1))
        self.product_type = int(kabu_station.get("product_type", 0))
        self.account_type = int(kabu_station.get("account_type", 4))
        self.order_type = str(kabu_station.get("order_type", "market"))
        self.time_in_force = str(kabu_station.get("time_in_force", "day"))
        self.live_conditions = {
            "kabu_station.enabled": bool(kabu_station.get("enabled", False)),
            "broker.live_trading_enabled": bool(broker.get("live_trading_enabled", False)),
            "safety.allow_live_trading": bool(safety.get("allow_live_trading", False)),
        }
        if not all(self.live_conditions.values()):
            raise LiveTradingDisabledError(self.disabled_message)
        raise NotImplementedError(self.disabled_message)

    def get_token(self) -> str:
        raise NotImplementedError(self.disabled_message)

    def place_buy_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(self.disabled_message)

    def place_sell_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(self.disabled_message)

    def get_positions(self) -> list[dict[str, Any]]:
        raise NotImplementedError(self.disabled_message)

    def get_cash(self) -> float:
        raise NotImplementedError(self.disabled_message)

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        raise NotImplementedError(self.disabled_message)


class TachibanaBrokerStub(BaseBroker):
    provider = "tachibana"
    disabled_message = "立花証券 e支店 API 接続は未実装です。現在はPaperBrokerのみ利用可能です。"

    def __init__(self, config: dict[str, Any], provider: str) -> None:
        self.config = config
        self.provider = provider
        self.tachibana = config.get("tachibana", {})
        self.environment = _clean_config_string(self.tachibana.get("environment", "demo"))
        self.demo_base_url = _clean_config_string(self.tachibana.get("demo_base_url", "https://demo-kabuka.e-shiten.jp/e_api_v4r9/"))
        self.live_base_url = _clean_config_string(self.tachibana.get("live_base_url", "https://kabuka.e-shiten.jp/e_api_v4r9/"))
        self.auth_method = _clean_config_string(self.tachibana.get("auth_method", "public_key_v4r9"))
        self.user_id_env = _clean_config_string(self.tachibana.get("user_id_env", "TACHIBANA_USER_ID"))
        self.password_env = _clean_config_string(self.tachibana.get("password_env", "TACHIBANA_PASSWORD"))
        self.second_password_env = _clean_config_string(self.tachibana.get("second_password_env", "TACHIBANA_SECOND_PASSWORD"))
        self.private_key_path_env = _clean_config_string(self.tachibana.get("private_key_path_env", "TACHIBANA_PRIVATE_KEY_PATH"))
        self.public_key_id_env = _clean_config_string(self.tachibana.get("public_key_id_env", "TACHIBANA_PUBLIC_KEY_ID"))
        self.request_timeout_seconds = int(self.tachibana.get("request_timeout_seconds", 10))
        self.account_type = _clean_config_string(self.tachibana.get("account_type", "cash"))
        self.product = _clean_config_string(self.tachibana.get("product", "stock"))
        self.market = _clean_config_string(self.tachibana.get("market", "tse"))

    def get_token(self) -> str:
        raise NotImplementedError(self.disabled_message)

    def place_buy_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(self.disabled_message)

    def place_sell_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(self.disabled_message)

    def get_positions(self) -> list[dict[str, Any]]:
        raise NotImplementedError(self.disabled_message)

    def get_cash(self) -> float:
        raise NotImplementedError(self.disabled_message)

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        raise NotImplementedError(self.disabled_message)


class TachibanaDemoBrokerStub(TachibanaBrokerStub):
    provider = "tachibana_demo"
    disabled_message = "立花証券 e支店 API デモ接続は未実装です。現在はPaperBrokerのみ利用可能です。"
    stub_message = "Tachibana demo API connection is not implemented yet."

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.provider)
        if self.environment != "demo":
            raise LiveTradingDisabledError("tachibana.environment が demo ではないため TachibanaDemoBroker は利用できません。")

    def login(self) -> dict[str, Any]:
        return self._stub_response("login", {"auth_method": self.auth_method})

    def logout(self) -> dict[str, Any]:
        return self._stub_response("logout")

    def load_auth_config(self) -> dict[str, Any]:
        auth_config = load_tachibana_auth_config(self.config)
        return {**auth_config.__dict__, "status": "stub", "broker": self.provider, "message": self.stub_message}

    def load_private_key(self) -> dict[str, Any]:
        auth_config = load_tachibana_auth_config(self.config)
        if not auth_config.private_key_path:
            return {
                "status": "stub",
                "broker": self.provider,
                "auth_method": self.auth_method,
                "path_set": False,
                "file_exists": False,
                "private_key_loaded": False,
                "message": "Private key path is not set.",
            }
        return {"status": "stub", "broker": self.provider, "auth_method": self.auth_method, **load_private_key(auth_config.private_key_path)}

    def build_auth_request(self) -> dict[str, Any]:
        return {"broker": self.provider, **build_login_payload(load_tachibana_auth_config(self.config))}

    def get_account_info(self) -> dict[str, Any]:
        return self._stub_response("get_account_info")

    def get_positions(self) -> dict[str, Any]:
        return self._stub_response("get_positions", {"positions": []})

    def get_cash(self) -> dict[str, Any]:
        return self._stub_response("get_cash", {"cash": None})

    def place_buy_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise LiveTradingDisabledError("Tachibana demo order sending is not implemented. No order was sent.")

    def place_sell_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise LiveTradingDisabledError("Tachibana demo order sending is not implemented. No order was sent.")

    def _stub_response(self, operation: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "status": "stub",
            "broker": self.provider,
            "environment": self.environment,
            "operation": operation,
            "message": self.stub_message,
            **(extra or {}),
        }


class TachibanaLiveBrokerStub(TachibanaBrokerStub):
    provider = "tachibana_live"
    disabled_message = "立花証券 e支店 API ライブ接続は未実装です。現在はPaperBrokerのみ利用可能です。"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config, self.provider)
        broker = config.get("broker", {})
        safety = config.get("safety", {})
        self.live_conditions = {
            "broker.live_trading_enabled": bool(broker.get("live_trading_enabled", False)),
            "safety.allow_live_trading": bool(safety.get("allow_live_trading", False)),
            "tachibana.environment == live": self.environment == "live",
        }
        if not all(self.live_conditions.values()):
            raise LiveTradingDisabledError("立花証券 live broker は安全条件を満たすまで利用できません。")
        raise NotImplementedError(self.disabled_message)


def build_broker(state: dict[str, Any], config: dict[str, Any]) -> BaseBroker:
    provider = config.get("broker", {}).get("provider", "paper")
    if provider == "paper":
        return PaperBroker(state, config)
    if provider == "tachibana_demo":
        return TachibanaDemoBrokerStub(config)
    if provider == "tachibana_live":
        return TachibanaLiveBrokerStub(config)
    if provider == "kabu_station":
        return KabuStationBrokerStub(config)
    raise ValueError(f"Unsupported broker provider: {provider}")


def _clean_config_string(value: Any) -> str:
    return str(value).strip().strip('"').strip("'")
