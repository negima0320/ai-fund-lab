"""Broker execution boundary.

PaperBroker can fill simulated orders. Tachibana brokers are read-only until
live trading is explicitly implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tachibana_auth import build_login_payload, load_private_key, load_tachibana_auth_config


class LiveTradingDisabledError(RuntimeError):
    """Raised whenever a live trading path is attempted while disabled."""


class BaseBroker(ABC):
    provider = "unknown"

    @abstractmethod
    def get_account_balance(self) -> dict[str, Any]:
        raise NotImplementedError

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
    def get_orders(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_executions(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_cash(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def place_order(self, order: dict[str, Any]) -> dict[str, Any]:
        action = str(order.get("action") or order.get("side") or "").upper()
        if action == "BUY":
            return self.place_buy_order(order)
        if action == "SELL":
            return self.place_sell_order(order)
        raise ValueError(f"Unsupported order action: {action}")


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

    def get_account_balance(self) -> dict[str, Any]:
        positions = self.get_positions()
        evaluation_amount = sum(float(item.get("market_value") or item.get("amount") or 0) for item in positions)
        return {
            "broker": self.provider,
            "cash": self.get_cash(),
            "evaluation_amount": evaluation_amount,
            "positions_count": len(positions),
        }

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.state.get("positions", []))

    def get_orders(self) -> list[dict[str, Any]]:
        return list(self.orders.values())

    def get_executions(self) -> list[dict[str, Any]]:
        return [order for order in self.orders.values() if order.get("order_status") == "FILLED"]

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

    def get_account_balance(self) -> dict[str, Any]:
        raise NotImplementedError(self.disabled_message)

    def get_orders(self) -> list[dict[str, Any]]:
        raise NotImplementedError(self.disabled_message)

    def get_executions(self) -> list[dict[str, Any]]:
        raise NotImplementedError(self.disabled_message)

    def get_cash(self) -> float:
        raise NotImplementedError(self.disabled_message)

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        raise NotImplementedError(self.disabled_message)


class TachibanaBroker(BaseBroker):
    provider = "tachibana"
    readonly_message = "Tachibana read only mode. Order sending is disabled."

    def __init__(self, config: dict[str, Any], provider: str, client: Any | None = None) -> None:
        self.config = config
        self.provider = provider
        self.client = client
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
        raise NotImplementedError("Tachibana authentication request is not implemented in read only mode.")

    def get_account_balance(self) -> dict[str, Any]:
        raw = self._client_call("get_account_balance", default={})
        return {
            "broker": self.provider,
            "environment": self.environment,
            "cash": _first_number(raw, ["cash", "Cash", "cash_balance", "available_cash"], 0.0),
            "evaluation_amount": _first_number(raw, ["evaluation_amount", "EvaluationAmount", "positions_value", "market_value"], 0.0),
            "raw": raw,
        }

    def get_account_info(self) -> dict[str, Any]:
        return {
            "broker": self.provider,
            "environment": self.environment,
            "read_only": True,
            "message": self.readonly_message,
        }

    def get_positions(self) -> list[dict[str, Any]]:
        raw = self._client_call("get_positions", default=[])
        return _list_payload(raw, "positions")

    def get_orders(self) -> list[dict[str, Any]]:
        raw = self._client_call("get_orders", default=[])
        return _list_payload(raw, "orders")

    def get_executions(self) -> list[dict[str, Any]]:
        raw = self._client_call("get_executions", default=[])
        return _list_payload(raw, "executions")

    def get_cash(self) -> float:
        return float(self.get_account_balance().get("cash") or 0.0)

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        orders = self.get_orders()
        return next((order for order in orders if str(order.get("order_id") or order.get("id")) == str(order_id)), {"order_id": order_id, "order_status": "UNKNOWN"})

    def place_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise LiveTradingDisabledError("Tachibana read only mode: place_order is not implemented. No order was sent.")

    def place_buy_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise LiveTradingDisabledError("Tachibana read only mode: buy order sending is disabled. No order was sent.")

    def place_sell_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise LiveTradingDisabledError("Tachibana read only mode: sell order sending is disabled. No order was sent.")

    def _client_call(self, method_name: str, default: Any) -> Any:
        if self.client is None:
            return default
        method = getattr(self.client, method_name, None)
        if method is None:
            return default
        return method()


class TachibanaBrokerStub(TachibanaBroker):
    provider = "tachibana"


class TachibanaDemoBroker(TachibanaBroker):
    provider = "tachibana_demo"
    stub_message = "Tachibana demo API read only mode is available. Order sending is disabled."
    order_message = "Tachibana demo order accepted by demo broker boundary."

    def __init__(self, config: dict[str, Any], client: Any | None = None) -> None:
        super().__init__(config, self.provider, client)
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
        return {**super().get_account_info(), "status": "read_only"}

    def place_order(self, order: dict[str, Any]) -> dict[str, Any]:
        action = str(order.get("action") or order.get("side") or "").upper()
        if action not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported order action: {action}")
        raw = self._client_call("place_order", default=None)
        if isinstance(raw, dict):
            return {
                **order,
                **raw,
                "action": action,
                "broker_provider": self.provider,
                "environment": "demo",
                "live_trading": False,
                "order_status": raw.get("order_status", raw.get("status", "DEMO_ACCEPTED")),
            }
        return {
            **order,
            "action": action,
            "order_id": order.get("order_id") or f"TACHIBANA-DEMO-{order.get('code', 'UNKNOWN')}-{action}",
            "order_status": "DEMO_ACCEPTED",
            "broker_provider": self.provider,
            "environment": "demo",
            "live_trading": False,
            "message": self.order_message,
        }

    def place_buy_order(self, order: dict[str, Any]) -> dict[str, Any]:
        return self.place_order({**order, "action": "BUY"})

    def place_sell_order(self, order: dict[str, Any]) -> dict[str, Any]:
        return self.place_order({**order, "action": "SELL"})

    def _stub_response(self, operation: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "status": "read_only",
            "broker": self.provider,
            "environment": self.environment,
            "operation": operation,
            "message": self.stub_message,
            **(extra or {}),
        }


class TachibanaDemoBrokerStub(TachibanaDemoBroker):
    pass


class TachibanaLiveBroker(TachibanaBroker):
    provider = "tachibana_live"
    stub_message = "Tachibana live API read only mode is available. Order sending is disabled."

    def __init__(self, config: dict[str, Any], client: Any | None = None) -> None:
        super().__init__(config, self.provider, client)
        if self.environment != "live":
            raise LiveTradingDisabledError("tachibana.environment が live ではないため TachibanaLiveBroker は利用できません。")

    def get_account_info(self) -> dict[str, Any]:
        return {**super().get_account_info(), "status": "read_only"}


class TachibanaLiveBrokerStub(TachibanaLiveBroker):
    pass


def account_snapshot(broker: BaseBroker) -> dict[str, Any]:
    balance = broker.get_account_balance()
    positions = broker.get_positions()
    orders = broker.get_orders()
    executions = broker.get_executions()
    return {
        "broker_provider": broker.provider,
        "account_balance": balance,
        "cash": balance.get("cash"),
        "evaluation_amount": balance.get("evaluation_amount"),
        "positions": positions,
        "orders": orders,
        "today_executions": executions,
        "read_only": broker.provider.startswith("tachibana"),
        "order_submission_enabled": broker.provider == "paper",
    }


def render_account_snapshot(snapshot: dict[str, Any]) -> str:
    lines = [
        "# Account Snapshot",
        "",
        f"Broker: {snapshot.get('broker_provider')}",
        f"Cash: {_format_money(snapshot.get('cash'))}",
        f"Evaluation Amount: {_format_money(snapshot.get('evaluation_amount'))}",
        "",
        "## Positions",
    ]
    lines.extend(_snapshot_item_lines(snapshot.get("positions", [])))
    lines.extend(["", "## Today's Executions"])
    lines.extend(_snapshot_item_lines(snapshot.get("today_executions", [])))
    return "\n".join(lines)


def build_broker(state: dict[str, Any], config: dict[str, Any]) -> BaseBroker:
    provider = config.get("broker", {}).get("provider", "paper")
    if provider == "paper":
        return PaperBroker(state, config)
    if provider == "tachibana_demo":
        return TachibanaDemoBroker(config)
    if provider == "tachibana_live":
        return TachibanaLiveBroker(config)
    if provider == "kabu_station":
        return KabuStationBrokerStub(config)
    raise ValueError(f"Unsupported broker provider: {provider}")


def _clean_config_string(value: Any) -> str:
    return str(value).strip().strip('"').strip("'")


def _first_number(payload: Any, keys: list[str], default: float) -> float:
    if not isinstance(payload, dict):
        return default
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return float(value)
    return default


def _list_payload(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    return []


def _snapshot_item_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- None"]
    return [
        "- " + ", ".join(f"{key}={value}" for key, value in item.items())
        for item in items
    ]


def _format_money(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.0f}"
