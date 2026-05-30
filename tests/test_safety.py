from __future__ import annotations

from safety import can_trade


def test_stop_trading_file_stops_new_buys(config_copy: dict, tmp_path) -> None:
    stop_file = tmp_path / "STOP_TRADING"
    stop_file.write_text("", encoding="utf-8")
    config_copy["safety"]["emergency_stop_file"] = str(stop_file)
    result = can_trade({"action": "BUY", "amount": 100000}, {"today_orders": []}, config_copy)
    assert not result["allowed"]
    assert result["safety_rule"] == "emergency_stop"


def test_max_single_order_amount_rejects_large_order(config_copy: dict, tmp_path) -> None:
    config_copy["safety"]["emergency_stop_file"] = str(tmp_path / "STOP_TRADING")
    result = can_trade({"action": "BUY", "amount": 250000}, {"today_orders": []}, config_copy)
    assert not result["allowed"]
    assert result["safety_rule"] == "max_single_order_amount"


def test_live_trading_not_allowed_when_safety_disables_it(config_copy: dict, tmp_path) -> None:
    config_copy["safety"]["mode"] = "live"
    config_copy["safety"]["allow_live_trading"] = False
    config_copy["safety"]["emergency_stop_file"] = str(tmp_path / "STOP_TRADING")
    result = can_trade({"action": "BUY", "amount": 100000}, {"today_orders": []}, config_copy)
    assert not result["allowed"]
    assert result["safety_rule"] == "live_trading_disabled"
