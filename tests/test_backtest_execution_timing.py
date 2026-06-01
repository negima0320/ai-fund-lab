from __future__ import annotations

from datetime import date

import main as main_module
from main import _entry_date_for_signal, _prepare_execution_candidates, write_trades_csv
from paper_trade import execute_real_data_paper_trade, initial_live_paper_state


def _selected_candidate(close: float = 1000.0) -> dict:
    return {
        "code": "1001",
        "name": "Timing Test",
        "sector_name": "Test",
        "section": "TSEPrime",
        "market_section": "TSEPrime",
        "listing_market": "TSEPrime",
        "date": "2026-01-05",
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000,
        "selected": True,
        "total_score": 99,
        "confidence": 1.0,
        "reason": "test",
    }


def test_signal_date_enters_on_next_business_day_open(config_copy: dict, monkeypatch) -> None:
    config_copy.setdefault("backtest", {})["entry_timing"] = "next_business_day_open"
    config_copy["execution"]["use_next_day_open_execution"] = False
    config_copy["selection"]["fallback_min_score"] = 1
    config_copy["portfolio"]["max_positions"] = 1
    monkeypatch.setattr(
        main_module,
        "load_cached_prime_prices",
        lambda target_date: [
            {
                "code": "1001",
                "name": "Timing Test",
                "open": 1030.0,
                "high": 1050.0,
                "low": 1020.0,
                "close": 1040.0,
                "volume": 2000,
            }
        ],
    )

    execution_candidates = _prepare_execution_candidates([_selected_candidate()], "2026-01-05", "2026-01-06", config_copy)
    state, _summary, trades = execute_real_data_paper_trade(
        execution_candidates,
        initial_live_paper_state(config_copy),
        config_copy,
        "2026-01-06",
    )

    buy = next(trade for trade in trades if trade.get("action") == "BUY")
    assert buy["signal_date"] == "2026-01-05"
    assert buy["entry_date"] == "2026-01-06"
    assert buy["entry_price"] == 1030.0
    assert buy["entry_price_source"] == "open"
    assert buy["signal_close_price"] == 1000.0
    assert buy["entry_open_price"] == 1030.0
    assert buy["entry_gap_rate"] == 0.03
    assert state["positions"][0]["current_price"] == 1040.0


def test_friday_signal_uses_next_cached_trading_day(config_copy: dict) -> None:
    trading_dates = [date(2026, 1, 9), date(2026, 1, 13)]
    assert _entry_date_for_signal(date(2026, 1, 9), trading_dates, config_copy) == date(2026, 1, 13)


def test_same_day_close_uses_signal_date_and_close(config_copy: dict, monkeypatch) -> None:
    config_copy.setdefault("backtest", {})["entry_timing"] = "same_day_close"
    config_copy["execution"]["use_next_day_open_execution"] = False
    config_copy["selection"]["fallback_min_score"] = 1
    monkeypatch.setattr(
        main_module,
        "load_cached_prime_prices",
        lambda target_date: [{"code": "1001", "open": 990.0, "high": 1010.0, "low": 980.0, "close": 1005.0}],
    )

    entry_date = _entry_date_for_signal(date(2026, 1, 5), [date(2026, 1, 5), date(2026, 1, 6)], config_copy)
    execution_candidates = _prepare_execution_candidates([_selected_candidate()], "2026-01-05", entry_date.isoformat(), config_copy)
    state, _summary, trades = execute_real_data_paper_trade(
        execution_candidates,
        initial_live_paper_state(config_copy),
        config_copy,
        "2026-01-05",
    )

    buy = next(trade for trade in trades if trade.get("action") == "BUY")
    assert buy["entry_date"] == "2026-01-05"
    assert buy["entry_price"] == 1005.0
    assert buy["entry_price_source"] == "close"
    assert state["positions"][0]["entry_price"] == 1005.0


def test_trades_csv_contains_signal_and_entry_columns(tmp_path) -> None:
    path = tmp_path / "trades.csv"
    write_trades_csv(
        path,
        [
            {
                "trade_id": "t1",
                "action": "SELL",
                "code": "1001",
                "signal_date": "2026-01-05",
                "entry_date": "2026-01-06",
                "entry_price_source": "open",
            }
        ],
    )

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "signal_date" in header
    assert "entry_date" in header
    assert "entry_price_source" in header
