from __future__ import annotations

import sqlite3

from db import initialize_database, save_trades
from main import count_csv_data_rows, write_trades_csv_from_db


def test_trades_csv_row_count_matches_database(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
    trades = [
        {
            "trade_id": "t-001",
            "action": "BUY",
            "code": "1001",
            "name": "Test One",
            "date": "2026-03-02",
            "entry_price": 1000,
            "shares": 100,
            "order_status": "FILLED",
        },
        {
            "trade_id": "t-002",
            "action": "SELL",
            "code": "1001",
            "name": "Test One",
            "entry_date": "2026-03-02",
            "exit_date": "2026-03-05",
            "entry_price": 1000,
            "exit_price": 1050,
            "shares": 100,
            "profit": 5000,
            "profit_rate": 0.05,
            "gross_profit": 5000,
            "net_profit": 3900,
            "result": "WIN",
            "order_status": "FILLED",
        },
    ]
    save_trades(config_copy, tmp_path, "2026-03-02", trades)

    csv_path, db_count, csv_count = write_trades_csv_from_db(config_copy, tmp_path)

    with sqlite3.connect(config_copy["database"]["path"]) as connection:
        actual_db_count = connection.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert db_count == actual_db_count == 2
    assert csv_count == actual_db_count
    assert count_csv_data_rows(csv_path) == actual_db_count
