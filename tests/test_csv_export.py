from __future__ import annotations

import sqlite3

from db import database_schema_check, initialize_database, save_pending_orders, save_scoring_results, save_trades
from main import count_csv_data_rows, write_trades_csv_from_db
from profile_loader import load_profile


def test_trades_csv_row_count_matches_database(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
    trades = [
        {
            "trade_id": "t-001",
            "action": "BUY",
            "code": "1001",
            "name": "Test One",
            "signal_date": "2026-03-01",
            "date": "2026-03-02",
            "entry_price": 1000,
            "entry_price_source": "open",
            "signal_close_price": 990,
            "entry_open_price": 1000,
            "entry_gap_rate": 0.0101,
            "shares": 100,
            "order_status": "FILLED",
        },
        {
            "trade_id": "t-002",
            "action": "SELL",
            "code": "1001",
            "name": "Test One",
            "signal_date": "2026-03-01",
            "entry_date": "2026-03-02",
            "exit_date": "2026-03-05",
            "entry_price": 1000,
            "entry_price_source": "open",
            "signal_close_price": 990,
            "entry_open_price": 1000,
            "entry_gap_rate": 0.0101,
            "exit_price": 1050,
            "shares": 100,
            "profit": 5000,
            "profit_rate": 0.05,
            "gross_profit": 5000,
            "net_profit": 3900,
            "result": "WIN",
            "order_status": "FILLED",
        },
        {
            "trade_id": "t-003",
            "action": "SELL",
            "code": "1002",
            "name": "Pending",
            "entry_date": "2026-03-02",
            "exit_date": "2026-03-03",
            "order_status": "PENDING",
        },
        {
            "trade_id": "t-004",
            "action": "BUY",
            "code": "1003",
            "name": "Rejected",
            "date": "2026-03-02",
            "order_status": "REJECTED",
        },
    ]
    save_trades(config_copy, tmp_path, "2026-03-02", trades)

    csv_path, db_count, csv_count = write_trades_csv_from_db(config_copy, tmp_path)

    with sqlite3.connect(config_copy["database"]["path"]) as connection:
        actual_db_count = connection.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert db_count == actual_db_count == 2
    assert csv_count == actual_db_count
    assert count_csv_data_rows(csv_path) == actual_db_count
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "signal_date" in csv_text.splitlines()[0]
    assert "2026-03-01" in csv_text
    with sqlite3.connect(config_copy["database"]["path"]) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(trades)")]
    assert "signal_date" in columns
    assert "entry_price_source" in columns


def test_profile_outputs_are_separated(tmp_path) -> None:
    profile_01 = load_profile("rookie_dealer_01")
    profile_02 = load_profile("rookie_dealer_02")
    profile_01["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    profile_02["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(profile_01, tmp_path)

    path_01, _count_01, _csv_01 = write_trades_csv_from_db(profile_01, tmp_path)
    path_02, _count_02, _csv_02 = write_trades_csv_from_db(profile_02, tmp_path)

    assert path_01 != path_02
    assert "rookie_dealer_01" in str(path_01)
    assert "rookie_dealer_02" in str(path_02)


def test_save_pending_orders_column_count_matches_schema(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)

    save_pending_orders(
        config_copy,
        tmp_path,
        [
            {
                "order_id": "order-001",
                "action": "BUY",
                "code": "1001",
                "name": "Pending Buy",
                "created_date": "2026-03-02",
                "scheduled_execution_date": "2026-03-03",
                "intended_price": 1000,
                "status": "PENDING",
                "score": 45,
                "reason": "manual approval",
            }
        ],
    )

    with sqlite3.connect(config_copy["database"]["path"]) as connection:
        count = connection.execute("SELECT COUNT(*) FROM pending_orders").fetchone()[0]

    assert count == 1


def test_db_check_reports_pending_order_insert_columns(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")

    result = database_schema_check(config_copy, tmp_path)
    pending = next(item for item in result["tables"] if item["table"] == "pending_orders")

    assert pending["status"] == "OK"
    assert pending["expected_insert_column_count"] == 12
    assert pending["insert_missing_in_schema"] == []


def test_db_save_with_scores_no_selection_and_pending_order(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)

    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-02",
            "source_provider": "jquants",
            "scores": [
                {
                    "code": "1001",
                    "name": "Rejected Candidate",
                    "rank": 1,
                    "total_score": 40,
                    "selected": False,
                    "rejected_reason": "score below threshold",
                    "relative_strength_score": 0,
                    "investor_context_score": 0,
                    "earnings_filter_checked": False,
                }
            ],
        },
    )
    save_trades(config_copy, tmp_path, "2026-03-02", [])
    save_pending_orders(
        config_copy,
        tmp_path,
        [
            {
                "order_id": "existing-pending",
                "action": "BUY",
                "code": "1002",
                "name": "Existing Pending",
                "created_date": "2026-03-01",
                "scheduled_execution_date": "2026-03-02",
                "status": "PENDING",
            }
        ],
    )

    with sqlite3.connect(config_copy["database"]["path"]) as connection:
        scoring_count = connection.execute("SELECT COUNT(*) FROM scoring_results").fetchone()[0]
        trade_count = connection.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        pending_count = connection.execute("SELECT COUNT(*) FROM pending_orders").fetchone()[0]

    assert scoring_count == 1
    assert trade_count == 0
    assert pending_count == 1
