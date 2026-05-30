from __future__ import annotations

from ai_analysis import build_decision_records, render_ai_summary
from db import initialize_database, save_market_context, save_scoring_results, save_trades


def test_ai_summary_uses_entry_market_context_for_closed_trade(config_copy: dict, tmp_path) -> None:
    config_copy["database"]["path"] = str(tmp_path / "ai_fund_lab.sqlite3")
    initialize_database(config_copy, tmp_path)
    save_market_context(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-01",
            "provider": "test",
            "market_regime": "risk_on",
            "advance_ratio": 0.66,
        },
    )
    save_scoring_results(
        config_copy,
        tmp_path,
        {
            "date": "2026-03-01",
            "scores": [
                {
                    "code": "1001",
                    "name": "Entry Context",
                    "rank": 1,
                    "selected": True,
                    "total_score": 76,
                }
            ],
        },
    )
    save_trades(
        config_copy,
        tmp_path,
        "2026-03-06",
        [
            {
                "trade_id": "closed-with-entry-context",
                "action": "SELL",
                "code": "1001",
                "name": "Entry Context",
                "entry_date": "2026-03-01",
                "exit_date": "2026-03-06",
                "entry_price": 1000,
                "profit": 4000,
                "gross_profit": 4000,
                "net_profit": 4000,
                "result": "WIN",
                "order_status": "FILLED",
            }
        ],
    )

    records = build_decision_records(config_copy, tmp_path, "2026-03-01", "2026-03-06")
    closed = [record for record in records if record["future_result"]["result_available"]]

    assert closed[0]["market_context"]["market_regime"] == "risk_on"
    assert closed[0]["market_context"]["advance_ratio"] == 0.66
    markdown = render_ai_summary(config_copy, "2026-03-01", "2026-03-06", records, tmp_path / "dataset.jsonl")
    assert "- risk_on: 1件" in markdown
