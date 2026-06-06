from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.exit_ai_trigger_audit import ExitAITriggerAudit


def _write_trades(root: Path, profile: str, rows: list[dict]) -> None:
    path = root / "logs" / "backtests" / profile / "2023-01-01_to_2026-05-31" / "trades.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_exit_ai_trigger_audit_matches_early_exit_by_entry_and_code(tmp_path: Path) -> None:
    base_rows = [
        {
            "action": "SELL",
            "trade_id": "2023-01-02_2023-01-20_1111",
            "code": "1111",
            "signal_date": "2023-01-01",
            "entry_date": "2023-01-02",
            "exit_date": "2023-01-20",
            "exit_reason": "max",
            "holding_days": 10,
            "net_profit": -10000,
            "net_profit_rate": -0.1,
        },
        {
            "action": "SELL",
            "trade_id": "2026-03-02_2026-03-20_2222",
            "code": "2222",
            "signal_date": "2026-03-01",
            "entry_date": "2026-03-02",
            "exit_date": "2026-03-20",
            "exit_reason": "max",
            "holding_days": 10,
            "net_profit": 20000,
            "net_profit_rate": 0.2,
        },
    ]
    exit_rows = [
        {
            "action": "SELL",
            "trade_id": "2023-01-02_2023-01-10_1111",
            "code": "1111",
            "signal_date": "2023-01-01",
            "entry_date": "2023-01-02",
            "exit_date": "2023-01-10",
            "exit_reason": "Exit AI",
            "holding_days": 5,
            "net_profit": -1000,
            "net_profit_rate": -0.01,
            "exit_ai_triggered": True,
            "exit_ai_probability": 0.61,
            "exit_ai_threshold": 0.5,
        },
        {
            "action": "SELL",
            "trade_id": "2026-03-02_2026-03-08_2222",
            "code": "2222",
            "signal_date": "2026-03-01",
            "entry_date": "2026-03-02",
            "exit_date": "2026-03-08",
            "exit_reason": "Exit AI",
            "holding_days": 4,
            "net_profit": 5000,
            "net_profit_rate": 0.05,
            "exit_ai_triggered": True,
            "exit_ai_probability": 0.62,
            "exit_ai_threshold": 0.5,
        },
    ]
    exit_060_rows = [{**row, "exit_ai_probability": 0.7, "exit_ai_threshold": 0.6} for row in exit_rows[:1]]

    _write_trades(tmp_path, "rookie_dealer_02_v2_66_ml_ranked", base_rows)
    _write_trades(tmp_path, "rookie_dealer_02_v2_68_ml_ranked_exit_ai_050", exit_rows)
    _write_trades(tmp_path, "rookie_dealer_02_v2_70_ml_ranked_exit_ai_060", exit_060_rows)

    dataset = pd.DataFrame(
        [
            {
                "trade_id": "x",
                "code": "1111",
                "entry_date": "2023-01-02",
                "current_date": "2023-01-10",
                "expected_return_10d": 0.01,
                "bad_entry_probability_10d": 0.7,
                "risk_adjusted_score": -0.34,
            }
        ]
    )
    dataset_path = tmp_path / "data" / "ml" / "exit_datasets" / "exit_dataset_v2_66_2023-01_to_2026-05.parquet"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(dataset_path)

    audit = ExitAITriggerAudit(root=tmp_path)
    result = audit.build()

    assert result["match_audit"]["v2_68_vs_v2_66"]["matched_count"] == 2
    assert result["v2_68_trigger_summary"]["trigger_count"] == 2
    assert result["v2_68_improvement_summary"]["improvement_total"] == 9000
    assert result["v2_68_improvement_summary"]["worsening_total"] == -15000
    assert result["march_2026_analysis"]["summary"]["triggered_delta"] == -15000
    assert result["v2_70_comparison"]["sets"][1]["set"] == "v2_68_only"

    paths = audit.save(result)
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.trigger_trades_csv.exists()
    assert paths.trade_delta_csv.exists()
