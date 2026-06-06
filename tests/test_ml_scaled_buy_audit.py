from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.scaled_buy_audit import ScaledBuyAudit


PERIOD = "2023-01-01_to_2026-05-31"
PROFILE = "rookie_dealer_02_v2_71_ml_ranked_exit_ai_050_scaled_buy"


def test_scaled_buy_audit_reports_concentration_and_outputs(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "ml"
    report_dir.mkdir(parents=True)
    backtest_dir = tmp_path / "logs" / "backtests" / PROFILE / PERIOD
    backtest_dir.mkdir(parents=True)

    scaled_rows = [
        {
            "action": "SELL",
            "code": "67400",
            "entry_date": "2026-03-09",
            "exit_date": "2026-03-10",
            "shares": 19100,
            "net_profit": 500000,
            "original_amount": 982300,
            "scaled_amount": 897700,
            "original_planned_shares": 20900,
            "scaled_shares": 19100,
            "scaled_buy_triggered": True,
        },
        {
            "action": "SELL",
            "code": "10010",
            "entry_date": "2026-03-11",
            "exit_date": "2026-03-15",
            "shares": 100,
            "net_profit": -10000,
            "original_amount": 500000,
            "scaled_amount": 300000,
            "original_planned_shares": 500,
            "scaled_shares": 300,
            "scaled_buy_triggered": True,
        },
    ]
    pd.DataFrame(scaled_rows).to_csv(report_dir / "scaled_buy_trades_2023-01_to_2026-05.csv", index=False)
    pd.DataFrame(scaled_rows).to_csv(backtest_dir / "trades.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2026-03-09", "total_assets": 1_000_000},
            {"date": "2026-03-10", "total_assets": 1_500_000},
            {"date": "2026-03-15", "total_assets": 1_490_000},
        ]
    ).to_csv(backtest_dir / "summary.csv", index=False)
    (backtest_dir / "backtest_summary.json").write_text(
        json.dumps({"final_assets": 1_490_000, "net_cumulative_profit": 490000, "all_trades": scaled_rows}),
        encoding="utf-8",
    )
    (report_dir / "scaled_buy_backtest_comparison_2023-01_to_2026-05.json").write_text(
        json.dumps(
            {
                "summary": [
                    {"profile": "rookie_dealer_02_v2_68_ml_ranked_exit_ai_050", "net_profit": 100000, "total_trades": 1},
                    {"profile": PROFILE, "net_profit": 490000, "total_trades": 2},
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = ScaledBuyAudit(root=tmp_path)
    result = audit.build()
    paths = audit.save(result)

    assert result["scaled_buy_stats"]["count"] == 2
    assert result["scaled_buy_stats"]["total_profit"] == 490000
    assert result["exclusion_sensitivity"][1]["case"] == "exclude_67400"
    assert result["exclusion_sensitivity"][1]["total_profit"] == -10000
    assert result["concentration"]["summary"]["67400_contribution_rate"] == 500000 / 490000
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.trades_csv.exists()
