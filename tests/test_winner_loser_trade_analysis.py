from __future__ import annotations

import csv
from pathlib import Path

from winner_loser_trade_analysis import build_winner_loser_trade_analysis, render_winner_loser_trade_analysis_markdown


def _write_trades(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "action",
        "code",
        "name",
        "sector_name",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "shares",
        "gross_profit",
        "gross_profit_rate",
        "net_profit",
        "net_profit_rate",
        "rsi",
        "volume_ratio",
        "holding_days",
        "exit_reason",
        "total_score",
        "market_regime",
        "relative_strength_score",
        "relative_strength_5d",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_winner_loser_trade_analysis_builds_summary_and_rules(tmp_path: Path) -> None:
    trades_path = tmp_path / "logs/backtests/test_profile/2021-01-01_to_2021-01-31/trades.csv"
    _write_trades(
        trades_path,
        [
            {
                "action": "SELL",
                "code": "1001",
                "name": "Winner",
                "sector_name": "機械",
                "entry_date": "2021-01-05",
                "exit_date": "2021-01-07",
                "entry_price": 1000,
                "exit_price": 1100,
                "shares": 100,
                "gross_profit": 10000,
                "gross_profit_rate": 0.10,
                "net_profit": 7968,
                "net_profit_rate": 0.079,
                "rsi": 55,
                "volume_ratio": 2.5,
                "holding_days": 2,
                "exit_reason": "利確",
                "total_score": 58,
                "market_regime": "risk_on",
                "relative_strength_score": 4,
                "relative_strength_5d": 0.04,
            },
            {
                "action": "SELL",
                "code": "1002",
                "name": "Loser",
                "sector_name": "小売業",
                "entry_date": "2021-01-06",
                "exit_date": "2021-01-08",
                "entry_price": 3000,
                "exit_price": 2850,
                "shares": 100,
                "gross_profit": -15000,
                "gross_profit_rate": -0.05,
                "net_profit": -15000,
                "net_profit_rate": -0.05,
                "rsi": 65,
                "volume_ratio": 4.5,
                "holding_days": 2,
                "exit_reason": "損切り",
                "total_score": 52,
                "market_regime": "neutral",
                "relative_strength_score": 0,
                "relative_strength_5d": -0.01,
            },
            {"action": "NO_BUY", "code": "", "gross_profit": "", "gross_profit_rate": ""},
        ],
    )

    analysis = build_winner_loser_trade_analysis(tmp_path, "test_profile", "2021-01-01", "2021-01-31")

    assert analysis["summary"]["total_trades"] == 2
    assert analysis["summary"]["win_count"] == 1
    assert analysis["summary"]["loss_count"] == 1
    assert analysis["summary"]["gross_profit_total"] == 10000
    assert analysis["summary"]["gross_loss_total"] == -15000
    assert analysis["profit_contribution"]["top_profit_sums"]["top_10"] == 10000
    assert "sector_name" in analysis["winner_common_features"]
    assert "loss_heavy_conditions" in analysis["difference_analysis"]

    markdown = render_winner_loser_trade_analysis_markdown(analysis)
    assert "## Closed Trade Summary" in markdown
    assert "## Winner / Loser Difference Ranking" in markdown
