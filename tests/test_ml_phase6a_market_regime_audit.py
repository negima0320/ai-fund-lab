from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase6a_market_regime_audit import Phase6AMarketRegimeAudit


PROFILE = "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
PERIOD = "2023-01-01_to_2026-05-31"


def _write_fixture(root: Path) -> None:
    log_dir = root / "logs" / "backtests" / PROFILE / PERIOD
    log_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2023-01-04", periods=100)
    assets = [1_000_000 + idx * 1_000 for idx in range(100)]
    assets[70] = 950_000
    pd.DataFrame(
        {
            "day": range(1, 101),
            "date": [day.strftime("%Y-%m-%d") for day in dates],
            "cash": [500_000] * 100,
            "positions_value": [500_000] * 100,
            "total_assets": assets,
            "daily_profit": [0] * 100,
            "open_positions_count": [2] * 100,
        }
    ).to_csv(log_dir / "summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "code": "11110",
                "entry_date": dates[10].strftime("%Y-%m-%d"),
                "exit_date": dates[13].strftime("%Y-%m-%d"),
                "net_profit": 10_000,
                "holding_days": 3,
                "entry_price": 100,
                "shares": 100,
                "pm_multiplier": 1.30,
            },
            {
                "action": "SELL",
                "code": "22220",
                "entry_date": dates[80].strftime("%Y-%m-%d"),
                "exit_date": dates[83].strftime("%Y-%m-%d"),
                "net_profit": -5_000,
                "holding_days": 3,
                "entry_price": 200,
                "shares": 100,
                "pm_multiplier": 0.80,
            },
        ]
    ).to_csv(log_dir / "trades.csv", index=False)
    pd.DataFrame(
        [
            {"entry_date": dates[10].strftime("%Y-%m-%d"), "buy_amount": 10_000},
            {"entry_date": dates[80].strftime("%Y-%m-%d"), "buy_amount": 20_000},
        ]
    ).to_csv(log_dir / "purchase_audit.csv", index=False)

    topix_dir = root / "data" / "cache" / "jquants" / "topix_prices"
    topix_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, day in enumerate(pd.bdate_range("2022-10-03", periods=180)):
        if idx < 100:
            close = 1000 + idx * 2
        elif idx < 135:
            close = 1200 - (idx - 100) * 2
        else:
            close = 1130 - (idx - 135) * 6
        records.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
            }
        )
    (topix_dir / "2022-10-03_to_2023-06-09.json").write_text(json.dumps({"records": records}), encoding="utf-8")

    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for day in dates[:5]:
        (processed / f"market_context_{day.strftime('%Y-%m-%d')}.json").write_text(
            json.dumps({"advance_ratio": 0.6, "average_change_rate": 0.01, "market_regime": "risk_on"}),
            encoding="utf-8",
        )


def test_phase6a_builds_regime_report_from_existing_logs(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = Phase6AMarketRegimeAudit(tmp_path).build_report()

    assert result["metadata"]["audit_only"] is True
    assert result["metadata"]["full_backtest_executed"] is False
    assert result["metadata"]["selected_count_in_day_used"] is False
    assert result["coverage"]["trades"] == 2
    regimes = {row["regime"]: row for row in result["regime_by_day"]}
    assert regimes["Bull"]["days"] > 0
    assert regimes["Bear"]["days"] > 0
    assert any(row["rule"] == "Rule D Bear new buys stopped" for row in result["virtual_strategy_audit"])
    assert result["drawdown_analysis"]["regime_at_dd"] in {"Bull", "Neutral", "Bear", "Unknown"}


def test_phase6a_saves_markdown_and_json(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = Phase6AMarketRegimeAudit(tmp_path)

    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 6-A" in paths.markdown.read_text(encoding="utf-8")
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["regime_definition"]["Bull"].startswith("TOPIX")
