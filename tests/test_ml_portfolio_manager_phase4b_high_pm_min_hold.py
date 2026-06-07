from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase4b_high_pm_min_hold import PortfolioManagerPhase4BHighPMMinHoldAudit


PERIOD = "2023-01-01_to_2026-05-31"


def _write_price_days(root: Path) -> None:
    price_dir = root / "data" / "raw"
    price_dir.mkdir(parents=True)
    dates = pd.bdate_range("2026-01-02", periods=12)
    closes = [100, 101, 103, 110, 112, 115, 118, 120, 122, 124, 126, 128]
    for date, close in zip(dates, closes):
        payload = {
            "date": date.strftime("%Y-%m-%d"),
            "prices": [
                {"code": "11110", "date": date.strftime("%Y-%m-%d"), "close": close},
                {"code": "22220", "date": date.strftime("%Y-%m-%d"), "close": close},
                {"code": "33330", "date": date.strftime("%Y-%m-%d"), "close": close},
            ],
        }
        (price_dir / f"prices_{date.strftime('%Y-%m-%d')}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_profile(root: Path, profile: str) -> None:
    base = root / "logs" / "backtests" / profile / PERIOD
    base.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-05",
                "code": "11110",
                "entry_price": 100,
                "exit_price": 101,
                "shares": 100,
                "holding_days": 2,
                "net_profit": 100,
                "net_profit_rate": 0.01,
                "pm_multiplier": 1.3,
            },
            {
                "action": "SELL",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-09",
                "code": "22220",
                "entry_price": 100,
                "exit_price": 115,
                "shares": 100,
                "holding_days": 6,
                "net_profit": 1500,
                "net_profit_rate": 0.15,
                "pm_multiplier": 1.15,
            },
            {
                "action": "SELL",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-05",
                "code": "33330",
                "entry_price": 100,
                "exit_price": 101,
                "shares": 100,
                "holding_days": 2,
                "net_profit": 100,
                "net_profit_rate": 0.01,
                "pm_multiplier": 0.8,
            },
        ]
    ).to_csv(base / "trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "decision": "BUY",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "code": "11110",
                "final_amount": 10_000,
                "pm_multiplier": 1.3,
            },
            {
                "decision": "BUY",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "code": "22220",
                "final_amount": 10_000,
                "pm_multiplier": 1.15,
            },
        ]
    ).to_csv(base / "purchase_audit.csv", index=False)


def test_phase4b_extends_only_high_pm_short_holds(tmp_path: Path) -> None:
    profile = "profile_v2_78"
    _write_price_days(tmp_path)
    _write_profile(tmp_path, profile)
    audit = PortfolioManagerPhase4BHighPMMinHoldAudit(root=tmp_path, profile=profile)

    result = audit.build()

    min3 = next(row for row in result["minimum_hold_simulation"] if row["minimum_hold_days"] == 3)
    assert min3["eligible_high_pm_trades"] == 2
    assert min3["changed_trade_count"] == 1
    assert min3["actual_net_profit"] == 1600
    assert min3["virtual_net_profit"] == 2500
    assert min3["profit_delta"] == 900
    assert any(row["pm_multiplier"] == 1.3 and row["trade_count"] == 1 for row in result["pm_multiplier_exit_quality"])


def test_phase4b_saves_reports_and_forbidden_flags(tmp_path: Path) -> None:
    profile = "profile_v2_78"
    _write_price_days(tmp_path)
    _write_profile(tmp_path, profile)
    audit = PortfolioManagerPhase4BHighPMMinHoldAudit(root=tmp_path, profile=profile)

    paths = audit.save(audit.build())

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["constraints"]["selected_count_in_day_used"] is False
    assert loaded["constraints"]["api_refetch"] is False
    assert loaded["constraints"]["openai_api"] is False
    assert loaded["constraints"]["trading_logic_changed"] is False
    assert loaded["constraints"]["new_profile_added"] is False
    assert loaded["constraints"]["full_backtest_executed"] is False
