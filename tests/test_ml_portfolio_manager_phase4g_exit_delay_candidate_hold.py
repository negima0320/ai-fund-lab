from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase4g_exit_delay_candidate_hold import (
    PERIOD,
    PRIMARY,
    REFERENCE,
    PortfolioManagerPhase4GExitDelayCandidateHoldAudit,
)


def _profile_dir(root: Path, profile: str) -> Path:
    return root / "logs" / "backtests" / profile / PERIOD


def _write_logs(root: Path, profile: str, trades: list[dict], audit: list[dict]) -> None:
    base = _profile_dir(root, profile)
    base.mkdir(parents=True)
    (base / "backtest_summary.json").write_text("{}", encoding="utf-8")
    pd.DataFrame([]).to_csv(base / "summary.csv", index=False)
    pd.DataFrame(trades).to_csv(base / "trades.csv", index=False)
    pd.DataFrame(audit).to_csv(base / "purchase_audit.csv", index=False)


def _write_prices(root: Path) -> None:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    for date, close in [("2026-01-06", 1000), ("2026-01-07", 1050), ("2026-01-08", 1030)]:
        payload = {
            "provider": "fixture",
            "date": date,
            "prices": [
                {"code": "71570", "date": date, "open": close, "high": close, "low": close, "close": close},
                {"code": "99990", "date": date, "open": close, "high": close, "low": close, "close": close},
            ],
        }
        (raw / f"prices_{date}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_phase4g_exit_delay_and_candidate_rules(tmp_path: Path) -> None:
    _write_prices(tmp_path)
    _write_logs(
        tmp_path,
        PRIMARY,
        [
            {
                "action": "SELL",
                "code": "71570",
                "entry_date": "2026-01-05",
                "exit_date": "2026-01-06",
                "exit_reason": "Exit AI avoid_loss_5d",
                "exit_ai_triggered": True,
                "actual_exit_price": 1000,
                "exit_price": 1000,
                "shares": 100,
                "net_profit": 1000,
                "holding_days": 2,
                "pm_score": 0.2,
                "pm_multiplier": 1.15,
            }
        ],
        [
            {
                "trade_id": "candidate",
                "signal_date": "2026-01-06",
                "entry_date": "2026-01-07",
                "code": "71570",
                "decision": "SKIP",
                "candidate_rank": 1,
                "pm_score": 0.2,
                "pm_multiplier": 1.15,
            }
        ],
    )
    _write_logs(tmp_path, REFERENCE, [], [])

    result = PortfolioManagerPhase4GExitDelayCandidateHoldAudit(tmp_path).build_report()

    summary = result["exit_delay_1d_audit"]["summary"]
    assert summary["trade_count"] == 1
    assert summary["profit_delta"] == 5000
    assert result["candidate_presence_audit"]["rows"][0]["same_day_candidate_present"] is True
    rules = {row["rule"]: row for row in result["candidate_presence_hold_virtual_audit"]["rules"]}
    assert rules["A_same_day_pm_score_gte_0"]["held_trade_count"] == 1
    assert rules["D_trade_pm_multiplier_gte_1_15"]["profit_delta"] == 5000
    assert result["check_71570"]["virtual_hold_sell_date"] == "2026-01-07"
    assert result["clean_v280_candidate_judgement"]["clean_v280_worth_implementing"] is True


def test_phase4g_saves_report_without_running_backtest(tmp_path: Path) -> None:
    _write_prices(tmp_path)
    _write_logs(tmp_path, PRIMARY, [], [])
    _write_logs(tmp_path, REFERENCE, [], [])

    audit = PortfolioManagerPhase4GExitDelayCandidateHoldAudit(tmp_path)
    result = audit.build_report()
    paths = audit.save_report(result)

    assert result["metadata"]["audit_only"] is True
    assert result["metadata"]["full_backtest_executed"] is False
    assert paths.markdown.exists()
    assert paths.json.exists()

