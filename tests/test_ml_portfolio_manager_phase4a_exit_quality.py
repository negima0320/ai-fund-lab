from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase4a_exit_quality import PortfolioManagerPhase4AExitQualityAudit


PERIOD = "2023-01-01_to_2026-05-31"


def _write_price_days(root: Path) -> None:
    price_dir = root / "data" / "raw"
    price_dir.mkdir(parents=True)
    dates = pd.bdate_range("2026-01-02", periods=25)
    closes = [
        100,
        110,
        112,
        114,
        116,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
    ]
    loss_closes = [
        100,
        90,
        89,
        88,
        87,
        86,
        85,
        84,
        83,
        82,
        81,
        80,
        79,
        78,
        77,
        76,
        75,
        74,
        73,
        72,
        71,
        70,
        69,
        68,
        67,
    ]
    flat_closes = [100, 95, 95, 95, 96, 96, 95, 95, 95, 95, 96, 96, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95]
    for date, close, loss_close, flat_close in zip(dates, closes, loss_closes, flat_closes):
        payload = {
            "provider": "fixture",
            "date": date.strftime("%Y-%m-%d"),
            "prices": [
                {"code": "11110", "date": date.strftime("%Y-%m-%d"), "close": close},
                {"code": "22220", "date": date.strftime("%Y-%m-%d"), "close": loss_close},
                {"code": "33330", "date": date.strftime("%Y-%m-%d"), "close": flat_close},
            ],
        }
        (price_dir / f"prices_{date.strftime('%Y-%m-%d')}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_profile(root: Path, profile: str) -> None:
    base = root / "logs" / "backtests" / profile / PERIOD
    base.mkdir(parents=True)
    (base / "backtest_summary.json").write_text(
        json.dumps(
            {
                "net_cumulative_profit": 4000.0,
                "profit_factor": 1.2,
                "max_drawdown": -0.05,
                "win_rate": 0.5,
                "closed_trades_count": 3,
                "daily_asset_curve": [],
                "all_trades": [],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-05",
                "code": "11110",
                "entry_price": 100,
                "exit_price": 110,
                "shares": 100,
                "holding_days": 2,
                "net_profit": 1000,
                "net_profit_rate": 0.10,
                "pm_multiplier": 1.15,
            },
            {
                "action": "SELL",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-05",
                "code": "22220",
                "entry_price": 100,
                "exit_price": 90,
                "shares": 100,
                "holding_days": 2,
                "net_profit": -1000,
                "net_profit_rate": -0.10,
                "pm_multiplier": 0.8,
            },
            {
                "action": "SELL",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-05",
                "code": "33330",
                "entry_price": 100,
                "exit_price": 95,
                "shares": 100,
                "holding_days": 2,
                "net_profit": 500,
                "net_profit_rate": 0.05,
                "pm_multiplier": 1.0,
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
                "pm_multiplier": 1.15,
            },
            {
                "decision": "BUY",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "code": "22220",
                "final_amount": 10_000,
                "pm_multiplier": 0.8,
            },
            {
                "decision": "BUY",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "code": "33330",
                "final_amount": 10_000,
                "pm_multiplier": 1.0,
            },
        ]
    ).to_csv(base / "purchase_audit.csv", index=False)


def _make_audit(tmp_path: Path) -> PortfolioManagerPhase4AExitQualityAudit:
    _write_price_days(tmp_path)
    profiles = {
        "v2_75": "profile_v2_75",
        "v2_76": "profile_v2_76",
        "v2_77": "profile_v2_77",
    }
    for profile in profiles.values():
        _write_profile(tmp_path, profile)
    return PortfolioManagerPhase4AExitQualityAudit(
        root=tmp_path,
        v2_75_profile=profiles["v2_75"],
        v2_76_profile=profiles["v2_76"],
        v2_77_cap_030_profile=profiles["v2_77"],
    )


def test_phase4a_computes_post_exit_returns_and_hypothetical_profit(tmp_path: Path) -> None:
    audit = _make_audit(tmp_path)

    result = audit.build()

    trade = next(row for row in result["exit_trades"] if row["profile"] == "v2_75" and row["code"] == "11110")
    assert trade["post_exit_return_1d"] == (112 / 110) - 1
    assert trade["post_exit_return_3d"] == (116 / 110) - 1
    assert trade["post_exit_return_20d"] == (134 / 110) - 1
    assert trade["max_post_exit_return_20d"] == (134 / 110) - 1
    assert trade["hypothetical_profit_hold_3d"] == (116 - 100) * 100
    assert trade["actual_minus_hold_3d"] == -600
    assert trade["exit_quality_label"] == "early_exit"


def test_phase4a_builds_profile_pm_and_holding_summaries(tmp_path: Path) -> None:
    audit = _make_audit(tmp_path)

    result = audit.build()

    profile_summary = {row["profile"]: row for row in result["profile_summary"]}
    assert profile_summary["v2_75"]["trade_count"] == 3
    assert profile_summary["v2_75"]["early_exit_count"] == 1
    assert profile_summary["v2_75"]["loss_cut_success_count"] == 1
    assert profile_summary["v2_75"]["good_exit_count"] == 1
    assert any(row["pm_multiplier"] == 1.15 and row["trade_count"] == 1 for row in result["pm_multiplier_summary"])
    assert any(row["holding_days_group"] == "2-3d" and row["trade_count"] == 3 for row in result["holding_days_summary"])
    assert result["hypothetical_hold_summary"][0]["actual_realized_profit"] == 500.0


def test_phase4a_saves_json_and_markdown_without_forbidden_behaviour(tmp_path: Path) -> None:
    audit = _make_audit(tmp_path)
    result = audit.build()

    paths = audit.save(result)

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 4-A" in paths.markdown.read_text(encoding="utf-8")
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["constraints"]["selected_count_in_day_used"] is False
    assert loaded["constraints"]["api_refetch"] is False
    assert loaded["constraints"]["openai_api"] is False
    assert loaded["constraints"]["trading_logic_changed"] is False
    assert loaded["constraints"]["exit_ai_logic_changed"] is False
