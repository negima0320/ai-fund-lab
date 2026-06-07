from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    script = Path("scripts/ml/report_portfolio_manager_phase4d_v278_vs_v279_diff_audit.py")
    spec = importlib.util.spec_from_file_location("phase4d_report", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile_dir(root: Path, profile: str) -> Path:
    return root / "logs" / "backtests" / profile / "2023-01-01_to_2026-05-31"


def _write_logs(root: Path, profile: str, summary: dict, trades: list[dict], audit: list[dict]) -> None:
    base = _profile_dir(root, profile)
    base.mkdir(parents=True)
    (base / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(trades).to_csv(base / "trades.csv", index=False)
    pd.DataFrame(audit).to_csv(base / "purchase_audit.csv", index=False)


def test_phase4d_report_handles_missing_logs(tmp_path: Path) -> None:
    module = _load_module()

    result = module.build_report(tmp_path)
    markdown, json_path = module.save_report(result, tmp_path)

    assert result["constraints"]["audit_only"] is True
    assert result["minimum_hold_effect"]["high_pm_min_hold_blocked_exit_count"] == 0
    assert markdown.exists()
    assert json_path.exists()


def test_phase4d_report_compares_buy_sell_and_contribution(tmp_path: Path) -> None:
    module = _load_module()
    common_trade_78 = {
        "action": "SELL",
        "entry_date": "2026-01-05",
        "exit_date": "2026-01-09",
        "code": "11110",
        "holding_days": 5,
        "entry_price": 1000,
        "shares": 100,
        "net_profit": 1000,
        "exit_reason": "最大保有期間到達",
        "pm_multiplier": 1.15,
        "high_pm_min_hold_blocked_exit_count": 0,
    }
    common_trade_79 = {
        **common_trade_78,
        "exit_date": "2026-01-10",
        "holding_days": 6,
        "net_profit": 1500,
    }
    only_79_trade = {
        "action": "SELL",
        "entry_date": "2026-01-06",
        "exit_date": "2026-01-08",
        "code": "22220",
        "holding_days": 3,
        "entry_price": 500,
        "shares": 100,
        "net_profit": 700,
        "exit_reason": "利確",
        "pm_multiplier": 1.0,
        "high_pm_min_hold_blocked_exit_count": 0,
    }
    _write_logs(
        tmp_path,
        module.V278,
        {
            "net_cumulative_profit": 1000,
            "profit_factor": 2.0,
            "max_drawdown": -0.1,
            "win_rate": 0.5,
            "closed_trades_count": 1,
        },
        [common_trade_78],
        [
            {
                "trade_id": "a",
                "entry_date": "2026-01-05",
                "code": "11110",
                "decision": "BUY",
                "final_shares": 100,
                "final_amount": 100000,
                "candidate_rank": 1,
                "pm_score": 0.3,
                "pm_multiplier": 1.15,
            },
            {
                "trade_id": "b",
                "entry_date": "2026-01-06",
                "code": "22220",
                "decision": "SKIP",
                "final_shares": 0,
                "final_amount": 0,
                "skip_reason": "selected_but_not_affordable",
            },
        ],
    )
    _write_logs(
        tmp_path,
        module.V279,
        {
            "net_cumulative_profit": 2200,
            "profit_factor": 2.5,
            "max_drawdown": -0.08,
            "win_rate": 0.6,
            "closed_trades_count": 2,
        },
        [common_trade_79, only_79_trade],
        [
            {
                "trade_id": "c",
                "entry_date": "2026-01-05",
                "code": "11110",
                "decision": "BUY",
                "final_shares": 100,
                "final_amount": 100000,
                "candidate_rank": 1,
                "pm_score": 0.3,
                "pm_multiplier": 1.15,
            },
            {
                "trade_id": "d",
                "entry_date": "2026-01-06",
                "code": "22220",
                "decision": "BUY",
                "final_shares": 100,
                "final_amount": 50000,
                "candidate_rank": 2,
                "pm_score": 0.1,
                "pm_multiplier": 1.0,
            },
        ],
    )

    result = module.build_report(tmp_path)

    assert result["buy_summary"] == {
        "only_v2_78_buy_count": 0,
        "only_v2_79_buy_count": 1,
        "common_buy_count": 1,
    }
    assert result["sell_summary"]["sell_date_changed_count"] == 1
    assert result["minimum_hold_effect"]["minimum_hold_activated"] is False
    assert result["profit_contribution"]["buckets"]["sell_timing_change"] == 500
    assert result["profit_contribution"]["buckets"]["affordability_change"] == 700
