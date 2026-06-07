from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase4f_side_effect_audit import PERIOD, V278, V279, PortfolioManagerPhase4FSideEffectAudit


def _profile_dir(root: Path, profile: str) -> Path:
    return root / "logs" / "backtests" / profile / PERIOD


def _write_logs(root: Path, profile: str, summary: list[dict], trades: list[dict], audit: list[dict]) -> None:
    base = _profile_dir(root, profile)
    base.mkdir(parents=True)
    (base / "backtest_summary.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(summary).to_csv(base / "summary.csv", index=False)
    pd.DataFrame(trades).to_csv(base / "trades.csv", index=False)
    pd.DataFrame(audit).to_csv(base / "purchase_audit.csv", index=False)


def test_phase4f_detects_buy_sell_divergence_and_min_hold_status(tmp_path: Path) -> None:
    _write_logs(
        tmp_path,
        V278,
        [
            {"date": "2026-01-05", "cash": 800000, "open_positions_count": 1},
            {"date": "2026-01-06", "cash": 810000, "open_positions_count": 1},
        ],
        [
            {
                "action": "SELL",
                "entry_date": "2026-01-05",
                "exit_date": "2026-01-08",
                "code": "11110",
                "holding_days": 4,
                "shares": 100,
                "net_profit": 1000,
                "exit_reason": "最大保有期間到達",
                "pm_multiplier": 1.15,
                "exit_ai_triggered": False,
            }
        ],
        [
            {
                "trade_id": "a",
                "entry_date": "2026-01-05",
                "code": "11110",
                "decision": "BUY",
                "final_shares": 100,
                "final_amount": 100000,
                "cash_before": 900000,
                "daily_buy_limit_remaining_before": 900000,
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
        V279,
        [
            {"date": "2026-01-05", "cash": 800000, "open_positions_count": 1},
            {"date": "2026-01-06", "cash": 700000, "open_positions_count": 2},
        ],
        [
            {
                "action": "SELL",
                "entry_date": "2026-01-05",
                "exit_date": "2026-01-09",
                "code": "11110",
                "holding_days": 5,
                "shares": 100,
                "net_profit": 1500,
                "exit_reason": "最大保有期間到達",
                "pm_multiplier": 1.15,
                "exit_ai_triggered": False,
                "high_pm_min_hold_applied": True,
                "high_pm_min_hold_blocked_exit_count": 0,
            },
            {
                "action": "SELL",
                "entry_date": "2026-01-06",
                "exit_date": "2026-01-10",
                "code": "22220",
                "holding_days": 4,
                "shares": 100,
                "net_profit": 700,
                "exit_reason": "利確",
                "pm_multiplier": 1.0,
                "exit_ai_triggered": False,
                "high_pm_min_hold_blocked_exit_count": 0,
            },
        ],
        [
            {
                "trade_id": "c",
                "entry_date": "2026-01-05",
                "code": "11110",
                "decision": "BUY",
                "final_shares": 100,
                "final_amount": 100000,
                "cash_before": 900000,
                "daily_buy_limit_remaining_before": 900000,
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
                "cash_before": 800000,
                "daily_buy_limit_remaining_before": 800000,
                "candidate_rank": 2,
                "pm_score": 0.1,
                "pm_multiplier": 1.0,
            },
        ],
    )

    result = PortfolioManagerPhase4FSideEffectAudit(tmp_path).build_report()

    assert result["metadata"]["audit_only"] is True
    assert result["daily_path_divergence"]["first_divergence_date"] == "2026-01-06"
    assert result["daily_path_divergence"]["divergence_type"] == "buy_difference"
    buy_summary = {row["metric"]: row["value"] for row in result["buy_root_cause"]["summary"]}
    assert buy_summary["only_v2_79_buy_count"] == 1
    assert result["buy_root_cause"]["rows"][0]["likely_cause"] == "cash_path_divergence"
    assert result["minimum_hold_confirmation"]["blocked_exit_count"] == 0
    assert result["minimum_hold_confirmation"]["blocked_exit_count_consistent"] is True
    assert result["side_effect_judgement"]["minimum_hold_directly_effective"] is False


def test_phase4f_saves_markdown_and_json(tmp_path: Path) -> None:
    _write_logs(tmp_path, V278, [], [], [])
    _write_logs(tmp_path, V279, [], [], [])

    audit = PortfolioManagerPhase4FSideEffectAudit(tmp_path)
    result = audit.build_report()
    paths = audit.save_report(result)

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 4-F" in paths.markdown.read_text(encoding="utf-8")

