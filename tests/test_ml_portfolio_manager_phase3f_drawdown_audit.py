from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase3f_drawdown import PortfolioManagerPhase3FDrawdownAudit


PERIOD = "2023-01-01_to_2026-05-31"


def _write_profile(root: Path, profile: str, *, curve: list[float], losses: list[tuple[str, float, float, float]]) -> None:
    base = root / "logs" / "backtests" / profile / PERIOD
    base.mkdir(parents=True)
    dates = pd.date_range("2023-01-02", periods=len(curve), freq="B")
    summary = {
        "net_cumulative_profit": curve[-1] - 1_000_000,
        "profit_factor": 1.5,
        "max_drawdown": min(value / max(curve[: idx + 1]) - 1 for idx, value in enumerate(curve)),
        "win_rate": 0.5,
        "closed_trades_count": len(losses),
        "daily_asset_curve": [
            {
                "date": date.strftime("%Y-%m-%d"),
                "day": idx + 1,
                "total_assets": value,
                "cumulative_profit": value - 1_000_000,
                "max_drawdown": min(curve[j] / max(curve[: j + 1]) - 1 for j in range(idx + 1)),
            }
            for idx, (date, value) in enumerate(zip(dates, curve))
        ],
        "all_trades": [],
    }
    (base / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = []
    audit_rows = []
    for idx, (code, profit, multiplier, score) in enumerate(losses):
        entry = dates[max(0, idx)].strftime("%Y-%m-%d")
        exit_date = dates[min(len(dates) - 1, idx + 2)].strftime("%Y-%m-%d")
        rows.append(
            {
                "action": "SELL",
                "code": code,
                "signal_date": entry,
                "entry_date": entry,
                "exit_date": exit_date,
                "entry_price": 100,
                "exit_price": 95 if profit < 0 else 110,
                "shares": 100,
                "net_profit": profit,
                "gross_profit": profit,
                "holding_days": 3,
                "final_amount": 10_000,
                "pm_multiplier": multiplier,
                "pm_score": score,
                "pm_high_conviction_proba": 0.7,
                "pm_avoid_proba": 0.2,
                "pm_model_version": "test",
                "pm_feature_count": 68,
            }
        )
        audit_rows.append(
            {
                "decision": "BUY",
                "code": code,
                "signal_date": entry,
                "entry_date": entry,
                "final_amount": 10_000,
                "pm_multiplier": multiplier,
                "pm_score": score,
                "pm_high_conviction_proba": 0.7,
                "pm_avoid_proba": 0.2,
                "pm_model_version": "test",
                "pm_feature_count": 68,
                "skip_reason": "",
            }
        )
    pd.DataFrame(rows).to_csv(base / "trades.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(base / "purchase_audit.csv", index=False)


def test_drawdown_audit_extracts_windows_and_contributions(tmp_path: Path) -> None:
    _write_profile(
        tmp_path,
        "baseline",
        curve=[1_000_000, 1_050_000, 1_020_000, 1_080_000],
        losses=[("11110", 10_000, 1.0, 0.1)],
    )
    _write_profile(
        tmp_path,
        "phase3d",
        curve=[1_000_000, 1_200_000, 1_100_000, 1_250_000],
        losses=[("22220", -20_000, 1.3, 0.5), ("33330", 40_000, 1.0, 0.1)],
    )
    _write_profile(
        tmp_path,
        "phase3e",
        curve=[1_000_000, 1_300_000, 1_000_000, 1_150_000],
        losses=[("44440", -80_000, 1.3, 0.5), ("55550", -10_000, 0.8, -0.1), ("66660", 20_000, 1.0, 0.1)],
    )

    audit = PortfolioManagerPhase3FDrawdownAudit(
        root=tmp_path,
        baseline_profile="baseline",
        phase3d_profile="phase3d",
        phase3e_profile="phase3e",
    )
    result = audit.build()

    assert result["constraints"]["api_refetch"] is False
    assert result["constraints"]["selected_count_in_day_used"] is False
    assert result["drawdown_windows"]["phase3e"]["start_date"] == "2023-01-03"
    assert result["drawdown_windows"]["phase3e"]["trough_date"] == "2023-01-04"
    assert result["same_period_comparison"]["phase3e_trade_count"] >= 1
    assert result["phase3e_drawdown_code_contribution"]
    assert result["phase3e_drawdown_pm_multiplier_contribution"]
    assert result["phase3e_drawdown_pm_score_band_contribution"]


def test_drawdown_audit_saves_markdown_and_json(tmp_path: Path) -> None:
    for profile in ["baseline", "phase3d", "phase3e"]:
        _write_profile(
            tmp_path,
            profile,
            curve=[1_000_000, 1_100_000, 900_000, 1_050_000],
            losses=[("11110", -20_000, 1.3, 0.5), ("22220", 30_000, 1.0, 0.1)],
        )
    audit = PortfolioManagerPhase3FDrawdownAudit(
        root=tmp_path,
        baseline_profile="baseline",
        phase3d_profile="phase3d",
        phase3e_profile="phase3e",
    )
    result = audit.build()
    paths = audit.save(result)

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Drawdown Root Cause" in paths.markdown.read_text(encoding="utf-8")
