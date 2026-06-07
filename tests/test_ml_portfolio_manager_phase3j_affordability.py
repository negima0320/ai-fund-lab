from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase3d_detail_audit import PERIOD, PHASE3D_PROFILE
from ml.portfolio_manager_phase3e import PHASE3E_PROFILE
from ml.portfolio_manager_phase3g import PHASE3G_PROFILE
from ml.portfolio_manager_phase3h_capital_utilization import PHASE3H_PROFILE
from ml.portfolio_manager_phase3i_candidate_pool import POOL_X2_PROFILE, POOL_X3_PROFILE
from ml.portfolio_manager_phase3j_affordability import PortfolioManagerPhase3JAffordabilityAudit


def _write_profile(root: Path, profile: str, *, target: bool = False) -> None:
    base = root / "logs" / "backtests" / profile / PERIOD
    base.mkdir(parents=True, exist_ok=True)
    (base / "backtest_summary.json").write_text(
        json.dumps(
            {
                "net_cumulative_profit": 1000,
                "profit_factor": 2.0,
                "max_drawdown": -0.05,
                "win_rate": 0.5,
                "closed_trades_count": 1,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "cash": 50000,
                "positions_value": 950000,
                "total_assets": 1000000,
                "daily_profit": 0,
                "open_positions_count": 3,
            },
            {
                "date": "2026-01-06",
                "cash": 800000,
                "positions_value": 200000,
                "total_assets": 1000000,
                "daily_profit": 0,
                "open_positions_count": 1,
            },
        ]
    ).to_csv(base / "summary.csv", index=False)
    pd.DataFrame(columns=["action", "signal_date", "entry_date", "code", "net_profit"]).to_csv(
        base / "trades.csv", index=False
    )
    rows = [
        {
            "signal_date": "2026-01-05",
            "entry_date": "2026-01-06",
            "code": "1111",
            "candidate_rank": 1,
            "score_rank": 1,
            "decision": "SKIP",
            "skip_reason": "selected_but_not_affordable",
            "reject_reason": "",
            "scale_reason": "",
            "cash_before": 50000,
            "daily_buy_limit_remaining_before": 600000,
            "planned_shares": 1000,
            "planned_amount": 4000000,
            "pm_base_planned_shares": 1000,
            "pm_base_planned_amount": 4000000,
            "pm_target_amount": 4000000,
            "allocation_limit": 4000000,
            "pm_score": 0.45,
            "pm_multiplier": 1.3,
        },
        {
            "signal_date": "2026-01-05",
            "entry_date": "2026-01-06",
            "code": "2222",
            "candidate_rank": 2,
            "score_rank": 2,
            "decision": "SKIP",
            "skip_reason": "selected_but_not_affordable",
            "reject_reason": "",
            "scale_reason": "per_code_exposure_cap_scaled_below_round_lot",
            "cash_before": 500000,
            "daily_buy_limit_remaining_before": 500000,
            "planned_shares": 1000,
            "planned_amount": 300000,
            "pm_base_planned_shares": 1000,
            "pm_base_planned_amount": 300000,
            "pm_target_amount": 300000,
            "allocation_limit": 10000,
            "pm_score": -0.1,
            "pm_multiplier": 0.8,
        },
    ]
    if target:
        rows.append(
            {
                "signal_date": "2026-01-05",
                "entry_date": "2026-01-06",
                "code": "3333",
                "candidate_rank": 3,
                "score_rank": 3,
                "decision": "BUY",
                "skip_reason": "",
                "reject_reason": "",
                "scale_reason": "",
                "cash_before": 500000,
                "daily_buy_limit_remaining_before": 500000,
                "planned_shares": 100,
                "planned_amount": 20000,
                "pm_base_planned_shares": 100,
                "pm_base_planned_amount": 20000,
                "pm_target_amount": 20000,
                "allocation_limit": 20000,
                "pm_score": 0.25,
                "pm_multiplier": 1.15,
            }
        )
    pd.DataFrame(rows).to_csv(base / "purchase_audit.csv", index=False)


def _write_prices(root: Path) -> None:
    price_dir = root / "data" / "cache" / "jquants" / "prices"
    price_dir.mkdir(parents=True)
    prices_by_day = {
        "2026-01-05": {"1111": 4000, "2222": 300, "3333": 200},
        "2026-01-06": {"1111": 4100, "2222": 310, "3333": 202},
        "2026-01-07": {"1111": 4200, "2222": 320, "3333": 204},
        "2026-01-08": {"1111": 4300, "2222": 330, "3333": 206},
        "2026-01-09": {"1111": 4400, "2222": 340, "3333": 208},
        "2026-01-12": {"1111": 4500, "2222": 350, "3333": 210},
        "2026-01-13": {"1111": 4600, "2222": 360, "3333": 212},
        "2026-01-14": {"1111": 4700, "2222": 370, "3333": 214},
        "2026-01-15": {"1111": 4800, "2222": 380, "3333": 216},
        "2026-01-16": {"1111": 4900, "2222": 390, "3333": 218},
        "2026-01-19": {"1111": 5000, "2222": 400, "3333": 220},
    }
    for date, prices in prices_by_day.items():
        records = [{"code": code, "close": close} for code, close in prices.items()]
        (price_dir / f"{date}.json").write_text(json.dumps({"prices": records}), encoding="utf-8")


def test_phase3j_affordability_audit_builds_report_without_trading_changes(tmp_path: Path) -> None:
    profiles = [PHASE3D_PROFILE, PHASE3E_PROFILE, PHASE3G_PROFILE, PHASE3H_PROFILE, POOL_X2_PROFILE, POOL_X3_PROFILE]
    for profile in profiles:
        _write_profile(tmp_path, profile, target=profile == PHASE3H_PROFILE)
    _write_prices(tmp_path)

    audit = PortfolioManagerPhase3JAffordabilityAudit(root=tmp_path, target_profile=PHASE3H_PROFILE)
    result = audit.build()
    paths = audit.save(result)

    assert result["constraints"]["selected_count_in_day_used"] is False
    assert result["constraints"]["api_refetch"] is False
    assert result["constraints"]["trading_logic_changed"] is False
    assert result["selected_but_not_affordable_count"] == 2
    reasons = {row["dominant_blocking_reason"] for row in result["reason_summary"]}
    assert "cash_shortage" in reasons
    assert "per_code_cap_shortage" in reasons or "below_round_lot_after_per_code_cap" in reasons
    assert result["pm_score_band_summary"]
    assert result["profile_comparison"]
    assert result["fallback_possibility"]["days_with_affordable_alternative_candidate"] == 1
    assert result["top_missed_opportunities"][0]["hypothetical_return_5d"] is not None
    assert paths.markdown.exists()
    assert paths.json.exists()
