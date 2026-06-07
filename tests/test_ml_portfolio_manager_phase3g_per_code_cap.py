from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from main import write_purchase_audit_csv
import paper_trade
from ml.portfolio_manager_phase3g import PortfolioManagerPhase3GReporter
from profile_loader import load_profile


PERIOD = "2023-01-01_to_2026-05-31"


def test_v2_77_profile_enables_per_code_cap_without_changing_v2_76() -> None:
    v76 = load_profile("rookie_dealer_02_v2_76")
    v77 = load_profile("rookie_dealer_02_v2_77")

    assert v76["profile_id"] == "rookie_dealer_02_v2_76_pm_ai_low_score_skip"
    assert not v76["portfolio_manager_ai_sizing"].get("per_code_exposure_cap_enabled", False)
    assert v77["profile_id"] == "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap"
    assert v77["portfolio_manager_ai_sizing"]["per_code_exposure_cap_enabled"] is True
    assert v77["portfolio_manager_ai_sizing"]["per_code_exposure_cap_rate"] == 0.20
    assert v77["portfolio_manager_ai_sizing"]["selected_count_in_day_forbidden"] is True


def test_per_code_cap_scales_buy_to_allowed_exposure() -> None:
    config = {
        "portfolio_manager_ai_sizing": {
            "per_code_exposure_cap_enabled": True,
            "per_code_exposure_cap_rate": 0.20,
        },
        "trading": {"use_round_lot": True, "round_lot_size": 100},
    }
    item = {"code": "62540"}

    shares, fields = paper_trade._apply_per_code_exposure_cap(
        item=item,
        shares=3000,
        entry_price=100.0,
        positions=[{"code": "62540", "market_value": 100_000.0}],
        total_assets=1_000_000.0,
        config=config,
    )

    assert shares == 1000
    assert fields["pm_per_code_cap_reduced"] is True
    assert fields["pm_per_code_cap_reason"] == "per_code_exposure_cap"
    assert fields["pm_per_code_cap_amount"] == 100_000.0
    assert item["pm_per_code_cap_rate"] == 0.20


def test_per_code_cap_skips_when_allowed_amount_is_below_round_lot() -> None:
    config = {
        "portfolio_manager_ai_sizing": {
            "per_code_exposure_cap_enabled": True,
            "per_code_exposure_cap_rate": 0.20,
        },
        "trading": {"use_round_lot": True, "round_lot_size": 100},
    }

    shares, fields = paper_trade._apply_per_code_exposure_cap(
        item={"code": "62540"},
        shares=100,
        entry_price=100.0,
        positions=[{"code": "62540", "market_value": 195_000.0}],
        total_assets=1_000_000.0,
        config=config,
    )

    assert shares == 0
    assert fields["pm_per_code_cap_skip"] is True
    assert fields["pm_per_code_cap_reason"] == "per_code_exposure_cap_scaled_below_round_lot"


def test_purchase_audit_csv_keeps_per_code_cap_columns(tmp_path: Path) -> None:
    path = tmp_path / "purchase_audit.csv"
    write_purchase_audit_csv(
        path,
        [
            {
                "trade_id": "t1",
                "profile_id": "p",
                "signal_date": "2026-01-01",
                "entry_date": "2026-01-02",
                "code": "62540",
                "decision": "SCALED_BUY",
                "pm_per_code_cap_enabled": True,
                "pm_per_code_cap_rate": 0.20,
                "pm_per_code_cap_reduced": True,
                "pm_per_code_cap_reason": "per_code_exposure_cap",
            }
        ],
    )

    df = pd.read_csv(path)

    assert "pm_per_code_cap_enabled" in df.columns
    assert "pm_per_code_cap_rate" in df.columns
    assert "pm_per_code_cap_reduced" in df.columns
    assert df.loc[0, "pm_per_code_cap_reason"] == "per_code_exposure_cap"


def _write_profile(root: Path, profile: str, *, net_profit: float, dd: float, cap_reductions: int = 0, cap_skips: int = 0) -> None:
    base = root / "logs" / "backtests" / profile / PERIOD
    base.mkdir(parents=True)
    summary = {
        "net_cumulative_profit": net_profit,
        "profit_factor": 1.5,
        "max_drawdown": dd,
        "win_rate": 0.5,
        "closed_trades_count": 2,
        "daily_asset_curve": [
            {"date": "2025-09-26", "total_assets": 1_100_000},
            {"date": "2025-09-29", "total_assets": 1_000_000},
            {"date": "2025-10-24", "total_assets": 1_120_000},
        ],
        "all_trades": [],
    }
    (base / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    trades = pd.DataFrame(
        [
            {
                "action": "SELL",
                "signal_date": "2025-09-26",
                "entry_date": "2025-09-26",
                "exit_date": "2025-09-29",
                "code": "62540",
                "entry_price": 100,
                "exit_price": 90,
                "shares": 1000,
                "net_profit": -10_000,
                "holding_days": 2,
                "final_amount": 100_000,
                "pm_multiplier": 1.0,
                "pm_score": 0.1,
            },
            {
                "action": "SELL",
                "signal_date": "2025-09-26",
                "entry_date": "2025-09-26",
                "exit_date": "2025-10-24",
                "code": "67400",
                "entry_price": 100,
                "exit_price": 120,
                "shares": 1000,
                "net_profit": net_profit + 10_000,
                "holding_days": 20,
                "final_amount": 100_000,
                "pm_multiplier": 1.3,
                "pm_score": 0.5,
            },
        ]
    )
    trades.to_csv(base / "trades.csv", index=False)
    audit_rows = [
        {"decision": "BUY", "skip_reason": "", "code": "62540", "signal_date": "2025-09-26", "entry_date": "2025-09-26", "final_amount": 100_000}
    ]
    audit_rows.extend(
        {"decision": "BUY", "skip_reason": "", "code": f"R{i}", "pm_per_code_cap_reduced": True, "pm_per_code_cap_reason": "per_code_exposure_cap"}
        for i in range(cap_reductions)
    )
    audit_rows.extend(
        {"decision": "SKIP", "skip_reason": "per_code_exposure_cap", "code": f"S{i}", "pm_per_code_cap_reduced": True, "pm_per_code_cap_reason": "per_code_exposure_cap"}
        for i in range(cap_skips)
    )
    pd.DataFrame(audit_rows).to_csv(base / "purchase_audit.csv", index=False)


def test_phase3g_report_generates_cap_sweep_outputs(tmp_path: Path) -> None:
    profiles = {
        "baseline": 100_000,
        "phase3d": 200_000,
        "phase3e": 300_000,
        "phase3g": 250_000,
        "cap015": 210_000,
        "cap025": 260_000,
        "cap030": 270_000,
    }
    for profile, profit in profiles.items():
        _write_profile(tmp_path, profile, net_profit=profit, dd=-0.1, cap_reductions=1 if profile.startswith("cap") or profile == "phase3g" else 0)

    reporter = PortfolioManagerPhase3GReporter(
        root=tmp_path,
        baseline_profile="baseline",
        phase3d_profile="phase3d",
        phase3e_profile="phase3e",
        phase3g_profile="phase3g",
        cap_rate_profiles={"0.15": "cap015", "0.20": "phase3g", "0.25": "cap025", "0.30": "cap030"},
    )
    result = reporter.build()
    paths = reporter.save(result)

    assert result["constraints"]["selected_count_in_day_used"] is False
    assert len(result["cap_rate_sweep"]) == 4
    assert result["cap_rate_sweep"][1]["per_code_exposure_cap_reduction_count"] == 1
    assert result["drawdown_focus_comparison"]
    assert paths.markdown.exists()
    assert paths.json.exists()
