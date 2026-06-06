from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ml.capital_allocation_phase8 import CapitalAllocationPhase8FallbackFilter
from ml.capital_allocation_phase8 import PRIORITY_CONDITIONS
from ml.capital_allocation_phase7 import V2_73_PROFILE, V2_74_PROFILE


def _write_profile(root: Path) -> None:
    profile_dir = root / "config" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_id": V2_74_PROFILE,
        "profile_name": "v2_74",
        "affordable_fallback_buy": {
            "enabled": True,
            "surplus_after_selection": True,
            "ranking": "risk_adjusted_score",
            "max_fallback_buys_per_day": 3,
        },
        "purchase_audit": {"enabled": True},
    }
    (profile_dir / f"{V2_74_PROFILE}.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write_backtest(root: Path, profile: str, *, profit: float, fallback_profit: float = 0.0) -> None:
    period = "2023-01-01_to_2026-05-31"
    out = root / "logs" / "backtests" / profile / period
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "final_assets": 1_000_000 + profit,
        "net_cumulative_profit": profit,
        "win_rate": 0.5,
        "profit_factor": 1.2 if profit >= 0 else 0.8,
        "max_drawdown": -0.1,
        "total_trades": 2,
    }
    (out / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    trades = pd.DataFrame(
        [
            {
                "action": "SELL",
                "code": "67400",
                "exit_date": "2023-02-01",
                "net_profit": profit - fallback_profit,
                "candidate_source": "selected",
                "holding_days": 10,
                "amount": 100000,
            },
            {
                "action": "SELL",
                "code": "90010",
                "exit_date": "2023-02-02",
                "net_profit": fallback_profit,
                "candidate_source": "fallback",
                "holding_days": 9,
                "amount": 50000,
            },
        ]
    )
    trades.to_csv(out / "trades.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2023-01-31", "total_assets": 1_000_000 + profit, "portfolio_value": 300_000, "open_positions_count": 2},
        ]
    ).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(
        [
            {"candidate_source": "fallback", "decision": "BUY", "final_amount": 50000},
            {"candidate_source": "selected", "decision": "BUY", "final_amount": 100000},
        ]
    ).to_csv(out / "purchase_audit.csv", index=False)


def test_phase8_generates_filtered_profiles(tmp_path: Path) -> None:
    _write_profile(tmp_path)
    runner = CapitalAllocationPhase8FallbackFilter(root=tmp_path, conditions=PRIORITY_CONDITIONS[:2])

    profiles = runner.ensure_profiles()

    assert profiles[:2] == [V2_73_PROFILE, V2_74_PROFILE]
    generated = tmp_path / "config" / "profiles" / f"{runner.profile_id_for(PRIORITY_CONDITIONS[0])}.yaml"
    payload = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert payload["affordable_fallback_buy"]["min_risk_adjusted_score"] == 0.05
    generated = tmp_path / "config" / "profiles" / f"{runner.profile_id_for(PRIORITY_CONDITIONS[1])}.yaml"
    payload = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert payload["affordable_fallback_buy"]["min_expected_return_10d"] == 0.02
    assert payload["affordable_fallback_buy"]["max_bad_entry_probability_10d"] == 0.70


def test_phase8_builds_and_saves_summary(tmp_path: Path) -> None:
    _write_profile(tmp_path)
    runner = CapitalAllocationPhase8FallbackFilter(root=tmp_path, conditions=PRIORITY_CONDITIONS[:1])
    profiles = runner.ensure_profiles()
    for index, profile in enumerate(profiles):
        _write_backtest(tmp_path, profile, profit=1000 + index * 100, fallback_profit=100 + index)

    result = runner.build()
    paths = runner.save(result)

    assert len(result["summary"]) == 3
    assert result["best_net_profit"]["net_profit"] == 1200
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.summary_csv.exists()
