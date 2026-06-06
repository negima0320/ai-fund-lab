from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ml.daily_buy_limit_sensitivity import DailyBuyLimitSensitivity


def _write_base_profile(root: Path) -> None:
    profile_dir = root / "config" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_id": "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue",
        "profile_name": "v2_73",
        "scaled_buy": {"enabled": True},
        "purchase_audit": {"enabled": True},
        "risk_margin": {"max_daily_buy_amount": 900000},
    }
    (profile_dir / "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write_backtest(root: Path, profile: str, net_profit: float) -> None:
    out = root / "logs" / "backtests" / profile / "2023-01-01_to_2026-05-31"
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest_summary.json").write_text(
        json.dumps({"final_assets": 1_000_000 + net_profit, "net_cumulative_profit": net_profit, "profit_factor": 1.2, "win_rate": 0.5, "max_drawdown": -0.1}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "code": "67400",
                "entry_date": "2026-03-09",
                "exit_date": "2026-03-10",
                "net_profit": net_profit,
                "amount": 100000,
                "scaled_buy_triggered": True,
            }
        ]
    ).to_csv(out / "trades.csv", index=False)
    pd.DataFrame([{"date": "2026-03-10", "total_assets": 1_000_000 + net_profit, "positions_value": 100000}]).to_csv(out / "summary.csv", index=False)
    pd.DataFrame([{"decision": "SCALED_BUY", "candidate_rank": 1}]).to_csv(out / "purchase_audit.csv", index=False)


def test_daily_buy_limit_sensitivity_generates_profiles_and_report(tmp_path: Path) -> None:
    _write_base_profile(tmp_path)
    runner = DailyBuyLimitSensitivity(
        root=tmp_path,
        conditions=[
            {"condition": "fixed_900000", "mode": "fixed", "limit": 900000, "profile": "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue"},
            {"condition": "asset_ratio_050", "mode": "asset_ratio", "ratio": 0.5},
            {"condition": "unlimited", "mode": "unlimited"},
        ],
    )
    profiles = runner.ensure_profiles()
    generated = tmp_path / "config" / "profiles" / "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue_asset_ratio_050.yaml"
    assert generated.exists()
    generated_payload = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert generated_payload["scaled_buy"]["limit_mode"] == "asset_ratio"
    assert generated_payload["scaled_buy"]["daily_buy_limit_ratio"] == 0.5
    for index, profile in enumerate(profiles):
        _write_backtest(tmp_path, profile, 10_000 * (index + 1))

    result = runner.build(profiles)
    paths = runner.save(result)

    assert len(result["summary"]) == 3
    assert result["best_net_profit"]["condition"] == "unlimited"
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.summary_csv.exists()
