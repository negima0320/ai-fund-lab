from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.capital_allocation_phase5 import CapitalAllocationPhase5Comparison


def _write_profile(root: Path, profile: str, *, net_profit: float, with_audit: bool = False) -> None:
    period = "2023-01-01_to_2026-05-31"
    out = root / "logs" / "backtests" / profile / period
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest_summary.json").write_text(
        json.dumps(
            {
                "final_assets": 1_000_000 + net_profit,
                "net_cumulative_profit": net_profit,
                "win_rate": 0.5,
                "profit_factor": 1.3,
                "max_drawdown": -0.1,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "code": "67400",
                "entry_date": "2026-03-09",
                "exit_date": "2026-03-10",
                "shares": 100,
                "net_profit": net_profit,
                "net_profit_rate": 0.1,
                "holding_days": 2,
                "exit_reason": "利確",
                "scaled_buy_triggered": with_audit,
            }
        ]
    ).to_csv(out / "trades.csv", index=False)
    pd.DataFrame([{"date": "2026-03-10", "total_assets": 1_000_000 + net_profit, "positions_value": 100000}]).to_csv(out / "summary.csv", index=False)
    if with_audit:
        pd.DataFrame(
            [
                {"decision": "SCALED_BUY", "candidate_rank": 1, "skip_reason": "", "code": "67400"},
                {"decision": "BUY", "candidate_rank": 2, "skip_reason": "", "code": "70010"},
            ]
        ).to_csv(out / "purchase_audit.csv", index=False)


def test_capital_allocation_phase5_report_uses_v2_73_outputs(tmp_path: Path) -> None:
    profiles = ["v2_71", "v2_73"]
    _write_profile(tmp_path, "v2_71", net_profit=10_000)
    _write_profile(tmp_path, "v2_73", net_profit=20_000, with_audit=True)

    comparison = CapitalAllocationPhase5Comparison(root=tmp_path, profiles=profiles, focus_profile="v2_73")
    result = comparison.build()
    paths = comparison.save(result)

    focus = [row for row in result["summary"] if row["profile"] == "v2_73"][0]
    assert focus["purchase_audit_rows"] == 2
    assert focus["continue_candidate_buy_count"] == 1
    assert paths.markdown.name == "capital_allocation_phase5_v2_73_comparison_2023-01_to_2026-05.md"
    assert paths.json.exists()
    assert paths.purchase_audit_summary_md.exists()
