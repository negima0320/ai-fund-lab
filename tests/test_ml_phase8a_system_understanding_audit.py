from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase8a_system_understanding_audit import Phase8ASystemUnderstandingAudit


PROFILE = "rookie_dealer_02_v2_82_cap38"
PERIOD = "2023-01-01_to_2026-05-31"


def _write_fixture(root: Path) -> None:
    core = root / "reports" / "final" / "v2_82_cap38" / "core_2023-01_to_2026-05"
    core.mkdir(parents=True, exist_ok=True)
    (root / "reports" / "final" / "v2_82_cap38").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "final" / "v2_82_cap38" / "final_summary.json").write_text(
        json.dumps(
            {
                "core_comparison": {
                    "v2_82_cap38": {
                        "net_profit": 3_777_545,
                        "profit_factor": 2.7309,
                        "max_drawdown": -0.0654,
                        "win_rate": 0.5511,
                        "cagr": 0.6674,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    dates = pd.bdate_range("2023-01-05", periods=280)
    pd.DataFrame(
        {
            "date": [day.strftime("%Y-%m-%d") for day in dates],
            "cash": [500_000] * len(dates),
            "positions_value": [500_000] * len(dates),
            "total_assets": [1_000_000 + idx * 1_000 for idx in range(len(dates))],
            "daily_profit": [1_000] * len(dates),
            "open_positions_count": [2] * len(dates),
        }
    ).to_csv(core / "summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "code": "11110",
                "entry_date": "2023-01-05",
                "exit_date": "2023-01-10",
                "net_profit": 20_000,
                "net_profit_rate": 0.05,
                "holding_days": 3,
                "exit_reason": "利確",
                "pm_multiplier": 1.30,
                "pm_score": 0.5,
                "expected_return_10d": 0.04,
                "bad_entry_probability_10d": 0.1,
                "risk_adjusted_score": 0.2,
                "score": 55,
                "volume_ratio": 2.5,
                "sector_name": "情報・通信業",
                "market_regime": "Bear",
            },
            {
                "action": "SELL",
                "code": "22220",
                "entry_date": "2023-02-01",
                "exit_date": "2023-02-06",
                "net_profit": -10_000,
                "net_profit_rate": -0.03,
                "holding_days": 4,
                "exit_reason": "損切り",
                "pm_multiplier": 0.80,
                "pm_score": -0.1,
                "expected_return_10d": 0.01,
                "bad_entry_probability_10d": 0.4,
                "risk_adjusted_score": -0.1,
                "score": 46,
                "volume_ratio": 1.5,
                "sector_name": "小売業",
                "market_regime": "Bull",
            },
        ]
    ).to_csv(core / "trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "entry_date": "2023-01-05",
                "code": "11110",
                "decision": "BUY",
                "candidate_source": "selected",
                "final_amount": 380_000,
                "pm_multiplier": 1.30,
                "pm_per_code_cap_reduced": True,
                "pm_per_code_cap_skip": False,
                "skip_reason": "",
                "risk_adjusted_score": 0.2,
                "expected_return_10d": 0.04,
                "bad_entry_probability_10d": 0.1,
                "market_regime": "Bear",
            },
            {
                "entry_date": "2023-02-01",
                "code": "22220",
                "decision": "SKIP",
                "candidate_source": "selected",
                "final_amount": 0,
                "pm_multiplier": 0.80,
                "pm_per_code_cap_reduced": False,
                "pm_per_code_cap_skip": False,
                "skip_reason": "selected_but_not_affordable",
                "risk_adjusted_score": -0.1,
                "expected_return_10d": 0.01,
                "bad_entry_probability_10d": 0.4,
                "market_regime": "Bull",
            },
        ]
    ).to_csv(core / "purchase_audit.csv", index=False)

    profile_dir = root / "config" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / f"{PROFILE}.yaml").write_text(
        """
profile_id: rookie_dealer_02_v2_82_cap38
trading:
  stop_loss_rate: -0.03
  take_profit_rate: 0.06
  max_holding_days: 5
risk_margin:
  max_daily_buy_amount: 900000
market_filter:
  risk_off_min_score: 50
ml_backtest:
  enabled: true
portfolio_manager_ai_sizing:
  enabled: true
  low_score_skip_enabled: true
  buy_ordering_mode: pm_aware
  fallback_to_next_affordable_selected: true
  fallback_min_pm_multiplier: 1.0
  per_code_exposure_cap_enabled: true
  per_code_exposure_cap_rate: 0.38
ml_exit_ai:
  enabled: true
capital_utilization_policy:
  enabled: true
purchase_audit:
  enabled: true
""",
        encoding="utf-8",
    )

    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "paper_trade.py").write_text("bear_pm_booster exit_ai_v2 minimum_hold earnings_filter", encoding="utf-8")
    (src / "scoring.py").write_text("earnings_filter", encoding="utf-8")
    (src / "profile_loader.py").write_text("v2_80", encoding="utf-8")
    (src / "main.py").write_text("", encoding="utf-8")


def test_phase8a_builds_system_understanding_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = Phase8ASystemUnderstandingAudit(tmp_path).build_report()

    assert result["metadata"]["audit_only"] is True
    assert result["metadata"]["logic_changed"] is False
    assert result["year_performance_table"][0]["year"] == 2023
    assert result["pm_ai_contribution"]["verdict"]["pm_ai_is_core_alpha"] is True
    assert result["stock_selection_ai_contribution"]["verdict"]["stock_ai_is_core_alpha"] is True
    assert result["bear_alpha"]["summary"]["bear_trades"] == 1
    assert any(row["logic"] == "Bear PM booster" for row in result["logic_inventory"])
    assert result["ai_alignment"]["ai_alignment_score"] in {"medium", "medium_high"}
    assert result["final_verdict"]["next_phase_recommended"].startswith("Phase 8-B")


def test_phase8a_saves_markdown_and_json(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = Phase8ASystemUnderstandingAudit(tmp_path)

    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 8-A" in paths.markdown.read_text(encoding="utf-8")
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["metadata"]["profile_added"] is False
    assert payload["removal_candidates"]

