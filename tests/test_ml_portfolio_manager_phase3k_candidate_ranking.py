from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.portfolio_manager_phase3d_detail_audit import PERIOD
from ml.portfolio_manager_phase3h_capital_utilization import PHASE3H_PROFILE
from ml.portfolio_manager_phase3k_candidate_ranking import PortfolioManagerPhase3KCandidateRankingAudit


def _write_backtest_fixture(root: Path) -> None:
    base = root / "logs" / "backtests" / PHASE3H_PROFILE / PERIOD
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
            }
        ]
    ).to_csv(base / "summary.csv", index=False)
    pd.DataFrame(columns=["action", "signal_date", "entry_date", "code", "net_profit"]).to_csv(
        base / "trades.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "signal_date": "2026-01-05",
                "entry_date": "2026-01-06",
                "code": "1111",
                "candidate_rank": 1,
                "score_rank": 1,
                "decision": "SKIP",
                "skip_reason": "selected_but_not_affordable",
                "reject_reason": "",
                "candidate_source": "selected",
                "planned_amount": 4000000,
                "pm_target_amount": 4000000,
                "risk_adjusted_score": 0.30,
                "expected_return_10d": 0.05,
                "pm_score": 0.45,
                "pm_multiplier": 1.3,
            },
            {
                "signal_date": "2026-01-05",
                "entry_date": "2026-01-06",
                "code": "2222",
                "candidate_rank": 2,
                "score_rank": 2,
                "decision": "BUY",
                "skip_reason": "",
                "reject_reason": "",
                "candidate_source": "selected",
                "planned_amount": 20000,
                "pm_target_amount": 20000,
                "risk_adjusted_score": 0.20,
                "expected_return_10d": 0.03,
                "pm_score": 0.25,
                "pm_multiplier": 1.15,
            },
            {
                "signal_date": "2026-01-06",
                "entry_date": "2026-01-07",
                "code": "3333",
                "candidate_rank": 1,
                "score_rank": 1,
                "decision": "SKIP",
                "skip_reason": "selected_but_not_affordable",
                "reject_reason": "",
                "candidate_source": "selected",
                "planned_amount": 3000000,
                "pm_target_amount": 3000000,
                "risk_adjusted_score": -0.05,
                "expected_return_10d": 0.01,
                "pm_score": -0.10,
                "pm_multiplier": 0.8,
            },
        ]
    ).to_csv(base / "purchase_audit.csv", index=False)


def _write_prices(root: Path) -> None:
    price_dir = root / "data" / "cache" / "jquants" / "prices"
    price_dir.mkdir(parents=True)
    prices_by_day = {
        "2026-01-05": {"1111": 100, "2222": 200, "3333": 300},
        "2026-01-06": {"1111": 101, "2222": 202, "3333": 303},
        "2026-01-07": {"1111": 102, "2222": 204, "3333": 306},
        "2026-01-08": {"1111": 103, "2222": 206, "3333": 309},
        "2026-01-09": {"1111": 104, "2222": 208, "3333": 312},
        "2026-01-12": {"1111": 105, "2222": 210, "3333": 315},
        "2026-01-13": {"1111": 106, "2222": 212, "3333": 318},
        "2026-01-14": {"1111": 107, "2222": 214, "3333": 321},
        "2026-01-15": {"1111": 108, "2222": 216, "3333": 324},
        "2026-01-16": {"1111": 109, "2222": 218, "3333": 327},
        "2026-01-19": {"1111": 110, "2222": 220, "3333": 330},
    }
    for date, prices in prices_by_day.items():
        records = [{"code": code, "close": close} for code, close in prices.items()]
        (price_dir / f"{date}.json").write_text(json.dumps({"prices": records}), encoding="utf-8")


def test_phase3k_candidate_ranking_audit_generates_reports_without_logic_changes(tmp_path: Path) -> None:
    _write_backtest_fixture(tmp_path)
    _write_prices(tmp_path)

    audit = PortfolioManagerPhase3KCandidateRankingAudit(root=tmp_path)
    result = audit.build()
    paths = audit.save(result)

    assert result["constraints"]["api_refetch"] is False
    assert result["constraints"]["openai_api"] is False
    assert result["constraints"]["selected_count_in_day_used"] is False
    assert result["constraints"]["trading_logic_changed"] is False
    assert result["ranking_basis"]["selected_sort_columns"] == "daily_score_rank ASC, risk_adjusted_score DESC, code ASC"
    classifications = {row["classification"] for row in result["candidate_path_classification"]}
    assert "top_candidate_unaffordable_but_next_candidate_bought" in classifications
    assert result["fallback_candidate_quality"]["candidate_count"] == 1
    flags = {row["flag"]: row["value"] for row in result["fallback_decision_flags"]}
    assert flags["ranking_log_insufficient"] is True
    assert result["rank_quality"]
    assert paths.markdown.exists()
    assert paths.json.exists()
