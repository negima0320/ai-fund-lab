from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase8g_pm_ai_ranking_audit import Phase8GPMAIRankingAudit


def _purchase_rows(run_pm130: bool = True, weak_pm130: bool = False) -> list[dict[str, object]]:
    if not run_pm130:
        multipliers = [1.0, 1.15, 0.8]
    elif weak_pm130:
        multipliers = [1.3, 1.0, 1.15]
    else:
        multipliers = [1.3, 0.8, 1.0]
    risk_scores = [0.90, 0.20, 0.55] if not weak_pm130 else [0.10, 0.80, 0.70]
    expected_returns = [0.10, -0.01, 0.04] if not weak_pm130 else [0.00, 0.09, 0.06]
    bad_entries = [0.02, 0.50, 0.20] if not weak_pm130 else [0.70, 0.03, 0.10]
    codes = ["11110", "22220", "33330"]
    rows: list[dict[str, object]] = []
    for idx, code in enumerate(codes):
        rows.append(
            {
                "entry_date": "2023-01-05",
                "signal_date": "2023-01-04",
                "code": code,
                "candidate_source": "selected",
                "decision": "BUY",
                "candidate_rank": idx + 1,
                "score_rank": idx + 1,
                "risk_adjusted_score": risk_scores[idx],
                "expected_return_10d": expected_returns[idx],
                "bad_entry_probability_10d": bad_entries[idx],
                "planned_amount": 100000 + idx * 10000,
                "final_amount": 90000 + idx * 10000,
                "pm_multiplier": multipliers[idx],
            }
        )
    rows.append(
        {
            "entry_date": "2023-01-06",
            "signal_date": "2023-01-05",
            "code": "44440",
            "candidate_source": "fallback",
            "decision": "SKIP",
            "candidate_rank": 1,
            "score_rank": 1,
            "risk_adjusted_score": 0.30,
            "expected_return_10d": 0.02,
            "bad_entry_probability_10d": 0.40,
            "planned_amount": 80000,
            "final_amount": 0,
            "pm_multiplier": 1.0,
        }
    )
    return rows


def _trade_rows(multiplier: float, profit: float) -> list[dict[str, object]]:
    return [
        {
            "action": "SELL",
            "entry_date": "2023-01-05",
            "exit_date": "2023-01-10",
            "code": "11110",
            "net_profit": profit,
            "profit": profit,
            "pm_multiplier": multiplier,
        }
    ]


def _write_run(root: Path, run_path: Path, purchase_rows: list[dict[str, object]], trade_rows: list[dict[str, object]]) -> None:
    purchase_path = root / run_path / "purchase_audit.csv"
    trades_path = root / run_path / "trades.csv"
    purchase_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(purchase_rows).to_csv(purchase_path, index=False)
    pd.DataFrame(trade_rows).to_csv(trades_path, index=False)


def _write_fixture(root: Path) -> None:
    _write_run(
        root,
        Path("reports/final/v2_82_cap38/core_2023-01_to_2026-05"),
        _purchase_rows(run_pm130=True),
        _trade_rows(1.3, 1000),
    )
    _write_run(
        root,
        Path("logs/backtests/rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38/2023-01-01_to_2026-05-31"),
        _purchase_rows(run_pm130=False),
        _trade_rows(1.0, 300),
    )
    _write_run(
        root,
        Path("logs/backtests/rookie_dealer_02_v2_91_pm_ai_v2_calibrated_rule_e_cap38/2023-01-01_to_2026-05-31"),
        _purchase_rows(run_pm130=True, weak_pm130=True),
        _trade_rows(1.3, 100),
    )
    for model_dir in [
        root / "models/ml/portfolio_manager/current_v2_73_phase3b_clean",
        root / "models/ml/portfolio_manager/candidate_v2_api_only",
    ]:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model_metadata.json").write_text(json.dumps({"name": model_dir.name}), encoding="utf-8")


def test_phase8g_builds_ranking_audit(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    audit = Phase8GPMAIRankingAudit(tmp_path)
    report = audit.build_report()

    assert report["metadata"]["audit_only"] is True
    assert report["problem_definition"]["pm_problem_type_recommended"] == "candidate_ranking_and_relative_capital_allocation"
    overview = {row["run"]: row for row in report["daily_candidate_summary"]["overview"]}
    assert overview["v2_82_current_pm_cap38"]["days"] == 2
    assert overview["v2_82_current_pm_cap38"]["avg_candidate_count"] == 2.0

    pm130_scores = {row["score_column"]: row for row in report["current_pm130_relative_rank_audit"]["score_summaries"]}
    assert pm130_scores["risk_adjusted_score"]["available"] is True
    assert pm130_scores["risk_adjusted_score"]["target_count"] == 1
    assert pm130_scores["risk_adjusted_score"]["rank_p50"] == 1.0
    assert pm130_scores["risk_adjusted_score"]["relative_top_candidate_rate"] == 1.0
    assert "expected_max_return_20d" in report["current_pm130_relative_rank_audit"]["missing_score_columns"]

    api_only = report["relative_feature_api_only_assessment"]
    assert api_only["relative_features_api_only_feasible"] is True
    assert "rank_in_day" in api_only["allowed_relative_features"]
    assert "selected_count_in_day" in api_only["forbidden_relative_features"]
    assert report["verdict"]["relative_allocation_worth_testing"] is True


def test_phase8g_saves_markdown_and_json(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = Phase8GPMAIRankingAudit(tmp_path)
    report = audit.build_report()
    paths = audit.save_report(report)

    assert paths.markdown.exists()
    assert paths.json.exists()
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["phase"] == "8-G"
    assert loaded["verdict"]["best_next_approach"] == "Phase 8-H Rule-Based Relative Allocator Backtest"
