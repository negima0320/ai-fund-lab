from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ml.portfolio_manager_sizing import PortfolioManagerV3SizingAdvisor
from ml.portfolio_manager_v3_backtest_candidate_audit_pm_sizing_universe import (
    BASELINE_PROFILE,
    CANDIDATE_PROFILES,
    PERIOD,
    PMAIV3BacktestCandidateAuditPMSizingUniverse,
)


FEATURES = ["rank_in_day", "percentile_in_day", "risk_adjusted_score"]


def _write_v3_fixture(root: Path) -> None:
    data_dir = root / "data/ml/portfolio_manager_v3"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for day in pd.bdate_range("2026-01-05", periods=4):
        for rank in range(1, 31):
            pct = 1.0 - (rank - 1) / 29.0
            rows.append(
                {
                    "prediction_date": day.strftime("%Y-%m-%d"),
                    "code": f"{10000 + rank}",
                    "rank_in_day": float(rank),
                    "percentile_in_day": pct,
                    "risk_adjusted_score": pct,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_parquet(data_dir / "portfolio_manager_v3_dataset_pm_sizing_universe_2023-01_to_2026-05.parquet", index=False)

    model_dir = root / "models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe"
    model_dir.mkdir(parents=True, exist_ok=True)
    x = frame[FEATURES]
    joblib.dump(HistGradientBoostingRegressor(max_iter=8, random_state=1).fit(x, frame["percentile_in_day"]), model_dir / "model_a_candidate_ranking_regressor.joblib")
    joblib.dump(HistGradientBoostingRegressor(max_iter=8, random_state=2).fit(x, frame["percentile_in_day"]), model_dir / "model_b_downside_utility_regressor.joblib")
    joblib.dump(HistGradientBoostingClassifier(max_iter=8, random_state=3).fit(x, (frame["rank_in_day"] <= 3).astype(int)), model_dir / "model_c_top_utility_classifier.joblib")
    (model_dir / "feature_columns.json").write_text(json.dumps(FEATURES), encoding="utf-8")


def _write_backtest_fixture(root: Path) -> None:
    labels = {"v2_82_cap38": BASELINE_PROFILE, **CANDIDATE_PROFILES}
    for label, profile in labels.items():
        base = root / "logs/backtests" / profile / PERIOD
        base.mkdir(parents=True, exist_ok=True)
        is_base = label == "v2_82_cap38"
        summary = {
            "initial_capital": 1_000_000,
            "final_assets": 1_300_000 if is_base else 1_220_000,
            "net_cumulative_profit": 300_000 if is_base else 220_000,
            "profit_factor": 2.0 if is_base else 1.5,
            "max_drawdown": -0.05 if is_base else -0.08,
            "win_rate": 0.55 if is_base else 0.50,
            "closed_trades_count": 4,
        }
        (base / "backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        pd.DataFrame(
            [
                {"date": "2026-01-05", "positions_value": 600_000, "total_assets": 1_000_000},
                {"date": "2026-01-06", "positions_value": 700_000, "total_assets": 1_100_000},
            ]
        ).to_csv(base / "summary.csv", index=False)
        pd.DataFrame(
            [
                {"action": "SELL", "signal_date": "2026-01-05", "exit_date": "2026-01-20", "code": "10001", "pm_multiplier": 1.3, "net_profit": 100_000 if is_base else 60_000, "holding_days": 5},
                {"action": "SELL", "signal_date": "2026-01-05", "exit_date": "2026-01-21", "code": "10002", "pm_multiplier": 1.15, "net_profit": 40_000, "holding_days": 4},
                {"action": "SELL", "signal_date": "2026-01-06", "exit_date": "2026-02-20", "code": "10003", "pm_multiplier": 1.0, "net_profit": -20_000, "holding_days": 3},
                {"action": "SELL", "signal_date": "2026-01-06", "exit_date": "2026-02-21", "code": "10004", "pm_multiplier": 0.8, "net_profit": -10_000, "holding_days": 2},
            ]
        ).to_csv(base / "trades.csv", index=False)
        pd.DataFrame(
            [
                {"decision": "BUY", "signal_date": "2026-01-05", "code": "10001", "pm_multiplier": 1.3, "pm_feature_found": True},
                {"decision": "BUY", "signal_date": "2026-01-05", "code": "10002", "pm_multiplier": 1.15, "pm_feature_found": True},
                {"decision": "BUY", "signal_date": "2026-01-06", "code": "10003", "pm_multiplier": 1.0, "pm_feature_found": True},
                {"decision": "BUY", "signal_date": "2026-01-06", "code": "10004", "pm_multiplier": 0.8, "pm_feature_found": True},
            ]
        ).to_csv(base / "purchase_audit.csv", index=False)


def test_pm_v3_runtime_lookup_supports_f2_mappings_and_missing_fallback(tmp_path: Path) -> None:
    _write_v3_fixture(tmp_path)
    advisor = PortfolioManagerV3SizingAdvisor(
        root=tmp_path,
        model_dir="models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe",
        dataset_path="data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_pm_sizing_universe_2023-01_to_2026-05.parquet",
        expected_feature_count=len(FEATURES),
        mapping_name="e_139_classifier_gate_recommended",
    )

    found = advisor.decision_for("2026-01-05", "10001")
    missing = advisor.decision_for("2026-01-05", "99999")

    assert found.feature_found is True
    assert found.multiplier in {0.6, 0.8, 1.0, 1.15, 1.3}
    assert found.model_version.startswith("pm_ai_v3_phase9d2_pm_sizing_universe")
    assert missing.feature_found is False
    assert missing.multiplier == 1.0
    assert missing.warning == "pm_v3_feature_row_missing"
    assert all("profit" not in column and "cash" not in column for column in advisor.feature_columns)


def test_phase9f2_report_generates_and_rejects_when_gates_fail(tmp_path: Path) -> None:
    _write_v3_fixture(tmp_path)
    _write_backtest_fixture(tmp_path)
    current_pm = tmp_path / "models/ml/portfolio_manager/current_v2_73_phase3b_clean/model_metadata.json"
    current_exit = tmp_path / "models/ml/exit/current_v2_66/model_metadata.json"
    v282_profile = tmp_path / f"config/profiles/{BASELINE_PROFILE}.yaml"
    current_pm.parent.mkdir(parents=True, exist_ok=True)
    current_exit.parent.mkdir(parents=True, exist_ok=True)
    v282_profile.parent.mkdir(parents=True, exist_ok=True)
    current_pm.write_text("pm-current", encoding="utf-8")
    current_exit.write_text("exit-current", encoding="utf-8")
    v282_profile.write_text(f"profile_id: {BASELINE_PROFILE}\n", encoding="utf-8")
    before = (current_pm.read_text(encoding="utf-8"), current_exit.read_text(encoding="utf-8"), v282_profile.read_text(encoding="utf-8"))

    audit = PMAIV3BacktestCandidateAuditPMSizingUniverse(tmp_path)
    paths = audit.save_report(audit.build_report())
    loaded = json.loads(paths.json.read_text(encoding="utf-8"))

    assert paths.markdown.exists()
    assert loaded["leakage_checklist"]["forbidden_feature_count"] == 0
    assert loaded["leakage_checklist"]["leakage_risk"] == "low"
    assert loaded["adoption_gate"]["adoption_recommendation"] == "reject"
    assert current_pm.read_text(encoding="utf-8") == before[0]
    assert current_exit.read_text(encoding="utf-8") == before[1]
    assert v282_profile.read_text(encoding="utf-8") == before[2]


def test_phase9f2_profiles_are_research_only_and_use_pm_sizing_universe() -> None:
    for profile in CANDIDATE_PROFILES.values():
        text = Path("config/profiles", f"{profile}.yaml").read_text(encoding="utf-8")
        assert "rule: pm_ai_v3_candidate" in text
        assert "model_dir: models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe" in text
        assert "dataset_path: data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_pm_sizing_universe_2023-01_to_2026-05.parquet" in text
        assert "model_dir: models/ml/exit/current_v2_66" in text
        assert "prediction_root: data/ml/walk_forward_predictions" in text

    v282 = Path("config/profiles", f"{BASELINE_PROFILE}.yaml").read_text(encoding="utf-8")
    assert "rule: high_minus_avoid" in v282
    assert "current_v2_73_phase3b_clean" in v282
