from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase7a_full_ai_state_audit import Phase7AFullAIStateAudit


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_model(root: Path, rel: str, features: list[str], *, target: str = "target") -> None:
    model_dir = root / rel
    _write_json(
        model_dir / "model_metadata.json",
        {
            "model_profile": model_dir.name,
            "feature_count": len(features),
            "feature_columns": features,
            "target": target,
            "train_start": "2021-06-01",
            "train_end": "2024-12-30",
            "valid_start": "2025-01-06",
            "valid_end": "2025-12-30",
            "test_start": "2026-01-05",
            "test_end": "2026-05-29",
        },
    )
    _write_json(model_dir / "feature_columns.json", features)
    _write_json(model_dir / "metrics.json", {target: {"auc": 0.6}})


def _write_fixture(root: Path, *, pm_backtest_columns: bool = True, stock_forbidden_feature: bool = False) -> None:
    stock_features = ["return_5d", "volume_ratio"]
    if stock_forbidden_feature:
        stock_features.append("selected_count_in_day")
    _write_model(root, "models/ml/current_enriched_v2", stock_features, target="risk_adjusted_score")
    _write_model(root, "models/ml/portfolio_manager/current_v2_73_phase3b_clean", ["risk_adjusted_score", "pm_score"], target="high_conviction_target")
    _write_model(root, "models/ml/exit/current_v2_66", ["holding_days", "unrealized_return"], target="avoid_loss_5d")
    _write_model(root, "models/ml/exit_ai_v2/candidate_v2_api_only", ["return_5d", "daily_range_ratio"], target="exit_quality_top_decile")

    for directory in [
        "data/ml/datasets",
        "data/ml/portfolio_manager",
        "data/ml/exit_datasets",
        "data/ml/exit_ai_v2",
    ]:
        (root / directory).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"date": "2021-06-01", "code": "11110", "return_5d": 0.01, "risk_adjusted_score": 0.2},
            {"date": "2026-05-29", "code": "22220", "return_5d": 0.02, "risk_adjusted_score": 0.3},
        ]
    ).to_parquet(root / "data/ml/datasets/ml_dataset.parquet")
    pd.DataFrame(
        [
            {"signal_date": "2023-01-04", "code": "11110", "high_conviction_target": True, **({"actual_net_profit": 1000} if pm_backtest_columns else {})},
            {"signal_date": "2026-05-28", "code": "22220", "high_conviction_target": False, **({"actual_net_profit": -1000} if pm_backtest_columns else {})},
        ]
    ).to_parquet(root / "data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet")
    pd.DataFrame(
        [
            {"current_date": "2023-01-04", "code": "11110", "avoid_loss_5d": False, "trade_id": "a"},
            {"current_date": "2026-05-28", "code": "22220", "avoid_loss_5d": True, "trade_id": "b"},
        ]
    ).to_parquet(root / "data/ml/exit_datasets/exit_dataset_v2_66_2023-01_to_2026-05.parquet")
    pd.DataFrame(
        [
            {"as_of_date": "2021-06-01", "code": "11110", "return_5d": 0.01, "exit_quality_top_decile": False},
            {"as_of_date": "2026-05-29", "code": "22220", "return_5d": 0.02, "exit_quality_top_decile": True},
        ]
    ).to_parquet(root / "data/ml/exit_ai_v2/exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet")
    for directory, prefix in [
        ("data/ml/features", "features"),
        ("data/ml/labels", "labels"),
        ("data/ml/walk_forward_predictions", "predictions"),
        ("data/cache/jquants/prices", "prices"),
        ("data/cache/jquants/topix_prices", "topix"),
        ("data/cache/jquants/financial_statements", "financial"),
        ("data/cache/jquants/listed_info", "listed"),
    ]:
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"date": "2021-06-01", "code": "11110", "value": 1}]).to_parquet(path / f"{prefix}_2021-06-01.parquet")
        pd.DataFrame([{"date": "2026-05-29", "code": "22220", "value": 2}]).to_parquet(path / f"{prefix}_2026-05-29.parquet")


def test_phase7a_inventories_all_ai_and_recommends_dataset_redesign(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = Phase7AFullAIStateAudit(tmp_path).build_report()

    assert result["metadata"]["audit_only"] is True
    assert len(result["ai_inventory"]) == 5
    assert result["metadata"]["current_main_candidate"] == "rookie_dealer_02_v2_82_cap38"
    assert result["pm_ai_dataset_audit"]["pm_dataset_needs_rebuild"] is True
    assert result["final_verdict"]["next_phase_recommended"] == "Phase 7-B PM AI API-only Dataset Rebuild"
    assert result["final_verdict"]["retraining_should_start_now"] is False


def test_phase7a_detects_2021_data_availability(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = Phase7AFullAIStateAudit(tmp_path).build_report()

    data = result["data_availability_2021"]
    assert data["available_from"] == "2021-06-01"
    assert data["available_to"] == "2026-05-29"
    assert data["api_only_2021_retraining_data_possible"] is True


def test_phase7a_leakage_audit_flags_forbidden_feature(tmp_path: Path) -> None:
    _write_fixture(tmp_path, stock_forbidden_feature=True)

    result = Phase7AFullAIStateAudit(tmp_path).build_report()

    checks = result["leakage_audit"]["checks"]
    assert any(check["status"] == "block" for check in checks)
    assert result["leakage_audit"]["blocking_issues"]


def test_phase7a_saves_report(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    audit = Phase7AFullAIStateAudit(tmp_path)
    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 7-A" in paths.markdown.read_text(encoding="utf-8")
