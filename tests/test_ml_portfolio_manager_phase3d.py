from __future__ import annotations

import pytest

from ml.portfolio_manager_sizing import PortfolioManagerSizingAdvisor
from ml.portfolio_manager_sizing import multiplier_from_high_minus_avoid


def test_high_minus_avoid_multiplier_thresholds() -> None:
    assert multiplier_from_high_minus_avoid(0.80, 0.35) == 1.30
    assert multiplier_from_high_minus_avoid(0.70, 0.45) == 1.15
    assert multiplier_from_high_minus_avoid(0.55, 0.55) == 1.00
    assert multiplier_from_high_minus_avoid(0.45, 0.55) == 0.80
    assert multiplier_from_high_minus_avoid(0.30, 0.60) == 0.60
    assert multiplier_from_high_minus_avoid(None, None) == 1.00


def test_clean_feature_count_and_selected_count_are_enforced(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "feature_columns.json").write_text('["selected_count_in_day"]', encoding="utf-8")

    with pytest.raises(ValueError, match="feature_count mismatch"):
        PortfolioManagerSizingAdvisor(root=tmp_path, model_dir=model_dir, dataset_path=tmp_path / "missing.parquet")


def test_selected_count_in_day_is_forbidden_after_count_match(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    cols = [f"f{i}" for i in range(67)] + ["selected_count_in_day"]
    (model_dir / "feature_columns.json").write_text(__import__("json").dumps(cols), encoding="utf-8")

    with pytest.raises(ValueError, match="selected_count_in_day"):
        PortfolioManagerSizingAdvisor(root=tmp_path, model_dir=model_dir, dataset_path=tmp_path / "missing.parquet")


def test_actual_and_audit_like_columns_are_forbidden_after_count_match(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    cols = [f"f{i}" for i in range(67)] + ["actual_net_profit"]
    (model_dir / "feature_columns.json").write_text(__import__("json").dumps(cols), encoding="utf-8")

    with pytest.raises(ValueError, match="Forbidden Portfolio Manager feature columns"):
        PortfolioManagerSizingAdvisor(root=tmp_path, model_dir=model_dir, dataset_path=tmp_path / "missing.parquet")
