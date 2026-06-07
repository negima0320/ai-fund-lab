from __future__ import annotations

import json
from pathlib import Path

from paper_trade import (
    ExitAIV2GateAdvisor,
    _apply_exit_ai_v2_gate_to_plan,
    _exit_ai_v2_gate_policy,
    _exit_ai_v2_gate_trade_fields,
)
from profile_loader import load_profile


class FakeGateAdvisor:
    def __init__(self, fields: dict[str, object]) -> None:
        self.fields = fields

    def decision_for(self, *, code: object, trade_date: str, pm_multiplier: object) -> dict[str, object]:
        return dict(self.fields)


def _base_plan(reason: str = "") -> dict[str, object]:
    return {
        "exit_reason": reason,
        "exit_price": None,
        "intended_exit_price": None,
        "execute_now": False,
        "mark_profit_rate": 0.01,
    }


def _gate_fields(*, signal: bool, available: bool = True) -> dict[str, object]:
    return {
        "exit_ai_v2_enabled": True,
        "exit_ai_v2_prediction_available": available,
        "exit_ai_v2_score": 0.24 if signal else 0.12,
        "exit_ai_v2_threshold": 0.20,
        "exit_ai_v2_gate_signal": signal,
        "exit_ai_v2_gate_reason": "score_above_threshold" if signal else "score_below_threshold",
        "exit_ai_v2_feature_missing_count": 0,
        "exit_ai_v2_model_version": "test",
        "exit_ai_v2_used_as_exit_trigger": False,
        "exit_ai_v2_high_pm_safe_mode": False,
        "exit_ai_v2_high_pm_threshold": None,
    }


def test_v280_profiles_load_and_v278_is_unchanged() -> None:
    v278 = load_profile("rookie_dealer_02_v2_78_pm_aware_order_fallback_w025")
    normal = load_profile("rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate")
    high_pm_safe = load_profile("rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate_high_pm_safe")
    alias = load_profile("rookie_dealer_02_v2_80")

    assert not v278.get("ml_exit_ai_v2_gate", {}).get("enabled", False)
    assert normal["profile_id"] == "rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate"
    assert high_pm_safe["ml_exit_ai_v2_gate"]["high_pm_safe_mode"] is True
    assert alias["profile_id"] == normal["profile_id"]


def test_exit_ai_v2_model_path_is_readable_and_current_model_is_not_targeted() -> None:
    profile = load_profile("rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate")
    policy = _exit_ai_v2_gate_policy(profile)

    model_dir = Path(policy["model_dir"])
    current_dir = Path(profile["ml_exit_ai"]["model_dir"])
    assert (model_dir / "exit_quality_top_decile_classifier.joblib").exists()
    assert model_dir != current_dir
    assert str(current_dir) == "models/ml/exit/current_v2_66"


def test_schema_mismatch_falls_back_safely(tmp_path: Path) -> None:
    model_dir = tmp_path / "models/ml/exit_ai_v2/candidate_v2_api_only"
    model_dir.mkdir(parents=True)
    (model_dir / "exit_quality_top_decile_classifier.joblib").write_bytes(b"placeholder")
    (model_dir / "model_metadata.json").write_text(json.dumps({"feature_columns": ["close", "volume"]}), encoding="utf-8")
    (model_dir / "preprocess.json").write_text(json.dumps({"feature_columns": ["close"]}), encoding="utf-8")

    advisor = ExitAIV2GateAdvisor(
        model_dir=model_dir,
        dataset_path=tmp_path / "missing.parquet",
        score_threshold=0.20,
        high_pm_safe_mode=False,
        high_pm_min_multiplier=1.15,
        high_pm_score_threshold=0.25,
        model_version="test",
    )

    assert advisor.available is False
    assert advisor.unavailable_reason == "feature_schema_mismatch"
    decision = advisor.decision_for(code="12340", trade_date="2025-01-06", pm_multiplier=1.0)
    assert decision["exit_ai_v2_prediction_available"] is False
    assert decision["exit_ai_v2_gate_signal"] is False


def test_prediction_unavailable_keeps_existing_exit_logic() -> None:
    advisor = FakeGateAdvisor(_gate_fields(signal=False, available=False))

    plan, fields = _apply_exit_ai_v2_gate_to_plan(
        _base_plan(),
        advisor,
        position={"code": "12340", "pm_multiplier": 1.0},
        trade_date="2025-01-06",
        current_price=100.0,
    )

    assert plan["exit_reason"] == ""
    assert fields["exit_ai_v2_prediction_available"] is False
    assert fields["exit_ai_v2_used_as_exit_trigger"] is False


def test_gate_signal_only_strengthens_when_no_existing_exit_reason() -> None:
    advisor = FakeGateAdvisor(_gate_fields(signal=True))

    plan, fields = _apply_exit_ai_v2_gate_to_plan(
        _base_plan(),
        advisor,
        position={"code": "12340", "pm_multiplier": 1.0},
        trade_date="2025-01-06",
        current_price=100.0,
    )

    assert plan["exit_reason"] == "exit_ai_v2_gate"
    assert plan["execute_now"] is True
    assert fields["exit_ai_v2_used_as_exit_trigger"] is True

    existing, existing_fields = _apply_exit_ai_v2_gate_to_plan(
        _base_plan("stop_loss"),
        advisor,
        position={"code": "12340", "pm_multiplier": 1.0},
        trade_date="2025-01-06",
        current_price=100.0,
    )
    assert existing["exit_reason"] == "stop_loss"
    assert existing_fields["exit_ai_v2_used_as_exit_trigger"] is False


def test_trade_fields_and_selected_count_in_day_guard() -> None:
    profile = load_profile("rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate")
    policy = _exit_ai_v2_gate_policy(profile)
    fields = _exit_ai_v2_gate_trade_fields(_gate_fields(signal=True))

    assert policy["selected_count_in_day_forbidden"] is True
    assert "selected_count_in_day" not in policy
    assert fields["exit_ai_v2_gate_signal"] is True
    assert "exit_ai_v2_score" in fields
