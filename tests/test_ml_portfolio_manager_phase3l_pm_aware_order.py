from __future__ import annotations

import importlib.util
from pathlib import Path

from profile_loader import load_profile

import paper_trade


class _Decision:
    def __init__(self, score: float, multiplier: float = 1.0) -> None:
        self.score = score
        self.multiplier = multiplier

    def as_fields(self) -> dict:
        return {
            "pm_ai_enabled": True,
            "pm_status": "ok",
            "pm_feature_count": 68,
            "pm_high_conviction_proba": 0.6,
            "pm_avoid_proba": 0.6 - self.score,
            "pm_score": self.score,
            "pm_multiplier": self.multiplier,
            "pm_model_version": "fake",
            "pm_feature_found": True,
            "pm_warning": "",
        }


class _Advisor:
    def decision_for(self, signal_date: str, code: str) -> _Decision:
        scores = {"1111": -0.2, "2222": 0.8, "3333": 0.1}
        multipliers = {"1111": 0.8, "2222": 1.3, "3333": 1.0}
        return _Decision(scores[str(code)], multipliers[str(code)])


def _base_config() -> dict:
    return {
        "profile_id": "test",
        "ai_purchase_policy": {"enabled": True},
        "portfolio_manager_ai_sizing": {
            "enabled": True,
            "buy_ordering_mode": "pm_aware",
            "pm_order_weight": 1.0,
            "fallback_to_next_affordable_selected": True,
            "fallback_min_pm_score": 0.0,
            "fallback_min_pm_multiplier": 1.0,
        },
    }


def test_v2_78_profiles_load_and_enable_pm_aware_ordering() -> None:
    config = load_profile("rookie_dealer_02_v2_78")
    policy = config["portfolio_manager_ai_sizing"]

    assert config["profile_id"] == "rookie_dealer_02_v2_78_pm_aware_order_fallback_w050"
    assert policy["buy_ordering_mode"] == "pm_aware"
    assert policy["pm_order_weight"] == 0.50
    assert policy["fallback_to_next_affordable_selected"] is True
    assert policy["selected_count_in_day_forbidden"] is True


def test_v2_77_sorting_is_unchanged_without_pm_aware() -> None:
    config = {
        "ai_purchase_policy": {"enabled": True},
        "portfolio_manager_ai_sizing": {"enabled": True},
    }
    selected = [
        {"code": "2222", "daily_score_rank": 2, "risk_adjusted_score": 0.9},
        {"code": "1111", "daily_score_rank": 1, "risk_adjusted_score": -0.1},
    ]

    ordered = paper_trade._sort_selected_candidates(selected, config)

    assert [item["code"] for item in ordered] == ["1111", "2222"]


def test_pm_aware_ordering_uses_pm_score(monkeypatch) -> None:
    monkeypatch.setattr(paper_trade, "_portfolio_manager_sizing_advisor", lambda config: _Advisor())
    selected = [
        {"code": "1111", "date": "2026-01-05", "daily_score_rank": 1, "risk_adjusted_score": 0.1},
        {"code": "2222", "date": "2026-01-05", "daily_score_rank": 2, "risk_adjusted_score": 0.2},
        {"code": "3333", "date": "2026-01-05", "daily_score_rank": 3, "risk_adjusted_score": 0.3},
    ]

    ordered = paper_trade._sort_selected_candidates(selected, _base_config())

    assert ordered[0]["code"] == "2222"
    assert ordered[0]["buy_ordering_mode"] == "pm_aware"
    assert ordered[0]["original_candidate_order"] == 2
    assert ordered[0]["pm_aware_candidate_order"] == 1
    assert "buy_priority_score" in ordered[0]
    assert "selected_count_in_day" not in ordered[0]


def test_selected_fallback_quality_filter() -> None:
    config = _base_config()
    allowed = {"code": "1111", "pm_score": -0.1, "pm_multiplier": 1.0}
    rejected = {"code": "2222", "pm_score": -0.1, "pm_multiplier": 0.8}

    assert paper_trade._phase3l_selected_fallback_quality_allowed(allowed, config) is True
    assert paper_trade._phase3l_selected_fallback_quality_allowed(rejected, config) is False


def test_phase3l_report_generates_json_and_markdown(tmp_path: Path) -> None:
    script = Path("scripts/ml/report_portfolio_manager_phase3l_pm_aware_order.py")
    spec = importlib.util.spec_from_file_location("phase3l_report", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.build_report(tmp_path)
    markdown, json_path = module.save_report(result, tmp_path)

    assert result["constraints"]["selected_count_in_day_used"] is False
    assert markdown.exists()
    assert json_path.exists()
