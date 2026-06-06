from __future__ import annotations

from typing import Any

import paper_trade
from profile_loader import load_profile
from ml.portfolio_manager_sizing import PortfolioManagerSizingDecision


class _FakeAdvisor:
    def __init__(self, decision: PortfolioManagerSizingDecision) -> None:
        self.decision = decision

    def decision_for(self, signal_date: str, code: str) -> PortfolioManagerSizingDecision:
        return self.decision


def _base_config(*, low_score_skip_enabled: bool = True) -> dict[str, Any]:
    return {
        "portfolio_manager_ai_sizing": {
            "enabled": True,
            "low_score_skip_enabled": low_score_skip_enabled,
            "low_score_skip_threshold": -0.20,
        },
        "trading": {"use_round_lot": True, "round_lot_size": 100},
    }


def test_low_pm_score_becomes_skip(monkeypatch) -> None:
    decision = PortfolioManagerSizingDecision(
        high_conviction_proba=0.30,
        avoid_proba=0.60,
        score=-0.30,
        multiplier=0.60,
        feature_count=68,
        model_version="phase3b_clean",
        feature_found=True,
    )
    monkeypatch.setattr(paper_trade, "_portfolio_manager_sizing_advisor", lambda config: _FakeAdvisor(decision))

    item = {"signal_date": "2026-03-01", "code": "12340"}
    shares, fields = paper_trade._apply_portfolio_manager_sizing(
        item=item,
        trade_date="2026-03-02",
        shares=1000,
        entry_price=100.0,
        cash=200_000.0,
        config=_base_config(low_score_skip_enabled=True),
    )

    assert shares == 0
    assert fields["pm_status"] == "skipped"
    assert fields["pm_resize_reason"] == "pm_low_score_skip"
    assert fields["pm_multiplier"] == 0.60
    assert fields["pm_score"] == -0.30
    assert item["pm_high_conviction_proba"] == 0.30
    assert item["pm_avoid_proba"] == 0.60


def test_low_pm_score_uses_v2_75_multiplier_when_skip_disabled(monkeypatch) -> None:
    decision = PortfolioManagerSizingDecision(
        high_conviction_proba=0.30,
        avoid_proba=0.60,
        score=-0.30,
        multiplier=0.60,
        feature_count=68,
        model_version="phase3b_clean",
        feature_found=True,
    )
    monkeypatch.setattr(paper_trade, "_portfolio_manager_sizing_advisor", lambda config: _FakeAdvisor(decision))

    shares, fields = paper_trade._apply_portfolio_manager_sizing(
        item={"signal_date": "2026-03-01", "code": "12340"},
        trade_date="2026-03-02",
        shares=1000,
        entry_price=100.0,
        cash=200_000.0,
        config=_base_config(low_score_skip_enabled=False),
    )

    assert shares == 600
    assert fields["pm_status"] == "ok"
    assert fields["pm_resize_reason"] == ""
    assert fields["pm_multiplier"] == 0.60


def test_threshold_boundary_is_not_skipped(monkeypatch) -> None:
    decision = PortfolioManagerSizingDecision(
        high_conviction_proba=0.40,
        avoid_proba=0.60,
        score=-0.20,
        multiplier=0.80,
        feature_count=68,
        model_version="phase3b_clean",
        feature_found=True,
    )
    monkeypatch.setattr(paper_trade, "_portfolio_manager_sizing_advisor", lambda config: _FakeAdvisor(decision))

    shares, fields = paper_trade._apply_portfolio_manager_sizing(
        item={"signal_date": "2026-03-01", "code": "12340"},
        trade_date="2026-03-02",
        shares=1000,
        entry_price=100.0,
        cash=200_000.0,
        config=_base_config(low_score_skip_enabled=True),
    )

    assert shares == 800
    assert fields["pm_status"] == "ok"
    assert fields["pm_resize_reason"] == ""


def test_v2_76_profile_enables_low_score_skip() -> None:
    profile = load_profile("rookie_dealer_02_v2_76")
    policy = profile["portfolio_manager_ai_sizing"]

    assert profile["profile_id"] == "rookie_dealer_02_v2_76_pm_ai_low_score_skip"
    assert policy["enabled"] is True
    assert policy["low_score_skip_enabled"] is True
    assert policy["low_score_skip_threshold"] == -0.20
    assert policy["selected_count_in_day_forbidden"] is True
    assert policy["data_lineage_audit_status"] == "PASS"
