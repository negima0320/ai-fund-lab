from __future__ import annotations

from copy import deepcopy

from profile_loader import load_profile
from paper_trade import (
    _apply_bear_pm_booster,
    _apply_per_code_exposure_cap,
    _bear_pm_booster_trade_fields,
    _portfolio_manager_trade_fields,
)


def _base_config() -> dict:
    return {
        "profile_id": "test_profile",
        "bear_pm_booster": {
            "enabled": True,
            "bear_pm_booster_enabled": True,
            "min_pm_multiplier": 1.15,
            "booster_multiplier": 1.5,
        },
        "portfolio_manager_ai_sizing": {
            "per_code_exposure_cap_enabled": True,
            "per_code_exposure_cap_rate": 0.30,
        },
        "trading": {"use_round_lot": True, "round_lot_size": 100},
        "capital_utilization_policy": {"buy_lot_size": 100},
    }


def _item(pm_multiplier: float = 1.15, pm_score: float = 0.2) -> dict:
    return {"code": "12340", "pm_multiplier": pm_multiplier, "pm_score": pm_score}


def test_v281_profile_and_aliases_load() -> None:
    profile = load_profile("rookie_dealer_02_v2_81_bear_pm115_booster_50")
    alias = load_profile("rookie_dealer_02_v2.81")
    underscore_alias = load_profile("rookie_dealer_02_v2_81")

    assert profile["profile_id"] == "rookie_dealer_02_v2_81_bear_pm115_booster_50"
    assert alias["profile_id"] == profile["profile_id"]
    assert underscore_alias["profile_id"] == profile["profile_id"]
    assert profile["bear_pm_booster"]["bear_pm_booster_enabled"] is True
    assert profile["bear_pm_booster"]["booster_multiplier"] == 1.5


def test_v278_profile_does_not_enable_bear_booster() -> None:
    profile = load_profile("rookie_dealer_02_v2_78_pm_aware_order_fallback_w025")

    assert "bear_pm_booster" not in profile or not profile.get("bear_pm_booster", {}).get("bear_pm_booster_enabled", False)


def test_bear_pm115_only_gets_boosted(monkeypatch) -> None:
    monkeypatch.setattr("paper_trade._bear_pm_booster_regime_for_date", lambda trade_date, config: "Bear")
    item = _item(1.15)

    shares, fields = _apply_bear_pm_booster(
        item=item,
        trade_date="2024-08-19",
        shares=200,
        entry_price=100,
        cash=200000,
        config=_base_config(),
    )

    assert shares == 300
    assert fields["market_regime"] == "Bear"
    assert fields["bear_pm_booster_applied"] is True
    assert fields["bear_pm_booster_before_amount"] == 20000
    assert fields["bear_pm_booster_after_amount"] == 30000


def test_bull_neutral_unknown_do_not_get_boosted(monkeypatch) -> None:
    config = _base_config()
    for regime in ["Bull", "Neutral", "Unknown", ""]:
        monkeypatch.setattr("paper_trade._bear_pm_booster_regime_for_date", lambda trade_date, config, regime=regime: regime)
        shares, fields = _apply_bear_pm_booster(
            item=_item(1.3),
            trade_date="2024-08-19",
            shares=200,
            entry_price=100,
            cash=200000,
            config=config,
        )
        assert shares == 200
        assert fields["bear_pm_booster_applied"] is False


def test_pm080_and_pm100_do_not_get_boosted(monkeypatch) -> None:
    monkeypatch.setattr("paper_trade._bear_pm_booster_regime_for_date", lambda trade_date, config: "Bear")
    config = _base_config()

    for multiplier in [0.8, 1.0]:
        shares, fields = _apply_bear_pm_booster(
            item=_item(multiplier),
            trade_date="2024-08-19",
            shares=200,
            entry_price=100,
            cash=200000,
            config=config,
        )
        assert shares == 200
        assert fields["bear_pm_booster_applied"] is False
        assert fields["bear_pm_booster_reason"] == "pm_multiplier_below_threshold"


def test_booster_after_amount_respects_cash_cap(monkeypatch) -> None:
    monkeypatch.setattr("paper_trade._bear_pm_booster_regime_for_date", lambda trade_date, config: "Bear")
    config = _base_config()
    config["trading"]["use_round_lot"] = False

    shares, fields = _apply_bear_pm_booster(
        item=_item(1.3),
        trade_date="2024-08-19",
        shares=100,
        entry_price=1000,
        cash=120000,
        config=config,
    )

    assert shares == 120
    assert fields["bear_pm_booster_after_amount"] == 120000
    assert "available_cash" in fields["bear_pm_booster_limited_by"]


def test_booster_does_not_break_per_code_cap(monkeypatch) -> None:
    monkeypatch.setattr("paper_trade._bear_pm_booster_regime_for_date", lambda trade_date, config: "Bear")
    config = _base_config()
    item = _item(1.3)
    boosted_shares, booster_fields = _apply_bear_pm_booster(
        item=item,
        trade_date="2024-08-19",
        shares=200,
        entry_price=1000,
        cash=400000,
        config=config,
    )

    capped_shares, cap_fields = _apply_per_code_exposure_cap(
        item=item,
        shares=boosted_shares,
        entry_price=1000,
        positions=[],
        total_assets=400000,
        config=config,
    )

    assert booster_fields["bear_pm_booster_applied"] is True
    assert capped_shares == 100
    assert cap_fields["pm_per_code_cap_amount"] == 100000


def test_audit_fields_can_be_generated(monkeypatch) -> None:
    monkeypatch.setattr("paper_trade._bear_pm_booster_regime_for_date", lambda trade_date, config: "Bear")
    item = _item(1.3)
    _, fields = _apply_bear_pm_booster(
        item=item,
        trade_date="2024-08-19",
        shares=200,
        entry_price=100,
        cash=200000,
        config=_base_config(),
    )
    item.update(fields)

    booster_fields = _bear_pm_booster_trade_fields(item)
    trade_fields = _portfolio_manager_trade_fields(item)

    assert booster_fields["bear_pm_booster_applied"] is True
    assert trade_fields["bear_pm_booster_multiplier"] == 1.5
    assert "selected_count_in_day" not in booster_fields
    assert "selected_count_in_day" not in trade_fields
