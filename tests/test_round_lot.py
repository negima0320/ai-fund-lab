from __future__ import annotations

from paper_trade import _calculate_buy_shares, _skipped_buy_attempt


def test_calculates_100_share_round_lot(config_copy: dict) -> None:
    shares, reason = _calculate_buy_shares(1500, 200000, config_copy)
    assert shares == 100
    assert reason == ""


def test_does_not_buy_when_round_lot_exceeds_limit(config_copy: dict) -> None:
    shares, reason = _calculate_buy_shares(3000, 200000, config_copy)
    assert shares == 0
    assert "100株購入に必要な金額" in reason


def test_skipped_reason_is_recorded(config_copy: dict) -> None:
    attempt = _skipped_buy_attempt(
        trade_id="T1",
        action="SKIP_BUY",
        code="1001",
        name="Test",
        trade_date="2026-03-06",
        price=3000,
        allocation_limit=200000,
        score=70,
        reason="test",
        skipped_reason="100株購入に必要な金額が1銘柄上限を超えるため買付不可",
        config=config_copy,
    )
    assert attempt["shares"] == 0
    assert attempt["skipped_reason"]
