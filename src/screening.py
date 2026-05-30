"""Screening log construction."""

from __future__ import annotations

from datetime import date
from typing import Any

from data_provider import BaseDataProvider


def generate_screening_log(
    config: dict[str, Any],
    run_date: date,
    run_id: str,
    provider: BaseDataProvider,
) -> dict[str, Any]:
    """Generate screening log from the configured data provider."""
    candidate_count = int(config["universe"]["candidate_count"])
    listed_stocks = provider.get_listed_stocks()
    prices = {item["code"]: item for item in provider.get_daily_prices(run_date)}
    candidates = []

    for index, stock in enumerate(listed_stocks[:candidate_count]):
        price = prices[stock["code"]]
        candidates.append(
            {
                "rank": index + 1,
                "code": stock["code"],
                "name": stock["name"],
                "market": stock["market"],
                "sector": stock["sector"],
                "close_price": price["close_price"],
                "momentum_5d": price["momentum_5d"],
                "volume_ratio_20d": price["volume_ratio_20d"],
                "volatility_20d": price["volatility_20d"],
                "screening_reason": _screening_reason(
                    price["momentum_5d"],
                    price["volume_ratio_20d"],
                    price["volatility_20d"],
                ),
            }
        )

    return {
        "run_id": run_id,
        "date": run_date.isoformat(),
        "dealer_id": config["dealer"]["id"],
        "data_provider": config["data_provider"],
        "market": config["universe"]["market"],
        "candidate_count": candidate_count,
        "conditions": [
            "東証プライム上場想定",
            "短期モメンタム確認",
            "出来高増加確認",
            "極端なボラティリティを回避",
        ],
        "candidates": candidates,
    }


def _screening_reason(momentum: float, volume_ratio: float, volatility: float) -> str:
    reasons = []
    if momentum > 0:
        reasons.append("短期モメンタムがプラス")
    if volume_ratio >= 1.5:
        reasons.append("出来高が平常時より増加")
    if volatility <= 0.06:
        reasons.append("許容範囲の値動き")
    return "、".join(reasons) if reasons else "条件確認用のダミー候補"
