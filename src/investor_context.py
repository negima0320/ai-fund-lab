"""Investor type context for market-wide supply and demand scoring."""

from __future__ import annotations

from datetime import date
from typing import Any


INVESTOR_CONTEXT_EMPTY: dict[str, Any] = {
    "investor_context_source": "unavailable",
    "investor_context_week": None,
    "overseas_net_buy": None,
    "overseas_net_buy_4w_sum": None,
    "overseas_net_buy_4w_trend": "unknown",
    "overseas_buy_sell_ratio": None,
    "individual_net_buy": None,
    "institution_net_buy": None,
    "trust_bank_net_buy": None,
    "proprietary_net_buy": None,
    "investor_context_score": 0,
}


def normalize_investor_type_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        week = _date_text(record, ["Date", "date", "EnDate", "end_date", "PublishedDate", "published_date", "PubDate", "Week", "week"])
        if not week:
            continue
        overseas_buy = _number(record, ["overseas_buy", "OverseasBuy", "ForeignersBuy", "ForeignersPurchases", "FrgnBuy"])
        overseas_sell = _number(record, ["overseas_sell", "OverseasSell", "ForeignersSell", "ForeignersSales", "FrgnSell"])
        normalized.append(
            {
                **record,
                "date": week,
                "overseas_net_buy": _net(
                    record,
                    ["overseas_net_buy", "OverseasNetBuy", "ForeignersBalance", "ForeignersNetBuy", "FrgnBal"],
                    overseas_buy,
                    overseas_sell,
                ),
                "overseas_buy": overseas_buy,
                "overseas_sell": overseas_sell,
                "individual_net_buy": _net(
                    record,
                    ["individual_net_buy", "IndividualNetBuy", "IndividualsBalance", "IndividualsNetBuy", "IndBal"],
                    _number(record, ["individual_buy", "IndividualBuy", "IndividualsBuy", "IndBuy"]),
                    _number(record, ["individual_sell", "IndividualSell", "IndividualsSell", "IndSell"]),
                ),
                "institution_net_buy": _net(
                    record,
                    ["institution_net_buy", "InstitutionNetBuy", "InstitutionsBalance", "InstitutionsNetBuy", "InstBal", "BrkBal"],
                    _number(record, ["institution_buy", "InstitutionBuy", "InstitutionsBuy", "InstBuy", "BrkBuy"]),
                    _number(record, ["institution_sell", "InstitutionSell", "InstitutionsSell", "InstSell", "BrkSell"]),
                ),
                "trust_bank_net_buy": _net(
                    record,
                    ["trust_bank_net_buy", "TrustBankNetBuy", "TrustBanksBalance", "TrustBanksNetBuy"],
                    _number(record, ["trust_bank_buy", "TrustBankBuy", "TrustBanksBuy"]),
                    _number(record, ["trust_bank_sell", "TrustBankSell", "TrustBanksSell"]),
                ),
                "proprietary_net_buy": _net(
                    record,
                    ["proprietary_net_buy", "ProprietaryNetBuy", "ProprietaryBalance", "ProprietaryNetBuy", "PropBal"],
                    _number(record, ["proprietary_buy", "ProprietaryBuy", "PropBuy"]),
                    _number(record, ["proprietary_sell", "ProprietarySell", "PropSell"]),
                ),
            }
        )
    return sorted(normalized, key=lambda item: item["date"])


def build_investor_context(records: list[dict[str, Any]], target_date: str | date) -> dict[str, Any]:
    normalized = normalize_investor_type_records(records)
    target = target_date.isoformat() if isinstance(target_date, date) else str(target_date)
    usable = [record for record in normalized if str(record.get("date")) <= target]
    if not usable:
        return dict(INVESTOR_CONTEXT_EMPTY)

    window = usable[-4:]
    latest = window[-1]
    overseas_values = [_number(record, ["overseas_net_buy"]) for record in window]
    overseas_values = [value for value in overseas_values if value is not None]
    overseas_4w_sum = sum(overseas_values) if overseas_values else None
    trend = _trend(overseas_values)
    ratio = _ratio(latest.get("overseas_buy"), latest.get("overseas_sell"))
    score = _investor_context_score(
        overseas_4w_sum=overseas_4w_sum,
        trend=trend,
        overseas_net_buy=_number(latest, ["overseas_net_buy"]),
        individual_net_buy=_number(latest, ["individual_net_buy"]),
    )

    return {
        "investor_context_source": "investor_types",
        "investor_context_week": latest.get("date"),
        "overseas_net_buy": _number(latest, ["overseas_net_buy"]),
        "overseas_net_buy_4w_sum": overseas_4w_sum,
        "overseas_net_buy_4w_trend": trend,
        "overseas_buy_sell_ratio": ratio,
        "individual_net_buy": _number(latest, ["individual_net_buy"]),
        "institution_net_buy": _number(latest, ["institution_net_buy"]),
        "trust_bank_net_buy": _number(latest, ["trust_bank_net_buy"]),
        "proprietary_net_buy": _number(latest, ["proprietary_net_buy"]),
        "investor_context_score": score,
    }


def _investor_context_score(
    overseas_4w_sum: float | None,
    trend: str,
    overseas_net_buy: float | None,
    individual_net_buy: float | None,
) -> int:
    score = 0
    if overseas_4w_sum is not None:
        if overseas_4w_sum > 0:
            score += 2
        elif overseas_4w_sum < 0:
            score -= 2
    if trend == "improving":
        score += 2
    elif trend == "worsening":
        score -= 1
    if (individual_net_buy or 0) < 0 and (overseas_net_buy or 0) > 0:
        score += 1
    return max(-3, min(5, score))


def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "unknown"
    if len(values) >= 4:
        previous = sum(values[:2])
        recent = sum(values[-2:])
    else:
        previous = values[-2]
        recent = values[-1]
    if recent > previous:
        return "improving"
    if recent < previous:
        return "worsening"
    return "flat"


def _net(record: dict[str, Any], net_keys: list[str], buy: float | None, sell: float | None) -> float | None:
    direct = _number(record, net_keys)
    if direct is not None:
        return direct
    if buy is None or sell is None:
        return None
    return buy - sell


def _ratio(buy: Any, sell: Any) -> float | None:
    buy_number = _to_float(buy)
    sell_number = _to_float(sell)
    if buy_number is None or sell_number in (None, 0):
        return None
    return round(buy_number / sell_number, 4)


def _number(record: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _to_float(record.get(key))
        if value is not None:
            return value
    return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _date_text(record: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None or value == "":
            continue
        text = str(value)
        if len(text) == 8 and text.isdigit():
            return f"{text[:4]}-{text[4:6]}-{text[6:]}"
        return text[:10]
    return None
