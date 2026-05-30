"""Build market context from cached J-Quants price data."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


def build_market_context(target_date_text: str, provider: str, root: Path) -> dict[str, Any]:
    target_date = date.fromisoformat(target_date_text)
    current_prices = _load_prices(root, target_date_text)
    previous = _load_previous_prices(root, target_date)
    stock_metadata = _load_stock_metadata(root)

    if not current_prices:
        return neutral_market_context(target_date_text, provider, "指定日の価格データがないためneutralとして扱います。")

    previous_by_code = {item.get("code"): item for item in previous}
    rows = []
    for item in current_prices:
        code = item.get("code")
        close = _to_float(item.get("close"))
        prev_close = _to_float((previous_by_code.get(code) or {}).get("close"))
        if close is None or prev_close is None or prev_close <= 0:
            continue
        change_rate = (close - prev_close) / prev_close
        turnover = _to_float(item.get("turnover_value"))
        if turnover is None:
            volume = _to_float(item.get("volume")) or 0
            turnover = close * volume
        volume = _to_float(item.get("volume")) or 0
        previous_volume = _to_float((previous_by_code.get(code) or {}).get("volume"))
        sector_name = (stock_metadata.get(code) or {}).get("sector_name") or "未分類"
        rows.append(
            {
                "code": code,
                "change_rate": change_rate,
                "turnover_value": turnover,
                "volume": volume,
                "previous_volume": previous_volume,
                "sector_name": sector_name,
            }
        )

    if not rows:
        return neutral_market_context(target_date_text, provider, "前日比を計算できる銘柄がないためneutralとして扱います。")

    advancers = sum(1 for item in rows if item["change_rate"] > 0)
    decliners = sum(1 for item in rows if item["change_rate"] < 0)
    unchanged = len(rows) - advancers - decliners
    advance_ratio = advancers / len(rows)
    average_change_rate = sum(item["change_rate"] for item in rows) / len(rows)
    turnover_value_total = sum(item["turnover_value"] for item in rows)
    market_regime = classify_market_regime(advance_ratio, average_change_rate)
    sector_momentum = build_sector_momentum(rows)

    return {
        "date": target_date_text,
        "provider": provider,
        "topix": None,
        "topix_change_rate": None,
        "nikkei": None,
        "nikkei_change_rate": None,
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "advance_ratio": round(advance_ratio, 4),
        "average_change_rate": round(average_change_rate, 6),
        "turnover_value_total": round(turnover_value_total, 2),
        "turnover_trend": None,
        "market_regime": market_regime,
        "usd_jpy": None,
        "us_market_summary": None,
        "important_news": [],
        "sector_momentum": sector_momentum,
        "top_sectors": sector_momentum[:5],
        "market_comment": market_comment(market_regime, advance_ratio, average_change_rate),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def neutral_market_context(target_date_text: str, provider: str, reason: str) -> dict[str, Any]:
    return {
        "date": target_date_text,
        "provider": provider,
        "topix": None,
        "topix_change_rate": None,
        "nikkei": None,
        "nikkei_change_rate": None,
        "advancers": 0,
        "decliners": 0,
        "unchanged": 0,
        "advance_ratio": None,
        "average_change_rate": None,
        "turnover_value_total": None,
        "turnover_trend": None,
        "market_regime": "neutral",
        "usd_jpy": None,
        "us_market_summary": None,
        "important_news": [],
        "sector_momentum": [],
        "top_sectors": [],
        "market_comment": reason,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def build_sector_momentum(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("sector_name") or "未分類", []).append(row)

    sectors = []
    max_turnover = max((sum(item["turnover_value"] for item in items) for items in grouped.values()), default=0)
    for sector_name, items in grouped.items():
        advancers = sum(1 for item in items if item["change_rate"] > 0)
        advance_ratio = advancers / len(items)
        average_change_rate = sum(item["change_rate"] for item in items) / len(items)
        turnover_value_total = sum(item["turnover_value"] for item in items)
        volume_increase_count = sum(
            1
            for item in items
            if item.get("previous_volume") is not None
            and float(item.get("previous_volume") or 0) > 0
            and float(item.get("volume") or 0) > float(item.get("previous_volume") or 0)
        )
        score = _sector_momentum_score(advance_ratio, average_change_rate, turnover_value_total, max_turnover, volume_increase_count, len(items))
        sectors.append(
            {
                "sector_name": sector_name,
                "stock_count": len(items),
                "advancers": advancers,
                "advance_ratio": round(advance_ratio, 4),
                "average_change_rate": round(average_change_rate, 6),
                "turnover_value_total": round(turnover_value_total, 2),
                "volume_increase_count": volume_increase_count,
                "sector_momentum_score": score,
                "sector_comment": sector_comment(sector_name, score, advance_ratio, average_change_rate, volume_increase_count),
            }
        )

    sectors.sort(key=lambda item: (item["sector_momentum_score"], item["turnover_value_total"]), reverse=True)
    for rank, item in enumerate(sectors, start=1):
        item["sector_rank"] = rank
    return sectors


def sector_comment(sector_name: str, score: float, advance_ratio: float, average_change_rate: float, volume_increase_count: int) -> str:
    if score >= 65:
        return f"{sector_name}は値上がり比率{advance_ratio:.1%}、平均騰落率{average_change_rate:.2%}で相対的に強い業種です。"
    if score <= 35:
        return f"{sector_name}は値上がり比率{advance_ratio:.1%}、平均騰落率{average_change_rate:.2%}で相対的に弱い業種です。"
    return f"{sector_name}は値上がり比率{advance_ratio:.1%}、出来高増加銘柄{volume_increase_count}件で中立圏です。"


def _sector_momentum_score(
    advance_ratio: float,
    average_change_rate: float,
    turnover_value_total: float,
    max_turnover: float,
    volume_increase_count: int,
    stock_count: int,
) -> float:
    advance_part = advance_ratio * 40
    change_part = max(0.0, min(1.0, (average_change_rate + 0.03) / 0.06)) * 30
    turnover_part = (turnover_value_total / max_turnover * 15) if max_turnover > 0 else 0
    volume_part = (volume_increase_count / stock_count * 15) if stock_count > 0 else 0
    return round(max(0.0, min(100.0, advance_part + change_part + turnover_part + volume_part)), 2)


def classify_market_regime(advance_ratio: float, average_change_rate: float) -> str:
    if advance_ratio >= 0.60 and average_change_rate > 0:
        return "risk_on"
    if advance_ratio <= 0.40 and average_change_rate < 0:
        return "risk_off"
    return "neutral"


def market_comment(market_regime: str, advance_ratio: float, average_change_rate: float) -> str:
    ratio = f"{advance_ratio:.1%}"
    change = f"{average_change_rate:.2%}"
    if market_regime == "risk_on":
        return f"値上がり銘柄比率は{ratio}、平均騰落率は{change}です。地合いはrisk_onと判断します。"
    if market_regime == "risk_off":
        return f"値上がり銘柄比率は{ratio}、平均騰落率は{change}です。地合いはrisk_offと判断し、買付は慎重に扱います。"
    return f"値上がり銘柄比率は{ratio}、平均騰落率は{change}です。地合いはneutralと判断し、個別銘柄スコアを優先します。"


def _load_prices(root: Path, target_date_text: str) -> list[dict[str, Any]]:
    path = root / "data" / "raw" / f"prices_{target_date_text}.json"
    if not path.exists():
        return []
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("prices", [])


def _load_stock_metadata(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "data" / "raw" / "prime_stocks_jquants.json"
    if not path.exists():
        return {}
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item.get("code"): item for item in payload.get("stocks", []) if item.get("code")}


def _load_previous_prices(root: Path, target_date: date) -> list[dict[str, Any]]:
    for offset in range(1, 11):
        previous_date = target_date - timedelta(days=offset)
        rows = _load_prices(root, previous_date.isoformat())
        if rows:
            return rows
    return []


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
