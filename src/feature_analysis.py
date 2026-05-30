"""Feature contribution analysis for closed paper trades."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from db import get_database_path
from trade_metrics import is_closed_trade_for_metrics


def build_feature_analysis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    db_path = get_database_path(config, root)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    profile_id = _profile_id(config)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        trade_rows = _rows(
            connection,
            """
            SELECT *
            FROM trades
            WHERE profile_id = ?
            ORDER BY entry_date, exit_date, id
            """,
            (profile_id,),
        )
    closed = [row for row in trade_rows if is_closed_trade_for_metrics(row)]
    records = [_feature_record(trade) for trade in closed]

    return {
        "profile_id": profile_id,
        "profile_name": _profile_name(config),
        "closed_trade_count": len(records),
        "rsi": _group_by(records, lambda item: _rsi_bucket(item.get("rsi")), ["0-30", "30-40", "40-50", "50-60", "60-70", "70+"]),
        "volume_ratio": _group_by(records, lambda item: _volume_bucket(item.get("volume_ratio")), ["<1", "1-2", "2-3", "3+"]),
        "market_regime": _group_by(records, lambda item: item.get("market_regime") or "unknown", ["risk_on", "neutral", "risk_off"]),
        "sector": _group_by(records, lambda item: item.get("sector_name") or "未分類"),
        "candlestick_signal": _group_by_signals(records),
        "score": _group_by(records, lambda item: _score_bucket(item.get("total_score")), ["60-65", "65-70", "70-75", "75-80", "80+"]),
        "records_used": records,
    }


def render_feature_analysis_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Feature Analysis",
        "",
        f"- profile_id: {analysis.get('profile_id')}",
        f"- profile_name: {analysis.get('profile_name')}",
        f"- closed_trade_count: {analysis.get('closed_trade_count')}",
        "",
        "## RSI別勝率・平均利益",
        "",
        *_group_lines(analysis.get("rsi", [])),
        "",
        "## volume_ratio別勝率",
        "",
        *_group_lines(analysis.get("volume_ratio", [])),
        "",
        "## market_regime別勝率",
        "",
        *_group_lines(analysis.get("market_regime", [])),
        "",
        "## sector別勝率",
        "",
        *_group_lines(analysis.get("sector", [])),
        "",
        "## candlestick_signal別勝率",
        "",
        *_group_lines(analysis.get("candlestick_signal", [])),
        "",
        "## score帯別勝率",
        "",
        *_group_lines(analysis.get("score", [])),
    ]
    return "\n".join(lines)


def _feature_record(trade: dict[str, Any]) -> dict[str, Any]:
    profit = _number(trade.get("net_profit")) or _number(trade.get("profit")) or _number(trade.get("gross_profit")) or 0.0
    profit_rate = _number(trade.get("net_profit_rate")) or _number(trade.get("profit_rate")) or _number(trade.get("gross_profit_rate"))
    return {
        "trade_id": trade.get("trade_id"),
        "code": trade.get("code"),
        "name": trade.get("name"),
        "entry_date": trade.get("entry_date"),
        "exit_date": trade.get("exit_date"),
        "result": trade.get("result"),
        "profit": profit,
        "profit_rate": profit_rate,
        "rsi": _number(trade.get("rsi")),
        "volume_ratio": _number(trade.get("volume_ratio")),
        "market_regime": trade.get("market_regime"),
        "sector_name": trade.get("sector_name"),
        "candlestick_signals": _json_list(trade.get("candlestick_signals")),
        "total_score": _number(trade.get("total_score") or trade.get("score")),
    }


def _group_by(records: list[dict[str, Any]], bucket_fn: Any, bucket_order: list[str] | None = None) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(bucket_fn(record))].append(record)
    if bucket_order:
        return [_group_stats(name, groups.get(name, [])) for name in bucket_order]
    return sorted((_group_stats(name, items) for name, items in groups.items()), key=lambda item: item["bucket"])


def _group_by_signals(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        signals = record.get("candlestick_signals") or ["no_signal"]
        for signal in signals:
            groups[str(signal)].append(record)
    return sorted((_group_stats(name, items) for name, items in groups.items()), key=lambda item: item["bucket"])


def _group_stats(name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [item for item in items if item.get("result") == "WIN" or float(item.get("profit") or 0) > 0]
    profit_values = [float(item.get("profit") or 0) for item in items]
    profit_rates = [float(item["profit_rate"]) for item in items if item.get("profit_rate") is not None]
    return {
        "bucket": name,
        "count": len(items),
        "win_count": len(wins),
        "loss_count": len(items) - len(wins),
        "win_rate": round(len(wins) / len(items), 4) if items else None,
        "average_profit": _average(profit_values),
        "average_profit_rate": _average(profit_rates),
        "total_profit": round(sum(profit_values), 2),
    }


def _group_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item['bucket']}: {item['count']}件, 勝率 {_format_percent(item.get('win_rate'))}, "
            f"平均利益 {_format_yen(item.get('average_profit'))}, "
            f"平均利益率 {_format_percent(item.get('average_profit_rate'))}, "
            f"合計損益 {_format_yen(item.get('total_profit'))}"
        )
        for item in items
    ]


def _rsi_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 30:
        return "0-30"
    if value < 40:
        return "30-40"
    if value < 50:
        return "40-50"
    if value < 60:
        return "50-60"
    if value < 70:
        return "60-70"
    return "70+"


def _volume_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 1:
        return "<1"
    if value < 2:
        return "1-2"
    if value < 3:
        return "2-3"
    return "3+"


def _score_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 60:
        return "under_60"
    if value < 65:
        return "60-65"
    if value < 70:
        return "65-70"
    if value < 75:
        return "70-75"
    if value < 80:
        return "75-80"
    return "80+"


def _rows(connection: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, params)]


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        return parsed if isinstance(parsed, list) else []
    return []


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _format_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2%}"


def _format_yen(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):,.0f}円"


def _profile_id(config: dict[str, Any]) -> str:
    return str(config.get("profile_id") or config.get("dealer", {}).get("id") or "rookie_dealer_01")


def _profile_name(config: dict[str, Any]) -> str:
    return str(config.get("profile_name") or config.get("dealer", {}).get("name") or _profile_id(config))
