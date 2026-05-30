"""Feature contribution analysis for closed paper trades."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from db import get_database_path
from trade_metrics import is_closed_trade_for_metrics


SCORE_DETAIL_BUCKET_ORDER = ["65-69", "70-71", "72-73", "74-75", "76-79", "80+"]
COMPONENT_SCORE_BUCKET_ORDER = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-40", "40-50", "50+"]
COMPONENT_DETAIL_BUCKET_ORDER = ["<0", "0", "0-3", "3-5", "5-8", "8-10", "10-15", "15+"]


def build_feature_analysis(
    config: dict[str, Any],
    root: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    db_path = get_database_path(config, root)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    profile_id = _profile_id(config)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        trade_where = "profile_id = ?"
        trade_params: list[Any] = [profile_id]
        if start_date and end_date:
            trade_where += " AND entry_date BETWEEN ? AND ?"
            trade_params.extend([start_date, end_date])
        trade_rows = _rows(
            connection,
            f"""
            SELECT *
            FROM trades
            WHERE {trade_where}
            ORDER BY entry_date, exit_date, id
            """,
            tuple(trade_params),
        )
        market_where = "profile_id = ?"
        market_params: list[Any] = [profile_id]
        if start_date and end_date:
            market_where += " AND date BETWEEN ? AND ?"
            market_params.extend([start_date, end_date])
        market_rows = _rows(
            connection,
            f"""
            SELECT date, market_regime, advance_ratio
            FROM market_contexts
            WHERE {market_where}
            ORDER BY date, id
            """,
            tuple(market_params),
        )
        scoring_where = "profile_id = ?"
        scoring_params: list[Any] = [profile_id]
        if start_date and end_date:
            scoring_where += " AND date BETWEEN ? AND ?"
            scoring_params.extend([start_date, end_date])
        scoring_rows = _rows(
            connection,
            f"""
            SELECT *
            FROM scoring_results
            WHERE {scoring_where}
            ORDER BY date, rank, id
            """,
            tuple(scoring_params),
        )
    closed = [row for row in trade_rows if is_closed_trade_for_metrics(row)]
    market_by_date = {row.get("date"): row for row in market_rows}
    records = [_feature_record(trade, market_by_date.get(trade.get("entry_date"), {})) for trade in closed]
    rsi_filter = _rsi_filter_rejection_summary(scoring_rows, config)
    component_validation = _score_component_validation(records, scoring_rows)

    return {
        "profile_id": profile_id,
        "profile_name": _profile_name(config),
        "start_date": start_date,
        "end_date": end_date,
        "closed_trade_count": len(records),
        "conditional_selected_trade_count": _conditional_selected_trade_count(records),
        "missing_feature_counts": _missing_feature_counts(records),
        "rsi_filter_rejected_count": rsi_filter["count"],
        "rsi_filter_rejected_avg_score": rsi_filter["average_score"],
        "rsi_filter_threshold": rsi_filter["threshold"],
        "rsi": _group_by(records, lambda item: _rsi_bucket(item.get("rsi")), ["0-30", "30-40", "40-50", "50-60", "60-70", "70+"]),
        "volume_ratio": _group_by(records, lambda item: _volume_bucket(item.get("volume_ratio")), ["<1", "1-2", "2-3", "3+"]),
        "market_regime": _group_by(records, lambda item: item.get("market_regime") or "unknown", ["risk_on", "neutral", "risk_off"]),
        "sector": _group_by(records, lambda item: item.get("sector_name") or "未分類"),
        "candlestick_signal": _group_by_signals(records),
        "score": _group_by(records, lambda item: _score_bucket(item.get("total_score")), ["60-65", "65-70", "70-75", "75-80", "80+"]),
        "score_detail": score_detail_groups(records),
        "score_contribution": {
            "selected_score_averages": _selected_score_averages(records),
            "technical_score": _group_by(records, lambda item: _component_score_bucket(item.get("technical_score")), COMPONENT_SCORE_BUCKET_ORDER),
            "financial_score": _group_by(records, lambda item: _component_score_bucket(item.get("financial_score")), COMPONENT_SCORE_BUCKET_ORDER),
            "news_score": _group_by(records, lambda item: _component_score_bucket(item.get("news_score")), COMPONENT_SCORE_BUCKET_ORDER),
        },
        "score_component_analysis": {
            "score_components_validation": component_validation,
            "rsi_score": _group_by(records, lambda item: _score_component_bucket(item.get("rsi_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "volume_score": _group_by(records, lambda item: _score_component_bucket(item.get("volume_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "candlestick_score": _group_by(records, lambda item: _score_component_bucket(item.get("candlestick_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "market_context_score": _group_by(records, lambda item: _score_component_bucket(item.get("market_context_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "penalty_score": _group_by(records, lambda item: _score_component_bucket(item.get("penalty_score")), COMPONENT_DETAIL_BUCKET_ORDER),
        },
        "records_used": records,
    }


def render_feature_analysis_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Feature Analysis",
        "",
        f"- profile_id: {analysis.get('profile_id')}",
        f"- profile_name: {analysis.get('profile_name')}",
        f"- start_date: {analysis.get('start_date') or 'all'}",
        f"- end_date: {analysis.get('end_date') or 'all'}",
        f"- closed_trade_count: {analysis.get('closed_trade_count')}",
        f"- conditional_selected_trade_count: {analysis.get('conditional_selected_trade_count', 0)}",
        f"- missing_feature_counts: {json.dumps(analysis.get('missing_feature_counts', {}), ensure_ascii=False)}",
        f"- rsi_filter_rejected_count: {analysis.get('rsi_filter_rejected_count')}",
        f"- rsi_filter_rejected_avg_score: {_format_number(analysis.get('rsi_filter_rejected_avg_score'))}",
        f"- rsi_filter_threshold: {_format_number(analysis.get('rsi_filter_threshold'))}",
        "",
        "## RSI別勝率・平均利益",
        "",
        *_group_lines(analysis.get("rsi", [])),
        "",
        "## score帯別勝率",
        "",
        *_group_lines(analysis.get("score", [])),
        "",
        "## score詳細分析",
        "",
        *_group_lines(analysis.get("score_detail", [])),
        "",
        "## Score Contribution Analysis",
        "",
        *_score_contribution_lines(analysis.get("score_contribution", {})),
        "",
        "## Score Component Analysis",
        "",
        *_score_component_analysis_lines(analysis.get("score_component_analysis", {})),
        "",
        "## market_regime別勝率",
        "",
        *_group_lines(analysis.get("market_regime", [])),
        "",
        "## volume_ratio別勝率",
        "",
        *_group_lines(analysis.get("volume_ratio", [])),
        "",
        "## sector別勝率",
        "",
        *_group_lines(analysis.get("sector", [])),
        "",
        "## candlestick_signal別勝率",
        "",
        *_group_lines(analysis.get("candlestick_signal", [])),
    ]
    return "\n".join(lines)


def _feature_record(trade: dict[str, Any], entry_market: dict[str, Any] | None = None) -> dict[str, Any]:
    entry_market = entry_market or {}
    score_components = _json_dict(trade.get("score_components"))
    profit = _number(trade.get("net_profit")) or _number(trade.get("profit")) or _number(trade.get("gross_profit")) or 0.0
    profit_rate = _number(trade.get("net_profit_rate")) or _number(trade.get("profit_rate")) or _number(trade.get("gross_profit_rate"))
    advance_ratio = _number(trade.get("advance_ratio"))
    if advance_ratio is None:
        advance_ratio = _number(entry_market.get("advance_ratio"))
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
        "market_regime": trade.get("market_regime") or entry_market.get("market_regime"),
        "advance_ratio": advance_ratio,
        "sector_name": trade.get("sector_name"),
        "candlestick_signals": _json_list(trade.get("candlestick_signals")),
        "selected_reason": trade.get("selected_reason") or trade.get("reason"),
        "total_score": _number(trade.get("total_score") or trade.get("score")),
        "technical_score": _number(trade.get("technical_score")),
        "financial_score": _number(trade.get("financial_score")),
        "news_score": _number(trade.get("news_score")),
        "score_components": score_components,
        "ma_score": _component_value(trade, score_components, "ma_score"),
        "rsi_score": _component_value(trade, score_components, "rsi_score"),
        "volume_score": _component_value(trade, score_components, "volume_score"),
        "candlestick_score": _component_value(trade, score_components, "candlestick_score"),
        "market_context_score": _component_value(trade, score_components, "market_context_score"),
        "sector_score": _component_value(trade, score_components, "sector_score"),
        "penalty_score": _component_value(trade, score_components, "penalty_score"),
        "score_components_total": _number(trade.get("score_components_total")) or _number(score_components.get("component_total")),
        "score_components_match": _bool_or_none(trade.get("score_components_match"), score_components.get("matches_total_score")),
    }


def _conditional_selected_trade_count(records: list[dict[str, Any]]) -> int:
    return sum(1 for record in records if str(record.get("selected_reason") or "").startswith("conditional selected"))


def _rsi_filter_rejection_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rejected = [row for row in rows if row.get("rejected_reason") == "RSI過熱のため新規買付見送り"]
    scores = [_number(row.get("total_score")) for row in rejected]
    scores = [score for score in scores if score is not None]
    return {
        "count": len(rejected),
        "average_score": _average(scores),
        "threshold": _number(config.get("selection", {}).get("max_rsi_for_new_position")),
    }


def _group_by(records: list[dict[str, Any]], bucket_fn: Any, bucket_order: list[str] | None = None) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(bucket_fn(record))].append(record)
    if bucket_order:
        ordered = [_group_stats(name, groups.get(name, [])) for name in bucket_order]
        extra_names = sorted(name for name in groups if name not in set(bucket_order))
        ordered.extend(_group_stats(name, groups[name]) for name in extra_names)
        return ordered
    return sorted((_group_stats(name, items) for name, items in groups.items()), key=lambda item: item["bucket"])


def _group_by_signals(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        signals = record.get("candlestick_signals") or ["no_signal"]
        for signal in signals:
            groups[str(signal)].append(record)
    return sorted((_group_stats(name, items) for name, items in groups.items()), key=lambda item: item["bucket"])


def _group_stats(name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [item for item in items if item.get("result") == "WIN" or (_record_profit(item) or 0) > 0]
    profit_values = [_record_profit(item) or 0.0 for item in items]
    profit_rates = [_record_profit_rate(item) for item in items]
    profit_rates = [rate for rate in profit_rates if rate is not None]
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


def score_detail_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _group_by(records, lambda item: _score_detail_bucket(item.get("total_score")), SCORE_DETAIL_BUCKET_ORDER)


def _missing_feature_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    keys = [
        "rsi",
        "volume_ratio",
        "market_regime",
        "advance_ratio",
        "sector_name",
        "candlestick_signals",
        "total_score",
        "technical_score",
        "financial_score",
        "news_score",
        "score_components",
        "rsi_score",
        "volume_score",
        "candlestick_score",
        "market_context_score",
        "penalty_score",
    ]
    return {key: sum(1 for record in records if _missing_feature(record.get(key))) for key in keys}


def _missing_feature(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, list) and not value:
        return True
    if isinstance(value, dict) and not value:
        return True
    return False


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


def _selected_score_averages(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "technical_score": _average(_valid_numbers(record.get("technical_score") for record in records)),
        "financial_score": _average(_valid_numbers(record.get("financial_score") for record in records)),
        "news_score": _average(_valid_numbers(record.get("news_score") for record in records)),
    }


def _score_contribution_lines(score_contribution: dict[str, Any]) -> list[str]:
    if not score_contribution:
        return ["- データなし"]
    averages = score_contribution.get("selected_score_averages", {})
    lines = [
        "### Selected Score Averages",
        "",
        f"- technical_score average: {_format_number(averages.get('technical_score'))}",
        f"- financial_score average: {_format_number(averages.get('financial_score'))}",
        f"- news_score average: {_format_number(averages.get('news_score'))}",
        "",
        "### technical_score帯",
        "",
        *_group_lines(score_contribution.get("technical_score", [])),
        "",
        "### financial_score帯",
        "",
        *_group_lines(score_contribution.get("financial_score", [])),
        "",
        "### news_score帯",
        "",
        *_group_lines(score_contribution.get("news_score", [])),
    ]
    return lines


def _score_component_analysis_lines(component_analysis: dict[str, Any]) -> list[str]:
    if not component_analysis:
        return ["- データなし"]
    validation = component_analysis.get("score_components_validation", {})
    warnings = validation.get("warnings", [])
    lines = [
        "### total_score内訳検証",
        "",
        f"- closed_trade_count: {validation.get('closed_trade_count', 0)}",
        f"- scoring_rows_count: {validation.get('scoring_rows_count', 0)}",
        f"- selected_scoring_rows_count: {validation.get('selected_scoring_rows_count', 0)}",
        f"- rejected_scoring_rows_count: {validation.get('rejected_scoring_rows_count', 0)}",
        f"- missing_score_components_count: {validation.get('missing_score_components_count', 0)}",
        f"- total_score_mismatch_count: {validation.get('total_score_mismatch_count', 0)}",
    ]
    if warnings:
        lines.extend(["", "### warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    sections = [
        ("rsi_score帯別", "rsi_score"),
        ("volume_score帯別", "volume_score"),
        ("candlestick_score帯別", "candlestick_score"),
        ("market_context_score帯別", "market_context_score"),
        ("penalty_score帯別", "penalty_score"),
    ]
    for title, key in sections:
        lines.extend(["", f"### {title}", ""])
        lines.extend(_group_lines(component_analysis.get(key, [])))
    return lines


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


def _score_detail_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 65:
        return "under_65"
    if value < 70:
        return "65-69"
    if value < 72:
        return "70-71"
    if value < 74:
        return "72-73"
    if value < 76:
        return "74-75"
    if value < 80:
        return "76-79"
    return "80+"


def _component_score_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 5:
        return "0-5"
    if value < 10:
        return "5-10"
    if value < 15:
        return "10-15"
    if value < 20:
        return "15-20"
    if value < 30:
        return "20-30"
    if value < 40:
        return "30-40"
    if value <= 50:
        return "40-50"
    return "50+"


def _score_component_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 0:
        return "<0"
    if value == 0:
        return "0"
    if value < 3:
        return "0-3"
    if value < 5:
        return "3-5"
    if value < 8:
        return "5-8"
    if value < 10:
        return "8-10"
    if value < 15:
        return "10-15"
    return "15+"


def _record_profit(item: dict[str, Any]) -> float | None:
    return _number(item.get("net_profit")) or _number(item.get("profit")) or _number(item.get("gross_profit"))


def _record_profit_rate(item: dict[str, Any]) -> float | None:
    return _number(item.get("net_profit_rate")) or _number(item.get("profit_rate")) or _number(item.get("gross_profit_rate"))


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


def _json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _component_value(row: dict[str, Any], components: dict[str, Any], key: str) -> float | None:
    value = _number(row.get(key))
    if value is not None:
        return value
    return _number(components.get(key))


def _bool_or_none(value: Any, fallback: Any = None) -> bool | None:
    if value is None or value == "":
        value = fallback
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _score_component_validation(records: list[dict[str, Any]], scoring_rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_records = [record for record in records if not record.get("score_components")]
    mismatch_records = [record for record in records if record.get("score_components_match") is False]
    scoring_missing = [row for row in scoring_rows if not _json_dict(row.get("score_components"))]
    scoring_mismatch = [row for row in scoring_rows if _bool_or_none(row.get("score_components_match")) is False]
    warnings = []
    if missing_records or scoring_missing:
        warnings.append("score_components がない古いレコードがあります。再バックテスト後のログで分析してください。")
    if mismatch_records or scoring_mismatch:
        warnings.append("total_score と score_components の合計が一致しないレコードがあります。clampや古い保存形式の可能性があります。")
    return {
        "closed_trade_count": len(records),
        "scoring_rows_count": len(scoring_rows),
        "selected_scoring_rows_count": sum(1 for row in scoring_rows if row.get("selected")),
        "rejected_scoring_rows_count": sum(1 for row in scoring_rows if not row.get("selected")),
        "missing_score_components_count": len(missing_records),
        "missing_scoring_score_components_count": len(scoring_missing),
        "total_score_mismatch_count": len(mismatch_records),
        "scoring_total_score_mismatch_count": len(scoring_mismatch),
        "warnings": warnings,
    }


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_numbers(values: Any) -> list[float]:
    result = []
    for value in values:
        number = _number(value)
        if number is not None:
            result.append(number)
    return result


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _format_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2%}"


def _format_yen(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):,.0f}円"


def _format_number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}"


def _profile_id(config: dict[str, Any]) -> str:
    return str(config.get("profile_id") or config.get("dealer", {}).get("id") or "rookie_dealer_01")


def _profile_name(config: dict[str, Any]) -> str:
    return str(config.get("profile_name") or config.get("dealer", {}).get("name") or _profile_id(config))
