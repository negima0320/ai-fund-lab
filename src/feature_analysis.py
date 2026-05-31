"""Feature contribution analysis for closed paper trades."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

try:  # pragma: no cover - PyYAML is part of the supported runtime.
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from db import get_database_path
from trade_metrics import is_closed_trade_for_metrics


SCORE_DETAIL_BUCKET_ORDER = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-71", "72-73", "74-75", "76-79", "80+"]
COMPONENT_SCORE_BUCKET_ORDER = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-40", "40-50", "50+"]
COMPONENT_DETAIL_BUCKET_ORDER = ["<0", "0", "0-3", "3-5", "5-8", "8-10", "10-15", "15+"]
RELATIVE_STRENGTH_BUCKET_ORDER = ["< -5%", "-5% to 0%", "0% to 3%", "3% to 5%", "5% to 10%", "10%+"]
RELATIVE_STRENGTH_SCORE_BUCKET_ORDER = ["0", "1-3", "4-6", "7-9", "10"]
SCORE_COMPONENT_KEYS = [
    "ma_score",
    "rsi_score",
    "volume_score",
    "candlestick_score",
    "sector_score",
    "market_context_score",
    "relative_strength_score",
    "investor_context_score",
    "penalty_score",
    "total_score",
]
EFFECTIVE_RANGE_COMPONENTS = [
    "technical_score",
    "relative_strength_score",
    "investor_context_score",
    "market_context_score",
    "penalty_score",
    "rsi_score",
    "volume_score",
    "candlestick_score",
    "sector_score",
]
COMPONENT_CONFIGURED_MAX = {
    "technical_score": 50,
    "relative_strength_score": 10,
    "investor_context_score": 5,
    "market_context_score": 0,
    "penalty_score": 0,
    "rsi_score": 10,
    "volume_score": 10,
    "candlestick_score": 15,
    "sector_score": 5,
}


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
    score_formula_audit = _score_formula_audit(config, records, scoring_rows, component_validation)
    score_effective_range_audit = _score_effective_range_audit(config, records, scoring_rows)
    earnings_exposure = _earnings_calendar_exposure(records, scoring_rows)
    feature_activation_audit = build_feature_activation_audit(config, records, scoring_rows, _registry_features(root, profile_id))
    relative_strength_debug = _relative_strength_debug(scoring_rows)
    relative_strength_pipeline = _relative_strength_pipeline(config, scoring_rows)

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
        "score": _group_by(records, lambda item: _score_bucket(item.get("total_score")), ["40-45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80+"]),
        "score_detail": score_detail_groups(records),
        "score_contribution": {
            "selected_score_averages": _selected_score_averages(records),
            "technical_score": _group_by(records, lambda item: _component_score_bucket(item.get("technical_score")), COMPONENT_SCORE_BUCKET_ORDER),
        },
        "score_component_analysis": {
            "score_components_validation": component_validation,
            "rsi_score": _group_by(records, lambda item: _score_component_bucket(item.get("rsi_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "volume_score": _group_by(records, lambda item: _score_component_bucket(item.get("volume_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "candlestick_score": _group_by(records, lambda item: _score_component_bucket(item.get("candlestick_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "market_context_score": _group_by(records, lambda item: _score_component_bucket(item.get("market_context_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "relative_strength_score": _group_by(records, lambda item: _score_component_bucket(item.get("relative_strength_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "investor_context_score": _group_by(records, lambda item: _score_component_bucket(item.get("investor_context_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "penalty_score": _group_by(records, lambda item: _score_component_bucket(item.get("penalty_score")), COMPONENT_DETAIL_BUCKET_ORDER),
        },
        "score_formula_audit": score_formula_audit,
        "score_effective_range_audit": score_effective_range_audit,
        "feature_activation_audit": feature_activation_audit,
        "relative_strength_pipeline": relative_strength_pipeline,
        "relative_strength_debug": relative_strength_debug,
        "earnings_calendar_exposure": earnings_exposure,
        "relative_strength_analysis": {
            "benchmark_source": _group_by(records, lambda item: item.get("benchmark_source") or "unknown"),
            "relative_strength_score": _group_by(
                records,
                lambda item: _relative_strength_score_bucket(item.get("relative_strength_score")),
                RELATIVE_STRENGTH_SCORE_BUCKET_ORDER,
            ),
            "relative_strength_5d": _group_by(
                records,
                lambda item: _relative_strength_bucket(item.get("relative_strength_5d")),
                RELATIVE_STRENGTH_BUCKET_ORDER,
            ),
            "relative_strength_10d": _group_by(
                records,
                lambda item: _relative_strength_bucket(item.get("relative_strength_10d")),
                RELATIVE_STRENGTH_BUCKET_ORDER,
            ),
            "relative_strength_20d": _group_by(
                records,
                lambda item: _relative_strength_bucket(item.get("relative_strength_20d")),
                RELATIVE_STRENGTH_BUCKET_ORDER,
            ),
        },
        "investor_context_analysis": {
            "investor_context_score": _group_by(
                records,
                lambda item: _investor_context_score_bucket(item.get("investor_context_score")),
                ["<0", "0", "1-2", "3-5"],
            ),
            "overseas_net_buy_4w_sum": _group_by(records, lambda item: _positive_negative_bucket(item.get("overseas_net_buy_4w_sum")), ["positive", "negative", "zero", "unknown"]),
            "overseas_net_buy_4w_trend": _group_by(records, lambda item: item.get("overseas_net_buy_4w_trend") or "unknown", ["improving", "worsening", "flat", "unknown"]),
            "investor_context_source": _group_by(records, lambda item: item.get("investor_context_source") or "unknown"),
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
        "## Score Formula Audit",
        "",
        *_score_formula_audit_lines(analysis.get("score_formula_audit", {})),
        "",
        "## Score Effective Range Audit",
        "",
        *_score_effective_range_audit_lines(analysis.get("score_effective_range_audit", {})),
        "",
        "## Feature Activation Audit",
        "",
        *_feature_activation_audit_lines(analysis.get("feature_activation_audit", {})),
        "",
        "## Relative Strength Analysis",
        "",
        *_relative_strength_analysis_lines(analysis.get("relative_strength_analysis", {})),
        "",
        "## Relative Strength Debug",
        "",
        *_relative_strength_pipeline_lines(analysis.get("relative_strength_pipeline", {})),
        "",
        *_relative_strength_debug_lines(analysis.get("relative_strength_debug", {})),
        "",
        "## Investor Context Analysis",
        "",
        *_investor_context_analysis_lines(analysis.get("investor_context_analysis", {})),
        "",
        "## Earnings Calendar Exposure",
        "",
        *_earnings_calendar_exposure_lines(analysis.get("earnings_calendar_exposure", {})),
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
        "exit_reason": trade.get("exit_reason"),
        "result": trade.get("result"),
        "profit": profit,
        "profit_rate": profit_rate,
        "rsi": _number(trade.get("rsi")),
        "volume_ratio": _number(trade.get("volume_ratio")),
        "stock_return_5d": _number(trade.get("stock_return_5d")),
        "stock_return_10d": _number(trade.get("stock_return_10d")),
        "stock_return_20d": _number(trade.get("stock_return_20d")),
        "benchmark_source": trade.get("benchmark_source") or "unknown",
        "benchmark_return_5d": _number(trade.get("benchmark_return_5d")),
        "benchmark_return_10d": _number(trade.get("benchmark_return_10d")),
        "benchmark_return_20d": _number(trade.get("benchmark_return_20d")),
        "relative_strength_5d": _number(trade.get("relative_strength_5d")),
        "relative_strength_10d": _number(trade.get("relative_strength_10d")),
        "relative_strength_20d": _number(trade.get("relative_strength_20d")),
        "relative_strength_score": _number(trade.get("relative_strength_score")),
        "investor_context_source": trade.get("investor_context_source") or "unknown",
        "investor_context_week": trade.get("investor_context_week"),
        "overseas_net_buy": _number(trade.get("overseas_net_buy")),
        "overseas_net_buy_4w_sum": _number(trade.get("overseas_net_buy_4w_sum")),
        "overseas_net_buy_4w_trend": trade.get("overseas_net_buy_4w_trend") or "unknown",
        "overseas_buy_sell_ratio": _number(trade.get("overseas_buy_sell_ratio")),
        "individual_net_buy": _number(trade.get("individual_net_buy")),
        "institution_net_buy": _number(trade.get("institution_net_buy")),
        "trust_bank_net_buy": _number(trade.get("trust_bank_net_buy")),
        "proprietary_net_buy": _number(trade.get("proprietary_net_buy")),
        "investor_context_score": _number(trade.get("investor_context_score")),
        "market_regime": trade.get("market_regime") or entry_market.get("market_regime"),
        "advance_ratio": advance_ratio,
        "sector_name": trade.get("sector_name"),
        "candlestick_signals": _json_list(trade.get("candlestick_signals")),
        "selected_reason": trade.get("selected_reason") or trade.get("reason"),
        "earnings_filter_checked": bool(trade.get("earnings_filter_checked")),
        "earnings_filter_blocked": bool(trade.get("earnings_filter_blocked")),
        "earnings_filter_reason": trade.get("earnings_filter_reason"),
        "earnings_announcement_date": trade.get("earnings_announcement_date"),
        "total_score": _number(trade.get("total_score") or trade.get("score")),
        "technical_score": _number(trade.get("technical_score")),
        "score_components": score_components,
        "ma_score": _component_value(trade, score_components, "ma_score"),
        "rsi_score": _component_value(trade, score_components, "rsi_score"),
        "volume_score": _component_value(trade, score_components, "volume_score"),
        "candlestick_score": _component_value(trade, score_components, "candlestick_score"),
        "market_context_score": _component_value(trade, score_components, "market_context_score"),
        "relative_strength_score": _component_value(trade, score_components, "relative_strength_score"),
        "investor_context_score": _component_value(trade, score_components, "investor_context_score"),
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


def _earnings_calendar_exposure(records: list[dict[str, Any]], scoring_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_exposure = sum(
        1 for row in scoring_rows
        if bool(row.get("selected")) and (row.get("earnings_filter_blocked") or row.get("earnings_announcement_date"))
    )
    stop_loss_exposure = sum(
        1 for record in records
        if _is_earnings_exposed(record) and "損切" in str(record.get("exit_reason") or "")
    )
    false_positive_exposure = sum(
        1 for record in records
        if _is_earnings_exposed(record) and _number(record.get("profit_rate")) is not None and float(record.get("profit_rate") or 0.0) < 0
    )
    return {
        "selected_earnings_exposure_count": selected_exposure,
        "stop_loss_earnings_exposure_count": stop_loss_exposure,
        "false_positive_earnings_exposure_count": false_positive_exposure,
    }


def _is_earnings_exposed(record: dict[str, Any]) -> bool:
    return bool(record.get("earnings_filter_blocked") or record.get("earnings_announcement_date"))


def build_feature_activation_audit(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    scoring_rows: list[dict[str, Any]],
    registry_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features = config.get("features", {}) if isinstance(config.get("features"), dict) else {}
    scoring = config.get("scoring", {}) if isinstance(config.get("scoring"), dict) else {}
    earnings_filter = config.get("earnings_filter", {}) if isinstance(config.get("earnings_filter"), dict) else {}
    registry_features = registry_features or {}
    items = {
        "relative_strength": _activation_item(
            data_enabled=bool(features.get("relative_strength")),
            scoring_enabled=bool(scoring.get("use_relative_strength_score")),
            registry_enabled=_registry_feature_enabled(registry_features, "relative_strength"),
            trigger_count=_non_zero_count(records, scoring_rows, "relative_strength_score"),
            trigger_label="non_zero_score_count",
        ),
        "investor_context": _activation_item(
            data_enabled=bool(features.get("investor_context")),
            scoring_enabled=bool(scoring.get("use_investor_context_score")),
            registry_enabled=_registry_feature_enabled(registry_features, "investor_context"),
            trigger_count=_non_zero_count(records, scoring_rows, "investor_context_score"),
            trigger_label="non_zero_score_count",
        ),
        "financial_context": _activation_item(
            data_enabled=bool(features.get("financial_context")),
            scoring_enabled=bool(scoring.get("use_financial_score")),
            registry_enabled=_registry_feature_enabled(registry_features, "financial_context"),
            trigger_count=_non_zero_count(records, scoring_rows, "financial_score"),
            trigger_label="non_zero_score_count",
        ),
        "earnings_filter": _activation_item(
            data_enabled=bool(earnings_filter.get("enabled")),
            scoring_enabled=None,
            registry_enabled=_registry_feature_enabled(registry_features, "earnings_filter"),
            trigger_count=_earnings_filter_rejected_count(scoring_rows),
            trigger_label="rejected_count",
        ),
    }
    return {
        "features": items,
        "feature_data_enabled": {
            name: item["data_enabled"]
            for name, item in items.items()
        },
        "feature_scoring_enabled": {
            name: item["scoring_enabled"]
            for name, item in items.items()
        },
        "feature_runtime_active": {
            name: item["runtime_active"]
            for name, item in items.items()
        },
        "feature_trigger_count": {
            name: item["actual_trigger_count"]
            for name, item in items.items()
        },
        "inactive_in_practice": [
            name for name, item in items.items()
            if item["status"] == "inactive_in_practice"
        ],
        "data_only": [
            name for name, item in items.items()
            if item["status"] == "data_only"
        ],
        "config_mismatch": [
            name for name, item in items.items()
            if item["status"] == "config_mismatch"
        ],
    }


def _activation_item(
    data_enabled: bool,
    scoring_enabled: bool | None,
    registry_enabled: bool | None,
    trigger_count: int,
    trigger_label: str,
) -> dict[str, Any]:
    runtime_active = trigger_count > 0
    if registry_enabled is True and not data_enabled:
        status = "config_mismatch"
    elif not data_enabled:
        status = "disabled"
    elif scoring_enabled is False:
        status = "data_only"
    elif runtime_active:
        status = "active"
    else:
        status = "inactive_in_practice"
    return {
        "data_enabled": data_enabled,
        "scoring_enabled": "N/A" if scoring_enabled is None else scoring_enabled,
        "registry_enabled": registry_enabled,
        "runtime_active": runtime_active,
        "actual_trigger_count": trigger_count,
        trigger_label: trigger_count,
        "status": status,
    }


def _non_zero_count(records: list[dict[str, Any]], scoring_rows: list[dict[str, Any]], key: str) -> int:
    count = 0
    for row in [*records, *scoring_rows]:
        value = _feature_value(row, key)
        if value is not None and abs(value) > 0:
            count += 1
    return count


def _feature_value(row: dict[str, Any], key: str) -> float | None:
    value = _number(row.get(key))
    if value is not None:
        return value
    components = _json_dict(row.get("score_components"))
    return _number(components.get(key))


def _earnings_filter_rejected_count(scoring_rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in scoring_rows
        if bool(row.get("earnings_filter_blocked"))
        or "決算予定日前後" in str(row.get("rejected_reason") or "")
        or "決算予定日前後" in str(row.get("earnings_filter_reason") or "")
    )


def _registry_feature_enabled(registry_features: dict[str, Any], feature_name: str) -> bool | None:
    if feature_name not in registry_features:
        return None
    return bool(registry_features.get(feature_name))


def _registry_features(root: Path, profile_id: str) -> dict[str, Any]:
    if yaml is None:
        return {}
    path = root / "config" / "profile_registry.yaml"
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    item = (payload.get("profiles") or {}).get(profile_id, {})
    features = item.get("features", {}) if isinstance(item, dict) else {}
    return features if isinstance(features, dict) else {}


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
        "score_components",
        "rsi_score",
        "volume_score",
        "candlestick_score",
        "market_context_score",
        "relative_strength_score",
        "relative_strength_5d",
        "relative_strength_10d",
        "relative_strength_20d",
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
        "relative_strength_score": _average(_valid_numbers(record.get("relative_strength_score") for record in records)),
    }


def _score_contribution_lines(score_contribution: dict[str, Any]) -> list[str]:
    if not score_contribution:
        return ["- データなし"]
    averages = score_contribution.get("selected_score_averages", {})
    lines = [
        "### Selected Score Averages",
        "",
        f"- technical_score average: {_format_number(averages.get('technical_score'))}",
        "",
        "### technical_score帯",
        "",
        *_group_lines(score_contribution.get("technical_score", [])),
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
        ("relative_strength_score帯別", "relative_strength_score"),
        ("investor_context_score帯別", "investor_context_score"),
        ("penalty_score帯別", "penalty_score"),
    ]
    for title, key in sections:
        lines.extend(["", f"### {title}", ""])
        lines.extend(_group_lines(component_analysis.get(key, [])))
    return lines


def _score_formula_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- データなし"]
    lines = [
        f"- total_score formula: {audit.get('total_score_formula')}",
        f"- expanded formula: {audit.get('expanded_formula')}",
        f"- expected_score_range: {audit.get('expected_score_range')}",
        f"- total_score mismatch count: {audit.get('total_score_mismatch_count', 0)}",
        f"- scoring total_score mismatch count: {audit.get('scoring_total_score_mismatch_count', 0)}",
        f"- profiles using relative_strength_score: {', '.join(audit.get('profiles_using_relative_strength_score', [])) or 'none'}",
        f"- profiles using investor_context_score: {', '.join(audit.get('profiles_using_investor_context_score', [])) or 'none'}",
        "",
        "### Component Averages",
        "",
    ]
    for item in audit.get("component_stats", []):
        lines.append(
            f"- {item['component']}: count {item['count']}, "
            f"average {_format_number(item.get('average'))}, "
            f"min {_format_number(item.get('min'))}, "
            f"max {_format_number(item.get('max'))}"
        )
    lines.extend(["", "### Duplicated Signal Warning", ""])
    warnings = audit.get("duplicated_signal_warnings", [])
    if not warnings:
        lines.append("- なし")
    else:
        for warning in warnings:
            lines.append(f"- {warning.get('level')}: {warning.get('message')}")
    return lines


def _score_effective_range_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- データなし"]
    lines = [
        f"- profile_id: {audit.get('profile_id')}",
        f"- theoretical_max_score: {_format_number(audit.get('theoretical_max_score'))}",
        f"- effective_max_score: {_format_number(audit.get('effective_max_score'))}",
        f"- observed_min_score: {_format_number(audit.get('observed_min_score'))}",
        f"- observed_max_score: {_format_number(audit.get('observed_max_score'))}",
        f"- observed_avg_score: {_format_number(audit.get('observed_avg_score'))}",
        f"- selected_min_score: {_format_number(audit.get('selected_min_score'))}",
        f"- selected_max_score: {_format_number(audit.get('selected_max_score'))}",
        f"- selected_avg_score: {_format_number(audit.get('selected_avg_score'))}",
        "",
        "### Component Effective Range",
        "",
    ]
    for item in audit.get("components", []):
        lines.append(
            f"- {item['component']}: configured_max {_format_number(item.get('configured_max'))}, "
            f"observed_min {_format_number(item.get('observed_min'))}, "
            f"observed_max {_format_number(item.get('observed_max'))}, "
            f"observed_avg {_format_number(item.get('observed_avg'))}, "
            f"non_zero_count {item.get('non_zero_count')}, "
            f"zero_count {item.get('zero_count')}, "
            f"status {item.get('status')}"
        )
    return lines


def _feature_activation_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- データなし"]
    lines = []
    features = audit.get("features", {})
    for name in ["relative_strength", "investor_context", "financial_context", "earnings_filter"]:
        item = features.get(name, {})
        lines.extend(
            [
                f"### {name}",
                "",
                f"- data_enabled: {_format_activation_bool(item.get('data_enabled'))}",
                f"- scoring_enabled: {_format_activation_bool(item.get('scoring_enabled'))}",
                f"- runtime_active: {_format_activation_bool(item.get('runtime_active'))}",
                f"- actual_trigger_count: {item.get('actual_trigger_count', 0)}",
            ]
        )
        if "rejected_count" in item:
            lines.append(f"- rejected_count: {item.get('rejected_count', 0)}")
        if "non_zero_score_count" in item:
            lines.append(f"- non_zero_score_count: {item.get('non_zero_score_count', 0)}")
        lines.extend([f"- status: {item.get('status', 'disabled')}", ""])
    inactive = audit.get("inactive_in_practice", [])
    lines.extend(["### inactive_in_practice", ""])
    if inactive:
        lines.extend(f"- {name}" for name in inactive)
    else:
        lines.append("- なし")
    data_only = audit.get("data_only", [])
    lines.extend(["", "### data_only", ""])
    if data_only:
        lines.extend(f"- {name}" for name in data_only)
    else:
        lines.append("- なし")
    mismatch = audit.get("config_mismatch", [])
    lines.extend(["", "### config_mismatch", ""])
    if mismatch:
        lines.extend(f"- {name}" for name in mismatch)
    else:
        lines.append("- なし")
    return lines


def _format_activation_bool(value: Any) -> str:
    if value == "N/A" or value is None:
        return "N/A"
    return str(bool(value)).lower()


def _relative_strength_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    sections = [
        ("benchmark_source別件数", "benchmark_source"),
        ("relative_strength_score帯別", "relative_strength_score"),
        ("relative_strength_5d帯別", "relative_strength_5d"),
        ("relative_strength_10d帯別", "relative_strength_10d"),
        ("relative_strength_20d帯別", "relative_strength_20d"),
    ]
    lines: list[str] = []
    for title, key in sections:
        if lines:
            lines.append("")
        lines.extend([f"### {title}", ""])
        lines.extend(_group_lines(analysis.get(key, [])))
    return lines


def _relative_strength_debug(scoring_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = list(scoring_rows)
    rs_available = [
        row for row in rows
        if _has_relative_strength_data(row)
    ]
    score_values = [_feature_value(row, "relative_strength_score") for row in rows]
    score_values = [value for value in score_values if value is not None]
    score_distribution = {
        "0": 0,
        "1-3": 0,
        "4-6": 0,
        "7-10": 0,
        "unknown": 0,
    }
    for row in rows:
        score_distribution[_debug_score_bucket(_feature_value(row, "relative_strength_score"))] += 1
    top_rows = sorted(
        rows,
        key=lambda row: (
            -(_feature_value(row, "relative_strength_score") or 0),
            str(row.get("date") or ""),
            str(row.get("code") or ""),
        ),
    )[:20]
    warnings = []
    if rows and all((value or 0) == 0 for value in score_values):
        warnings.append("relative_strength_score is zero for all candidates")
    if rows and not rs_available:
        warnings.append("relative_strength_5d/10d/20d are missing for all candidates")
    return {
        "candidate_count": len(rows),
        "topix_records_loaded": int(max([_feature_value(row, "topix_records_loaded") or 0 for row in rows], default=0)),
        "topix_api_calls": int(max([_feature_value(row, "topix_api_calls") or 0 for row in rows], default=0)),
        "rs_data_available_count": len(rs_available),
        "rs_data_missing_count": len(rows) - len(rs_available),
        "relative_strength_score_distribution": score_distribution,
        "relative_strength_5d_stats": _numeric_stats([_feature_value(row, "relative_strength_5d") for row in rows]),
        "relative_strength_10d_stats": _numeric_stats([_feature_value(row, "relative_strength_10d") for row in rows]),
        "relative_strength_20d_stats": _numeric_stats([_feature_value(row, "relative_strength_20d") for row in rows]),
        "benchmark_source_distribution": _count_values(row.get("benchmark_source") or "unknown" for row in rows),
        "top_20_relative_strength_score": [_relative_strength_debug_record(row) for row in top_rows],
        "warnings": warnings,
    }


def _relative_strength_pipeline(config: dict[str, Any], scoring_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = list(scoring_rows)
    feature_enabled = bool(config.get("features", {}).get("relative_strength"))
    scoring_enabled = bool(config.get("scoring", {}).get("use_relative_strength_score"))
    cache_paths = [str(row.get("topix_cache_path") or "") for row in rows if row.get("topix_cache_path")]
    records_loaded = int(max([_feature_value(row, "topix_records_loaded") or 0 for row in rows], default=0))
    benchmark_provider_called = _any_truthy(rows, "relative_strength_benchmark_provider_called") or records_loaded > 0 or any(
        str(row.get("benchmark_source") or "") in {"topix", "prime_average", "candidate_median"} for row in rows
    )
    cache_exists = _any_truthy(rows, "relative_strength_cache_exists")
    if not cache_exists:
        cache_exists = any(Path(path).exists() for path in cache_paths)
    benchmark_sources = [str(row.get("benchmark_source") or "unknown") for row in rows]
    benchmark_source = next((source for source in benchmark_sources if source and source != "unknown"), "unknown")
    rs_calculated = _any_truthy(rows, "relative_strength_calculated") or any(_has_relative_strength_data(row) for row in rows)
    return {
        "feature_enabled": feature_enabled,
        "scoring_enabled": scoring_enabled,
        "benchmark_provider_called": bool(benchmark_provider_called),
        "cache_path": cache_paths[0] if cache_paths else "",
        "cache_exists": bool(cache_exists),
        "records_loaded": records_loaded,
        "benchmark_source": benchmark_source,
        "rs_calculated": bool(rs_calculated),
    }


def _any_truthy(rows: list[dict[str, Any]], key: str) -> bool:
    return any(_boolish(row.get(key)) for row in rows)


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _relative_strength_pipeline_lines(pipeline: dict[str, Any]) -> list[str]:
    if not pipeline:
        return ["### Relative Strength Pipeline", "", "- データなし"]
    return [
        "### Relative Strength Pipeline",
        "",
        f"- feature enabled: {_format_activation_bool(pipeline.get('feature_enabled'))}",
        f"- scoring enabled: {_format_activation_bool(pipeline.get('scoring_enabled'))}",
        f"- benchmark provider called: {_format_activation_bool(pipeline.get('benchmark_provider_called'))}",
        f"- cache path: {pipeline.get('cache_path') or 'N/A'}",
        f"- cache exists: {_format_activation_bool(pipeline.get('cache_exists'))}",
        f"- records loaded: {pipeline.get('records_loaded', 0)}",
        f"- benchmark source: {pipeline.get('benchmark_source') or 'unknown'}",
        f"- rs calculated: {_format_activation_bool(pipeline.get('rs_calculated'))}",
        "",
    ]


def _has_relative_strength_data(row: dict[str, Any]) -> bool:
    return any(_feature_value(row, key) is not None for key in ["relative_strength_5d", "relative_strength_10d", "relative_strength_20d"])


def _debug_score_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value <= 0:
        return "0"
    if value <= 3:
        return "1-3"
    if value <= 6:
        return "4-6"
    return "7-10"


def _numeric_stats(values: list[float | None]) -> dict[str, Any]:
    numbers = [value for value in values if value is not None]
    return {
        "average": _average(numbers),
        "max": max(numbers) if numbers else None,
        "min": min(numbers) if numbers else None,
    }


def _count_values(values: Any) -> dict[str, int]:
    counts = {"topix": 0, "prime_average": 0, "candidate_median": 0, "unknown": 0}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _relative_strength_debug_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": row.get("date"),
        "code": row.get("code"),
        "score": _feature_value(row, "relative_strength_score"),
        "rs5": _feature_value(row, "relative_strength_5d"),
        "rs10": _feature_value(row, "relative_strength_10d"),
        "rs20": _feature_value(row, "relative_strength_20d"),
    }


def _relative_strength_debug_lines(debug: dict[str, Any]) -> list[str]:
    if not debug:
        return ["- データなし"]
    distribution = debug.get("relative_strength_score_distribution", {})
    benchmark = debug.get("benchmark_source_distribution", {})
    lines = [
        f"- candidate_count: {debug.get('candidate_count', 0)}",
        f"- topix_records_loaded: {debug.get('topix_records_loaded', 0)}",
        f"- topix_api_calls: {debug.get('topix_api_calls', 0)}",
        f"- rs_data_available_count: {debug.get('rs_data_available_count', 0)}",
        f"- rs_data_missing_count: {debug.get('rs_data_missing_count', 0)}",
        "",
        "### warnings",
        "",
    ]
    warnings = debug.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- なし")
    lines.extend(
        [
            "",
            "### relative_strength_score",
            "",
            f"- 0点: {distribution.get('0', 0)}件",
            f"- 1-3点: {distribution.get('1-3', 0)}件",
            f"- 4-6点: {distribution.get('4-6', 0)}件",
            f"- 7-10点: {distribution.get('7-10', 0)}件",
            f"- unknown: {distribution.get('unknown', 0)}件",
            "",
            "### relative_strength_5d",
            "",
            *_rs_stat_lines(debug.get("relative_strength_5d_stats", {})),
            "",
            "### relative_strength_10d",
            "",
            *_rs_stat_lines(debug.get("relative_strength_10d_stats", {})),
            "",
            "### relative_strength_20d",
            "",
            *_rs_stat_lines(debug.get("relative_strength_20d_stats", {})),
            "",
            "### benchmark_source",
            "",
            f"- topix: {benchmark.get('topix', 0)}件",
            f"- prime_average: {benchmark.get('prime_average', 0)}件",
            f"- candidate_median: {benchmark.get('candidate_median', 0)}件",
            f"- unknown: {benchmark.get('unknown', 0)}件",
            "",
            "### Top 20 relative_strength_score",
            "",
            "| date | code | score | rs5 | rs10 | rs20 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    top = debug.get("top_20_relative_strength_score", [])
    if top:
        for row in top:
            lines.append(
                "| "
                f"{row.get('date')} | {row.get('code')} | {_format_number(row.get('score'))} | "
                f"{_format_percent(row.get('rs5'))} | {_format_percent(row.get('rs10'))} | {_format_percent(row.get('rs20'))} |"
            )
    else:
        lines.append("| なし |  |  |  |  |  |")
    return lines


def _rs_stat_lines(stats: dict[str, Any]) -> list[str]:
    return [
        f"- average: {_format_percent(stats.get('average'))}",
        f"- max: {_format_percent(stats.get('max'))}",
        f"- min: {_format_percent(stats.get('min'))}",
    ]


def _investor_context_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    sections = [
        ("investor_context_score帯別", "investor_context_score"),
        ("overseas_net_buy_4w_sum正負別", "overseas_net_buy_4w_sum"),
        ("overseas_net_buy_4w_trend別", "overseas_net_buy_4w_trend"),
        ("investor_context_source別件数", "investor_context_source"),
    ]
    lines: list[str] = []
    for title, key in sections:
        if lines:
            lines.append("")
        lines.extend([f"### {title}", ""])
        lines.extend(_group_lines(analysis.get(key, [])))
    return lines


def _earnings_calendar_exposure_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    return [
        f"- selected銘柄のうち決算前後だった件数: {analysis.get('selected_earnings_exposure_count', 0)}",
        f"- stop_loss銘柄のうち決算前後だった件数: {analysis.get('stop_loss_earnings_exposure_count', 0)}",
        f"- false_positive銘柄のうち決算前後だった件数: {analysis.get('false_positive_earnings_exposure_count', 0)}",
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
    if value < 40:
        return "under_40"
    if value < 45:
        return "40-45"
    if value < 50:
        return "45-50"
    if value < 55:
        return "50-55"
    if value < 60:
        return "55-60"
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
    if value < 40:
        return "under_40"
    if value < 45:
        return "40-44"
    if value < 50:
        return "45-49"
    if value < 55:
        return "50-54"
    if value < 60:
        return "55-59"
    if value < 65:
        return "60-64"
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


def _relative_strength_score_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value <= 0:
        return "0"
    if value <= 3:
        return "1-3"
    if value <= 6:
        return "4-6"
    if value < 10:
        return "7-9"
    return "10"


def _relative_strength_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < -0.05:
        return "< -5%"
    if value < 0:
        return "-5% to 0%"
    if value < 0.03:
        return "0% to 3%"
    if value < 0.05:
        return "3% to 5%"
    if value < 0.10:
        return "5% to 10%"
    return "10%+"


def _investor_context_score_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 0:
        return "<0"
    if value == 0:
        return "0"
    if value <= 2:
        return "1-2"
    return "3-5"


def _positive_negative_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _positive_negative_unknown_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _bool_bucket(value: Any) -> str:
    bool_value = _bool_or_none(value)
    if bool_value is None:
        return "unknown"
    return "true" if bool_value else "false"


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


def _score_formula_audit(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    scoring_rows: list[dict[str, Any]],
    component_validation: dict[str, Any],
) -> dict[str, Any]:
    relative_strength_enabled = _relative_strength_enabled(config)
    investor_context_enabled = _investor_context_enabled(config)
    return {
        "total_score_formula": "total_score = technical_score + relative_strength_score + investor_context_score + market_context_score + penalty_score",
        "expanded_formula": (
            "technical_score = clamp(ma_score + rsi_score + volume_score + "
            "candlestick_score + sector_score, 0, 50); "
            "total_score = technical_score + relative_strength_score + investor_context_score + market_context_score + penalty_score"
        ),
        "expected_score_range": _expected_score_range(relative_strength_enabled, investor_context_enabled),
        "relative_strength_enabled": relative_strength_enabled,
        "investor_context_enabled": investor_context_enabled,
        "profiles_using_relative_strength_score": [_profile_id(config)] if relative_strength_enabled else [],
        "profiles_using_investor_context_score": [_profile_id(config)] if investor_context_enabled else [],
        "component_stats": _score_component_stats(records, scoring_rows),
        "total_score_mismatch_count": component_validation.get("total_score_mismatch_count", 0),
        "scoring_total_score_mismatch_count": component_validation.get("scoring_total_score_mismatch_count", 0),
        "duplicated_signal_warnings": _duplicated_signal_warnings(config, relative_strength_enabled, investor_context_enabled),
    }


def _score_effective_range_audit(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    scoring_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    relative_strength_enabled = _relative_strength_enabled(config)
    investor_context_enabled = _investor_context_enabled(config)
    rows = _effective_score_records(records, scoring_rows)
    total_scores = _valid_numbers(row.get("total_score") for row in rows)
    selected_rows = [row for row in rows if row.get("selected")]
    selected_scores = _valid_numbers(row.get("total_score") for row in selected_rows)
    component_summaries = [
        _effective_component_summary(component, rows, relative_strength_enabled, investor_context_enabled)
        for component in EFFECTIVE_RANGE_COMPONENTS
    ]
    top_level_components = {
        "technical_score",
        "relative_strength_score",
        "investor_context_score",
        "market_context_score",
        "penalty_score",
    }
    effective_max_score = sum(
        float(item.get("observed_max") or 0)
        for item in component_summaries
        if item.get("component") in top_level_components and item.get("status") != "inactive"
    )
    return {
        "profile_id": _profile_id(config),
        "theoretical_max_score": _theoretical_max_score(relative_strength_enabled, investor_context_enabled),
        "effective_max_score": round(effective_max_score, 4),
        "observed_min_score": _min_or_none(total_scores),
        "observed_max_score": _max_or_none(total_scores),
        "observed_avg_score": _average(total_scores),
        "selected_min_score": _min_or_none(selected_scores),
        "selected_max_score": _max_or_none(selected_scores),
        "selected_avg_score": _average(selected_scores),
        "components": component_summaries,
    }


def _effective_score_records(records: list[dict[str, Any]], scoring_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if scoring_rows:
        return [_effective_record_from_scoring(row) for row in scoring_rows]
    return [_effective_record_from_trade(record) for record in records]


def _effective_record_from_scoring(row: dict[str, Any]) -> dict[str, Any]:
    components = _json_dict(row.get("score_components"))
    record = _component_record(row, components)
    record["total_score"] = _number(row.get("total_score")) or _number(components.get("total_score"))
    record["selected"] = bool(row.get("selected"))
    return record


def _effective_record_from_trade(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **{component: _number(record.get(component)) for component in EFFECTIVE_RANGE_COMPONENTS},
        "total_score": _number(record.get("total_score")),
        "selected": True,
    }


def _component_record(row: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    record = {key: _component_value(row, components, key) for key in SCORE_COMPONENT_KEYS}
    technical_score = _number(row.get("technical_score"))
    if technical_score is None:
        technical_score = _technical_score_from_components(record)
    record["technical_score"] = technical_score
    return record


def _technical_score_from_components(record: dict[str, Any]) -> float | None:
    values = [
        _number(record.get("ma_score")),
        _number(record.get("rsi_score")),
        _number(record.get("volume_score")),
        _number(record.get("candlestick_score")),
        _number(record.get("sector_score")),
    ]
    if any(value is None for value in values):
        return None
    return round(max(0.0, min(50.0, sum(float(value) for value in values))), 4)


def _effective_component_summary(
    component: str,
    rows: list[dict[str, Any]],
    relative_strength_enabled: bool,
    investor_context_enabled: bool,
) -> dict[str, Any]:
    values = _valid_numbers(row.get(component) for row in rows)
    non_zero_count = sum(1 for value in values if abs(value) > 0)
    zero_count = sum(1 for value in values if value == 0)
    return {
        "component": component,
        "configured_max": _configured_component_max(component, relative_strength_enabled, investor_context_enabled),
        "observed_min": _min_or_none(values),
        "observed_max": _max_or_none(values),
        "observed_avg": _average(values),
        "non_zero_count": non_zero_count,
        "zero_count": zero_count,
        "status": _component_status(component, values),
    }


def _configured_component_max(component: str, relative_strength_enabled: bool, investor_context_enabled: bool) -> float:
    if component == "relative_strength_score" and not relative_strength_enabled:
        return 0.0
    if component == "investor_context_score" and not investor_context_enabled:
        return 0.0
    return float(COMPONENT_CONFIGURED_MAX.get(component, 0))


def _component_status(component: str, values: list[float]) -> str:
    if not values:
        return "no_data"
    if all(value == 0 for value in values):
        if component in {"market_context_score", "penalty_score", "relative_strength_score", "investor_context_score"}:
            return "inactive"
        return "zero"
    return "active"


def _theoretical_max_score(relative_strength_enabled: bool, investor_context_enabled: bool) -> float:
    return 50.0 + (10.0 if relative_strength_enabled else 0.0) + (5.0 if investor_context_enabled else 0.0)


def _expected_score_range(relative_strength_enabled: bool, investor_context_enabled: bool) -> str:
    return f"0-{int(_theoretical_max_score(relative_strength_enabled, investor_context_enabled))}"


def _relative_strength_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("features", {}).get("relative_strength")) and bool(
        config.get("scoring", {}).get("use_relative_strength_score")
    )


def _investor_context_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("features", {}).get("investor_context")) and bool(
        config.get("scoring", {}).get("use_investor_context_score")
    )


def _score_component_stats(records: list[dict[str, Any]], scoring_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(records)
    rows.extend(_scoring_component_record(row) for row in scoring_rows)
    stats = []
    for key in SCORE_COMPONENT_KEYS:
        values = _valid_numbers(row.get(key) for row in rows)
        stats.append(
            {
                "component": key,
                "count": len(values),
                "average": _average(values),
                "min": round(min(values), 4) if values else None,
                "max": round(max(values), 4) if values else None,
            }
        )
    return stats


def _scoring_component_record(row: dict[str, Any]) -> dict[str, Any]:
    components = _json_dict(row.get("score_components"))
    record = _component_record(row, components)
    record["total_score"] = _number(row.get("total_score")) or _number(components.get("total_score"))
    return record


def _duplicated_signal_warnings(config: dict[str, Any], relative_strength_enabled: bool, investor_context_enabled: bool) -> list[dict[str, str]]:
    warnings = []
    selection = config.get("selection", {})
    volume_filter = config.get("volume_filter", {})
    market_filter = config.get("market_filter", {})
    if selection.get("max_rsi_for_new_position") is not None:
        warnings.append(
            {
                "level": "review",
                "message": "RSI is used in rsi_score and also in the RSI selection filter; this is not double-added to total_score, but it affects eligibility.",
            }
        )
    if volume_filter.get("enabled"):
        warnings.append(
            {
                "level": "review",
                "message": "volume_ratio is used in volume_score and also in the volume selection filter; this is not double-added to total_score, but it affects eligibility.",
            }
        )
    if market_filter.get("enabled"):
        warnings.append(
            {
                "level": "ok",
                "message": "market_regime is used by selection filtering while market_context_score is currently 0, so no score double-add was detected.",
            }
        )
    warnings.append(
        {
            "level": "ok",
            "message": "candlestick signals are decomposed into candlestick_score inside technical_score; no separate candlestick add-on was detected.",
        }
    )
    if relative_strength_enabled:
        warnings.append(
            {
                "level": "ok",
                "message": "relative_strength_score is not part of technical_score and is added once as a separate component.",
            }
        )
    if investor_context_enabled:
        warnings.append(
            {
                "level": "ok",
                "message": "investor_context_score is market-wide weekly context and is added once outside technical_score.",
            }
        )
    return warnings


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


def _min_or_none(values: list[float]) -> float | None:
    return round(min(values), 4) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return round(max(values), 4) if values else None


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
