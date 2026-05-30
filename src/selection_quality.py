"""Selection quality analysis for screened, scored, selected, and rejected stocks."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from db import get_database_path


NUMERIC_LIFT_FIELDS = [
    ("rsi", "RSI"),
    ("volume_ratio", "volume_ratio"),
    ("total_score", "total_score"),
]

CATEGORICAL_LIFT_FIELDS = [
    ("market_regime", "market_regime"),
    ("sector", "sector"),
    ("candlestick_signal", "candlestick_signal"),
]

DEEP_ANALYSIS_FEATURES = [
    ("rsi", "RSI"),
    ("volume_ratio", "volume_ratio"),
    ("total_score", "total_score"),
    ("sector", "sector"),
    ("market_regime", "market_regime"),
    ("candlestick_signal", "candlestick_signal"),
]

MIN_RULE_SAMPLE_SIZE = 5
POSITIVE_LIFT_THRESHOLD = 0.005


def build_selection_quality_analysis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    db_path = get_database_path(config, root)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    profile_id = _profile_id(config)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        screening_rows = _rows(
            connection,
            """
            SELECT *
            FROM screening_results
            WHERE profile_id = ?
            ORDER BY date, id
            """,
            (profile_id,),
        )
        scoring_rows = _rows(
            connection,
            """
            SELECT *
            FROM scoring_results
            WHERE profile_id = ?
            ORDER BY date, rank, id
            """,
            (profile_id,),
        )

    prices_by_code = _load_price_history(root)
    screening_by_key = {(row.get("date"), row.get("code")): row for row in screening_rows}
    screen_records = [
        _quality_record(row, row, prices_by_code, source_stage="screen")
        for row in screening_rows
    ]
    score_records = [
        _quality_record(row, screening_by_key.get((row.get("date"), row.get("code")), {}), prices_by_code, source_stage="score")
        for row in scoring_rows
    ]
    selected_records = [record for record in score_records if record.get("selected")]
    rejected_records = [record for record in score_records if not record.get("selected")]

    return {
        "profile_id": profile_id,
        "profile_name": _profile_name(config),
        "screen_candidate_count": len(screen_records),
        "score_candidate_count": len(score_records),
        "selected_count": len(selected_records),
        "rejected_count": len(rejected_records),
        "conditional_selected_count": _conditional_selected_count(selected_records),
        "conditional_rejected_count": _conditional_rejected_count(rejected_records),
        "screen_candidates": _return_summary(screen_records),
        "score_candidates": _return_summary(score_records),
        "selected": _return_summary(selected_records),
        "rejected": _return_summary(rejected_records),
        "selection_lift": {
            "return_5d": _lift(selected_records, rejected_records, "return_5d"),
            "return_10d": _lift(selected_records, rejected_records, "return_10d"),
        },
        "selection_lift_optimization_analysis": _selection_lift_optimization_analysis(selected_records, rejected_records),
        "selection_lift_deep_analysis": _selection_lift_deep_analysis(selected_records, rejected_records),
        "rejected_reason_analysis": _rejected_reason_analysis(selected_records, rejected_records),
        "missed_opportunity_by_reason": _missed_opportunity_by_reason(rejected_records),
        "top_missed_opportunities": _top_records(
            rejected_records,
            key="return_10d",
            reverse=True,
            limit=10,
        ),
        "top_false_positives": _top_records(
            selected_records,
            key="return_10d",
            reverse=False,
            limit=10,
        ),
    }


def render_selection_quality_markdown(analysis: dict[str, Any]) -> str:
    selected = analysis.get("selected", {})
    rejected = analysis.get("rejected", {})
    lift = analysis.get("selection_lift", {})
    lines = [
        "# Selection Quality Analysis",
        "",
        f"- profile_id: {analysis.get('profile_id')}",
        f"- profile_name: {analysis.get('profile_name')}",
        f"- screen_candidate_count: {analysis.get('screen_candidate_count')}",
        f"- score_candidate_count: {analysis.get('score_candidate_count')}",
        f"- selected_count: {analysis.get('selected_count')}",
        f"- rejected_count: {analysis.get('rejected_count')}",
        f"- conditional_selected_count: {analysis.get('conditional_selected_count', 0)}",
        f"- conditional_rejected_count: {analysis.get('conditional_rejected_count', 0)}",
        "",
        "## Future Return Summary",
        "",
        f"- selected平均5日リターン: {_format_percent(selected.get('average_return_5d'))}",
        f"- rejected平均5日リターン: {_format_percent(rejected.get('average_return_5d'))}",
        f"- selected平均10日リターン: {_format_percent(selected.get('average_return_10d'))}",
        f"- rejected平均10日リターン: {_format_percent(rejected.get('average_return_10d'))}",
        f"- Selection Lift 5d: {_format_percent(lift.get('return_5d'))}",
        f"- Selection Lift 10d: {_format_percent(lift.get('return_10d'))}",
        "",
        "## Selection Lift Optimization Analysis",
        "",
        "### Selected vs Rejected Average",
        "",
        *_feature_average_lines(analysis.get("selection_lift_optimization_analysis", {}).get("feature_averages", [])),
        "",
        "### Category Share Difference",
        "",
        *_category_share_lines(analysis.get("selection_lift_optimization_analysis", {}).get("category_share_differences", [])),
        "",
        "### Selected側だけに多い特徴",
        "",
        *_feature_lift_lines(analysis.get("selection_lift_optimization_analysis", {}).get("selected_heavy_features", [])),
        "",
        "### Rejected側だけに多い特徴",
        "",
        *_feature_lift_lines(analysis.get("selection_lift_optimization_analysis", {}).get("rejected_heavy_features", [])),
        "",
        "## Selection Lift Deep Analysis",
        "",
        *_deep_feature_section_lines(analysis.get("selection_lift_deep_analysis", {}).get("features", {})),
        "",
        "### Positive Lift Features",
        "",
        *_deep_lift_feature_lines(analysis.get("selection_lift_deep_analysis", {}).get("positive_lift_features", [])),
        "",
        "### Negative Lift Features",
        "",
        *_deep_lift_feature_lines(analysis.get("selection_lift_deep_analysis", {}).get("negative_lift_features", [])),
        "",
        "### Candidate New Rules",
        "",
        *_candidate_rule_lines(analysis.get("selection_lift_deep_analysis", {}).get("candidate_new_rules", [])),
        "",
        "## Stage Comparison",
        "",
        *_stage_lines(analysis),
        "",
        "## Rejected Reason Analysis",
        "",
        *_rejected_reason_lines(analysis.get("rejected_reason_analysis", [])),
        "",
        "## Missed Opportunity by Reason",
        "",
        *_missed_opportunity_lines(analysis.get("missed_opportunity_by_reason", [])),
        "",
        "## False Rejection Check",
        "",
        *_false_rejection_lines(analysis.get("rejected_reason_analysis", [])),
        "",
        "## Top Missed Opportunities",
        "",
        *_record_lines(analysis.get("top_missed_opportunities", [])),
        "",
        "## Top False Positives",
        "",
        *_record_lines(analysis.get("top_false_positives", [])),
    ]
    return "\n".join(lines)


def _quality_record(
    row: dict[str, Any],
    screening: dict[str, Any],
    prices_by_code: dict[str, list[dict[str, Any]]],
    source_stage: str,
) -> dict[str, Any]:
    code = str(row.get("code") or "")
    base_date = row.get("date")
    base_close = _number(screening.get("close")) or _price_on_date(prices_by_code.get(code, []), base_date)
    price_5d = _future_price(prices_by_code.get(code, []), base_date, 5)
    price_10d = _future_price(prices_by_code.get(code, []), base_date, 10)
    return {
        "source_stage": source_stage,
        "date": base_date,
        "code": code,
        "name": row.get("name") or screening.get("name"),
        "selected": bool(row.get("selected")) if source_stage == "score" else None,
        "rank": _int(row.get("rank")),
        "total_score": _number(row.get("total_score")),
        "rsi": _first_number(row.get("rsi"), screening.get("rsi")),
        "volume_ratio": _first_number(row.get("volume_ratio"), screening.get("volume_ratio")),
        "market_regime": row.get("market_regime") or screening.get("market_regime") or "unknown",
        "sector": row.get("sector_name") or screening.get("sector_name") or "未分類",
        "candlestick_signals": _json_list(row.get("candlestick_signals")) or _json_list(screening.get("candlestick_signals")),
        "rejected_reason": row.get("rejected_reason"),
        "reason": row.get("reason"),
        "base_close": base_close,
        "price_5d": price_5d,
        "price_10d": price_10d,
        "return_5d": _return_rate(base_close, price_5d),
        "return_10d": _return_rate(base_close, price_10d),
    }


def _selection_lift_optimization_analysis(
    selected_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "feature_averages": _feature_averages(selected_records, rejected_records),
        "category_share_differences": _category_share_differences(selected_records, rejected_records),
        "selected_heavy_features": _feature_bias_ranking(selected_records, rejected_records, reverse=True),
        "rejected_heavy_features": _feature_bias_ranking(selected_records, rejected_records, reverse=False),
    }


def _conditional_selected_count(records: list[dict[str, Any]]) -> int:
    return sum(1 for record in records if str(record.get("reason") or "").startswith("conditional selected"))


def _conditional_rejected_count(records: list[dict[str, Any]]) -> int:
    return sum(1 for record in records if str(record.get("rejected_reason") or "").startswith("conditional rejected"))


def _selection_lift_deep_analysis(
    selected_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = _deep_feature_rows(selected_records, rejected_records)
    positive = [
        row for row in rows
        if _is_positive_lift_feature(row)
    ]
    negative = [
        row for row in rows
        if not _is_positive_lift_feature(row)
        and row.get("selected_count", 0) > 0
        and row.get("rejected_count", 0) > 0
        and row.get("return_lift_10d") is not None
    ]
    positive.sort(key=_deep_feature_sort_key, reverse=True)
    negative.sort(key=lambda row: (float(row.get("return_lift_10d") or -999), float(row.get("win_rate_lift") or -999)))
    return {
        "features": _deep_rows_by_feature(rows),
        "positive_lift_features": positive[:20],
        "negative_lift_features": negative[:20],
        "candidate_new_rules": _candidate_new_rules(positive),
    }


def _deep_feature_rows(selected_records: list[dict[str, Any]], rejected_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, label in DEEP_ANALYSIS_FEATURES:
        selected_groups = _group_records_by_deep_feature(selected_records, key)
        rejected_groups = _group_records_by_deep_feature(rejected_records, key)
        values = sorted(set(selected_groups) | set(rejected_groups))
        for value in values:
            selected_summary = _future_return_summary(selected_groups.get(value, []))
            rejected_summary = _future_return_summary(rejected_groups.get(value, []))
            row = {
                "feature": label,
                "value": value,
                "selected_average_future_return_10d": selected_summary["average_future_return_10d"],
                "selected_win_rate": selected_summary["win_rate"],
                "selected_count": selected_summary["count"],
                "rejected_average_future_return_10d": rejected_summary["average_future_return_10d"],
                "rejected_win_rate": rejected_summary["win_rate"],
                "rejected_count": rejected_summary["count"],
                "return_lift_10d": _difference(selected_summary["average_future_return_10d"], rejected_summary["average_future_return_10d"]),
                "win_rate_lift": _difference(selected_summary["win_rate"], rejected_summary["win_rate"]),
            }
            row["sample_note"] = _sample_note(row)
            rows.append(row)
    return rows


def _deep_rows_by_feature(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["feature"])].append(row)
    return {
        feature: sorted(items, key=lambda item: (-int(item.get("selected_count") or 0) - int(item.get("rejected_count") or 0), str(item.get("value"))))
        for feature, items in grouped.items()
    }


def _group_records_by_deep_feature(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for value in _deep_feature_values(record, key):
            grouped[value].append(record)
    return grouped


def _deep_feature_values(record: dict[str, Any], key: str) -> list[str]:
    if key == "rsi":
        return [_rsi_bucket(record.get("rsi"))]
    if key == "volume_ratio":
        return [_volume_bucket(record.get("volume_ratio"))]
    if key == "total_score":
        return [_score_bucket(record.get("total_score"))]
    if key == "candlestick_signal":
        return _candlestick_signal_values(record)
    if key == "sector":
        return [str(record.get("sector") or "未分類")]
    if key == "market_regime":
        return [str(record.get("market_regime") or "unknown")]
    return ["unknown"]


def _future_return_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    returns = _valid_numbers(record.get("return_10d") for record in records)
    return {
        "average_future_return_10d": _average(returns),
        "win_rate": _positive_rate(returns),
        "count": len(returns),
    }


def _is_positive_lift_feature(row: dict[str, Any]) -> bool:
    return_lift = _number(row.get("return_lift_10d"))
    selected_average = _number(row.get("selected_average_future_return_10d"))
    win_rate_lift = _number(row.get("win_rate_lift"))
    return (
        return_lift is not None
        and selected_average is not None
        and return_lift >= POSITIVE_LIFT_THRESHOLD
        and selected_average > 0
        and (win_rate_lift is None or win_rate_lift >= 0)
        and int(row.get("selected_count") or 0) > 0
    )


def _deep_feature_sort_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(row.get("return_lift_10d") or 0.0),
        float(row.get("win_rate_lift") or 0.0),
        int(row.get("selected_count") or 0),
    )


def _candidate_new_rules(positive_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules = []
    for row in sorted(positive_rows, key=_deep_feature_sort_key, reverse=True):
        rule = _rule_text(row)
        if not rule:
            continue
        rules.append(
            {
                "rule": rule,
                "feature": row.get("feature"),
                "value": row.get("value"),
                "selected_average_future_return_10d": row.get("selected_average_future_return_10d"),
                "selected_win_rate": row.get("selected_win_rate"),
                "selected_count": row.get("selected_count"),
                "rejected_average_future_return_10d": row.get("rejected_average_future_return_10d"),
                "rejected_win_rate": row.get("rejected_win_rate"),
                "rejected_count": row.get("rejected_count"),
                "return_lift_10d": row.get("return_lift_10d"),
                "win_rate_lift": row.get("win_rate_lift"),
                "sample_note": row.get("sample_note"),
                "confidence": "reference" if row.get("sample_note") else "candidate",
            }
        )
    return rules[:10]


def _rule_text(row: dict[str, Any]) -> str:
    feature = str(row.get("feature") or "")
    value = str(row.get("value") or "")
    if feature == "RSI":
        return _bucket_rule("RSI", value)
    if feature == "volume_ratio":
        return _bucket_rule("volume_ratio", value)
    if feature == "total_score":
        return _bucket_rule("total_score", value)
    if feature == "sector":
        return f"sector = {value}"
    if feature == "market_regime":
        return f"market_regime = {value}"
    if feature == "candlestick_signal":
        return f"candlestick_signal = {value}"
    return ""


def _bucket_rule(feature: str, bucket: str) -> str:
    if bucket.startswith("<"):
        return f"{feature} < {bucket[1:]}"
    if bucket.endswith("+"):
        return f"{feature} >= {bucket[:-1]}"
    if "-" in bucket:
        lower, upper = bucket.split("-", 1)
        return f"{feature} >= {lower} and {feature} < {upper}"
    if bucket == "unknown":
        return ""
    return f"{feature} = {bucket}"


def _sample_note(row: dict[str, Any]) -> str:
    selected_count = int(row.get("selected_count") or 0)
    rejected_count = int(row.get("rejected_count") or 0)
    if selected_count < MIN_RULE_SAMPLE_SIZE or rejected_count < MIN_RULE_SAMPLE_SIZE:
        return f"参考扱い: selected/rejectedのいずれかが{MIN_RULE_SAMPLE_SIZE}件未満"
    return ""


def _feature_averages(selected_records: list[dict[str, Any]], rejected_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for key, label in NUMERIC_LIFT_FIELDS:
        selected_average = _average(_valid_numbers(record.get(key) for record in selected_records))
        rejected_average = _average(_valid_numbers(record.get(key) for record in rejected_records))
        result.append(
            {
                "feature": label,
                "selected_average": selected_average,
                "rejected_average": rejected_average,
                "difference": _difference(selected_average, rejected_average),
            }
        )
    return result


def _category_share_differences(selected_records: list[dict[str, Any]], rejected_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, label in CATEGORICAL_LIFT_FIELDS:
        rows.extend(_category_share_rows(selected_records, rejected_records, key, label))
    return rows


def _category_share_rows(
    selected_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
    key: str,
    label: str,
) -> list[dict[str, Any]]:
    selected_counts = _feature_value_counts(selected_records, key)
    rejected_counts = _feature_value_counts(rejected_records, key)
    values = sorted(set(selected_counts) | set(rejected_counts))
    selected_total = sum(selected_counts.values())
    rejected_total = sum(rejected_counts.values())
    rows = []
    for value in values:
        selected_share = _share(selected_counts.get(value, 0), selected_total)
        rejected_share = _share(rejected_counts.get(value, 0), rejected_total)
        rows.append(
            {
                "feature": label,
                "value": value,
                "selected_count": selected_counts.get(value, 0),
                "rejected_count": rejected_counts.get(value, 0),
                "selected_share": selected_share,
                "rejected_share": rejected_share,
                "share_difference": _difference(selected_share, rejected_share),
            }
        )
    return sorted(rows, key=lambda item: (str(item["feature"]), -abs(float(item.get("share_difference") or 0.0)), str(item["value"])))


def _feature_bias_ranking(
    selected_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
    reverse: bool,
    limit: int = 10,
) -> list[dict[str, Any]]:
    selected_counts: dict[tuple[str, str], int] = defaultdict(int)
    rejected_counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in selected_records:
        for feature in _bias_features(record):
            selected_counts[feature] += 1
    for record in rejected_records:
        for feature in _bias_features(record):
            rejected_counts[feature] += 1

    selected_total = len(selected_records)
    rejected_total = len(rejected_records)
    rows = []
    for feature in sorted(set(selected_counts) | set(rejected_counts)):
        selected_share = _share(selected_counts.get(feature, 0), selected_total)
        rejected_share = _share(rejected_counts.get(feature, 0), rejected_total)
        share_difference = _difference(selected_share, rejected_share)
        rows.append(
            {
                "feature": feature[0],
                "value": feature[1],
                "selected_count": selected_counts.get(feature, 0),
                "rejected_count": rejected_counts.get(feature, 0),
                "selected_share": selected_share,
                "rejected_share": rejected_share,
                "share_difference": share_difference,
            }
        )
    rows.sort(key=lambda item: (float(item.get("share_difference") or 0.0), int(item.get("selected_count") or 0) + int(item.get("rejected_count") or 0)), reverse=reverse)
    return rows[:limit]


def _bias_features(record: dict[str, Any]) -> list[tuple[str, str]]:
    features = [
        ("RSI", _rsi_bucket(record.get("rsi"))),
        ("volume_ratio", _volume_bucket(record.get("volume_ratio"))),
        ("total_score", _score_bucket(record.get("total_score"))),
        ("market_regime", str(record.get("market_regime") or "unknown")),
        ("sector", str(record.get("sector") or "未分類")),
    ]
    signals = _candlestick_signal_values(record)
    features.extend(("candlestick_signal", signal) for signal in signals)
    return [(name, value) for name, value in features if value]


def _feature_value_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        values = _candlestick_signal_values(record) if key == "candlestick_signal" else [str(record.get(key) or "unknown")]
        for value in values:
            counts[value] += 1
    return counts


def _candlestick_signal_values(record: dict[str, Any]) -> list[str]:
    signals = [str(signal) for signal in record.get("candlestick_signals", []) if signal]
    return signals or ["no_signal"]


def _return_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    returns_5d = _valid_numbers(record.get("return_5d") for record in records)
    returns_10d = _valid_numbers(record.get("return_10d") for record in records)
    return {
        "count": len(records),
        "return_5d_count": len(returns_5d),
        "return_10d_count": len(returns_10d),
        "average_return_5d": _average(returns_5d),
        "average_return_10d": _average(returns_10d),
    }


def _lift(selected: list[dict[str, Any]], rejected: list[dict[str, Any]], key: str) -> float | None:
    selected_average = _average(_valid_numbers(record.get(key) for record in selected))
    rejected_average = _average(_valid_numbers(record.get(key) for record in rejected))
    if selected_average is None or rejected_average is None:
        return None
    return round(selected_average - rejected_average, 6)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 6)


def _share(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(count / total, 6)


def _rejected_reason_analysis(selected_records: list[dict[str, Any]], rejected_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_average_10d = _average(_valid_numbers(record.get("return_10d") for record in selected_records))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rejected_records:
        groups[_reason(record)].append(record)
    result = []
    for reason, records in groups.items():
        returns_5d = _valid_numbers(record.get("return_5d") for record in records)
        returns_10d = _valid_numbers(record.get("return_10d") for record in records)
        average_10d = _average(returns_10d)
        result.append(
            {
                "rejected_reason": reason,
                "count": len(records),
                "avg_future_return_5d": _average(returns_5d),
                "avg_future_return_10d": average_10d,
                "median_future_return_5d": _median(returns_5d),
                "median_future_return_10d": _median(returns_10d),
                "top_10d_return": max(returns_10d) if returns_10d else None,
                "bottom_10d_return": min(returns_10d) if returns_10d else None,
                "positive_rate_5d": _positive_rate(returns_5d),
                "positive_rate_10d": _positive_rate(returns_10d),
                "selected_avg_future_return_10d": selected_average_10d,
                "filter_effectiveness": _filter_effectiveness(average_10d, selected_average_10d),
            }
        )
    return sorted(result, key=lambda item: (-int(item["count"]), str(item["rejected_reason"])))


def _missed_opportunity_by_reason(rejected_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rejected_records:
        return_10d = _number(record.get("return_10d"))
        if return_10d is not None and return_10d >= 0.2:
            groups[_reason(record)].append(record)
    result = [
        {
            "rejected_reason": reason,
            "count": len(records),
            "top_10d_return": max(_valid_numbers(record.get("return_10d") for record in records)),
        }
        for reason, records in groups.items()
    ]
    return sorted(result, key=lambda item: (-int(item["count"]), str(item["rejected_reason"])))


def _filter_effectiveness(rejected_average_10d: float | None, selected_average_10d: float | None) -> str:
    if rejected_average_10d is None or selected_average_10d is None:
        return "unknown"
    diff = rejected_average_10d - selected_average_10d
    if diff > 0.005:
        return "harmful"
    if diff >= -0.005:
        return "questionable"
    return "effective"


def _top_records(records: list[dict[str, Any]], key: str, reverse: bool, limit: int) -> list[dict[str, Any]]:
    ranked = [record for record in records if _number(record.get(key)) is not None]
    ranked.sort(key=lambda record: float(record.get(key) or 0.0), reverse=reverse)
    return [
        {
            "date": record.get("date"),
            "code": record.get("code"),
            "name": record.get("name"),
            "rank": record.get("rank"),
            "total_score": record.get("total_score"),
            "rejected_reason": record.get("rejected_reason"),
            "return_5d": record.get("return_5d"),
            "return_10d": record.get("return_10d"),
        }
        for record in ranked[:limit]
    ]


def _reason(record: dict[str, Any]) -> str:
    return str(record.get("rejected_reason") or "unknown")


def _load_price_history(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_dir = root / "data" / "raw"
    for path in sorted(raw_dir.glob("prices_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("prices", []):
            code = str(row.get("code") or row.get("Code") or row.get("LocalCode") or "")
            row_date = row.get("date") or row.get("Date")
            close = _number(row.get("close") or row.get("Close") or row.get("AdjustmentClose"))
            if code and row_date and close is not None:
                result[code].append({"date": str(row_date), "close": close})
    for code in result:
        result[code] = sorted(result[code], key=lambda row: row.get("date") or "")
    return result


def _future_price(prices: list[dict[str, Any]], base_date: str | None, offset: int) -> float | None:
    if not base_date:
        return None
    future = [row for row in prices if row.get("date") and row.get("date") > base_date]
    if len(future) < offset:
        return None
    return _number(future[offset - 1].get("close"))


def _price_on_date(prices: list[dict[str, Any]], target_date: str | None) -> float | None:
    if not target_date:
        return None
    for row in prices:
        if row.get("date") == target_date:
            return _number(row.get("close"))
    return None


def _return_rate(base: float | None, future: float | None) -> float | None:
    if base is None or future is None or base == 0:
        return None
    return round((future - base) / base, 6)


def _stage_lines(analysis: dict[str, Any]) -> list[str]:
    labels = [
        ("screen_candidates", "screen候補"),
        ("score_candidates", "score候補"),
        ("selected", "selected銘柄"),
        ("rejected", "rejected銘柄"),
    ]
    lines = []
    for key, label in labels:
        summary = analysis.get(key, {})
        lines.append(
            f"- {label}: {summary.get('count', 0)}件, "
            f"平均5日リターン {_format_percent(summary.get('average_return_5d'))}, "
            f"平均10日リターン {_format_percent(summary.get('average_return_10d'))}"
        )
    return lines


def _record_lines(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return ["- データなし"]
    return [
        (
            f"- {record.get('date')} {record.get('code')} {record.get('name')}: "
            f"score {_format_number(record.get('total_score'))}, "
            f"5d {_format_percent(record.get('return_5d'))}, "
            f"10d {_format_percent(record.get('return_10d'))}, "
            f"reason {record.get('rejected_reason') or 'N/A'}"
        )
        for record in records
    ]


def _rejected_reason_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('rejected_reason')}: count {item.get('count')}, "
            f"avg5d {_format_percent(item.get('avg_future_return_5d'))}, "
            f"avg10d {_format_percent(item.get('avg_future_return_10d'))}, "
            f"median5d {_format_percent(item.get('median_future_return_5d'))}, "
            f"median10d {_format_percent(item.get('median_future_return_10d'))}, "
            f"top10d {_format_percent(item.get('top_10d_return'))}, "
            f"bottom10d {_format_percent(item.get('bottom_10d_return'))}, "
            f"positive5d {_format_percent(item.get('positive_rate_5d'))}, "
            f"positive10d {_format_percent(item.get('positive_rate_10d'))}, "
            f"effectiveness {item.get('filter_effectiveness')}"
        )
        for item in items
    ]


def _missed_opportunity_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('rejected_reason')}: {item.get('count')}件, "
            f"top10d {_format_percent(item.get('top_10d_return'))}"
        )
        for item in items
    ]


def _false_rejection_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('rejected_reason')}: {item.get('filter_effectiveness')} "
            f"(rejected avg10d {_format_percent(item.get('avg_future_return_10d'))}, "
            f"selected avg10d {_format_percent(item.get('selected_avg_future_return_10d'))})"
        )
        for item in items
    ]


def _rows(connection: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, params)]


def _valid_numbers(values: Any) -> list[float]:
    result = []
    for value in values:
        number = _number(value)
        if number is not None:
            result.append(number)
    return result


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else []
    return []


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[midpoint], 6)
    return round((ordered[midpoint - 1] + ordered[midpoint]) / 2, 6)


def _positive_rate(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(1 for value in values if value > 0) / len(values), 6)


def _format_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2%}"


def _format_number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}"


def _feature_average_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('feature')}: selected平均 {_format_number(item.get('selected_average'))}, "
            f"rejected平均 {_format_number(item.get('rejected_average'))}, "
            f"差分 {_format_signed_number(item.get('difference'))}"
        )
        for item in items
    ]


def _category_share_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('feature')}={item.get('value')}: selected {_format_percent(item.get('selected_share'))} "
            f"({item.get('selected_count')}件), rejected {_format_percent(item.get('rejected_share'))} "
            f"({item.get('rejected_count')}件), 差分 {_format_signed_percent(item.get('share_difference'))}"
        )
        for item in items
    ]


def _feature_lift_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('feature')}={item.get('value')}: "
            f"selected {_format_percent(item.get('selected_share'))} ({item.get('selected_count')}件), "
            f"rejected {_format_percent(item.get('rejected_share'))} ({item.get('rejected_count')}件), "
            f"差分 {_format_signed_percent(item.get('share_difference'))}"
        )
        for item in items
    ]


def _deep_feature_section_lines(features: dict[str, list[dict[str, Any]]]) -> list[str]:
    if not features:
        return ["- データなし"]
    lines = []
    for feature in [label for _, label in DEEP_ANALYSIS_FEATURES]:
        items = features.get(feature, [])
        lines.extend([f"### {feature}", "", *_deep_feature_row_lines(items), ""])
    return lines[:-1] if lines else ["- データなし"]


def _deep_feature_row_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('value')}: "
            f"selected avg10d {_format_percent(item.get('selected_average_future_return_10d'))}, "
            f"win_rate {_format_percent(item.get('selected_win_rate'))}, "
            f"count {item.get('selected_count')}; "
            f"rejected avg10d {_format_percent(item.get('rejected_average_future_return_10d'))}, "
            f"win_rate {_format_percent(item.get('rejected_win_rate'))}, "
            f"count {item.get('rejected_count')}; "
            f"lift {_format_signed_percent(item.get('return_lift_10d'))}, "
            f"win_rate_lift {_format_signed_percent(item.get('win_rate_lift'))}"
            f"{_sample_note_suffix(item)}"
        )
        for item in items
    ]


def _deep_lift_feature_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('feature')}={item.get('value')}: "
            f"selected avg10d {_format_percent(item.get('selected_average_future_return_10d'))}, "
            f"win_rate {_format_percent(item.get('selected_win_rate'))}, "
            f"count {item.get('selected_count')}; "
            f"rejected avg10d {_format_percent(item.get('rejected_average_future_return_10d'))}, "
            f"win_rate {_format_percent(item.get('rejected_win_rate'))}, "
            f"count {item.get('rejected_count')}; "
            f"lift {_format_signed_percent(item.get('return_lift_10d'))}"
            f"{_sample_note_suffix(item)}"
        )
        for item in items
    ]


def _candidate_rule_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('rule')}: "
            f"selected avg10d {_format_percent(item.get('selected_average_future_return_10d'))}, "
            f"win_rate {_format_percent(item.get('selected_win_rate'))}, "
            f"count {item.get('selected_count')}; "
            f"lift {_format_signed_percent(item.get('return_lift_10d'))}, "
            f"confidence {item.get('confidence')}"
            f"{_sample_note_suffix(item)}"
        )
        for item in items
    ]


def _sample_note_suffix(item: dict[str, Any]) -> str:
    note = item.get("sample_note")
    return f", {note}" if note else ""


def _format_signed_number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):+.2f}"


def _format_signed_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):+.2%}"


def _rsi_bucket(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    if number < 40:
        return "<40"
    if number < 50:
        return "40-50"
    if number < 60:
        return "50-60"
    if number < 65:
        return "60-65"
    if number < 70:
        return "65-70"
    return "70+"


def _volume_bucket(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    if number < 1:
        return "<1"
    if number < 2:
        return "1-2"
    if number < 3:
        return "2-3"
    return "3+"


def _score_bucket(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    if number < 60:
        return "<60"
    if number < 65:
        return "60-65"
    if number < 70:
        return "65-70"
    if number < 75:
        return "70-75"
    if number < 80:
        return "75-80"
    return "80+"


def _profile_id(config: dict[str, Any]) -> str:
    return str(config.get("profile_id") or config.get("dealer", {}).get("id") or "rookie_dealer_01")


def _profile_name(config: dict[str, Any]) -> str:
    return str(config.get("profile_name") or config.get("dealer", {}).get("name") or _profile_id(config))
