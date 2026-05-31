"""Selection quality analysis for screened, scored, selected, and rejected stocks."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from db import get_database_path
from earnings_calendar import EARNINGS_FILTER_REJECTED_REASON


NUMERIC_LIFT_FIELDS = [
    ("rsi", "RSI"),
    ("volume_ratio", "volume_ratio"),
    ("total_score", "total_score"),
    ("relative_strength_score", "relative_strength_score"),
    ("relative_strength_5d", "relative_strength_5d"),
    ("relative_strength_10d", "relative_strength_10d"),
    ("relative_strength_20d", "relative_strength_20d"),
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
    ("relative_strength_score", "relative_strength_score"),
    ("relative_strength_5d", "relative_strength_5d"),
    ("relative_strength_10d", "relative_strength_10d"),
    ("relative_strength_20d", "relative_strength_20d"),
    ("sector", "sector"),
    ("market_regime", "market_regime"),
    ("candlestick_signal", "candlestick_signal"),
]

MIN_RULE_SAMPLE_SIZE = 5
MIN_SECTOR_SAMPLE_SIZE = 10
POSITIVE_LIFT_THRESHOLD = 0.005
LOW_SCORE_MIN = 65
LOW_SCORE_MAX = 69
LOW_SCORE_RECORD_LIMIT = 20
LOW_SCORE_SEPARATION_LIMIT = 20


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
        "relative_strength_selection_quality": _relative_strength_selection_quality(selected_records, rejected_records),
        "investor_context_selection_quality": _investor_context_selection_quality(selected_records, rejected_records),
        "sector_lift_analysis": _sector_lift_analysis(selected_records, rejected_records),
        "low_score_deep_analysis": _low_score_deep_analysis(score_records, config),
        "earnings_filter_analysis": _earnings_filter_analysis(selected_records, rejected_records),
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
        "## Relative Strength Selection Quality",
        "",
        *_relative_strength_selection_quality_lines(analysis.get("relative_strength_selection_quality", {})),
        "",
        "## Investor Context Selection Quality",
        "",
        *_investor_context_selection_quality_lines(analysis.get("investor_context_selection_quality", {})),
        "",
        "## Sector Lift Analysis",
        "",
        *_sector_lift_lines(analysis.get("sector_lift_analysis", {}).get("sectors", [])),
        "",
        "### Positive Sector Filters",
        "",
        *_sector_filter_lines(analysis.get("sector_lift_analysis", {}).get("positive_sector_filters", [])),
        "",
        "### Negative Sector Filters",
        "",
        *_sector_filter_lines(analysis.get("sector_lift_analysis", {}).get("negative_sector_filters", [])),
        "",
        "## Low Score Deep Analysis",
        "",
        *_low_score_deep_analysis_lines(analysis.get("low_score_deep_analysis", {})),
        "",
        "## Earnings Filter Analysis",
        "",
        *_earnings_filter_analysis_lines(analysis.get("earnings_filter_analysis", {})),
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
        "stock_return_5d": _first_number(row.get("stock_return_5d"), screening.get("stock_return_5d")),
        "stock_return_10d": _first_number(row.get("stock_return_10d"), screening.get("stock_return_10d")),
        "stock_return_20d": _first_number(row.get("stock_return_20d"), screening.get("stock_return_20d")),
        "benchmark_source": row.get("benchmark_source") or screening.get("benchmark_source") or "unknown",
        "benchmark_return_5d": _first_number(row.get("benchmark_return_5d"), screening.get("benchmark_return_5d")),
        "benchmark_return_10d": _first_number(row.get("benchmark_return_10d"), screening.get("benchmark_return_10d")),
        "benchmark_return_20d": _first_number(row.get("benchmark_return_20d"), screening.get("benchmark_return_20d")),
        "relative_strength_5d": _first_number(row.get("relative_strength_5d"), screening.get("relative_strength_5d")),
        "relative_strength_10d": _first_number(row.get("relative_strength_10d"), screening.get("relative_strength_10d")),
        "relative_strength_20d": _first_number(row.get("relative_strength_20d"), screening.get("relative_strength_20d")),
        "relative_strength_score": _first_number(row.get("relative_strength_score"), screening.get("relative_strength_score")),
        "investor_context_source": row.get("investor_context_source") or "unknown",
        "investor_context_week": row.get("investor_context_week"),
        "overseas_net_buy": _number(row.get("overseas_net_buy")),
        "overseas_net_buy_4w_sum": _number(row.get("overseas_net_buy_4w_sum")),
        "overseas_net_buy_4w_trend": row.get("overseas_net_buy_4w_trend") or "unknown",
        "overseas_buy_sell_ratio": _number(row.get("overseas_buy_sell_ratio")),
        "individual_net_buy": _number(row.get("individual_net_buy")),
        "institution_net_buy": _number(row.get("institution_net_buy")),
        "investor_context_score": _number(row.get("investor_context_score")),
        "market_regime": row.get("market_regime") or screening.get("market_regime") or "unknown",
        "sector": row.get("sector_name") or screening.get("sector_name") or "未分類",
        "candlestick_signals": _json_list(row.get("candlestick_signals")) or _json_list(screening.get("candlestick_signals")),
        "rejected_reason": row.get("rejected_reason"),
        "reason": row.get("reason"),
        "earnings_filter_checked": bool(row.get("earnings_filter_checked")),
        "earnings_filter_blocked": bool(row.get("earnings_filter_blocked")),
        "earnings_filter_reason": row.get("earnings_filter_reason"),
        "earnings_announcement_date": row.get("earnings_announcement_date"),
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


def _earnings_filter_analysis(selected_records: list[dict[str, Any]], rejected_records: list[dict[str, Any]]) -> dict[str, Any]:
    rejected = [
        record for record in rejected_records
        if record.get("earnings_filter_blocked") or record.get("rejected_reason") == EARNINGS_FILTER_REJECTED_REASON
    ]
    selected_avg10d = _average(_valid_numbers(record.get("return_10d") for record in selected_records))
    rejected_avg10d = _average(_valid_numbers(record.get("return_10d") for record in rejected))
    return {
        "rejected_by_earnings_filter_count": len(rejected),
        "rejected_by_earnings_filter_avg5d": _average(_valid_numbers(record.get("return_5d") for record in rejected)),
        "rejected_by_earnings_filter_avg10d": rejected_avg10d,
        "earnings_filter_effectiveness": _filter_effectiveness(rejected_avg10d, selected_avg10d),
    }


def _relative_strength_selection_quality(selected_records: list[dict[str, Any]], rejected_records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "relative_strength_5d",
        "relative_strength_10d",
        "relative_strength_20d",
        "relative_strength_score",
    ]
    return {
        field: {
            "selected_average": _average(_valid_numbers(record.get(field) for record in selected_records)),
            "rejected_average": _average(_valid_numbers(record.get(field) for record in rejected_records)),
        }
        for field in fields
    }


def _investor_context_selection_quality(selected_records: list[dict[str, Any]], rejected_records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["investor_context_score", "overseas_net_buy_4w_sum"]
    return {
        field: {
            "selected_average": _average(_valid_numbers(record.get(field) for record in selected_records)),
            "rejected_average": _average(_valid_numbers(record.get(field) for record in rejected_records)),
        }
        for field in fields
    }


def _sector_lift_analysis(selected_records: list[dict[str, Any]], rejected_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _sector_lift_rows(selected_records, rejected_records)
    positive = [
        row for row in rows
        if _number(row.get("lift")) is not None
        and float(row.get("lift") or 0.0) >= POSITIVE_LIFT_THRESHOLD
        and _number(row.get("selected_avg10d")) is not None
        and float(row.get("selected_avg10d") or 0.0) > 0
    ]
    negative = [
        row for row in rows
        if _number(row.get("lift")) is not None
        and (float(row.get("lift") or 0.0) <= 0 or float(row.get("selected_avg10d") or 0.0) <= 0)
        and int(row.get("selected_count") or 0) > 0
        and int(row.get("rejected_count") or 0) > 0
    ]
    positive.sort(key=_sector_lift_sort_key, reverse=True)
    negative.sort(key=lambda row: (float(row.get("lift") or 999), float(row.get("selected_avg10d") or 999)))
    return {
        "sectors": rows,
        "positive_sector_filters": [_sector_filter_candidate(row, "positive") for row in positive[:20]],
        "negative_sector_filters": [_sector_filter_candidate(row, "negative") for row in negative[:20]],
    }


def _sector_lift_rows(selected_records: list[dict[str, Any]], rejected_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_groups = _group_records_by_deep_feature(selected_records, "sector")
    rejected_groups = _group_records_by_deep_feature(rejected_records, "sector")
    sectors = sorted(set(selected_groups) | set(rejected_groups))
    rows = []
    for sector in sectors:
        selected_summary = _future_return_summary(selected_groups.get(sector, []))
        rejected_summary = _future_return_summary(rejected_groups.get(sector, []))
        lift = _difference(selected_summary["average_future_return_10d"], rejected_summary["average_future_return_10d"])
        row = {
            "sector": sector,
            "selected_avg10d": selected_summary["average_future_return_10d"],
            "rejected_avg10d": rejected_summary["average_future_return_10d"],
            "lift": lift,
            "selected_count": selected_summary["count"],
            "rejected_count": rejected_summary["count"],
            "confidence": _sector_lift_confidence(selected_summary["count"], rejected_summary["count"], lift),
        }
        row["sample_note"] = _sector_sample_note(row)
        rows.append(row)
    return sorted(rows, key=lambda row: (float(row.get("lift") or -999), int(row.get("selected_count") or 0) + int(row.get("rejected_count") or 0)), reverse=True)


def _sector_lift_sort_key(row: dict[str, Any]) -> tuple[float, int]:
    return (
        float(row.get("lift") or 0.0),
        int(row.get("selected_count") or 0) + int(row.get("rejected_count") or 0),
    )


def _sector_lift_confidence(selected_count: int, rejected_count: int, lift: float | None) -> str:
    if lift is None:
        return "unknown"
    if selected_count < MIN_SECTOR_SAMPLE_SIZE or rejected_count < MIN_SECTOR_SAMPLE_SIZE:
        return "reference"
    lift_abs = abs(lift)
    if lift_abs >= 0.02:
        return "strong"
    if lift_abs >= POSITIVE_LIFT_THRESHOLD:
        return "moderate"
    return "weak"


def _sector_sample_note(row: dict[str, Any]) -> str:
    if int(row.get("selected_count") or 0) < MIN_SECTOR_SAMPLE_SIZE or int(row.get("rejected_count") or 0) < MIN_SECTOR_SAMPLE_SIZE:
        return f"参考扱い: selected/rejectedのいずれかが{MIN_SECTOR_SAMPLE_SIZE}件未満"
    return ""


def _sector_filter_candidate(row: dict[str, Any], direction: str) -> dict[str, Any]:
    return {
        "filter": f"sector = {row.get('sector')}",
        "sector": row.get("sector"),
        "selected_avg10d": row.get("selected_avg10d"),
        "rejected_avg10d": row.get("rejected_avg10d"),
        "lift": row.get("lift"),
        "selected_count": row.get("selected_count"),
        "rejected_count": row.get("rejected_count"),
        "confidence": "reference" if row.get("sample_note") else direction,
        "sample_note": row.get("sample_note"),
    }


def _low_score_deep_analysis(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    low_score_min, low_score_max = _low_score_range(config)
    low_score_records = [
        record for record in records
        if _is_low_score_record(record, low_score_min, low_score_max) and _number(record.get("return_10d")) is not None
    ]
    winners = [
        record for record in low_score_records
        if float(record.get("return_10d") or 0.0) > 0
    ]
    losers = [
        record for record in low_score_records
        if float(record.get("return_10d") or 0.0) <= 0
    ]
    winners.sort(key=lambda record: float(record.get("return_10d") or 0.0), reverse=True)
    losers.sort(key=lambda record: float(record.get("return_10d") or 0.0))
    separations = _low_score_feature_separations(winners, losers)
    return {
        "score_range": {"min": low_score_min, "max": low_score_max},
        "low_score_count": len(low_score_records),
        "winner_count": len(winners),
        "loser_count": len(losers),
        "winners": _low_score_record_summaries(winners),
        "losers": _low_score_record_summaries(losers),
        "feature_separation": separations[:LOW_SCORE_SEPARATION_LIMIT],
        "candidate_rescue_rules": _low_score_candidate_rescue_rules(separations, winners, losers, low_score_min, low_score_max),
    }


def _low_score_range(config: dict[str, Any]) -> tuple[float, float]:
    selection = config.get("selection", {})
    return (
        float(selection.get("fallback_min_score", LOW_SCORE_MIN)),
        float(selection.get("min_score", LOW_SCORE_MAX + 1)) - 1,
    )


def _is_low_score_record(record: dict[str, Any], low_score_min: float, low_score_max: float) -> bool:
    total_score = _number(record.get("total_score"))
    return total_score is not None and low_score_min <= total_score <= low_score_max


def _low_score_record_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "date": record.get("date"),
            "code": record.get("code"),
            "name": record.get("name"),
            "rank": record.get("rank"),
            "total_score": record.get("total_score"),
            "return_10d": record.get("return_10d"),
            "rsi": record.get("rsi"),
            "volume_ratio": record.get("volume_ratio"),
            "market_regime": record.get("market_regime"),
            "sector": record.get("sector"),
            "candlestick_signals": record.get("candlestick_signals", []),
        }
        for record in records[:LOW_SCORE_RECORD_LIMIT]
    ]


def _low_score_feature_separations(
    winners: list[dict[str, Any]],
    losers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    winner_counts = _low_score_feature_counts(winners)
    loser_counts = _low_score_feature_counts(losers)
    rows = []
    for feature in sorted(set(winner_counts) | set(loser_counts)):
        winner_count = winner_counts.get(feature, 0)
        loser_count = loser_counts.get(feature, 0)
        winner_share = _share(winner_count, len(winners))
        loser_share = _share(loser_count, len(losers))
        separation = _difference(winner_share, loser_share)
        rows.append(
            {
                "feature": feature[0],
                "value": feature[1],
                "winner_count": winner_count,
                "loser_count": loser_count,
                "winner_share": winner_share,
                "loser_share": loser_share,
                "separation": separation,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            abs(float(row.get("separation") or 0.0)),
            int(row.get("winner_count") or 0) + int(row.get("loser_count") or 0),
            str(row.get("feature")),
            str(row.get("value")),
        ),
        reverse=True,
    )


def _low_score_feature_counts(records: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        for feature in set(_low_score_features(record)):
            counts[feature] += 1
    return counts


def _low_score_features(record: dict[str, Any]) -> list[tuple[str, str]]:
    signals = set(_candlestick_signal_values(record))
    features = [
        ("RSI", _rsi_bucket(record.get("rsi"))),
        ("volume_ratio", _volume_bucket(record.get("volume_ratio"))),
        ("market_regime", str(record.get("market_regime") or "unknown")),
        ("sector", str(record.get("sector") or "未分類")),
        ("volume_confirmed_breakout", "yes" if "volume_confirmed_breakout" in signals else "no"),
        ("long_lower_shadow_support", "yes" if "long_lower_shadow_support" in signals else "no"),
    ]
    features.extend(("candlestick_signal", signal) for signal in signals)
    return [(name, value) for name, value in features if value]


def _low_score_candidate_rescue_rules(
    rows: list[dict[str, Any]],
    winners: list[dict[str, Any]],
    losers: list[dict[str, Any]],
    low_score_min: float,
    low_score_max: float,
) -> list[dict[str, Any]]:
    positive_rows = [
        row for row in rows
        if _number(row.get("separation")) is not None
        and float(row.get("separation") or 0.0) > 0
        and int(row.get("winner_count") or 0) > 0
        and str(row.get("value") or "") not in {"no", "unknown", "no_signal"}
    ]
    rules = []
    rule_texts = set()
    volume_rule = _find_low_score_row(positive_rows, "volume_ratio", "3+")
    breakout_rule = _find_low_score_row(positive_rows, "volume_confirmed_breakout", "yes")
    if volume_rule and breakout_rule:
        combined = _combined_low_score_rule(winners, losers, low_score_min, low_score_max)
        rules.append(combined)
        rule_texts.add(str(combined["rule"]))

    for row in positive_rows:
        condition = _low_score_condition_text(row)
        if not condition:
            continue
        rule_text = f"total_score {_format_score_range_value(low_score_min)}-{_format_score_range_value(low_score_max)} でも {condition} なら採用候補"
        if rule_text in rule_texts:
            continue
        rule_texts.add(rule_text)
        rules.append(_low_score_rule_candidate(row, rule_text))
        if len(rules) >= 10:
            break
    return rules[:10]


def _find_low_score_row(rows: list[dict[str, Any]], feature: str, value: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("feature") == feature and row.get("value") == value:
            return row
    return None


def _combined_low_score_rule(
    winners: list[dict[str, Any]],
    losers: list[dict[str, Any]],
    low_score_min: float,
    low_score_max: float,
) -> dict[str, Any]:
    winner_count = _combined_low_score_count(winners)
    loser_count = _combined_low_score_count(losers)
    winner_share = _share(winner_count, len(winners))
    loser_share = _share(loser_count, len(losers))
    return {
        "rule": f"total_score {_format_score_range_value(low_score_min)}-{_format_score_range_value(low_score_max)} でも volume_ratio >= 3 かつ volume_confirmed_breakout なら採用候補",
        "feature": "combined",
        "value": "volume_ratio >= 3 AND volume_confirmed_breakout",
        "winner_count": winner_count,
        "loser_count": loser_count,
        "winner_share": winner_share,
        "loser_share": loser_share,
        "separation": _difference(winner_share, loser_share),
        "confidence": "candidate",
    }


def _format_score_range_value(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _combined_low_score_count(records: list[dict[str, Any]]) -> int:
    return sum(
        1 for record in records
        if (_number(record.get("volume_ratio")) or 0.0) >= 3
        and "volume_confirmed_breakout" in set(_candlestick_signal_values(record))
    )


def _low_score_rule_candidate(row: dict[str, Any], rule_text: str) -> dict[str, Any]:
    return {
        "rule": rule_text,
        "feature": row.get("feature"),
        "value": row.get("value"),
        "winner_count": row.get("winner_count"),
        "loser_count": row.get("loser_count"),
        "winner_share": row.get("winner_share"),
        "loser_share": row.get("loser_share"),
        "separation": row.get("separation"),
        "confidence": "candidate",
    }


def _low_score_condition_text(row: dict[str, Any]) -> str:
    feature = str(row.get("feature") or "")
    value = str(row.get("value") or "")
    if feature == "RSI":
        return _bucket_rule("RSI", value)
    if feature == "volume_ratio":
        return _bucket_rule("volume_ratio", value)
    if feature == "market_regime":
        return f"market_regime = {value}"
    if feature == "sector":
        return f"sector = {value}"
    if feature == "candlestick_signal":
        return f"candlestick_signal = {value}"
    if feature in {"volume_confirmed_breakout", "long_lower_shadow_support"} and value == "yes":
        return feature
    return ""


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
    if key == "relative_strength_score":
        return [_relative_strength_score_bucket(record.get("relative_strength_score"))]
    if key in {"relative_strength_5d", "relative_strength_10d", "relative_strength_20d"}:
        return [_relative_strength_bucket(record.get(key))]
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
    if feature == "relative_strength_score":
        return _bucket_rule("relative_strength_score", value)
    if feature in {"relative_strength_5d", "relative_strength_10d", "relative_strength_20d"}:
        return _relative_strength_rule(feature, value)
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


def _relative_strength_rule(feature: str, bucket: str) -> str:
    mapping = {
        "< -5%": f"{feature} < -5%",
        "-5% to 0%": f"{feature} >= -5% and {feature} < 0%",
        "0% to 3%": f"{feature} >= 0% and {feature} < 3%",
        "3% to 5%": f"{feature} >= 3% and {feature} < 5%",
        "5% to 10%": f"{feature} >= 5% and {feature} < 10%",
        "10%+": f"{feature} >= 10%",
    }
    return mapping.get(bucket, "")


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
        ("relative_strength_score", _relative_strength_score_bucket(record.get("relative_strength_score"))),
        ("relative_strength_5d", _relative_strength_bucket(record.get("relative_strength_5d"))),
        ("relative_strength_10d", _relative_strength_bucket(record.get("relative_strength_10d"))),
        ("relative_strength_20d", _relative_strength_bucket(record.get("relative_strength_20d"))),
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


def _sector_lift_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('sector')}: "
            f"selected avg10d {_format_percent(item.get('selected_avg10d'))}, "
            f"rejected avg10d {_format_percent(item.get('rejected_avg10d'))}, "
            f"lift {_format_signed_percent(item.get('lift'))}, "
            f"selected count {item.get('selected_count')}, "
            f"rejected count {item.get('rejected_count')}, "
            f"confidence {item.get('confidence')}"
            f"{_sample_note_suffix(item)}"
        )
        for item in items
    ]


def _sector_filter_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('filter')}: "
            f"selected avg10d {_format_percent(item.get('selected_avg10d'))}, "
            f"rejected avg10d {_format_percent(item.get('rejected_avg10d'))}, "
            f"lift {_format_signed_percent(item.get('lift'))}, "
            f"selected count {item.get('selected_count')}, "
            f"rejected count {item.get('rejected_count')}, "
            f"confidence {item.get('confidence')}"
            f"{_sample_note_suffix(item)}"
        )
        for item in items
    ]


def _low_score_deep_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    score_range = analysis.get("score_range", {})
    score_range_text = (
        f"{_format_score_range_value(float(score_range.get('min', LOW_SCORE_MIN)))}-"
        f"{_format_score_range_value(float(score_range.get('max', LOW_SCORE_MAX)))}"
    )
    return [
        f"- target_score_range: {score_range_text}",
        f"- low_score_count: {analysis.get('low_score_count', 0)}",
        f"- winner_count: {analysis.get('winner_count', 0)}",
        f"- loser_count: {analysis.get('loser_count', 0)}",
        "",
        f"### Winners in {score_range_text}",
        "",
        *_low_score_record_lines(analysis.get("winners", [])),
        "",
        f"### Losers in {score_range_text}",
        "",
        *_low_score_record_lines(analysis.get("losers", [])),
        "",
        "### Features with highest separation",
        "",
        *_low_score_separation_lines(analysis.get("feature_separation", [])),
        "",
        "### Candidate Rescue Rules",
        "",
        *_low_score_rule_lines(analysis.get("candidate_rescue_rules", [])),
    ]


def _low_score_record_lines(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return ["- データなし"]
    return [
        (
            f"- {record.get('date')} {record.get('code')} {record.get('name')}: "
            f"score {_format_number(record.get('total_score'))}, "
            f"10d {_format_percent(record.get('return_10d'))}, "
            f"RSI {_format_number(record.get('rsi'))}, "
            f"volume_ratio {_format_number(record.get('volume_ratio'))}, "
            f"market_regime {record.get('market_regime')}, "
            f"sector {record.get('sector')}, "
            f"signals {_signal_text(record.get('candlestick_signals', []))}"
        )
        for record in records
    ]


def _low_score_separation_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('feature')}={item.get('value')}: "
            f"winners {_format_percent(item.get('winner_share'))} ({item.get('winner_count')}件), "
            f"losers {_format_percent(item.get('loser_share'))} ({item.get('loser_count')}件), "
            f"separation {_format_signed_percent(item.get('separation'))}"
        )
        for item in items
    ]


def _low_score_rule_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('rule')}: "
            f"winners {_format_percent(item.get('winner_share'))} ({item.get('winner_count')}件), "
            f"losers {_format_percent(item.get('loser_share'))} ({item.get('loser_count')}件), "
            f"separation {_format_signed_percent(item.get('separation'))}, "
            f"confidence {item.get('confidence')}"
        )
        for item in items
    ]


def _earnings_filter_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    return [
        f"- rejected_by_earnings_filter_count: {analysis.get('rejected_by_earnings_filter_count', 0)}",
        f"- rejected_by_earnings_filter_avg5d: {_format_percent(analysis.get('rejected_by_earnings_filter_avg5d'))}",
        f"- rejected_by_earnings_filter_avg10d: {_format_percent(analysis.get('rejected_by_earnings_filter_avg10d'))}",
        f"- earnings_filter_effectiveness: {analysis.get('earnings_filter_effectiveness') or 'N/A'}",
    ]


def _relative_strength_selection_quality_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    lines = []
    for field in ["relative_strength_5d", "relative_strength_10d", "relative_strength_20d", "relative_strength_score"]:
        item = analysis.get(field, {})
        lines.append(f"- selected平均 {field}: {_format_number(item.get('selected_average'))}")
        lines.append(f"- rejected平均 {field}: {_format_number(item.get('rejected_average'))}")
    return lines


def _investor_context_selection_quality_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    lines = []
    for field in ["investor_context_score", "overseas_net_buy_4w_sum"]:
        item = analysis.get(field, {})
        lines.append(f"- selected平均 {field}: {_format_number(item.get('selected_average'))}")
        lines.append(f"- rejected平均 {field}: {_format_number(item.get('rejected_average'))}")
    return lines


def _signal_text(signals: Any) -> str:
    values = [str(signal) for signal in signals if signal]
    return ", ".join(values) if values else "no_signal"


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
    if number < 40:
        return "<40"
    if number < 45:
        return "40-45"
    if number < 50:
        return "45-50"
    if number < 55:
        return "50-55"
    if number < 60:
        return "55-60"
    if number < 65:
        return "60-65"
    if number < 70:
        return "65-70"
    if number < 75:
        return "70-75"
    if number < 80:
        return "75-80"
    return "80+"


def _relative_strength_score_bucket(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    if number <= 0:
        return "0"
    if number <= 3:
        return "1-3"
    if number <= 6:
        return "4-6"
    if number < 10:
        return "7-9"
    return "10"


def _relative_strength_bucket(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    if number < -0.05:
        return "< -5%"
    if number < 0:
        return "-5% to 0%"
    if number < 0.03:
        return "0% to 3%"
    if number < 0.05:
        return "3% to 5%"
    if number < 0.10:
        return "5% to 10%"
    return "10%+"


def _profile_id(config: dict[str, Any]) -> str:
    return str(config.get("profile_id") or config.get("dealer", {}).get("id") or "rookie_dealer_01")


def _profile_name(config: dict[str, Any]) -> str:
    return str(config.get("profile_name") or config.get("dealer", {}).get("name") or _profile_id(config))
