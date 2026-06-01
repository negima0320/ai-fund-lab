"""Feature contribution analysis for closed paper trades."""

from __future__ import annotations

import json
import csv
import sqlite3
import time
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
PROFIT_RATE_BUCKET_ORDER = [
    "<= -10%",
    "-10% to -5%",
    "-5% to -3%",
    "-3% to -1%",
    "-1% to 0%",
    "0% to 1%",
    "1% to 3%",
    "3% to 5%",
    "5% to 10%",
    "10%+",
]
HOLDING_PERIOD_BUCKET_ORDER = ["1 day", "2 days", "3 days", "4-5 days", "6-10 days", "11+ days"]
TRADE_SCORE_BUCKET_ORDER = ["<40", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70+"]
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
    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    def timed(key: str, action: Any) -> Any:
        started = time.perf_counter()
        try:
            return action()
        finally:
            timings[key] = round(timings.get(key, 0.0) + time.perf_counter() - started, 6)

    db_path = get_database_path(config, root)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    profile_id = _profile_id(config)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        baseline_profile_id = _baseline_profile_id(root, profile_id)
        trade_where = "profile_id = ?"
        trade_params: list[Any] = [profile_id]
        if start_date and end_date:
            trade_where += " AND entry_date BETWEEN ? AND ?"
            trade_params.extend([start_date, end_date])
        trade_rows = timed("feature_analysis_load_logs_sec", lambda: _rows(
            connection,
            f"""
            SELECT *
            FROM trades
            WHERE {trade_where}
            ORDER BY entry_date, exit_date, id
            """,
            tuple(trade_params),
        ))
        baseline_trade_rows: list[dict[str, Any]] = []
        if baseline_profile_id and baseline_profile_id != profile_id:
            baseline_where = "profile_id = ?"
            baseline_params: list[Any] = [baseline_profile_id]
            if start_date and end_date:
                baseline_where += " AND entry_date BETWEEN ? AND ?"
                baseline_params.extend([start_date, end_date])
            baseline_trade_rows = timed("feature_analysis_load_logs_sec", lambda: _rows(
                connection,
                f"""
                SELECT *
                FROM trades
                WHERE {baseline_where}
                ORDER BY entry_date, exit_date, id
                """,
                tuple(baseline_params),
            ))
        market_where = "profile_id = ?"
        market_params: list[Any] = [profile_id]
        if start_date and end_date:
            market_where += " AND date BETWEEN ? AND ?"
            market_params.extend([start_date, end_date])
        market_rows = timed("feature_analysis_load_logs_sec", lambda: _rows(
            connection,
            f"""
            SELECT date, market_regime, advance_ratio
            FROM market_contexts
            WHERE {market_where}
            ORDER BY date, id
            """,
            tuple(market_params),
        ))
        scoring_where = "profile_id = ?"
        scoring_params: list[Any] = [profile_id]
        if start_date and end_date:
            scoring_where += " AND date BETWEEN ? AND ?"
            scoring_params.extend([start_date, end_date])
        scoring_rows = timed("feature_analysis_load_processed_sec", lambda: _rows(
            connection,
            f"""
            SELECT *
            FROM scoring_results
            WHERE {scoring_where}
            ORDER BY date, rank, id
            """,
            tuple(scoring_params),
        ))
    closed = timed("feature_analysis_result_integrity_sec", lambda: [row for row in trade_rows if is_closed_trade_for_metrics(row)])
    baseline_closed = timed("feature_analysis_result_integrity_sec", lambda: [row for row in baseline_trade_rows if is_closed_trade_for_metrics(row)])
    market_by_date = timed("feature_analysis_load_processed_sec", lambda: {row.get("date"): row for row in market_rows})
    records = timed("feature_analysis_result_integrity_sec", lambda: [_feature_record(trade, market_by_date.get(trade.get("entry_date"), {})) for trade in closed])
    baseline_records = timed("feature_analysis_result_integrity_sec", lambda: [_feature_record(trade, {}) for trade in baseline_closed])
    backtest_summary = timed("feature_analysis_load_logs_sec", lambda: _backtest_summary_payload(root, profile_id, start_date, end_date))
    if backtest_summary:
        records = timed("feature_analysis_result_integrity_sec", lambda: _feature_records_from_backtest_summary(backtest_summary))
    baseline_backtest_summary: dict[str, Any] = {}
    if baseline_profile_id and baseline_profile_id != profile_id:
        baseline_backtest_summary = timed("feature_analysis_load_logs_sec", lambda: _backtest_summary_payload(root, baseline_profile_id, start_date, end_date))
        if baseline_backtest_summary:
            baseline_records = timed("feature_analysis_result_integrity_sec", lambda: _feature_records_from_backtest_summary(baseline_backtest_summary))
    rsi_filter = _rsi_filter_rejection_summary(scoring_rows, config)
    component_validation = timed("feature_analysis_score_integrity_sec", lambda: _score_component_validation(records, scoring_rows))
    score_formula_audit = timed("feature_analysis_score_integrity_sec", lambda: _score_formula_audit(config, records, scoring_rows, component_validation))
    score_effective_range_audit = timed("feature_analysis_score_integrity_sec", lambda: _score_effective_range_audit(config, records, scoring_rows))
    earnings_exposure = timed("feature_analysis_earnings_filter_sec", lambda: _earnings_calendar_exposure(records, scoring_rows))
    earnings_filter_debug = timed("feature_analysis_earnings_filter_sec", lambda: _earnings_filter_debug(config, scoring_rows))
    earnings_pipeline = timed("feature_analysis_earnings_filter_sec", lambda: _earnings_pipeline(scoring_rows, earnings_filter_debug))
    feature_activation_audit = build_feature_activation_audit(config, records, scoring_rows, _registry_features(root, profile_id))
    relative_strength_debug = timed("feature_analysis_relative_strength_sec", lambda: _relative_strength_debug(scoring_rows))
    relative_strength_pipeline = timed("feature_analysis_relative_strength_sec", lambda: _relative_strength_pipeline(config, scoring_rows))
    investor_context_filter = timed("feature_analysis_investor_context_sec", lambda: _investor_context_filter_analysis(scoring_rows, records))
    integrity_audits = timed("feature_analysis_market_filter_audit_sec", lambda: _backtest_integrity_audits(root, profile_id, start_date, end_date))
    capital_utilization_audit = timed(
        "feature_analysis_result_integrity_sec",
        lambda: _capital_utilization_audit(root, profile_id, start_date, end_date, backtest_summary, config),
    )
    api_field_usage_audit = timed(
        "feature_analysis_score_component_sec",
        lambda: _api_field_usage_audit(config, root, profile_id, start_date, end_date, scoring_rows),
    )
    volume_filter_audit = timed(
        "feature_analysis_score_component_sec",
        lambda: _volume_filter_audit(config, root, profile_id, start_date, end_date, scoring_rows, records, baseline_records, baseline_profile_id),
    )
    rsi_volume_hot_zone_audit = timed(
        "feature_analysis_score_component_sec",
        lambda: _rsi_volume_hot_zone_audit(config, scoring_rows, records),
    )

    relative_strength_analysis = timed(
        "feature_analysis_relative_strength_sec",
        lambda: {
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
    )
    relative_strength_effect_analysis = timed("feature_analysis_relative_strength_sec", lambda: _relative_strength_effect_analysis(records, baseline_records, baseline_profile_id))
    trade_profit_distribution_analysis = timed("feature_analysis_score_component_sec", lambda: _trade_profit_distribution_analysis(records, backtest_summary))
    exit_reason_analysis = timed("feature_analysis_score_component_sec", lambda: _exit_reason_analysis(records))
    profit_concentration_analysis = timed("feature_analysis_score_component_sec", lambda: _profit_concentration_analysis(records))
    holding_period_analysis = timed("feature_analysis_score_component_sec", lambda: _holding_period_analysis(records))
    score_vs_profit_analysis = timed("feature_analysis_score_component_sec", lambda: _score_vs_profit_analysis(records))
    baseline_vs_relative_strength_trade_analysis = timed(
        "feature_analysis_relative_strength_sec",
        lambda: _baseline_vs_relative_strength_trade_analysis(records, baseline_records, baseline_profile_id),
    )
    expectancy_formula_audit = timed("feature_analysis_score_component_sec", lambda: _expectancy_formula_audit(records, backtest_summary))
    profit_consistency_audit = timed(
        "feature_analysis_score_component_sec",
        lambda: _profit_consistency_audit(root, profile_id, start_date, end_date, records, backtest_summary, trade_rows),
    )
    trade_expectancy_deep_analysis = timed("feature_analysis_score_component_sec", lambda: _trade_expectancy_deep_analysis(records, backtest_summary))
    exit_reason_profit_analysis = timed("feature_analysis_score_component_sec", lambda: _exit_reason_profit_analysis(records))
    profit_capture_analysis = timed("feature_analysis_score_component_sec", lambda: _profit_capture_analysis(records))
    opportunity_loss_analysis = timed(
        "feature_analysis_score_component_sec",
        lambda: _opportunity_loss_analysis(root, profile_id, records, start_date, end_date),
    )
    stop_loss_pattern_analysis = timed("feature_analysis_score_component_sec", lambda: _stop_loss_pattern_analysis(records))
    max_holding_pattern_analysis = timed("feature_analysis_score_component_sec", lambda: _max_holding_pattern_analysis(records))
    winner_vs_stop_loss_contrast = timed("feature_analysis_score_component_sec", lambda: _winner_vs_stop_loss_contrast(records))
    rule_candidate_proposal = timed(
        "feature_analysis_score_component_sec",
        lambda: _rule_candidate_proposal(records, stop_loss_pattern_analysis, max_holding_pattern_analysis, winner_vs_stop_loss_contrast),
    )
    profit_analysis_conclusion = timed(
        "feature_analysis_score_component_sec",
        lambda: _profit_analysis_conclusion(
            trade_expectancy_deep_analysis,
            exit_reason_profit_analysis,
            profit_capture_analysis,
            opportunity_loss_analysis,
            profit_consistency_audit,
        ),
    )
    investor_context_analysis = timed(
        "feature_analysis_investor_context_sec",
        lambda: {
            "investor_context_score": _group_by(
                records,
                lambda item: _investor_context_score_bucket(item.get("investor_context_score")),
                ["<0", "0", "1-2", "3-5"],
            ),
            "overseas_net_buy_4w_sum": _group_by(records, lambda item: _positive_negative_bucket(item.get("overseas_net_buy_4w_sum")), ["positive", "negative", "zero", "unknown"]),
            "overseas_net_buy_4w_trend": _group_by(records, lambda item: item.get("overseas_net_buy_4w_trend") or "unknown", ["improving", "worsening", "flat", "unknown"]),
            "investor_context_source": _group_by(records, lambda item: item.get("investor_context_source") or "unknown"),
            "top_candidates": _top_investor_context_candidates(records),
            "effect_analysis": _investor_context_effect_analysis(records),
        },
    )
    score_contribution = timed(
        "feature_analysis_score_component_sec",
        lambda: {
            "selected_score_averages": _selected_score_averages(records),
            "technical_score": _group_by(records, lambda item: _component_score_bucket(item.get("technical_score")), COMPONENT_SCORE_BUCKET_ORDER),
        },
    )
    score_component_analysis = timed(
        "feature_analysis_score_component_sec",
        lambda: {
            "score_components_validation": component_validation,
            "rsi_score": _group_by(records, lambda item: _score_component_bucket(item.get("rsi_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "volume_score": _group_by(records, lambda item: _score_component_bucket(item.get("volume_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "candlestick_score": _group_by(records, lambda item: _score_component_bucket(item.get("candlestick_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "market_context_score": _group_by(records, lambda item: _score_component_bucket(item.get("market_context_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "relative_strength_score": _group_by(records, lambda item: _score_component_bucket(item.get("relative_strength_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "investor_context_score": _group_by(records, lambda item: _score_component_bucket(item.get("investor_context_score")), COMPONENT_DETAIL_BUCKET_ORDER),
            "penalty_score": _group_by(records, lambda item: _score_component_bucket(item.get("penalty_score")), COMPONENT_DETAIL_BUCKET_ORDER),
        },
    )
    timings["feature_analysis_total"] = round(time.perf_counter() - total_started, 6)
    measured_without_total = sum(value for key, value in timings.items() if key != "feature_analysis_total")
    residual = max(0.0, timings["feature_analysis_total"] - measured_without_total)
    timings["feature_analysis_score_component_sec"] = round(timings.get("feature_analysis_score_component_sec", 0.0) + residual, 6)
    timings.setdefault("feature_analysis_load_reports_sec", 0.0)
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
        "score_contribution": score_contribution,
        "score_component_analysis": score_component_analysis,
        "score_formula_audit": score_formula_audit,
        "score_effective_range_audit": score_effective_range_audit,
        "feature_activation_audit": feature_activation_audit,
        "api_field_usage_audit": api_field_usage_audit,
        "volume_filter_audit": volume_filter_audit,
        "rsi_volume_hot_zone_audit": rsi_volume_hot_zone_audit,
        "relative_strength_pipeline": relative_strength_pipeline,
        "relative_strength_debug": relative_strength_debug,
        "earnings_calendar_exposure": earnings_exposure,
        "earnings_filter_debug": earnings_filter_debug,
        "earnings_pipeline": earnings_pipeline,
        "relative_strength_analysis": relative_strength_analysis,
        "relative_strength_effect_analysis": relative_strength_effect_analysis,
        "trade_profit_distribution_analysis": trade_profit_distribution_analysis,
        "exit_reason_analysis": exit_reason_analysis,
        "profit_concentration_analysis": profit_concentration_analysis,
        "holding_period_analysis": holding_period_analysis,
        "score_vs_profit_analysis": score_vs_profit_analysis,
        "baseline_vs_relative_strength_trade_analysis": baseline_vs_relative_strength_trade_analysis,
        "expectancy_formula_audit": expectancy_formula_audit,
        "profit_consistency_audit": profit_consistency_audit,
        "trade_expectancy_deep_analysis": trade_expectancy_deep_analysis,
        "exit_reason_profit_analysis": exit_reason_profit_analysis,
        "profit_capture_analysis": profit_capture_analysis,
        "opportunity_loss_analysis": opportunity_loss_analysis,
        "stop_loss_pattern_analysis": stop_loss_pattern_analysis,
        "max_holding_pattern_analysis": max_holding_pattern_analysis,
        "winner_vs_stop_loss_contrast": winner_vs_stop_loss_contrast,
        "rule_candidate_proposal": rule_candidate_proposal,
        "profit_analysis_conclusion": profit_analysis_conclusion,
        "investor_context_analysis": investor_context_analysis,
        "investor_context_filter": investor_context_filter,
        "capital_utilization_audit": capital_utilization_audit,
        "market_filter_audit": integrity_audits.get("market_filter_audit", {}),
        "backtest_result_integrity_audit": integrity_audits.get("backtest_result_integrity_audit", {}),
        "score_integrity_audit": integrity_audits.get("score_integrity_audit", {}),
        "feature_analysis_performance": timings,
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
        "## API Field Usage Audit",
        "",
        *_api_field_usage_audit_lines(analysis.get("api_field_usage_audit", {})),
        "",
        "## Volume Filter Audit",
        "",
        *_volume_filter_audit_lines(analysis.get("volume_filter_audit", {})),
        "",
        "## RSI Volume Hot Zone Audit",
        "",
        *_rsi_volume_hot_zone_audit_lines(analysis.get("rsi_volume_hot_zone_audit", {})),
        "",
        "## Capital Utilization Audit",
        "",
        *_capital_utilization_audit_lines(analysis.get("capital_utilization_audit", {})),
        "",
        "## Backtest Result Integrity Audit",
        "",
        *_generic_audit_lines(analysis.get("backtest_result_integrity_audit", {})),
        "",
        "## Market Filter Audit",
        "",
        *_generic_audit_lines(analysis.get("market_filter_audit", {})),
        "",
        "## Score Integrity Audit",
        "",
        *_generic_audit_lines(analysis.get("score_integrity_audit", {})),
        "",
        "## Performance Audit",
        "",
        *_performance_audit_lines(analysis.get("performance_audit", {})),
        "",
        "## JSON Read Ranking",
        "",
        *_json_read_ranking_lines(analysis.get("json_read_ranking") or (analysis.get("performance_audit", {}) or {}).get("json_read_ranking") or (analysis.get("performance_audit", {}) or {}).get("top_json_read_files") or []),
        "",
        "## Profile Read Reason",
        "",
        *_profile_read_reason_lines(analysis.get("profile_read_reason", {})),
        "",
        "## Indicator Field Audit",
        "",
        *_indicator_field_audit_lines(analysis.get("indicator_field_audit") or (analysis.get("performance_audit", {}) or {}).get("indicator_field_audit") or {}),
        "",
        "## Runtime Memory Cache Audit",
        "",
        *_runtime_memory_cache_audit_lines((analysis.get("performance_audit", {}) or {}).get("runtime_memory_cache_audit", {})),
        "",
        "## Relative Strength Analysis",
        "",
        *_relative_strength_analysis_lines(analysis.get("relative_strength_analysis", {})),
        "",
        "## Relative Strength Effect Analysis",
        "",
        *_relative_strength_effect_analysis_lines(analysis.get("relative_strength_effect_analysis", {})),
        "",
        "## Trade Profit Distribution Analysis",
        "",
        *_trade_profit_distribution_analysis_lines(analysis.get("trade_profit_distribution_analysis", {})),
        "",
        "## Exit Reason Analysis",
        "",
        *_exit_reason_analysis_table_lines(analysis.get("exit_reason_analysis", [])),
        "",
        "## Profit Concentration Analysis",
        "",
        *_profit_concentration_analysis_lines(analysis.get("profit_concentration_analysis", {})),
        "",
        "## Profit Consistency Audit",
        "",
        *_profit_consistency_audit_lines(analysis.get("profit_consistency_audit", {})),
        "",
        "## Trade Expectancy Deep Analysis",
        "",
        *_trade_expectancy_deep_analysis_lines(analysis.get("trade_expectancy_deep_analysis", {})),
        "",
        "## Exit Reason Profit Analysis",
        "",
        *_exit_reason_profit_analysis_lines(analysis.get("exit_reason_profit_analysis", [])),
        "",
        "## Profit Capture Analysis",
        "",
        *_profit_capture_analysis_lines(analysis.get("profit_capture_analysis", {})),
        "",
        "## Opportunity Loss Analysis",
        "",
        *_opportunity_loss_analysis_lines(analysis.get("opportunity_loss_analysis", {})),
        "",
        "## Stop Loss Pattern Analysis",
        "",
        *_stop_loss_pattern_analysis_lines(analysis.get("stop_loss_pattern_analysis", {})),
        "",
        "## Max Holding Pattern Analysis",
        "",
        *_max_holding_pattern_analysis_lines(analysis.get("max_holding_pattern_analysis", {})),
        "",
        "## Winner vs Stop Loss Contrast",
        "",
        *_winner_vs_stop_loss_contrast_lines(analysis.get("winner_vs_stop_loss_contrast", [])),
        "",
        "## Rule Candidate Proposal",
        "",
        *_rule_candidate_proposal_lines(analysis.get("rule_candidate_proposal", [])),
        "",
        "## Conclusion",
        "",
        *_profit_analysis_conclusion_lines(analysis.get("profit_analysis_conclusion", [])),
        "",
        "## Holding Period Analysis",
        "",
        *_effect_table_lines(analysis.get("holding_period_analysis", [])),
        "",
        "## Score vs Profit Analysis",
        "",
        *_effect_table_lines(analysis.get("score_vs_profit_analysis", [])),
        "",
        "## Baseline vs Relative Strength Trade Analysis",
        "",
        *_baseline_vs_relative_strength_trade_analysis_lines(analysis.get("baseline_vs_relative_strength_trade_analysis", {})),
        "",
        "## Expectancy Formula Audit",
        "",
        *_expectancy_formula_audit_lines(analysis.get("expectancy_formula_audit", {})),
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
        "## Investor Context Filter",
        "",
        *_investor_context_filter_lines(analysis.get("investor_context_filter", {})),
        "",
        "## Earnings Calendar Exposure",
        "",
        *_earnings_calendar_exposure_lines(analysis.get("earnings_calendar_exposure", {})),
        "",
        "## Earnings Filter Debug",
        "",
        *_earnings_filter_debug_lines(analysis.get("earnings_filter_debug", {})),
        "",
        "## Earnings Pipeline",
        "",
        *_earnings_pipeline_lines(analysis.get("earnings_pipeline", {})),
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
    investor_context_score_raw = _number(trade.get("investor_context_score"))
    investor_context_score_component = _component_value(trade, score_components, "investor_context_score")
    return {
        "trade_id": trade.get("trade_id"),
        "code": trade.get("code"),
        "name": trade.get("name"),
        "date": trade.get("signal_date") or trade.get("entry_date"),
        "signal_date": trade.get("signal_date"),
        "entry_date": trade.get("entry_date"),
        "exit_date": trade.get("exit_date"),
        "exit_reason": trade.get("exit_reason"),
        "result": trade.get("result"),
        "profit": profit,
        "profit_rate": profit_rate,
        "gross_profit": _number(trade.get("gross_profit")),
        "gross_profit_rate": _number(trade.get("gross_profit_rate")),
        "net_profit": _number(trade.get("net_profit")),
        "net_profit_rate": _number(trade.get("net_profit_rate")),
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
        "investor_context_score": investor_context_score_raw,
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
        "investor_context_score": investor_context_score_raw if investor_context_score_raw is not None else investor_context_score_component,
        "investor_context_component_score": investor_context_score_component,
        "selected": bool(trade.get("selected", True)),
        "entry_price": _number(trade.get("entry_price")),
        "exit_price": _number(trade.get("exit_price") or trade.get("actual_exit_price")),
        "holding_days": _number(trade.get("holding_days")),
        "sector_score": _component_value(trade, score_components, "sector_score"),
        "penalty_score": _component_value(trade, score_components, "penalty_score"),
        "score_components_total": _number(trade.get("score_components_total")) or _number(score_components.get("component_total")),
        "score_components_match": _bool_or_none(trade.get("score_components_match"), score_components.get("matches_total_score")),
    }


def _backtest_summary_payload(root: Path, profile_id: str, start_date: str | None, end_date: str | None) -> dict[str, Any]:
    if not start_date or not end_date:
        return {}
    path = root / "logs" / "backtests" / profile_id / f"{start_date}_to_{end_date}" / "backtest_summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _feature_records_from_backtest_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("all_trades", [])
    if not isinstance(rows, list):
        rows = []
    return [_feature_record(row, {}) for row in rows if isinstance(row, dict) and is_closed_trade_for_metrics(row)]


def _trade_profit_distribution_analysis(records: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    profits = [_record_profit(record) or 0.0 for record in records]
    gross_profits = [_number(record.get("gross_profit")) or 0.0 for record in records]
    net_profits = [_number(record.get("net_profit")) or _record_profit(record) or 0.0 for record in records]
    profit_rates = [_record_profit_rate(record) for record in records]
    profit_rates = [value for value in profit_rates if value is not None]
    holding_days = [_number(record.get("holding_days")) for record in records]
    holding_days = [value for value in holding_days if value is not None]
    sorted_by_profit = sorted(records, key=lambda item: _record_profit(item) or 0.0, reverse=True)
    sorted_by_loss = sorted(records, key=lambda item: _record_profit(item) or 0.0)
    return {
        "total_trades": int(summary.get("total_trades") or len(records) or 0),
        "closed_trade_count": int(summary.get("closed_trade_count") or summary.get("closed_trades_count") or len(records) or 0),
        "avg_profit": _average(profits),
        "avg_profit_rate": _average(profit_rates),
        "median_profit_rate": _median(profit_rates),
        "win_rate": _win_rate(records),
        "profit_factor": _number(summary.get("profit_factor")) or _profit_factor(records),
        "computed_trade_profit_factor": _profit_factor(records),
        "gross_profit_sum": round(sum(gross_profits), 2),
        "net_profit_sum": round(sum(net_profits), 2),
        "summary_net_cumulative_profit": _number(summary.get("net_cumulative_profit")),
        "summary_gross_cumulative_profit": _number(summary.get("gross_cumulative_profit")),
        "average_holding_days": _average(holding_days),
        "median_holding_days": _median(holding_days),
        "max_win_profit_rate": _max_or_none([value for value in profit_rates if value > 0]),
        "max_loss_profit_rate": _min_or_none([value for value in profit_rates if value < 0]),
        "largest_win_profit": _record_profit(sorted_by_profit[0]) if sorted_by_profit else None,
        "largest_loss_profit": _record_profit(sorted_by_loss[0]) if sorted_by_loss else None,
        "histogram": _group_by(records, lambda item: _profit_rate_bucket(_record_profit_rate(item)), PROFIT_RATE_BUCKET_ORDER),
        "top_winners": [_trade_table_record(row) for row in sorted_by_profit[:20]],
        "top_losers": [_trade_table_record(row) for row in sorted_by_loss[:20]],
    }


def _exit_reason_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("exit_reason") or record.get("result") or "unknown")].append(record)
    result = []
    for reason, items in sorted(groups.items()):
        stats = _group_stats(reason, items)
        stats["reason"] = reason
        stats["avg_holding_days"] = _average(_valid_numbers(item.get("holding_days") for item in items))
        result.append(stats)
    return result


def _profit_concentration_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    profits = sorted([_record_profit(record) or 0.0 for record in records], reverse=True)
    losses = sorted([value for value in profits if value < 0])
    net_profit = round(sum(profits), 2)

    def top_sum(count: int) -> float:
        return round(sum(profits[:count]), 2)

    def bottom_sum(count: int) -> float:
        return round(sum(losses[:count]), 2)

    return {
        "net_profit": net_profit,
        "top_5_profit_sum": top_sum(5),
        "top_10_profit_sum": top_sum(10),
        "top_20_profit_sum": top_sum(20),
        "bottom_5_loss_sum": bottom_sum(5),
        "bottom_10_loss_sum": bottom_sum(10),
        "bottom_20_loss_sum": bottom_sum(20),
        "net_profit_without_top_5": round(net_profit - top_sum(5), 2),
        "net_profit_without_top_10": round(net_profit - top_sum(10), 2),
        "net_profit_without_top_20": round(net_profit - top_sum(20), 2),
        "net_profit_without_bottom_5": round(net_profit - bottom_sum(5), 2),
        "net_profit_without_bottom_10": round(net_profit - bottom_sum(10), 2),
        "net_profit_without_bottom_20": round(net_profit - bottom_sum(20), 2),
    }


def _holding_period_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _group_by(records, lambda item: _holding_period_bucket(item.get("holding_days")), HOLDING_PERIOD_BUCKET_ORDER)


def _score_vs_profit_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _group_by(records, lambda item: _trade_score_bucket(item.get("total_score")), TRADE_SCORE_BUCKET_ORDER)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_trade_score_bucket(record.get("total_score"))].append(record)
    for row in rows:
        row["profit_factor"] = _profit_factor(groups.get(row["bucket"], []))
    return rows


def _baseline_vs_relative_strength_trade_analysis(
    records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    baseline_profile_id: str | None,
) -> dict[str, Any]:
    target_by_key = {_trade_key(record): record for record in records if _trade_key(record)}
    base_by_key = {_trade_key(record): record for record in baseline_records if _trade_key(record)}
    newly_selected = [target_by_key[key] for key in sorted(set(target_by_key) - set(base_by_key))]
    removed = [base_by_key[key] for key in sorted(set(base_by_key) - set(target_by_key))]
    common = [target_by_key[key] for key in sorted(set(target_by_key) & set(base_by_key))]
    return {
        "baseline_profile_id": baseline_profile_id,
        "target_trade_count": len(records),
        "baseline_trade_count": len(baseline_records),
        "common_trade_count": len(common),
        "newly_selected_count": len(newly_selected),
        "removed_count": len(removed),
        "newly_selected_summary": _trade_subset_summary(newly_selected),
        "removed_if_kept_summary": _trade_subset_summary(removed),
        "common_trade_summary": _trade_subset_summary(common),
        "improvement_type": _relative_strength_improvement_type(newly_selected, removed),
        "newly_selected_histogram": _group_by(newly_selected, lambda item: _profit_rate_bucket(_record_profit_rate(item)), PROFIT_RATE_BUCKET_ORDER),
        "removed_histogram": _group_by(removed, lambda item: _profit_rate_bucket(_record_profit_rate(item)), PROFIT_RATE_BUCKET_ORDER),
    }


def _expectancy_formula_audit(records: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    win_records = [record for record in records if (_record_profit(record) or 0.0) > 0]
    loss_records = [record for record in records if (_record_profit(record) or 0.0) <= 0]
    win_rate = _win_rate(records)
    avg_win_rate = _average([_record_profit_rate(record) for record in win_records if _record_profit_rate(record) is not None])
    avg_loss_rate = _average([_record_profit_rate(record) for record in loss_records if _record_profit_rate(record) is not None])
    expectancy = None
    if win_rate is not None:
        expectancy = round((win_rate * (avg_win_rate or 0.0)) + ((1 - win_rate) * (avg_loss_rate or 0.0)), 4)
    net_profit = _number(summary.get("net_cumulative_profit"))
    total_trades = _number(summary.get("total_trades")) or len(records)
    avg_profit_values = [_record_profit(record) or 0.0 for record in records]
    avg_profit_rates = [_record_profit_rate(record) for record in records if _record_profit_rate(record) is not None]
    warnings = []
    if net_profit is not None and net_profit > 0 and expectancy is not None and expectancy < 0:
        warnings.append("expectancy is rate-based while net profit is yen-based; position sizing/taxes can make signs differ")
    stored_expectancy = _number(summary.get("expectancy"))
    if stored_expectancy is not None and expectancy is not None and abs(stored_expectancy - expectancy) > 0.0002:
        warnings.append("stored expectancy differs from recomputed rate expectancy")
    return {
        "expectancy_formula": "win_rate * average_win_profit_rate + (1 - win_rate) * average_loss_profit_rate",
        "expectancy_value": expectancy,
        "stored_expectancy": stored_expectancy,
        "avg_profit_per_trade": _average(avg_profit_values),
        "avg_profit_rate_per_trade": _average(avg_profit_rates),
        "net_profit_per_trade": round(net_profit / total_trades, 4) if net_profit is not None and total_trades else None,
        "net_profit": net_profit,
        "total_trades": int(total_trades or 0),
        "win_rate": win_rate,
        "average_win_profit_rate": avg_win_rate,
        "average_loss_profit_rate": avg_loss_rate,
        "warnings": warnings,
    }


def _profit_consistency_audit(
    root: Path,
    profile_id: str,
    start_date: str | None,
    end_date: str | None,
    records: list[dict[str, Any]],
    backtest_summary: dict[str, Any],
    sqlite_trade_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    range_key = f"{start_date}_to_{end_date}" if start_date and end_date else ""
    backtest_dir = root / "logs" / "backtests" / profile_id / range_key if range_key else root / "logs" / "backtests" / profile_id
    summary_csv_rows = _read_csv_rows(backtest_dir / "summary.csv")
    trades_csv_rows = _read_csv_rows(backtest_dir / "trades.csv")
    summary_csv_row = summary_csv_rows[-1] if summary_csv_rows else {}
    initial_capital = _number(backtest_summary.get("initial_capital")) or 0.0
    final_assets = _number(backtest_summary.get("final_assets"))
    mark_to_market_profit = round(final_assets - initial_capital, 2) if final_assets is not None else _number(summary_csv_row.get("cumulative_profit"))
    realized_gross_profit = _number(backtest_summary.get("gross_cumulative_profit")) or _number(summary_csv_row.get("gross_cumulative_profit"))
    realized_period_tax_net_profit = _number(backtest_summary.get("net_cumulative_profit")) or _number(summary_csv_row.get("net_cumulative_profit"))
    open_position_unrealized_profit = (
        round(mark_to_market_profit - realized_gross_profit, 2)
        if mark_to_market_profit is not None and realized_gross_profit is not None
        else None
    )
    sources = [
        _profit_source_from_summary_csv(summary_csv_row),
        _profit_source_from_trade_rows("trades.csv", trades_csv_rows),
        _profit_source_from_backtest_summary(backtest_summary),
        _profit_source_from_feature_analysis(records),
        _profit_source_from_trade_rows("sqlite.trades", sqlite_trade_rows),
    ]
    by_source = {item["source"]: item for item in sources}
    period_net = _number((by_source.get("backtest_summary.json") or {}).get("net_profit"))
    trade_net = _number((by_source.get("trades.csv") or {}).get("net_profit"))
    feature_net = _number((by_source.get("feature_analysis.records") or {}).get("net_profit"))
    warnings: list[str] = []
    if period_net is not None and trade_net is not None and abs(period_net - trade_net) > 1:
        warnings.append("backtest_summary net_cumulative_profit differs from trades.csv per-trade net_profit sum")
    if trade_net is not None and feature_net is not None and abs(trade_net - feature_net) > 1:
        warnings.append("feature_analysis net_profit_sum matches trades.csv per-trade net basis, not period-level net_cumulative_profit")
    cause = _profit_consistency_cause(by_source)
    return {
        "status": "WARNING" if warnings else "OK",
        "true_5y_profit_source": "final_assets - initial_capital for mark-to-market total P/L; net_cumulative_profit for realized period-tax net P/L",
        "true_5y_profit": mark_to_market_profit,
        "mark_to_market_profit": mark_to_market_profit,
        "realized_gross_profit": realized_gross_profit,
        "realized_period_tax_net_profit": realized_period_tax_net_profit,
        "open_position_unrealized_profit": open_position_unrealized_profit,
        "source_rows": sources,
        "warnings": warnings,
        "cause": cause,
        "notes": [
            "backtest_summary net_cumulative_profit uses period-level tax netting: gross_cumulative_profit - period_estimated_tax - commission",
            "final_assets includes open positions marked to market; net_cumulative_profit is realized closed-trade profit after period-level estimated tax",
            "trades.csv and feature_analysis net_profit sums use each trade row's net_profit, where winning trades already include per-trade estimated tax",
            "profit_factor is gross-profit based in backtest_summary, while feature_analysis computed_trade_profit_factor was net-profit based",
        ],
    }


def _trade_expectancy_deep_analysis(records: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    wins = [record for record in records if (_record_gross_profit(record) or 0.0) > 0]
    losses = [record for record in records if (_record_gross_profit(record) or 0.0) < 0]
    gross_profit = sum(_record_gross_profit(record) or 0.0 for record in wins)
    gross_loss = sum(_record_gross_profit(record) or 0.0 for record in losses)
    period_net = _number(summary.get("net_cumulative_profit"))
    return {
        "basis": "gross trade P/L and gross_profit_rate; period net uses backtest_summary tax netting",
        "trade_count": len(records),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": _win_rate_gross(records),
        "profit_factor": _profit_factor_gross(records),
        "average_profit_rate": _average([_record_gross_profit_rate(record) for record in wins if _record_gross_profit_rate(record) is not None]),
        "average_loss_rate": _average([_record_gross_profit_rate(record) for record in losses if _record_gross_profit_rate(record) is not None]),
        "average_profit_amount": _average([_record_gross_profit(record) or 0.0 for record in wins]),
        "average_loss_amount": _average([_record_gross_profit(record) or 0.0 for record in losses]),
        "winning_trade_average_holding_days": _average(_valid_numbers(record.get("holding_days") for record in wins)),
        "losing_trade_average_holding_days": _average(_valid_numbers(record.get("holding_days") for record in losses)),
        "gross_profit_total": round(gross_profit, 2),
        "gross_loss_total": round(gross_loss, 2),
        "gross_net_profit": round(gross_profit + gross_loss, 2),
        "estimated_tax_total": _number(summary.get("estimated_tax_total")) or 0.0,
        "total_commission": _number(summary.get("total_commission")) or 0.0,
        "period_tax_net_profit": period_net,
        "period_net_profit_per_trade": round(period_net / len(records), 4) if period_net is not None and records else None,
    }


def _exit_reason_profit_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_normalized_exit_reason(record.get("exit_reason") or record.get("reason") or record.get("result"))].append(record)
    rows = []
    for reason in ["take_profit", "stop_loss", "max_holding_days", "market_exit", "other"]:
        items = groups.get(reason, [])
        rows.append({
            "reason": reason,
            "count": len(items),
            "win_rate": _win_rate_gross(items),
            "average_profit_rate": _average([_record_gross_profit_rate(item) for item in items if _record_gross_profit_rate(item) is not None]),
            "total_profit": round(sum(_record_gross_profit(item) or 0.0 for item in items), 2),
        })
    return rows


def _profit_capture_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    take_profit = [record for record in records if _normalized_exit_reason(record.get("exit_reason")) == "take_profit"]
    stop_loss = [record for record in records if _normalized_exit_reason(record.get("exit_reason")) == "stop_loss"]
    sorted_by_profit = sorted(records, key=lambda item: _record_gross_profit(item) or 0.0, reverse=True)
    sorted_by_loss = sorted(records, key=lambda item: _record_gross_profit(item) or 0.0)
    return {
        "basis": "gross trade P/L",
        "take_profit_count": len(take_profit),
        "stop_loss_count": len(stop_loss),
        "take_profit_average_capture_rate": _average([_record_gross_profit_rate(record) for record in take_profit if _record_gross_profit_rate(record) is not None]),
        "stop_loss_average_loss_rate": _average([_record_gross_profit_rate(record) for record in stop_loss if _record_gross_profit_rate(record) is not None]),
        "take_profit_average_profit": _average([_record_gross_profit(record) or 0.0 for record in take_profit]),
        "stop_loss_average_profit": _average([_record_gross_profit(record) or 0.0 for record in stop_loss]),
        "top_winners": [_trade_table_record(row, profit_basis="gross") for row in sorted_by_profit[:20]],
        "top_losers": [_trade_table_record(row, profit_basis="gross") for row in sorted_by_loss[:20]],
    }


def _opportunity_loss_analysis(
    root: Path,
    profile_id: str,
    records: list[dict[str, Any]],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    win_records = [record for record in records if (_record_gross_profit(record) or 0.0) > 0 and record.get("exit_date")]
    date_paths = _indicator_date_paths(root, profile_id, start_date, end_date)
    dates = sorted(date_paths)
    close_cache: dict[str, dict[str, float]] = {}

    def close_for(date_value: str | None, code: Any) -> float | None:
        if not date_value:
            return None
        if date_value not in close_cache:
            close_cache[date_value] = _indicator_close_map(date_paths.get(date_value))
        return close_cache[date_value].get(_normalize_code(code))

    def next_date_after(exit_date: str, offset: int) -> str | None:
        index = _date_index(dates, exit_date)
        if index is None:
            return None
        target_index = index + offset
        return dates[target_index] if target_index < len(dates) else None

    rows = []
    for record in win_records:
        exit_date = str(record.get("exit_date") or "")
        exit_price = _number(record.get("exit_price")) or close_for(exit_date, record.get("code"))
        if not exit_price:
            continue
        row = {
            "entry_date": record.get("entry_date"),
            "exit_date": exit_date,
            "code": record.get("code"),
            "name": record.get("name"),
            "exit_price": exit_price,
            "profit": _record_gross_profit(record),
            "profit_rate": _record_gross_profit_rate(record),
            "holding_days": record.get("holding_days"),
            "exit_reason": record.get("exit_reason"),
        }
        for offset in (5, 20):
            target_date = next_date_after(exit_date, offset)
            target_close = close_for(target_date, record.get("code")) if target_date else None
            row[f"after_{offset}d_date"] = target_date
            row[f"after_{offset}d_return"] = round((target_close / exit_price) - 1, 4) if target_close is not None else None
        rows.append(row)
    after_5_values = [row["after_5d_return"] for row in rows if row.get("after_5d_return") is not None]
    after_20_values = [row["after_20d_return"] for row in rows if row.get("after_20d_return") is not None]
    return {
        "basis": "winning closed trades; post-exit return uses indicator close on the 5th/20th available indicator date after exit",
        "winning_trade_count": len(win_records),
        "covered_after_5d_count": len(after_5_values),
        "covered_after_20d_count": len(after_20_values),
        "average_after_5d_return": _average(after_5_values),
        "median_after_5d_return": _median(after_5_values),
        "positive_after_5d_rate": round(sum(1 for value in after_5_values if value > 0) / len(after_5_values), 4) if after_5_values else None,
        "average_after_20d_return": _average(after_20_values),
        "median_after_20d_return": _median(after_20_values),
        "positive_after_20d_rate": round(sum(1 for value in after_20_values if value > 0) / len(after_20_values), 4) if after_20_values else None,
        "top_after_20d_opportunity_losses": sorted(
            rows,
            key=lambda item: item.get("after_20d_return") if item.get("after_20d_return") is not None else -999,
            reverse=True,
        )[:20],
        "notes": _opportunity_loss_notes(len(win_records), len(after_5_values), len(after_20_values)),
    }


def _profit_analysis_conclusion(
    expectancy: dict[str, Any],
    exit_reason_rows: list[dict[str, Any]],
    capture: dict[str, Any],
    opportunity: dict[str, Any],
    consistency: dict[str, Any],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    avg_win = _number(expectancy.get("average_profit_rate")) or 0.0
    avg_loss = _number(expectancy.get("average_loss_rate")) or 0.0
    win_rate = _number(expectancy.get("win_rate")) or 0.0
    if avg_win and avg_loss:
        reasons.append({
            "factor": "thin_expectancy_edge",
            "evidence": (
                f"PF is only {_format_profit_factor(expectancy.get('profit_factor'))}; "
                f"win_rate={_format_percent(win_rate)}, avg_win={_format_percent(avg_win)}, avg_loss={_format_percent(avg_loss)}"
            ),
        })
    period_net = _number(expectancy.get("period_tax_net_profit"))
    gross_net = _number(expectancy.get("gross_net_profit"))
    if period_net is not None and gross_net is not None:
        reasons.append({
            "factor": "tax_and_cost_compression",
            "evidence": (
                f"gross closed-trade net was {_format_yen(gross_net)}, "
                f"but period-tax net was {_format_yen(period_net)}"
            ),
        })
    max_holding = next((row for row in exit_reason_rows if row.get("reason") == "max_holding_days"), {})
    if (max_holding.get("count") or 0) > 0:
        reasons.append({
            "factor": "exit_mix_limits_capture",
            "evidence": (
                f"max_holding_days exits={max_holding.get('count', 0)}, "
                f"total_profit={_format_yen(max_holding.get('total_profit'))}; "
                f"take_profit average capture={_format_percent(capture.get('take_profit_average_capture_rate'))}"
            ),
        })
    after_20 = _number(opportunity.get("average_after_20d_return"))
    if after_20 and after_20 > 0:
        reasons.append({
            "factor": "possible_early_exit_opportunity_loss",
            "evidence": f"winning trades rose another {_format_percent(after_20)} on average over the next 20 available indicator days",
        })
    if consistency.get("cause"):
        reasons.append({
            "factor": "accounting_basis_matters",
            "evidence": str(consistency.get("cause")),
        })
    return [{**row, "rank": index + 1} for index, row in enumerate(reasons[:3])]


def _stop_loss_pattern_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    stop_records = [record for record in records if _normalized_exit_reason(record.get("exit_reason")) == "stop_loss"]
    return {
        "count": len(stop_records),
        "average_loss_rate": _average([_record_gross_profit_rate(record) for record in stop_records if _record_gross_profit_rate(record) is not None]),
        "total_loss": round(sum(_record_gross_profit(record) or 0.0 for record in stop_records), 2),
        "total_score": _pattern_group_by(stop_records, lambda item: _trade_score_bucket(item.get("total_score")), TRADE_SCORE_BUCKET_ORDER),
        "relative_strength_score": _pattern_group_by(stop_records, lambda item: _relative_strength_score_bucket(item.get("relative_strength_score")), RELATIVE_STRENGTH_SCORE_BUCKET_ORDER + ["unknown"]),
        "relative_strength_5d": _pattern_group_by(stop_records, lambda item: _relative_strength_bucket(item.get("relative_strength_5d")), RELATIVE_STRENGTH_BUCKET_ORDER + ["unknown"]),
        "relative_strength_10d": _pattern_group_by(stop_records, lambda item: _relative_strength_bucket(item.get("relative_strength_10d")), RELATIVE_STRENGTH_BUCKET_ORDER + ["unknown"]),
        "relative_strength_20d": _pattern_group_by(stop_records, lambda item: _relative_strength_bucket(item.get("relative_strength_20d")), RELATIVE_STRENGTH_BUCKET_ORDER + ["unknown"]),
        "rsi": _pattern_group_by(stop_records, lambda item: _rsi_bucket(item.get("rsi")), ["0-30", "30-40", "40-50", "50-60", "60-70", "70+"]),
        "volume_ratio": _pattern_group_by(stop_records, lambda item: _volume_bucket(item.get("volume_ratio")), ["<1", "1-2", "2-3", "3+"]),
        "sector_name": _pattern_group_by(stop_records, lambda item: item.get("sector_name") or "unknown"),
        "candlestick_signals": _pattern_group_by_signals(stop_records),
        "market_context_score": _pattern_group_by(stop_records, lambda item: _score_component_bucket(item.get("market_context_score")), COMPONENT_DETAIL_BUCKET_ORDER),
    }


def _max_holding_pattern_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    max_records = [record for record in records if _normalized_exit_reason(record.get("exit_reason")) == "max_holding_days"]
    return {
        "count": len(max_records),
        "win_rate": _win_rate_gross(max_records),
        "average_profit_rate": _average([_record_gross_profit_rate(record) for record in max_records if _record_gross_profit_rate(record) is not None]),
        "total_profit": round(sum(_record_gross_profit(record) or 0.0 for record in max_records), 2),
        "total_score": _pattern_group_by(max_records, lambda item: _trade_score_bucket(item.get("total_score")), TRADE_SCORE_BUCKET_ORDER),
        "relative_strength_score": _pattern_group_by(max_records, lambda item: _relative_strength_score_bucket(item.get("relative_strength_score")), RELATIVE_STRENGTH_SCORE_BUCKET_ORDER + ["unknown"]),
        "rsi": _pattern_group_by(max_records, lambda item: _rsi_bucket(item.get("rsi")), ["0-30", "30-40", "40-50", "50-60", "60-70", "70+"]),
        "volume_ratio": _pattern_group_by(max_records, lambda item: _volume_bucket(item.get("volume_ratio")), ["<1", "1-2", "2-3", "3+"]),
        "sector_name": _pattern_group_by(max_records, lambda item: item.get("sector_name") or "unknown"),
        "candlestick_signals": _pattern_group_by_signals(max_records),
    }


def _winner_vs_stop_loss_contrast(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners = [record for record in records if (_record_gross_profit(record) or 0.0) > 0]
    stops = [record for record in records if _normalized_exit_reason(record.get("exit_reason")) == "stop_loss"]
    contrasts: list[dict[str, Any]] = []
    for feature in [
        "relative_strength_score",
        "relative_strength_5d",
        "relative_strength_10d",
        "relative_strength_20d",
        "rsi",
        "volume_ratio",
        "total_score",
        "technical_score",
        "market_context_score",
    ]:
        win_values = _valid_numbers(record.get(feature) for record in winners)
        stop_values = _valid_numbers(record.get(feature) for record in stops)
        if not win_values and not stop_values:
            contrasts.append({
                "feature": feature,
                "type": "numeric",
                "winner_value": None,
                "stop_loss_value": None,
                "difference": None,
                "impact_score": 0,
                "note": "missing in existing artifacts",
            })
            continue
        win_avg = _average(win_values)
        stop_avg = _average(stop_values)
        diff = round((win_avg or 0.0) - (stop_avg or 0.0), 4) if win_avg is not None and stop_avg is not None else None
        contrasts.append({
            "feature": feature,
            "type": "numeric",
            "winner_value": win_avg,
            "stop_loss_value": stop_avg,
            "difference": diff,
            "impact_score": abs(diff or 0.0),
            "note": "",
        })
    contrasts.extend(_categorical_contrast_rows("sector_name", winners, stops, lambda item: item.get("sector_name") or "unknown"))
    contrasts.extend(_categorical_contrast_rows("candlestick_signals", winners, stops, lambda item: item.get("candlestick_signals") or ["no_signal"], multi=True))
    return sorted(contrasts, key=lambda item: item.get("impact_score") or 0.0, reverse=True)[:10]


def _rule_candidate_proposal(
    records: list[dict[str, Any]],
    stop_loss: dict[str, Any],
    max_holding: dict[str, Any],
    contrast: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stops = [record for record in records if _normalized_exit_reason(record.get("exit_reason")) == "stop_loss"]
    max_records = [record for record in records if _normalized_exit_reason(record.get("exit_reason")) == "max_holding_days"]
    rs_available = any(_number(record.get("relative_strength_score")) is not None for record in records)
    high_rsi = [record for record in records if (_number(record.get("rsi")) or 0.0) >= 70]
    hot_volume = [record for record in records if (_number(record.get("volume_ratio")) or 0.0) >= 3]
    worst_sector_row = _worst_stop_loss_sector(stop_loss.get("sector_name", []))
    max_holding_losers = [record for record in max_records if (_record_gross_profit(record) or 0.0) < 0]
    return [
        {
            "rule_id": "candidate_rs_low_exclusion",
            "rule": "relative_strength_score が低い銘柄を新規候補から除外する閾値をA/Bテストする",
            "aim": "相対的に弱い銘柄の追随失敗を減らす",
            "side_effect": "existing 5-year artifacts lack relative_strength fields, so impact cannot be estimated until the field is persisted",
            "estimated_affected_count": 0 if not rs_available else sum(1 for record in records if (_number(record.get("relative_strength_score")) or 0.0) <= 3),
            "priority": "Low" if not rs_available else "High",
        },
        {
            "rule_id": "candidate_high_rsi_guard",
            "rule": "entry時RSIが70以上の銘柄を除外または減点する",
            "aim": "短期過熱後の反落による損切りを減らす",
            "side_effect": "強いブレイクアウトの勝ちトレードも落とす可能性がある",
            "estimated_affected_count": len(high_rsi),
            "estimated_stop_loss_count": sum(1 for record in high_rsi if record in stops),
            "priority": "High" if sum(1 for record in high_rsi if record in stops) >= 20 else "Low",
        },
        {
            "rule_id": "candidate_hot_volume_guard",
            "rule": "volume_ratioが3以上の銘柄を除外または減点する",
            "aim": "出来高急増後の短期反落を避ける",
            "side_effect": "初動の大勝ち銘柄も削る可能性がある",
            "estimated_affected_count": len(hot_volume),
            "estimated_stop_loss_count": sum(1 for record in hot_volume if record in stops),
            "priority": "High" if sum(1 for record in hot_volume if record in stops) >= 30 else "Medium",
        },
        {
            "rule_id": "max_holding_exit_policy_ab",
            "rule": "max_holding_days 到達銘柄の延長条件または早期撤退条件をA/Bテストする",
            "aim": "最大保有到達197件の利益捕捉を改善する",
            "side_effect": "延長は資金拘束を増やし、早期撤退は勝ち転換を逃す可能性がある",
            "estimated_affected_count": len(max_records),
            "estimated_loss_count": len(max_holding_losers),
            "priority": "Medium" if (max_holding.get("total_profit") or 0) > 0 else "High",
        },
        {
            "rule_id": "sector_specific_stop_guard",
            "rule": f"{worst_sector_row.get('bucket', 'worst_sector')} など損切り損失が大きいsectorを除外または減点する",
            "aim": "sector固有の反落しやすさを抑える",
            "side_effect": "セクター循環局面の勝ち銘柄を逃す可能性がある",
            "estimated_affected_count": int(worst_sector_row.get("count") or 0),
            "estimated_total_loss": worst_sector_row.get("total_profit"),
            "priority": "Low" if (worst_sector_row.get("count") or 0) < 15 else "Medium",
        },
    ]


def _pattern_group_by(records: list[dict[str, Any]], bucket_func: Any, bucket_order: list[str] | None = None) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(bucket_func(record))].append(record)
    keys = list(bucket_order or [])
    keys.extend(sorted(key for key in groups if key not in set(keys)))
    return [_pattern_stats(key, groups.get(key, [])) for key in keys if key in groups or bucket_order]


def _pattern_group_by_signals(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        signals = record.get("candlestick_signals") or ["no_signal"]
        for signal in signals:
            groups[str(signal)].append(record)
    return sorted((_pattern_stats(key, value) for key, value in groups.items()), key=lambda item: (-(item.get("count") or 0), item.get("bucket") or ""))


def _pattern_stats(bucket: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "count": len(items),
        "win_rate": _win_rate_gross(items),
        "average_profit_rate": _average([_record_gross_profit_rate(item) for item in items if _record_gross_profit_rate(item) is not None]),
        "total_profit": round(sum(_record_gross_profit(item) or 0.0 for item in items), 2),
    }


def _categorical_contrast_rows(feature: str, winners: list[dict[str, Any]], stops: list[dict[str, Any]], value_func: Any, multi: bool = False) -> list[dict[str, Any]]:
    win_counts = _category_counts(winners, value_func, multi)
    stop_counts = _category_counts(stops, value_func, multi)
    total_wins = max(1, len(winners))
    total_stops = max(1, len(stops))
    rows = []
    for category in set(win_counts) | set(stop_counts):
        win_rate = win_counts.get(category, 0) / total_wins
        stop_rate = stop_counts.get(category, 0) / total_stops
        diff = round(stop_rate - win_rate, 4)
        rows.append({
            "feature": feature,
            "type": "categorical",
            "category": category,
            "winner_value": round(win_rate, 4),
            "stop_loss_value": round(stop_rate, 4),
            "difference": diff,
            "impact_score": abs(diff),
            "note": "positive difference means more common in stop_loss trades",
        })
    return rows


def _category_counts(records: list[dict[str, Any]], value_func: Any, multi: bool) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        values = value_func(record)
        if not multi:
            values = [values]
        for value in values or ["unknown"]:
            counts[str(value or "unknown")] += 1
    return counts


def _worst_stop_loss_sector(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return min(rows, key=lambda row: _number(row.get("total_profit")) or 0.0)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _profit_source_from_summary_csv(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "summary.csv",
        "trade_count": int(_number(row.get("closed_trades_count")) or 0),
        "gross_profit": _number(row.get("gross_cumulative_profit")),
        "gross_loss": None,
        "net_profit": _number(row.get("net_cumulative_profit")),
        "pf": None,
        "basis": "last row cumulative period-level P/L",
    }


def _profit_source_from_backtest_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "backtest_summary.json",
        "trade_count": int(_number(summary.get("closed_trade_count") or summary.get("total_trades")) or 0),
        "gross_profit": _number(summary.get("gross_win_total")),
        "gross_loss": _number(summary.get("gross_loss_total")),
        "net_profit": _number(summary.get("net_cumulative_profit")),
        "pf": _number(summary.get("profit_factor")),
        "basis": "period-level tax netting",
    }


def _profit_source_from_feature_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _profit_source_from_trade_rows("feature_analysis.records", records)


def _profit_source_from_trade_rows(source: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in rows if _row_is_closed_trade(row)]
    gross_values = [_number(row.get("gross_profit")) for row in closed]
    gross_source = "gross_profit"
    if not any(value is not None for value in gross_values):
        gross_values = [_number(row.get("profit")) for row in closed]
        gross_source = "profit fallback"
    gross_values = [value or 0.0 for value in gross_values]
    net_values = [
        (_number(row.get("net_profit")) if _number(row.get("net_profit")) is not None else _number(row.get("profit")) or 0.0)
        for row in closed
    ]
    gross_profit = round(sum(value for value in gross_values if value > 0), 2)
    gross_loss = round(sum(value for value in gross_values if value < 0), 2)
    pf = round(gross_profit / abs(gross_loss), 4) if gross_loss < 0 else None
    return {
        "source": source,
        "trade_count": len(closed),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": round(sum(net_values), 2),
        "pf": pf,
        "basis": f"closed trade rows ({gross_source})",
    }


def _row_is_closed_trade(row: dict[str, Any]) -> bool:
    action = str(row.get("action") or "").upper()
    result = str(row.get("result") or "")
    status = str(row.get("order_status") or row.get("status") or "").upper()
    if action and action != "SELL":
        return False
    if result and result not in {"WIN", "LOSS"}:
        return False
    if status and status != "FILLED":
        return False
    return bool(row.get("exit_date") or action == "SELL")


def _profit_consistency_cause(by_source: dict[str, dict[str, Any]]) -> str:
    summary = by_source.get("backtest_summary.json") or {}
    trades = by_source.get("trades.csv") or {}
    if _number(summary.get("gross_profit")) is not None and _number(trades.get("gross_profit")) is not None:
        summary_pf = _number(summary.get("pf"))
        trades_pf = _number(trades.get("pf"))
        if summary_pf is not None and trades_pf is not None and abs(summary_pf - trades_pf) <= 0.0001:
            return "gross P/L and PF match; mismatch is caused by period-level tax netting versus per-trade net_profit aggregation"
    return "source mismatch requires manual review"


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


def _earnings_filter_debug(config: dict[str, Any], scoring_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = list(scoring_rows)
    earnings_filter = config.get("earnings_filter", {}) if isinstance(config.get("earnings_filter"), dict) else {}
    enabled = bool(earnings_filter.get("enabled"))
    records_count = int(max([_number(row.get("earnings_calendar_records_count")) or 0 for row in rows], default=0))
    info_found = [row for row in rows if _boolish(row.get("earnings_info_found")) or row.get("earnings_candidate_date")]
    rejected = [row for row in rows if _earnings_filter_row_rejected(row)]
    checked = [row for row in rows if _boolish(row.get("earnings_filter_checked"))]
    status = _earnings_filter_status(enabled, records_count, len(info_found), len(rejected))
    warnings: list[str] = []
    if enabled and status != "active":
        warnings.append(status)
    return {
        "enabled": enabled,
        "status": status,
        "earnings_calendar_records": records_count,
        "candidate_count": len(rows),
        "earnings_info_found_count": len(info_found),
        "earnings_info_missing_count": len(rows) - len(info_found),
        "earnings_filter_candidate_count": len(rejected),
        "earnings_filter_rejected_count": len(rejected),
        "earnings_filter_applied_count": len(checked),
        "unknown_earnings_count": len(rows) - len(info_found),
        "days_to_earnings_distribution": _days_to_earnings_distribution(rows),
        "selection_diff_relation": (
            "earnings_filter_rejected_count is the direct selection-effect candidate count; "
            "compare-profiles selection_diff should increase when these rows would otherwise pass selection."
        ),
        "warnings": sorted(set(warnings)),
        "top_rejected_candidates": [_earnings_rejected_debug_record(row) for row in rejected[:20]],
        "nearest_earnings_candidates": [_earnings_nearest_debug_record(row) for row in _nearest_earnings_rows(rows)[:20]],
    }


def _earnings_filter_status(enabled: bool, records_count: int, found_count: int, rejected_count: int) -> str:
    if not enabled:
        return "disabled"
    if records_count == 0:
        return "earnings_data_unavailable"
    if found_count == 0:
        return "no_candidate_match"
    if rejected_count == 0:
        return "inactive_in_practice"
    return "active"


def _earnings_pipeline(scoring_rows: list[dict[str, Any]], earnings_debug: dict[str, Any]) -> dict[str, Any]:
    rows = list(scoring_rows)
    first = rows[0] if rows else {}
    cache_records = int(max([_number(row.get("earnings_pipeline_cache_records")) or 0 for row in rows], default=0))
    records_loaded = int(max([_number(row.get("earnings_pipeline_records_loaded")) or 0 for row in rows], default=0))
    matched = int(max([_number(row.get("earnings_pipeline_matched_candidates")) or 0 for row in rows], default=0))
    rejected = int(max([_number(row.get("earnings_pipeline_rejected_candidates")) or 0 for row in rows], default=0))
    return {
        "feature_enabled": any(_boolish(row.get("earnings_pipeline_feature_enabled")) for row in rows)
        or bool(earnings_debug.get("enabled")),
        "fetch_start": first.get("earnings_pipeline_fetch_start") or "",
        "fetch_end": first.get("earnings_pipeline_fetch_end") or "",
        "cache_path": first.get("earnings_pipeline_cache_path") or "",
        "cache_exists": any(_boolish(row.get("earnings_pipeline_cache_exists")) for row in rows),
        "cache_records": cache_records,
        "cache_loaded": any(_boolish(row.get("earnings_pipeline_cache_loaded")) for row in rows),
        "index_built": any(_boolish(row.get("earnings_pipeline_index_built")) for row in rows),
        "candidate_matching_called": any(_boolish(row.get("earnings_pipeline_candidate_matching_called")) for row in rows),
        "earnings_records_loaded": records_loaded or int(earnings_debug.get("earnings_calendar_records") or 0),
        "matched_candidates": matched or int(earnings_debug.get("earnings_info_found_count") or 0),
        "rejected_candidates": rejected or int(earnings_debug.get("earnings_filter_rejected_count") or 0),
        "reason": first.get("earnings_pipeline_reason") or "",
    }


def _days_to_earnings_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    buckets = {"<= -3": 0, "-2 to +2": 0, "+3 to +7": 0, "+8 to +14": 0, "+15+": 0, "unknown": 0}
    for row in rows:
        days = _number(row.get("earnings_days_until_earnings"))
        if days is None:
            buckets["unknown"] += 1
        elif days <= -3:
            buckets["<= -3"] += 1
        elif -2 <= days <= 2:
            buckets["-2 to +2"] += 1
        elif 3 <= days <= 7:
            buckets["+3 to +7"] += 1
        elif 8 <= days <= 14:
            buckets["+8 to +14"] += 1
        else:
            buckets["+15+"] += 1
    return buckets


def _earnings_filter_row_rejected(row: dict[str, Any]) -> bool:
    return bool(row.get("earnings_filter_blocked")) or "決算予定日前後" in str(row.get("earnings_filter_reason") or "")


def _earnings_rejected_debug_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": row.get("date"),
        "code": row.get("code"),
        "company_name": row.get("name"),
        "earnings_date": row.get("earnings_candidate_date") or row.get("earnings_announcement_date"),
        "days_to_earnings": _number(row.get("earnings_days_until_earnings")),
        "reason": row.get("earnings_filter_reason") or row.get("rejected_reason"),
        "action": "rejected",
    }


def _nearest_earnings_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = [
        row for row in rows
        if _number(row.get("earnings_days_until_earnings")) is not None
    ]
    return sorted(
        found,
        key=lambda row: (
            abs(float(_number(row.get("earnings_days_until_earnings")) or 0)),
            str(row.get("date") or ""),
            str(row.get("code") or ""),
        ),
    )


def _earnings_nearest_debug_record(row: dict[str, Any]) -> dict[str, Any]:
    if _earnings_filter_row_rejected(row):
        action = "rejected"
    elif _boolish(row.get("selected")):
        action = "selected"
    else:
        action = "candidate"
    return {
        "date": row.get("date"),
        "code": row.get("code"),
        "company_name": row.get("name"),
        "earnings_date": row.get("earnings_candidate_date") or row.get("earnings_announcement_date"),
        "days_to_earnings": _number(row.get("earnings_days_until_earnings")),
        "action": action,
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


def _baseline_profile_id(root: Path, profile_id: str) -> str | None:
    if yaml is None:
        return "rookie_dealer_02_v2_1" if profile_id != "rookie_dealer_02_v2_1" else None
    path = root / "config" / "profile_registry.yaml"
    if path.exists():
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            item = (payload.get("profiles") or {}).get(profile_id, {})
            compare_to = item.get("compare_to") if isinstance(item, dict) else None
            if compare_to:
                return str(compare_to)
        except Exception:
            pass
    return "rookie_dealer_02_v2_1" if profile_id != "rookie_dealer_02_v2_1" else None


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


def _profit_factor(items: list[dict[str, Any]]) -> float | None:
    profits = [_record_profit(item) or 0.0 for item in items]
    gross_profit = sum(value for value in profits if value > 0)
    gross_loss = abs(sum(value for value in profits if value < 0))
    if gross_loss == 0:
        return None if gross_profit == 0 else float("inf")
    return round(gross_profit / gross_loss, 4)


def _profit_factor_gross(items: list[dict[str, Any]]) -> float | None:
    profits = [_record_gross_profit(item) or 0.0 for item in items]
    gross_profit = sum(value for value in profits if value > 0)
    gross_loss = abs(sum(value for value in profits if value < 0))
    if gross_loss == 0:
        return None if gross_profit == 0 else float("inf")
    return round(gross_profit / gross_loss, 4)


def _win_rate(items: list[dict[str, Any]]) -> float | None:
    if not items:
        return None
    wins = [item for item in items if item.get("result") == "WIN" or (_record_profit(item) or 0.0) > 0]
    return round(len(wins) / len(items), 4)


def _win_rate_gross(items: list[dict[str, Any]]) -> float | None:
    if not items:
        return None
    wins = [item for item in items if (_record_gross_profit(item) or 0.0) > 0]
    return round(len(wins) / len(items), 4)


def _profit_rate_bucket(value: Any) -> str:
    rate = _number(value)
    if rate is None:
        return "unknown"
    if rate <= -0.10:
        return "<= -10%"
    if rate <= -0.05:
        return "-10% to -5%"
    if rate <= -0.03:
        return "-5% to -3%"
    if rate <= -0.01:
        return "-3% to -1%"
    if rate < 0:
        return "-1% to 0%"
    if rate < 0.01:
        return "0% to 1%"
    if rate < 0.03:
        return "1% to 3%"
    if rate < 0.05:
        return "3% to 5%"
    if rate < 0.10:
        return "5% to 10%"
    return "10%+"


def _holding_period_bucket(value: Any) -> str:
    days = _number(value)
    if days is None:
        return "unknown"
    if days <= 1:
        return "1 day"
    if days == 2:
        return "2 days"
    if days == 3:
        return "3 days"
    if days <= 5:
        return "4-5 days"
    if days <= 10:
        return "6-10 days"
    return "11+ days"


def _trade_score_bucket(value: Any) -> str:
    score = _number(value)
    if score is None:
        return "unknown"
    if score < 40:
        return "<40"
    if score < 45:
        return "40-44"
    if score < 50:
        return "45-49"
    if score < 55:
        return "50-54"
    if score < 60:
        return "55-59"
    if score < 65:
        return "60-64"
    if score < 70:
        return "65-69"
    return "70+"


def _trade_table_record(record: dict[str, Any], profit_basis: str = "net") -> dict[str, Any]:
    profit = _record_gross_profit(record) if profit_basis == "gross" else _record_profit(record)
    profit_rate = _record_gross_profit_rate(record) if profit_basis == "gross" else _record_profit_rate(record)
    return {
        "entry_date": record.get("entry_date"),
        "exit_date": record.get("exit_date"),
        "code": record.get("code"),
        "name": record.get("name"),
        "entry_price": record.get("entry_price"),
        "exit_price": record.get("exit_price"),
        "profit": profit,
        "profit_rate": profit_rate,
        "holding_days": record.get("holding_days"),
        "exit_reason": record.get("exit_reason"),
        "total_score": record.get("total_score"),
        "ma_score": record.get("ma_score"),
        "rsi_score": record.get("rsi_score"),
        "volume_score": record.get("volume_score"),
        "candlestick_score": record.get("candlestick_score"),
        "relative_strength_score": record.get("relative_strength_score"),
        "technical_score": record.get("technical_score"),
    }


def _trade_key(record: dict[str, Any]) -> str:
    signal_date = str(record.get("signal_date") or record.get("entry_date") or "")
    code = str(record.get("code") or "")
    return f"{signal_date}|{code}" if signal_date and code else ""


def _trade_subset_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    stats = _group_stats("summary", records)
    stats["profit_factor"] = _profit_factor(records)
    stats["avg_holding_days"] = _average(_valid_numbers(record.get("holding_days") for record in records))
    return stats


def _relative_strength_improvement_type(newly_selected: list[dict[str, Any]], removed: list[dict[str, Any]]) -> str:
    newly_profit = sum(_record_profit(record) or 0.0 for record in newly_selected)
    removed_profit = sum(_record_profit(record) or 0.0 for record in removed)
    if newly_profit > 0 and removed_profit < 0:
        return "winner_addition_and_loss_reduction"
    if newly_profit > 0:
        return "winner_addition"
    if removed_profit < 0:
        return "loss_reduction"
    return "mixed_or_no_clear_effect"


def _investor_context_effect_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [
        ("investor_context_score >= 4", [record for record in records if (_number(record.get("investor_context_score")) or 0) >= 4]),
        ("investor_context_score <= 0", [record for record in records if (_number(record.get("investor_context_score")) or 0) <= 0]),
    ]
    result = []
    for name, items in groups:
        stats = _group_stats(name, items)
        stats["profit_factor"] = _profit_factor(items)
        result.append(stats)
    return result


def _investor_context_filter_analysis(scoring_rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    rejected = [
        row for row in scoring_rows
        if str(row.get("rejected_reason") or "") == "investor_context_negative"
        or bool(row.get("investor_context_filter_blocked"))
    ]
    rejected_codes = sorted({str(row.get("code")) for row in rejected if row.get("code")})
    profit_if_kept_values = [
        value for value in (_profit_if_kept_value(row) for row in rejected)
        if value is not None
    ]
    accepted = [
        record for record in records
        if (_number(record.get("investor_context_score")) is None or (_number(record.get("investor_context_score")) or 0) >= 0)
    ]
    return {
        "rejected_count": len(rejected),
        "rejected_codes": rejected_codes,
        "rejected_profit_if_kept": round(sum(profit_if_kept_values), 2) if profit_if_kept_values else None,
        "accepted_profit": round(sum((_record_profit(record) or 0.0) for record in accepted), 2),
    }


def _api_field_usage_audit(
    config: dict[str, Any],
    root: Path,
    profile_id: str,
    start_date: str | None,
    end_date: str | None,
    scoring_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    samples = _api_field_usage_samples(root, profile_id, start_date, end_date)
    field_rows = _api_field_usage_rows(samples)
    return {
        "profile_id": profile_id,
        "source": "static implementation audit with lightweight local sample inspection",
        "api_spec_reference": {
            "version": "J-Quants API v2 field names observed from implementation/API docs",
            "endpoints": [
                "/equities/master",
                "/equities/bars/daily",
                "/indices/bars/daily/topix",
                "/equities/investor-types",
                "/equities/earnings-calendar",
                "/fins/summary",
                "/markets/margin-interest",
                "/markets/breakdown",
            ],
        },
        "adjusted_price_usage": _adjusted_price_usage_audit(samples),
        "field_usage": field_rows,
        "financial_context_audit": _financial_context_audit(config, root, start_date, end_date, scoring_rows),
        "disabled_by_default_candidates": {
            "limit_up_entry_guard.enabled": False,
            "limit_down_guard.enabled": False,
            "product_category_filter.enabled": False,
            "margin_type_score.enabled": False,
            "note": "config候補として監査に出すだけで、現時点では売買ロジック・スコアに未接続。",
        },
        "future_candidates": [
            {
                "endpoint": "/markets/margin-interest",
                "purpose": "信用買残/売残・信用倍率による需給悪化チェック候補",
                "status": "future_candidate_not_implemented",
                "future_leak_risk": "medium: signal_date以前に公表済みの週次/日次データだけに制限が必要",
            },
            {
                "endpoint": "/markets/breakdown",
                "purpose": "現物/信用/空売り系の売買内訳による短期需給補助候補",
                "status": "future_candidate_not_implemented",
                "future_leak_risk": "medium: pubdate/asof制御が必要",
            },
        ],
        "summary": _api_field_usage_summary(field_rows),
    }


def _api_field_usage_samples(root: Path, profile_id: str, start_date: str | None, end_date: str | None) -> dict[str, Any]:
    raw_price = _first_payload_from_files(root / "data" / "raw", "prices_*.json", start_date, end_date)
    listed = _read_json_if_exists(root / "data" / "raw" / "listed_stocks_jquants.json")
    indicators = _first_payload_from_files(root / "data" / "processed" / profile_id, "indicators_*.json", start_date, end_date)
    candidates = _first_payload_from_files(root / "data" / "processed" / profile_id, "candidates_*.json", start_date, end_date)
    scored = _first_payload_from_files(root / "data" / "processed" / profile_id, "scored_candidates_*.json", start_date, end_date)
    return {
        "raw_price_record": _first_record(raw_price, "prices"),
        "listed_record": _first_record(listed, "stocks"),
        "indicator_record": _first_record(indicators, "indicators"),
        "candidate_record": _first_record(candidates, "candidates"),
        "scored_record": _first_record(scored, "scores"),
    }


def _api_field_usage_rows(samples: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _field_usage("/equities/master", "Code", "code", "code", "yes", "yes", "yes", "yes", "no", "drop row when absent", "low"),
        _field_usage("/equities/master", "CoName", "name", "name", "report", "yes", "report", "yes", "no", "empty string", "low"),
        _field_usage("/equities/master", "CoNameEn", "name_en", "", "no", "no", "no", "no", "yes", "empty string; currently only raw normalized", "low"),
        _field_usage("/equities/master", "Mkt/MktNm", "market_section", "market_section", "no", "yes", "yes", "yes", "no", "Unknown is excluded when allow_unknown_market=false", "low"),
        _field_usage("/equities/master", "S17/S17Nm", "sector17_code/sector17_name", "sector17_code/sector17_name", "no", "no", "no", "audit/report", "partly", "empty string; saved for future sector stability", "low"),
        _field_usage("/equities/master", "S33/S33Nm", "sector33_code/sector33_name", "sector33_code/sector33_name + sector_name", "sector_score via sector_name", "yes", "no", "yes", "no", "empty string; sector momentum becomes neutral", "low"),
        _field_usage("/equities/master", "ScaleCat", "scale_category", "scale_category", "no", "no", "no", "audit", "yes", "empty string; saved as future size-bias candidate", "low"),
        _field_usage("/equities/master", "Mrgn/MrgnNm", "margin_type", "margin_type", "no", "no", "no", "audit", "yes", "empty string; saved as future liquidity/supply-demand candidate", "low"),
        _field_usage("/equities/master", "ProdCat", "product_category", "product_category", "no", "no", "disabled_candidate", "audit", "yes", "empty string; product_category_filter disabled by default", "low"),
        _field_usage("/equities/bars/daily", "O/H/L/C", "open/high/low/close", "open/high/low/close", "yes", "yes", "no", "yes", "no", "row skipped for indicator calc if close missing", "low"),
        _field_usage("/equities/bars/daily", "Vo", "volume", "volume", "yes", "yes", "yes", "yes", "no", "row skipped for indicator calc if volume missing", "low"),
        _field_usage("/equities/bars/daily", "AdjO/AdjH/AdjL/AdjC", "adjusted_open/high/low/close", "adjusted_open/high/low/close", "no", "no", "no", "audit", "yes", "kept when available; current MA/RSI/candles use unadjusted OHLC", "low"),
        _field_usage("/equities/bars/daily", "AdjVo", "adjusted_volume", "adjusted_volume", "no", "no", "no", "audit", "yes", "kept when available; current volume_ratio uses Vo", "low"),
        _field_usage("/equities/bars/daily", "Va", "turnover_value", "direct_turnover_value", "not currently; turnover_value is estimated", "no", "no", "audit", "partly", "direct Va saved; scoring still uses close*volume estimate", "low"),
        _field_usage("/equities/bars/daily", "UL", "limit_up_flag", "limit_up_flag", "no", "no", "disabled_candidate", "audit", "yes", "saved when available; limit_up_entry_guard disabled by default", "low"),
        _field_usage("/equities/bars/daily", "LL", "limit_down_flag", "limit_down_flag", "no", "no", "disabled_candidate", "audit", "yes", "saved when available; limit_down_guard disabled by default", "low"),
        _field_usage("/indices/bars/daily/topix", "Date/O/H/L/C", "records", "benchmark_return_*d", "yes when relative_strength enabled", "no", "no", "yes", "no", "relative_strength_score becomes 0/fallback if unavailable", "low"),
        _field_usage("/equities/investor-types", "Foreigners/Individuals buy-sell fields", "records", "investor_context_*", "yes for v2_8", "no", "yes for v2_11", "yes", "no", "disabled/unavailable context is score 0", "low if as-of date is respected"),
        _field_usage("/equities/earnings-calendar", "Date/Code", "records", "earnings_*", "no", "no", "yes when enabled", "yes", "no", "fail_open setting controls unavailable data", "medium: future events are intentionally used only for event-risk blocking"),
        _field_usage("/fins/summary", "DisclosedDate/Code and summary fields", "records", "", "no", "no", "no", "audit only", "yes", "not joined into scoring; disabled for short-term score", "high unless DisclosedDate <= signal_date"),
    ]


def _field_usage(
    endpoint: str,
    api_field: str,
    raw_field: str,
    processed_field: str,
    scoring: str,
    screening: str,
    filtering: str,
    report_audit: str,
    unused: str,
    missing: str,
    future_leak_risk: str,
) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "api_response_field": api_field,
        "raw_field": raw_field,
        "processed_field": processed_field,
        "scoring_used": scoring,
        "screening_used": screening,
        "filter_used": filtering,
        "report_or_audit_only": report_audit,
        "unused": unused,
        "missing_handling": missing,
        "future_leak_risk": future_leak_risk,
    }


def _adjusted_price_usage_audit(samples: dict[str, Any]) -> dict[str, Any]:
    raw = samples.get("raw_price_record", {})
    indicator = samples.get("indicator_record", {})
    candidate = samples.get("candidate_record", {})
    adjusted_fields = ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close", "adjusted_volume"]
    return {
        "raw_adjusted_fields_present": any(raw.get(field) is not None for field in adjusted_fields),
        "processed_adjusted_fields_present": any(indicator.get(field) is not None for field in adjusted_fields),
        "candidate_adjusted_fields_present": any(candidate.get(field) is not None for field in adjusted_fields),
        "ma_basis": "close (unadjusted C/Close/OHLC priority in _normalize_daily_price)",
        "rsi_basis": "close (unadjusted C/Close/OHLC priority in _normalize_daily_price)",
        "relative_strength_stock_return_basis": "close (unadjusted C/Close/OHLC priority in _normalize_daily_price)",
        "candlestick_basis": "open/high/low/close (unadjusted O/H/L/C priority)",
        "volume_ratio_basis": "volume (unadjusted Vo priority)",
        "turnover_value_usage": "scoring/screening uses estimated close*volume; direct API Va is saved as direct_turnover_value when available",
        "limit_up_down_saved": bool(indicator.get("limit_up_flag") is not None or indicator.get("limit_down_flag") is not None),
        "recommendation": "AdjC/AdjVoが十分に保存されていることを確認後、株式分割をまたぐbacktestでは調整後価格ベースへ切替検証する候補。",
    }


def _financial_context_audit(
    config: dict[str, Any],
    root: Path,
    start_date: str | None,
    end_date: str | None,
    scoring_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    records = _financial_records_for_period(root, start_date, end_date)
    selected_codes = sorted({str(row.get("code")) for row in scoring_rows if row.get("selected") and row.get("code")})
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        code = str(_first_existing(record, ["code", "Code", "LocalCode"]) or "")
        if code:
            by_code[code].append(record)
    latest_per_code = {}
    for code, rows in by_code.items():
        latest = max((str(_first_existing(row, ["DisclosedDate", "disclosed_date", "Date", "date"]) or "") for row in rows), default="")
        latest_per_code[code] = latest
    signal_date = end_date or start_date or ""
    future_leak_samples = [
        {"code": code, "latest_disclosure_date": disclosed}
        for code, disclosed in sorted(latest_per_code.items())
        if signal_date and disclosed and disclosed > signal_date
    ][:20]
    available_fields = sorted({key for record in records[:100] for key in record.keys()})
    joinable_selected = [code for code in selected_codes if code in by_code]
    return {
        "feature_enabled": bool(config.get("features", {}).get("financial_context")),
        "scoring_enabled": bool(config.get("scoring", {}).get("use_financial_score")),
        "data_exists": bool(records),
        "record_count": len(records),
        "available_fields": available_fields,
        "latest_disclosure_date_per_code_sample": dict(list(sorted(latest_per_code.items()))[:20]),
        "selected_code_count": len(selected_codes),
        "joinable_selected_code_count": len(joinable_selected),
        "joinable_selected_code_sample": joinable_selected[:20],
        "future_data_leak_risk": "high if DisclosedDate after signal_date is joined into score",
        "future_data_leak_sample_count": len(future_leak_samples),
        "future_data_leak_samples": future_leak_samples,
        "status": "audit_only_not_scored",
    }


def _api_field_usage_summary(field_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_fields": len(field_rows),
        "scoring_used_count": sum(1 for row in field_rows if str(row.get("scoring_used")) not in {"no", "not currently; turnover_value is estimated"}),
        "screening_used_count": sum(1 for row in field_rows if str(row.get("screening_used")) == "yes"),
        "filter_used_count": sum(1 for row in field_rows if str(row.get("filter_used")) not in {"no"}),
        "unused_or_future_count": sum(1 for row in field_rows if str(row.get("unused")) in {"yes", "partly"}),
        "high_future_leak_risk_count": sum(1 for row in field_rows if str(row.get("future_leak_risk")).startswith("high")),
    }


def _volume_filter_audit(
    config: dict[str, Any],
    root: Path,
    profile_id: str,
    start_date: str | None,
    end_date: str | None,
    scoring_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    baseline_profile_id: str | None,
) -> dict[str, Any]:
    volume_filter = config.get("volume_filter", {}) if isinstance(config.get("volume_filter"), dict) else {}
    max_volume_ratio = _number(volume_filter.get("max_volume_ratio"))
    min_volume_ratio = _number(volume_filter.get("min_volume_ratio"))
    pre_filter_rows = _processed_candidate_rows(root, profile_id, start_date, end_date)
    post_filter_rows = _processed_scored_rows(root, profile_id, start_date, end_date)
    if not post_filter_rows:
        post_filter_rows = scoring_rows
    baseline_selected_rows = _processed_scored_rows(root, baseline_profile_id or "", start_date, end_date) if baseline_profile_id else []
    target_selected_keys = {_selection_row_key(row) for row in post_filter_rows if _selection_row_key(row)}
    target_trade_keys = {_selection_row_key(row) for row in records if _selection_row_key(row)}
    baseline_above_max_selected = [
        row for row in baseline_selected_rows
        if max_volume_ratio is not None
        and (_number(row.get("volume_ratio")) or 0.0) > max_volume_ratio
        and _selection_row_key(row) not in target_selected_keys
    ]
    rows_above_max = [
        row for row in pre_filter_rows
        if max_volume_ratio is not None and (_number(row.get("volume_ratio")) or 0.0) > max_volume_ratio
    ]
    post_rows_above_max = [
        row for row in post_filter_rows
        if max_volume_ratio is not None and (_number(row.get("volume_ratio")) or 0.0) > max_volume_ratio
    ]
    rejected_by_max = [
        row for row in post_filter_rows
        if str(row.get("rejected_reason") or row.get("reason") or "") == "volume_ratio_above_max"
        or str(row.get("volume_filter_reason") or "") == "volume_ratio_above_max"
    ]
    baseline_impacted = [
        record for record in baseline_records
        if max_volume_ratio is not None
        and (_number(record.get("volume_ratio")) or 0.0) > max_volume_ratio
        and _selection_row_key(record) not in target_trade_keys
    ]
    top_pick_blocked = [row for row in baseline_above_max_selected if _is_top_pick_selection_reason(row)]
    conditional_blocked = [row for row in baseline_above_max_selected if _is_conditional_selection_reason(row)]
    compare_diff = _compare_profile_diff_analysis(root, baseline_profile_id, profile_id, start_date, end_date)
    selection_diff_count = int(_number(compare_diff.get("selection_diff_count")) or 0)
    compare_base_selected_count = int(_number(compare_diff.get("base_selected_count")) or 0)
    compare_target_selected_count = int(_number(compare_diff.get("target_selected_count")) or 0)
    removed_by_max_count = len(baseline_above_max_selected)
    selection_diff_reason_breakdown = {
        "volume_ratio_above_max": removed_by_max_count,
        "top_pick_blocked_by_max_volume_ratio": len(top_pick_blocked),
        "conditional_blocked_by_max_volume_ratio": len(conditional_blocked),
        "other_selection_diff": max(0, selection_diff_count - removed_by_max_count),
    }
    return {
        "enabled": bool(volume_filter.get("enabled", False)),
        "min_volume_ratio": min_volume_ratio,
        "max_volume_ratio": max_volume_ratio,
        "scoring_rows_count": len(scoring_rows),
        "pre_filter_scoring_rows_count": len(pre_filter_rows),
        "post_filter_scoring_rows_count": len(post_filter_rows),
        "volume_ratio_above_max_count": len(rows_above_max),
        "volume_ratio_above_max_pre_filter_count": len(rows_above_max),
        "volume_ratio_above_max_post_filter_count": len(post_rows_above_max),
        "volume_ratio_above_max_rejected_count": len(rejected_by_max),
        "volume_ratio_above_max_selected_blocked_count": len(baseline_above_max_selected),
        "volume_ratio_above_max_top_pick_blocked_count": len(top_pick_blocked),
        "volume_ratio_above_max_conditional_blocked_count": len(conditional_blocked),
        "selection_diff_count": selection_diff_count,
        "compare_base_selected_count": compare_base_selected_count,
        "compare_target_selected_count": compare_target_selected_count,
        "selection_diff_reason_breakdown": selection_diff_reason_breakdown,
        "removed_by_max_volume_ratio_count": removed_by_max_count,
        "removed_by_max_volume_ratio_profit_if_baseline_kept": round(sum(_record_gross_profit(record) or 0.0 for record in baseline_impacted), 2),
        "rejected_by_max_volume_ratio_samples": [_volume_filter_sample(row) for row in rejected_by_max[:20]],
        "above_max_samples": [_volume_filter_sample(row) for row in rows_above_max[:20]],
        "removed_by_max_volume_ratio_samples": [_volume_filter_sample(row) for row in baseline_above_max_selected[:20]],
        "baseline_profile_id": baseline_profile_id,
        "baseline_above_max_trade_count": len(baseline_impacted),
        "baseline_above_max_trade_summary": _gross_trade_subset_summary(baseline_impacted),
        "applied_stage": "pre-selection scoring: max_volume_ratio candidates are removed before final scored_candidates/selected artifacts",
        "selection_diff_interpretation": _volume_filter_selection_diff_interpretation(
            selection_diff_count,
            compare_base_selected_count,
            removed_by_max_count,
            len(rows_above_max),
            len(post_rows_above_max),
        ),
        "note": "pre_filter counts use data/processed/<profile>/candidates_*.json; post_filter counts use scored_candidates/scoring rows. baseline summaries are audit-only estimates from existing logs.",
    }


def _volume_filter_sample(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": row.get("date") or row.get("signal_date") or row.get("entry_date"),
        "code": row.get("code"),
        "name": row.get("name"),
        "volume_ratio": _number(row.get("volume_ratio")),
        "total_score": _number(row.get("total_score")),
        "rejected_reason": row.get("rejected_reason") or row.get("reason"),
    }


def _processed_candidate_rows(root: Path, profile_id: str, start_date: str | None, end_date: str | None) -> list[dict[str, Any]]:
    return _processed_json_stage_rows(root, profile_id, "candidates", start_date, end_date)


def _processed_scored_rows(root: Path, profile_id: str, start_date: str | None, end_date: str | None) -> list[dict[str, Any]]:
    rows = []
    for row in _processed_json_stage_rows(root, profile_id, "scored_candidates", start_date, end_date):
        if row.get("selected") is False and not row.get("rejected_reason"):
            continue
        rows.append(row)
    return rows


def _processed_json_stage_rows(root: Path, profile_id: str, stage: str, start_date: str | None, end_date: str | None) -> list[dict[str, Any]]:
    if not profile_id:
        return []
    directory = root / "data" / "processed" / profile_id
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"{stage}_*.json")):
        date_text = _date_from_filename(path)
        if start_date and date_text and date_text < start_date:
            continue
        if end_date and date_text and date_text > end_date:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in _rows_from_processed_payload(payload, stage):
            if isinstance(row, dict):
                item = dict(row)
                item.setdefault("date", date_text or payload.get("date") if isinstance(payload, dict) else date_text)
                rows.append(item)
    return rows


def _rows_from_processed_payload(payload: Any, stage: str) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    if stage == "scored_candidates":
        rows = payload.get("scores")
        if isinstance(rows, list) and rows:
            return rows
        rows = payload.get("selected")
        return rows if isinstance(rows, list) else []
    for key in [stage, "candidates", "scores", "selected"]:
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    return []


def _selection_row_key(row: dict[str, Any]) -> str:
    date = str(row.get("signal_date") or row.get("date") or row.get("entry_date") or "")
    code = str(row.get("code") or "")
    return f"{date}|{code}" if date and code else ""


def _is_top_pick_selection_reason(row: dict[str, Any]) -> bool:
    reason = str(row.get("selection_reason") or row.get("selected_reason") or row.get("reason") or "")
    return "ノートレード回避" in reason or "最上位候補" in reason or "top_pick" in reason


def _is_conditional_selection_reason(row: dict[str, Any]) -> bool:
    reason = str(row.get("selection_reason") or row.get("selected_reason") or row.get("reason") or "")
    return "conditional" in reason or "低スコア例外" in reason


def _compare_profile_diff_analysis(
    root: Path,
    baseline_profile_id: str | None,
    profile_id: str,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    if not baseline_profile_id or not start_date or not end_date:
        return {}
    path = root / "reports" / "experiments" / f"{start_date}_to_{end_date}" / baseline_profile_id / "compare_profiles.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    analyses = payload.get("profile_diff_analyses") if isinstance(payload, dict) else []
    if not isinstance(analyses, list):
        return {}
    for item in analyses:
        if isinstance(item, dict) and item.get("target_profile_id") == profile_id:
            return item
    return {}


def _volume_filter_selection_diff_interpretation(
    selection_diff_count: int,
    compare_base_selected_count: int,
    removed_by_max_count: int,
    pre_above_max_count: int,
    post_above_max_count: int,
) -> str:
    if pre_above_max_count > 0 and post_above_max_count == 0:
        stage = "max_volume_ratio is visible in pre-filter candidates and absent after final scoring artifacts, so it is applied before final scored_candidates are persisted"
    elif pre_above_max_count > 0:
        stage = "max_volume_ratio candidates remain after final scoring artifacts; check scoring persistence"
    else:
        stage = "no candidates above max_volume_ratio were found in pre-filter candidates"
    if compare_base_selected_count == 0 and selection_diff_count > 0:
        return f"{stage}; compare_profiles baseline selected count is 0, so selection_diff_count cannot be attributed only to max_volume_ratio"
    if removed_by_max_count == 0 and selection_diff_count > 0:
        return f"{stage}; selection_diff_count exists, but no baseline selected rows above max_volume_ratio were available in existing artifacts"
    return stage


def _gross_trade_subset_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "win_rate": _win_rate_gross(records),
        "average_profit": _average([_record_gross_profit(record) or 0.0 for record in records]),
        "average_profit_rate": _average([_record_gross_profit_rate(record) for record in records if _record_gross_profit_rate(record) is not None]),
        "total_profit": round(sum(_record_gross_profit(record) or 0.0 for record in records), 2),
        "profit_factor": _profit_factor_gross(records),
    }


def _volume_filter_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- データなし"]
    summary = audit.get("baseline_above_max_trade_summary", {})
    lines = [
        f"- enabled: {str(bool(audit.get('enabled'))).lower()}",
        f"- min_volume_ratio: {_format_number(audit.get('min_volume_ratio'))}",
        f"- max_volume_ratio: {_format_number(audit.get('max_volume_ratio'))}",
        f"- scoring_rows_count: {audit.get('scoring_rows_count', 0)}",
        f"- pre_filter_scoring_rows_count: {audit.get('pre_filter_scoring_rows_count', 0)}",
        f"- post_filter_scoring_rows_count: {audit.get('post_filter_scoring_rows_count', 0)}",
        f"- volume_ratio_above_max_count: {audit.get('volume_ratio_above_max_count', 0)}",
        f"- volume_ratio_above_max_pre_filter_count: {audit.get('volume_ratio_above_max_pre_filter_count', 0)}",
        f"- volume_ratio_above_max_post_filter_count: {audit.get('volume_ratio_above_max_post_filter_count', 0)}",
        f"- volume_ratio_above_max_rejected_count: {audit.get('volume_ratio_above_max_rejected_count', 0)}",
        f"- volume_ratio_above_max_selected_blocked_count: {audit.get('volume_ratio_above_max_selected_blocked_count', 0)}",
        f"- volume_ratio_above_max_top_pick_blocked_count: {audit.get('volume_ratio_above_max_top_pick_blocked_count', 0)}",
        f"- volume_ratio_above_max_conditional_blocked_count: {audit.get('volume_ratio_above_max_conditional_blocked_count', 0)}",
        f"- selection_diff_count: {audit.get('selection_diff_count', 0)}",
        f"- compare_base_selected_count: {audit.get('compare_base_selected_count', 0)}",
        f"- compare_target_selected_count: {audit.get('compare_target_selected_count', 0)}",
        f"- removed_by_max_volume_ratio_count: {audit.get('removed_by_max_volume_ratio_count', 0)}",
        f"- removed_by_max_volume_ratio_profit_if_baseline_kept: {_format_yen(audit.get('removed_by_max_volume_ratio_profit_if_baseline_kept'))}",
        f"- baseline_profile_id: {audit.get('baseline_profile_id') or 'N/A'}",
        f"- baseline_above_max_trade_count: {audit.get('baseline_above_max_trade_count', 0)}",
        f"- baseline_above_max_total_profit: {_format_yen(summary.get('total_profit'))}",
        f"- baseline_above_max_win_rate: {_format_percent(summary.get('win_rate'))}",
        f"- baseline_above_max_avg_profit_rate: {_format_percent(summary.get('average_profit_rate'))}",
        f"- applied_stage: {audit.get('applied_stage')}",
        f"- selection_diff_interpretation: {audit.get('selection_diff_interpretation')}",
        f"- note: {audit.get('note')}",
        "",
        "### selection_diff_reason_breakdown",
        "",
        *_key_value_lines(audit.get("selection_diff_reason_breakdown", {})),
        "",
        "### rejected_by_max_volume_ratio samples",
        "",
        *_volume_filter_sample_lines(audit.get("rejected_by_max_volume_ratio_samples", [])),
        "",
        "### removed_by_max_volume_ratio samples",
        "",
        *_volume_filter_sample_lines(audit.get("removed_by_max_volume_ratio_samples", [])),
        "",
        "### above_max samples",
        "",
        *_volume_filter_sample_lines(audit.get("above_max_samples", [])),
    ]
    return lines


def _volume_filter_sample_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| date | code | name | volume_ratio | total_score | rejected_reason |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    if not rows:
        lines.append("| - | - | - | N/A | N/A | - |")
        return lines
    for row in rows:
        lines.append(
            "| "
            f"{row.get('date') or ''} | "
            f"{row.get('code') or ''} | "
            f"{row.get('name') or ''} | "
            f"{_format_number(row.get('volume_ratio'))} | "
            f"{_format_number(row.get('total_score'))} | "
            f"{row.get('rejected_reason') or ''} |"
        )
    return lines


def _rsi_volume_hot_zone_audit(config: dict[str, Any], scoring_rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    hot_zone = config.get("rsi_volume_hot_zone_filter", {}) if isinstance(config.get("rsi_volume_hot_zone_filter"), dict) else {}
    min_rsi = _number(hot_zone.get("min_rsi")) or 60.0
    min_volume_ratio = _number(hot_zone.get("min_volume_ratio")) or 3.0
    max_volume_ratio = _number(hot_zone.get("max_volume_ratio")) or 5.0
    scoring_matches = [
        row for row in scoring_rows
        if _matches_rsi_volume_hot_zone(row, min_rsi, min_volume_ratio, max_volume_ratio)
    ]
    rejected_rows = [
        row for row in scoring_matches
        if str(row.get("rejected_reason") or row.get("reason") or row.get("rsi_volume_hot_zone_reason") or "") == "rsi_volume_hot_zone"
        or bool(row.get("rsi_volume_hot_zone_excluded"))
    ]
    trade_matches = [
        row for row in records
        if _matches_rsi_volume_hot_zone(row, min_rsi, min_volume_ratio, max_volume_ratio)
    ]
    return {
        "enabled": bool(hot_zone.get("enabled", False)),
        "min_rsi": min_rsi,
        "min_volume_ratio": min_volume_ratio,
        "max_volume_ratio": max_volume_ratio,
        "scoring_rows_count": len(scoring_rows),
        "matched_scoring_rows_count": len(scoring_matches),
        "rejected_count": len(rejected_rows),
        "rejected_trade_count": len(trade_matches),
        "rejected_profit_estimate": round(sum(_record_gross_profit(row) or 0.0 for row in trade_matches), 2),
        "gross_loss_reduction_estimate": round(abs(sum(_record_gross_profit(row) or 0.0 for row in trade_matches if (_record_gross_profit(row) or 0.0) < 0)), 2),
        "missed_profit_estimate": round(sum(_record_gross_profit(row) or 0.0 for row in trade_matches if (_record_gross_profit(row) or 0.0) > 0), 2),
        "sample_rows": [_rsi_volume_hot_zone_sample(row) for row in (rejected_rows[:20] or scoring_matches[:20] or trade_matches[:20])],
        "note": "rejected_count is based on scoring rows/rejected_reason; rejected_trade_count and rejected_profit_estimate are an audit-only estimate from existing closed trades matching the configured hot zone.",
    }


def _matches_rsi_volume_hot_zone(row: dict[str, Any], min_rsi: float, min_volume_ratio: float, max_volume_ratio: float) -> bool:
    rsi = _number(row.get("rsi"))
    volume_ratio = _number(row.get("volume_ratio"))
    return rsi is not None and volume_ratio is not None and rsi >= min_rsi and volume_ratio >= min_volume_ratio and volume_ratio <= max_volume_ratio


def _rsi_volume_hot_zone_sample(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": row.get("date") or row.get("signal_date") or row.get("entry_date"),
        "code": row.get("code"),
        "name": row.get("name"),
        "rsi": _number(row.get("rsi")),
        "volume_ratio": _number(row.get("volume_ratio")),
        "total_score": _number(row.get("total_score")),
        "profit": _record_gross_profit(row),
        "rejected_reason": row.get("rejected_reason") or row.get("reason") or row.get("rsi_volume_hot_zone_reason"),
    }


def _rsi_volume_hot_zone_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- データなし"]
    lines = [
        f"- enabled: {str(bool(audit.get('enabled'))).lower()}",
        f"- min_rsi: {_format_number(audit.get('min_rsi'))}",
        f"- min_volume_ratio: {_format_number(audit.get('min_volume_ratio'))}",
        f"- max_volume_ratio: {_format_number(audit.get('max_volume_ratio'))}",
        f"- scoring_rows_count: {audit.get('scoring_rows_count', 0)}",
        f"- matched_scoring_rows_count: {audit.get('matched_scoring_rows_count', 0)}",
        f"- rejected_count: {audit.get('rejected_count', 0)}",
        f"- rejected_trade_count: {audit.get('rejected_trade_count', 0)}",
        f"- rejected_profit_estimate: {_format_yen(audit.get('rejected_profit_estimate'))}",
        f"- gross_loss_reduction_estimate: {_format_yen(audit.get('gross_loss_reduction_estimate'))}",
        f"- missed_profit_estimate: {_format_yen(audit.get('missed_profit_estimate'))}",
        f"- note: {audit.get('note')}",
        "",
        "### sample rows",
        "",
        *_rsi_volume_hot_zone_sample_lines(audit.get("sample_rows", [])),
    ]
    return lines


def _rsi_volume_hot_zone_sample_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| date | code | name | rsi | volume_ratio | total_score | profit | rejected_reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    if not rows:
        lines.append("| - | - | - | N/A | N/A | N/A | N/A | - |")
        return lines
    for row in rows:
        lines.append(
            "| "
            f"{row.get('date') or ''} | "
            f"{row.get('code') or ''} | "
            f"{row.get('name') or ''} | "
            f"{_format_number(row.get('rsi'))} | "
            f"{_format_number(row.get('volume_ratio'))} | "
            f"{_format_number(row.get('total_score'))} | "
            f"{_format_yen(row.get('profit'))} | "
            f"{row.get('rejected_reason') or ''} |"
        )
    return lines


def _key_value_lines(values: dict[str, Any]) -> list[str]:
    if not values:
        return ["- データなし"]
    return [f"- {key}: {value}" for key, value in values.items()]


def _api_field_usage_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- データなし"]
    adjusted = audit.get("adjusted_price_usage", {})
    summary = audit.get("summary", {})
    lines = [
        f"- source: {audit.get('source')}",
        f"- total_fields: {summary.get('total_fields', 0)}",
        f"- scoring_used_count: {summary.get('scoring_used_count', 0)}",
        f"- unused_or_future_count: {summary.get('unused_or_future_count', 0)}",
        "",
        "### Adjusted Price Usage",
        "",
        f"- raw_adjusted_fields_present: {adjusted.get('raw_adjusted_fields_present')}",
        f"- processed_adjusted_fields_present: {adjusted.get('processed_adjusted_fields_present')}",
        f"- ma_basis: {adjusted.get('ma_basis')}",
        f"- rsi_basis: {adjusted.get('rsi_basis')}",
        f"- relative_strength_stock_return_basis: {adjusted.get('relative_strength_stock_return_basis')}",
        f"- candlestick_basis: {adjusted.get('candlestick_basis')}",
        f"- volume_ratio_basis: {adjusted.get('volume_ratio_basis')}",
        f"- turnover_value_usage: {adjusted.get('turnover_value_usage')}",
        f"- limit_up_down_saved: {adjusted.get('limit_up_down_saved')}",
        f"- recommendation: {adjusted.get('recommendation')}",
        "",
        "### Field Matrix",
        "",
        "| endpoint | API field | raw field | processed field | scoring | screening | filter | report/audit | unused | missing | future leak risk |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in audit.get("field_usage", []):
        lines.append(
            "| "
            + " | ".join(
                _escape_table_cell(row.get(key, ""))
                for key in [
                    "endpoint",
                    "api_response_field",
                    "raw_field",
                    "processed_field",
                    "scoring_used",
                    "screening_used",
                    "filter_used",
                    "report_or_audit_only",
                    "unused",
                    "missing_handling",
                    "future_leak_risk",
                ]
            )
            + " |"
        )
    financial = audit.get("financial_context_audit", {})
    lines.extend(
        [
            "",
            "### Financial Context Audit",
            "",
            f"- data_exists: {financial.get('data_exists')}",
            f"- record_count: {financial.get('record_count', 0)}",
            f"- feature_enabled: {financial.get('feature_enabled')}",
            f"- scoring_enabled: {financial.get('scoring_enabled')}",
            f"- selected_code_count: {financial.get('selected_code_count', 0)}",
            f"- joinable_selected_code_count: {financial.get('joinable_selected_code_count', 0)}",
            f"- available_fields: {', '.join(financial.get('available_fields', [])[:40])}",
            f"- future_data_leak_risk: {financial.get('future_data_leak_risk')}",
            f"- future_data_leak_sample_count: {financial.get('future_data_leak_sample_count', 0)}",
            "",
            "### Disabled By Default Candidates",
            "",
        ]
    )
    disabled = audit.get("disabled_by_default_candidates", {})
    for key, value in disabled.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "### Future Candidates", ""])
    for item in audit.get("future_candidates", []):
        lines.append(
            f"- {item.get('endpoint')}: {item.get('purpose')} "
            f"(status={item.get('status')}, future_leak_risk={item.get('future_leak_risk')})"
        )
    return lines


def _escape_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _first_payload_from_files(directory: Path, pattern: str, start_date: str | None, end_date: str | None) -> dict[str, Any]:
    if not directory.exists():
        return {}
    for path in sorted(directory.glob(pattern)):
        date_text = _date_from_filename(path)
        if start_date and date_text and date_text < start_date:
            continue
        if end_date and date_text and date_text > end_date:
            continue
        payload = _read_json_if_exists(path)
        if payload:
            return payload
    return {}


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _first_record(payload: dict[str, Any], key: str) -> dict[str, Any]:
    records = payload.get(key)
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict):
                return record
    return {}


def _date_from_filename(path: Path) -> str:
    stem = path.stem
    for prefix in ["prices_", "indicators_", "candidates_", "scored_candidates_"]:
        if stem.startswith(prefix):
            return stem.removeprefix(prefix)
    return ""


def _financial_records_for_period(root: Path, start_date: str | None, end_date: str | None) -> list[dict[str, Any]]:
    base = root / "data" / "cache" / "jquants" / "financial_statements"
    if not base.exists():
        return []
    records = []
    for path in sorted(base.glob("*.json")):
        payload = _read_json_if_exists(path)
        for record in payload.get("records", []):
            if not isinstance(record, dict):
                continue
            disclosed = str(_first_existing(record, ["DisclosedDate", "disclosed_date", "Date", "date"]) or "")
            if start_date and disclosed and disclosed < start_date:
                continue
            if end_date and disclosed and disclosed > end_date:
                continue
            records.append(record)
    return records


def _first_existing(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _profit_if_kept_value(row: dict[str, Any]) -> float | None:
    for key in ["profit_if_kept", "future_profit_if_kept", "future_profit", "future_profit_10d", "future_return_profit_10d"]:
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _top_investor_context_candidates(records: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda item: (
            _number(item.get("investor_context_score")) if _number(item.get("investor_context_score")) is not None else -999,
            _number(item.get("overseas_net_buy_4w_sum")) if _number(item.get("overseas_net_buy_4w_sum")) is not None else -999999999999,
        ),
        reverse=True,
    )
    rows = []
    for item in ranked[:limit]:
        rows.append(
            {
                "date": item.get("date") or item.get("entry_date") or "",
                "code": item.get("code") or "",
                "investor_context_score": item.get("investor_context_score"),
                "overseas_net_buy_4w_sum": item.get("overseas_net_buy_4w_sum"),
                "trend": item.get("overseas_net_buy_4w_trend") or "unknown",
                "selected": bool(item.get("selected", True)),
                "result": item.get("result") or "",
                "profit": _record_profit(item),
            }
        )
    return rows


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


def _relative_strength_effect_analysis(
    records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    baseline_profile_id: str | None,
) -> dict[str, Any]:
    return {
        "buckets": _relative_strength_effect_buckets(records),
        "top_selected_trades": _top_relative_strength_selected_trades(records),
        "selected_vs_baseline": _relative_strength_selected_vs_baseline(records, baseline_records, baseline_profile_id),
    }


def _relative_strength_effect_buckets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [
        ("relative_strength_score = 0", [record for record in records if (_number(record.get("relative_strength_score")) or 0) <= 0]),
        ("relative_strength_score 1-3", [record for record in records if 0 < (_number(record.get("relative_strength_score")) or 0) <= 3]),
        ("relative_strength_score 4-6", [record for record in records if 3 < (_number(record.get("relative_strength_score")) or 0) <= 6]),
        ("relative_strength_score 7-9", [record for record in records if 6 < (_number(record.get("relative_strength_score")) or 0) < 10]),
        ("relative_strength_score 10", [record for record in records if (_number(record.get("relative_strength_score")) or 0) >= 10]),
    ]
    rows = []
    for bucket, items in groups:
        stats = _group_stats(bucket, items)
        stats["profit_factor"] = _profit_factor(items)
        rows.append(stats)
    return rows


def _top_relative_strength_selected_trades(records: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda item: (
            _number(item.get("relative_strength_score")) if _number(item.get("relative_strength_score")) is not None else -999,
            _record_profit(item) if _record_profit(item) is not None else -999999999,
        ),
        reverse=True,
    )
    return [
        {
            "date": item.get("signal_date") or item.get("date") or item.get("entry_date"),
            "code": item.get("code"),
            "relative_strength_score": item.get("relative_strength_score"),
            "rs5": item.get("relative_strength_5d"),
            "rs10": item.get("relative_strength_10d"),
            "rs20": item.get("relative_strength_20d"),
            "selected": bool(item.get("selected", True)),
            "result": item.get("result") or "",
            "profit": _record_profit(item),
        }
        for item in ranked[:limit]
    ]


def _relative_strength_selected_vs_baseline(
    records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    baseline_profile_id: str | None,
) -> dict[str, Any]:
    target_by_key = {_trade_selection_key(record): record for record in records if _trade_selection_key(record)}
    baseline_by_key = {_trade_selection_key(record): record for record in baseline_records if _trade_selection_key(record)}
    newly_selected_keys = sorted(set(target_by_key) - set(baseline_by_key))
    removed_keys = sorted(set(baseline_by_key) - set(target_by_key))
    newly_selected_profit = round(sum((_record_profit(target_by_key[key]) or 0.0) for key in newly_selected_keys), 2)
    removed_profit_if_kept = round(sum((_record_profit(baseline_by_key[key]) or 0.0) for key in removed_keys), 2)
    return {
        "baseline_profile_id": baseline_profile_id,
        "newly_selected_count": len(newly_selected_keys),
        "removed_count": len(removed_keys),
        "newly_selected_profit": newly_selected_profit,
        "removed_profit_if_kept": removed_profit_if_kept,
        "net_selection_effect_profit": round(newly_selected_profit - removed_profit_if_kept, 2),
    }


def _trade_selection_key(record: dict[str, Any]) -> tuple[str, str] | None:
    code = str(record.get("code") or "")
    date_text = str(record.get("signal_date") or record.get("date") or record.get("entry_date") or "")
    if not code or not date_text:
        return None
    return date_text, code


def _relative_strength_effect_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    lines = [
        "| bucket | count | win_rate | avg_profit_rate | PF | total_profit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis.get("buckets", []):
        lines.append(_effect_bucket_table_row(row))
    lines.extend(["", "## Top Relative Strength Selected Trades", ""])
    lines.extend(_top_relative_strength_trade_lines(analysis.get("top_selected_trades", [])))
    comparison = analysis.get("selected_vs_baseline", {})
    lines.extend(["", "## Relative Strength Selected vs Baseline", ""])
    if comparison:
        lines.extend(
            [
                f"- baseline_profile_id: {comparison.get('baseline_profile_id') or 'N/A'}",
                f"- newly_selected_count: {comparison.get('newly_selected_count', 0)}",
                f"- removed_count: {comparison.get('removed_count', 0)}",
                f"- newly_selected_profit: {_format_yen(comparison.get('newly_selected_profit'))}",
                f"- removed_profit_if_kept: {_format_yen(comparison.get('removed_profit_if_kept'))}",
                f"- net_selection_effect_profit: {_format_yen(comparison.get('net_selection_effect_profit'))}",
            ]
        )
    else:
        lines.append("- データなし")
    return lines


def _effect_bucket_table_row(row: dict[str, Any]) -> str:
    pf = row.get("profit_factor")
    pf_text = "inf" if pf == float("inf") else _format_number(pf)
    return (
        "| "
        f"{row.get('bucket', '')} | "
        f"{row.get('count', 0)} | "
        f"{_format_percent(row.get('win_rate'))} | "
        f"{_format_percent(row.get('average_profit_rate'))} | "
        f"{pf_text} | "
        f"{_format_yen(row.get('total_profit'))} |"
    )


def _effect_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| bucket | count | win_rate | avg_profit | avg_profit_rate | total_profit | PF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| - | 0 | N/A | N/A | N/A | N/A | N/A |")
        return lines
    for row in rows:
        lines.append(
            "| "
            f"{row.get('bucket', row.get('reason', ''))} | "
            f"{row.get('count', 0)} | "
            f"{_format_percent(row.get('win_rate'))} | "
            f"{_format_yen(row.get('average_profit'))} | "
            f"{_format_percent(row.get('average_profit_rate'))} | "
            f"{_format_yen(row.get('total_profit'))} | "
            f"{_format_profit_factor(row.get('profit_factor'))} |"
        )
    return lines


def _trade_profit_distribution_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    lines = [
        f"- total_trades: {analysis.get('total_trades', 0)}",
        f"- closed_trade_count: {analysis.get('closed_trade_count', 0)}",
        f"- avg_profit: {_format_yen(analysis.get('avg_profit'))}",
        f"- avg_profit_rate: {_format_percent(analysis.get('avg_profit_rate'))}",
        f"- median_profit_rate: {_format_percent(analysis.get('median_profit_rate'))}",
        f"- win_rate: {_format_percent(analysis.get('win_rate'))}",
        f"- profit_factor: {_format_profit_factor(analysis.get('profit_factor'))}",
        f"- computed_trade_profit_factor: {_format_profit_factor(analysis.get('computed_trade_profit_factor'))}",
        f"- gross_profit_sum: {_format_yen(analysis.get('gross_profit_sum'))}",
        f"- net_profit_sum: {_format_yen(analysis.get('net_profit_sum'))}",
        f"- summary_gross_cumulative_profit: {_format_yen(analysis.get('summary_gross_cumulative_profit'))}",
        f"- summary_net_cumulative_profit: {_format_yen(analysis.get('summary_net_cumulative_profit'))}",
        f"- average_holding_days: {_format_number(analysis.get('average_holding_days'))}",
        f"- median_holding_days: {_format_number(analysis.get('median_holding_days'))}",
        f"- max_win_profit_rate: {_format_percent(analysis.get('max_win_profit_rate'))}",
        f"- max_loss_profit_rate: {_format_percent(analysis.get('max_loss_profit_rate'))}",
        f"- largest_win_profit: {_format_yen(analysis.get('largest_win_profit'))}",
        f"- largest_loss_profit: {_format_yen(analysis.get('largest_loss_profit'))}",
        "",
        "### Profit Rate Histogram",
        "",
        *_effect_table_lines(analysis.get("histogram", [])),
        "",
        "### Top Winners",
        "",
        *_trade_table_lines(analysis.get("top_winners", [])),
        "",
        "### Top Losers",
        "",
        *_trade_table_lines(analysis.get("top_losers", [])),
    ]
    return lines


def _exit_reason_analysis_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| reason | count | win_rate | avg_profit | avg_profit_rate | total_profit | avg_holding_days |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| - | 0 | N/A | N/A | N/A | N/A | N/A |")
        return lines
    for row in rows:
        lines.append(
            "| "
            f"{row.get('reason', row.get('bucket', ''))} | "
            f"{row.get('count', 0)} | "
            f"{_format_percent(row.get('win_rate'))} | "
            f"{_format_yen(row.get('average_profit'))} | "
            f"{_format_percent(row.get('average_profit_rate'))} | "
            f"{_format_yen(row.get('total_profit'))} | "
            f"{_format_number(row.get('avg_holding_days'))} |"
        )
    return lines


def _profit_concentration_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    keys = [
        "top_5_profit_sum",
        "top_10_profit_sum",
        "top_20_profit_sum",
        "bottom_5_loss_sum",
        "bottom_10_loss_sum",
        "bottom_20_loss_sum",
        "net_profit_without_top_5",
        "net_profit_without_top_10",
        "net_profit_without_top_20",
        "net_profit_without_bottom_5",
        "net_profit_without_bottom_10",
        "net_profit_without_bottom_20",
    ]
    return [f"- {key}: {_format_yen(analysis.get(key))}" for key in keys]


def _profit_consistency_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- データなし"]
    lines = [
        f"- status: {audit.get('status')}",
        f"- true_5y_profit_source: {audit.get('true_5y_profit_source')}",
        f"- true_5y_profit: {_format_yen(audit.get('true_5y_profit'))}",
        f"- mark_to_market_profit: {_format_yen(audit.get('mark_to_market_profit'))}",
        f"- realized_gross_profit: {_format_yen(audit.get('realized_gross_profit'))}",
        f"- realized_period_tax_net_profit: {_format_yen(audit.get('realized_period_tax_net_profit'))}",
        f"- open_position_unrealized_profit: {_format_yen(audit.get('open_position_unrealized_profit'))}",
        f"- cause: {audit.get('cause')}",
        "",
        "| source | trade_count | gross_profit | gross_loss | net_profit | PF | basis |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in audit.get("source_rows", []) or []:
        lines.append(
            "| "
            f"{row.get('source')} | "
            f"{row.get('trade_count', 0)} | "
            f"{_format_yen(row.get('gross_profit'))} | "
            f"{_format_yen(row.get('gross_loss'))} | "
            f"{_format_yen(row.get('net_profit'))} | "
            f"{_format_profit_factor(row.get('pf'))} | "
            f"{row.get('basis', '')} |"
        )
    for warning in audit.get("warnings", []) or []:
        lines.append(f"- audit_warning: {warning}")
    for note in audit.get("notes", []) or []:
        lines.append(f"- note: {note}")
    return lines


def _trade_expectancy_deep_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    return [
        f"- basis: {analysis.get('basis')}",
        f"- trade_count: {analysis.get('trade_count', 0)}",
        f"- win_count: {analysis.get('win_count', 0)}",
        f"- loss_count: {analysis.get('loss_count', 0)}",
        f"- win_rate: {_format_percent(analysis.get('win_rate'))}",
        f"- profit_factor: {_format_profit_factor(analysis.get('profit_factor'))}",
        f"- 平均利益率: {_format_percent(analysis.get('average_profit_rate'))}",
        f"- 平均損失率: {_format_percent(analysis.get('average_loss_rate'))}",
        f"- 平均利益額: {_format_yen(analysis.get('average_profit_amount'))}",
        f"- 平均損失額: {_format_yen(analysis.get('average_loss_amount'))}",
        f"- 勝ちトレード平均保有日数: {_format_number(analysis.get('winning_trade_average_holding_days'))}",
        f"- 負けトレード平均保有日数: {_format_number(analysis.get('losing_trade_average_holding_days'))}",
        f"- gross_profit_total: {_format_yen(analysis.get('gross_profit_total'))}",
        f"- gross_loss_total: {_format_yen(analysis.get('gross_loss_total'))}",
        f"- gross_net_profit: {_format_yen(analysis.get('gross_net_profit'))}",
        f"- estimated_tax_total: {_format_yen(analysis.get('estimated_tax_total'))}",
        f"- total_commission: {_format_yen(analysis.get('total_commission'))}",
        f"- period_tax_net_profit: {_format_yen(analysis.get('period_tax_net_profit'))}",
        f"- period_net_profit_per_trade: {_format_yen(analysis.get('period_net_profit_per_trade'))}",
    ]


def _exit_reason_profit_analysis_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| reason | count | win_rate | average_profit_rate | total_profit |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| - | 0 | N/A | N/A | N/A |")
        return lines
    for row in rows:
        lines.append(
            "| "
            f"{row.get('reason')} | "
            f"{row.get('count', 0)} | "
            f"{_format_percent(row.get('win_rate'))} | "
            f"{_format_percent(row.get('average_profit_rate'))} | "
            f"{_format_yen(row.get('total_profit'))} |"
        )
    return lines


def _profit_capture_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    return [
        f"- basis: {analysis.get('basis')}",
        f"- take_profit_count: {analysis.get('take_profit_count', 0)}",
        f"- stop_loss_count: {analysis.get('stop_loss_count', 0)}",
        f"- 利確平均獲得率: {_format_percent(analysis.get('take_profit_average_capture_rate'))}",
        f"- 損切平均損失率: {_format_percent(analysis.get('stop_loss_average_loss_rate'))}",
        f"- take_profit_average_profit: {_format_yen(analysis.get('take_profit_average_profit'))}",
        f"- stop_loss_average_profit: {_format_yen(analysis.get('stop_loss_average_profit'))}",
        "",
        "### 最大利益トレード上位20件",
        "",
        *_trade_table_lines(analysis.get("top_winners", [])),
        "",
        "### 最大損失トレード上位20件",
        "",
        *_trade_table_lines(analysis.get("top_losers", [])),
    ]


def _opportunity_loss_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    lines = [
        f"- basis: {analysis.get('basis')}",
        f"- winning_trade_count: {analysis.get('winning_trade_count', 0)}",
        f"- covered_after_5d_count: {analysis.get('covered_after_5d_count', 0)}",
        f"- covered_after_20d_count: {analysis.get('covered_after_20d_count', 0)}",
        f"- 保有終了後5営業日 平均追加上昇率: {_format_percent(analysis.get('average_after_5d_return'))}",
        f"- 保有終了後5営業日 中央追加上昇率: {_format_percent(analysis.get('median_after_5d_return'))}",
        f"- 保有終了後5営業日 プラス率: {_format_percent(analysis.get('positive_after_5d_rate'))}",
        f"- 保有終了後20営業日 平均追加上昇率: {_format_percent(analysis.get('average_after_20d_return'))}",
        f"- 保有終了後20営業日 中央追加上昇率: {_format_percent(analysis.get('median_after_20d_return'))}",
        f"- 保有終了後20営業日 プラス率: {_format_percent(analysis.get('positive_after_20d_rate'))}",
        "",
        "### Post Exit 20D Opportunity Top 20",
        "",
        *_opportunity_loss_table_lines(analysis.get("top_after_20d_opportunity_losses", [])),
    ]
    for note in analysis.get("notes", []) or []:
        lines.append(f"- note: {note}")
    return lines


def _opportunity_loss_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| exit_date | code | name | exit_price | profit | profit_rate | after_5d_date | after_5d_return | after_20d_date | after_20d_return | exit_reason |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |",
    ]
    if not rows:
        lines.append("| - | - | - |  |  |  | - |  | - |  | - |")
        return lines
    for row in rows:
        lines.append(
            "| "
            f"{row.get('exit_date') or ''} | "
            f"{row.get('code') or ''} | "
            f"{row.get('name') or ''} | "
            f"{_format_number(row.get('exit_price'))} | "
            f"{_format_yen(row.get('profit'))} | "
            f"{_format_percent(row.get('profit_rate'))} | "
            f"{row.get('after_5d_date') or ''} | "
            f"{_format_percent(row.get('after_5d_return'))} | "
            f"{row.get('after_20d_date') or ''} | "
            f"{_format_percent(row.get('after_20d_return'))} | "
            f"{row.get('exit_reason') or ''} |"
        )
    return lines


def _profit_analysis_conclusion_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- データなし"]
    return [f"{row.get('rank')}. {row.get('factor')}: {row.get('evidence')}" for row in rows]


def _stop_loss_pattern_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    lines = [
        f"- 件数: {analysis.get('count', 0)}",
        f"- 平均損失率: {_format_percent(analysis.get('average_loss_rate'))}",
        f"- 合計損失: {_format_yen(analysis.get('total_loss'))}",
    ]
    sections = [
        ("エントリー時 total_score 帯別", "total_score"),
        ("relative_strength_score 帯別", "relative_strength_score"),
        ("relative_strength_5d 帯別", "relative_strength_5d"),
        ("relative_strength_10d 帯別", "relative_strength_10d"),
        ("relative_strength_20d 帯別", "relative_strength_20d"),
        ("RSI 帯別", "rsi"),
        ("volume_ratio 帯別", "volume_ratio"),
        ("sector_name 別", "sector_name"),
        ("candlestick_signals 別", "candlestick_signals"),
        ("market_context_score 帯別", "market_context_score"),
    ]
    for title, key in sections:
        lines.extend(["", f"### {title}", ""])
        lines.extend(_pattern_table_lines(analysis.get(key, [])))
    return lines


def _max_holding_pattern_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    lines = [
        f"- 件数: {analysis.get('count', 0)}",
        f"- 勝率: {_format_percent(analysis.get('win_rate'))}",
        f"- 平均利益率: {_format_percent(analysis.get('average_profit_rate'))}",
        f"- 合計損益: {_format_yen(analysis.get('total_profit'))}",
    ]
    sections = [
        ("エントリー時 total_score 帯別", "total_score"),
        ("relative_strength_score 帯別", "relative_strength_score"),
        ("RSI 帯別", "rsi"),
        ("volume_ratio 帯別", "volume_ratio"),
        ("sector_name 別", "sector_name"),
        ("candlestick_signals 別", "candlestick_signals"),
    ]
    for title, key in sections:
        lines.extend(["", f"### {title}", ""])
        lines.extend(_pattern_table_lines(analysis.get(key, [])))
    return lines


def _pattern_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| bucket | count | win_rate | average_profit_rate | total_profit |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| - | 0 | N/A | N/A | N/A |")
        return lines
    for row in rows:
        lines.append(
            "| "
            f"{row.get('bucket', '')} | "
            f"{row.get('count', 0)} | "
            f"{_format_percent(row.get('win_rate'))} | "
            f"{_format_percent(row.get('average_profit_rate'))} | "
            f"{_format_yen(row.get('total_profit'))} |"
        )
    return lines


def _winner_vs_stop_loss_contrast_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| feature | type | category | winner_value | stop_loss_value | difference | note |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    if not rows:
        lines.append("| - | - | - | N/A | N/A | N/A | - |")
        return lines
    for row in rows:
        lines.append(
            "| "
            f"{row.get('feature')} | "
            f"{row.get('type')} | "
            f"{row.get('category', '')} | "
            f"{_format_contrast_value(row.get('winner_value'), row.get('feature'))} | "
            f"{_format_contrast_value(row.get('stop_loss_value'), row.get('feature'))} | "
            f"{_format_contrast_value(row.get('difference'), row.get('feature'))} | "
            f"{row.get('note', '')} |"
        )
    return lines


def _rule_candidate_proposal_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| rule_id | ルール内容 | 狙い | 想定される副作用 | 既存5年データ上で影響しそうな件数 | 優先度 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    if not rows:
        lines.append("| - | - | - | - | 0 | Low |")
        return lines
    for row in rows:
        extra = []
        if "estimated_stop_loss_count" in row:
            extra.append(f"stop_loss {row.get('estimated_stop_loss_count')}")
        if "estimated_loss_count" in row:
            extra.append(f"loss {row.get('estimated_loss_count')}")
        if "estimated_total_loss" in row:
            extra.append(f"total_loss {_format_yen(row.get('estimated_total_loss'))}")
        affected = str(row.get("estimated_affected_count", 0))
        if extra:
            affected = f"{affected} ({', '.join(extra)})"
        lines.append(
            "| "
            f"{row.get('rule_id')} | "
            f"{row.get('rule')} | "
            f"{row.get('aim')} | "
            f"{row.get('side_effect')} | "
            f"{affected} | "
            f"{row.get('priority')} |"
        )
    return lines


def _format_contrast_value(value: Any, feature: Any) -> str:
    if value is None:
        return "N/A"
    feature_text = str(feature or "")
    if "relative_strength_" in feature_text and feature_text != "relative_strength_score":
        return _format_percent(value)
    if feature_text in {"winner_value", "stop_loss_value"}:
        return _format_number(value)
    return _format_number(value)


def _baseline_vs_relative_strength_trade_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    lines = [
        f"- baseline_profile_id: {analysis.get('baseline_profile_id') or 'N/A'}",
        f"- target_trade_count: {analysis.get('target_trade_count', 0)}",
        f"- baseline_trade_count: {analysis.get('baseline_trade_count', 0)}",
        f"- common_trade_count: {analysis.get('common_trade_count', 0)}",
        f"- newly_selected_count: {analysis.get('newly_selected_count', 0)}",
        f"- removed_count: {analysis.get('removed_count', 0)}",
        f"- improvement_type: {analysis.get('improvement_type')}",
        "",
        "### Newly Selected Summary",
        "",
        *_trade_subset_summary_lines(analysis.get("newly_selected_summary", {})),
        "",
        "### Removed If Kept Summary",
        "",
        *_trade_subset_summary_lines(analysis.get("removed_if_kept_summary", {})),
        "",
        "### Newly Selected Profit Histogram",
        "",
        *_effect_table_lines(analysis.get("newly_selected_histogram", [])),
        "",
        "### Removed Profit Histogram",
        "",
        *_effect_table_lines(analysis.get("removed_histogram", [])),
    ]
    return lines


def _trade_subset_summary_lines(summary: dict[str, Any]) -> list[str]:
    return [
        f"- count: {summary.get('count', 0)}",
        f"- win_rate: {_format_percent(summary.get('win_rate'))}",
        f"- avg_profit: {_format_yen(summary.get('average_profit'))}",
        f"- avg_profit_rate: {_format_percent(summary.get('average_profit_rate'))}",
        f"- total_profit: {_format_yen(summary.get('total_profit'))}",
        f"- profit_factor: {_format_profit_factor(summary.get('profit_factor'))}",
        f"- avg_holding_days: {_format_number(summary.get('avg_holding_days'))}",
    ]


def _expectancy_formula_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- データなし"]
    lines = [
        f"- expectancy_formula: {audit.get('expectancy_formula')}",
        f"- expectancy_value: {_format_percent(audit.get('expectancy_value'))}",
        f"- stored_expectancy: {_format_percent(audit.get('stored_expectancy'))}",
        f"- avg_profit_per_trade: {_format_yen(audit.get('avg_profit_per_trade'))}",
        f"- avg_profit_rate_per_trade: {_format_percent(audit.get('avg_profit_rate_per_trade'))}",
        f"- net_profit / total_trades: {_format_yen(audit.get('net_profit_per_trade'))}",
        f"- win_rate: {_format_percent(audit.get('win_rate'))}",
        f"- average_win_profit_rate: {_format_percent(audit.get('average_win_profit_rate'))}",
        f"- average_loss_profit_rate: {_format_percent(audit.get('average_loss_profit_rate'))}",
    ]
    for warning in audit.get("warnings", []) or []:
        lines.append(f"- audit_warning: {warning}")
    return lines


def _trade_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| entry_date | exit_date | code | name | entry_price | exit_price | profit | profit_rate | holding_days | exit_reason | total_score | score_components |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    if not rows:
        lines.append("| - | - | - | - |  |  |  |  |  | - |  | - |")
        return lines
    for row in rows:
        components = {
            key: row.get(key)
            for key in ["technical_score", "ma_score", "rsi_score", "volume_score", "candlestick_score", "relative_strength_score"]
            if row.get(key) is not None
        }
        lines.append(
            "| "
            f"{row.get('entry_date') or ''} | "
            f"{row.get('exit_date') or ''} | "
            f"{row.get('code') or ''} | "
            f"{row.get('name') or ''} | "
            f"{_format_number(row.get('entry_price'))} | "
            f"{_format_number(row.get('exit_price'))} | "
            f"{_format_yen(row.get('profit'))} | "
            f"{_format_percent(row.get('profit_rate'))} | "
            f"{_format_number(row.get('holding_days'))} | "
            f"{row.get('exit_reason') or ''} | "
            f"{_format_number(row.get('total_score'))} | "
            f"{_compact_json(components)} |"
        )
    return lines


def _top_relative_strength_trade_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| date | code | relative_strength_score | rs5 | rs10 | rs20 | selected | result | profit |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    if not rows:
        lines.append("| なし |  |  |  |  |  |  |  |  |")
        return lines
    for row in rows:
        lines.append(
            "| "
            f"{row.get('date', '')} | "
            f"{row.get('code', '')} | "
            f"{_format_number(row.get('relative_strength_score'))} | "
            f"{_format_percent(row.get('rs5'))} | "
            f"{_format_percent(row.get('rs10'))} | "
            f"{_format_percent(row.get('rs20'))} | "
            f"{str(bool(row.get('selected'))).lower()} | "
            f"{row.get('result', '')} | "
            f"{_format_yen(row.get('profit'))} |"
        )
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
    lines.extend(["", "### Top Investor Context Candidates", ""])
    lines.extend(_top_investor_context_candidate_lines(analysis.get("top_candidates", [])))
    lines.extend(["", "### Investor Context Effect Analysis", ""])
    lines.extend(_investor_context_effect_lines(analysis.get("effect_analysis", [])))
    return lines


def _top_investor_context_candidate_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- データなし"]
    lines = [
        "| date | code | investor_context_score | overseas_net_buy_4w_sum | trend | selected | result | profit |",
        "| --- | --- | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('date', '')} | "
            f"{row.get('code', '')} | "
            f"{_format_number(row.get('investor_context_score'))} | "
            f"{_format_number_with_commas(row.get('overseas_net_buy_4w_sum'))} | "
            f"{row.get('trend', '')} | "
            f"{str(bool(row.get('selected'))).lower()} | "
            f"{row.get('result', '')} | "
            f"{_format_yen(row.get('profit'))} |"
        )
    return lines


def _format_number_with_commas(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):,.2f}"


def _investor_context_effect_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- データなし"]
    lines = [
        "| bucket | count | win_rate | avg_profit_rate | PF | total_profit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        pf = row.get("profit_factor")
        pf_text = "inf" if pf == float("inf") else _format_number(pf)
        lines.append(
            "| "
            f"{row.get('bucket', '')} | "
            f"{row.get('count', 0)} | "
            f"{_format_percent(row.get('win_rate'))} | "
            f"{_format_percent(row.get('average_profit_rate'))} | "
            f"{pf_text} | "
            f"{_format_yen(row.get('total_profit'))} |"
        )
    return lines


def _investor_context_filter_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    codes = analysis.get("rejected_codes") or []
    code_text = ", ".join(str(code) for code in codes[:30]) if codes else "なし"
    if len(codes) > 30:
        code_text = f"{code_text}, ... (+{len(codes) - 30})"
    return [
        f"- rejected_count: {analysis.get('rejected_count', 0)}",
        f"- rejected_codes: {code_text}",
        f"- rejected_profit_if_kept: {_format_yen(analysis.get('rejected_profit_if_kept'))}",
        f"- accepted_profit: {_format_yen(analysis.get('accepted_profit'))}",
    ]


def _earnings_calendar_exposure_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- データなし"]
    return [
        f"- selected銘柄のうち決算前後だった件数: {analysis.get('selected_earnings_exposure_count', 0)}",
        f"- stop_loss銘柄のうち決算前後だった件数: {analysis.get('stop_loss_earnings_exposure_count', 0)}",
        f"- false_positive銘柄のうち決算前後だった件数: {analysis.get('false_positive_earnings_exposure_count', 0)}",
    ]


def _earnings_filter_debug_lines(debug: dict[str, Any]) -> list[str]:
    if not debug:
        return ["- データなし"]
    lines = [
        f"- status: {debug.get('status') or 'unknown'}",
        f"- earnings_calendar_records: {debug.get('earnings_calendar_records', 0)}",
        f"- candidate_count: {debug.get('candidate_count', 0)}",
        f"- earnings_info_found_count: {debug.get('earnings_info_found_count', 0)}",
        f"- earnings_info_missing_count: {debug.get('earnings_info_missing_count', 0)}",
        f"- earnings_filter_candidate_count: {debug.get('earnings_filter_candidate_count', 0)}",
        f"- earnings_filter_rejected_count: {debug.get('earnings_filter_rejected_count', 0)}",
        f"- earnings_filter_applied_count: {debug.get('earnings_filter_applied_count', 0)}",
        f"- unknown_earnings_count: {debug.get('unknown_earnings_count', 0)}",
        f"- selection_diff_relation: {debug.get('selection_diff_relation') or 'N/A'}",
        "",
        "### warnings",
        "",
    ]
    warnings = debug.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- なし")
    distribution = debug.get("days_to_earnings_distribution", {})
    lines.extend(
        [
            "",
            "### days_to_earnings distribution",
            "",
            f"- <= -3: {distribution.get('<= -3', 0)}",
            f"- -2 to +2: {distribution.get('-2 to +2', 0)}",
            f"- +3 to +7: {distribution.get('+3 to +7', 0)}",
            f"- +8 to +14: {distribution.get('+8 to +14', 0)}",
            f"- +15+: {distribution.get('+15+', 0)}",
            f"- unknown: {distribution.get('unknown', 0)}",
            "",
            "### nearest earnings candidates",
            "",
            "| date | code | company_name | earnings_date | days_to_earnings | action |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    nearest = debug.get("nearest_earnings_candidates", [])
    if nearest:
        for row in nearest:
            lines.append(
                "| "
                f"{row.get('date')} | {row.get('code')} | {row.get('company_name') or ''} | "
                f"{row.get('earnings_date') or ''} | {_format_number(row.get('days_to_earnings'))} | {row.get('action') or ''} |"
            )
    else:
        lines.append("| なし |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "### Top rejected candidates",
            "",
            "| date | code | earnings_date | days_to_earnings | reason |",
            "|---|---:|---|---:|---|",
        ]
    )
    top = debug.get("top_rejected_candidates", [])
    if top:
        for row in top:
            lines.append(
                "| "
                f"{row.get('date')} | {row.get('code')} | {row.get('earnings_date') or ''} | "
                f"{_format_number(row.get('days_to_earnings'))} | {row.get('reason') or ''} |"
            )
    else:
        lines.append("| なし |  |  |  |  |")
    return lines


def _earnings_pipeline_lines(pipeline: dict[str, Any]) -> list[str]:
    if not pipeline:
        return ["- データなし"]
    return [
        f"- feature enabled: {str(bool(pipeline.get('feature_enabled'))).lower()}",
        f"- fetch_start: {pipeline.get('fetch_start') or 'N/A'}",
        f"- fetch_end: {pipeline.get('fetch_end') or 'N/A'}",
        f"- cache path: {pipeline.get('cache_path') or 'N/A'}",
        f"- cache exists: {str(bool(pipeline.get('cache_exists'))).lower()}",
        f"- cache records: {pipeline.get('cache_records', 0)}",
        f"- cache loaded: {str(bool(pipeline.get('cache_loaded'))).lower()}",
        f"- index built: {str(bool(pipeline.get('index_built'))).lower()}",
        f"- candidate matching called: {str(bool(pipeline.get('candidate_matching_called'))).lower()}",
        f"- earnings records loaded: {pipeline.get('earnings_records_loaded', 0)}",
        f"- matched candidates: {pipeline.get('matched_candidates', 0)}",
        f"- rejected candidates: {pipeline.get('rejected_candidates', 0)}",
        f"- reason: {pipeline.get('reason') or 'N/A'}",
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


def _record_gross_profit(item: dict[str, Any]) -> float | None:
    return _number(item.get("gross_profit")) or _number(item.get("profit"))


def _record_gross_profit_rate(item: dict[str, Any]) -> float | None:
    return _number(item.get("gross_profit_rate")) or _number(item.get("profit_rate"))


def _normalized_exit_reason(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "other"
    if "利確" in text or "take" in text:
        return "take_profit"
    if "損切" in text or "stop" in text:
        return "stop_loss"
    if "最大保有" in text or "max_holding" in text or "holding" in text:
        return "max_holding_days"
    if "market" in text or "forced" in text or "close" in text or "期間終了" in text:
        return "market_exit"
    return "other"


def _normalize_code(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _indicator_date_paths(root: Path, profile_id: str, start_date: str | None, end_date: str | None) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    dirs = [root / "data" / "processed" / profile_id]
    common_root = root / "data" / "processed" / "common" / "indicators"
    if common_root.exists():
        dirs.extend(sorted(path for path in common_root.iterdir() if path.is_dir()))
    for directory in dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("indicators_*.json")):
            date_value = path.stem.replace("indicators_", "")
            if start_date and date_value < start_date:
                continue
            if end_date and date_value > end_date:
                continue
            candidates.setdefault(date_value, path)
    return candidates


def _indicator_close_map(path: Path | None) -> dict[str, float]:
    if not path or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("indicators") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}
    result: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = _normalize_code(row.get("code") or row.get("Code"))
        close = _number(row.get("adjusted_close")) or _number(row.get("adj_close")) or _number(row.get("AdjC")) or _number(row.get("close")) or _number(row.get("C"))
        if code and close is not None:
            result[code] = close
    return result


def _date_index(dates: list[str], date_value: str) -> int | None:
    if not dates:
        return None
    for index, candidate in enumerate(dates):
        if candidate > date_value:
            return index - 1 if index > 0 and dates[index - 1] == date_value else index - 1 if index > 0 else 0
        if candidate == date_value:
            return index
    return len(dates) - 1


def _opportunity_loss_notes(winning_count: int, after_5_count: int, after_20_count: int) -> list[str]:
    notes = [
        "This is an audit-only post-exit mark using existing indicator close data; it does not change backtest decisions.",
        "If adjusted_close is absent, indicator close is used.",
    ]
    if after_5_count < winning_count:
        notes.append(f"after_5d coverage is partial: {after_5_count}/{winning_count}")
    if after_20_count < winning_count:
        notes.append(f"after_20d coverage is partial: {after_20_count}/{winning_count}")
    return notes


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


def _backtest_integrity_audits(root: Path, profile_id: str, start_date: str | None, end_date: str | None) -> dict[str, Any]:
    if not start_date or not end_date:
        return {}
    summary_path = root / "logs" / "backtests" / profile_id / f"{start_date}_to_{end_date}" / "backtest_summary.json"
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return {}
    result_audit = _repair_backtest_result_integrity_audit(root, profile_id, start_date, end_date, payload)
    return {
        "market_filter_audit": payload.get("market_filter_audit", {}),
        "backtest_result_integrity_audit": result_audit,
        "score_integrity_audit": payload.get("score_integrity_audit", {}),
    }


def _repair_backtest_result_integrity_audit(
    root: Path,
    profile_id: str,
    start_date: str,
    end_date: str,
    backtest_summary: dict[str, Any],
) -> dict[str, Any]:
    audit = dict(backtest_summary.get("backtest_result_integrity_audit", {}) or {})
    all_trades = backtest_summary.get("all_trades", [])
    if not isinstance(all_trades, list):
        all_trades = []
    scored_by_key, selected_by_key, key_source = _processed_scored_candidate_keys(root, profile_id, start_date, end_date)
    buy_trades = [
        trade
        for trade in all_trades
        if isinstance(trade, dict)
        and str(trade.get("action") or "").upper() == "BUY"
        and str(trade.get("order_status") or trade.get("status") or "FILLED").upper() == "FILLED"
    ]
    run_dates = sorted({key.split("|", 1)[0] for key in set(scored_by_key) | set(selected_by_key) if "|" in key})
    buy_keys = {_feature_trade_selection_key(trade) for trade in buy_trades if _feature_trade_selection_key(trade)}
    missing_keys = sorted(buy_keys - set(selected_by_key))
    debug_sample = _feature_trade_without_selected_debug_sample(
        missing_keys,
        buy_trades,
        scored_by_key,
        selected_by_key,
        key_source,
        run_dates,
        root,
        profile_id,
        start_date,
        end_date,
    )
    audit["trade_without_selected_count"] = len(missing_keys)
    audit["trade_without_selected_sample"] = missing_keys[:20]
    audit["trade_without_selected_debug_sample"] = debug_sample
    audit["market_trade_samples"] = _feature_market_trade_samples(all_trades, selected_by_key, set(missing_keys))
    if buy_trades:
        audit["trade_selected_match_rate"] = round((len(buy_trades) - len(missing_keys)) / len(buy_trades), 4)
    warnings = [
        item
        for item in audit.get("warnings", []) or []
        if "buy trade exists without selected candidate" not in str(item)
    ]
    errors = list(audit.get("errors", []) or [])
    if missing_keys:
        warnings.append("buy trade exists without selected candidate in the run period")
    audit["warnings"] = warnings
    audit["errors"] = errors
    audit["integrity_warning_count"] = len(warnings)
    audit["integrity_error_count"] = len(errors)
    audit["result_integrity_status"] = "WARNING" if warnings or errors else "OK"
    return audit


def _feature_market_trade_samples(
    all_trades: list[Any],
    selected_by_key: dict[str, dict[str, Any]],
    missing_keys: set[str],
) -> list[dict[str, Any]]:
    samples = []
    selected_keys = set(selected_by_key)
    buy_trades = [
        trade
        for trade in all_trades
        if isinstance(trade, dict) and str(trade.get("action") or "").upper() == "BUY"
    ]
    sell_trades = [
        trade
        for trade in all_trades
        if isinstance(trade, dict) and str(trade.get("action") or "").upper() == "SELL"
    ]
    for trade in [*buy_trades, *sell_trades]:
        if not isinstance(trade, dict):
            continue
        status = str(trade.get("order_status") or trade.get("status") or "FILLED").upper()
        if status != "FILLED":
            continue
        key = _feature_trade_selection_key(trade)
        selected_found = bool(key and key in selected_keys)
        reasons = []
        if key in missing_keys:
            reasons.append("trade_without_selected")
        section = (
            trade.get("market_section")
            or trade.get("listing_market")
            or trade.get("section")
            or (selected_by_key.get(key or "") or {}).get("market_section")
            or "Unknown"
        )
        samples.append(
            {
                "trade_date": trade.get("entry_date") or trade.get("exit_date") or trade.get("date"),
                "code": trade.get("code"),
                "name": trade.get("name"),
                "market_section": section,
                "selected_found": selected_found,
                "reason": ",".join(reasons) or "ok",
            }
        )
        if len(samples) >= 20:
            break
    return samples


def _processed_scored_candidate_keys(
    root: Path,
    profile_id: str,
    start_date: str,
    end_date: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    scored_by_key: dict[str, dict[str, Any]] = {}
    selected_by_key: dict[str, dict[str, Any]] = {}
    key_source: dict[str, str] = {}
    processed_dir = root / "data" / "processed" / profile_id
    for path in sorted(processed_dir.glob("scored_candidates_*.json")):
        day = path.stem.replace("scored_candidates_", "")
        if day < start_date or day > end_date:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        scores = payload.get("scores", [])
        if not isinstance(scores, list):
            scores = []
        selected = payload.get("selected", [])
        if not selected:
            selected = [row for row in scores if isinstance(row, dict) and row.get("selected")]
        if not isinstance(selected, list):
            selected = []
        source = str(path.relative_to(root))
        for row in scores:
            if not isinstance(row, dict):
                continue
            key = _feature_selection_key(row, day)
            if key:
                scored_by_key[key] = row
                key_source.setdefault(key, source)
        for row in selected:
            if not isinstance(row, dict):
                continue
            key = _feature_selection_key(row, day)
            if key:
                selected_by_key[key] = row
                key_source[key] = source
    return scored_by_key, selected_by_key, key_source


def _feature_selection_key(row: dict[str, Any], fallback_day: str = "") -> str:
    code = str(row.get("code") or "")
    day = str(row.get("signal_date") or row.get("date") or fallback_day or "")
    return f"{day}|{code}" if day and code else ""


def _feature_trade_selection_key(trade: dict[str, Any]) -> str:
    code = str(trade.get("code") or "")
    day = str(trade.get("signal_date") or trade.get("date") or "")
    return f"{day}|{code}" if day and code else ""


def _feature_trade_lookup_keys(trade: dict[str, Any], run_dates: list[str]) -> list[str]:
    code = str(trade.get("code") or "")
    if not code:
        return []
    days = [
        str(trade.get("signal_date") or ""),
        str(trade.get("date") or ""),
        str(trade.get("trade_date") or ""),
        str(trade.get("entry_date") or ""),
    ]
    entry_date = str(trade.get("entry_date") or trade.get("trade_date") or "")
    previous = [day for day in run_dates if day < entry_date]
    if previous:
        days.append(previous[-1])
    keys = []
    seen = set()
    for day in days:
        if not day:
            continue
        key = f"{day}|{code}"
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def _feature_trade_without_selected_debug_sample(
    missing_keys: list[str],
    buy_trades: list[dict[str, Any]],
    scored_by_key: dict[str, dict[str, Any]],
    selected_by_key: dict[str, dict[str, Any]],
    key_source: dict[str, str],
    run_dates: list[str],
    root: Path,
    profile_id: str,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    by_key = {_feature_trade_selection_key(trade): trade for trade in buy_trades if _feature_trade_selection_key(trade)}
    samples = []
    for key in missing_keys[:20]:
        trade = by_key.get(key, {})
        lookup_keys = _feature_trade_lookup_keys(trade, run_dates)
        scored_key = next((item for item in lookup_keys if item in scored_by_key), "")
        selected_key = next((item for item in lookup_keys if item in selected_by_key), "")
        samples.append(
            {
                "code": trade.get("code"),
                "signal_date": trade.get("signal_date"),
                "trade_date": trade.get("trade_date") or trade.get("entry_date") or trade.get("date"),
                "entry_date": trade.get("entry_date"),
                "selected_lookup_keys_checked": lookup_keys,
                "found_in_scored_candidates": bool(scored_key),
                "found_in_selected_candidates": bool(selected_key),
                "matched_scored_key": scored_key,
                "matched_selected_key": selected_key,
                "source_log_file": str((root / "logs" / "backtests" / profile_id / f"{start_date}_to_{end_date}" / "backtest_summary.json").relative_to(root)),
                "scored_candidate_file_checked": key_source.get(scored_key or selected_key, ""),
            }
        )
    return samples


def _capital_utilization_audit(
    root: Path,
    profile_id: str,
    start_date: str | None,
    end_date: str | None,
    backtest_summary: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not start_date or not end_date:
        return {"status": "unavailable", "reason": "start_date/end_date are required"}
    log_dir = root / "logs" / "backtests" / profile_id / f"{start_date}_to_{end_date}"
    summary_csv = log_dir / "summary.csv"
    if not summary_csv.exists() and not backtest_summary:
        return {"status": "unavailable", "reason": "backtest summary artifacts not found"}

    cash_ratios: list[float] = []
    exposure_ratios: list[float] = []
    position_counts: list[float] = []
    target_exposure = None
    if isinstance(config, dict):
        policy = config.get("capital_utilization_policy", {})
        if isinstance(policy, dict):
            target_exposure = _number(policy.get("target_exposure"))
    target_exposure_reached_days = 0
    target_exposure_gaps: list[float] = []
    cash_after_buy_candidates: list[tuple[str, float]] = []
    if summary_csv.exists():
        try:
            with summary_csv.open("r", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    day = str(row.get("date") or "")
                    total_assets = _number(row.get("total_assets"))
                    cash = _number(row.get("cash"))
                    positions_value = _number(row.get("positions_value"))
                    position_count = _number(row.get("open_positions_count"))
                    if total_assets and total_assets > 0:
                        if cash is not None:
                            cash_ratios.append(cash / total_assets)
                        if positions_value is not None:
                            exposure = positions_value / total_assets
                            exposure_ratios.append(exposure)
                            if target_exposure is not None and exposure >= target_exposure:
                                target_exposure_reached_days += 1
                            if target_exposure is not None:
                                target_exposure_gaps.append(max(0.0, target_exposure - exposure))
                        if day and cash is not None:
                            cash_after_buy_candidates.append((day, cash))
                    if position_count is not None:
                        position_counts.append(position_count)
        except Exception:
            pass

    events = backtest_summary.get("all_trades", []) if isinstance(backtest_summary, dict) else []
    if not isinstance(events, list):
        events = []
    skipped = [item for item in events if isinstance(item, dict) and str(item.get("action") or "") == "SKIP_BUY"]
    skipped_reasons: dict[str, int] = defaultdict(int)
    for item in skipped:
        reason = _capital_skip_reason(item.get("skipped_reason") or item.get("reason"))
        skipped_reasons[reason] += 1
    buys = [item for item in events if isinstance(item, dict) and str(item.get("action") or "") == "BUY"]
    buy_dates = {
        str(item.get("entry_date") or item.get("date") or item.get("signal_date") or "")
        for item in buys
        if item.get("entry_date") or item.get("date") or item.get("signal_date")
    }
    event_dates: dict[str, set[str]] = defaultdict(set)
    for item in events:
        if not isinstance(item, dict):
            continue
        day = str(item.get("entry_date") or item.get("date") or item.get("signal_date") or "")
        action = str(item.get("action") or "")
        if day and action:
            event_dates[day].add(action)
    order_amounts = [_number(item.get("amount") or item.get("notional")) for item in buys]
    order_quantities = [_number(item.get("shares") or item.get("quantity")) for item in buys]
    buy_as_much_as_possible_count = sum(
        1
        for item in buys
        if str(item.get("allocation_reason") or "") == "capital_utilization_policy"
        or bool(item.get("buy_as_much_as_possible"))
    )
    cash_after_buy = [cash for day, cash in cash_after_buy_candidates if day in buy_dates]
    unaffordable_reasons = {"insufficient_available_cash", "round_lot_unaffordable", "selected_but_not_affordable"}
    unaffordable_selected_count = sum(skipped_reasons.get(reason, 0) for reason in unaffordable_reasons)
    affordable_selected_count = len(buys) + sum(
        skipped_reasons.get(reason, 0)
        for reason in ["target_exposure_limit", "max_positions_limit", "risk_policy_limit"]
    )
    target_exposure_blocked_reason_breakdown = {
        reason: count
        for reason, count in sorted(skipped_reasons.items(), key=lambda item: item[0])
        if count
    }
    no_candidate_days = sum(1 for actions in event_dates.values() if "NO_BUY" in actions and "BUY" not in actions and "SKIP_BUY" not in actions)
    no_affordable_candidate_days = sum(1 for actions in event_dates.values() if "SKIP_BUY" in actions and "BUY" not in actions)

    return {
        "status": "OK",
        "source": {
            "summary_csv": str(summary_csv.relative_to(root)) if summary_csv.exists() else "",
            "backtest_summary": str((log_dir / "backtest_summary.json").relative_to(root)),
        },
        "average_cash_ratio": _average(cash_ratios),
        "average_market_exposure": _average(exposure_ratios),
        "average_position_count": _average(position_counts),
        "max_position_count": max(position_counts) if position_counts else None,
        "skipped_buy_count": len(skipped),
        "skipped_buy_reason_breakdown": dict(sorted(skipped_reasons.items(), key=lambda item: item[0])),
        "average_order_amount": _average([value for value in order_amounts if value is not None]),
        "average_order_quantity": _average([value for value in order_quantities if value is not None]),
        "buy_as_much_as_possible_count": buy_as_much_as_possible_count,
        "target_exposure_reached_days": target_exposure_reached_days,
        "target_exposure_gap_average": _average(target_exposure_gaps),
        "target_exposure_blocked_reason_breakdown": target_exposure_blocked_reason_breakdown,
        "affordable_selected_count": affordable_selected_count,
        "unaffordable_selected_count": unaffordable_selected_count,
        "cash_after_buy_average": _average(cash_after_buy),
        "position_value_cap_hit_count": int(skipped_reasons.get("selected_but_not_affordable", 0) or 0),
        "min_cash_buffer_hit_count": int(skipped_reasons.get("insufficient_available_cash", 0) or 0),
        "no_candidate_days": no_candidate_days,
        "no_affordable_candidate_days": no_affordable_candidate_days,
        "target_exposure_note": (
            "target exposure is blocked by the reasons above; no_candidate_days and no_affordable_candidate_days "
            "show days where exposure could not increase even before changing strategy logic"
        ),
    }


def _capital_skip_reason(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    lowered = text.lower()
    if "price_exceeds_single_order_limit" in lowered:
        return "selected_but_not_affordable"
    if "round_lot_unaffordable" in lowered:
        return "round_lot_unaffordable"
    if "insufficient_available_cash" in lowered or "cash_buffer_limit" in lowered:
        return "insufficient_available_cash"
    if "max_positions_limit" in lowered:
        return "max_positions_limit"
    if "risk_policy_limit" in lowered:
        return "risk_policy_limit"
    if "selected_but_not_affordable" in lowered:
        return "selected_but_not_affordable"
    if "1銘柄上限" in text or "上限を超える" in text or "max_single_order_amount" in lowered:
        return "price_exceeds_single_order_limit"
    if "1日の買付上限" in text or "daily" in lowered:
        return "risk_policy_limit"
    if "現金" in text or "余力" in text or "cash" in lowered:
        return "insufficient_available_cash"
    if "最大保有" in text or "ポジション" in text or "max_positions" in lowered:
        return "max_positions_limit"
    return text


def _capital_utilization_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- audit: unavailable"]
    if audit.get("status") != "OK":
        return [f"- status: {audit.get('status', 'unavailable')}", f"- reason: {audit.get('reason', '')}"]
    lines = [
        f"- average_cash_ratio: {_format_number(audit.get('average_cash_ratio'))}",
        f"- average_market_exposure: {_format_number(audit.get('average_market_exposure'))}",
        f"- average_position_count: {_format_number(audit.get('average_position_count'))}",
        f"- max_position_count: {_format_number(audit.get('max_position_count'))}",
        f"- skipped_buy_count: {audit.get('skipped_buy_count', 0)}",
        f"- skipped_buy_reason_breakdown: {json.dumps(audit.get('skipped_buy_reason_breakdown', {}), ensure_ascii=False, sort_keys=True)}",
        f"- average_order_amount: {_format_number(audit.get('average_order_amount'))}",
        f"- average_order_quantity: {_format_number(audit.get('average_order_quantity'))}",
        f"- buy_as_much_as_possible_count: {audit.get('buy_as_much_as_possible_count', 0)}",
        f"- target_exposure_reached_days: {audit.get('target_exposure_reached_days', 0)}",
        f"- target_exposure_gap_average: {_format_number(audit.get('target_exposure_gap_average'))}",
        f"- target_exposure_blocked_reason_breakdown: {json.dumps(audit.get('target_exposure_blocked_reason_breakdown', {}), ensure_ascii=False, sort_keys=True)}",
        f"- affordable_selected_count: {audit.get('affordable_selected_count', 0)}",
        f"- unaffordable_selected_count: {audit.get('unaffordable_selected_count', 0)}",
        f"- cash_after_buy_average: {_format_number(audit.get('cash_after_buy_average'))}",
        f"- position_value_cap_hit_count: {audit.get('position_value_cap_hit_count', 0)}",
        f"- min_cash_buffer_hit_count: {audit.get('min_cash_buffer_hit_count', 0)}",
        f"- no_candidate_days: {audit.get('no_candidate_days', 0)}",
        f"- no_affordable_candidate_days: {audit.get('no_affordable_candidate_days', 0)}",
        f"- target_exposure_note: {audit.get('target_exposure_note', '')}",
    ]
    source = audit.get("source", {})
    if isinstance(source, dict):
        lines.append(f"- source: {json.dumps(source, ensure_ascii=False, sort_keys=True)}")
    return lines


def _generic_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- audit: unavailable"]
    lines = []
    for key, value in audit.items():
        if key in {"warnings", "errors", "stale_score_cache_files"}:
            continue
        if isinstance(value, (dict, list)):
            lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        else:
            lines.append(f"- {key}: {value}")
    for warning in audit.get("warnings", []) or []:
        lines.append(f"- warning: {warning}")
    for error in audit.get("errors", []) or []:
        lines.append(f"- error: {error}")
    stale_files = audit.get("stale_score_cache_files", []) or []
    if stale_files:
        lines.append("- stale_score_cache_files:")
        for item in stale_files[:20]:
            lines.append(
                "  - "
                f"date={item.get('date')} profile={item.get('profile')} cache_path={item.get('cache_path') or item.get('path')} "
                f"expected_config_hash={item.get('expected_config_hash')} actual_config_hash={item.get('actual_config_hash')} "
                f"expected_market_filter_hash={item.get('expected_market_filter_hash')} actual_market_filter_hash={item.get('actual_market_filter_hash')} "
                f"reason={item.get('reason')}"
            )
    return lines


def _performance_audit_lines(audit: dict[str, Any]) -> list[str]:
    phase_keys = [
        "json_read",
        "json_file_io_sec",
        "json_postprocess_sec",
        "indicator_parse_or_transform_sec",
        "analysis_json_scan_sec",
        "misc",
        "daily_loop_total",
        "per_day_pipeline_total",
        "candidate_generation_total",
        "indicator_generation_total",
        "scoring_total",
        "cache_lookup_total",
        "cache_materialize_total",
        "materialize_indicators_sec",
        "materialize_candidates_sec",
        "materialize_scored_candidates_sec",
        "materialize_market_context_sec",
        "materialize_file_exists_check_sec",
        "materialize_copy_write_sec",
        "materialize_unknown_sec",
        "file_copy_or_link_total",
        "backtest_day_iteration_total",
        "feature_analysis_total",
        "relative_strength_analysis_total",
        "investor_context_analysis_total",
        "markdown_render_total",
        "csv_read_write_total",
        "db_read_write_total",
        "candidate_load",
        "indicator_load",
        "score_load",
        "indicator_file_io_sec",
        "indicator_json_parse_sec",
        "indicator_record_normalize_sec",
        "indicator_filter_by_date_or_code_sec",
        "indicator_copy_from_common_sec",
        "indicator_unused_or_unknown_sec",
        "market_filter",
        "score_integrity_audit",
        "result_integrity_audit",
        "feature_analysis_generation",
        "feature_analysis_load_logs_sec",
        "feature_analysis_load_processed_sec",
        "feature_analysis_load_reports_sec",
        "feature_analysis_market_filter_audit_sec",
        "feature_analysis_score_integrity_sec",
        "feature_analysis_result_integrity_sec",
        "feature_analysis_relative_strength_sec",
        "feature_analysis_investor_context_sec",
        "feature_analysis_score_component_sec",
        "feature_analysis_earnings_filter_sec",
        "feature_analysis_markdown_render_sec",
        "feature_analysis_json_render_sec",
        "feature_analysis_json_write_sec",
        "feature_analysis_write_sec",
        "experiment_summary_generation",
        "cache_copy_or_write",
        "report_write",
        "trade_simulation",
        "comparison_analysis",
        "other_misc",
    ]
    scope_keys = ["common", "profile", "logs", "reports", "data_other", "outside_root", "other"]
    if not isinstance(audit, dict) or not audit:
        audit = {}
    phase_elapsed = audit.get("phase_elapsed_sec", {}) if isinstance(audit.get("phase_elapsed_sec"), dict) else {}
    json_scope = audit.get("json_read_scope", {}) if isinstance(audit.get("json_read_scope"), dict) else {}
    lines = [
        f"- total_elapsed_sec: {audit.get('total_elapsed_sec', 0)}",
        "",
        "### phase_elapsed_sec",
        "",
        "| phase | elapsed_sec |",
        "| --- | ---: |",
    ]
    for key in phase_keys:
        lines.append(f"| {key} | {phase_elapsed.get(key, 0)} |")
    lines.extend(["", "### json_read_scope", "", "| scope | count | bytes | elapsed_sec |", "| --- | ---: | ---: | ---: |"])
    for scope in scope_keys:
        item = json_scope.get(scope, {}) if isinstance(json_scope, dict) else {}
        if not isinstance(item, dict):
            item = {}
        lines.append(f"| {scope} | {item.get('count', 0)} | {item.get('bytes', 0)} | {item.get('elapsed_sec', 0)} |")
    by_profile = audit.get("json_read_by_profile", [])
    lines.extend(["", "### JSON Read by Profile", "", "| profile_id | scope | count | bytes | elapsed_sec |", "| --- | --- | ---: | ---: | ---: |"])
    if isinstance(by_profile, list) and by_profile:
        for row in by_profile:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {row.get('profile_id', '')} | {row.get('scope', '')} | {row.get('count', 0)} | "
                f"{row.get('bytes', 0)} | {row.get('elapsed_sec', 0)} |"
            )
    else:
        lines.append("| unavailable | - | 0 | 0 | 0 |")
    targets = audit.get("optimization_targets_top3", [])
    materialize = audit.get("materialize_audit", {}) if isinstance(audit.get("materialize_audit"), dict) else {}
    lines.extend(
        [
            "",
            "### Materialize Audit",
            "",
            f"- indicator_materialize_skipped_count: {materialize.get('indicator_materialize_skipped_count', 0)}",
            f"- indicator_materialize_required_count: {materialize.get('indicator_materialize_required_count', 0)}",
            f"- candidate_materialize_skipped_count: {materialize.get('candidate_materialize_skipped_count', 0)}",
            f"- candidate_materialize_required_count: {materialize.get('candidate_materialize_required_count', 0)}",
            f"- materialize_skip_reason: {_compact_json(materialize.get('materialize_skip_reason', {}))}",
            "",
            f"- optimization_targets_top3: {_compact_json(targets)}",
        ]
    )
    return lines


def _json_read_ranking_lines(rows: Any) -> list[str]:
    if not isinstance(rows, list) or not rows:
        return ["- ranking: unavailable"]
    lines = [
        "| rank | path | scope | read_count | total_bytes | total_elapsed_sec | avg_elapsed_ms | suspected_reason |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(rows[:20], start=1):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row.get('rank', index)} | {row.get('path', '')} | {row.get('scope', '')} | "
            f"{row.get('read_count', row.get('count', 0))} | {row.get('total_bytes', row.get('bytes', 0))} | "
            f"{row.get('total_elapsed_sec', row.get('elapsed_sec', 0))} | {row.get('avg_elapsed_ms', 0)} | "
            f"{row.get('suspected_reason', '')} |"
        )
    return lines


def _profile_read_reason_lines(audit: dict[str, Any]) -> list[str]:
    if not isinstance(audit, dict) or not audit:
        audit = {}
    breakdown = audit.get("breakdown", {}) if isinstance(audit.get("breakdown"), dict) else {}
    lines = [
        f"- profile_read_total_bytes: {audit.get('profile_read_total_bytes', 0)}",
        f"- profile_read_total_count: {audit.get('profile_read_total_count', 0)}",
        f"- profile_read_total_elapsed_sec: {audit.get('profile_read_total_elapsed_sec', 0)}",
        f"- profile_cache_used_count: {audit.get('profile_cache_used_count', 0)}",
        f"- common_cache_used_count: {audit.get('common_cache_used_count', 0)}",
    ]
    if audit.get("note"):
        lines.append(f"- note: {audit.get('note')}")
    lines.extend(["", "### Breakdown", "", "| reason | count | bytes | elapsed_sec | note |", "| --- | ---: | ---: | ---: | --- |"])
    for reason in [
        "runtime_indicators_read",
        "runtime_candidates_read",
        "scored_candidates_read",
        "runtime_market_context_read",
        "report_or_analysis_read",
        "unknown",
    ]:
        item = breakdown.get(reason, {}) if isinstance(breakdown, dict) else {}
        if not isinstance(item, dict):
            item = {}
        lines.append(
            f"| {reason} | {item.get('count', 0)} | {item.get('bytes', 0)} | "
            f"{item.get('elapsed_sec', 0)} | {item.get('note', '')} |"
        )
    return lines


def _indicator_field_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not isinstance(audit, dict) or not audit:
        audit = {}
    return [
        f"- total_fields: {audit.get('total_fields', 0)}",
        f"- used_fields: {_compact_json(audit.get('used_fields', []))}",
        f"- maybe_unused_fields: {_compact_json(audit.get('maybe_unused_fields', []))}",
        f"- sample_record_size_bytes: {audit.get('sample_record_size_bytes', 0)}",
        f"- estimated_reducible_fields_count: {audit.get('estimated_reducible_fields_count', 0)}",
        f"- confidence: {audit.get('confidence', 'unknown')}",
        f"- note: {audit.get('note', 'No indicator fields are removed by this audit.')}",
    ]


def _runtime_memory_cache_audit_lines(audit: dict[str, Any]) -> list[str]:
    rows = audit if isinstance(audit, dict) else {}
    lines = [
        "| cache_name | hit_count | miss_count | size | note |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name in ["listed_stocks_raw", "listed_stocks_lookup", "raw_prices_by_date", "indicator_runtime_cache"]:
        item = rows.get(name, {}) if isinstance(rows, dict) else {}
        if not isinstance(item, dict):
            item = {}
        lines.append(
            f"| {name} | {item.get('hit_count', 0)} | {item.get('miss_count', 0)} | "
            f"{item.get('size', 0)} | {item.get('note', '')} |"
        )
    return lines


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 4)
    return round((ordered[middle - 1] + ordered[middle]) / 2, 4)


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


def _format_profit_factor(value: Any) -> str:
    if value is None:
        return "N/A"
    if value == float("inf"):
        return "inf"
    return _format_number(value)


def _profile_id(config: dict[str, Any]) -> str:
    return str(config.get("profile_id") or config.get("dealer", {}).get("id") or "rookie_dealer_01")


def _profile_name(config: dict[str, Any]) -> str:
    return str(config.get("profile_name") or config.get("dealer", {}).get("name") or _profile_id(config))
