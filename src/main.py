"""Entry point for AI Fund Lab first implementation."""

from __future__ import annotations

import csv
import calendar
import hashlib
import json
import os
import plistlib
import re
import shutil
import sqlite3
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only on minimal local environments.
    yaml = None

from article import generate_note_article
from benchmark_provider import build_relative_strength_benchmark
from ai_decision import apply_ai_decision, build_ai_decision_log, build_ai_decision_provider
from ai_analysis import export_ai_dataset, export_ai_summary, record_ai_analysis_export
from charts import generate_charts_from_summary
from broker import LiveTradingDisabledError, account_snapshot, build_broker, render_account_snapshot
from commentary import (
    generate_buy_comment,
    generate_daily_comment,
    generate_no_trade_comment,
    generate_reflection_comment,
    generate_sell_comment,
)
from config_version import attach_config_version, config_version_from, load_config as load_versioned_config
from data_provider import DummyDataProvider, JQuantsApiError, JQuantsDataProvider
from demo_auto_order import DemoAutoOrderBlocked, execute_demo_auto_orders, latest_order_preview_path, load_operation_schedule
from earnings_calendar import earnings_counts
from db import (
    analyze_operation_data,
    database_schema_check,
    get_database_path,
    initialize_database,
    save_article,
    save_ai_decision,
    save_market_context,
    save_portfolio_snapshot,
    save_reflections,
    save_pending_orders,
    save_safety_events,
    save_scoring_results,
    save_screening_results,
    save_trades,
)
from feature_analysis import build_feature_analysis, render_feature_analysis_markdown, score_detail_groups
from indicators import calculate_indicators
from investor_context import INVESTOR_CONTEXT_EMPTY, build_investor_context
from market_sections import (
    allowed_market_sections,
    attach_market_section_fields,
    market_section_allowed,
    market_section_counts,
    market_section_from_row,
    normalize_market_section,
)
from jquants_plan import (
    DISPLAY_CAPABILITIES,
    PlanResolution,
    jquants_capability_status,
    jquants_earliest_supported_date,
    jquants_has_capability,
    jquants_profile_compatibility,
    jquants_supported_date_ranges,
    normalize_jquants_plan,
    resolve_jquants_plan,
)
from market_context import build_market_context, neutral_market_context
from news_provider import build_news_provider
from paper_trade import execute_paper_trade_day, execute_real_data_paper_trade, initial_live_paper_state, initial_paper_state
from portfolio import build_daily_summary
import profile_registry as profile_registry_service
from profile_loader import DEFAULT_PROFILE_ID, list_profiles, load_profile
from reflection import generate_reflections
from report import generate_daily_report
from real_screening import screen_candidates
from release_notes import generate_release_notes, render_release_notes_markdown
from safety import can_trade
from scoring import build_trade_decisions, score_candidates, score_real_candidates
from selection_quality import build_selection_quality_analysis, render_selection_quality_markdown
from screening import generate_screening_log
from tachibana_auth import load_private_key, load_tachibana_auth_config
from technical_indicators import TechnicalIndicatorDependencyError
from tax import calculate_period_profit_summary
from trade_metrics import profit_factor_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "rookie_dealer.yaml"
PROVIDER_CONFIG_PATH = ROOT / "config" / "provider.yaml"
ACTIVE_PROFILE_ID = DEFAULT_PROFILE_ID
BACKTEST_MODE_ACTIVE = False
BACKTEST_DAY_LOG_PREFIX = ""
FAST_ANALYSIS_ACTIVE = False
SUMMARY_ONLY_ACTIVE = False
JQUANTS_PLAN_OVERRIDE: str | None = None
FORCE_REFRESH_ACTIVE = False
STORAGE_MODE_OVERRIDE: str | None = None
BACKTEST_PROFILE_TIMINGS: dict[str, float] = {}
RUNTIME_SETTINGS: dict[str, Any] = {}
RUN_EXPERIMENTS_SHARED_STAGE_ACTIVE = False
RUN_EXPERIMENTS_SCORING_REUSE_SOURCE_BY_PROFILE: dict[str, str] = {}
RUN_EXPERIMENTS_PERFORMANCE_REPORT: dict[str, Any] = {}
COMMON_CACHE_METRICS: dict[str, int] = {
    "cache_reused_from_common_count": 0,
    "profile_specific_cache_count": 0,
    "generated_cache_size": 0,
}
LIGHT_API_CALL_LIMITS = {
    "topix_prices": 4,
    "investor_types": 12,
    "earnings_calendar": 3,
    "financial_statements": 3,
}
JQUANTS_API_SESSION: dict[str, Any] = {}


def main() -> None:
    global ACTIVE_PROFILE_ID, FAST_ANALYSIS_ACTIVE, SUMMARY_ONLY_ACTIVE, JQUANTS_PLAN_OVERRIDE, FORCE_REFRESH_ACTIVE, STORAGE_MODE_OVERRIDE, RUNTIME_SETTINGS
    _enable_line_buffered_output()
    args = parse_args()
    RUNTIME_SETTINGS = resolve_runtime_settings(args)
    ACTIVE_PROFILE_ID = RUNTIME_SETTINGS["profile_id"]
    FAST_ANALYSIS_ACTIVE = bool(args.fast_analysis)
    SUMMARY_ONLY_ACTIVE = bool(args.summary_only)
    JQUANTS_PLAN_OVERRIDE = args.jquants_plan
    FORCE_REFRESH_ACTIVE = bool(args.force_refresh)
    STORAGE_MODE_OVERRIDE = args.storage_mode
    provider_name = RUNTIME_SETTINGS["provider"]
    if args.mode == "help":
        run_help()
        return
    if args.mode == "preflight":
        run_preflight(RUNTIME_SETTINGS["profile_id"], with_smoke_test=args.with_smoke_test)
        return
    if args.mode == "simulate-operation":
        run_simulate_operation(RUNTIME_SETTINGS["profile_id"], args.days)
        return
    if args.mode == "list-profiles":
        run_list_profiles()
        return
    if args.mode == "profile-info":
        run_profile_info(RUNTIME_SETTINGS["profile_id"])
        return
    if args.mode == "compare-profiles":
        run_compare_profiles(args.profiles, args.start_date, args.end_date)
        return
    if args.mode == "compare-experiments":
        run_compare_experiments(args.base_profile, args.start_date, args.end_date)
        return
    if args.mode == "run-experiments":
        run_experiments(args.base_profile, args.start_date, args.end_date, args.profiles, args.skip_backtest, args.skip_analyze)
        return
    if args.mode == "jquants-api-summary":
        run_jquants_api_summary()
        return
    if args.mode == "jquants-smoke-test":
        run_jquants_smoke_test(args.endpoint)
        return
    if args.mode == "validate-config":
        run_validate_config(args.profile, args.strict)
        return
    if args.mode == "storage-audit":
        run_storage_audit()
        return
    if args.mode == "cleanup-storage":
        run_cleanup_storage(
            apply=args.apply,
            keep_latest_experiments=args.keep_latest_experiments,
            keep_days=args.keep_days,
            include_reports=args.include_reports,
            include_logs=args.include_logs,
            include_processed=args.include_processed,
            exclude_raw_prices=args.exclude_raw_prices,
            exclude_jquants_cache=args.exclude_jquants_cache,
            verbose=args.verbose,
        )
        return
    if args.mode == "compact-processed-cache":
        run_compact_processed_cache(apply=args.apply, verbose=args.verbose)
        return
    if args.mode == "inspect-cache":
        run_inspect_cache(args.target, args.date)
        return
    if args.mode == "performance-audit":
        run_performance_audit()
        return
    if args.mode in {"clean-reports", "clean-experiments", "clean-cache", "clean-articles"}:
        run_clean_command(
            args.mode,
            profile_id=args.profile,
            provider_name=provider_name,
            yes=args.yes and not args.dry_run,
            older_than_days=args.older_than_days,
            include_latest=args.include_latest,
            cache_kind=args.cache_kind,
            verbose=args.verbose,
        )
        return
    config = load_config(CONFIG_PATH)
    if args.mode == "status":
        run_status(config, args.output_format)
        return
    if args.mode == "healthcheck":
        run_healthcheck(provider_name)
        return
    if args.mode == "tachibana-healthcheck":
        run_tachibana_healthcheck(args.tachibana_env)
        return
    if args.mode == "db-check":
        run_db_check(config)
        return
    if args.mode == "account-snapshot":
        run_account_snapshot(config)
        return
    if args.mode == "demo-auto-order":
        run_demo_auto_order(config)
        return
    if args.mode == "init-db":
        run_init_db(config)
        return
    if args.mode == "analyze":
        run_analyze(config, args.start_date, args.end_date)
        return
    if args.mode == "release-notes":
        run_release_notes(args.since, args.until)
        return
    if args.mode == "full-paper-run":
        run_full_paper_run(provider_name, args.start_date, args.end_date)
        return
    if args.mode == "list-stocks":
        run_list_stocks(provider_name)
        return
    if args.mode == "fetch-prices":
        run_fetch_prices(provider_name, args.date)
        return
    if args.mode == "calculate-indicators":
        run_calculate_indicators(provider_name, args.date)
        return
    if args.mode == "screen":
        run_screen(provider_name, args.date)
        return
    if args.mode == "score":
        run_score(provider_name, args.date)
        return
    if args.mode == "trade":
        run_trade(provider_name, args.date)
        return
    if args.mode == "preview-orders":
        run_preview_orders(provider_name, args.date)
        return
    if args.mode == "publish-article":
        run_publish_article(args.date, args.note_url)
        return
    if args.mode == "run-daily":
        run_daily(provider_name, args.date)
        return
    if args.mode == "backtest":
        run_backtest(provider_name, args.start_date, args.end_date)
        return
    if args.mode == "export-ai-dataset":
        run_export_ai_dataset(config, args.start_date, args.end_date)
        return
    if args.mode == "export-ai-summary":
        run_export_ai_summary(config, args.start_date, args.end_date)
        return

    now = datetime.now()
    run_id = now.strftime("%Y%m%d-%H%M%S")
    state = initial_paper_state(config)
    start_date = next_business_day(now.date())
    last_paths = {}
    last_scoring_log = {}
    last_paper_trade_log = {}
    daily_summaries = []
    all_closed_trades = []

    for day_number in range(1, args.days + 1):
        run_date = add_business_days(start_date, day_number - 1)
        day_key = f"day_{day_number:03d}"
        data_provider = build_data_provider(config, run_date, run_id)
        screening_log = generate_screening_log(config, run_date, run_id, data_provider)
        attach_config_version(screening_log, config)
        screening_log["mode"] = args.mode
        screening_log["day_number"] = day_number

        scoring_log = score_candidates(screening_log, config)
        attach_config_version(scoring_log, config)
        scoring_log["mode"] = args.mode
        scoring_log["day_number"] = day_number

        trade_decision_log = build_trade_decisions(scoring_log, config)
        attach_config_version(trade_decision_log, config)
        trade_decision_log["mode"] = args.mode
        trade_decision_log["day_number"] = day_number

        paper_trade_log = execute_paper_trade_day(scoring_log, trade_decision_log, config, state, day_number)
        attach_config_version(paper_trade_log, config)
        paper_trade_log["mode"] = args.mode

        portfolio_summary = build_daily_summary(paper_trade_log, config)
        attach_config_version(portfolio_summary, config)
        portfolio_summary["mode"] = args.mode
        portfolio_summary["safety_events"] = paper_trade_log.get("safety_events", [])
        portfolio_summary["dealer_comment"] = generate_daily_comment(
            portfolio_summary,
            scoring_log.get("selected", []),
            paper_trade_log.get("orders", []) + paper_trade_log.get("order_attempts", []) + paper_trade_log.get("closed_trades", []),
            config,
        )

        reflection_log = generate_reflections(paper_trade_log, config)
        attach_config_version(reflection_log, config)
        reflection_log["mode"] = args.mode
        reflection_log["day_number"] = day_number

        report_md = generate_daily_report(portfolio_summary, paper_trade_log, trade_decision_log, config)
        article_md = generate_note_article(portfolio_summary, paper_trade_log, config)

        paths = build_day_paths(run_date, day_key)
        write_json(paths["screening_run"], {key: screening_log[key] for key in screening_log if key != "candidates"})
        write_json(paths["candidates"], screening_log)
        write_json(paths["scoring"], scoring_log)
        write_json(paths["decisions"], trade_decision_log)
        write_json(paths["orders"], _pick_paper_trade_fields(paper_trade_log, ["run_id", "date", "day_number", "dealer_id", "orders", "order_attempts"]))
        write_json(paths["trades_daily"], build_daily_trade_log(paper_trade_log, reflection_log))
        write_json(paths["portfolio"], _pick_paper_trade_fields(paper_trade_log, ["run_id", "date", "day_number", "dealer_id", "positions", "closed_trades", "all_closed_trades", "asset_history", "pnl"]))
        write_json(paths["closed_trades"], _pick_paper_trade_fields(paper_trade_log, ["run_id", "date", "day_number", "dealer_id", "closed_trades", "all_closed_trades"]))
        write_json(paths["pnl"], _pick_paper_trade_fields(paper_trade_log, ["run_id", "date", "day_number", "dealer_id", "pnl"]))
        write_json(paths["portfolio_summary"], portfolio_summary)
        write_json(paths["safety"], {"date": paper_trade_log["date"], "safety_events": paper_trade_log.get("safety_events", [])})
        write_json(paths["reflections"], reflection_log)
        write_text(paths["report"], report_md)
        write_text(paths["article"], article_md)
        _update_article_index(paths["article"], run_date.isoformat(), "draft", config)

        daily_summaries.append(portfolio_summary)
        all_closed_trades = paper_trade_log["all_closed_trades"]
        last_paths = paths
        last_scoring_log = scoring_log
        last_paper_trade_log = paper_trade_log

    summary_csv = ROOT / "reports" / profile_id_from(config) / "summary.csv"
    trades_csv = ROOT / "reports" / profile_id_from(config) / "trades.csv"
    charts_dir = ROOT / "reports" / profile_id_from(config) / "charts"
    write_summary_csv(summary_csv, daily_summaries)
    write_trades_csv(trades_csv, all_closed_trades)
    chart_paths = generate_charts_from_summary(summary_csv, charts_dir)

    print("AIファンド研究所 MVPデモ実行が完了しました。")
    print("mode: demo")
    print("real_trading: disabled")
    print("api_connection: disabled")
    print(f"run_id: {run_id}")
    print(f"days: {args.days}")
    print(f"last_day_selected: {len(last_scoring_log['selected'])}")
    print(f"open_positions: {len(last_paper_trade_log['positions'])}")
    print(f"closed_trades_total: {len(last_paper_trade_log['all_closed_trades'])}")
    print(f"daily_report: {last_paths['report'].relative_to(ROOT)}")
    print(f"note_draft: {last_paths['article'].relative_to(ROOT)}")
    print(f"summary_csv: {summary_csv.relative_to(ROOT)}")
    print(f"trades_csv: {trades_csv.relative_to(ROOT)}")
    print(f"assets_chart: {chart_paths['assets_curve'].relative_to(ROOT)}")


def parse_args() -> Any:
    parser = ArgumentParser(description="AIファンド研究所 MVP runner")
    parser.add_argument(
        "--mode",
        choices=[
            "demo",
            "help",
            "status",
            "init-db",
            "analyze",
            "release-notes",
            "full-paper-run",
            "preflight",
            "simulate-operation",
            "list-profiles",
            "profile-info",
            "healthcheck",
            "tachibana-healthcheck",
            "db-check",
            "account-snapshot",
            "demo-auto-order",
            "list-stocks",
            "fetch-prices",
            "calculate-indicators",
            "screen",
            "score",
            "trade",
            "preview-orders",
            "publish-article",
            "run-daily",
            "backtest",
            "export-ai-dataset",
            "export-ai-summary",
            "compare-profiles",
            "compare-experiments",
            "run-experiments",
            "jquants-api-summary",
            "jquants-smoke-test",
            "validate-config",
            "storage-audit",
            "cleanup-storage",
            "compact-processed-cache",
            "inspect-cache",
            "performance-audit",
            "clean-reports",
            "clean-experiments",
            "clean-cache",
            "clean-articles",
        ],
        default="demo",
        help="Execution mode. Use demo, healthcheck, list-stocks, fetch-prices, or calculate-indicators.",
    )
    parser.add_argument(
        "--provider",
        choices=["dummy", "jquants"],
        default=None,
        help="Temporary provider override. If omitted, config/provider.yaml is used.",
    )
    parser.add_argument(
        "--jquants-plan",
        choices=["free", "light"],
        default=None,
        help="Override J-Quants contract plan. Config default: free.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Temporary profile override. If omitted, config/provider.yaml profile.default is used.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Profile ids for compare-profiles mode.",
    )
    parser.add_argument(
        "--base",
        dest="base_profile",
        default=None,
        help="Baseline profile id for compare-experiments mode.",
    )
    parser.add_argument(
        "--base-profile",
        dest="base_profile",
        default=None,
        help="Baseline profile id for experiment batch modes.",
    )
    parser.add_argument(
        "--env",
        dest="tachibana_env",
        choices=["demo", "live"],
        default="demo",
        help="Tachibana environment for tachibana-healthcheck mode.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of business days to simulate in demo mode.",
    )
    parser.add_argument(
        "--date",
        help="Target date in YYYY-MM-DD format for fetch-prices mode.",
    )
    parser.add_argument(
        "--target",
        choices=["indicators", "candidates", "market_context"],
        default="indicators",
        help="Cache target for inspect-cache.",
    )
    parser.add_argument(
        "--start-date",
        help="Backtest start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        help="Backtest end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--period",
        choices=["6m", "1y", "3y", "5y"],
        help="Date range preset. If --end-date is omitted, today in Asia/Tokyo is used.",
    )
    parser.add_argument(
        "--since",
        help="Release notes start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--until",
        help="Release notes end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--note-url",
        help="Published note URL for publish-article mode.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format for status mode.",
    )
    parser.add_argument(
        "--fast-analysis",
        action="store_true",
        help="Reduce heavy backtest analysis logs without changing trading results.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip daily markdown/articles and heavy analyze steps for experiment/backtest summary generation.",
    )
    parser.add_argument(
        "--storage-mode",
        choices=["full_debug", "analysis", "compact"],
        default=None,
        help="Saved JSON/DB detail level. full_debug keeps all fields, analysis prunes debug-only fields, compact saves minimal runtime rows.",
    )
    parser.add_argument(
        "--entry-timing",
        choices=["same-day-close", "next-business-day-open", "next-business-day-close"],
        default=None,
        help="Temporary backtest execution timing override. Config default is next_business_day_open.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refresh cached provider data such as J-Quants earnings calendar.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Use existing backtest/analyze results for run-experiments.",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip analyze in run-experiments and use existing analysis outputs where required.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files for clean modes. Without this flag clean modes are dry-run.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete cleanup-storage targets. Without this flag cleanup-storage is dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit dry-run for clean modes. This is the default unless --yes is specified.",
    )
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=None,
        help="Clean only files older than N days.",
    )
    parser.add_argument(
        "--keep-days",
        type=int,
        default=30,
        help="Keep cleanup-storage files newer than N days.",
    )
    parser.add_argument(
        "--keep-latest-experiments",
        type=int,
        default=3,
        help="Keep the newest N experiment result directories in cleanup-storage.",
    )
    parser.add_argument(
        "--include-reports",
        action="store_true",
        help="Include old safe report artifacts in cleanup-storage.",
    )
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Include old safe log artifacts in cleanup-storage.",
    )
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help="Include profile-specific processed indicators/candidates in cleanup-storage.",
    )
    parser.add_argument(
        "--exclude-raw-prices",
        action="store_true",
        default=True,
        help="Keep J-Quants raw price files in cleanup-storage. This is the default.",
    )
    parser.add_argument(
        "--exclude-jquants-cache",
        action="store_true",
        default=True,
        help="Keep data/cache/jquants in cleanup-storage. This is the default.",
    )
    parser.add_argument(
        "--include-latest",
        action="store_true",
        help="Include latest files in clean modes. Latest files are kept by default.",
    )
    parser.add_argument(
        "--cache-kind",
        choices=["prices", "topix_prices", "earnings_calendar", "investor_types", "financial_statements", "all"],
        default="all",
        help="Cache category for clean-cache.",
    )
    parser.add_argument(
        "--endpoint",
        choices=[
            "all",
            "listed_info",
            "prices",
            "topix_prices",
            "investor_types",
            "earnings_calendar",
            "financial_statements",
            "trading_calendar",
        ],
        default="topix_prices",
        help="J-Quants endpoint for jquants-smoke-test.",
    )
    parser.add_argument(
        "--with-smoke-test",
        action="store_true",
        help="Run J-Quants smoke tests during preflight. Regular preflight does not call every API.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show each clean target path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat validate-config warnings as failures.",
    )
    args = parser.parse_args()
    _mark_cli_date_sources(args)
    _apply_period_preset(args)
    _apply_configured_date_defaults(args)
    _apply_jquants_supported_date_limits(args)
    if args.days < 1:
        parser.error("--days must be 1 or greater.")
    if args.mode in {"fetch-prices", "calculate-indicators", "screen", "score", "trade", "preview-orders", "publish-article", "run-daily", "inspect-cache"} and not args.date:
        parser.error(f"--date YYYY-MM-DD is required for {args.mode} mode.")
    if args.mode == "publish-article" and not args.note_url:
        parser.error("--note-url URL is required for publish-article mode.")
    if args.mode == "analyze" and (args.start_date or args.end_date):
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date must be specified together for analyze mode.")
        try:
            start_date = date.fromisoformat(args.start_date)
            end_date = date.fromisoformat(args.end_date)
        except ValueError:
            parser.error("--start-date and --end-date must be in YYYY-MM-DD format.")
        if start_date > end_date:
            parser.error("--start-date must be earlier than or equal to --end-date.")
    if args.mode in {"backtest", "full-paper-run", "export-ai-dataset", "export-ai-summary", "compare-profiles"}:
        if args.mode == "compare-profiles" and not args.profiles:
            parser.error("--profiles PROFILE_ID [PROFILE_ID ...] is required for compare-profiles mode.")
        if not args.start_date or not args.end_date:
            parser.error(f"--start-date YYYY-MM-DD and --end-date YYYY-MM-DD are required for {args.mode} mode.")
        try:
            start_date = date.fromisoformat(args.start_date)
            end_date = date.fromisoformat(args.end_date)
        except ValueError:
            parser.error("--start-date and --end-date must be in YYYY-MM-DD format.")
        if start_date > end_date:
            parser.error("--start-date must be earlier than or equal to --end-date.")
    if args.mode == "compare-experiments" and (args.start_date or args.end_date):
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date must be specified together for compare-experiments mode.")
        try:
            start_date = date.fromisoformat(args.start_date)
            end_date = date.fromisoformat(args.end_date)
        except ValueError:
            parser.error("--start-date and --end-date must be in YYYY-MM-DD format.")
        if start_date > end_date:
            parser.error("--start-date must be earlier than or equal to --end-date.")
    if args.mode == "run-experiments":
        if not args.start_date or not args.end_date:
            parser.error("--start-date YYYY-MM-DD and --end-date YYYY-MM-DD are required for run-experiments mode.")
        try:
            start_date = date.fromisoformat(args.start_date)
            end_date = date.fromisoformat(args.end_date)
        except ValueError:
            parser.error("--start-date and --end-date must be in YYYY-MM-DD format.")
        if start_date > end_date:
            parser.error("--start-date must be earlier than or equal to --end-date.")
    if args.mode == "release-notes":
        if not args.since or not args.until:
            parser.error("--since YYYY-MM-DD and --until YYYY-MM-DD are required for release-notes mode.")
        try:
            since_date = date.fromisoformat(args.since)
            until_date = date.fromisoformat(args.until)
        except ValueError:
            parser.error("--since and --until must be in YYYY-MM-DD format.")
        if since_date > until_date:
            parser.error("--since must be earlier than or equal to --until.")
    return args


def _mark_cli_date_sources(args: Any) -> None:
    argv = sys.argv[1:]
    args.start_date_source = "cli" if "--start-date" in argv else None
    args.end_date_source = "cli" if "--end-date" in argv else None
    args.requested_start_date = args.start_date
    args.requested_end_date = args.end_date


def _apply_period_preset(args: Any) -> None:
    if not getattr(args, "period", None):
        return
    end = date.fromisoformat(args.end_date) if getattr(args, "end_date", None) else _today_jst()
    months = {"6m": 6, "1y": 12, "3y": 36, "5y": 60}[args.period]
    if not getattr(args, "start_date", None):
        args.start_date = _shift_months(end, -months).isoformat()
        args.start_date_source = "cli"
        args.requested_start_date = args.start_date
    if not getattr(args, "end_date", None):
        args.end_date = end.isoformat()
        args.end_date_source = "cli"
        args.requested_end_date = args.end_date


def _today_jst() -> date:
    return datetime.now(ZoneInfo("Asia/Tokyo")).date()


def _shift_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _apply_configured_date_defaults(args: Any) -> None:
    if args.mode not in {"backtest", "full-paper-run", "export-ai-dataset", "export-ai-summary", "compare-profiles", "run-experiments"}:
        return
    provider_config = _load_provider_runtime_config()
    if not args.start_date:
        args.start_date = _config_get(provider_config, ("backtest", "start_date"))
        args.start_date_source = "config" if args.start_date else "default"
        args.requested_start_date = args.start_date
    if not args.end_date:
        args.end_date = _config_get(provider_config, ("backtest", "end_date"))
        args.end_date_source = "config" if args.end_date else "default"
        args.requested_end_date = args.end_date


def _apply_jquants_supported_date_limits(args: Any) -> None:
    if getattr(args, "provider", None) not in {None, "jquants"}:
        return
    if getattr(args, "mode", "") not in {"backtest", "fetch-prices", "run-experiments"}:
        return
    resolution = resolve_jquants_plan(args=args, config_root=ROOT, provider_config=_load_provider_runtime_config())
    earliest = jquants_earliest_supported_date({"jquants": resolution.config}, "prices")
    if earliest is None:
        return
    if getattr(args, "start_date", None):
        requested = date.fromisoformat(args.start_date)
        if requested < earliest:
            args.requested_start_date = args.start_date
            args.start_date = earliest.isoformat()
            print(
                "warning: requested start-date is before J-Quants supported range; "
                f"adjusted to {args.start_date}"
            )
    if getattr(args, "date", None) and args.mode == "fetch-prices":
        requested_date = date.fromisoformat(args.date)
        if requested_date < earliest:
            args.requested_date = args.date
            args.date = earliest.isoformat()
            print(
                "warning: requested date is before J-Quants supported range; "
                f"adjusted to {args.date}"
            )


def resolve_runtime_settings(args: Any, provider_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = provider_config if provider_config is not None else _load_provider_runtime_config()
    plan_resolution = resolve_jquants_plan(
        args=args,
        config=config if provider_config is not None else None,
        config_root=None if provider_config is not None else ROOT,
        provider_config=config,
    )
    profile_id, profile_source = _resolve_setting_value(
        getattr(args, "profile", None),
        _config_get(config, ("profile", "default")),
        DEFAULT_PROFILE_ID,
    )
    provider_name, provider_source = _resolve_setting_value(
        getattr(args, "provider", None),
        config.get("data_provider") or _config_get(config, ("provider", "default")),
        "jquants",
    )
    broker_mode, broker_source = _resolve_setting_value(
        None,
        _config_get(config, ("broker", "mode")) or _config_get(config, ("broker", "provider")),
        "paper",
    )
    auto_order_enabled, auto_order_source = _resolve_setting_value(
        None,
        _config_get(config, ("operation", "auto_order_enabled")),
        False,
    )
    return {
        "profile_id": str(profile_id),
        "provider": str(provider_name),
        "jquants_plan": plan_resolution.plan,
        "jquants_plan_resolution": plan_resolution,
        "entry_timing": getattr(args, "entry_timing", None),
        "broker_mode": str(broker_mode),
        "auto_order_enabled": bool(auto_order_enabled),
        "date_resolution": {
            "requested_start_date": getattr(args, "requested_start_date", getattr(args, "start_date", None)),
            "requested_end_date": getattr(args, "requested_end_date", getattr(args, "end_date", None)),
            "start_date_source": getattr(args, "start_date_source", None) or "default",
            "end_date_source": getattr(args, "end_date_source", None) or "default",
            "effective_start_date": getattr(args, "start_date", None),
            "effective_end_date": getattr(args, "end_date", None),
        },
        "sources": {
            "profile": profile_source,
            "provider": provider_source,
            "jquants_plan": plan_resolution.source,
            "broker": broker_source,
            "auto_order_enabled": auto_order_source,
        },
    }


def _resolve_setting_value(cli_value: Any, config_value: Any, default_value: Any) -> tuple[Any, str]:
    if cli_value is not None and cli_value != "":
        return cli_value, "cli"
    if config_value is not None and config_value != "":
        return config_value, "config"
    return default_value, "default"


def run_simulate_operation(profile_id: str, days: int) -> None:
    simulation = build_operation_simulation(profile_id, days)
    print(render_operation_simulation(simulation))


def build_operation_simulation(profile_id: str, days: int, start_date: date | None = None) -> dict[str, Any]:
    if days < 1:
        raise ValueError("days must be 1 or greater")
    schedule_path = ROOT / "config" / "operation_schedule.yaml"
    schedule = _load_operation_schedule_for_validation()
    config = load_profile(profile_id)
    _apply_runtime_provider_settings(config)
    _apply_jquants_plan_settings(config)
    start = start_date or _today_jst()
    simulation_days = [_simulate_operation_day(start + timedelta(days=offset), schedule) for offset in range(days)]
    api_usage = estimate_operation_api_usage(config, days)
    return {
        "mode": "simulate-operation",
        "dry_run": True,
        "profile_id": profile_id,
        "days": days,
        "schedule_path": str(schedule_path.relative_to(ROOT)),
        "schedule": schedule,
        "operation_days": simulation_days,
        "api_usage": api_usage,
        "files": expected_operation_files(profile_id, config),
        "launchd_validation": validate_launchd_files(),
        "orders": {
            "actual_orders": 0,
            "preview_only": True,
            "expected_order_events": ["preview_orders", "paper/demo order check"],
        },
    }


def _simulate_operation_day(day: date, schedule: dict[str, Any]) -> dict[str, Any]:
    flow = schedule.get("daily_flow", {}) if isinstance(schedule.get("daily_flow"), dict) else {}
    events = []
    event_specs = [
        ("data_fetch_time", "data-fetch", ["listed info cache check", "price cache check"]),
        ("screening_time", "run-daily", ["screening", "scoring", "candidate generation"]),
        ("report_time", "report", ["daily paper report", "analysis summary files"]),
        ("evening_paper_run_time", "evening-selection", ["preflight", "screen", "score", "preview-orders", "analyze"]),
        ("order_review_time", "order-review", ["preview_orders review", "risk check result"]),
        ("demo_order_time", "paper order", ["cash check", "position check", "duplicate order check", "demo/paper order preview"]),
        ("article_time", "article", ["daily note article"]),
        ("analyze_time", "analyze", ["analysis_latest.md", "feature_analysis.md", "selection_quality.md"]),
    ]
    seen = set()
    for key, name, expected in event_specs:
        time_text = flow.get(key)
        if not time_text or (time_text, name) in seen:
            continue
        seen.add((time_text, name))
        events.append({"time": str(time_text), "name": name, "expected": expected})
    return {"date": day.isoformat(), "events": sorted(events, key=lambda item: item["time"])}


def estimate_operation_api_usage(config: dict[str, Any], days: int) -> dict[str, Any]:
    plan = _jquants_plan(config)
    features = config.get("features", {}) if isinstance(config.get("features"), dict) else {}
    earnings_filter = config.get("earnings_filter", {}) if isinstance(config.get("earnings_filter"), dict) else {}
    daily = {
        "prices_calls": 20,
        "topix_calls": 1 if features.get("relative_strength") and jquants_has_capability(plan, "topix_prices") else 0,
        "investor_types_calls": 1 if features.get("investor_context") and jquants_has_capability(plan, "investor_types") else 0,
        "earnings_calendar_calls": 1 if earnings_filter.get("enabled") else 0,
    }
    daily_total = sum(daily.values())
    monthly_total = daily_total * 20
    limit_per_minute = _jquants_requests_per_minute(config)
    return {
        "plan": plan,
        "daily": daily,
        "daily_total": daily_total,
        "monthly_total": monthly_total,
        "limit_per_minute": limit_per_minute,
        "limit_status": "OK" if daily_total <= limit_per_minute * 10 else "REVIEW",
        "simulated_total": daily_total * days,
    }


def expected_operation_files(profile_id: str, config: dict[str, Any]) -> dict[str, list[str]]:
    files = {
        "reports": [
            f"reports/{profile_id}/backtests/analysis_latest.md",
            f"reports/{profile_id}/backtests/feature_analysis.md",
            f"reports/{profile_id}/backtests/selection_quality.md",
            f"reports/{profile_id}/broker/account_snapshot_latest.md",
        ],
        "articles": [],
        "experiments": [],
        "logs": [
            "logs/paper_run.log",
            "logs/demo_orders.log",
        ],
    }
    reporting = config.get("reporting", {}) if isinstance(config.get("reporting"), dict) else {}
    if reporting.get("article_output_mode", "daily_only") == "daily_only":
        files["articles"].append("reports/articles/daily/YYYY/MM/daily_note.md")
    return files


def validate_launchd_files() -> dict[str, Any]:
    if str(os.environ.get("CI", "")).lower() == "true":
        return {
            "status": "SKIP",
            "reason": "launchd validation skipped in CI",
            "checked": 0,
            "checks": [],
        }
    launchd_dir = ROOT / "docs" / "launchd"
    plist_paths = sorted(launchd_dir.glob("*.plist")) if launchd_dir.exists() else []
    checks = []
    for path in plist_paths:
        try:
            payload = plistlib.loads(path.read_bytes())
        except Exception as exc:
            checks.append({"status": "ERROR", "path": str(path.relative_to(ROOT)), "message": f"plist parse failed: {exc}"})
            continue
        program_args = payload.get("ProgramArguments") or []
        missing_scripts = []
        for arg in program_args:
            if isinstance(arg, str) and arg.endswith(".sh"):
                script_path = Path(arg)
                if not script_path.is_absolute():
                    script_path = ROOT / script_path
                if not script_path.exists():
                    missing_scripts.append(str(script_path))
        working_dir = Path(payload.get("WorkingDirectory") or ROOT)
        checks.append(
            {
                "status": "OK" if not missing_scripts and working_dir.exists() else "ERROR",
                "path": str(path.relative_to(ROOT)),
                "label": payload.get("Label", ""),
                "working_directory": str(working_dir),
                "missing_scripts": missing_scripts,
                "message": "OK" if not missing_scripts and working_dir.exists() else f"missing script: {', '.join(missing_scripts) or 'working directory missing'}",
            }
        )
    return {
        "status": "OK" if checks and all(item["status"] == "OK" for item in checks) else "ERROR" if checks else "WARN",
        "reason": "checked launchd plist files" if checks else "no launchd plist files found",
        "checked": len(checks),
        "checks": checks,
    }


def render_operation_simulation(simulation: dict[str, Any]) -> str:
    lines = [
        "# Operation Simulation",
        "",
        f"- dry_run: {str(simulation['dry_run']).lower()}",
        f"- profile_id: {simulation['profile_id']}",
        f"- days: {simulation['days']}",
        f"- schedule: {simulation['schedule_path']}",
        "",
        "## Daily Timeline",
        "",
    ]
    for day in simulation["operation_days"]:
        lines.append(day["date"])
        lines.append("")
        for event in day["events"]:
            lines.append(event["time"])
            lines.append(event["name"])
            lines.append("expected:")
            for expected in event["expected"]:
                lines.append(f"- {expected}")
            lines.append("")
    api = simulation["api_usage"]
    lines.extend(
        [
            "## Estimated API Usage",
            "",
            "J-Quants:",
            f"- prices calls: {api['daily']['prices_calls']}",
            f"- topix calls: {api['daily']['topix_calls']}",
            f"- investor_types calls: {api['daily']['investor_types_calls']}",
            f"- earnings_calendar calls: {api['daily']['earnings_calendar_calls']}",
            "",
            f"daily: {api['daily_total']} calls",
            f"monthly: {api['monthly_total']} calls",
            f"plan: {api['plan']}",
            f"limit: {api['limit_status']}",
            "",
            "## Expected Files",
            "",
        ]
    )
    for group, paths in simulation["files"].items():
        lines.append(f"{group}:")
        if paths:
            for path in paths:
                lines.append(f"- {path}")
        else:
            lines.append("- none")
        lines.append("")
    launchd = simulation["launchd_validation"]
    lines.extend(["## launchd validation", "", launchd["status"], f"reason: {launchd.get('reason', '')}", f"checked: {launchd.get('checked', 0)}", ""])
    for check in launchd["checks"]:
        lines.append(f"- {check['path']}: {check['message']}")
    lines.extend(["", "## Orders", "", f"- actual_orders: {simulation['orders']['actual_orders']}", "- broker/API execution: not_called"])
    return "\n".join(lines)


def run_validate_config(profile_id: str | None = None, strict: bool = False) -> None:
    result = build_config_validation(profile_id, RUNTIME_SETTINGS or None, strict=strict)
    print("Config Validation")
    print(f"status: {result['status']}")
    print(f"strict: {str(strict).lower()}")
    print(f"profiles: {', '.join(result['profile_ids'])}")
    print(f"J-Quants Plan: {result['jquants_plan']}")
    print(f"Source: {result['jquants_plan_source']}")
    print("J-Quants Plan Resolution:")
    print(f"- plan: {result['jquants_plan']}")
    print(f"- source: {result['jquants_plan_source']}")
    print(f"- config_path: {result.get('jquants_config_path') or 'N/A'}")
    print(f"- capabilities: {json.dumps(result['capabilities'], ensure_ascii=False, sort_keys=True)}")
    print("Capabilities:")
    for capability in DISPLAY_CAPABILITIES:
        print(f"- {capability}: {result['capabilities'].get(capability, 'disabled')}")
    for check in result["checks"]:
        print(f"[{check['status']}] {check['name']}: {check['message']}")
    print("Validation Summary:")
    print(f"- errors: {result['fail_count']}")
    print(f"- warnings: {result['warn_count']}")
    print(f"- checked_profiles: {result['checked_profiles']}")
    if result["exit_code"]:
        raise SystemExit(1)


def run_jquants_api_summary() -> None:
    summary = build_jquants_api_summary()
    print("J-Quants API Summary")
    for row in summary["endpoints"]:
        print(
            f"- {row['endpoint']}: cache_files={row['cache_files']} "
            f"total_records={row['total_records']} usable_cache_files={row['usable_cache_files']} "
            f"empty_cache_files={row['empty_cache_files']} latest_cache_date={row['latest_cache_date'] or 'N/A'} "
            f"last_status={row['last_status'] or 'N/A'} last_records={row['last_records'] or 'N/A'} "
            f"last_error={row['last_error'] or 'N/A'}"
        )


JQUANTS_SMOKE_ENDPOINTS = [
    "listed_info",
    "prices",
    "topix_prices",
    "investor_types",
    "earnings_calendar",
    "financial_statements",
    "trading_calendar",
]


def run_jquants_smoke_test(endpoint: str) -> None:
    result = build_jquants_smoke_test(endpoint, load_config(CONFIG_PATH))
    if endpoint == "all":
        print("J-Quants Smoke Test Summary")
        print("")
        print("| endpoint | status_code | records | cache_saved | result |")
        print("| --- | --- | ---: | --- | --- |")
        for row in result.get("endpoints", []):
            print(
                f"| {row.get('endpoint')} | {row.get('status_code') or 'N/A'} | "
                f"{row.get('records', 0)} | {str(bool(row.get('cache_saved'))).lower()} | {row.get('result')} |"
            )
        return
    print("J-Quants Smoke Test")
    keys = [
        "endpoint",
        "url",
        "params",
        "status_code",
        "records",
        "first_record_keys",
        "response_body_sample",
        "cache_saved",
        "cache_path",
        "error_reason",
        "result",
    ]
    for key in keys:
        value = result.get(key)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        print(f"{key}: {value if value not in (None, '') else 'N/A'}")


def build_jquants_smoke_test(endpoint: str, config: dict[str, Any]) -> dict[str, Any]:
    if endpoint == "all":
        return {
            "endpoint": "all",
            "plan": _jquants_plan(config),
            "endpoints": [
                build_jquants_smoke_test(item, config)
                for item in JQUANTS_SMOKE_ENDPOINTS
            ],
        }
    if endpoint not in set(JQUANTS_SMOKE_ENDPOINTS):
        raise ValueError(f"unsupported smoke endpoint: {endpoint}")
    plan = _jquants_plan(config)
    if not jquants_has_capability(plan, endpoint):
        result = {
            "endpoint": endpoint,
            "url": "",
            "params": {},
            "status_code": "",
            "records": 0,
            "first_record_keys": [],
            "response_body_sample": "",
            "cache_saved": False,
            "cache_path": "",
            "error_reason": "capability disabled for current plan",
            "result": "SKIPPED_PLAN",
        }
        _log_jquants_api_event(
            endpoint=endpoint,
            plan=plan,
            cache_hit=False,
            status="SKIPPED_PLAN",
            records=0,
            saved=False,
            reason="capability_disabled",
            result="SKIPPED_PLAN",
        )
        return result
    end_date = _business_day_on_or_before(date.today())
    start_date = _jquants_smoke_start_date(endpoint, end_date)
    provider = JQuantsDataProvider(
        ROOT / ".env",
        timeout_seconds=int(config.get("jquants", {}).get("request_timeout_seconds", 20)),
        plan=plan,
        requests_per_minute=_jquants_requests_per_minute(config),
        parallel_fetch=_jquants_parallel_fetch(config),
        max_parallel_requests=_jquants_max_parallel_requests(config),
    )
    payload: dict[str, Any]
    try:
        payload = _fetch_jquants_smoke_payload(provider, endpoint, start_date, end_date)
    except Exception as exc:
        payload = {
            "records": [],
            "saved": False,
            "reason": _api_error_status(exc),
            "api_status": _api_error_status(exc),
            **_api_error_log_fields_raw(exc),
        }
    metadata = dict(getattr(provider, "last_request_metadata", {}) or {})
    records = payload.get("records", []) if isinstance(payload.get("records"), list) else []
    request_params = payload.get("request_params") or metadata.get("params") or {}
    status_code = payload.get("http_status") or metadata.get("status_code") or payload.get("api_status") or ""
    result_name = _jquants_smoke_result(status_code, records, payload)
    error_reason = _jquants_smoke_error_reason(result_name, payload, records)
    result = {
        "endpoint": endpoint,
        "url": payload.get("request_url") or metadata.get("url") or "",
        "params": request_params,
        "status_code": status_code,
        "records": len(records),
        "first_record_keys": sorted(records[0].keys()) if records else [],
        "cache_saved": bool(payload.get("saved")),
        "cache_path": payload.get("cache_path") or "",
        "result": result_name,
        "error_reason": error_reason,
        "response_body_sample": str(payload.get("response_body") or metadata.get("response_body") or "")[:500],
    }
    _log_jquants_api_event(
        endpoint=endpoint,
        plan=plan,
        cache_hit=bool(payload.get("from_cache")),
        status=_provider_payload_status(payload),
        records=len(records),
        saved=bool(payload.get("saved")),
        cache_path=payload.get("cache_path"),
        reason=error_reason,
        result=result_name,
        **_payload_http_log_fields(payload, metadata),
    )
    return result


def _fetch_jquants_smoke_payload(
    provider: JQuantsDataProvider,
    endpoint: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    if endpoint == "listed_info":
        records = provider.get_listed_stocks()
        return _jquants_smoke_cache_payload(endpoint, records, f"{end_date.isoformat()}.json", provider, end_date, end_date)
    if endpoint == "prices":
        records = provider.get_daily_prices(end_date)
        return _jquants_smoke_cache_payload(endpoint, records, f"{end_date.isoformat()}.json", provider, end_date, end_date)
    if endpoint == "topix_prices":
        return provider.fetch_topix_prices_cached(ROOT / "data" / "cache", start_date=start_date, end_date=end_date, force_refresh=True)
    if endpoint == "investor_types":
        return provider.fetch_investor_types_cached(ROOT / "data" / "cache", start_date=start_date, end_date=end_date, force_refresh=True)
    if endpoint == "earnings_calendar":
        return provider.fetch_earnings_calendar_cached(ROOT / "data" / "cache", target_date=end_date, force_refresh=True)
    if endpoint == "financial_statements":
        return provider.fetch_financial_statements_cached(ROOT / "data" / "cache", start_date=start_date, end_date=end_date, force_refresh=True)
    if endpoint == "trading_calendar":
        records = provider._get_paginated_records(
            "/markets/calendar",
            {"from": start_date.strftime("%Y%m%d"), "to": end_date.strftime("%Y%m%d")},
        )
        return _jquants_smoke_cache_payload(
            endpoint,
            records,
            f"{start_date.isoformat()}_to_{end_date.isoformat()}.json",
            provider,
            start_date,
            end_date,
        )
    raise ValueError(f"unsupported smoke endpoint: {endpoint}")


def _jquants_smoke_cache_payload(
    endpoint: str,
    records: list[dict[str, Any]],
    filename: str,
    provider: JQuantsDataProvider,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    cache_path = ROOT / "data" / "cache" / "jquants" / endpoint / filename
    saved = bool(records)
    if saved:
        write_json(cache_path, {"records": records})
    else:
        _record_jquants_empty_marker(endpoint, start_date, end_date, "empty_response")
    return {
        "records": records,
        "cache_path": str(cache_path),
        "from_cache": False,
        "saved": saved,
        "usable": saved,
        "available": saved,
        "api_status": "200",
        "reason": "" if saved else "empty_response",
        "request_url": (getattr(provider, "last_request_metadata", {}) or {}).get("url", ""),
        "request_params": (getattr(provider, "last_request_metadata", {}) or {}).get("params", {}),
        "http_status": (getattr(provider, "last_request_metadata", {}) or {}).get("status_code", ""),
        "response_body": (getattr(provider, "last_request_metadata", {}) or {}).get("response_body", ""),
    }


def _jquants_smoke_result(status_code: Any, records: list[dict[str, Any]], payload: dict[str, Any]) -> str:
    if str(payload.get("api_status") or "") == "SKIPPED_PLAN":
        return "SKIPPED_PLAN"
    try:
        code = int(status_code)
    except (TypeError, ValueError):
        status = str(payload.get("api_status") or payload.get("reason") or "").lower()
        if status == "auth_or_plan_error":
            return "AUTH_OR_PLAN_ERROR"
        if status == "bad_request":
            return "BAD_REQUEST"
        if status == "endpoint_not_found":
            return "ENDPOINT_NOT_FOUND"
        if status == "rate_limit":
            return "RATE_LIMITED"
        if status:
            return "ERROR"
        return "OK" if records else "EMPTY"
    if code in {401, 403}:
        return "AUTH_OR_PLAN_ERROR"
    if code == 400:
        return "BAD_REQUEST"
    if code == 404:
        return "ENDPOINT_NOT_FOUND"
    if code == 429:
        return "RATE_LIMITED"
    if 500 <= code <= 599:
        return "SERVER_ERROR"
    if code == 200 and not records:
        return "EMPTY"
    if code == 200:
        return "OK"
    return "ERROR"


def _jquants_smoke_error_reason(result: str, payload: dict[str, Any], records: list[dict[str, Any]]) -> str:
    if result == "OK":
        return ""
    if result == "EMPTY":
        return "empty_response"
    return str(payload.get("reason") or payload.get("warning") or result.lower())


def _jquants_smoke_start_date(endpoint: str, end_date: date) -> date:
    if endpoint == "investor_types":
        return end_date - timedelta(weeks=52)
    if endpoint == "financial_statements":
        return end_date
    if endpoint == "trading_calendar":
        return end_date - timedelta(days=30)
    return end_date - timedelta(days=14)


def build_jquants_api_summary() -> dict[str, Any]:
    last_events = _last_jquants_api_events()
    endpoints = []
    for endpoint in JQUANTS_SMOKE_ENDPOINTS:
        cache = _jquants_cache_directory_summary(endpoint)
        event = last_events.get(endpoint, {})
        endpoints.append(
            {
                "endpoint": endpoint,
                **cache,
                "last_status": event.get("status"),
                "last_records": event.get("records"),
                "last_error": event.get("error"),
            }
        )
    return {"endpoints": endpoints}


def _jquants_cache_directory_summary(endpoint: str) -> dict[str, Any]:
    directory = ROOT / "data" / "cache" / "jquants" / endpoint
    files = sorted(directory.glob("*.json")) if directory.exists() else []
    total_records = 0
    usable_cache_files = 0
    empty_cache_files = 0
    latest_dates = []
    for path in files:
        state = _cache_file_state(str(path))
        total_records += int(state.get("records") or 0)
        if state.get("usable"):
            usable_cache_files += 1
        else:
            empty_cache_files += 1
        if state.get("latest_date"):
            latest_dates.append(str(state["latest_date"]))
        else:
            latest_dates.append(path.stem.split("_to_")[-1])
    return {
        "cache_files": len(files),
        "total_records": total_records,
        "usable_cache_files": usable_cache_files,
        "empty_cache_files": empty_cache_files,
        "latest_cache_date": max(latest_dates) if latest_dates else None,
    }


def _last_jquants_api_events() -> dict[str, dict[str, str]]:
    path = ROOT / "logs" / "jquants_api.log"
    if not path.exists():
        return {}
    events: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_key_value_log_line(line)
        endpoint = parsed.get("endpoint")
        if endpoint:
            events[endpoint] = parsed
    return events


def _parse_key_value_log_line(line: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in line.split():
        if "=" in part:
            key, value = part.split("=", 1)
            parsed[key] = value
    return parsed


def build_config_validation(
    profile_id: str | None,
    runtime_settings: dict[str, Any] | None = None,
    provider_config: dict[str, Any] | None = None,
    registry: dict[str, Any] | None = None,
    operation_schedule: dict[str, Any] | None = None,
    profile_config: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    settings = runtime_settings or _runtime_settings_or_defaults()
    provider_payload = provider_config if provider_config is not None else _load_provider_runtime_config()
    registry_payload = registry if registry is not None else _load_profile_registry_for_validation()
    jquants_validation = _load_jquants_validation_config(provider_payload, settings)
    plan = jquants_validation["plan"]
    checks: list[dict[str, str]] = []
    _validate_jquants_config(checks, jquants_validation)
    target_profile_ids = _validation_target_profiles(profile_id, registry_payload)
    if profile_config is not None and profile_id:
        profile_configs = {profile_id: profile_config}
    else:
        profile_configs = {}
    _validate_registry_consistency(checks, registry_payload, profile_id)
    _validate_registry_details(checks, registry_payload)
    for target_profile_id in target_profile_ids:
        config = profile_configs.get(target_profile_id)
        if config is None:
            config = _load_profile_for_validation(checks, target_profile_id)
        _validate_plan_capabilities(checks, target_profile_id, plan, registry_payload)
        _validate_profile_config(checks, target_profile_id, config, plan, registry_payload)
    schedule = operation_schedule if operation_schedule is not None else _load_operation_schedule_for_validation()
    _validate_safety_config(checks, provider_payload, schedule, settings, strict)

    fail_count = sum(1 for item in checks if item["status"] == "FAIL")
    warn_count = sum(1 for item in checks if item["status"] == "WARN")
    exit_code = 1 if fail_count or (strict and warn_count) else 0
    return {
        "status": "FAILED" if fail_count else "OK_WITH_WARNINGS" if warn_count else "OK",
        "profile_id": profile_id or "all",
        "profile_ids": target_profile_ids,
        "jquants_plan": plan,
        "jquants_plan_source": jquants_validation["source"],
        "jquants_config_path": jquants_validation.get("config_path"),
        "capabilities": jquants_validation["capabilities"],
        "fail_count": fail_count,
        "warn_count": warn_count,
        "checked_profiles": len(target_profile_ids),
        "strict": strict,
        "exit_code": exit_code,
        "checks": checks,
    }


def _validation_check(checks: list[dict[str, str]], ok: bool, name: str, ok_message: str, fail_message: str, fail_status: str = "FAIL") -> None:
    checks.append({"status": "OK" if ok else fail_status, "name": name, "message": ok_message if ok else fail_message})


def _load_profile_registry_for_validation() -> dict[str, Any]:
    path = ROOT / "config" / "profile_registry.yaml"
    if not path.exists():
        return {"profiles": {}}
    try:
        return load_profile_registry(path)
    except Exception:
        return {"profiles": {}}


def _validation_target_profiles(profile_id: str | None, registry: dict[str, Any]) -> list[str]:
    if profile_id:
        return [profile_id]
    return sorted(registry_profiles(registry).keys())


def _load_profile_for_validation(checks: list[dict[str, str]], profile_id: str) -> dict[str, Any]:
    profile_path = ROOT / "config" / "profiles" / f"{profile_id}.yaml"
    _validation_check(
        checks,
        profile_path.exists(),
        f"profile.{profile_id}.yaml",
        f"{profile_path.relative_to(ROOT)} exists",
        f"{profile_path.relative_to(ROOT)} is missing",
    )
    try:
        config = load_profile(profile_id)
        _validation_check(checks, True, f"profile.{profile_id}.load", f"{profile_id} can be loaded", "")
        return config
    except Exception as exc:
        _validation_check(checks, False, f"profile.{profile_id}.load", "", f"{profile_id} cannot be loaded: {exc}")
        return {}


def _load_jquants_validation_config(provider_config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    class Args:
        jquants_plan = settings.get("jquants_plan") if settings.get("sources", {}).get("jquants_plan") == "cli" else None

    resolution = resolve_jquants_plan(args=Args(), config_root=ROOT, provider_config=provider_config)
    return {
        "plan": resolution.plan,
        "raw_plan": resolution.plan,
        "source": resolution.source,
        "config_path": resolution.config_path,
        "capabilities": resolution.capabilities,
        "requests_per_minute": resolution.requests_per_minute,
        "parallel_fetch": resolution.parallel_fetch,
        "supported_date_ranges": resolution.supported_date_ranges,
        "warnings": resolution.warnings,
    }


def _validate_jquants_config(checks: list[dict[str, str]], jquants_validation: dict[str, Any]) -> None:
    raw_plan = str(jquants_validation.get("raw_plan") or "")
    plan_ok = raw_plan in {"free", "light"}
    _validation_check(
        checks,
        plan_ok,
        "jquants.plan",
        f"jquants.plan is {jquants_validation['plan']}",
        f"jquants.plan must be free or light: {raw_plan}",
    )
    for capability, status in jquants_validation.get("capabilities", {}).items():
        checks.append({"status": "OK", "name": f"jquants.capability.{capability}", "message": status})
    for warning in jquants_validation.get("warnings", []):
        checks.append({"status": "WARN", "name": "jquants.plan_resolution", "message": warning})
    ranges = jquants_validation.get("supported_date_ranges", {})
    for endpoint, earliest in ranges.items():
        checks.append({"status": "OK", "name": f"jquants.supported_date.{endpoint}", "message": f"{endpoint} earliest: {earliest}"})


def _validate_registry_consistency(checks: list[dict[str, str]], registry: dict[str, Any], profile_id: str | None) -> None:
    registry_path = ROOT / "config" / "profile_registry.yaml"
    _validation_check(
        checks,
        registry_path.exists(),
        "profile_registry_exists",
        "config/profile_registry.yaml exists",
        "config/profile_registry.yaml is missing",
    )
    profiles = registry_profiles(registry)
    registered_missing = []
    for registered_id in profiles:
        if not (ROOT / "config" / "profiles" / f"{registered_id}.yaml").exists():
            registered_missing.append(registered_id)
    profile_files = {path.stem for path in (ROOT / "config" / "profiles").glob("*.yaml")}
    unregistered_experiments = sorted(
        profile_file
        for profile_file in profile_files - set(profiles)
        if profile_file.startswith("rookie_dealer_02_v2_")
    )
    _validation_check(
        checks,
        not registered_missing,
        "profile_registry_files",
        "all registry profiles have yaml files",
        f"registry profiles missing yaml: {', '.join(registered_missing)}",
    )
    checks.append(
        {
            "status": "WARN" if unregistered_experiments else "OK",
            "name": "profile_registry_unregistered_files",
            "message": (
                f"unregistered experiment profile files: {', '.join(unregistered_experiments)}"
                if unregistered_experiments
                else "no unregistered experiment profile files"
            ),
        }
    )
    if profile_id is None:
        checks.append({"status": "OK", "name": "profile_registry_target", "message": "all registry profiles selected"})
    elif profile_id in profiles:
        checks.append({"status": "OK", "name": "profile_registry_target", "message": f"{profile_id} is registered"})
    else:
        checks.append({"status": "WARN", "name": "profile_registry_target", "message": f"{profile_id} is not listed in profile_registry.yaml"})


def _validate_registry_details(checks: list[dict[str, str]], registry: dict[str, Any]) -> None:
    result = profile_registry_service.validate_registry(registry)
    for item in result.get("checks", []):
        checks.append(
            {
                "status": item["status"],
                "name": f"registry.{item['name']}",
                "message": item["message"],
            }
        )


def _validate_plan_capabilities(checks: list[dict[str, str]], profile_id: str, plan: str, registry: dict[str, Any] | None = None) -> None:
    compatibility = jquants_profile_compatibility(profile_id, plan)
    registry_item = registry_profiles(registry).get(profile_id, {}) if registry else {}
    required_plan = registry_item.get("required_plan")
    if required_plan == "light" and plan == "free":
        checks.append(
            {
                "status": "WARN",
                "name": f"profile.{profile_id}.required_plan",
                "message": f"{profile_id} requires light but current plan is free",
            }
        )
    missing = compatibility.get("missing_capabilities", [])
    fallback = compatibility.get("fallback_applied", [])
    unresolved = compatibility.get("unresolved_missing_capabilities", [])
    if not missing:
        checks.append({"status": "OK", "name": f"profile.{profile_id}.jquants_capabilities", "message": "required capabilities are available"})
    elif unresolved:
        checks.append({"status": "FAIL", "name": f"profile.{profile_id}.jquants_capabilities", "message": f"missing capabilities: {', '.join(unresolved)}"})
    else:
        fallback_text = ", ".join(f"{item['capability']} ({item['policy']})" for item in fallback)
        checks.append({"status": "WARN", "name": f"profile.{profile_id}.jquants_capabilities", "message": f"missing capabilities have fallback: {fallback_text}"})


def _validate_profile_config(
    checks: list[dict[str, str]],
    profile_id: str,
    config: dict[str, Any],
    plan: str,
    registry: dict[str, Any] | None = None,
) -> None:
    prefix = f"profile.{profile_id}"
    _validation_check(checks, bool(config.get("profile_id")), f"{prefix}.profile_id", "profile_id is set", "profile_id is missing")
    _validation_check(checks, bool(config.get("profile_name")), f"{prefix}.profile_name", "profile_name is set", "profile_name is missing")
    _validation_check(checks, isinstance(config.get("scoring", {}), dict), f"{prefix}.scoring", "scoring config is present", "scoring config is missing")
    selection = config.get("selection", {}) if isinstance(config.get("selection"), dict) else {}
    has_threshold = any(key in selection for key in ["min_score", "fallback_min_score", "top_pick_min_score"])
    _validation_check(checks, has_threshold, f"{prefix}.selection_threshold", "selection threshold is set", "selection threshold is missing")
    _validate_score_config(checks, config, prefix)
    _validate_profile_feature_consistency(checks, profile_id, config, plan, registry)


def _validate_score_config(checks: list[dict[str, str]], config: dict[str, Any], prefix: str = "profile") -> None:
    stale_paths = _stale_score_config_paths(config)
    _validation_check(
        checks,
        not stale_paths,
        f"{prefix}.stale_score_components",
        "news_score, fixed financial_score, and base_score are not configured",
        f"stale score settings found: {', '.join(stale_paths)}",
    )
    theoretical_max = _configured_theoretical_max_score(config)
    min_score = _to_float(_config_get(config, ("selection", "min_score")))
    if min_score is None:
        checks.append({"status": "WARN", "name": f"{prefix}.selection_threshold_value", "message": "selection.min_score is not set"})
    else:
        _validation_check(
            checks,
            min_score <= theoretical_max,
            f"{prefix}.selection_threshold_value",
            f"selection.min_score {min_score:g} is within theoretical max {theoretical_max:g}",
            f"selection.min_score {min_score:g} exceeds theoretical max {theoretical_max:g}",
        )
    formula = str(_config_get(config, ("scoring", "total_score_formula"), "technical_score + market_context_score + penalty_score"))
    stale_terms = [term for term in ["news_score", "base_score"] if term in formula]
    if "financial_score" in formula and not _config_get(config, ("scoring", "use_financial_score"), False):
        stale_terms.append("financial_score")
    _validation_check(
        checks,
        not stale_terms,
        f"{prefix}.score_formula_terms",
        f"score formula is active-component only: {formula}",
        f"score formula contains inactive terms: {', '.join(stale_terms)}",
    )


def _validate_profile_feature_consistency(
    checks: list[dict[str, str]],
    profile_id: str,
    config: dict[str, Any],
    plan: str,
    registry: dict[str, Any] | None = None,
) -> None:
    prefix = f"profile.{profile_id}"
    features = config.get("features", {}) if isinstance(config.get("features"), dict) else {}
    scoring = config.get("scoring", {}) if isinstance(config.get("scoring"), dict) else {}
    registry_item = registry_profiles(registry).get(profile_id, {}) if registry else {}
    registry_features = registry_item.get("features", {}) if isinstance(registry_item.get("features"), dict) else {}
    _validate_registry_profile_feature_match(checks, prefix, "relative_strength", bool(registry_features.get("relative_strength")), bool(features.get("relative_strength")))
    _validate_registry_profile_feature_match(checks, prefix, "investor_context", bool(registry_features.get("investor_context")), bool(features.get("investor_context")))
    _validate_registry_profile_feature_match(checks, prefix, "financial_context", bool(registry_features.get("financial_context")), bool(features.get("financial_context")))
    earnings_filter = config.get("earnings_filter", {}) if isinstance(config.get("earnings_filter"), dict) else {}
    _validate_registry_profile_feature_match(checks, prefix, "earnings_filter", bool(registry_features.get("earnings_filter")), bool(earnings_filter.get("enabled")))
    _validation_check(
        checks,
        not scoring.get("use_relative_strength_score") or bool(features.get("relative_strength")),
        f"{prefix}.relative_strength_feature",
        "relative_strength score feature is consistent",
        "use_relative_strength_score=true but features.relative_strength is not enabled",
    )
    _validation_check(
        checks,
        not scoring.get("use_investor_context_score") or bool(features.get("investor_context")),
        f"{prefix}.investor_context_feature",
        "investor_context score feature is consistent",
        "use_investor_context_score=true but features.investor_context is not enabled",
    )
    _validation_check(
        checks,
        not scoring.get("use_financial_score") or bool(features.get("financial_context")),
        f"{prefix}.financial_context_feature",
        "financial score feature is consistent",
        "use_financial_score=true but features.financial_context is not enabled",
    )
    for feature_name, data_enabled, scoring_enabled in [
        ("relative_strength", bool(features.get("relative_strength")), bool(scoring.get("use_relative_strength_score"))),
        ("investor_context", bool(features.get("investor_context")), bool(scoring.get("use_investor_context_score"))),
        ("financial_context", bool(features.get("financial_context")), bool(scoring.get("use_financial_score"))),
    ]:
        if data_enabled and not scoring_enabled:
            checks.append(
                {
                    "status": "OK",
                    "name": f"{prefix}.{feature_name}_data_only",
                    "message": f"{feature_name} data_enabled=true and scoring_enabled=false; data_only mode",
                }
            )
    if earnings_filter.get("enabled"):
        if jquants_has_capability(plan, "earnings_calendar"):
            checks.append({"status": "OK", "name": f"{prefix}.earnings_filter_capability", "message": "earnings_calendar capability is available"})
        else:
            checks.append({"status": "FAIL", "name": f"{prefix}.earnings_filter_capability", "message": "earnings_filter requires earnings_calendar capability"})
    light_features = []
    if features.get("relative_strength") and not jquants_has_capability(plan, "topix_prices"):
        light_features.append("relative_strength/topix_prices")
    if features.get("investor_context") and not jquants_has_capability(plan, "investor_types"):
        light_features.append("investor_context/investor_types")
    if light_features:
        checks.append({"status": "WARN", "name": f"{prefix}.light_features_on_free", "message": f"light-only features need fallback or disable: {', '.join(light_features)}"})


def _validate_registry_profile_feature_match(
    checks: list[dict[str, str]],
    prefix: str,
    feature_name: str,
    registry_enabled: bool,
    profile_data_enabled: bool,
) -> None:
    if registry_enabled and not profile_data_enabled:
        checks.append(
            {
                "status": "WARN",
                "name": f"{prefix}.{feature_name}_registry_mismatch",
                "message": f"registry features.{feature_name}=true but profile data_enabled=false",
            }
        )
    else:
        checks.append(
            {
                "status": "OK",
                "name": f"{prefix}.{feature_name}_registry_match",
                "message": f"registry/profile {feature_name} settings are consistent",
            }
        )


def _stale_score_config_paths(config: dict[str, Any]) -> list[str]:
    stale: list[str] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            current = (*path, str(key))
            if key in {"news_score", "base_score"}:
                stale.append(".".join(current))
            if key == "financial_score" and isinstance(child, (int, float)):
                stale.append(".".join(current))
            visit(child, current)

    visit(config, ())
    return stale


def _configured_theoretical_max_score(config: dict[str, Any]) -> float:
    scoring = config.get("scoring", {}) if isinstance(config.get("scoring"), dict) else {}
    max_score = 50.0
    if scoring.get("use_relative_strength_score"):
        max_score += float(scoring.get("relative_strength_score_weight", 10) or 10)
    if scoring.get("use_investor_context_score"):
        max_score += float(scoring.get("investor_context_score_weight", 5) or 5)
    if scoring.get("use_financial_score"):
        max_score += float(scoring.get("financial_score_weight", 10) or 10)
    return max_score


def _load_operation_schedule_for_validation() -> dict[str, Any]:
    path = ROOT / "config" / "operation_schedule.yaml"
    if not path.exists():
        return {}
    try:
        return load_operation_schedule(path)
    except Exception:
        return {}


def _validate_safety_config(
    checks: list[dict[str, str]],
    provider_config: dict[str, Any],
    schedule: dict[str, Any],
    runtime_settings: dict[str, Any],
    strict: bool = False,
) -> None:
    schedule_policy = schedule.get("execution_policy", {}) if isinstance(schedule.get("execution_policy"), dict) else {}
    schedule_safety = schedule.get("safety", {}) if isinstance(schedule.get("safety"), dict) else {}
    broker_mode = str(runtime_settings.get("broker_mode") or _config_get(provider_config, ("broker", "mode")) or schedule_policy.get("broker") or "paper")
    auto_order_enabled = bool(runtime_settings.get("auto_order_enabled") or _config_get(provider_config, ("operation", "auto_order_enabled")) or schedule_policy.get("auto_order_enabled", False))
    checks.append({"status": "OK", "name": "safety.broker_mode", "message": f"broker_mode: {broker_mode}"})
    if broker_mode == "tachibana_live":
        checks.append(
            {
                "status": "FAIL" if strict else "WARN",
                "name": "safety.live_broker_mode",
                "message": "broker mode is tachibana_live",
            }
        )
    checks.append(
        {
            "status": "OK" if broker_mode in {"paper", "tachibana_demo"} else "WARN",
            "name": "safety.broker_mode_allowed",
            "message": f"broker mode {broker_mode}",
        }
    )
    checks.append(
        {
            "status": "OK" if not auto_order_enabled else "WARN",
            "name": "safety.auto_order_enabled",
            "message": f"auto_order_enabled: {str(auto_order_enabled).lower()}",
        }
    )
    live_auto_order = broker_mode == "tachibana_live" and auto_order_enabled
    _validation_check(checks, not live_auto_order, "safety.live_auto_order", "live auto order is disabled", "live auto order is enabled")
    require_manual = bool(schedule_safety.get("require_manual_approval", False))
    forbid_live = bool(schedule_safety.get("forbid_live_auto_order", schedule_policy.get("forbid_live_auto_order", False)))
    _validation_check(checks, require_manual, "safety.require_manual_approval", "manual approval is required", "require_manual_approval is false")
    _validation_check(checks, forbid_live, "safety.forbid_live_auto_order", "live auto order is forbidden", "forbid_live_auto_order is false")


def run_clean_command(
    mode: str,
    profile_id: str | None = None,
    provider_name: str | None = None,
    yes: bool = False,
    older_than_days: int | None = None,
    include_latest: bool = False,
    cache_kind: str = "all",
    verbose: bool = False,
) -> None:
    plan = build_clean_plan(mode, profile_id, provider_name, older_than_days, include_latest, cache_kind)
    result = execute_clean_targets([Path(item["path"]) for item in plan["targets"]], yes=yes)
    title = _clean_title(mode, yes)
    print(title)
    print(f"- target_profile: {profile_id or 'all'}")
    print(f"- files: {plan['file_count']}")
    print(f"- total_size: {_format_bytes(plan['total_size'])}")
    print(f"- latest_kept: {plan['latest_kept']}")
    if plan.get("oldest_mtime"):
        print(f"- oldest: {plan['oldest_mtime']}")
        print(f"- newest: {plan['newest_mtime']}")
    if verbose:
        for item in plan["targets"]:
            print(f"- {item['relative_path']} ({_format_bytes(item['size'])})")
    if not yes:
        print("- use --yes to delete")
    print(f"deleted_count: {result['deleted_count']}")


def _clean_title(mode: str, yes: bool) -> str:
    names = {
        "clean-reports": "Clean Reports",
        "clean-cache": "Clean Cache",
        "clean-experiments": "Clean Experiments",
        "clean-articles": "Clean Articles",
    }
    return f"{names.get(mode, mode)} {'Delete' if yes else 'Dry Run'}"


def build_clean_plan(
    mode: str,
    profile_id: str | None = None,
    provider_name: str | None = None,
    older_than_days: int | None = None,
    include_latest: bool = False,
    cache_kind: str = "all",
) -> dict[str, Any]:
    roots = _clean_roots(mode, profile_id, provider_name, cache_kind)
    cutoff = (datetime.now() - timedelta(days=older_than_days)).timestamp() if older_than_days is not None else None
    targets = []
    latest_kept = 0
    skipped_symlink = 0
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if path.is_symlink():
                skipped_symlink += 1
                continue
            if not path.is_file():
                continue
            if not _clean_path_allowed(path, mode):
                continue
            if not include_latest and _is_latest_file(path):
                latest_kept += 1
                continue
            stat = path.stat()
            if cutoff is not None and stat.st_mtime >= cutoff:
                continue
            if mode == "clean-articles" and profile_id and profile_id not in path.parts and profile_id not in path.name:
                continue
            targets.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(ROOT)) if _is_relative_to(path, ROOT) else str(path),
                    "size": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                }
            )
    mtimes = [item["mtime"] for item in targets]
    return {
        "mode": mode,
        "dry_run": True,
        "targets": sorted(targets, key=lambda item: item["relative_path"]),
        "file_count": len(targets),
        "total_size": sum(int(item["size"]) for item in targets),
        "latest_kept": latest_kept,
        "skipped_symlink": skipped_symlink,
        "oldest_mtime": min(mtimes) if mtimes else None,
        "newest_mtime": max(mtimes) if mtimes else None,
    }


def _clean_roots(mode: str, profile_id: str | None = None, provider_name: str | None = None, cache_kind: str = "all") -> list[Path]:
    if mode == "clean-reports":
        if profile_id:
            return [ROOT / "reports" / profile_id]
        reports = ROOT / "reports"
        roots = [reports / "profile_comparisons"]
        if reports.exists():
            roots.extend(
                path
                for path in reports.iterdir()
                if path.is_dir() and path.name not in {"articles", "experiments", "profile_comparisons"}
            )
        return roots
    if mode == "clean-experiments":
        return [ROOT / "reports" / "experiments"]
    if mode == "clean-articles":
        return [ROOT / "reports" / "articles"]
    if mode == "clean-cache":
        provider = provider_name or "jquants"
        base = ROOT / "data" / "cache" / provider
        if provider != "jquants":
            return []
        if cache_kind and cache_kind != "all":
            if cache_kind == "investor_types":
                return [base / cache_kind, base / "empty_ranges.json"]
            return [base / cache_kind]
        return [base]
    raise SystemExit(f"Unsupported clean mode: {mode}")


def build_clean_targets(mode: str, profile_id: str | None = None, provider_name: str | None = None) -> list[Path]:
    return [Path(item["path"]) for item in build_clean_plan(mode, profile_id, provider_name)["targets"]]


def execute_clean_targets(targets: list[Path], yes: bool = False) -> dict[str, Any]:
    deleted = []
    if yes:
        for target in targets:
            if target.is_symlink() or not target.is_file() or not _clean_path_allowed_for_any_mode(target):
                continue
            target.unlink()
            deleted.append(str(target))
    return {"dry_run": not yes, "target_count": len(targets), "deleted_count": len(deleted), "deleted": deleted}


def _clean_path_allowed(path: Path, mode: str) -> bool:
    allowed = {
        "clean-reports": [ROOT / "reports"],
        "clean-experiments": [ROOT / "reports" / "experiments"],
        "clean-articles": [ROOT / "reports" / "articles"],
        "clean-cache": [ROOT / "data" / "cache" / "jquants"],
    }
    return any(_is_relative_to(path.resolve(), root.resolve()) for root in allowed.get(mode, []))


def _clean_path_allowed_for_any_mode(path: Path) -> bool:
    return any(
        _clean_path_allowed(path, mode)
        for mode in ["clean-reports", "clean-experiments", "clean-articles", "clean-cache"]
    )


def _is_latest_file(path: Path) -> bool:
    return "latest" in path.name.lower()


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024


def run_storage_audit() -> None:
    audit = build_storage_audit()
    print("## Storage Audit")
    print()
    print("| path | size | file_count | category | safe_to_delete |")
    print("| --- | ---: | ---: | --- | --- |")
    for row in audit["paths"]:
        print(
            f"| {row['path']} | {_format_bytes(row['size'])} | {row['file_count']} | "
            f"{row['category']} | {row['safe_to_delete']} |"
        )
    print()
    print("## Largest Files")
    for item in audit["largest_files"][:20]:
        print(f"- {item['path']}: {_format_bytes(item['size'])}")
    print()
    print("## File Type Ranking")
    for item in audit["file_types"][:20]:
        print(f"- {item['extension']}: {_format_bytes(item['size'])} ({item['file_count']} files)")
    print()
    print("## Old Experiment Artifacts")
    for item in audit["experiments"][:20]:
        print(f"- {item['path']}: {_format_bytes(item['size'])} ({item['file_count']} files)")
    print()
    print("## Cache Duplication")
    duplication = audit["cache_duplication"]
    for key in [
        "profile_processed_size",
        "common_processed_size",
        "profile_indicator_files",
        "profile_candidate_files",
        "duplicate_indicator_dates",
        "duplicate_candidate_dates",
        "potential_savings",
    ]:
        value = duplication.get(key)
        if isinstance(value, int) and (key.endswith("_size") or key == "potential_savings"):
            value = _format_bytes(value)
        print(f"- {key}: {value}")
    print()
    print("## Cleanup Recommendation")
    for item in audit["cleanup_recommendation"]:
        print(f"- {item}")


def build_storage_audit() -> dict[str, Any]:
    snapshot = _storage_file_snapshot()
    paths = [
        ("data", ROOT / "data", "data", "no"),
        ("data/raw", ROOT / "data" / "raw", "raw_prices_and_master", "no"),
        ("data/processed", ROOT / "data" / "processed", "processed_cache", "partial"),
        ("data/cache", ROOT / "data" / "cache", "provider_cache", "caution"),
        ("reports", ROOT / "reports", "reports", "partial"),
        ("logs", ROOT / "logs", "logs", "yes"),
        ("storage", ROOT / "storage", "database", "no"),
        (".pytest_cache", ROOT / ".pytest_cache", "test_cache", "yes"),
    ]
    rows = []
    for label, path, category, safe in paths:
        summary = _snapshot_path_size_summary(snapshot, path)
        rows.append({"path": label, "category": category, "safe_to_delete": safe, **summary})
    pycache_summary = _snapshot_pycache_summary(snapshot)
    rows.append({"path": "__pycache__", "category": "python_cache", "safe_to_delete": "yes", **pycache_summary})
    return {
        "paths": sorted(rows, key=lambda item: item["size"], reverse=True),
        "largest_files": _snapshot_largest_files(snapshot, limit=30),
        "file_types": _snapshot_file_type_ranking(snapshot),
        "experiments": _snapshot_experiment_artifact_sizes(snapshot),
        "cache_duplication": _snapshot_processed_cache_duplication(snapshot),
        "cleanup_recommendation": _storage_cleanup_recommendations(rows),
    }


def run_cleanup_storage(
    *,
    apply: bool = False,
    keep_latest_experiments: int = 3,
    keep_days: int = 30,
    include_reports: bool = False,
    include_logs: bool = False,
    include_processed: bool = False,
    exclude_raw_prices: bool = True,
    exclude_jquants_cache: bool = True,
    verbose: bool = False,
) -> None:
    plan = build_cleanup_storage_plan(
        keep_latest_experiments=keep_latest_experiments,
        keep_days=keep_days,
        include_reports=include_reports,
        include_logs=include_logs,
        include_processed=include_processed,
        exclude_raw_prices=exclude_raw_prices,
        exclude_jquants_cache=exclude_jquants_cache,
    )
    result = execute_cleanup_storage_plan(plan, apply=apply)
    print(f"Cleanup Storage {'Apply' if apply else 'Dry Run'}")
    print(f"- files: {plan['file_count']}")
    print(f"- total_size: {_format_bytes(plan['total_size'])}")
    print(f"- keep_days: {keep_days}")
    print(f"- keep_latest_experiments: {keep_latest_experiments}")
    print(f"- deleted_count: {result['deleted_count']}")
    if verbose:
        for item in plan["targets"]:
            print(f"- {item['relative_path']} ({_format_bytes(item['size'])}) reason={item['reason']}")
    if not apply:
        print("- dry-run only; use --apply to delete")


def run_compact_processed_cache(apply: bool = False, verbose: bool = False) -> None:
    plan = build_compact_processed_cache_plan()
    result = execute_compact_processed_cache_plan(plan, apply=apply, verbose=verbose)
    print(f"Compact Processed Cache {'Apply' if apply else 'Dry Run'}")
    print(f"- candidate_files: {plan['file_count']}")
    print(f"- estimated_savings: {_format_bytes(plan['estimated_savings'])}")
    print(f"- common_targets: {plan['common_target_count']}")
    print(f"- compacted_count: {result['compacted_count']}")
    print(f"- skipped_count: {result['skipped_count']}")
    print(f"- saved_size: {_format_bytes(result['saved_size'])}")
    if result.get("skip_reasons"):
        print(f"- skip_reasons: {_compact_json(result['skip_reasons'])}")
    if verbose:
        for item in plan["targets"][:200]:
            print(f"- {item['relative_path']} -> {item['common_relative_path']} ({_format_bytes(item['size'])})")
    if not apply:
        print("- dry-run only; use --apply to compact")


def run_inspect_cache(target: str, date_text: str) -> None:
    inspection = inspect_cache(target, date_text, load_config(CONFIG_PATH))
    print(f"## Cache Inspection: {target} {date_text}")
    print()
    print(f"- path: {inspection.get('path') or '-'}")
    print(f"- row_count: {inspection['row_count']}")
    print(f"- column_count: {inspection['column_count']}")
    print(f"- file_size: {_format_bytes(inspection['file_size'])}")
    print()
    print("## Largest Fields")
    for item in inspection["largest_fields"][:20]:
        print(f"- {item['field']}: {_format_bytes(item['size'])} ({item['non_null_count']} non-null)")
    print()
    print("## Required Fields")
    for field in inspection["required_fields"]:
        print(f"- {field}")
    print()
    print("## Removable Fields")
    for field in inspection["removable_fields"]:
        print(f"- {field}")


def run_performance_audit() -> None:
    audit = build_performance_audit()
    print("## Performance Audit")
    print()
    for key in [
        "processed_indicator_total_size",
        "processed_candidate_total_size",
        "scored_candidate_total_size",
        "logs_total_size",
        "reports_total_size",
    ]:
        print(f"- {key}: {_format_bytes(audit[key])}")
    print()
    print("## Hot Files")
    for section in [
        "largest_indicator_files",
        "largest_candidate_files",
        "largest_scored_candidate_files",
        "largest_log_files",
    ]:
        print(f"### {section}")
        for item in audit[section]:
            print(f"- {item['path']}: {_format_bytes(item['size'])}")
    print()
    print("## Runtime Cost Estimate")
    for key, value in audit["runtime_cost_estimate"].items():
        print(f"- {key}: {value}")
    print()
    print("## Optimization Recommendation")
    for item in audit["optimization_recommendation"]:
        print(f"- {item}")


def build_performance_audit() -> dict[str, Any]:
    indicator_files = _find_named_json_files(ROOT / "data" / "processed", "indicators_*.json")
    candidate_files = _find_named_json_files(ROOT / "data" / "processed", "candidates_*.json")
    scored_files = _find_named_json_files(ROOT / "data" / "processed", "scored_candidates_*.json")
    log_files = _files_under(ROOT / "logs")
    report_files = _files_under(ROOT / "reports")
    samples = {
        "indicator_load_time_sample": _json_load_sample(indicator_files),
        "candidate_load_time_sample": _json_load_sample(candidate_files),
        "scoring_load_time_sample": _json_load_sample(scored_files),
        "json_parse_time_sample": _json_parse_sample(indicator_files + candidate_files + scored_files),
    }
    return {
        "processed_indicator_total_size": _unique_file_size_sum(indicator_files),
        "processed_candidate_total_size": _unique_file_size_sum(candidate_files),
        "scored_candidate_total_size": _unique_file_size_sum(scored_files),
        "logs_total_size": _unique_file_size_sum(log_files),
        "reports_total_size": _unique_file_size_sum(report_files),
        "largest_indicator_files": _largest_from_paths(indicator_files),
        "largest_candidate_files": _largest_from_paths(candidate_files),
        "largest_scored_candidate_files": _largest_from_paths(scored_files),
        "largest_log_files": _largest_from_paths(log_files),
        "runtime_cost_estimate": samples,
        "optimization_recommendation": _performance_recommendations(indicator_files, candidate_files, scored_files, log_files),
    }


def _find_named_json_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob(pattern) if path.is_file() and not path.is_symlink()]


def _files_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]


def _unique_file_size_sum(paths: list[Path]) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen:
            continue
        seen.add(inode)
        total += stat.st_size
    return total


def _largest_from_paths(paths: list[Path], limit: int = 10) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[int, int]] = set()
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen:
            continue
        seen.add(inode)
        rows.append({"path": str(path.relative_to(ROOT)) if _is_relative_to(path, ROOT) else str(path), "size": stat.st_size})
    return sorted(rows, key=lambda item: item["size"], reverse=True)[:limit]


def _sample_paths(paths: list[Path], limit: int = 5) -> list[Path]:
    if not paths:
        return []
    largest = [Path(item["path"]) for item in _largest_from_paths(paths, limit=limit)]
    resolved = []
    for item in largest:
        resolved.append(ROOT / item if not item.is_absolute() else item)
    return resolved


def _json_load_sample(paths: list[Path], limit: int = 5) -> dict[str, Any]:
    samples = []
    for path in _sample_paths(paths, limit):
        started = time.perf_counter()
        try:
            payload = read_json(path)
            elapsed = time.perf_counter() - started
            rows = 0
            if isinstance(payload, dict):
                for key in ["indicators", "candidates", "scores", "trades"]:
                    value = payload.get(key)
                    if isinstance(value, list):
                        rows = len(value)
                        break
            samples.append(
                {
                    "path": str(path.relative_to(ROOT)) if _is_relative_to(path, ROOT) else str(path),
                    "size": path.stat().st_size,
                    "seconds": round(elapsed, 4),
                    "rows": rows,
                }
            )
        except Exception as exc:
            samples.append({"path": str(path), "error": str(exc)})
    avg = sum(float(item.get("seconds", 0)) for item in samples) / len(samples) if samples else 0.0
    return {
        "sample_count": len(samples),
        "avg_seconds": round(avg, 4),
        "samples": samples,
    }


def _json_parse_sample(paths: list[Path], limit: int = 5) -> dict[str, Any]:
    samples = []
    for path in _sample_paths(paths, limit):
        try:
            text = path.read_text(encoding="utf-8")
            started = time.perf_counter()
            json.loads(text)
            elapsed = time.perf_counter() - started
            samples.append(
                {
                    "path": str(path.relative_to(ROOT)) if _is_relative_to(path, ROOT) else str(path),
                    "size": path.stat().st_size,
                    "seconds": round(elapsed, 4),
                }
            )
        except Exception as exc:
            samples.append({"path": str(path), "error": str(exc)})
    avg = sum(float(item.get("seconds", 0)) for item in samples) / len(samples) if samples else 0.0
    return {"sample_count": len(samples), "avg_seconds": round(avg, 4), "samples": samples}


def _performance_recommendations(indicators: list[Path], candidates: list[Path], scored: list[Path], logs: list[Path]) -> list[str]:
    recs = []
    indicator_size = _unique_file_size_sum(indicators)
    scored_size = _unique_file_size_sum(scored)
    log_size = _unique_file_size_sum(logs)
    if indicator_size > 1024**3:
        recs.append("common cache: keep indicators/candidates in data/processed/common and hardlink profile paths.")
        recs.append("compact cache: prune indicators to profile-required fields; avoid debug-only MACD/BB/ATR in fast mode.")
    if scored_size > 512 * 1024**2:
        recs.append("compact storage: use --storage-mode compact for run-experiments --fast-analysis.")
    if log_size > 1024**3:
        recs.append("debug log削減: avoid full_debug except while investigating a specific failure.")
    recs.extend(
        [
            "gzip/zstd compression: consider compressing cold JSON artifacts after experiments finish.",
            "SQLite/parquet検討: large score matrices are better stored column-wise for analysis scans.",
        ]
    )
    return recs


def inspect_cache(target: str, date_text: str, config: dict[str, Any]) -> dict[str, Any]:
    path = _inspect_cache_path(target, date_text, config)
    if path is None or not path.exists():
        raise SystemExit(f"Cache file not found for target={target} date={date_text}")
    payload = read_json(path)
    rows = _inspect_cache_rows(target, payload)
    columns = sorted({key for row in rows if isinstance(row, dict) for key in row})
    required = sorted(_inspect_required_fields(target, config))
    removable = sorted(set(columns) - set(required))
    return {
        "target": target,
        "date": date_text,
        "path": str(path.relative_to(ROOT)) if _is_relative_to(path, ROOT) else str(path),
        "row_count": len(rows),
        "column_count": len(columns),
        "file_size": path.stat().st_size,
        "largest_fields": _largest_row_fields(rows),
        "required_fields": required,
        "removable_fields": removable,
    }


def _inspect_cache_path(target: str, date_text: str, config: dict[str, Any]) -> Path | None:
    if target == "indicators":
        candidates = [
            _common_processed_cache_path(config, "indicators", date_text),
            processed_profile_path(config, f"indicators_{date_text}.json"),
            ROOT / "data" / "processed" / f"indicators_{date_text}.json",
        ]
    elif target == "candidates":
        candidates = [
            _common_processed_cache_path(config, "candidates", date_text),
            processed_profile_path(config, f"candidates_{date_text}.json"),
        ]
    elif target == "market_context":
        candidates = [ROOT / "data" / "processed" / f"market_context_{date_text}.json"]
    else:
        raise SystemExit(f"Unsupported inspect-cache target: {target}")
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def _inspect_cache_rows(target: str, payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    if target == "indicators":
        rows = payload.get("indicators", [])
    elif target == "candidates":
        rows = payload.get("candidates", [])
    else:
        rows = [payload]
    return [row for row in rows if isinstance(row, dict)]


def _inspect_required_fields(target: str, config: dict[str, Any]) -> set[str]:
    if target == "market_context":
        return {"date", "market_regime", "advance_ratio", "decline_ratio"}
    if target == "candidates":
        return _candidate_required_fields(config)
    if target != "indicators":
        return set()
    required = {
        "code",
        "name",
        "sector_name",
        "section",
        "market_section",
        "listing_market",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma5",
        "ma25",
        "rsi",
        "volume_ratio",
        "turnover_value",
        "five_day_volatility",
        "five_day_change_rate",
        "previous_close",
        "previous_ma5",
        "previous_ma25",
        "candle_type",
        "candle_body_rate",
        "upper_shadow_rate",
        "lower_shadow_rate",
        "close_position_in_range",
        "gap_rate",
        "candlestick_signals",
        "candlestick_score",
        "trend_score",
        "volume_score",
        "rsi_score",
    }
    if bool(config.get("features", {}).get("sector_analysis", True)):
        required.update({"sector_momentum_score", "sector_rank", "sector_comment"})
    if bool(config.get("features", {}).get("relative_strength")) or bool(config.get("scoring", {}).get("use_relative_strength_score")):
        required.update(
            {
                "stock_return_5d",
                "stock_return_10d",
                "stock_return_20d",
                "benchmark_source",
                "benchmark_return_5d",
                "benchmark_return_10d",
                "benchmark_return_20d",
                "relative_strength_5d",
                "relative_strength_10d",
                "relative_strength_20d",
                "relative_strength_score",
                "topix_records_loaded",
                "topix_api_calls",
            }
        )
    if _backtest_indicator_mode(config) == "full":
        required.update({"macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle", "bb_lower", "bb_position", "atr"})
    return required


def _candidate_required_fields(config: dict[str, Any]) -> set[str]:
    required = {
        "code",
        "name",
        "sector_name",
        "section",
        "market_section",
        "listing_market",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma5",
        "ma25",
        "volume_ratio",
        "rsi",
        "turnover_value",
        "five_day_volatility",
        "five_day_change_rate",
        "candle_type",
        "candle_body_rate",
        "upper_shadow_rate",
        "lower_shadow_rate",
        "close_position_in_range",
        "gap_rate",
        "candlestick_signals",
        "candlestick_score",
        "trend_score",
        "volume_score",
        "rsi_score",
        "fallback",
        "pass_reason",
        "ma_spread",
    }
    if bool(config.get("features", {}).get("sector_analysis", True)):
        required.update({"sector_momentum_score", "sector_rank", "sector_comment"})
    if bool(config.get("features", {}).get("relative_strength")) or bool(config.get("scoring", {}).get("use_relative_strength_score")):
        required.update(
            {
                "stock_return_5d",
                "stock_return_10d",
                "stock_return_20d",
                "benchmark_source",
                "benchmark_return_5d",
                "benchmark_return_10d",
                "benchmark_return_20d",
                "relative_strength_5d",
                "relative_strength_10d",
                "relative_strength_20d",
                "relative_strength_score",
                "topix_records_loaded",
                "topix_api_calls",
            }
        )
    return required


def _largest_row_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    field_sizes: dict[str, int] = {}
    non_null: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            field_sizes[key] = field_sizes.get(key, 0) + len(encoded.encode("utf-8"))
            if value is not None:
                non_null[key] = non_null.get(key, 0) + 1
    return [
        {"field": key, "size": size, "non_null_count": non_null.get(key, 0)}
        for key, size in sorted(field_sizes.items(), key=lambda item: item[1], reverse=True)
    ]


def build_cleanup_storage_plan(
    *,
    keep_latest_experiments: int = 3,
    keep_days: int = 30,
    include_reports: bool = False,
    include_logs: bool = False,
    include_processed: bool = False,
    exclude_raw_prices: bool = True,
    exclude_jquants_cache: bool = True,
) -> dict[str, Any]:
    cutoff = (datetime.now() - timedelta(days=max(0, keep_days))).timestamp()
    targets: list[dict[str, Any]] = []
    targets.extend(_cleanup_pycache_targets(cutoff))
    targets.extend(_cleanup_pytest_cache_targets(cutoff))
    if include_logs:
        targets.extend(_cleanup_files_under(ROOT / "logs" / "backtests", cutoff, "old_backtest_logs"))
        targets.extend(_cleanup_files_under(ROOT / "logs" / "scoring", cutoff, "old_scoring_logs"))
    if include_reports:
        targets.extend(_cleanup_files_under(ROOT / "reports" / "articles" / "backtests", cutoff, "old_backtest_articles"))
        targets.extend(_cleanup_matching_files(ROOT / "reports" / "backtests", cutoff, "day_*.md", "old_backtest_day_reports"))
        targets.extend(_cleanup_old_experiment_targets(cutoff, keep_latest_experiments))
    if include_processed:
        targets.extend(_cleanup_processed_profile_targets(cutoff))
    if not exclude_jquants_cache:
        targets.extend(_cleanup_files_under(ROOT / "data" / "cache" / "jquants" / "unsupported_ranges.json", cutoff, "old_unsupported_cache"))
    if not exclude_raw_prices:
        targets.extend(_cleanup_files_under(ROOT / "data" / "raw", cutoff, "raw_prices_explicitly_included"))
    deduped = {item["path"]: item for item in targets if _cleanup_storage_path_allowed(Path(item["path"]))}
    ordered = sorted(deduped.values(), key=lambda item: item["relative_path"])
    return {
        "targets": ordered,
        "file_count": len(ordered),
        "total_size": sum(int(item["size"]) for item in ordered),
    }


def execute_cleanup_storage_plan(plan: dict[str, Any], apply: bool = False) -> dict[str, Any]:
    deleted = []
    if apply:
        for item in plan.get("targets", []):
            path = Path(item["path"])
            if path.is_symlink() or not path.is_file() or not _cleanup_storage_path_allowed(path):
                continue
            path.unlink()
            deleted.append(str(path))
    return {"dry_run": not apply, "deleted_count": len(deleted), "deleted": deleted}


def _path_size_summary(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"size": 0, "file_count": 0}
    if path.is_file():
        return {"size": path.stat().st_size, "file_count": 1}
    size = 0
    count = 0
    seen_inodes: set[tuple[int, int]] = set()
    for root, dirs, files in os.walk(path):
        dirs[:] = [item for item in dirs if item not in {".git", ".venv"}]
        for name in files:
            file_path = Path(root) / name
            if file_path.is_symlink():
                continue
            try:
                stat = file_path.stat()
                inode = (stat.st_dev, stat.st_ino)
                if inode in seen_inodes:
                    count += 1
                    continue
                seen_inodes.add(inode)
                size += stat.st_size
                count += 1
            except OSError:
                continue
    return {"size": size, "file_count": count}


def _storage_file_snapshot() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_errors = 0
    for current_root, dirs, names in os.walk(ROOT):
        dirs[:] = [item for item in dirs if item not in {".git", ".venv"}]
        for name in names:
            path = Path(current_root) / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                stat = path.stat()
                rel = path.relative_to(ROOT)
            except OSError:
                seen_errors += 1
                continue
            except ValueError:
                continue
            entries.append(
                {
                    "path": path,
                    "relative_path": str(rel),
                    "parts": rel.parts,
                    "suffix": path.suffix.lower() or "(no extension)",
                    "size": stat.st_size,
                    "inode": (stat.st_dev, stat.st_ino),
                }
            )
    if seen_errors:
        entries.append(
            {
                "path": ROOT,
                "relative_path": "__scan_errors__",
                "parts": ("__scan_errors__",),
                "suffix": "(no extension)",
                "size": 0,
                "inode": (-1, -1),
                "scan_errors": seen_errors,
            }
        )
    return entries


def _snapshot_path_size_summary(snapshot: list[dict[str, Any]], path: Path) -> dict[str, int]:
    if not path.exists():
        return {"size": 0, "file_count": 0}
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return _path_size_summary(path)
    if path.is_file():
        try:
            return {"size": path.stat().st_size, "file_count": 1}
        except OSError:
            return {"size": 0, "file_count": 0}
    prefix = rel.parts
    size = 0
    count = 0
    seen_inodes: set[tuple[int, int]] = set()
    for item in snapshot:
        parts = item.get("parts", ())
        if len(parts) < len(prefix) or parts[: len(prefix)] != prefix:
            continue
        inode = item["inode"]
        count += 1
        if inode in seen_inodes:
            continue
        seen_inodes.add(inode)
        size += int(item["size"])
    return {"size": size, "file_count": count}


def _snapshot_pycache_summary(snapshot: list[dict[str, Any]]) -> dict[str, int]:
    size = 0
    count = 0
    seen_inodes: set[tuple[int, int]] = set()
    for item in snapshot:
        if "__pycache__" not in item.get("parts", ()):
            continue
        count += 1
        inode = item["inode"]
        if inode in seen_inodes:
            continue
        seen_inodes.add(inode)
        size += int(item["size"])
    return {"size": size, "file_count": count}


def _snapshot_largest_files(snapshot: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    files = []
    seen_inodes: set[tuple[int, int]] = set()
    for item in snapshot:
        inode = item["inode"]
        if inode in seen_inodes:
            continue
        seen_inodes.add(inode)
        files.append({"path": item["relative_path"], "size": int(item["size"])})
    return sorted(files, key=lambda row: row["size"], reverse=True)[:limit]


def _snapshot_file_type_ranking(snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ext: dict[str, dict[str, int]] = {}
    seen_inodes: set[tuple[int, int]] = set()
    for item in snapshot:
        inode = item["inode"]
        if inode in seen_inodes:
            continue
        seen_inodes.add(inode)
        row = by_ext.setdefault(str(item["suffix"]), {"size": 0, "file_count": 0})
        row["size"] += int(item["size"])
        row["file_count"] += 1
    return sorted(
        [{"extension": ext, **values} for ext, values in by_ext.items()],
        key=lambda row: row["size"],
        reverse=True,
    )


def _snapshot_experiment_artifact_sizes(snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_experiment: dict[str, dict[str, Any]] = {}
    seen_by_experiment: dict[str, set[tuple[int, int]]] = {}
    prefix = ("reports", "experiments")
    for item in snapshot:
        parts = item.get("parts", ())
        if len(parts) < 3 or parts[:2] != prefix:
            continue
        key = "/".join(parts[:3])
        seen = seen_by_experiment.setdefault(key, set())
        row = by_experiment.setdefault(key, {"path": key, "size": 0, "file_count": 0})
        row["file_count"] += 1
        inode = item["inode"]
        if inode in seen:
            continue
        seen.add(inode)
        row["size"] += int(item["size"])
    return sorted(by_experiment.values(), key=lambda row: row["size"], reverse=True)


def _snapshot_processed_cache_duplication(snapshot: list[dict[str, Any]]) -> dict[str, Any]:
    processed_prefix = ("data", "processed")
    common_prefix = ("data", "processed", "common")
    profile_size = 0
    common_size = 0
    profile_indicator_dates: dict[str, int] = {}
    profile_candidate_dates: dict[str, int] = {}
    duplicate_groups: dict[tuple[str, str], list[tuple[int, tuple[int, int]]]] = {}
    seen_profile_inodes: set[tuple[int, int]] = set()
    seen_common_inodes: set[tuple[int, int]] = set()
    for item in snapshot:
        parts = item.get("parts", ())
        if len(parts) < 4 or parts[:2] != processed_prefix:
            continue
        name = parts[-1]
        inode = item["inode"]
        size = int(item["size"])
        if parts[:3] == common_prefix:
            if inode not in seen_common_inodes:
                seen_common_inodes.add(inode)
                common_size += size
            continue
        if inode not in seen_profile_inodes:
            seen_profile_inodes.add(inode)
            profile_size += size
        if name.startswith("indicators_") and name.endswith(".json"):
            date_key = name.removeprefix("indicators_").removesuffix(".json")
            profile_indicator_dates[date_key] = profile_indicator_dates.get(date_key, 0) + 1
            duplicate_groups.setdefault(("indicators", date_key), []).append((size, inode))
        elif name.startswith("candidates_") and name.endswith(".json"):
            date_key = name.removeprefix("candidates_").removesuffix(".json")
            profile_candidate_dates[date_key] = profile_candidate_dates.get(date_key, 0) + 1
            duplicate_groups.setdefault(("candidates", date_key), []).append((size, inode))
    potential_savings = 0
    for rows in duplicate_groups.values():
        unique_by_inode = {}
        for size, inode in rows:
            unique_by_inode[inode] = size
        sizes = list(unique_by_inode.values())
        if len(sizes) > 1:
            potential_savings += sum(sizes) - max(sizes)
    return {
        "profile_processed_size": profile_size,
        "common_processed_size": common_size,
        "profile_indicator_files": sum(profile_indicator_dates.values()),
        "profile_candidate_files": sum(profile_candidate_dates.values()),
        "duplicate_indicator_dates": sum(1 for count in profile_indicator_dates.values() if count > 1),
        "duplicate_candidate_dates": sum(1 for count in profile_candidate_dates.values() if count > 1),
        "potential_savings": potential_savings,
    }


def _pycache_summary() -> dict[str, int]:
    size = 0
    count = 0
    for path in ROOT.rglob("__pycache__"):
        if ".venv" in path.parts:
            continue
        summary = _path_size_summary(path)
        size += summary["size"]
        count += summary["file_count"]
    return {"size": size, "file_count": count}


def _largest_files(root: Path, limit: int = 20) -> list[dict[str, Any]]:
    files = []
    seen_inodes: set[tuple[int, int]] = set()
    for current_root, dirs, names in os.walk(root):
        dirs[:] = [item for item in dirs if item not in {".git", ".venv"}]
        for name in names:
            path = Path(current_root) / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                stat = path.stat()
                inode = (stat.st_dev, stat.st_ino)
                if inode in seen_inodes:
                    continue
                seen_inodes.add(inode)
                files.append({"path": str(path.relative_to(ROOT)), "size": stat.st_size})
            except OSError:
                continue
    return sorted(files, key=lambda item: item["size"], reverse=True)[:limit]


def _file_type_ranking(root: Path) -> list[dict[str, Any]]:
    by_ext: dict[str, dict[str, int]] = {}
    seen_inodes: set[tuple[int, int]] = set()
    for current_root, dirs, names in os.walk(root):
        dirs[:] = [item for item in dirs if item not in {".git", ".venv"}]
        for name in names:
            path = Path(current_root) / name
            if path.is_symlink() or not path.is_file():
                continue
            ext = path.suffix.lower() or "(no extension)"
            try:
                stat = path.stat()
            except OSError:
                continue
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen_inodes:
                continue
            seen_inodes.add(inode)
            row = by_ext.setdefault(ext, {"size": 0, "file_count": 0})
            row["size"] += stat.st_size
            row["file_count"] += 1
    return sorted(
        [{"extension": ext, **values} for ext, values in by_ext.items()],
        key=lambda item: item["size"],
        reverse=True,
    )


def _experiment_artifact_sizes() -> list[dict[str, Any]]:
    root = ROOT / "reports" / "experiments"
    if not root.exists():
        return []
    items = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        summary = _path_size_summary(path)
        items.append({"path": str(path.relative_to(ROOT)), **summary})
    return sorted(items, key=lambda item: item["size"], reverse=True)


def _processed_cache_duplication() -> dict[str, Any]:
    processed = ROOT / "data" / "processed"
    common = processed / "common"
    profile_size = 0
    profile_indicator_dates: dict[str, int] = {}
    profile_candidate_dates: dict[str, int] = {}
    if processed.exists():
        seen_inodes: set[tuple[int, int]] = set()
        for profile_dir in processed.iterdir():
            if not profile_dir.is_dir() or profile_dir.name == "common":
                continue
            for path in profile_dir.glob("*.json"):
                try:
                    stat = path.stat()
                    inode = (stat.st_dev, stat.st_ino)
                    if inode not in seen_inodes:
                        seen_inodes.add(inode)
                        profile_size += stat.st_size
                except OSError:
                    continue
                if path.name.startswith("indicators_"):
                    profile_indicator_dates[path.name.removeprefix("indicators_").removesuffix(".json")] = profile_indicator_dates.get(path.name.removeprefix("indicators_").removesuffix(".json"), 0) + 1
                if path.name.startswith("candidates_"):
                    profile_candidate_dates[path.name.removeprefix("candidates_").removesuffix(".json")] = profile_candidate_dates.get(path.name.removeprefix("candidates_").removesuffix(".json"), 0) + 1
    return {
        "profile_processed_size": profile_size,
        "common_processed_size": _path_size_summary(common)["size"],
        "profile_indicator_files": sum(profile_indicator_dates.values()),
        "profile_candidate_files": sum(profile_candidate_dates.values()),
        "duplicate_indicator_dates": sum(1 for count in profile_indicator_dates.values() if count > 1),
        "duplicate_candidate_dates": sum(1 for count in profile_candidate_dates.values() if count > 1),
        "potential_savings": _processed_cache_potential_savings(),
    }


def _processed_cache_potential_savings() -> int:
    groups: dict[tuple[str, str], list[int]] = {}
    processed = ROOT / "data" / "processed"
    if not processed.exists():
        return 0
    for profile_dir in processed.iterdir():
        if not profile_dir.is_dir() or profile_dir.name == "common":
            continue
        for stage in ["indicators", "candidates"]:
            for path in profile_dir.glob(f"{stage}_*.json"):
                try:
                    config = load_profile(profile_dir.name)
                    common_path = _common_processed_cache_path(
                        config,
                        stage,
                        path.stem.removeprefix(f"{stage}_"),
                    )
                    if _same_inode(path, common_path):
                        continue
                except Exception:
                    pass
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                groups.setdefault((stage, path.name), []).append(size)
    savings = 0
    for sizes in groups.values():
        if len(sizes) > 1:
            savings += sum(sizes) - max(sizes)
    return savings


def build_compact_processed_cache_plan() -> dict[str, Any]:
    targets = []
    common_targets: set[str] = set()
    processed = ROOT / "data" / "processed"
    if not processed.exists():
        return {"targets": [], "file_count": 0, "estimated_savings": 0, "common_target_count": 0}
    for profile_dir in processed.iterdir():
        if not profile_dir.is_dir() or profile_dir.name == "common":
            continue
        profile_id = profile_dir.name
        try:
            config = load_profile(profile_id)
        except Exception:
            continue
        for stage in ["indicators", "candidates"]:
            for path in sorted(profile_dir.glob(f"{stage}_*.json")):
                target_date_text = path.stem.removeprefix(f"{stage}_")
                common_path = _common_processed_cache_path(config, stage, target_date_text)
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if _same_inode(path, common_path):
                    continue
                common_targets.add(str(common_path))
                targets.append(
                    {
                        "profile_id": profile_id,
                        "stage": stage,
                        "date": target_date_text,
                        "path": str(path),
                        "relative_path": str(path.relative_to(ROOT)),
                        "common_path": str(common_path),
                        "common_relative_path": str(common_path.relative_to(ROOT)),
                        "size": stat.st_size,
                    }
                )
    estimated_savings = _estimate_compact_savings(targets)
    return {
        "targets": targets,
        "file_count": len(targets),
        "estimated_savings": estimated_savings,
        "common_target_count": len(common_targets),
    }


def execute_compact_processed_cache_plan(plan: dict[str, Any], apply: bool = False, verbose: bool = False) -> dict[str, Any]:
    compacted = 0
    skipped = 0
    saved_size = 0
    skip_reasons: dict[str, int] = {}
    if not apply:
        return {
            "dry_run": True,
            "compacted_count": 0,
            "skipped_count": 0,
            "saved_size": 0,
            "skip_reasons": {},
        }
    for item in plan.get("targets", []):
        path = Path(item["path"])
        common_path = Path(item["common_path"])
        result = _compact_one_processed_file(path, common_path)
        if result["status"] == "compacted":
            compacted += 1
            saved_size += int(item.get("size", 0))
            if verbose:
                print(f"compacted: {item['relative_path']} -> {item['common_relative_path']}")
        else:
            skipped += 1
            reason = result.get("reason", "unknown")
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if verbose:
                print(f"skipped: {item['relative_path']} reason={reason}")
    return {
        "dry_run": False,
        "compacted_count": compacted,
        "skipped_count": skipped,
        "saved_size": saved_size,
        "skip_reasons": skip_reasons,
    }


def _estimate_compact_savings(targets: list[dict[str, Any]]) -> int:
    groups: dict[str, list[int]] = {}
    for item in targets:
        groups.setdefault(str(item["common_path"]), []).append(int(item.get("size", 0)))
    savings = 0
    for sizes in groups.values():
        if len(sizes) > 1:
            savings += sum(sizes) - max(sizes)
    return savings


def _compact_one_processed_file(path: Path, common_path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        return {"status": "skipped", "reason": "invalid_source"}
    if not _is_relative_to(path.resolve(), (ROOT / "data" / "processed").resolve()):
        return {"status": "skipped", "reason": "outside_processed"}
    if _same_inode(path, common_path):
        return {"status": "skipped", "reason": "already_compacted"}
    common_path.parent.mkdir(parents=True, exist_ok=True)
    if not common_path.exists():
        try:
            path.replace(common_path)
            os.link(common_path, path)
            return {"status": "compacted", "reason": "moved_to_common"}
        except OSError as exc:
            return {"status": "skipped", "reason": f"link_failed:{exc.__class__.__name__}"}
    if not common_path.is_file():
        return {"status": "skipped", "reason": "common_not_file"}
    try:
        if path.stat().st_size != common_path.stat().st_size:
            return {"status": "skipped", "reason": "content_mismatch"}
        if _file_sha1(path) != _file_sha1(common_path):
            return {"status": "skipped", "reason": "content_mismatch"}
        path.unlink()
        os.link(common_path, path)
        return {"status": "compacted", "reason": "linked_to_common"}
    except OSError as exc:
        return {"status": "skipped", "reason": f"link_failed:{exc.__class__.__name__}"}


def _same_inode(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and left.stat().st_ino == right.stat().st_ino and left.stat().st_dev == right.stat().st_dev
    except OSError:
        return False


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _storage_cleanup_recommendations(rows: list[dict[str, Any]]) -> list[str]:
    recs = [
        "Run `python src/main.py --mode cleanup-storage --dry-run --include-logs --include-reports` to inspect safe log/report cleanup.",
        "Keep `data/raw/prices_*.json`, `data/cache/jquants/*`, and `storage/ai_fund_lab.sqlite3` unless explicitly rebuilding data.",
    ]
    processed = next((row for row in rows if row["path"] == "data/processed"), None)
    if processed and processed["size"] > 5 * 1024**3:
        recs.append("`data/processed` is large; enable common processed cache and consider `--include-processed` after confirming raw prices are preserved.")
    return recs


def _cleanup_pycache_targets(cutoff: float) -> list[dict[str, Any]]:
    targets = []
    for directory in ROOT.rglob("__pycache__"):
        if ".venv" in directory.parts:
            continue
        targets.extend(_cleanup_files_under(directory, cutoff, "python_bytecode_cache"))
    return targets


def _cleanup_pytest_cache_targets(cutoff: float) -> list[dict[str, Any]]:
    return _cleanup_files_under(ROOT / ".pytest_cache", cutoff, "pytest_cache")


def _cleanup_files_under(root: Path, cutoff: float, reason: str) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    paths = [root] if root.is_file() else list(root.rglob("*"))
    targets = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime > cutoff:
            continue
        targets.append(_cleanup_target(path, stat.st_size, reason))
    return targets


def _cleanup_matching_files(root: Path, cutoff: float, pattern: str, reason: str) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    targets = []
    for path in root.rglob(pattern):
        if path.is_symlink() or not path.is_file():
            continue
        stat = path.stat()
        if stat.st_mtime <= cutoff:
            targets.append(_cleanup_target(path, stat.st_size, reason))
    return targets


def _cleanup_old_experiment_targets(cutoff: float, keep_latest: int) -> list[dict[str, Any]]:
    root = ROOT / "reports" / "experiments"
    if not root.exists():
        return []
    dirs = sorted(
        [path for path in root.rglob("*") if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    keep = set(dirs[: max(0, keep_latest)])
    targets = []
    for directory in dirs:
        if directory in keep:
            continue
        targets.extend(_cleanup_files_under(directory, cutoff, "old_experiment_artifact"))
    return targets


def _cleanup_processed_profile_targets(cutoff: float) -> list[dict[str, Any]]:
    root = ROOT / "data" / "processed"
    if not root.exists():
        return []
    targets = []
    for profile_dir in root.iterdir():
        if not profile_dir.is_dir() or profile_dir.name == "common":
            continue
        for pattern in ["indicators_*.json", "candidates_*.json"]:
            targets.extend(_cleanup_matching_files(profile_dir, cutoff, pattern, "profile_processed_cache_rebuildable"))
    return targets


def _cleanup_target(path: Path, size: int, reason: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(ROOT)) if _is_relative_to(path, ROOT) else str(path),
        "size": int(size),
        "reason": reason,
    }


def _cleanup_storage_path_allowed(path: Path) -> bool:
    if path.is_symlink() or not _is_relative_to(path.resolve(), ROOT.resolve()):
        return False
    forbidden = [ROOT / "src", ROOT / "config", ROOT / "docs", ROOT / ".git", ROOT / ".venv"]
    if any(_is_relative_to(path.resolve(), item.resolve()) for item in forbidden if item.exists()):
        return False
    allowed = [
        ROOT / ".pytest_cache",
        ROOT / "logs" / "backtests",
        ROOT / "logs" / "scoring",
        ROOT / "reports" / "articles" / "backtests",
        ROOT / "reports" / "backtests",
        ROOT / "reports" / "experiments",
        ROOT / "data" / "processed",
        ROOT / "data" / "cache" / "jquants" / "unsupported_ranges.json",
    ]
    if path.name.endswith((".pyc", ".pyo")) and "__pycache__" in path.parts and ".venv" not in path.parts:
        return True
    return any(_is_relative_to(path.resolve(), item.resolve()) for item in allowed if item.exists())


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def run_healthcheck(provider_name: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("healthcheck mode currently supports --provider jquants only.")

    config = load_config(CONFIG_PATH)
    try:
        provider = JQuantsDataProvider(
            ROOT / ".env",
            plan=_jquants_plan(config),
            requests_per_minute=_jquants_requests_per_minute(config),
            parallel_fetch=_jquants_parallel_fetch(config),
            max_parallel_requests=_jquants_max_parallel_requests(config),
        )
        listed_stocks = provider.get_listed_stocks()
    except RuntimeError as exc:
        print(f"J-Quants connection failed: {exc}")
        raise SystemExit(1) from exc

    payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "provider": "jquants",
        "plan": _jquants_plan(config),
        "endpoint": "/equities/master",
        "listed_stocks_count": len(listed_stocks),
        "sample": listed_stocks[:5],
    }
    write_json(ROOT / "data" / "raw" / "jquants_healthcheck.json", payload)

    print("J-Quants connection successful")
    print("provider: jquants")
    print(f"plan: {_jquants_plan(config)}")
    print(f"listed stocks: {len(listed_stocks)}")


def run_tachibana_healthcheck(environment: str) -> None:
    config = load_config(CONFIG_PATH)
    payload = build_tachibana_healthcheck(config, environment)
    output_dir = ROOT / "reports" / profile_id_from(config) / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tachibana_healthcheck_latest.json"
    markdown_path = output_dir / "tachibana_healthcheck_latest.md"
    write_json(json_path, payload)
    write_text(markdown_path, render_tachibana_healthcheck_markdown(payload))

    print("Tachibana healthcheck completed")
    print(f"environment: {payload['environment']}")
    print(f"status: {payload['status']}")
    for check in payload["checks"]:
        print(f"[{check['status']}] {check['message']}")
    print(f"json: {json_path.relative_to(ROOT)}")
    print(f"markdown: {markdown_path.relative_to(ROOT)}")


def run_db_check(config: dict[str, Any]) -> None:
    result = database_schema_check(config, ROOT)
    print("DB Check")
    print(f"database_path: {Path(result['database_path']).relative_to(ROOT) if Path(result['database_path']).is_relative_to(ROOT) else result['database_path']}")
    for table in result.get("tables", []):
        expected = table.get("expected_insert_columns", [])
        status = table.get("status", "OK")
        print(f"- {table['table']}: {status} columns={table['column_count']}")
        if expected:
            print(f"  expected_insert_columns: {len(expected)}")
            missing = table.get("insert_missing_in_schema", [])
            if missing:
                print(f"  missing: {', '.join(missing)}")
    errors = result.get("errors", [])
    print("DB Check Summary")
    print(f"- tables: {len(result.get('tables', []))}")
    print(f"- errors: {len(errors)}")
    if errors:
        raise SystemExit(1)


def build_tachibana_healthcheck(config: dict[str, Any], environment: str = "demo") -> dict[str, Any]:
    checks = []
    tachibana = config.get("tachibana")
    if not isinstance(tachibana, dict):
        checks.append({"status": "FAIL", "message": "tachibana config is missing"})
        return _tachibana_healthcheck_payload(environment, checks, {})

    checks.append({"status": "OK", "message": "tachibana config exists"})
    configured_environment = _clean_config_string(tachibana.get("environment", "demo"))
    if configured_environment == environment:
        checks.append({"status": "OK", "message": f"tachibana.environment is {configured_environment}"})
    else:
        checks.append({"status": "WARN", "message": f"requested env is {environment}, config environment is {configured_environment}"})

    auth_method = _clean_config_string(tachibana.get("auth_method", ""))
    if auth_method == "public_key_v4r9":
        checks.append({"status": "OK", "message": "auth_method is public_key_v4r9"})
    else:
        checks.append({"status": "FAIL", "message": "auth_method must be public_key_v4r9"})

    demo_base_url = _clean_config_string(tachibana.get("demo_base_url", ""))
    if demo_base_url.endswith("/e_api_v4r9/"):
        checks.append({"status": "OK", "message": "demo_base_url is v4r9 URL"})
    else:
        checks.append({"status": "FAIL", "message": "demo_base_url must point to e_api_v4r9"})

    _load_env_file_to_process(ROOT / ".env")
    auth_config = load_tachibana_auth_config(config)
    env_values = _read_env_file(ROOT / ".env") if (ROOT / ".env").exists() else {}
    user_id_env = _clean_config_string(tachibana.get("user_id_env", "TACHIBANA_USER_ID"))
    password_env = _clean_config_string(tachibana.get("password_env", "TACHIBANA_PASSWORD"))
    second_password_env = _clean_config_string(tachibana.get("second_password_env", "TACHIBANA_SECOND_PASSWORD"))
    private_key_path_env = _clean_config_string(tachibana.get("private_key_path_env", "TACHIBANA_PRIVATE_KEY_PATH"))
    public_key_id_env = _clean_config_string(tachibana.get("public_key_id_env", "TACHIBANA_PUBLIC_KEY_ID"))
    for env_name in [user_id_env, password_env, second_password_env, private_key_path_env, public_key_id_env]:
        is_set = bool(env_values.get(env_name) or os.getenv(env_name))
        checks.append({"status": "OK" if is_set else "WARN", "message": f"{env_name} is {'set' if is_set else 'not set'}"})
    private_key_path_value = env_values.get(private_key_path_env) or os.getenv(private_key_path_env)
    if private_key_path_value:
        private_key_check = load_private_key(private_key_path_value)
        checks.append({"status": "OK" if private_key_check["file_exists"] else "WARN", "message": f"{private_key_path_env} file {'exists' if private_key_check['file_exists'] else 'does not exist'}"})
    else:
        checks.append({"status": "WARN", "message": f"{private_key_path_env} file check skipped because path is not set"})

    settings = {
        "environment": configured_environment,
        "demo_base_url": demo_base_url,
        "live_base_url": _clean_config_string(tachibana.get("live_base_url", "")),
        "auth_method": auth_config.auth_method,
        "request_timeout_seconds": auth_config.request_timeout_seconds,
        "account_type": tachibana.get("account_type"),
        "product": tachibana.get("product"),
        "market": tachibana.get("market"),
    }
    return _tachibana_healthcheck_payload(environment, checks, settings)


def _tachibana_healthcheck_payload(environment: str, checks: list[dict[str, str]], settings: dict[str, Any]) -> dict[str, Any]:
    failures = sum(1 for item in checks if item["status"] == "FAIL")
    warnings = sum(1 for item in checks if item["status"] == "WARN")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": environment,
        "status": "FAILED" if failures else "OK_WITH_WARNINGS" if warnings else "OK",
        "api_connection": "not_attempted",
        "order_sending": "disabled",
        "settings": settings,
        "checks": checks,
    }


def render_tachibana_healthcheck_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Tachibana Healthcheck",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Environment: {payload['environment']}",
        f"- Status: {payload['status']}",
        f"- API connection: {payload['api_connection']}",
        f"- Order sending: {payload['order_sending']}",
        "",
        "## Settings",
        "",
    ]
    settings = payload.get("settings", {})
    if settings:
        for key, value in settings.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- 設定なし")
    lines.extend(["", "## Checks", ""])
    for check in payload["checks"]:
        lines.append(f"- [{check['status']}] {check['message']}")
    lines.extend(["", "## Notes", "", "- 認証情報の値は表示・保存していません。", "- 実API通信と実発注は行っていません。"])
    return "\n".join(lines)


def run_init_db(config: dict[str, Any]) -> None:
    db_path = initialize_database(config, ROOT)
    print("SQLite database initialized")
    print(f"path: {db_path.relative_to(ROOT)}")


def run_account_snapshot(config: dict[str, Any]) -> None:
    state_path = ROOT / "logs" / "portfolio" / profile_id_from(config) / "state.json"
    state = read_json(state_path) if state_path.exists() else initial_live_paper_state(config)
    try:
        broker = build_broker(state, config)
        snapshot = account_snapshot(broker)
    except (LiveTradingDisabledError, NotImplementedError, ValueError) as exc:
        raise SystemExit(f"account-snapshot failed: {exc}") from exc
    output_dir = ROOT / "reports" / profile_id_from(config) / "broker"
    json_path = output_dir / "account_snapshot_latest.json"
    markdown_path = output_dir / "account_snapshot_latest.md"
    write_json(json_path, snapshot)
    write_text(markdown_path, render_account_snapshot(snapshot))
    print(render_account_snapshot(snapshot))
    print(f"json: {json_path.relative_to(ROOT)}")
    print(f"markdown: {markdown_path.relative_to(ROOT)}")


def run_demo_auto_order(config: dict[str, Any]) -> None:
    schedule = load_operation_schedule(ROOT / "config" / "operation_schedule.yaml")
    profile_id = profile_id_from(config)
    preview_path = latest_order_preview_path(ROOT, profile_id)
    preview = read_json(preview_path)
    orders = preview.get("preview_orders", [])
    state_path = ROOT / "logs" / "portfolio" / profile_id / "state.json"
    state = read_json(state_path) if state_path.exists() else initial_live_paper_state(config)
    try:
        broker = build_broker(state, config)
        result = execute_demo_auto_orders(config, schedule, orders, broker)
    except (DemoAutoOrderBlocked, LiveTradingDisabledError, NotImplementedError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"demo-auto-order stopped: {exc}") from exc
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "demo_orders.log"
    record = {
        "generated_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "profile_id": profile_id,
        "preview_path": str(preview_path.relative_to(ROOT)),
        **result,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"demo auto order status: {result['status']}")
    print(f"orders: {len(result['orders'])}")
    print(f"log: {log_path.relative_to(ROOT)}")


def run_analyze(config: dict[str, Any], start_date: str | None = None, end_date: str | None = None) -> None:
    try:
        analysis = analyze_operation_data(config, ROOT)
    except FileNotFoundError as exc:
        print(f"Analysis failed: {exc}")
        print("Run `python src/main.py --mode init-db` and then run daily operations first.")
        raise SystemExit(1) from exc
    except ValueError as exc:
        print(f"Analysis failed: {exc}")
        print("Run `python src/main.py --mode run-daily --provider jquants --date YYYY-MM-DD` first.")
        raise SystemExit(1) from exc

    output_dir = ROOT / "reports" / profile_id_from(config) / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis_latest.json"
    markdown_path = output_dir / "analysis_latest.md"
    feature_json_path = output_dir / "feature_analysis.json"
    feature_markdown_path = output_dir / "feature_analysis.md"
    selection_quality_json_path = output_dir / "selection_quality.json"
    selection_quality_markdown_path = output_dir / "selection_quality.md"
    feature_analysis = build_feature_analysis(config, ROOT, start_date, end_date)
    selection_quality_analysis = build_selection_quality_analysis(config, ROOT)
    analysis["selection_quality_analysis"] = selection_quality_analysis
    write_json(json_path, analysis)
    write_text(markdown_path, render_analysis_markdown(analysis))
    write_json(feature_json_path, feature_analysis)
    write_text(feature_markdown_path, render_feature_analysis_markdown(feature_analysis))
    write_json(selection_quality_json_path, selection_quality_analysis)
    write_text(selection_quality_markdown_path, render_selection_quality_markdown(selection_quality_analysis))

    portfolio = analysis["portfolio_analysis"]
    trades = analysis["trade_analysis"]
    trades_csv, db_trade_count, csv_trade_count = write_trades_csv_from_db(config)
    print("analysis completed")
    print(f"profile_id: {analysis.get('current_profile_id')}")
    print(f"profile_name: {analysis.get('current_profile_name')}")
    print(f"latest_total_assets: {_format_optional_number(portfolio['latest_total_assets'])}")
    print(f"cumulative_profit: {_format_optional_number(portfolio['cumulative_profit'])}")
    print(f"gross_cumulative_profit: {_format_optional_number(portfolio.get('gross_cumulative_profit'))}")
    print(f"net_cumulative_profit: {_format_optional_number(portfolio.get('net_cumulative_profit'))}")
    print(f"realized_profit: {_format_optional_number(portfolio.get('realized_profit'))}")
    print(f"unrealized_profit: {_format_optional_number(portfolio.get('unrealized_profit'))}")
    print(f"cash: {_format_optional_number(portfolio.get('cash'))}")
    print(f"positions_value: {_format_optional_number(portfolio.get('positions_value'))}")
    print(f"reconciliation_difference: {_format_optional_number(portfolio.get('reconciliation_difference'))}")
    print(f"reconciliation_ok: {portfolio.get('reconciliation_ok')}")
    print(f"estimated_tax_total: {_format_optional_number(portfolio.get('estimated_tax_total'))}")
    print(f"total_commission: {_format_optional_number(portfolio.get('total_commission'))}")
    print(f"win_rate: {_format_optional_percent(trades['win_rate'])}")
    print(f"gross_profit_total: {_format_optional_number(trades.get('gross_profit_total'))}")
    print(f"gross_loss_total: {_format_optional_number(trades.get('gross_loss_total'))}")
    print(f"profit_factor: {_format_optional_number(trades.get('profit_factor'))}")
    print(f"closed_trade_count: {trades.get('closed_trade_count')}")
    print(f"win_count: {trades.get('win_count')}")
    print(f"loss_count: {trades.get('loss_count')}")
    print(f"excluded_order_event_count: {trades.get('excluded_order_event_count')}")
    print(f"profit_ratio: {_format_optional_number(trades.get('profit_ratio'))}")
    print(f"expectancy: {_format_optional_percent(trades.get('expectancy'))}")
    print(f"worst_loss_profit_rate: {_format_optional_percent(trades.get('worst_loss_profit_rate'))}")
    print(f"loss_over_stop_count: {trades.get('loss_over_stop_count', 0)}")
    print(f"max_drawdown: {_format_optional_percent(portfolio['max_drawdown'])}")
    print(f"total_trades: {trades['total_trades']}")
    print(f"trades_csv: {trades_csv.relative_to(ROOT)} ({csv_trade_count}/{db_trade_count} rows)")
    print(f"markdown: {markdown_path.relative_to(ROOT)}")
    print(f"json: {json_path.relative_to(ROOT)}")
    print(f"feature_analysis_markdown: {feature_markdown_path.relative_to(ROOT)}")
    print(f"feature_analysis_json: {feature_json_path.relative_to(ROOT)}")
    print(f"selection_quality_markdown: {selection_quality_markdown_path.relative_to(ROOT)}")
    print(f"selection_quality_json: {selection_quality_json_path.relative_to(ROOT)}")


def run_compare_profiles(profile_ids: list[str], start_date_text: str, end_date_text: str) -> tuple[Path, Path]:
    _print_date_resolution("Compare Profiles Date Resolution", _runtime_date_resolution(start_date_text, end_date_text))
    profiles = [load_profile(profile_id) for profile_id in profile_ids]
    db_path = get_database_path(profiles[0], ROOT)
    if not db_path.exists():
        raise SystemExit(f"SQLite DB not found: {db_path}")

    rows = [_profile_compare_row(config, db_path, start_date_text, end_date_text) for config in profiles]
    ranking = build_profile_ranking(rows)
    profile_diff_analyses = build_profile_diff_analyses(profiles, db_path, start_date_text, end_date_text)
    profile_diff_analysis = profile_diff_analyses[0] if profile_diff_analyses else None
    payload = {
        "start_date": start_date_text,
        "end_date": end_date_text,
        "date_resolution": _runtime_date_resolution(start_date_text, end_date_text),
        "profiles": rows,
        "ranking": ranking,
        "profile_diff_analyses": profile_diff_analyses,
        "profile_diff_analysis": profile_diff_analysis,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    output_dir = ROOT / "reports" / "profile_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    key = f"{start_date_text}_to_{end_date_text}_{'_vs_'.join(profile_ids)}"
    json_path = output_dir / f"compare_{key}.json"
    markdown_path = output_dir / f"compare_{key}.md"
    write_json(json_path, payload)
    write_text(markdown_path, render_compare_profiles_markdown(payload))

    print("profile comparison completed")
    for row in rows:
        print(
            f"- {row['profile_id']}: final_assets={_format_optional_number(row.get('final_assets'))}, "
            f"net_cumulative_profit={_format_optional_number(row.get('net_cumulative_profit'))}, "
            f"win_rate={_format_optional_percent(row.get('win_rate'))}, "
            f"profit_factor={_format_optional_number(row.get('profit_factor'))}, "
            f"expectancy={_format_optional_percent(row.get('expectancy'))}, "
            f"total_trades={row.get('total_trades')}"
        )
    if ranking:
        print("Profile Ranking")
        for item in ranking:
            print(f"{item['rank']}位 {item['profile_id']} score={_format_optional_number(item['score'])}")
    if profile_diff_analyses:
        print("Profile Diff Analysis")
        for analysis in profile_diff_analyses:
            print(f"base_profile: {analysis['base_profile_id']}")
            print(f"target_profile: {analysis['target_profile_id']}")
            print(f"newly selected by target: {analysis['newly_selected_count']}")
            print(f"removed by target: {analysis['removed_count']}")
            print(f"outcome diff count: {analysis.get('outcome_diff_count', 0)}")
            print(f"practical effect: {analysis.get('practical_effect')}")
            if analysis.get("no_practical_effect"):
                print("No practical effect")
    print(f"markdown: {markdown_path.relative_to(ROOT)}")
    print(f"json: {json_path.relative_to(ROOT)}")
    return markdown_path, json_path


PROFILE_REGISTRY_PATH = ROOT / "config" / "profile_registry.yaml"


def load_profile_registry(path: Path = PROFILE_REGISTRY_PATH) -> dict[str, Any]:
    return profile_registry_service.load_profile_registry(path)


def registry_profiles(registry: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    return profile_registry_service.registry_profiles(registry)


def registry_profile_ids(registry: dict[str, Any] | None = None) -> list[str]:
    return sorted(registry_profiles(registry).keys())


def registry_experiment_profile_ids(base_profile_id: str, registry: dict[str, Any] | None = None) -> list[str]:
    return profile_registry_service.get_experiments(base_profile_id, registry)


def run_list_profiles() -> None:
    rows = build_profile_registry_rows(load_profile_registry())
    print("profile_id | role | required_plan | enabled_features | compare_to | description")
    for row in rows:
        print(
            f"{row['profile_id']} | {row['role']} | {row['required_plan']} | "
            f"{row['enabled_features'] or '-'} | {row['compare_to'] or '-'} | {row['description']}"
        )


def build_profile_registry_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "enabled_features": ", ".join(row.get("enabled_features") or []),
        }
        for row in profile_registry_service.list_profiles(registry, include_deprecated=True)
    ]


def run_profile_info(profile_id: str) -> None:
    info = build_profile_info(profile_id, load_profile_registry())
    print(f"profile_id: {info['profile_id']}")
    print(f"role: {info['role']}")
    print(f"required_plan: {info['required_plan']}")
    print(f"compare_to: {info['compare_to'] or '-'}")
    print(f"description: {info['description']}")
    print(f"enabled features: {', '.join(info['enabled_features']) or 'none'}")
    print(f"profile yaml path: {info['profile_yaml_path']}")
    print(f"score formula: {info['score_formula']}")
    print(f"required capabilities: {', '.join(info['required_capabilities']) or 'none'}")
    print(f"recommended backtest command: {info['recommended_backtest_command']}")
    print(f"recommended compare command: {info['recommended_compare_command']}")


def build_profile_info(profile_id: str, registry: dict[str, Any]) -> dict[str, Any]:
    try:
        return profile_registry_service.get_profile_info(profile_id, registry)
    except KeyError as exc:
        raise SystemExit(f"Profile not found in registry: {profile_id}")
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


def run_compare_experiments(base_profile_id: str | None, start_date_text: str | None = None, end_date_text: str | None = None) -> None:
    registry = load_profile_registry()
    base = base_profile_id or _registry_baseline_profile_id(registry)
    summary = build_experiment_comparison_summary(base, registry, start_date_text, end_date_text)
    output_dir = ROOT / "reports" / "profile_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "experiment_summary.md"
    json_path = output_dir / "experiment_summary.json"
    write_text(markdown_path, render_experiment_summary_markdown(summary))
    write_json(json_path, summary)
    print("experiment comparison summary completed")
    print(f"base_profile: {base}")
    print(f"experiments: {', '.join(summary['experiment_profiles']) or 'none'}")
    print(f"markdown: {markdown_path.relative_to(ROOT)}")
    print(f"json: {json_path.relative_to(ROOT)}")


def _runtime_date_resolution(start_date_text: str | None, end_date_text: str | None) -> dict[str, Any]:
    runtime = _runtime_settings_or_defaults().get("date_resolution", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    return {
        "requested_start_date": runtime.get("requested_start_date") or start_date_text,
        "requested_end_date": runtime.get("requested_end_date") or end_date_text,
        "effective_start_date": start_date_text,
        "effective_end_date": end_date_text,
        "start_date_source": runtime.get("start_date_source") or "default",
        "end_date_source": runtime.get("end_date_source") or "default",
    }


def _print_date_resolution(title: str, resolution: dict[str, Any]) -> None:
    print(title)
    print(f"requested_start_date: {resolution.get('requested_start_date')}")
    print(f"requested_end_date: {resolution.get('requested_end_date')}")
    print(f"effective_start_date: {resolution.get('effective_start_date')}")
    print(f"effective_end_date: {resolution.get('effective_end_date')}")
    print(f"source: start={resolution.get('start_date_source')} end={resolution.get('end_date_source')}")


def run_experiments(
    base_profile_id: str | None,
    start_date_text: str,
    end_date_text: str,
    requested_profiles: list[str] | None = None,
    skip_backtest: bool = False,
    skip_analyze: bool = False,
) -> None:
    global ACTIVE_PROFILE_ID, RUN_EXPERIMENTS_SHARED_STAGE_ACTIVE, RUN_EXPERIMENTS_SCORING_REUSE_SOURCE_BY_PROFILE, RUN_EXPERIMENTS_PERFORMANCE_REPORT
    registry = load_profile_registry()
    base = base_profile_id or _registry_baseline_profile_id(registry)
    experiment_ids = select_experiment_profiles(base, registry, requested_profiles)
    plan_resolution = _runtime_settings_or_defaults().get("jquants_plan_resolution")
    current_plan = (plan_resolution.plan if isinstance(plan_resolution, PlanResolution) else _runtime_settings_or_defaults().get("jquants_plan", "free"))
    date_resolution = _runtime_date_resolution(start_date_text, end_date_text)
    _print_date_resolution("Run Experiments Date Resolution", date_resolution)
    print("J-Quants Plan Resolution:")
    if isinstance(plan_resolution, PlanResolution):
        print(f"- plan: {plan_resolution.plan}")
        print(f"- source: {plan_resolution.source}")
        print(f"- config_path: {plan_resolution.config_path or 'N/A'}")
        print(f"- capabilities: {json.dumps(plan_resolution.capabilities, ensure_ascii=False, sort_keys=True)}")
    else:
        print(f"- plan: {current_plan}")
    capability = resolve_experiment_capabilities(experiment_ids, current_plan)
    runnable_experiment_ids = [item["profile_id"] for item in capability["runnable"]]
    skipped_profiles = capability["skipped"]
    profile_ids = [base, *runnable_experiment_ids]
    if skip_backtest:
        _ensure_experiment_db_rows(profile_ids, start_date_text, end_date_text)
    if skip_backtest and skip_analyze:
        _ensure_experiment_analysis_outputs(profile_ids)
    previous_profile = ACTIVE_PROFILE_ID
    compare_paths: tuple[Path, Path] | None = None
    experiment_api_summary = {
        "api_calls_by_endpoint": {},
        "api_errors_by_endpoint": {},
        "api_retry_count": {},
        "api_retry_success_count": {},
        "disabled_features_reason": {},
        "investor_types_chunks_total": 0,
        "investor_types_chunks_success": 0,
        "investor_types_chunks_failed": 0,
        "investor_types_records_loaded": 0,
        "investor_types_fetch_requested_start": "",
        "investor_types_fetch_clamped_start": "",
        "investor_types_fetch_start": "",
        "investor_types_fetch_end": "",
        "investor_types_disabled_reason": "",
    }
    try:
        if not skip_backtest:
            performance_report = prepare_run_experiments_common_stages(profile_ids, start_date_text, end_date_text)
            performance_report["skipped_profiles"] = [item.get("profile_id") for item in skipped_profiles]
            RUN_EXPERIMENTS_PERFORMANCE_REPORT = performance_report
            RUN_EXPERIMENTS_SCORING_REUSE_SOURCE_BY_PROFILE = _experiment_scoring_reuse_sources(profile_ids)
            RUN_EXPERIMENTS_SHARED_STAGE_ACTIVE = True
            for profile_id in profile_ids:
                ACTIVE_PROFILE_ID = profile_id
                print(f"run-experiments backtest start: {profile_id}")
                profile_started = time.perf_counter()
                run_backtest("jquants", start_date_text, end_date_text)
                profile_total = round(time.perf_counter() - profile_started, 4)
                scoring_seconds = round(float(BACKTEST_PROFILE_TIMINGS.get("scoring", 0.0)), 4)
                trade_seconds = round(float(BACKTEST_PROFILE_TIMINGS.get("trading", 0.0)), 4)
                performance_report["scoring_time"] = round(float(performance_report.get("scoring_time", 0.0)) + scoring_seconds, 4)
                performance_report["trade_time"] = round(float(performance_report.get("trade_time", 0.0)) + trade_seconds, 4)
                performance_report.setdefault("profile_scoring_time_by_profile", {})[profile_id] = scoring_seconds
                performance_report.setdefault("profile_trade_time_by_profile", {})[profile_id] = trade_seconds
                performance_report.setdefault("profile_total_time_by_profile", {})[profile_id] = profile_total
                _merge_jquants_api_session_summary(experiment_api_summary)
                if not skip_analyze and not SUMMARY_ONLY_ACTIVE:
                    print(f"run-experiments analyze start: {profile_id}")
                    run_analyze(load_config(CONFIG_PATH), start_date_text, end_date_text)
        elif not skip_analyze and not SUMMARY_ONLY_ACTIVE:
            for profile_id in profile_ids:
                ACTIVE_PROFILE_ID = profile_id
                print(f"run-experiments analyze start: {profile_id}")
                run_analyze(load_config(CONFIG_PATH), start_date_text, end_date_text)
        try:
            _print_date_resolution("Compare Profiles Date Resolution", date_resolution)
            compare_paths = run_compare_profiles(profile_ids, start_date_text, end_date_text)
        except SystemExit as exc:
            print(f"run-experiments compare warning: {exc}")
        summary = build_experiment_batch_summary(
            base,
            runnable_experiment_ids,
            registry,
            start_date_text,
            end_date_text,
            skipped_profiles=skipped_profiles,
            capability_warnings=capability["warnings"],
        )
        summary.update(experiment_api_summary)
        if not skip_backtest:
            summary["performance_report"] = performance_report
        write_experiment_batch_outputs(summary, start_date_text, end_date_text, base, compare_paths)
    finally:
        RUN_EXPERIMENTS_SHARED_STAGE_ACTIVE = False
        RUN_EXPERIMENTS_SCORING_REUSE_SOURCE_BY_PROFILE = {}
        RUN_EXPERIMENTS_PERFORMANCE_REPORT = {}
        ACTIVE_PROFILE_ID = previous_profile


def resolve_experiment_capabilities(experiment_ids: list[str], current_plan: str) -> dict[str, Any]:
    runnable = []
    skipped = []
    warnings = []
    for profile_id in experiment_ids:
        compatibility = jquants_profile_compatibility(profile_id, current_plan)
        unresolved = compatibility.get("unresolved_missing_capabilities", [])
        fallback = compatibility.get("fallback_applied", [])
        if unresolved:
            skipped.append(
                {
                    "profile_id": profile_id,
                    "status": "skipped",
                    "skip_reason": f"missing capabilities: {', '.join(unresolved)}",
                    "required_plan": registry_profiles().get(profile_id, {}).get("required_plan", ""),
                }
            )
            continue
        if fallback:
            warnings.append(
                {
                    "profile_id": profile_id,
                    "warning": "; ".join(f"{item['capability']}: {item['policy']}" for item in fallback),
                }
            )
        runnable.append({"profile_id": profile_id, "compatibility": compatibility})
    return {"runnable": runnable, "skipped": skipped, "warnings": warnings}


def prepare_run_experiments_common_stages(profile_ids: list[str], start_date_text: str, end_date_text: str) -> dict[str, Any]:
    global ACTIVE_PROFILE_ID, BACKTEST_MODE_ACTIVE, BACKTEST_DAY_LOG_PREFIX
    global COMMON_CACHE_METRICS
    started = time.perf_counter()
    previous_profile = ACTIVE_PROFILE_ID
    previous_backtest_mode = BACKTEST_MODE_ACTIVE
    report: dict[str, Any] = {
        "price_fetch_time": 0.0,
        "shared_price_fetch_time": 0.0,
        "indicator_time": 0.0,
        "shared_indicator_time": 0.0,
        "candidate_time": 0.0,
        "shared_candidate_time": 0.0,
        "market_context_time": 0.0,
        "scoring_time": 0.0,
        "trade_time": 0.0,
        "profile_scoring_time_by_profile": {},
        "profile_trade_time_by_profile": {},
        "profile_total_time_by_profile": {},
        "reused_indicator_count": 0,
        "reused_candidate_count": 0,
        "reused_scoring_count": 0,
        "cache_reused_from_common_count": 0,
        "profile_specific_cache_count": 0,
        "generated_cache_size": 0,
        "cleanup_hint": "",
        "skipped_profiles": [],
        "stage_groups": [],
    }
    if not profile_ids:
        return report
    end_date = date.fromisoformat(end_date_text)
    groups = _experiment_common_stage_groups(profile_ids)
    try:
        COMMON_CACHE_METRICS = {
            "cache_reused_from_common_count": 0,
            "profile_specific_cache_count": 0,
            "generated_cache_size": 0,
        }
        BACKTEST_MODE_ACTIVE = True
        price_start = time.perf_counter()
        indicator_fetch_start_dates = []
        trade_start_dates = []
        for group in groups:
            config = load_profile(group["representative"])
            parsed_start = date.fromisoformat(start_date_text)
            trade_start = max(parsed_start, jquants_earliest_supported_date(config, "prices") or parsed_start)
            trade_start_dates.append(trade_start)
            indicator_fetch_start_dates.append(_indicator_fetch_start_date(trade_start, config))
        fetch_start = min(indicator_fetch_start_dates) if indicator_fetch_start_dates else date.fromisoformat(start_date_text)
        trade_start = min(trade_start_dates) if trade_start_dates else date.fromisoformat(start_date_text)
        ACTIVE_PROFILE_ID = groups[0]["representative"]
        ensure_price_history_for_backtest("jquants", fetch_start, end_date, trade_start)
        trading_dates = available_cached_price_dates(trade_start, end_date)
        report["price_fetch_time"] = round(time.perf_counter() - price_start, 4)
        report["shared_price_fetch_time"] = report["price_fetch_time"]
        report["target_trading_days"] = len(trading_dates)
        for group in groups:
            representative = group["representative"]
            group_profiles = group["profiles"]
            ACTIVE_PROFILE_ID = representative
            config = load_config(CONFIG_PATH)
            parsed_start = date.fromisoformat(start_date_text)
            group_trade_start = max(parsed_start, jquants_earliest_supported_date(config, "prices") or parsed_start)
            indicator_fetch_start = _indicator_fetch_start_date(group_trade_start, config)
            group_dates = [item for item in trading_dates if item >= group_trade_start]
            _preload_light_api_context(config, group_trade_start, end_date, indicator_fetch_start_date=indicator_fetch_start)
            group_report = {
                "signature": group["signature"],
                "representative": representative,
                "profiles": group_profiles,
                "trading_days": len(group_dates),
                "indicator_generated_or_reused": 0,
                "candidate_generated_or_reused": 0,
                "copied_indicator_count": 0,
                "copied_candidate_count": 0,
            }
            for index, trading_date in enumerate(group_dates, start=1):
                target_date_text = trading_date.isoformat()
                BACKTEST_DAY_LOG_PREFIX = f"[common {representative} {index}/{len(group_dates)}] {target_date_text}"
                indicator_start = time.perf_counter()
                ensure_indicators("jquants", target_date_text)
                report["indicator_time"] += time.perf_counter() - indicator_start
                group_report["indicator_generated_or_reused"] += 1
                market_start = time.perf_counter()
                ensure_market_context("jquants", target_date_text)
                report["market_context_time"] += time.perf_counter() - market_start
                candidate_start = time.perf_counter()
                ensure_screen("jquants", target_date_text)
                report["candidate_time"] += time.perf_counter() - candidate_start
                group_report["candidate_generated_or_reused"] += 1
                for profile_id in group_profiles:
                    if profile_id == representative:
                        continue
                    copied_indicator = _copy_common_indicator_stage(representative, profile_id, target_date_text)
                    copied_candidate = _copy_common_candidate_stage(representative, profile_id, target_date_text)
                    if copied_indicator:
                        report["reused_indicator_count"] += 1
                        group_report["copied_indicator_count"] += 1
                    if copied_candidate:
                        report["reused_candidate_count"] += 1
                        group_report["copied_candidate_count"] += 1
            report["stage_groups"].append(group_report)
        report["indicator_time"] = round(float(report["indicator_time"]), 4)
        report["candidate_time"] = round(float(report["candidate_time"]), 4)
        report["shared_indicator_time"] = report["indicator_time"]
        report["shared_candidate_time"] = report["candidate_time"]
        report["market_context_time"] = round(float(report["market_context_time"]), 4)
        report.update(COMMON_CACHE_METRICS)
        report["cleanup_hint"] = (
            "Run `python src/main.py --mode storage-audit` and "
            "`python src/main.py --mode cleanup-storage --dry-run --include-logs --include-reports` "
            "when repeated experiments grow logs/reports."
        )
        report["total_common_stage_time"] = round(time.perf_counter() - started, 4)
        print_run_experiments_performance_report(report)
        return report
    finally:
        ACTIVE_PROFILE_ID = previous_profile
        BACKTEST_MODE_ACTIVE = previous_backtest_mode
        BACKTEST_DAY_LOG_PREFIX = ""


def _experiment_common_stage_groups(profile_ids: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for profile_id in profile_ids:
        signature = _experiment_common_stage_signature(load_profile(profile_id))
        grouped.setdefault(signature, []).append(profile_id)
    return [
        {"signature": signature, "representative": profiles[0], "profiles": profiles}
        for signature, profiles in grouped.items()
    ]


def _experiment_common_stage_signature(config: dict[str, Any]) -> str:
    return json.dumps(
        {
            "provider": "jquants",
            "indicator_mode": _backtest_indicator_mode(config),
            "relative_strength": _relative_strength_enabled_for_indicators(config),
            "sector_analysis": bool(config.get("features", {}).get("sector_analysis", True)),
            "market_filter": {
                "allowed_sections": sorted(allowed_market_sections(config)),
                "allow_unknown_market": bool(config.get("market_filter", {}).get("allow_unknown_market", False)),
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _experiment_scoring_reuse_sources(profile_ids: list[str]) -> dict[str, str]:
    source_by_signature: dict[str, str] = {}
    reuse: dict[str, str] = {}
    for profile_id in profile_ids:
        signature = _experiment_scoring_signature(load_profile(profile_id))
        source = source_by_signature.get(signature)
        if source:
            reuse[profile_id] = source
        else:
            source_by_signature[signature] = profile_id
    return reuse


def _experiment_scoring_signature(config: dict[str, Any]) -> str:
    keys = {
        "features": config.get("features", {}),
        "scoring": config.get("scoring", {}),
        "selection": config.get("selection", {}),
        "volume_filter": config.get("volume_filter", {}),
        "market_filter": {
            "allowed_sections": sorted(allowed_market_sections(config)),
            "allow_unknown_market": bool(config.get("market_filter", {}).get("allow_unknown_market", False)),
            "risk_off_buy_policy": config.get("market_filter", {}).get("risk_off_buy_policy"),
            "risk_off_max_buy_orders": config.get("market_filter", {}).get("risk_off_max_buy_orders"),
            "risk_off_min_score": config.get("market_filter", {}).get("risk_off_min_score"),
            "risk_off_disable_top_pick": config.get("market_filter", {}).get("risk_off_disable_top_pick"),
        },
        "earnings_filter": config.get("earnings_filter", {}),
        "investor_context_filter": config.get("investor_context_filter", {}),
    }
    return json.dumps(keys, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _copy_common_indicator_stage(source_profile_id: str, target_profile_id: str, target_date_text: str) -> bool:
    source_config = load_profile(source_profile_id)
    target_config = load_profile(target_profile_id)
    source_path = processed_profile_path(source_config, f"indicators_{target_date_text}.json")
    target_path = processed_profile_path(target_config, f"indicators_{target_date_text}.json")
    if not source_path.exists():
        return False
    if target_path.exists():
        return True
    payload = read_json(source_path)
    write_json(target_path, payload)
    return True


def _copy_common_candidate_stage(source_profile_id: str, target_profile_id: str, target_date_text: str) -> bool:
    source_config = load_profile(source_profile_id)
    target_config = load_profile(target_profile_id)
    source_path = processed_profile_path(source_config, f"candidates_{target_date_text}.json")
    target_path = processed_profile_path(target_config, f"candidates_{target_date_text}.json")
    if not source_path.exists():
        return False
    if not target_path.exists():
        payload = _with_profile_metadata(read_json(source_path), target_config)
        write_json(target_path, payload)
    source_log = ROOT / "logs" / "screening" / source_profile_id / f"screening_{target_date_text}.json"
    target_log = ROOT / "logs" / "screening" / target_profile_id / f"screening_{target_date_text}.json"
    if source_log.exists() and not target_log.exists():
        write_json(target_log, _with_profile_metadata(read_json(source_log), target_config))
    return True


def _with_profile_metadata(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(payload)
    updated["profile_id"] = profile_id_from(config)
    updated["profile_name"] = profile_name_from(config)
    updated["config_version"] = config_version_from(config)
    return updated


def _common_processed_cache_key(config: dict[str, Any], stage: str) -> str:
    if stage == "indicators":
        signature = _experiment_common_stage_signature(config)
    elif stage == "candidates":
        signature = json.dumps(
            {
                "common_stage": _experiment_common_stage_signature(config),
                "candidate_settings": {
                    "screening": config.get("screening", {}),
                    "features": {"sector_analysis": bool(config.get("features", {}).get("sector_analysis", True))},
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        signature = stage
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]


def _common_processed_cache_path(config: dict[str, Any], stage: str, target_date_text: str) -> Path:
    filename = f"{stage}_{target_date_text}.json"
    return ROOT / "data" / "processed" / "common" / stage / _common_processed_cache_key(config, stage) / filename


def _restore_common_processed_cache(config: dict[str, Any], stage: str, target_date_text: str, target_path: Path) -> bool:
    common_path = _common_processed_cache_path(config, stage, target_date_text)
    if not common_path.exists():
        return False
    payload = read_json(common_path)
    if stage == "candidates":
        payload = _with_profile_metadata(payload, config)
    write_json(target_path, payload)
    COMMON_CACHE_METRICS["cache_reused_from_common_count"] = COMMON_CACHE_METRICS.get("cache_reused_from_common_count", 0) + 1
    return True


def _save_common_processed_cache(config: dict[str, Any], stage: str, target_date_text: str, payload: dict[str, Any]) -> None:
    common_path = _common_processed_cache_path(config, stage, target_date_text)
    if not common_path.exists():
        write_json(common_path, payload)
        try:
            COMMON_CACHE_METRICS["generated_cache_size"] = COMMON_CACHE_METRICS.get("generated_cache_size", 0) + common_path.stat().st_size
        except OSError:
            pass


def _link_profile_processed_cache_to_common(config: dict[str, Any], stage: str, target_date_text: str, profile_path: Path) -> None:
    common_path = _common_processed_cache_path(config, stage, target_date_text)
    if not common_path.exists() or not profile_path.exists() or _same_inode(profile_path, common_path):
        return
    try:
        if profile_path.stat().st_size != common_path.stat().st_size:
            return
        if _file_sha1(profile_path) != _file_sha1(common_path):
            return
        profile_path.unlink()
        os.link(common_path, profile_path)
    except OSError:
        return


def print_run_experiments_performance_report(report: dict[str, Any]) -> None:
    print("Run Experiments Performance Report")
    for key in [
        "shared_price_fetch_time",
        "shared_indicator_time",
        "shared_candidate_time",
        "price_fetch_time",
        "indicator_time",
        "candidate_time",
        "market_context_time",
        "scoring_time",
        "trade_time",
        "profile_scoring_time_by_profile",
        "profile_trade_time_by_profile",
        "reused_scoring_count",
        "reused_indicator_count",
        "reused_candidate_count",
        "cache_reused_from_common_count",
        "profile_specific_cache_count",
        "generated_cache_size",
        "cleanup_hint",
        "skipped_profiles",
        "target_trading_days",
        "total_common_stage_time",
    ]:
        value = report.get(key, 0)
        if isinstance(value, (dict, list)):
            value = _compact_json(value)
        print(f"- {key}: {value}")


def _ensure_experiment_db_rows(profile_ids: list[str], start_date_text: str, end_date_text: str) -> None:
    db_path = get_database_path(load_profile(profile_ids[0]), ROOT)
    if not db_path.exists():
        raise SystemExit(f"--skip-backtest requires existing SQLite DB: {db_path}")
    missing = []
    with sqlite3.connect(db_path) as connection:
        for profile_id in profile_ids:
            count = connection.execute(
                """
                SELECT COUNT(*)
                FROM portfolio_snapshots
                WHERE profile_id = ? AND date BETWEEN ? AND ?
                """,
                (profile_id, start_date_text, end_date_text),
            ).fetchone()[0]
            if not count:
                missing.append(profile_id)
    if missing:
        raise SystemExit(f"--skip-backtest requires existing backtest rows for: {', '.join(missing)}")


def _ensure_experiment_analysis_outputs(profile_ids: list[str]) -> None:
    missing = []
    for profile_id in profile_ids:
        path = ROOT / "reports" / profile_id / "backtests" / "analysis_latest.json"
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
    if missing:
        raise SystemExit(f"--skip-analyze requires existing analysis outputs: {', '.join(missing)}")


def select_experiment_profiles(
    base_profile_id: str,
    registry: dict[str, Any],
    requested_profiles: list[str] | None = None,
) -> list[str]:
    experiments = registry_experiment_profile_ids(base_profile_id, registry)
    if not requested_profiles:
        return experiments
    requested = [profile_id for profile_id in requested_profiles if profile_id != base_profile_id]
    unknown = sorted(set(requested) - set(experiments))
    if unknown:
        raise SystemExit(f"Profiles are not experiments for base {base_profile_id}: {', '.join(unknown)}")
    return [profile_id for profile_id in experiments if profile_id in set(requested)]


def build_experiment_batch_summary(
    base_profile_id: str,
    experiment_ids: list[str],
    registry: dict[str, Any],
    start_date_text: str,
    end_date_text: str,
    skipped_profiles: list[dict[str, Any]] | None = None,
    capability_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    profiles = registry_profiles(registry)
    db_path = get_database_path(load_profile(base_profile_id), ROOT)
    rows_by_profile: dict[str, dict[str, Any]] = {}
    if db_path.exists():
        for profile_id in [base_profile_id, *experiment_ids]:
            rows_by_profile[profile_id] = _profile_compare_row(load_profile(profile_id), db_path, start_date_text, end_date_text)
    base_row = rows_by_profile.get(base_profile_id, {})
    base_date_audit = _experiment_date_range_audit(base_profile_id, start_date_text, end_date_text)
    experiment_rows = []
    for profile_id in experiment_ids:
        registry_item = profiles[profile_id]
        row = rows_by_profile.get(profile_id, {"profile_id": profile_id})
        row_market_coverage = row.get("market_coverage", {}) if isinstance(row.get("market_coverage"), dict) else {}
        diff = _experiment_diff(base_profile_id, profile_id, db_path, start_date_text, end_date_text) if db_path.exists() else {}
        feature_analysis = _experiment_feature_analysis(load_profile(profile_id), start_date_text, end_date_text) if db_path.exists() else {}
        activation = feature_analysis.get("feature_activation_audit", {})
        earnings_debug = feature_analysis.get("earnings_filter_debug", {})
        processed_audit = _experiment_processed_data_audit(profile_id, start_date_text, end_date_text)
        date_audit = _experiment_date_range_audit(profile_id, start_date_text, end_date_text)
        coverage_audit = date_audit.get("backtest_coverage_audit", {}) if isinstance(date_audit, dict) else {}
        if "selection_diff_count" in diff:
            selection_diff_count = int(diff.get("selection_diff_count") or 0)
        else:
            selection_diff_count = int(diff.get("newly_selected_count") or 0) + int(diff.get("removed_count") or 0)
        outcome_diff_count = int(diff.get("outcome_diff_count") or 0)
        practical_effect = diff.get("practical_effect") or _profile_practical_effect(selection_diff_count, outcome_diff_count)
        judgement = _experiment_judgement(base_row, row, diff)
        experiment_rows.append(
            {
                "profile_id": profile_id,
                "role": registry_item.get("role", ""),
                "description": registry_item.get("description", ""),
                "required_plan": registry_item.get("required_plan", ""),
                "enabled_features": _enabled_registry_features(registry_item),
                "final_assets": row.get("final_assets"),
                "net_cumulative_profit": row.get("net_cumulative_profit"),
                "win_rate": row.get("win_rate"),
                "profit_factor": row.get("profit_factor"),
                "expectancy": row.get("expectancy"),
                "max_drawdown": row.get("max_drawdown"),
                "total_trades": row.get("total_trades"),
                "newly_selected_count": diff.get("newly_selected_count", 0),
                "removed_count": diff.get("removed_count", 0),
                "investor_filter_rejected_count": diff.get("investor_filter_rejected_count", 0),
                "market_filter": row.get("market_filter", {}),
                "market_candidate_count": row_market_coverage.get("candidate_count", {}),
                "market_selected_count": row_market_coverage.get("selected_count", {}),
                "market_filter_excluded_count": row_market_coverage.get("market_filter_excluded_count", 0),
                "selection_diff_count": selection_diff_count,
                "outcome_diff_count": outcome_diff_count,
                "feature_data_enabled": activation.get("feature_data_enabled", {}),
                "feature_scoring_enabled": activation.get("feature_scoring_enabled", {}),
                "feature_trigger_count": activation.get("feature_trigger_count", {}),
                "earnings_calendar_records": earnings_debug.get("earnings_calendar_records", 0),
                "earnings_filter_rejected_count": earnings_debug.get("earnings_filter_rejected_count", 0),
                "earnings_filter_status": earnings_debug.get("status", ""),
                "processed_data_audit": processed_audit,
                "backtest_coverage_audit": coverage_audit,
                "first_price_date": coverage_audit.get("first_price_date"),
                "last_price_date": coverage_audit.get("last_price_date"),
                "first_candidate_date": coverage_audit.get("first_candidate_date"),
                "last_candidate_date": coverage_audit.get("last_candidate_date"),
                "first_trade_date": coverage_audit.get("first_trade_date"),
                "last_trade_date": coverage_audit.get("last_trade_date"),
                "candidate_days": coverage_audit.get("candidate_days"),
                "trade_days": coverage_audit.get("trade_days"),
                "coverage_ratio": coverage_audit.get("coverage_ratio"),
                "coverage_warning": coverage_audit.get("coverage_warning", ""),
                "indicators_last_date": processed_audit.get("indicators_last_date"),
                "candidates_last_date": processed_audit.get("candidates_last_date"),
                "scored_candidates_last_date": processed_audit.get("scored_candidates_last_date"),
                "feature_status": {
                    name: item.get("status")
                    for name, item in (activation.get("features", {}) or {}).items()
                },
                "inactive_in_practice": activation.get("inactive_in_practice", []),
                "practical_effect": practical_effect,
                "effect_reason": diff.get("effect_reason") or _profile_effect_reason(selection_diff_count, outcome_diff_count),
                "no_practical_effect": "yes" if practical_effect == "no_practical_effect" else "no",
                "judgement": judgement["judgement"],
                "judgement_reasons": judgement["reasons"],
                "verdict": judgement["judgement"],
                "verdict_reason": "; ".join(judgement["reasons"]),
                "candidate": "yes" if judgement["judgement"] == "candidate" else "no",
            }
        )
    for skipped in skipped_profiles or []:
        item = profiles.get(skipped["profile_id"], {})
        experiment_rows.append(
            {
                "profile_id": skipped["profile_id"],
                "role": item.get("role", "experiment"),
                "description": item.get("description", ""),
                "required_plan": item.get("required_plan", skipped.get("required_plan", "")),
                "enabled_features": _enabled_registry_features(item),
                "status": "skipped",
                "skip_reason": skipped.get("skip_reason", ""),
                "verdict": "skipped",
                "verdict_reason": skipped.get("skip_reason", ""),
                "practical_effect": "no",
                "candidate": "no",
            }
        )
    batch_date_resolution = _date_resolution_with_coverage(
        _runtime_date_resolution(start_date_text, end_date_text),
        base_date_audit,
    )
    return {
        "base_profile": base_profile_id,
        "start_date": start_date_text,
        "end_date": end_date_text,
        "date_resolution": batch_date_resolution,
        "backtest_coverage_audit": base_date_audit.get("backtest_coverage_audit", {}) if isinstance(base_date_audit, dict) else {},
        "profiles": [base_profile_id, *experiment_ids],
        "base": {
            "profile_id": base_profile_id,
            "description": profiles.get(base_profile_id, {}).get("description", ""),
            "required_plan": profiles.get(base_profile_id, {}).get("required_plan", ""),
            "role": profiles.get(base_profile_id, {}).get("role", ""),
            "enabled_features": _enabled_registry_features(profiles.get(base_profile_id, {})),
            "backtest_coverage_audit": base_date_audit.get("backtest_coverage_audit", {}) if isinstance(base_date_audit, dict) else {},
            **base_row,
        },
        "experiments": experiment_rows,
        "skipped_profiles": skipped_profiles or [],
        "capability_warnings": capability_warnings or [],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _experiment_date_range_audit(profile_id: str, start_date_text: str, end_date_text: str) -> dict[str, Any]:
    summary_path = ROOT / "logs" / "backtests" / profile_id / f"{start_date_text}_to_{end_date_text}" / "backtest_summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = read_json(summary_path)
    except Exception:
        return {}
    audit = summary.get("date_range_audit", {})
    return audit if isinstance(audit, dict) else {}


def _experiment_processed_data_audit(profile_id: str, start_date_text: str, end_date_text: str) -> dict[str, Any]:
    audit = _experiment_date_range_audit(profile_id, start_date_text, end_date_text).get("processed_data_audit", {})
    if isinstance(audit, dict) and audit:
        return audit
    try:
        trading_dates = available_cached_price_dates(date.fromisoformat(start_date_text), date.fromisoformat(end_date_text))
    except Exception:
        trading_dates = []
    return build_processed_data_audit(load_profile(profile_id), trading_dates)


def _experiment_diff(base_profile_id: str, target_profile_id: str, db_path: Path, start_date_text: str, end_date_text: str) -> dict[str, Any]:
    try:
        payload = build_profile_diff_analysis(
            [load_profile(base_profile_id), load_profile(target_profile_id)],
            db_path,
            start_date_text,
            end_date_text,
        )
    except Exception:
        return {}
    return payload or {}


def _experiment_feature_analysis(config: dict[str, Any], start_date_text: str, end_date_text: str) -> dict[str, Any]:
    try:
        return build_feature_analysis(config, ROOT, start_date_text, end_date_text)
    except Exception:
        return {}


def _enabled_registry_features(item: dict[str, Any]) -> list[str]:
    features = item.get("features", {}) if isinstance(item.get("features"), dict) else {}
    return [feature for feature, enabled in features.items() if bool(enabled)]


def _experiment_candidate(base_row: dict[str, Any], row: dict[str, Any]) -> bool:
    return _experiment_judgement(base_row, row)["judgement"] == "candidate"


def _experiment_judgement(base_row: dict[str, Any], row: dict[str, Any], diff: dict[str, Any] | None = None) -> dict[str, Any]:
    base_profit = _to_float(base_row.get("net_cumulative_profit"))
    target_profit = _to_float(row.get("net_cumulative_profit"))
    base_pf = _to_float(base_row.get("profit_factor"))
    target_pf = _to_float(row.get("profit_factor"))
    base_dd = _to_float(base_row.get("max_drawdown"))
    target_dd = _to_float(row.get("max_drawdown"))
    base_trades = _to_float(base_row.get("total_trades"))
    target_trades = _to_float(row.get("total_trades"))
    reasons: list[str] = []
    if None in {base_profit, target_profit, base_pf, target_pf, base_dd, target_dd, base_trades, target_trades}:
        return {"judgement": "needs_review", "reasons": ["missing_metrics"]}
    selection_diff_count = 0
    outcome_diff_count = 0
    if diff is not None:
        if "selection_diff_count" in diff:
            selection_diff_count = int(diff.get("selection_diff_count") or 0)
        else:
            selection_diff_count = int(diff.get("newly_selected_count") or 0) + int(diff.get("removed_count") or 0)
        outcome_diff_count = int(diff.get("outcome_diff_count") or 0)
    no_practical_diff = diff is not None and selection_diff_count == 0 and outcome_diff_count == 0
    metrics_identical = (
        target_profit == base_profit
        and target_pf == base_pf
        and target_dd == base_dd
        and target_trades == base_trades
    )
    if no_practical_diff:
        reasons.append("no_practical_effect")
    elif diff is not None and selection_diff_count == 0 and outcome_diff_count > 0:
        reasons.append("execution_or_exit_effect")
    elif diff is not None and selection_diff_count > 0:
        reasons.append("selection_effect")
    if no_practical_diff and metrics_identical:
        return {"judgement": "no_practical_effect", "reasons": reasons}
    trade_count_ok = target_trades >= base_trades * 0.5
    if not trade_count_ok:
        reasons.append("trade_count_below_50_percent")
    profit_ok = target_profit >= base_profit
    pf_ok = target_pf >= base_pf * 0.95
    drawdown_ok = target_dd >= base_dd - 0.05
    if not profit_ok:
        reasons.append("net_cumulative_profit_below_base")
    if not pf_ok:
        reasons.append("profit_factor_deteriorated")
    if not drawdown_ok:
        reasons.append("max_drawdown_deteriorated")
    if not trade_count_ok:
        return {"judgement": "needs_review", "reasons": reasons}
    if profit_ok and pf_ok and drawdown_ok:
        if reasons == ["no_practical_effect"]:
            return {"judgement": "needs_review", "reasons": reasons}
        return {"judgement": "candidate", "reasons": reasons or ["meets_candidate_criteria"]}
    if profit_ok:
        return {"judgement": "needs_review", "reasons": reasons}
    return {"judgement": "rejected", "reasons": reasons}


def write_experiment_batch_outputs(
    summary: dict[str, Any],
    start_date_text: str,
    end_date_text: str,
    base_profile_id: str | None = None,
    compare_paths: tuple[Path, Path] | None = None,
) -> tuple[Path, Path]:
    output_dir = experiment_batch_output_dir(start_date_text, end_date_text, base_profile_id or str(summary.get("base_profile") or "base"))
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "experiment_summary.md"
    json_path = output_dir / "experiment_summary.json"
    write_text(markdown_path, render_experiment_batch_markdown(summary))
    write_json(json_path, summary)
    if compare_paths:
        compare_markdown, compare_json = compare_paths
        if compare_markdown.exists():
            shutil.copyfile(compare_markdown, output_dir / "compare_profiles.md")
        if compare_json.exists():
            shutil.copyfile(compare_json, output_dir / "compare_profiles.json")
    for row in [summary.get("base", {}), *summary.get("experiments", [])]:
        profile_id = row.get("profile_id")
        if profile_id:
            write_text(profiles_dir / f"{profile_id}.md", render_experiment_profile_markdown(row))
    print("experiment batch completed")
    print(f"summary_md: {markdown_path.relative_to(ROOT)}")
    print(f"summary_json: {json_path.relative_to(ROOT)}")
    return markdown_path, json_path


def experiment_batch_output_dir(start_date_text: str, end_date_text: str, base_profile_id: str) -> Path:
    return ROOT / "reports" / "experiments" / f"{start_date_text}_to_{end_date_text}" / base_profile_id


def render_experiment_batch_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Experiment Batch Summary",
        "",
        f"- base_profile: {summary.get('base_profile')}",
        f"- period: {summary.get('start_date')} to {summary.get('end_date')}",
        f"- generated_at: {summary.get('generated_at')}",
    ]
    resolution = summary.get("date_resolution", {}) if isinstance(summary.get("date_resolution"), dict) else {}
    lines.extend(
        [
            "",
            "## Backtest Date Resolution",
            "",
            f"- requested_start_date: {resolution.get('requested_start_date', summary.get('start_date'))}",
            f"- requested_end_date: {resolution.get('requested_end_date', summary.get('end_date'))}",
            f"- effective_start_date: {resolution.get('effective_start_date', summary.get('start_date'))}",
            f"- effective_end_date: {resolution.get('effective_end_date', summary.get('end_date'))}",
            f"- source: start={resolution.get('start_date_source', 'default')} end={resolution.get('end_date_source', 'default')}",
        ]
    )
    if summary.get("capability_warnings"):
        lines.extend(["", "## Capability Warnings", ""])
        for warning in summary.get("capability_warnings", []):
            lines.append(f"- {warning.get('profile_id')}: {warning.get('warning')}")
    lines.extend(["", "## Backtest Coverage Audit", ""])
    base_coverage = summary.get("backtest_coverage_audit", {}) if isinstance(summary.get("backtest_coverage_audit"), dict) else {}
    if base_coverage:
        lines.extend(["### Base", ""])
        lines.extend(_backtest_coverage_audit_lines(base_coverage))
        lines.append("")
    lines.extend(
        [
            "| profile_id | first_price_date | last_price_date | first_candidate_date | last_candidate_date | first_trade_date | last_trade_date | candidate_days | trade_days | coverage_ratio | coverage_warning |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary.get("experiments", []):
        lines.append(
            f"| {row.get('profile_id')} | {row.get('first_price_date', '')} | {row.get('last_price_date', '')} | "
            f"{row.get('first_candidate_date', '')} | {row.get('last_candidate_date', '')} | "
            f"{row.get('first_trade_date', '')} | {row.get('last_trade_date', '')} | "
            f"{row.get('candidate_days', 0)} | {row.get('trade_days', 0)} | "
            f"{_format_optional_percent(row.get('coverage_ratio'))} | {row.get('coverage_warning') or '-'} |"
        )
    performance = summary.get("performance_report", {}) if isinstance(summary.get("performance_report"), dict) else {}
    lines.extend(
        [
            "",
            "## Run Experiments Performance Report",
            "",
            f"- shared_price_fetch_time: {performance.get('shared_price_fetch_time', performance.get('price_fetch_time', 0))}",
            f"- shared_indicator_time: {performance.get('shared_indicator_time', performance.get('indicator_time', 0))}",
            f"- shared_candidate_time: {performance.get('shared_candidate_time', performance.get('candidate_time', 0))}",
            f"- price_fetch_time: {performance.get('price_fetch_time', 0)}",
            f"- indicator_time: {performance.get('indicator_time', 0)}",
            f"- candidate_time: {performance.get('candidate_time', 0)}",
            f"- scoring_time: {performance.get('scoring_time', 0)}",
            f"- trade_time: {performance.get('trade_time', 0)}",
            f"- profile_scoring_time_by_profile: {_compact_json(performance.get('profile_scoring_time_by_profile', {}))}",
            f"- profile_trade_time_by_profile: {_compact_json(performance.get('profile_trade_time_by_profile', {}))}",
            f"- reused_indicator_count: {performance.get('reused_indicator_count', 0)}",
            f"- reused_candidate_count: {performance.get('reused_candidate_count', 0)}",
            f"- reused_scoring_count: {performance.get('reused_scoring_count', 0)}",
            f"- cache_reused_from_common_count: {performance.get('cache_reused_from_common_count', 0)}",
            f"- profile_specific_cache_count: {performance.get('profile_specific_cache_count', 0)}",
            f"- generated_cache_size: {_format_bytes(int(performance.get('generated_cache_size', 0) or 0))}",
            f"- cleanup_hint: {performance.get('cleanup_hint', '-')}",
            f"- skipped_profiles: {_compact_json(performance.get('skipped_profiles', []))}",
        ]
    )
    lines.extend(
        [
            "",
            "## API Usage",
            "",
            f"- api_calls_by_endpoint: {_compact_json(summary.get('api_calls_by_endpoint', {}))}",
            f"- api_errors_by_endpoint: {_compact_json(summary.get('api_errors_by_endpoint', {}))}",
            f"- api_retry_count: {_compact_json(summary.get('api_retry_count', {}))}",
            f"- api_retry_success_count: {_compact_json(summary.get('api_retry_success_count', {}))}",
            f"- disabled_features_reason: {_compact_json(summary.get('disabled_features_reason', {}))}",
            f"- investor_types_chunks_total: {summary.get('investor_types_chunks_total', 0)}",
            f"- investor_types_chunks_success: {summary.get('investor_types_chunks_success', 0)}",
            f"- investor_types_chunks_failed: {summary.get('investor_types_chunks_failed', 0)}",
            f"- investor_types_records_loaded: {summary.get('investor_types_records_loaded', 0)}",
            f"- investor_types_fetch_requested_start: {summary.get('investor_types_fetch_requested_start') or '-'}",
            f"- investor_types_fetch_clamped_start: {summary.get('investor_types_fetch_clamped_start') or '-'}",
            f"- investor_types_fetch_start: {summary.get('investor_types_fetch_start') or '-'}",
            f"- investor_types_fetch_end: {summary.get('investor_types_fetch_end') or '-'}",
            f"- investor_types_disabled_reason: {summary.get('investor_types_disabled_reason') or '-'}",
        ]
    )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| profile_id | role | description | required_plan | enabled_features | final_assets | net_cumulative_profit | win_rate | profit_factor | expectancy | max_drawdown | total_trades | newly_selected_count | removed_count | investor_filter_rejected_count | market_filter | market_candidate_count | market_selected_count | market_filter_excluded_count | selection_diff_count | outcome_diff_count | indicators_last_date | candidates_last_date | scored_candidates_last_date | feature_data_enabled | feature_scoring_enabled | feature_trigger_count | earnings_calendar_records | earnings_filter_rejected_count | earnings_filter_status | practical_effect | effect_reason | verdict | verdict_reason |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary.get("experiments", []):
        lines.append(_experiment_summary_table_row(row))
    return "\n".join(lines)


def _experiment_summary_table_row(row: dict[str, Any]) -> str:
    enabled = ", ".join(row.get("enabled_features") or []) or "-"
    return (
        f"| {row.get('profile_id')} | {row.get('role', '')} | {row.get('description', '')} | {row.get('required_plan', '')} | {enabled} | "
        f"{_format_optional_number(row.get('final_assets'))} | {_format_optional_number(row.get('net_cumulative_profit'))} | "
        f"{_format_optional_percent(row.get('win_rate'))} | {_format_optional_number(row.get('profit_factor'))} | "
        f"{_format_optional_percent(row.get('expectancy'))} | {_format_optional_percent(row.get('max_drawdown'))} | "
        f"{row.get('total_trades')} | {row.get('newly_selected_count')} | {row.get('removed_count')} | {row.get('investor_filter_rejected_count', 0)} | "
        f"{_compact_json(row.get('market_filter', {}))} | {_compact_json(row.get('market_candidate_count', {}))} | "
        f"{_compact_json(row.get('market_selected_count', {}))} | {row.get('market_filter_excluded_count', 0)} | "
        f"{row.get('selection_diff_count')} | {row.get('outcome_diff_count')} | "
        f"{row.get('indicators_last_date', '')} | {row.get('candidates_last_date', '')} | {row.get('scored_candidates_last_date', '')} | "
        f"{_compact_json(row.get('feature_data_enabled', {}))} | {_compact_json(row.get('feature_scoring_enabled', {}))} | "
        f"{_compact_json(row.get('feature_trigger_count', {}))} | "
        f"{row.get('earnings_calendar_records', 0)} | {row.get('earnings_filter_rejected_count', 0)} | {row.get('earnings_filter_status', '')} | "
        f"{row.get('practical_effect')} | {row.get('effect_reason', '-')} | "
        f"{row.get('verdict', row.get('judgement', '-'))} | {row.get('verdict_reason', '-')} |"
    )


def _compact_json(value: Any) -> str:
    if not value:
        return "{}"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_experiment_profile_markdown(row: dict[str, Any]) -> str:
    lines = [
            f"# {row.get('profile_id')}",
            "",
            f"- role: {row.get('role', '')}",
            f"- description: {row.get('description', '')}",
            f"- required_plan: {row.get('required_plan', '')}",
            f"- enabled_features: {', '.join(row.get('enabled_features') or []) or '-'}",
            f"- status: {row.get('status', 'completed')}",
            f"- skip_reason: {row.get('skip_reason', '-')}",
            f"- final_assets: {_format_optional_number(row.get('final_assets'))}",
            f"- net_cumulative_profit: {_format_optional_number(row.get('net_cumulative_profit'))}",
            f"- win_rate: {_format_optional_percent(row.get('win_rate'))}",
            f"- profit_factor: {_format_optional_number(row.get('profit_factor'))}",
            f"- expectancy: {_format_optional_percent(row.get('expectancy'))}",
            f"- max_drawdown: {_format_optional_percent(row.get('max_drawdown'))}",
            f"- total_trades: {row.get('total_trades')}",
            f"- newly_selected_count: {row.get('newly_selected_count', 0)}",
            f"- removed_count: {row.get('removed_count', 0)}",
            f"- investor_filter_rejected_count: {row.get('investor_filter_rejected_count', 0)}",
            f"- market_filter: {_compact_json(row.get('market_filter', {}))}",
            f"- market_candidate_count: {_compact_json(row.get('market_candidate_count', {}))}",
            f"- market_selected_count: {_compact_json(row.get('market_selected_count', {}))}",
            f"- market_filter_excluded_count: {row.get('market_filter_excluded_count', 0)}",
            f"- selection_diff_count: {row.get('selection_diff_count', 0)}",
            f"- outcome_diff_count: {row.get('outcome_diff_count', 0)}",
            f"- indicators_last_date: {row.get('indicators_last_date', '-')}",
            f"- candidates_last_date: {row.get('candidates_last_date', '-')}",
            f"- scored_candidates_last_date: {row.get('scored_candidates_last_date', '-')}",
            f"- feature_data_enabled: {_compact_json(row.get('feature_data_enabled', {}))}",
            f"- feature_scoring_enabled: {_compact_json(row.get('feature_scoring_enabled', {}))}",
            f"- feature_trigger_count: {_compact_json(row.get('feature_trigger_count', {}))}",
            f"- feature_status: {_compact_json(row.get('feature_status', {}))}",
            f"- inactive_in_practice: {', '.join(row.get('inactive_in_practice') or []) or '-'}",
            f"- practical_effect: {row.get('practical_effect', 'no')}",
            f"- effect_reason: {row.get('effect_reason', '-')}",
            f"- verdict: {row.get('verdict', row.get('judgement', '-'))}",
            f"- verdict_reason: {row.get('verdict_reason', ', '.join(row.get('judgement_reasons') or []) or '-')}",
            f"- candidate: {row.get('candidate', 'no')}",
        ]
    coverage = row.get("backtest_coverage_audit", {}) if isinstance(row.get("backtest_coverage_audit"), dict) else {}
    if coverage:
        lines.extend(["", "## Backtest Coverage Audit", ""])
        lines.extend(_backtest_coverage_audit_lines(coverage))
    return "\n".join(lines)


def _registry_baseline_profile_id(registry: dict[str, Any]) -> str:
    for profile_id, item in registry_profiles(registry).items():
        if item.get("role") == "baseline":
            return profile_id
    raise SystemExit("No baseline profile found in profile_registry.yaml")


def build_experiment_comparison_summary(
    base_profile_id: str,
    registry: dict[str, Any],
    start_date_text: str | None = None,
    end_date_text: str | None = None,
) -> dict[str, Any]:
    profiles = registry_profiles(registry)
    if base_profile_id not in profiles:
        raise SystemExit(f"Base profile not found in registry: {base_profile_id}")
    experiment_ids = registry_experiment_profile_ids(base_profile_id, registry)
    rows = []
    metrics = []
    db_warning = ""
    if start_date_text and end_date_text:
        try:
            base_config = load_profile(base_profile_id)
            db_path = get_database_path(base_config, ROOT)
            if db_path.exists():
                metrics = [_profile_compare_row(load_profile(profile_id), db_path, start_date_text, end_date_text) for profile_id in [base_profile_id, *experiment_ids]]
            else:
                db_warning = f"SQLite DB not found: {db_path}"
        except Exception as exc:
            db_warning = str(exc)
    metrics_by_profile = {row.get("profile_id"): row for row in metrics}
    base_metrics = metrics_by_profile.get(base_profile_id, {})
    for profile_id in experiment_ids:
        item = profiles[profile_id]
        target_metrics = metrics_by_profile.get(profile_id, {})
        judgement = _experiment_judgement(base_metrics, target_metrics) if target_metrics else {"judgement": "needs_review", "reasons": ["missing_metrics"]}
        rows.append(
            {
                "profile_id": profile_id,
                "description": item.get("description", ""),
                "required_plan": item.get("required_plan", ""),
                "compare_to": item.get("compare_to") or base_profile_id,
                "judgement": judgement["judgement"],
                "judgement_reasons": judgement["reasons"],
                "recommended_command": (
                    f"python src/main.py --mode compare-profiles --profiles {base_profile_id} {profile_id} "
                    "--start-date YYYY-MM-DD --end-date YYYY-MM-DD"
                ),
            }
        )
    return {
        "base_profile": base_profile_id,
        "experiment_profiles": experiment_ids,
        "experiments": rows,
        "start_date": start_date_text,
        "end_date": end_date_text,
        "metrics": metrics,
        "db_warning": db_warning,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def render_experiment_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Experiment Summary",
        "",
        f"- base_profile: {summary.get('base_profile')}",
        f"- generated_at: {summary.get('generated_at')}",
        "",
        "## Experiments",
        "",
        "| profile | required_plan | compare_to | judgement | description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in summary.get("experiments", []):
        lines.append(f"| {item['profile_id']} | {item['required_plan']} | {item['compare_to']} | {item.get('judgement', '-')} | {item['description']} |")
    lines.extend(["", "## Recommended Commands", ""])
    for item in summary.get("experiments", []):
        lines.append(f"- `{item['recommended_command']}`")
    if summary.get("metrics"):
        lines.extend(["", "## Result Summary", ""])
        for item in build_profile_ranking(summary["metrics"]):
            lines.append(f"- {item['rank']}位 {item['profile_id']}: score={_format_optional_number(item.get('score'))}")
    if summary.get("db_warning"):
        lines.extend(["", f"[WARN] {summary['db_warning']}"])
    return "\n".join(lines)


def _profile_compare_row(config: dict[str, Any], db_path: Path, start_date_text: str, end_date_text: str) -> dict[str, Any]:
    profile_id = profile_id_from(config)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        portfolio_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM portfolio_snapshots
                WHERE profile_id = ? AND date BETWEEN ? AND ?
                ORDER BY date, id
                """,
                (profile_id, start_date_text, end_date_text),
            )
        ]
        trade_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM trades
                WHERE profile_id = ?
                  AND COALESCE(exit_date, entry_date) BETWEEN ? AND ?
                ORDER BY entry_date, exit_date, id
                """,
                (profile_id, start_date_text, end_date_text),
            )
        ]
        scoring_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM scoring_results
                WHERE profile_id = ? AND date BETWEEN ? AND ?
                ORDER BY date, rank, id
                """,
                (profile_id, start_date_text, end_date_text),
            )
        ]

    latest = portfolio_rows[-1] if portfolio_rows else {}
    pf_metrics = profit_factor_metrics(trade_rows)
    closed = pf_metrics["closed_trades"]
    wins = pf_metrics["wins"]
    win_rates = [float(row["profit_rate"]) for row in wins if row.get("profit_rate") is not None]
    loss_rows = pf_metrics["losses"]
    loss_rates = [float(row["profit_rate"]) for row in loss_rows if row.get("profit_rate") is not None]
    holding_days = [float(row["holding_days"]) for row in closed if row.get("holding_days") is not None]
    average_win_profit_rate = _average_number(win_rates)
    average_loss_profit_rate = _average_number(loss_rates)
    win_rate = pf_metrics["win_rate"]
    expectancy = (
        round((win_rate * (average_win_profit_rate or 0.0)) + ((1 - win_rate) * (average_loss_profit_rate or 0.0)), 4)
        if win_rate is not None
        else None
    )
    max_drawdown = min((float(row.get("max_drawdown") or 0) for row in portfolio_rows), default=None) if portfolio_rows else None
    return {
        "profile_id": profile_id,
        "profile_name": profile_name_from(config),
        "market_filter": {
            "allowed_sections": sorted(allowed_market_sections(config)),
            "allow_unknown_market": bool(config.get("market_filter", {}).get("allow_unknown_market", False)),
        },
        "market_coverage": _backtest_summary_market_coverage(profile_id, start_date_text, end_date_text),
        "stop_loss_execution": config.get("execution", {}).get("stop_loss_execution", "next_day_open"),
        "final_assets": latest.get("total_assets"),
        "net_cumulative_profit": latest.get("net_cumulative_profit"),
        "win_rate": win_rate,
        "profit_factor": pf_metrics["profit_factor"],
        "average_win_profit_rate": average_win_profit_rate,
        "average_loss_profit_rate": average_loss_profit_rate,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "average_holding_days": _average_number(holding_days),
        "total_trades": pf_metrics["total_trades"],
        "closed_trade_count": pf_metrics["closed_trade_count"],
        "win_count": pf_metrics["win_count"],
        "loss_count": pf_metrics["loss_count"],
        "excluded_order_event_count": pf_metrics["excluded_order_event_count"],
        "loss_over_stop_count": len(
            [
                row
                for row in loss_rows
                if row.get("profit_rate") is not None and float(row["profit_rate"]) < float(config.get("risk", {}).get("stop_loss_pct", -0.03))
            ]
        ),
        "conditional_selected_count": _conditional_selected_count(scoring_rows),
        "conditional_rejected_count": _conditional_rejected_count(scoring_rows),
        "score_detail": score_detail_groups(closed),
    }


def _backtest_summary_market_coverage(profile_id: str, start_date_text: str, end_date_text: str) -> dict[str, Any]:
    path = ROOT / "logs" / "backtests" / profile_id / f"{start_date_text}_to_{end_date_text}" / "backtest_summary.json"
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    coverage = payload.get("market_coverage", {})
    return coverage if isinstance(coverage, dict) else {}


def build_profile_diff_analysis(
    configs: list[dict[str, Any]],
    db_path: Path,
    start_date_text: str,
    end_date_text: str,
) -> dict[str, Any] | None:
    if len(configs) < 2:
        return None
    pair = _profile_diff_pair(configs)
    if pair is None:
        return None
    return _build_profile_diff_analysis_for_pair(pair[0], pair[1], db_path, start_date_text, end_date_text)


def build_profile_diff_analyses(
    configs: list[dict[str, Any]],
    db_path: Path,
    start_date_text: str,
    end_date_text: str,
) -> list[dict[str, Any]]:
    if len(configs) < 2:
        return []
    base_config = configs[0]
    analyses = []
    for target_config in configs[1:]:
        analyses.append(_build_profile_diff_analysis_for_pair(base_config, target_config, db_path, start_date_text, end_date_text))
    return analyses


def _build_profile_diff_analysis_for_pair(
    base_config: dict[str, Any],
    target_config: dict[str, Any],
    db_path: Path,
    start_date_text: str,
    end_date_text: str,
) -> dict[str, Any]:
    base_profile_id = profile_id_from(base_config)
    target_profile_id = profile_id_from(target_config)
    base_rows = _load_scoring_rows_for_profile(db_path, base_profile_id, start_date_text, end_date_text)
    target_rows = _load_scoring_rows_for_profile(db_path, target_profile_id, start_date_text, end_date_text)
    base_trade_rows = _load_trade_rows_for_profile(db_path, base_profile_id, start_date_text, end_date_text)
    target_trade_rows = _load_trade_rows_for_profile(db_path, target_profile_id, start_date_text, end_date_text)
    base_selected = _selected_key_map(base_rows)
    target_selected = _selected_key_map(target_rows)
    newly_selected_keys = sorted(set(target_selected) - set(base_selected))
    removed_keys = sorted(set(base_selected) - set(target_selected))
    outcome_diff = _trade_outcome_diff(base_trade_rows, target_trade_rows)
    summary_diff = _summary_diff(
        _profile_compare_row(base_config, db_path, start_date_text, end_date_text),
        _profile_compare_row(target_config, db_path, start_date_text, end_date_text),
    )
    target_market_coverage = _backtest_summary_market_coverage(target_profile_id, start_date_text, end_date_text)
    selection_diff_count = len(newly_selected_keys) + len(removed_keys)
    outcome_diff_count = outcome_diff["outcome_diff_count"]
    practical_effect = _profile_practical_effect(selection_diff_count, outcome_diff_count)
    return {
        "base_profile_id": base_profile_id,
        "base_profile_name": profile_name_from(base_config),
        "target_profile_id": target_profile_id,
        "target_profile_name": profile_name_from(target_config),
        "base_selected_count": len(base_selected),
        "target_selected_count": len(target_selected),
        "base_risk_off_candidate_count": _risk_off_candidate_count(base_rows),
        "target_risk_off_candidate_count": _risk_off_candidate_count(target_rows),
        "base_risk_off_rejected_count": _risk_off_rejected_count(base_rows),
        "target_risk_off_rejected_count": _risk_off_rejected_count(target_rows),
        "base_conditional_selected_count": _conditional_selected_count(base_rows),
        "target_conditional_selected_count": _conditional_selected_count(target_rows),
        "base_conditional_rejected_count": _conditional_rejected_count(base_rows),
        "target_conditional_rejected_count": _conditional_rejected_count(target_rows),
        "investor_filter_rejected_count": _investor_filter_rejected_count(target_rows),
        "market_filter": {
            "base_allowed_sections": sorted(allowed_market_sections(base_config)),
            "target_allowed_sections": sorted(allowed_market_sections(target_config)),
        },
        "market_candidate_count": target_market_coverage.get("candidate_count", {}),
        "market_selected_count": target_market_coverage.get("selected_count", {}),
        "market_filter_excluded_count": target_market_coverage.get("market_filter_excluded_count", 0),
        "newly_selected_count": len(newly_selected_keys),
        "removed_count": len(removed_keys),
        "selection_diff_count": selection_diff_count,
        "newly_selected": [_selection_diff_record(target_selected[key]) for key in newly_selected_keys],
        "removed": [_selection_diff_record(base_selected[key]) for key in removed_keys],
        "trade_outcome_diff": outcome_diff,
        "outcome_diff_count": outcome_diff_count,
        "summary_diff": summary_diff,
        "practical_effect": practical_effect,
        "effect_reason": _profile_effect_reason(selection_diff_count, outcome_diff_count),
        "effective_config_differences": _effective_config_differences(base_config, target_config),
        "no_practical_effect": selection_diff_count == 0 and outcome_diff_count == 0,
    }


def _profile_diff_pair(configs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    by_id = {profile_id_from(config): config for config in configs}
    if "rookie_dealer_02_v2_1" in by_id and "rookie_dealer_02_v2_2" in by_id:
        return by_id["rookie_dealer_02_v2_1"], by_id["rookie_dealer_02_v2_2"]
    if len(configs) >= 2:
        return configs[0], configs[1]
    return None


def _investor_filter_rejected_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if str(row.get("rejected_reason") or "") == "investor_context_negative")


def _load_scoring_rows_for_profile(db_path: Path, profile_id: str, start_date_text: str, end_date_text: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM scoring_results
                WHERE profile_id = ? AND date BETWEEN ? AND ?
                ORDER BY date, rank, id
                """,
                (profile_id, start_date_text, end_date_text),
            )
        ]


def _load_trade_rows_for_profile(db_path: Path, profile_id: str, start_date_text: str, end_date_text: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM trades
                WHERE profile_id = ?
                  AND COALESCE(exit_date, entry_date) BETWEEN ? AND ?
                ORDER BY entry_date, exit_date, id
                """,
                (profile_id, start_date_text, end_date_text),
            )
        ]


def _selected_key_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("date")), str(row.get("code"))): row
        for row in rows
        if bool(row.get("selected"))
    }


def _closed_trade_key_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        if not row.get("entry_date") or not row.get("code"):
            continue
        if not (row.get("exit_date") or str(row.get("action") or "").upper() == "SELL"):
            continue
        key = (str(row.get("entry_date")), str(row.get("code")))
        if key not in result:
            result[key] = row
            continue
        suffix = 2
        while (key[0], f"{key[1]}#{suffix}") in result:
            suffix += 1
        result[(key[0], f"{key[1]}#{suffix}")] = row
    return result


def _trade_outcome_diff(base_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_trades = _closed_trade_key_map(base_rows)
    target_trades = _closed_trade_key_map(target_rows)
    common_keys = sorted(set(base_trades) & set(target_trades))
    different_exit_date = []
    different_exit_reason = []
    different_profit = []
    different_holding_days = []
    outcome_diffs = []
    for key in common_keys:
        base = base_trades[key]
        target = target_trades[key]
        changes = []
        if _normalized_value(base.get("exit_date")) != _normalized_value(target.get("exit_date")):
            changes.append("exit_date")
            different_exit_date.append(_trade_outcome_diff_record(key, base, target, ["exit_date"]))
        if _normalized_value(base.get("exit_reason")) != _normalized_value(target.get("exit_reason")):
            changes.append("exit_reason")
            different_exit_reason.append(_trade_outcome_diff_record(key, base, target, ["exit_reason"]))
        if not _numbers_close(_trade_profit_value(base), _trade_profit_value(target)):
            changes.append("profit")
            different_profit.append(_trade_outcome_diff_record(key, base, target, ["profit"]))
        if _normalized_value(base.get("holding_days")) != _normalized_value(target.get("holding_days")):
            changes.append("holding_days")
            different_holding_days.append(_trade_outcome_diff_record(key, base, target, ["holding_days"]))
        if changes:
            outcome_diffs.append(_trade_outcome_diff_record(key, base, target, changes))
    return {
        "same_entry_count": len(common_keys),
        "outcome_diff_count": len(outcome_diffs),
        "same_entry_different_exit_date_count": len(different_exit_date),
        "same_entry_different_exit_reason_count": len(different_exit_reason),
        "same_entry_different_profit_count": len(different_profit),
        "same_entry_different_holding_days_count": len(different_holding_days),
        "same_entry_different_exit_date": different_exit_date[:50],
        "same_entry_different_exit_reason": different_exit_reason[:50],
        "same_entry_different_profit": different_profit[:50],
        "same_entry_different_holding_days": different_holding_days[:50],
        "outcome_diffs": outcome_diffs[:50],
    }


def _trade_outcome_diff_record(
    key: tuple[str, str],
    base: dict[str, Any],
    target: dict[str, Any],
    changed_fields: list[str],
) -> dict[str, Any]:
    code = str(base.get("code") or target.get("code") or key[1]).split("#", 1)[0]
    return {
        "entry_date": key[0],
        "code": code,
        "name": base.get("name") or target.get("name"),
        "changed_fields": changed_fields,
        "base_exit_date": base.get("exit_date"),
        "target_exit_date": target.get("exit_date"),
        "base_exit_reason": base.get("exit_reason"),
        "target_exit_reason": target.get("exit_reason"),
        "base_profit": _trade_profit_value(base),
        "target_profit": _trade_profit_value(target),
        "base_profit_rate": _trade_profit_rate_value(base),
        "target_profit_rate": _trade_profit_rate_value(target),
        "base_holding_days": base.get("holding_days"),
        "target_holding_days": target.get("holding_days"),
    }


def _trade_profit_value(row: dict[str, Any]) -> float | None:
    return _to_float(row.get("net_profit") if row.get("net_profit") is not None else row.get("profit"))


def _trade_profit_rate_value(row: dict[str, Any]) -> float | None:
    return _to_float(row.get("net_profit_rate") if row.get("net_profit_rate") is not None else row.get("profit_rate"))


def _numbers_close(left: float | None, right: float | None, tolerance: float = 0.0001) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= tolerance


def _normalized_value(value: Any) -> str:
    return "" if value is None else str(value)


def _summary_diff(base_row: dict[str, Any], target_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "net_profit_diff": _number_diff(target_row.get("net_cumulative_profit"), base_row.get("net_cumulative_profit")),
        "profit_factor_diff": _number_diff(target_row.get("profit_factor"), base_row.get("profit_factor")),
        "win_rate_diff": _number_diff(target_row.get("win_rate"), base_row.get("win_rate")),
        "trade_count_diff": _number_diff(target_row.get("total_trades"), base_row.get("total_trades")),
    }


def _number_diff(target: Any, base: Any) -> float | None:
    target_number = _to_float(target)
    base_number = _to_float(base)
    if target_number is None or base_number is None:
        return None
    return round(target_number - base_number, 6)


def _profile_practical_effect(selection_diff_count: int, outcome_diff_count: int) -> str:
    if selection_diff_count == 0 and outcome_diff_count == 0:
        return "no_practical_effect"
    if selection_diff_count == 0 and outcome_diff_count > 0:
        return "execution_or_exit_effect"
    return "selection_effect"


def _profile_effect_reason(selection_diff_count: int, outcome_diff_count: int) -> str:
    if selection_diff_count == 0 and outcome_diff_count == 0:
        return "selection_diff_count=0 and outcome_diff_count=0"
    if selection_diff_count == 0:
        return "same entries but trade outcomes differ"
    if outcome_diff_count == 0:
        return "entry selection differs"
    return "entry selection and trade outcomes differ"


def _risk_off_candidate_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("market_regime") == "risk_off")


def _risk_off_rejected_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("market_regime") == "risk_off"
        and not bool(row.get("selected"))
        and row.get("rejected_reason") == "risk_offのため買付抑制"
    )


def _conditional_selected_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if bool(row.get("selected")) and str(row.get("reason") or "").startswith("conditional selected")
    )


def _conditional_rejected_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if not bool(row.get("selected")) and str(row.get("rejected_reason") or "").startswith("conditional rejected")
    )


def _selection_diff_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": row.get("date"),
        "code": row.get("code"),
        "name": row.get("name"),
        "rank": row.get("rank"),
        "total_score": row.get("total_score"),
        "market_regime": row.get("market_regime"),
        "rejected_reason": row.get("rejected_reason"),
        "reason": row.get("reason"),
    }


def _effective_config_differences(base_config: dict[str, Any], target_config: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        ("market_filter", "risk_off_buy_policy"),
        ("market_filter", "risk_off_max_buy_orders"),
        ("market_filter", "risk_off_min_score"),
        ("market_filter", "risk_off_disable_top_pick"),
        ("selection", "min_score"),
        ("selection", "fallback_min_score"),
        ("selection", "top_pick_min_score"),
        ("selection", "conditional_selection"),
        ("selection", "max_rsi_for_new_position"),
        ("volume_filter", "min_volume_ratio"),
        ("execution", "stop_loss_execution"),
    ]
    differences = []
    for section, key in keys:
        base_value = (base_config.get(section) or {}).get(key)
        target_value = (target_config.get(section) or {}).get(key)
        if base_value != target_value:
            differences.append(
                {
                    "key": f"{section}.{key}",
                    "base": base_value,
                    "target": target_value,
                }
            )
    return differences


def build_profile_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weights = {
        "net_cumulative_profit": 0.35,
        "profit_factor": 0.30,
        "max_drawdown": 0.20,
        "expectancy": 0.15,
    }
    scored = []
    for row in rows:
        components = {
            "net_cumulative_profit": _normalized_score(row.get("net_cumulative_profit"), rows, "net_cumulative_profit"),
            "profit_factor": _normalized_score(row.get("profit_factor"), rows, "profit_factor"),
            "max_drawdown": _normalized_score(row.get("max_drawdown"), rows, "max_drawdown"),
            "expectancy": _normalized_score(row.get("expectancy"), rows, "expectancy"),
        }
        score = round(sum(components[key] * weight for key, weight in weights.items()), 4)
        scored.append(
            {
                "profile_id": row["profile_id"],
                "profile_name": row.get("profile_name"),
                "score": score,
                "components": components,
                "weights": weights,
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["profile_id"]))
    for index, item in enumerate(scored, start=1):
        item["rank"] = index
    return scored


def _normalized_score(value: Any, rows: list[dict[str, Any]], key: str) -> float:
    numeric_values = [_to_float(row.get(key)) for row in rows]
    numeric_values = [item for item in numeric_values if item is not None]
    numeric_value = _to_float(value)
    if numeric_value is None or not numeric_values:
        return 0.0
    minimum = min(numeric_values)
    maximum = max(numeric_values)
    if maximum == minimum:
        return 1.0
    return round((numeric_value - minimum) / (maximum - minimum), 4)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_compare_profiles_markdown(payload: dict[str, Any]) -> str:
    resolution = payload.get("date_resolution", {}) if isinstance(payload.get("date_resolution"), dict) else {}
    lines = [
        f"# Profile比較 {payload['start_date']} to {payload['end_date']}",
        "",
        "## Backtest Date Resolution",
        "",
        f"- requested_start_date: {resolution.get('requested_start_date', payload['start_date'])}",
        f"- requested_end_date: {resolution.get('requested_end_date', payload['end_date'])}",
        f"- effective_start_date: {resolution.get('effective_start_date', payload['start_date'])}",
        f"- effective_end_date: {resolution.get('effective_end_date', payload['end_date'])}",
        f"- source: start={resolution.get('start_date_source', 'default')} end={resolution.get('end_date_source', 'default')}",
        "",
        "| profile | stop_loss_execution | final_assets | net_cumulative_profit | win_rate | profit_factor | expectancy | max_drawdown | avg_holding_days | closed | wins | losses | excluded | avg_win | avg_loss | total_trades | loss_over_stop_count | conditional_selected | conditional_rejected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["profiles"]:
        lines.append(
            "| "
            f"{row['profile_id']} {row['profile_name']} | "
            f"{row.get('stop_loss_execution')} | "
            f"{_format_optional_yen(row.get('final_assets'))} | "
            f"{_format_optional_yen(row.get('net_cumulative_profit'))} | "
            f"{_format_optional_percent(row.get('win_rate'))} | "
            f"{_format_optional_number(row.get('profit_factor'))} | "
            f"{_format_optional_percent(row.get('expectancy'))} | "
            f"{_format_optional_percent(row.get('max_drawdown'))} | "
            f"{_format_optional_number(row.get('average_holding_days'))} | "
            f"{row.get('closed_trade_count')} | "
            f"{row.get('win_count')} | "
            f"{row.get('loss_count')} | "
            f"{row.get('excluded_order_event_count')} | "
            f"{_format_optional_percent(row.get('average_win_profit_rate'))} | "
            f"{_format_optional_percent(row.get('average_loss_profit_rate'))} | "
            f"{row.get('total_trades')} | "
            f"{row.get('loss_over_stop_count')} | "
            f"{row.get('conditional_selected_count')} | "
            f"{row.get('conditional_rejected_count')} |"
        )
    if payload.get("ranking"):
        lines.extend(["", "## Profile Ranking", ""])
        for item in payload["ranking"]:
            lines.extend(
                [
                    f"{item['rank']}位 {item['profile_id']} {item.get('profile_name') or ''}".strip(),
                    f"score: {_format_optional_number(item.get('score'))}",
                    "",
                ]
            )
        lines.extend(
            [
                "ranking score = net_cumulative_profit 35% + profit_factor 30% + max_drawdown 20% + expectancy 15%",
                "",
            ]
        )
    analyses = payload.get("profile_diff_analyses") or ([payload.get("profile_diff_analysis")] if payload.get("profile_diff_analysis") else [])
    if analyses:
        lines.extend(["", "## Profile Diff Analysis", ""])
        for analysis in analyses:
            lines.extend(_profile_diff_analysis_lines(analysis))
    lines.extend(["", "## Score Detail", ""])
    for row in payload["profiles"]:
        lines.extend(
            [
                f"### {row['profile_id']} {row['profile_name']}",
                "",
                "| score | count | win_rate | average_profit_rate | total_profit |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in row.get("score_detail", []):
            lines.append(
                "| "
                f"{item.get('bucket')} | "
                f"{item.get('count')} | "
                f"{_format_optional_percent(item.get('win_rate'))} | "
                f"{_format_optional_percent(item.get('average_profit_rate'))} | "
                f"{_format_optional_yen(item.get('total_profit'))} |"
            )
        lines.append("")
    lines.append("")
    return "\n".join(lines)


def _profile_diff_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    outcome = analysis.get("trade_outcome_diff") or {}
    summary = analysis.get("summary_diff") or {}
    lines = [
        f"### {analysis.get('base_profile_id')} vs {analysis.get('target_profile_id')}",
        "",
        f"- base_profile: {analysis.get('base_profile_id')} {analysis.get('base_profile_name') or ''}",
        f"- target_profile: {analysis.get('target_profile_id')} {analysis.get('target_profile_name') or ''}",
        f"- base selected count: {analysis.get('base_selected_count')}",
        f"- target selected count: {analysis.get('target_selected_count')}",
        f"- base risk_off candidate count: {analysis.get('base_risk_off_candidate_count')}",
        f"- target risk_off candidate count: {analysis.get('target_risk_off_candidate_count')}",
        f"- base risk_off rejected count: {analysis.get('base_risk_off_rejected_count')}",
        f"- target risk_off rejected count: {analysis.get('target_risk_off_rejected_count')}",
        f"- base conditional selected count: {analysis.get('base_conditional_selected_count')}",
        f"- target conditional selected count: {analysis.get('target_conditional_selected_count')}",
        f"- base conditional rejected count: {analysis.get('base_conditional_rejected_count')}",
        f"- target conditional rejected count: {analysis.get('target_conditional_rejected_count')}",
        f"- investor_filter_rejected_count: {analysis.get('investor_filter_rejected_count', 0)}",
        f"- market_filter: {_compact_json(analysis.get('market_filter', {}))}",
        f"- market_candidate_count: {_compact_json(analysis.get('market_candidate_count', {}))}",
        f"- market_selected_count: {_compact_json(analysis.get('market_selected_count', {}))}",
        f"- market_filter_excluded_count: {analysis.get('market_filter_excluded_count', 0)}",
        f"- newly selected by target: {analysis.get('newly_selected_count')}",
        f"- removed by target: {analysis.get('removed_count')}",
        f"- selection_diff_count: {analysis.get('selection_diff_count')}",
        f"- outcome_diff_count: {analysis.get('outcome_diff_count')}",
        f"- practical_effect: {analysis.get('practical_effect')}",
        f"- effect_reason: {analysis.get('effect_reason')}",
    ]
    if analysis.get("no_practical_effect"):
        lines.extend(["", "No practical effect"])
    lines.extend(
        [
            "",
            "### Summary Diff",
            "",
            f"- net profit diff: {_format_optional_yen(summary.get('net_profit_diff'))}",
            f"- PF diff: {_format_optional_number(summary.get('profit_factor_diff'))}",
            f"- win_rate diff: {_format_optional_percent(summary.get('win_rate_diff'))}",
            f"- trade_count diff: {_format_optional_number(summary.get('trade_count_diff'))}",
            "",
            "### Entry Selection Diff",
            "",
            f"- newly_selected: {analysis.get('newly_selected_count')}",
            f"- removed: {analysis.get('removed_count')}",
            "",
            "### Trade Outcome Diff",
            "",
            f"- same entry but different exit_date: {outcome.get('same_entry_different_exit_date_count', 0)}",
            f"- same entry but different exit_reason: {outcome.get('same_entry_different_exit_reason_count', 0)}",
            f"- same entry but different profit: {outcome.get('same_entry_different_profit_count', 0)}",
            f"- same entry but different holding_days: {outcome.get('same_entry_different_holding_days_count', 0)}",
        ]
    )
    lines.extend(_trade_outcome_diff_lines(outcome.get("outcome_diffs", [])))
    lines.extend(["", "### Effective Config Differences", ""])
    differences = analysis.get("effective_config_differences") or []
    if differences:
        lines.extend(
            [
                f"- {item.get('key')}: {item.get('base')} -> {item.get('target')}"
                for item in differences
            ]
        )
    else:
        lines.append("- 差分なし")
    lines.extend(["", "### Newly Selected by Target", ""])
    lines.extend(_selection_diff_lines(analysis.get("newly_selected", [])))
    lines.extend(["", "### Removed by Target", ""])
    lines.extend(_selection_diff_lines(analysis.get("removed", [])))
    return lines


def _trade_outcome_diff_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- なし"]
    return [
        (
            f"- {item.get('entry_date')} {item.get('code')} {item.get('name')}: "
            f"fields {', '.join(item.get('changed_fields') or [])}, "
            f"exit {item.get('base_exit_date')} -> {item.get('target_exit_date')}, "
            f"reason {item.get('base_exit_reason') or 'N/A'} -> {item.get('target_exit_reason') or 'N/A'}, "
            f"profit {_format_optional_yen(item.get('base_profit'))} -> {_format_optional_yen(item.get('target_profit'))}, "
            f"holding {item.get('base_holding_days')} -> {item.get('target_holding_days')}"
        )
        for item in items[:50]
    ]


def _selection_diff_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- なし"]
    return [
        (
            f"- {item.get('date')} {item.get('code')} {item.get('name')}: "
            f"rank {item.get('rank')}, score {_format_optional_number(item.get('total_score'))}, "
            f"market {item.get('market_regime') or 'unknown'}"
        )
        for item in items[:50]
    ]


def _average_number(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def run_export_ai_dataset(config: dict[str, Any], start_date: str, end_date: str) -> None:
    initialize_database(config, ROOT)
    try:
        result = export_ai_dataset(config, ROOT, start_date, end_date)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"AI dataset export failed: {exc}")
        raise SystemExit(1) from exc
    record_ai_analysis_export(config, ROOT, start_date, end_date, result["path"], None, result["record_count"])
    print("AI analysis dataset exported")
    print(f"records: {result['record_count']}")
    print(f"jsonl: {result['path'].relative_to(ROOT)}")


def run_export_ai_summary(config: dict[str, Any], start_date: str, end_date: str) -> None:
    initialize_database(config, ROOT)
    try:
        result = export_ai_summary(config, ROOT, start_date, end_date)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"AI summary export failed: {exc}")
        raise SystemExit(1) from exc
    record_ai_analysis_export(config, ROOT, start_date, end_date, result["dataset_path"], result["path"], result["record_count"])
    print("AI analysis summary exported")
    print(f"records: {result['record_count']}")
    print(f"jsonl: {result['dataset_path'].relative_to(ROOT)}")
    print(f"markdown: {result['path'].relative_to(ROOT)}")


def write_daily_ai_dataset(config: dict[str, Any], target_date_text: str) -> None:
    try:
        result = export_ai_dataset(
            config,
            ROOT,
            target_date_text,
            target_date_text,
            file_name=f"decision_dataset_{target_date_text}.jsonl",
        )
    except Exception as exc:
        print(f"[WARN] AI analysis dataset export skipped: {exc}")
        return
    print(f"AI analysis dataset: {result['path'].relative_to(ROOT)}")


def run_release_notes(since: str, until: str) -> None:
    try:
        notes = generate_release_notes(since, until, ROOT)
    except (RuntimeError, ValueError) as exc:
        print(f"Release notes generation failed: {exc}")
        raise SystemExit(1) from exc

    output_dir = ROOT / "reports" / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"release_notes_{since}_to_{until}"
    markdown_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    write_json(json_path, notes)
    write_text(markdown_path, render_release_notes_markdown(notes))

    print("release notes generated")
    print(f"since: {since}")
    print(f"until: {until}")
    print(f"total_commits: {notes['total_commits']}")
    print(f"markdown: {markdown_path.relative_to(ROOT)}")
    print(f"json: {json_path.relative_to(ROOT)}")


def run_full_paper_run(provider_name: str, start_date_text: str, end_date_text: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("full-paper-run mode currently supports --provider jquants only.")

    config = load_config(CONFIG_PATH)
    _assert_full_paper_run_is_paper_only(config)
    range_key = f"{start_date_text}_to_{end_date_text}"

    print("Full Paper Run start")
    print(f"period: {start_date_text} to {end_date_text}")
    print("broker: paper")
    print("live_trading: disabled")

    _full_paper_step(1, 7, "preflight", run_preflight)

    db_path = get_database_path(config, ROOT)
    if db_path.exists():
        print(f"database: existing {db_path.relative_to(ROOT)}")
    _full_paper_step(2, 7, "init-db", lambda: initialize_database(config, ROOT))

    _full_paper_step(3, 7, "jquants-healthcheck", lambda: run_healthcheck(provider_name))
    _full_paper_step(4, 7, "list-stocks", lambda: run_list_stocks(provider_name))
    _full_paper_step(5, 7, "backtest", lambda: run_backtest(provider_name, start_date_text, end_date_text))
    _full_paper_step(6, 7, "analyze", lambda: run_analyze(config))
    _full_paper_step(7, 7, "release-notes", lambda: run_release_notes(start_date_text, end_date_text))

    summary = build_full_paper_run_summary(config, provider_name, start_date_text, end_date_text)
    output_dir = ROOT / "reports" / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"full_paper_run_{range_key}.md"
    json_path = output_dir / f"full_paper_run_{range_key}.json"
    summary["summary_markdown_path"] = str(markdown_path)
    summary["summary_json_path"] = str(json_path)
    write_json(json_path, summary)
    write_text(markdown_path, render_full_paper_run_markdown(summary))

    print("")
    print("Full Paper Run completed")
    print("")
    print(f"- Backtest report: {_relative_path_text(summary.get('backtest_report_path'))}")
    print("- Analysis report: reports/backtests/analysis_latest.md")
    print(f"- Article drafts: {summary['article_drafts_dir']}")
    print(f"- Summary: {markdown_path.relative_to(ROOT)}")


def _assert_full_paper_run_is_paper_only(config: dict[str, Any]) -> None:
    broker = config.get("broker", {})
    safety = config.get("safety", {})
    violations = []
    if broker.get("provider") != "paper":
        violations.append("broker.provider must be paper")
    if bool(safety.get("allow_live_trading", False)):
        violations.append("safety.allow_live_trading must be false")
    if bool(broker.get("live_trading_enabled", False)):
        violations.append("broker.live_trading_enabled must be false")
    if violations:
        raise SystemExit("full-paper-run aborted: " + "; ".join(violations))


def _full_paper_step(step_number: int, total_steps: int, step_name: str, action: Any) -> Any:
    print(f"[full-paper-run {step_number}/{total_steps}] {step_name} start")
    try:
        result = action()
    except SystemExit as exc:
        print(f"[full-paper-run {step_number}/{total_steps}] {step_name} failed")
        raise
    except Exception as exc:
        print(f"[full-paper-run {step_number}/{total_steps}] {step_name} failed")
        print(f"reason: {exc}")
        raise SystemExit(1) from exc
    print(f"[full-paper-run {step_number}/{total_steps}] {step_name} done")
    return result


def build_full_paper_run_summary(
    config: dict[str, Any],
    provider_name: str,
    start_date_text: str,
    end_date_text: str,
) -> dict[str, Any]:
    range_key = f"{start_date_text}_to_{end_date_text}"
    backtest_json_path = ROOT / "reports" / f"backtest_{range_key}.json"
    backtest_log_path = ROOT / "logs" / "backtests" / range_key / "backtest_summary.json"
    analysis_json_path = ROOT / "reports" / "backtests" / "analysis_latest.json"
    release_notes_path = ROOT / "reports" / "backtests" / f"release_notes_{range_key}.md"
    article_dir = ROOT / "reports" / "articles" / "daily"

    backtest_summary = read_json(backtest_json_path) if backtest_json_path.exists() else {}
    backtest_log = read_json(backtest_log_path) if backtest_log_path.exists() else {}
    analysis = read_json(analysis_json_path) if analysis_json_path.exists() else {}

    closed_trades = backtest_log.get("state", {}).get("closed_trades", [])
    daily_asset_curve = backtest_summary.get("daily_asset_curve", [])
    scoring_files = sorted((ROOT / "logs" / "backtests" / range_key).glob("scoring_*.json"))
    selected_count_total = 0
    no_trade_days = 0
    for scoring_file in scoring_files:
        scoring = read_json(scoring_file)
        selected_count = len(scoring.get("selected", []))
        selected_count_total += selected_count
        if selected_count == 0:
            no_trade_days += 1

    generated_articles_count = _article_index_count(profile_id_from(config), start_date_text, end_date_text)
    trade_analysis = analysis.get("trade_analysis", {})
    portfolio_analysis = analysis.get("portfolio_analysis", {})
    gross_profit = sum(float(trade.get("gross_profit", trade.get("profit", 0)) or 0) for trade in closed_trades)
    net_profit = sum(float(trade.get("net_profit", trade.get("profit", 0)) or 0) for trade in closed_trades)

    execution_model = _backtest_execution_model(config)
    execution_model.update(_backtest_execution_model_stats(all_trades, execution_model))
    summary = {
        "start_date": start_date_text,
        "end_date": end_date_text,
        "provider": provider_name,
        "broker": "paper",
        "config_version": config_version_from(config),
        "initial_capital": backtest_summary.get("initial_capital", config.get("portfolio", {}).get("initial_cash")),
        "final_assets": backtest_summary.get("final_assets"),
        "gross_profit": round(gross_profit, 2),
        "net_profit": round(net_profit, 2),
        "win_rate": backtest_summary.get("win_rate"),
        "max_drawdown": backtest_summary.get("max_drawdown"),
        "total_trades": backtest_summary.get("total_trades", trade_analysis.get("total_trades")),
        "take_profit_count": backtest_summary.get("take_profit_count"),
        "stop_loss_count": backtest_summary.get("stop_loss_count"),
        "max_holding_exit_count": backtest_summary.get("max_holding_exit_count"),
        "selected_count_total": selected_count_total,
        "no_trade_days": no_trade_days,
        "generated_articles_count": generated_articles_count,
        "daily_count": len(daily_asset_curve),
        "dealer_comment": _full_paper_run_dealer_comment(backtest_summary),
        "next_checks": [
            "selected_count が少ない場合はスコアリング基準と単元株制約を確認する",
            "税引後損益とスリッページを見て短期売買コストに耐えられるか確認する",
            "最大ドローダウンが許容範囲内か確認する",
            "生成されたnote記事が公開前レビューに耐える内容か確認する",
            "立花証券API接続前にPaperBrokerで複数期間のバックテストを繰り返す",
        ],
        "backtest_report_path": str(backtest_json_path.with_suffix(".md")),
        "backtest_json_path": str(backtest_json_path),
        "analysis_report_path": str(analysis_json_path.with_suffix(".md")),
        "analysis_json_path": str(analysis_json_path),
        "release_notes_path": str(release_notes_path),
        "article_drafts_dir": str(article_dir.relative_to(ROOT)) if article_dir.exists() else str(article_dir.relative_to(ROOT)),
        "live_trading": False,
    }
    summary["net_cumulative_profit_from_analysis"] = portfolio_analysis.get("net_cumulative_profit")
    summary["gross_cumulative_profit_from_analysis"] = portfolio_analysis.get("gross_cumulative_profit")
    return summary


def _full_paper_run_dealer_comment(backtest_summary: dict[str, Any]) -> str:
    final_assets = backtest_summary.get("final_assets")
    initial_capital = backtest_summary.get("initial_capital")
    total_trades = backtest_summary.get("total_trades", 0)
    if final_assets is None or initial_capital is None:
        return "実行結果の集計が不足しています。まずはログとDB保存状況を確認します。"
    if total_trades == 0:
        return "本期間では約定取引がありません。無理な売買は避け、条件が整うまで待機する判断です。"
    if float(final_assets) >= float(initial_capital):
        return "本期間はPaperBrokerで一連の運用確認が完了しました。利益は確認できますが、感情は考慮せず再現性を優先します。"
    return "本期間は損益が初期資金を下回りました。ルールに従った結果として受け止め、損失要因を分析します。"


def render_full_paper_run_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Full Paper Run {summary['start_date']} to {summary['end_date']}",
        "",
        "## 実行条件",
        "",
        f"- 実行期間: {summary['start_date']} 〜 {summary['end_date']}",
        f"- provider: {summary['provider']}",
        f"- broker: {summary['broker']}",
        f"- config_version: {summary['config_version']}",
        "- live_trading: disabled",
        "",
        "## 最終サマリ",
        "",
        f"- 初期資金: {_format_optional_yen(summary.get('initial_capital'))}",
        f"- 最終資産: {_format_optional_yen(summary.get('final_assets'))}",
        f"- 税引前損益: {_format_optional_yen(summary.get('gross_profit'))}",
        f"- 税引後損益: {_format_optional_yen(summary.get('net_profit'))}",
        f"- 勝率: {_format_optional_rate(summary.get('win_rate'))}",
        f"- 最大ドローダウン: {_format_optional_rate(summary.get('max_drawdown'))}",
        f"- 総取引数: {summary.get('total_trades', 'N/A')}",
        f"- 利確回数: {summary.get('take_profit_count', 'N/A')}",
        f"- 損切り回数: {summary.get('stop_loss_count', 'N/A')}",
        f"- 最大保有期間売却回数: {summary.get('max_holding_exit_count', 'N/A')}",
        f"- selected_count合計: {summary.get('selected_count_total', 0)}",
        f"- no trade日数: {summary.get('no_trade_days', 0)}",
        f"- 生成された記事数: {summary.get('generated_articles_count', 0)}",
        "",
        "## 新人ディーラー1号コメント",
        "",
        summary["dealer_comment"],
        "",
        "## 生成物",
        "",
        f"- Backtest report: {_relative_path_text(summary.get('backtest_report_path'))}",
        f"- Analysis report: {_relative_path_text(summary.get('analysis_report_path'))}",
        f"- Release notes: {_relative_path_text(summary.get('release_notes_path'))}",
        f"- Article drafts: {summary['article_drafts_dir']}",
        "",
        "## 次に確認すべき課題",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["next_checks"])
    lines.extend(
        [
            "",
            "## 注意",
            "",
            "- この実行はPaperBrokerのみを使用します。",
            "- 実売買、立花証券API接続、kabuステーションAPI接続は行っていません。",
            "- APIキーやパスワードなどの秘密情報は保存していません。",
        ]
    )
    return "\n".join(lines)


def run_help() -> None:
    print(
        """AI Fund Lab / AIファンド研究所

主要モード:
- preflight: 環境、設定、DB、APIキー、セーフティ設定を確認
- init-db: SQLite DBを初期化またはマイグレーション
- healthcheck: J-Quants APIキーの疎通確認
- tachibana-healthcheck: 立花証券デモ接続前の設定確認
- account-snapshot: Broker共通IFで口座スナップショットを読み取り表示
- demo-auto-order: tachibana_demo向け自動発注。live環境では停止
- list-stocks: J-Quantsから東証プライム銘柄一覧を取得
- fetch-prices: 指定日の株価データを取得
- calculate-indicators: 指定日のテクニカル指標を計算
- screen: 実データ指標から候補銘柄を抽出
- score: 候補銘柄をスコアリング
- trade: スコアリング済み銘柄で仮想売買
- run-daily: 日次運用フローを1コマンドで実行
- backtest: 指定期間のバックテストを実行
- full-paper-run: PaperBrokerで事前チェックから分析まで通し実行
- analyze: SQLiteから分析レポートを生成
- compare-profiles: profile別のバックテスト結果を横並び比較
- export-ai-dataset: AI改善用JSONLデータセットを生成
- export-ai-summary: AI改善用Markdownサマリを生成
- release-notes: Gitコミット履歴から開発ノートを生成
- preview-orders: 発注せず注文候補をプレビュー
- publish-article: note手動投稿後の記事をpublishedへ記録
- status: 現在の設定、DB、記事、セーフティ状態を表示
- demo: ダミーデータでMVPフローを実行

推奨実行順:
1. python src/main.py --mode preflight
2. python src/main.py --mode init-db
3. python src/main.py --mode tachibana-healthcheck --env demo
4. python src/main.py --mode list-stocks --provider jquants
5. python src/main.py --mode run-daily --provider jquants --date YYYY-MM-DD
6. python src/main.py --mode preview-orders --provider jquants --date YYYY-MM-DD
7. python src/main.py --mode analyze
8. python src/main.py --mode backtest --provider jquants --start-date YYYY-MM-DD --end-date YYYY-MM-DD
9. python src/main.py --mode compare-profiles --profiles rookie_dealer_01 rookie_dealer_02 --start-date YYYY-MM-DD --end-date YYYY-MM-DD
10. python src/main.py --mode export-ai-summary --start-date YYYY-MM-DD --end-date YYYY-MM-DD
11. python src/main.py --mode full-paper-run --provider jquants --start-date YYYY-MM-DD --end-date YYYY-MM-DD
12. python src/main.py --mode release-notes --since YYYY-MM-DD --until YYYY-MM-DD

補足:
- 実売買は未実装です。
- 現在はPaperBrokerのみ使用します。
- Broker候補: paper / tachibana_demo / tachibana_live / kabu_station
- Mac/Linuxでの実売買API候補は、立花証券 e支店 API を第一候補にします。
- tachibana_demo / tachibana_live はRead Onlyモードで、実注文は出しません。
- demo-auto-order は tachibana_demo のみ対象です。tachibana_live では必ず停止します。
- APIキーや口座情報は表示しません。
"""
    )


def run_status(config: dict[str, Any], output_format: str = "text") -> None:
    payload = build_status_payload(config)
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(render_status_text(payload))


def build_status_payload(config: dict[str, Any]) -> dict[str, Any]:
    db_status = _status_database(config)
    article_status = _article_index_status(profile_id_from(config))
    safety = config.get("safety", {})
    broker = config.get("broker", {})
    stop_file = Path(safety.get("emergency_stop_file", "storage/STOP_TRADING"))
    if not stop_file.is_absolute():
        stop_file = ROOT / stop_file
    live_enabled = bool(broker.get("live_trading_enabled", False) and safety.get("allow_live_trading", False))
    tachibana = config.get("tachibana", {})
    tachibana_env_names = [
        _clean_config_string(tachibana.get("user_id_env", "TACHIBANA_USER_ID")),
        _clean_config_string(tachibana.get("password_env", "TACHIBANA_PASSWORD")),
        _clean_config_string(tachibana.get("second_password_env", "TACHIBANA_SECOND_PASSWORD")),
        _clean_config_string(tachibana.get("private_key_path_env", "TACHIBANA_PRIVATE_KEY_PATH")),
        _clean_config_string(tachibana.get("public_key_id_env", "TACHIBANA_PUBLIC_KEY_ID")),
    ]
    env_values = _read_env_file(ROOT / ".env") if (ROOT / ".env").exists() else {}
    return {
        "project": "AI Fund Lab / AIファンド研究所",
        "config": {
            "profile_id": profile_id_from(config),
            "profile_name": profile_name_from(config),
            "data_provider": config.get("data_provider"),
            "broker_provider": broker.get("provider", "paper"),
            "safety_mode": safety.get("mode", "paper"),
            "allow_live_trading": bool(safety.get("allow_live_trading", False)),
            "ai_commentary_provider": config.get("ai_commentary", {}).get("provider", "rule_based"),
            "ai_commentary_fallback_to_rule_based": bool(config.get("ai_commentary", {}).get("fallback_to_rule_based", True)),
            "ai_decision_enabled": bool(config.get("ai_decision", {}).get("enabled", False)),
            "ai_decision_provider": config.get("ai_decision", {}).get("provider", "openai"),
            "ai_decision_daily_call_limit": config.get("ai_decision", {}).get("daily_call_limit", 3),
            "ai_decision_fallback_to_rule_based": bool(config.get("ai_decision", {}).get("fallback_to_rule_based", True)),
            "openai_api_configured": bool(env_values.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")),
            "news_provider": config.get("news", {}).get("provider"),
        },
        "database": db_status,
        "articles": article_status,
        "safety": {
            "stop_trading_exists": stop_file.exists(),
            "emergency_stop_file": str(stop_file.relative_to(ROOT)) if stop_file.is_relative_to(ROOT) else str(stop_file),
            "live_trading_enabled": live_enabled,
            "live_trading_status": "enabled" if live_enabled else "disabled",
        },
        "tachibana": {
            "environment": tachibana.get("environment", "demo"),
            "demo_base_url": tachibana.get("demo_base_url"),
            "live_base_url": tachibana.get("live_base_url"),
            "credentials": {name: bool(env_values.get(name) or os.getenv(name)) for name in tachibana_env_names},
        },
    }


def _status_database(config: dict[str, Any]) -> dict[str, Any]:
    db_path = _database_path_from_config(config)
    status: dict[str, Any] = {
        "exists": db_path.exists(),
        "path": str(db_path.relative_to(ROOT)) if db_path.is_relative_to(ROOT) else str(db_path),
        "counts": {
            "portfolio_snapshots": 0,
            "trades": 0,
            "scoring_results": 0,
            "articles": 0,
            "pending_orders": 0,
            "ai_decisions": 0,
            "market_contexts": 0,
        },
        "latest_portfolio": None,
    }
    if not db_path.exists():
        return status
    try:
        with sqlite3.connect(db_path) as connection:
            connection.row_factory = sqlite3.Row
            profile_id = profile_id_from(config)
            for table in status["counts"]:
                if _table_has_column(connection, table, "profile_id"):
                    status["counts"][table] = connection.execute(f"SELECT COUNT(*) FROM {table} WHERE profile_id = ?", (profile_id,)).fetchone()[0]
                else:
                    status["counts"][table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if _table_has_column(connection, "portfolio_snapshots", "profile_id"):
                row = connection.execute("SELECT * FROM portfolio_snapshots WHERE profile_id = ? ORDER BY date DESC, id DESC LIMIT 1", (profile_id,)).fetchone()
            else:
                row = connection.execute("SELECT * FROM portfolio_snapshots ORDER BY date DESC, id DESC LIMIT 1").fetchone()
            if row:
                latest = dict(row)
                status["latest_portfolio"] = {
                    "date": latest.get("date"),
                    "total_assets": latest.get("total_assets"),
                    "cumulative_profit": latest.get("cumulative_profit"),
                    "open_positions_count": latest.get("open_positions_count"),
                    "closed_trades_count": latest.get("closed_trades_count"),
                }
    except sqlite3.Error as exc:
        status["error"] = str(exc)
    return status


def _table_has_column(connection: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _article_index_status(profile_id: str) -> dict[str, Any]:
    index_path = ROOT / "reports" / "articles" / "index.json"
    if not index_path.exists():
        return {
            "drafts_count": 0,
            "published_count": 0,
            "latest_draft": None,
            "latest_published": None,
        }
    articles = [
        item
        for item in read_json(index_path).get("articles", [])
        if item.get("profile_id") == profile_id
    ]
    drafts = sorted((item for item in articles if item.get("status") == "draft"), key=lambda item: item.get("updated_at", ""))
    published = sorted((item for item in articles if item.get("status") == "published"), key=lambda item: item.get("updated_at", ""))
    return {
        "drafts_count": len(drafts),
        "published_count": len(published),
        "latest_draft": drafts[-1].get("path") if drafts else None,
        "latest_published": published[-1].get("path") if published else None,
    }


def _relative_or_none(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def render_status_text(payload: dict[str, Any]) -> str:
    config = payload["config"]
    database = payload["database"]
    counts = database["counts"]
    latest = database.get("latest_portfolio") or {}
    articles = payload["articles"]
    safety = payload["safety"]
    tachibana = payload.get("tachibana", {})
    lines = [
        "AI Fund Lab Status",
        "",
        "## Profile",
        f"- profile_id: {config['profile_id']}",
        f"- profile_name: {config['profile_name']}",
        "",
        "## Current Config",
        f"- data_provider: {config['data_provider']}",
        f"- broker.provider: {config['broker_provider']}",
        f"- safety.mode: {config['safety_mode']}",
        f"- allow_live_trading: {str(config['allow_live_trading']).lower()}",
        f"- OpenAI API: {'configured' if config['openai_api_configured'] else 'not configured'}",
        f"- ai_commentary.provider: {config['ai_commentary_provider']}",
        f"- ai_commentary.fallback_to_rule_based: {str(config['ai_commentary_fallback_to_rule_based']).lower()}",
        f"- ai_decision.enabled: {str(config['ai_decision_enabled']).lower()}",
        f"- ai_decision.provider: {config['ai_decision_provider']}",
        f"- ai_decision.daily_call_limit: {config['ai_decision_daily_call_limit']}",
        f"- ai_decision.fallback_to_rule_based: {str(config['ai_decision_fallback_to_rule_based']).lower()}",
        f"- news.provider: {config['news_provider']}",
        "",
        "## Database",
        f"- exists: {str(database['exists']).lower()}",
        f"- portfolio_snapshots: {counts['portfolio_snapshots']}",
        f"- trades: {counts['trades']}",
        f"- scoring_results: {counts['scoring_results']}",
        f"- articles: {counts['articles']}",
        f"- pending_orders: {counts.get('pending_orders', 0)}",
        f"- ai_decisions: {counts.get('ai_decisions', 0)}",
        f"- market_contexts: {counts.get('market_contexts', 0)}",
        "",
        "## Latest Portfolio",
        f"- date: {latest.get('date', 'N/A')}",
        f"- total_assets: {_format_status_number(latest.get('total_assets'))}",
        f"- cumulative_profit: {_format_status_number(latest.get('cumulative_profit'))}",
        f"- open_positions_count: {latest.get('open_positions_count', 'N/A')}",
        f"- closed_trades_count: {latest.get('closed_trades_count', 'N/A')}",
        "",
        "## Latest Articles",
        f"- drafts_count: {articles['drafts_count']}",
        f"- published_count: {articles['published_count']}",
        f"- latest_draft: {articles.get('latest_draft') or 'N/A'}",
        f"- latest_published: {articles.get('latest_published') or 'N/A'}",
        "",
        "## Safety",
        f"- STOP_TRADING exists: {str(safety['stop_trading_exists']).lower()}",
        f"- live trading: {safety['live_trading_status']}",
        "",
        "## Tachibana",
        f"- environment: {tachibana.get('environment', 'N/A')}",
        f"- demo_base_url: {tachibana.get('demo_base_url') or 'N/A'}",
        f"- live_base_url: {tachibana.get('live_base_url') or 'N/A'}",
        *[f"- {name}: {'set' if is_set else 'not set'}" for name, is_set in (tachibana.get("credentials") or {}).items()],
    ]
    if database.get("error"):
        lines.extend(["", f"[WARN] database error: {database['error']}"])
    return "\n".join(lines)


def _format_status_number(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def run_preflight(profile_id: str | None = None, with_smoke_test: bool = False) -> None:
    global ACTIVE_PROFILE_ID
    if profile_id:
        ACTIVE_PROFILE_ID = profile_id
    results: list[dict[str, Any]] = []
    config: dict[str, Any] = {}
    env_values: dict[str, str] = {}

    try:
        config = load_config(CONFIG_PATH)
        _preflight_add(results, "OK", f"profile loaded: {profile_id_from(config)}", {"path": config.get("_profile_path")})
        _preflight_add(results, "OK", f"available profiles: {', '.join(item['profile_id'] for item in list_profiles())}")
    except Exception as exc:
        _preflight_add(results, "FAIL", f"profile load failed: {exc}")
        config = {}

    if config:
        _check_required_config(results, config)
        _check_runtime_value_sources(results, config)
        _check_jquants_plan_capabilities(results, config)
        _check_earnings_filter_preflight(results, config)
        _check_investor_context_preflight(results, config)
        if with_smoke_test:
            _check_jquants_smoke_preflight(results, config)

    env_path = ROOT / ".env"
    if env_path.exists():
        env_values = _read_env_file(env_path)
        _preflight_add(results, "OK", ".env exists", {"path": ".env"})
    else:
        _preflight_add(results, "WARN", ".env not found", {"path": ".env"})

    jquants_key_set = bool(env_values.get("JQUANTS_API_KEY") or os.getenv("JQUANTS_API_KEY"))
    openai_key_set = bool(env_values.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if jquants_key_set:
        _preflight_add(results, "OK", "JQUANTS_API_KEY is set")
    else:
        _preflight_add(results, "WARN", "JQUANTS_API_KEY not set")

    commentary = config.get("ai_commentary", {}) if config else {}
    commentary_provider = commentary.get("provider", "rule_based")
    fallback_enabled = bool(commentary.get("fallback_to_rule_based", True))
    ai_decision = config.get("ai_decision", {}) if config else {}
    ai_decision_enabled = bool(ai_decision.get("enabled", False))
    ai_decision_provider = ai_decision.get("provider", "openai")
    ai_decision_fallback = bool(ai_decision.get("fallback_to_rule_based", True))
    _preflight_add(results, "OK", f"ai_decision.enabled is {str(ai_decision_enabled).lower()}")
    _preflight_add(results, "OK", f"ai_decision.provider is {ai_decision_provider}")
    _preflight_add(results, "OK", f"ai_decision.daily_call_limit is {ai_decision.get('daily_call_limit', 3)}")
    _preflight_add(results, "OK" if ai_decision_fallback else "WARN", f"ai_decision.fallback_to_rule_based is {str(ai_decision_fallback).lower()}")
    _preflight_add(results, "OK", f"ai_commentary.provider is {commentary_provider}")
    _preflight_add(results, "OK" if fallback_enabled else "WARN", f"ai_commentary.fallback_to_rule_based is {str(fallback_enabled).lower()}")
    if commentary_provider == "openai":
        if openai_key_set:
            _preflight_add(results, "OK", "OPENAI_API_KEY is set for OpenAI commentary")
        else:
            _preflight_add(results, "WARN", "OPENAI_API_KEY is not set. Falling back to rule_based decision/commentary.")
            _preflight_add(results, "OK" if fallback_enabled else "WARN", "ai_commentary fallback_to_rule_based checked")
    else:
        _preflight_add(results, "SKIP", "OpenAI commentary skipped because ai_commentary.provider is rule_based")

    if ai_decision_enabled and ai_decision_provider == "openai":
        if openai_key_set:
            _preflight_add(results, "OK", "OPENAI_API_KEY is set for AI decision")
        else:
            _preflight_add(results, "WARN", "OPENAI_API_KEY is not set. Falling back to rule_based decision/commentary.")
            _preflight_add(results, "OK" if ai_decision_fallback else "WARN", "ai_decision fallback_to_rule_based checked")
    elif not ai_decision_enabled:
        _preflight_add(results, "SKIP", "OpenAI decision skipped because ai_decision.enabled is false")
    else:
        _preflight_add(results, "SKIP", f"OpenAI decision skipped because ai_decision.provider is {ai_decision_provider}")

    _ensure_required_directories(results)
    _check_gitignore_rules(results)
    if config:
        _check_database(results, config)
    else:
        _preflight_add(results, "SKIP", "database check skipped because config is unavailable")

    if jquants_key_set:
        _check_jquants_health(results, config)
    else:
        _preflight_add(results, "WARN", "J-Quants healthcheck skipped because JQUANTS_API_KEY is not set")

    if openai_key_set and (commentary_provider == "openai" or (ai_decision_enabled and ai_decision_provider == "openai")):
        _check_openai_configuration(results, openai_key_set, True)
    else:
        _preflight_add(results, "SKIP", "OpenAI connection skipped because it is optional or not configured")

    output = _preflight_payload(results)
    output["profile_id"] = profile_id_from(config) if config else ACTIVE_PROFILE_ID
    output["profile_name"] = profile_name_from(config) if config else ACTIVE_PROFILE_ID
    output_dir = ROOT / "reports" / (profile_id_from(config) if config else ACTIVE_PROFILE_ID) / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "preflight_latest.json"
    markdown_path = output_dir / "preflight_latest.md"
    write_json(json_path, output)
    write_text(markdown_path, render_preflight_markdown(output))

    warnings = output["summary"]["warnings"]
    failures = output["summary"]["failures"]
    if failures:
        final = f"Preflight completed: FAILED with {failures} failures and {warnings} warnings"
    elif warnings:
        final = f"Preflight completed: OK with {warnings} warnings"
    else:
        final = "Preflight completed: OK"
    print(final)
    print(f"json: {json_path.relative_to(ROOT)}")
    print(f"markdown: {markdown_path.relative_to(ROOT)}")


def _preflight_add(results: list[dict[str, Any]], status: str, message: str, details: dict[str, Any] | None = None) -> None:
    normalized = status.upper()
    results.append(
        {
            "status": normalized,
            "message": message,
            "details": details or {},
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    print(f"[{normalized}] {message}")


def _check_required_config(results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    required = {
        "initial_capital": ("portfolio", "initial_cash"),
        "data_provider": ("data_provider",),
        "broker": ("broker",),
        "tachibana": ("tachibana",),
        "kabu_station": ("kabu_station",),
        "trading": ("trading",),
        "selection": ("selection",),
        "costs": ("costs",),
        "database": ("database",),
        "news": ("news",),
        "ai_commentary": ("ai_commentary",),
        "ai_decision": ("ai_decision",),
        "safety": ("safety",),
    }
    missing = []
    for label, path in required.items():
        if not _config_has_path(config, path):
            missing.append(label)
    if missing:
        _preflight_add(results, "FAIL", f"required config missing: {', '.join(missing)}")
    else:
        _preflight_add(results, "OK", "required config keys present")
    if "safety" in config:
        safety = config.get("safety", {})
        stop_file = Path(safety.get("emergency_stop_file", "storage/STOP_TRADING"))
        if not stop_file.is_absolute():
            stop_file = ROOT / stop_file
        if stop_file.parent.exists():
            _preflight_add(results, "OK", "safety emergency stop directory exists", {"path": str(stop_file.parent.relative_to(ROOT))})
        else:
            _preflight_add(results, "WARN", "safety emergency stop directory does not exist", {"path": str(stop_file.parent)})
        live_message = "allow_live_trading is false" if not safety.get("allow_live_trading", False) else "allow_live_trading is true"
        _preflight_add(results, "OK" if not safety.get("allow_live_trading", False) else "WARN", live_message)
    if "broker" in config:
        broker = config.get("broker", {})
        broker_provider = broker.get("provider", "paper")
        live_enabled = bool(broker.get("live_trading_enabled", False))
        safety_live = bool(config.get("safety", {}).get("allow_live_trading", False))
        _preflight_add(results, "OK", f"broker.provider is {broker_provider}")
        _preflight_add(results, "OK" if not live_enabled else "WARN", f"broker.live_trading_enabled is {str(live_enabled).lower()}")
        _preflight_add(results, "OK" if broker_provider == "paper" else "WARN", "current broker is paper" if broker_provider == "paper" else "current broker is not paper")
        if live_enabled and safety_live:
            _preflight_add(results, "WARN", "live trading double lock is open, but live broker stubs are still not implemented")
        else:
            _preflight_add(results, "OK", "live trading is disabled by double lock")
    if "tachibana" in config:
        _check_tachibana_configuration(results, config)
    if "kabu_station" in config:
        _check_kabu_station_configuration(results, config)


def _check_runtime_value_sources(results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    sources = config.get("_value_sources", {}) if isinstance(config.get("_value_sources"), dict) else {}
    broker_mode = config.get("broker", {}).get("mode") or config.get("broker", {}).get("provider", "paper")
    auto_order_enabled = bool(config.get("operation", {}).get("auto_order_enabled", False))
    _preflight_add(results, "OK", f"Profile: {profile_id_from(config)} (source={sources.get('profile', 'default')})")
    _preflight_add(results, "OK", f"Data Provider: {config.get('data_provider', 'jquants')} (source={sources.get('provider', 'default')})")
    _preflight_add(results, "OK", f"Broker: {broker_mode} (source={sources.get('broker', 'default')})")
    _preflight_add(
        results,
        "OK" if not auto_order_enabled else "WARN",
        f"operation.auto_order_enabled: {str(auto_order_enabled).lower()} (source={sources.get('auto_order_enabled', 'default')})",
    )


def _config_has_path(config: dict[str, Any], path: tuple[str, ...]) -> bool:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return current is not None


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _load_env_file_to_process(path: Path) -> None:
    if not path.exists():
        return
    for key, value in _read_env_file(path).items():
        os.environ.setdefault(key, value)


def _ensure_required_directories(results: list[dict[str, Any]]) -> None:
    directories = [
        ROOT / "storage",
        ROOT / "data" / "raw",
        ROOT / "data" / "processed",
        ROOT / "data" / "raw" / "news",
        ROOT / "reports",
        ROOT / "reports" / "charts",
        ROOT / "reports" / "backtests",
        ROOT / "reports" / "articles",
        ROOT / "reports" / "articles" / "daily",
        ROOT / "reports" / "paper",
        ROOT / "articles" / "drafts",
        ROOT / "articles" / "published",
        ROOT / "logs",
        ROOT / "logs" / "safety",
        ROOT / "logs" / "ai_decision",
        ROOT / "logs" / "market_context",
    ]
    created = []
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(str(directory.relative_to(ROOT)))
    details = {"created": created, "checked": [str(directory.relative_to(ROOT)) for directory in directories]}
    _preflight_add(results, "OK", "required directories ready", details)


def _check_gitignore_rules(results: list[dict[str, Any]]) -> None:
    gitignore_path = ROOT / ".gitignore"
    required_rules = [".env", "logs/", "analysis_logs/", "data/raw/", "data/processed/", "articles/drafts/", "*.sqlite3", "*.db"]
    if not gitignore_path.exists():
        _preflight_add(results, "FAIL", ".gitignore not found", {"required_rules": required_rules})
        return
    lines = {line.strip() for line in gitignore_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")}
    missing = [rule for rule in required_rules if rule not in lines]
    if missing:
        _preflight_add(results, "FAIL", f"gitignore rules missing: {', '.join(missing)}", {"missing": missing})
        return
    _preflight_add(results, "OK", "gitignore rules checked")


def _check_database(results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    db_path = _database_path_from_config(config)
    if not db_path.exists():
        _preflight_add(
            results,
            "WARN",
            "database does not exist; run `python src/main.py --mode init-db`",
            {"path": str(db_path.relative_to(ROOT)) if db_path.is_relative_to(ROOT) else str(db_path)},
        )
        return
    required_tables = {
        "portfolio_snapshots",
        "trades",
        "scoring_results",
        "screening_results",
        "reflections",
        "articles",
        "safety_events",
        "pending_orders",
        "ai_decisions",
        "market_contexts",
        "ai_analysis_exports",
    }
    try:
        with sqlite3.connect(db_path) as connection:
            rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    except sqlite3.Error as exc:
        _preflight_add(results, "FAIL", f"database check failed: {exc}")
        return
    tables = {row[0] for row in rows}
    missing = sorted(required_tables - tables)
    if missing:
        _preflight_add(results, "FAIL", f"database tables missing: {', '.join(missing)}", {"path": str(db_path), "missing": missing})
        return
    _preflight_add(results, "OK", "database exists", {"path": str(db_path.relative_to(ROOT)) if db_path.is_relative_to(ROOT) else str(db_path)})
    _preflight_add(results, "OK", "database tables checked")


def _database_path_from_config(config: dict[str, Any]) -> Path:
    configured = config.get("database", {}).get("path", "storage/ai_fund_lab.sqlite3")
    path = Path(configured)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _reset_jquants_api_session() -> None:
    JQUANTS_API_SESSION.clear()
    JQUANTS_API_SESSION.update(
        {
            "api_calls_by_endpoint": {},
            "api_errors_by_endpoint": {},
            "api_retry_count": {},
            "api_retry_success_count": {},
            "disabled_features_reason": {},
            "payloads": {},
        }
    )


def _jquants_api_session() -> dict[str, Any]:
    if not JQUANTS_API_SESSION:
        _reset_jquants_api_session()
    return JQUANTS_API_SESSION


def _api_call_allowed(endpoint: str) -> tuple[bool, str]:
    session = _jquants_api_session()
    disabled = session.setdefault("disabled_features_reason", {})
    if endpoint in disabled:
        return False, str(disabled[endpoint])
    calls = session.setdefault("api_calls_by_endpoint", {})
    limit = LIGHT_API_CALL_LIMITS.get(endpoint, 3)
    if int(calls.get(endpoint, 0) or 0) >= limit:
        reason = "api_call_limit_reached"
        disabled[endpoint] = reason
        return False, reason
    calls[endpoint] = int(calls.get(endpoint, 0) or 0) + 1
    return True, ""


def _record_api_retry(endpoint: str) -> None:
    retries = _jquants_api_session().setdefault("api_retry_count", {})
    retries[endpoint] = int(retries.get(endpoint, 0) or 0) + 1


def _record_api_retry_success(endpoint: str) -> None:
    retries = _jquants_api_session().setdefault("api_retry_success_count", {})
    retries[endpoint] = int(retries.get(endpoint, 0) or 0) + 1


def _record_api_error(endpoint: str, reason: str) -> None:
    session = _jquants_api_session()
    errors = session.setdefault("api_errors_by_endpoint", {})
    errors[endpoint] = int(errors.get(endpoint, 0) or 0) + 1
    if reason:
        session.setdefault("disabled_features_reason", {})[endpoint] = reason


def _api_unavailable_payload(endpoint: str, reason: str) -> dict[str, Any]:
    return {
        "records": [],
        "cache_path": "",
        "from_cache": False,
        "fallback_used": False,
        "warning": reason,
        "available": False,
        "usable": False,
        "saved": False,
        "reason": reason,
        "endpoint": endpoint,
    }


def _preload_light_api_context(
    config: dict[str, Any],
    start_date: date,
    end_date: date,
    indicator_fetch_start_date: date | None = None,
) -> None:
    if bool(config.get("earnings_filter", {}).get("enabled", False)):
        earnings_start, earnings_end = _earnings_calendar_preload_range(start_date, end_date)
        payload = _load_earnings_calendar_for_period(earnings_start, earnings_end, config)
        _jquants_api_session().setdefault("payloads", {})["earnings_calendar"] = {
            "records": payload.get("records", []),
            "metadata": payload.get("metadata", {}),
            "start_date": earnings_start.isoformat(),
            "end_date": earnings_end.isoformat(),
        }
    if _relative_strength_enabled_for_indicators(config) and _expected_relative_strength_benchmark_source(config) == "topix":
        topix_start = indicator_fetch_start_date or _topix_preload_start_date(start_date)
        payload = _load_topix_prices_for_period_with_options(topix_start, end_date, config, use_preloaded=False)
        _jquants_api_session().setdefault("payloads", {})["topix_prices"] = {
            "records": payload.get("records", []),
            "metadata": {
                "available": bool(payload.get("available")),
                "cache_path": payload.get("cache_path"),
                "from_cache": payload.get("from_cache"),
                "fallback_used": payload.get("fallback_used"),
                "warning": payload.get("warning"),
                "reason": payload.get("reason"),
                "topix_records_loaded": len(payload.get("records", [])),
            },
            "start_date": topix_start.isoformat(),
            "end_date": end_date.isoformat(),
        }
    if bool(config.get("features", {}).get("investor_context")) and bool(config.get("scoring", {}).get("use_investor_context_score")):
        payload = _load_investor_types_for_period(start_date, end_date, config)
        _jquants_api_session().setdefault("payloads", {})["investor_types"] = {
            "records": payload.get("records", []),
            "metadata": payload.get("metadata", {}),
            "start_date": payload.get("start_date") or start_date.isoformat(),
            "end_date": payload.get("end_date") or end_date.isoformat(),
        }


def _earnings_calendar_preload_range(start_date: date, end_date: date) -> tuple[date, date]:
    return start_date - timedelta(days=14), end_date + timedelta(days=14)


def _topix_preload_start_date(start_date: date) -> date:
    dates = previous_business_dates(start_date, 35)
    return dates[0] if dates else start_date


def _merge_jquants_api_session_summary(target: dict[str, Any]) -> None:
    session = _jquants_api_session()
    for key in ["api_calls_by_endpoint", "api_errors_by_endpoint", "api_retry_count", "api_retry_success_count"]:
        merged = target.setdefault(key, {})
        for endpoint, count in session.get(key, {}).items():
            merged[endpoint] = int(merged.get(endpoint, 0) or 0) + int(count or 0)
    disabled = target.setdefault("disabled_features_reason", {})
    disabled.update(session.get("disabled_features_reason", {}))
    investor_summary = session.get("investor_types_fetch_summary")
    if isinstance(investor_summary, dict):
        _merge_investor_types_fetch_summary(target, investor_summary)


def _merge_investor_types_fetch_summary(target: dict[str, Any], summary: dict[str, Any]) -> None:
    target["investor_types_chunks_total"] = int(target.get("investor_types_chunks_total") or 0) + int(summary.get("investor_types_chunks_total") or 0)
    target["investor_types_chunks_success"] = int(target.get("investor_types_chunks_success") or 0) + int(summary.get("investor_types_chunks_success") or 0)
    target["investor_types_chunks_failed"] = int(target.get("investor_types_chunks_failed") or 0) + int(summary.get("investor_types_chunks_failed") or 0)
    target["investor_types_records_loaded"] = int(target.get("investor_types_records_loaded") or 0) + int(summary.get("investor_types_records_loaded") or 0)
    requested_starts = [
        value
        for value in [target.get("investor_types_fetch_requested_start"), summary.get("investor_types_fetch_requested_start")]
        if value
    ]
    clamped_starts = [
        value
        for value in [target.get("investor_types_fetch_clamped_start"), summary.get("investor_types_fetch_clamped_start")]
        if value
    ]
    starts = [value for value in [target.get("investor_types_fetch_start"), summary.get("investor_types_fetch_start")] if value]
    ends = [value for value in [target.get("investor_types_fetch_end"), summary.get("investor_types_fetch_end")] if value]
    disabled_reasons = [
        str(value)
        for value in [target.get("investor_types_disabled_reason"), summary.get("investor_types_disabled_reason")]
        if value
    ]
    if requested_starts:
        target["investor_types_fetch_requested_start"] = min(str(value) for value in requested_starts)
    if clamped_starts:
        target["investor_types_fetch_clamped_start"] = min(str(value) for value in clamped_starts)
    if starts:
        target["investor_types_fetch_start"] = min(str(value) for value in starts)
    if ends:
        target["investor_types_fetch_end"] = max(str(value) for value in ends)
    if disabled_reasons:
        target["investor_types_disabled_reason"] = ", ".join(sorted(set(disabled_reasons)))


def _check_jquants_plan_capabilities(results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    plan = _jquants_plan(config)
    sources = config.get("_value_sources", {}) if isinstance(config.get("_value_sources"), dict) else {}
    source = sources.get("jquants_plan", "default")
    resolution = config.get("_jquants_plan_resolution", {}) if isinstance(config.get("_jquants_plan_resolution"), dict) else {}
    status = jquants_capability_status(plan)
    compatibility = jquants_profile_compatibility(profile_id_from(config), plan)
    _preflight_add(
        results,
        "OK",
        "J-Quants Plan Resolution:",
        {
            "plan": plan,
            "source": source,
            "config_path": resolution.get("config_path"),
            "capabilities": status,
        },
    )
    _preflight_add(results, "OK", f"J-Quants Plan: {plan} (source={source})", {"plan": plan, "source": source})
    _preflight_add(results, "OK", f"Rate Limit: {_jquants_requests_per_minute(config)} requests/min")
    _preflight_add(results, "OK", f"Parallel Fetch: {'enabled' if _jquants_parallel_fetch(config) else 'disabled'}")
    _preflight_add(results, "OK", "Available capabilities:", status)
    _preflight_add(results, "OK", "J-Quants supported date range:", jquants_supported_date_ranges(config))
    for endpoint, earliest in jquants_supported_date_ranges(config).items():
        _preflight_add(results, "OK", f"{endpoint} earliest: {earliest}")
    for capability in ["prices", "earnings_calendar", "topix_prices", "investor_types"]:
        capability_status = status.get(capability, "disabled")
        _preflight_add(
            results,
            "OK" if capability_status == "OK" else "SKIP",
            f"J-Quants capability {capability}: {capability_status}",
            {"plan": plan, "capability": capability},
        )
    for endpoint in ["topix_prices", "investor_types", "earnings_calendar", "financial_statements"]:
        cache_status = _jquants_endpoint_cache_status(endpoint, date.today())
        _preflight_add(
            results,
            "OK" if cache_status["usable"] else "WARN",
            f"{endpoint} cache: exists={str(cache_status['exists']).lower()} records={cache_status['records']} usable={str(cache_status['usable']).lower()} status={cache_status['status']}",
            cache_status,
        )
    missing = compatibility["missing_capabilities"]
    fallback_applied = compatibility["fallback_applied"]
    _preflight_add(results, "OK", f"profile required capabilities: {', '.join(compatibility['profile_required_capabilities']) or 'none'}")
    _preflight_add(results, "OK", f"current plan capabilities: {', '.join(compatibility['current_plan_capabilities']) or 'none'}")
    _preflight_add(results, "WARN" if missing else "OK", f"missing capabilities: {', '.join(missing) if missing else 'none'}")
    _preflight_add(
        results,
        "WARN" if fallback_applied else "OK",
        "fallback applied: "
        + (
            "; ".join(f"{item['capability']} -> {item['policy']}" for item in fallback_applied)
            if fallback_applied
            else "none"
        ),
        compatibility,
    )
    _preflight_add(results, "OK" if compatibility["can_run_backtest"] else "FAIL", f"can_run_backtest: {str(compatibility['can_run_backtest']).lower()}")
    _preflight_add(results, "OK" if compatibility["can_run_paper"] else "FAIL", f"can_run_live/paper: {str(compatibility['can_run_paper']).lower()}")


def _check_jquants_smoke_preflight(results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    try:
        summary = build_jquants_smoke_test("all", config)
    except Exception as exc:
        _preflight_add(results, "FAIL", f"J-Quants smoke test failed: {exc}")
        return
    for row in summary.get("endpoints", []):
        result = str(row.get("result") or "ERROR")
        status = "OK" if result == "OK" else "SKIP" if result == "SKIPPED_PLAN" else "WARN"
        _preflight_add(
            results,
            status,
            f"smoke {row.get('endpoint')}: {result} records={row.get('records', 0)}",
            row,
        )


def _jquants_endpoint_cache_status(endpoint: str, reference_date: date) -> dict[str, Any]:
    cache_path = _jquants_expected_cache_path(endpoint, reference_date)
    state = _cache_file_state(str(cache_path))
    status = "missing"
    if state["exists"]:
        status = "usable" if state["usable"] else "empty_cache"
    return {
        "endpoint": endpoint,
        "path": _relative_path_text(cache_path),
        "status": status,
        **state,
    }


def _jquants_expected_cache_path(endpoint: str, reference_date: date) -> Path:
    cache_root = ROOT / "data" / "cache" / "jquants"
    if endpoint == "topix_prices":
        start_date = reference_date - timedelta(days=45)
        return cache_root / endpoint / f"{start_date.isoformat()}_to_{reference_date.isoformat()}.json"
    if endpoint == "investor_types":
        start_date, end_date = _investor_types_fetch_ranges(reference_date)[0]
        return cache_root / endpoint / f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"
    if endpoint == "financial_statements":
        start_date = reference_date - timedelta(days=365)
        return cache_root / endpoint / f"{start_date.isoformat()}_to_{reference_date.isoformat()}.json"
    if endpoint == "earnings_calendar":
        return cache_root / endpoint / f"{reference_date.isoformat()}.json"
    return cache_root / endpoint


def _cache_file_state(path_text: Any) -> dict[str, Any]:
    if not path_text:
        return {"exists": False, "records": 0, "latest_date": None}
    path = Path(str(path_text))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() or not path.is_file():
        return {"exists": False, "records": 0, "usable": False, "latest_date": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"exists": True, "records": 0, "usable": False, "latest_date": None}
    records = payload.get("records", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        records = []
    return {
        "exists": True,
        "records": len(records),
        "usable": len(records) > 0,
        "latest_date": _latest_record_date(records),
    }


def _latest_record_date(records: list[dict[str, Any]]) -> str | None:
    dates = []
    for record in records:
        value = record.get("date") or record.get("Date") or record.get("TargetDate") or record.get("DisclosedDate")
        if value:
            dates.append(str(value))
    return max(dates) if dates else None


def _check_investor_context_preflight(results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    enabled = bool(config.get("features", {}).get("investor_context")) and bool(config.get("scoring", {}).get("use_investor_context_score"))
    capability_status = config.get("jquants", {}).get("capability_status", {}).get("investor_types", "disabled")
    _preflight_add(results, "OK" if capability_status == "OK" else "SKIP", f"investor_types capability: {capability_status}")
    _preflight_add(results, "OK" if enabled else "SKIP", f"investor_context {'enabled' if enabled else 'disabled'}")
    if not enabled:
        return
    payload = _load_investor_context_for_date(date.today(), config, force_refresh=False)
    metadata = payload.get("metadata", {})
    status = "OK" if metadata.get("available", False) else "WARN"
    _preflight_add(
        results,
        status,
        f"investor_types cache: {metadata.get('cache_path') or 'N/A'}",
        {
            **_cache_file_state(metadata.get("cache_path")),
            "from_cache": metadata.get("from_cache"),
            "fallback_used": metadata.get("fallback_used"),
            "warning": metadata.get("warning"),
        },
    )
    _preflight_add(results, status, f"latest investor data week: {metadata.get('latest_investor_data_week') or 'N/A'}")


def _check_earnings_filter_preflight(results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    filter_config = config.get("earnings_filter", {})
    enabled = bool(filter_config.get("enabled", False))
    capability_status = config.get("jquants", {}).get("capability_status", {}).get("earnings_calendar", "OK")
    _preflight_add(results, "OK" if capability_status == "OK" else "SKIP", f"earnings_calendar capability: {capability_status}")
    _preflight_add(results, "OK" if enabled else "SKIP", f"earnings_filter {'enabled' if enabled else 'disabled'}")
    if not enabled:
        return
    payload = _load_earnings_calendar_for_date(date.today(), config, force_refresh=False)
    metadata = payload.get("metadata", {})
    records = payload.get("records", [])
    counts = earnings_counts(records, date.today()) if records else {"today": 0, "next_business_day": 0}
    status = "OK" if metadata.get("filter_available", False) else "WARN"
    _preflight_add(
        results,
        status,
        f"earnings calendar cache date: {metadata.get('cache_date') or 'N/A'}",
        {
            **_cache_file_state(metadata.get("cache_path")),
            "cache_path": metadata.get("cache_path"),
            "from_cache": metadata.get("from_cache"),
            "fallback_used": metadata.get("fallback_used"),
            "warning": metadata.get("warning"),
        },
    )
    _preflight_add(results, "OK", f"today earnings count: {counts['today']}")
    _preflight_add(results, "OK", f"next business day earnings count: {counts['next_business_day']}")


def _check_jquants_health(results: list[dict[str, Any]], config: dict[str, Any] | None = None) -> None:
    try:
        plan_config = config or {}
        provider = JQuantsDataProvider(
            ROOT / ".env",
            plan=_jquants_plan(plan_config),
            requests_per_minute=_jquants_requests_per_minute(plan_config),
            parallel_fetch=_jquants_parallel_fetch(plan_config),
            max_parallel_requests=_jquants_max_parallel_requests(plan_config),
        )
        listed = provider.get_listed_stocks()
    except Exception as exc:
        _preflight_add(results, "WARN", f"J-Quants healthcheck skipped or failed: {exc}")
        return
    _preflight_add(results, "OK", "J-Quants healthcheck successful", {"listed_stocks": len(listed), "plan": provider.plan})


def _check_openai_configuration(results: list[dict[str, Any]], api_key_set: bool, fallback_enabled: bool) -> None:
    if not api_key_set:
        _preflight_add(results, "WARN", "OpenAI connection skipped because OPENAI_API_KEY is not set; rule_based fallback is available")
        return
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        _preflight_add(results, "WARN", "OpenAI package is not installed; rule_based fallback is available")
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ModuleNotFoundError:
        pass
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        client.models.list()
    except Exception as exc:
        _preflight_add(results, "WARN", f"OpenAI connection failed: {exc}; rule_based fallback is available")
        return
    _preflight_add(results, "OK", "OpenAI connection successful")


def _check_kabu_station_configuration(results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    kabu_station = config.get("kabu_station", {})
    enabled = bool(kabu_station.get("enabled", False))
    api_base_url = _clean_config_string(kabu_station.get("api_base_url", ""))
    password_env = _clean_config_string(kabu_station.get("api_password_env", "KABU_STATION_API_PASSWORD"))
    env_values = _read_env_file(ROOT / ".env") if (ROOT / ".env").exists() else {}
    password_set = bool(env_values.get(password_env) or os.getenv(password_env))
    _preflight_add(results, "OK" if not enabled else "WARN", f"kabu_station.enabled is {str(enabled).lower()}")
    _preflight_add(results, "OK", f"kabu_station.api_base_url is {api_base_url}")
    _preflight_add(results, "OK" if password_set else "WARN", f"{password_env} is {'set' if password_set else 'not set'}")
    _preflight_add(results, "OK", "kabuステーションAPI is not connected; PaperBroker only")


def _check_tachibana_configuration(results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    tachibana = config.get("tachibana", {})
    broker = config.get("broker", {})
    safety = config.get("safety", {})
    environment = _clean_config_string(tachibana.get("environment", "demo"))
    demo_base_url = _clean_config_string(tachibana.get("demo_base_url", ""))
    live_base_url = _clean_config_string(tachibana.get("live_base_url", ""))
    user_id_env = _clean_config_string(tachibana.get("user_id_env", "TACHIBANA_USER_ID"))
    password_env = _clean_config_string(tachibana.get("password_env", "TACHIBANA_PASSWORD"))
    second_password_env = _clean_config_string(tachibana.get("second_password_env", "TACHIBANA_SECOND_PASSWORD"))
    private_key_path_env = _clean_config_string(tachibana.get("private_key_path_env", "TACHIBANA_PRIVATE_KEY_PATH"))
    public_key_id_env = _clean_config_string(tachibana.get("public_key_id_env", "TACHIBANA_PUBLIC_KEY_ID"))
    env_values = _read_env_file(ROOT / ".env") if (ROOT / ".env").exists() else {}
    live_allowed = (
        bool(broker.get("live_trading_enabled", False))
        and bool(safety.get("allow_live_trading", False))
        and environment == "live"
    )
    _preflight_add(results, "OK", f"tachibana.environment is {environment}")
    _preflight_add(results, "OK", f"tachibana.demo_base_url is {demo_base_url}")
    _preflight_add(results, "OK", f"tachibana.live_base_url is {live_base_url}")
    for env_name in [user_id_env, password_env, second_password_env, private_key_path_env, public_key_id_env]:
        is_set = bool(env_values.get(env_name) or os.getenv(env_name))
        _preflight_add(results, "OK" if is_set else "WARN", f"{env_name} is {'set' if is_set else 'not set'}")
    _preflight_add(results, "WARN" if live_allowed else "OK", "tachibana live trading gate is open" if live_allowed else "tachibana live trading is disabled")
    _preflight_add(results, "OK", "立花証券 e支店 API is not connected; PaperBroker only")
    healthcheck = build_tachibana_healthcheck(config, "demo")
    status = "FAIL" if healthcheck["status"] == "FAILED" else "WARN" if healthcheck["status"] == "OK_WITH_WARNINGS" else "OK"
    _preflight_add(results, status, f"tachibana-healthcheck demo summary: {healthcheck['status']}")


def _clean_config_string(value: Any) -> str:
    return str(value).strip().strip('"').strip("'")


def _preflight_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
    warnings = sum(1 for item in results if item["status"] == "WARN")
    failures = sum(1 for item in results if item["status"] == "FAIL")
    skipped = sum(1 for item in results if item["status"] == "SKIP")
    ok = sum(1 for item in results if item["status"] == "OK")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "ok": ok,
            "warnings": warnings,
            "failures": failures,
            "skipped": skipped,
            "status": "FAILED" if failures else "OK",
        },
        "results": results,
    }


def render_preflight_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Preflight Check",
        "",
        f"- Profile: {payload.get('profile_id', 'unknown')} {payload.get('profile_name', '')}",
        f"- Generated at: {payload['generated_at']}",
        f"- Status: {summary['status']}",
        f"- OK: {summary['ok']}",
        f"- Warnings: {summary['warnings']}",
        f"- Failures: {summary['failures']}",
        f"- Skipped: {summary['skipped']}",
        "",
        "## Results",
        "",
    ]
    for item in payload["results"]:
        lines.append(f"- [{item['status']}] {item['message']}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- APIキーの値は表示・保存していません。",
            "- `WARN` がある場合は、run-daily / backtest の前に内容を確認してください。",
            "- DBがない場合は `python src/main.py --mode init-db` を実行してください。",
        ]
    )
    return "\n".join(lines)


def run_list_stocks(provider_name: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("list-stocks mode currently supports --provider jquants only.")

    config = load_config(CONFIG_PATH)
    try:
        provider = JQuantsDataProvider(
            ROOT / ".env",
            plan=_jquants_plan(config),
            requests_per_minute=_jquants_requests_per_minute(config),
            parallel_fetch=_jquants_parallel_fetch(config),
            max_parallel_requests=_jquants_max_parallel_requests(config),
        )
        listed_stocks = provider.get_listed_stocks()
        normalized_stocks = [_normalize_listed_stock(record) for record in listed_stocks]
        prime_stocks = [stock for stock in normalized_stocks if stock.get("section") == "TSEPrime"]
    except RuntimeError as exc:
        print(f"J-Quants listed stocks fetch failed: {exc}")
        raise SystemExit(1) from exc

    listed_output_path = ROOT / "data" / "raw" / "listed_stocks_jquants.json"
    write_json(
        listed_output_path,
        {
            "provider": "jquants",
            "source": "listed stocks",
            "total_count": len(listed_stocks),
            "market_counts": market_section_counts(normalized_stocks),
            "stocks": normalized_stocks,
        },
    )

    output_path = ROOT / "data" / "raw" / "prime_stocks_jquants.json"
    write_json(
        output_path,
        {
            "provider": "jquants",
            "source": "listed stocks",
            "total_count": len(listed_stocks),
            "prime_count": len(prime_stocks),
            "stocks": prime_stocks,
        },
    )

    print(f"all listed stocks: {len(listed_stocks)}")
    print(f"prime stocks: {len(prime_stocks)}")
    print(f"listed saved: {listed_output_path.relative_to(ROOT)}")
    print(f"saved: {output_path.relative_to(ROOT)}")


def run_fetch_prices(provider_name: str, target_date_text: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("fetch-prices mode currently supports --provider jquants only.")

    try:
        target_date = date.fromisoformat(target_date_text)
    except ValueError as exc:
        raise SystemExit("--date must be in YYYY-MM-DD format.") from exc

    config = load_config(CONFIG_PATH)
    listed_stock_by_code = {str(stock["code"]): stock for stock in _listed_stock_master() if stock.get("code")}
    allowed_codes = set(_allowed_stock_master_by_code(config))
    if not listed_stock_by_code:
        raise SystemExit("Listed stock master is empty. Re-run list-stocks.")
    if not allowed_codes:
        raise SystemExit(_market_filter_empty_message(config))
    try:
        provider = JQuantsDataProvider(
            ROOT / ".env",
            plan=_jquants_plan(config),
            requests_per_minute=_jquants_requests_per_minute(config),
            parallel_fetch=_jquants_parallel_fetch(config),
            max_parallel_requests=_jquants_max_parallel_requests(config),
        )
        daily_prices = provider.get_daily_prices(target_date)
    except RuntimeError as exc:
        print(f"J-Quants daily prices fetch failed: {exc}")
        raise SystemExit(1) from exc

    if not daily_prices:
        raise SystemExit(f"No daily price data found for {target_date_text}. The date may be weekend, holiday, or not updated yet.")

    listed_prices = [
        _normalize_daily_price_with_market(record, listed_stock_by_code)
        for record in daily_prices
        if _get_first(record, ["code", "Code", "LocalCode"]) in listed_stock_by_code
    ]
    if not listed_prices:
        raise SystemExit(f"No listed stock daily price data found for {target_date_text}. The date may be weekend, holiday, or not updated yet.")

    output_path = ROOT / "data" / "raw" / f"prices_{target_date_text}.json"
    write_json(
        output_path,
        {
            "provider": "jquants",
            "date": target_date_text,
            "total_count": len(daily_prices),
            "listed_count": len(listed_prices),
            "allowed_count": sum(1 for item in listed_prices if item.get("code") in allowed_codes),
            "market_counts": market_section_counts(listed_prices),
            "prices": listed_prices,
        },
    )

    print(f"date: {target_date_text}")
    print(f"all prices: {len(daily_prices)}")
    print(f"listed prices: {len(listed_prices)}")
    print(f"allowed prices: {sum(1 for item in listed_prices if item.get('code') in allowed_codes)}")
    print(f"saved: {output_path.relative_to(ROOT)}")


def run_calculate_indicators(provider_name: str, target_date_text: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("calculate-indicators mode currently supports --provider jquants only.")

    try:
        target_date = date.fromisoformat(target_date_text)
    except ValueError as exc:
        raise SystemExit("--date must be in YYYY-MM-DD format.") from exc

    config = load_config(CONFIG_PATH)
    stock_by_code = _allowed_stock_master_by_code(config)
    prime_codes = set(stock_by_code)
    stock_names = {code: stock.get("name", "") for code, stock in stock_by_code.items()}
    stock_sectors = {code: stock.get("sector_name", "") for code, stock in stock_by_code.items()}
    stock_sections = {code: stock.get("section", "Unknown") for code, stock in stock_by_code.items()}
    if not prime_codes:
        raise SystemExit(_market_filter_empty_message(config))
    indicator_mode = _backtest_indicator_mode(config) if BACKTEST_MODE_ACTIVE else "full"
    enable_relative_strength = _relative_strength_enabled_for_indicators(config)
    output_path = ROOT / "data" / "processed" / f"indicators_{target_date_text}.json"
    profile_output_path = processed_profile_path(config, f"indicators_{target_date_text}.json")
    if BACKTEST_MODE_ACTIVE and profile_output_path.exists():
        cached_payload = read_json(profile_output_path)
        if _indicator_cache_matches_current_scoring(cached_payload, config, indicator_mode, enable_relative_strength):
            _ensure_relative_strength_benchmark_cache(config, target_date, indicator_mode, enable_relative_strength)
            write_json(output_path, cached_payload)
            _save_common_processed_cache(config, "indicators", target_date_text, cached_payload)
            _link_profile_processed_cache_to_common(config, "indicators", target_date_text, profile_output_path)
            print(f"{BACKTEST_DAY_LOG_PREFIX} indicators cache hit: {profile_output_path.relative_to(ROOT)}")
            return

    lookback_days = 60 if indicator_mode == "full" else 35
    fetch_dates = previous_business_dates(target_date, lookback_days)
    if BACKTEST_MODE_ACTIVE:
        price_rows = load_cached_price_history(fetch_dates)
        if not price_rows:
            fetch_start = fetch_dates[0] if fetch_dates else target_date
            print(
                f"{BACKTEST_DAY_LOG_PREFIX} warning: cached price history missing for indicators; "
                f"attempting backtest price fetch from {fetch_start.isoformat()} to {target_date_text}."
            )
            ensure_price_history_for_backtest("jquants", fetch_start, target_date, fetch_start)
            price_rows = load_cached_price_history(fetch_dates)
        if not price_rows:
            missing_dates = [
                item.isoformat()
                for item in fetch_dates
                if load_cached_prime_prices(item) is None
            ]
            sample = ", ".join(missing_dates[:5])
            detail = f" missing_dates_sample=[{sample}]" if sample else ""
            raise SystemExit(
                f"No cached price history found for {target_date_text} after fetch attempt. "
                f"Run fetch-period-prices first.{detail}"
            )
    else:
        try:
            provider = JQuantsDataProvider(
                ROOT / ".env",
                timeout_seconds=int(config.get("jquants", {}).get("request_timeout_seconds", 20)),
                plan=_jquants_plan(config),
                requests_per_minute=_jquants_requests_per_minute(config),
                parallel_fetch=_jquants_parallel_fetch(config),
                max_parallel_requests=_jquants_max_parallel_requests(config),
            )
            price_rows = fetch_price_history(
                provider,
                target_date,
                prime_codes,
                lookback_business_days=60,
                rate_limit_per_minute=_jquants_requests_per_minute(config),
                fetch_dates=fetch_dates,
                continue_on_error=True,
                verbose=False,
            )
            _print_fetch_statistics(provider)
        except RuntimeError as exc:
            price_rows = load_cached_price_history(fetch_dates)
            if price_rows:
                print(f"J-Quants indicator source fetch failed; using cached price history. reason={exc}")
            else:
                print(f"J-Quants indicator source fetch failed: {exc}")
                raise SystemExit(1) from exc

    if not price_rows:
        raise SystemExit(f"No price history found for {target_date_text}. The date may be weekend, holiday, or not updated yet.")

    input_days = len({row.get("date") for row in price_rows if row.get("date")})
    target_stocks = len({row.get("code") for row in price_rows if row.get("date") == target_date_text})
    benchmark_payload = _relative_strength_benchmark_payload(price_rows, target_date, fetch_dates, config) if enable_relative_strength else {"benchmark_returns": {}, "benchmark_source": "unavailable"}
    if BACKTEST_MODE_ACTIVE:
        print(f"{BACKTEST_DAY_LOG_PREFIX} indicators mode: {indicator_mode}")
        print(f"{BACKTEST_DAY_LOG_PREFIX} relative_strength indicators: {'enabled' if enable_relative_strength else 'disabled'}")
        if enable_relative_strength:
            print(f"{BACKTEST_DAY_LOG_PREFIX} benchmark_source: {benchmark_payload.get('benchmark_source')}")
        print(f"{BACKTEST_DAY_LOG_PREFIX} indicators input days: {input_days}")
        print(f"{BACKTEST_DAY_LOG_PREFIX} indicators input rows: {len(price_rows)}")
        print(f"{BACKTEST_DAY_LOG_PREFIX} indicators target stocks: {target_stocks}")

    def progress_callback(index: int, total: int, code: str) -> None:
        if BACKTEST_MODE_ACTIVE:
            print(f"{BACKTEST_DAY_LOG_PREFIX} indicators progress: {index}/{total} code={code}")

    try:
        indicators, insufficient_history_count = calculate_indicators(
            price_rows,
            stock_names,
            target_date_text,
            stock_sectors,
            stock_sections=stock_sections,
            indicator_mode=indicator_mode,
            progress_callback=progress_callback if BACKTEST_MODE_ACTIVE else None,
            enable_relative_strength=enable_relative_strength,
            benchmark_returns=benchmark_payload.get("benchmark_returns"),
            benchmark_source=str(benchmark_payload.get("benchmark_source") or "unavailable"),
        )
    except TechnicalIndicatorDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if enable_relative_strength:
        topix_cache_path = benchmark_payload.get("topix_cache_path", "")
        topix_cache_exists = bool(topix_cache_path and Path(str(topix_cache_path)).exists())
        benchmark_provider_called = True
        rs_calculated = benchmark_payload.get("benchmark_source") not in {None, "", "unavailable"}
        for item in indicators:
            item["topix_records_loaded"] = benchmark_payload.get("topix_records_loaded", 0)
            item["topix_api_calls"] = benchmark_payload.get("topix_api_calls", 0)
            item["topix_cache_path"] = topix_cache_path
            item["relative_strength_feature_enabled"] = True
            item["relative_strength_scoring_enabled"] = True
            item["relative_strength_benchmark_provider_called"] = benchmark_provider_called
            item["relative_strength_cache_exists"] = topix_cache_exists
            item["relative_strength_calculated"] = rs_calculated and item.get("relative_strength_5d") is not None
    excluded_count = len(prime_codes) - len(indicators)
    if not indicators:
        if BACKTEST_MODE_ACTIVE:
            print(
                f"{BACKTEST_DAY_LOG_PREFIX} warning: No indicators calculated for {target_date_text}. "
                "Price history may be insufficient or target date may have no data; skipping day."
            )
            payload = {
                "provider": "jquants",
                "date": target_date_text,
                "indicator_mode": indicator_mode,
                "relative_strength_enabled": enable_relative_strength,
                "benchmark_source": benchmark_payload.get("benchmark_source"),
                "lookback_business_days": lookback_days,
                "input_days": input_days,
                "input_rows": len(price_rows),
                "target_stocks": target_stocks,
                "calculated_count": 0,
                "excluded_count": excluded_count,
                "insufficient_history_count": insufficient_history_count,
                "skip_reason": "insufficient_indicator_history",
                "indicators": [],
            }
            write_json(output_path, payload)
            write_json(profile_output_path, payload)
            _save_common_processed_cache(config, "indicators", target_date_text, payload)
            _link_profile_processed_cache_to_common(config, "indicators", target_date_text, profile_output_path)
            return
        raise SystemExit(f"No indicators calculated for {target_date_text}. Price history may be insufficient or target date may have no data.")

    payload = {
        "provider": "jquants",
        "date": target_date_text,
        "indicator_mode": indicator_mode,
        "relative_strength_enabled": enable_relative_strength,
        "benchmark_source": benchmark_payload.get("benchmark_source"),
        "lookback_business_days": lookback_days,
        "input_days": input_days,
        "input_rows": len(price_rows),
        "target_stocks": target_stocks,
        "calculated_count": len(indicators),
        "excluded_count": excluded_count,
        "insufficient_history_count": insufficient_history_count,
        "indicators": indicators,
    }
    write_json(output_path, payload)
    if BACKTEST_MODE_ACTIVE:
        write_json(profile_output_path, payload)
        _save_common_processed_cache(config, "indicators", target_date_text, payload)
        _link_profile_processed_cache_to_common(config, "indicators", target_date_text, profile_output_path)

    print(f"date: {target_date_text}")
    print(f"indicator mode: {indicator_mode}")
    print(f"price rows: {len(price_rows)}")
    print(f"input days: {input_days}")
    print(f"target stocks: {target_stocks}")
    print(f"calculated indicators: {len(indicators)}")
    print(f"excluded insufficient data: {excluded_count}")
    print(f"saved: {output_path.relative_to(ROOT)}")
    if BACKTEST_MODE_ACTIVE:
        print(f"profile cache saved: {profile_output_path.relative_to(ROOT)}")


def run_screen(provider_name: str, target_date_text: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("screen mode currently supports --provider jquants only.")
    try:
        date.fromisoformat(target_date_text)
    except ValueError as exc:
        raise SystemExit("--date must be in YYYY-MM-DD format.") from exc

    indicators_path = ROOT / "data" / "processed" / f"indicators_{target_date_text}.json"
    if not indicators_path.exists():
        raise SystemExit(
            f"Indicator file not found: {indicators_path.relative_to(ROOT)}. "
            f"Run `python src/main.py --mode calculate-indicators --provider jquants --date {target_date_text}` first."
        )

    config = load_config(CONFIG_PATH)
    payload = read_json(indicators_path)
    indicators = payload.get("indicators", [])
    market_filtered = _apply_market_section_filter(indicators, config)
    indicators = market_filtered["allowed"]

    if indicators:
        indicators = enrich_indicators_with_sector_momentum(indicators, target_date_text, provider_name)
        result = screen_candidates(indicators, target_count=50)
    else:
        result = {
            "candidates": [],
            "conditions": [],
            "strict_passed_count": 0,
            "fallback_used": False,
            "fallback_passed_count": 0,
            "excluded_summary": {
                "reason": "empty_indicator_payload",
                "source_indicator_file": str(indicators_path.relative_to(ROOT)),
            },
        }
    candidates = result["candidates"]
    screening_log = {
        "date": target_date_text,
        "provider": provider_name,
        "config_version": config_version_from(config),
        "conditions": result["conditions"],
        "total_target_count": len(indicators),
        "condition_passed_count": result["strict_passed_count"],
        "fallback_used": result["fallback_used"],
        "fallback_passed_count": result["fallback_passed_count"],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "excluded_summary": {
            **result["excluded_summary"],
            "market_filter_excluded": market_filtered["excluded_count"],
        },
        "market_coverage": {
            "allowed_sections": sorted(allowed_market_sections(config)),
            "allow_unknown_market": bool(config.get("market_filter", {}).get("allow_unknown_market", False)),
            "input_counts": market_filtered["input_counts"],
            "candidate_counts": market_section_counts(candidates),
            "market_filter_excluded_count": market_filtered["excluded_count"],
        },
    }

    screening_log.setdefault("profile_id", profile_id_from(config))
    screening_log.setdefault("profile_name", profile_name_from(config))
    screening_path = ROOT / "logs" / "screening" / profile_id_from(config) / f"screening_{target_date_text}.json"
    candidates_path = processed_profile_path(config, f"candidates_{target_date_text}.json")
    write_json(screening_path, screening_log)
    candidate_payload = {
        "date": target_date_text,
        "provider": provider_name,
        "profile_id": profile_id_from(config),
        "profile_name": profile_name_from(config),
        "config_version": config_version_from(config),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "market_coverage": screening_log["market_coverage"],
    }
    write_json(candidates_path, candidate_payload)
    if BACKTEST_MODE_ACTIVE:
        _save_common_processed_cache(config, "candidates", target_date_text, candidate_payload)
        _link_profile_processed_cache_to_common(config, "candidates", target_date_text, candidates_path)
    save_screening_results(config, ROOT, screening_log)

    print(f"target stocks: {len(indicators)}")
    print(f"condition passed: {result['strict_passed_count']}")
    print(f"market_filter_excluded: {market_filtered['excluded_count']}")
    print(f"fallback used: {result['fallback_used']}")
    print(f"candidates: {len(candidates)}")
    if not indicators:
        print("screen warning: indicator payload is empty; saved empty candidates.")
    print(f"screening saved: {screening_path.relative_to(ROOT)}")
    print(f"candidates saved: {candidates_path.relative_to(ROOT)}")


def enrich_indicators_with_sector_momentum(
    indicators: list[dict[str, Any]],
    target_date_text: str,
    provider_name: str,
) -> list[dict[str, Any]]:
    context = load_market_context_for_date(target_date_text, provider_name)
    config = load_config(CONFIG_PATH)
    if not bool(config.get("features", {}).get("sector_analysis", True)):
        return [
            {
                **item,
                "sector_momentum_score": 50,
                "sector_rank": None,
                "sector_comment": "profileでsector_analysisが無効化されています。",
            }
            for item in indicators
        ]
    sector_by_name = {item.get("sector_name"): item for item in context.get("sector_momentum", []) if item.get("sector_name")}
    enriched = []
    for item in indicators:
        sector = sector_by_name.get(item.get("sector_name"))
        if sector:
            enriched.append(
                {
                    **item,
                    "sector_momentum_score": sector.get("sector_momentum_score"),
                    "sector_rank": sector.get("sector_rank"),
                    "sector_comment": sector.get("sector_comment", ""),
                }
            )
        else:
            enriched.append(
                {
                    **item,
                    "sector_momentum_score": item.get("sector_momentum_score", 50),
                    "sector_rank": item.get("sector_rank"),
                    "sector_comment": item.get("sector_comment", "業種モメンタムは中立扱いです。"),
                }
            )
    return enriched


def _apply_market_section_filter(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    allowed = []
    excluded = []
    for row in rows:
        section = market_section_from_row(row)
        enriched = attach_market_section_fields(row, section)
        if market_section_allowed(enriched, config):
            allowed.append(enriched)
        else:
            excluded.append({**enriched, "rejected_reason": "market_filter_excluded"})
    return {
        "allowed": allowed,
        "excluded": excluded,
        "excluded_count": len(excluded),
        "input_counts": market_section_counts(rows),
        "allowed_counts": market_section_counts(allowed),
        "excluded_counts": market_section_counts(excluded),
    }


def run_score(provider_name: str, target_date_text: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("score mode currently supports --provider jquants only.")
    try:
        date.fromisoformat(target_date_text)
    except ValueError as exc:
        raise SystemExit("--date must be in YYYY-MM-DD format.") from exc

    config = load_config(CONFIG_PATH)
    candidates_path = processed_profile_path(config, f"candidates_{target_date_text}.json")
    if not candidates_path.exists():
        raise SystemExit(
            f"Candidate file not found: {candidates_path.relative_to(ROOT)}. "
            f"Run `python src/main.py --mode screen --provider jquants --date {target_date_text}` first."
        )

    payload = read_json(candidates_path)
    candidates = payload.get("candidates", [])
    market_context = load_market_context_for_date(target_date_text, provider_name)
    target_date = date.fromisoformat(target_date_text)
    if candidates:
        earnings_payload = _load_earnings_calendar_for_date(target_date, config)
        if config.get("earnings_filter", {}).get("enabled") and earnings_payload["metadata"].get("filter_available", False):
            config["_earnings_calendar_records"] = earnings_payload["records"]
        config["_earnings_calendar_metadata"] = earnings_payload["metadata"]
        investor_context_payload = _load_investor_context_for_date(target_date, config)
        config["_investor_context"] = investor_context_payload["context"]
        config["_investor_context_metadata"] = investor_context_payload["metadata"]
        scoring_log = score_real_candidates(
            candidates,
            target_date_text,
            config,
            provider_name,
            market_context=market_context,
        )
    else:
        earnings_payload = {"records": [], "metadata": {"filter_available": False, "skip_reason": "empty_candidates"}}
        investor_context_payload = {"context": {}, "metadata": {"available": False, "skip_reason": "empty_candidates"}}
        scoring_log = {
            "date": target_date_text,
            "provider": provider_name,
            "scores": [],
            "selected": [],
            "candidate_count": 0,
            "selected_count": 0,
            "selection_config": {},
            "market_context": market_context,
            "market_filter": {},
            "skip_reason": "empty_candidates",
        }
    scoring_log["earnings_calendar"] = earnings_payload.get("metadata", {})
    scoring_log["investor_context"] = investor_context_payload.get("metadata", {})
    attach_config_version(scoring_log, config)
    ai_decision_log = run_ai_decision_if_enabled(scoring_log, config, target_date_text)
    scores = scoring_log["scores"]
    selected = scoring_log["selected"]
    highest_score = max((item["total_score"] for item in scores), default=0)

    scoring_log.setdefault("profile_id", profile_id_from(config))
    scoring_log.setdefault("profile_name", profile_name_from(config))
    scoring_path = ROOT / "logs" / "scoring" / profile_id_from(config) / f"scoring_{target_date_text}.json"
    scored_candidates_path = processed_profile_path(config, f"scored_candidates_{target_date_text}.json")
    storage_scoring_log = _scoring_log_for_storage(scoring_log, config)
    write_json(scoring_path, storage_scoring_log)
    write_json(
        scored_candidates_path,
        {
            "date": target_date_text,
            "provider": provider_name,
            "profile_id": profile_id_from(config),
            "profile_name": profile_name_from(config),
            "config_version": config_version_from(config),
            "candidate_count": len(candidates),
            "scored_count": len(scores),
            "selected_count": len(selected),
            "earnings_calendar": earnings_payload.get("metadata", {}),
            "investor_context": investor_context_payload.get("metadata", {}),
            "selection_config": scoring_log.get("selection_config", {}),
            "market_context": scoring_log.get("market_context", {}),
            "market_filter": scoring_log.get("market_filter", {}),
            "market_coverage": {
                "candidate_count": market_section_counts(scores),
                "selected_count": market_section_counts(selected),
                "market_filter_excluded_count": (scoring_log.get("market_filter", {}) or {}).get("market_filter_excluded_count", 0),
            },
            "ai_decision": scoring_log.get("ai_decision", {}),
            "scores": _scores_for_storage(scores, config),
        },
    )
    save_scoring_results(config, ROOT, storage_scoring_log)
    if ai_decision_log:
        save_ai_decision(config, ROOT, ai_decision_log)
    write_daily_ai_dataset(config, target_date_text)

    print(f"candidates: {len(candidates)}")
    print(f"scored: {len(scores)}")
    print(f"selected: {len(selected)}")
    print(f"highest_score: {highest_score}")
    print(f"storage_mode: {_storage_save_mode(config)}")
    if len(storage_scoring_log.get("scores", [])) != len(scores):
        print(f"scoring storage: saved {len(storage_scoring_log.get('scores', []))}/{len(scores)} scores (rejected candidate detail disabled)")
    if ai_decision_log:
        print(f"ai_decision: {ai_decision_log['provider']} fallback={ai_decision_log['fallback_used']}")
    print(f"scoring saved: {scoring_path.relative_to(ROOT)}")
    print(f"scored candidates saved: {scored_candidates_path.relative_to(ROOT)}")


def run_ai_decision_if_enabled(scoring_log: dict[str, Any], config: dict[str, Any], target_date_text: str) -> dict[str, Any] | None:
    ai_config = config.get("ai_decision", {})
    if BACKTEST_MODE_ACTIVE and _backtest_disable_openai(config):
        scoring_log["ai_decision"] = {
            "enabled": False,
            "provider": "rule_based",
            "fallback_used": False,
            "reason": "backtest.disable_openai is true",
        }
        return None
    if not ai_config.get("enabled", False):
        scoring_log["ai_decision"] = {
            "enabled": False,
            "provider": ai_config.get("provider", "openai"),
            "fallback_used": False,
        }
        return None

    provider = build_ai_decision_provider(config, ROOT)
    config_version = scoring_log.get("config_version") or config_version_from(config)
    decision_result = provider.decide(
        target_date=target_date_text,
        config_version=config_version,
        market_context=load_market_context_for_date(target_date_text, scoring_log.get("source_provider", "jquants")),
        portfolio_summary=_latest_portfolio_summary_for_ai_decision(),
        scored_candidates=scoring_log.get("scores", []),
    )
    apply_ai_decision(scoring_log, decision_result, config)
    ai_log = build_ai_decision_log(
        target_date=target_date_text,
        config_version=config_version,
        decision_result=decision_result,
        candidates_count=len(scoring_log.get("scores", [])),
        selected_count=len(scoring_log.get("selected", [])),
    )
    ai_log["profile_id"] = profile_id_from(config)
    ai_log["profile_name"] = profile_name_from(config)
    ai_log_path = ROOT / "logs" / "ai_decision" / profile_id_from(config) / f"ai_decision_{target_date_text}.json"
    write_json(ai_log_path, ai_log)
    return ai_log


def run_market_context(provider_name: str, target_date_text: str) -> dict[str, Any]:
    if provider_name != "jquants":
        raise SystemExit("market_context currently supports --provider jquants only.")
    config = load_config(CONFIG_PATH)
    try:
        context = build_market_context(target_date_text, provider_name, ROOT)
    except Exception as exc:
        context = neutral_market_context(target_date_text, provider_name, f"market_context生成に失敗したためneutralとして扱います: {exc}")
    context.setdefault("config_version", config_version_from(config))
    processed_path = ROOT / "data" / "processed" / f"market_context_{target_date_text}.json"
    context["profile_id"] = profile_id_from(config)
    context["profile_name"] = profile_name_from(config)
    log_path = ROOT / "logs" / "market_context" / profile_id_from(config) / f"market_context_{target_date_text}.json"
    write_json(processed_path, context)
    write_json(log_path, context)
    save_market_context(config, ROOT, context)
    return context


def ensure_market_context(provider_name: str, target_date_text: str) -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    if not bool(config.get("features", {}).get("market_context", True)):
        return neutral_market_context(target_date_text, provider_name, "profileでmarket_contextが無効化されています。")
    processed_path = ROOT / "data" / "processed" / f"market_context_{target_date_text}.json"
    if processed_path.exists():
        context = read_json(processed_path)
        config = load_config(CONFIG_PATH)
        save_market_context(config, ROOT, context)
        return context
    return run_market_context(provider_name, target_date_text)


def load_market_context_for_date(target_date_text: str, provider_name: str) -> dict[str, Any]:
    path = ROOT / "data" / "processed" / f"market_context_{target_date_text}.json"
    if path.exists():
        return read_json(path)
    return neutral_market_context(target_date_text, provider_name, "market_context未生成のためneutralとして扱います。")


def _latest_portfolio_summary_for_ai_decision() -> dict[str, Any]:
    portfolio_dir = ROOT / "logs" / "portfolio" / ACTIVE_PROFILE_ID
    if not portfolio_dir.exists():
        return {}
    latest = sorted(portfolio_dir.glob("portfolio_*.json"))
    if not latest:
        return {}
    try:
        payload = read_json(latest[-1])
    except Exception:
        return {}
    return {
        "date": payload.get("date"),
        "cash": payload.get("cash"),
        "total_assets": payload.get("total_assets"),
        "open_positions_count": payload.get("open_positions_count"),
        "closed_trades_count": payload.get("closed_trades_count"),
        "max_drawdown": payload.get("max_drawdown"),
    }


def fetch_candidate_news(candidates: list[dict[str, Any]], target_date_text: str, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if BACKTEST_MODE_ACTIVE and _backtest_disable_news(config):
        print(f"news fetch skipped for backtest: candidates={len(candidates)}")
        return {}
    if not bool(config.get("features", {}).get("news", config.get("news", {}).get("enabled", True))):
        return {}
    provider = build_news_provider(config, ROOT)
    news_by_code = {}
    for candidate in candidates:
        code = candidate["code"]
        try:
            news_by_code[code] = provider.get_news(code, candidate["name"], target_date_text)
        except Exception as exc:
            news_by_code[code] = {
                "code": code,
                "name": candidate["name"],
                "date": target_date_text,
                "provider": config.get("news", {}).get("provider", ""),
                "query": "",
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "articles": [],
                "limitation": f"ニュース取得に失敗: {exc}",
            }
    return news_by_code


def run_trade(provider_name: str, target_date_text: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("trade mode currently supports --provider jquants only.")
    try:
        date.fromisoformat(target_date_text)
    except ValueError as exc:
        raise SystemExit("--date must be in YYYY-MM-DD format.") from exc

    config = load_config(CONFIG_PATH)
    scored_path = processed_profile_path(config, f"scored_candidates_{target_date_text}.json")
    if not scored_path.exists():
        raise SystemExit(
            f"Scored candidates file not found: {scored_path.relative_to(ROOT)}. "
            f"Run `python src/main.py --mode score --provider jquants --date {target_date_text}` first."
        )

    scored_payload = read_json(scored_path)
    scored_candidates = scored_payload.get("scores", [])
    if not scored_candidates:
        raise SystemExit(f"No scored candidates found in {scored_path.relative_to(ROOT)}.")

    state_path = ROOT / "logs" / "portfolio" / profile_id_from(config) / "state.json"
    state = read_json(state_path) if state_path.exists() else initial_live_paper_state(config)
    trades_path = ROOT / "logs" / "trades" / profile_id_from(config) / f"trades_{target_date_text}.json"
    portfolio_path = ROOT / "logs" / "portfolio" / profile_id_from(config) / f"portfolio_{target_date_text}.json"

    if trades_path.exists() and portfolio_path.exists():
        portfolio_summary = read_json(portfolio_path)
        portfolio_summary.setdefault("config_version", config_version_from(config))
        trades = read_json(trades_path).get("trades", [])
        for trade in trades:
            trade.setdefault("config_version", config_version_from(config))
        if not _round_lot_cache_is_stale(config, state, trades):
            storage_trades = _trades_for_storage(trades, config)
            write_json(trades_path, {"date": target_date_text, "provider": provider_name, "profile_id": profile_id_from(config), "profile_name": profile_name_from(config), "config_version": config_version_from(config), "storage_mode": _storage_save_mode(config), "trades": storage_trades})
            write_json(portfolio_path, portfolio_summary)
            save_portfolio_snapshot(config, ROOT, portfolio_summary)
            save_trades(config, ROOT, target_date_text, storage_trades)
            save_pending_orders(config, ROOT, state.get("pending_orders", []))
            selected_count = scored_payload.get("selected_count", sum(1 for item in scored_candidates if item.get("selected")))
            print(f"date: {target_date_text}")
            print(f"selected candidates: {selected_count}")
            print(f"trade logs: {len(trades)}")
            print(f"open positions: {portfolio_summary['open_positions_count']}")
            print(f"total assets: {portfolio_summary['total_assets']}")
            print(f"trades saved: {trades_path.relative_to(ROOT)}")
            print(f"portfolio saved: {portfolio_path.relative_to(ROOT)}")
            print("status: already processed")
            return
        state = initial_live_paper_state(config)

    scored_candidates = enrich_candidates_with_position_prices(scored_candidates, state, target_date_text)
    updated_state, portfolio_summary, trades = execute_real_data_paper_trade(scored_candidates, state, config, target_date_text)
    attach_config_version(portfolio_summary, config)
    for trade in trades:
        trade.setdefault("config_version", config_version_from(config))

    write_json(state_path, updated_state)
    storage_trades = _trades_for_storage(trades, config)
    write_json(
        trades_path,
        {
            "date": target_date_text,
            "provider": provider_name,
            "config_version": config_version_from(config),
            "storage_mode": _storage_save_mode(config),
            "trades": storage_trades,
        },
    )
    write_json(portfolio_path, portfolio_summary)
    save_portfolio_snapshot(config, ROOT, portfolio_summary)
    save_trades(config, ROOT, target_date_text, storage_trades)
    save_pending_orders(config, ROOT, updated_state.get("pending_orders", []))

    selected_count = sum(1 for item in scored_candidates if item.get("selected"))
    print(f"date: {target_date_text}")
    print(f"selected candidates: {selected_count}")
    print(f"trade logs: {len(trades)}")
    print(f"open positions: {portfolio_summary['open_positions_count']}")
    print(f"total assets: {portfolio_summary['total_assets']}")
    print(f"trades saved: {trades_path.relative_to(ROOT)}")
    print(f"portfolio saved: {portfolio_path.relative_to(ROOT)}")
    print(f"state saved: {state_path.relative_to(ROOT)}")


def run_preview_orders(provider_name: str, target_date_text: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("preview-orders mode currently supports --provider jquants only.")
    _validate_date(target_date_text)
    config = load_config(CONFIG_PATH)
    scored_path = processed_profile_path(config, f"scored_candidates_{target_date_text}.json")
    if not scored_path.exists():
        raise SystemExit(f"Scored candidates not found: {scored_path.relative_to(ROOT)}. Run score mode first.")

    scored_payload = read_json(scored_path)
    state_path = ROOT / "logs" / "portfolio" / profile_id_from(config) / "state.json"
    state = read_json(state_path) if state_path.exists() else initial_live_paper_state(config)
    scored_candidates = enrich_candidates_with_position_prices(scored_payload.get("scores", []), state, target_date_text)
    preview = build_order_preview(scored_candidates, state, config, target_date_text)

    output_dir = ROOT / "reports" / profile_id_from(config) / "order_previews"
    json_path = output_dir / f"order_preview_{target_date_text}.json"
    markdown_path = output_dir / f"order_preview_{target_date_text}.md"
    write_json(json_path, preview)
    write_text(markdown_path, render_order_preview_markdown(preview))
    print(render_order_preview_console(preview))
    print(f"json: {json_path.relative_to(ROOT)}")
    print(f"markdown: {markdown_path.relative_to(ROOT)}")


def run_publish_article(target_date_text: str, note_url: str) -> None:
    _validate_date(target_date_text)
    config = load_config(CONFIG_PATH)
    migrate_report_articles_layout(config)
    draft_path = _find_article_draft(target_date_text)
    if draft_path is None:
        raise SystemExit(f"Draft article not found for {target_date_text}. Run run-daily first.")

    published_at = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")
    body = draft_path.read_text(encoding="utf-8")
    published_body = upsert_article_front_matter(
        body,
        {
            "date": target_date_text,
            "status": "published",
            "note_url": note_url,
            "published_at": published_at,
            "config_version": config_version_from(config),
        },
    )
    published_path = _published_article_path(config, target_date_text, profile_id_from(config))
    write_text(published_path, published_body)
    keep_draft = bool(config.get("articles", {}).get("keep_draft_after_publish", False))
    if not keep_draft:
        draft_path.unlink()

    save_article(
        config,
        ROOT,
        {
            "date": target_date_text,
            "day": _day_from_article_body(published_body),
            "title": _extract_markdown_title(published_body),
            "status": "published",
            "path": str(published_path.relative_to(ROOT)),
            "note_url": note_url,
            "published_at": published_at,
            "config_version": config_version_from(config),
            "body": published_body,
        },
    )
    _update_article_index(published_path, target_date_text, "published", config)
    print(f"published article path: {published_path.relative_to(ROOT)}")
    print(f"note_url: {note_url}")
    print(f"draft removed: {str(not keep_draft).lower()}")


def _find_article_draft(target_date_text: str) -> Path | None:
    config = load_config(CONFIG_PATH)
    profile_id = profile_id_from(config)
    run_date = date.fromisoformat(target_date_text)
    compact = target_date_text.replace("-", "")
    candidates = [
        _daily_article_path(config, target_date_text, profile_id),
        ROOT / "reports" / "articles" / "daily" / run_date.strftime("%Y") / run_date.strftime("%m") / profile_id / f"{compact}_day.md",
        ROOT / "articles" / "drafts" / profile_id / f"day_{target_date_text}.md",
        ROOT / "articles" / "drafts" / f"day_{target_date_text}.md",
        ROOT / "articles" / "drafts" / profile_id / run_date.strftime("%Y-%m") / f"{compact}_day_001.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _daily_article_path(config: dict[str, Any], target_date_text: str, profile_id: str) -> Path:
    target_date = date.fromisoformat(target_date_text)
    return (
        ROOT
        / "reports"
        / "articles"
        / "daily"
        / target_date.strftime("%Y")
        / target_date.strftime("%m")
        / profile_id
        / f"day_{target_date_text}.md"
    )


def _backtest_article_path(config: dict[str, Any], target_date_text: str, profile_id: str) -> Path:
    target_date = date.fromisoformat(target_date_text)
    return (
        ROOT
        / "reports"
        / "articles"
        / "backtests"
        / target_date.strftime("%Y")
        / target_date.strftime("%m")
        / profile_id
        / f"day_{target_date_text}.md"
    )


def _published_article_path(config: dict[str, Any], target_date_text: str, profile_id: str) -> Path:
    target_date = date.fromisoformat(target_date_text)
    return (
        ROOT
        / "reports"
        / "articles"
        / "published"
        / target_date.strftime("%Y")
        / target_date.strftime("%m")
        / profile_id
        / f"day_{target_date_text}.md"
    )


def _paper_report_path(config: dict[str, Any], target_date_text: str) -> Path:
    target_date = date.fromisoformat(target_date_text)
    return ROOT / "reports" / "paper" / profile_id_from(config) / target_date.strftime("%Y-%m") / f"day_{target_date_text}.md"


def _update_article_index(path: Path, target_date_text: str, status: str, config: dict[str, Any]) -> None:
    index_path = ROOT / "reports" / "articles" / "index.json"
    existing = read_json(index_path) if index_path.exists() else {"articles": []}
    articles = [
        item
        for item in existing.get("articles", [])
        if not (item.get("date") == target_date_text and item.get("profile_id") == profile_id_from(config) and item.get("status") == status)
    ]
    articles.append(
        {
            "date": target_date_text,
            "profile_id": profile_id_from(config),
            "profile_name": profile_name_from(config),
            "status": status,
            "path": str(path.relative_to(ROOT)),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    latest_item = articles[-1]
    write_json(index_path, {"articles": sorted(articles, key=lambda item: (item.get("date", ""), item.get("profile_id", ""), item.get("status", "")))})
    latest_json = ROOT / "reports" / "articles" / "latest.json"
    write_json(latest_json, latest_item)
    if path.exists():
        write_text(ROOT / "reports" / "articles" / "latest.md", path.read_text(encoding="utf-8"))


def _article_index_count(profile_id: str, start_date_text: str, end_date_text: str) -> int:
    index_path = ROOT / "reports" / "articles" / "index.json"
    if not index_path.exists():
        return 0
    payload = read_json(index_path)
    return sum(
        1
        for item in payload.get("articles", [])
        if item.get("profile_id") == profile_id
        and start_date_text <= str(item.get("date", "")) <= end_date_text
        and item.get("status") == "draft"
    )


def upsert_article_front_matter(body: str, metadata: dict[str, str]) -> str:
    lines = body.splitlines()
    content_lines = lines
    existing: dict[str, str] = {}
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                content_lines = lines[index + 1 :]
                break
            if ":" in line:
                key, value = line.split(":", 1)
                existing[key.strip()] = value.strip()
    existing.update(metadata)
    front_matter = ["---"]
    for key in ["date", "status", "note_url", "published_at", "config_version"]:
        if key in existing:
            front_matter.append(f"{key}: {existing[key]}")
    front_matter.append("---")
    return "\n".join(front_matter + [""] + content_lines) + "\n"


def _day_from_article_body(body: str) -> int | None:
    for line in body.splitlines():
        if "Day " not in line:
            continue
        marker = line.split("Day ", 1)[1]
        digits = ""
        for char in marker:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            return int(digits)
    return None


def build_order_preview(
    scored_candidates: list[dict[str, Any]],
    state: dict[str, Any],
    config: dict[str, Any],
    target_date_text: str,
) -> dict[str, Any]:
    initial_cash = float(config["portfolio"]["initial_cash"])
    max_positions = int(config["portfolio"]["max_positions"])
    allocation_limit = initial_cash * float(config["portfolio"]["max_allocation_per_symbol"])
    take_profit_pct = float(config["risk"]["take_profit_pct"])
    stop_loss_pct = float(config["risk"]["stop_loss_pct"])
    max_holding_days = int(config["risk"]["max_holding_business_days"])
    cash = float(state.get("cash", initial_cash))
    price_by_code = {item["code"]: item for item in scored_candidates}
    today_orders: list[dict[str, Any]] = []
    sell_candidates = []
    buy_candidates = []
    skipped = []
    safety_results = []
    positions = _preview_position_rows(state.get("positions", []), price_by_code)

    for position in state.get("positions", []):
        market = price_by_code.get(position.get("code"), {})
        current_price = float(market.get("close") or position.get("current_price") or position.get("entry_price"))
        entry_price = float(position.get("entry_price", current_price))
        shares = int(position.get("shares") or position.get("quantity") or 0)
        holding_days = int(position.get("holding_days") or position.get("holding_business_days") or 0) + 1
        profit_rate = (current_price - entry_price) / entry_price if entry_price else 0
        reason = _preview_exit_reason(profit_rate, take_profit_pct, stop_loss_pct, holding_days, max_holding_days)
        if not reason:
            continue
        order = {
            "action": "SELL",
            "code": position.get("code"),
            "name": position.get("name"),
            "shares": shares,
            "exit_price": current_price,
            "amount": round(shares * current_price, 2),
            "reason": reason,
            "sell_reason": reason,
            "profit_rate": round(profit_rate, 4),
            "safety_checked": True,
            "live_trading": False,
            "broker_provider": "paper",
            "order_status": "PREVIEW",
        }
        validation = can_trade(order, _preview_safety_portfolio(state, today_orders, order), config)
        safety_results.append(_preview_safety_result(order, validation))
        if validation["allowed"]:
            sell_candidates.append(order)
            today_orders.append(order)
        else:
            skipped.append(_preview_skipped(order, validation))

    held_codes = {position.get("code") for position in state.get("positions", [])}
    open_after_sells = len(state.get("positions", [])) - len(sell_candidates)
    selected = _preview_selected_candidates(scored_candidates, config)
    for item in selected:
        if open_after_sells + len(buy_candidates) >= max_positions:
            skipped.append({"code": item.get("code"), "name": item.get("name"), "reason": "最大保有銘柄数に到達するため見送り"})
            continue
        if item.get("code") in held_codes:
            skipped.append({"code": item.get("code"), "name": item.get("name"), "reason": "すでに保有中のため買い増ししない"})
            continue
        price = float(item.get("close") or item.get("close_price") or 0)
        allocation = min(allocation_limit, cash - sum(order["estimated_amount"] for order in buy_candidates))
        shares, reason = _preview_buy_shares(price, allocation, config)
        if shares <= 0:
            skipped.append({"code": item.get("code"), "name": item.get("name"), "reason": reason})
            continue
        amount = round(shares * price, 2)
        order = {
            "action": "BUY",
            "code": item.get("code"),
            "name": item.get("name"),
            "shares": shares,
            "entry_price": price,
            "amount": amount,
            "score": item.get("total_score"),
            "reason": item.get("selection_reason") or item.get("selected_reason") or item.get("reason", ""),
            "buy_reason": item.get("selection_reason") or item.get("selected_reason") or item.get("reason", ""),
            "excluded_reason": item.get("rejected_reason") or "",
            "sector_name": item.get("sector_name"),
            "market_regime": item.get("market_regime"),
            "candlestick_signals": item.get("candlestick_signals", []),
            "safety_checked": True,
            "live_trading": False,
            "broker_provider": "paper",
            "order_status": "PREVIEW",
        }
        validation = can_trade(order, _preview_safety_portfolio(state, today_orders, order), config)
        safety_results.append(_preview_safety_result(order, validation))
        if validation["allowed"]:
            buy_candidates.append(
                {
                    "code": order["code"],
                    "name": order["name"],
                    "shares": shares,
                    "estimated_price": price,
                    "estimated_amount": amount,
                    "score": item.get("total_score"),
                    "reason": order["reason"],
                    "buy_reason": order["buy_reason"],
                    "sector_name": item.get("sector_name"),
                    "market_regime": item.get("market_regime"),
                    "candlestick_signals": item.get("candlestick_signals", []),
                }
            )
            today_orders.append(order)
        else:
            skipped.append(_preview_skipped(order, validation))

    skipped.extend(_preview_rejected_candidate_rows(scored_candidates, held_codes))
    live_enabled = bool(config.get("broker", {}).get("live_trading_enabled", False) and config.get("safety", {}).get("allow_live_trading", False))
    broker_provider = config.get("broker", {}).get("provider", "paper")
    preview_orders = _preview_orders(sell_candidates, buy_candidates)
    risk_check_summary = _preview_risk_check_summary(safety_results)
    return {
        "date": target_date_text,
        "mode": "MANUAL_APPROVAL_PREVIEW",
        "provider": "jquants",
        "broker_provider": "paper",
        "broker_candidates": ["paper"],
        "order_submission_enabled": False,
        "manual_approval_required": True,
        "manual_approval_flow": {
            "status": "PENDING_MANUAL_APPROVAL",
            "order_submission_enabled": False,
            "broker_provider": "paper",
            "live_trading_enabled": False,
            "message": "preview_orders を確認するだけです。このモードでは発注しません。",
        },
        "positions": positions,
        "sell_candidates": [
            {
                "code": item["code"],
                "name": item["name"],
                "shares": item["shares"],
                "estimated_price": item["exit_price"],
                "estimated_amount": item["amount"],
                "reason": item["reason"],
                "sell_reason": item["sell_reason"],
                "profit_rate": item["profit_rate"],
            }
            for item in sell_candidates
        ],
        "buy_candidates": buy_candidates,
        "preview_orders": preview_orders,
        "skipped": skipped,
        "safety": safety_results,
        "risk_check_summary": risk_check_summary,
        "summary": {
            "buy_count": len(buy_candidates),
            "sell_count": len(sell_candidates),
            "position_count": len(positions),
            "unrealized_pnl": round(sum(float(item.get("unrealized_pnl") or 0.0) for item in positions), 2),
            "estimated_buy_amount": round(sum(item["estimated_amount"] for item in buy_candidates), 2),
            "estimated_sell_amount": round(sum(item["amount"] for item in sell_candidates), 2),
            "risk_check_passed": risk_check_summary["rejected_count"] == 0,
            "live_trading_enabled": False,
            "broker_provider": "paper",
            "configured_broker_provider": broker_provider,
            "configured_live_trading_enabled": live_enabled,
        },
    }


def _preview_position_rows(positions: list[dict[str, Any]], price_by_code: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for position in positions:
        market = price_by_code.get(position.get("code"), {})
        current_price = float(market.get("close") or position.get("current_price") or position.get("entry_price") or 0)
        entry_price = float(position.get("entry_price") or current_price or 0)
        shares = int(position.get("shares") or position.get("quantity") or 0)
        market_value = round(current_price * shares, 2)
        unrealized_pnl = round((current_price - entry_price) * shares, 2)
        unrealized_pnl_rate = round((current_price - entry_price) / entry_price, 4) if entry_price else None
        rows.append(
            {
                "code": position.get("code"),
                "name": position.get("name"),
                "shares": shares,
                "entry_price": entry_price,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_rate": unrealized_pnl_rate,
                "holding_days": int(position.get("holding_days") or position.get("holding_business_days") or 0),
                "sector_name": position.get("sector_name") or market.get("sector_name"),
            }
        )
    return rows


def _preview_rejected_candidate_rows(scored_candidates: list[dict[str, Any]], held_codes: set[Any]) -> list[dict[str, Any]]:
    rows = []
    for item in scored_candidates:
        if item.get("selected") or item.get("code") in held_codes:
            continue
        reason = item.get("rejected_reason") or item.get("reason") or item.get("selection_reason") or "selected=false のため買い候補から除外"
        rows.append(
            {
                "code": item.get("code"),
                "name": item.get("name"),
                "reason": reason,
                "score": item.get("total_score"),
                "market_regime": item.get("market_regime"),
                "sector_name": item.get("sector_name"),
            }
        )
    return rows


def _preview_orders(sell_candidates: list[dict[str, Any]], buy_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    orders = []
    for item in sell_candidates:
        orders.append(
            {
                "approval_status": "PENDING_MANUAL_APPROVAL",
                "order_status": "PREVIEW",
                "action": "SELL",
                "code": item.get("code"),
                "name": item.get("name"),
                "shares": item.get("shares"),
                "estimated_price": item.get("exit_price"),
                "estimated_amount": item.get("amount"),
                "reason": item.get("sell_reason") or item.get("reason"),
                "broker_provider": "paper",
                "live_trading": False,
            }
        )
    for item in buy_candidates:
        orders.append(
            {
                "approval_status": "PENDING_MANUAL_APPROVAL",
                "order_status": "PREVIEW",
                "action": "BUY",
                "code": item.get("code"),
                "name": item.get("name"),
                "shares": item.get("shares"),
                "estimated_price": item.get("estimated_price"),
                "estimated_amount": item.get("estimated_amount"),
                "reason": item.get("buy_reason") or item.get("reason"),
                "broker_provider": "paper",
                "live_trading": False,
            }
        )
    return orders


def _preview_risk_check_summary(safety_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "checked_count": len(safety_results),
        "passed_count": sum(1 for item in safety_results if item.get("passed")),
        "rejected_count": sum(1 for item in safety_results if not item.get("passed")),
    }


def _preview_selected_candidates(scored_candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    selection = config.get("selection", {})
    min_score = float(selection.get("min_score", 70))
    top_pick_min = float(selection.get("top_pick_min_score", 65))
    min_confidence = float(selection.get("min_confidence", 0.7))
    max_selected = int(selection.get("max_selected", config["portfolio"].get("max_positions", 5)))
    selected = [
        item
        for item in scored_candidates
        if item.get("selected") and float(item.get("total_score", 0)) >= min_score and float(item.get("confidence", 0)) >= min_confidence
    ]
    if not selected and selection.get("allow_top_pick_when_no_selection", True):
        selected = [
            item
            for item in scored_candidates
            if item.get("selected") and float(item.get("total_score", 0)) >= top_pick_min and float(item.get("confidence", 0)) >= min_confidence
        ][:1]
    selected.sort(key=lambda item: (float(item.get("total_score", 0)), float(item.get("confidence", 0))), reverse=True)
    return selected[:max_selected]


def _preview_buy_shares(price: float, allocation: float, config: dict[str, Any]) -> tuple[int, str]:
    if price <= 0 or allocation <= 0:
        return 0, "買付余力が不足しているため見送り"
    if bool(config.get("trading", {}).get("use_round_lot", False)):
        lot_size = int(config.get("trading", {}).get("round_lot_size", 100))
        minimum = price * lot_size
        if minimum > allocation:
            return 0, f"{lot_size}株購入に必要な金額が1銘柄上限を超えるため見送り"
        return int(allocation // minimum) * lot_size, ""
    shares = int(allocation // price)
    if shares <= 0:
        return 0, "買付可能株数が0のため見送り"
    return shares, ""


def _preview_exit_reason(profit_rate: float, take_profit_pct: float, stop_loss_pct: float, holding_days: int, max_holding_days: int) -> str:
    if profit_rate >= take_profit_pct:
        return "利確"
    if profit_rate <= stop_loss_pct:
        return "損切り"
    if holding_days >= max_holding_days:
        return "最大保有期間到達"
    return ""


def _preview_safety_portfolio(state: dict[str, Any], today_orders: list[dict[str, Any]], pending_order: dict[str, Any]) -> dict[str, Any]:
    return {
        "cash": state.get("cash"),
        "total_assets": state.get("total_assets"),
        "max_drawdown": state.get("max_drawdown", 0),
        "daily_profit_rate": state.get("daily_profit_rate", 0),
        "today_orders": today_orders,
        "_pending_order": pending_order,
    }


def _preview_safety_result(order: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": order.get("code"),
        "name": order.get("name"),
        "action": order.get("action"),
        "passed": validation["allowed"],
        "rule": validation.get("safety_rule") or "passed",
        "reason": validation.get("reason"),
    }


def _preview_skipped(order: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": order.get("code"),
        "name": order.get("name"),
        "reason": validation.get("reason", "セーフティチェックにより見送り"),
    }


def render_order_preview_console(preview: dict[str, Any]) -> str:
    return render_order_preview_markdown(preview)


def render_order_preview_markdown(preview: dict[str, Any]) -> str:
    lines = [
        "# Daily Paper Report",
        "",
        f"Date: {preview['date']}",
        f"Mode: {preview['mode']}",
        f"Broker: {preview.get('broker_provider', 'paper')}",
        f"Order Submission Enabled: {str(preview.get('order_submission_enabled', False)).lower()}",
        f"Manual Approval Required: {str(preview.get('manual_approval_required', True)).lower()}",
        "",
        "## Manual Approval Flow",
        "",
        *_manual_approval_flow_lines(preview.get("manual_approval_flow", {})),
        "",
        "## preview_orders",
        "",
        *_preview_order_approval_lines(preview.get("preview_orders", [])),
        "",
        "## 今日の買い候補",
        "",
    ]
    lines.extend(_preview_order_lines(preview["buy_candidates"], is_sell=False))
    lines.extend(["", "## 今日の売り候補", ""])
    lines.extend(_preview_order_lines(preview["sell_candidates"], is_sell=True))
    lines.extend(["", "## 保有中ポジション", ""])
    lines.extend(_preview_position_lines(preview.get("positions", [])))
    lines.extend(["", "## 含み損益", ""])
    lines.extend(_preview_unrealized_lines(preview.get("summary", {})))
    lines.extend(["", "## 除外理由", ""])
    lines.extend(_preview_skipped_lines(preview["skipped"]))
    lines.extend(["", "## リスクチェック結果", ""])
    lines.extend(_preview_risk_check_summary_lines(preview.get("risk_check_summary", {})))
    lines.extend(_preview_safety_lines(preview["safety"]))
    summary = preview["summary"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- buy_count: {summary['buy_count']}",
            f"- sell_count: {summary['sell_count']}",
            f"- position_count: {summary.get('position_count', 0)}",
            f"- unrealized_pnl: {summary.get('unrealized_pnl', 0):,.0f}",
            f"- estimated_buy_amount: {summary['estimated_buy_amount']:,.0f}",
            f"- estimated_sell_amount: {summary['estimated_sell_amount']:,.0f}",
            f"- risk_check_passed: {str(summary.get('risk_check_passed', True)).lower()}",
            f"- broker_provider: {summary.get('broker_provider', 'paper')}",
            f"- live_trading_enabled: {str(summary['live_trading_enabled']).lower()}",
        ]
    )
    return "\n".join(lines)


def _manual_approval_flow_lines(flow: dict[str, Any]) -> list[str]:
    if not flow:
        return [
            "- status: PENDING_MANUAL_APPROVAL",
            "- order_submission_enabled: false",
            "- broker_provider: paper",
            "- live_trading_enabled: false",
        ]
    return [
        f"- status: {flow.get('status', 'PENDING_MANUAL_APPROVAL')}",
        f"- order_submission_enabled: {str(flow.get('order_submission_enabled', False)).lower()}",
        f"- broker_provider: {flow.get('broker_provider', 'paper')}",
        f"- live_trading_enabled: {str(flow.get('live_trading_enabled', False)).lower()}",
        f"- message: {flow.get('message', '')}",
    ]


def _preview_order_approval_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- preview order なし"]
    return [
        (
            f"- [{item.get('approval_status')}] {item.get('action')} {item.get('code')} {item.get('name')}: "
            f"shares={item.get('shares')}, estimated_price={float(item.get('estimated_price') or 0):,.0f}, "
            f"estimated_amount={float(item.get('estimated_amount') or 0):,.0f}, "
            f"reason={item.get('reason')}, broker={item.get('broker_provider', 'paper')}, "
            f"live_trading={str(item.get('live_trading', False)).lower()}"
        )
        for item in items
    ]


def _preview_order_lines(items: list[dict[str, Any]], is_sell: bool) -> list[str]:
    if not items:
        return ["- None"]
    lines = []
    for item in items:
        if is_sell:
            lines.append(
                f"- {item['code']} {item['name']}: shares={item['shares']}, "
                f"estimated_price={item['estimated_price']:,.0f}, estimated_amount={item['estimated_amount']:,.0f}, "
                f"売り理由={item.get('sell_reason', item['reason'])}, profit_rate={item['profit_rate']:.2%}"
            )
        else:
            lines.append(
                f"- {item['code']} {item['name']}: shares={item['shares']}, "
                f"estimated_price={item['estimated_price']:,.0f}, estimated_amount={item['estimated_amount']:,.0f}, "
                f"score={item['score']}, 買い理由={item.get('buy_reason', item['reason'])}"
            )
    return lines


def _preview_position_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 保有中ポジションなし"]
    return [
        (
            f"- {item.get('code')} {item.get('name')}: shares={item.get('shares')}, "
            f"entry_price={float(item.get('entry_price') or 0):,.0f}, "
            f"current_price={float(item.get('current_price') or 0):,.0f}, "
            f"market_value={float(item.get('market_value') or 0):,.0f}, "
            f"含み損益={float(item.get('unrealized_pnl') or 0):+,.0f} "
            f"({_format_optional_percent(item.get('unrealized_pnl_rate'))}), "
            f"holding_days={item.get('holding_days')}, sector={item.get('sector_name') or 'N/A'}"
        )
        for item in items
    ]


def _preview_unrealized_lines(summary: dict[str, Any]) -> list[str]:
    return [f"- 含み損益合計: {float(summary.get('unrealized_pnl') or 0):+,.0f}"]


def _preview_skipped_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- None"]
    return [
        (
            f"- {item.get('code')} {item.get('name')}: {item.get('reason')}"
            + (f" (score={item.get('score')})" if item.get("score") is not None else "")
        )
        for item in items
    ]


def _preview_risk_check_summary_lines(summary: dict[str, Any]) -> list[str]:
    if not summary:
        return ["- checked_count: 0", "- passed_count: 0", "- rejected_count: 0"]
    return [
        f"- checked_count: {summary.get('checked_count', 0)}",
        f"- passed_count: {summary.get('passed_count', 0)}",
        f"- rejected_count: {summary.get('rejected_count', 0)}",
    ]


def _preview_safety_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- passed: true / rule: no_order"]
    return [
        (
            f"- {item['action']} {item['code']} {item['name']}: "
            f"{'passed' if item['passed'] else 'rejected'} / rule: {item['rule']} / reason: {item.get('reason') or ''}"
        )
        for item in items
    ]


def run_daily(provider_name: str, target_date_text: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("run-daily mode currently supports --provider jquants only.")
    _validate_date(target_date_text)

    try:
        _run_daily_step(1, 11, "fetch-prices", lambda: ensure_prices(provider_name, target_date_text))
        _run_daily_step(2, 11, "calculate-indicators", lambda: ensure_indicators(provider_name, target_date_text))
        _run_daily_step(3, 11, "market-context", lambda: ensure_market_context(provider_name, target_date_text))
        _run_daily_step(4, 11, "screen", lambda: ensure_screen(provider_name, target_date_text))
        scoring_log = _run_daily_step(5, 11, "score/ai-decision", lambda: ensure_score(provider_name, target_date_text))
        trade_result = _run_daily_step(6, 11, "trade", lambda: execute_trade_for_date(provider_name, target_date_text))
        _run_daily_step(7, 11, "reflections", lambda: write_real_reflections(target_date_text, trade_result))
        report_path, article_path = _run_daily_step(8, 11, "reports/articles", lambda: write_real_daily_markdown(target_date_text, trade_result, scoring_log))
        summary_csv, trades_csv = _run_daily_step(9, 11, "csv", write_real_summary_csvs)
        _run_daily_step(10, 11, "charts", lambda: generate_charts_from_summary(summary_csv, ROOT / "reports" / profile_id_from(load_config(CONFIG_PATH)) / "charts"))
        _run_daily_step(11, 11, "done", lambda: None)
    except SystemExit as exc:
        print(f"run-daily failed during step: {getattr(exc, 'step_name', 'unknown')}")
        raise
    except Exception as exc:
        print(f"run-daily failed during step: {getattr(exc, 'step_name', 'unknown')}")
        print(f"reason: {exc}")
        raise SystemExit(1) from exc

    summary = trade_result["portfolio_summary"]
    trades = trade_result["trades"]
    print(f"date: {target_date_text}")
    print(f"provider: {provider_name}")
    print(f"candidates_count: {trade_result['candidate_count']}")
    print(f"selected_count: {trade_result['selected_count']}")
    print(f"buy_count: {sum(1 for trade in trades if trade.get('action') == 'BUY')}")
    print(f"sell_count: {sum(1 for trade in trades if trade.get('action') == 'SELL')}")
    print(f"total_assets: {summary['total_assets']}")
    print(f"cumulative_profit: {summary['cumulative_profit']}")
    print(f"report_path: {report_path.relative_to(ROOT)}")
    print(f"article_path: {article_path.relative_to(ROOT)}")


def run_backtest(provider_name: str, start_date_text: str, end_date_text: str) -> None:
    global BACKTEST_MODE_ACTIVE, BACKTEST_PROFILE_TIMINGS
    if provider_name != "jquants":
        raise SystemExit("backtest mode currently supports --provider jquants only.")
    runtime_resolution = _runtime_date_resolution(start_date_text, end_date_text)
    requested_start_text = str(runtime_resolution.get("requested_start_date") or start_date_text)
    requested_end_text = str(runtime_resolution.get("requested_end_date") or end_date_text)
    requested_start_date = date.fromisoformat(requested_start_text)
    end_date = date.fromisoformat(end_date_text)
    config = load_config(CONFIG_PATH)
    parsed_start_date = date.fromisoformat(start_date_text)
    effective_start_date = max(parsed_start_date, jquants_earliest_supported_date(config, "prices") or parsed_start_date)
    if effective_start_date != requested_start_date:
        print(
            "warning: requested start-date is before J-Quants supported range; "
            f"requested={requested_start_date.isoformat()} effective={effective_start_date.isoformat()}"
        )
    start_date = effective_start_date
    start_date_text = start_date.isoformat()
    indicator_fetch_start_date = _indicator_fetch_start_date(start_date, config)
    range_key = f"{start_date_text}_to_{end_date_text}"
    BACKTEST_MODE_ACTIVE = True
    BACKTEST_PROFILE_TIMINGS = {}
    _reset_jquants_api_session()
    total_started_at = time.perf_counter()
    profile_id = profile_id_from(config)
    backtest_dir = ROOT / "logs" / "backtests" / profile_id / range_key
    report_dir = ROOT / "reports" / "backtests" / profile_id / range_key
    backtest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    state = initial_live_paper_state(config)
    daily_summaries = []
    all_trades = []
    processed_dates = []
    skipped_days: list[dict[str, str]] = []
    trading_dates: list[date] = []
    price_history_dates: list[date] = []

    try:
        print("backtest date range:")
        print(f"- requested_start_date: {requested_start_date.isoformat()}")
        print(f"- requested_end_date: {requested_end_text}")
        print(f"- effective_start_date: {start_date.isoformat()}")
        print(f"- effective_end_date: {end_date.isoformat()}")
        print(f"- source: start={runtime_resolution.get('start_date_source')} end={runtime_resolution.get('end_date_source')}")
        print(f"- effective_trade_start_date: {start_date.isoformat()}")
        print(f"- indicator_fetch_start_date: {indicator_fetch_start_date.isoformat()}")
        print(f"- indicator_fetch_lookback_days: {_backtest_indicator_fetch_lookback_days(config)}")
        print(f"- indicator_min_history_days: {_backtest_indicator_min_history_days(config)}")
        price_fetch_audit = _run_daily_step(
            1,
            3,
            "fetch-period-prices",
                lambda: ensure_price_history_for_backtest(
                    provider_name,
                    indicator_fetch_start_date,
                    end_date,
                    start_date,
                ),
        )
        trading_dates = _run_daily_step(2, 3, "detect-trading-days", lambda: available_cached_price_dates(start_date, end_date))
        if not trading_dates:
            raise SystemExit("No cached trading days found for the backtest period. The period may be weekend, holiday, or unavailable.")
        price_history_dates = available_cached_price_dates(indicator_fetch_start_date, end_date)
        price_history_days = len(price_history_dates)

        print(f"backtest trading_days: {len(trading_dates)}")
        print(f"backtest price_history_days: {price_history_days}")
        print(f"backtest target_trading_days: {len(trading_dates)}")
        print(f"backtest news fetch: {'disabled' if _backtest_disable_news(config) else 'enabled'}")
        print(f"backtest OpenAI: {'disabled' if _backtest_disable_openai(config) else 'configured by profile'}")
        print(f"backtest indicator_mode: {_backtest_indicator_mode(config)}")
        execution_model = _backtest_execution_model(config)
        print("backtest execution model:")
        print(f"- signal_timing: {execution_model['signal_timing']}")
        print(f"- entry_timing: {execution_model['entry_timing']}")
        print(f"- entry_price_source: {execution_model['entry_price_source']}")
        print(f"- same_day_execution: {execution_model['same_day_execution']}")
        print(f"backtest relative_strength: {'enabled' if _relative_strength_enabled_for_indicators(config) else 'disabled'}")
        print(f"backtest fast_analysis: {'enabled' if _fast_analysis_enabled(config) else 'disabled'}")
        _preload_light_api_context(config, start_date, end_date, indicator_fetch_start_date=indicator_fetch_start_date)
        for index, trading_date in enumerate(trading_dates, start=1):
            target_date_text = trading_date.isoformat()
            global BACKTEST_DAY_LOG_PREFIX
            BACKTEST_DAY_LOG_PREFIX = f"[day {index}/{len(trading_dates)}] {target_date_text}"
            print(f"[day {index}/{len(trading_dates)}] {target_date_text} start")
            has_history, indicator_input_days, indicator_min_days = _has_minimum_indicator_history(indicator_fetch_start_date, trading_date, config)
            indicator_cache_exists = (
                processed_profile_path(config, f"indicators_{target_date_text}.json").exists()
                or (ROOT / "data" / "processed" / f"indicators_{target_date_text}.json").exists()
            )
            if not has_history and not indicator_cache_exists:
                print(
                    f"[day {index}/{len(trading_dates)}] {target_date_text} warning: "
                    f"indicator history insufficient; input_days={indicator_input_days} "
                    f"min_required={indicator_min_days}. skipping day."
                )
                skipped_days.append({"date": target_date_text, "reason": "insufficient_indicator_history"})
                continue
            if not has_history and indicator_cache_exists:
                print(
                    f"[day {index}/{len(trading_dates)}] {target_date_text} warning: "
                    f"indicator history insufficient; input_days={indicator_input_days} "
                    f"min_required={indicator_min_days}, but cached indicator payload exists. continuing to screen empty stage if needed."
                )
            _run_backtest_day_step(index, len(trading_dates), target_date_text, "calculate-indicators", lambda: ensure_indicators(provider_name, target_date_text), lambda: _backtest_indicator_metrics(target_date_text))
            if not _indicator_payload_has_rows(config, target_date_text):
                print(
                    f"[day {index}/{len(trading_dates)}] {target_date_text} warning: "
                    "indicator payload is empty after calculation; continuing with empty candidates/scored_candidates."
                )
            _run_backtest_day_step(index, len(trading_dates), target_date_text, "market-context", lambda: ensure_market_context(provider_name, target_date_text), lambda: _backtest_market_context_metrics(target_date_text))
            screen_step = ensure_screen if RUN_EXPERIMENTS_SHARED_STAGE_ACTIVE else run_screen
            _run_backtest_day_step(index, len(trading_dates), target_date_text, "screen", lambda: screen_step(provider_name, target_date_text), lambda: _backtest_screen_metrics(config, target_date_text))
            scoring_log = _run_backtest_day_step(index, len(trading_dates), target_date_text, "score", lambda: score_for_date(provider_name, target_date_text), lambda: _backtest_score_metrics(config, target_date_text))
            scoring_log["signal_date"] = target_date_text
            for key in ("scores", "selected"):
                for row in scoring_log.get(key, []):
                    row.setdefault("signal_date", target_date_text)
            _run_backtest_day_step(index, len(trading_dates), target_date_text, "ai-decision", lambda: _backtest_ai_decision_status(scoring_log, config), lambda: _backtest_ai_decision_metrics(scoring_log))

            trade_context: dict[str, Any] = {}
            entry_date = _entry_date_for_signal(trading_date, trading_dates, config)
            if entry_date is None:
                print(
                    f"[day {index}/{len(trading_dates)}] {target_date_text} warning: "
                    "entry date not found after signal date; trade execution skipped."
                )
                skipped_days.append({"date": target_date_text, "reason": "missing_entry_date"})
                continue
            entry_date_text = entry_date.isoformat()

            def run_trade_step() -> dict[str, Any]:
                nonlocal state
                execution_scores = _prepare_execution_candidates(scoring_log.get("scores", []), target_date_text, entry_date_text, config)
                scored_candidates = enrich_candidates_with_position_prices(execution_scores, state, entry_date_text)
                trade_config = {
                    **config,
                    "execution": {**config.get("execution", {}), "use_next_day_open_execution": False},
                }
                state, portfolio_summary, trades = execute_real_data_paper_trade(scored_candidates, state, trade_config, entry_date_text)
                portfolio_summary["signal_date"] = target_date_text
                portfolio_summary["entry_date"] = entry_date_text
                portfolio_summary["entry_price_source"] = _backtest_execution_model(config)["entry_price_source"]
                attach_config_version(portfolio_summary, config)
                for trade in trades:
                    trade.setdefault("signal_date", target_date_text)
                    trade.setdefault("entry_price_source", _backtest_execution_model(config)["entry_price_source"])
                    trade.setdefault("config_version", config_version_from(config))
                scoring_log.setdefault("config_version", config_version_from(config))
                scoring_log["entry_date"] = entry_date_text
                scoring_log["entry_price_source"] = _backtest_execution_model(config)["entry_price_source"]
                attach_commentary(portfolio_summary, trades, scored_candidates, config)
                safety_events = portfolio_summary.get("safety_events", [])
                trade_result = {
                    "state": state,
                    "portfolio_summary": portfolio_summary,
                    "trades": trades,
                    "safety_events": safety_events,
                    "candidate_count": scoring_log.get("candidate_count", len(scoring_log.get("scores", []))),
                    "selected_count": len(scoring_log.get("selected", [])),
                }
                trade_context.update({"portfolio_summary": portfolio_summary, "trades": trades, "trade_result": trade_result})
                return trade_result

            trade_result = _run_backtest_day_step(index, len(trading_dates), target_date_text, "trade", run_trade_step, lambda: _backtest_trade_metrics(trade_context))

            report_context: dict[str, Any] = {}

            def run_reports_step() -> tuple[Path, Path, Path]:
                reflection_path = backtest_dir / f"reflections_{target_date_text}.json"
                report_path = report_dir / f"day_{target_date_text}.md"
                article_path = _backtest_article_path(config, target_date_text, profile_id)
                if _generate_backtest_daily_markdown(config) and _generate_articles_in_backtest(config):
                    reflection_path = write_backtest_reflections(backtest_dir, target_date_text, trade_result)
                    report_path, article_path = write_backtest_daily_markdown(report_dir, article_path.parent, target_date_text, trade_result, scoring_log)
                elif _generate_backtest_daily_markdown(config):
                    reflection_path = write_backtest_reflections(backtest_dir, target_date_text, trade_result)
                    report_path = write_backtest_report_markdown(report_dir, target_date_text, trade_result, scoring_log)
                elif _generate_articles_in_backtest(config):
                    reflection_path = write_backtest_reflections(backtest_dir, target_date_text, trade_result)
                    article_path = write_backtest_article_markdown(article_path, target_date_text, trade_result)
                report_context.update({"reflection_path": reflection_path, "report_path": report_path, "article_path": article_path})
                return reflection_path, report_path, article_path

            reflection_path, report_path, article_path = _run_backtest_day_step(index, len(trading_dates), target_date_text, "reports/articles", run_reports_step, lambda: _backtest_report_metrics(report_context))

            portfolio_summary = trade_context["portfolio_summary"]
            trades = trade_context["trades"]

            def run_db_save_step() -> None:
                storage_trades = _trades_for_storage(trades, config)
                storage_scoring_log = _scoring_log_for_storage(scoring_log, config)
                write_json(
                    backtest_dir / f"trades_{target_date_text}.json",
                    {
                        "date": target_date_text,
                        "signal_date": target_date_text,
                        "entry_date": entry_date_text,
                        "provider": provider_name,
                        "config_version": config_version_from(config),
                        "storage_mode": _storage_save_mode(config),
                        "trades": storage_trades,
                    },
                )
                write_json(backtest_dir / f"portfolio_{target_date_text}.json", portfolio_summary)
                write_json(backtest_dir / f"safety_{target_date_text}.json", {"date": target_date_text, "config_version": config_version_from(config), "safety_events": trade_result.get("safety_events", [])})
                write_json(backtest_dir / f"scoring_{target_date_text}.json", storage_scoring_log)
                save_portfolio_snapshot(config, ROOT, portfolio_summary)
                save_trades(config, ROOT, entry_date_text, storage_trades)
                save_pending_orders(config, ROOT, state.get("pending_orders", []))
                save_safety_events(config, ROOT, target_date_text, trade_result.get("safety_events", []))

            _run_backtest_day_step(index, len(trading_dates), target_date_text, "db-save", run_db_save_step, lambda: _backtest_db_save_metrics(trades, trade_result))
            daily_summaries.append(portfolio_summary)
            all_trades.extend(trades)
            processed_dates.append(target_date_text)
            print(
                f"[day {index}/{len(trading_dates)}] {target_date_text} done "
                f"selected={trade_result['selected_count']} buy={sum(1 for trade in trades if trade.get('action') == 'BUY')} "
                f"sell={sum(1 for trade in trades if trade.get('action') == 'SELL')} assets={portfolio_summary['total_assets']}"
            )
            print(f"  report: {report_path.relative_to(ROOT)}")
            print(f"  article: {article_path.relative_to(ROOT)}")
            print(f"  reflections: {reflection_path.relative_to(ROOT)}")

        summary = _run_daily_step(
            3,
            3,
            "backtest-summary",
            lambda: (
                write_backtest_summary(
                    range_key,
                    start_date_text,
                    end_date_text,
                    config,
                    state,
                    daily_summaries,
                    all_trades,
                    backtest_dir,
                    {
                        **runtime_resolution,
                        "requested_start_date": requested_start_date.isoformat(),
                        "requested_end_date": requested_end_text,
                        "effective_start_date": (trading_dates[0].isoformat() if trading_dates else start_date.isoformat()),
                        "effective_end_date": end_date.isoformat(),
                    },
                    build_backtest_date_range_audit(
                        config=config,
                        requested_start_date=requested_start_date,
                        requested_end_date=end_date,
                        effective_trade_start_date=start_date,
                        effective_trade_end_date=end_date,
                        indicator_fetch_start_date=indicator_fetch_start_date,
                        price_history_dates=price_history_dates,
                        trading_dates=trading_dates,
                        processed_dates=processed_dates,
                        skipped_days=skipped_days,
                        all_trades=all_trades,
                        price_fetch_audit=price_fetch_audit,
                    ),
                )
            ),
        )
    except SystemExit as exc:
        print(f"backtest failed during step: {getattr(exc, 'step_name', 'unknown')}")
        BACKTEST_MODE_ACTIVE = False
        raise
    except KeyboardInterrupt as exc:
        print("")
        print("backtest interrupted by Ctrl+C")
        print("No live orders were sent. Partial cache and logs remain on disk.")
        BACKTEST_MODE_ACTIVE = False
        raise SystemExit(130) from exc
    except Exception as exc:
        print(f"backtest failed during step: {getattr(exc, 'step_name', 'unknown')}")
        print(f"reason: {exc}")
        BACKTEST_MODE_ACTIVE = False
        raise SystemExit(1) from exc

    print("backtest completed")
    print(f"provider: {provider_name}")
    print(f"start_date: {start_date_text}")
    print(f"end_date: {end_date_text}")
    audit = summary.get("date_range_audit", {})
    if audit:
        print("Backtest Date Range Audit")
        print(f"- first_price_date: {audit.get('first_price_date')}")
        print(f"- last_price_date: {audit.get('last_price_date')}")
        print(f"- first_trading_day: {audit.get('first_trading_day')}")
        print(f"- last_trading_day: {audit.get('last_trading_day')}")
        print(f"- target_trading_days: {audit.get('target_trading_days')}")
        print(f"- processed_days: {audit.get('processed_days')}")
        print(f"- last_processed_day: {audit.get('last_processed_day')}")
        print(f"- coverage_ok: {((audit.get('data_coverage') or {}).get('prices') or {}).get('coverage_ok')}")
        execution_audit = audit.get("backtest_execution_audit", {})
        if execution_audit:
            print("Backtest Execution Audit")
            print(f"- status: {execution_audit.get('status')}")
            print(f"- last_processed_day: {execution_audit.get('last_processed_day')}")
            print(f"- last_candidate_date: {execution_audit.get('last_candidate_date')}")
            print(f"- last_scored_candidate_date: {execution_audit.get('last_scored_candidate_date')}")
            print(f"- last_trade_date: {execution_audit.get('last_trade_date')}")
            print(f"- date_range_limited_reason: {execution_audit.get('date_range_limited_reason')}")
        coverage_audit = audit.get("backtest_coverage_audit", {})
        if coverage_audit:
            print("Backtest Coverage Audit")
            print(f"- first_candidate_date: {coverage_audit.get('first_candidate_date')}")
            print(f"- first_trade_date: {coverage_audit.get('first_trade_date')}")
            print(f"- candidate_days: {coverage_audit.get('candidate_days')}")
            print(f"- trade_days: {coverage_audit.get('trade_days')}")
            print(f"- coverage_ratio: {_format_optional_percent(coverage_audit.get('coverage_ratio'))}")
            print(f"- coverage_warning: {coverage_audit.get('coverage_warning') or '-'}")
    print(f"processed_days: {len(processed_dates)}")
    print(f"final_assets: {summary['final_assets']}")
    print(f"cumulative_profit: {summary['cumulative_profit']}")
    print(f"summary_md: {Path(summary['report_markdown_path']).relative_to(ROOT)}")
    print(f"summary_json: {Path(summary['report_json_path']).relative_to(ROOT)}")
    _print_backtest_profile_timings(time.perf_counter() - total_started_at)
    if _is_rule_based_backtest(config):
        print("")
        print(_rule_based_backtest_completed_label(start_date, end_date))
        print("")
        print(f"- final_assets: {summary['final_assets']}")
        print(f"- net_cumulative_profit: {summary.get('net_cumulative_profit')}")
        print(f"- net_cumulative_profit_rate: {summary.get('net_cumulative_profit_rate')}")
        print(f"- win_rate: {summary.get('win_rate')}")
        print(f"- max_drawdown: {summary.get('max_drawdown')}")
        print(f"- total_trades: {summary.get('total_trades')}")
        print(f"- report_path: {Path(summary['rule_based_90d_report_path']).relative_to(ROOT)}")
    BACKTEST_MODE_ACTIVE = False


def _run_daily_step(step_number: int, total_steps: int, step_name: str, action: Any) -> Any:
    print(f"[{step_number}/{total_steps}] {step_name} start")
    started_at = time.perf_counter()
    try:
        result = action()
    except SystemExit as exc:
        setattr(exc, "step_name", step_name)
        print(f"[{step_number}/{total_steps}] {step_name} failed")
        raise
    except Exception as exc:
        setattr(exc, "step_name", step_name)
        print(f"[{step_number}/{total_steps}] {step_name} failed")
        raise
    elapsed = time.perf_counter() - started_at
    _record_backtest_profile_timing(step_name, elapsed)
    print(f"[{step_number}/{total_steps}] {step_name} done")
    return result


def _enable_line_buffered_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except AttributeError:
            pass


def _run_backtest_day_step(
    day_number: int,
    total_days: int,
    target_date_text: str,
    step_name: str,
    action: Any,
    metrics: Any | None = None,
) -> Any:
    prefix = f"[day {day_number}/{total_days}] {target_date_text} {step_name}"
    print(f"{prefix} start")
    started_at = time.perf_counter()
    try:
        result = action()
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - started_at
        print(f"{prefix} interrupted after {elapsed:.1f}s")
        raise
    except Exception:
        elapsed = time.perf_counter() - started_at
        print(f"{prefix} failed after {elapsed:.1f}s")
        raise
    elapsed = time.perf_counter() - started_at
    _record_backtest_profile_timing(step_name, elapsed)
    print(f"{prefix} done in {elapsed:.1f}s")
    if metrics:
        for line in metrics() or []:
            print(f"  {line}")
    return result


def _record_backtest_profile_timing(step_name: str, elapsed: float) -> None:
    if not BACKTEST_MODE_ACTIVE:
        return
    bucket = {
        "fetch-period-prices": "fetch_prices",
        "calculate-indicators": "indicator",
        "screen": "screening",
        "score": "scoring",
        "trade": "trading",
        "reports/articles": "report",
        "backtest-summary": "report",
    }.get(step_name)
    if not bucket:
        return
    BACKTEST_PROFILE_TIMINGS[bucket] = BACKTEST_PROFILE_TIMINGS.get(bucket, 0.0) + elapsed


def _print_backtest_profile_timings(total_seconds: float) -> None:
    print("backtest profile timings:")
    for key in ["fetch_prices", "indicator", "screening", "scoring", "trading", "report"]:
        print(f"- {key} seconds: {BACKTEST_PROFILE_TIMINGS.get(key, 0.0):.2f}")
    print(f"- total seconds: {total_seconds:.2f}")
    print("")
    print("Performance Summary")
    for key in ["fetch_prices", "indicator", "screening", "scoring", "trading", "report"]:
        print(f"- {key}: {BACKTEST_PROFILE_TIMINGS.get(key, 0.0):.2f}s")
    print(f"- total_runtime: {total_seconds:.2f}s")


def _print_fetch_statistics(provider: Any) -> None:
    stats = getattr(provider, "fetch_stats", {}) or {}
    api_calls = int(stats.get("api_calls", 0) or 0)
    total_fetch_time = float(stats.get("total_fetch_time", 0.0) or 0.0)
    average_fetch_time = total_fetch_time / api_calls if api_calls else 0.0
    print("fetch statistics")
    print(f"- api_calls: {api_calls}")
    print(f"- cache_hits: {int(stats.get('cache_hits', 0) or 0)}")
    print(f"- cache_misses: {int(stats.get('cache_misses', 0) or 0)}")
    print(f"- average_fetch_time: {average_fetch_time:.3f}s")
    print(f"- rate_limit_wait_time: {float(stats.get('rate_limit_wait_time', 0.0) or 0.0):.3f}s")


def _backtest_indicator_metrics(target_date_text: str) -> list[str]:
    path = ROOT / "data" / "processed" / f"indicators_{target_date_text}.json"
    if not path.exists():
        return ["indicators file: missing"]
    payload = read_json(path)
    return [
        f"indicators mode: {payload.get('indicator_mode', 'full')}",
        f"indicators input days: {payload.get('input_days', 'N/A')}",
        f"indicators input rows: {payload.get('input_rows', 'N/A')}",
        f"indicators target stocks: {payload.get('target_stocks', 'N/A')}",
        f"indicators calculated: {payload.get('calculated_count', len(payload.get('indicators', [])))}",
        f"indicators excluded: {payload.get('excluded_count', 'N/A')}",
    ]


def _backtest_market_context_metrics(target_date_text: str) -> list[str]:
    context = load_market_context_for_date(target_date_text, "jquants")
    return [
        f"market regime: {context.get('market_regime', 'N/A')}",
        f"advance ratio: {_format_optional_percent(context.get('advance_ratio'))}",
        f"sector count: {len(context.get('sector_momentum', []))}",
    ]


def _backtest_screen_metrics(config: dict[str, Any], target_date_text: str) -> list[str]:
    indicator_path = ROOT / "data" / "processed" / f"indicators_{target_date_text}.json"
    indicators = read_json(indicator_path).get("indicators", []) if indicator_path.exists() else []
    candidates_path = processed_profile_path(config, f"candidates_{target_date_text}.json")
    candidates = read_json(candidates_path).get("candidates", []) if candidates_path.exists() else []
    return [
        f"screen target stocks: {len(indicators)}",
        f"screen candidates: {len(candidates)}",
    ]


def _backtest_score_metrics(config: dict[str, Any], target_date_text: str) -> list[str]:
    path = processed_profile_path(config, f"scored_candidates_{target_date_text}.json")
    payload = read_json(path) if path.exists() else {}
    scores = payload.get("scores", [])
    return [
        f"scoring candidates: {payload.get('candidate_count', len(scores))}",
        f"scored candidates: {payload.get('scored_count', len(scores))}",
        f"selected candidates: {payload.get('selected_count', sum(1 for item in scores if item.get('selected')))}",
    ]


def _backtest_ai_decision_status(scoring_log: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if _backtest_disable_openai(config):
        scoring_log["ai_decision"] = {
            "enabled": False,
            "provider": "rule_based",
            "fallback_used": False,
            "reason": "backtest.disable_openai is true",
        }
    return scoring_log.get("ai_decision", {})


def _backtest_ai_decision_metrics(scoring_log: dict[str, Any]) -> list[str]:
    decision = scoring_log.get("ai_decision", {})
    return [
        f"ai_decision enabled: {str(bool(decision.get('enabled', False))).lower()}",
        f"ai_decision provider: {decision.get('provider', 'rule_based')}",
        f"ai_decision fallback: {str(bool(decision.get('fallback_used', False))).lower()}",
    ]


def _backtest_trade_metrics(context: dict[str, Any]) -> list[str]:
    trades = context.get("trades", [])
    summary = context.get("portfolio_summary", {})
    return [
        f"trade input scores: {context.get('trade_result', {}).get('candidate_count', 'N/A')}",
        f"buy orders: {sum(1 for trade in trades if trade.get('action') == 'BUY')}",
        f"sell orders: {sum(1 for trade in trades if trade.get('action') == 'SELL')}",
        f"total assets: {summary.get('total_assets', 'N/A')}",
    ]


def _backtest_report_metrics(context: dict[str, Any]) -> list[str]:
    return [
        f"report: {_relative_path_text(context.get('report_path'))}",
        f"article: {_relative_path_text(context.get('article_path'))}",
        f"reflections: {_relative_path_text(context.get('reflection_path'))}",
    ]


def _backtest_db_save_metrics(trades: list[dict[str, Any]], trade_result: dict[str, Any]) -> list[str]:
    return [
        f"trades saved: {len(trades)}",
        f"safety events saved: {len(trade_result.get('safety_events', []))}",
        f"selected count saved: {trade_result.get('selected_count', 0)}",
    ]


def _backtest_disable_news(config: dict[str, Any]) -> bool:
    return bool(config.get("backtest", {}).get("disable_news_fetch", True))


def _backtest_disable_openai(config: dict[str, Any]) -> bool:
    return bool(config.get("backtest", {}).get("disable_openai", True))


def _fast_analysis_enabled(config: dict[str, Any]) -> bool:
    return SUMMARY_ONLY_ACTIVE or FAST_ANALYSIS_ACTIVE or bool(config.get("backtest", {}).get("fast_analysis")) or bool(config.get("analysis", {}).get("fast_analysis"))


def _backtest_indicator_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("backtest", {}).get("indicator_mode", "fast"))
    return mode if mode in {"full", "fast", "minimal"} else "fast"


def _normalize_entry_timing(value: Any) -> str:
    timing = str(value or "next_business_day_open").strip().lower().replace("-", "_")
    if timing in {"same_day_close", "next_business_day_open", "next_business_day_close"}:
        return timing
    return "next_business_day_open"


def _backtest_execution_model(config: dict[str, Any]) -> dict[str, Any]:
    backtest = config.get("backtest", {})
    entry_timing = _normalize_entry_timing(backtest.get("entry_timing", "next_business_day_open"))
    if entry_timing in {"same_day_close", "next_business_day_close"}:
        entry_price_source = "close"
    else:
        entry_price_source = str(backtest.get("entry_price_source", "open") or "open").lower()
        if entry_price_source not in {"open", "close"}:
            entry_price_source = "open"
    return {
        "signal_timing": "after_close",
        "entry_timing": entry_timing,
        "entry_price_source": entry_price_source,
        "same_day_execution": entry_timing == "same_day_close",
    }


def _entry_date_for_signal(signal_date: date, trading_dates: list[date], config: dict[str, Any]) -> date | None:
    if _backtest_execution_model(config)["entry_timing"] == "same_day_close":
        return signal_date
    for trading_date in trading_dates:
        if trading_date > signal_date:
            return trading_date
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare_execution_candidates(
    scored_candidates: list[dict[str, Any]],
    signal_date_text: str,
    entry_date_text: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    model = _backtest_execution_model(config)
    price_source = model["entry_price_source"]
    entry_prices = {str(item.get("code")): item for item in load_cached_prime_prices(date.fromisoformat(entry_date_text))}
    prepared: list[dict[str, Any]] = []
    for item in scored_candidates:
        code = str(item.get("code"))
        market = entry_prices.get(code)
        signal_close = _safe_float(item.get("close"))
        row = dict(item)
        row["signal_date"] = signal_date_text
        row["entry_date"] = entry_date_text
        row["entry_price_source"] = price_source
        row["signal_close_price"] = signal_close
        if market:
            entry_open = _safe_float(market.get("open"))
            entry_close = _safe_float(market.get("close"))
            entry_price = entry_open if price_source == "open" else entry_close
            row.update(
                {
                    "execution_date": entry_date_text,
                    "open": market.get("open"),
                    "high": market.get("high"),
                    "low": market.get("low"),
                    "close": market.get("close"),
                    "volume": market.get("volume", row.get("volume")),
                    "entry_price": entry_price,
                    "entry_open_price": entry_open,
                    "entry_close_price": entry_close,
                    "entry_gap_rate": round((entry_open - signal_close) / signal_close, 4)
                    if entry_open is not None and signal_close
                    else None,
                    "entry_price_available": entry_price is not None,
                }
            )
        else:
            row.update(
                {
                    "execution_date": entry_date_text,
                    "entry_price": None,
                    "entry_open_price": None,
                    "entry_close_price": None,
                    "entry_gap_rate": None,
                    "entry_price_available": False,
                }
            )
        prepared.append(row)
    return prepared


def _backtest_indicator_fetch_lookback_days(config: dict[str, Any]) -> int:
    return max(0, int(config.get("backtest", {}).get("indicator_fetch_lookback_days", 180)))


def _backtest_indicator_min_history_days(config: dict[str, Any]) -> int:
    return max(1, int(config.get("backtest", {}).get("indicator_min_history_days", 60)))


def _indicator_fetch_start_date(trade_start_date: date, config: dict[str, Any]) -> date:
    requested = trade_start_date - timedelta(days=_backtest_indicator_fetch_lookback_days(config))
    earliest = jquants_earliest_supported_date(config, "prices")
    return max(requested, earliest) if earliest else requested


def _indicator_history_day_count(fetch_start_date: date, target_date: date) -> int:
    return len(available_cached_price_dates(fetch_start_date, target_date))


def _has_minimum_indicator_history(
    fetch_start_date: date,
    target_date: date,
    config: dict[str, Any],
) -> tuple[bool, int, int]:
    input_days = _indicator_history_day_count(fetch_start_date, target_date)
    min_days = _backtest_indicator_min_history_days(config)
    return input_days >= min_days, input_days, min_days


def _indicator_payload_has_rows(config: dict[str, Any], target_date_text: str) -> bool:
    profile_path = processed_profile_path(config, f"indicators_{target_date_text}.json")
    path = profile_path if profile_path.exists() else ROOT / "data" / "processed" / f"indicators_{target_date_text}.json"
    if not path.exists():
        return False
    payload = read_json(path)
    return bool(payload.get("indicators"))


def _relative_strength_enabled_for_indicators(config: dict[str, Any]) -> bool:
    return bool(config.get("features", {}).get("relative_strength")) and bool(
        config.get("scoring", {}).get("use_relative_strength_score")
    )


def _indicator_cache_matches_current_scoring(
    payload: dict[str, Any],
    config: dict[str, Any],
    indicator_mode: str,
    enable_relative_strength: bool,
) -> bool:
    if payload.get("indicator_mode") != indicator_mode:
        return False
    if bool(payload.get("relative_strength_enabled")) != enable_relative_strength:
        return False
    if not enable_relative_strength:
        return True

    expected_source = _expected_relative_strength_benchmark_source(config)
    if str(payload.get("benchmark_source") or "") != expected_source:
        return False
    indicators = payload.get("indicators", [])
    if not indicators:
        return False
    return _relative_strength_indicator_rows_are_populated(indicators, expected_source)


def _relative_strength_indicator_rows_are_populated(indicators: list[dict[str, Any]], expected_source: str) -> bool:
    required_fields = [
        "stock_return_5d",
        "stock_return_10d",
        "stock_return_20d",
        "benchmark_return_5d",
        "benchmark_return_10d",
        "benchmark_return_20d",
        "relative_strength_5d",
        "relative_strength_10d",
        "relative_strength_20d",
        "relative_strength_score",
    ]
    for item in indicators:
        if str(item.get("benchmark_source") or "") != expected_source:
            continue
        if all(field in item for field in required_fields) and any(
            item.get(field) is not None for field in required_fields if field != "relative_strength_score"
        ):
            return True
    return False


def _ensure_relative_strength_benchmark_cache(
    config: dict[str, Any],
    target_date: date,
    indicator_mode: str,
    enable_relative_strength: bool,
) -> None:
    if not enable_relative_strength:
        return
    if _expected_relative_strength_benchmark_source(config) != "topix":
        return
    if _preloaded_topix_payload_for(target_date):
        return
    lookback_days = 60 if indicator_mode == "full" else 35
    fetch_dates = previous_business_dates(target_date, lookback_days)
    start_date = fetch_dates[0] if fetch_dates else target_date
    _load_topix_prices_for_period(start_date, target_date, config)


def _rule_based_backtest_completed_label(start_date: date, end_date: date) -> str:
    days = (end_date - start_date).days + 1
    if 80 <= days <= 100:
        return "Rule-based 90d backtest completed"
    return "Rule-based backtest completed"


def ensure_prices(provider_name: str, target_date_text: str) -> None:
    price_path = ROOT / "data" / "raw" / f"prices_{target_date_text}.json"
    if price_path.exists():
        return
    run_fetch_prices(provider_name, target_date_text)


def ensure_indicators(provider_name: str, target_date_text: str) -> None:
    config = load_config(CONFIG_PATH)
    path = ROOT / "data" / "processed" / f"indicators_{target_date_text}.json"
    profile_path = processed_profile_path(config, f"indicators_{target_date_text}.json")
    indicator_mode = _backtest_indicator_mode(config) if BACKTEST_MODE_ACTIVE else "full"
    enable_relative_strength = _relative_strength_enabled_for_indicators(config)
    if BACKTEST_MODE_ACTIVE and profile_path.exists():
        payload = read_json(profile_path)
        if _indicator_cache_matches_current_scoring(payload, config, indicator_mode, enable_relative_strength):
            _ensure_relative_strength_benchmark_cache(config, date.fromisoformat(target_date_text), indicator_mode, enable_relative_strength)
            write_json(path, payload)
            _save_common_processed_cache(config, "indicators", target_date_text, payload)
            _link_profile_processed_cache_to_common(config, "indicators", target_date_text, profile_path)
            COMMON_CACHE_METRICS["profile_specific_cache_count"] = COMMON_CACHE_METRICS.get("profile_specific_cache_count", 0) + 1
            print(f"{BACKTEST_DAY_LOG_PREFIX} indicators cache hit: {profile_path.relative_to(ROOT)}")
            return
    if BACKTEST_MODE_ACTIVE and _restore_common_processed_cache(config, "indicators", target_date_text, profile_path):
        payload = read_json(profile_path)
        if _indicator_cache_matches_current_scoring(payload, config, indicator_mode, enable_relative_strength):
            _ensure_relative_strength_benchmark_cache(config, date.fromisoformat(target_date_text), indicator_mode, enable_relative_strength)
            write_json(path, payload)
            print(f"{BACKTEST_DAY_LOG_PREFIX} indicators common cache hit: {profile_path.relative_to(ROOT)}")
            return
    if path.exists() and not BACKTEST_MODE_ACTIVE:
        return
    if path.exists() and BACKTEST_MODE_ACTIVE:
        payload = read_json(path)
        if _indicator_cache_matches_current_scoring(payload, config, indicator_mode, enable_relative_strength):
            _ensure_relative_strength_benchmark_cache(config, date.fromisoformat(target_date_text), indicator_mode, enable_relative_strength)
            return
    run_calculate_indicators(provider_name, target_date_text)


def ensure_screen(provider_name: str, target_date_text: str) -> None:
    config = load_config(CONFIG_PATH)
    path = processed_profile_path(config, f"candidates_{target_date_text}.json")
    if path.exists():
        screening_path = ROOT / "logs" / "screening" / profile_id_from(config) / f"screening_{target_date_text}.json"
        payload = read_json(screening_path) if screening_path.exists() else read_json(path)
        payload.setdefault("config_version", config_version_from(config))
        save_screening_results(config, ROOT, payload)
        _save_common_processed_cache(config, "candidates", target_date_text, read_json(path))
        _link_profile_processed_cache_to_common(config, "candidates", target_date_text, path)
        COMMON_CACHE_METRICS["profile_specific_cache_count"] = COMMON_CACHE_METRICS.get("profile_specific_cache_count", 0) + 1
        return
    if BACKTEST_MODE_ACTIVE and _restore_common_processed_cache(config, "candidates", target_date_text, path):
        payload = read_json(path)
        save_screening_results(config, ROOT, payload)
        print(f"{BACKTEST_DAY_LOG_PREFIX} candidates common cache hit: {path.relative_to(ROOT)}")
        return
    run_screen(provider_name, target_date_text)


def ensure_score(provider_name: str, target_date_text: str) -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    path = processed_profile_path(config, f"scored_candidates_{target_date_text}.json")
    if not path.exists():
        run_score(provider_name, target_date_text)
    payload = read_json(path)
    payload.setdefault("config_version", config_version_from(config))
    save_scoring_results(
        config,
        ROOT,
        {
            "date": payload.get("date", target_date_text),
            "config_version": payload.get("config_version"),
            "source_provider": provider_name,
            "scores": payload.get("scores", []),
        },
    )
    return payload


def score_for_date(provider_name: str, target_date_text: str) -> dict[str, Any]:
    copied = _copy_reusable_scoring_for_active_profile(target_date_text)
    if copied is not None:
        return copied
    run_score(provider_name, target_date_text)
    config = load_config(CONFIG_PATH)
    scoring_path = ROOT / "logs" / "scoring" / profile_id_from(config) / f"scoring_{target_date_text}.json"
    processed_path = processed_profile_path(config, f"scored_candidates_{target_date_text}.json")
    scoring_log = read_json(scoring_path)
    processed = read_json(processed_path)
    return {
        **scoring_log,
        "candidate_count": processed.get("candidate_count", len(scoring_log.get("scores", []))),
        "scored_count": processed.get("scored_count", len(scoring_log.get("scores", []))),
        "selected_count": processed.get("selected_count", len(scoring_log.get("selected", []))),
    }


def _copy_reusable_scoring_for_active_profile(target_date_text: str) -> dict[str, Any] | None:
    if not RUN_EXPERIMENTS_SHARED_STAGE_ACTIVE:
        return None
    target_config = load_config(CONFIG_PATH)
    target_profile_id = profile_id_from(target_config)
    source_profile_id = RUN_EXPERIMENTS_SCORING_REUSE_SOURCE_BY_PROFILE.get(target_profile_id)
    if not source_profile_id:
        return None
    source_config = load_profile(source_profile_id)
    source_processed = processed_profile_path(source_config, f"scored_candidates_{target_date_text}.json")
    source_log = ROOT / "logs" / "scoring" / source_profile_id / f"scoring_{target_date_text}.json"
    if not source_processed.exists() or not source_log.exists():
        return None
    target_processed = processed_profile_path(target_config, f"scored_candidates_{target_date_text}.json")
    target_log = ROOT / "logs" / "scoring" / target_profile_id / f"scoring_{target_date_text}.json"
    processed_payload = _with_profile_metadata(read_json(source_processed), target_config)
    scoring_payload = _with_profile_metadata(read_json(source_log), target_config)
    write_json(target_processed, processed_payload)
    write_json(target_log, scoring_payload)
    save_scoring_results(
        target_config,
        ROOT,
        {
            "date": scoring_payload.get("date", target_date_text),
            "config_version": scoring_payload.get("config_version"),
            "source_provider": scoring_payload.get("source_provider", "jquants"),
            "scores": scoring_payload.get("scores", []),
        },
    )
    RUN_EXPERIMENTS_PERFORMANCE_REPORT["reused_scoring_count"] = int(RUN_EXPERIMENTS_PERFORMANCE_REPORT.get("reused_scoring_count", 0) or 0) + 1
    return {
        **scoring_payload,
        "candidate_count": processed_payload.get("candidate_count", len(scoring_payload.get("scores", []))),
        "scored_count": processed_payload.get("scored_count", len(scoring_payload.get("scores", []))),
        "selected_count": processed_payload.get("selected_count", len(scoring_payload.get("selected", []))),
    }


def _save_rejected_candidates(config: dict[str, Any]) -> bool:
    return bool(config.get("analysis", {}).get("save_rejected_candidates", True))


def _storage_save_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("storage", {}).get("save_mode") or "analysis")
    return mode if mode in {"full_debug", "analysis", "compact"} else "analysis"


RUNTIME_SCORE_FIELDS = {
    "code", "name", "sector_name", "section", "market_section", "listing_market",
    "date", "open", "high", "low", "close", "volume", "ma5", "ma25", "rsi",
    "volume_ratio", "turnover_value", "total_score", "technical_score",
    "confidence", "rank", "selected", "reason", "score_reason",
    "selection_reason", "selected_reason", "rejected_reason", "fallback",
    "candlestick_signals", "candlestick_score", "trend_score", "ma_score",
    "volume_score", "rsi_score", "market_context_score", "sector_score",
    "penalty_score", "score_components_total", "score_components_match",
    "market_regime", "market_filter_applied", "market_filter_reason",
    "source_provider", "config_version",
}

ANALYSIS_SCORE_FIELDS = RUNTIME_SCORE_FIELDS | {
    "sector_momentum_score", "sector_rank", "sector_comment", "sector_score_adjustment",
    "five_day_volatility", "five_day_change_rate",
    "stock_return_5d", "stock_return_10d", "stock_return_20d",
    "benchmark_source", "benchmark_return_5d", "benchmark_return_10d", "benchmark_return_20d",
    "relative_strength_5d", "relative_strength_10d", "relative_strength_20d",
    "relative_strength_score", "topix_records_loaded", "topix_api_calls",
    "investor_context_source", "investor_context_week", "overseas_net_buy",
    "overseas_net_buy_4w_sum", "overseas_net_buy_4w_trend", "overseas_buy_sell_ratio",
    "individual_net_buy", "institution_net_buy", "trust_bank_net_buy",
    "proprietary_net_buy", "investor_context_score",
    "candle_type", "candle_body_rate", "upper_shadow_rate", "lower_shadow_rate",
    "close_position_in_range", "gap_rate",
    "earnings_filter_checked", "earnings_filter_blocked", "earnings_filter_reason",
    "earnings_announcement_date", "earnings_calendar_records_count",
    "earnings_info_found", "earnings_candidate_date", "earnings_days_until_earnings",
}

RUNTIME_TRADE_FIELDS = {
    "trade_id", "profile_id", "profile_name", "action", "code", "name", "sector_name",
    "signal_date", "entry_date", "exit_date", "holding_days", "entry_price",
    "entry_price_source", "signal_close_price", "entry_open_price", "entry_gap_rate",
    "exit_price", "shares", "amount", "profit", "profit_rate", "exit_reason",
    "result", "score", "total_score", "technical_score", "rsi", "volume_ratio",
    "selected_reason", "reason", "broker_provider", "order_status", "status",
    "config_version",
}

ANALYSIS_TRADE_FIELDS = RUNTIME_TRADE_FIELDS | {
    "stock_return_5d", "stock_return_10d", "stock_return_20d",
    "benchmark_source", "benchmark_return_5d", "benchmark_return_10d", "benchmark_return_20d",
    "relative_strength_5d", "relative_strength_10d", "relative_strength_20d",
    "relative_strength_score", "topix_records_loaded", "topix_api_calls",
    "investor_context_source", "investor_context_week", "overseas_net_buy",
    "overseas_net_buy_4w_sum", "overseas_net_buy_4w_trend", "investor_context_score",
    "ma_score", "rsi_score", "volume_score", "candlestick_score",
    "market_context_score", "sector_score", "penalty_score",
    "score_components_total", "score_components_match", "market_regime",
    "advance_ratio", "candlestick_signals",
    "earnings_filter_checked", "earnings_filter_blocked", "earnings_filter_reason",
    "earnings_announcement_date", "earnings_info_found", "earnings_candidate_date",
    "earnings_days_until_earnings", "gross_profit", "gross_profit_rate",
    "buy_commission", "sell_commission", "total_commission", "estimated_tax",
    "net_profit", "net_profit_rate",
}


def _save_backtest_daily_reports(config: dict[str, Any]) -> bool:
    return _generate_backtest_daily_markdown(config)


def _reporting_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("reporting", {}) if isinstance(config.get("reporting"), dict) else {}


def _generate_articles_in_backtest(config: dict[str, Any]) -> bool:
    if SUMMARY_ONLY_ACTIVE:
        return False
    reporting = _reporting_config(config)
    if "generate_articles_in_backtest" in reporting:
        return bool(reporting.get("generate_articles_in_backtest"))
    return bool(config.get("analysis", {}).get("save_backtest_daily_reports", False))


def _generate_backtest_daily_markdown(config: dict[str, Any]) -> bool:
    if SUMMARY_ONLY_ACTIVE:
        return False
    reporting = _reporting_config(config)
    if "generate_daily_markdown_in_backtest" in reporting:
        return bool(reporting.get("generate_daily_markdown_in_backtest"))
    return bool(config.get("analysis", {}).get("save_backtest_daily_reports", False))


def _scores_for_storage(scores: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    mode = _storage_save_mode(config)
    if mode == "full_debug":
        rows = scores
    elif mode == "compact":
        rows = [item for item in scores if item.get("selected") or _is_always_saved_rejected_score(item)]
    elif _save_rejected_candidates(config):
        rows = scores
    else:
        rows = [item for item in scores if item.get("selected") or _is_always_saved_rejected_score(item)]
    return [_score_row_for_storage(item, mode) for item in rows]


def _score_row_for_storage(item: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "full_debug":
        return dict(item)
    fields = RUNTIME_SCORE_FIELDS if mode == "compact" else ANALYSIS_SCORE_FIELDS
    return {key: item.get(key) for key in fields if key in item}


def _trades_for_storage(trades: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    mode = _storage_save_mode(config)
    if mode == "full_debug":
        return trades
    fields = RUNTIME_TRADE_FIELDS if mode == "compact" else ANALYSIS_TRADE_FIELDS
    return [{key: trade.get(key) for key in fields if key in trade} for trade in trades]


def _is_always_saved_rejected_score(item: dict[str, Any]) -> bool:
    return str(item.get("rejected_reason") or item.get("reason") or "") == "investor_context_negative"


def _scoring_log_for_storage(scoring_log: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    mode = _storage_save_mode(config)
    if mode == "full_debug" and _save_rejected_candidates(config):
        return scoring_log
    scores = _scores_for_storage(scoring_log.get("scores", []), config)
    omitted = len(scoring_log.get("scores", [])) - len(scores)
    return {
        **scoring_log,
        "storage_mode": mode,
        "scores": scores,
        "selected": [item for item in scores if item.get("selected")],
        "rejected_candidate_detail_saved": mode == "analysis" and _save_rejected_candidates(config),
        "storage_omitted_score_count": omitted,
        "storage_note": _storage_note(mode, omitted),
    }


def _storage_note(mode: str, omitted: int) -> str:
    if mode == "full_debug":
        return "Full debug storage keeps all score fields and rows."
    if mode == "compact":
        return f"Compact storage keeps selected and important rejected rows only; omitted={omitted}."
    return "Analysis storage prunes debug-only fields while keeping analysis rows."


def attach_commentary(
    portfolio_summary: dict[str, Any],
    trades: list[dict[str, Any]],
    scored_candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    candidate_by_code = {item.get("code"): item for item in scored_candidates}
    selected_candidates = [item for item in scored_candidates if item.get("selected")]
    for trade in trades:
        action = trade.get("action")
        candidate = candidate_by_code.get(trade.get("code"), {})
        if action == "BUY" and not trade.get("dealer_comment"):
            trade["dealer_comment"] = generate_buy_comment({**candidate, **trade}, config)
        elif action == "SELL" and not trade.get("dealer_comment"):
            trade["dealer_comment"] = generate_sell_comment(trade, config)
        elif action == "SKIP_BUY" and not trade.get("dealer_comment"):
            trade["dealer_comment"] = generate_no_trade_comment(trade.get("skipped_reason") or "買付条件を満たしません。", config)
        elif action == "NO_BUY" and not trade.get("dealer_comment"):
            trade["dealer_comment"] = generate_no_trade_comment(trade.get("reason") or "本日は買付対象なし", config)
    if not portfolio_summary.get("dealer_comment"):
        portfolio_summary["dealer_comment"] = generate_daily_comment(portfolio_summary, selected_candidates, trades, config)


def execute_trade_for_date(provider_name: str, target_date_text: str) -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    scored_path = processed_profile_path(config, f"scored_candidates_{target_date_text}.json")
    scored_payload = read_json(scored_path)
    state_path = ROOT / "logs" / "portfolio" / profile_id_from(config) / "state.json"
    state = read_json(state_path) if state_path.exists() else initial_live_paper_state(config)
    scored_candidates = enrich_candidates_with_position_prices(scored_payload.get("scores", []), state, target_date_text)
    trades_path = ROOT / "logs" / "trades" / profile_id_from(config) / f"trades_{target_date_text}.json"
    portfolio_path = ROOT / "logs" / "portfolio" / profile_id_from(config) / f"portfolio_{target_date_text}.json"
    safety_path = ROOT / "logs" / "safety" / profile_id_from(config) / f"safety_{target_date_text}.json"

    if trades_path.exists() and portfolio_path.exists():
        cached_trades = read_json(trades_path).get("trades", [])
        selected_count = scored_payload.get("selected_count", sum(1 for item in scored_candidates if item.get("selected")))
        stale_no_buy_cache = (
            selected_count > 0
            and any(trade.get("action") == "NO_BUY" for trade in cached_trades)
            and not state.get("positions")
            and not state.get("closed_trades")
            and int(state.get("current_day", 0)) <= 1
        )
        stale_round_lot_cache = _round_lot_cache_is_stale(config, state, cached_trades)
        if not stale_no_buy_cache and not stale_round_lot_cache:
            portfolio_summary = read_json(portfolio_path)
            portfolio_summary.setdefault("config_version", config_version_from(config))
            safety_events = read_json(safety_path).get("safety_events", []) if safety_path.exists() else portfolio_summary.get("safety_events", [])
            portfolio_summary["safety_events"] = safety_events
            attach_commentary(portfolio_summary, cached_trades, scored_candidates, config)
            for trade in cached_trades:
                trade.setdefault("config_version", config_version_from(config))
            storage_trades = _trades_for_storage(cached_trades, config)
            write_json(trades_path, {"date": target_date_text, "provider": provider_name, "profile_id": profile_id_from(config), "profile_name": profile_name_from(config), "config_version": config_version_from(config), "storage_mode": _storage_save_mode(config), "trades": storage_trades})
            write_json(portfolio_path, portfolio_summary)
            write_json(safety_path, {"date": target_date_text, "provider": provider_name, "profile_id": profile_id_from(config), "profile_name": profile_name_from(config), "config_version": config_version_from(config), "safety_events": safety_events})
            save_portfolio_snapshot(config, ROOT, portfolio_summary)
            save_trades(config, ROOT, target_date_text, storage_trades)
            save_pending_orders(config, ROOT, state.get("pending_orders", []))
            save_safety_events(config, ROOT, target_date_text, safety_events)
            write_daily_ai_dataset(config, target_date_text)
            return {
                "state": state,
                "portfolio_summary": portfolio_summary,
                "trades": cached_trades,
                "safety_events": safety_events,
                "candidate_count": scored_payload.get("candidate_count", len(scored_candidates)),
                "selected_count": selected_count,
            }
        state = initial_live_paper_state(config)
        scored_candidates = enrich_candidates_with_position_prices(scored_payload.get("scores", []), state, target_date_text)

    updated_state, portfolio_summary, trades = execute_real_data_paper_trade(scored_candidates, state, config, target_date_text)
    attach_config_version(portfolio_summary, config)
    for trade in trades:
        trade.setdefault("config_version", config_version_from(config))
    attach_commentary(portfolio_summary, trades, scored_candidates, config)

    storage_trades = _trades_for_storage(trades, config)
    write_json(state_path, updated_state)
    write_json(trades_path, {"date": target_date_text, "provider": provider_name, "profile_id": profile_id_from(config), "profile_name": profile_name_from(config), "config_version": config_version_from(config), "storage_mode": _storage_save_mode(config), "trades": storage_trades})
    write_json(portfolio_path, portfolio_summary)
    write_json(safety_path, {"date": target_date_text, "provider": provider_name, "profile_id": profile_id_from(config), "profile_name": profile_name_from(config), "config_version": config_version_from(config), "safety_events": portfolio_summary.get("safety_events", [])})
    save_portfolio_snapshot(config, ROOT, portfolio_summary)
    save_trades(config, ROOT, target_date_text, storage_trades)
    save_pending_orders(config, ROOT, updated_state.get("pending_orders", []))
    save_safety_events(config, ROOT, target_date_text, portfolio_summary.get("safety_events", []))
    write_daily_ai_dataset(config, target_date_text)
    return {
        "state": updated_state,
        "portfolio_summary": portfolio_summary,
        "trades": trades,
        "safety_events": portfolio_summary.get("safety_events", []),
        "candidate_count": scored_payload.get("candidate_count", len(scored_candidates)),
        "selected_count": scored_payload.get("selected_count", sum(1 for item in scored_candidates if item.get("selected"))),
    }


def _round_lot_cache_is_stale(config: dict[str, Any], state: dict[str, Any], trades: list[dict[str, Any]]) -> bool:
    trading = config.get("trading", {})
    if not trading.get("use_round_lot"):
        return False
    lot_size = int(trading.get("round_lot_size", 100))
    if state.get("closed_trades") or int(state.get("current_day", 0)) > 1:
        return False
    for trade in trades:
        if trade.get("action") != "BUY":
            continue
        shares = int(trade.get("shares") or trade.get("quantity") or 0)
        if trade.get("use_round_lot") is not True:
            return True
        if shares > 0 and shares % lot_size != 0:
            return True
    for position in state.get("positions", []):
        shares = int(position.get("shares") or position.get("quantity") or 0)
        if shares > 0 and shares % lot_size != 0:
            return True
    return False


def write_real_reflections(target_date_text: str, trade_result: dict[str, Any]) -> Path:
    config = load_config(CONFIG_PATH)
    closed_trades = []
    for trade in trade_result["trades"]:
        if trade.get("action") != "SELL":
            continue
        closed_trades.append(
            {
                **trade,
                "buy_reason": trade.get("buy_reason", ""),
            }
        )
    reflection_log = generate_reflections(
        {
            "run_id": f"jquants-{target_date_text}",
            "date": target_date_text,
            "config_version": config_version_from(config),
            "closed_trades": closed_trades,
        },
        config,
    )
    attach_config_version(reflection_log, config)
    reflection_log["profile_id"] = profile_id_from(config)
    reflection_log["profile_name"] = profile_name_from(config)
    path = ROOT / "logs" / "reflections" / profile_id_from(config) / f"reflections_{target_date_text}.json"
    write_json(path, reflection_log)
    save_reflections(config, ROOT, reflection_log, closed_trades)
    return path


def write_backtest_reflections(backtest_dir: Path, target_date_text: str, trade_result: dict[str, Any]) -> Path:
    config = load_config(CONFIG_PATH)
    closed_trades = [
        {
            **trade,
            "buy_reason": trade.get("buy_reason", ""),
        }
        for trade in trade_result["trades"]
        if trade.get("action") == "SELL"
    ]
    reflection_log = generate_reflections(
        {
            "run_id": f"backtest-{target_date_text}",
            "date": target_date_text,
            "config_version": config_version_from(config),
            "closed_trades": closed_trades,
        },
        config,
    )
    attach_config_version(reflection_log, config)
    path = backtest_dir / f"reflections_{target_date_text}.json"
    write_json(path, reflection_log)
    save_reflections(config, ROOT, reflection_log, closed_trades)
    return path


def write_real_daily_markdown(
    target_date_text: str,
    trade_result: dict[str, Any],
    scoring_log: dict[str, Any],
) -> tuple[Path, Path]:
    config = load_config(CONFIG_PATH)
    migrate_report_articles_layout(config)
    summary = normalize_real_summary_for_markdown(trade_result["portfolio_summary"], target_date_text)
    summary["market_context"] = load_market_context_for_date(target_date_text, "jquants")
    attach_paper_trading_report_context(summary, config)
    paper_trade_log = normalize_real_trade_for_markdown(
        trade_result["state"],
        trade_result["trades"],
        target_date_text,
        trade_result.get("safety_events", []),
    )
    decisions = {
        "date": target_date_text,
        "decisions": [
            {
                "code": item["code"],
                "name": item["name"],
                "sector_name": item.get("sector_name"),
                "sector_momentum_score": item.get("sector_momentum_score"),
                "sector_rank": item.get("sector_rank"),
                "sector_comment": item.get("sector_comment"),
                "sector_score_adjustment": item.get("sector_score_adjustment"),
                "decision": "BUY" if item.get("selected") else "PASS",
                "reason": item.get("selection_reason") or item.get("selected_reason") or item.get("rejected_reason") or item.get("reason", ""),
                "total_score": item["total_score"],
                "technical_score": item.get("technical_score"),
                "candle_type": item.get("candle_type"),
                "candlestick_signals": item.get("candlestick_signals", []),
                "candlestick_score": item.get("candlestick_score"),
                "trend_score": item.get("trend_score"),
                "volume_score": item.get("volume_score"),
                "rsi_score": item.get("rsi_score"),
                "ma5": item.get("ma5"),
                "ma25": item.get("ma25"),
                "volume_ratio": item.get("volume_ratio"),
                "macd_hist": item.get("macd_hist"),
                "bb_position": item.get("bb_position"),
                "atr": item.get("atr"),
                "confidence": item["confidence"],
                "market_filter_applied": item.get("market_filter_applied", False),
                "market_regime": item.get("market_regime"),
                "market_filter_reason": item.get("market_filter_reason", ""),
                "dealer_comment": generate_buy_comment(item, config) if item.get("selected") else "",
            }
            for item in scoring_log.get("scores", [])
        ],
    }
    report_md = generate_daily_report(summary, paper_trade_log, decisions, config)
    article_md = generate_note_article(summary, paper_trade_log, config)
    report_path = _paper_report_path(config, target_date_text)
    article_path = _daily_article_path(config, target_date_text, profile_id_from(config))
    write_text(report_path, report_md)
    write_text(article_path, article_md)
    save_article(
        config,
        ROOT,
        {
            "date": target_date_text,
            "day": summary["day_number"],
            "title": _extract_markdown_title(article_md),
            "status": "draft",
            "path": str(article_path.relative_to(ROOT)),
            "config_version": config_version_from(config),
            "body": article_md,
        },
    )
    _update_article_index(article_path, target_date_text, "draft", config)
    return report_path, article_path


def migrate_report_articles_layout(config: dict[str, Any]) -> None:
    base_dir = ROOT / "reports" / "articles"
    if not base_dir.exists():
        return
    profile_id = profile_id_from(config)
    for path in base_dir.glob("*.md"):
        date_text = _date_text_from_filename(path.name)
        if not date_text:
            continue
        target = _daily_article_path(config, date_text, profile_id)
        if target == path:
            continue
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        _update_article_index(target, date_text, "draft", config)


def _date_text_from_filename(name: str) -> str | None:
    match = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})", name)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def write_backtest_daily_markdown(
    report_dir: Path,
    article_dir: Path,
    target_date_text: str,
    trade_result: dict[str, Any],
    scoring_log: dict[str, Any],
) -> tuple[Path, Path]:
    config = load_config(CONFIG_PATH)
    summary = normalize_real_summary_for_markdown(trade_result["portfolio_summary"], target_date_text)
    summary["market_context"] = load_market_context_for_date(target_date_text, "jquants")
    attach_paper_trading_report_context(summary, config)
    paper_trade_log = normalize_real_trade_for_markdown(
        trade_result["state"],
        trade_result["trades"],
        target_date_text,
        trade_result.get("safety_events", []),
    )
    decisions = {
        "date": target_date_text,
        "decisions": [
            {
                "code": item["code"],
                "name": item["name"],
                "sector_name": item.get("sector_name"),
                "sector_momentum_score": item.get("sector_momentum_score"),
                "sector_rank": item.get("sector_rank"),
                "sector_comment": item.get("sector_comment"),
                "sector_score_adjustment": item.get("sector_score_adjustment"),
                "decision": "BUY" if item.get("selected") else "PASS",
                "reason": item.get("selection_reason") or item.get("selected_reason") or item.get("rejected_reason") or item.get("reason", ""),
                "total_score": item["total_score"],
                "technical_score": item.get("technical_score"),
                "candle_type": item.get("candle_type"),
                "candlestick_signals": item.get("candlestick_signals", []),
                "candlestick_score": item.get("candlestick_score"),
                "trend_score": item.get("trend_score"),
                "volume_score": item.get("volume_score"),
                "rsi_score": item.get("rsi_score"),
                "ma5": item.get("ma5"),
                "ma25": item.get("ma25"),
                "volume_ratio": item.get("volume_ratio"),
                "macd_hist": item.get("macd_hist"),
                "bb_position": item.get("bb_position"),
                "atr": item.get("atr"),
                "confidence": item["confidence"],
                "market_filter_applied": item.get("market_filter_applied", False),
                "market_regime": item.get("market_regime"),
                "market_filter_reason": item.get("market_filter_reason", ""),
                "dealer_comment": generate_buy_comment(item, config) if item.get("selected") else "",
            }
            for item in scoring_log.get("scores", [])
        ],
    }
    report_md = generate_daily_report(summary, paper_trade_log, decisions, config)
    article_md = generate_note_article(summary, paper_trade_log, config)
    report_path = report_dir / f"day_{target_date_text}.md"
    article_path = article_dir / f"day_{target_date_text}.md"
    write_text(report_path, report_md)
    write_text(article_path, article_md)
    save_article(
        config,
        ROOT,
        {
            "date": target_date_text,
            "day": summary["day_number"],
            "title": _extract_markdown_title(article_md),
            "status": "draft",
            "path": str(article_path.relative_to(ROOT)),
            "config_version": config_version_from(config),
            "body": article_md,
        },
    )
    _update_article_index(article_path, target_date_text, "draft", config)
    return report_path, article_path


def write_backtest_report_markdown(
    report_dir: Path,
    target_date_text: str,
    trade_result: dict[str, Any],
    scoring_log: dict[str, Any],
) -> Path:
    config = load_config(CONFIG_PATH)
    summary = normalize_real_summary_for_markdown(trade_result["portfolio_summary"], target_date_text)
    summary["market_context"] = load_market_context_for_date(target_date_text, "jquants")
    attach_paper_trading_report_context(summary, config)
    paper_trade_log = normalize_real_trade_for_markdown(
        trade_result["state"],
        trade_result["trades"],
        target_date_text,
        trade_result.get("safety_events", []),
    )
    decisions = {"date": target_date_text, "decisions": _daily_decision_rows(scoring_log, config)}
    report_path = report_dir / f"day_{target_date_text}.md"
    write_text(report_path, generate_daily_report(summary, paper_trade_log, decisions, config))
    return report_path


def write_backtest_article_markdown(
    article_path: Path,
    target_date_text: str,
    trade_result: dict[str, Any],
) -> Path:
    config = load_config(CONFIG_PATH)
    summary = normalize_real_summary_for_markdown(trade_result["portfolio_summary"], target_date_text)
    summary["market_context"] = load_market_context_for_date(target_date_text, "jquants")
    attach_paper_trading_report_context(summary, config)
    paper_trade_log = normalize_real_trade_for_markdown(
        trade_result["state"],
        trade_result["trades"],
        target_date_text,
        trade_result.get("safety_events", []),
    )
    article_md = generate_note_article(summary, paper_trade_log, config)
    write_text(article_path, article_md)
    save_article(
        config,
        ROOT,
        {
            "date": target_date_text,
            "day": summary["day_number"],
            "title": _extract_markdown_title(article_md),
            "status": "draft",
            "path": str(article_path.relative_to(ROOT)),
            "config_version": config_version_from(config),
            "body": article_md,
        },
    )
    _update_article_index(article_path, target_date_text, "draft", config)
    return article_path


def write_fast_backtest_daily_artifacts(
    backtest_dir: Path,
    report_dir: Path,
    article_dir: Path,
    target_date_text: str,
    trade_result: dict[str, Any],
) -> tuple[Path, Path, Path]:
    reflection_path = backtest_dir / f"reflections_{target_date_text}.json"
    report_path = report_dir / f"day_{target_date_text}.md"
    article_path = article_dir / f"day_{target_date_text}.md"
    portfolio_summary = trade_result.get("portfolio_summary", {})
    trades = trade_result.get("trades", [])
    buy_count = sum(1 for trade in trades if trade.get("action") == "BUY")
    sell_count = sum(1 for trade in trades if trade.get("action") == "SELL")
    write_json(
        reflection_path,
        {
            "date": target_date_text,
            "skipped": True,
            "reason": "fast_analysis disabled daily reflection generation",
        },
    )
    write_text(
        report_path,
        "\n".join(
            [
                f"# Backtest Daily Report {target_date_text}",
                "",
                "fast_analysis: true",
                "detail: skipped daily markdown generation",
                f"total_assets: {portfolio_summary.get('total_assets')}",
                f"candidate_count: {trade_result.get('candidate_count', 0)}",
                f"selected_count: {trade_result.get('selected_count', 0)}",
                f"buy_count: {buy_count}",
                f"sell_count: {sell_count}",
                "",
            ]
        ),
    )
    write_text(
        article_path,
        "\n".join(
            [
                f"# Backtest Article {target_date_text}",
                "",
                "fast_analysis: true",
                "detail: skipped note article generation",
                "",
            ]
        ),
    )
    return reflection_path, report_path, article_path


def _daily_decision_rows(scoring_log: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "code": item["code"],
            "name": item["name"],
            "sector_name": item.get("sector_name"),
            "sector_momentum_score": item.get("sector_momentum_score"),
            "sector_rank": item.get("sector_rank"),
            "sector_comment": item.get("sector_comment"),
            "sector_score_adjustment": item.get("sector_score_adjustment"),
            "decision": "BUY" if item.get("selected") else "PASS",
            "reason": item.get("selection_reason") or item.get("selected_reason") or item.get("rejected_reason") or item.get("reason", ""),
            "total_score": item["total_score"],
            "technical_score": item.get("technical_score"),
            "candle_type": item.get("candle_type"),
            "candlestick_signals": item.get("candlestick_signals", []),
            "candlestick_score": item.get("candlestick_score"),
            "trend_score": item.get("trend_score"),
            "volume_score": item.get("volume_score"),
            "rsi_score": item.get("rsi_score"),
            "ma5": item.get("ma5"),
            "ma25": item.get("ma25"),
            "volume_ratio": item.get("volume_ratio"),
            "macd_hist": item.get("macd_hist"),
            "bb_position": item.get("bb_position"),
            "atr": item.get("atr"),
            "confidence": item["confidence"],
            "market_filter_applied": item.get("market_filter_applied", False),
            "market_regime": item.get("market_regime"),
            "market_filter_reason": item.get("market_filter_reason", ""),
            "dealer_comment": generate_buy_comment(item, config) if item.get("selected") else "",
        }
        for item in scoring_log.get("scores", [])
    ]


def normalize_real_summary_for_markdown(summary: dict[str, Any], target_date_text: str) -> dict[str, Any]:
    return {
        **summary,
        "date": target_date_text,
        "day": summary.get("day", 1),
        "day_number": summary.get("day", 1),
        "day_change": summary["daily_profit"],
        "day_change_pct": summary["daily_profit"] / (summary["total_assets"] - summary["daily_profit"]) if summary["total_assets"] != summary["daily_profit"] else 0,
        "cumulative_pnl": summary["cumulative_profit"],
        "cumulative_return_pct": summary["cumulative_profit_rate"],
        "max_drawdown_note": "日次total_assets履歴から過去ピーク比の最大下落率を計算する。",
    }


def attach_paper_trading_report_context(summary: dict[str, Any], config: dict[str, Any]) -> None:
    try:
        analysis = analyze_operation_data(config, ROOT)
    except (FileNotFoundError, ValueError, sqlite3.Error):
        return
    summary["walk_forward_validation"] = analysis.get("walk_forward_validation", {})


def _extract_markdown_title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def normalize_real_trade_for_markdown(
    state: dict[str, Any],
    trades: list[dict[str, Any]],
    target_date_text: str,
    safety_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    buy_orders = [
        {
            "code": trade["code"],
            "name": trade["name"],
            "sector_name": trade.get("sector_name", ""),
            "quantity": trade["shares"],
            "price": trade["entry_price"],
            "entry_date": trade["entry_date"],
            "dealer_comment": trade.get("dealer_comment", ""),
            "technical_score": trade.get("technical_score"),
            "trend_score": trade.get("trend_score"),
            "volume_score": trade.get("volume_score"),
            "rsi_score": trade.get("rsi_score"),
            "candlestick_score": trade.get("candlestick_score"),
            "candle_type": trade.get("candle_type"),
            "candlestick_signals": trade.get("candlestick_signals", []),
            "ma5": trade.get("ma5"),
            "ma25": trade.get("ma25"),
            "volume_ratio": trade.get("volume_ratio"),
            "sector_momentum_score": trade.get("sector_momentum_score"),
            "sector_rank": trade.get("sector_rank"),
            "sector_comment": trade.get("sector_comment"),
            "macd_hist": trade.get("macd_hist"),
            "bb_position": trade.get("bb_position"),
            "atr": trade.get("atr"),
        }
        for trade in trades
        if trade.get("action") == "BUY" and trade.get("order_status", trade.get("status")) == "FILLED"
    ]
    closed = [
        {
            **trade,
            "dealer_comment": trade.get("dealer_comment") or generate_sell_comment(trade),
            "reflection_comment": trade.get("reflection_comment") or generate_reflection_comment(trade),
        }
        for trade in trades
        if trade.get("action") == "SELL" and trade.get("order_status", trade.get("status")) == "FILLED"
    ]
    skipped_buys = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
    pending_orders = [trade for trade in trades if trade.get("order_status") == "PENDING" or trade.get("status") == "PENDING"]
    executed_orders = [trade for trade in trades if trade.get("order_status") == "FILLED" or trade.get("status") == "FILLED"]
    positions = [
        {
            "code": position["code"],
            "name": position["name"],
            "quantity": position["shares"],
            "market_value": position["market_value"],
            "unrealized_pnl": position.get("unrealized_profit", 0),
            "sector_name": position.get("sector_name", ""),
        }
        for position in state["positions"]
    ]
    return {
        "date": target_date_text,
        "day_number": state.get("current_day", 1),
        "orders": buy_orders,
        "pending_orders": pending_orders,
        "executed_orders": executed_orders,
        "skipped_buys": skipped_buys,
        "closed_trades": closed,
        "all_closed_trades": state.get("closed_trades", closed),
        "safety_events": safety_events or [],
        "positions": positions,
    }


def write_real_summary_csvs() -> tuple[Path, Path]:
    config = load_config(CONFIG_PATH)
    portfolio_dir = ROOT / "logs" / "portfolio" / profile_id_from(config)
    summaries = []
    for path in sorted(portfolio_dir.glob("portfolio_*.json")):
        summaries.append(read_json(path))
    summary_csv = ROOT / "reports" / profile_id_from(config) / "summary.csv"
    write_summary_csv(summary_csv, summaries)
    trades_csv, db_trade_count, _csv_trade_count = write_trades_csv_from_db(config)
    if db_trade_count == 0:
        state_path = portfolio_dir / "state.json"
        state = read_json(state_path) if state_path.exists() else {"closed_trades": []}
        write_trades_csv(trades_csv, state.get("closed_trades", []))
    return summary_csv, trades_csv


def load_trade_rows_for_csv(config: dict[str, Any], root: Path = ROOT) -> list[dict[str, Any]]:
    initialize_database(config, root)
    db_path = get_database_path(config, root)
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT *
            FROM trades
            WHERE profile_id = ?
              AND order_status = 'FILLED'
              AND action IN ('BUY', 'SELL')
            ORDER BY entry_date, exit_date, id
            """,
            (profile_id_from(config),),
        ).fetchall()
    return [_normalize_trade_row_for_csv(dict(row)) for row in rows]


def _normalize_trade_row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("signal_date"):
        trade_id = str(row.get("trade_id") or "")
        first_token = trade_id.split("_", 1)[0]
        try:
            date.fromisoformat(first_token)
        except ValueError:
            first_token = ""
        if first_token and first_token != str(row.get("entry_date") or ""):
            row["signal_date"] = first_token
    return row


def write_trades_csv_from_db(config: dict[str, Any], root: Path = ROOT) -> tuple[Path, int, int]:
    rows = load_trade_rows_for_csv(config, root)
    trades_csv = root / "reports" / profile_id_from(config) / "trades.csv"
    write_trades_csv(trades_csv, rows)
    return trades_csv, len(rows), count_csv_data_rows(trades_csv)


def ensure_price_history_for_backtest(
    provider_name: str,
    start_date: date,
    end_date: date,
    price_fetch_min_start: date | None = None,
) -> dict[str, Any]:
    if provider_name != "jquants":
        raise SystemExit("backtest mode currently supports --provider jquants only.")

    requested_start_date = start_date

    stock_master_path = _listed_stock_master_path()
    if not stock_master_path.exists():
        print("fetch-period-prices listed stock cache: missing; fetching J-Quants master")
        try:
            run_list_stocks(provider_name)
        except (RuntimeError, SystemExit) as exc:
            cached_dates = available_cached_price_dates(start_date, end_date)
            if cached_dates:
                print(
                    "fetch-period-prices warning: prime stock master fetch failed; "
                    f"continuing with {len(cached_dates)} cached backtest day(s). reason={exc}"
                )
                return {
                    "price_fetch_requested_start": requested_start_date.isoformat(),
                    "price_fetch_clamped_start": start_date.isoformat(),
                    "first_fetch_attempt_date": None,
                    "target_business_days": len(cached_dates),
                    "warning": str(exc),
                }
            raise

    config = load_config(CONFIG_PATH)
    stock_by_code = _allowed_stock_master_by_code(config)
    prime_codes = set(stock_by_code)
    if not prime_codes:
        raise SystemExit(_market_filter_empty_message(config))
    prices_supported_start = jquants_earliest_supported_date(config, "prices")
    effective_start_date = max(
        start_date,
        price_fetch_min_start or start_date,
        prices_supported_start or start_date,
    )
    if effective_start_date != start_date:
        print(
            "warning: price fetch start-date clamped; "
            f"requested={start_date.isoformat()} effective={effective_start_date.isoformat()}"
        )
        start_date = effective_start_date
    requested_lookback_start = previous_business_dates(start_date, 30)[0]
    lookback_start = max(
        requested_lookback_start,
        price_fetch_min_start or requested_lookback_start,
        prices_supported_start or requested_lookback_start,
    )
    if lookback_start != requested_lookback_start:
        print(
            "fetch-period-prices lookback clamped: "
            f"requested_lookback_start={requested_lookback_start.isoformat()} "
            f"effective_lookback_start={lookback_start.isoformat()}"
        )
    fetch_dates = business_dates_between(lookback_start, end_date)
    target_dates = business_dates_between(start_date, end_date)
    first_fetch_attempt_date = fetch_dates[0].isoformat() if fetch_dates else None
    audit = {
        "price_fetch_requested_start": requested_start_date.isoformat(),
        "price_fetch_min_start": price_fetch_min_start.isoformat() if price_fetch_min_start else None,
        "price_fetch_clamped_start": lookback_start.isoformat(),
        "first_fetch_attempt_date": first_fetch_attempt_date,
        "price_fetch_end": end_date.isoformat(),
        "target_business_days": len(fetch_dates),
        "trade_business_days": len(target_dates),
    }
    print(f"price_fetch_requested_start: {audit['price_fetch_requested_start']}")
    print(f"price_fetch_min_start: {audit['price_fetch_min_start'] or '-'}")
    print(f"price_fetch_clamped_start: {audit['price_fetch_clamped_start']}")
    print(f"first_fetch_attempt_date: {audit['first_fetch_attempt_date'] or '-'}")
    print(f"fetch-period-prices target period: {lookback_start.isoformat()} to {end_date.isoformat()}")
    print(f"fetch-period-prices target business days: {len(fetch_dates)}")
    cached_dates = [fetch_date for fetch_date in fetch_dates if load_cached_prime_prices(fetch_date) is not None]
    no_data_cache = load_no_data_days_cache()
    unsupported_cache = load_unsupported_days_cache()
    no_data_dates = [
        fetch_date
        for fetch_date in fetch_dates
        if load_cached_prime_prices(fetch_date) is None and no_data_cache_entry(fetch_date, no_data_cache) is not None
    ]
    unsupported_dates = [
        fetch_date
        for fetch_date in fetch_dates
        if (
            load_cached_prime_prices(fetch_date) is None
            and no_data_cache_entry(fetch_date, no_data_cache) is None
            and unsupported_cache_entry(fetch_date, unsupported_cache) is not None
        )
    ]
    missing_dates = [
        fetch_date
        for fetch_date in fetch_dates
        if (
            load_cached_prime_prices(fetch_date) is None
            and no_data_cache_entry(fetch_date, no_data_cache) is None
            and unsupported_cache_entry(fetch_date, unsupported_cache) is None
        )
    ]
    print(
        "fetch-period-prices price cache: "
        f"cached={len(cached_dates)} no_data={len(no_data_dates)} "
        f"unsupported={len(unsupported_dates)} missing={len(missing_dates)} total={len(fetch_dates)} "
        f"lookback_start={lookback_start.isoformat()}"
    )
    for fetch_date in cached_dates:
        print(f"fetch-period-prices cache hit: {fetch_date.isoformat()}")
    for fetch_date in no_data_dates:
        entry = no_data_cache_entry(fetch_date, no_data_cache) or {}
        print(f"fetch-period-prices skip no-data cache: {fetch_date.isoformat()} reason={entry.get('reason', 'unknown')}")
    for fetch_date in unsupported_dates:
        entry = unsupported_cache_entry(fetch_date, unsupported_cache) or {}
        print(f"fetch-period-prices skip unsupported cache: {fetch_date.isoformat()} reason={entry.get('reason', 'unknown')}")
    if fetch_dates and not missing_dates:
        print("backtest price history: using cached price files")
        return audit

    try:
        provider = JQuantsDataProvider(
            ROOT / ".env",
            timeout_seconds=int(config.get("jquants", {}).get("request_timeout_seconds", 20)),
            plan=_jquants_plan(config),
            requests_per_minute=_jquants_requests_per_minute(config),
            parallel_fetch=_jquants_parallel_fetch(config),
            max_parallel_requests=_jquants_max_parallel_requests(config),
        )
    except RuntimeError as exc:
        usable_dates = available_cached_price_dates(start_date, end_date)
        if usable_dates:
            print(
                "fetch-period-prices warning: J-Quants provider unavailable; "
                f"continuing with {len(usable_dates)} cached backtest day(s). reason={exc}"
            )
            audit["warning"] = str(exc)
            return audit
        raise
    try:
        fetch_price_history(
            provider,
            end_date,
            prime_codes,
            lookback_business_days=len(fetch_dates),
            rate_limit_per_minute=_jquants_requests_per_minute(config),
            fetch_dates=fetch_dates,
            continue_on_error=True,
            verbose=True,
            stop_on_consecutive_unsupported=False,
        )
        _print_fetch_statistics(provider)
    except KeyboardInterrupt:
        print("fetch-period-prices interrupted. Existing cache is preserved.")
        _print_fetch_statistics(provider)
        raise

    usable_dates = available_cached_price_dates(start_date, end_date)
    if usable_dates:
        print(
            "fetch-period-prices completed with cached usable days: "
            f"{len(usable_dates)}/{len(target_dates)}"
        )
        return audit
    print("fetch-period-prices completed, but no usable cached dates were found for the target period.")
    return audit


def available_cached_price_dates(start_date: date, end_date: date) -> list[date]:
    dates = []
    for current in business_dates_between(start_date, end_date):
        cached_rows = load_cached_prime_prices(current)
        if cached_rows:
            dates.append(current)
    return dates


def all_cached_price_dates() -> list[date]:
    dates: set[date] = set()
    for directory in [ROOT / "data" / "raw", ROOT / "data" / "cache" / "jquants" / "prices"]:
        if not directory.exists():
            continue
        for path in directory.glob("prices_*.json"):
            date_text = path.stem.removeprefix("prices_")
            try:
                dates.add(date.fromisoformat(date_text))
            except ValueError:
                continue
        if directory.name == "prices":
            for path in directory.glob("*.json"):
                try:
                    dates.add(date.fromisoformat(path.stem))
                except ValueError:
                    continue
    return sorted(dates)


def build_backtest_date_range_audit(
    *,
    config: dict[str, Any],
    requested_start_date: date,
    requested_end_date: date,
    effective_trade_start_date: date,
    effective_trade_end_date: date,
    indicator_fetch_start_date: date,
    price_history_dates: list[date],
    trading_dates: list[date],
    processed_dates: list[str],
    skipped_days: list[dict[str, str]],
    all_trades: list[dict[str, Any]],
    price_fetch_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_trading_days = business_dates_between(effective_trade_start_date, effective_trade_end_date)
    latest_available_price_date = price_history_dates[-1].isoformat() if price_history_dates else None
    trade_dates = sorted(
        {
            str(value)
            for trade in all_trades
            for value in [trade.get("entry_date") or trade.get("date"), trade.get("exit_date")]
            if value
        }
    )
    processed_set = set(processed_dates)
    missing_processed_dates = [item.isoformat() for item in trading_dates if item.isoformat() not in processed_set]
    processed_data_audit = build_processed_data_audit(config, trading_dates)
    coverage_audit = build_backtest_coverage_audit(
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
        price_history_dates=price_history_dates,
        trading_dates=trading_dates,
        expected_trading_days=expected_trading_days,
        processed_data_audit=processed_data_audit,
        trade_dates=trade_dates,
        price_fetch_audit=price_fetch_audit or {},
    )
    execution_audit = build_backtest_execution_audit(
        processed_dates=processed_dates,
        skipped_days=skipped_days,
        trading_dates=trading_dates,
        expected_trading_days=expected_trading_days,
        processed_data_audit=processed_data_audit,
        all_trades=all_trades,
        effective_trade_end_date=effective_trade_end_date,
    )
    coverage_ok = bool(latest_available_price_date and latest_available_price_date >= requested_end_date.isoformat())
    audit = {
        "requested_start_date": requested_start_date.isoformat(),
        "requested_end_date": requested_end_date.isoformat(),
        "effective_trade_start_date": effective_trade_start_date.isoformat(),
        "effective_trade_end_date": effective_trade_end_date.isoformat(),
        "indicator_fetch_start_date": indicator_fetch_start_date.isoformat(),
        "price_fetch_requested_start": (price_fetch_audit or {}).get("price_fetch_requested_start"),
        "price_fetch_clamped_start": (price_fetch_audit or {}).get("price_fetch_clamped_start"),
        "first_fetch_attempt_date": (price_fetch_audit or {}).get("first_fetch_attempt_date"),
        "raw_price_first_date": price_history_dates[0].isoformat() if price_history_dates else None,
        "raw_price_last_date": latest_available_price_date,
        "first_price_date": price_history_dates[0].isoformat() if price_history_dates else None,
        "last_price_date": latest_available_price_date,
        "first_trading_day": trading_dates[0].isoformat() if trading_dates else None,
        "last_trading_day": trading_dates[-1].isoformat() if trading_dates else None,
        "target_trading_days": len(expected_trading_days),
        "target_trading_days_source": "raw_price_cache",
        "detected_trading_days": len(trading_dates),
        "processed_first_date": processed_dates[0] if processed_dates else None,
        "processed_last_date": processed_dates[-1] if processed_dates else None,
        "missing_processed_dates_count": len(missing_processed_dates),
        "first_missing_processed_date": missing_processed_dates[0] if missing_processed_dates else None,
        "last_missing_processed_date": missing_processed_dates[-1] if missing_processed_dates else None,
        "processed_days": len(processed_dates),
        "skipped_days": len(skipped_days),
        "skipped_day_details": skipped_days,
        "last_processed_day": processed_dates[-1] if processed_dates else None,
        "first_trade_date": trade_dates[0] if trade_dates else None,
        "last_trade_date": trade_dates[-1] if trade_dates else None,
        "data_coverage": {
            "prices": {
                "requested_end_date": requested_end_date.isoformat(),
                "latest_available_price_date": latest_available_price_date,
                "coverage_ok": coverage_ok,
                "warning": ""
                if coverage_ok
                else (
                    "price data ends before requested_end_date; backtest can only process cached/fetched price dates"
                    if latest_available_price_date
                    else "no price data available"
                ),
            }
        },
        "hardcoded_date_audit": hardcoded_date_audit(),
        "processed_data_audit": processed_data_audit,
        "backtest_coverage_audit": coverage_audit,
        "backtest_execution_audit": execution_audit,
    }
    if audit["last_processed_day"] and audit["last_processed_day"] < requested_end_date.isoformat():
        audit["effective_range_warning"] = (
            "processed days end before requested_end_date; check latest_available_price_date and fetch-period-prices logs"
        )
    else:
        audit["effective_range_warning"] = ""
    return audit


def build_backtest_coverage_audit(
    *,
    requested_start_date: date,
    requested_end_date: date,
    price_history_dates: list[date],
    trading_dates: list[date],
    expected_trading_days: list[date],
    processed_data_audit: dict[str, Any],
    trade_dates: list[str],
    price_fetch_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    first_price_date = price_history_dates[0].isoformat() if price_history_dates else None
    last_price_date = price_history_dates[-1].isoformat() if price_history_dates else None
    raw_cache_dates = all_cached_price_dates()
    raw_cache_first_date = raw_cache_dates[0].isoformat() if raw_cache_dates else None
    raw_cache_last_date = raw_cache_dates[-1].isoformat() if raw_cache_dates else None
    first_indicator_date = (
        processed_data_audit.get("indicators_first_date_in_range")
        or processed_data_audit.get("indicators_first_date")
    )
    last_indicator_date = (
        processed_data_audit.get("indicators_last_date_in_range")
        or processed_data_audit.get("indicators_last_date")
    )
    first_candidate_date = (
        processed_data_audit.get("candidates_first_date_in_range")
        or processed_data_audit.get("candidates_first_date")
    )
    last_candidate_date = (
        processed_data_audit.get("candidates_last_date_in_range")
        or processed_data_audit.get("candidates_last_date")
    )
    warnings = _backtest_coverage_warnings(
        requested_start=requested_start_date.isoformat(),
        requested_end=requested_end_date.isoformat(),
        first_price=first_price_date,
        last_price=last_price_date,
        first_candidate=first_candidate_date,
        last_candidate=last_candidate_date,
        first_trade=trade_dates[0] if trade_dates else None,
    )
    coverage_ratio = round(len(trading_dates) / len(expected_trading_days), 4) if expected_trading_days else 0.0
    return {
        "requested_start_date": requested_start_date.isoformat(),
        "requested_end_date": requested_end_date.isoformat(),
        "price_fetch_requested_start": (price_fetch_audit or {}).get("price_fetch_requested_start"),
        "price_fetch_clamped_start": (price_fetch_audit or {}).get("price_fetch_clamped_start"),
        "first_fetch_attempt_date": (price_fetch_audit or {}).get("first_fetch_attempt_date"),
        "actual_start_date": first_price_date,
        "actual_end_date": last_price_date,
        "raw_cache_first_date": raw_cache_first_date,
        "raw_cache_last_date": raw_cache_last_date,
        "first_price_date": first_price_date,
        "last_price_date": last_price_date,
        "first_indicator_date": first_indicator_date,
        "last_indicator_date": last_indicator_date,
        "first_candidate_date": first_candidate_date,
        "last_candidate_date": last_candidate_date,
        "first_trade_date": trade_dates[0] if trade_dates else None,
        "last_trade_date": trade_dates[-1] if trade_dates else None,
        "candidate_days": processed_data_audit.get("candidate_days_in_range", 0),
        "trade_days": len(set(trade_dates)),
        "price_days": len(trading_dates),
        "expected_business_days": len(expected_trading_days),
        "coverage_ratio": coverage_ratio,
        "coverage_warnings": warnings,
        "coverage_warning": "; ".join(warnings) if warnings else "",
    }


def _backtest_coverage_warnings(
    *,
    requested_start: str,
    requested_end: str,
    first_price: str | None,
    last_price: str | None,
    first_candidate: str | None,
    last_candidate: str | None,
    first_trade: str | None,
) -> list[str]:
    warnings: list[str] = []
    if not first_price:
        return ["no price data available"]
    if requested_start < first_price:
        warnings.append("requested_start_date is earlier than first_price_date")
        warnings.append("historical price coverage incomplete")
    if last_price and last_price < requested_end:
        warnings.append("last_price_date is earlier than requested_end_date")
    if first_candidate and first_price and first_candidate > first_price:
        warnings.append("first_candidate_date is later than first_price_date")
    if last_candidate and last_price and last_candidate < last_price:
        warnings.append("last_candidate_date is earlier than last_price_date")
    if first_trade and first_candidate and first_trade > first_candidate:
        warnings.append("first_trade_date is later than first_candidate_date")
    return warnings


def build_backtest_execution_audit(
    *,
    processed_dates: list[str],
    skipped_days: list[dict[str, str]],
    trading_dates: list[date],
    expected_trading_days: list[date],
    processed_data_audit: dict[str, Any],
    all_trades: list[dict[str, Any]],
    effective_trade_end_date: date,
) -> dict[str, Any]:
    expected_processing_end_date = expected_trading_days[-1].isoformat() if expected_trading_days else effective_trade_end_date.isoformat()
    actual_trading_days = len(trading_dates)
    trade_dates = sorted(
        {
            str(value)
            for trade in all_trades
            for value in [trade.get("entry_date") or trade.get("date"), trade.get("exit_date")]
            if value
        }
    )
    last_processed_day = processed_dates[-1] if processed_dates else None
    limited_reason = _backtest_date_range_limited_reason(
        last_processed_day=last_processed_day,
        expected_processing_end_date=expected_processing_end_date,
        processed_data_audit=processed_data_audit,
        actual_trading_days=actual_trading_days,
    )
    status = "ERROR" if limited_reason != "none" else "OK"
    return {
        "status": status,
        "first_processed_day": processed_dates[0] if processed_dates else None,
        "last_processed_day": last_processed_day,
        "processed_days": len(processed_dates),
        "skipped_days": len(skipped_days),
        "first_indicator_date": processed_data_audit.get("indicators_first_date_in_range") or processed_data_audit.get("indicators_first_date"),
        "last_indicator_date": processed_data_audit.get("indicators_last_date_in_range") or processed_data_audit.get("indicators_last_date"),
        "first_candidate_date": processed_data_audit.get("candidates_first_date_in_range") or processed_data_audit.get("candidates_first_date"),
        "last_candidate_date": processed_data_audit.get("candidates_last_date_in_range") or processed_data_audit.get("candidates_last_date"),
        "first_scored_candidate_date": processed_data_audit.get("scored_candidates_first_date_in_range") or processed_data_audit.get("scored_candidates_first_date"),
        "last_scored_candidate_date": processed_data_audit.get("scored_candidates_last_date_in_range") or processed_data_audit.get("scored_candidates_last_date"),
        "first_trade_date": trade_dates[0] if trade_dates else None,
        "last_trade_date": trade_dates[-1] if trade_dates else None,
        "target_trading_days": len(expected_trading_days),
        "actual_trading_days": actual_trading_days,
        "effective_end_date": effective_trade_end_date.isoformat(),
        "expected_processing_end_date": expected_processing_end_date,
        "date_range_limited_reason": limited_reason,
    }


def _backtest_date_range_limited_reason(
    *,
    last_processed_day: str | None,
    expected_processing_end_date: str,
    processed_data_audit: dict[str, Any],
    actual_trading_days: int,
) -> str:
    if actual_trading_days == 0:
        return "no_actual_trading_days"
    if not last_processed_day:
        return "no_processed_days"
    if last_processed_day < expected_processing_end_date:
        return "processed_ends_before_expected_processing_end_date"
    candidates_last = processed_data_audit.get("candidates_last_date")
    scored_last = processed_data_audit.get("scored_candidates_last_date")
    indicators_last = processed_data_audit.get("indicators_last_date")
    if indicators_last and candidates_last and candidates_last < indicators_last:
        return "candidates_end_before_indicators"
    if candidates_last and scored_last and scored_last < candidates_last:
        return "scored_candidates_end_before_candidates"
    if processed_data_audit.get("dates_with_indicators_but_no_candidates_count"):
        return "indicators_without_candidates"
    if processed_data_audit.get("dates_with_candidates_but_no_scored_count"):
        return "candidates_without_scored_candidates"
    return "none"


def build_processed_data_audit(config: dict[str, Any], trading_dates: list[date]) -> dict[str, Any]:
    profile_dir = ROOT / "data" / "processed" / profile_id_from(config)
    expected_set = {item.isoformat() for item in trading_dates}
    indicators = _processed_stage_dates(profile_dir, "indicators")
    candidates = _processed_stage_dates(profile_dir, "candidates")
    scored = _processed_stage_dates(profile_dir, "scored_candidates")
    indicators_in_range = [item for item in indicators if item in expected_set]
    candidates_in_range = [item for item in candidates if item in expected_set]
    scored_in_range = [item for item in scored if item in expected_set]
    candidates_set = set(candidates)
    scored_set = set(scored)
    dates_with_indicators_but_no_candidates = [item for item in indicators if item in expected_set and item not in candidates_set]
    dates_with_candidates_but_no_scored = [item for item in candidates if item in expected_set and item not in scored_set]
    return {
        "indicators_first_date": indicators[0] if indicators else None,
        "indicators_last_date": indicators[-1] if indicators else None,
        "candidates_first_date": candidates[0] if candidates else None,
        "candidates_last_date": candidates[-1] if candidates else None,
        "scored_candidates_first_date": scored[0] if scored else None,
        "scored_candidates_last_date": scored[-1] if scored else None,
        "indicators_first_date_in_range": indicators_in_range[0] if indicators_in_range else None,
        "indicators_last_date_in_range": indicators_in_range[-1] if indicators_in_range else None,
        "candidates_first_date_in_range": candidates_in_range[0] if candidates_in_range else None,
        "candidates_last_date_in_range": candidates_in_range[-1] if candidates_in_range else None,
        "scored_candidates_first_date_in_range": scored_in_range[0] if scored_in_range else None,
        "scored_candidates_last_date_in_range": scored_in_range[-1] if scored_in_range else None,
        "indicators_count": len(indicators),
        "candidates_file_count": len(candidates),
        "scored_candidates_file_count": len(scored),
        "indicator_days_in_range": len(indicators_in_range),
        "candidate_days_in_range": len(candidates_in_range),
        "scored_candidate_days_in_range": len(scored_in_range),
        "dates_with_indicators_but_no_candidates": dates_with_indicators_but_no_candidates,
        "dates_with_candidates_but_no_scored": dates_with_candidates_but_no_scored,
        "dates_with_indicators_but_no_candidates_count": len(dates_with_indicators_but_no_candidates),
        "dates_with_candidates_but_no_scored_count": len(dates_with_candidates_but_no_scored),
    }


def _processed_stage_dates(profile_dir: Path, stage: str) -> list[str]:
    prefix = f"{stage}_"
    dates = []
    for path in sorted(profile_dir.glob(f"{prefix}*.json")):
        date_text = path.stem.removeprefix(prefix)
        try:
            date.fromisoformat(date_text)
        except ValueError:
            continue
        dates.append(date_text)
    return sorted(dates)


def hardcoded_date_audit(target: str = "2026-03-06") -> dict[str, Any]:
    roots = [ROOT / "config", ROOT / "src", ROOT / "docs", ROOT / "reports", ROOT / "README.md"]
    matches: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in paths:
            try:
                if target in path.read_text(encoding="utf-8", errors="ignore"):
                    matches.append(str(path.relative_to(ROOT)))
            except OSError:
                continue
    return {
        "target": target,
        "match_count": len(matches),
        "matches_sample": matches[:30],
        "warning": f"{target} remains in config/src/docs/reports/README" if matches else "",
    }


def business_dates_between(start_date: date, end_date: date) -> list[date]:
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def enrich_candidates_with_position_prices(
    scored_candidates: list[dict[str, Any]],
    state: dict[str, Any],
    target_date_text: str,
) -> list[dict[str, Any]]:
    enriched = list(scored_candidates)
    existing_codes = {item["code"] for item in enriched}
    held_codes = {position["code"] for position in state.get("positions", [])}
    missing_codes = held_codes - existing_codes
    if not missing_codes:
        return enriched

    indicators_path = ROOT / "data" / "processed" / f"indicators_{target_date_text}.json"
    if indicators_path.exists():
        indicators = read_json(indicators_path).get("indicators", [])
    else:
        indicators = load_cached_prime_prices(date.fromisoformat(target_date_text))
    by_code = {item["code"]: item for item in indicators}
    for code in sorted(missing_codes):
        indicator = by_code.get(code)
        if not indicator:
            continue
        enriched.append(
            {
                "code": indicator["code"],
                "name": indicator.get("name", indicator["code"]),
                "sector_name": indicator.get("sector_name", ""),
                "date": target_date_text,
                "open": indicator.get("open"),
                "high": indicator.get("high"),
                "low": indicator.get("low"),
                "close": indicator["close"],
                "volume": indicator.get("volume"),
                "ma5": indicator.get("ma5"),
                "ma25": indicator.get("ma25"),
                "volume_ratio": indicator.get("volume_ratio"),
                "rsi": indicator.get("rsi"),
                "macd": indicator.get("macd"),
                "macd_signal": indicator.get("macd_signal"),
                "macd_hist": indicator.get("macd_hist"),
                "bb_upper": indicator.get("bb_upper"),
                "bb_middle": indicator.get("bb_middle"),
                "bb_lower": indicator.get("bb_lower"),
                "bb_position": indicator.get("bb_position"),
                "atr": indicator.get("atr"),
                "turnover_value": indicator.get("turnover_value"),
                "five_day_volatility": indicator.get("five_day_volatility"),
                "candle_type": indicator.get("candle_type"),
                "candlestick_signals": indicator.get("candlestick_signals", []),
                "sector_momentum_score": indicator.get("sector_momentum_score"),
                "sector_rank": indicator.get("sector_rank"),
                "sector_comment": indicator.get("sector_comment", ""),
                "total_score": 0,
                "technical_score": 0,
                "confidence": 0.0,
                "rank": 0,
                "selected": False,
                "reason": "保有銘柄の価格更新用データ",
                "selection_reason": "",
                "selected_reason": "",
                "rejected_reason": "保有銘柄の価格更新用データのため新規買付対象外",
                "source_provider": "jquants",
                "fallback": False,
            }
        )
    return enriched


def write_backtest_summary(
    range_key: str,
    start_date_text: str,
    end_date_text: str,
    config: dict[str, Any],
    state: dict[str, Any],
    daily_summaries: list[dict[str, Any]],
    all_trades: list[dict[str, Any]],
    backtest_dir: Path,
    date_resolution: dict[str, Any] | None = None,
    date_range_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    initial_capital = float(config["portfolio"]["initial_cash"])
    final_assets = float(state.get("total_assets", initial_capital))
    closed_trades = state.get("closed_trades", [])
    period_profit = calculate_period_profit_summary(closed_trades, config)
    pf_metrics = profit_factor_metrics(all_trades)
    gross_profit_total = pf_metrics["gross_profit_total"]
    gross_win_total = pf_metrics["gross_win_total"]
    gross_loss_total = pf_metrics["gross_loss_total"]
    net_cumulative_profit = period_profit["net_cumulative_profit"]
    win_rate = pf_metrics["win_rate"]
    take_profit_count = sum(1 for trade in closed_trades if trade.get("exit_reason") == "利確")
    stop_loss_count = sum(1 for trade in closed_trades if trade.get("exit_reason") == "損切り")
    max_holding_exit_count = sum(1 for trade in closed_trades if trade.get("exit_reason") == "最大保有期間到達")
    best_trade = max(closed_trades, key=lambda item: item.get("profit", 0), default=None)
    worst_trade = min(closed_trades, key=lambda item: item.get("profit", 0), default=None)
    daily_asset_curve = [
        {
            "date": item["date"],
            "day": item["day"],
            "total_assets": item["total_assets"],
            "cumulative_profit": item["cumulative_profit"],
            "max_drawdown": item["max_drawdown"],
        }
        for item in daily_summaries
    ]
    selected_count_total = 0
    no_trade_days = 0
    for scoring_file in sorted(backtest_dir.glob("scoring_*.json")):
        scoring = read_json(scoring_file)
        selected_count = len(scoring.get("selected", []))
        selected_count_total += selected_count
        if selected_count == 0:
            no_trade_days += 1
    net_cumulative_profit_rate = round(net_cumulative_profit / initial_capital, 4) if initial_capital else None
    execution_model = _backtest_execution_model(config)
    execution_model.update(_backtest_execution_model_stats(all_trades, execution_model))
    resolved_dates = _date_resolution_with_coverage(
        date_resolution
        or {
            "requested_start_date": start_date_text,
            "requested_end_date": end_date_text,
            "effective_start_date": start_date_text,
            "effective_end_date": end_date_text,
            "start_date_source": "default",
            "end_date_source": "default",
        },
        date_range_audit or {},
    )
    summary = {
        "start_date": start_date_text,
        "end_date": end_date_text,
        "date_resolution": resolved_dates,
        "date_range_audit": date_range_audit or {},
        "execution_model": execution_model,
        "market_coverage": build_market_coverage_summary(backtest_dir),
        "profile_id": profile_id_from(config),
        "profile_name": profile_name_from(config),
        "provider": "jquants",
        "broker": config.get("broker", {}).get("provider", "paper"),
        "openai": "disabled" if _is_rule_based_backtest(config) else "enabled",
        "config_version": config_version_from(config),
        "initial_capital": initial_capital,
        "final_assets": round(final_assets, 2),
        "cumulative_profit": round(final_assets - initial_capital, 2),
        "cumulative_profit_rate": round((final_assets - initial_capital) / initial_capital, 4),
        "gross_cumulative_profit": period_profit["gross_cumulative_profit"],
        "net_cumulative_profit": net_cumulative_profit,
        "net_cumulative_profit_rate": net_cumulative_profit_rate,
        "total_commission": period_profit["total_commission"],
        "estimated_tax_total": period_profit["estimated_tax_total"],
        "profit_factor": pf_metrics["profit_factor"],
        "gross_profit_total": gross_profit_total,
        "gross_win_total": gross_win_total,
        "gross_loss_total": gross_loss_total,
        "win_rate": win_rate,
        "closed_trade_count": pf_metrics["closed_trade_count"],
        "win_count": pf_metrics["win_count"],
        "loss_count": pf_metrics["loss_count"],
        "excluded_order_event_count": pf_metrics["excluded_order_event_count"],
        "total_trades": pf_metrics["total_trades"],
        "closed_trades_count": len(closed_trades),
        "take_profit_count": take_profit_count,
        "stop_loss_count": stop_loss_count,
        "max_holding_exit_count": max_holding_exit_count,
        "no_trade_days": no_trade_days,
        "selected_count_total": selected_count_total,
        "max_drawdown": daily_summaries[-1]["max_drawdown"] if daily_summaries else 0.0,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "daily_asset_curve": daily_asset_curve,
        "dealer_comment": _backtest_dealer_comment(final_assets, initial_capital, closed_trades),
    }
    integrity_audit = build_backtest_integrity_audit(config, all_trades, backtest_dir, date_range_audit or {})
    summary["backtest_integrity_audit"] = integrity_audit
    summary["evaluation_label"] = "final_evaluation" if integrity_audit.get("overall_status") == "OK" else "experimental"
    report_json_path = ROOT / "reports" / profile_id_from(config) / f"backtest_{range_key}.json"
    report_md_path = ROOT / "reports" / profile_id_from(config) / f"backtest_{range_key}.md"
    rule_based_90d_report_path = ROOT / "reports" / "backtests" / f"rule_based_90d_summary_{range_key}.md"
    log_summary_path = backtest_dir / "backtest_summary.json"
    summary["report_json_path"] = str(report_json_path)
    summary["report_markdown_path"] = str(report_md_path)
    summary["rule_based_90d_report_path"] = str(rule_based_90d_report_path)
    summary["log_summary_path"] = str(log_summary_path)

    write_json(report_json_path, summary)
    write_json(log_summary_path, {**summary, "all_trades": all_trades, "state": state})
    write_text(report_md_path, render_backtest_summary_markdown(summary, config))
    if _is_rule_based_backtest(config):
        write_text(rule_based_90d_report_path, render_rule_based_90d_summary_markdown(summary))
    write_summary_csv(backtest_dir / "summary.csv", daily_summaries)
    write_trades_csv(backtest_dir / "trades.csv", closed_trades)
    return summary


def _date_resolution_with_coverage(
    date_resolution: dict[str, Any],
    date_range_audit: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(date_resolution)
    coverage = date_range_audit.get("backtest_coverage_audit", {}) if isinstance(date_range_audit, dict) else {}
    first_price_date = coverage.get("first_price_date") or date_range_audit.get("first_trading_day") or date_range_audit.get("first_price_date")
    requested_start = str(resolved.get("requested_start_date") or resolved.get("effective_start_date") or "")
    if first_price_date and requested_start and requested_start < str(first_price_date):
        resolved["effective_start_date"] = first_price_date
        resolved["effective_start_date_source"] = "price_coverage"
    return resolved


def build_market_coverage_summary(backtest_dir: Path) -> dict[str, Any]:
    candidate_counts = {"Prime": 0, "Standard": 0, "Growth": 0, "Unknown": 0}
    selected_counts = {"Prime": 0, "Standard": 0, "Growth": 0, "Unknown": 0}
    excluded_count = 0
    for scoring_file in sorted(backtest_dir.glob("scoring_*.json")):
        try:
            payload = read_json(scoring_file)
        except Exception:
            continue
        scores = payload.get("scores", [])
        selected = [item for item in scores if item.get("selected")]
        _merge_count_dict(candidate_counts, market_section_counts(scores))
        _merge_count_dict(selected_counts, market_section_counts(selected))
        market_filter = payload.get("market_filter", {})
        if isinstance(market_filter, dict):
            excluded_count += int(market_filter.get("market_filter_excluded_count") or 0)
        else:
            excluded_count += sum(1 for item in scores if item.get("market_section_filter_blocked"))
    return {
        "candidate_count": candidate_counts,
        "selected_count": selected_counts,
        "market_filter_excluded_count": excluded_count,
    }


def _merge_count_dict(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value or 0)


def _is_rule_based_backtest(config: dict[str, Any]) -> bool:
    ai_decision = config.get("ai_decision", {})
    ai_commentary = config.get("ai_commentary", {})
    broker = config.get("broker", {})
    safety = config.get("safety", {})
    return (
        not bool(ai_decision.get("enabled", False))
        and ai_commentary.get("provider", "rule_based") == "rule_based"
        and broker.get("provider", "paper") == "paper"
        and not bool(broker.get("live_trading_enabled", False))
        and not bool(safety.get("allow_live_trading", False))
    )


def render_backtest_summary_markdown(summary: dict[str, Any], config: dict[str, Any]) -> str:
    lines = [
        f"# バックテスト結果 {summary['start_date']} to {summary['end_date']}",
        "",
        f"担当AI: {config['dealer']['name']}",
        f"config_version: {summary.get('config_version', config_version_from(config))}",
        "",
        "## サマリ",
        "",
        f"- 開始日: {summary['start_date']}",
        f"- 終了日: {summary['end_date']}",
        "",
        "## Backtest Date Resolution",
        "",
        f"- requested_start_date: {summary.get('date_resolution', {}).get('requested_start_date', summary['start_date'])}",
        f"- requested_end_date: {summary.get('date_resolution', {}).get('requested_end_date', summary['end_date'])}",
        f"- effective_start_date: {summary.get('date_resolution', {}).get('effective_start_date', summary['start_date'])}",
        f"- effective_end_date: {summary.get('date_resolution', {}).get('effective_end_date', summary['end_date'])}",
        f"- source: start={summary.get('date_resolution', {}).get('start_date_source', 'default')} end={summary.get('date_resolution', {}).get('end_date_source', 'default')}",
        "",
        "## Backtest Date Range Audit",
        "",
        *_backtest_date_range_audit_lines(summary.get("date_range_audit", {})),
        "",
        "## Backtest Coverage Audit",
        "",
        *_backtest_coverage_audit_lines(summary.get("date_range_audit", {}).get("backtest_coverage_audit", {})),
        "",
        "## Backtest Execution Audit",
        "",
        *_backtest_execution_audit_lines(summary.get("date_range_audit", {}).get("backtest_execution_audit", {})),
        "",
        "## Backtest Execution Model",
        "",
        *_backtest_execution_model_lines(summary.get("execution_model") or _backtest_execution_model(config)),
        "",
        "## Market Coverage",
        "",
        *_market_coverage_lines(summary.get("market_coverage", {})),
        "",
        "## Backtest Integrity Audit",
        "",
        *_backtest_integrity_audit_lines(summary.get("backtest_integrity_audit", {})),
        "",
        "## 成績",
        "",
        f"- evaluation_label: {summary.get('evaluation_label', 'experimental')}",
        f"- 初期資金: {summary['initial_capital']:,.0f}円",
        f"- 最終資産: {summary['final_assets']:,.0f}円",
        f"- 累計損益: {summary['cumulative_profit']:,.0f}円",
        f"- 累計損益率: {summary['cumulative_profit_rate']:.2%}",
        f"- 税引前損益: {_format_optional_yen(summary.get('gross_cumulative_profit'))}",
        f"- 概算税額: {_format_optional_yen(summary.get('estimated_tax_total'))}",
        f"- 手数料合計: {_format_optional_yen(summary.get('total_commission'))}",
        f"- 税引後損益: {_format_optional_yen(summary.get('net_cumulative_profit'))}",
        f"- 税引後損益率: {_format_optional_percent(summary.get('net_cumulative_profit_rate'))}",
        f"- 勝率: {_format_optional_rate(summary['win_rate'])}",
        f"- profit factor: {_format_optional_number(summary.get('profit_factor'))}",
        f"- closed_trade_count: {summary.get('closed_trade_count')}",
        f"- win_count: {summary.get('win_count')}",
        f"- loss_count: {summary.get('loss_count')}",
        f"- excluded_order_event_count: {summary.get('excluded_order_event_count')}",
        f"- 総取引数: {summary['total_trades']}",
        f"- 利確回数: {summary['take_profit_count']}",
        f"- 損切り回数: {summary['stop_loss_count']}",
        f"- 最大保有期間売却回数: {summary['max_holding_exit_count']}",
        f"- no trade日数: {summary.get('no_trade_days', 0)}",
        f"- selected_count合計: {summary.get('selected_count_total', 0)}",
        f"- 最大ドローダウン: {summary['max_drawdown']:.2%}",
        "",
        "## ベスト取引",
        "",
        _format_trade_summary(summary["best_trade"]),
        "",
        "## ワースト取引",
        "",
        _format_trade_summary(summary["worst_trade"]),
        "",
        "## 日別資産推移",
        "",
    ]
    if not summary["daily_asset_curve"]:
        lines.append("- 日別資産推移なし")
    else:
        for item in summary["daily_asset_curve"]:
            lines.append(
                f"- Day {item['day']} {item['date']}: 総資産 {item['total_assets']:,.0f}円, "
                f"累計損益 {item['cumulative_profit']:,.0f}円, DD {item['max_drawdown']:.2%}"
            )
    lines.extend(["", "## 新人ディーラー1号コメント", "", summary["dealer_comment"], ""])
    return "\n".join(lines)


def _backtest_date_range_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- audit: unavailable"]
    prices = (audit.get("data_coverage") or {}).get("prices", {})
    hardcoded = audit.get("hardcoded_date_audit", {})
    processed = audit.get("processed_data_audit", {}) if isinstance(audit.get("processed_data_audit"), dict) else {}
    lines = [
        f"- requested_start_date: {audit.get('requested_start_date')}",
        f"- requested_end_date: {audit.get('requested_end_date')}",
        f"- effective_trade_start_date: {audit.get('effective_trade_start_date')}",
        f"- effective_trade_end_date: {audit.get('effective_trade_end_date')}",
        f"- indicator_fetch_start_date: {audit.get('indicator_fetch_start_date')}",
        f"- price_fetch_requested_start: {audit.get('price_fetch_requested_start')}",
        f"- price_fetch_clamped_start: {audit.get('price_fetch_clamped_start')}",
        f"- first_fetch_attempt_date: {audit.get('first_fetch_attempt_date')}",
        f"- raw_price_first_date: {audit.get('raw_price_first_date')}",
        f"- raw_price_last_date: {audit.get('raw_price_last_date')}",
        f"- first_price_date: {audit.get('first_price_date')}",
        f"- last_price_date: {audit.get('last_price_date')}",
        f"- first_trading_day: {audit.get('first_trading_day')}",
        f"- last_trading_day: {audit.get('last_trading_day')}",
        f"- target_trading_days: {audit.get('target_trading_days')}",
        f"- target_trading_days_source: {audit.get('target_trading_days_source')}",
        f"- processed_first_date: {audit.get('processed_first_date')}",
        f"- processed_last_date: {audit.get('processed_last_date')}",
        f"- missing_processed_dates_count: {audit.get('missing_processed_dates_count')}",
        f"- first_missing_processed_date: {audit.get('first_missing_processed_date')}",
        f"- last_missing_processed_date: {audit.get('last_missing_processed_date')}",
        f"- processed_days: {audit.get('processed_days')}",
        f"- skipped_days: {audit.get('skipped_days')}",
        f"- last_processed_day: {audit.get('last_processed_day')}",
        f"- first_trade_date: {audit.get('first_trade_date')}",
        f"- last_trade_date: {audit.get('last_trade_date')}",
        "",
        "### Data Coverage Audit",
        "",
        f"- prices.requested_end_date: {prices.get('requested_end_date')}",
        f"- prices.latest_available_price_date: {prices.get('latest_available_price_date')}",
        f"- prices.coverage_ok: {str(bool(prices.get('coverage_ok'))).lower()}",
        f"- prices.warning: {prices.get('warning') or '-'}",
        "",
        "### Requested vs Effective Period",
        "",
        f"- requested_period: {audit.get('requested_start_date')} to {audit.get('requested_end_date')}",
        f"- effective_period: {audit.get('first_trading_day')} to {audit.get('last_processed_day')}",
        f"- effective_range_warning: {audit.get('effective_range_warning') or '-'}",
        "",
        "### Hardcoded Date Audit",
        "",
        f"- target: {hardcoded.get('target', '2026-03-06')}",
        f"- match_count: {hardcoded.get('match_count', 0)}",
        f"- warning: {hardcoded.get('warning') or '-'}",
        "",
        "### Processed Data Audit",
        "",
        f"- indicators_last_date: {processed.get('indicators_last_date')}",
        f"- candidates_last_date: {processed.get('candidates_last_date')}",
        f"- scored_candidates_last_date: {processed.get('scored_candidates_last_date')}",
        f"- indicators_count: {processed.get('indicators_count')}",
        f"- candidates_file_count: {processed.get('candidates_file_count')}",
        f"- scored_candidates_file_count: {processed.get('scored_candidates_file_count')}",
        f"- dates_with_indicators_but_no_candidates: {processed.get('dates_with_indicators_but_no_candidates_count', 0)}",
        f"- dates_with_candidates_but_no_scored: {processed.get('dates_with_candidates_but_no_scored_count', 0)}",
    ]
    for value in (processed.get("dates_with_indicators_but_no_candidates") or [])[:10]:
        lines.append(f"- indicator_without_candidate: {value}")
    for value in (processed.get("dates_with_candidates_but_no_scored") or [])[:10]:
        lines.append(f"- candidate_without_scored: {value}")
    for match in hardcoded.get("matches_sample", [])[:10]:
        lines.append(f"- match: {match}")
    return lines


def _backtest_coverage_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- audit: unavailable"]
    lines = [
        f"- requested_start_date: {audit.get('requested_start_date')}",
        f"- requested_end_date: {audit.get('requested_end_date')}",
        f"- price_fetch_requested_start: {audit.get('price_fetch_requested_start')}",
        f"- price_fetch_clamped_start: {audit.get('price_fetch_clamped_start')}",
        f"- first_fetch_attempt_date: {audit.get('first_fetch_attempt_date')}",
        f"- actual_start_date: {audit.get('actual_start_date')}",
        f"- actual_end_date: {audit.get('actual_end_date')}",
        f"- raw_cache_first_date: {audit.get('raw_cache_first_date')}",
        f"- raw_cache_last_date: {audit.get('raw_cache_last_date')}",
        f"- first_price_date: {audit.get('first_price_date')}",
        f"- last_price_date: {audit.get('last_price_date')}",
        f"- first_indicator_date: {audit.get('first_indicator_date')}",
        f"- last_indicator_date: {audit.get('last_indicator_date')}",
        f"- first_candidate_date: {audit.get('first_candidate_date')}",
        f"- last_candidate_date: {audit.get('last_candidate_date')}",
        f"- first_trade_date: {audit.get('first_trade_date')}",
        f"- last_trade_date: {audit.get('last_trade_date')}",
        f"- candidate_days: {audit.get('candidate_days')}",
        f"- trade_days: {audit.get('trade_days')}",
        f"- price_days: {audit.get('price_days')}",
        f"- expected_business_days: {audit.get('expected_business_days')}",
        f"- coverage_ratio: {_format_optional_percent(audit.get('coverage_ratio'))}",
        f"- coverage_warning: {audit.get('coverage_warning') or '-'}",
    ]
    for warning in audit.get("coverage_warnings") or []:
        lines.append(f"- warning: {warning}")
    return lines


def _backtest_execution_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- audit: unavailable"]
    status = str(audit.get("status") or "")
    lines = [
        f"- status: {status}",
        f"- first_processed_day: {audit.get('first_processed_day')}",
        f"- last_processed_day: {audit.get('last_processed_day')}",
        f"- processed_days: {audit.get('processed_days')}",
        f"- skipped_days: {audit.get('skipped_days')}",
        f"- first_indicator_date: {audit.get('first_indicator_date')}",
        f"- last_indicator_date: {audit.get('last_indicator_date')}",
        f"- first_candidate_date: {audit.get('first_candidate_date')}",
        f"- last_candidate_date: {audit.get('last_candidate_date')}",
        f"- first_scored_candidate_date: {audit.get('first_scored_candidate_date')}",
        f"- last_scored_candidate_date: {audit.get('last_scored_candidate_date')}",
        f"- first_trade_date: {audit.get('first_trade_date')}",
        f"- last_trade_date: {audit.get('last_trade_date')}",
        f"- target_trading_days: {audit.get('target_trading_days')}",
        f"- actual_trading_days: {audit.get('actual_trading_days')}",
        f"- effective_end_date: {audit.get('effective_end_date')}",
        f"- expected_processing_end_date: {audit.get('expected_processing_end_date')}",
        f"- date_range_limited_reason: {audit.get('date_range_limited_reason')}",
    ]
    if status == "ERROR":
        lines.append("- ERROR: processed_last_date is before the expected processing end date or a processed stage is incomplete.")
    return lines


def _backtest_execution_model_lines(model: dict[str, Any]) -> list[str]:
    return [
        f"- signal_timing: {model.get('signal_timing', 'after_close')}",
        f"- entry_timing: {model.get('entry_timing', 'next_business_day_open')}",
        f"- entry_price_source: {model.get('entry_price_source', 'open')}",
        f"- same_day_execution: {str(bool(model.get('same_day_execution'))).lower()}",
        f"- signal_date_entry_date_separated: {str(bool(model.get('signal_date_entry_date_separated', False))).lower()}",
        f"- signal_entry_same_day_count: {model.get('signal_entry_same_day_count', 0)}",
        f"- signal_entry_next_day_count: {model.get('signal_entry_next_day_count', 0)}",
        f"- signal_entry_gap_average: {_format_optional_percent(model.get('signal_entry_gap_average'))}",
    ]


def _market_coverage_lines(coverage: dict[str, Any]) -> list[str]:
    if not coverage:
        return ["- market coverage: unavailable"]
    candidates = coverage.get("candidate_count", {}) if isinstance(coverage.get("candidate_count"), dict) else {}
    selected = coverage.get("selected_count", {}) if isinstance(coverage.get("selected_count"), dict) else {}
    return [
        f"- Prime candidate count: {candidates.get('Prime', 0)}",
        f"- Standard candidate count: {candidates.get('Standard', 0)}",
        f"- Growth candidate count: {candidates.get('Growth', 0)}",
        f"- Unknown candidate count: {candidates.get('Unknown', 0)}",
        f"- Prime selected count: {selected.get('Prime', 0)}",
        f"- Standard selected count: {selected.get('Standard', 0)}",
        f"- Growth selected count: {selected.get('Growth', 0)}",
        f"- Unknown selected count: {selected.get('Unknown', 0)}",
        f"- market_filter_excluded_count: {coverage.get('market_filter_excluded_count', 0)}",
    ]


def _backtest_execution_model_stats(all_trades: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    filled_buys = [
        trade
        for trade in all_trades
        if trade.get("action") == "BUY" and (trade.get("order_status") or trade.get("status")) in {"FILLED", "filled", "Filled", None}
    ]
    same_day_count = 0
    next_day_count = 0
    missing_count = 0
    gaps: list[float] = []
    for trade in filled_buys:
        signal_text = trade.get("signal_date")
        entry_text = trade.get("entry_date")
        if not signal_text or not entry_text:
            missing_count += 1
            continue
        try:
            signal = date.fromisoformat(str(signal_text))
            entry = date.fromisoformat(str(entry_text))
        except ValueError:
            missing_count += 1
            continue
        if signal == entry:
            same_day_count += 1
        elif signal < entry:
            next_day_count += 1
        gap = _safe_float(trade.get("entry_gap_rate"))
        if gap is not None:
            gaps.append(gap)
    same_day_execution = bool(model.get("same_day_execution"))
    separated = missing_count == 0 and (same_day_execution or same_day_count == 0)
    return {
        "signal_date_entry_date_separated": separated,
        "signal_entry_same_day_count": same_day_count,
        "signal_entry_next_day_count": next_day_count,
        "signal_entry_missing_count": missing_count,
        "signal_entry_gap_average": round(sum(gaps) / len(gaps), 4) if gaps else None,
    }


def build_backtest_integrity_audit(
    config: dict[str, Any],
    all_trades: list[dict[str, Any]],
    backtest_dir: Path | None = None,
    date_range_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = _backtest_execution_model(config)
    filled_buys = [
        trade
        for trade in all_trades
        if trade.get("action") == "BUY" and (trade.get("order_status") or trade.get("status")) in {"FILLED", "filled", "Filled", None}
    ]
    filled_sells = [
        trade
        for trade in all_trades
        if trade.get("action") == "SELL" and (trade.get("order_status") or trade.get("status")) in {"FILLED", "filled", "Filled", None}
    ]
    checks: dict[str, dict[str, Any]] = {}
    same_day_execution = bool(model.get("same_day_execution"))
    checks["same_day_execution"] = _integrity_check(
        "WARN" if same_day_execution else "OK",
        same_day_execution,
        "same_day_close is enabled; useful for comparison but optimistic." if same_day_execution else "buy execution is not same-day close.",
    )

    separated_violations = [
        trade
        for trade in filled_buys
        if trade.get("signal_date") and trade.get("entry_date") and trade.get("signal_date") == trade.get("entry_date")
    ]
    missing_signal = [trade for trade in filled_buys if not trade.get("signal_date") or not trade.get("entry_date")]
    if same_day_execution:
        separated_status = "WARN"
        separated_message = "same_day_close intentionally keeps signal_date and entry_date on the same date."
    elif separated_violations:
        separated_status = "NG"
        separated_message = f"{len(separated_violations)} buy trade(s) executed on signal_date."
    elif missing_signal:
        separated_status = "WARN"
        separated_message = f"{len(missing_signal)} buy trade(s) lack signal_date or entry_date."
    else:
        separated_status = "OK"
        separated_message = "filled buy trades separate signal_date and entry_date."
    checks["signal_date_entry_date_separated"] = _integrity_check(
        separated_status,
        not separated_violations and not missing_signal and not same_day_execution,
        separated_message,
    )

    price_source = str(model.get("entry_price_source") or "")
    entry_price_mismatches = []
    for trade in filled_buys:
        entry_price = _safe_float(trade.get("entry_price"))
        if price_source == "open":
            expected = _safe_float(trade.get("entry_open_price"))
        elif price_source == "close":
            expected = _safe_float(trade.get("signal_close_price" if same_day_execution else "entry_price"))
        else:
            expected = None
        if expected is not None and entry_price is not None and abs(entry_price - expected) > 0.0001:
            entry_price_mismatches.append(trade)
    checks["entry_price_source"] = _integrity_check(
        "NG" if entry_price_mismatches else "OK",
        price_source,
        f"{len(entry_price_mismatches)} buy trade(s) do not match configured entry price source."
        if entry_price_mismatches
        else f"entry price source is {price_source}.",
    )

    stop_loss_execution = str(config.get("execution", {}).get("stop_loss_execution", "next_day_open"))
    checks["exit_price_source"] = _integrity_check(
        "OK",
        stop_loss_execution,
        "close-based exit checks are applied after holding_days >= 2; intraday stop uses same-day low only when configured.",
    )
    checks["entry_day_exit_policy"] = _integrity_check(
        "OK",
        "disabled",
        "entry-day immediate TP/SL/max-holding exit is disabled by holding_days < 2.",
    )

    investor_enabled = bool(config.get("features", {}).get("investor_context")) and bool(
        config.get("scoring", {}).get("use_investor_context_score")
    )
    earnings_enabled = bool(config.get("earnings_filter", {}).get("enabled"))
    financial_enabled = bool(config.get("features", {}).get("financial_context"))
    context_warnings = investor_enabled or earnings_enabled or financial_enabled
    checks["uses_future_price_data"] = _integrity_check(
        "WARN" if context_warnings else "OK",
        context_warnings,
        "price path uses signal_date/entry_date only, but point-in-time safety warnings exist in non-price context data."
        if context_warnings
        else "no evidence of future price data use in the execution model.",
    )
    checks["investor_context_pubdate_safe"] = _integrity_check(
        "WARN" if investor_enabled else "OK",
        not investor_enabled,
        "investor_types is filtered by record date, but PubDate availability is not explicitly enforced."
        if investor_enabled
        else "investor_context disabled for this profile.",
    )
    checks["financial_context_disclosure_date_safe"] = _integrity_check(
        "WARN" if financial_enabled else "OK",
        not financial_enabled,
        "financial_context requires explicit disclosure-date point-in-time verification before final evaluation."
        if financial_enabled
        else "financial_context disabled for this profile.",
    )
    checks["earnings_calendar_point_in_time_safe"] = _integrity_check(
        "WARN" if earnings_enabled else "OK",
        not earnings_enabled,
        "earnings calendar uses available schedule cache; historical point-in-time schedule availability is not guaranteed."
        if earnings_enabled
        else "earnings_filter disabled for this profile.",
    )

    costs = config.get("costs", {})
    commission_enabled = float(costs.get("commission_rate", 0.0) or 0.0) > 0 or float(costs.get("min_commission", 0.0) or 0.0) > 0
    checks["transaction_costs_enabled"] = _integrity_check(
        "OK" if commission_enabled else "WARN",
        commission_enabled,
        "commission is modeled." if commission_enabled else "commission is configured as zero; results exclude brokerage fees.",
    )
    tax_enabled = bool(costs.get("apply_tax_on_profit", True)) and float(costs.get("tax_rate", 0.0) or 0.0) > 0
    checks["tax_enabled"] = _integrity_check(
        "OK" if tax_enabled else "WARN",
        tax_enabled,
        "tax on realized profit is modeled." if tax_enabled else "tax is disabled or tax_rate is zero.",
    )
    gap_recorded = any(trade.get("entry_gap_rate") is not None for trade in filled_buys)
    checks["slippage_enabled"] = _integrity_check(
        "WARN" if not gap_recorded else "OK",
        gap_recorded,
        "entry gap is recorded; additional order-book slippage is not modeled."
        if gap_recorded
        else "no entry_gap_rate found in filled buys; slippage/gap effect may be missing.",
    )
    checks["cash_constraint_enabled"] = _integrity_check("OK", True, "buy sizing is constrained by cash and commission.")
    checks["round_lot_enabled"] = _integrity_check(
        "OK" if bool(config.get("trading", {}).get("use_round_lot")) else "WARN",
        bool(config.get("trading", {}).get("use_round_lot")),
        "round-lot sizing is enabled." if bool(config.get("trading", {}).get("use_round_lot")) else "round-lot sizing is disabled.",
    )
    checks["max_positions_enabled"] = _integrity_check(
        "OK" if int(config.get("portfolio", {}).get("max_positions", 0) or 0) > 0 else "NG",
        config.get("portfolio", {}).get("max_positions"),
        f"max_positions={config.get('portfolio', {}).get('max_positions')}",
    )
    checks["same_day_cash_reuse"] = _integrity_check(
        "WARN",
        True,
        "sell proceeds are added to cash before same-day buys in the paper engine.",
    )
    checks["listed_info_point_in_time_safe"] = _integrity_check(
        "WARN",
        False,
        "listed_info uses the current cached universe for historical periods; delisted names may be missing.",
    )
    processed = (date_range_audit or {}).get("processed_data_audit") or {}
    missing_count = int(processed.get("dates_with_indicators_but_no_candidates_count") or 0) + int(
        processed.get("dates_with_candidates_but_no_scored_count") or 0
    )
    checks["missing_price_handling"] = _integrity_check(
        "WARN" if missing_count else "OK",
        missing_count,
        f"processed stage gaps={missing_count}." if missing_count else "missing stage files are not detected in the audited range.",
    )
    checks["survivorship_bias_risk"] = _integrity_check(
        "WARN",
        True,
        "historical universe is likely based on current listed_info; treat long backtests as experimental.",
    )

    status_counts = {
        "OK": sum(1 for item in checks.values() if item["status"] == "OK"),
        "WARN": sum(1 for item in checks.values() if item["status"] == "WARN"),
        "NG": sum(1 for item in checks.values() if item["status"] == "NG"),
    }
    overall_status = "NG" if status_counts["NG"] else "WARN" if status_counts["WARN"] else "OK"
    return {
        "overall_status": overall_status,
        "evaluation_label": "final_evaluation" if overall_status == "OK" else "experimental",
        "status_counts": status_counts,
        "checks": checks,
    }


def _integrity_check(status: str, value: Any, message: str) -> dict[str, Any]:
    return {"status": status, "value": value, "message": message}


def _backtest_integrity_audit_lines(audit: dict[str, Any]) -> list[str]:
    if not audit:
        return ["- audit: unavailable"]
    lines = [
        f"- overall_status: {audit.get('overall_status')}",
        f"- evaluation_label: {audit.get('evaluation_label')}",
    ]
    checks = audit.get("checks") or {}
    for key in [
        "same_day_execution",
        "signal_date_entry_date_separated",
        "entry_price_source",
        "exit_price_source",
        "uses_future_price_data",
        "investor_context_pubdate_safe",
        "financial_context_disclosure_date_safe",
        "earnings_calendar_point_in_time_safe",
        "transaction_costs_enabled",
        "tax_enabled",
        "slippage_enabled",
        "cash_constraint_enabled",
        "round_lot_enabled",
        "max_positions_enabled",
        "same_day_cash_reuse",
        "listed_info_point_in_time_safe",
        "missing_price_handling",
        "survivorship_bias_risk",
    ]:
        item = checks.get(key, {})
        lines.append(f"- {key}: {item.get('status', 'WARN')} - {item.get('message', '')}")
    if audit.get("overall_status") in {"WARN", "NG"}:
        lines.append("- conclusion: WARN/NG項目があるため、このbacktest結果はfinal evaluationではなくexperimentalとして扱います。")
    return lines


def render_rule_based_90d_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Rule-based 90d Backtest Summary {summary['start_date']} to {summary['end_date']}",
        "",
        "## 実行条件",
        "",
        f"- 期間: {summary['start_date']} 〜 {summary['end_date']}",
        f"- profile: {summary.get('profile_id')} {summary.get('profile_name')}",
        f"- provider: {summary.get('provider', 'jquants')}",
        "- ChatGPT/OpenAI: disabled",
        f"- broker: {summary.get('broker', 'paper')}",
        f"- config_version: {summary.get('config_version')}",
        "",
        "## Backtest Date Range Audit",
        "",
        *_backtest_date_range_audit_lines(summary.get("date_range_audit", {})),
        "",
        "## 結果サマリ",
        "",
        f"- 初期資金: {_format_optional_yen(summary.get('initial_capital'))}",
        f"- 最終資産: {_format_optional_yen(summary.get('final_assets'))}",
        f"- 税引前損益: {_format_optional_yen(summary.get('gross_cumulative_profit'))}",
        f"- 税引後損益: {_format_optional_yen(summary.get('net_cumulative_profit'))}",
        f"- 税引後損益率: {_format_optional_percent(summary.get('net_cumulative_profit_rate'))}",
        f"- 勝率: {_format_optional_percent(summary.get('win_rate'))}",
        f"- profit factor: {_format_optional_number(summary.get('profit_factor'))}",
        f"- 最大ドローダウン: {_format_optional_percent(summary.get('max_drawdown'))}",
        f"- 総取引数: {summary.get('total_trades')}",
        f"- 利確回数: {summary.get('take_profit_count')}",
        f"- 損切り回数: {summary.get('stop_loss_count')}",
        f"- 最大保有期間売却回数: {summary.get('max_holding_exit_count')}",
        f"- no trade日数: {summary.get('no_trade_days')}",
        f"- selected_count合計: {summary.get('selected_count_total')}",
        "",
        "## 新人ディーラー1号コメント",
        "",
        summary.get("dealer_comment", ""),
        "",
        "## 次に見るべき改善ポイント",
        "",
        "- selected_count_total と no_trade_days を見て、スクリーニングが厳しすぎないか確認する。",
        "- profit factor と勝率を合わせて、利幅と損切り幅のバランスを確認する。",
        "- 最大ドローダウンがrisk_marginの許容範囲内か確認する。",
        "- 利確、損切り、最大保有期間売却の比率を見て、出口ルールが短期売買に合っているか確認する。",
        "- J-Quants Freeプランの12週間遅延データ前提で、複数期間でも再現するか確認する。",
        "",
        "## 注意",
        "",
        "OpenAI / ChatGPT APIは使わず、ルールベースのみで検証しています。PaperBrokerによる仮想売買であり、実売買は行いません。",
        "",
    ]
    return "\n".join(lines)


def render_analysis_markdown(analysis: dict[str, Any]) -> str:
    portfolio = analysis["portfolio_analysis"]
    trades = analysis["trade_analysis"]
    scores = analysis["score_analysis"]
    reflections = analysis["reflection_analysis"]
    config_versions = analysis.get("config_version_analysis", [])
    sector_win_rates = analysis.get("sector_win_rate_analysis", [])
    profile_analysis = analysis.get("profile_analysis", [])
    yearly_performance = analysis.get("yearly_performance", [])
    monthly_performance = analysis.get("monthly_performance", [])
    walk_forward_validation = analysis.get("walk_forward_validation", {})
    market_regime_performance = analysis.get("market_regime_performance", {})
    selection_quality = analysis.get("selection_quality_analysis", {})
    bands = scores["score_bands"]
    lines = [
        "# 新人ディーラー1号 分析レポート",
        "",
        f"profile: {analysis.get('current_profile_id', 'unknown')} {analysis.get('current_profile_name', '')}",
        f"生成日時: {analysis['generated_at']}",
        f"現在のconfig_version: {analysis.get('current_config_version', 'unknown')}",
        "",
        "## ポートフォリオ分析",
        "",
        f"- 初期資金: {_format_optional_yen(portfolio['initial_capital'])}",
        f"- 最新総資産: {_format_optional_yen(portfolio['latest_total_assets'])}",
        f"- 累計損益: {_format_optional_yen(portfolio['cumulative_profit'])}",
        f"- 累計損益率: {_format_optional_percent(portfolio['cumulative_profit_rate'])}",
        f"- 税引前累計損益: {_format_optional_yen(portfolio.get('gross_cumulative_profit'))}",
        f"- 税引後累計損益: {_format_optional_yen(portfolio.get('net_cumulative_profit'))}",
        f"- 概算税額合計: {_format_optional_yen(portfolio.get('estimated_tax_total'))}",
        f"- 手数料合計: {_format_optional_yen(portfolio.get('total_commission'))}",
        f"- 最大ドローダウン: {_format_optional_percent(portfolio['max_drawdown'])}",
        f"- 運用日数: {portfolio['operation_days']}",
        "",
        "## Reconciliation Report",
        "",
        f"- initial_capital: {_format_optional_yen(portfolio.get('initial_capital'))}",
        f"- realized_profit: {_format_optional_yen(portfolio.get('realized_profit'))}",
        f"- unrealized_profit: {_format_optional_yen(portfolio.get('unrealized_profit'))}",
        f"- gross_profit_total: {_format_optional_yen(trades.get('gross_profit_total'))}",
        f"- gross_loss_total: {_format_optional_yen(trades.get('gross_loss_total'))}",
        f"- cash: {_format_optional_yen(portfolio.get('cash'))}",
        f"- positions_value: {_format_optional_yen(portfolio.get('positions_value'))}",
        f"- final_assets: {_format_optional_yen(portfolio.get('latest_total_assets'))}",
        f"- formula: initial_capital + realized_profit + unrealized_profit = {_format_optional_yen(portfolio.get('reconciled_assets'))}",
        f"- reconciliation_difference: {_format_optional_yen(portfolio.get('reconciliation_difference'))}",
        f"- reconciliation_ok: {portfolio.get('reconciliation_ok')}",
        f"- open_positions_count: {portfolio.get('open_positions_count')}",
        f"- closed_trades_count: {portfolio.get('closed_trades_count')}",
        "",
        "## Profile別集計",
        "",
        *_profile_analysis_lines(profile_analysis),
        "",
        "## 取引分析",
        "",
        f"- 総取引数: {trades['total_trades']}",
        f"- 勝ち取引数: {trades.get('win_count', trades['winning_trades'])}",
        f"- 負け取引数: {trades.get('loss_count', trades['losing_trades'])}",
        f"- 勝率: {_format_optional_percent(trades['win_rate'])}",
        f"- gross_profit_total: {_format_optional_yen(trades.get('gross_profit_total'))}",
        f"- gross_loss_total: {_format_optional_yen(trades.get('gross_loss_total'))}",
        f"- profit factor: {_format_optional_number(trades.get('profit_factor'))}",
        f"- closed_trade_count: {trades.get('closed_trade_count')}",
        f"- win_count: {trades.get('win_count')}",
        f"- loss_count: {trades.get('loss_count')}",
        f"- excluded_order_event_count: {trades.get('excluded_order_event_count')}",
        f"- profit ratio: {_format_optional_number(trades.get('profit_ratio'))}",
        f"- 期待値: {_format_optional_percent(trades.get('expectancy'))}",
        f"- 平均勝ち利益率: {_format_optional_percent(trades.get('average_win_profit_rate', trades['average_profit_rate']))}",
        f"- 平均負け損失率: {_format_optional_percent(trades.get('average_loss_profit_rate', trades['average_loss_rate']))}",
        f"- 平均保有日数: {_format_optional_number(trades['average_holding_days'])}",
        f"- largest_win: {_format_optional_yen(trades.get('largest_win'))}",
        f"- largest_loss: {_format_optional_yen(trades.get('largest_loss'))}",
        f"- 最大負け損失率: {_format_optional_percent(trades.get('worst_loss_profit_rate'))}",
        f"- best_trade: {_format_trade_summary_inline(trades.get('best_trade'))}",
        f"- worst_trade: {_format_trade_summary_inline(trades.get('worst_trade'))}",
        f"- 利確回数: {trades['take_profit_count']}",
        f"- 損切り回数: {trades['stop_loss_count']}",
        f"- 損切り乖離平均: {_format_optional_percent(trades.get('stop_loss_slippage_average'))}",
        f"- 損切り乖離最大: {_format_optional_percent(trades.get('stop_loss_slippage_max'))}",
        f"- 設定損切り超過件数: {trades.get('loss_over_stop_count', 0)}",
        f"- 設定損切り超過率: {_format_optional_percent(trades.get('loss_over_stop_rate'))}",
        f"- 最大保有期間売却回数: {trades['max_holding_exit_count']}",
        f"- 平均スリッページ: {_format_optional_percent(trades.get('average_slippage'))}",
        f"- 最大スリッページ: {_format_optional_percent(trades.get('max_slippage'))}",
        f"- ギャップアップ回数: {trades.get('gap_up_count', 0)}",
        f"- ギャップダウン回数: {trades.get('gap_down_count', 0)}",
        f"- 利確到達前に売った取引の割合: {_format_optional_percent(trades.get('sold_before_take_profit_rate'))}",
        "",
        "## Exit Reason Analysis",
        "",
        *_exit_reason_analysis_lines(trades.get("exit_reason_analysis", [])),
        "",
        "## Exit Efficiency",
        "",
        *_exit_efficiency_lines(trades.get("exit_efficiency", {})),
        "",
        "## Holding Period Analysis",
        "",
        *_holding_period_analysis_lines(trades.get("holding_period_analysis", [])),
        "",
        "## Holding Period Optimization",
        "",
        *_holding_period_optimization_lines(trades.get("holding_period_optimization", {})),
        "",
        "## Candidate Exit Improvements",
        "",
        *_candidate_exit_improvement_lines(trades.get("candidate_exit_improvements", [])),
        "",
        "## Trade Replay Analysis",
        "",
        *_trade_replay_analysis_lines(trades.get("trade_replay_analysis", {})),
        "",
        "## Stop Loss Recovery Analysis",
        "",
        *_stop_loss_recovery_analysis_lines(trades.get("stop_loss_recovery_analysis", {})),
        "",
        "## Walk Forward Validation",
        "",
        *_walk_forward_validation_lines(walk_forward_validation),
        "",
        "## Market Regime Performance Analysis",
        "",
        *_market_regime_performance_lines(market_regime_performance),
        "",
        "## Yearly Performance",
        "",
        *_yearly_performance_lines(yearly_performance),
        "",
        "## Monthly Performance",
        "",
        *_monthly_performance_lines(monthly_performance),
        "",
        "## config_version別集計",
        "",
        *_config_version_lines(config_versions),
        "",
        "## 業種別勝率",
        "",
        *_sector_win_rate_lines(sector_win_rates),
        "",
        "## Selection Quality Analysis",
        "",
        *_selection_quality_lines(selection_quality),
        "",
        "## スコア分析",
        "",
        f"- selected銘柄数: {scores['selected_count']}",
        f"- conditional selected count: {scores.get('conditional_selected_count', 0)}",
        f"- conditional rejected count: {scores.get('conditional_rejected_count', 0)}",
        f"- selected銘柄の平均スコア: {_format_optional_number(scores['selected_average_score'])}",
        f"- rejected銘柄の平均スコア: {_format_optional_number(scores['rejected_average_score'])}",
        "",
        "### スコア帯別件数",
        "",
        f"- 90点以上: {bands['90_or_more']}",
        f"- 80点台: {bands['80s']}",
        f"- 70点台: {bands['70s']}",
        f"- 60点台: {bands['60s']}",
        f"- 60点未満: {bands['under_60']}",
        "",
        "## AI振り返り分析",
        "",
        f"- reflection件数: {reflections['reflection_count']}",
        "",
        "### WIN時のよくある good_points",
        "",
        *_count_lines(reflections["win_common_good_points"]),
        "",
        "### LOSS時のよくある bad_points",
        "",
        *_count_lines(reflections["loss_common_bad_points"]),
        "",
        "### suggestions",
        "",
        *_plain_lines(reflections["suggestions"], "suggestionsなし"),
        "",
    ]
    return "\n".join(lines)


def _format_optional_rate(rate: Any) -> str:
    if rate is None:
        return "N/A（売却済み取引なし）"
    return f"{float(rate):.2%}"


def _format_optional_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2%}"


def _format_optional_number(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.2f}"


def _format_optional_yen(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.0f}円"


def _format_trade_summary_inline(trade: Any) -> str:
    if not trade:
        return "N/A"
    code = trade.get("code", "")
    name = trade.get("name", "")
    result = trade.get("result", "")
    profit = _format_optional_yen(trade.get("profit"))
    profit_rate = _format_optional_percent(trade.get("profit_rate"))
    exit_reason = trade.get("exit_reason") or "N/A"
    return f"{code} {name} {result} {profit} ({profit_rate}) / {exit_reason}"


def _relative_path_text(path_text: Any) -> str:
    if not path_text:
        return "N/A"
    path = Path(str(path_text))
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _config_version_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- config_version別の取引データなし"]
    return [
        (
            f"- {item['config_version']}: 取引数 {item['trade_count']}件, "
            f"勝率 {_format_optional_percent(item['win_rate'])}, "
            f"累計損益 {_format_optional_yen(item['cumulative_profit'])}"
        )
        for item in items
    ]


def _sector_win_rate_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 業種別の売却済み取引データなし"]
    return [
        (
            f"- {item['sector_name']}: 売却済み {item['closed_trades']}件, "
            f"勝率 {_format_optional_percent(item['win_rate'])}, "
            f"平均損益率 {_format_optional_percent(item.get('average_profit_rate'))}, "
            f"税引後損益 {_format_optional_yen(item.get('net_profit_total'))}"
        )
        for item in items
    ]


def _selection_quality_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- selection qualityデータなし"]
    selected = analysis.get("selected", {})
    rejected = analysis.get("rejected", {})
    lift = analysis.get("selection_lift", {})
    return [
        f"- screen候補: {analysis.get('screen_candidate_count', 0)}件",
        f"- score候補: {analysis.get('score_candidate_count', 0)}件",
        f"- selected銘柄: {analysis.get('selected_count', 0)}件",
        f"- rejected銘柄: {analysis.get('rejected_count', 0)}件",
        f"- selected平均5日リターン: {_format_optional_percent(selected.get('average_return_5d'))}",
        f"- rejected平均5日リターン: {_format_optional_percent(rejected.get('average_return_5d'))}",
        f"- selected平均10日リターン: {_format_optional_percent(selected.get('average_return_10d'))}",
        f"- rejected平均10日リターン: {_format_optional_percent(rejected.get('average_return_10d'))}",
        f"- Selection Lift 5d: {_format_optional_percent(lift.get('return_5d'))}",
        f"- Selection Lift 10d: {_format_optional_percent(lift.get('return_10d'))}",
    ]


def _profile_analysis_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- profile別データなし"]
    return [
        (
            f"- {item['profile_id']} {item.get('profile_name') or ''}: "
            f"総資産 {_format_optional_yen(item.get('latest_total_assets'))}, "
            f"勝率 {_format_optional_percent(item.get('win_rate'))}, "
            f"最大DD {_format_optional_percent(item.get('max_drawdown'))}, "
            f"総取引数 {item.get('total_trades', 0)}"
        )
        for item in items
    ]


def _yearly_performance_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 年別パフォーマンスデータなし"]
    lines = []
    for item in items:
        lines.extend(
            [
                f"### {item['year']}",
                "",
                f"- profit: {_format_optional_yen(item.get('profit'))}",
                f"- win_rate: {_format_optional_percent(item.get('win_rate'))}",
                f"- profit_factor: {_format_optional_number(item.get('profit_factor'))}",
                f"- max_drawdown: {_format_optional_percent(item.get('max_drawdown'))}",
                "",
            ]
        )
    return lines


def _monthly_performance_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 月別パフォーマンスデータなし"]
    lines = []
    for item in items:
        lines.extend(
            [
                f"### {item['month']}",
                "",
                f"- profit: {_format_optional_yen(item.get('profit'))}",
                f"- trades: {item.get('trades')}",
                f"- win_rate: {_format_optional_percent(item.get('win_rate'))}",
                "",
            ]
        )
    return lines


def _walk_forward_validation_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- Walk Forward Validation データなし"]
    return [
        "### Period Results",
        "",
        *_walk_forward_period_lines(analysis.get("periods", [])),
        "",
        "## Stable Periods",
        "",
        *_walk_forward_period_lines(analysis.get("stable_periods", [])),
        "",
        "## Weak Periods",
        "",
        *_walk_forward_period_lines(analysis.get("weak_periods", [])),
        "",
        "## Overfit Risk",
        "",
        *_overfit_risk_lines(analysis.get("overfit_risk", {})),
    ]


def _market_regime_performance_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- Market Regime Performance データなし"]
    lines = ["### Regime Results", ""]
    lines.extend(_market_regime_result_lines(analysis.get("regimes", [])))
    lines.extend(["", "## Best Regime", ""])
    lines.extend(_market_regime_single_lines(analysis.get("best_regime")))
    lines.extend(["", "## Worst Regime", ""])
    lines.extend(_market_regime_single_lines(analysis.get("worst_regime")))
    lines.extend(["", "## Candidate Regime Filters", ""])
    lines.extend(_candidate_regime_filter_lines(analysis.get("candidate_regime_filters", [])))
    return lines


def _market_regime_result_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- レジーム別データなし"]
    return [
        (
            f"- {item.get('market_regime')}: "
            f"profit {_format_optional_yen(item.get('profit'))}, "
            f"win_rate {_format_optional_percent(item.get('win_rate'))}, "
            f"PF {_format_optional_number(item.get('profit_factor'))}, "
            f"expectancy {_format_optional_percent(item.get('expectancy'))}, "
            f"DD {_format_optional_percent(item.get('max_drawdown'))}, "
            f"trade_count {item.get('trade_count')}"
        )
        for item in items
    ]


def _market_regime_single_lines(item: dict[str, Any] | None) -> list[str]:
    if not item:
        return ["- 該当なし"]
    return _market_regime_result_lines([item])


def _candidate_regime_filter_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 候補なし"]
    return [
        f"- {item.get('rule')}: {item.get('reason')}"
        for item in items
    ]


def _walk_forward_period_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 該当期間なし"]
    return [
        (
            f"- {item.get('start_date')} to {item.get('end_date')}: "
            f"net_cumulative_profit {_format_optional_yen(item.get('net_cumulative_profit'))}, "
            f"win_rate {_format_optional_percent(item.get('win_rate'))}, "
            f"profit_factor {_format_optional_number(item.get('profit_factor'))}, "
            f"max_drawdown {_format_optional_percent(item.get('max_drawdown'))}, "
            f"total_trades {item.get('total_trades')}, "
            f"expectancy {_format_optional_percent(item.get('expectancy'))}"
        )
        for item in items
    ]


def _overfit_risk_lines(item: dict[str, Any]) -> list[str]:
    if not item:
        return ["- overfit risk データなし"]
    return [
        f"- risk_level: {item.get('risk_level')}",
        f"- stable_period_count: {item.get('stable_period_count')}",
        f"- weak_period_count: {item.get('weak_period_count')}",
        f"- reason: {item.get('reason')}",
    ]


def _exit_reason_analysis_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 売却理由別データなし"]
    return [
        (
            f"- {item['exit_reason']}: 件数 {item['count']}件, "
            f"勝率 {_format_optional_percent(item.get('win_rate'))}, "
            f"平均利益 {_format_optional_yen(item.get('avg_profit'))}, "
            f"平均利益率 {_format_optional_percent(item.get('avg_profit_rate', item.get('average_profit_rate')))}, "
            f"合計利益 {_format_optional_yen(item.get('total_profit'))}, "
            f"平均保有日数 {_format_optional_number(item.get('avg_holding_days'))}"
        )
        for item in items
    ]


def _exit_efficiency_lines(item: dict[str, Any]) -> list[str]:
    if not item:
        return ["- Exit Efficiency データなし"]
    return [
        f"- 利確到達件数: {item.get('take_profit_count', 0)}",
        f"- 損切り到達件数: {item.get('stop_loss_count', 0)}",
        f"- 最大保有期間到達件数: {item.get('max_holding_count', 0)}",
        f"- 最大保有期間到達のうち利益で終わった件数: {item.get('max_holding_profit_count', 0)}",
        f"- 最大保有期間到達のうち損失で終わった件数: {item.get('max_holding_loss_count', 0)}",
    ]


def _holding_period_analysis_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 保有期間別データなし"]
    return [
        (
            f"- {item.get('holding_days')}日: count {item.get('count')}, "
            f"win_rate {_format_optional_percent(item.get('win_rate'))}, "
            f"avg_profit_rate {_format_optional_percent(item.get('avg_profit_rate'))}, "
            f"total_profit {_format_optional_yen(item.get('total_profit'))}"
        )
        for item in items
    ]


def _holding_period_optimization_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- Holding Period Optimization データなし"]
    return [
        f"- current_max_holding_days: {analysis.get('current_max_holding_days')}",
        f"- current_profit: {_format_optional_yen(analysis.get('current_profit'))}",
        "",
        "### calculation_details",
        "",
        *_holding_period_calculation_detail_lines(analysis.get("calculation_details", {})),
        "",
        "### 推定利益ランキング",
        "",
        *_holding_period_simulation_lines(analysis.get("estimated_profit_ranking", [])),
        "",
        "### Candidate Holding Days",
        "",
        *_candidate_holding_day_lines(analysis.get("candidate_holding_days", [])),
    ]


def _holding_period_calculation_detail_lines(item: dict[str, Any]) -> list[str]:
    if not item:
        return ["- calculation_details なし"]
    return [
        f"- current_profit_formula: {item.get('current_profit_formula')}",
        f"- simulated_profit_formula: {item.get('simulated_profit_formula')}",
        f"- lift_vs_current_formula: {item.get('lift_vs_current_formula')}",
        f"- current_profit: {_format_optional_yen(item.get('current_profit'))}",
        f"- simulated_profit: {_format_optional_yen(item.get('simulated_profit'))}",
        f"- profit_difference: {_format_optional_yen(item.get('profit_difference'))}",
        f"- lift_vs_current: {_format_optional_yen(item.get('lift_vs_current'))}",
        f"- base_trade_count: {item.get('base_trade_count')}",
        f"- simulated_trade_count: {item.get('simulated_trade_count')}",
    ]


def _holding_period_simulation_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 推定データなし"]
    return [
        (
            f"- max_holding_days={item.get('max_holding_days')}: "
            f"推定利益 {_format_optional_yen(item.get('estimated_profit'))}, "
            f"推定PF {_format_optional_number(item.get('estimated_profit_factor'))}, "
            f"推定勝率 {_format_optional_percent(item.get('estimated_win_rate'))}, "
            f"推定DD {_format_optional_percent(item.get('estimated_drawdown'))}, "
            f"sample_count {item.get('sample_count')}"
        )
        for item in items
    ]


def _candidate_holding_day_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 推奨候補なし"]
    return [
        (
            f"- recommended_max_holding_days: {item.get('recommended_max_holding_days')}, "
            f"estimated_profit {_format_optional_yen(item.get('estimated_profit'))}, "
            f"lift_vs_current {_format_optional_yen(item.get('estimated_profit_lift_vs_current'))}, "
            f"estimated_pf {_format_optional_number(item.get('estimated_profit_factor'))}, "
            f"estimated_win_rate {_format_optional_percent(item.get('estimated_win_rate'))}, "
            f"estimated_dd {_format_optional_percent(item.get('estimated_drawdown'))}, "
            f"reason {item.get('reason')}"
        )
        for item in items
    ]


def _candidate_exit_improvement_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 改善候補なし"]
    return [
        (
            f"- {item.get('suggestion')}: {item.get('reason')} "
            f"(current_value: {_format_exit_current_value(item.get('current_value'))})"
        )
        for item in items
    ]


def _format_exit_current_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2%}"
    return str(value)


def _trade_replay_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- Trade Replay データなし"]
    return [
        "### TOP10利益トレード",
        "",
        *_trade_replay_record_lines(analysis.get("top_profit_trades", [])),
        "",
        "### TOP10損失トレード",
        "",
        *_trade_replay_record_lines(analysis.get("top_loss_trades", [])),
        "",
        "### 勝ち組平均推移",
        "",
        *_average_replay_lines(analysis.get("winner_average_replay", [])),
        "",
        "### 負け組平均推移",
        "",
        *_average_replay_lines(analysis.get("loser_average_replay", [])),
    ]


def _trade_replay_record_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- replayデータなし"]
    lines = []
    for item in items:
        lines.append(
            (
                f"- entry_date {item.get('entry_date')}, code {item.get('code')}, "
                f"holding_days {item.get('holding_days')}, "
                f"profit {_format_optional_yen(item.get('profit'))}, "
                f"profit_rate {_format_optional_percent(item.get('profit_rate'))}"
            )
        )
        lines.append(f"  - entry: {_format_optional_percent(item.get('entry_return_rate'))}")
        for day_item in item.get("day_returns", []):
            lines.append(
                f"  - Day{day_item.get('day')}: {_format_optional_percent(day_item.get('return_rate'))}"
            )
    return lines


def _average_replay_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 平均推移データなし"]
    return [
        (
            f"- {item.get('label')}: {_format_optional_percent(item.get('return_rate'))} "
            f"(count {item.get('count')})"
        )
        for item in items
    ]


def _stop_loss_recovery_analysis_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- Stop Loss Recovery データなし"]
    return [
        f"- stop_loss_count: {analysis.get('stop_loss_count', 0)}",
        f"- replay_count: {analysis.get('replay_count', 0)}",
        f"- Day5回復率: {_format_optional_percent(analysis.get('day5_recovery_rate'))}",
        f"- Day10回復率: {_format_optional_percent(analysis.get('day10_recovery_rate'))}",
        "",
        "### Recovery Winners",
        "",
        *_recovery_trade_lines(analysis.get("recovery_winners", [])),
        "",
        "### Recovery Losers",
        "",
        *_recovery_trade_lines(analysis.get("recovery_losers", [])),
        "",
        "### Recovery Signals",
        "",
        *_recovery_signal_lines(analysis.get("recovery_signals", [])),
        "",
        "### Candidate Dynamic Stop Rules",
        "",
        *_dynamic_stop_rule_lines(analysis.get("candidate_dynamic_stop_rules", [])),
    ]


def _recovery_trade_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('entry_date')} {item.get('code')} {item.get('name')}: "
            f"holding_days {item.get('holding_days')}, "
            f"Day1 {_format_optional_percent(item.get('day1_return'))}, "
            f"Day5 {_format_optional_percent(item.get('day5_return'))}, "
            f"Day10 {_format_optional_percent(item.get('day10_return'))}, "
            f"RSI {_format_optional_number(item.get('rsi'))}, "
            f"volume_ratio {_format_optional_number(item.get('volume_ratio'))}, "
            f"market_regime {item.get('market_regime')}, "
            f"sector {item.get('sector')}, "
            f"signals {', '.join(item.get('candlestick_signals') or ['no_signal'])}"
        )
        for item in items
    ]


def _recovery_signal_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [
        (
            f"- {item.get('feature')}={item.get('value')}: "
            f"recovery {_format_optional_percent(item.get('winner_share'))} ({item.get('winner_count')}件), "
            f"non_recovery {_format_optional_percent(item.get('loser_share'))} ({item.get('loser_count')}件), "
            f"difference {_format_optional_signed_percent(item.get('share_difference'))}"
        )
        for item in items
    ]


def _dynamic_stop_rule_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 候補なし"]
    return [
        (
            f"- {item.get('rule')}: recovery {_format_optional_percent(item.get('winner_share'))} "
            f"({item.get('winner_count')}件), non_recovery {_format_optional_percent(item.get('loser_share'))} "
            f"({item.get('loser_count')}件), reason {item.get('reason')}"
        )
        for item in items
    ]


def _format_optional_signed_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.2%}"


def _count_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 該当なし"]
    return [f"- {item['text']}: {item['count']}回" for item in items]


def _plain_lines(items: list[str], empty_text: str) -> list[str]:
    if not items:
        return [f"- {empty_text}"]
    return [f"- {item}" for item in items]


def _format_trade_summary(trade: Any) -> str:
    if not trade:
        return "- 該当取引なし"
    return (
        f"- {trade['trade_id']} {trade['code']} {trade['name']}: {trade['result']}, "
        f"損益 {trade['profit']:,.0f}円 ({trade['profit_rate']:.2%}), 理由: {trade['exit_reason']}"
    )


def _backtest_dealer_comment(final_assets: float, initial_capital: float, closed_trades: list[dict[str, Any]]) -> str:
    if not closed_trades:
        return "売却済み取引がないため、勝率や売却ルールの評価はまだ保留します。ルールに従い、検証期間を広げます。"
    if final_assets > initial_capital:
        return "検証期間では初期資金を上回りました。感情は考慮せず、同じ条件で再現性を確認します。"
    if final_assets < initial_capital:
        return "検証期間では初期資金を下回りました。損切りを含めてルール通りに執行し、改善案は記録に留めます。"
    return "検証期間では損益がほぼ中立でした。資金効率と売買頻度を継続観察します。"


def _validate_date(target_date_text: str) -> None:
    try:
        date.fromisoformat(target_date_text)
    except ValueError as exc:
        raise SystemExit("--date must be in YYYY-MM-DD format.") from exc


def fetch_price_history(
    provider: JQuantsDataProvider,
    target_date: date,
    prime_codes: set[str],
    lookback_business_days: int,
    rate_limit_per_minute: int,
    fetch_dates: list[date] | None = None,
    continue_on_error: bool = False,
    verbose: bool = False,
    stop_on_consecutive_unsupported: bool = True,
) -> list[dict[str, Any]]:
    target_fetch_dates = fetch_dates or previous_business_dates(target_date, lookback_business_days)
    total_dates = len(target_fetch_dates)
    if getattr(provider, "requests_per_minute", None) is None:
        setattr(provider, "requests_per_minute", rate_limit_per_minute)
    if getattr(provider, "parallel_fetch", False) and not continue_on_error:
        if verbose:
            print(
                "fetch-period-prices parallel fetch enabled "
                f"workers={getattr(provider, 'max_parallel_requests', 4)} rpm={getattr(provider, 'requests_per_minute', rate_limit_per_minute)}"
            )
        with ThreadPoolExecutor(max_workers=int(getattr(provider, "max_parallel_requests", 4))) as executor:
            parts = list(
                executor.map(
                    lambda item: _fetch_price_history_for_date(
                        provider,
                        item[1],
                        prime_codes,
                        item[0],
                        total_dates,
                        continue_on_error,
                        verbose,
                    ),
                    enumerate(target_fetch_dates, start=1),
                )
            )
        return [row for part in parts for row in part]

    rows = []
    consecutive_unsupported = 0
    unsupported_range_start: date | None = None
    for index, fetch_date in enumerate(target_fetch_dates, start=1):
        part = _fetch_price_history_for_date(
            provider,
            fetch_date,
            prime_codes,
            index,
            total_dates,
            continue_on_error,
            verbose,
        )
        rows.extend(part)
        if unsupported_cache_entry(fetch_date) is not None and not part:
            consecutive_unsupported += 1
            unsupported_range_start = unsupported_range_start or fetch_date
        else:
            consecutive_unsupported = 0
            unsupported_range_start = None
        if continue_on_error and consecutive_unsupported >= 3 and unsupported_range_start is not None:
            save_unsupported_range(
                unsupported_range_start,
                fetch_date,
                reason="bad_request_or_out_of_range",
                source="consecutive-400",
            )
            if verbose:
                action = "remaining dates skipped" if stop_on_consecutive_unsupported else "range recorded, continuing because later dates may be available"
                print(
                    "fetch-period-prices warning: consecutive bad_request_or_out_of_range "
                    f"from {unsupported_range_start.isoformat()} to {fetch_date.isoformat()}; {action}"
                )
            if stop_on_consecutive_unsupported:
                break
            consecutive_unsupported = 0
            unsupported_range_start = None
    return rows


def _fetch_price_history_for_date(
    provider: JQuantsDataProvider,
    fetch_date: date,
    prime_codes: set[str],
    index: int,
    total_dates: int,
    continue_on_error: bool,
    verbose: bool,
) -> list[dict[str, Any]]:
    cached_rows = load_cached_prime_prices(fetch_date)
    if cached_rows is not None:
        _increment_provider_stat(provider, "cache_hits")
        if verbose:
            print(
                f"fetch-period-prices [{index}/{total_dates}] "
                f"{fetch_date.isoformat()} cache ({len(cached_rows)} prime rows)"
            )
        return cached_rows
    no_data_entry = load_no_data_day(fetch_date)
    if no_data_entry is not None:
        _increment_provider_stat(provider, "cache_hits")
        if verbose:
            print(
                f"fetch-period-prices [{index}/{total_dates}] "
                f"{fetch_date.isoformat()} no-data cache hit reason={no_data_entry.get('reason', 'unknown')}"
            )
            print(
                f"fetch-period-prices skip no-data cache: "
                f"{fetch_date.isoformat()} reason={no_data_entry.get('reason', 'unknown')}"
            )
        return []
    unsupported_entry = load_unsupported_day(fetch_date)
    if unsupported_entry is not None:
        _increment_provider_stat(provider, "cache_hits")
        if verbose:
            print(
                f"fetch-period-prices [{index}/{total_dates}] "
                f"{fetch_date.isoformat()} unsupported cache hit reason={unsupported_entry.get('reason', 'unknown')}"
            )
            print(
                f"fetch-period-prices skip unsupported cache: "
                f"{fetch_date.isoformat()} reason={unsupported_entry.get('reason', 'unknown')}"
            )
        return []

    _increment_provider_stat(provider, "cache_misses")
    if verbose:
        print(
            f"fetch-period-prices [{index}/{total_dates}] "
            f"{fetch_date.isoformat()} fetching J-Quants API"
        )
    try:
        daily_prices = fetch_daily_prices_with_rate_limit_retry(provider, fetch_date, verbose=verbose)
    except KeyboardInterrupt:
        if verbose:
            print(f"fetch-period-prices {fetch_date.isoformat()} interrupted by Ctrl+C")
        raise
    except RuntimeError as exc:
        if is_bad_request_or_out_of_range_error(exc):
            save_unsupported_day(
                fetch_date,
                reason="bad_request_or_out_of_range",
                source="fetch-period-prices",
            )
            if verbose:
                print(
                    f"fetch-period-prices {fetch_date.isoformat()} "
                    "HTTP 400 bad_request_or_out_of_range; saved unsupported cache"
                )
            return []
        if verbose:
            print(f"fetch-period-prices {fetch_date.isoformat()} temporary API error; not cached: {exc}")
        if continue_on_error:
            return []
        raise
    config = load_config(CONFIG_PATH)
    listed_stock_by_code = {str(stock["code"]): stock for stock in _listed_stock_master() if stock.get("code")}
    all_listed_prices = [
        _normalize_daily_price_with_market(record, listed_stock_by_code)
        for record in daily_prices
        if _get_first(record, ["code", "Code", "LocalCode"]) in listed_stock_by_code
    ]
    allowed_codes = set(_allowed_stock_master_by_code(config))
    allowed_prices = [row for row in all_listed_prices if row.get("code") in allowed_codes]
    if all_listed_prices:
        cache_price_snapshot(fetch_date, len(daily_prices), all_listed_prices)
        if verbose:
            print(
                f"fetch-period-prices {fetch_date.isoformat()} saved "
                f"{len(all_listed_prices)} listed rows from {len(daily_prices)} rows"
            )
        return allowed_prices
    save_no_data_day(fetch_date, reason="no_listed_rows", source="fetch-period-prices")
    if verbose:
        print(f"fetch-period-prices {fetch_date.isoformat()} no listed rows; saved no-data cache")
    return []


def _increment_provider_stat(provider: Any, key: str, amount: float = 1.0) -> None:
    stats = getattr(provider, "fetch_stats", None)
    if isinstance(stats, dict):
        stats[key] = stats.get(key, 0) + amount


def fetch_daily_prices_with_rate_limit_retry(
    provider: JQuantsDataProvider,
    fetch_date: date,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    retry_waits = [12, 24, 48]
    attempt = 0
    while True:
        try:
            return provider.get_daily_prices(fetch_date)
        except RuntimeError as exc:
            if is_bad_request_or_out_of_range_error(exc) or not is_retryable_jquants_error(exc) or attempt >= len(retry_waits):
                raise
            wait_seconds = retry_waits[attempt]
            attempt += 1
            if verbose:
                print(
                    f"fetch-period-prices {fetch_date.isoformat()} "
                    f"{jquants_error_category(exc)}; retry {attempt}/{len(retry_waits)} "
                    f"after {wait_seconds}s"
                )
            time.sleep(wait_seconds)


def is_retryable_jquants_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        is_rate_limit_error(exc)
        or "network error" in message
        or "timed out" in message
        or "timeout" in message
        or any(f"http {code}" in message for code in range(500, 600))
    )


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "rate limit" in message or "http 429" in message


def is_bad_request_or_out_of_range_error(exc: Exception) -> bool:
    return "http 400" in str(exc).lower()


def jquants_error_category(exc: Exception) -> str:
    if is_rate_limit_error(exc):
        return "rate limit exceeded"
    if is_bad_request_or_out_of_range_error(exc):
        return "bad_request_or_out_of_range"
    return "temporary API error"


def load_cached_prime_prices(fetch_date: date) -> Any:
    path = ROOT / "data" / "raw" / f"prices_{fetch_date.isoformat()}.json"
    if not path.exists():
        return load_cached_prime_prices_from_jquants_cache(fetch_date)
    payload = read_json(path)
    return _enrich_cached_prices_with_market(payload.get("prices", [])) or None


def load_cached_prime_prices_from_jquants_cache(fetch_date: date) -> list[dict[str, Any]] | None:
    path = ROOT / "data" / "cache" / "jquants" / "prices" / f"{fetch_date.isoformat()}.json"
    if not path.exists():
        return None
    payload = read_json(path)
    records = payload.get("records", []) if isinstance(payload, dict) else []
    if not records:
        return None
    prime_codes = _cached_prime_stock_codes()
    normalized = [_normalize_daily_price(record) for record in records]
    if prime_codes:
        normalized = [record for record in normalized if record.get("code") in prime_codes]
    normalized = _enrich_cached_prices_with_market(normalized)
    return normalized or None


def _enrich_cached_prices_with_market(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stock_by_code = _allowed_stock_master_by_code(load_config(CONFIG_PATH))
    if not stock_by_code:
        return rows
    enriched = []
    for row in rows:
        code = str(row.get("code") or "")
        if code not in stock_by_code:
            continue
        if row.get("section") and row.get("market_section") and row.get("listing_market"):
            enriched.append(row)
        else:
            enriched.append(attach_market_section_fields(row, stock_by_code[code].get("section", "Unknown")))
    return enriched


def _cached_prime_stock_codes() -> set[str]:
    return set(_allowed_stock_master_by_code(load_config(CONFIG_PATH)).keys())


def _listed_stock_master_path() -> Path:
    listed = ROOT / "data" / "raw" / "listed_stocks_jquants.json"
    if listed.exists():
        return listed
    return ROOT / "data" / "raw" / "prime_stocks_jquants.json"


def _listed_stock_master() -> list[dict[str, Any]]:
    path = _listed_stock_master_path()
    if not path.exists():
        return []
    payload = read_json(path)
    stocks = [_normalize_listed_stock(stock) for stock in payload.get("stocks", []) if stock.get("code")]
    if path.name == "prime_stocks_jquants.json":
        return [
            attach_market_section_fields(stock, "TSEPrime") if market_section_from_row(stock) == "Unknown" else stock
            for stock in stocks
        ]
    return stocks


def _allowed_stock_master_by_code(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stocks = _listed_stock_master()
    allowed = [stock for stock in stocks if market_section_allowed(stock, config)]
    return {str(stock["code"]): stock for stock in allowed if stock.get("code")}


def _market_filter_empty_message(config: dict[str, Any]) -> str:
    stocks = _listed_stock_master()
    allowed_sections = sorted(allowed_market_sections(config))
    section_value_counts: dict[str, int] = {}
    raw_market_counts: dict[str, int] = {}
    for stock in stocks:
        section = market_section_from_row(stock)
        section_value_counts[section] = section_value_counts.get(section, 0) + 1
        raw_value = str(stock.get("market") or stock.get("section") or stock.get("market_section") or stock.get("listing_market") or "Unknown")
        raw_market_counts[raw_value] = raw_market_counts.get(raw_value, 0) + 1
    return (
        "Allowed stock list is empty. Re-run list-stocks and check market_filter settings. "
        f"listed_stock_count={len(stocks)} "
        f"allowed_sections={allowed_sections} "
        f"allow_unknown_market={bool(config.get('market_filter', {}).get('allow_unknown_market', False))} "
        f"section_value_counts={dict(sorted(section_value_counts.items()))} "
        f"raw_market_value_counts={dict(sorted(raw_market_counts.items()))}"
    )


def no_data_days_cache_path() -> Path:
    return ROOT / "data" / "raw" / "no_data_days_jquants.json"


def unsupported_days_cache_path() -> Path:
    return ROOT / "data" / "cache" / "jquants" / "unsupported_ranges.json"


def load_no_data_days_cache() -> dict[str, Any]:
    path = no_data_days_cache_path()
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def load_unsupported_days_cache() -> dict[str, Any]:
    path = unsupported_days_cache_path()
    if not path.exists():
        return {"prices": []}
    payload = read_json(path)
    if isinstance(payload, dict) and "prices" in payload:
        return payload
    if isinstance(payload, dict):
        ranges = [
            {"start": key, "end": key, "reason": value.get("reason", "unknown")}
            for key, value in payload.items()
            if isinstance(value, dict)
        ]
        return {"prices": ranges}
    return {"prices": []}


def no_data_cache_entry(fetch_date: date, cache: dict[str, Any] | None = None) -> dict[str, Any] | None:
    cache = cache if cache is not None else load_no_data_days_cache()
    entry = cache.get(fetch_date.isoformat())
    return entry if isinstance(entry, dict) else None


def unsupported_cache_entry(fetch_date: date, cache: dict[str, Any] | None = None) -> dict[str, Any] | None:
    cache = cache if cache is not None else load_unsupported_days_cache()
    target = fetch_date.isoformat()
    for item in cache.get("prices", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("start", "")) <= target <= str(item.get("end", "")):
            return {
                "provider": "jquants",
                "reason": item.get("reason", "unknown"),
                "source": item.get("source", "unsupported_ranges"),
                "range_start": item.get("start"),
                "range_end": item.get("end"),
            }
    return None


def load_no_data_day(fetch_date: date) -> dict[str, Any] | None:
    return no_data_cache_entry(fetch_date)


def load_unsupported_day(fetch_date: date) -> dict[str, Any] | None:
    return unsupported_cache_entry(fetch_date)


def save_no_data_day(fetch_date: date, reason: str, source: str) -> None:
    cache = load_no_data_days_cache()
    cache[fetch_date.isoformat()] = {
        "provider": "jquants",
        "reason": reason,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
    }
    write_json(no_data_days_cache_path(), cache)


def save_unsupported_day(fetch_date: date, reason: str, source: str) -> None:
    save_unsupported_range(fetch_date, fetch_date, reason, source)


def save_unsupported_range(start_date: date, end_date: date, reason: str, source: str, endpoint: str = "prices") -> None:
    cache = load_unsupported_days_cache()
    ranges = list(cache.get(endpoint, []))
    new_range = {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "reason": reason,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
    }
    ranges.append(new_range)
    cache[endpoint] = _merge_unsupported_ranges(ranges)
    write_json(unsupported_days_cache_path(), cache)


def _merge_unsupported_ranges(ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = sorted(
        [item for item in ranges if item.get("start") and item.get("end")],
        key=lambda item: (str(item["start"]), str(item["end"])),
    )
    merged: list[dict[str, Any]] = []
    for item in normalized:
        if not merged or item.get("reason") != merged[-1].get("reason"):
            merged.append(dict(item))
            continue
        previous_end = date.fromisoformat(str(merged[-1]["end"]))
        current_start = date.fromisoformat(str(item["start"]))
        if current_start <= previous_end + timedelta(days=1):
            if str(item["end"]) > str(merged[-1]["end"]):
                merged[-1]["end"] = item["end"]
            continue
        merged.append(dict(item))
    return merged


def load_cached_price_history(fetch_dates: list[date]) -> list[dict[str, Any]]:
    rows = []
    for fetch_date in fetch_dates:
        cached_rows = load_cached_prime_prices(fetch_date)
        if cached_rows:
            rows.extend(cached_rows)
    return rows


def cache_price_snapshot(fetch_date: date, total_count: int, listed_prices: list[dict[str, Any]]) -> None:
    path = ROOT / "data" / "raw" / f"prices_{fetch_date.isoformat()}.json"
    write_json(
        path,
        {
            "provider": "jquants",
            "date": fetch_date.isoformat(),
            "total_count": total_count,
            "listed_count": len(listed_prices),
            "market_counts": market_section_counts(listed_prices),
            "prices": listed_prices,
        },
    )


def previous_business_dates(target_date: date, count: int) -> list[date]:
    dates = []
    current = target_date
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current -= timedelta(days=1)
    return list(reversed(dates))


def _is_prime_stock(record: dict[str, Any]) -> bool:
    return _normalize_listed_stock(record).get("section") == "TSEPrime"


def _normalize_prime_stock(record: dict[str, Any]) -> dict[str, Any]:
    stock = _normalize_listed_stock(record)
    return stock


def _normalize_listed_stock(record: dict[str, Any]) -> dict[str, Any]:
    section = normalize_market_section(
        _get_first(record, ["section", "Section", "market", "market_name", "MarketName", "MarketCodeName", "market_segment", "MktNm", "market_code", "MarketCode", "market_segment_code", "Mkt"])
    )
    return {
        "code": _get_first(record, ["code", "Code", "local_code", "LocalCode"]),
        "name": _get_first(record, ["name", "Name", "company_name", "CompanyName", "issue_name", "IssueName", "CoName"]),
        "market": _get_first(record, ["market", "market_name", "MarketName", "MarketCodeName", "market_segment", "MktNm"]),
        "section": section,
        "market_section": section,
        "listing_market": section,
        "sector_code": _get_first(record, ["sector_code", "SectorCode", "sector_33_code", "Sector33Code", "sector17_code", "Sector17Code", "S33", "S17"]),
        "sector_name": _get_first(record, ["sector_name", "SectorName", "sector_33_name", "Sector33Name", "sector17_name", "Sector17Name", "S33Nm", "S17Nm"]),
        "scale_category": _get_first(record, ["scale_category", "ScaleCategory", "scale_category_name", "ScaleCategoryName", "ScaleCat"]),
    }


def _normalize_daily_price(record: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "code": _get_first(record, ["code", "Code", "LocalCode"]),
        "date": _format_jquants_date(_get_first(record, ["date", "Date"])),
        "open": _get_number(record, ["open", "Open", "O", "AdjustmentOpen"]),
        "high": _get_number(record, ["high", "High", "H", "AdjustmentHigh"]),
        "low": _get_number(record, ["low", "Low", "L", "AdjustmentLow"]),
        "close": _get_number(record, ["close", "Close", "C", "AdjustmentClose"]),
        "volume": _get_number(record, ["volume", "Volume", "Vo", "AdjustmentVolume"]),
    }
    turnover_value = _get_number(record, ["turnover_value", "TurnoverValue", "Va"])
    if turnover_value is not None:
        normalized["turnover_value"] = turnover_value
    return normalized


def _normalize_daily_price_with_market(record: dict[str, Any], stock_by_code: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normalized = _normalize_daily_price(record)
    stock = stock_by_code.get(str(normalized.get("code")), {})
    section = stock.get("section") or market_section_from_row(record)
    normalized.update(
        {
            "name": stock.get("name", ""),
            "sector_name": stock.get("sector_name", ""),
            "section": normalize_market_section(section),
            "market_section": normalize_market_section(section),
            "listing_market": normalize_market_section(section),
        }
    )
    return normalized


def _get_first(record: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def _get_number(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _format_jquants_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def add_business_days(start_date: date, offset: int) -> date:
    current = start_date
    added = 0
    while added < offset:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def next_business_day(value: date) -> date:
    current = value
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def build_day_paths(run_date: date, day_key: str) -> dict[str, Path]:
    config = load_config(CONFIG_PATH)
    profile_id = profile_id_from(config)
    month_key = run_date.strftime("%Y-%m")
    date_key = run_date.strftime("%Y%m%d")
    file_key = f"{date_key}_{day_key}"
    return {
        "screening_run": ROOT / "logs" / "screening" / profile_id / month_key / f"{file_key}_screening_run.json",
        "candidates": ROOT / "logs" / "screening" / profile_id / month_key / f"{file_key}_candidates_50.json",
        "scoring": ROOT / "logs" / "scoring" / profile_id / month_key / f"{file_key}_ai_scores.json",
        "decisions": ROOT / "logs" / "scoring" / profile_id / month_key / f"{file_key}_trade_decisions.json",
        "orders": ROOT / "logs" / "trades" / profile_id / month_key / f"{file_key}_paper_orders.json",
        "trades_daily": ROOT / "logs" / "trades" / profile_id / month_key / f"{file_key}.json",
        "closed_trades": ROOT / "logs" / "trades" / profile_id / month_key / f"{file_key}_closed_trades.json",
        "pnl": ROOT / "logs" / "trades" / profile_id / month_key / f"{file_key}_pnl.json",
        "safety": ROOT / "logs" / "safety" / profile_id / month_key / f"{file_key}.json",
        "portfolio": ROOT / "logs" / "portfolio" / profile_id / month_key / f"{file_key}.json",
        "portfolio_summary": ROOT / "logs" / "portfolio" / profile_id / month_key / f"{file_key}_summary.json",
        "reflections": ROOT / "logs" / "reflections" / profile_id / month_key / f"{file_key}_reflections.json",
        "report": ROOT / "reports" / "paper" / profile_id / month_key / f"{file_key}.md",
        "article": ROOT / "reports" / "articles" / "daily" / run_date.strftime("%Y") / run_date.strftime("%m") / profile_id / f"{file_key}.md",
    }


def build_data_provider(config: dict[str, Any], run_date: date, run_id: str) -> Any:
    provider_name = config.get("data_provider", "dummy")
    if provider_name == "dummy":
        return DummyDataProvider(config, run_date, run_id)
    if provider_name == "jquants":
        return JQuantsDataProvider(
            ROOT / ".env",
            plan=_jquants_plan(config),
            requests_per_minute=_jquants_requests_per_minute(config),
            parallel_fetch=_jquants_parallel_fetch(config),
            max_parallel_requests=_jquants_max_parallel_requests(config),
        )
    raise ValueError(f"Unsupported data_provider: {provider_name}")


def load_config(path: Path) -> dict[str, Any]:
    runtime_settings = _runtime_settings_or_defaults()
    profile_id = ACTIVE_PROFILE_ID
    if not RUNTIME_SETTINGS and profile_id == DEFAULT_PROFILE_ID:
        profile_id = str(runtime_settings["profile_id"])
    config = load_profile(profile_id)
    _apply_runtime_provider_settings(config)
    _apply_jquants_plan_settings(config)
    if STORAGE_MODE_OVERRIDE:
        config.setdefault("storage", {})["save_mode"] = STORAGE_MODE_OVERRIDE
    if _fast_analysis_enabled(config):
        config.setdefault("storage", {}).setdefault("save_mode", "compact")
        analysis = config.setdefault("analysis", {})
        analysis["save_rejected_candidates"] = False
        analysis["save_selection_quality_detail"] = False
        analysis["save_replay_detail"] = False
        analysis["save_recovery_detail"] = False
        analysis["save_score_audit_detail"] = False
        analysis["save_backtest_daily_reports"] = False
        reporting = config.setdefault("reporting", {})
        reporting["generate_articles_in_backtest"] = False
        reporting["generate_daily_markdown_in_backtest"] = False
    return config


def _load_provider_runtime_config() -> dict[str, Any]:
    if not PROVIDER_CONFIG_PATH.exists():
        return {}
    try:
        payload = load_versioned_config(PROVIDER_CONFIG_PATH)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _config_get(config: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _runtime_settings_or_defaults() -> dict[str, Any]:
    if RUNTIME_SETTINGS:
        return RUNTIME_SETTINGS
    class EmptyArgs:
        provider = None
        profile = None
        jquants_plan = None

    return resolve_runtime_settings(EmptyArgs())


def _apply_runtime_provider_settings(config: dict[str, Any]) -> None:
    settings = _runtime_settings_or_defaults()
    sources = dict(settings.get("sources", {}))
    config["_value_sources"] = sources
    config["data_provider"] = settings["provider"]
    broker = config.setdefault("broker", {})
    broker["mode"] = settings["broker_mode"]
    broker["provider"] = settings["broker_mode"]
    operation = config.setdefault("operation", {})
    operation["auto_order_enabled"] = bool(settings["auto_order_enabled"])
    jquants = config.setdefault("jquants", {})
    jquants["plan"] = settings["jquants_plan"]
    if settings.get("entry_timing"):
        config.setdefault("backtest", {})["entry_timing"] = _normalize_entry_timing(settings["entry_timing"])


def _apply_jquants_plan_settings(config: dict[str, Any]) -> None:
    jquants = config.setdefault("jquants", {})
    class Args:
        jquants_plan = JQUANTS_PLAN_OVERRIDE

    resolution = resolve_jquants_plan(
        args=Args(),
        config={"jquants": jquants},
        config_root=ROOT,
        provider_config=_load_provider_runtime_config(),
    )
    plan = resolution.plan
    warnings = config.setdefault("_warnings", [])
    warnings.extend(resolution.warnings)

    jquants.update(resolution.config)
    jquants["plan"] = plan
    jquants["capability_status"] = resolution.capabilities
    jquants["requests_per_minute"] = resolution.requests_per_minute
    jquants["rate_limit_per_minute"] = resolution.requests_per_minute
    jquants["parallel_fetch"] = resolution.parallel_fetch
    jquants["max_parallel_requests"] = resolution.max_parallel_requests
    config.setdefault("_value_sources", {})["jquants_plan"] = resolution.source
    config["_jquants_plan_resolution"] = {
        "plan": resolution.plan,
        "source": resolution.source,
        "config_path": resolution.config_path,
        "capabilities": resolution.capabilities,
        "requests_per_minute": resolution.requests_per_minute,
        "parallel_fetch": resolution.parallel_fetch,
        "supported_date_ranges": resolution.supported_date_ranges,
    }

    features = config.setdefault("features", {})
    if not jquants_has_capability(plan, "topix_prices"):
        if features.get("topix_relative_strength"):
            warnings.append("topix_relative_strength disabled because J-Quants plan is free")
        features["topix_relative_strength"] = False
        jquants["relative_strength_benchmark"] = "prime_market_average"
    else:
        jquants["relative_strength_benchmark"] = "topix" if features.get("topix_relative_strength") else "prime_market_average"

    if not jquants_has_capability(plan, "investor_types"):
        if features.get("investor_context"):
            warnings.append("investor_context disabled because J-Quants plan is free")
        features["investor_context"] = False

    if plan != "light" and features.get("longer_history_backtest"):
        warnings.append("longer_history_backtest disabled because J-Quants plan is free")
        features["longer_history_backtest"] = False


def _load_jquants_config_file() -> dict[str, Any]:
    path = ROOT / "config" / "jquants.yaml"
    if not path.exists():
        return {}
    try:
        if yaml is not None:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            payload = _load_simple_yaml(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    section = payload.get("jquants", payload)
    return section if isinstance(section, dict) else {}


def _jquants_plan(config: dict[str, Any]) -> str:
    return normalize_jquants_plan(config.get("jquants", {}).get("plan", "free"))


def jquants_earliest_supported_date(config: dict[str, Any], endpoint: str = "prices") -> date | None:
    jquants = config.get("jquants", {}) if isinstance(config.get("jquants"), dict) else {}
    plan = _jquants_plan(config)
    payload = jquants.get("earliest_supported_date", {})
    value = None
    if isinstance(payload, dict):
        endpoint_payload = payload.get(endpoint)
        if isinstance(endpoint_payload, dict):
            value = endpoint_payload.get(plan)
        if value is None:
            value = payload.get(plan)
    elif isinstance(payload, str):
        value = payload
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def jquants_supported_date_ranges(config: dict[str, Any]) -> dict[str, str]:
    return {
        endpoint: (jquants_earliest_supported_date(config, endpoint) or jquants_earliest_supported_date(config, "prices") or "").isoformat()
        if (jquants_earliest_supported_date(config, endpoint) or jquants_earliest_supported_date(config, "prices"))
        else "unknown"
        for endpoint in ["prices", "topix_prices", "investor_types", "earnings_calendar", "financial_statements"]
    }


def _jquants_plan_settings(jquants: dict[str, Any], plan: str) -> dict[str, Any]:
    plans = jquants.get("plans", {})
    if isinstance(plans, dict) and isinstance(plans.get(plan), dict):
        return dict(plans[plan])
    defaults = {
        "free": {"requests_per_minute": 5, "parallel_fetch": False},
        "light": {"requests_per_minute": 60, "parallel_fetch": True, "max_parallel_requests": 4},
    }
    return defaults.get(plan, defaults["free"])


def _jquants_requests_per_minute(config: dict[str, Any]) -> int:
    jquants = config.get("jquants", {}) if isinstance(config.get("jquants"), dict) else {}
    plan_settings = _jquants_plan_settings(jquants, _jquants_plan(config))
    return int(plan_settings.get("requests_per_minute", jquants.get("requests_per_minute", jquants.get("rate_limit_per_minute", 5))))


def _jquants_parallel_fetch(config: dict[str, Any]) -> bool:
    jquants = config.get("jquants", {}) if isinstance(config.get("jquants"), dict) else {}
    plan = _jquants_plan(config)
    plan_settings = _jquants_plan_settings(jquants, plan)
    return plan == "light" and bool(plan_settings.get("parallel_fetch", jquants.get("parallel_fetch", False)))


def _jquants_max_parallel_requests(config: dict[str, Any]) -> int:
    return max(1, int(config.get("jquants", {}).get("max_parallel_requests", 4)))


def _load_earnings_calendar_for_period(
    start_date: date,
    end_date: date,
    config: dict[str, Any],
    force_refresh: bool | None = None,
) -> dict[str, Any]:
    cache_root = ROOT / "data" / "cache"
    cache_path = cache_root / "jquants" / "earnings_calendar" / f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"
    allowed, stop_reason = _api_call_allowed("earnings_calendar")
    if not allowed:
        pipeline = _earnings_pipeline_metadata(
            enabled=True,
            cache_path=str(cache_path),
            cache_exists=cache_path.exists(),
            cache_records=0,
            cache_loaded=False,
            records_loaded=0,
            fetch_start=start_date.isoformat(),
            fetch_end=end_date.isoformat(),
            reason=stop_reason,
        )
        return {"records": [], "metadata": {"filter_available": False, "warning": stop_reason, "disabled_reason": stop_reason, "pipeline": pipeline}}
    try:
        provider = JQuantsDataProvider(
            ROOT / ".env",
            timeout_seconds=int(config.get("jquants", {}).get("request_timeout_seconds", 20)),
            plan=_jquants_plan(config),
            requests_per_minute=_jquants_requests_per_minute(config),
            parallel_fetch=_jquants_parallel_fetch(config),
            max_parallel_requests=_jquants_max_parallel_requests(config),
        )
        payload = provider.fetch_earnings_calendar_period_cached(
            cache_root,
            start_date=start_date,
            end_date=end_date,
            force_refresh=FORCE_REFRESH_ACTIVE if force_refresh is None else force_refresh,
        )
        records = payload.get("records", []) if isinstance(payload.get("records"), list) else []
        _log_jquants_api_event(
            endpoint="earnings_calendar",
            plan=_jquants_plan(config),
            cache_hit=bool(payload.get("from_cache")),
            status=_provider_payload_status(payload),
            records=len(records),
            saved=bool(payload.get("saved")),
            cache_path=payload.get("cache_path"),
            reason=str(payload.get("reason") or ""),
            **_payload_http_log_fields(payload, getattr(provider, "last_request_metadata", {}) or {}),
        )
    except Exception as exc:
        _record_api_error("earnings_calendar", _api_error_status(exc))
        _log_jquants_api_event(
            endpoint="earnings_calendar",
            plan=_jquants_plan(config),
            cache_hit=False,
            status=_api_error_status(exc),
            records=0,
            saved=False,
            cache_path=str(cache_path),
            error=str(exc),
            **_api_error_log_fields(exc),
        )
        payload = {
            "records": [],
            "cache_path": str(cache_path),
            "from_cache": False,
            "fallback_used": False,
            "warning": str(exc),
            "filter_available": False,
            "available": False,
            "saved": False,
            "reason": _api_error_status(exc),
        }
        records = []
    records = payload.get("records", []) if isinstance(payload.get("records"), list) else []
    raw_reason = str(payload.get("reason") or payload.get("warning") or "")
    cache_exists = Path(str(payload.get("cache_path") or cache_path)).exists()
    reason = raw_reason
    if not records:
        if raw_reason == "empty_response":
            reason = "empty_response"
        elif not cache_exists:
            reason = "cache_missing"
        else:
            reason = raw_reason or "empty_response"
    pipeline = _earnings_pipeline_metadata(
        enabled=True,
        cache_path=str(payload.get("cache_path") or cache_path),
        cache_exists=cache_exists,
        cache_records=len(records),
        cache_loaded=bool(records),
        records_loaded=len(records),
        fetch_start=start_date.isoformat(),
        fetch_end=end_date.isoformat(),
        index_built=bool(records),
        reason=reason,
    )
    if not payload.get("filter_available", payload.get("available", False)) and bool(config.get("earnings_filter", {}).get("fail_open", True)):
        print(f"earnings filter warning: disabled by fail_open. reason={reason}")
    return {
        "records": records,
        "metadata": {
            "cache_path": payload.get("cache_path"),
            "from_cache": payload.get("from_cache"),
            "fallback_used": payload.get("fallback_used"),
            "warning": payload.get("warning"),
            "reason": reason,
            "filter_available": bool(payload.get("filter_available", payload.get("available", False))),
            "available": bool(payload.get("available", False)),
            "fetch_start": start_date.isoformat(),
            "fetch_end": end_date.isoformat(),
            "cache_exists": pipeline["cache_exists"],
            "cache_records": pipeline["cache_records"],
            "pipeline": pipeline,
        },
    }


def _load_earnings_calendar_for_date(target_date: date, config: dict[str, Any], force_refresh: bool | None = None) -> dict[str, Any]:
    enabled = bool(config.get("earnings_filter", {}).get("enabled", False))
    if not bool(config.get("earnings_filter", {}).get("enabled", False)):
        return {
            "records": [],
            "metadata": {
                "filter_available": False,
                "reason": "earnings_filter disabled",
                "pipeline": _earnings_pipeline_metadata(
                    enabled=False,
                    cache_path="",
                    cache_exists=False,
                    cache_records=0,
                    cache_loaded=False,
                    records_loaded=0,
                    reason="earnings_filter disabled",
                ),
            },
        }
    preloaded = _jquants_api_session().setdefault("payloads", {}).get("earnings_calendar")
    if isinstance(preloaded, dict):
        records = preloaded.get("records", []) if isinstance(preloaded.get("records"), list) else []
        metadata = preloaded.get("metadata", {}) if isinstance(preloaded.get("metadata"), dict) else {}
        pipeline = _earnings_pipeline_metadata(
            enabled=True,
            cache_path=str(metadata.get("cache_path") or ""),
            cache_exists=bool(metadata.get("cache_exists")),
            cache_records=int(metadata.get("cache_records") or len(records)),
            cache_loaded=bool(metadata.get("filter_available") or metadata.get("available")),
            records_loaded=len(records),
            fetch_start=str(preloaded.get("start_date") or metadata.get("fetch_start") or ""),
            fetch_end=str(preloaded.get("end_date") or metadata.get("fetch_end") or ""),
            index_built=bool(records),
            reason=str(metadata.get("warning") or metadata.get("reason") or ""),
        )
        return {
            "records": records,
            "metadata": {
                **metadata,
                "filter_available": bool(metadata.get("filter_available", metadata.get("available", bool(records)))),
                "from_preloaded": True,
                "pipeline": pipeline,
            },
        }
    cache_root = ROOT / "data" / "cache"
    cache_path = cache_root / "jquants" / "earnings_calendar" / f"{target_date.isoformat()}.json"
    requested_force_refresh = FORCE_REFRESH_ACTIVE if force_refresh is None else force_refresh
    if target_date < date.today():
        resolved_cache_path = _find_earnings_calendar_cache_for_date(cache_root, target_date)
        cache_load_called = resolved_cache_path is not None
        if resolved_cache_path is not None:
            payload = _read_earnings_calendar_cache(resolved_cache_path, target_date)
        else:
            payload = {
                "records": [],
                "cache_path": str(cache_path),
                "cache_date": target_date.isoformat(),
                "from_cache": False,
                "fallback_used": False,
                "warning": "historical earnings calendar cache unavailable; disabled to avoid future leak",
                "filter_available": False,
            }
        records = payload.get("records", []) if isinstance(payload.get("records"), list) else []
        payload["pipeline"] = _earnings_pipeline_metadata(
            enabled=True,
            cache_path=payload.get("cache_path") or str(cache_path),
            cache_exists=bool(resolved_cache_path and resolved_cache_path.exists()),
            cache_records=len(records),
            cache_loaded=bool(cache_load_called and payload.get("filter_available", False)),
            records_loaded=len(records),
            fetch_start=str(payload.get("cache_date") or target_date.isoformat()),
            fetch_end=str(payload.get("cache_date") or target_date.isoformat()),
            index_built=bool(records),
            reason=str(payload.get("warning") or payload.get("reason") or ""),
        )
        if not payload.get("filter_available", False) and bool(config.get("earnings_filter", {}).get("fail_open", True)):
            print(f"earnings filter warning: disabled by fail_open. reason={payload.get('warning')}")
        return {
            "records": records,
            "metadata": {
                "cache_path": payload.get("cache_path"),
                "cache_date": payload.get("cache_date"),
                "from_cache": payload.get("from_cache"),
                "fallback_used": payload.get("fallback_used"),
                "warning": payload.get("warning"),
                "filter_available": payload.get("filter_available", False),
                "pipeline": payload.get("pipeline", {}),
            },
        }
    allowed, stop_reason = _api_call_allowed("earnings_calendar")
    if not allowed:
        return {"records": [], "metadata": {"filter_available": False, "warning": stop_reason, "disabled_reason": stop_reason}}
    try:
        provider = JQuantsDataProvider(
            ROOT / ".env",
            timeout_seconds=int(config.get("jquants", {}).get("request_timeout_seconds", 20)),
            plan=_jquants_plan(config),
            requests_per_minute=_jquants_requests_per_minute(config),
            parallel_fetch=_jquants_parallel_fetch(config),
            max_parallel_requests=_jquants_max_parallel_requests(config),
        )
        payload = provider.fetch_earnings_calendar_cached(
            cache_root,
            target_date=target_date,
            force_refresh=requested_force_refresh,
        )
        _log_jquants_api_event(
            endpoint="earnings_calendar",
            plan=_jquants_plan(config),
            cache_hit=bool(payload.get("from_cache")),
            status=_provider_payload_status(payload),
            records=len(payload.get("records", [])),
            saved=bool(payload.get("saved")),
            cache_path=payload.get("cache_path"),
            reason=str(payload.get("reason") or ""),
        )
    except Exception as exc:
        _record_api_error("earnings_calendar", _api_error_status(exc))
        _log_jquants_api_event(
            endpoint="earnings_calendar",
            plan=_jquants_plan(config),
            cache_hit=False,
            status=_api_error_status(exc),
            records=0,
            saved=False,
            cache_path=str(cache_path),
            error=str(exc),
            **_api_error_log_fields(exc),
        )
        payload = {
            "records": [],
            "cache_path": "",
            "cache_date": target_date.isoformat(),
            "from_cache": False,
            "fallback_used": False,
            "warning": str(exc),
            "filter_available": False,
        }
    if not payload.get("filter_available", False) and bool(config.get("earnings_filter", {}).get("fail_open", True)):
        print(f"earnings filter warning: disabled by fail_open. reason={payload.get('warning')}")
    records = payload.get("records", []) if isinstance(payload.get("records"), list) else []
    payload["pipeline"] = _earnings_pipeline_metadata(
        enabled=True,
        cache_path=payload.get("cache_path") or str(cache_path),
        cache_exists=Path(str(payload.get("cache_path") or cache_path)).exists(),
        cache_records=len(records),
        cache_loaded=bool(payload.get("from_cache") and payload.get("filter_available", False)),
        records_loaded=len(records),
        fetch_start=target_date.isoformat(),
        fetch_end=target_date.isoformat(),
        index_built=bool(records),
        reason=str(payload.get("warning") or payload.get("reason") or ""),
    )
    return {
        "records": records,
        "metadata": {
            "cache_path": payload.get("cache_path"),
            "cache_date": payload.get("cache_date"),
            "from_cache": payload.get("from_cache"),
            "fallback_used": payload.get("fallback_used"),
            "warning": payload.get("warning"),
            "filter_available": payload.get("filter_available", False),
            "pipeline": payload.get("pipeline", {}),
        },
    }


def _find_earnings_calendar_cache_for_date(cache_root: Path, target_date: date) -> Path | None:
    directory = cache_root / "jquants" / "earnings_calendar"
    exact = directory / f"{target_date.isoformat()}.json"
    if exact.exists():
        return exact
    if not directory.exists():
        return None
    candidates: list[tuple[date, Path]] = []
    for path in directory.glob("*.json"):
        try:
            cache_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if cache_date <= target_date:
            candidates.append((cache_date, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _earnings_pipeline_metadata(
    *,
    enabled: bool,
    cache_path: str,
    cache_exists: bool,
    cache_records: int,
    cache_loaded: bool,
    records_loaded: int,
    fetch_start: str = "",
    fetch_end: str = "",
    index_built: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "feature_enabled": enabled,
        "fetch_start": fetch_start,
        "fetch_end": fetch_end,
        "cache_path": cache_path,
        "cache_exists": cache_exists,
        "cache_records": int(cache_records or 0),
        "cache_loaded": cache_loaded,
        "index_built": index_built,
        "candidate_matching_called": False,
        "earnings_records_loaded": int(records_loaded or 0),
        "matched_candidates": 0,
        "rejected_candidates": 0,
        "reason": reason,
    }


def _read_earnings_calendar_cache(cache_path: Path, target_date: date) -> dict[str, Any]:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "records": [],
            "cache_path": str(cache_path),
            "cache_date": target_date.isoformat(),
            "from_cache": False,
            "fallback_used": False,
            "warning": str(exc),
            "filter_available": False,
        }
    records = payload.get("records", []) if isinstance(payload, dict) else []
    usable = isinstance(records, list) and len(records) > 0
    return {
        "records": records if isinstance(records, list) else [],
        "cache_path": str(cache_path),
        "cache_date": target_date.isoformat(),
        "from_cache": True,
        "fallback_used": False,
        "warning": "" if usable else "empty_cache",
        "filter_available": usable,
        "available": usable,
        "usable": usable,
        "reason": "" if usable else "empty_cache",
    }


def _load_investor_context_for_date(target_date: date, config: dict[str, Any], force_refresh: bool | None = None) -> dict[str, Any]:
    enabled = bool(config.get("features", {}).get("investor_context")) and bool(config.get("scoring", {}).get("use_investor_context_score"))
    if not enabled:
        return {
            "records": [],
            "context": dict(INVESTOR_CONTEXT_EMPTY),
            "metadata": {"available": False, "reason": "investor_context disabled"},
        }
    preloaded = _jquants_api_session().setdefault("payloads", {}).get("investor_types")
    if isinstance(preloaded, dict):
        records = preloaded.get("records", [])
        context = build_investor_context(records, target_date)
        metadata = preloaded.get("metadata", {}) if isinstance(preloaded.get("metadata"), dict) else {}
        if records:
            return {
                "records": records,
                "context": context,
                "metadata": {
                    **metadata,
                    "available": True,
                    "from_preloaded": True,
                    "latest_investor_data_week": context.get("investor_context_week"),
                    "investor_context_source": context.get("investor_context_source"),
                    "investor_context_score": context.get("investor_context_score"),
                },
            }
        return {
            "records": [],
            "context": dict(INVESTOR_CONTEXT_EMPTY),
            "metadata": {
                **metadata,
                "available": False,
                "from_preloaded": True,
                "warning": metadata.get("warning") or metadata.get("reason") or "investor_types unavailable",
                "disabled_reason": metadata.get("reason") or metadata.get("disabled_reason"),
            },
        }
    disabled_reason = _jquants_api_session().setdefault("disabled_features_reason", {}).get("investor_types")
    if disabled_reason:
        return {
            "records": [],
            "context": dict(INVESTOR_CONTEXT_EMPTY),
            "metadata": {"available": False, "warning": disabled_reason, "disabled_reason": disabled_reason},
        }
    if not jquants_has_capability(_jquants_plan(config), "investor_types"):
        print("investor_context warning: investor_types disabled for current J-Quants plan.")
        return {
            "records": [],
            "context": dict(INVESTOR_CONTEXT_EMPTY),
            "metadata": {"available": False, "warning": "investor_types disabled for current J-Quants plan"},
        }
    try:
        provider: JQuantsDataProvider | None = None
        payload = {"records": [], "available": False, "warning": "investor_types unavailable"}
        ranges = _investor_types_fetch_ranges(target_date)
        consecutive_empty = 0
        for index, (start_date, end_date) in enumerate(ranges):
            cached = _read_investor_types_cache_payload(start_date, end_date)
            if cached is not None:
                payload = cached
                retry_range = ""
                _log_jquants_api_event(
                    endpoint="investor_types",
                    plan=_jquants_plan(config),
                    cache_hit=True,
                    status="200",
                    records=len(payload.get("records", [])),
                    saved=False,
                    cache_path=payload.get("cache_path"),
                    reason="",
                )
                break
            allowed, stop_reason = _api_call_allowed("investor_types")
            if not allowed:
                payload = _api_unavailable_payload("investor_types", stop_reason)
                break
            if provider is None:
                provider = JQuantsDataProvider(
                    ROOT / ".env",
                    timeout_seconds=int(config.get("jquants", {}).get("request_timeout_seconds", 20)),
                    plan=_jquants_plan(config),
                    requests_per_minute=_jquants_requests_per_minute(config),
                    parallel_fetch=_jquants_parallel_fetch(config),
                    max_parallel_requests=_jquants_max_parallel_requests(config),
                )
            payload = _fetch_investor_types_with_retries(provider, start_date, end_date, config, force_refresh=FORCE_REFRESH_ACTIVE if force_refresh is None else force_refresh)
            retry_range = ""
            status = _provider_payload_status(payload)
            terminal_error = status in {"auth_or_plan_error", "bad_request", "endpoint_not_found"}
            if not payload.get("records") and not terminal_error and index + 1 < len(ranges):
                consecutive_empty += 1
                retry_start, retry_end = ranges[index + 1]
                retry_range = f"{retry_start.isoformat()}_to_{retry_end.isoformat()}"
                payload["reason"] = payload.get("reason") or "empty_response"
                payload["warning"] = payload.get("warning") or "api_success_but_empty"
            elif not payload.get("records"):
                consecutive_empty += 1
            _log_jquants_api_event(
                endpoint="investor_types",
                plan=_jquants_plan(config),
                cache_hit=bool(payload.get("from_cache")),
                status=_provider_payload_status(payload),
                records=len(payload.get("records", [])),
                saved=bool(payload.get("saved")),
                cache_path=payload.get("cache_path"),
                reason=str(payload.get("reason") or ""),
                retry_range=retry_range,
                attempt=payload.get("attempt") or "",
                error_reason=str(payload.get("reason") or ""),
                **_payload_http_log_fields(payload, getattr(provider, "last_request_metadata", {}) if provider is not None else {}),
            )
            if payload.get("records"):
                break
            if terminal_error:
                _record_api_error("investor_types", status)
                break
            if consecutive_empty >= 3:
                payload["warning"] = payload.get("warning") or "api_success_but_empty"
                payload["reason"] = payload.get("reason") or "empty_response"
                _record_api_error("investor_types", "empty_response")
                break
    except Exception as exc:
        _record_api_error("investor_types", _api_error_status(exc))
        _log_jquants_api_event(
            endpoint="investor_types",
            plan=_jquants_plan(config),
            cache_hit=False,
            status=_api_error_status(exc),
            records=0,
            saved=False,
            cache_path="",
            error=str(exc),
            **_api_error_log_fields(exc),
        )
        payload = {
            "records": [],
            "cache_path": "",
            "from_cache": False,
            "fallback_used": False,
            "warning": str(exc),
            "available": False,
        }
    records = payload.get("records", [])
    context = build_investor_context(records, target_date) if payload.get("available", False) else dict(INVESTOR_CONTEXT_EMPTY)
    if context.get("investor_context_source") == "unavailable" and payload.get("available", False):
        payload["warning"] = payload.get("warning") or "investor_context unavailable"
    return {
        "records": records,
        "context": context,
        "metadata": {
            "available": payload.get("available", False),
            "cache_path": payload.get("cache_path"),
            "from_cache": payload.get("from_cache"),
            "fallback_used": payload.get("fallback_used"),
            "warning": payload.get("warning"),
            "disabled_reason": payload.get("reason"),
            "latest_investor_data_week": context.get("investor_context_week"),
            "investor_context_source": context.get("investor_context_source"),
            "investor_context_score": context.get("investor_context_score"),
        },
    }


def _load_investor_types_for_period(start_date: date, end_date: date, config: dict[str, Any]) -> dict[str, Any]:
    requested_fetch_start = start_date - timedelta(weeks=26)
    earliest = _investor_types_earliest_supported_date(config)
    fetch_start = max(requested_fetch_start, earliest) if earliest else requested_fetch_start
    clamp_reason = "plan_supported_date" if earliest and requested_fetch_start < earliest else ""
    fetch_end = _business_day_on_or_before(end_date - timedelta(days=14))
    if fetch_end < fetch_start:
        fetch_end = _business_day_on_or_before(end_date)
    metadata: dict[str, Any] = {
        "available": False,
        "warning": "investor_types unavailable",
        "reason": "unavailable",
        "from_period_preload": True,
        "requested_fetch_start": requested_fetch_start.isoformat(),
        "clamped_fetch_start": fetch_start.isoformat(),
        "clamp_reason": clamp_reason,
    }
    if clamp_reason:
        print(
            "investor_types fetch_start clamped: "
            f"requested_fetch_start={requested_fetch_start.isoformat()} "
            f"clamped_fetch_start={fetch_start.isoformat()} "
            f"fetch_end={fetch_end.isoformat()} "
            f"clamp_reason={clamp_reason}"
        )
    if not jquants_has_capability(_jquants_plan(config), "investor_types"):
        metadata["warning"] = "investor_types disabled for current J-Quants plan"
        metadata["reason"] = "capability_disabled"
        return {
            "records": [],
            "metadata": metadata,
            "start_date": fetch_start.isoformat(),
            "end_date": fetch_end.isoformat(),
        }
    provider = JQuantsDataProvider(
        ROOT / ".env",
        timeout_seconds=int(config.get("jquants", {}).get("request_timeout_seconds", 20)),
        plan=_jquants_plan(config),
        requests_per_minute=_jquants_requests_per_minute(config),
        parallel_fetch=_jquants_parallel_fetch(config),
        max_parallel_requests=_jquants_max_parallel_requests(config),
    )
    chunks = _investor_types_period_chunks(fetch_start, fetch_end)
    records: list[dict[str, Any]] = []
    success_chunks = []
    failed_chunks = []
    cache_paths = []
    for chunk_start, chunk_end in chunks:
        payload: dict[str, Any]
        cached = _read_investor_types_cache_payload(chunk_start, chunk_end)
        if cached is not None:
            payload = cached
            status = "200"
            reason = ""
            _log_jquants_api_event(
                endpoint="investor_types",
                plan=_jquants_plan(config),
                cache_hit=True,
                status=status,
                records=len(payload.get("records", [])),
                saved=False,
                cache_path=payload.get("cache_path"),
                reason=reason,
                fetch_from=chunk_start.isoformat(),
                fetch_to=chunk_end.isoformat(),
            )
        else:
            allowed, stop_reason = _api_call_allowed("investor_types")
            if not allowed:
                payload = _api_unavailable_payload("investor_types", stop_reason)
                status = stop_reason
                reason = stop_reason
            else:
                payload = _fetch_investor_types_with_retries(provider, chunk_start, chunk_end, config, force_refresh=FORCE_REFRESH_ACTIVE)
                status = _provider_payload_status(payload)
                reason = str(payload.get("reason") or status)
            _log_jquants_api_event(
                endpoint="investor_types",
                plan=_jquants_plan(config),
                cache_hit=bool(payload.get("from_cache")),
                status=status,
                records=len(payload.get("records", [])),
                saved=bool(payload.get("saved")),
                cache_path=payload.get("cache_path"),
                reason=reason,
                attempt=payload.get("attempt") or "",
                error_reason=reason,
                fetch_from=chunk_start.isoformat(),
                fetch_to=chunk_end.isoformat(),
                **_payload_http_log_fields(payload, getattr(provider, "last_request_metadata", {}) or {}),
            )
        chunk_records = payload.get("records", []) if isinstance(payload.get("records"), list) else []
        if chunk_records:
            records.extend(chunk_records)
            success_chunks.append({"start": chunk_start.isoformat(), "end": chunk_end.isoformat(), "records": len(chunk_records)})
            if payload.get("cache_path"):
                cache_paths.append(payload.get("cache_path"))
        else:
            failed_chunks.append({"start": chunk_start.isoformat(), "end": chunk_end.isoformat(), "reason": reason or "empty_response"})
    records = _dedupe_investor_type_records(records)
    chunk_summary = {
        "investor_types_chunks_total": len(chunks),
        "investor_types_chunks_success": len(success_chunks),
        "investor_types_chunks_failed": len(failed_chunks),
        "investor_types_records_loaded": len(records),
        "investor_types_fetch_requested_start": requested_fetch_start.isoformat(),
        "investor_types_fetch_clamped_start": fetch_start.isoformat(),
        "investor_types_fetch_start": fetch_start.isoformat(),
        "investor_types_fetch_end": fetch_end.isoformat(),
        "investor_types_clamp_reason": clamp_reason,
        "investor_types_failed_chunks": failed_chunks,
        "investor_types_disabled_reason": "",
    }
    _jquants_api_session()["investor_types_fetch_summary"] = chunk_summary
    if records:
        metadata.update(
            {
                "available": True,
                "warning": "",
                "reason": "",
                "cache_path": cache_paths[0] if cache_paths else "",
                "cache_paths": cache_paths,
                "from_cache": len(success_chunks) == len(chunks),
                "chunks_total": len(chunks),
                "chunks_success": len(success_chunks),
                "chunks_failed": len(failed_chunks),
                "records_loaded": len(records),
                "failed_chunks": failed_chunks,
            }
        )
    else:
        reason = failed_chunks[0]["reason"] if failed_chunks else "unavailable"
        metadata["warning"] = reason
        metadata["reason"] = reason
        chunk_summary["investor_types_disabled_reason"] = reason
        metadata["chunks_total"] = len(chunks)
        metadata["chunks_success"] = 0
        metadata["chunks_failed"] = len(failed_chunks)
        metadata["records_loaded"] = 0
        metadata["failed_chunks"] = failed_chunks
        _record_api_error("investor_types", reason)
    return {
        "records": records if isinstance(records, list) else [],
        "metadata": metadata,
        "start_date": fetch_start.isoformat(),
        "end_date": fetch_end.isoformat(),
        "requested_start_date": requested_fetch_start.isoformat(),
        "clamped_start_date": fetch_start.isoformat(),
        "clamp_reason": clamp_reason,
    }


def _investor_types_earliest_supported_date(config: dict[str, Any]) -> date | None:
    earliest = jquants_earliest_supported_date(config, "investor_types") or jquants_earliest_supported_date(config, "prices")
    if earliest is None and _jquants_plan(config) == "light":
        return date(2021, 5, 31)
    return earliest


def _investor_types_period_chunks(start_date: date, end_date: date) -> list[tuple[date, date]]:
    if end_date < start_date:
        return []
    chunks = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=364), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def _dedupe_investor_type_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for record in records:
        if not isinstance(record, dict):
            continue
        key = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _investor_types_fetch_ranges(target_date: date) -> list[tuple[date, date]]:
    end_date = _business_day_on_or_before(target_date - timedelta(days=14))
    return [
        (end_date - timedelta(weeks=26), end_date),
        (end_date - timedelta(weeks=52), end_date),
        (end_date - timedelta(weeks=104), end_date),
    ]


def _read_investor_types_cache_payload(start_date: date, end_date: date) -> dict[str, Any] | None:
    if FORCE_REFRESH_ACTIVE:
        return None
    cache_path = ROOT / "data" / "cache" / "jquants" / "investor_types" / f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"
    if not cache_path.exists():
        return None
    try:
        records = read_json(cache_path).get("records", [])
    except Exception:
        return None
    if not isinstance(records, list) or not records:
        return None
    return {
        "records": records,
        "cache_path": str(cache_path),
        "from_cache": True,
        "fallback_used": False,
        "warning": "",
        "available": True,
        "saved": False,
        "usable": True,
        "reason": "",
    }


def _fetch_investor_types_with_retries(
    provider: JQuantsDataProvider,
    start_date: date,
    end_date: date,
    config: dict[str, Any],
    force_refresh: bool = False,
) -> dict[str, Any]:
    max_attempts = int(config.get("jquants", {}).get("investor_types_retry_attempts", 4) or 4)
    retry_waits = _jquants_retry_waits(config)
    last_payload: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        payload = provider.fetch_investor_types_cached(
            ROOT / "data" / "cache",
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh or attempt > 1,
        )
        payload["attempt"] = attempt
        last_payload = payload
        status = _provider_payload_status(payload)
        retryable = _retryable_jquants_status(status)
        if payload.get("records") or payload.get("from_cache") or not retryable or attempt >= max_attempts:
            if attempt > 1 and payload.get("records"):
                _record_api_retry_success("investor_types")
                payload["retry_success"] = True
            return payload
        _record_api_retry("investor_types")
        retry_after_header = _retry_after_seconds(payload)
        retry_after = retry_after_header if retry_after_header is not None else _retry_wait_for_attempt(retry_waits, attempt)
        payload["retry_after"] = str(retry_after)
        _log_jquants_api_event(
            endpoint="investor_types",
            plan=_jquants_plan(config),
            cache_hit=bool(payload.get("from_cache")),
            status=status,
            records=len(payload.get("records", [])),
            saved=bool(payload.get("saved")),
            cache_path=payload.get("cache_path"),
            reason=str(payload.get("reason") or ""),
            attempt=attempt,
            error_reason=str(payload.get("reason") or status),
            **_payload_http_log_fields(payload, getattr(provider, "last_request_metadata", {}) or {}),
        )
        if retry_after > 0:
            time.sleep(retry_after)
        allowed, stop_reason = _api_call_allowed("investor_types")
        if not allowed:
            last_payload = _api_unavailable_payload("investor_types", stop_reason)
            break
    return last_payload


def _business_day_on_or_before(value: date) -> date:
    current = value
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _load_financial_statements_for_period(
    start_date: date,
    end_date: date,
    config: dict[str, Any],
    force_refresh: bool | None = None,
) -> dict[str, Any]:
    allowed, stop_reason = _api_call_allowed("financial_statements")
    if not allowed:
        return _api_unavailable_payload("financial_statements", stop_reason)
    try:
        provider = JQuantsDataProvider(
            ROOT / ".env",
            timeout_seconds=int(config.get("jquants", {}).get("request_timeout_seconds", 20)),
            plan=_jquants_plan(config),
            requests_per_minute=_jquants_requests_per_minute(config),
            parallel_fetch=_jquants_parallel_fetch(config),
            max_parallel_requests=_jquants_max_parallel_requests(config),
        )
        payload = provider.fetch_financial_statements_cached(
            ROOT / "data" / "cache",
            start_date=start_date,
            end_date=end_date,
            force_refresh=FORCE_REFRESH_ACTIVE if force_refresh is None else force_refresh,
        )
        _log_jquants_api_event(
            endpoint="financial_statements",
            plan=_jquants_plan(config),
            cache_hit=bool(payload.get("from_cache")),
            status=_provider_payload_status(payload),
            records=len(payload.get("records", [])),
            saved=bool(payload.get("saved")),
            cache_path=payload.get("cache_path"),
            reason=str(payload.get("reason") or ""),
        )
        return payload
    except Exception as exc:
        _record_api_error("financial_statements", _api_error_status(exc))
        _log_jquants_api_event(
            endpoint="financial_statements",
            plan=_jquants_plan(config),
            cache_hit=False,
            status=_api_error_status(exc),
            records=0,
            saved=False,
            cache_path="",
            error=str(exc),
            **_api_error_log_fields(exc),
        )
        return {
            "records": [],
            "cache_path": "",
            "from_cache": False,
            "fallback_used": False,
            "warning": str(exc),
            "available": False,
            "saved": False,
        }


def _relative_strength_benchmark_payload(
    price_rows: list[dict[str, Any]],
    target_date: date,
    fetch_dates: list[date],
    config: dict[str, Any],
) -> dict[str, Any]:
    topix_records: list[dict[str, Any]] = []
    scoring_config = config.get("scoring", {}).get("relative_strength", {})
    wants_topix = str(scoring_config.get("benchmark", "topix")) == "topix"
    source_payload: dict[str, Any] = {}
    if wants_topix and _jquants_plan(config) == "light" and jquants_has_capability(_jquants_plan(config), "topix_prices"):
        preloaded = _preloaded_topix_payload_for(target_date)
        if preloaded:
            topix_records = preloaded.get("records", [])
            source_payload = preloaded
        else:
            start_date = fetch_dates[0] if fetch_dates else target_date
            payload = _load_topix_prices_for_period(start_date, target_date, config)
            topix_records = payload.get("records", [])
            source_payload = payload
        if not topix_records:
            print(f"relative_strength warning: TOPIX benchmark unavailable; fallback benchmark will be used. reason={source_payload.get('warning')}")
    elif wants_topix:
        print("relative_strength warning: TOPIX benchmark disabled for current J-Quants plan; fallback benchmark will be used.")
    benchmark = build_relative_strength_benchmark(price_rows, target_date.isoformat(), topix_records)
    benchmark["topix_records_loaded"] = len(topix_records)
    benchmark["topix_api_calls"] = int(_jquants_api_session().setdefault("api_calls_by_endpoint", {}).get("topix_prices", 0) or 0)
    benchmark["topix_cache_path"] = source_payload.get("cache_path", "")
    if benchmark["benchmark_source"] == "unavailable":
        print("relative_strength warning: benchmark unavailable; relative_strength_score will be 0.")
    return benchmark


def _preloaded_topix_payload_for(target_date: date) -> dict[str, Any] | None:
    preloaded = _jquants_api_session().setdefault("payloads", {}).get("topix_prices")
    if not isinstance(preloaded, dict):
        return None
    end_text = str(preloaded.get("end_date") or "")
    if end_text and target_date.isoformat() > end_text:
        return None
    metadata = preloaded.get("metadata", {}) if isinstance(preloaded.get("metadata"), dict) else {}
    records = preloaded.get("records", [])
    return {
        "records": records,
        "available": metadata.get("available", bool(records)),
        "from_cache": metadata.get("from_cache"),
        "fallback_used": metadata.get("fallback_used"),
        "warning": metadata.get("warning", ""),
        "reason": metadata.get("reason", ""),
        "topix_records_loaded": metadata.get("topix_records_loaded", len(records)),
    }


def _expected_relative_strength_benchmark_source(config: dict[str, Any]) -> str:
    scoring_config = config.get("scoring", {}).get("relative_strength", {})
    wants_topix = str(scoring_config.get("benchmark", "topix")) == "topix"
    if wants_topix and _jquants_plan(config) == "light" and jquants_has_capability(_jquants_plan(config), "topix_prices"):
        return "topix"
    return "prime_average"


def _load_topix_prices_for_period(start_date: date, end_date: date, config: dict[str, Any]) -> dict[str, Any]:
    return _load_topix_prices_for_period_with_options(start_date, end_date, config, use_preloaded=True)


def _load_topix_prices_for_period_with_options(
    start_date: date,
    end_date: date,
    config: dict[str, Any],
    use_preloaded: bool = True,
) -> dict[str, Any]:
    if use_preloaded:
        preloaded = _preloaded_topix_payload_for(end_date)
        if preloaded:
            return preloaded
    cached = _read_topix_cache_payload(start_date, end_date)
    if cached is not None:
        _log_jquants_api_event(
            endpoint="topix_prices",
            plan=_jquants_plan(config),
            cache_hit=True,
            status="200",
            records=len(cached.get("records", [])),
            saved=False,
            cache_path=cached.get("cache_path"),
            reason="",
        )
        return cached
    allowed, stop_reason = _api_call_allowed("topix_prices")
    if not allowed:
        return _api_unavailable_payload("topix_prices", stop_reason)
    try:
        provider = JQuantsDataProvider(
            ROOT / ".env",
            timeout_seconds=int(config.get("jquants", {}).get("request_timeout_seconds", 20)),
            plan=_jquants_plan(config),
            requests_per_minute=_jquants_requests_per_minute(config),
            parallel_fetch=_jquants_parallel_fetch(config),
            max_parallel_requests=_jquants_max_parallel_requests(config),
        )
        payload = _fetch_topix_prices_with_retries(provider, start_date, end_date, config)
        _log_jquants_api_event(
            endpoint="topix_prices",
            plan=_jquants_plan(config),
            cache_hit=bool(payload.get("from_cache")),
            status=_provider_payload_status(payload),
            records=len(payload.get("records", [])),
            saved=bool(payload.get("saved")),
            cache_path=payload.get("cache_path"),
            reason=str(payload.get("reason") or ""),
            attempt=payload.get("attempt") or "",
            error_reason=str(payload.get("reason") or ""),
            **_payload_http_log_fields(payload, getattr(provider, "last_request_metadata", {}) or {}),
        )
        status = _provider_payload_status(payload)
        if status == "auth_or_plan_error":
            _record_api_error("topix_prices", "auth_or_plan_error")
        elif not payload.get("records") and _retryable_jquants_status(status):
            _record_api_error("topix_prices", status)
        return payload
    except Exception as exc:
        reason = _api_error_status(exc)
        if reason == "auth_or_plan_error":
            _record_api_error("topix_prices", "auth_or_plan_error")
        else:
            _record_api_error("topix_prices", reason)
        _log_jquants_api_event(
            endpoint="topix_prices",
            plan=_jquants_plan(config),
            cache_hit=False,
            status=_api_error_status(exc),
            records=0,
            saved=False,
            cache_path="",
            error=str(exc),
            **_api_error_log_fields(exc),
        )
        return {
            "records": [],
            "cache_path": "",
            "from_cache": False,
            "fallback_used": False,
            "warning": str(exc),
            "available": False,
        }


def _read_topix_cache_payload(start_date: date, end_date: date) -> dict[str, Any] | None:
    if FORCE_REFRESH_ACTIVE:
        return None
    cache_path = ROOT / "data" / "cache" / "jquants" / "topix_prices" / f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"
    if not cache_path.exists():
        return None
    try:
        records = read_json(cache_path).get("records", [])
    except Exception:
        return None
    if not isinstance(records, list) or not records:
        return None
    return {
        "records": records,
        "cache_path": str(cache_path),
        "from_cache": True,
        "fallback_used": False,
        "warning": "",
        "available": True,
        "saved": False,
        "usable": True,
        "reason": "",
    }


def _fetch_topix_prices_with_retries(
    provider: JQuantsDataProvider,
    start_date: date,
    end_date: date,
    config: dict[str, Any],
) -> dict[str, Any]:
    max_attempts = int(config.get("jquants", {}).get("topix_retry_attempts", 4) or 4)
    retry_waits = _jquants_retry_waits(config)
    last_payload: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        payload = provider.fetch_topix_prices_cached(
            ROOT / "data" / "cache",
            start_date=start_date,
            end_date=end_date,
            force_refresh=FORCE_REFRESH_ACTIVE or attempt > 1,
        )
        payload["attempt"] = attempt
        last_payload = payload
        status = _provider_payload_status(payload)
        retryable = _retryable_jquants_status(status)
        if payload.get("records") or payload.get("from_cache") or not retryable or attempt >= max_attempts:
            if attempt > 1 and payload.get("records"):
                _record_api_retry_success("topix_prices")
                payload["retry_success"] = True
            return payload
        _record_api_retry("topix_prices")
        retry_after_header = _retry_after_seconds(payload)
        retry_after = retry_after_header if retry_after_header is not None else _retry_wait_for_attempt(retry_waits, attempt)
        payload["retry_after"] = str(retry_after)
        _log_jquants_api_event(
            endpoint="topix_prices",
            plan=_jquants_plan(config),
            cache_hit=bool(payload.get("from_cache")),
            status=status,
            records=len(payload.get("records", [])),
            saved=bool(payload.get("saved")),
            cache_path=payload.get("cache_path"),
            reason=str(payload.get("reason") or ""),
            attempt=attempt,
            error_reason=str(payload.get("reason") or status),
            **_payload_http_log_fields(payload, getattr(provider, "last_request_metadata", {}) or {}),
        )
        if retry_after > 0:
            time.sleep(retry_after)
        allowed, stop_reason = _api_call_allowed("topix_prices")
        if not allowed:
            last_payload = _api_unavailable_payload("topix_prices", stop_reason)
            break
    return last_payload


def _jquants_retry_waits(config: dict[str, Any]) -> list[int]:
    raw = config.get("jquants", {}).get("retry_backoff_seconds", [30, 60, 120])
    if not isinstance(raw, list):
        return [30, 60, 120]
    waits: list[int] = []
    for value in raw:
        try:
            waits.append(max(0, int(value)))
        except (TypeError, ValueError):
            continue
    return waits or [30, 60, 120]


def _retry_wait_for_attempt(waits: list[int], attempt: int) -> int:
    index = max(0, min(len(waits) - 1, attempt - 1))
    return waits[index]


def _retry_after_seconds(payload: dict[str, Any]) -> int | None:
    value = payload.get("retry_after")
    try:
        if value not in (None, ""):
            return max(0, int(float(value)))
    except (TypeError, ValueError):
        return None
    return None


def _retryable_jquants_status(status: str) -> bool:
    return status in {"rate_limit", "network_error", "timeout", "api_error", "api_call_limit_reached"}


def _provider_payload_status(payload: dict[str, Any]) -> str:
    if payload.get("api_status"):
        return str(payload.get("api_status"))
    if payload.get("available"):
        return "200"
    if payload.get("filter_available"):
        return "200"
    warning = str(payload.get("warning") or "").strip()
    return warning or "unavailable"


def _api_error_status(exc: Exception) -> str:
    if isinstance(exc, JQuantsApiError):
        return exc.category
    return "error"


def _record_jquants_empty_marker(endpoint: str, start_date: date, end_date: date, reason: str) -> None:
    marker_path = ROOT / "data" / "cache" / "jquants" / "empty_ranges.json"
    payload: dict[str, Any] = {}
    if marker_path.exists():
        try:
            payload = read_json(marker_path)
        except Exception:
            payload = {}
    ranges = payload.setdefault(endpoint, [])
    ranges.append({"start": start_date.isoformat(), "end": end_date.isoformat(), "reason": reason})
    write_json(marker_path, payload)


def _api_error_log_fields(exc: Exception) -> dict[str, str]:
    if not isinstance(exc, JQuantsApiError):
        return {}
    return {
        "http_status": str(exc.status_code or ""),
        "request_url": exc.request_url,
        "request_params": json.dumps(exc.request_params, ensure_ascii=False, sort_keys=True),
        "response_body": exc.response_body,
        "retry_after": str(getattr(exc, "retry_after", "") or ""),
    }


def _api_error_log_fields_raw(exc: Exception) -> dict[str, Any]:
    if not isinstance(exc, JQuantsApiError):
        return {}
    return {
        "http_status": exc.status_code,
        "request_url": exc.request_url,
        "request_params": exc.request_params,
        "response_body": exc.response_body,
        "retry_after": getattr(exc, "retry_after", "") or "",
    }


def _payload_http_log_fields(payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, str]:
    metadata = metadata or {}
    request_params = payload.get("request_params") or metadata.get("params") or {}
    if isinstance(request_params, str):
        request_params_text = request_params
    else:
        request_params_text = json.dumps(request_params, ensure_ascii=False, sort_keys=True)
    status_code = payload.get("http_status") or metadata.get("status_code") or ""
    return {
        "http_status": str(status_code or ""),
        "request_url": str(payload.get("request_url") or metadata.get("url") or ""),
        "request_params": request_params_text,
        "response_body": str(payload.get("response_body") or metadata.get("response_body") or ""),
        "retry_after": str(payload.get("retry_after") or metadata.get("retry_after") or ""),
    }


def _log_jquants_api_event(
    endpoint: str,
    plan: str,
    cache_hit: bool,
    status: str,
    records: int,
    saved: bool = False,
    cache_path: Any = "",
    error: str = "",
    reason: str = "",
    retry_range: str = "",
    http_status: str = "",
    request_url: str = "",
    request_params: str = "",
    response_body: str = "",
    attempt: Any = "",
    retry_after: str = "",
    error_reason: str = "",
    result: str = "",
    fetch_from: str = "",
    fetch_to: str = "",
) -> None:
    path = ROOT / "logs" / "jquants_api.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"timestamp={datetime.now().isoformat(timespec='seconds')} "
        f"endpoint={endpoint} plan={plan} cache_hit={str(cache_hit).lower()} "
        f"status={status} records={records} saved={str(saved).lower()} "
        f"cache_path={_relative_path_text(cache_path) if cache_path else 'N/A'}"
    )
    if reason:
        line = f"{line} reason={reason}"
    if retry_range:
        line = f"{line} retry_range={retry_range}"
    if attempt:
        line = f"{line} attempt={attempt}"
    if http_status:
        line = f"{line} http_status={http_status}"
        line = f"{line} status_code={http_status}"
    if retry_after:
        line = f"{line} retry_after={retry_after}"
    if error_reason:
        line = f"{line} error_reason={error_reason}"
    if result:
        line = f"{line} result={result}"
    if fetch_from:
        line = f"{line} from={fetch_from}"
    if fetch_to:
        line = f"{line} to={fetch_to}"
    if request_url:
        line = f"{line} request_url={request_url}"
    if request_params:
        line = f"{line} request_params={_log_safe_value(request_params)}"
    if response_body:
        line = f"{line} response_body={_log_safe_value(response_body)}"
    if error:
        line = f"{line} error={_log_safe_value(error)}"
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{line}\n")


def _log_safe_value(value: Any) -> str:
    return str(value).replace("\n", " ").replace(" ", "_")[:500]


def profile_id_from(config: dict[str, Any] | None = None) -> str:
    config = config or load_config(CONFIG_PATH)
    return str(config.get("profile_id") or DEFAULT_PROFILE_ID)


def profile_name_from(config: dict[str, Any] | None = None) -> str:
    config = config or load_config(CONFIG_PATH)
    return str(config.get("profile_name") or config.get("dealer", {}).get("name") or profile_id_from(config))


def add_profile_metadata(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload["profile_id"] = profile_id_from(config)
    payload["profile_name"] = profile_name_from(config)
    return payload


def profile_path(config: dict[str, Any], *parts: str) -> Path:
    profile_id = profile_id_from(config)
    if len(parts) >= 2 and parts[0] == "articles" and parts[1] == "drafts":
        return ROOT / "articles" / "drafts" / profile_id / Path(*parts[2:])
    return ROOT / parts[0] / profile_id / Path(*parts[1:])


def processed_profile_path(config: dict[str, Any], filename: str) -> Path:
    return ROOT / "data" / "processed" / profile_id_from(config) / filename


def build_daily_trade_log(paper_trade_log: dict[str, Any], reflection_log: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": paper_trade_log["run_id"],
        "date": paper_trade_log["date"],
        "config_version": paper_trade_log.get("config_version") or reflection_log.get("config_version"),
        "day": paper_trade_log["day_number"],
        "day_number": paper_trade_log["day_number"],
        "dealer_id": paper_trade_log["dealer_id"],
        "orders": paper_trade_log["orders"],
        "closed_trades": paper_trade_log["closed_trades"],
        "safety_events": paper_trade_log.get("safety_events", []),
        "reflections": reflection_log["reflections"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "day",
        "date",
        "cash",
        "positions_value",
        "total_assets",
        "daily_profit",
        "cumulative_profit",
        "cumulative_profit_rate",
        "gross_cumulative_profit",
        "net_cumulative_profit",
        "total_commission",
        "estimated_tax_total",
        "net_total_assets",
        "win_rate",
        "max_drawdown",
        "open_positions_count",
        "closed_trades_count",
    ]
    rows = []
    for summary in summaries:
        rows.append(
            {
                "day": summary["day"],
                "date": summary["date"],
                "cash": summary["cash"],
                "positions_value": summary["positions_value"],
                "total_assets": summary["total_assets"],
                "daily_profit": summary["daily_profit"],
                "cumulative_profit": summary["cumulative_profit"],
                "cumulative_profit_rate": summary.get("cumulative_profit_rate", summary.get("cumulative_return_pct", 0)),
                "gross_cumulative_profit": summary.get("gross_cumulative_profit", summary.get("cumulative_profit", 0)),
                "net_cumulative_profit": summary.get("net_cumulative_profit", summary.get("cumulative_profit", 0)),
                "total_commission": summary.get("total_commission", 0),
                "estimated_tax_total": summary.get("estimated_tax_total", 0),
                "net_total_assets": summary.get("net_total_assets", summary.get("total_assets", 0)),
                "win_rate": "" if summary["win_rate"] is None else summary["win_rate"],
                "max_drawdown": summary["max_drawdown"],
                "open_positions_count": summary["open_positions_count"],
                "closed_trades_count": summary.get("closed_trades_count", summary.get("closed_trade_count", 0)),
            }
        )
    write_csv(path, fieldnames, rows)


def write_trades_csv(path: Path, trades: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "trade_id",
        "profile_id",
        "profile_name",
        "action",
        "code",
        "name",
        "sector_name",
        "signal_date",
        "entry_date",
        "exit_date",
        "holding_days",
        "entry_price",
        "entry_price_source",
        "signal_close_price",
        "entry_open_price",
        "entry_gap_rate",
        "exit_price",
        "intended_exit_price",
        "actual_exit_price",
        "shares",
        "profit",
        "profit_rate",
        "stop_loss_rate",
        "stop_loss_trigger_price",
        "stop_loss_triggered_date",
        "gap_slippage_rate",
        "stop_loss_slippage_rate",
        "gross_profit",
        "gross_profit_rate",
        "buy_commission",
        "sell_commission",
        "total_commission",
        "taxable_profit",
        "estimated_tax",
        "net_profit",
        "net_profit_rate",
        "exit_reason",
        "result",
        "score",
        "rsi",
        "volume_ratio",
        "total_score",
        "technical_score",
        "ma_score",
        "rsi_score",
        "volume_score",
        "candlestick_score",
        "market_context_score",
        "investor_context_score",
        "sector_score",
        "penalty_score",
        "score_components",
        "score_components_total",
        "score_components_match",
        "market_regime",
        "advance_ratio",
        "investor_context_source",
        "investor_context_week",
        "overseas_net_buy",
        "overseas_net_buy_4w_sum",
        "overseas_net_buy_4w_trend",
        "overseas_buy_sell_ratio",
        "individual_net_buy",
        "institution_net_buy",
        "trust_bank_net_buy",
        "proprietary_net_buy",
        "candlestick_signals",
        "earnings_filter_checked",
        "earnings_filter_blocked",
        "earnings_filter_reason",
        "earnings_announcement_date",
        "selected_reason",
        "reason",
        "broker_provider",
        "order_status",
        "config_version",
        "created_at",
    ]
    rows = [{field: trade.get(field, "") for field in fieldnames} for trade in trades]
    write_csv(path, fieldnames, rows)


def count_csv_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return sum(1 for _row in reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pick_paper_trade_fields(paper_trade_log: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: paper_trade_log[key] for key in keys}


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small two-level YAML config used by this project."""
    config: dict[str, Any] = {}
    current_section = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and ":" in line and not line.endswith(":"):
            key, value = line.split(":", 1)
            config[key.strip()] = _parse_scalar(value.strip())
            current_section = ""
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1]
            config[current_section] = {}
            continue
        if current_section and line.startswith("  "):
            key, value = line.strip().split(":", 1)
            config[current_section][key] = _parse_scalar(value.strip())
    return config


def _parse_scalar(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
