"""Entry point for AI Fund Lab first implementation."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
import time
from argparse import ArgumentParser
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only on minimal local environments.
    yaml = None

from article import generate_note_article
from ai_decision import apply_ai_decision, build_ai_decision_log, build_ai_decision_provider
from ai_analysis import export_ai_dataset, export_ai_summary, record_ai_analysis_export
from charts import generate_charts_from_summary
from commentary import (
    generate_buy_comment,
    generate_daily_comment,
    generate_no_trade_comment,
    generate_reflection_comment,
    generate_sell_comment,
)
from config_version import attach_config_version, config_version_from, load_config as load_versioned_config
from data_provider import DummyDataProvider, JQuantsDataProvider
from db import (
    analyze_operation_data,
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
from feature_analysis import build_feature_analysis, render_feature_analysis_markdown
from indicators import calculate_indicators
from market_context import build_market_context, neutral_market_context
from news_provider import build_news_provider
from paper_trade import execute_paper_trade_day, execute_real_data_paper_trade, initial_live_paper_state, initial_paper_state
from portfolio import build_daily_summary
from profile_loader import DEFAULT_PROFILE_ID, list_profiles, load_profile
from reflection import generate_reflections
from report import generate_daily_report
from real_screening import screen_candidates
from release_notes import generate_release_notes, render_release_notes_markdown
from safety import can_trade
from scoring import build_trade_decisions, score_candidates, score_real_candidates
from screening import generate_screening_log
from tachibana_auth import load_private_key, load_tachibana_auth_config
from technical_indicators import TechnicalIndicatorDependencyError
from tax import calculate_period_profit_summary
from trade_metrics import profit_factor_metrics


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "rookie_dealer.yaml"
ACTIVE_PROFILE_ID = DEFAULT_PROFILE_ID
BACKTEST_MODE_ACTIVE = False
BACKTEST_DAY_LOG_PREFIX = ""


def main() -> None:
    global ACTIVE_PROFILE_ID
    _enable_line_buffered_output()
    args = parse_args()
    ACTIVE_PROFILE_ID = args.profile
    if args.mode == "help":
        run_help()
        return
    if args.mode == "preflight":
        run_preflight(args.profile)
        return
    if args.mode == "compare-profiles":
        run_compare_profiles(args.profiles, args.start_date, args.end_date)
        return
    config = load_config(CONFIG_PATH)
    if args.mode == "status":
        run_status(config, args.output_format)
        return
    if args.mode == "healthcheck":
        run_healthcheck(args.provider)
        return
    if args.mode == "tachibana-healthcheck":
        run_tachibana_healthcheck(args.tachibana_env)
        return
    if args.mode == "init-db":
        run_init_db(config)
        return
    if args.mode == "analyze":
        run_analyze(config)
        return
    if args.mode == "release-notes":
        run_release_notes(args.since, args.until)
        return
    if args.mode == "full-paper-run":
        run_full_paper_run(args.provider, args.start_date, args.end_date)
        return
    if args.mode == "list-stocks":
        run_list_stocks(args.provider)
        return
    if args.mode == "fetch-prices":
        run_fetch_prices(args.provider, args.date)
        return
    if args.mode == "calculate-indicators":
        run_calculate_indicators(args.provider, args.date)
        return
    if args.mode == "screen":
        run_screen(args.provider, args.date)
        return
    if args.mode == "score":
        run_score(args.provider, args.date)
        return
    if args.mode == "trade":
        run_trade(args.provider, args.date)
        return
    if args.mode == "preview-orders":
        run_preview_orders(args.provider, args.date)
        return
    if args.mode == "publish-article":
        run_publish_article(args.date, args.note_url)
        return
    if args.mode == "run-daily":
        run_daily(args.provider, args.date)
        return
    if args.mode == "backtest":
        run_backtest(args.provider, args.start_date, args.end_date)
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
            "healthcheck",
            "tachibana-healthcheck",
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
        ],
        default="demo",
        help="Execution mode. Use demo, healthcheck, list-stocks, fetch-prices, or calculate-indicators.",
    )
    parser.add_argument(
        "--provider",
        choices=["dummy", "jquants"],
        default="dummy",
        help="Provider for healthcheck mode.",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_ID,
        help="AI fund profile id under config/profiles. Default: rookie_dealer_01.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Profile ids for compare-profiles mode.",
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
        "--start-date",
        help="Backtest start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        help="Backtest end date in YYYY-MM-DD format.",
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
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be 1 or greater.")
    if args.mode in {"fetch-prices", "calculate-indicators", "screen", "score", "trade", "preview-orders", "publish-article", "run-daily"} and not args.date:
        parser.error(f"--date YYYY-MM-DD is required for {args.mode} mode.")
    if args.mode == "publish-article" and not args.note_url:
        parser.error("--note-url URL is required for publish-article mode.")
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


def run_healthcheck(provider_name: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("healthcheck mode currently supports --provider jquants only.")

    try:
        provider = JQuantsDataProvider(ROOT / ".env")
        listed_stocks = provider.get_listed_stocks()
    except RuntimeError as exc:
        print(f"J-Quants connection failed: {exc}")
        raise SystemExit(1) from exc

    payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "provider": "jquants",
        "endpoint": "/equities/master",
        "listed_stocks_count": len(listed_stocks),
        "sample": listed_stocks[:5],
    }
    write_json(ROOT / "data" / "raw" / "jquants_healthcheck.json", payload)

    print("J-Quants connection successful")
    print("provider: jquants")
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


def run_analyze(config: dict[str, Any]) -> None:
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
    feature_analysis = build_feature_analysis(config, ROOT)
    write_json(json_path, analysis)
    write_text(markdown_path, render_analysis_markdown(analysis))
    write_json(feature_json_path, feature_analysis)
    write_text(feature_markdown_path, render_feature_analysis_markdown(feature_analysis))

    portfolio = analysis["portfolio_analysis"]
    trades = analysis["trade_analysis"]
    trades_csv, db_trade_count, csv_trade_count = write_trades_csv_from_db(config)
    print("analysis completed")
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


def run_compare_profiles(profile_ids: list[str], start_date_text: str, end_date_text: str) -> None:
    profiles = [load_profile(profile_id) for profile_id in profile_ids]
    db_path = get_database_path(profiles[0], ROOT)
    if not db_path.exists():
        raise SystemExit(f"SQLite DB not found: {db_path}")

    rows = [_profile_compare_row(config, db_path, start_date_text, end_date_text) for config in profiles]
    ranking = build_profile_ranking(rows)
    payload = {
        "start_date": start_date_text,
        "end_date": end_date_text,
        "profiles": rows,
        "ranking": ranking,
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
    print(f"markdown: {markdown_path.relative_to(ROOT)}")
    print(f"json: {json_path.relative_to(ROOT)}")


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
                  AND order_status = 'FILLED'
                  AND action IN ('BUY', 'SELL')
                  AND COALESCE(exit_date, entry_date) BETWEEN ? AND ?
                ORDER BY entry_date, exit_date, id
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
    win_rate = round(len(wins) / len(closed), 4) if closed else None
    expectancy = (
        round((win_rate * (average_win_profit_rate or 0.0)) + ((1 - win_rate) * (average_loss_profit_rate or 0.0)), 4)
        if win_rate is not None
        else None
    )
    max_drawdown = min((float(row.get("max_drawdown") or 0) for row in portfolio_rows), default=None) if portfolio_rows else None
    return {
        "profile_id": profile_id,
        "profile_name": profile_name_from(config),
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
        "total_trades": len(trade_rows),
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
    }


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
    lines = [
        f"# Profile比較 {payload['start_date']} to {payload['end_date']}",
        "",
        "| profile | stop_loss_execution | final_assets | net_cumulative_profit | win_rate | profit_factor | expectancy | max_drawdown | avg_holding_days | closed | wins | losses | excluded | avg_win | avg_loss | total_trades | loss_over_stop_count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"{row.get('loss_over_stop_count')} |"
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
    lines.append("")
    return "\n".join(lines)


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
    article_dir = ROOT / "articles" / "drafts" / "backtests" / range_key

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

    generated_articles_count = len(list(article_dir.glob("*.md"))) if article_dir.exists() else 0
    trade_analysis = analysis.get("trade_analysis", {})
    portfolio_analysis = analysis.get("portfolio_analysis", {})
    gross_profit = sum(float(trade.get("gross_profit", trade.get("profit", 0)) or 0) for trade in closed_trades)
    net_profit = sum(float(trade.get("net_profit", trade.get("profit", 0)) or 0) for trade in closed_trades)

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
- tachibana_demo / tachibana_live は未実装スタブで、実注文は出しません。
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
    drafts = _article_files(ROOT / "articles" / "drafts" / profile_id_from(config))
    published = _article_files(ROOT / "articles" / "published" / profile_id_from(config))
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
        "articles": {
            "drafts_count": len(drafts),
            "published_count": len(published),
            "latest_draft": _relative_or_none(drafts[-1]) if drafts else None,
            "latest_published": _relative_or_none(published[-1]) if published else None,
        },
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


def _article_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted((path for path in directory.glob("**/*.md") if path.is_file()), key=lambda path: path.stat().st_mtime)


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


def run_preflight(profile_id: str | None = None) -> None:
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
        _check_jquants_health(results)
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


def _check_jquants_health(results: list[dict[str, Any]]) -> None:
    try:
        provider = JQuantsDataProvider(ROOT / ".env")
        listed = provider.get_listed_stocks()
    except Exception as exc:
        _preflight_add(results, "WARN", f"J-Quants healthcheck skipped or failed: {exc}")
        return
    _preflight_add(results, "OK", "J-Quants healthcheck successful", {"listed_stocks": len(listed)})


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

    try:
        provider = JQuantsDataProvider(ROOT / ".env")
        listed_stocks = provider.get_listed_stocks()
        prime_stocks = [_normalize_prime_stock(record) for record in listed_stocks if _is_prime_stock(record)]
    except RuntimeError as exc:
        print(f"J-Quants listed stocks fetch failed: {exc}")
        raise SystemExit(1) from exc

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
    print(f"saved: {output_path.relative_to(ROOT)}")


def run_fetch_prices(provider_name: str, target_date_text: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("fetch-prices mode currently supports --provider jquants only.")

    try:
        target_date = date.fromisoformat(target_date_text)
    except ValueError as exc:
        raise SystemExit("--date must be in YYYY-MM-DD format.") from exc

    prime_path = ROOT / "data" / "raw" / "prime_stocks_jquants.json"
    if not prime_path.exists():
        raise SystemExit("Prime stock list not found. Run `python src/main.py --mode list-stocks --provider jquants` first.")

    prime_payload = read_json(prime_path)
    prime_codes = {stock["code"] for stock in prime_payload.get("stocks", [])}
    if not prime_codes:
        raise SystemExit("Prime stock list is empty. Re-run list-stocks and check the result.")

    try:
        provider = JQuantsDataProvider(ROOT / ".env")
        daily_prices = provider.get_daily_prices(target_date)
    except RuntimeError as exc:
        print(f"J-Quants daily prices fetch failed: {exc}")
        raise SystemExit(1) from exc

    if not daily_prices:
        raise SystemExit(f"No daily price data found for {target_date_text}. The date may be weekend, holiday, or not updated yet.")

    prime_prices = [_normalize_daily_price(record) for record in daily_prices if _get_first(record, ["code", "Code", "LocalCode"]) in prime_codes]
    if not prime_prices:
        raise SystemExit(f"No Tokyo Prime daily price data found for {target_date_text}. The date may be weekend, holiday, or not updated yet.")

    output_path = ROOT / "data" / "raw" / f"prices_{target_date_text}.json"
    write_json(
        output_path,
        {
            "provider": "jquants",
            "date": target_date_text,
            "total_count": len(daily_prices),
            "prime_count": len(prime_prices),
            "prices": prime_prices,
        },
    )

    print(f"date: {target_date_text}")
    print(f"all prices: {len(daily_prices)}")
    print(f"prime prices: {len(prime_prices)}")
    print(f"saved: {output_path.relative_to(ROOT)}")


def run_calculate_indicators(provider_name: str, target_date_text: str) -> None:
    if provider_name != "jquants":
        raise SystemExit("calculate-indicators mode currently supports --provider jquants only.")

    try:
        target_date = date.fromisoformat(target_date_text)
    except ValueError as exc:
        raise SystemExit("--date must be in YYYY-MM-DD format.") from exc

    prime_path = ROOT / "data" / "raw" / "prime_stocks_jquants.json"
    if not prime_path.exists():
        raise SystemExit("Prime stock list not found. Run `python src/main.py --mode list-stocks --provider jquants` first.")

    prime_payload = read_json(prime_path)
    prime_stocks = prime_payload.get("stocks", [])
    prime_codes = {stock["code"] for stock in prime_stocks}
    stock_names = {stock["code"]: stock["name"] for stock in prime_stocks}
    stock_sectors = {stock["code"]: stock.get("sector_name", "") for stock in prime_stocks}
    if not prime_codes:
        raise SystemExit("Prime stock list is empty. Re-run list-stocks and check the result.")

    config = load_config(CONFIG_PATH)
    indicator_mode = _backtest_indicator_mode(config) if BACKTEST_MODE_ACTIVE else "full"
    output_path = ROOT / "data" / "processed" / f"indicators_{target_date_text}.json"
    profile_output_path = processed_profile_path(config, f"indicators_{target_date_text}.json")
    if BACKTEST_MODE_ACTIVE and profile_output_path.exists():
        cached_payload = read_json(profile_output_path)
        if cached_payload.get("indicator_mode") == indicator_mode:
            write_json(output_path, cached_payload)
            print(f"{BACKTEST_DAY_LOG_PREFIX} indicators cache hit: {profile_output_path.relative_to(ROOT)}")
            return

    lookback_days = 60 if indicator_mode == "full" else 35
    fetch_dates = previous_business_dates(target_date, lookback_days)
    if BACKTEST_MODE_ACTIVE:
        price_rows = load_cached_price_history(fetch_dates)
        if not price_rows:
            raise SystemExit(f"No cached price history found for {target_date_text}. Run fetch-period-prices first.")
    else:
        try:
            provider = JQuantsDataProvider(
                ROOT / ".env",
                timeout_seconds=int(config.get("jquants", {}).get("request_timeout_seconds", 20)),
            )
            price_rows = fetch_price_history(
                provider,
                target_date,
                prime_codes,
                lookback_business_days=60,
                rate_limit_per_minute=int(config.get("jquants", {}).get("rate_limit_per_minute", 5)),
                fetch_dates=fetch_dates,
                continue_on_error=True,
                verbose=False,
            )
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
    if BACKTEST_MODE_ACTIVE:
        print(f"{BACKTEST_DAY_LOG_PREFIX} indicators mode: {indicator_mode}")
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
            indicator_mode=indicator_mode,
            progress_callback=progress_callback if BACKTEST_MODE_ACTIVE else None,
        )
    except TechnicalIndicatorDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    excluded_count = len(prime_codes) - len(indicators)
    if not indicators:
        raise SystemExit(f"No indicators calculated for {target_date_text}. Price history may be insufficient or target date may have no data.")

    payload = {
        "provider": "jquants",
        "date": target_date_text,
        "indicator_mode": indicator_mode,
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
    if not indicators:
        raise SystemExit(f"No indicators found in {indicators_path.relative_to(ROOT)}.")

    indicators = enrich_indicators_with_sector_momentum(indicators, target_date_text, provider_name)
    result = screen_candidates(indicators, target_count=50)
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
        "excluded_summary": result["excluded_summary"],
    }

    screening_log.setdefault("profile_id", profile_id_from(config))
    screening_log.setdefault("profile_name", profile_name_from(config))
    screening_path = ROOT / "logs" / "screening" / profile_id_from(config) / f"screening_{target_date_text}.json"
    candidates_path = processed_profile_path(config, f"candidates_{target_date_text}.json")
    write_json(screening_path, screening_log)
    write_json(
        candidates_path,
        {
            "date": target_date_text,
            "provider": provider_name,
            "profile_id": profile_id_from(config),
            "profile_name": profile_name_from(config),
            "config_version": config_version_from(config),
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
    )
    save_screening_results(config, ROOT, screening_log)

    print(f"target stocks: {len(indicators)}")
    print(f"condition passed: {result['strict_passed_count']}")
    print(f"fallback used: {result['fallback_used']}")
    print(f"candidates: {len(candidates)}")
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
    if not candidates:
        raise SystemExit(f"No candidates found in {candidates_path.relative_to(ROOT)}.")

    market_context = load_market_context_for_date(target_date_text, provider_name)
    news_by_code = fetch_candidate_news(candidates, target_date_text, config)
    scoring_log = score_real_candidates(
        candidates,
        target_date_text,
        config,
        provider_name,
        news_by_code=news_by_code,
        market_context=market_context,
    )
    attach_config_version(scoring_log, config)
    ai_decision_log = run_ai_decision_if_enabled(scoring_log, config, target_date_text)
    scores = scoring_log["scores"]
    selected = scoring_log["selected"]
    highest_score = max((item["total_score"] for item in scores), default=0)

    scoring_log.setdefault("profile_id", profile_id_from(config))
    scoring_log.setdefault("profile_name", profile_name_from(config))
    scoring_path = ROOT / "logs" / "scoring" / profile_id_from(config) / f"scoring_{target_date_text}.json"
    scored_candidates_path = processed_profile_path(config, f"scored_candidates_{target_date_text}.json")
    write_json(scoring_path, scoring_log)
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
            "selection_config": scoring_log.get("selection_config", {}),
            "market_context": scoring_log.get("market_context", {}),
            "market_filter": scoring_log.get("market_filter", {}),
            "news_config": config.get("news", {}),
            "ai_decision": scoring_log.get("ai_decision", {}),
            "scores": scores,
        },
    )
    save_scoring_results(config, ROOT, scoring_log)
    if ai_decision_log:
        save_ai_decision(config, ROOT, ai_decision_log)
    write_daily_ai_dataset(config, target_date_text)

    print(f"candidates: {len(candidates)}")
    print(f"scored: {len(scores)}")
    print(f"selected: {len(selected)}")
    print(f"highest_score: {highest_score}")
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
            write_json(trades_path, {"date": target_date_text, "provider": provider_name, "profile_id": profile_id_from(config), "profile_name": profile_name_from(config), "config_version": config_version_from(config), "trades": trades})
            write_json(portfolio_path, portfolio_summary)
            save_portfolio_snapshot(config, ROOT, portfolio_summary)
            save_trades(config, ROOT, target_date_text, trades)
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
    write_json(
        trades_path,
        {
            "date": target_date_text,
            "provider": provider_name,
            "config_version": config_version_from(config),
            "trades": trades,
        },
    )
    write_json(portfolio_path, portfolio_summary)
    save_portfolio_snapshot(config, ROOT, portfolio_summary)
    save_trades(config, ROOT, target_date_text, trades)
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
    draft_path = _find_article_draft(target_date_text)
    if draft_path is None:
        raise SystemExit(f"Draft article not found: articles/drafts/day_{target_date_text}.md")

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
    published_path = ROOT / "articles" / "published" / profile_id_from(config) / f"day_{target_date_text}.md"
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
    print(f"published article path: {published_path.relative_to(ROOT)}")
    print(f"note_url: {note_url}")
    print(f"draft removed: {str(not keep_draft).lower()}")


def _find_article_draft(target_date_text: str) -> Path | None:
    profile_id = ACTIVE_PROFILE_ID
    candidates = [
        ROOT / "articles" / "drafts" / profile_id / f"day_{target_date_text}.md",
        ROOT / "articles" / "drafts" / f"day_{target_date_text}.md",
    ]
    candidates.extend(sorted((ROOT / "articles" / "drafts" / profile_id).glob(f"**/*{target_date_text}*.md")))
    compact = target_date_text.replace("-", "")
    candidates.extend(sorted((ROOT / "articles" / "drafts" / profile_id).glob(f"**/*{compact}*.md")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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
            "profit_rate": round(profit_rate, 4),
            "safety_checked": True,
            "live_trading": False,
            "broker_provider": "preview",
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
            "safety_checked": True,
            "live_trading": False,
            "broker_provider": "preview",
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
                }
            )
            today_orders.append(order)
        else:
            skipped.append(_preview_skipped(order, validation))

    live_enabled = bool(config.get("broker", {}).get("live_trading_enabled", False) and config.get("safety", {}).get("allow_live_trading", False))
    broker_provider = config.get("broker", {}).get("provider", "paper")
    return {
        "date": target_date_text,
        "mode": "LIVE_DISABLED" if not live_enabled else "PAPER",
        "provider": "jquants",
        "broker_provider": broker_provider,
        "broker_candidates": ["paper", "tachibana_demo", "tachibana_live", "kabu_station"],
        "sell_candidates": [
            {
                "code": item["code"],
                "name": item["name"],
                "shares": item["shares"],
                "estimated_price": item["exit_price"],
                "estimated_amount": item["amount"],
                "reason": item["reason"],
                "profit_rate": item["profit_rate"],
            }
            for item in sell_candidates
        ],
        "buy_candidates": buy_candidates,
        "skipped": skipped,
        "safety": safety_results,
        "summary": {
            "buy_count": len(buy_candidates),
            "sell_count": len(sell_candidates),
            "estimated_buy_amount": round(sum(item["estimated_amount"] for item in buy_candidates), 2),
            "estimated_sell_amount": round(sum(item["estimated_amount"] for item in sell_candidates), 2),
            "live_trading_enabled": live_enabled,
            "broker_provider": broker_provider,
        },
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
        "# Order Preview",
        "",
        f"Date: {preview['date']}",
        f"Mode: {preview['mode']}",
        f"Broker: {preview.get('broker_provider', 'paper')}",
        "",
        "## Sell Candidates",
        "",
    ]
    lines.extend(_preview_order_lines(preview["sell_candidates"], is_sell=True))
    lines.extend(["", "## Buy Candidates", ""])
    lines.extend(_preview_order_lines(preview["buy_candidates"], is_sell=False))
    lines.extend(["", "## Skipped", ""])
    lines.extend(_preview_skipped_lines(preview["skipped"]))
    lines.extend(["", "## Safety", ""])
    lines.extend(_preview_safety_lines(preview["safety"]))
    summary = preview["summary"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- buy_count: {summary['buy_count']}",
            f"- sell_count: {summary['sell_count']}",
            f"- estimated_buy_amount: {summary['estimated_buy_amount']:,.0f}",
            f"- estimated_sell_amount: {summary['estimated_sell_amount']:,.0f}",
            f"- broker_provider: {summary.get('broker_provider', 'paper')}",
            f"- live_trading_enabled: {str(summary['live_trading_enabled']).lower()}",
        ]
    )
    return "\n".join(lines)


def _preview_order_lines(items: list[dict[str, Any]], is_sell: bool) -> list[str]:
    if not items:
        return ["- None"]
    lines = []
    for item in items:
        if is_sell:
            lines.append(
                f"- {item['code']} {item['name']}: shares={item['shares']}, "
                f"estimated_price={item['estimated_price']:,.0f}, estimated_amount={item['estimated_amount']:,.0f}, "
                f"reason={item['reason']}, profit_rate={item['profit_rate']:.2%}"
            )
        else:
            lines.append(
                f"- {item['code']} {item['name']}: shares={item['shares']}, "
                f"estimated_price={item['estimated_price']:,.0f}, estimated_amount={item['estimated_amount']:,.0f}, "
                f"score={item['score']}, reason={item['reason']}"
            )
    return lines


def _preview_skipped_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- None"]
    return [f"- {item.get('code')} {item.get('name')}: {item.get('reason')}" for item in items]


def _preview_safety_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- passed: true / rule: no_order"]
    return [f"- {item['action']} {item['code']} {item['name']}: {'passed' if item['passed'] else 'rejected'} / rule: {item['rule']}" for item in items]


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
    global BACKTEST_MODE_ACTIVE
    if provider_name != "jquants":
        raise SystemExit("backtest mode currently supports --provider jquants only.")
    start_date = date.fromisoformat(start_date_text)
    end_date = date.fromisoformat(end_date_text)
    range_key = f"{start_date_text}_to_{end_date_text}"
    config = load_config(CONFIG_PATH)
    BACKTEST_MODE_ACTIVE = True
    profile_id = profile_id_from(config)
    backtest_dir = ROOT / "logs" / "backtests" / profile_id / range_key
    report_dir = ROOT / "reports" / "backtests" / profile_id / range_key
    article_dir = ROOT / "articles" / "drafts" / "backtests" / profile_id / range_key
    backtest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    article_dir.mkdir(parents=True, exist_ok=True)
    state = initial_live_paper_state(config)
    daily_summaries = []
    all_trades = []
    processed_dates = []

    try:
        _run_daily_step(1, 3, "fetch-period-prices", lambda: ensure_price_history_for_backtest(provider_name, start_date, end_date))
        trading_dates = _run_daily_step(2, 3, "detect-trading-days", lambda: available_cached_price_dates(start_date, end_date))
        if not trading_dates:
            raise SystemExit("No cached trading days found for the backtest period. The period may be weekend, holiday, or unavailable.")

        print(f"backtest trading_days: {len(trading_dates)}")
        print(f"backtest news fetch: {'disabled' if _backtest_disable_news(config) else 'enabled'}")
        print(f"backtest OpenAI: {'disabled' if _backtest_disable_openai(config) else 'configured by profile'}")
        print(f"backtest indicator_mode: {_backtest_indicator_mode(config)}")
        for index, trading_date in enumerate(trading_dates, start=1):
            target_date_text = trading_date.isoformat()
            global BACKTEST_DAY_LOG_PREFIX
            BACKTEST_DAY_LOG_PREFIX = f"[day {index}/{len(trading_dates)}] {target_date_text}"
            print(f"[day {index}/{len(trading_dates)}] {target_date_text} start")
            _run_backtest_day_step(index, len(trading_dates), target_date_text, "calculate-indicators", lambda: ensure_indicators(provider_name, target_date_text), lambda: _backtest_indicator_metrics(target_date_text))
            _run_backtest_day_step(index, len(trading_dates), target_date_text, "market-context", lambda: ensure_market_context(provider_name, target_date_text), lambda: _backtest_market_context_metrics(target_date_text))
            _run_backtest_day_step(index, len(trading_dates), target_date_text, "screen", lambda: run_screen(provider_name, target_date_text), lambda: _backtest_screen_metrics(config, target_date_text))
            scoring_log = _run_backtest_day_step(index, len(trading_dates), target_date_text, "score", lambda: score_for_date(provider_name, target_date_text), lambda: _backtest_score_metrics(config, target_date_text))
            _run_backtest_day_step(index, len(trading_dates), target_date_text, "ai-decision", lambda: _backtest_ai_decision_status(scoring_log, config), lambda: _backtest_ai_decision_metrics(scoring_log))

            trade_context: dict[str, Any] = {}

            def run_trade_step() -> dict[str, Any]:
                nonlocal state
                scored_candidates = enrich_candidates_with_position_prices(scoring_log.get("scores", []), state, target_date_text)
                state, portfolio_summary, trades = execute_real_data_paper_trade(scored_candidates, state, config, target_date_text)
                attach_config_version(portfolio_summary, config)
                for trade in trades:
                    trade.setdefault("config_version", config_version_from(config))
                scoring_log.setdefault("config_version", config_version_from(config))
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
                reflection_path = write_backtest_reflections(backtest_dir, target_date_text, trade_result)
                report_path, article_path = write_backtest_daily_markdown(report_dir, article_dir, target_date_text, trade_result, scoring_log)
                report_context.update({"reflection_path": reflection_path, "report_path": report_path, "article_path": article_path})
                return reflection_path, report_path, article_path

            reflection_path, report_path, article_path = _run_backtest_day_step(index, len(trading_dates), target_date_text, "reports/articles", run_reports_step, lambda: _backtest_report_metrics(report_context))

            portfolio_summary = trade_context["portfolio_summary"]
            trades = trade_context["trades"]

            def run_db_save_step() -> None:
                write_json(backtest_dir / f"trades_{target_date_text}.json", {"date": target_date_text, "provider": provider_name, "config_version": config_version_from(config), "trades": trades})
                write_json(backtest_dir / f"portfolio_{target_date_text}.json", portfolio_summary)
                write_json(backtest_dir / f"safety_{target_date_text}.json", {"date": target_date_text, "config_version": config_version_from(config), "safety_events": trade_result.get("safety_events", [])})
                write_json(backtest_dir / f"scoring_{target_date_text}.json", scoring_log)
                save_portfolio_snapshot(config, ROOT, portfolio_summary)
                save_trades(config, ROOT, target_date_text, trades)
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
            lambda: write_backtest_summary(range_key, start_date_text, end_date_text, config, state, daily_summaries, all_trades, backtest_dir),
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
    print(f"processed_days: {len(processed_dates)}")
    print(f"final_assets: {summary['final_assets']}")
    print(f"cumulative_profit: {summary['cumulative_profit']}")
    print(f"summary_md: {Path(summary['report_markdown_path']).relative_to(ROOT)}")
    print(f"summary_json: {Path(summary['report_json_path']).relative_to(ROOT)}")
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
    print(f"{prefix} done in {elapsed:.1f}s")
    if metrics:
        for line in metrics() or []:
            print(f"  {line}")
    return result


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


def _backtest_indicator_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("backtest", {}).get("indicator_mode", "fast"))
    return mode if mode in {"full", "fast", "minimal"} else "fast"


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
    if BACKTEST_MODE_ACTIVE and profile_path.exists():
        payload = read_json(profile_path)
        if payload.get("indicator_mode") == indicator_mode:
            write_json(path, payload)
            print(f"{BACKTEST_DAY_LOG_PREFIX} indicators cache hit: {profile_path.relative_to(ROOT)}")
            return
    if path.exists() and (not BACKTEST_MODE_ACTIVE or read_json(path).get("indicator_mode") == indicator_mode):
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
            write_json(trades_path, {"date": target_date_text, "provider": provider_name, "profile_id": profile_id_from(config), "profile_name": profile_name_from(config), "config_version": config_version_from(config), "trades": cached_trades})
            write_json(portfolio_path, portfolio_summary)
            write_json(safety_path, {"date": target_date_text, "provider": provider_name, "profile_id": profile_id_from(config), "profile_name": profile_name_from(config), "config_version": config_version_from(config), "safety_events": safety_events})
            save_portfolio_snapshot(config, ROOT, portfolio_summary)
            save_trades(config, ROOT, target_date_text, cached_trades)
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

    write_json(state_path, updated_state)
    write_json(trades_path, {"date": target_date_text, "provider": provider_name, "profile_id": profile_id_from(config), "profile_name": profile_name_from(config), "config_version": config_version_from(config), "trades": trades})
    write_json(portfolio_path, portfolio_summary)
    write_json(safety_path, {"date": target_date_text, "provider": provider_name, "profile_id": profile_id_from(config), "profile_name": profile_name_from(config), "config_version": config_version_from(config), "safety_events": portfolio_summary.get("safety_events", [])})
    save_portfolio_snapshot(config, ROOT, portfolio_summary)
    save_trades(config, ROOT, target_date_text, trades)
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
    summary = normalize_real_summary_for_markdown(trade_result["portfolio_summary"], target_date_text)
    summary["market_context"] = load_market_context_for_date(target_date_text, "jquants")
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
                "news_score": item.get("news_score"),
                "financial_score": item.get("financial_score"),
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
                "news_reason": item.get("news_reason", "ニュース材料は中立です"),
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
    report_path = ROOT / "reports" / profile_id_from(config) / f"day_{target_date_text}.md"
    article_path = ROOT / "articles" / "drafts" / profile_id_from(config) / f"day_{target_date_text}.md"
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
    return report_path, article_path


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
                "news_score": item.get("news_score"),
                "financial_score": item.get("financial_score"),
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
                "news_reason": item.get("news_reason", "ニュース材料は中立です"),
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
    return report_path, article_path


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
    return [dict(row) for row in rows]


def write_trades_csv_from_db(config: dict[str, Any], root: Path = ROOT) -> tuple[Path, int, int]:
    rows = load_trade_rows_for_csv(config, root)
    trades_csv = root / "reports" / profile_id_from(config) / "trades.csv"
    write_trades_csv(trades_csv, rows)
    return trades_csv, len(rows), count_csv_data_rows(trades_csv)


def ensure_price_history_for_backtest(provider_name: str, start_date: date, end_date: date) -> None:
    if provider_name != "jquants":
        raise SystemExit("backtest mode currently supports --provider jquants only.")

    target_dates = business_dates_between(start_date, end_date)
    print(f"fetch-period-prices target period: {start_date.isoformat()} to {end_date.isoformat()}")
    print(f"fetch-period-prices target business days: {len(target_dates)}")
    if target_dates and all((ROOT / "data" / "processed" / f"indicators_{target.isoformat()}.json").exists() for target in target_dates):
        print("backtest price history: using cached indicator files")
        return

    prime_path = ROOT / "data" / "raw" / "prime_stocks_jquants.json"
    if not prime_path.exists():
        print("fetch-period-prices prime stock cache: missing; fetching J-Quants master")
        try:
            run_list_stocks(provider_name)
        except (RuntimeError, SystemExit) as exc:
            cached_dates = available_cached_price_dates(start_date, end_date)
            if cached_dates:
                print(
                    "fetch-period-prices warning: prime stock master fetch failed; "
                    f"continuing with {len(cached_dates)} cached backtest day(s). reason={exc}"
                )
                return
            raise

    prime_payload = read_json(prime_path)
    prime_codes = {stock["code"] for stock in prime_payload.get("stocks", [])}
    if not prime_codes:
        raise SystemExit("Prime stock list is empty. Re-run list-stocks and check the result.")

    config = load_config(CONFIG_PATH)
    lookback_start = previous_business_dates(start_date, 30)[0]
    fetch_dates = business_dates_between(lookback_start, end_date)
    cached_dates = [fetch_date for fetch_date in fetch_dates if load_cached_prime_prices(fetch_date) is not None]
    missing_dates = [fetch_date for fetch_date in fetch_dates if load_cached_prime_prices(fetch_date) is None]
    print(
        "fetch-period-prices price cache: "
        f"cached={len(cached_dates)} missing={len(missing_dates)} total={len(fetch_dates)} "
        f"lookback_start={lookback_start.isoformat()}"
    )
    for fetch_date in cached_dates:
        print(f"fetch-period-prices cache hit: {fetch_date.isoformat()}")
    if fetch_dates and not missing_dates:
        print("backtest price history: using cached price files")
        return

    try:
        provider = JQuantsDataProvider(
            ROOT / ".env",
            timeout_seconds=int(config.get("jquants", {}).get("request_timeout_seconds", 20)),
        )
    except RuntimeError as exc:
        usable_dates = available_cached_price_dates(start_date, end_date)
        if usable_dates:
            print(
                "fetch-period-prices warning: J-Quants provider unavailable; "
                f"continuing with {len(usable_dates)} cached backtest day(s). reason={exc}"
            )
            return
        raise
    try:
        fetch_price_history(
            provider,
            end_date,
            prime_codes,
            lookback_business_days=len(fetch_dates),
            rate_limit_per_minute=int(config.get("jquants", {}).get("rate_limit_per_minute", 5)),
            fetch_dates=fetch_dates,
            continue_on_error=True,
            verbose=True,
        )
    except KeyboardInterrupt:
        print("fetch-period-prices interrupted. Existing cache is preserved.")
        raise

    usable_dates = available_cached_price_dates(start_date, end_date)
    if usable_dates:
        print(
            "fetch-period-prices completed with cached usable days: "
            f"{len(usable_dates)}/{len(target_dates)}"
        )
        return
    print("fetch-period-prices completed, but no usable cached dates were found for the target period.")


def available_cached_price_dates(start_date: date, end_date: date) -> list[date]:
    dates = []
    for current in business_dates_between(start_date, end_date):
        cached_rows = load_cached_prime_prices(current)
        if cached_rows:
            dates.append(current)
    return dates


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
    if not indicators_path.exists():
        return enriched

    indicators = read_json(indicators_path).get("indicators", [])
    by_code = {item["code"]: item for item in indicators}
    for code in sorted(missing_codes):
        indicator = by_code.get(code)
        if not indicator:
            continue
        enriched.append(
            {
                "code": indicator["code"],
                "name": indicator["name"],
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
                "news_score": 0,
                "financial_score": 0,
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
) -> dict[str, Any]:
    initial_capital = float(config["portfolio"]["initial_cash"])
    final_assets = float(state.get("total_assets", initial_capital))
    closed_trades = state.get("closed_trades", [])
    executed_trade_actions = [
        trade
        for trade in all_trades
        if trade.get("action") in {"BUY", "SELL"} and str(trade.get("order_status") or trade.get("status") or "").upper() == "FILLED"
    ]
    period_profit = calculate_period_profit_summary(closed_trades, config)
    pf_metrics = profit_factor_metrics(all_trades)
    gross_profit_total = pf_metrics["gross_profit_total"]
    gross_win_total = pf_metrics["gross_win_total"]
    gross_loss_total = pf_metrics["gross_loss_total"]
    net_cumulative_profit = period_profit["net_cumulative_profit"]
    wins = pf_metrics["win_count"]
    win_rate = round(wins / pf_metrics["closed_trade_count"], 4) if pf_metrics["closed_trade_count"] else None
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
    summary = {
        "start_date": start_date_text,
        "end_date": end_date_text,
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
        "total_trades": len(executed_trade_actions),
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
        "",
        "### exit_reason別集計",
        "",
        *_exit_reason_analysis_lines(trades.get("exit_reason_analysis", [])),
        "",
        "## config_version別集計",
        "",
        *_config_version_lines(config_versions),
        "",
        "## 業種別勝率",
        "",
        *_sector_win_rate_lines(sector_win_rates),
        "",
        "## スコア分析",
        "",
        f"- selected銘柄数: {scores['selected_count']}",
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


def _exit_reason_analysis_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 売却理由別データなし"]
    return [
        (
            f"- {item['exit_reason']}: 件数 {item['count']}件, "
            f"平均損益率 {_format_optional_percent(item.get('average_profit_rate'))}"
        )
        for item in items
    ]


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
) -> list[dict[str, Any]]:
    rows = []
    api_requests = 0
    request_interval = 60 / max(rate_limit_per_minute, 1)
    target_fetch_dates = fetch_dates or previous_business_dates(target_date, lookback_business_days)
    total_dates = len(target_fetch_dates)
    for index, fetch_date in enumerate(target_fetch_dates, start=1):
        cached_rows = load_cached_prime_prices(fetch_date)
        if cached_rows is not None:
            if verbose:
                print(
                    f"fetch-period-prices [{index}/{total_dates}] "
                    f"{fetch_date.isoformat()} cache ({len(cached_rows)} prime rows)"
                )
            rows.extend(cached_rows)
            continue

        if api_requests > 0:
            if verbose:
                print(f"fetch-period-prices waiting {request_interval:.1f}s for rate limit")
            time.sleep(request_interval)
        if verbose:
            print(
                f"fetch-period-prices [{index}/{total_dates}] "
                f"{fetch_date.isoformat()} fetching J-Quants API"
            )
        try:
            daily_prices = provider.get_daily_prices(fetch_date)
        except KeyboardInterrupt:
            if verbose:
                print(f"fetch-period-prices {fetch_date.isoformat()} interrupted by Ctrl+C")
            raise
        except RuntimeError as exc:
            if verbose:
                print(f"fetch-period-prices {fetch_date.isoformat()} failed: {exc}")
            if continue_on_error:
                api_requests += 1
                continue
            raise
        prime_prices = [
            _normalize_daily_price(record)
            for record in daily_prices
            if _get_first(record, ["code", "Code", "LocalCode"]) in prime_codes
        ]
        if prime_prices:
            cache_price_snapshot(fetch_date, len(daily_prices), prime_prices)
            rows.extend(prime_prices)
            if verbose:
                print(
                    f"fetch-period-prices {fetch_date.isoformat()} saved "
                    f"{len(prime_prices)} prime rows from {len(daily_prices)} rows"
                )
        elif verbose:
            print(f"fetch-period-prices {fetch_date.isoformat()} no prime rows; not cached")
        api_requests += 1
    return rows


def load_cached_prime_prices(fetch_date: date) -> Any:
    path = ROOT / "data" / "raw" / f"prices_{fetch_date.isoformat()}.json"
    if not path.exists():
        return None
    payload = read_json(path)
    return payload.get("prices", [])


def load_cached_price_history(fetch_dates: list[date]) -> list[dict[str, Any]]:
    rows = []
    for fetch_date in fetch_dates:
        cached_rows = load_cached_prime_prices(fetch_date)
        if cached_rows:
            rows.extend(cached_rows)
    return rows


def cache_price_snapshot(fetch_date: date, total_count: int, prime_prices: list[dict[str, Any]]) -> None:
    path = ROOT / "data" / "raw" / f"prices_{fetch_date.isoformat()}.json"
    write_json(
        path,
        {
            "provider": "jquants",
            "date": fetch_date.isoformat(),
            "total_count": total_count,
            "prime_count": len(prime_prices),
            "prices": prime_prices,
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
    market_values = [
        _get_first(record, ["market", "market_name", "MarketName", "MarketCodeName", "market_segment", "MktNm"]),
        _get_first(record, ["market_code", "MarketCode", "market_segment_code", "Mkt"]),
    ]
    joined = " ".join(str(value) for value in market_values if value)
    return "プライム" in joined or "Prime" in joined or "prime" in joined or "0111" in joined


def _normalize_prime_stock(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": _get_first(record, ["code", "Code", "local_code", "LocalCode"]),
        "name": _get_first(record, ["name", "Name", "company_name", "CompanyName", "issue_name", "IssueName", "CoName"]),
        "market": _get_first(record, ["market", "market_name", "MarketName", "MarketCodeName", "market_segment", "MktNm"]),
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
        "report": ROOT / "reports" / profile_id / month_key / f"{file_key}.md",
        "article": ROOT / "articles" / "drafts" / profile_id / month_key / f"{file_key}.md",
    }


def build_data_provider(config: dict[str, Any], run_date: date, run_id: str) -> Any:
    provider_name = config.get("data_provider", "dummy")
    if provider_name == "dummy":
        return DummyDataProvider(config, run_date, run_id)
    if provider_name == "jquants":
        return JQuantsDataProvider(ROOT / ".env")
    raise ValueError(f"Unsupported data_provider: {provider_name}")


def load_config(path: Path) -> dict[str, Any]:
    return load_profile(ACTIVE_PROFILE_ID)


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
        "entry_date",
        "exit_date",
        "holding_days",
        "entry_price",
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
