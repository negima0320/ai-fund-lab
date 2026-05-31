"""SQLite persistence helpers for AI Fund Lab operation data."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from config_version import config_version_from
from tax import calculate_period_estimated_tax
from trade_metrics import is_filled_trade, profit_factor_metrics


HOLDING_PERIOD_SIMULATION_DAYS = [4, 5, 6, 7, 8, 10]
WALK_FORWARD_PERIODS = [
    ("2025-03-01", "2025-06-30"),
    ("2025-07-01", "2025-10-31"),
    ("2025-11-01", "2026-03-06"),
]


def get_database_path(config: dict[str, Any], root: Path) -> Path:
    configured_path = config.get("database", {}).get("path", "storage/ai_fund_lab.sqlite3")
    path = Path(configured_path)
    if not path.is_absolute():
        path = root / path
    return path


def _profile_id(config: dict[str, Any]) -> str:
    return str(config.get("profile_id") or config.get("dealer", {}).get("id") or "rookie_dealer_01")


def _profile_name(config: dict[str, Any]) -> str:
    return str(config.get("profile_name") or config.get("dealer", {}).get("name") or _profile_id(config))


def initialize_database(config: dict[str, Any], root: Path) -> Path:
    db_path = get_database_path(config, root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT,
                profile_name TEXT,
                date TEXT NOT NULL,
                cash REAL,
                positions_value REAL,
                total_assets REAL,
                daily_profit REAL,
                cumulative_profit REAL,
                cumulative_profit_rate REAL,
                win_rate REAL,
                max_drawdown REAL,
                open_positions_count INTEGER,
                closed_trades_count INTEGER,
                gross_cumulative_profit REAL,
                net_cumulative_profit REAL,
                total_commission REAL,
                estimated_tax_total REAL,
                net_total_assets REAL,
                dealer_comment TEXT,
                config_version TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                profile_id TEXT,
                profile_name TEXT,
                action TEXT,
                code TEXT,
                name TEXT,
                sector_name TEXT,
                entry_date TEXT,
                exit_date TEXT,
                holding_days INTEGER,
                entry_price REAL,
                exit_price REAL,
                shares INTEGER,
                amount REAL,
                profit REAL,
                profit_rate REAL,
                exit_reason TEXT,
                result TEXT,
                score REAL,
                rsi REAL,
                volume_ratio REAL,
                stock_return_5d REAL,
                stock_return_10d REAL,
                stock_return_20d REAL,
                benchmark_source TEXT,
                benchmark_return_5d REAL,
                benchmark_return_10d REAL,
                benchmark_return_20d REAL,
                relative_strength_5d REAL,
                relative_strength_10d REAL,
                relative_strength_20d REAL,
                relative_strength_score REAL,
                topix_records_loaded REAL,
                topix_api_calls REAL,
                investor_context_source TEXT,
                investor_context_week TEXT,
                overseas_net_buy REAL,
                overseas_net_buy_4w_sum REAL,
                overseas_net_buy_4w_trend TEXT,
                overseas_buy_sell_ratio REAL,
                individual_net_buy REAL,
                institution_net_buy REAL,
                trust_bank_net_buy REAL,
                proprietary_net_buy REAL,
                investor_context_score REAL,
                total_score REAL,
                technical_score REAL,
                ma_score REAL,
                rsi_score REAL,
                volume_score REAL,
                candlestick_score REAL,
                market_context_score REAL,
                sector_score REAL,
                penalty_score REAL,
                score_components TEXT,
                score_components_total REAL,
                score_components_match INTEGER,
                market_regime TEXT,
                advance_ratio REAL,
                candlestick_signals TEXT,
                selected_reason TEXT,
                reason TEXT,
                round_lot_size INTEGER,
                use_round_lot INTEGER,
                skipped_reason TEXT,
                intended_price REAL,
                executed_price REAL,
                slippage_amount REAL,
                slippage_rate REAL,
                stop_loss_rate REAL,
                stop_loss_trigger_price REAL,
                stop_loss_triggered_date TEXT,
                intended_exit_price REAL,
                actual_exit_price REAL,
                gap_slippage_rate REAL,
                stop_loss_slippage_rate REAL,
                gross_profit REAL,
                gross_profit_rate REAL,
                buy_commission REAL,
                sell_commission REAL,
                total_commission REAL,
                taxable_profit REAL,
                estimated_tax REAL,
                net_profit REAL,
                net_profit_rate REAL,
                dealer_comment TEXT,
                broker_provider TEXT,
                order_status TEXT,
                live_trading INTEGER,
                safety_checked INTEGER,
                config_version TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scoring_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT,
                profile_name TEXT,
                date TEXT NOT NULL,
                code TEXT,
                name TEXT,
                sector_name TEXT,
                sector_momentum_score REAL,
                sector_rank INTEGER,
                sector_comment TEXT,
                sector_score_adjustment REAL,
                rank INTEGER,
                total_score REAL,
                technical_score REAL,
                confidence REAL,
                selected INTEGER,
                reason TEXT,
                rejected_reason TEXT,
                fallback INTEGER,
                macd REAL,
                macd_signal REAL,
                macd_hist REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                atr REAL,
                stock_return_5d REAL,
                stock_return_10d REAL,
                stock_return_20d REAL,
                benchmark_source TEXT,
                benchmark_return_5d REAL,
                benchmark_return_10d REAL,
                benchmark_return_20d REAL,
                relative_strength_5d REAL,
                relative_strength_10d REAL,
                relative_strength_20d REAL,
                relative_strength_score REAL,
                investor_context_source TEXT,
                investor_context_week TEXT,
                overseas_net_buy REAL,
                overseas_net_buy_4w_sum REAL,
                overseas_net_buy_4w_trend TEXT,
                overseas_buy_sell_ratio REAL,
                individual_net_buy REAL,
                institution_net_buy REAL,
                trust_bank_net_buy REAL,
                proprietary_net_buy REAL,
                investor_context_score REAL,
                candle_type TEXT,
                candle_body_rate REAL,
                upper_shadow_rate REAL,
                lower_shadow_rate REAL,
                close_position_in_range REAL,
                gap_rate REAL,
                candlestick_signals TEXT,
                candlestick_score REAL,
                trend_score REAL,
                volume_score REAL,
                rsi_score REAL,
                ma_score REAL,
                market_context_score REAL,
                sector_score REAL,
                penalty_score REAL,
                score_components TEXT,
                score_components_total REAL,
                score_components_match INTEGER,
                market_filter_applied INTEGER,
                market_regime TEXT,
                market_filter_reason TEXT,
                earnings_filter_checked INTEGER,
                earnings_filter_blocked INTEGER,
                earnings_filter_reason TEXT,
                earnings_announcement_date TEXT,
                source_provider TEXT,
                config_version TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS screening_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT,
                profile_name TEXT,
                date TEXT NOT NULL,
                code TEXT,
                name TEXT,
                sector_name TEXT,
                sector_momentum_score REAL,
                sector_rank INTEGER,
                sector_comment TEXT,
                close REAL,
                volume REAL,
                ma5 REAL,
                ma25 REAL,
                rsi REAL,
                volume_ratio REAL,
                turnover_value REAL,
                five_day_volatility REAL,
                stock_return_5d REAL,
                stock_return_10d REAL,
                stock_return_20d REAL,
                benchmark_return_5d REAL,
                benchmark_return_10d REAL,
                benchmark_return_20d REAL,
                relative_strength_5d REAL,
                relative_strength_10d REAL,
                relative_strength_20d REAL,
                relative_strength_score REAL,
                macd REAL,
                macd_signal REAL,
                macd_hist REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                atr REAL,
                candle_type TEXT,
                candle_body_rate REAL,
                upper_shadow_rate REAL,
                lower_shadow_rate REAL,
                close_position_in_range REAL,
                gap_rate REAL,
                candlestick_signals TEXT,
                candlestick_score REAL,
                trend_score REAL,
                volume_score REAL,
                rsi_score REAL,
                fallback INTEGER,
                pass_reason TEXT,
                rejected_reason TEXT,
                config_version TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT,
                profile_name TEXT,
                trade_id TEXT,
                code TEXT,
                name TEXT,
                result TEXT,
                profit REAL,
                profit_rate REAL,
                summary TEXT,
                good_points TEXT,
                bad_points TEXT,
                lesson TEXT,
                suggestions TEXT,
                reflection_comment TEXT,
                config_version TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT,
                profile_name TEXT,
                date TEXT NOT NULL,
                day INTEGER,
                title TEXT,
                status TEXT,
                path TEXT,
                note_url TEXT,
                body TEXT,
                published_at TEXT,
                config_version TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS safety_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                action TEXT,
                code TEXT,
                name TEXT,
                rejected INTEGER,
                rejected_reason TEXT,
                safety_rule TEXT,
                broker_provider TEXT,
                order_status TEXT,
                live_trading INTEGER,
                safety_checked INTEGER,
                order_payload TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pending_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT,
                action TEXT,
                code TEXT,
                name TEXT,
                created_date TEXT,
                scheduled_execution_date TEXT,
                intended_price REAL,
                status TEXT,
                score REAL,
                reason TEXT,
                config_version TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ai_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT,
                profile_name TEXT,
                date TEXT NOT NULL,
                config_version TEXT,
                provider TEXT,
                model TEXT,
                candidates_count INTEGER,
                selected_count INTEGER,
                decision_summary TEXT,
                rookie_comment TEXT,
                fallback_used INTEGER,
                token_input INTEGER,
                token_output INTEGER,
                estimated_cost REAL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS market_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT,
                profile_name TEXT,
                date TEXT NOT NULL,
                provider TEXT,
                market_regime TEXT,
                advancers INTEGER,
                decliners INTEGER,
                advance_ratio REAL,
                average_change_rate REAL,
                turnover_value_total REAL,
                topix_change_rate REAL,
                nikkei_change_rate REAL,
                usd_jpy REAL,
                us_market_summary TEXT,
                important_news TEXT,
                sector_momentum TEXT,
                market_comment TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ai_analysis_exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT,
                start_date TEXT,
                end_date TEXT,
                dataset_path TEXT,
                summary_path TEXT,
                record_count INTEGER,
                created_at TEXT NOT NULL
            );
            """
        )
        for table in [
            "portfolio_snapshots",
            "trades",
            "scoring_results",
            "screening_results",
            "reflections",
            "articles",
            "ai_decisions",
            "market_contexts",
        ]:
            _add_column_if_missing(connection, table, "profile_id", "TEXT")
            _add_column_if_missing(connection, table, "profile_name", "TEXT")
        _add_column_if_missing(connection, "trades", "round_lot_size", "INTEGER")
        _add_column_if_missing(connection, "trades", "use_round_lot", "INTEGER")
        _add_column_if_missing(connection, "trades", "skipped_reason", "TEXT")
        _add_column_if_missing(connection, "trades", "intended_price", "REAL")
        _add_column_if_missing(connection, "trades", "executed_price", "REAL")
        _add_column_if_missing(connection, "trades", "slippage_amount", "REAL")
        _add_column_if_missing(connection, "trades", "slippage_rate", "REAL")
        _add_column_if_missing(connection, "trades", "stop_loss_rate", "REAL")
        _add_column_if_missing(connection, "trades", "stop_loss_trigger_price", "REAL")
        _add_column_if_missing(connection, "trades", "stop_loss_triggered_date", "TEXT")
        _add_column_if_missing(connection, "trades", "intended_exit_price", "REAL")
        _add_column_if_missing(connection, "trades", "actual_exit_price", "REAL")
        _add_column_if_missing(connection, "trades", "gap_slippage_rate", "REAL")
        _add_column_if_missing(connection, "trades", "stop_loss_slippage_rate", "REAL")
        _add_column_if_missing(connection, "trades", "gross_profit", "REAL")
        _add_column_if_missing(connection, "trades", "gross_profit_rate", "REAL")
        _add_column_if_missing(connection, "trades", "buy_commission", "REAL")
        _add_column_if_missing(connection, "trades", "sell_commission", "REAL")
        _add_column_if_missing(connection, "trades", "total_commission", "REAL")
        _add_column_if_missing(connection, "trades", "taxable_profit", "REAL")
        _add_column_if_missing(connection, "trades", "estimated_tax", "REAL")
        _add_column_if_missing(connection, "trades", "net_profit", "REAL")
        _add_column_if_missing(connection, "trades", "net_profit_rate", "REAL")
        _add_column_if_missing(connection, "trades", "dealer_comment", "TEXT")
        _add_column_if_missing(connection, "trades", "sector_name", "TEXT")
        _add_column_if_missing(connection, "trades", "rsi", "REAL")
        _add_column_if_missing(connection, "trades", "volume_ratio", "REAL")
        _add_relative_strength_columns(connection, "trades")
        _add_investor_context_columns(connection, "trades")
        _drop_removed_score_columns(connection)
        _add_column_if_missing(connection, "trades", "total_score", "REAL")
        _add_column_if_missing(connection, "trades", "technical_score", "REAL")
        _add_column_if_missing(connection, "trades", "ma_score", "REAL")
        _add_column_if_missing(connection, "trades", "rsi_score", "REAL")
        _add_column_if_missing(connection, "trades", "volume_score", "REAL")
        _add_column_if_missing(connection, "trades", "candlestick_score", "REAL")
        _add_column_if_missing(connection, "trades", "market_context_score", "REAL")
        _add_column_if_missing(connection, "trades", "sector_score", "REAL")
        _add_column_if_missing(connection, "trades", "penalty_score", "REAL")
        _add_column_if_missing(connection, "trades", "score_components", "TEXT")
        _add_column_if_missing(connection, "trades", "score_components_total", "REAL")
        _add_column_if_missing(connection, "trades", "score_components_match", "INTEGER")
        _add_column_if_missing(connection, "trades", "market_regime", "TEXT")
        _add_column_if_missing(connection, "trades", "advance_ratio", "REAL")
        _add_column_if_missing(connection, "trades", "candlestick_signals", "TEXT")
        _add_column_if_missing(connection, "trades", "earnings_filter_checked", "INTEGER")
        _add_column_if_missing(connection, "trades", "earnings_filter_blocked", "INTEGER")
        _add_column_if_missing(connection, "trades", "earnings_filter_reason", "TEXT")
        _add_column_if_missing(connection, "trades", "earnings_announcement_date", "TEXT")
        _add_column_if_missing(connection, "trades", "selected_reason", "TEXT")
        _add_column_if_missing(connection, "trades", "broker_provider", "TEXT")
        _add_column_if_missing(connection, "trades", "order_status", "TEXT")
        _add_column_if_missing(connection, "trades", "live_trading", "INTEGER")
        _add_column_if_missing(connection, "trades", "safety_checked", "INTEGER")
        _add_column_if_missing(connection, "trades", "config_version", "TEXT")
        _add_column_if_missing(connection, "portfolio_snapshots", "gross_cumulative_profit", "REAL")
        _add_column_if_missing(connection, "portfolio_snapshots", "net_cumulative_profit", "REAL")
        _add_column_if_missing(connection, "portfolio_snapshots", "total_commission", "REAL")
        _add_column_if_missing(connection, "portfolio_snapshots", "estimated_tax_total", "REAL")
        _add_column_if_missing(connection, "portfolio_snapshots", "net_total_assets", "REAL")
        _add_column_if_missing(connection, "portfolio_snapshots", "dealer_comment", "TEXT")
        _add_column_if_missing(connection, "portfolio_snapshots", "config_version", "TEXT")
        _add_column_if_missing(connection, "reflections", "reflection_comment", "TEXT")
        _add_column_if_missing(connection, "reflections", "config_version", "TEXT")
        _add_column_if_missing(connection, "safety_events", "broker_provider", "TEXT")
        _add_column_if_missing(connection, "safety_events", "order_status", "TEXT")
        _add_column_if_missing(connection, "safety_events", "live_trading", "INTEGER")
        _add_column_if_missing(connection, "safety_events", "safety_checked", "INTEGER")
        _add_column_if_missing(connection, "articles", "published_at", "TEXT")
        _add_column_if_missing(connection, "articles", "config_version", "TEXT")
        _delete_non_trade_order_rows(connection)
        _add_column_if_missing(connection, "scoring_results", "ai_reason", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "ai_risk", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "ai_confidence", "REAL")
        _add_column_if_missing(connection, "scoring_results", "ai_score", "REAL")
        _add_column_if_missing(connection, "scoring_results", "macd", "REAL")
        _add_column_if_missing(connection, "scoring_results", "sector_name", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "sector_momentum_score", "REAL")
        _add_column_if_missing(connection, "scoring_results", "sector_rank", "INTEGER")
        _add_column_if_missing(connection, "scoring_results", "sector_comment", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "sector_score_adjustment", "REAL")
        _add_column_if_missing(connection, "scoring_results", "macd_signal", "REAL")
        _add_column_if_missing(connection, "scoring_results", "macd_hist", "REAL")
        _add_column_if_missing(connection, "scoring_results", "bb_upper", "REAL")
        _add_column_if_missing(connection, "scoring_results", "bb_middle", "REAL")
        _add_column_if_missing(connection, "scoring_results", "bb_lower", "REAL")
        _add_column_if_missing(connection, "scoring_results", "atr", "REAL")
        _add_relative_strength_columns(connection, "scoring_results")
        _add_investor_context_columns(connection, "scoring_results")
        _add_column_if_missing(connection, "scoring_results", "candle_type", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "candle_body_rate", "REAL")
        _add_column_if_missing(connection, "scoring_results", "upper_shadow_rate", "REAL")
        _add_column_if_missing(connection, "scoring_results", "lower_shadow_rate", "REAL")
        _add_column_if_missing(connection, "scoring_results", "close_position_in_range", "REAL")
        _add_column_if_missing(connection, "scoring_results", "gap_rate", "REAL")
        _add_column_if_missing(connection, "scoring_results", "candlestick_signals", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "candlestick_score", "REAL")
        _add_column_if_missing(connection, "scoring_results", "trend_score", "REAL")
        _add_column_if_missing(connection, "scoring_results", "volume_score", "REAL")
        _add_column_if_missing(connection, "scoring_results", "rsi_score", "REAL")
        _add_column_if_missing(connection, "scoring_results", "ma_score", "REAL")
        _add_column_if_missing(connection, "scoring_results", "market_context_score", "REAL")
        _add_column_if_missing(connection, "scoring_results", "sector_score", "REAL")
        _add_column_if_missing(connection, "scoring_results", "penalty_score", "REAL")
        _add_column_if_missing(connection, "scoring_results", "score_components", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "score_components_total", "REAL")
        _add_column_if_missing(connection, "scoring_results", "score_components_match", "INTEGER")
        _add_column_if_missing(connection, "scoring_results", "earnings_filter_checked", "INTEGER")
        _add_column_if_missing(connection, "scoring_results", "earnings_filter_blocked", "INTEGER")
        _add_column_if_missing(connection, "scoring_results", "earnings_filter_reason", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "earnings_announcement_date", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "market_filter_applied", "INTEGER")
        _add_column_if_missing(connection, "scoring_results", "market_regime", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "market_filter_reason", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "config_version", "TEXT")
        _add_column_if_missing(connection, "screening_results", "candle_type", "TEXT")
        _add_column_if_missing(connection, "screening_results", "candle_body_rate", "REAL")
        _add_column_if_missing(connection, "screening_results", "upper_shadow_rate", "REAL")
        _add_column_if_missing(connection, "screening_results", "lower_shadow_rate", "REAL")
        _add_column_if_missing(connection, "screening_results", "close_position_in_range", "REAL")
        _add_column_if_missing(connection, "screening_results", "gap_rate", "REAL")
        _add_column_if_missing(connection, "screening_results", "candlestick_signals", "TEXT")
        _add_column_if_missing(connection, "screening_results", "candlestick_score", "REAL")
        _add_column_if_missing(connection, "screening_results", "trend_score", "REAL")
        _add_column_if_missing(connection, "screening_results", "volume_score", "REAL")
        _add_column_if_missing(connection, "screening_results", "rsi_score", "REAL")
        _add_column_if_missing(connection, "screening_results", "config_version", "TEXT")
        _add_column_if_missing(connection, "screening_results", "sector_name", "TEXT")
        _add_column_if_missing(connection, "screening_results", "sector_momentum_score", "REAL")
        _add_column_if_missing(connection, "screening_results", "sector_rank", "INTEGER")
        _add_column_if_missing(connection, "screening_results", "sector_comment", "TEXT")
        _add_column_if_missing(connection, "screening_results", "macd", "REAL")
        _add_column_if_missing(connection, "screening_results", "macd_signal", "REAL")
        _add_column_if_missing(connection, "screening_results", "macd_hist", "REAL")
        _add_column_if_missing(connection, "screening_results", "bb_upper", "REAL")
        _add_column_if_missing(connection, "screening_results", "bb_middle", "REAL")
        _add_column_if_missing(connection, "screening_results", "bb_lower", "REAL")
        _add_column_if_missing(connection, "screening_results", "atr", "REAL")
        _add_relative_strength_columns(connection, "screening_results")
        _add_column_if_missing(connection, "pending_orders", "config_version", "TEXT")
        _add_column_if_missing(connection, "ai_decisions", "config_version", "TEXT")
        _add_column_if_missing(connection, "market_contexts", "sector_momentum", "TEXT")
    return db_path


def save_market_context(config: dict[str, Any], root: Path, market_context: dict[str, Any]) -> None:
    target_date = market_context.get("date")
    profile_id = market_context.get("profile_id") or _profile_id(config)
    profile_name = market_context.get("profile_name") or _profile_name(config)
    with _connect(config, root) as connection:
        connection.execute("DELETE FROM market_contexts WHERE date = ? AND profile_id = ?", (target_date, profile_id))
        connection.execute(
            """
            INSERT INTO market_contexts (
                profile_id, profile_name, date, provider, market_regime, advancers, decliners,
                advance_ratio, average_change_rate, turnover_value_total,
                topix_change_rate, nikkei_change_rate, usd_jpy,
                us_market_summary, important_news, sector_momentum, market_comment, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                profile_name,
                target_date,
                market_context.get("provider"),
                market_context.get("market_regime"),
                market_context.get("advancers"),
                market_context.get("decliners"),
                market_context.get("advance_ratio"),
                market_context.get("average_change_rate"),
                market_context.get("turnover_value_total"),
                market_context.get("topix_change_rate"),
                market_context.get("nikkei_change_rate"),
                market_context.get("usd_jpy"),
                market_context.get("us_market_summary"),
                _json(market_context.get("important_news", [])),
                _json(market_context.get("sector_momentum", [])),
                market_context.get("market_comment"),
                market_context.get("created_at") or _now(),
            ),
        )


def save_pending_orders(config: dict[str, Any], root: Path, pending_orders: list[dict[str, Any]]) -> None:
    config_version = config_version_from(config)
    with _connect(config, root) as connection:
        connection.execute("DELETE FROM pending_orders")
        connection.executemany(
            """
            INSERT INTO pending_orders (
                order_id, action, code, name, created_date,
                scheduled_execution_date, intended_price, status, score,
                reason, config_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    order.get("order_id"),
                    order.get("action"),
                    order.get("code"),
                    order.get("name"),
                    order.get("created_date"),
                    order.get("scheduled_execution_date"),
                    order.get("intended_price"),
                    order.get("status"),
                    order.get("score"),
                    order.get("reason"),
                    order.get("config_version") or config_version,
                    _now(),
                )
                for order in pending_orders
            ],
        )


def save_safety_events(config: dict[str, Any], root: Path, event_date: str, events: list[dict[str, Any]]) -> None:
    with _connect(config, root) as connection:
        connection.execute("DELETE FROM safety_events WHERE date = ?", (event_date,))
        if not events:
            return
        connection.executemany(
            """
            INSERT INTO safety_events (
                date, action, code, name, rejected, rejected_reason,
                safety_rule, broker_provider, order_status, live_trading,
                safety_checked, order_payload, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.get("date", event_date),
                    (item.get("order") or {}).get("action") or (item.get("order") or {}).get("side"),
                    (item.get("order") or {}).get("code"),
                    (item.get("order") or {}).get("name"),
                    1 if item.get("rejected") else 0,
                    item.get("rejected_reason"),
                    item.get("safety_rule"),
                    (item.get("order") or {}).get("broker_provider"),
                    (item.get("order") or {}).get("order_status") or (item.get("order") or {}).get("status"),
                    1 if (item.get("order") or {}).get("live_trading") else 0,
                    1 if (item.get("order") or {}).get("safety_checked") else 0,
                    _json(item.get("order", {})),
                    _now(),
                )
                for item in events
            ],
        )


def save_portfolio_snapshot(config: dict[str, Any], root: Path, snapshot: dict[str, Any]) -> None:
    config_version = snapshot.get("config_version") or config_version_from(config)
    profile_id = snapshot.get("profile_id") or _profile_id(config)
    profile_name = snapshot.get("profile_name") or _profile_name(config)
    with _connect(config, root) as connection:
        connection.execute("DELETE FROM portfolio_snapshots WHERE date = ? AND profile_id = ?", (snapshot.get("date"), profile_id))
        connection.execute(
            """
            INSERT INTO portfolio_snapshots (
                profile_id, profile_name, date, cash, positions_value, total_assets, daily_profit,
                cumulative_profit, cumulative_profit_rate, win_rate, max_drawdown,
                open_positions_count, closed_trades_count, gross_cumulative_profit,
                net_cumulative_profit, total_commission, estimated_tax_total,
                net_total_assets, dealer_comment, config_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                profile_name,
                snapshot.get("date"),
                snapshot.get("cash"),
                snapshot.get("positions_value"),
                snapshot.get("total_assets"),
                snapshot.get("daily_profit"),
                snapshot.get("cumulative_profit"),
                snapshot.get("cumulative_profit_rate"),
                snapshot.get("win_rate"),
                snapshot.get("max_drawdown"),
                snapshot.get("open_positions_count"),
                snapshot.get("closed_trades_count"),
                snapshot.get("gross_cumulative_profit", snapshot.get("cumulative_profit", 0)),
                snapshot.get("net_cumulative_profit", snapshot.get("cumulative_profit", 0)),
                snapshot.get("total_commission", 0),
                snapshot.get("estimated_tax_total", 0),
                snapshot.get("net_total_assets", snapshot.get("total_assets")),
                snapshot.get("dealer_comment"),
                config_version,
                _now(),
            ),
        )


def save_trades(config: dict[str, Any], root: Path, trade_date: str, trades: list[dict[str, Any]]) -> None:
    default_config_version = config_version_from(config)
    profile_id = _profile_id(config)
    profile_name = _profile_name(config)
    with _connect(config, root) as connection:
        connection.execute("DELETE FROM trades WHERE profile_id = ? AND (entry_date = ? OR exit_date = ?)", (profile_id, trade_date, trade_date))
        rows = []
        for trade in trades:
            if not is_filled_trade(trade):
                continue
            rows.append(
                (
                    trade.get("trade_id"),
                    trade.get("profile_id") or profile_id,
                    trade.get("profile_name") or profile_name,
                    trade.get("action"),
                    trade.get("code"),
                    trade.get("name"),
                    trade.get("sector_name"),
                    trade.get("entry_date") or trade.get("date"),
                    trade.get("exit_date"),
                    trade.get("holding_days"),
                    trade.get("entry_price"),
                    trade.get("exit_price"),
                    trade.get("shares"),
                    trade.get("amount") or trade.get("notional"),
                    trade.get("profit"),
                    trade.get("profit_rate"),
                    trade.get("exit_reason"),
                    trade.get("result"),
                    trade.get("score"),
                    trade.get("rsi"),
                    trade.get("volume_ratio"),
                    trade.get("stock_return_5d"),
                    trade.get("stock_return_10d"),
                    trade.get("stock_return_20d"),
                    trade.get("benchmark_source"),
                    trade.get("benchmark_return_5d"),
                    trade.get("benchmark_return_10d"),
                    trade.get("benchmark_return_20d"),
                    trade.get("relative_strength_5d"),
                    trade.get("relative_strength_10d"),
                    trade.get("relative_strength_20d"),
                    trade.get("relative_strength_score"),
                    trade.get("topix_records_loaded"),
                    trade.get("topix_api_calls"),
                    trade.get("investor_context_source"),
                    trade.get("investor_context_week"),
                    trade.get("overseas_net_buy"),
                    trade.get("overseas_net_buy_4w_sum"),
                    trade.get("overseas_net_buy_4w_trend"),
                    trade.get("overseas_buy_sell_ratio"),
                    trade.get("individual_net_buy"),
                    trade.get("institution_net_buy"),
                    trade.get("trust_bank_net_buy"),
                    trade.get("proprietary_net_buy"),
                    trade.get("investor_context_score"),
                    trade.get("total_score", trade.get("score")),
                    trade.get("technical_score"),
                    trade.get("ma_score") or trade.get("trend_score"),
                    trade.get("rsi_score"),
                    trade.get("volume_score"),
                    trade.get("candlestick_score"),
                    trade.get("market_context_score"),
                    trade.get("sector_score") or trade.get("sector_score_adjustment"),
                    trade.get("penalty_score"),
                    _json(trade.get("score_components", {})),
                    trade.get("score_components_total"),
                    1 if trade.get("score_components_match") else 0 if trade.get("score_components_match") is not None else None,
                    trade.get("market_regime"),
                    trade.get("advance_ratio"),
                    _json(trade.get("candlestick_signals", [])),
                    1 if trade.get("earnings_filter_checked") else 0,
                    1 if trade.get("earnings_filter_blocked") else 0,
                    trade.get("earnings_filter_reason"),
                    trade.get("earnings_announcement_date"),
                    trade.get("selected_reason") or trade.get("reason") or trade.get("buy_reason"),
                    trade.get("reason") or trade.get("buy_reason"),
                    trade.get("round_lot_size"),
                    1 if trade.get("use_round_lot") else 0,
                    trade.get("skipped_reason"),
                    trade.get("intended_price"),
                    trade.get("executed_price"),
                    trade.get("slippage_amount"),
                    trade.get("slippage_rate"),
                    trade.get("stop_loss_rate"),
                    trade.get("stop_loss_trigger_price"),
                    trade.get("stop_loss_triggered_date"),
                    trade.get("intended_exit_price"),
                    trade.get("actual_exit_price"),
                    trade.get("gap_slippage_rate"),
                    trade.get("stop_loss_slippage_rate"),
                    trade.get("gross_profit"),
                    trade.get("gross_profit_rate"),
                    trade.get("buy_commission"),
                    trade.get("sell_commission"),
                    trade.get("total_commission"),
                    trade.get("taxable_profit"),
                    trade.get("estimated_tax"),
                    trade.get("net_profit"),
                    trade.get("net_profit_rate"),
                    trade.get("dealer_comment"),
                    trade.get("broker_provider"),
                    trade.get("order_status") or trade.get("status"),
                    1 if trade.get("live_trading") else 0,
                    1 if trade.get("safety_checked") else 0,
                    trade.get("config_version") or default_config_version,
                    _now(),
                )
            )
        connection.executemany(
            """
            INSERT INTO trades (
                trade_id, profile_id, profile_name, action, code, name, sector_name, entry_date, exit_date, holding_days,
                entry_price, exit_price, shares, amount, profit, profit_rate,
                exit_reason, result, score, rsi, volume_ratio,
                stock_return_5d, stock_return_10d, stock_return_20d,
                benchmark_source, benchmark_return_5d, benchmark_return_10d, benchmark_return_20d,
                relative_strength_5d, relative_strength_10d, relative_strength_20d,
                relative_strength_score, topix_records_loaded, topix_api_calls,
                investor_context_source, investor_context_week, overseas_net_buy,
                overseas_net_buy_4w_sum, overseas_net_buy_4w_trend, overseas_buy_sell_ratio,
                individual_net_buy, institution_net_buy, trust_bank_net_buy,
                proprietary_net_buy, investor_context_score,
                total_score,
                technical_score, ma_score, rsi_score,
                volume_score, candlestick_score, market_context_score, sector_score,
                penalty_score, score_components, score_components_total,
                score_components_match, market_regime,
                advance_ratio, candlestick_signals,
                earnings_filter_checked, earnings_filter_blocked,
                earnings_filter_reason, earnings_announcement_date,
                selected_reason, reason, round_lot_size,
                use_round_lot, skipped_reason, intended_price, executed_price,
                slippage_amount, slippage_rate, stop_loss_rate,
                stop_loss_trigger_price, stop_loss_triggered_date,
                intended_exit_price, actual_exit_price, gap_slippage_rate,
                stop_loss_slippage_rate, gross_profit, gross_profit_rate,
                buy_commission, sell_commission, total_commission, taxable_profit,
                estimated_tax, net_profit, net_profit_rate, dealer_comment,
                broker_provider, order_status, live_trading, safety_checked,
                config_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def save_scoring_results(config: dict[str, Any], root: Path, scoring_log: dict[str, Any]) -> None:
    target_date = scoring_log.get("date")
    config_version = scoring_log.get("config_version") or config_version_from(config)
    profile_id = scoring_log.get("profile_id") or _profile_id(config)
    profile_name = scoring_log.get("profile_name") or _profile_name(config)
    scores = scoring_log.get("scores", [])
    if not bool(config.get("analysis", {}).get("save_rejected_candidates", True)):
        scores = [item for item in scores if item.get("selected")]
    with _connect(config, root) as connection:
        connection.execute("DELETE FROM scoring_results WHERE date = ? AND profile_id = ?", (target_date, profile_id))
        connection.executemany(
            """
            INSERT INTO scoring_results (
                profile_id, profile_name, date, code, name, sector_name, sector_momentum_score,
                sector_rank, sector_comment, sector_score_adjustment,
                rank, total_score, technical_score,
                confidence, selected, reason, rejected_reason,
                fallback, macd, macd_signal, macd_hist, bb_upper, bb_middle,
                bb_lower, atr,
                stock_return_5d, stock_return_10d, stock_return_20d,
                benchmark_source, benchmark_return_5d, benchmark_return_10d, benchmark_return_20d,
                relative_strength_5d, relative_strength_10d, relative_strength_20d,
                relative_strength_score, topix_records_loaded, topix_api_calls,
                investor_context_source, investor_context_week, overseas_net_buy,
                overseas_net_buy_4w_sum, overseas_net_buy_4w_trend, overseas_buy_sell_ratio,
                individual_net_buy, institution_net_buy, trust_bank_net_buy,
                proprietary_net_buy, investor_context_score,
                candle_type, candle_body_rate, upper_shadow_rate,
                lower_shadow_rate, close_position_in_range, gap_rate,
                candlestick_signals, candlestick_score, trend_score, volume_score,
                rsi_score, ma_score, market_context_score, sector_score,
                penalty_score, score_components, score_components_total,
                score_components_match, market_filter_applied, market_regime, market_filter_reason,
                earnings_filter_checked, earnings_filter_blocked, earnings_filter_reason, earnings_announcement_date,
                source_provider, ai_reason, ai_risk, ai_confidence,
                ai_score, config_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    profile_id,
                    profile_name,
                    target_date,
                    item.get("code"),
                    item.get("name"),
                    item.get("sector_name"),
                    item.get("sector_momentum_score"),
                    item.get("sector_rank"),
                    item.get("sector_comment"),
                    item.get("sector_score_adjustment"),
                    item.get("rank"),
                    item.get("total_score"),
                    item.get("technical_score"),
                    item.get("confidence"),
                    1 if item.get("selected") else 0,
                    item.get("selection_reason") or item.get("selected_reason") or item.get("reason"),
                    item.get("rejected_reason") or item.get("rejection_reason"),
                    1 if item.get("fallback") else 0,
                    item.get("macd"),
                    item.get("macd_signal"),
                    item.get("macd_hist"),
                    item.get("bb_upper"),
                    item.get("bb_middle"),
                    item.get("bb_lower"),
                    item.get("atr"),
                    item.get("stock_return_5d"),
                    item.get("stock_return_10d"),
                    item.get("stock_return_20d"),
                    item.get("benchmark_source"),
                    item.get("benchmark_return_5d"),
                    item.get("benchmark_return_10d"),
                    item.get("benchmark_return_20d"),
                    item.get("relative_strength_5d"),
                    item.get("relative_strength_10d"),
                    item.get("relative_strength_20d"),
                    item.get("relative_strength_score"),
                    item.get("topix_records_loaded"),
                    item.get("topix_api_calls"),
                    item.get("investor_context_source"),
                    item.get("investor_context_week"),
                    item.get("overseas_net_buy"),
                    item.get("overseas_net_buy_4w_sum"),
                    item.get("overseas_net_buy_4w_trend"),
                    item.get("overseas_buy_sell_ratio"),
                    item.get("individual_net_buy"),
                    item.get("institution_net_buy"),
                    item.get("trust_bank_net_buy"),
                    item.get("proprietary_net_buy"),
                    item.get("investor_context_score"),
                    item.get("candle_type"),
                    item.get("candle_body_rate"),
                    item.get("upper_shadow_rate"),
                    item.get("lower_shadow_rate"),
                    item.get("close_position_in_range"),
                    item.get("gap_rate"),
                    _json(item.get("candlestick_signals", [])),
                    item.get("candlestick_score"),
                    item.get("trend_score"),
                    item.get("volume_score"),
                    item.get("rsi_score"),
                    item.get("ma_score") or item.get("trend_score"),
                    item.get("market_context_score"),
                    item.get("sector_score") or item.get("sector_score_adjustment"),
                    item.get("penalty_score"),
                    _json(item.get("score_components", {})),
                    item.get("score_components_total"),
                    1 if item.get("score_components_match") else 0 if item.get("score_components_match") is not None else None,
                    1 if item.get("market_filter_applied") else 0,
                    item.get("market_regime"),
                    item.get("market_filter_reason"),
                    1 if item.get("earnings_filter_checked") else 0,
                    1 if item.get("earnings_filter_blocked") else 0,
                    item.get("earnings_filter_reason"),
                    item.get("earnings_announcement_date"),
                    item.get("source_provider") or scoring_log.get("source_provider"),
                    item.get("ai_reason"),
                    item.get("ai_risk"),
                    item.get("ai_confidence"),
                    item.get("ai_score"),
                    item.get("config_version") or config_version,
                    _now(),
                )
                for item in scores
            ],
        )


def save_ai_decision(config: dict[str, Any], root: Path, ai_decision_log: dict[str, Any]) -> None:
    target_date = ai_decision_log.get("date")
    profile_id = ai_decision_log.get("profile_id") or _profile_id(config)
    profile_name = ai_decision_log.get("profile_name") or _profile_name(config)
    with _connect(config, root) as connection:
        connection.execute("DELETE FROM ai_decisions WHERE date = ? AND profile_id = ?", (target_date, profile_id))
        connection.execute(
            """
            INSERT INTO ai_decisions (
                profile_id, profile_name, date, config_version, provider, model, candidates_count,
                selected_count, decision_summary, rookie_comment, fallback_used,
                token_input, token_output, estimated_cost, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                profile_name,
                target_date,
                ai_decision_log.get("config_version"),
                ai_decision_log.get("provider"),
                ai_decision_log.get("model"),
                ai_decision_log.get("candidates_count"),
                ai_decision_log.get("selected_count"),
                ai_decision_log.get("decision_summary"),
                ai_decision_log.get("rookie_comment"),
                1 if ai_decision_log.get("fallback_used") else 0,
                ai_decision_log.get("token_input"),
                ai_decision_log.get("token_output"),
                ai_decision_log.get("estimated_cost"),
                ai_decision_log.get("created_at") or _now(),
            ),
        )


def save_screening_results(config: dict[str, Any], root: Path, screening_log: dict[str, Any]) -> None:
    target_date = screening_log.get("date")
    config_version = screening_log.get("config_version") or config_version_from(config)
    profile_id = screening_log.get("profile_id") or _profile_id(config)
    profile_name = screening_log.get("profile_name") or _profile_name(config)
    with _connect(config, root) as connection:
        connection.execute("DELETE FROM screening_results WHERE date = ? AND profile_id = ?", (target_date, profile_id))
        connection.executemany(
            """
            INSERT INTO screening_results (
                profile_id, profile_name, date, code, name, sector_name, sector_momentum_score,
                sector_rank, sector_comment, close, volume, ma5, ma25, rsi, volume_ratio,
                turnover_value, five_day_volatility,
                stock_return_5d, stock_return_10d, stock_return_20d,
                benchmark_source, benchmark_return_5d, benchmark_return_10d, benchmark_return_20d,
                relative_strength_5d, relative_strength_10d, relative_strength_20d,
                relative_strength_score,
                macd, macd_signal,
                macd_hist, bb_upper, bb_middle, bb_lower, atr, candle_type,
                candle_body_rate, upper_shadow_rate, lower_shadow_rate,
                close_position_in_range, gap_rate, candlestick_signals,
                candlestick_score, trend_score, volume_score, rsi_score,
                fallback, pass_reason, rejected_reason, config_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    profile_id,
                    profile_name,
                    target_date,
                    item.get("code"),
                    item.get("name"),
                    item.get("sector_name"),
                    item.get("sector_momentum_score"),
                    item.get("sector_rank"),
                    item.get("sector_comment"),
                    item.get("close"),
                    item.get("volume"),
                    item.get("ma5"),
                    item.get("ma25"),
                    item.get("rsi"),
                    item.get("volume_ratio"),
                    item.get("turnover_value"),
                    item.get("five_day_volatility"),
                    item.get("stock_return_5d"),
                    item.get("stock_return_10d"),
                    item.get("stock_return_20d"),
                    item.get("benchmark_source"),
                    item.get("benchmark_return_5d"),
                    item.get("benchmark_return_10d"),
                    item.get("benchmark_return_20d"),
                    item.get("relative_strength_5d"),
                    item.get("relative_strength_10d"),
                    item.get("relative_strength_20d"),
                    item.get("relative_strength_score"),
                    item.get("macd"),
                    item.get("macd_signal"),
                    item.get("macd_hist"),
                    item.get("bb_upper"),
                    item.get("bb_middle"),
                    item.get("bb_lower"),
                    item.get("atr"),
                    item.get("candle_type"),
                    item.get("candle_body_rate"),
                    item.get("upper_shadow_rate"),
                    item.get("lower_shadow_rate"),
                    item.get("close_position_in_range"),
                    item.get("gap_rate"),
                    _json(item.get("candlestick_signals", [])),
                    item.get("candlestick_score"),
                    item.get("trend_score"),
                    item.get("volume_score"),
                    item.get("rsi_score"),
                    1 if item.get("fallback") else 0,
                    item.get("pass_reason") or item.get("reason"),
                    item.get("rejected_reason"),
                    item.get("config_version") or config_version,
                    _now(),
                )
                for item in screening_log.get("candidates", [])
            ],
        )


def save_reflections(config: dict[str, Any], root: Path, reflection_log: dict[str, Any], trades: list[dict[str, Any]] | None = None) -> None:
    profit_by_trade_id = {trade.get("trade_id"): trade for trade in trades or []}
    config_version = reflection_log.get("config_version") or config_version_from(config)
    profile_id = reflection_log.get("profile_id") or _profile_id(config)
    profile_name = reflection_log.get("profile_name") or _profile_name(config)
    with _connect(config, root) as connection:
        for item in reflection_log.get("reflections", []):
            trade = profit_by_trade_id.get(item.get("trade_id"), {})
            connection.execute("DELETE FROM reflections WHERE trade_id = ? AND profile_id = ?", (item.get("trade_id"), profile_id))
            connection.execute(
                """
                INSERT INTO reflections (
                    profile_id, profile_name, trade_id, code, name, result, profit, profit_rate, summary,
                    good_points, bad_points, lesson, suggestions,
                    reflection_comment, config_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    profile_name,
                    item.get("trade_id"),
                    item.get("code"),
                    item.get("name"),
                    item.get("result"),
                    trade.get("profit"),
                    item.get("profit_rate"),
                    _reflection_summary(item),
                    _json(item.get("good_points", [])),
                    _json(item.get("bad_points", [])),
                    item.get("lesson_for_next_trade"),
                    _json(item.get("suggestions", [])),
                    item.get("reflection_comment"),
                    item.get("config_version") or trade.get("config_version") or config_version,
                    _now(),
                ),
            )


def save_article(config: dict[str, Any], root: Path, article: dict[str, Any]) -> None:
    path = str(article.get("path") or "")
    now = _now()
    config_version = article.get("config_version") or config_version_from(config)
    profile_id = article.get("profile_id") or _profile_id(config)
    profile_name = article.get("profile_name") or _profile_name(config)
    with _connect(config, root) as connection:
        existing = connection.execute("SELECT created_at FROM articles WHERE path = ?", (path,)).fetchone()
        created_at = existing[0] if existing else now
        connection.execute("DELETE FROM articles WHERE path = ?", (path,))
        connection.execute(
            """
            INSERT INTO articles (
                profile_id, profile_name, date, day, title, status, path, note_url, body,
                published_at, config_version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                profile_name,
                article.get("date"),
                article.get("day"),
                article.get("title"),
                article.get("status", "draft"),
                path,
                article.get("note_url"),
                article.get("body"),
                article.get("published_at"),
                config_version,
                created_at,
                now,
            ),
        )


def analyze_operation_data(config: dict[str, Any], root: Path) -> dict[str, Any]:
    db_path = get_database_path(config, root)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    profile_id = _profile_id(config)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        _delete_non_trade_order_rows(connection)
        connection.commit()
        portfolio_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM portfolio_snapshots WHERE profile_id = ? ORDER BY date, id",
                (profile_id,),
            )
        ]
        trade_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM trades WHERE profile_id = ? ORDER BY entry_date, exit_date, id",
                (profile_id,),
            )
        ]
        scoring_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM scoring_results WHERE profile_id = ? ORDER BY date, rank, id",
                (profile_id,),
            )
        ]
        reflection_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM reflections WHERE profile_id = ? ORDER BY created_at, id",
                (profile_id,),
            )
        ]

    if not portfolio_rows and not trade_rows and not scoring_rows and not reflection_rows:
        raise ValueError("SQLite DB has no operation data.")

    prices_by_code = _load_price_history(root)
    return {
        "portfolio_analysis": _portfolio_analysis(config, portfolio_rows),
        "trade_analysis": _trade_analysis(config, trade_rows, prices_by_code),
        "score_analysis": _score_analysis(scoring_rows),
        "reflection_analysis": _reflection_analysis(reflection_rows),
        "config_version_analysis": _config_version_analysis(trade_rows),
        "sector_win_rate_analysis": _sector_win_rate_analysis(trade_rows),
        "profile_analysis": _profile_analysis(portfolio_rows, trade_rows),
        "yearly_performance": _yearly_performance_analysis(portfolio_rows, trade_rows),
        "monthly_performance": _monthly_performance_analysis(trade_rows),
        "walk_forward_validation": _walk_forward_validation(config, portfolio_rows, trade_rows),
        "market_regime_performance": _market_regime_performance_analysis(config, trade_rows),
        "current_profile_id": _profile_id(config),
        "current_profile_name": _profile_name(config),
        "current_config_version": config_version_from(config),
        "generated_at": _now(),
        "database_path": str(db_path),
    }


def _connect(config: dict[str, Any], root: Path) -> sqlite3.Connection:
    db_path = initialize_database(config, root)
    return sqlite3.connect(db_path)


def _add_column_if_missing(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _drop_removed_score_columns(connection: sqlite3.Connection) -> None:
    removed_columns = {
        "trades": [
            "news" + "_score",
            "financial" + "_score",
        ],
        "scoring_results": [
            "news" + "_score",
            "news" + "_reason",
            "news" + "_articles_count",
            "positive_" + "news_count",
            "negative_" + "news_count",
            "news_provider",
            "news" + "_limitation",
            "financial" + "_score",
        ],
    }
    for table, columns in removed_columns.items():
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        for column in columns:
            if column in existing:
                connection.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
                existing.remove(column)


def _add_relative_strength_columns(connection: sqlite3.Connection, table: str) -> None:
    _add_column_if_missing(connection, table, "benchmark_source", "TEXT")
    for column in [
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
        "topix_records_loaded",
        "topix_api_calls",
    ]:
        _add_column_if_missing(connection, table, column, "REAL")


def _add_investor_context_columns(connection: sqlite3.Connection, table: str) -> None:
    for column in ["investor_context_source", "investor_context_week", "overseas_net_buy_4w_trend"]:
        _add_column_if_missing(connection, table, column, "TEXT")
    for column in [
        "overseas_net_buy",
        "overseas_net_buy_4w_sum",
        "overseas_buy_sell_ratio",
        "individual_net_buy",
        "institution_net_buy",
        "trust_bank_net_buy",
        "proprietary_net_buy",
        "investor_context_score",
    ]:
        _add_column_if_missing(connection, table, column, "REAL")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _reflection_summary(item: dict[str, Any]) -> str:
    return (
        f"result={item.get('result')}, profit_rate={item.get('profit_rate')}, "
        f"sell_reason={item.get('sell_reason')}"
    )


def _portfolio_analysis(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    initial_capital = float(config["portfolio"]["initial_cash"])
    if not rows:
        return {
            "initial_capital": initial_capital,
            "latest_total_assets": None,
            "cumulative_profit": None,
            "cumulative_profit_rate": None,
            "max_drawdown": None,
            "operation_days": 0,
            "gross_cumulative_profit": None,
            "net_cumulative_profit": None,
            "total_commission": 0.0,
            "estimated_tax_total": 0.0,
            "net_total_assets": None,
            "cash": None,
            "positions_value": None,
            "open_positions_count": 0,
            "closed_trades_count": 0,
            "realized_profit": 0.0,
            "unrealized_profit": 0.0,
            "reconciliation_difference": None,
            "reconciliation_ok": False,
        }

    latest = rows[-1]
    max_drawdown = min((float(row.get("max_drawdown") or 0) for row in rows), default=0.0)
    latest_assets = float(latest.get("total_assets") or initial_capital)
    cumulative_profit = latest_assets - initial_capital
    cash = float(latest.get("cash") or 0.0)
    positions_value = float(latest.get("positions_value") or 0.0)
    gross_cumulative_profit = latest.get("gross_cumulative_profit")
    total_commission = float(latest.get("total_commission") or 0.0)
    if gross_cumulative_profit is not None:
        gross_cumulative_profit = round(float(gross_cumulative_profit), 2)
        estimated_tax_total = calculate_period_estimated_tax(gross_cumulative_profit, total_commission, config)
        net_cumulative_profit = round(gross_cumulative_profit - estimated_tax_total - total_commission, 2)
        net_total_assets = round(initial_capital + net_cumulative_profit, 2)
    else:
        estimated_tax_total = float(latest.get("estimated_tax_total") or 0.0)
        net_cumulative_profit = latest.get("net_cumulative_profit")
        net_total_assets = latest.get("net_total_assets")
    realized_profit = float(net_cumulative_profit or 0.0)
    unrealized_profit = round(cumulative_profit - realized_profit, 2)
    reconciled_assets = round(initial_capital + realized_profit + unrealized_profit, 2)
    reconciliation_difference = round(reconciled_assets - latest_assets, 2)
    return {
        "initial_capital": initial_capital,
        "latest_total_assets": latest_assets,
        "cash": cash,
        "positions_value": positions_value,
        "cumulative_profit": round(cumulative_profit, 2),
        "cumulative_profit_rate": round(cumulative_profit / initial_capital, 4),
        "max_drawdown": max_drawdown,
        "operation_days": len({row["date"] for row in rows if row.get("date")}),
        "gross_cumulative_profit": gross_cumulative_profit,
        "net_cumulative_profit": net_cumulative_profit,
        "total_commission": total_commission,
        "estimated_tax_total": estimated_tax_total,
        "net_total_assets": net_total_assets,
        "open_positions_count": latest.get("open_positions_count") or 0,
        "closed_trades_count": latest.get("closed_trades_count") or 0,
        "realized_profit": round(realized_profit, 2),
        "unrealized_profit": unrealized_profit,
        "reconciled_assets": reconciled_assets,
        "reconciliation_difference": reconciliation_difference,
        "reconciliation_ok": abs(reconciliation_difference) < 0.01,
    }


def _trade_analysis(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    prices_by_code: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    filled_rows = [row for row in rows if is_filled_trade(row)]
    pf_metrics = profit_factor_metrics(rows)
    closed = pf_metrics["closed_trades"]
    wins = pf_metrics["wins"]
    losses = pf_metrics["losses"]
    gross_profits = [float(row.get("gross_profit") or row.get("profit") or 0) for row in closed]
    realized_profit_total = pf_metrics["realized_profit_total"]
    gross_profit_total = pf_metrics["gross_profit_total"]
    gross_win_total = pf_metrics["gross_win_total"]
    gross_loss_total = pf_metrics["gross_loss_total"]
    profit_rates = [float(row["profit_rate"]) for row in closed if row.get("profit_rate") is not None]
    win_rates = [float(row["profit_rate"]) for row in wins if row.get("profit_rate") is not None]
    loss_rates = [float(row["profit_rate"]) for row in losses if row.get("profit_rate") is not None]
    stop_loss_rate = float(config.get("risk", {}).get("stop_loss_pct", -0.03))
    stop_loss_rows = [row for row in closed if row.get("exit_reason") == "損切り"]
    stop_loss_slippages = [
        float(row["stop_loss_slippage_rate"]) for row in stop_loss_rows if row.get("stop_loss_slippage_rate") is not None
    ]
    loss_over_stop_rows = [row for row in losses if row.get("profit_rate") is not None and float(row["profit_rate"]) < stop_loss_rate]
    holding_days = [float(row["holding_days"]) for row in closed if row.get("holding_days") is not None]
    slippages = [float(row["slippage_rate"]) for row in filled_rows if row.get("slippage_rate") is not None]
    total_commission = round(sum(float(row.get("total_commission") or 0) for row in closed), 2)
    trade_estimated_tax_total = round(sum(float(row.get("estimated_tax") or 0) for row in closed), 2)
    estimated_tax_total = calculate_period_estimated_tax(realized_profit_total, total_commission, config)
    net_profit_total = round(realized_profit_total - estimated_tax_total - total_commission, 2)
    average_win_profit_rate = _average(win_rates)
    average_loss_profit_rate = _average(loss_rates)
    win_count = pf_metrics["win_count"]
    loss_count = pf_metrics["loss_count"]
    win_rate = pf_metrics["win_rate"]
    expectancy = None
    if win_rate is not None:
        expectancy = round((win_rate * (average_win_profit_rate or 0.0)) + ((1 - win_rate) * (average_loss_profit_rate or 0.0)), 4)
    profit_ratio = None
    if average_win_profit_rate is not None and average_loss_profit_rate not in (None, 0):
        profit_ratio = round(average_win_profit_rate / abs(average_loss_profit_rate), 4)
    largest_win = max((value for value in gross_profits if value > 0), default=None)
    largest_loss = min((value for value in gross_profits if value < 0), default=None)
    best_trade = max(closed, key=lambda row: float(row.get("gross_profit") or row.get("profit") or 0), default=None)
    worst_trade = min(closed, key=lambda row: float(row.get("gross_profit") or row.get("profit") or 0), default=None)
    return {
        "total_trades": pf_metrics["total_trades"],
        "closed_trades": pf_metrics["closed_trade_count"],
        "closed_trade_count": pf_metrics["closed_trade_count"],
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "excluded_order_event_count": pf_metrics["excluded_order_event_count"],
        "win_rate": win_rate,
        "average_profit_rate": average_win_profit_rate,
        "average_loss_rate": average_loss_profit_rate,
        "average_win_profit_rate": average_win_profit_rate,
        "average_loss_profit_rate": average_loss_profit_rate,
        "worst_loss_profit_rate": min(loss_rates) if loss_rates else None,
        "average_all_profit_rate": _average(profit_rates),
        "expectancy": expectancy,
        "profit_ratio": profit_ratio,
        "best_trade": _trade_summary(best_trade),
        "worst_trade": _trade_summary(worst_trade),
        "exit_reason_analysis": _exit_reason_analysis(closed),
        "exit_efficiency": _exit_efficiency_analysis(closed),
        "holding_period_analysis": _holding_period_analysis(closed),
        "holding_period_optimization": _holding_period_optimization(config, closed),
        "candidate_exit_improvements": _candidate_exit_improvements(config, closed),
        "trade_replay_analysis": _trade_replay_analysis(closed, prices_by_code or {}),
        "stop_loss_recovery_analysis": _stop_loss_recovery_analysis(closed, prices_by_code or {}),
        "sold_before_take_profit_rate": _sold_before_take_profit_rate(closed),
        "average_holding_days": _average(holding_days),
        "take_profit_count": sum(1 for row in closed if row.get("exit_reason") == "利確"),
        "stop_loss_count": sum(1 for row in closed if row.get("exit_reason") == "損切り"),
        "stop_loss_slippage_average": _average(stop_loss_slippages),
        "stop_loss_slippage_max": max(stop_loss_slippages, key=abs) if stop_loss_slippages else None,
        "loss_over_stop_count": len(loss_over_stop_rows),
        "loss_over_stop_rate": round(len(loss_over_stop_rows) / loss_count, 4) if loss_count else None,
        "max_holding_exit_count": sum(1 for row in closed if row.get("exit_reason") == "最大保有期間到達"),
        "total_commission": total_commission,
        "estimated_tax_total": estimated_tax_total,
        "trade_estimated_tax_total": trade_estimated_tax_total,
        "gross_profit_total": gross_profit_total,
        "gross_win_total": gross_win_total,
        "gross_loss_total": gross_loss_total,
        "realized_profit_total": realized_profit_total,
        "profit_factor": pf_metrics["profit_factor"],
        "largest_win": round(largest_win, 2) if largest_win is not None else None,
        "largest_loss": round(largest_loss, 2) if largest_loss is not None else None,
        "net_profit_total": net_profit_total,
        "average_slippage": _average(slippages),
        "max_slippage": max(slippages, key=abs) if slippages else None,
        "gap_up_count": sum(1 for row in filled_rows if row.get("slippage_rate") is not None and float(row["slippage_rate"]) > 0),
        "gap_down_count": sum(1 for row in filled_rows if row.get("slippage_rate") is not None and float(row["slippage_rate"]) < 0),
    }


def _exit_reason_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary_reasons = ["利確", "損切り", "最大保有期間到達", "その他"]
    summaries = []
    for reason in primary_reasons:
        if reason == "その他":
            reason_rows = [row for row in rows if _exit_reason_group(row) == "その他"]
        else:
            reason_rows = [row for row in rows if _exit_reason_group(row) == reason]
        profit_rates = [float(row["profit_rate"]) for row in reason_rows if row.get("profit_rate") is not None]
        profits = [_trade_profit(row) for row in reason_rows]
        profits = [profit for profit in profits if profit is not None]
        holding_days = [float(row["holding_days"]) for row in reason_rows if row.get("holding_days") is not None]
        summaries.append(
            {
                "exit_reason": reason,
                "count": len(reason_rows),
                "win_rate": _win_rate(reason_rows),
                "avg_profit": _average(profits),
                "avg_profit_rate": _average(profit_rates),
                "average_profit_rate": _average(profit_rates),
                "total_profit": round(sum(profits), 2) if profits else 0.0,
                "avg_holding_days": _average(holding_days),
            }
        )
    return summaries


def _exit_reason_group(row: dict[str, Any]) -> str:
    reason = str(row.get("exit_reason") or "その他")
    if reason in {"利確", "損切り", "最大保有期間到達"}:
        return reason
    return "その他"


def _exit_efficiency_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_holding_rows = [row for row in rows if _exit_reason_group(row) == "最大保有期間到達"]
    return {
        "take_profit_count": sum(1 for row in rows if _exit_reason_group(row) == "利確"),
        "stop_loss_count": sum(1 for row in rows if _exit_reason_group(row) == "損切り"),
        "max_holding_count": len(max_holding_rows),
        "max_holding_profit_count": sum(1 for row in max_holding_rows if _profit_rate(row) is not None and float(_profit_rate(row) or 0.0) > 0),
        "max_holding_loss_count": sum(1 for row in max_holding_rows if _profit_rate(row) is not None and float(_profit_rate(row) or 0.0) <= 0),
    }


def _holding_period_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("holding_days") is None:
            continue
        grouped.setdefault(int(row["holding_days"]), []).append(row)
    summaries = []
    for holding_days, period_rows in sorted(grouped.items()):
        profit_rates = [float(row["profit_rate"]) for row in period_rows if row.get("profit_rate") is not None]
        profits = [_trade_profit(row) for row in period_rows]
        profits = [profit for profit in profits if profit is not None]
        summaries.append(
            {
                "holding_days": holding_days,
                "count": len(period_rows),
                "win_rate": _win_rate(period_rows),
                "avg_profit_rate": _average(profit_rates),
                "total_profit": round(sum(profits), 2) if profits else 0.0,
            }
        )
    return summaries


def _holding_period_optimization(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    current_max_holding_days = int(config.get("risk", {}).get("max_holding_business_days", 5))
    current_rows = [row for row in rows if _trade_profit(row) is not None]
    current_profit = round(sum(float(_trade_profit(row) or 0.0) for row in current_rows), 2)
    simulations = [
        _simulate_max_holding_days(config, rows, max_holding_days)
        for max_holding_days in HOLDING_PERIOD_SIMULATION_DAYS
    ]
    ranked = sorted(
        simulations,
        key=lambda item: (
            float(item.get("estimated_profit") or 0.0),
            float(item.get("estimated_profit_factor") or 0.0),
            float(item.get("estimated_win_rate") or 0.0),
            float(item.get("estimated_drawdown") or -1.0),
        ),
        reverse=True,
    )
    return {
        "current_max_holding_days": current_max_holding_days,
        "current_profit": current_profit,
        "holding_period_analysis": _holding_period_analysis(rows),
        "estimated_profit_ranking": ranked,
        "calculation_details": _holding_period_calculation_details(ranked, current_profit, len(current_rows)),
        "candidate_holding_days": _candidate_holding_days(ranked, current_profit),
    }


def _simulate_max_holding_days(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    max_holding_days: int,
) -> dict[str, Any]:
    simulated_rows = [
        row for row in rows
        if row.get("holding_days") is not None
        and int(row["holding_days"]) <= max_holding_days
        and _trade_profit(row) is not None
    ]
    simulated_rows = sorted(simulated_rows, key=_trade_timeline_key)
    profits = [_trade_profit(row) for row in simulated_rows]
    profits = [profit for profit in profits if profit is not None]
    wins = [profit for profit in profits if profit > 0]
    losses = [profit for profit in profits if profit < 0]
    gross_win_total = round(sum(wins), 2)
    gross_loss_total = round(sum(losses), 2)
    profit_factor = round(gross_win_total / abs(gross_loss_total), 4) if gross_loss_total < 0 else None
    return {
        "max_holding_days": max_holding_days,
        "sample_count": len(simulated_rows),
        "estimated_profit": round(sum(profits), 2) if profits else 0.0,
        "estimated_profit_factor": profit_factor,
        "estimated_win_rate": round(len(wins) / len(profits), 4) if profits else None,
        "estimated_drawdown": _estimated_drawdown(profits, _initial_capital(config)),
    }


def _holding_period_calculation_details(
    ranked: list[dict[str, Any]],
    current_profit: float,
    base_trade_count: int,
) -> dict[str, Any]:
    if not ranked:
        return {
            "current_profit_formula": "sum(profit for all closed trades)",
            "simulated_profit_formula": "sum(profit for closed trades with holding_days <= max_holding_days)",
            "lift_vs_current_formula": "simulated_profit - current_profit",
            "current_profit": current_profit,
            "simulated_profit": None,
            "profit_difference": None,
            "lift_vs_current": None,
            "base_trade_count": base_trade_count,
            "simulated_trade_count": 0,
        }
    best = ranked[0]
    simulated_profit = float(best.get("estimated_profit") or 0.0)
    profit_difference = round(simulated_profit - current_profit, 2)
    return {
        "current_profit_formula": "sum(profit for all closed trades)",
        "simulated_profit_formula": "sum(profit for closed trades with holding_days <= max_holding_days)",
        "lift_vs_current_formula": "simulated_profit - current_profit",
        "current_profit": current_profit,
        "simulated_profit": round(simulated_profit, 2),
        "profit_difference": profit_difference,
        "lift_vs_current": profit_difference,
        "base_trade_count": base_trade_count,
        "simulated_trade_count": best.get("sample_count", 0),
    }


def _candidate_holding_days(ranked: list[dict[str, Any]], current_profit: float) -> list[dict[str, Any]]:
    if not ranked:
        return []
    best = ranked[0]
    simulated_profit = float(best.get("estimated_profit") or 0.0)
    lift = round(simulated_profit - current_profit, 2)
    if simulated_profit <= current_profit:
        return []
    reason = (
        f"max_holding_days={best.get('max_holding_days')} が推定利益 "
        f"{simulated_profit:,.0f}円で、実績利益 {current_profit:,.0f}円を上回ります。"
    )
    return [
        {
            "recommended_max_holding_days": best.get("max_holding_days"),
            "reason": reason,
            "estimated_profit": best.get("estimated_profit"),
            "estimated_profit_lift_vs_current": lift,
            "estimated_profit_factor": best.get("estimated_profit_factor"),
            "estimated_win_rate": best.get("estimated_win_rate"),
            "estimated_drawdown": best.get("estimated_drawdown"),
        }
    ]


def _estimated_drawdown(profits: list[float], initial_capital: float) -> float | None:
    if not profits or initial_capital <= 0:
        return None
    equity = initial_capital
    peak = initial_capital
    max_drawdown = 0.0
    for profit in profits:
        equity += profit
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak if peak else 0.0
        max_drawdown = min(max_drawdown, drawdown)
    return round(max_drawdown, 4)


def _initial_capital(config: dict[str, Any]) -> float:
    return float(config.get("portfolio", {}).get("initial_cash") or config.get("initial_capital") or 1_000_000)


def _trade_timeline_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("exit_date") or row.get("entry_date") or row.get("date") or ""),
        str(row.get("trade_id") or row.get("code") or ""),
    )


def _load_price_history(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
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
                result.setdefault(code, []).append({"date": str(row_date), "close": close})
    for code in result:
        result[code] = sorted(result[code], key=lambda row: row.get("date") or "")
    return result


def _candidate_exit_improvements(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    risk = config.get("risk", {})
    take_profit_rate = float(risk.get("take_profit_pct", 0.06))
    stop_loss_rate = float(risk.get("stop_loss_pct", -0.03))
    max_holding_days = int(risk.get("max_holding_business_days", 5))
    efficiency = _exit_efficiency_analysis(rows)
    max_holding_rows = [row for row in rows if _exit_reason_group(row) == "最大保有期間到達"]
    stop_loss_rows = [row for row in rows if _exit_reason_group(row) == "損切り"]
    suggestions = []

    if efficiency["take_profit_count"] < efficiency["max_holding_profit_count"]:
        suggestions.append(
            {
                "suggestion": "take_profit_rate を下げる候補",
                "reason": f"利確件数 {efficiency['take_profit_count']}件に対して、最大保有期間到達の利益終了が {efficiency['max_holding_profit_count']}件あります。",
                "current_value": take_profit_rate,
            }
        )

    max_holding_profit_rates = [
        float(row["profit_rate"]) for row in max_holding_rows if row.get("profit_rate") is not None
    ]
    max_holding_avg = _average(max_holding_profit_rates)
    max_holding_win_rate = _win_rate(max_holding_rows)
    if max_holding_avg is not None and max_holding_avg > 0 and (max_holding_win_rate or 0.0) >= 0.5:
        suggestions.append(
            {
                "suggestion": "max_holding_days を延ばす候補",
                "reason": f"最大保有期間到達の平均利益率が {max_holding_avg:.2%}、勝率が {(max_holding_win_rate or 0.0):.2%} です。",
                "current_value": max_holding_days,
            }
        )

    stop_loss_profit_rates = [
        float(row["profit_rate"]) for row in stop_loss_rows if row.get("profit_rate") is not None
    ]
    stop_loss_avg = _average(stop_loss_profit_rates)
    if stop_loss_rows and stop_loss_avg is not None and abs(stop_loss_avg - stop_loss_rate) <= 0.010001:
        suggestions.append(
            {
                "suggestion": "stop_loss は機能",
                "reason": f"損切り {len(stop_loss_rows)}件の平均損失率が {stop_loss_avg:.2%} で、設定値 {stop_loss_rate:.2%} 付近です。",
                "current_value": stop_loss_rate,
            }
        )

    if efficiency["max_holding_loss_count"] > efficiency["max_holding_profit_count"]:
        suggestions.append(
            {
                "suggestion": "time stop 強化",
                "reason": f"最大保有期間到達の損失終了が {efficiency['max_holding_loss_count']}件で、利益終了 {efficiency['max_holding_profit_count']}件を上回っています。",
                "current_value": max_holding_days,
            }
        )

    return suggestions or [
        {
            "suggestion": "現行出口ルールを継続検証",
            "reason": "利確、損切り、最大保有期間到達の偏りが小さく、明確な変更候補はまだ弱いです。",
            "current_value": None,
        }
    ]


def _trade_replay_analysis(
    rows: list[dict[str, Any]],
    prices_by_code: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if not prices_by_code:
        return {
            "top_profit_trades": [],
            "top_loss_trades": [],
            "winner_average_replay": _average_replay([]),
            "loser_average_replay": _average_replay([]),
        }
    ranked = [
        row for row in rows
        if _trade_profit(row) is not None
        and row.get("code")
        and row.get("entry_date")
    ]
    winners = sorted(
        [row for row in ranked if float(_trade_profit(row) or 0.0) > 0],
        key=lambda row: float(_trade_profit(row) or 0.0),
        reverse=True,
    )[:10]
    losers = sorted(
        [row for row in ranked if float(_trade_profit(row) or 0.0) < 0],
        key=lambda row: float(_trade_profit(row) or 0.0),
    )[:10]
    winner_replays = [_trade_replay_record(row, prices_by_code) for row in winners]
    loser_replays = [_trade_replay_record(row, prices_by_code) for row in losers]
    winner_replays = [item for item in winner_replays if item]
    loser_replays = [item for item in loser_replays if item]
    return {
        "top_profit_trades": winner_replays,
        "top_loss_trades": loser_replays,
        "winner_average_replay": _average_replay(winner_replays),
        "loser_average_replay": _average_replay(loser_replays),
    }


def _trade_replay_record(
    row: dict[str, Any],
    prices_by_code: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    code = str(row.get("code") or "")
    entry_date = str(row.get("entry_date") or "")
    prices = prices_by_code.get(code, [])
    entry_price = _trade_entry_price(row, prices)
    if not code or not entry_date or entry_price is None or entry_price == 0:
        return None
    day_returns = []
    for day in range(1, 11):
        close = _future_close(prices, entry_date, day)
        day_returns.append(
            {
                "day": day,
                "close": close,
                "return_rate": _close_return_rate(entry_price, close),
            }
        )
    return {
        "entry_date": entry_date,
        "code": code,
        "name": row.get("name"),
        "holding_days": row.get("holding_days"),
        "profit": _trade_profit(row),
        "profit_rate": _profit_rate(row),
        "entry_price": entry_price,
        "entry_return_rate": 0.0,
        "day_returns": day_returns,
    }


def _trade_entry_price(row: dict[str, Any], prices: list[dict[str, Any]]) -> float | None:
    entry_price = row.get("entry_price")
    if entry_price is not None:
        return float(entry_price)
    return _close_on_or_after(prices, str(row.get("entry_date") or ""))


def _close_on_or_after(prices: list[dict[str, Any]], target_date: str) -> float | None:
    for row in prices:
        if str(row.get("date") or "") >= target_date:
            return _number(row.get("close"))
    return None


def _future_close(prices: list[dict[str, Any]], entry_date: str, offset: int) -> float | None:
    future_prices = [row for row in prices if str(row.get("date") or "") > entry_date]
    if len(future_prices) < offset:
        return None
    return _number(future_prices[offset - 1].get("close"))


def _close_return_rate(entry_price: float, close: float | None) -> float | None:
    if close is None or entry_price == 0:
        return None
    return round((close - entry_price) / entry_price, 6)


def _average_replay(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [{"label": "entry", "return_rate": 0.0, "count": len(items)}]
    for day in range(1, 11):
        returns = []
        for item in items:
            day_item = next((entry for entry in item.get("day_returns", []) if entry.get("day") == day), None)
            if day_item and day_item.get("return_rate") is not None:
                returns.append(float(day_item["return_rate"]))
        result.append({"label": f"day{day}", "return_rate": _average(returns), "count": len(returns)})
    return result


def _stop_loss_recovery_analysis(
    rows: list[dict[str, Any]],
    prices_by_code: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    stop_loss_rows = [row for row in rows if _exit_reason_group(row) == "損切り"]
    records = [
        record for record in (
            _stop_loss_recovery_record(row, prices_by_code)
            for row in stop_loss_rows
        )
        if record
    ]
    winners = [record for record in records if record.get("day10_recovered")]
    losers = [record for record in records if not record.get("day10_recovered")]
    return {
        "stop_loss_count": len(stop_loss_rows),
        "replay_count": len(records),
        "day5_recovery_rate": _recovery_rate(records, "day5_recovered"),
        "day10_recovery_rate": _recovery_rate(records, "day10_recovered"),
        "recovery_winners": winners,
        "recovery_losers": losers,
        "recovery_signals": _recovery_signal_rows(winners, losers),
        "candidate_dynamic_stop_rules": _candidate_dynamic_stop_rules(winners, losers),
    }


def _stop_loss_recovery_record(
    row: dict[str, Any],
    prices_by_code: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    replay = _trade_replay_record(row, prices_by_code)
    if not replay:
        return None
    day1_return = _replay_day_return(replay, 1)
    day5_return = _replay_day_return(replay, 5)
    day10_return = _replay_day_return(replay, 10)
    return {
        "entry_date": replay.get("entry_date"),
        "code": replay.get("code"),
        "name": replay.get("name"),
        "holding_days": replay.get("holding_days"),
        "profit": replay.get("profit"),
        "profit_rate": replay.get("profit_rate"),
        "day1_return": day1_return,
        "day5_return": day5_return,
        "day10_return": day10_return,
        "day5_recovered": day5_return is not None and day5_return > 0,
        "day10_recovered": day10_return is not None and day10_return > 0,
        "rsi": row.get("rsi"),
        "volume_ratio": row.get("volume_ratio"),
        "market_regime": row.get("market_regime") or "unknown",
        "sector": row.get("sector_name") or "未分類",
        "candlestick_signals": _load_json_list(row.get("candlestick_signals")),
    }


def _replay_day_return(replay: dict[str, Any], day: int) -> float | None:
    day_item = next((item for item in replay.get("day_returns", []) if item.get("day") == day), None)
    if not day_item:
        return None
    return day_item.get("return_rate")


def _recovery_rate(records: list[dict[str, Any]], key: str) -> float | None:
    valid = [record for record in records if record.get(key) is not None]
    if not valid:
        return None
    return round(sum(1 for record in valid if record.get(key)) / len(valid), 4)


def _recovery_signal_rows(winners: list[dict[str, Any]], losers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winner_counts = _recovery_feature_counts(winners)
    loser_counts = _recovery_feature_counts(losers)
    rows = []
    for feature in sorted(set(winner_counts) | set(loser_counts)):
        winner_count = winner_counts.get(feature, 0)
        loser_count = loser_counts.get(feature, 0)
        winner_share = _share_count(winner_count, len(winners))
        loser_share = _share_count(loser_count, len(losers))
        rows.append(
            {
                "feature": feature[0],
                "value": feature[1],
                "winner_count": winner_count,
                "loser_count": loser_count,
                "winner_share": winner_share,
                "loser_share": loser_share,
                "share_difference": _difference_rates(winner_share, loser_share),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            abs(float(item.get("share_difference") or 0.0)),
            int(item.get("winner_count") or 0) + int(item.get("loser_count") or 0),
        ),
        reverse=True,
    )


def _recovery_feature_counts(records: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        for feature in set(_recovery_features(record)):
            counts[feature] = counts.get(feature, 0) + 1
    return counts


def _recovery_features(record: dict[str, Any]) -> list[tuple[str, str]]:
    signals = record.get("candlestick_signals") or ["no_signal"]
    features = [
        ("RSI", _rsi_bucket(record.get("rsi"))),
        ("volume_ratio", _volume_bucket(record.get("volume_ratio"))),
        ("market_regime", str(record.get("market_regime") or "unknown")),
        ("sector", str(record.get("sector") or "未分類")),
    ]
    features.extend(("candlestick_signal", str(signal)) for signal in signals)
    return [(name, value) for name, value in features if value]


def _candidate_dynamic_stop_rules(
    winners: list[dict[str, Any]],
    losers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rules = []
    day1_soft_winners = [
        record for record in winners
        if record.get("day1_return") is not None and float(record["day1_return"]) >= -0.02
    ]
    day1_soft_losers = [
        record for record in losers
        if record.get("day1_return") is not None and float(record["day1_return"]) >= -0.02
    ]
    winner_share = _share_count(len(day1_soft_winners), len(winners))
    loser_share = _share_count(len(day1_soft_losers), len(losers))
    if winner_share is not None and winner_share >= 0.5 and (loser_share is None or winner_share > loser_share):
        rules.append(
            {
                "rule": "Day1 -2%以内なら保有継続候補",
                "winner_share": winner_share,
                "loser_share": loser_share,
                "winner_count": len(day1_soft_winners),
                "loser_count": len(day1_soft_losers),
                "reason": "損切り後に回復した銘柄はDay1の下落が浅い傾向があります。",
            }
        )
    day1_deep_losers = [
        record for record in losers
        if record.get("day1_return") is not None and float(record["day1_return"]) < -0.02
    ]
    deep_loser_share = _share_count(len(day1_deep_losers), len(losers))
    if deep_loser_share is not None and deep_loser_share >= 0.5:
        rules.append(
            {
                "rule": "Day1 -2%超なら通常損切り維持候補",
                "winner_share": _share_count(
                    sum(1 for record in winners if record.get("day1_return") is not None and float(record["day1_return"]) < -0.02),
                    len(winners),
                ),
                "loser_share": deep_loser_share,
                "winner_count": sum(1 for record in winners if record.get("day1_return") is not None and float(record["day1_return"]) < -0.02),
                "loser_count": len(day1_deep_losers),
                "reason": "未回復組はDay1から大きく崩れる傾向があります。",
            }
        )
    return rules


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
    if number < 1.5:
        return "<1.5"
    if number < 2:
        return "1.5-2"
    if number < 3:
        return "2-3"
    return "3+"


def _share_count(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(count / total, 4)


def _difference_rates(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 4)


def _sold_before_take_profit_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    before_take_profit = [row for row in rows if row.get("exit_reason") != "利確"]
    return round(len(before_take_profit) / len(rows), 4)


def _profit_rate(row: dict[str, Any]) -> float | None:
    if row.get("profit_rate") is None:
        return None
    return float(row["profit_rate"])


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _win_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    wins = 0
    evaluated = 0
    for row in rows:
        profit_rate = row.get("profit_rate")
        if profit_rate is not None:
            evaluated += 1
            if float(profit_rate) > 0:
                wins += 1
            continue
        result = row.get("result")
        if result in {"WIN", "LOSS"}:
            evaluated += 1
            if result == "WIN":
                wins += 1
    if evaluated == 0:
        return None
    return round(wins / evaluated, 4)


def _trade_profit(row: dict[str, Any]) -> float | None:
    for key in ["net_profit", "gross_profit", "profit"]:
        if row.get(key) is not None:
            return float(row[key])
    return None


def _yearly_performance_analysis(portfolio_rows: list[dict[str, Any]], trade_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closed = profit_factor_metrics(trade_rows)["closed_trades"]
    years = sorted(
        {
            key
            for key in [_year_key(row.get("exit_date") or row.get("entry_date") or row.get("date")) for row in closed]
            if key
        }
        | {
            key
            for key in [_year_key(row.get("date")) for row in portfolio_rows]
            if key
        }
    )
    result = []
    for year in years:
        year_trades = [row for row in closed if _year_key(row.get("exit_date") or row.get("entry_date") or row.get("date")) == year]
        metrics = profit_factor_metrics(year_trades)
        win_rate = round(metrics["win_count"] / metrics["closed_trade_count"], 4) if metrics["closed_trade_count"] else None
        drawdowns = [
            float(row.get("max_drawdown") or 0)
            for row in portfolio_rows
            if _year_key(row.get("date")) == year and row.get("max_drawdown") is not None
        ]
        result.append(
            {
                "year": year,
                "profit": metrics["realized_profit_total"],
                "win_rate": win_rate,
                "profit_factor": metrics["profit_factor"],
                "max_drawdown": min(drawdowns) if drawdowns else None,
                "trades": metrics["closed_trade_count"],
            }
        )
    return result


def _walk_forward_validation(
    config: dict[str, Any],
    portfolio_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    periods = [
        _walk_forward_period(config, portfolio_rows, trade_rows, start_date, end_date)
        for start_date, end_date in WALK_FORWARD_PERIODS
    ]
    stable = [period for period in periods if _is_stable_walk_forward_period(period)]
    weak = [period for period in periods if not _is_stable_walk_forward_period(period)]
    return {
        "periods": periods,
        "stable_periods": stable,
        "weak_periods": weak,
        "overfit_risk": _walk_forward_overfit_risk(periods),
    }


def _market_regime_performance_analysis(config: dict[str, Any], trade_rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = profit_factor_metrics(trade_rows)["closed_trades"]
    regimes = [
        _market_regime_summary(config, closed, regime)
        for regime in ["risk_on", "neutral", "risk_off"]
    ]
    evaluated = [item for item in regimes if int(item.get("trade_count") or 0) > 0]
    best = max(evaluated, key=_market_regime_rank_key, default=None)
    worst = min(evaluated, key=_market_regime_rank_key, default=None)
    return {
        "regimes": regimes,
        "best_regime": best,
        "worst_regime": worst,
        "candidate_regime_filters": _candidate_regime_filters(regimes, best, worst),
    }


def _market_regime_summary(
    config: dict[str, Any],
    closed: list[dict[str, Any]],
    regime: str,
) -> dict[str, Any]:
    rows = [
        row for row in closed
        if str(row.get("market_regime") or "unknown") == regime
    ]
    metrics = profit_factor_metrics(rows)
    profits = [_trade_profit(row) for row in metrics["closed_trades"]]
    profits = [profit for profit in profits if profit is not None]
    win_rates = [
        float(row["profit_rate"]) for row in rows
        if row.get("profit_rate") is not None and float(row["profit_rate"]) > 0
    ]
    loss_rates = [
        float(row["profit_rate"]) for row in rows
        if row.get("profit_rate") is not None and float(row["profit_rate"]) <= 0
    ]
    expectancy = None
    if metrics["win_rate"] is not None:
        expectancy = round((metrics["win_rate"] * (_average(win_rates) or 0.0)) + ((1 - metrics["win_rate"]) * (_average(loss_rates) or 0.0)), 4)
    sorted_profits = [
        _trade_profit(row) for row in sorted(rows, key=_trade_timeline_key)
        if _trade_profit(row) is not None
    ]
    return {
        "market_regime": regime,
        "profit": round(sum(profits), 2) if profits else 0.0,
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "expectancy": expectancy,
        "max_drawdown": _estimated_drawdown([float(value) for value in sorted_profits], _initial_capital(config)),
        "trade_count": metrics["closed_trade_count"],
    }


def _market_regime_rank_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(item.get("profit") or 0.0),
        float(item.get("expectancy") or 0.0),
        float(item.get("profit_factor") or 0.0),
        float(item.get("win_rate") or 0.0),
    )


def _candidate_regime_filters(
    regimes: list[dict[str, Any]],
    best: dict[str, Any] | None,
    worst: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates = []
    if best and _is_positive_regime(best):
        candidates.append(
            {
                "rule": f"market_regime = {best['market_regime']} は採用維持",
                "reason": (
                    f"{best['market_regime']} は profit {float(best.get('profit') or 0):,.0f}円、"
                    f"PF {best.get('profit_factor')}、expectancy {best.get('expectancy')} で最も強い相場です。"
                ),
            }
        )
    if worst and _is_weak_regime(worst):
        candidates.append(
            {
                "rule": f"market_regime = {worst['market_regime']} は買付抑制候補",
                "reason": (
                    f"{worst['market_regime']} は profit {float(worst.get('profit') or 0):,.0f}円、"
                    f"PF {worst.get('profit_factor')}、expectancy {worst.get('expectancy')} で最も弱い相場です。"
                ),
            }
        )
    risk_off = next((item for item in regimes if item.get("market_regime") == "risk_off"), None)
    if risk_off and int(risk_off.get("trade_count") or 0) > 0 and _is_weak_regime(risk_off):
        candidates.append(
            {
                "rule": "risk_off の新規買付制限を維持または強化",
                "reason": "risk_off で profit / PF / expectancy のいずれかが弱く、守りを優先する候補です。",
            }
        )
    return candidates


def _is_positive_regime(item: dict[str, Any]) -> bool:
    return (
        int(item.get("trade_count") or 0) > 0
        and float(item.get("profit") or 0.0) > 0
        and (item.get("profit_factor") is None or float(item.get("profit_factor") or 0.0) >= 1.0)
        and (item.get("expectancy") is None or float(item.get("expectancy") or 0.0) > 0)
    )


def _is_weak_regime(item: dict[str, Any]) -> bool:
    return (
        int(item.get("trade_count") or 0) > 0
        and (
            float(item.get("profit") or 0.0) < 0
            or (item.get("profit_factor") is not None and float(item.get("profit_factor") or 0.0) < 1.0)
            or (item.get("expectancy") is not None and float(item.get("expectancy") or 0.0) < 0)
        )
    )


def _walk_forward_period(
    config: dict[str, Any],
    portfolio_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    period_trades = [
        row for row in trade_rows
        if _date_in_range(_trade_metric_date(row), start_date, end_date)
    ]
    metrics = profit_factor_metrics(period_trades)
    closed = metrics["closed_trades"]
    profits = [_trade_profit(row) for row in closed]
    profits = [profit for profit in profits if profit is not None]
    win_rates = [
        float(row["profit_rate"]) for row in closed
        if row.get("profit_rate") is not None and float(row["profit_rate"]) > 0
    ]
    loss_rates = [
        float(row["profit_rate"]) for row in closed
        if row.get("profit_rate") is not None and float(row["profit_rate"]) <= 0
    ]
    win_rate = metrics["win_rate"]
    expectancy = None
    if win_rate is not None:
        expectancy = round((win_rate * (_average(win_rates) or 0.0)) + ((1 - win_rate) * (_average(loss_rates) or 0.0)), 4)
    period_portfolio = [
        row for row in portfolio_rows
        if _date_in_range(str(row.get("date") or ""), start_date, end_date)
    ]
    portfolio_drawdowns = [
        float(row.get("max_drawdown") or 0.0)
        for row in period_portfolio
        if row.get("max_drawdown") is not None
    ]
    max_drawdown = min(portfolio_drawdowns) if portfolio_drawdowns else _estimated_drawdown(profits, _initial_capital(config))
    return {
        "start_date": start_date,
        "end_date": end_date,
        "net_cumulative_profit": round(sum(profits), 2) if profits else 0.0,
        "win_rate": win_rate,
        "profit_factor": metrics["profit_factor"],
        "max_drawdown": max_drawdown,
        "total_trades": metrics["closed_trade_count"],
        "expectancy": expectancy,
    }


def _is_stable_walk_forward_period(period: dict[str, Any]) -> bool:
    return (
        int(period.get("total_trades") or 0) > 0
        and float(period.get("net_cumulative_profit") or 0.0) > 0
        and (period.get("profit_factor") is None or float(period.get("profit_factor") or 0.0) >= 1.0)
        and (period.get("expectancy") is None or float(period.get("expectancy") or 0.0) > 0)
    )


def _walk_forward_overfit_risk(periods: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [period for period in periods if int(period.get("total_trades") or 0) > 0]
    stable = [period for period in evaluated if _is_stable_walk_forward_period(period)]
    weak = [period for period in evaluated if not _is_stable_walk_forward_period(period)]
    if not evaluated:
        return {
            "risk_level": "unknown",
            "stable_period_count": 0,
            "weak_period_count": 0,
            "reason": "評価可能な取引期間がありません。",
        }
    if len(stable) == len(evaluated):
        risk_level = "low"
        reason = "評価可能な全期間がプラス期待値でした。"
    elif len(stable) >= len(weak):
        risk_level = "moderate"
        reason = "安定期間と弱い期間が混在しています。"
    else:
        risk_level = "high"
        reason = "弱い期間が安定期間を上回っており、期間依存の可能性があります。"
    return {
        "risk_level": risk_level,
        "stable_period_count": len(stable),
        "weak_period_count": len(weak),
        "reason": reason,
    }


def _trade_metric_date(row: dict[str, Any]) -> str:
    return str(row.get("exit_date") or row.get("entry_date") or row.get("date") or "")


def _date_in_range(value: str, start_date: str, end_date: str) -> bool:
    return bool(value) and start_date <= value <= end_date


def _monthly_performance_analysis(trade_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closed = profit_factor_metrics(trade_rows)["closed_trades"]
    months = sorted(
        {
            key
            for key in [_month_key(row.get("exit_date") or row.get("entry_date") or row.get("date")) for row in closed]
            if key
        }
    )
    result = []
    for month in months:
        month_trades = [row for row in closed if _month_key(row.get("exit_date") or row.get("entry_date") or row.get("date")) == month]
        metrics = profit_factor_metrics(month_trades)
        win_rate = round(metrics["win_count"] / metrics["closed_trade_count"], 4) if metrics["closed_trade_count"] else None
        result.append(
            {
                "month": month,
                "profit": metrics["realized_profit_total"],
                "trades": metrics["closed_trade_count"],
                "win_rate": win_rate,
            }
        )
    return result


def _year_key(value: Any) -> str | None:
    text = str(value or "")
    return text[:4] if len(text) >= 4 else None


def _month_key(value: Any) -> str | None:
    text = str(value or "")
    return text[:7] if len(text) >= 7 else None


def _trade_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "trade_id": row.get("trade_id"),
        "code": row.get("code"),
        "name": row.get("name"),
        "entry_date": row.get("entry_date"),
        "exit_date": row.get("exit_date"),
        "holding_days": row.get("holding_days"),
        "entry_price": row.get("entry_price"),
        "exit_price": row.get("exit_price"),
        "shares": row.get("shares"),
        "profit": row.get("profit"),
        "profit_rate": row.get("profit_rate"),
        "gross_profit": row.get("gross_profit"),
        "net_profit": row.get("net_profit"),
        "exit_reason": row.get("exit_reason"),
        "result": row.get("result"),
    }


def _score_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("selected")]
    rejected = [row for row in rows if not row.get("selected")]
    selected_scores = [float(row["total_score"]) for row in selected if row.get("total_score") is not None]
    rejected_scores = [float(row["total_score"]) for row in rejected if row.get("total_score") is not None]
    all_scores = [float(row["total_score"]) for row in rows if row.get("total_score") is not None]
    return {
        "selected_count": len(selected),
        "conditional_selected_count": _conditional_selected_count(selected),
        "conditional_rejected_count": _conditional_rejected_count(rejected),
        "selected_average_score": _average(selected_scores),
        "rejected_average_score": _average(rejected_scores),
        "score_bands": {
            "90_or_more": sum(1 for score in all_scores if score >= 90),
            "80s": sum(1 for score in all_scores if 80 <= score < 90),
            "70s": sum(1 for score in all_scores if 70 <= score < 80),
            "60s": sum(1 for score in all_scores if 60 <= score < 70),
            "under_60": sum(1 for score in all_scores if score < 60),
        },
    }


def _conditional_selected_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if str(row.get("reason") or "").startswith("conditional selected"))


def _conditional_rejected_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if str(row.get("rejected_reason") or "").startswith("conditional rejected"))


def _reflection_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    win_good_points: dict[str, int] = {}
    loss_bad_points: dict[str, int] = {}
    suggestions: list[str] = []
    for row in rows:
        if row.get("result") == "WIN":
            _count_json_items(win_good_points, row.get("good_points"))
        if row.get("result") == "LOSS":
            _count_json_items(loss_bad_points, row.get("bad_points"))
        suggestions.extend(_load_json_list(row.get("suggestions")))
    return {
        "reflection_count": len(rows),
        "win_common_good_points": _top_counts(win_good_points),
        "loss_common_bad_points": _top_counts(loss_bad_points),
        "suggestions": sorted(set(suggestions)),
    }


def _config_version_analysis(trade_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in [item for item in trade_rows if is_filled_trade(item)]:
        version = row.get("config_version") or "unknown"
        grouped.setdefault(version, []).append(row)

    summaries = []
    for version, rows in sorted(grouped.items()):
        closed = [row for row in rows if row.get("action") == "SELL" or row.get("exit_date")]
        wins = [row for row in closed if row.get("result") == "WIN"]
        cumulative_profit = sum(float(row.get("net_profit") or row.get("profit") or 0) for row in closed)
        summaries.append(
            {
                "config_version": version,
                "trade_count": len([row for row in rows if row.get("action") in {"BUY", "SELL"}]),
                "closed_trades": len(closed),
                "win_rate": round(len(wins) / len(closed), 4) if closed else None,
                "cumulative_profit": round(cumulative_profit, 2),
            }
        )
    return summaries


def _sector_win_rate_analysis(trade_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in [item for item in trade_rows if is_filled_trade(item)]:
        if not (row.get("action") == "SELL" or row.get("exit_date")):
            continue
        sector = row.get("sector_name") or "未分類"
        grouped.setdefault(sector, []).append(row)

    summaries = []
    for sector, rows in sorted(grouped.items()):
        wins = [row for row in rows if row.get("result") == "WIN"]
        profit_rates = [float(row["profit_rate"]) for row in rows if row.get("profit_rate") is not None]
        net_profit = sum(float(row.get("net_profit") or row.get("profit") or 0) for row in rows)
        summaries.append(
            {
                "sector_name": sector,
                "closed_trades": len(rows),
                "winning_trades": len(wins),
                "win_rate": round(len(wins) / len(rows), 4) if rows else None,
                "average_profit_rate": _average(profit_rates),
                "net_profit_total": round(net_profit, 2),
            }
        )
    return sorted(summaries, key=lambda item: (item["win_rate"] or 0, item["closed_trades"]), reverse=True)


def _profile_analysis(portfolio_rows: list[dict[str, Any]], trade_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filled_trade_rows = [row for row in trade_rows if is_filled_trade(row)]
    profile_ids = sorted(
        {
            row.get("profile_id") or "unknown"
            for row in portfolio_rows + filled_trade_rows
            if row.get("profile_id") or row.get("profile_name")
        }
    )
    summaries = []
    for profile_id in profile_ids:
        p_rows = [row for row in portfolio_rows if (row.get("profile_id") or "unknown") == profile_id]
        t_rows = [row for row in filled_trade_rows if (row.get("profile_id") or "unknown") == profile_id]
        latest = p_rows[-1] if p_rows else {}
        closed = [row for row in t_rows if row.get("action") == "SELL" or row.get("exit_date")]
        wins = [row for row in closed if row.get("result") == "WIN"]
        summaries.append(
            {
                "profile_id": profile_id,
                "profile_name": latest.get("profile_name") or (t_rows[0].get("profile_name") if t_rows else profile_id),
                "latest_total_assets": latest.get("total_assets"),
                "win_rate": round(len(wins) / len(closed), 4) if closed else None,
                "max_drawdown": min((float(row.get("max_drawdown") or 0) for row in p_rows), default=None) if p_rows else None,
                "total_trades": len([row for row in t_rows if row.get("action") in {"BUY", "SELL"}]),
            }
        )
    return summaries


def _delete_non_trade_order_rows(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        DELETE FROM trades
        WHERE UPPER(COALESCE(order_status, '')) IN ('PENDING', 'REJECTED', 'CANCELLED', 'PREVIEW')
           OR action NOT IN ('BUY', 'SELL')
        """
    )


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _load_json_list(value: Any) -> list[str]:
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded]


def _count_json_items(counter: dict[str, int], value: Any) -> None:
    for item in _load_json_list(value):
        counter[item] = counter.get(item, 0) + 1


def _top_counts(counter: dict[str, int], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {"text": text, "count": count}
        for text, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]
