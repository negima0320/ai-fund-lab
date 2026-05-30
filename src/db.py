"""SQLite persistence helpers for AI Fund Lab operation data."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from config_version import config_version_from


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
                reason TEXT,
                round_lot_size INTEGER,
                use_round_lot INTEGER,
                skipped_reason TEXT,
                intended_price REAL,
                executed_price REAL,
                slippage_amount REAL,
                slippage_rate REAL,
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
                news_score REAL,
                news_reason TEXT,
                news_articles_count INTEGER,
                positive_news_count INTEGER,
                negative_news_count INTEGER,
                news_provider TEXT,
                news_limitation TEXT,
                financial_score REAL,
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
        _add_column_if_missing(connection, "scoring_results", "news_reason", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "news_articles_count", "INTEGER")
        _add_column_if_missing(connection, "scoring_results", "positive_news_count", "INTEGER")
        _add_column_if_missing(connection, "scoring_results", "negative_news_count", "INTEGER")
        _add_column_if_missing(connection, "scoring_results", "news_provider", "TEXT")
        _add_column_if_missing(connection, "scoring_results", "news_limitation", "TEXT")
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
                    trade.get("reason") or trade.get("buy_reason"),
                    trade.get("round_lot_size"),
                    1 if trade.get("use_round_lot") else 0,
                    trade.get("skipped_reason"),
                    trade.get("intended_price"),
                    trade.get("executed_price"),
                    trade.get("slippage_amount"),
                    trade.get("slippage_rate"),
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
                exit_reason, result, score, reason, round_lot_size,
                use_round_lot, skipped_reason, intended_price, executed_price,
                slippage_amount, slippage_rate, gross_profit, gross_profit_rate,
                buy_commission, sell_commission, total_commission, taxable_profit,
                estimated_tax, net_profit, net_profit_rate, dealer_comment,
                broker_provider, order_status, live_trading, safety_checked,
                config_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def save_scoring_results(config: dict[str, Any], root: Path, scoring_log: dict[str, Any]) -> None:
    target_date = scoring_log.get("date")
    config_version = scoring_log.get("config_version") or config_version_from(config)
    profile_id = scoring_log.get("profile_id") or _profile_id(config)
    profile_name = scoring_log.get("profile_name") or _profile_name(config)
    with _connect(config, root) as connection:
        connection.execute("DELETE FROM scoring_results WHERE date = ? AND profile_id = ?", (target_date, profile_id))
        connection.executemany(
            """
            INSERT INTO scoring_results (
                profile_id, profile_name, date, code, name, sector_name, sector_momentum_score,
                sector_rank, sector_comment, sector_score_adjustment,
                rank, total_score, technical_score, news_score,
                news_reason, news_articles_count, positive_news_count,
                negative_news_count, news_provider, news_limitation,
                financial_score, confidence, selected, reason, rejected_reason,
                fallback, macd, macd_signal, macd_hist, bb_upper, bb_middle,
                bb_lower, atr, candle_type, candle_body_rate, upper_shadow_rate,
                lower_shadow_rate, close_position_in_range, gap_rate,
                candlestick_signals, candlestick_score, trend_score, volume_score,
                rsi_score, source_provider, ai_reason, ai_risk, ai_confidence,
                ai_score, config_version, created_at
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
                    item.get("sector_score_adjustment"),
                    item.get("rank"),
                    item.get("total_score"),
                    item.get("technical_score"),
                    item.get("news_score"),
                    item.get("news_reason"),
                    item.get("news_articles_count"),
                    item.get("positive_news_count"),
                    item.get("negative_news_count"),
                    item.get("news_provider"),
                    item.get("news_limitation"),
                    item.get("financial_score"),
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
                    item.get("source_provider") or scoring_log.get("source_provider"),
                    item.get("ai_reason"),
                    item.get("ai_risk"),
                    item.get("ai_confidence"),
                    item.get("ai_score"),
                    item.get("config_version") or config_version,
                    _now(),
                )
                for item in scoring_log.get("scores", [])
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
                turnover_value, five_day_volatility, macd, macd_signal,
                macd_hist, bb_upper, bb_middle, bb_lower, atr, candle_type,
                candle_body_rate, upper_shadow_rate, lower_shadow_rate,
                close_position_in_range, gap_rate, candlestick_signals,
                candlestick_score, trend_score, volume_score, rsi_score,
                fallback, pass_reason, rejected_reason, config_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        portfolio_rows = [dict(row) for row in connection.execute("SELECT * FROM portfolio_snapshots ORDER BY date, id")]
        trade_rows = [dict(row) for row in connection.execute("SELECT * FROM trades ORDER BY entry_date, exit_date, id")]
        scoring_rows = [dict(row) for row in connection.execute("SELECT * FROM scoring_results ORDER BY date, rank, id")]
        reflection_rows = [dict(row) for row in connection.execute("SELECT * FROM reflections ORDER BY created_at, id")]

    if not portfolio_rows and not trade_rows and not scoring_rows and not reflection_rows:
        raise ValueError("SQLite DB has no operation data.")

    return {
        "portfolio_analysis": _portfolio_analysis(config, portfolio_rows),
        "trade_analysis": _trade_analysis(trade_rows),
        "score_analysis": _score_analysis(scoring_rows),
        "reflection_analysis": _reflection_analysis(reflection_rows),
        "config_version_analysis": _config_version_analysis(trade_rows),
        "sector_win_rate_analysis": _sector_win_rate_analysis(trade_rows),
        "profile_analysis": _profile_analysis(portfolio_rows, trade_rows),
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
        }

    latest = rows[-1]
    max_drawdown = min((float(row.get("max_drawdown") or 0) for row in rows), default=0.0)
    latest_assets = float(latest.get("total_assets") or initial_capital)
    cumulative_profit = latest_assets - initial_capital
    return {
        "initial_capital": initial_capital,
        "latest_total_assets": latest_assets,
        "cumulative_profit": round(cumulative_profit, 2),
        "cumulative_profit_rate": round(cumulative_profit / initial_capital, 4),
        "max_drawdown": max_drawdown,
        "operation_days": len({row["date"] for row in rows if row.get("date")}),
        "gross_cumulative_profit": latest.get("gross_cumulative_profit"),
        "net_cumulative_profit": latest.get("net_cumulative_profit"),
        "total_commission": latest.get("total_commission") or 0.0,
        "estimated_tax_total": latest.get("estimated_tax_total") or 0.0,
        "net_total_assets": latest.get("net_total_assets"),
    }


def _trade_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in rows if row.get("action") == "SELL" or row.get("exit_date")]
    wins = [row for row in closed if row.get("result") == "WIN"]
    losses = [row for row in closed if row.get("result") == "LOSS"]
    gross_profits = [float(row.get("gross_profit") or row.get("profit") or 0) for row in closed]
    gross_profit_total = round(sum(gross_profits), 2)
    gross_win_total = round(sum(value for value in gross_profits if value > 0), 2)
    gross_loss_total = round(sum(value for value in gross_profits if value < 0), 2)
    profit_rates = [float(row["profit_rate"]) for row in closed if row.get("profit_rate") is not None]
    win_rates = [float(row["profit_rate"]) for row in wins if row.get("profit_rate") is not None]
    loss_rates = [float(row["profit_rate"]) for row in losses if row.get("profit_rate") is not None]
    holding_days = [float(row["holding_days"]) for row in closed if row.get("holding_days") is not None]
    slippages = [float(row["slippage_rate"]) for row in rows if row.get("slippage_rate") is not None]
    return {
        "total_trades": len([row for row in rows if row.get("action") in {"BUY", "SELL"}]),
        "closed_trades": len(closed),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(len(wins) / len(closed), 4) if closed else None,
        "average_profit_rate": _average(win_rates),
        "average_loss_rate": _average(loss_rates),
        "average_all_profit_rate": _average(profit_rates),
        "average_holding_days": _average(holding_days),
        "take_profit_count": sum(1 for row in closed if row.get("exit_reason") == "利確"),
        "stop_loss_count": sum(1 for row in closed if row.get("exit_reason") == "損切り"),
        "max_holding_exit_count": sum(1 for row in closed if row.get("exit_reason") == "最大保有期間到達"),
        "total_commission": round(sum(float(row.get("total_commission") or 0) for row in closed), 2),
        "estimated_tax_total": round(sum(float(row.get("estimated_tax") or 0) for row in closed), 2),
        "gross_profit_total": gross_profit_total,
        "gross_win_total": gross_win_total,
        "gross_loss_total": gross_loss_total,
        "profit_factor": round(gross_win_total / abs(gross_loss_total), 4) if gross_loss_total < 0 else None,
        "net_profit_total": round(sum(float(row.get("net_profit") or row.get("profit") or 0) for row in closed), 2),
        "average_slippage": _average(slippages),
        "max_slippage": max(slippages, key=abs) if slippages else None,
        "gap_up_count": sum(1 for row in rows if row.get("slippage_rate") is not None and float(row["slippage_rate"]) > 0),
        "gap_down_count": sum(1 for row in rows if row.get("slippage_rate") is not None and float(row["slippage_rate"]) < 0),
    }


def _score_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("selected")]
    rejected = [row for row in rows if not row.get("selected")]
    selected_scores = [float(row["total_score"]) for row in selected if row.get("total_score") is not None]
    rejected_scores = [float(row["total_score"]) for row in rejected if row.get("total_score") is not None]
    all_scores = [float(row["total_score"]) for row in rows if row.get("total_score") is not None]
    return {
        "selected_count": len(selected),
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
    for row in trade_rows:
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
    for row in trade_rows:
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
    profile_ids = sorted(
        {
            row.get("profile_id") or "unknown"
            for row in portfolio_rows + trade_rows
            if row.get("profile_id") or row.get("profile_name")
        }
    )
    summaries = []
    for profile_id in profile_ids:
        p_rows = [row for row in portfolio_rows if (row.get("profile_id") or "unknown") == profile_id]
        t_rows = [row for row in trade_rows if (row.get("profile_id") or "unknown") == profile_id]
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
