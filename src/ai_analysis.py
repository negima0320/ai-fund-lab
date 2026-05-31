"""Machine-readable exports for later AI analysis."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from db import get_database_path


SCHEMA_VERSION = "1.0"


def export_ai_dataset(config: dict[str, Any], root: Path, start_date: str, end_date: str, file_name: str | None = None) -> dict[str, Any]:
    records = build_decision_records(config, root, start_date, end_date)
    output_dir = root / "analysis_logs" / _profile_id(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (file_name or f"decision_dataset_{start_date}_to_{end_date}.jsonl")
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
    return {"path": path, "record_count": len(records), "records": records}


def export_ai_summary(config: dict[str, Any], root: Path, start_date: str, end_date: str) -> dict[str, Any]:
    dataset = export_ai_dataset(config, root, start_date, end_date)
    records = dataset["records"]
    output_dir = root / "analysis_logs" / _profile_id(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"ai_summary_{start_date}_to_{end_date}.md"
    markdown = render_ai_summary(config, start_date, end_date, records, dataset["path"])
    path.write_text(markdown + "\n", encoding="utf-8")
    return {"path": path, "dataset_path": dataset["path"], "record_count": dataset["record_count"], "records": records}


def build_decision_records(config: dict[str, Any], root: Path, start_date: str, end_date: str) -> list[dict[str, Any]]:
    db_path = get_database_path(config, root)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")
    profile_id = _profile_id(config)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        scoring_rows = _rows(
            connection,
            """
            SELECT * FROM scoring_results
            WHERE profile_id = ? AND date BETWEEN ? AND ?
            ORDER BY date, rank, id
            """,
            (profile_id, start_date, end_date),
        )
        screening_rows = _rows(
            connection,
            """
            SELECT * FROM screening_results
            WHERE profile_id = ? AND date BETWEEN ? AND ?
            ORDER BY date, id
            """,
            (profile_id, start_date, end_date),
        )
        market_rows = _rows(
            connection,
            """
            SELECT * FROM market_contexts
            WHERE profile_id = ? AND date BETWEEN ? AND ?
            ORDER BY date, id
            """,
            (profile_id, start_date, end_date),
        )
        ai_rows = _rows(
            connection,
            """
            SELECT * FROM ai_decisions
            WHERE profile_id = ? AND date BETWEEN ? AND ?
            ORDER BY date, id
            """,
            (profile_id, start_date, end_date),
        )
        trade_rows = _rows(
            connection,
            """
            SELECT * FROM trades
            WHERE profile_id = ? AND (entry_date BETWEEN ? AND ? OR exit_date BETWEEN ? AND ?)
            ORDER BY entry_date, exit_date, id
            """,
            (profile_id, start_date, end_date, start_date, end_date),
        )
        portfolio_rows = _rows(
            connection,
            """
            SELECT * FROM portfolio_snapshots
            WHERE profile_id = ? AND date BETWEEN ? AND ?
            ORDER BY date, id
            """,
            (profile_id, start_date, end_date),
        )

    screening_by_key = {(row.get("date"), row.get("code")): row for row in screening_rows}
    market_by_date = {row.get("date"): row for row in market_rows}
    ai_by_date = {row.get("date"): row for row in ai_rows}
    portfolio_by_date = {row.get("date"): row for row in portfolio_rows}
    buy_by_key = _buy_trades_by_key(trade_rows)
    buy_by_code = _trades_by_code([row for row in trade_rows if row.get("action") in {"BUY", "SKIP_BUY"}])
    closed_by_key = _closed_trades_by_key(trade_rows)
    closed_by_code = _trades_by_code([row for row in trade_rows if row.get("action") == "SELL"])
    prices_by_code = _prices_by_code(screening_rows)

    records = []
    for scoring in scoring_rows:
        key = (scoring.get("date"), scoring.get("code"))
        screening = screening_by_key.get(key, {})
        ai = ai_by_date.get(scoring.get("date"), {})
        buy_trade = buy_by_key.get(key)
        if not buy_trade and scoring.get("selected"):
            buy_trade = _first_trade_on_or_after(buy_by_code.get(scoring.get("code"), []), scoring.get("date"))
        closed_trade = closed_by_key.get(key) or (closed_by_key.get((scoring.get("code"), buy_trade.get("entry_date"))) if buy_trade else None)
        if not closed_trade and scoring.get("selected"):
            closed_trade = _first_trade_on_or_after(closed_by_code.get(scoring.get("code"), []), (buy_trade or {}).get("entry_date") or scoring.get("date"))
        entry_date = (buy_trade or closed_trade or {}).get("entry_date") or scoring.get("date")
        market = market_by_date.get(entry_date) or market_by_date.get(scoring.get("date"), {})
        records.append(
            build_decision_record(
                config=config,
                scoring=scoring,
                screening=screening,
                market=market,
                ai=ai,
                buy_trade=buy_trade,
                closed_trade=closed_trade,
                prices=prices_by_code.get(scoring.get("code"), []),
                portfolio=portfolio_by_date.get(scoring.get("date"), {}),
            )
        )
    return records


def build_decision_record(
    config: dict[str, Any],
    scoring: dict[str, Any],
    screening: dict[str, Any],
    market: dict[str, Any],
    ai: dict[str, Any],
    buy_trade: dict[str, Any] | None,
    closed_trade: dict[str, Any] | None,
    prices: list[dict[str, Any]],
    portfolio: dict[str, Any],
) -> dict[str, Any]:
    selected = bool(scoring.get("selected"))
    action = _decision_action(selected, buy_trade)
    safety_rejected_reason = _safety_rejected_reason(buy_trade)
    entry_date = (buy_trade or closed_trade or {}).get("entry_date") or scoring.get("date")
    entry_price = _number((buy_trade or {}).get("entry_price")) or _number(screening.get("close"))
    future = _future_result(closed_trade, prices, entry_date, entry_price)
    market_regime = (
        (buy_trade or {}).get("market_regime")
        or (closed_trade or {}).get("market_regime")
        or market.get("market_regime")
    )
    advance_ratio = _number((buy_trade or {}).get("advance_ratio"))
    if advance_ratio is None:
        advance_ratio = _number((closed_trade or {}).get("advance_ratio"))
    if advance_ratio is None:
        advance_ratio = _number(market.get("advance_ratio"))
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": scoring.get("profile_id") or _profile_id(config),
        "profile_name": scoring.get("profile_name") or _profile_name(config),
        "config_version": scoring.get("config_version"),
        "date": scoring.get("date"),
        "code": scoring.get("code"),
        "name": scoring.get("name"),
        "market_context": {
            "market_regime": market_regime,
            "advance_ratio": advance_ratio,
            "average_change_rate": _number(market.get("average_change_rate")),
            "turnover_value_total": _number(market.get("turnover_value_total")),
            "topix_change_rate": _number(market.get("topix_change_rate")),
            "nikkei_change_rate": _number(market.get("nikkei_change_rate")),
        },
        "sector_context": {
            "sector_name": scoring.get("sector_name") or screening.get("sector_name"),
            "sector_momentum_score": _number(scoring.get("sector_momentum_score") or screening.get("sector_momentum_score")),
            "sector_rank": _int(scoring.get("sector_rank") or screening.get("sector_rank")),
            "sector_comment": scoring.get("sector_comment") or screening.get("sector_comment"),
        },
        "technical_features": {
            "close": _number(screening.get("close")),
            "ma5": _number(screening.get("ma5")),
            "ma25": _number(screening.get("ma25")),
            "rsi": _number(screening.get("rsi")),
            "macd": _number(scoring.get("macd") or screening.get("macd")),
            "macd_signal": _number(scoring.get("macd_signal") or screening.get("macd_signal")),
            "macd_hist": _number(scoring.get("macd_hist") or screening.get("macd_hist")),
            "bb_position": _bb_position(screening),
            "atr": _number(scoring.get("atr") or screening.get("atr")),
            "volume": _number(screening.get("volume")),
            "volume_ratio": _number(screening.get("volume_ratio")),
            "turnover_value": _number(screening.get("turnover_value")),
            "five_day_volatility": _number(screening.get("five_day_volatility")),
        },
        "candlestick_features": {
            "candle_type": scoring.get("candle_type") or screening.get("candle_type"),
            "candle_body_rate": _number(scoring.get("candle_body_rate") or screening.get("candle_body_rate")),
            "upper_shadow_rate": _number(scoring.get("upper_shadow_rate") or screening.get("upper_shadow_rate")),
            "lower_shadow_rate": _number(scoring.get("lower_shadow_rate") or screening.get("lower_shadow_rate")),
            "close_position_in_range": _number(scoring.get("close_position_in_range") or screening.get("close_position_in_range")),
            "gap_rate": _number(scoring.get("gap_rate") or screening.get("gap_rate")),
            "candlestick_signals": _json_list(scoring.get("candlestick_signals") or screening.get("candlestick_signals")),
        },
        "rule_based_score": {
            "total_score": _number(scoring.get("total_score")),
            "technical_score": _number(scoring.get("technical_score")),
            "trend_score": _number(scoring.get("trend_score")),
            "volume_score": _number(scoring.get("volume_score")),
            "rsi_score": _number(scoring.get("rsi_score")),
            "candlestick_score": _number(scoring.get("candlestick_score")),
            "ma_score": _number(scoring.get("ma_score") or scoring.get("trend_score")),
            "market_context_score": _number(scoring.get("market_context_score")),
            "sector_score": _number(scoring.get("sector_score") or scoring.get("sector_score_adjustment")),
            "penalty_score": _number(scoring.get("penalty_score")),
            "score_components": _json_dict(scoring.get("score_components")),
            "score_components_total": _number(scoring.get("score_components_total")),
            "score_components_match": _bool_or_none(scoring.get("score_components_match")),
            "confidence": _number(scoring.get("confidence")),
            "rank": _int(scoring.get("rank")),
            "reason": scoring.get("reason"),
        },
        "ai_decision": {
            "used": bool(ai and ai.get("provider") == "openai" and not ai.get("fallback_used")),
            "model": ai.get("model") if ai else None,
            "ai_score": _number(scoring.get("ai_score")),
            "ai_rank": None,
            "ai_confidence": _number(scoring.get("ai_confidence")),
            "ai_reason": scoring.get("ai_reason"),
            "ai_risk": scoring.get("ai_risk"),
        },
        "decision": {
            "selected": selected,
            "action": action,
            "rejected_reason": scoring.get("rejected_reason"),
            "market_filter_applied": bool(scoring.get("market_filter_applied")),
            "market_regime": scoring.get("market_regime") or market_regime,
            "market_filter_reason": scoring.get("market_filter_reason"),
            "safety_passed": safety_rejected_reason is None,
            "safety_rejected_reason": safety_rejected_reason,
            "order_created": buy_trade is not None,
            "order_executed": bool(buy_trade and buy_trade.get("order_status") == "FILLED"),
            "skipped_reason": (buy_trade or {}).get("skipped_reason"),
        },
        "position_context": {
            "already_holding": False,
            "holding_days": 0,
            "entry_price": None,
            "current_profit_rate": None,
        },
        "future_result": future,
        "portfolio_context": {
            "total_assets": _number(portfolio.get("total_assets")),
            "open_positions_count": _int(portfolio.get("open_positions_count")),
            "max_drawdown": _number(portfolio.get("max_drawdown")),
        },
    }


def render_ai_summary(config: dict[str, Any], start_date: str, end_date: str, records: list[dict[str, Any]], dataset_path: Path) -> str:
    closed = [record for record in records if record["future_result"].get("result_available")]
    total_trades = len(closed)
    wins = [record for record in closed if _result_profit(record) > 0]
    losses = [record for record in closed if _result_profit(record) <= 0]
    exit_reasons = Counter(record["future_result"].get("exit_reason") or "unknown" for record in closed)
    lines = [
        "# AI改善用サマリ",
        "",
        f"- profile_id: {_profile_id(config)}",
        f"- profile_name: {_profile_name(config)}",
        f"- 期間: {start_date} to {end_date}",
        f"- dataset: {_relative_path(dataset_path)}",
        f"- レコード数: {len(records)}",
        f"- 総取引数: {total_trades}",
        f"- 勝率: {_rate(len(wins), total_trades)}",
        f"- 最大ドローダウン: {_summary_max_drawdown(records)}",
        "",
        "## スコア帯別勝率",
        "",
        *_group_lines(_bucket_win_rates(closed, lambda r: _score_bucket(r["rule_based_score"].get("total_score")))),
        "",
        "## RSI帯別勝率",
        "",
        *_group_lines(_bucket_win_rates(closed, lambda r: _rsi_bucket(r["technical_features"].get("rsi")))),
        "",
        "## 出来高倍率別勝率",
        "",
        *_group_lines(_bucket_win_rates(closed, lambda r: _volume_bucket(r["technical_features"].get("volume_ratio")))),
        "",
        "## ローソク足シグナル別勝率",
        "",
        *_group_lines(_signal_win_rates(closed)),
        "",
        "## market_regime別勝率",
        "",
        *_group_lines(_bucket_win_rates(closed, lambda r: r["market_context"].get("market_regime") or "unknown")),
        "",
        "## sector別勝率",
        "",
        *_group_lines(_bucket_win_rates(closed, lambda r: r["sector_context"].get("sector_name") or "unknown")),
        "",
        "## 利確/損切り/期限切れ件数",
        "",
        *_counter_lines(exit_reasons),
        "",
        "## ChatGPTに改善案を聞くためのプロンプト雛形",
        "",
        "以下のAIファンド売買ログを分析し、勝ちパターン・負けパターン・改善すべきルール・過剰なフィルター・不足している指標を提案してください。ただし、過学習を避け、再現性の高い改善案だけを出してください。",
    ]
    if losses:
        lines.extend(["", "## 補足", "", f"- 負け取引数: {len(losses)}"])
    return "\n".join(lines)


def record_ai_analysis_export(
    config: dict[str, Any],
    root: Path,
    start_date: str,
    end_date: str,
    dataset_path: Path | None,
    summary_path: Path | None,
    record_count: int,
) -> None:
    db_path = get_database_path(config, root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_analysis_exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT,
                start_date TEXT,
                end_date TEXT,
                dataset_path TEXT,
                summary_path TEXT,
                record_count INTEGER,
                created_at TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO ai_analysis_exports (
                profile_id, start_date, end_date, dataset_path, summary_path, record_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _profile_id(config),
                start_date,
                end_date,
                str(dataset_path) if dataset_path else None,
                str(summary_path) if summary_path else None,
                record_count,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def _rows(connection: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, params)]


def _buy_trades_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        if row.get("action") in {"BUY", "SKIP_BUY"}:
            result[(row.get("entry_date"), row.get("code"))] = row
    return result


def _closed_trades_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        if row.get("action") == "SELL":
            result[(row.get("entry_date"), row.get("code"))] = row
            result[(row.get("code"), row.get("entry_date"))] = row
    return result


def _prices_by_code(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("code") and row.get("date"):
            result[row["code"]].append(row)
    for code in result:
        result[code] = sorted(result[code], key=lambda row: row.get("date") or "")
    return result


def _trades_by_code(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("code"):
            result[row["code"]].append(row)
    for code in result:
        result[code] = sorted(result[code], key=lambda row: (row.get("entry_date") or row.get("exit_date") or "", row.get("id") or 0))
    return result


def _first_trade_on_or_after(rows: list[dict[str, Any]], target_date: str | None) -> dict[str, Any] | None:
    if not target_date:
        return None
    for row in rows:
        row_date = row.get("entry_date") or row.get("exit_date")
        if row_date and row_date >= target_date:
            return row
    return None


def _future_result(closed: dict[str, Any] | None, prices: list[dict[str, Any]], entry_date: str, entry_price: float | None) -> dict[str, Any]:
    result = {
        "result_available": bool(closed),
        "exit_date": (closed or {}).get("exit_date"),
        "exit_reason": (closed or {}).get("exit_reason"),
        "holding_days": _int((closed or {}).get("holding_days")),
        "gross_profit": _number((closed or {}).get("gross_profit") or (closed or {}).get("profit")),
        "gross_profit_rate": _number((closed or {}).get("gross_profit_rate") or (closed or {}).get("profit_rate")),
        "net_profit": _number((closed or {}).get("net_profit")),
        "net_profit_rate": _number((closed or {}).get("net_profit_rate")),
        "max_favorable_excursion": None,
        "max_adverse_excursion": None,
        "price_after_1d": _future_price(prices, entry_date, 1),
        "price_after_3d": _future_price(prices, entry_date, 3),
        "price_after_5d": _future_price(prices, entry_date, 5),
    }
    if entry_price:
        future_prices = [
            _number(row.get("close"))
            for row in prices
            if row.get("date") and row.get("date") > entry_date and (not closed or row.get("date") <= closed.get("exit_date"))
        ]
        future_prices = [price for price in future_prices if price is not None]
        if future_prices:
            result["max_favorable_excursion"] = round((max(future_prices) - entry_price) / entry_price, 6)
            result["max_adverse_excursion"] = round((min(future_prices) - entry_price) / entry_price, 6)
    return result


def _future_price(prices: list[dict[str, Any]], entry_date: str, offset: int) -> float | None:
    future = [row for row in prices if row.get("date") and row.get("date") > entry_date]
    if len(future) < offset:
        return None
    return _number(future[offset - 1].get("close"))


def _decision_action(selected: bool, buy_trade: dict[str, Any] | None) -> str:
    if buy_trade and buy_trade.get("action") == "BUY":
        return "BUY"
    if buy_trade and buy_trade.get("action") == "SKIP_BUY":
        return "SKIP"
    return "HOLD" if selected else "SKIP"


def _safety_rejected_reason(trade: dict[str, Any] | None) -> str | None:
    if not trade:
        return None
    if trade.get("order_status") == "REJECTED" or trade.get("skipped_reason"):
        return trade.get("skipped_reason") or trade.get("reason")
    return None


def _bb_position(row: dict[str, Any]) -> float | None:
    close = _number(row.get("close"))
    lower = _number(row.get("bb_lower"))
    upper = _number(row.get("bb_upper"))
    if close is None or lower is None or upper is None or upper == lower:
        return None
    return round((close - lower) / (upper - lower), 6)


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


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _profile_id(config: dict[str, Any]) -> str:
    return str(config.get("profile_id") or config.get("dealer", {}).get("id") or "rookie_dealer_01")


def _profile_name(config: dict[str, Any]) -> str:
    return str(config.get("profile_name") or config.get("dealer", {}).get("name") or _profile_id(config))


def _result_profit(record: dict[str, Any]) -> float:
    return float(record["future_result"].get("net_profit") or record["future_result"].get("gross_profit") or 0)


def _rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "N/A"
    return f"{numerator / denominator:.2%}"


def _summary_max_drawdown(records: list[dict[str, Any]]) -> str:
    values = [record.get("portfolio_context", {}).get("max_drawdown") for record in records]
    values = [float(value) for value in values if value is not None]
    return "N/A" if not values else f"{min(values):.2%}"


def _bucket_win_rates(records: list[dict[str, Any]], bucket_fn: Any) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(bucket_fn(record))].append(record)
    return [_group_stats(name, items) for name, items in sorted(groups.items())]


def _signal_win_rates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        signals = record["candlestick_features"].get("candlestick_signals") or ["no_signal"]
        for signal in signals:
            groups[str(signal)].append(record)
    return [_group_stats(name, items) for name, items in sorted(groups.items())]


def _group_stats(name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for item in items if _result_profit(item) > 0)
    return {"name": name, "count": len(items), "win_rate": wins / len(items) if items else None}


def _group_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- データなし"]
    return [f"- {item['name']}: {item['count']}件, 勝率 {_format_rate(item['win_rate'])}" for item in items]


def _counter_lines(counter: Counter) -> list[str]:
    if not counter:
        return ["- データなし"]
    return [f"- {key}: {count}件" for key, count in counter.most_common()]


def _format_rate(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2%}"


def _score_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    lower = int(value // 10 * 10)
    return f"{lower}-{lower + 9}"


def _rsi_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 40:
        return "rsi_under_40"
    if value < 50:
        return "rsi_40_49"
    if value < 65:
        return "rsi_50_64"
    if value < 70:
        return "rsi_65_69"
    return "rsi_70_plus"


def _volume_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 1:
        return "volume_under_1x"
    if value < 1.5:
        return "volume_1_1.49x"
    if value < 2:
        return "volume_1.5_1.99x"
    return "volume_2x_plus"


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
