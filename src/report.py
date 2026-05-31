"""Daily report Markdown generation."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

from commentary import generate_buy_comment, generate_daily_comment, generate_sell_comment


def generate_daily_report(
    summary: dict[str, Any],
    paper_trade_log: dict[str, Any],
    trade_decision_log: dict[str, Any],
    config: dict[str, Any],
) -> str:
    buys = [decision for decision in trade_decision_log["decisions"] if decision["decision"] == "BUY"]
    orders = paper_trade_log.get("orders", [])
    positions = paper_trade_log["positions"]
    closed = paper_trade_log["closed_trades"]
    skipped_buys = paper_trade_log.get("skipped_buys", [])
    safety_events = paper_trade_log.get("safety_events") or summary.get("safety_events", [])
    win_rate = _format_rate(summary["win_rate"])
    daily_comment = summary.get("dealer_comment") or generate_daily_comment(summary, buys, orders + closed + skipped_buys, config)

    lines = [
        f"# 日報 Day {summary['day_number']} - {summary['date']}",
        "",
        f"担当AI: {config['dealer']['name']}",
        "",
        *_market_context_section(summary.get("market_context", {})),
        "",
        "## サマリ",
        "",
        f"- 総資産: {summary['total_assets']:,.0f}円",
        f"- 前日比: {summary['day_change']:,.0f}円 ({summary['day_change_pct']:.2%})",
        f"- 累計損益: {summary['cumulative_pnl']:,.0f}円",
        f"- 税引前累計損益: {summary.get('gross_cumulative_profit', summary['cumulative_pnl']):+,.0f}円",
        f"- 概算税額: {summary.get('estimated_tax_total', 0):,.0f}円",
        f"- 税引後累計損益: {summary.get('net_cumulative_profit', summary['cumulative_pnl']):+,.0f}円",
        f"- 手数料合計: {summary.get('total_commission', 0):,.0f}円",
        f"- 勝率: {win_rate}",
        f"- 最大ドローダウン: {summary['max_drawdown']:.2%}",
        f"- 最大ドローダウン計算: {summary['max_drawdown_note']}",
        "",
        "## Daily Report",
        "",
        *_paper_daily_report_lines(summary, paper_trade_log),
        "",
        "## Weekly Report",
        "",
        *_paper_weekly_report_lines(summary, paper_trade_log),
        "",
        "## Walk Forward Validation",
        "",
        *_walk_forward_validation_lines(summary.get("walk_forward_validation", {})),
        "",
        "## 本日の売買判断",
        "",
    ]
    lines.extend(_buy_lines(buys, config))
    lines.extend(["", "## 約定待ち注文", ""])
    lines.extend(_pending_order_lines(paper_trade_log.get("pending_orders", [])))
    lines.extend(["", "## 本日約定", ""])
    lines.extend(_executed_order_lines(paper_trade_log.get("executed_orders", [])))
    lines.extend(["", "## 選定されたが買えなかった銘柄", ""])
    lines.extend(_skipped_buy_lines(skipped_buys))
    lines.extend(["", "## 保有銘柄", ""])
    lines.extend(_position_lines(positions))
    lines.extend(["", "## 売却済み取引", ""])
    lines.extend(_closed_trade_lines(closed, config))
    lines.extend(["", "## セーフティガード", ""])
    lines.extend(_safety_lines(safety_events))
    lines.extend(
        [
            "",
            "## グラフ",
            "",
            "![資産推移](../charts/assets_curve.png)",
            "",
            "![累計損益](../charts/cumulative_profit.png)",
            "",
            "![最大ドローダウン](../charts/max_drawdown.png)",
        ]
    )
    lines.extend(
        [
            "",
            "## 新人ディーラー1号コメント",
            "",
            daily_comment,
            "",
        ]
    )
    return "\n".join(lines)


def _market_context_section(market_context: dict[str, Any]) -> list[str]:
    if not market_context:
        return [
            "## 今日の市場環境",
            "",
            "- 地合い: neutral",
            "- 値上がり銘柄比率: N/A",
            "- 平均騰落率: N/A",
            "- 新人ディーラー1号の市場コメント: 市場環境データは未生成です。個別銘柄スコアを優先します。",
        ]
    lines = [
        "## 今日の市場環境",
        "",
        f"- 地合い: {market_context.get('market_regime', 'neutral')}",
        f"- 値上がり銘柄比率: {_format_optional_percent(market_context.get('advance_ratio'))}",
        f"- 平均騰落率: {_format_optional_percent(market_context.get('average_change_rate'))}",
        f"- 新人ディーラー1号の市場コメント: {market_context.get('market_comment') or '市場環境はneutralとして扱います。'}",
    ]
    lines.extend(["", "### 今日強かった業種", ""])
    lines.extend(_top_sector_lines(market_context.get("top_sectors") or market_context.get("sector_momentum", [])))
    return lines


def _top_sector_lines(sectors: list[dict[str, Any]]) -> list[str]:
    if not sectors:
        return ["- 業種別モメンタムは未生成です"]
    return [
        (
            f"- {item.get('sector_rank', index)}位 {item.get('sector_name', '未分類')}: "
            f"score={float(item.get('sector_momentum_score') or 0):.1f}, "
            f"値上がり比率={_format_optional_percent(item.get('advance_ratio'))}, "
            f"平均騰落率={_format_optional_percent(item.get('average_change_rate'))}"
        )
        for index, item in enumerate(sectors[:5], start=1)
    ]


def _paper_daily_report_lines(summary: dict[str, Any], paper_trade_log: dict[str, Any]) -> list[str]:
    all_closed = paper_trade_log.get("all_closed_trades") or paper_trade_log.get("closed_trades", [])
    lines = ["### 今日買った銘柄", ""]
    lines.extend(_today_bought_lines(paper_trade_log.get("orders", [])))
    lines.extend(["", "### 今日売った銘柄", ""])
    lines.extend(_today_sold_lines(paper_trade_log.get("closed_trades", [])))
    lines.extend(
        [
            "",
            "### 含み損益",
            "",
            f"- 含み損益: {_format_yen(_unrealized_profit_total(paper_trade_log.get('positions', [])))}",
            "",
            "### セクター比率",
            "",
            *_sector_ratio_lines(paper_trade_log.get("positions", [])),
            "",
            "### 勝率 / PF",
            "",
            f"- 勝率: {_format_optional_percent(summary.get('win_rate'))}",
            f"- PF: {_format_optional_number(_profit_factor(all_closed))}",
        ]
    )
    return lines


def _paper_weekly_report_lines(summary: dict[str, Any], paper_trade_log: dict[str, Any]) -> list[str]:
    all_closed = paper_trade_log.get("all_closed_trades") or paper_trade_log.get("closed_trades", [])
    report_date = _parse_date(str(summary.get("date") or paper_trade_log.get("date") or ""))
    if report_date is None:
        return ["- 期間集計データなし"]
    week_start = report_date - timedelta(days=report_date.weekday())
    month_start = report_date.replace(day=1)
    year_start = report_date.replace(month=1, day=1)
    return [
        "### 今週の成績",
        "",
        *_period_performance_lines(all_closed, week_start, report_date),
        "",
        "### 月初来成績",
        "",
        *_period_performance_lines(all_closed, month_start, report_date),
        "",
        "### 年初来成績",
        "",
        *_period_performance_lines(all_closed, year_start, report_date),
    ]


def _walk_forward_validation_lines(analysis: dict[str, Any]) -> list[str]:
    if not analysis:
        return ["- Walk Forward Validation データなし"]
    lines = ["### Period Results", ""]
    lines.extend(_walk_forward_period_lines(analysis.get("periods", [])))
    lines.extend(["", "### Stable Periods", ""])
    lines.extend(_walk_forward_period_lines(analysis.get("stable_periods", [])))
    lines.extend(["", "### Weak Periods", ""])
    lines.extend(_walk_forward_period_lines(analysis.get("weak_periods", [])))
    lines.extend(["", "### Overfit Risk", ""])
    risk = analysis.get("overfit_risk", {})
    if risk:
        lines.extend(
            [
                f"- risk_level: {risk.get('risk_level')}",
                f"- stable_period_count: {risk.get('stable_period_count')}",
                f"- weak_period_count: {risk.get('weak_period_count')}",
                f"- reason: {risk.get('reason')}",
            ]
        )
    else:
        lines.append("- overfit risk データなし")
    return lines


def _walk_forward_period_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 該当期間なし"]
    return [
        (
            f"- {item.get('start_date')} to {item.get('end_date')}: "
            f"net_cumulative_profit {_format_yen(item.get('net_cumulative_profit'))}, "
            f"win_rate {_format_optional_percent(item.get('win_rate'))}, "
            f"profit_factor {_format_optional_number(item.get('profit_factor'))}, "
            f"max_drawdown {_format_optional_percent(item.get('max_drawdown'))}, "
            f"total_trades {item.get('total_trades')}, "
            f"expectancy {_format_optional_percent(item.get('expectancy'))}"
        )
        for item in items
    ]


def _today_bought_lines(orders: list[dict[str, Any]]) -> list[str]:
    buys = [item for item in orders if item.get("action") == "BUY"]
    if not buys:
        return ["- 本日買付なし"]
    return [
        (
            f"- {item.get('code')} {item.get('name')}: "
            f"{item.get('quantity', item.get('shares', 0))}株, "
            f"entry_price={float(item.get('price') or item.get('entry_price') or 0):,.0f}円"
        )
        for item in buys
    ]


def _today_sold_lines(closed: list[dict[str, Any]]) -> list[str]:
    if not closed:
        return ["- 本日売却なし"]
    return [
        (
            f"- {item.get('code')} {item.get('name')}: "
            f"損益 {_format_yen(item.get('net_profit', item.get('profit')))}, "
            f"損益率 {_format_optional_percent(item.get('net_profit_rate', item.get('profit_rate')))}, "
            f"理由 {item.get('exit_reason') or 'N/A'}"
        )
        for item in closed
    ]


def _sector_ratio_lines(positions: list[dict[str, Any]]) -> list[str]:
    if not positions:
        return ["- 保有銘柄なし"]
    total_value = sum(float(item.get("market_value") or 0.0) for item in positions)
    if total_value <= 0:
        return ["- 評価額データなし"]
    sectors: dict[str, float] = {}
    for item in positions:
        sector = str(item.get("sector_name") or "未分類")
        sectors[sector] = sectors.get(sector, 0.0) + float(item.get("market_value") or 0.0)
    return [
        f"- {sector}: {value / total_value:.2%} ({value:,.0f}円)"
        for sector, value in sorted(sectors.items(), key=lambda item: item[1], reverse=True)
    ]


def _period_performance_lines(closed: list[dict[str, Any]], start: date, end: date) -> list[str]:
    rows = [
        item for item in closed
        if (trade_date := _parse_date(str(item.get("exit_date") or item.get("entry_date") or ""))) is not None
        and start <= trade_date <= end
    ]
    profit = sum(_trade_profit(item) for item in rows)
    return [
        f"- 期間: {start.isoformat()} to {end.isoformat()}",
        f"- 損益: {_format_yen(profit)}",
        f"- 勝率: {_format_optional_percent(_win_rate(rows))}",
        f"- PF: {_format_optional_number(_profit_factor(rows))}",
        f"- 取引数: {len(rows)}",
    ]


def _unrealized_profit_total(positions: list[dict[str, Any]]) -> float:
    return sum(float(item.get("unrealized_pnl", item.get("unrealized_profit", 0.0)) or 0.0) for item in positions)


def _trade_profit(item: dict[str, Any]) -> float:
    for key in ["net_profit", "gross_profit", "profit"]:
        if item.get(key) is not None:
            return float(item[key])
    return 0.0


def _win_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    wins = sum(1 for item in rows if _trade_profit(item) > 0)
    return round(wins / len(rows), 4)


def _profit_factor(rows: list[dict[str, Any]]) -> float | None:
    profits = [_trade_profit(item) for item in rows]
    gross_win = sum(value for value in profits if value > 0)
    gross_loss = sum(value for value in profits if value < 0)
    if gross_loss >= 0:
        return None
    return round(gross_win / abs(gross_loss), 4)


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _format_yen(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.0f}円"


def _format_optional_number(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.2f}"


def _buy_lines(buys: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    if not buys:
        return ["- 買付判断なし"]
    lines = []
    for item in buys:
        lines.append(f"- {item['code']} {item['name']}: {item['reason']} (score={item['total_score']}, confidence={item['confidence']})")
        lines.extend(_technical_detail_lines(item))
        lines.append(f"  - コメント: {item.get('dealer_comment') or generate_buy_comment(item, config)}")
    return lines


def _technical_detail_lines(item: dict[str, Any]) -> list[str]:
    if not any(item.get(field) is not None for field in ["candle_type", "ma5", "ma25", "volume_ratio"]):
        return []
    signals = item.get("candlestick_signals") or []
    warning_signals = [signal for signal in signals if "warning" in signal or signal == "overheated_warning"]
    ma_state = _ma_state(item)
    volume_state = _volume_state(item)
    return [
        f"  - 業種: {item.get('sector_name') or 'N/A'} / sector_score={item.get('sector_momentum_score', 'N/A')} / rank={item.get('sector_rank', 'N/A')}",
        f"  - market_filter: {_market_filter_state(item)}",
        f"  - ローソク足タイプ: {item.get('candle_type') or 'N/A'}",
        f"  - 移動平均線の状態: {ma_state}",
        f"  - 出来高確認: {volume_state}",
        f"  - 注意シグナル: {', '.join(warning_signals) if warning_signals else 'なし'}",
    ]


def _market_filter_state(item: dict[str, Any]) -> str:
    if not item.get("market_filter_applied"):
        return f"{item.get('market_regime') or 'neutral'} / not applied"
    return f"{item.get('market_regime') or 'unknown'} / {item.get('market_filter_reason') or 'applied'}"


def _ma_state(item: dict[str, Any]) -> str:
    close = _to_float(item.get("close"))
    ma5 = _to_float(item.get("ma5"))
    ma25 = _to_float(item.get("ma25"))
    if close is not None and ma5 is not None and ma25 is not None and close > ma5 > ma25:
        return "close > ma5 > ma25"
    if close is not None and ma5 is not None and close > ma5:
        return "close > ma5"
    return "N/A"


def _volume_state(item: dict[str, Any]) -> str:
    ratio = _to_float(item.get("volume_ratio"))
    if ratio is None:
        return "N/A"
    if ratio >= 1.8:
        return f"前日比 {ratio:.2f}倍で増加"
    return f"前日比 {ratio:.2f}倍"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _skipped_buy_lines(skipped_buys: list[dict[str, Any]]) -> list[str]:
    if not skipped_buys:
        return ["- 該当なし"]
    lines = []
    for item in skipped_buys:
        lines.append(
            f"- {item['code']} {item['name']}: {item['skipped_reason']} "
            f"(price={item['entry_price']:,.0f}円, round_lot={item['round_lot_size']}株)"
        )
        if item.get("dealer_comment"):
            lines.append(f"  - コメント: {item['dealer_comment']}")
    return lines


def _position_lines(positions: list[dict[str, Any]]) -> list[str]:
    if not positions:
        return ["- 保有銘柄なし"]
    return [
        f"- {item['code']} {item['name']}: {item['quantity']}株, 評価額 {item['market_value']:,.0f}円, 含み損益 {item['unrealized_pnl']:,.0f}円"
        for item in positions
    ]


def _closed_trade_lines(closed: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    if not closed:
        return ["- 売却済み取引なし"]
    lines = []
    for item in closed:
        lines.append(
            f"- {item['trade_id']} {item['code']} {item['name']}: {item['result']}, "
            f"entry_date={item['entry_date']}, exit_date={item['exit_date']}, "
            f"holding_days={item['holding_days']}, shares={item['shares']}, "
            f"税引前 {item.get('gross_profit', item['profit']):,.0f}円, "
            f"税引後 {item.get('net_profit', item['profit']):,.0f}円 "
            f"({item.get('net_profit_rate', item['profit_rate']):.2%}), "
            f"概算税額 {item.get('estimated_tax', 0):,.0f}円, 理由: {item['exit_reason']}"
        )
        lines.append(f"  - コメント: {item.get('dealer_comment') or generate_sell_comment(item, config)}")
    return lines


def _pending_order_lines(orders: list[dict[str, Any]]) -> list[str]:
    if not orders:
        return ["- 約定待ち注文なし"]
    lines = []
    for item in orders:
        lines.append(
            f"- {item.get('action')} {item.get('code')} {item.get('name')}: "
            f"予定日={item.get('scheduled_execution_date')}, 想定価格={float(item.get('intended_price') or 0):,.0f}円"
        )
        lines.extend(_technical_detail_lines(item))
    return lines


def _executed_order_lines(orders: list[dict[str, Any]]) -> list[str]:
    if not orders:
        return ["- 本日約定なし"]
    lines = []
    for item in orders:
        intended = item.get("intended_price")
        executed = item.get("executed_price") or item.get("entry_price") or item.get("exit_price")
        if intended is None or executed is None:
            lines.append(f"- {item.get('action')} {item.get('code')} {item.get('name')}: 約定価格={float(executed or 0):,.0f}円")
            continue
        lines.append(
            f"- {item.get('action')} {item.get('code')} {item.get('name')}: "
            f"約定価格={float(executed):,.0f}円, 想定価格との差={float(item.get('slippage_amount') or 0):+,.0f}円 "
            f"({float(item.get('slippage_rate') or 0):+.2%})"
        )
    return lines


def _safety_lines(events: list[dict[str, Any]]) -> list[str]:
    if not events:
        return ["- 発動なし"]
    lines = []
    for event in events:
        order = event.get("order", {})
        action = order.get("action") or order.get("side") or "UNKNOWN"
        if event.get("safety_rule") == "emergency_stop" and action == "BUY":
            lines.append("- 新規買付停止")
        else:
            lines.append(f"- {action} 注文停止")
        lines.append(f"  - 理由: {event.get('rejected_reason', '')}")
    return lines


def _format_rate(rate: Optional[float]) -> str:
    if rate is None:
        return "N/A（売却済み取引なし）"
    return f"{rate:.2%}"


def _format_optional_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2%}"
