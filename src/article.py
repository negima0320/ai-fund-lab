"""note draft Markdown generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from commentary import generate_daily_comment, generate_note_title, generate_reflection_comment, generate_sell_comment
from devlog import generate_devlog_section


def generate_note_article(
    summary: dict[str, Any],
    paper_trade_log: dict[str, Any],
    config: dict[str, Any],
    repo_root: Optional[Path] = None,
) -> str:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    one_line_title = generate_note_title(summary, paper_trade_log, config)
    lines = [
        f"# AIファンド1号 Day {summary['day_number']}：{one_line_title}",
        "",
        *_result_section(summary),
        "",
        *_market_context_section(summary.get("market_context", {})),
        "",
        *_decision_section(summary, paper_trade_log, config),
        "",
        *_reflection_section(paper_trade_log, config),
        "",
        *_safety_section(paper_trade_log.get("safety_events") or summary.get("safety_events", [])),
        "",
        *_chart_section(),
        "",
        *_execution_config_section(summary, config),
        "",
        generate_devlog_section(summary["date"], repo_root),
        "",
        "## ネギマコメント",
        "",
        "> ここに一言コメントを書く",
        "",
        "## 免責事項",
        "",
        "本記事は投資助言を目的としたものではありません。",
        "AI自動売買システムの開発・検証記録です。",
        "",
    ]
    return "\n".join(lines)


def _generate_one_line_title(summary: dict[str, Any], paper_trade_log: dict[str, Any]) -> str:
    closed = paper_trade_log["closed_trades"]
    if any(trade["exit_reason"] == "利確ルールに到達" for trade in closed):
        return "新人ディーラー、初めての利確"
    if any(trade["exit_reason"] == "損切りルールに到達" for trade in closed):
        return "ルール通りに損切りしました"
    if not paper_trade_log["orders"] and not closed:
        return "本日は静観です"
    if summary["daily_profit"] > 0:
        return "資産は増えたが、油断は禁物です"
    if summary["daily_profit"] < 0:
        return "資産は減少、ルール確認を継続します"
    return "ルールに従い、淡々と初日を終えました"


def _result_section(summary: dict[str, Any]) -> list[str]:
    return [
        "## 今日の結果",
        "",
        f"- 総資産: {summary['total_assets']:,.0f}円",
        f"- 前日比: {summary['daily_profit']:,.0f}円 ({summary['day_change_pct']:.2%})",
        f"- 累計損益: {summary['cumulative_profit']:,.0f}円",
        f"- 税引前累計損益: {summary.get('gross_cumulative_profit', summary['cumulative_profit']):+,.0f}円",
        f"- 概算税額: {summary.get('estimated_tax_total', 0):,.0f}円",
        f"- 税引後累計損益: {summary.get('net_cumulative_profit', summary['cumulative_profit']):+,.0f}円",
        f"- 手数料合計: {summary.get('total_commission', 0):,.0f}円",
        f"- 勝率: {_format_rate(summary['win_rate'])}",
        f"- 最大ドローダウン: {summary['max_drawdown']:.2%}",
    ]


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


def _decision_section(summary: dict[str, Any], paper_trade_log: dict[str, Any], config: dict[str, Any]) -> list[str]:
    lines = ["## 新人ディーラー1号の判断", "", "### 本日の買付", ""]
    lines.extend(_buy_lines(paper_trade_log["orders"]))
    lines.extend(["", "### 約定待ち注文", ""])
    lines.extend(_pending_order_lines(paper_trade_log.get("pending_orders", [])))
    lines.extend(["", "### 本日約定", ""])
    lines.extend(_executed_order_lines(paper_trade_log.get("executed_orders", [])))
    lines.extend(["", "### 選定されたが買えなかった銘柄", ""])
    lines.extend(_skipped_buy_lines(paper_trade_log.get("skipped_buys", [])))
    lines.extend(["", "### 本日の売却", ""])
    lines.extend(_sell_lines(paper_trade_log["closed_trades"], config))
    lines.extend(["", "### 保有銘柄", ""])
    lines.extend(_position_lines(paper_trade_log["positions"]))
    lines.extend(
        [
            "",
            "### 判断理由",
            "",
            summary.get("dealer_comment")
            or generate_daily_comment(
                summary,
                paper_trade_log.get("orders", []),
                paper_trade_log.get("orders", []) + paper_trade_log["closed_trades"] + paper_trade_log.get("skipped_buys", []),
                config,
            ),
        ]
    )
    return lines


def _reflection_section(paper_trade_log: dict[str, Any], config: dict[str, Any]) -> list[str]:
    lines = ["## 売却があった取引の振り返り", ""]
    if not paper_trade_log["closed_trades"]:
        lines.append("本日は売却がなかったため、確定取引の振り返りはありません。")
        return lines

    for trade in paper_trade_log["closed_trades"]:
        lines.extend(
            [
                f"### {trade['code']} {trade['name']}",
                "",
                f"- 結果: {trade['result']}",
                f"- 税引前損益: {trade.get('gross_profit', trade['profit']):,.0f}円 ({trade.get('gross_profit_rate', trade['profit_rate']):.2%})",
                f"- 概算税額: {trade.get('estimated_tax', 0):,.0f}円",
                f"- 税引後損益: {trade.get('net_profit', trade['profit']):,.0f}円 ({trade.get('net_profit_rate', trade['profit_rate']):.2%})",
                f"- 売却理由: {trade['exit_reason']}",
                f"- AI振り返り: {trade.get('reflection_comment') or generate_reflection_comment(trade, config)}",
                "",
            ]
        )
    return lines


def _chart_section() -> list[str]:
    return [
        "## グラフ",
        "",
        "![資産推移](../../../reports/charts/assets_curve.png)",
        "",
        "![累計損益](../../../reports/charts/cumulative_profit.png)",
        "",
        "![最大ドローダウン](../../../reports/charts/max_drawdown.png)",
    ]


def _execution_config_section(summary: dict[str, Any], config: dict[str, Any]) -> list[str]:
    safety = config.get("safety", {})
    broker = config.get("broker", {})
    return [
        "## 実行設定",
        "",
        f"- config_version: {summary.get('config_version') or config.get('_config_version', 'unknown')}",
        f"- data_provider: {config.get('data_provider', 'unknown')}",
        f"- broker: {broker.get('provider', 'paper')}",
        f"- safety_mode: {safety.get('mode', 'paper')}",
    ]


def _safety_section(events: list[dict[str, Any]]) -> list[str]:
    lines = ["## セーフティガード", ""]
    if not events:
        lines.append("- 発動なし")
        return lines
    for event in events:
        order = event.get("order", {})
        action = order.get("action") or order.get("side") or "UNKNOWN"
        if event.get("safety_rule") == "emergency_stop" and action == "BUY":
            lines.append("- 新規買付停止")
        else:
            lines.append(f"- {action} 注文停止")
        lines.append(f"  - 理由: {event.get('rejected_reason', '')}")
    return lines


def _buy_lines(orders: list[dict[str, Any]]) -> list[str]:
    if not orders:
        return ["- 買付なし"]
    lines = []
    for order in orders:
        lines.append(f"- {order['code']} {order['name']}: {order['quantity']}株 @ {order['price']:,.0f}円")
        lines.extend(_technical_detail_lines(order))
        if order.get("dealer_comment"):
            lines.append(f"  - コメント: {order['dealer_comment']}")
    return lines


def _technical_detail_lines(item: dict[str, Any]) -> list[str]:
    if not any(item.get(field) is not None for field in ["candle_type", "ma5", "ma25", "volume_ratio"]):
        return []
    signals = item.get("candlestick_signals") or []
    warning_signals = [signal for signal in signals if "warning" in signal or signal == "overheated_warning"]
    return [
        f"  - 業種: {item.get('sector_name') or 'N/A'} / sector_score={item.get('sector_momentum_score', 'N/A')} / rank={item.get('sector_rank', 'N/A')}",
        f"  - ローソク足タイプ: {item.get('candle_type') or 'N/A'}",
        f"  - 移動平均線の状態: {_ma_state(item)}",
        f"  - 出来高確認: {_volume_state(item)}",
        f"  - 注意シグナル: {', '.join(warning_signals) if warning_signals else 'なし'}",
    ]


def _ma_state(item: dict[str, Any]) -> str:
    close = _to_float(item.get("close") or item.get("entry_price") or item.get("price"))
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
        lines.append(f"- {item['code']} {item['name']}: {item['skipped_reason']}")
        if item.get("dealer_comment"):
            lines.append(f"  - コメント: {item['dealer_comment']}")
    return lines


def _sell_lines(closed_trades: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    if not closed_trades:
        return ["- 売却なし"]
    lines = []
    for trade in closed_trades:
        lines.append(f"- {trade['trade_id']} {trade['code']} {trade['name']}: {trade['result']} {trade['profit_rate']:.2%}, 理由: {trade['exit_reason']}")
        lines.append(f"  - コメント: {trade.get('dealer_comment') or generate_sell_comment(trade, config)}")
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
        executed = item.get("executed_price") or item.get("entry_price") or item.get("exit_price")
        lines.append(
            f"- {item.get('action')} {item.get('code')} {item.get('name')}: "
            f"約定価格={float(executed or 0):,.0f}円, 想定価格との差={float(item.get('slippage_amount') or 0):+,.0f}円 "
            f"({float(item.get('slippage_rate') or 0):+.2%})"
        )
    return lines


def _position_lines(positions: list[dict[str, Any]]) -> list[str]:
    if not positions:
        return ["- 保有銘柄なし"]
    return [
        f"- {item['code']} {item['name']}: {item['quantity']}株, 評価額 {item['market_value']:,.0f}円"
        for item in positions
    ]


def _format_rate(rate: Optional[float]) -> str:
    if rate is None:
        return "N/A（売却済み取引なし）"
    return f"{rate:.2%}"


def _format_optional_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2%}"
