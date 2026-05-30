"""Portfolio summaries."""

from __future__ import annotations

from typing import Any


def build_daily_summary(paper_trade_log: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    closed_trades = paper_trade_log.get("all_closed_trades", paper_trade_log["closed_trades"])
    wins = sum(1 for trade in closed_trades if trade["result"] == "WIN")
    losses = sum(1 for trade in closed_trades if trade["result"] == "LOSS")
    trade_count = len(closed_trades)
    win_rate = round(wins / trade_count, 4) if trade_count else None
    initial_cash = float(config["portfolio"]["initial_cash"])
    total_assets = float(paper_trade_log["pnl"]["total_assets"])
    asset_history = [initial_cash] + paper_trade_log.get("asset_history", [total_assets])
    previous_day_assets = asset_history[-2] if len(asset_history) >= 2 else initial_cash
    daily_profit = round(total_assets - previous_day_assets, 2)
    drawdown = _calculate_max_drawdown(asset_history)

    return {
        "run_id": paper_trade_log["run_id"],
        "date": paper_trade_log["date"],
        "dealer_id": config["dealer"]["id"],
        "day": paper_trade_log.get("day_number", 1),
        "day_number": paper_trade_log.get("day_number", 1),
        "cash": paper_trade_log["pnl"]["cash"],
        "positions_value": paper_trade_log["pnl"]["portfolio_market_value"],
        "total_assets": total_assets,
        "daily_profit": daily_profit,
        "cumulative_profit": paper_trade_log["pnl"]["cumulative_pnl"],
        "gross_cumulative_profit": paper_trade_log["pnl"].get("gross_cumulative_profit", paper_trade_log["pnl"]["cumulative_pnl"]),
        "net_cumulative_profit": paper_trade_log["pnl"].get("net_cumulative_profit", paper_trade_log["pnl"]["cumulative_pnl"]),
        "total_commission": paper_trade_log["pnl"].get("total_commission", 0.0),
        "estimated_tax_total": paper_trade_log["pnl"].get("estimated_tax_total", 0.0),
        "net_total_assets": paper_trade_log["pnl"].get("net_total_assets", total_assets),
        "previous_day_assets": previous_day_assets,
        "day_change": daily_profit,
        "day_change_pct": round(daily_profit / previous_day_assets, 4),
        "cumulative_pnl": paper_trade_log["pnl"]["cumulative_pnl"],
        "cumulative_return_pct": paper_trade_log["pnl"]["cumulative_return_pct"],
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "closed_trade_count": trade_count,
        "max_drawdown": round(drawdown, 4),
        "max_drawdown_note": "初期資産と各Day終了時の総資産履歴から、過去ピーク比の最大下落率を計算する。履歴が少ないDay1は下落がなければ0.00%。",
        "open_positions_count": len(paper_trade_log["positions"]),
    }


def _calculate_max_drawdown(asset_history: list[float]) -> float:
    peak = asset_history[0]
    max_drawdown = 0.0
    for asset in asset_history:
        peak = max(peak, asset)
        drawdown = (asset - peak) / peak
        max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown
