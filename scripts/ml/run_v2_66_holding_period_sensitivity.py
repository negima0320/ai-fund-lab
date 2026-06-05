#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.data_loader import JQuantsDataLoader


PROFILE_ID = "rookie_dealer_02_v2_66_ml_ranked"
DEFAULT_HOLDING_DAYS = [5, 10, 15, 20, 25, 30]
INITIAL_CAPITAL = 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc holding-period sensitivity for v2_66 ML ranked trades.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--holding-days", nargs="*", type=int, default=DEFAULT_HOLDING_DAYS)
    args = parser.parse_args()

    started_at = time.perf_counter()
    config = _load_profile_config()
    trades = _load_trades(args.start, args.end)
    prices = _load_prices_for_trades(trades, max(args.holding_days))
    simulations = []
    details = []
    for max_holding_days in args.holding_days:
        simulated = _simulate_holding_days(trades, prices, config, max_holding_days)
        simulations.append(_summarize(simulated, max_holding_days))
        details.append(simulated)

    analysis = {
        "profile": PROFILE_ID,
        "period": {"start": args.start, "end": args.end},
        "method": {
            "type": "post_hoc_exit_resimulation",
            "note": (
                "Existing v2_66 entries are fixed and only exit timing is recomputed from price cache. "
                "This is faster than a full backtest, but it does not model capital/position-slot changes that "
                "could alter future entries under longer holding periods."
            ),
            "prediction_source": "existing v2_66 entries already include walk-forward ML-ranked selection",
            "price_source": "data/cache/jquants/prices",
        },
        "parameters": {
            "stop_loss_rate": float(config["trading"].get("stop_loss_rate", -0.03)),
            "take_profit_rate": float(config["trading"].get("take_profit_rate", 0.06)),
            "tax_rate": float((config.get("costs") or {}).get("tax_rate", 0.20315)),
            "initial_capital": INITIAL_CAPITAL,
        },
        "elapsed_sec": round(time.perf_counter() - started_at, 2),
        "results": simulations,
        "best_by_net_profit": max(simulations, key=lambda row: row.get("net_profit") or -10**18),
        "best_by_profit_factor": max(simulations, key=lambda row: row.get("profit_factor") or -10**18),
        "best_by_drawdown": max(simulations, key=lambda row: row.get("max_drawdown") if row.get("max_drawdown") is not None else -10**18),
        "baseline_actual": _baseline_actual(trades),
    }
    paths = _save(analysis, pd.concat(details, ignore_index=True), args.start, args.end)
    print(f"markdown={paths['markdown']}")
    print(f"json={paths['json']}")
    print(f"trades_csv={paths['trades_csv']}")
    for row in simulations:
        print(
            f"hold={row['max_holding_days']} final_assets={row['final_assets']} "
            f"net_profit={row['net_profit']} win_rate={row['win_rate']} "
            f"pf={row['profit_factor']} dd={row['max_drawdown']} "
            f"trades={row['total_trades']} avg_hold={row['average_holding_days']}"
        )


def _load_profile_config() -> dict[str, Any]:
    base = yaml.safe_load((ROOT / "config" / "rookie_dealer.yaml").read_text(encoding="utf-8"))
    profile = yaml.safe_load((ROOT / "config" / "profiles" / f"{PROFILE_ID}.yaml").read_text(encoding="utf-8"))
    merged = {**base, **profile}
    merged["trading"] = {**base.get("trading", {}), **profile.get("trading", {})}
    merged["costs"] = {**base.get("costs", {}), **profile.get("costs", {})}
    return merged


def _load_trades(start: str, end: str) -> pd.DataFrame:
    path = ROOT / "logs" / "backtests" / PROFILE_ID / f"{start}_to_{end}" / "trades.csv"
    df = pd.read_csv(path)
    if "action" in df.columns:
        df = df[df["action"].astype(str).eq("SELL")].copy()
    df["code"] = df["code"].astype(str)
    for column in ["signal_date", "entry_date", "exit_date"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    numeric_columns = ["entry_price", "exit_price", "shares", "net_profit", "net_profit_rate", "holding_days"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["entry_date", "entry_price", "shares"]).reset_index(drop=True)


def _load_prices_for_trades(trades: pd.DataFrame, max_holding_days: int) -> pd.DataFrame:
    start = trades["entry_date"].min().strftime("%Y-%m-%d")
    end = (trades["entry_date"].max() + pd.Timedelta(days=max_holding_days * 3)).strftime("%Y-%m-%d")
    loader = JQuantsDataLoader(ROOT / "data" / "cache" / "jquants")
    prices = loader.load_prices(start, end)
    codes = set(trades["code"].astype(str))
    prices["code"] = prices["code"].astype(str)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    return prices[prices["code"].isin(codes)].sort_values(["date", "code"]).reset_index(drop=True)


def _simulate_holding_days(trades: pd.DataFrame, prices: pd.DataFrame, config: dict[str, Any], max_holding_days: int) -> pd.DataFrame:
    trading_dates = sorted(pd.Timestamp(value) for value in prices["date"].dropna().unique())
    date_index = {date: index for index, date in enumerate(trading_dates)}
    price_by_key = {(str(row.code), pd.Timestamp(row.date)): row for row in prices.itertuples(index=False)}
    rows = []
    for trade in trades.to_dict("records"):
        rows.append(_simulate_trade(trade, trading_dates, date_index, price_by_key, config, max_holding_days))
    return pd.DataFrame(rows)


def _simulate_trade(
    trade: dict[str, Any],
    trading_dates: list[pd.Timestamp],
    date_index: dict[pd.Timestamp, int],
    price_by_key: dict[tuple[str, pd.Timestamp], Any],
    config: dict[str, Any],
    max_holding_days: int,
) -> dict[str, Any]:
    entry_date = pd.Timestamp(trade["entry_date"])
    entry_index = date_index.get(entry_date)
    entry_price = float(trade["entry_price"])
    shares = float(trade["shares"])
    stop_loss_rate = float(config["trading"].get("stop_loss_rate", -0.03))
    take_profit_rate = float(config["trading"].get("take_profit_rate", 0.06))
    stop_price = round(entry_price * (1 + stop_loss_rate), 4)

    exit_date = pd.Timestamp(trade.get("exit_date")) if pd.notna(trade.get("exit_date")) else entry_date
    exit_price = _float(trade.get("exit_price"), entry_price)
    exit_reason = str(trade.get("exit_reason") or "actual_exit")
    holding_days = int(_float(trade.get("holding_days"), 1))
    data_end_used = False

    if entry_index is not None:
        last_index = min(entry_index + max_holding_days - 1, len(trading_dates) - 1)
        for index in range(entry_index + 1, last_index + 1):
            date_value = trading_dates[index]
            holding_day = index - entry_index + 1
            market = price_by_key.get((str(trade["code"]), date_value))
            if market is None:
                continue
            close_price = _float(getattr(market, "close"), None)
            low_price = _float(getattr(market, "low"), None)
            if close_price is None:
                continue
            mark_profit_rate = (close_price - entry_price) / entry_price if entry_price else 0.0
            if holding_day >= 2 and low_price is not None and low_price <= stop_price:
                exit_date = date_value
                exit_price = stop_price
                exit_reason = "損切り"
                holding_days = holding_day
                break
            if holding_day >= 2 and mark_profit_rate >= take_profit_rate:
                exit_date = date_value
                exit_price = close_price
                exit_reason = "利確"
                holding_days = holding_day
                break
            if holding_day >= max_holding_days:
                exit_date = date_value
                exit_price = close_price
                exit_reason = "最大保有期間到達"
                holding_days = holding_day
                break
        else:
            if last_index < entry_index + max_holding_days - 1 and last_index >= entry_index:
                market = price_by_key.get((str(trade["code"]), trading_dates[last_index]))
                if market is not None:
                    exit_date = trading_dates[last_index]
                    exit_price = _float(getattr(market, "close"), exit_price)
                    exit_reason = "data_end"
                    holding_days = last_index - entry_index + 1
                    data_end_used = True

    gross_profit = (exit_price - entry_price) * shares
    tax_rate = float((config.get("costs") or {}).get("tax_rate", 0.20315))
    estimated_tax = gross_profit * tax_rate if gross_profit > 0 else 0.0
    net_profit = gross_profit - estimated_tax
    entry_notional = entry_price * shares
    row = {
        "max_holding_days": max_holding_days,
        "code": str(trade["code"]),
        "name": trade.get("name"),
        "signal_date": _date_text(trade.get("signal_date")),
        "entry_date": _date_text(entry_date),
        "simulated_exit_date": _date_text(exit_date),
        "actual_exit_date": _date_text(trade.get("exit_date")),
        "entry_price": round(entry_price, 4),
        "simulated_exit_price": round(exit_price, 4),
        "actual_exit_price": _round(trade.get("exit_price")),
        "shares": shares,
        "simulated_exit_reason": exit_reason,
        "actual_exit_reason": trade.get("exit_reason"),
        "simulated_holding_days": holding_days,
        "actual_holding_days": _round(trade.get("holding_days"), 0),
        "gross_profit": round(gross_profit, 2),
        "estimated_tax": round(estimated_tax, 2),
        "net_profit": round(net_profit, 2),
        "net_profit_rate": round(net_profit / entry_notional, 4) if entry_notional else 0.0,
        "actual_net_profit": _round(trade.get("net_profit")),
        "profit_delta_vs_actual": round(net_profit - _float(trade.get("net_profit"), 0.0), 2),
        "data_end_used": data_end_used,
    }
    return row


def _summarize(df: pd.DataFrame, max_holding_days: int) -> dict[str, Any]:
    profits = pd.to_numeric(df["net_profit"], errors="coerce").fillna(0.0)
    wins = profits > 0
    gross_profit = float(profits[wins].sum())
    gross_loss = float(-profits[profits < 0].sum())
    ordered = df.assign(_exit_date=pd.to_datetime(df["simulated_exit_date"], errors="coerce")).sort_values("_exit_date")
    equity = INITIAL_CAPITAL + pd.to_numeric(ordered["net_profit"], errors="coerce").fillna(0.0).cumsum()
    drawdown = (equity - equity.cummax()) / equity.cummax()
    monthly = _monthly_summary(df)
    return {
        "max_holding_days": max_holding_days,
        "final_assets": round(INITIAL_CAPITAL + float(profits.sum()), 2),
        "net_profit": round(float(profits.sum()), 2),
        "win_rate": round(float(wins.mean()), 4) if len(wins) else None,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss else None,
        "max_drawdown": round(float(drawdown.min()), 4) if len(drawdown) else None,
        "total_trades": int(len(df)),
        "average_holding_days": round(float(pd.to_numeric(df["simulated_holding_days"], errors="coerce").mean()), 4),
        "monthly_win_rate": round(monthly["monthly_win_rate"], 4) if monthly["monthly_win_rate"] is not None else None,
        "losing_months": monthly["losing_months"],
        "worst_month": monthly["worst_month"],
        "best_month": monthly["best_month"],
        "data_end_exit_count": int(df["data_end_used"].sum()),
    }


def _monthly_summary(df: pd.DataFrame) -> dict[str, Any]:
    monthly = df.copy()
    monthly["simulated_exit_date"] = pd.to_datetime(monthly["simulated_exit_date"], errors="coerce")
    monthly["month"] = monthly["simulated_exit_date"].dt.to_period("M").astype(str)
    grouped = monthly.groupby("month", dropna=True)["net_profit"].sum().reset_index()
    if grouped.empty:
        return {"monthly_win_rate": None, "losing_months": 0, "worst_month": None, "best_month": None}
    grouped["is_win"] = grouped["net_profit"] > 0
    worst = grouped.sort_values("net_profit").iloc[0]
    best = grouped.sort_values("net_profit", ascending=False).iloc[0]
    return {
        "monthly_win_rate": float(grouped["is_win"].mean()),
        "losing_months": int((grouped["net_profit"] < 0).sum()),
        "worst_month": {"month": str(worst["month"]), "net_profit": round(float(worst["net_profit"]), 2)},
        "best_month": {"month": str(best["month"]), "net_profit": round(float(best["net_profit"]), 2)},
    }


def _baseline_actual(trades: pd.DataFrame) -> dict[str, Any]:
    df = trades.copy()
    df["net_profit"] = pd.to_numeric(df["net_profit"], errors="coerce").fillna(0.0)
    df["simulated_exit_date"] = df["exit_date"]
    df["simulated_holding_days"] = df["holding_days"]
    baseline = _summarize(df.assign(data_end_used=False), 5)
    baseline["source"] = "actual_v2_66_backtest"
    return baseline


def _save(analysis: dict[str, Any], detail: pd.DataFrame, start: str, end: str) -> dict[str, Path]:
    report_root = ROOT / "reports" / "ml"
    report_root.mkdir(parents=True, exist_ok=True)
    stem = "v2_66_holding_period_sensitivity_2023-01_to_2026-05"
    markdown = report_root / f"{stem}.md"
    json_path = report_root / f"{stem}.json"
    trades_csv = report_root / f"{stem}_trades.csv"
    markdown.write_text(_format_markdown(analysis), encoding="utf-8")
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    detail.to_csv(trades_csv, index=False)
    return {"markdown": markdown, "json": json_path, "trades_csv": trades_csv}


def _format_markdown(analysis: dict[str, Any]) -> str:
    columns = [
        "max_holding_days",
        "final_assets",
        "net_profit",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "total_trades",
        "average_holding_days",
        "monthly_win_rate",
        "losing_months",
        "worst_month",
        "best_month",
        "data_end_exit_count",
    ]
    lines = [
        "# v2_66 ML Ranked Holding Period Sensitivity",
        "",
        f"- profile: `{analysis['profile']}`",
        f"- period: `{analysis['period']['start']}` to `{analysis['period']['end']}`",
        f"- method: `{analysis['method']['type']}`",
        f"- note: {analysis['method']['note']}",
        "- no trading logic change, no API refetch, no current model prediction generation",
        "",
        "## Baseline Actual Backtest",
        "",
        _table([analysis["baseline_actual"]], columns),
        "",
        "## Holding Period Results",
        "",
        _table(analysis["results"], columns),
        "",
        "## Best Conditions",
        "",
        f"- best_by_net_profit: `{analysis['best_by_net_profit']['max_holding_days']}` days",
        f"- best_by_profit_factor: `{analysis['best_by_profit_factor']['max_holding_days']}` days",
        f"- best_by_drawdown: `{analysis['best_by_drawdown']['max_holding_days']}` days",
        "",
    ]
    return "\n".join(lines)


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, dict):
        return f"{value.get('month')} ({value.get('net_profit')})"
    if value is None:
        return ""
    return str(value)


def _float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 2) -> float | None:
    value = _float(value, None)
    return round(value, digits) if value is not None else None


def _date_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
