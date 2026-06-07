#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase6b_bear_market_winner_audit import Phase6BBearMarketWinnerAudit
from profile_loader import load_profile


PERIOD = "2023-01-01_to_2026-05-31"
REPORT_STEM = "portfolio_manager_phase6e_bear_booster_2023-01_to_2026-05"
PROFILES = {
    "v2_78_baseline": "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025",
    "v2_81_bear_pm115_booster_50": "rookie_dealer_02_v2_81_bear_pm115_booster_50",
}


def _profile_dir(profile: str) -> Path:
    return ROOT / "logs" / "backtests" / profile / PERIOD


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _truthy(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    return series.reindex(index).fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})


def _sell_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    if "action" not in trades.columns:
        return trades.copy()
    return trades[trades["action"].fillna("").astype(str).eq("SELL")].copy()


def _profit_factor(values: pd.Series | None) -> float | None:
    profits = _numeric(values).dropna()
    if profits.empty:
        return None
    gross_profit = float(profits[profits > 0].sum())
    gross_loss = abs(float(profits[profits < 0].sum()))
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def _win_rate(values: pd.Series | None) -> float | None:
    profits = _numeric(values).dropna()
    if profits.empty:
        return None
    return float((profits > 0).mean())


def _mean(series: pd.Series | None) -> float | None:
    values = _numeric(series).dropna()
    return float(values.mean()) if not values.empty else None


def _monthly_win_rate(trades: pd.DataFrame) -> float | None:
    sells = _sell_trades(trades)
    if sells.empty or "exit_date" not in sells.columns:
        return None
    months = pd.to_datetime(sells["exit_date"], errors="coerce").dt.strftime("%Y-%m")
    monthly = _numeric(sells.get("net_profit")).groupby(months).sum()
    monthly = monthly[monthly.index.notna()]
    if monthly.empty:
        return None
    return float((monthly > 0).mean())


def _capital_distribution(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or not {"positions_value", "total_assets"}.issubset(daily.columns):
        return {"average_capital_utilization": None}
    total = _numeric(daily.get("total_assets")).replace(0, pd.NA)
    positions = _numeric(daily.get("positions_value"))
    utilization = (positions / total).dropna()
    return {"average_capital_utilization": float(utilization.mean()) if not utilization.empty else None}


def _skip_counts(audit: pd.DataFrame) -> dict[str, int]:
    if audit.empty:
        return {}
    reason = audit.get("skip_reason", pd.Series(dtype=str)).fillna("").astype(str)
    scale = audit.get("scale_reason", pd.Series(dtype=str)).fillna("").astype(str)
    cap = audit.get("pm_per_code_cap_reason", pd.Series(dtype=str)).fillna("").astype(str)
    combined = reason.where(reason.ne(""), scale.where(scale.ne(""), cap))
    return {str(key): int(value) for key, value in combined[combined.ne("")].value_counts().items()}


def _enriched_sells(profile: str) -> pd.DataFrame:
    audit = Phase6BBearMarketWinnerAudit(ROOT, profile=profile, period=PERIOD)
    trades = audit._load_trades()
    purchase = audit._load_purchase_audit()
    listed = audit._load_listed_info()
    regime = audit._load_regime()
    return audit._enrich_trades(trades, purchase, listed, regime)


def _bear_stats(enriched: pd.DataFrame) -> dict[str, Any]:
    bear = enriched[enriched.get("regime", pd.Series(dtype=str)).eq("Bear")].copy() if not enriched.empty else pd.DataFrame()
    profits = _numeric(bear.get("profit"))
    return {
        "bear_trade_count": int(len(bear)),
        "bear_profit": float(profits.sum()) if not profits.empty else 0.0,
        "bear_profit_factor": _profit_factor(profits),
        "bear_win_rate": _win_rate(profits),
    }


def _booster_stats(sells: pd.DataFrame) -> dict[str, Any]:
    if sells.empty:
        return {
            "bear_pm_booster_applied_count": 0,
            "bear_pm_booster_total_incremental_amount": 0.0,
            "boosted_trades_profit": None,
            "boosted_trades_win_rate": None,
            "single_code_concentration": None,
        }
    applied = _truthy(sells.get("bear_pm_booster_applied"), sells.index)
    boosted = sells[applied].copy()
    before = _numeric(boosted.get("bear_pm_booster_before_amount"))
    after = _numeric(boosted.get("bear_pm_booster_after_amount"))
    profit = _numeric(boosted.get("net_profit"))
    by_code = profit.groupby(boosted.get("code")).sum() if not boosted.empty and "code" in boosted.columns else pd.Series(dtype=float)
    total_abs = float(by_code.abs().sum()) if not by_code.empty else 0.0
    return {
        "bear_pm_booster_applied_count": int(applied.sum()),
        "bear_pm_booster_total_incremental_amount": float((after - before).clip(lower=0).sum()) if not boosted.empty else 0.0,
        "boosted_trades_profit": float(profit.sum()) if not profit.empty else None,
        "boosted_trades_win_rate": _win_rate(profit),
        "single_code_concentration": float(by_code.abs().max() / total_abs) if total_abs else None,
    }


def _pm080_bear_stats(enriched: pd.DataFrame) -> dict[str, Any]:
    if enriched.empty:
        return {"pm080_bear_trade_count": 0, "pm080_bear_profit": 0.0, "pm080_bear_profit_preserved": None}
    pm = _numeric(enriched.get("pm_multiplier")).round(2)
    bear = enriched.get("regime", pd.Series(dtype=str)).eq("Bear")
    rows = enriched[bear & pm.eq(0.80)]
    profit = float(_numeric(rows.get("profit")).sum()) if not rows.empty else 0.0
    return {
        "pm080_bear_trade_count": int(len(rows)),
        "pm080_bear_profit": profit,
        "pm080_bear_profit_preserved": profit > 0,
    }


def _dd_period(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or "total_assets" not in daily.columns:
        return {"dd_start": "", "dd_trough": ""}
    assets = _numeric(daily.get("total_assets"))
    if assets.empty:
        return {"dd_start": "", "dd_trough": ""}
    peak = assets.cummax()
    dd = assets / peak - 1.0
    trough_idx = dd.idxmin()
    start_idx = assets.loc[:trough_idx].idxmax()
    date_col = daily.get("date", pd.Series("", index=daily.index))
    return {"dd_start": str(date_col.loc[start_idx]), "dd_trough": str(date_col.loc[trough_idx])}


def _profile_row(label: str, profile: str) -> dict[str, Any]:
    base = _profile_dir(profile)
    summary = _read_json(base / "backtest_summary.json")
    trades = _read_csv(base / "trades.csv")
    daily = _read_csv(base / "summary.csv")
    audit = _read_csv(base / "purchase_audit.csv")
    config = load_profile(profile)
    sells = _sell_trades(trades)
    enriched = _enriched_sells(profile) if (base / "trades.csv").exists() else pd.DataFrame()
    skip_counts = _skip_counts(audit)
    booster_policy = config.get("bear_pm_booster", {})
    return {
        "variant": label,
        "profile": profile,
        "status": "ok" if summary else "missing_backtest_logs",
        "bear_pm_booster_enabled": bool(booster_policy.get("bear_pm_booster_enabled", booster_policy.get("enabled", False))),
        "bear_pm_booster_min_pm_multiplier": booster_policy.get("min_pm_multiplier"),
        "bear_pm_booster_multiplier": booster_policy.get("booster_multiplier"),
        "net_profit": summary.get("net_cumulative_profit"),
        "profit_factor": summary.get("profit_factor"),
        "max_drawdown": summary.get("max_drawdown"),
        "win_rate": summary.get("win_rate"),
        "monthly_win_rate": _monthly_win_rate(trades),
        "total_trades": summary.get("closed_trades_count"),
        "average_holding_days": summary.get("average_holding_days"),
        **_capital_distribution(daily),
        "selected_but_not_affordable": skip_counts.get("selected_but_not_affordable", 0),
        "insufficient_available_cash": skip_counts.get("insufficient_available_cash", 0),
        "per_code_cap_skip_or_reduction_count": int(_truthy(audit.get("pm_per_code_cap_skip"), audit.index).sum()) if not audit.empty else 0,
        **_bear_stats(enriched),
        **_booster_stats(sells),
        **_pm080_bear_stats(enriched),
        **_dd_period(daily),
    }


def build_report(root: Path = ROOT) -> dict[str, Any]:
    rows = [_profile_row(label, profile) for label, profile in PROFILES.items()]
    return {
        "phase": "6-E",
        "purpose": "Bear PM>=1.15 booster profile comparison",
        "period": PERIOD,
        "constraints": {
            "report_only": True,
            "full_backtest_executed_by_report": False,
            "full_pytest_executed_by_report": False,
            "current_model_overwritten": False,
            "api_refetch": False,
            "openai_used": False,
        },
        "profiles": rows,
        "adoption_criteria": {
            "net_profit": "improve vs v2_78",
            "profit_factor": ">= 2.5",
            "max_drawdown": "within -10%",
            "win_rate": ">= 53%",
            "bear_profit": "improve vs v2_78",
            "pm080_bear_profit": "preserved",
            "concentration": "acceptable",
        },
    }


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows_"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_markdown(result: dict[str, Any]) -> str:
    columns = [
        "variant",
        "status",
        "net_profit",
        "profit_factor",
        "max_drawdown",
        "win_rate",
        "monthly_win_rate",
        "total_trades",
        "average_holding_days",
        "average_capital_utilization",
        "selected_but_not_affordable",
        "insufficient_available_cash",
        "per_code_cap_skip_or_reduction_count",
        "bear_trade_count",
        "bear_profit",
        "bear_profit_factor",
        "bear_win_rate",
        "bear_pm_booster_applied_count",
        "bear_pm_booster_total_incremental_amount",
        "boosted_trades_profit",
        "boosted_trades_win_rate",
        "pm080_bear_profit_preserved",
        "pm080_bear_profit",
        "single_code_concentration",
        "dd_start",
        "dd_trough",
    ]
    lines = [
        "# Portfolio Manager Phase 6-E Bear PM Booster",
        "",
        "## Scope",
        "",
        "- report only",
        "- no backtest is executed by this script",
        "- compares existing v2_78 and v2_81 backtest logs",
        "",
        "## Profile Comparison",
        "",
        _table(result["profiles"], columns),
        "",
        "## Adoption Criteria",
        "",
        _table([result["adoption_criteria"]], list(result["adoption_criteria"].keys())),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    result = build_report(ROOT)
    report_dir = ROOT / "reports" / "ml"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{REPORT_STEM}.json"
    md_path = report_dir / f"{REPORT_STEM}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(format_markdown(result), encoding="utf-8")
    print(f"generated markdown={md_path}")
    print(f"generated json={json_path}")


if __name__ == "__main__":
    main()
