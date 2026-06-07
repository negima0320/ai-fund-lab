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

from profile_loader import load_profile


PERIOD = "2023-01-01_to_2026-05-31"
REPORT_STEM = "portfolio_manager_phase4c_high_pm_min_hold_2023-01_to_2026-05"
PROFILES = {
    "v2_78_w025": "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025",
    "v2_79_min_hold_5d": "rookie_dealer_02_v2_79_high_pm_min_hold_5d",
    "v2_79_min_hold_7d": "rookie_dealer_02_v2_79_high_pm_min_hold_7d",
}


def _profile_dir(root: Path, profile: str) -> Path:
    return root / "logs" / "backtests" / profile / PERIOD


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _profit_factor(values: pd.Series) -> float | None:
    profits = pd.to_numeric(values, errors="coerce").dropna()
    if profits.empty:
        return None
    gross_profit = float(profits[profits > 0].sum())
    gross_loss = abs(float(profits[profits < 0].sum()))
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def _win_rate(values: pd.Series) -> float | None:
    profits = pd.to_numeric(values, errors="coerce").dropna()
    if profits.empty:
        return None
    return float((profits > 0).mean())


def _monthly_win_rate(trades: pd.DataFrame) -> float | None:
    sells = _sell_trades(trades)
    if sells.empty or "exit_date" not in sells.columns:
        return None
    months = pd.to_datetime(sells["exit_date"], errors="coerce").dt.strftime("%Y-%m")
    monthly = pd.to_numeric(sells["net_profit"], errors="coerce").groupby(months).sum()
    monthly = monthly[monthly.index.notna()]
    if monthly.empty:
        return None
    return float((monthly > 0).mean())


def _sell_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "action" not in trades.columns:
        return pd.DataFrame()
    return trades[trades["action"].fillna("").astype(str).eq("SELL")].copy()


def _truthy(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    return series.reindex(index).fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def _skip_counts(audit: pd.DataFrame) -> dict[str, int]:
    if audit.empty:
        return {}
    reason = audit.get("skip_reason", pd.Series(dtype=str)).fillna("").astype(str)
    scale = audit.get("scale_reason", pd.Series(dtype=str)).fillna("").astype(str)
    combined = reason.where(reason.ne(""), scale)
    return {str(key): int(value) for key, value in combined[combined.ne("")].value_counts().items()}


def _capital_distribution(daily: pd.DataFrame) -> dict[str, Any]:
    empty = {
        "average_capital_utilization": None,
        "median_capital_utilization": None,
        "cash_idle_days": None,
        "average_holding_count": None,
    }
    if daily.empty or not {"cash", "positions_value", "total_assets"}.issubset(daily.columns):
        return empty
    total = pd.to_numeric(daily["total_assets"], errors="coerce").replace(0, pd.NA)
    positions = pd.to_numeric(daily["positions_value"], errors="coerce")
    utilization = (positions / total).dropna()
    holding = pd.to_numeric(daily.get("open_positions_count"), errors="coerce") if "open_positions_count" in daily.columns else pd.Series(dtype=float)
    return {
        "average_capital_utilization": float(utilization.mean()) if not utilization.empty else None,
        "median_capital_utilization": float(utilization.median()) if not utilization.empty else None,
        "cash_idle_days": int((utilization <= 0.05).sum()) if not utilization.empty else None,
        "average_holding_count": float(holding.dropna().mean()) if not holding.dropna().empty else None,
    }


def _high_pm_stats(trades: pd.DataFrame) -> dict[str, Any]:
    sells = _sell_trades(trades)
    if sells.empty or "pm_multiplier" not in sells.columns:
        return {
            "high_pm_trade_count": 0,
            "high_pm_net_profit": None,
            "high_pm_profit_factor": None,
            "high_pm_win_rate": None,
            "high_pm_average_holding_days": None,
            "high_pm_min_hold_blocked_exit_count": 0,
        }
    high_pm = sells[pd.to_numeric(sells["pm_multiplier"], errors="coerce") >= 1.15].copy()
    profits = pd.to_numeric(high_pm.get("net_profit"), errors="coerce")
    holding = pd.to_numeric(high_pm.get("holding_days"), errors="coerce")
    if "high_pm_min_hold_blocked_exit_count" in high_pm.columns:
        blocked_count = int(pd.to_numeric(high_pm["high_pm_min_hold_blocked_exit_count"], errors="coerce").fillna(0).sum())
    else:
        blocked_count = int(_truthy(high_pm.get("high_pm_min_hold_blocked_exit"), high_pm.index).sum())
    return {
        "high_pm_trade_count": int(profits.notna().sum()),
        "high_pm_net_profit": float(profits.sum()) if not profits.empty else None,
        "high_pm_profit_factor": _profit_factor(profits),
        "high_pm_win_rate": _win_rate(profits),
        "high_pm_average_holding_days": float(holding.dropna().mean()) if not holding.dropna().empty else None,
        "high_pm_min_hold_blocked_exit_count": blocked_count,
    }


def build_report(root: Path = ROOT) -> dict[str, Any]:
    comparison: list[dict[str, Any]] = []
    for label, profile in PROFILES.items():
        base = _profile_dir(root, profile)
        summary = _read_summary(base / "backtest_summary.json")
        daily = _read_csv(base / "summary.csv")
        trades = _read_csv(base / "trades.csv")
        audit = _read_csv(base / "purchase_audit.csv")
        config = load_profile(profile)
        policy = config.get("portfolio_manager_ai_sizing", {})
        skip_counts = _skip_counts(audit)
        row = {
            "variant": label,
            "profile": profile,
            "status": "ok" if summary else "missing_backtest_logs",
            "high_pm_min_hold_enabled": bool(policy.get("high_pm_min_hold_enabled", False)),
            "high_pm_min_hold_days": policy.get("high_pm_min_hold_days"),
            "net_profit": summary.get("net_cumulative_profit"),
            "profit_factor": summary.get("profit_factor"),
            "max_drawdown": summary.get("max_drawdown"),
            "win_rate": summary.get("win_rate"),
            "total_trades": summary.get("closed_trades_count"),
            "monthly_win_rate": _monthly_win_rate(trades),
            "average_holding_days": summary.get("average_holding_days"),
            **_capital_distribution(daily),
            **_high_pm_stats(trades),
            "selected_but_not_affordable": skip_counts.get("selected_but_not_affordable", 0),
            "insufficient_available_cash": skip_counts.get("insufficient_available_cash", 0),
        }
        comparison.append(row)
    return {
        "purpose": "Portfolio Manager AI Phase 4-C high PM minimum hold profile comparison",
        "period": PERIOD,
        "constraints": {
            "api_refetch": False,
            "openai_api": False,
            "historical_predictions_source": "data/ml/walk_forward_predictions/",
            "current_model_historical_regeneration": False,
            "selected_count_in_day_used": False,
            "live_order_placement": False,
        },
        "profiles": PROFILES,
        "comparison": comparison,
    }


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def format_markdown(result: dict[str, Any]) -> str:
    columns = [
        "variant",
        "status",
        "high_pm_min_hold_enabled",
        "high_pm_min_hold_days",
        "net_profit",
        "profit_factor",
        "max_drawdown",
        "win_rate",
        "total_trades",
        "monthly_win_rate",
        "average_capital_utilization",
        "median_capital_utilization",
        "average_holding_days",
        "average_holding_count",
        "high_pm_trade_count",
        "high_pm_net_profit",
        "high_pm_profit_factor",
        "high_pm_win_rate",
        "high_pm_average_holding_days",
        "high_pm_min_hold_blocked_exit_count",
        "selected_but_not_affordable",
        "insufficient_available_cash",
        "cash_idle_days",
    ]
    return "\n".join(
        [
            "# Portfolio Manager AI Phase 4-C High PM Minimum Hold",
            "",
            "## Scope",
            "",
            f"- period: `{result['period']}`",
            "- reads existing backtest logs only",
            "- no API refetch / no OpenAI API / no current model historical regeneration",
            "- `selected_count_in_day` used: `False`",
            "",
            "## Comparison",
            "",
            _table(result["comparison"], columns),
            "",
            "## Implementation Note",
            "",
            "The new profiles only suppress Exit AI exits for positions with `pm_multiplier >= 1.15` before the configured minimum holding days. Existing stop loss, take profit, max holding, and forced exits are not suppressed.",
            "",
        ]
    )


def save_report(result: dict[str, Any], root: Path = ROOT) -> tuple[Path, Path]:
    report_dir = root / "reports" / "ml"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = report_dir / f"{REPORT_STEM}.md"
    json_path = report_dir / f"{REPORT_STEM}.json"
    markdown.write_text(format_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return markdown, json_path


def main() -> int:
    result = build_report(ROOT)
    markdown, json_path = save_report(result, ROOT)
    print(f"markdown={markdown}")
    print(f"json={json_path}")
    for row in result["comparison"]:
        print(
            f"{row['variant']}: status={row['status']} net_profit={row['net_profit']} "
            f"pf={row['profit_factor']} dd={row['max_drawdown']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
