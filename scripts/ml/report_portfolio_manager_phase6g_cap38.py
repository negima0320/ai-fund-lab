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
REPORT_STEM = "portfolio_manager_phase6g_cap38_2023-01_to_2026-05"
PROFILES = {
    "v2_78_baseline": "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025",
    "v2_82_cap38": "rookie_dealer_02_v2_82_cap38",
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


def _concentration_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    sells = _sell_trades(trades)
    if sells.empty or "code" not in sells.columns:
        return {"single_code_profit_concentration": None, "top5_code_profit_concentration": None}
    by_code = _numeric(sells.get("net_profit")).groupby(sells["code"].astype(str)).sum()
    total_abs = float(by_code.abs().sum())
    if total_abs == 0:
        return {"single_code_profit_concentration": None, "top5_code_profit_concentration": None}
    abs_profit = by_code.abs().sort_values(ascending=False)
    return {
        "single_code_profit_concentration": float(abs_profit.iloc[0] / total_abs) if not abs_profit.empty else None,
        "top5_code_profit_concentration": float(abs_profit.head(5).sum() / total_abs),
    }


def _cagr(summary: dict[str, Any]) -> float | None:
    initial = summary.get("initial_assets")
    final = summary.get("final_assets")
    days = summary.get("processed_days")
    try:
        initial_f = float(initial)
        final_f = float(final)
        years = float(days) / 245.0
    except (TypeError, ValueError):
        return None
    if initial_f <= 0 or final_f <= 0 or years <= 0:
        return None
    return (final_f / initial_f) ** (1 / years) - 1


def _profile_row(label: str, profile: str) -> dict[str, Any]:
    base = _profile_dir(profile)
    summary = _read_json(base / "backtest_summary.json")
    trades = _read_csv(base / "trades.csv")
    daily = _read_csv(base / "summary.csv")
    audit = _read_csv(base / "purchase_audit.csv")
    config = load_profile(profile)
    sizing = config.get("portfolio_manager_ai_sizing", {})
    skip_counts = _skip_counts(audit)
    return {
        "variant": label,
        "profile": profile,
        "status": "ok" if summary else "missing_backtest_logs",
        "per_code_exposure_cap_rate": sizing.get("per_code_exposure_cap_rate"),
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
        **_concentration_metrics(trades),
        "initial_assets": summary.get("initial_assets"),
        "final_assets": summary.get("final_assets"),
        "cagr": summary.get("cagr", _cagr(summary)),
    }


def build_report(root: Path = ROOT) -> dict[str, Any]:
    rows = [_profile_row(label, profile) for label, profile in PROFILES.items()]
    return {
        "phase": "6-G",
        "purpose": "per-code exposure cap 38% profile comparison",
        "period": PERIOD,
        "constraints": {
            "report_only": True,
            "full_backtest_executed_by_report": False,
            "full_pytest_executed_by_report": False,
            "current_model_overwritten": False,
            "api_refetch": False,
            "openai_used": False,
            "bear_booster_mixed": False,
            "exit_ai_v2_mixed": False,
        },
        "profiles": rows,
        "comparison_targets": PROFILES,
        "adoption_criteria": {
            "net_profit": "improve vs v2_78",
            "profit_factor": "maintain around v2_78 / above 2.5",
            "max_drawdown": "avoid material deterioration",
            "win_rate": "maintain around v2_78 / above 53%",
            "capital_utilization": "improve without excessive concentration",
            "concentration": "single-code and top5 concentration acceptable",
        },
    }


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows_"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def format_markdown(result: dict[str, Any]) -> str:
    columns = [
        "variant",
        "status",
        "per_code_exposure_cap_rate",
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
        "single_code_profit_concentration",
        "top5_code_profit_concentration",
        "initial_assets",
        "final_assets",
        "cagr",
    ]
    lines = [
        "# Portfolio Manager Phase 6-G Cap 38%",
        "",
        "## Scope",
        "",
        "- report only",
        "- no backtest is executed by this script",
        "- compares existing v2_78 and v2_82 backtest logs",
        "- v2_82 changes only per-code exposure cap from 0.30 to 0.38",
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
