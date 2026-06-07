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
REPORT_STEM = "portfolio_manager_phase5h_exit_ai_v2_gate_2023-01_to_2026-05"
PROFILES = {
    "v2_78_baseline": "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025",
    "v2_80_conservative_gate": "rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate",
    "v2_80_high_pm_safe": "rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate_high_pm_safe",
}


def _profile_dir(root: Path, profile: str) -> Path:
    return root / "logs" / "backtests" / profile / PERIOD


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


def _truthy(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    return series.reindex(index).fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _sell_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    if "action" not in trades.columns:
        return trades.copy()
    return trades[trades["action"].fillna("").astype(str).eq("SELL")].copy()


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
    combined = reason.where(reason.ne(""), scale)
    return {str(key): int(value) for key, value in combined[combined.ne("")].value_counts().items()}


def _exit_ai_v2_stats(trades: pd.DataFrame) -> dict[str, Any]:
    sells = _sell_trades(trades)
    if sells.empty:
        return {
            "exit_ai_v2_prediction_coverage": None,
            "exit_ai_v2_gate_signal_count": 0,
            "exit_ai_v2_used_as_exit_trigger_count": 0,
            "exit_ai_v2_triggered_profit": None,
            "exit_ai_v2_triggered_win_rate": None,
            "high_pm_exit_count": 0,
            "high_pm_exit_ai_v2_trigger_count": 0,
            "post_exit_return_5d_for_v2_triggered": None,
            "post_exit_return_10d_for_v2_triggered": None,
            "post_exit_return_20d_for_v2_triggered": None,
        }
    index = sells.index
    available = _truthy(sells.get("exit_ai_v2_prediction_available"), index)
    signal = _truthy(sells.get("exit_ai_v2_gate_signal"), index)
    triggered = _truthy(sells.get("exit_ai_v2_used_as_exit_trigger"), index)
    high_pm = _numeric(sells.get("pm_multiplier")).ge(1.15)
    triggered_rows = sells[triggered]
    return {
        "exit_ai_v2_prediction_coverage": float(available.mean()) if len(available) else None,
        "exit_ai_v2_gate_signal_count": int(signal.sum()),
        "exit_ai_v2_used_as_exit_trigger_count": int(triggered.sum()),
        "exit_ai_v2_triggered_profit": float(_numeric(triggered_rows.get("net_profit")).sum()) if not triggered_rows.empty else None,
        "exit_ai_v2_triggered_win_rate": _win_rate(triggered_rows.get("net_profit", pd.Series(dtype=float))),
        "high_pm_exit_count": int(high_pm.sum()) if len(high_pm) else 0,
        "high_pm_exit_ai_v2_trigger_count": int((high_pm & triggered).sum()) if len(high_pm) else 0,
        "post_exit_return_5d_for_v2_triggered": _mean(triggered_rows.get("post_exit_return_5d")),
        "post_exit_return_10d_for_v2_triggered": _mean(triggered_rows.get("post_exit_return_10d")),
        "post_exit_return_20d_for_v2_triggered": _mean(triggered_rows.get("post_exit_return_20d")),
    }


def _mean(series: pd.Series | None) -> float | None:
    values = _numeric(series).dropna()
    return float(values.mean()) if not values.empty else None


def _profile_row(root: Path, label: str, profile: str) -> dict[str, Any]:
    base = _profile_dir(root, profile)
    summary = _read_json(base / "backtest_summary.json")
    trades = _read_csv(base / "trades.csv")
    daily = _read_csv(base / "summary.csv")
    audit = _read_csv(base / "purchase_audit.csv")
    config = load_profile(profile)
    skip_counts = _skip_counts(audit)
    policy = config.get("ml_exit_ai_v2_gate", {})
    return {
        "variant": label,
        "profile": profile,
        "status": "ok" if summary else "missing_backtest_logs",
        "exit_ai_v2_gate_enabled": bool(policy.get("enabled", False)),
        "exit_ai_v2_score_threshold": policy.get("score_threshold"),
        "exit_ai_v2_high_pm_safe_mode": bool(policy.get("high_pm_safe_mode", False)),
        "exit_ai_v2_high_pm_threshold": policy.get("high_pm_score_threshold"),
        "net_profit": summary.get("net_cumulative_profit"),
        "profit_factor": summary.get("profit_factor"),
        "max_drawdown": summary.get("max_drawdown"),
        "win_rate": summary.get("win_rate"),
        "total_trades": summary.get("closed_trades_count"),
        "monthly_win_rate": _monthly_win_rate(trades),
        "average_holding_days": summary.get("average_holding_days"),
        **_capital_distribution(daily),
        "selected_but_not_affordable": skip_counts.get("selected_but_not_affordable", 0),
        **_exit_ai_v2_stats(trades),
    }


def build_report(root: Path = ROOT) -> dict[str, Any]:
    rows = [_profile_row(root, label, profile) for label, profile in PROFILES.items()]
    return {
        "phase": "5-H",
        "purpose": "Exit AI v2 conservative gate profile comparison",
        "period": PERIOD,
        "constraints": {
            "report_only": True,
            "full_backtest_executed_by_report": False,
            "full_pytest_executed_by_report": False,
            "current_exit_model_overwritten": False,
            "api_refetch": False,
        },
        "profiles": rows,
        "adoption_criteria": {
            "net_profit": "improve vs v2_78",
            "profit_factor": ">= 2.5",
            "max_drawdown": "within -10%",
            "win_rate": ">= 53%",
            "v2_triggered_post_exit_return": "negative leaning",
            "high_pm": "no excessive early exits",
        },
    }


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows_"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def format_markdown(result: dict[str, Any]) -> str:
    columns = [
        "variant",
        "status",
        "net_profit",
        "profit_factor",
        "max_drawdown",
        "win_rate",
        "total_trades",
        "monthly_win_rate",
        "average_holding_days",
        "average_capital_utilization",
        "selected_but_not_affordable",
        "exit_ai_v2_prediction_coverage",
        "exit_ai_v2_gate_signal_count",
        "exit_ai_v2_used_as_exit_trigger_count",
        "exit_ai_v2_triggered_profit",
        "exit_ai_v2_triggered_win_rate",
        "high_pm_exit_count",
        "high_pm_exit_ai_v2_trigger_count",
    ]
    lines = [
        "# Portfolio Manager Phase 5-H Exit AI v2 Conservative Gate",
        "",
        "## Scope",
        "",
        "- report only",
        "- no full backtest is executed by this script",
        "- current Exit AI model is not overwritten",
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
