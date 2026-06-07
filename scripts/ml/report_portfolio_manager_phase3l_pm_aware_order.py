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
BASELINE_PROFILE = "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030"
PHASE3L_PROFILES = {
    "v2_77_current": BASELINE_PROFILE,
    "v2_78_w025": "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025",
    "v2_78_w050": "rookie_dealer_02_v2_78_pm_aware_order_fallback_w050",
    "v2_78_w100": "rookie_dealer_02_v2_78_pm_aware_order_fallback_w100",
}
REPORT_STEM = "portfolio_manager_phase3l_pm_aware_order_2023-01_to_2026-05"


def _profile_dir(root: Path, profile: str) -> Path:
    return root / "logs" / "backtests" / profile / PERIOD


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
    if trades.empty or "exit_date" not in trades.columns or "net_profit" not in trades.columns:
        return None
    data = trades.copy()
    data["month"] = pd.to_datetime(data["exit_date"], errors="coerce").dt.strftime("%Y-%m")
    monthly = pd.to_numeric(data["net_profit"], errors="coerce").groupby(data["month"]).sum().dropna()
    monthly = monthly[monthly.index.notna()]
    if monthly.empty:
        return None
    return float((monthly > 0).mean())


def _capital_distribution(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or not {"cash", "positions_value", "total_assets"}.issubset(daily.columns):
        return {
            "average_capital_utilization": None,
            "median_capital_utilization": None,
            "days_below_50pct": None,
            "cash_idle_days": None,
            "average_holding_count": None,
        }
    total = pd.to_numeric(daily["total_assets"], errors="coerce").replace(0, pd.NA)
    positions = pd.to_numeric(daily["positions_value"], errors="coerce")
    utilization = (positions / total).dropna()
    holding = pd.to_numeric(daily.get("open_positions_count"), errors="coerce") if "open_positions_count" in daily.columns else pd.Series(dtype=float)
    return {
        "average_capital_utilization": float(utilization.mean()) if not utilization.empty else None,
        "median_capital_utilization": float(utilization.median()) if not utilization.empty else None,
        "days_below_50pct": int((utilization < 0.5).sum()) if not utilization.empty else None,
        "cash_idle_days": int((utilization <= 0.05).sum()) if not utilization.empty else None,
        "average_holding_count": float(holding.dropna().mean()) if not holding.dropna().empty else None,
    }


def _skip_counts(audit: pd.DataFrame) -> dict[str, int]:
    if audit.empty:
        return {}
    reason = audit.get("skip_reason", pd.Series(dtype=str)).fillna("").astype(str)
    scale = audit.get("scale_reason", pd.Series(dtype=str)).fillna("").astype(str)
    combined = reason.where(reason.ne(""), scale)
    return {str(key): int(value) for key, value in combined[combined.ne("")].value_counts().items()}


def _truthy(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    return series.reindex(index).fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def _fallback_stats(audit: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    if audit.empty:
        return {
            "fallback_triggered_count": 0,
            "fallback_bought_count": 0,
            "fallback_quality_filtered_count": 0,
            "fallback_average_pm_score": None,
            "fallback_average_pm_multiplier": None,
            "fallback_realized_profit": None,
            "fallback_win_rate": None,
        }
    fallback = audit[_truthy(audit.get("fallback_triggered"), audit.index)].copy()
    bought = fallback[fallback.get("decision", pd.Series(dtype=str)).fillna("").astype(str).isin(["BUY", "SCALED_BUY"])]
    quality_filtered = audit[_truthy(audit.get("skipped_by_fallback_quality_filter"), audit.index)]
    if bought.empty or trades.empty:
        realized = pd.Series(dtype=float)
    else:
        key_cols = ["signal_date", "code"]
        bought_keys = set(zip(bought.get("signal_date", []), bought.get("code", [])))
        trade_data = trades[trades.get("action", pd.Series(dtype=str)).fillna("").astype(str).eq("SELL")].copy()
        realized = pd.to_numeric(
            trade_data[[*key_cols, "net_profit"]].apply(
                lambda row: row["net_profit"] if (row["signal_date"], row["code"]) in bought_keys else pd.NA,
                axis=1,
            ),
            errors="coerce",
        ).dropna()
    return {
        "fallback_triggered_count": int(len(fallback)),
        "fallback_bought_count": int(len(bought)),
        "fallback_quality_filtered_count": int(len(quality_filtered)),
        "fallback_average_pm_score": _safe_float(pd.to_numeric(bought.get("pm_score"), errors="coerce").mean()) if not bought.empty else None,
        "fallback_average_pm_multiplier": _safe_float(pd.to_numeric(bought.get("pm_multiplier"), errors="coerce").mean()) if not bought.empty else None,
        "fallback_realized_profit": float(realized.sum()) if not realized.empty else None,
        "fallback_win_rate": _win_rate(realized),
    }


def _group_profit(trades: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if trades.empty or column not in trades.columns or "net_profit" not in trades.columns:
        return []
    sells = trades[trades.get("action", pd.Series(dtype=str)).fillna("").astype(str).eq("SELL")].copy()
    rows = []
    for key, group in sells.groupby(column, dropna=False):
        profits = pd.to_numeric(group["net_profit"], errors="coerce")
        rows.append(
            {
                column: str(key),
                "trade_count": int(profits.notna().sum()),
                "net_profit": float(profits.sum()),
                "profit_factor": _profit_factor(profits),
                "win_rate": _win_rate(profits),
            }
        )
    return rows


def _concentration(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty or "code" not in trades.columns or "net_profit" not in trades.columns:
        return {"top1_contribution": None, "top3_contribution": None, "top5_contribution": None, "code_67400_contribution": None}
    sells = trades[trades.get("action", pd.Series(dtype=str)).fillna("").astype(str).eq("SELL")].copy()
    total = float(pd.to_numeric(sells.get("net_profit"), errors="coerce").sum())
    by_code = pd.to_numeric(sells["net_profit"], errors="coerce").groupby(sells["code"].astype(str)).sum().sort_values(ascending=False)
    def contribution(n: int) -> float | None:
        if total == 0 or by_code.empty:
            return None
        return float(by_code.head(n).sum() / total)
    return {
        "top1_contribution": contribution(1),
        "top3_contribution": contribution(3),
        "top5_contribution": contribution(5),
        "code_67400_contribution": float(by_code.get("67400", 0.0) / total) if total else None,
    }


def build_report(root: Path = ROOT) -> dict[str, Any]:
    comparison = []
    multiplier_rows = []
    score_rows = []
    for label, profile in PHASE3L_PROFILES.items():
        base = _profile_dir(root, profile)
        summary = _read_summary(base / "backtest_summary.json")
        daily = _read_csv(base / "summary.csv")
        trades = _read_csv(base / "trades.csv")
        audit = _read_csv(base / "purchase_audit.csv")
        config = load_profile(profile)
        skip_counts = _skip_counts(audit)
        row = {
            "variant": label,
            "profile": profile,
            "status": "ok" if summary else "missing_backtest_logs",
            "pm_order_weight": config.get("portfolio_manager_ai_sizing", {}).get("pm_order_weight"),
            "net_profit": summary.get("net_cumulative_profit"),
            "profit_factor": summary.get("profit_factor"),
            "max_drawdown": summary.get("max_drawdown"),
            "win_rate": summary.get("win_rate"),
            "total_trades": summary.get("closed_trades_count"),
            "monthly_win_rate": _monthly_win_rate(trades),
            **_capital_distribution(daily),
            "selected_but_not_affordable": skip_counts.get("selected_but_not_affordable", 0),
            "insufficient_available_cash": skip_counts.get("insufficient_available_cash", 0),
            "daily_buy_limit_scaled_below_round_lot": skip_counts.get("daily_buy_limit_scaled_below_round_lot", 0),
            "per_code_exposure_cap_scaled_below_round_lot": skip_counts.get("per_code_exposure_cap_scaled_below_round_lot", 0),
            "pm_low_score_skip": skip_counts.get("pm_low_score_skip", 0),
            **_fallback_stats(audit, trades),
            **_concentration(trades),
        }
        comparison.append(row)
        for group in _group_profit(trades, "pm_multiplier"):
            multiplier_rows.append({"variant": label, **group})
        if not trades.empty and "pm_score" in trades.columns:
            scored = trades.copy()
            scored["pm_score_band"] = pd.cut(
                pd.to_numeric(scored["pm_score"], errors="coerce"),
                bins=[-999, -0.2, 0.0, 0.2, 0.4, 999],
                labels=["< -0.20", "-0.20 to 0", "0 to 0.20", "0.20 to 0.40", ">= 0.40"],
            )
            for group in _group_profit(scored, "pm_score_band"):
                score_rows.append({"variant": label, **group})
    return {
        "purpose": "Portfolio Manager AI Phase 3-L PM-aware buy ordering report",
        "period": PERIOD,
        "constraints": {
            "api_refetch": False,
            "openai_api": False,
            "historical_predictions_source": "data/ml/walk_forward_predictions/",
            "current_model_historical_regeneration": False,
            "selected_count_in_day_used": False,
            "live_order_placement": False,
        },
        "profiles": PHASE3L_PROFILES,
        "comparison": comparison,
        "pm_multiplier_performance": multiplier_rows,
        "pm_score_band_performance": score_rows,
    }


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def format_markdown(result: dict[str, Any]) -> str:
    comparison_cols = [
        "variant",
        "profile",
        "status",
        "pm_order_weight",
        "net_profit",
        "profit_factor",
        "max_drawdown",
        "win_rate",
        "total_trades",
        "monthly_win_rate",
        "average_capital_utilization",
        "median_capital_utilization",
        "days_below_50pct",
        "cash_idle_days",
        "selected_but_not_affordable",
        "fallback_triggered_count",
        "fallback_bought_count",
        "fallback_realized_profit",
        "code_67400_contribution",
    ]
    return "\n".join(
        [
            "# Portfolio Manager AI Phase 3-L PM-aware Buy Ordering",
            "",
            "## Scope",
            "",
            f"- period: `{result['period']}`",
            "- report reads existing backtest logs only",
            "- no API refetch / no OpenAI API / no current model historical regeneration",
            "- `selected_count_in_day` used: `False`",
            "",
            "## Summary Comparison",
            "",
            _table(result["comparison"], comparison_cols),
            "",
            "## PM Multiplier Performance",
            "",
            _table(result["pm_multiplier_performance"], ["variant", "pm_multiplier", "trade_count", "net_profit", "profit_factor", "win_rate"]),
            "",
            "## PM Score Band Performance",
            "",
            _table(result["pm_score_band_performance"], ["variant", "pm_score_band", "trade_count", "net_profit", "profit_factor", "win_rate"]),
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
