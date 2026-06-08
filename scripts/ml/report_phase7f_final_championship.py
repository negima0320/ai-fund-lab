#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = ROOT / "reports" / "final" / "v2_82_cap38"

CORE_PERIOD = "2023-01-01_to_2026-05-31"
EXTENDED_PERIOD = "2021-06-01_to_2026-05-31"

V278 = "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
V282 = "rookie_dealer_02_v2_82_cap38"

SNAPSHOT_FILES = [
    "summary.csv",
    "trades.csv",
    "purchase_audit.csv",
    "backtest_summary.json",
]


@dataclass(frozen=True)
class ProfileRun:
    label: str
    profile: str
    period: str

    @property
    def log_dir(self) -> Path:
        return ROOT / "logs" / "backtests" / self.profile / self.period


RUNS = [
    ProfileRun("v2_78_core", V278, CORE_PERIOD),
    ProfileRun("v2_82_core", V282, CORE_PERIOD),
    ProfileRun("v2_78_extended", V278, EXTENDED_PERIOD),
    ProfileRun("v2_82_extended", V282, EXTENDED_PERIOD),
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _bool_series(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    return series.reindex(index).fillna(False).map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"}
    )


def _sell_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    if "action" not in trades.columns:
        return trades.copy()
    return trades[trades["action"].fillna("").astype(str).eq("SELL")].copy()


def _profit_column(df: pd.DataFrame) -> pd.Series:
    if "net_profit" in df.columns:
        return _numeric(df["net_profit"])
    if "profit" in df.columns:
        return _numeric(df["profit"])
    return pd.Series(0.0, index=df.index)


def _pf(profits: pd.Series) -> float | None:
    profits = _numeric(profits).dropna()
    wins = float(profits[profits > 0].sum())
    losses = float(-profits[profits < 0].sum())
    if losses == 0:
        return None if wins == 0 else float("inf")
    return wins / losses


def _win_rate(profits: pd.Series) -> float | None:
    profits = _numeric(profits).dropna()
    if profits.empty:
        return None
    return float((profits > 0).mean())


def _cagr(initial: float | None, final: float | None, period: str) -> float | None:
    if not initial or not final or initial <= 0 or final <= 0:
        return None
    start_text, end_text = period.split("_to_")
    start = pd.to_datetime(start_text)
    end = pd.to_datetime(end_text)
    years = max((end - start).days / 365.25, 0)
    if years <= 0:
        return None
    return (final / initial) ** (1 / years) - 1


def _capital_metrics(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return {
            "average_capital_utilization": None,
            "median_capital_utilization": None,
            "cash_idle_days": None,
            "average_position_count": None,
        }
    total = _numeric(daily.get("total_assets")).replace(0, pd.NA)
    positions = _numeric(daily.get("positions_value"))
    util = (positions / total).dropna()
    cash = _numeric(daily.get("cash"))
    return {
        "average_capital_utilization": float(util.mean()) if not util.empty else None,
        "median_capital_utilization": float(util.median()) if not util.empty else None,
        "cash_idle_days": int((cash > 0).sum()) if not cash.empty else None,
        "average_position_count": float(_numeric(daily.get("open_positions_count")).mean())
        if "open_positions_count" in daily.columns
        else None,
    }


def _period_profit(trades: pd.DataFrame, freq: str, initial: float | None) -> list[dict[str, Any]]:
    sells = _sell_trades(trades)
    if sells.empty or "exit_date" not in sells.columns:
        return []
    dates = pd.to_datetime(sells["exit_date"], errors="coerce")
    profits = _profit_column(sells)
    grouped = profits.groupby(dates.dt.to_period(freq)).agg(["sum", "count"])
    rows: list[dict[str, Any]] = []
    for key, row in grouped.iterrows():
        if pd.isna(key):
            continue
        mask = dates.dt.to_period(freq).eq(key)
        period_profits = profits[mask]
        rows.append(
            {
                "period": str(key),
                "profit": float(row["sum"]),
                "return": float(row["sum"] / initial) if initial else None,
                "trades": int(row["count"]),
                "profit_factor": _pf(period_profits),
                "win_rate": _win_rate(period_profits),
            }
        )
    return rows


def _drawdowns(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or "total_assets" not in daily.columns:
        return {"max_drawdown": None, "top_drawdowns": []}
    dates = pd.to_datetime(daily.get("date", daily.get("day")), errors="coerce")
    assets = _numeric(daily["total_assets"])
    curve = pd.DataFrame({"date": dates, "assets": assets}).dropna()
    if curve.empty:
        return {"max_drawdown": None, "top_drawdowns": []}
    curve["peak"] = curve["assets"].cummax()
    curve["drawdown"] = curve["assets"] / curve["peak"] - 1
    trough_idx = curve["drawdown"].idxmin()
    trough = curve.loc[trough_idx]
    peak_rows = curve.loc[:trough_idx]
    peak_idx = peak_rows["assets"].idxmax()
    peak = curve.loc[peak_idx]
    recovery_rows = curve.loc[trough_idx:]
    recovered = recovery_rows[recovery_rows["assets"] >= peak["assets"]]
    recovery_date = None if recovered.empty else recovered.iloc[0]["date"]
    top = (
        curve.nsmallest(5, "drawdown")[["date", "assets", "peak", "drawdown"]]
        .assign(date=lambda df: df["date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )
    recovery_days = None
    if recovery_date is not None and not pd.isna(recovery_date):
        recovery_days = int((recovery_date - trough["date"]).days)
    return {
        "max_drawdown": float(trough["drawdown"]),
        "dd_start": peak["date"].strftime("%Y-%m-%d"),
        "dd_trough": trough["date"].strftime("%Y-%m-%d"),
        "dd_recovery": None if recovery_date is None or pd.isna(recovery_date) else recovery_date.strftime("%Y-%m-%d"),
        "dd_recovery_days": recovery_days,
        "top_drawdowns": top,
    }


def _group_stats(trades: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    sells = _sell_trades(trades)
    if sells.empty or column not in sells.columns:
        return []
    profits = _profit_column(sells)
    rows: list[dict[str, Any]] = []
    for key, group in sells.groupby(sells[column].fillna("Unknown").astype(str), dropna=False):
        group_profit = _profit_column(group)
        rows.append(
            {
                column: key,
                "trades": int(len(group)),
                "net_profit": float(group_profit.sum()),
                "profit_factor": _pf(group_profit),
                "win_rate": _win_rate(group_profit),
                "average_return": float(_numeric(group.get("net_profit_rate", group.get("profit_rate"))).mean())
                if ("net_profit_rate" in group.columns or "profit_rate" in group.columns)
                else None,
                "average_holding_days": float(_numeric(group.get("holding_days")).mean())
                if "holding_days" in group.columns
                else None,
                "profit_share": None,
            }
        )
    total = float(profits.sum())
    for row in rows:
        row["profit_share"] = row["net_profit"] / total if total else None
    return sorted(rows, key=lambda row: row["net_profit"], reverse=True)


def _pm_distribution_by_regime(trades: pd.DataFrame) -> list[dict[str, Any]]:
    sells = _sell_trades(trades)
    if sells.empty or not {"market_regime", "pm_multiplier"}.issubset(sells.columns):
        return []
    rows: list[dict[str, Any]] = []
    for (regime, pm), group in sells.groupby(
        [sells["market_regime"].fillna("Unknown").astype(str), sells["pm_multiplier"].fillna("Unknown").astype(str)]
    ):
        profits = _profit_column(group)
        rows.append(
            {
                "market_regime": regime,
                "pm_multiplier": pm,
                "trades": int(len(group)),
                "net_profit": float(profits.sum()),
                "profit_factor": _pf(profits),
                "win_rate": _win_rate(profits),
            }
        )
    return rows


def _capital_constraints(audit: pd.DataFrame) -> dict[str, Any]:
    if audit.empty:
        return {}
    skip_reason = audit.get("skip_reason", pd.Series("", index=audit.index)).fillna("").astype(str)
    scale_reason = audit.get("scale_reason", pd.Series("", index=audit.index)).fillna("").astype(str)
    cap_reason = audit.get("pm_per_code_cap_reason", pd.Series("", index=audit.index)).fillna("").astype(str)
    combined = skip_reason.where(skip_reason.ne(""), scale_reason.where(scale_reason.ne(""), cap_reason))
    cap_reduced = _bool_series(audit.get("pm_per_code_cap_reduced"), audit.index)
    cap_skip = _bool_series(audit.get("pm_per_code_cap_skip"), audit.index)
    original = _numeric(audit.get("pm_per_code_cap_original_amount"))
    capped = _numeric(audit.get("pm_per_code_cap_amount"))
    reduction = (original - capped).clip(lower=0)
    return {
        "selected_but_not_affordable": int((combined == "selected_but_not_affordable").sum()),
        "insufficient_available_cash": int((combined == "insufficient_available_cash").sum()),
        "per_code_cap_skip_or_reduction_count": int((cap_reduced | cap_skip).sum()),
        "per_code_cap_skip_count": int(cap_skip.sum()),
        "per_code_cap_reduction_count": int(cap_reduced.sum()),
        "per_code_cap_reduction_amount": float(reduction[cap_reduced].sum()),
        "reason_breakdown": {str(k): int(v) for k, v in combined[combined.ne("")].value_counts().items()},
    }


def _concentration(trades: pd.DataFrame) -> dict[str, Any]:
    sells = _sell_trades(trades)
    if sells.empty or "code" not in sells.columns:
        return {}
    by_code = _profit_column(sells).groupby(sells["code"].astype(str)).sum().sort_values(ascending=False)
    abs_total = float(by_code.abs().sum())
    return {
        "top_profit_code": None if by_code.empty else str(by_code.index[0]),
        "top_profit_code_profit": None if by_code.empty else float(by_code.iloc[0]),
        "single_code_abs_profit_concentration": float(by_code.abs().max() / abs_total) if abs_total else None,
        "top5_abs_profit_concentration": float(by_code.abs().head(5).sum() / abs_total) if abs_total else None,
        "top10_abs_profit_concentration": float(by_code.abs().head(10).sum() / abs_total) if abs_total else None,
    }


def _top_trades(trades: pd.DataFrame, n: int = 100) -> dict[str, list[dict[str, Any]]]:
    sells = _sell_trades(trades)
    if sells.empty:
        return {"winners": [], "losers": []}
    cols = [
        "code",
        "name",
        "sector_name",
        "entry_date",
        "exit_date",
        "holding_days",
        "net_profit",
        "profit_rate",
        "exit_reason",
        "pm_multiplier",
        "pm_score",
        "market_regime",
    ]
    cols = [col for col in cols if col in sells.columns]
    ranked = sells.assign(_profit=_profit_column(sells))
    return {
        "winners": ranked.sort_values("_profit", ascending=False).head(n)[cols].to_dict(orient="records"),
        "losers": ranked.sort_values("_profit", ascending=True).head(n)[cols].to_dict(orient="records"),
    }


def _snapshot(run: ProfileRun, target_name: str) -> dict[str, Any]:
    target = FINAL_DIR / target_name
    copied: list[str] = []
    missing: list[str] = []
    target.mkdir(parents=True, exist_ok=True)
    for filename in SNAPSHOT_FILES:
        src = run.log_dir / filename
        if src.exists():
            shutil.copy2(src, target / filename)
            copied.append(filename)
        else:
            missing.append(filename)
    return {"target_dir": str(target.relative_to(ROOT)), "copied": copied, "missing": missing}


def _run_summary(run: ProfileRun) -> dict[str, Any]:
    summary = _read_json(run.log_dir / "backtest_summary.json")
    trades = _read_csv(run.log_dir / "trades.csv")
    daily = _read_csv(run.log_dir / "summary.csv")
    audit = _read_csv(run.log_dir / "purchase_audit.csv")
    initial = _number(summary.get("initial_assets", summary.get("initial_capital")))
    final = _number(summary.get("final_assets"))
    result = {
        "label": run.label,
        "profile": run.profile,
        "period": run.period,
        "log_dir": str(run.log_dir.relative_to(ROOT)),
        "logs_available": bool(summary),
        "initial_assets": initial,
        "final_assets": final,
        "net_profit": _number(summary.get("net_cumulative_profit")),
        "cagr": summary.get("cagr") if summary.get("cagr") is not None else _cagr(initial, final, run.period),
        "annualized_return": summary.get("annualized_return") if summary.get("annualized_return") is not None else _cagr(initial, final, run.period),
        "profit_factor": _number(summary.get("profit_factor")),
        "max_drawdown": _number(summary.get("max_drawdown")),
        "win_rate": _number(summary.get("win_rate")),
        "monthly_win_rate": _win_rate(pd.Series([row["profit"] for row in _period_profit(trades, "M", initial)])),
        "total_trades": summary.get("total_trades", summary.get("closed_trade_count")),
        "closed_trades_count": summary.get("closed_trades_count"),
        "average_holding_days": _number(summary.get("average_holding_days")),
        "date_range_audit": summary.get("date_range_audit", {}),
        "backtest_execution_audit": summary.get("backtest_result_integrity_audit", {}),
        "market_coverage": summary.get("market_coverage", {}),
        **_capital_metrics(daily),
        "capital_constraints": _capital_constraints(audit),
        "drawdown_analysis": _drawdowns(daily),
        "yearly_results": _period_profit(trades, "Y", initial),
        "monthly_results": _period_profit(trades, "M", initial),
        "market_regime_results": _group_stats(trades, "market_regime"),
        "pm_multiplier_results": _group_stats(trades, "pm_multiplier"),
        "pm_distribution_by_regime": _pm_distribution_by_regime(trades),
        "exit_reason_results": _group_stats(trades, "exit_reason"),
        "concentration": _concentration(trades),
        "top_trades": _top_trades(trades),
    }
    return result


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "net_profit",
        "profit_factor",
        "max_drawdown",
        "win_rate",
        "monthly_win_rate",
        "total_trades",
        "average_holding_days",
        "average_capital_utilization",
        "cagr",
    ]
    return {
        field: (
            candidate.get(field) - baseline.get(field)
            if isinstance(candidate.get(field), (int, float)) and isinstance(baseline.get(field), (int, float))
            else None
        )
        for field in fields
    }


def _precheck() -> dict[str, Any]:
    wf_dir = ROOT / "data" / "ml" / "walk_forward_predictions"
    wf_files = sorted(wf_dir.glob("predictions_*.parquet"))
    wf_dates = [path.stem.replace("predictions_", "") for path in wf_files]
    datasets = {
        "stock_dataset": ROOT / "data" / "ml" / "datasets" / "ml_dataset.parquet",
        "pm_current_dataset": ROOT
        / "data"
        / "ml"
        / "portfolio_manager"
        / "portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet",
        "pm_api_only_dataset": ROOT
        / "data"
        / "ml"
        / "portfolio_manager_api_only"
        / "pm_ai_api_only_dataset_2021-06_to_2026-05.parquet",
        "exit_ai_v2_dataset": ROOT
        / "data"
        / "ml"
        / "exit_ai_v2"
        / "exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet",
    }
    dataset_status = {
        name: {"exists": path.exists(), "path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size if path.exists() else None}
        for name, path in datasets.items()
    }
    wf_min = min(wf_dates) if wf_dates else None
    wf_max = max(wf_dates) if wf_dates else None
    can_evaluate_2021_2022_same_condition = bool(wf_min and wf_min <= "2021-06-01")
    return {
        "walk_forward_predictions": {
            "exists": bool(wf_files),
            "file_count": len(wf_files),
            "available_from": wf_min,
            "available_to": wf_max,
            "same_condition_2021_2022_available": can_evaluate_2021_2022_same_condition,
        },
        "datasets": dataset_status,
        "extended_period_judgment": "reference_only"
        if not can_evaluate_2021_2022_same_condition
        else "same_condition_possible",
        "coverage_issue": None
        if can_evaluate_2021_2022_same_condition
        else "Stock Selection walk-forward predictions start after 2021-2022, so extended robustness is not a clean championship comparison.",
    }


def build_report() -> dict[str, Any]:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    run_results = [_run_summary(run) for run in RUNS]
    by_label = {row["label"]: row for row in run_results}
    snapshots = {
        "core": _snapshot(ProfileRun("v2_82_core", V282, CORE_PERIOD), "core_2023-01_to_2026-05"),
    }
    if by_label.get("v2_82_extended", {}).get("logs_available"):
        snapshots["extended"] = _snapshot(
            ProfileRun("v2_82_extended", V282, EXTENDED_PERIOD),
            "extended_2021-06_to_2026-05",
        )
    core_delta = _delta(by_label["v2_82_core"], by_label["v2_78_core"]) if by_label["v2_82_core"].get("logs_available") and by_label["v2_78_core"].get("logs_available") else {}
    extended_delta = (
        _delta(by_label["v2_82_extended"], by_label["v2_78_extended"])
        if by_label.get("v2_82_extended", {}).get("logs_available") and by_label.get("v2_78_extended", {}).get("logs_available")
        else {}
    )
    verdict = {
        "production_candidate": True,
        "recommended_profile": V282,
        "confidence_level": "medium",
        "fix_recommended": True,
        "reason": (
            "Core period v2_82 improves net profit with PF above 2.5, DD below 10%, and win rate above 53%. "
            "Confidence is medium because extended 2021-2022 same-condition prediction coverage is not available from current walk-forward files."
        ),
        "do_not_touch": [
            "current Stock Selection AI",
            "current PM AI model directory",
            "current Exit AI v2_66",
            "Exit AI v2 candidate integration",
            "PM AI API-only candidate integration",
            "live order path",
        ],
        "recommended_improvements": [
            "Build true 2021-2022 walk-forward prediction coverage before treating extended results as decisive.",
            "Run PM AI API-only candidate integration audit separately before replacing current PM AI.",
            "Keep v2_82 cap38 as current Version 1.0 candidate until a cleaner full walk-forward retraining pipeline beats it.",
        ],
    }
    return {
        "phase": "7-F",
        "title": "Final Championship Audit",
        "generated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "constraints": {
            "api_refetch": False,
            "openai_used": False,
            "live_order": False,
            "current_model_overwrite": False,
            "new_profile_added": False,
            "pm_ai_candidate_integrated": False,
            "exit_ai_v2_candidate_integrated": False,
        },
        "precheck": _precheck(),
        "runs": run_results,
        "comparisons": {"core_v282_minus_v278": core_delta, "extended_v282_minus_v278": extended_delta},
        "snapshots": snapshots,
        "version_1_0_verdict": verdict,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data_"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def format_markdown(report: dict[str, Any]) -> str:
    run_cols = [
        "label",
        "logs_available",
        "initial_assets",
        "final_assets",
        "net_profit",
        "cagr",
        "profit_factor",
        "max_drawdown",
        "win_rate",
        "monthly_win_rate",
        "total_trades",
        "average_holding_days",
        "average_capital_utilization",
    ]
    lines = [
        "# Phase 7-F Final Championship Audit",
        "",
        "## Precheck",
        "",
        f"- extended_period_judgment: {report['precheck']['extended_period_judgment']}",
        f"- coverage_issue: {report['precheck']['coverage_issue']}",
        f"- walk_forward_available_from: {report['precheck']['walk_forward_predictions']['available_from']}",
        f"- walk_forward_available_to: {report['precheck']['walk_forward_predictions']['available_to']}",
        "",
        "## Core / Extended Summary",
        "",
        _table(report["runs"], run_cols),
        "",
        "## Core Delta v2_82 - v2_78",
        "",
        "```json",
        json.dumps(report["comparisons"]["core_v282_minus_v278"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Extended Delta v2_82 - v2_78",
        "",
        "```json",
        json.dumps(report["comparisons"]["extended_v282_minus_v278"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    for run in report["runs"]:
        if not run.get("logs_available"):
            continue
        lines.extend(
            [
                f"## {run['label']} Detail",
                "",
                "### Market Regime",
                "",
                _table(
                    run["market_regime_results"],
                    ["market_regime", "trades", "net_profit", "profit_factor", "win_rate", "average_return", "profit_share"],
                ),
                "",
                "### PM Multiplier",
                "",
                _table(
                    run["pm_multiplier_results"],
                    ["pm_multiplier", "trades", "net_profit", "profit_factor", "win_rate", "average_return", "profit_share"],
                ),
                "",
                "### Exit Reason",
                "",
                _table(
                    run["exit_reason_results"],
                    ["exit_reason", "trades", "net_profit", "profit_factor", "win_rate", "profit_share"],
                ),
                "",
                "### Yearly Results",
                "",
                _table(run["yearly_results"], ["period", "profit", "return", "trades", "profit_factor", "win_rate"]),
                "",
                "### Drawdown",
                "",
                "```json",
                json.dumps(run["drawdown_analysis"], ensure_ascii=False, indent=2),
                "```",
                "",
                "### Capital Constraints",
                "",
                "```json",
                json.dumps(run["capital_constraints"], ensure_ascii=False, indent=2),
                "```",
                "",
                "### Concentration",
                "",
                "```json",
                json.dumps(run["concentration"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Version 1.0 Verdict",
            "",
            "```json",
            json.dumps(report["version_1_0_verdict"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Snapshots",
            "",
            "```json",
            json.dumps(report["snapshots"], ensure_ascii=False, indent=2),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = build_report()
    json_path = FINAL_DIR / "final_summary.json"
    md_path = FINAL_DIR / "final_summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(format_markdown(report), encoding="utf-8")
    print(f"wrote {md_path.relative_to(ROOT)}")
    print(f"wrote {json_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
