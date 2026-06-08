#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase8h_relative_allocator_backtest_2023-01_to_2026-05"
PERIOD = "2023-01-01_to_2026-05-31"

V282 = "rookie_dealer_02_v2_82_cap38"
V290 = "rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38"
V291 = "rookie_dealer_02_v2_91_pm_ai_v2_calibrated_rule_e_cap38"
V292 = "rookie_dealer_02_v2_92_relative_allocator_cap38"
PM_BUCKETS = [1.30, 1.15, 1.00, 0.80, 0.60]


@dataclass(frozen=True)
class RunPaths:
    label: str
    profile: str
    log_dir: Path


def _run_paths() -> list[RunPaths]:
    return [
        RunPaths("v2_82_current_pm_cap38", V282, ROOT / "reports" / "final" / "v2_82_cap38" / "core_2023-01_to_2026-05"),
        RunPaths("v2_90_pm_ai_v2_raw_cap38", V290, ROOT / "logs" / "backtests" / V290 / PERIOD),
        RunPaths("v2_91_pm_ai_v2_calibrated_cap38", V291, ROOT / "logs" / "backtests" / V291 / PERIOD),
        RunPaths("v2_92_relative_allocator_cap38", V292, ROOT / "logs" / "backtests" / V292 / PERIOD),
    ]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _sell_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "action" not in trades.columns:
        return trades.copy()
    return trades[trades["action"].fillna("").astype(str).str.upper().eq("SELL")].copy()


def _profit_column(trades: pd.DataFrame) -> pd.Series:
    if "net_profit" in trades.columns:
        return _numeric(trades["net_profit"])
    if "profit" in trades.columns:
        return _numeric(trades["profit"])
    return pd.Series(0.0, index=trades.index)


def _profit_factor(profits: pd.Series) -> float | None:
    values = _numeric(profits).dropna()
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(-values[values < 0].sum())
    if gross_loss == 0:
        return None if gross_profit == 0 else float("inf")
    return gross_profit / gross_loss


def _win_rate(profits: pd.Series) -> float | None:
    values = _numeric(profits).dropna()
    if values.empty:
        return None
    return float((values > 0).mean())


def _summary_number(summary: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = summary.get(key)
        if value is None:
            continue
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if not pd.isna(numeric):
            return float(numeric)
    return None


def _monthly_win_rate(trades: pd.DataFrame) -> float | None:
    if trades.empty or "exit_date" not in trades.columns:
        return None
    dates = pd.to_datetime(trades["exit_date"], errors="coerce")
    monthly = _profit_column(trades).groupby(dates.dt.to_period("M")).sum()
    monthly = monthly[monthly.index.notna()]
    if monthly.empty:
        return None
    return float((monthly > 0).mean())


def _cagr(initial: float | None, final: float | None) -> float | None:
    if not initial or not final or initial <= 0 or final <= 0:
        return None
    years = max((pd.Timestamp("2026-05-31") - pd.Timestamp("2023-01-01")).days / 365.25, 0)
    if years <= 0:
        return None
    return (final / initial) ** (1 / years) - 1


def _drawdown(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or "total_assets" not in daily.columns:
        return {"max_drawdown": None, "dd_start": None, "dd_trough": None}
    curve = pd.DataFrame({"date": pd.to_datetime(daily.get("date"), errors="coerce"), "assets": _numeric(daily["total_assets"])}).dropna()
    if curve.empty:
        return {"max_drawdown": None, "dd_start": None, "dd_trough": None}
    curve["peak"] = curve["assets"].cummax()
    curve["drawdown"] = curve["assets"] / curve["peak"] - 1.0
    trough_idx = curve["drawdown"].idxmin()
    peak_idx = curve.loc[:trough_idx, "assets"].idxmax()
    return {
        "max_drawdown": float(curve.loc[trough_idx, "drawdown"]),
        "dd_start": curve.loc[peak_idx, "date"].strftime("%Y-%m-%d"),
        "dd_trough": curve.loc[trough_idx, "date"].strftime("%Y-%m-%d"),
    }


def _load_run(run: RunPaths) -> dict[str, Any]:
    daily = _read_csv(run.log_dir / "summary.csv")
    trades = _sell_trades(_read_csv(run.log_dir / "trades.csv"))
    purchase = _read_csv(run.log_dir / "purchase_audit.csv")
    summary = _read_json(run.log_dir / "backtest_summary.json")
    return {"run": run, "daily": daily, "trades": trades, "purchase": purchase, "summary": summary}


def _basic_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    run: RunPaths = payload["run"]
    daily = payload["daily"]
    trades = payload["trades"]
    summary = payload["summary"]
    profits = _profit_column(trades)
    total_assets = _numeric(daily.get("total_assets")) if not daily.empty else pd.Series(dtype=float)
    positions = _numeric(daily.get("positions_value", daily.get("investment_amount"))) if not daily.empty else pd.Series(dtype=float)
    utilization = positions / total_assets.replace(0, pd.NA) if not total_assets.empty else pd.Series(dtype=float)
    final_assets = _summary_number(summary, "final_assets") or (float(total_assets.dropna().iloc[-1]) if not total_assets.dropna().empty else None)
    initial_assets = _summary_number(summary, "initial_capital", "initial_assets") or 1_000_000.0
    dd = _drawdown(daily)
    return {
        "label": run.label,
        "profile": run.profile,
        "net_profit": _summary_number(summary, "net_cumulative_profit", "net_profit") if summary else float(profits.sum()),
        "profit_factor": _summary_number(summary, "profit_factor") if summary else _profit_factor(profits),
        "max_drawdown": _summary_number(summary, "max_drawdown") if summary else dd["max_drawdown"],
        "dd_start": dd["dd_start"],
        "dd_trough": dd["dd_trough"],
        "win_rate": _summary_number(summary, "win_rate") if summary else _win_rate(profits),
        "monthly_win_rate": _monthly_win_rate(trades),
        "total_trades": int(_summary_number(summary, "total_trades") or len(trades)),
        "average_holding_days": float(_numeric(trades.get("holding_days")).mean()) if not trades.empty else None,
        "average_capital_utilization": float(utilization.dropna().mean()) if not utilization.dropna().empty else None,
        "initial_assets": initial_assets,
        "final_assets": final_assets,
        "cagr": _cagr(initial_assets, final_assets),
    }


def _pm_distribution(payload: dict[str, Any]) -> list[dict[str, Any]]:
    trades = payload["trades"]
    rows = []
    for bucket in PM_BUCKETS:
        selected = trades[_numeric(trades.get("pm_multiplier")).round(2).eq(bucket)] if not trades.empty else pd.DataFrame()
        profits = _profit_column(selected)
        rows.append(
            {
                "label": payload["run"].label,
                "pm_multiplier": bucket,
                "trades": int(len(selected)),
                "profit": float(profits.sum()) if not profits.empty else 0.0,
                "profit_factor": _profit_factor(profits),
                "win_rate": _win_rate(profits),
            }
        )
    return rows


def _quantiles(series: pd.Series | None) -> dict[str, float | None]:
    values = _numeric(series).dropna()
    if values.empty:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "mean": None}
    return {
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "mean": float(values.mean()),
    }


def _relative_allocator_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    purchase = payload["purchase"]
    trades = payload["trades"]
    if purchase.empty:
        return {"label": payload["run"].label, "relative_allocator_operationally_valid": False}
    enabled = purchase.get("relative_allocator_enabled")
    relative_enabled_count = int(enabled.fillna("").astype(str).str.lower().isin({"true", "1", "yes"}).sum()) if enabled is not None else 0
    pm130 = purchase[_numeric(purchase.get("pm_multiplier")).round(2).eq(1.30)] if "pm_multiplier" in purchase.columns else pd.DataFrame()
    pm080 = purchase[_numeric(purchase.get("pm_multiplier")).round(2).eq(0.80)] if "pm_multiplier" in purchase.columns else pd.DataFrame()
    return {
        "label": payload["run"].label,
        "relative_allocator_operationally_valid": relative_enabled_count > 0,
        "relative_allocator_rows": relative_enabled_count,
        "relative_rank_distribution": _quantiles(purchase.get("relative_rank")),
        "pm130_relative_rank_distribution": _quantiles(pm130.get("relative_rank")) if not pm130.empty else _quantiles(None),
        "pm080_relative_rank_distribution": _quantiles(pm080.get("relative_rank")) if not pm080.empty else _quantiles(None),
        "candidate_count_distribution": _quantiles(purchase.get("relative_candidate_count")),
        "relative_score_distribution": _quantiles(purchase.get("relative_score")),
        "pm130_profit": float(_profit_column(trades[_numeric(trades.get("pm_multiplier")).round(2).eq(1.30)]).sum()) if not trades.empty else 0.0,
    }


def _risk_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    purchase = payload["purchase"]
    trades = payload["trades"]
    if purchase.empty:
        return {"label": payload["run"].label}
    skip_reason = purchase.get("skip_reason", pd.Series("", index=purchase.index)).fillna("").astype(str)
    cap_reduced = purchase.get("pm_per_code_cap_reduced", pd.Series(False, index=purchase.index)).fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    cap_skip = purchase.get("pm_per_code_cap_skip", pd.Series(False, index=purchase.index)).fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    profits = _profit_column(trades)
    by_code = trades.assign(_profit=profits).groupby("code")["_profit"].sum() if not trades.empty and "code" in trades.columns else pd.Series(dtype=float)
    total_abs = float(by_code.abs().sum()) if not by_code.empty else 0.0
    max_abs_share = float(by_code.abs().max() / total_abs) if total_abs else None
    return {
        "label": payload["run"].label,
        "selected_but_not_affordable": int(skip_reason.eq("selected_but_not_affordable").sum()),
        "insufficient_available_cash": int(skip_reason.eq("insufficient_available_cash").sum()),
        "per_code_cap_skip_or_reduction": int((cap_reduced | cap_skip).sum()),
        "single_code_concentration_abs_profit_share": max_abs_share,
    }


def build_report() -> dict[str, Any]:
    payloads = [_load_run(run) for run in _run_paths()]
    basics = [_basic_metrics(payload) for payload in payloads]
    pm_rows = [row for payload in payloads for row in _pm_distribution(payload)]
    relative_rows = [_relative_allocator_metrics(payload) for payload in payloads]
    risks = [_risk_metrics(payload) for payload in payloads]
    verdict = _verdict(basics, relative_rows)
    return {
        "metadata": {
            "phase": "8-H",
            "backtest_report_only": True,
            "current_model_overwritten": False,
            "full_pytest_executed": False,
            "period": PERIOD,
        },
        "basic_comparison": basics,
        "pm_multiplier_comparison": pm_rows,
        "relative_allocator_metrics": relative_rows,
        "risk_metrics": risks,
        "verdict": verdict,
    }


def _verdict(basics: list[dict[str, Any]], relative_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {row["label"]: row for row in basics}
    v82 = by_label.get("v2_82_current_pm_cap38", {})
    v90 = by_label.get("v2_90_pm_ai_v2_raw_cap38", {})
    v91 = by_label.get("v2_91_pm_ai_v2_calibrated_cap38", {})
    v92 = by_label.get("v2_92_relative_allocator_cap38", {})
    v92_relative = next((row for row in relative_rows if row["label"] == "v2_92_relative_allocator_cap38"), {})
    v92_profit = float(v92.get("net_profit") or 0.0)
    v90_profit = float(v90.get("net_profit") or 0.0)
    v91_profit = float(v91.get("net_profit") or 0.0)
    v82_profit = float(v82.get("net_profit") or 0.0)
    acceptable = (
        v92_profit > max(v90_profit, v91_profit)
        and float(v92.get("profit_factor") or 0.0) >= 2.0
        and float(v92.get("max_drawdown") or -1.0) >= -0.10
        and float(v92.get("win_rate") or 0.0) >= 0.50
    )
    close_to_v282 = v82_profit > 0 and v92_profit >= v82_profit * 0.80
    return {
        "relative_allocator_operationally_valid": bool(v92_relative.get("relative_allocator_operationally_valid")),
        "relative_allocator_result_acceptable": bool(acceptable),
        "relative_allocator_beats_pm_ai_v2": bool(v92_profit > max(v90_profit, v91_profit)),
        "relative_allocator_close_to_v282": bool(close_to_v282),
        "pm_ai_ranking_hypothesis_supported": bool(acceptable or close_to_v282),
        "next_phase_recommended": "Phase 8-I Relative Allocator Refinement" if acceptable else "Stay v2_82 current PM",
    }


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return f"{value:.4f}"
    if isinstance(value, dict):
        return ", ".join(f"{key}:{_format(val)}" for key, val in value.items())
    return str(value).replace("|", "\\|")


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(_format(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def format_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Portfolio Manager AI Phase 8-H Relative Allocator Backtest",
            "",
            "## Basic Comparison",
            "",
            _table(report["basic_comparison"], ["label", "net_profit", "profit_factor", "max_drawdown", "win_rate", "monthly_win_rate", "total_trades", "average_holding_days", "average_capital_utilization", "final_assets", "cagr"]),
            "",
            "## PM Multiplier Comparison",
            "",
            _table(report["pm_multiplier_comparison"], ["label", "pm_multiplier", "trades", "profit", "profit_factor", "win_rate"]),
            "",
            "## Relative Allocator",
            "",
            _table(report["relative_allocator_metrics"], ["label", "relative_allocator_operationally_valid", "relative_allocator_rows", "relative_rank_distribution", "pm130_relative_rank_distribution", "pm080_relative_rank_distribution", "candidate_count_distribution", "relative_score_distribution"]),
            "",
            "## Risk",
            "",
            _table(report["risk_metrics"], ["label", "selected_but_not_affordable", "insufficient_available_cash", "per_code_cap_skip_or_reduction", "single_code_concentration_abs_profit_share"]),
            "",
            "## Verdict",
            "",
            _table([report["verdict"]], ["relative_allocator_operationally_valid", "relative_allocator_result_acceptable", "relative_allocator_beats_pm_ai_v2", "relative_allocator_close_to_v282", "pm_ai_ranking_hypothesis_supported", "next_phase_recommended"]),
            "",
        ]
    )


def save_report(report: dict[str, Any]) -> tuple[Path, Path]:
    report_dir = ROOT / "reports" / "ml"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{REPORT_STEM}.json"
    md_path = report_dir / f"{REPORT_STEM}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown(report), encoding="utf-8")
    return md_path, json_path


def main() -> int:
    report = build_report()
    md_path, json_path = save_report(report)
    verdict = report["verdict"]
    print(f"markdown={md_path}")
    print(f"json={json_path}")
    print(f"relative_allocator_operationally_valid={verdict.get('relative_allocator_operationally_valid')}")
    print(f"relative_allocator_result_acceptable={verdict.get('relative_allocator_result_acceptable')}")
    print(f"relative_allocator_beats_pm_ai_v2={verdict.get('relative_allocator_beats_pm_ai_v2')}")
    print(f"next_phase_recommended={verdict.get('next_phase_recommended')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
