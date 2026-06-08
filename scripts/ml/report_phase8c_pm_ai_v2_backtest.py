#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase8c_pm_ai_v2_backtest_2023-01_to_2026-05"
PERIOD = "2023-01-01_to_2026-05-31"
V282 = "rookie_dealer_02_v2_82_cap38"
V290 = "rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38"
PM_BUCKETS = [1.30, 1.15, 1.00, 0.80, 0.60]


@dataclass(frozen=True)
class RunPaths:
    label: str
    profile: str
    log_dir: Path


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
    return trades[trades["action"].fillna("").astype(str).eq("SELL")].copy()


def _profit_column(trades: pd.DataFrame) -> pd.Series:
    if "net_profit" in trades.columns:
        return _numeric(trades["net_profit"])
    if "profit" in trades.columns:
        return _numeric(trades["profit"])
    return pd.Series(0.0, index=trades.index)


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


def _cagr(initial: float | None, final: float | None) -> float | None:
    if not initial or not final or initial <= 0 or final <= 0:
        return None
    start, end = PERIOD.split("_to_")
    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0)
    if years <= 0:
        return None
    return (final / initial) ** (1 / years) - 1


def _bool_series(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    return series.reindex(index).fillna(False).map(lambda v: str(v).strip().lower() in {"1", "true", "yes", "y"})


def _run_paths() -> list[RunPaths]:
    return [
        RunPaths(
            "v2_82_cap38",
            V282,
            ROOT / "reports" / "final" / "v2_82_cap38" / "core_2023-01_to_2026-05",
        ),
        RunPaths(
            "v2_90_pm_ai_v2_api_only_cap38",
            V290,
            ROOT / "logs" / "backtests" / V290 / PERIOD,
        ),
    ]


def _load_run(run: RunPaths) -> dict[str, Any]:
    daily = _read_csv(run.log_dir / "summary.csv")
    trades = _sell_trades(_read_csv(run.log_dir / "trades.csv"))
    purchase = _read_csv(run.log_dir / "purchase_audit.csv")
    summary = _read_json(run.log_dir / "backtest_summary.json")
    return {
        "run": run,
        "daily": daily,
        "trades": trades,
        "purchase": purchase,
        "summary": summary,
    }


def _basic_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    run: RunPaths = payload["run"]
    daily = payload["daily"]
    trades = payload["trades"]
    summary = payload["summary"]
    profits = _profit_column(trades)
    initial = 1_000_000.0
    final_assets = None
    if not daily.empty and "total_assets" in daily.columns:
        final_assets = float(_numeric(daily["total_assets"]).dropna().iloc[-1])
    total_assets = _numeric(daily.get("total_assets")) if not daily.empty else pd.Series(dtype=float)
    positions = _numeric(daily.get("positions_value")) if not daily.empty else pd.Series(dtype=float)
    utilization = positions / total_assets.replace(0, pd.NA) if not total_assets.empty else pd.Series(dtype=float)
    return {
        "label": run.label,
        "profile": run.profile,
        "net_profit": _summary_number(summary, "net_cumulative_profit", "net_profit") if summary else (float(profits.sum()) if not profits.empty else None),
        "profit_factor": _summary_number(summary, "profit_factor") if summary else _pf(profits),
        "max_drawdown": _summary_number(summary, "max_drawdown") if summary else _drawdown(daily).get("max_drawdown"),
        "win_rate": _summary_number(summary, "win_rate") if summary else _win_rate(profits),
        "monthly_win_rate": _monthly_win_rate(trades),
        "total_trades": int(_summary_number(summary, "total_trades") or len(trades)),
        "average_holding_days": float(_numeric(trades.get("holding_days")).mean()) if not trades.empty else None,
        "average_capital_utilization": float(utilization.dropna().mean()) if not utilization.dropna().empty else None,
        "initial_assets": initial,
        "final_assets": final_assets,
        "cagr": _cagr(initial, final_assets),
    }


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


def _drawdown(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or "total_assets" not in daily.columns:
        return {"max_drawdown": None}
    dates = pd.to_datetime(daily.get("date"), errors="coerce")
    assets = _numeric(daily["total_assets"])
    curve = pd.DataFrame({"date": dates, "assets": assets}).dropna()
    if curve.empty:
        return {"max_drawdown": None}
    curve["peak"] = curve["assets"].cummax()
    curve["drawdown"] = curve["assets"] / curve["peak"] - 1.0
    trough_idx = curve["drawdown"].idxmin()
    peak_idx = curve.loc[:trough_idx, "assets"].idxmax()
    return {
        "max_drawdown": float(curve.loc[trough_idx, "drawdown"]),
        "dd_start": curve.loc[peak_idx, "date"].strftime("%Y-%m-%d"),
        "dd_trough": curve.loc[trough_idx, "date"].strftime("%Y-%m-%d"),
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
                "profit_factor": _pf(profits),
                "win_rate": _win_rate(profits),
            }
        )
    return rows


def _pm_quality(payload: dict[str, Any]) -> dict[str, Any]:
    trades = payload["trades"]
    high = _numeric(trades.get("pm_high_conviction_proba")) if not trades.empty else pd.Series(dtype=float)
    avoid = _numeric(trades.get("pm_avoid_proba")) if not trades.empty else pd.Series(dtype=float)
    return {
        "label": payload["run"].label,
        "high_conviction_rate": float(high.ge(0.5).mean()) if not high.empty else None,
        "avoid_rate": float(avoid.ge(0.5).mean()) if not avoid.empty else None,
        "candidate_prediction_coverage": _candidate_prediction_coverage(trades),
        "candidate_pm130_exists": bool(_numeric(trades.get("pm_multiplier")).round(2).eq(1.30).any()) if not trades.empty else False,
    }


def _candidate_prediction_coverage(trades: pd.DataFrame) -> float | None:
    if trades.empty:
        return None
    if "pm_candidate_prediction_available" in trades.columns and trades["pm_candidate_prediction_available"].notna().any():
        values = _bool_series(trades["pm_candidate_prediction_available"], trades.index)
        return float(values.mean()) if len(values) else None
    model_version = trades.get("pm_model_version", pd.Series("", index=trades.index)).fillna("").astype(str)
    if model_version.str.contains("api_only|candidate_v2_api_only", regex=True).any() and "pm_feature_found" in trades.columns:
        values = _bool_series(trades["pm_feature_found"], trades.index)
        return float(values.mean()) if len(values) else None
    if "pm_candidate_prediction_available" not in trades.columns:
        return None
    values = _bool_series(trades["pm_candidate_prediction_available"], trades.index)
    return float(values.mean()) if len(values) else None


def _risk_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    purchase = payload["purchase"]
    trades = payload["trades"]
    if purchase.empty:
        selected_but_not_affordable = insufficient = cap_count = 0
    else:
        selected_but_not_affordable = int(purchase.get("skip_reason", pd.Series("", index=purchase.index)).fillna("").astype(str).eq("selected_but_not_affordable").sum())
        insufficient = int(purchase.get("skip_reason", pd.Series("", index=purchase.index)).fillna("").astype(str).eq("insufficient_available_cash").sum())
        reduced = _bool_series(purchase.get("pm_per_code_cap_reduced"), purchase.index)
        skipped = _bool_series(purchase.get("pm_per_code_cap_skip"), purchase.index)
        cap_count = int((reduced | skipped).sum())
    concentration = _concentration(trades)
    dd = _drawdown(payload["daily"])
    return {
        "label": payload["run"].label,
        "selected_but_not_affordable": selected_but_not_affordable,
        "insufficient_available_cash": insufficient,
        "per_code_cap_skip_or_reduction_count": cap_count,
        **concentration,
        **dd,
    }


def _concentration(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty or "code" not in trades.columns:
        return {"top1_profit_share": None, "top5_profit_share": None}
    by_code = _profit_column(trades).groupby(trades["code"].astype(str)).sum().sort_values(ascending=False)
    total = float(by_code.sum())
    if total == 0:
        return {"top1_profit_share": None, "top5_profit_share": None}
    return {
        "top1_profit_share": float(by_code.head(1).sum() / total),
        "top5_profit_share": float(by_code.head(5).sum() / total),
    }


def _diff(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "net_profit",
        "profit_factor",
        "max_drawdown",
        "win_rate",
        "average_capital_utilization",
    ]
    row = {"base": base["label"], "candidate": candidate["label"]}
    for key in keys:
        b = base.get(key)
        c = candidate.get(key)
        row[f"{key}_delta"] = None if b is None or c is None else c - b
    return row


def _quality_comparison(v282: dict[str, Any], v290: dict[str, Any]) -> dict[str, Any]:
    base_pm = {row["pm_multiplier"]: row for row in _pm_distribution(v282)}
    cand_pm = {row["pm_multiplier"]: row for row in _pm_distribution(v290)}
    return {
        "candidate_pm130_exists": bool(cand_pm.get(1.30, {}).get("trades")),
        "current_pm130_profit_reference": base_pm.get(1.30, {}).get("profit"),
        "candidate_pm130_profit": cand_pm.get(1.30, {}).get("profit"),
        "candidate_pm080_trades": cand_pm.get(0.80, {}).get("trades"),
        "candidate_pm060_trades": cand_pm.get(0.60, {}).get("trades"),
        "pm080_overuse_risk": (cand_pm.get(0.80, {}).get("trades") or 0) > (base_pm.get(0.80, {}).get("trades") or 0) * 1.5,
    }


def _verdict(basic: list[dict[str, Any]], pm_quality: list[dict[str, Any]], quality: dict[str, Any]) -> dict[str, Any]:
    by_label = {row["label"]: row for row in basic}
    v290 = by_label.get("v2_90_pm_ai_v2_api_only_cap38", {})
    profit = v290.get("net_profit")
    pf = v290.get("profit_factor")
    dd = v290.get("max_drawdown")
    wr = v290.get("win_rate")
    coverage = {row["label"]: row for row in pm_quality}.get("v2_90_pm_ai_v2_api_only_cap38", {}).get("candidate_prediction_coverage")
    operational = bool(coverage is None or coverage >= 0.90)
    acceptable = bool(
        profit is not None
        and pf is not None
        and dd is not None
        and wr is not None
        and pf >= 2.0
        and dd >= -0.10
        and wr >= 0.50
        and not quality.get("pm080_overuse_risk")
        and quality.get("candidate_pm130_exists")
    )
    return {
        "pm_ai_v2_operationally_valid": operational,
        "pm_ai_v2_result_acceptable": acceptable,
        "pm_ai_v2_should_replace_current": bool(operational and acceptable and profit and profit >= 3_777_545),
        "if_not_why": "" if acceptable else "PM AI v2 result must preserve PF/DD/win-rate and avoid extreme conservative multiplier collapse.",
        "next_phase_recommended": "Promote PM AI v2 candidate profile review" if acceptable else "Stay current PM and redesign PM AI v2 calibration",
    }


def build_report() -> dict[str, Any]:
    loaded = [_load_run(run) for run in _run_paths()]
    basic = [_basic_metrics(payload) for payload in loaded]
    pm_distribution = [row for payload in loaded for row in _pm_distribution(payload)]
    pm_quality = [_pm_quality(payload) for payload in loaded]
    risks = [_risk_metrics(payload) for payload in loaded]
    by_label = {payload["run"].label: payload for payload in loaded}
    basic_by_label = {row["label"]: row for row in basic}
    quality = _quality_comparison(by_label["v2_82_cap38"], by_label["v2_90_pm_ai_v2_api_only_cap38"])
    return {
        "metadata": {
            "phase": "8-C",
            "current_model_overwritten": False,
            "exit_ai_v2_candidate_integrated": False,
            "live_order_placement": False,
            "period": PERIOD,
        },
        "sources": {payload["run"].label: str(payload["run"].log_dir) for payload in loaded},
        "basic_comparison": basic,
        "difference": _diff(basic_by_label["v2_82_cap38"], basic_by_label["v2_90_pm_ai_v2_api_only_cap38"]),
        "pm_multiplier_distribution": pm_distribution,
        "pm_quality": pm_quality,
        "risk_metrics": risks,
        "quality_checks": quality,
        "verdict": _verdict(basic, pm_quality, quality),
    }


def save_report(report: dict[str, Any]) -> tuple[Path, Path]:
    report_dir = ROOT / "reports" / "ml"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{REPORT_STEM}.json"
    md_path = report_dir / f"{REPORT_STEM}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown(report), encoding="utf-8")
    return md_path, json_path


def format_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Portfolio Manager AI Phase 8-C PM AI v2 Backtest Report",
            "",
            "## Basic Comparison",
            "",
            _table(report["basic_comparison"], ["label", "net_profit", "profit_factor", "max_drawdown", "win_rate", "monthly_win_rate", "total_trades", "average_holding_days", "average_capital_utilization", "final_assets", "cagr"]),
            "",
            "## Difference",
            "",
            _table([report["difference"]], ["net_profit_delta", "profit_factor_delta", "max_drawdown_delta", "win_rate_delta", "average_capital_utilization_delta"]),
            "",
            "## PM Multiplier Distribution",
            "",
            _table(report["pm_multiplier_distribution"], ["label", "pm_multiplier", "trades", "profit", "profit_factor", "win_rate"]),
            "",
            "## PM Quality",
            "",
            _table(report["pm_quality"], ["label", "high_conviction_rate", "avoid_rate", "candidate_prediction_coverage", "candidate_pm130_exists"]),
            "",
            "## Risk Metrics",
            "",
            _table(report["risk_metrics"], ["label", "selected_but_not_affordable", "insufficient_available_cash", "per_code_cap_skip_or_reduction_count", "top1_profit_share", "top5_profit_share", "max_drawdown", "dd_start", "dd_trough"]),
            "",
            "## Quality Checks",
            "",
            _table([report["quality_checks"]], ["candidate_pm130_exists", "current_pm130_profit_reference", "candidate_pm130_profit", "candidate_pm080_trades", "candidate_pm060_trades", "pm080_overuse_risk"]),
            "",
            "## Verdict",
            "",
            _table([report["verdict"]], ["pm_ai_v2_operationally_valid", "pm_ai_v2_result_acceptable", "pm_ai_v2_should_replace_current", "if_not_why", "next_phase_recommended"]),
            "",
        ]
    )


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")
    return "\n".join([header, sep, *body])


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def main() -> int:
    report = build_report()
    md_path, json_path = save_report(report)
    verdict = report["verdict"]
    print(f"markdown={md_path}")
    print(f"json={json_path}")
    print(f"pm_ai_v2_operationally_valid={verdict.get('pm_ai_v2_operationally_valid')}")
    print(f"pm_ai_v2_result_acceptable={verdict.get('pm_ai_v2_result_acceptable')}")
    print(f"pm_ai_v2_should_replace_current={verdict.get('pm_ai_v2_should_replace_current')}")
    print(f"next_phase_recommended={verdict.get('next_phase_recommended')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
