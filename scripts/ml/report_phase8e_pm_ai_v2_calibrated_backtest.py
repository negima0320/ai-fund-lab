#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase8e_pm_ai_v2_calibrated_backtest_2023-01_to_2026-05"
PERIOD = "2023-01-01_to_2026-05-31"
V282 = "rookie_dealer_02_v2_82_cap38"
V290 = "rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38"
V291 = "rookie_dealer_02_v2_91_pm_ai_v2_calibrated_rule_e_cap38"
PM_BUCKETS = [1.30, 1.15, 1.00, 0.80, 0.60]


@dataclass(frozen=True)
class RunPaths:
    label: str
    profile: str
    log_dir: Path


def _run_paths() -> list[RunPaths]:
    return [
        RunPaths("v2_82_cap38", V282, ROOT / "reports" / "final" / "v2_82_cap38" / "core_2023-01_to_2026-05"),
        RunPaths("v2_90_pm_ai_v2_api_only_cap38", V290, ROOT / "logs" / "backtests" / V290 / PERIOD),
        RunPaths("v2_91_pm_ai_v2_calibrated_rule_e_cap38", V291, ROOT / "logs" / "backtests" / V291 / PERIOD),
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


def _bool_series(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    return series.reindex(index).fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})


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
    curve = pd.DataFrame(
        {
            "date": pd.to_datetime(daily.get("date"), errors="coerce"),
            "assets": _numeric(daily["total_assets"]),
        }
    ).dropna()
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
    positions = _numeric(daily.get("positions_value")) if not daily.empty else pd.Series(dtype=float)
    utilization = positions / total_assets.replace(0, pd.NA) if not total_assets.empty else pd.Series(dtype=float)
    final_assets = float(total_assets.dropna().iloc[-1]) if not total_assets.dropna().empty else None
    dd = _drawdown(daily)
    return {
        "label": run.label,
        "profile": run.profile,
        "net_profit": _summary_number(summary, "net_cumulative_profit", "net_profit") if summary else float(profits.sum()),
        "profit_factor": _summary_number(summary, "profit_factor") if summary else _profit_factor(profits),
        "max_drawdown": _summary_number(summary, "max_drawdown") if summary else dd["max_drawdown"],
        "win_rate": _summary_number(summary, "win_rate") if summary else _win_rate(profits),
        "monthly_win_rate": _monthly_win_rate(trades),
        "total_trades": int(_summary_number(summary, "total_trades") or len(trades)),
        "average_holding_days": float(_numeric(trades.get("holding_days")).mean()) if not trades.empty else None,
        "average_capital_utilization": float(utilization.dropna().mean()) if not utilization.dropna().empty else None,
        "final_assets": final_assets,
        "cagr": _cagr(1_000_000.0, final_assets),
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


def _pm_quality(payload: dict[str, Any]) -> dict[str, Any]:
    trades = payload["trades"]
    high = _numeric(trades.get("pm_high_conviction_proba")) if not trades.empty else pd.Series(dtype=float)
    avoid = _numeric(trades.get("pm_avoid_proba")) if not trades.empty else pd.Series(dtype=float)
    raw = _numeric(trades.get("pm_candidate_multiplier_raw")) if not trades.empty else pd.Series(dtype=float)
    calibrated = _numeric(trades.get("pm_candidate_multiplier_calibrated")) if not trades.empty else pd.Series(dtype=float)
    raw_calibrated_changed = 0
    if not raw.empty and not calibrated.empty:
        comparable = pd.DataFrame({"raw": raw, "calibrated": calibrated}).dropna()
        raw_calibrated_changed = int(comparable["raw"].ne(comparable["calibrated"]).sum())
    return {
        "label": payload["run"].label,
        "high_conviction_rate": float(high.ge(0.5).mean()) if not high.empty else None,
        "avoid_rate": float(avoid.ge(0.5).mean()) if not avoid.empty else None,
        "candidate_prediction_coverage": _candidate_prediction_coverage(trades),
        "raw_to_calibrated_changed_count": raw_calibrated_changed,
        "calibration_thresholds_used": _first_non_empty(trades.get("pm_calibration_thresholds")),
    }


def _candidate_prediction_coverage(trades: pd.DataFrame) -> float | None:
    if trades.empty:
        return None
    if "pm_candidate_prediction_available" in trades.columns and trades["pm_candidate_prediction_available"].notna().any():
        return float(_bool_series(trades["pm_candidate_prediction_available"], trades.index).mean())
    if "pm_feature_found" in trades.columns:
        return float(_bool_series(trades["pm_feature_found"], trades.index).mean())
    return None


def _first_non_empty(series: pd.Series | None) -> str:
    if series is None:
        return ""
    values = series.dropna().astype(str)
    values = values[values.ne("")]
    return values.iloc[0] if not values.empty else ""


def _risk_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    purchase = payload["purchase"]
    trades = payload["trades"]
    selected = int(purchase.get("skip_reason", pd.Series("", index=purchase.index)).fillna("").astype(str).eq("selected_but_not_affordable").sum()) if not purchase.empty else 0
    insufficient = int(purchase.get("skip_reason", pd.Series("", index=purchase.index)).fillna("").astype(str).eq("insufficient_available_cash").sum()) if not purchase.empty else 0
    cap = 0
    if not purchase.empty:
        cap = int((_bool_series(purchase.get("pm_per_code_cap_reduced"), purchase.index) | _bool_series(purchase.get("pm_per_code_cap_skip"), purchase.index)).sum())
    by_code = _profit_column(trades).groupby(trades.get("code", pd.Series(dtype=str)).astype(str)).sum().sort_values(ascending=False) if not trades.empty and "code" in trades.columns else pd.Series(dtype=float)
    total = float(by_code.sum()) if not by_code.empty else 0.0
    dd = _drawdown(payload["daily"])
    return {
        "label": payload["run"].label,
        "selected_but_not_affordable": selected,
        "insufficient_available_cash": insufficient,
        "per_code_cap_skip_or_reduction_count": cap,
        "top1_profit_share": float(by_code.head(1).sum() / total) if total else None,
        "top5_profit_share": float(by_code.head(5).sum() / total) if total else None,
        **dd,
    }


def _trade_key_frame(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["key", "pm_multiplier", "profit"])
    buy_date = trades.get("entry_date", trades.get("buy_date", pd.Series("", index=trades.index))).astype(str)
    key = trades.get("code", pd.Series("", index=trades.index)).astype(str) + "|" + buy_date
    return pd.DataFrame({"key": key, "pm_multiplier": _numeric(trades.get("pm_multiplier")).round(2), "profit": _profit_column(trades)})


def _quality_checks(v282: dict[str, Any], v291: dict[str, Any]) -> dict[str, Any]:
    base = _trade_key_frame(v282["trades"])
    cand = _trade_key_frame(v291["trades"])
    if base.empty or cand.empty:
        return {}
    merged = base.merge(cand, on="key", how="outer", suffixes=("_v282", "_v291"))
    base_pm130 = merged["pm_multiplier_v282"].eq(1.30)
    cand_pm130 = merged["pm_multiplier_v291"].eq(1.30)
    recovered = merged[base_pm130 & cand_pm130]
    missed = merged[base_pm130 & ~cand_pm130]
    base_pm080 = merged["pm_multiplier_v282"].eq(0.80)
    cand_pm080 = merged["pm_multiplier_v291"].eq(0.80)
    promoted = merged[merged["pm_multiplier_v291"].fillna(0) > merged["pm_multiplier_v282"].fillna(0)]
    demoted = merged[merged["pm_multiplier_v291"].fillna(0) < merged["pm_multiplier_v282"].fillna(0)]
    return {
        "current_pm130_recovery_count": int(len(recovered)),
        "current_pm130_recall": float(len(recovered) / base_pm130.sum()) if base_pm130.sum() else None,
        "pm130_recovered_profit": float(recovered["profit_v282"].fillna(0).sum()),
        "pm130_missed_profit": float(missed["profit_v282"].fillna(0).sum()),
        "pm080_overuse_risk": bool(cand_pm080.sum() > base_pm080.sum() * 1.5),
        "promoted_trades": int(len(promoted)),
        "promoted_trade_profit": float(promoted["profit_v291"].fillna(0).sum()),
        "demoted_trades": int(len(demoted)),
        "demoted_trade_profit": float(demoted["profit_v291"].fillna(0).sum()),
    }


def _diff(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    row = {"base": base["label"], "candidate": candidate["label"]}
    for key in ["net_profit", "profit_factor", "max_drawdown", "win_rate", "average_capital_utilization"]:
        row[f"{key}_delta"] = None if base.get(key) is None or candidate.get(key) is None else candidate[key] - base[key]
    return row


def _verdict(basic: list[dict[str, Any]], pm_quality: list[dict[str, Any]], quality: dict[str, Any]) -> dict[str, Any]:
    by_label = {row["label"]: row for row in basic}
    v291 = by_label.get("v2_91_pm_ai_v2_calibrated_rule_e_cap38", {})
    v290 = by_label.get("v2_90_pm_ai_v2_api_only_cap38", {})
    v282 = by_label.get("v2_82_cap38", {})
    coverage = {row["label"]: row for row in pm_quality}.get("v2_91_pm_ai_v2_calibrated_rule_e_cap38", {}).get("candidate_prediction_coverage")
    operational = bool(coverage is None or coverage >= 0.90)
    acceptable = bool(
        operational
        and (v291.get("net_profit") or 0) > (v290.get("net_profit") or 0)
        and (v291.get("profit_factor") or 0) >= 2.0
        and (v291.get("max_drawdown") or -1.0) >= -0.10
        and (v291.get("win_rate") or 0) >= 0.50
        and (v291.get("average_capital_utilization") or 0) > (v290.get("average_capital_utilization") or 0)
        and (quality.get("current_pm130_recovery_count") or 0) > 0
        and not quality.get("pm080_overuse_risk")
    )
    replace = bool(acceptable and (v291.get("net_profit") or 0) >= (v282.get("net_profit") or 0))
    why = ""
    if not acceptable:
        why = "Calibrated PM AI v2 must improve v2_90 while preserving PF/DD/win-rate, capital utilization, and PM1.30 recovery."
    elif not replace:
        why = "Operationally acceptable, but still below v2_82 replacement bar."
    return {
        "pm_ai_v2_calibrated_operationally_valid": operational,
        "pm_ai_v2_calibrated_result_acceptable": acceptable,
        "pm_ai_v2_calibrated_should_replace_current": replace,
        "if_not_why": why,
        "next_phase_recommended": "Promote v2_91 review" if replace else ("Tune calibration further" if acceptable else "Stay current PM and revisit PM AI v2 calibration"),
    }


def build_report() -> dict[str, Any]:
    loaded = [_load_run(run) for run in _run_paths()]
    basic = [_basic_metrics(payload) for payload in loaded]
    by_label = {payload["run"].label: payload for payload in loaded}
    basic_by_label = {row["label"]: row for row in basic}
    pm_distribution = [row for payload in loaded for row in _pm_distribution(payload)]
    pm_quality = [_pm_quality(payload) for payload in loaded]
    risk = [_risk_metrics(payload) for payload in loaded]
    quality = _quality_checks(by_label["v2_82_cap38"], by_label["v2_91_pm_ai_v2_calibrated_rule_e_cap38"])
    return {
        "metadata": {
            "phase": "8-E",
            "current_model_overwritten": False,
            "exit_ai_v2_candidate_integrated": False,
            "live_order_placement": False,
            "period": PERIOD,
        },
        "sources": {payload["run"].label: str(payload["run"].log_dir) for payload in loaded},
        "basic_comparison": basic,
        "difference_vs_v2_90": _diff(basic_by_label["v2_90_pm_ai_v2_api_only_cap38"], basic_by_label["v2_91_pm_ai_v2_calibrated_rule_e_cap38"]),
        "difference_vs_v2_82": _diff(basic_by_label["v2_82_cap38"], basic_by_label["v2_91_pm_ai_v2_calibrated_rule_e_cap38"]),
        "pm_multiplier_distribution": pm_distribution,
        "pm_quality": pm_quality,
        "quality_checks": quality,
        "risk_metrics": risk,
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
            "# Portfolio Manager AI Phase 8-E PM AI v2 Calibrated Backtest Report",
            "",
            "## Basic Comparison",
            _table(report["basic_comparison"], ["label", "net_profit", "profit_factor", "max_drawdown", "win_rate", "monthly_win_rate", "total_trades", "average_holding_days", "average_capital_utilization", "final_assets", "cagr"]),
            "",
            "## Difference vs v2_90",
            _table([report["difference_vs_v2_90"]], ["net_profit_delta", "profit_factor_delta", "max_drawdown_delta", "win_rate_delta", "average_capital_utilization_delta"]),
            "",
            "## Difference vs v2_82",
            _table([report["difference_vs_v2_82"]], ["net_profit_delta", "profit_factor_delta", "max_drawdown_delta", "win_rate_delta", "average_capital_utilization_delta"]),
            "",
            "## PM Multiplier Distribution",
            _table(report["pm_multiplier_distribution"], ["label", "pm_multiplier", "trades", "profit", "profit_factor", "win_rate"]),
            "",
            "## PM Quality",
            _table(report["pm_quality"], ["label", "high_conviction_rate", "avoid_rate", "candidate_prediction_coverage", "raw_to_calibrated_changed_count", "calibration_thresholds_used"]),
            "",
            "## Quality Checks",
            _table([report["quality_checks"]], ["current_pm130_recovery_count", "current_pm130_recall", "pm130_recovered_profit", "pm130_missed_profit", "pm080_overuse_risk", "promoted_trades", "promoted_trade_profit", "demoted_trades", "demoted_trade_profit"]),
            "",
            "## Risk Metrics",
            _table(report["risk_metrics"], ["label", "selected_but_not_affordable", "insufficient_available_cash", "per_code_cap_skip_or_reduction_count", "top1_profit_share", "top5_profit_share", "max_drawdown", "dd_start", "dd_trough"]),
            "",
            "## Verdict",
            _table([report["verdict"]], ["pm_ai_v2_calibrated_operationally_valid", "pm_ai_v2_calibrated_result_acceptable", "pm_ai_v2_calibrated_should_replace_current", "if_not_why", "next_phase_recommended"]),
            "",
        ]
    )


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(_format(row.get(column)) for column in columns) + " |" for row in rows]
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
    print(f"pm_ai_v2_calibrated_operationally_valid={verdict.get('pm_ai_v2_calibrated_operationally_valid')}")
    print(f"pm_ai_v2_calibrated_result_acceptable={verdict.get('pm_ai_v2_calibrated_result_acceptable')}")
    print(f"pm_ai_v2_calibrated_should_replace_current={verdict.get('pm_ai_v2_calibrated_should_replace_current')}")
    print(f"next_phase_recommended={verdict.get('next_phase_recommended')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
