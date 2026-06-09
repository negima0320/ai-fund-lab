"""Phase 9-F2 PM AI v3 backtest candidate audit on PM sizing universe.

This report reads existing backtest artifacts for research-only evaluation. It
does not train models, fetch J-Quants data, regenerate historical predictions,
or overwrite current PM/Exit/v2_82 artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_v3_dataset_builder import FORBIDDEN_TOKENS, LABEL_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
PERIOD = "2023-01-01_to_2026-05-31"
REPORT_STEM = "phase9f2_pm_ai_v3_backtest_candidate_2023-01_to_2026-05"
BASELINE_PROFILE = "rookie_dealer_02_v2_82_cap38"
CANDIDATE_PROFILES = {
    "v2_94_e139": "rookie_dealer_02_v2_94_pm_ai_v3_e139_candidate",
    "v2_94b_e140": "rookie_dealer_02_v2_94b_pm_ai_v3_e140_candidate",
    "v2_94c_e120": "rookie_dealer_02_v2_94c_pm_ai_v3_e120_candidate",
    "v2_94d_rank_score": "rookie_dealer_02_v2_94d_pm_ai_v3_rank_score_candidate",
    "v2_94e_rank_downside_blend": "rookie_dealer_02_v2_94e_pm_ai_v3_rank_downside_blend_candidate",
}
PROFILE_LABELS = {"v2_82_cap38": BASELINE_PROFILE, **CANDIDATE_PROFILES}
PM_V3_FEATURE_COLUMNS = Path("models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe/feature_columns.json")


@dataclass(frozen=True)
class Phase9F2Paths:
    markdown: Path
    json: Path


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


def _mean(series: pd.Series | None) -> float | None:
    values = _numeric(series).dropna()
    return float(values.mean()) if not values.empty else None


def _profit_factor(values: pd.Series | None) -> float | None:
    profits = _numeric(values).dropna()
    if profits.empty:
        return None
    gross_profit = float(profits[profits > 0].sum())
    gross_loss = abs(float(profits[profits < 0].sum()))
    return gross_profit / gross_loss if gross_loss else None


def _win_rate(values: pd.Series | None) -> float | None:
    profits = _numeric(values).dropna()
    return float((profits > 0).mean()) if not profits.empty else None


class PMAIV3BacktestCandidateAuditPMSizingUniverse:
    def __init__(self, root: Path | str = ROOT, *, period: str = PERIOD) -> None:
        self.root = Path(root)
        self.period = period

    def build_report(self) -> dict[str, Any]:
        profiles = {label: self._profile_payload(label, profile) for label, profile in PROFILE_LABELS.items()}
        leakage = self._leakage_guard()
        baseline = profiles["v2_82_cap38"]["summary"]
        candidate_rows = [payload["summary"] for label, payload in profiles.items() if label != "v2_82_cap38"]
        return {
            "metadata": {
                "phase": "9-F2",
                "research_only": True,
                "backtest_candidate_audit": True,
                "training_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_model_regenerated_historical_predictions": False,
                "old_candidate_phase9d_model_used": False,
                "old_top10_fixed_dataset_used": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
            },
            "period": self.period,
            "profiles": PROFILE_LABELS,
            "profile_summaries": [payload["summary"] for payload in profiles.values()],
            "pm_v3_coverage_by_profile": {label: payload["pm_v3_coverage"] for label, payload in profiles.items()},
            "missing_reason_distribution_by_profile": {label: payload["missing_reason_distribution"] for label, payload in profiles.items()},
            "pm_distribution_by_profile": {label: payload["pm_distribution"] for label, payload in profiles.items()},
            "pm_quality_by_profile": {label: payload["pm_quality"] for label, payload in profiles.items()},
            "current_pm130_comparison": self._current_pm130_comparison(profiles),
            "year2026_pm130_audit": self._year2026_pm130_audit(profiles),
            "leakage_checklist": leakage,
            "adoption_gate": self._adoption_gate(candidate_rows, baseline, profiles, leakage),
        }

    def save_report(self, report: dict[str, Any]) -> Phase9F2Paths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9F2Paths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 9-F2 Backtest Candidate Audit",
                "",
                "## Scope",
                "",
                "- research-only coverage-fixed PM AI v3 backtest candidate audit",
                "- current PM AI, current Exit AI, and v2_82 profile are not overwritten",
                "- backtest results are used for evaluation only, never as PM AI v3 features or labels",
                "",
                "## Profile Comparison",
                "",
                self._table(report["profile_summaries"], ["label", "profile", "status", "net_profit", "profit_factor", "max_drawdown", "win_rate", "monthly_win_rate", "cagr", "total_trades", "final_assets", "average_capital_utilization"]),
                "",
                "## PM v3 Coverage",
                "",
                self._table(report["pm_v3_coverage_by_profile"].values(), ["label", "buy_count", "pm_feature_found_count", "pm_feature_coverage", "coverage_gate_passed", "top_missing_reason"]),
                "",
                "## PM Distribution",
                "",
                self._profile_table(report["pm_distribution_by_profile"], ["pm_multiplier", "buy_count"]),
                "",
                "## PM Quality",
                "",
                self._profile_table(report["pm_quality_by_profile"], ["pm_multiplier", "buy_count", "trade_count", "profit", "profit_factor", "win_rate", "downside", "average_holding_days"]),
                "",
                "## Current PM1.30 Comparison",
                "",
                self._table(report["current_pm130_comparison"], ["label", "profile", "pm130_trade_count", "pm130_profit", "pm130_profit_factor", "pm130_win_rate", "pm130_downside", "pm130_buy_count", "overlap_count", "overlap_rate", "meets_current_pm130_quality"]),
                "",
                "## 2026 PM1.30 Audit",
                "",
                self._table(report["year2026_pm130_audit"], ["label", "profile", "pm130_buy_count", "pm130_trade_count", "pm130_profit", "pm130_profit_factor", "pm130_win_rate", "pm130_downside", "pm130_firing_risk"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_checklist"]], ["forbidden_feature_count", "forbidden_feature_columns", "label_columns_in_features", "future_columns_in_features", "leakage_risk", "backtest_results_used_as_features", "current_pm_multiplier_used_as_label"]),
                "",
                "## Adoption Gate",
                "",
                self._table([report["adoption_gate"]], ["best_candidate_label", "coverage_gate", "pf_gate", "drawdown_gate", "profit_gate", "pm130_quality_gate", "leakage_gate", "all_gates_passed", "adoption_recommendation", "beats_v2_82"]),
                "",
            ]
        )

    def _profile_payload(self, label: str, profile: str) -> dict[str, Any]:
        base = self.root / "logs" / "backtests" / profile / self.period
        summary = _read_json(base / "backtest_summary.json")
        trades = _read_csv(base / "trades.csv")
        daily = _read_csv(base / "summary.csv")
        audit = _read_csv(base / "purchase_audit.csv")
        sells = self._sell_trades(trades)
        return {
            "summary": self._summary_row(label, profile, summary, trades, daily),
            "pm_v3_coverage": self._pm_v3_coverage(label, audit),
            "missing_reason_distribution": self._missing_reason_distribution(audit),
            "pm_distribution": self._pm_distribution(audit),
            "pm_quality": self._pm_quality(audit, sells),
            "buy_audit": audit,
            "sell_trades": sells,
        }

    def _summary_row(self, label: str, profile: str, summary: dict[str, Any], trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
        final_assets = summary.get("final_assets")
        initial = summary.get("initial_capital") or 1_000_000
        return {
            "label": label,
            "profile": profile,
            "status": "ok" if summary else "missing_backtest_logs",
            "net_profit": summary.get("net_cumulative_profit"),
            "profit_factor": summary.get("profit_factor"),
            "max_drawdown": summary.get("max_drawdown"),
            "win_rate": summary.get("win_rate"),
            "monthly_win_rate": self._monthly_win_rate(trades),
            "cagr": self._cagr(final_assets, initial),
            "total_trades": summary.get("closed_trades_count") or summary.get("closed_trade_count") or summary.get("total_trades"),
            "final_assets": final_assets,
            "average_capital_utilization": self._average_capital_utilization(daily),
        }

    def _sell_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()
        if "action" not in trades.columns:
            return trades.copy()
        return trades[trades["action"].fillna("").astype(str).eq("SELL")].copy()

    def _buy_rows(self, audit: pd.DataFrame) -> pd.DataFrame:
        if audit.empty:
            return audit
        if "decision" not in audit.columns:
            return audit.copy()
        return audit[audit["decision"].fillna("").astype(str).eq("BUY")].copy()

    def _pm_v3_coverage(self, label: str, audit: pd.DataFrame) -> dict[str, Any]:
        buys = self._buy_rows(audit)
        if buys.empty:
            return {"label": label, "buy_count": 0, "pm_feature_found_count": 0, "pm_feature_coverage": None, "coverage_gate_passed": False, "top_missing_reason": ""}
        found = buys.get("pm_feature_found", pd.Series(False, index=buys.index)).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})
        missing = buys.get("pm_missing_reason", pd.Series("", index=buys.index)).fillna("").astype(str)
        top_missing = ""
        counts = missing[missing.ne("")].value_counts()
        if not counts.empty:
            top_missing = str(counts.index[0])
        coverage = float(found.mean()) if len(buys) else None
        return {
            "label": label,
            "buy_count": int(len(buys)),
            "pm_feature_found_count": int(found.sum()),
            "pm_feature_coverage": coverage,
            "coverage_gate_passed": bool(coverage is not None and coverage >= 0.95),
            "top_missing_reason": top_missing,
        }

    def _missing_reason_distribution(self, audit: pd.DataFrame) -> list[dict[str, Any]]:
        buys = self._buy_rows(audit)
        if buys.empty or "pm_missing_reason" not in buys.columns:
            return []
        values = buys["pm_missing_reason"].fillna("").astype(str)
        counts = values[values.ne("")].value_counts()
        return [{"missing_reason": str(reason), "count": int(count)} for reason, count in counts.items()]

    def _pm_distribution(self, audit: pd.DataFrame) -> list[dict[str, Any]]:
        buys = self._buy_rows(audit)
        if buys.empty or "pm_multiplier" not in buys.columns:
            return []
        counts = _numeric(buys.get("pm_multiplier")).round(2).dropna().value_counts().sort_index()
        return [{"pm_multiplier": float(multiplier), "buy_count": int(count)} for multiplier, count in counts.items()]

    def _pm_quality(self, audit: pd.DataFrame, sells: pd.DataFrame) -> list[dict[str, Any]]:
        buys = self._pm_distribution(audit)
        buy_by = {round(row["pm_multiplier"], 2): row["buy_count"] for row in buys}
        rows = []
        pm = _numeric(sells.get("pm_multiplier")).round(2) if not sells.empty else pd.Series(dtype=float)
        profit = _numeric(sells.get("net_profit") if "net_profit" in sells.columns else sells.get("profit"))
        for multiplier in [0.60, 0.80, 1.00, 1.15, 1.30]:
            group = sells[pm.eq(multiplier)] if not sells.empty else pd.DataFrame()
            values = profit.loc[group.index] if not group.empty else pd.Series(dtype=float)
            rows.append(
                {
                    "pm_multiplier": multiplier,
                    "buy_count": int(buy_by.get(round(multiplier, 2), 0)),
                    "trade_count": int(len(group)),
                    "profit": float(values.sum()) if not values.empty else 0.0,
                    "profit_factor": _profit_factor(values),
                    "win_rate": _win_rate(values),
                    "downside": float(values[values < 0].sum()) if not values.empty else 0.0,
                    "average_holding_days": _mean(group.get("holding_days")) if not group.empty else None,
                }
            )
        return rows

    def _current_pm130_comparison(self, profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        baseline_quality = self._quality_row(profiles["v2_82_cap38"], 1.30)
        baseline_buys = self._pm130_buy_keys(profiles["v2_82_cap38"]["buy_audit"], minimum=1.30)
        rows = []
        for label, profile in PROFILE_LABELS.items():
            payload = profiles[label]
            row = self._quality_row(payload, 1.30)
            buy_keys = self._pm130_buy_keys(payload["buy_audit"], minimum=1.30)
            overlap = baseline_buys.intersection(buy_keys)
            rows.append(
                {
                    "label": label,
                    "profile": profile,
                    "pm130_trade_count": row.get("trade_count"),
                    "pm130_profit": row.get("profit"),
                    "pm130_profit_factor": row.get("profit_factor"),
                    "pm130_win_rate": row.get("win_rate"),
                    "pm130_downside": row.get("downside"),
                    "pm130_buy_count": row.get("buy_count"),
                    "overlap_count": int(len(overlap)),
                    "overlap_rate": float(len(overlap) / len(buy_keys)) if buy_keys else None,
                    "meets_current_pm130_quality": self._meets_pm130_quality(row, baseline_quality),
                }
            )
        return rows

    def _year2026_pm130_audit(self, profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for label, profile in PROFILE_LABELS.items():
            buys = self._buy_rows(profiles[label]["buy_audit"])
            sells = profiles[label]["sell_trades"]
            buy_dates = pd.to_datetime(buys.get("signal_date"), errors="coerce") if not buys.empty else pd.Series(dtype="datetime64[ns]")
            sell_dates = pd.to_datetime(sells.get("signal_date"), errors="coerce") if not sells.empty else pd.Series(dtype="datetime64[ns]")
            buy_2026 = buys[buy_dates.dt.year.eq(2026)] if not buys.empty else buys
            sell_2026 = sells[sell_dates.dt.year.eq(2026)] if not sells.empty else sells
            quality = self._pm_quality(buy_2026, sell_2026)
            pm130 = next((row for row in quality if row["pm_multiplier"] == 1.30), {})
            rows.append(
                {
                    "label": label,
                    "profile": profile,
                    "pm130_buy_count": pm130.get("buy_count", 0),
                    "pm130_trade_count": pm130.get("trade_count", 0),
                    "pm130_profit": pm130.get("profit", 0.0),
                    "pm130_profit_factor": pm130.get("profit_factor"),
                    "pm130_win_rate": pm130.get("win_rate"),
                    "pm130_downside": pm130.get("downside", 0.0),
                    "pm130_firing_risk": "high" if int(pm130.get("buy_count", 0) or 0) < 10 else "low",
                }
            )
        return rows

    def _quality_row(self, payload: dict[str, Any], multiplier: float) -> dict[str, Any]:
        return next((row for row in payload["pm_quality"] if round(row["pm_multiplier"], 2) == round(multiplier, 2)), {})

    def _pm130_buy_keys(self, audit: pd.DataFrame, *, minimum: float) -> set[str]:
        buys = self._buy_rows(audit)
        if buys.empty or "pm_multiplier" not in buys.columns:
            return set()
        rows = buys[_numeric(buys.get("pm_multiplier")).round(2).ge(minimum)].copy()
        dates = pd.to_datetime(rows.get("signal_date"), errors="coerce").dt.strftime("%Y-%m-%d")
        codes = rows.get("code", pd.Series("", index=rows.index)).astype(str)
        return set((dates + "|" + codes).dropna())

    def _meets_pm130_quality(self, row: dict[str, Any], baseline: dict[str, Any]) -> bool:
        if not row or not baseline:
            return False
        checks = [
            (row.get("trade_count") or 0) > 0,
            (row.get("profit") or -10**18) >= (baseline.get("profit") or 0),
            (row.get("profit_factor") or -1) >= (baseline.get("profit_factor") or 0),
            (row.get("win_rate") or -1) >= (baseline.get("win_rate") or 0),
        ]
        return all(checks)

    def _adoption_gate(self, candidate_rows: list[dict[str, Any]], baseline: dict[str, Any], profiles: dict[str, dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        baseline_pm130 = self._quality_row(profiles["v2_82_cap38"], 1.30)
        best = max(candidate_rows, key=lambda row: row.get("net_profit") or -10**18, default={})
        best_label = best.get("label")
        coverage = profiles.get(best_label, {}).get("pm_v3_coverage", {}) if best_label else {}
        best_pm130 = self._quality_row(profiles.get(best_label, {}), 1.30) if best_label else {}
        gates = {
            "coverage_gate": bool((coverage.get("pm_feature_coverage") or 0.0) >= 0.95),
            "pf_gate": bool((best.get("profit_factor") or -1) >= (baseline.get("profit_factor") or 0)),
            "drawdown_gate": bool((best.get("max_drawdown") or -1) >= (baseline.get("max_drawdown") or 0)),
            "profit_gate": bool((best.get("net_profit") or -10**18) >= 0.95 * (baseline.get("net_profit") or 0)),
            "pm130_quality_gate": self._meets_pm130_quality(best_pm130, baseline_pm130),
            "leakage_gate": leakage["leakage_risk"] == "low",
        }
        passed = all(gates.values())
        return {
            "best_candidate_label": best_label,
            **gates,
            "all_gates_passed": passed,
            "adoption_recommendation": "adopt" if passed else "reject",
            "beats_v2_82": bool(
                (best.get("net_profit") or -10**18) >= (baseline.get("net_profit") or 0)
                and (best.get("profit_factor") or -1) >= (baseline.get("profit_factor") or 0)
                and (best.get("max_drawdown") or -1) >= (baseline.get("max_drawdown") or 0)
            ),
        }

    def _leakage_guard(self) -> dict[str, Any]:
        feature_path = self.root / PM_V3_FEATURE_COLUMNS
        features = [str(column) for column in json.loads(feature_path.read_text(encoding="utf-8"))] if feature_path.exists() else []
        forbidden = [column for column in features if any(token in column.lower() for token in FORBIDDEN_TOKENS)]
        labels = [column for column in features if column in LABEL_COLUMNS or "label" in column.lower() or "target" in column.lower()]
        future = [column for column in features if column.lower().startswith("future_")]
        return {
            "forbidden_feature_count": len(forbidden),
            "forbidden_feature_columns": forbidden,
            "label_columns_in_features": labels,
            "future_columns_in_features": future,
            "leakage_risk": "high" if forbidden or labels or future else "low",
            "backtest_results_used_as_features": False,
            "current_pm_multiplier_used_as_label": False,
            "current_pm_ai_overwritten": False,
            "current_exit_ai_overwritten": False,
            "v2_82_profile_overwritten": False,
        }

    def _monthly_win_rate(self, trades: pd.DataFrame) -> float | None:
        if trades.empty:
            return None
        date_col = "exit_date" if "exit_date" in trades.columns else "entry_date"
        dates = pd.to_datetime(trades.get(date_col), errors="coerce")
        profits = _numeric(trades.get("net_profit") if "net_profit" in trades.columns else trades.get("profit"))
        monthly = profits.groupby(dates.dt.to_period("M")).sum().dropna()
        return float((monthly > 0).mean()) if not monthly.empty else None

    def _cagr(self, final_assets: Any, initial: Any) -> float | None:
        try:
            final_value = float(final_assets)
            initial_value = float(initial)
        except (TypeError, ValueError):
            return None
        years = 3.414
        return (final_value / initial_value) ** (1 / years) - 1 if initial_value > 0 and final_value > 0 else None

    def _average_capital_utilization(self, daily: pd.DataFrame) -> float | None:
        if daily.empty:
            return None
        if {"positions_value", "total_assets"}.issubset(daily.columns):
            ratio = _numeric(daily["positions_value"]) / _numeric(daily["total_assets"]).replace(0, pd.NA)
            return _mean(ratio)
        if "capital_utilization" in daily.columns:
            return _mean(daily["capital_utilization"])
        return None

    def _profile_table(self, values: dict[str, list[dict[str, Any]]], columns: list[str]) -> str:
        rows = []
        for label, entries in values.items():
            for row in entries:
                rows.append({"label": label, **row})
        return self._table(rows, ["label", *columns])

    def _table(self, rows: Any, columns: list[str]) -> str:
        rows = list(rows)
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(self._format(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, list):
            return ", ".join(str(item) for item in value[:10])
        return str(value).replace("\n", " ")


def build_phase9f2_pm_ai_v3_backtest_candidate(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3BacktestCandidateAuditPMSizingUniverse(root).build_report()

