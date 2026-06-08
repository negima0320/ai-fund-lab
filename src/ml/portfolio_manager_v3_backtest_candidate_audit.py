"""Phase 9-F PM AI v3 backtest candidate audit.

This report reads existing backtest artifacts only. It does not train models,
fetch J-Quants data, regenerate historical predictions, or overwrite current
PM/Exit/profile artifacts.
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
REPORT_STEM = "phase9f_pm_ai_v3_backtest_candidate_2023-01_to_2026-05"
BASELINE_PROFILE = "rookie_dealer_02_v2_82_cap38"
CANDIDATE_PROFILES = {
    "v2_93_a_mapping_a_rank_score_only": "rookie_dealer_02_v2_93_pm_ai_v3_candidate",
    "v2_93_b_conservative_pm130_threshold": "rookie_dealer_02_v2_93b_pm_ai_v3_candidate_conservative",
    "v2_93_c_half_pm130_candidates": "rookie_dealer_02_v2_93c_pm_ai_v3_candidate_half_pm130",
}
PROFILE_LABELS = {"v2_82_cap38": BASELINE_PROFILE, **CANDIDATE_PROFILES}
PM_V3_FEATURE_COLUMNS = Path("models/ml/portfolio_manager_v3/candidate_phase9d/feature_columns.json")


@dataclass(frozen=True)
class Phase9FPaths:
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


class PMAIV3BacktestCandidateAudit:
    def __init__(self, root: Path | str = ROOT, *, period: str = PERIOD) -> None:
        self.root = Path(root)
        self.period = period

    def build_report(self) -> dict[str, Any]:
        profiles = {label: self._profile_payload(label, profile) for label, profile in PROFILE_LABELS.items()}
        leakage = self._leakage_guard()
        baseline = profiles["v2_82_cap38"]["summary"]
        candidate_rows = [payload["summary"] for label, payload in profiles.items() if label != "v2_82_cap38"]
        adoption = self._adoption(candidate_rows, baseline, profiles, leakage)
        return {
            "metadata": {
                "phase": "9-F",
                "research_only": True,
                "backtest_audit_only": True,
                "training_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_model_regenerated_historical_predictions": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
            },
            "period": self.period,
            "profiles": PROFILE_LABELS,
            "profile_summaries": [payload["summary"] for payload in profiles.values()],
            "pm_distribution_by_profile": {label: payload["pm_distribution"] for label, payload in profiles.items()},
            "pm_lookup_status_by_profile": {label: payload["pm_lookup_status"] for label, payload in profiles.items()},
            "pm_quality_by_profile": {label: payload["pm_quality"] for label, payload in profiles.items()},
            "pm130_comparison": self._pm130_comparison(profiles),
            "leakage_checklist": leakage,
            "adoption": adoption,
        }

    def save_report(self, report: dict[str, Any]) -> Phase9FPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9FPaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 9-F Backtest Candidate Audit",
                "",
                "## Scope",
                "",
                "- research-only comparison of v2_82_cap38 versus v2_82 + PM AI v3 candidate mappings",
                "- current PM AI, current Exit AI, and v2_82 profile are not overwritten",
                "- backtest artifacts are used for evaluation only, never as PM AI v3 features",
                "",
                "## Profile Comparison",
                "",
                self._table(report["profile_summaries"], ["label", "profile", "status", "net_profit", "profit_factor", "max_drawdown", "win_rate", "monthly_win_rate", "cagr", "total_trades", "final_assets", "average_capital_utilization"]),
                "",
                "## PM Distribution",
                "",
                self._profile_table(report["pm_distribution_by_profile"], ["pm_multiplier", "buy_count"]),
                "",
                "## PM v3 Lookup Status",
                "",
                self._table(report["pm_lookup_status_by_profile"].values(), ["label", "buy_count", "pm_feature_found_count", "pm_feature_coverage", "top_missing_reason"]),
                "",
                "## PM Quality",
                "",
                self._profile_table(report["pm_quality_by_profile"], ["pm_multiplier", "trade_count", "profit", "profit_factor", "win_rate", "downside"]),
                "",
                "## PM1.30 Comparison",
                "",
                self._table(report["pm130_comparison"], ["label", "profile", "trade_count", "profit", "profit_factor", "win_rate", "downside", "meets_current_pm130_quality"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_checklist"]], ["forbidden_feature_count", "forbidden_feature_columns", "label_columns_in_features", "leakage_risk"]),
                "",
                "## Adoption",
                "",
                self._table([report["adoption"]], ["best_candidate_label", "strict_conditions_passed", "beats_v2_82", "adoption_recommendation", "reason"]),
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
            "pm_distribution": self._pm_distribution(audit),
            "pm_lookup_status": self._pm_lookup_status(label, audit),
            "pm_quality": self._pm_quality(sells),
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

    def _pm_distribution(self, audit: pd.DataFrame) -> list[dict[str, Any]]:
        if audit.empty or "pm_multiplier" not in audit.columns:
            return []
        rows = audit.copy()
        if "decision" in rows.columns:
            rows = rows[rows["decision"].fillna("").astype(str).eq("BUY")]
        counts = _numeric(rows.get("pm_multiplier")).round(2).dropna().value_counts().sort_index()
        return [{"pm_multiplier": float(multiplier), "buy_count": int(count)} for multiplier, count in counts.items()]

    def _pm_lookup_status(self, label: str, audit: pd.DataFrame) -> dict[str, Any]:
        if audit.empty:
            return {"label": label, "buy_count": 0, "pm_feature_found_count": 0, "pm_feature_coverage": None, "top_missing_reason": ""}
        rows = audit.copy()
        if "decision" in rows.columns:
            rows = rows[rows["decision"].fillna("").astype(str).eq("BUY")]
        found = rows.get("pm_feature_found", pd.Series(False, index=rows.index)).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})
        missing = rows.get("pm_missing_reason", pd.Series("", index=rows.index)).fillna("").astype(str)
        top_missing = ""
        if not missing.empty:
            counts = missing[missing.ne("")].value_counts()
            top_missing = str(counts.index[0]) if not counts.empty else ""
        return {
            "label": label,
            "buy_count": int(len(rows)),
            "pm_feature_found_count": int(found.sum()),
            "pm_feature_coverage": float(found.mean()) if len(rows) else None,
            "top_missing_reason": top_missing,
        }

    def _pm_quality(self, sells: pd.DataFrame) -> list[dict[str, Any]]:
        if sells.empty or "pm_multiplier" not in sells.columns:
            return []
        rows = []
        pm = _numeric(sells.get("pm_multiplier")).round(2)
        profit = _numeric(sells.get("net_profit") if "net_profit" in sells.columns else sells.get("profit"))
        for multiplier in [0.60, 0.80, 1.00, 1.15, 1.30]:
            group = sells[pm.eq(multiplier)]
            values = profit.loc[group.index]
            rows.append(
                {
                    "pm_multiplier": multiplier,
                    "trade_count": int(len(group)),
                    "profit": float(values.sum()) if not values.empty else 0.0,
                    "profit_factor": _profit_factor(values),
                    "win_rate": _win_rate(values),
                    "downside": float(values[values < 0].sum()) if not values.empty else 0.0,
                }
            )
        return rows

    def _pm130_comparison(self, profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        baseline = next((row for row in profiles["v2_82_cap38"]["pm_quality"] if row["pm_multiplier"] == 1.30), {})
        baseline_pf = baseline.get("profit_factor")
        baseline_profit = baseline.get("profit")
        rows = []
        for label, payload in profiles.items():
            row = next((item for item in payload["pm_quality"] if item["pm_multiplier"] == 1.30), {})
            meets = True
            if label != "v2_82_cap38":
                meets = (
                    row.get("profit_factor") is not None
                    and baseline_pf is not None
                    and float(row.get("profit_factor")) >= float(baseline_pf)
                    and float(row.get("profit") or 0.0) >= float(baseline_profit or 0.0)
                )
            rows.append({"label": label, "profile": payload["summary"]["profile"], **row, "meets_current_pm130_quality": meets})
        return rows

    def _leakage_guard(self) -> dict[str, Any]:
        path = self.root / PM_V3_FEATURE_COLUMNS
        feature_columns = []
        if path.exists():
            feature_columns = [str(column) for column in json.loads(path.read_text(encoding="utf-8"))]
        forbidden = [column for column in feature_columns if any(token in column.lower() for token in FORBIDDEN_TOKENS)]
        label_like = [column for column in feature_columns if column in LABEL_COLUMNS or column.lower().startswith("future_") or "label" in column.lower() or "target" in column.lower()]
        return {
            "feature_columns": feature_columns,
            "forbidden_feature_count": len(forbidden),
            "forbidden_feature_columns": forbidden,
            "label_columns_in_features": label_like,
            "leakage_risk": "high" if forbidden or label_like else "low",
        }

    def _adoption(self, candidates: list[dict[str, Any]], baseline: dict[str, Any], profiles: dict[str, dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        available = [row for row in candidates if row.get("status") == "ok"]
        if not available:
            return {
                "best_candidate_label": "",
                "strict_conditions_passed": False,
                "beats_v2_82": False,
                "adoption_recommendation": "reject",
                "reason": "candidate backtest logs are missing",
            }
        best = max(available, key=lambda row: (float(row.get("profit_factor") or -10**9), float(row.get("net_profit") or -10**9)))
        pm130_rows = self._pm130_comparison(profiles)
        pm130_best = next((row for row in pm130_rows if row["label"] == best["label"]), {})
        lookup = profiles.get(best["label"], {}).get("pm_lookup_status", {})
        conditions = {
            "leakage_low": leakage["leakage_risk"] == "low",
            "pm_v3_lookup_coverage_positive": float(lookup.get("pm_feature_coverage") or 0.0) > 0.0,
            "pf_at_least_v2_82": float(best.get("profit_factor") or -10**9) >= float(baseline.get("profit_factor") or 10**9),
            "dd_not_worse_than_v2_82": float(best.get("max_drawdown") or -10**9) >= float(baseline.get("max_drawdown") or 0.0),
            "profit_at_least_95pct_v2_82": float(best.get("net_profit") or -10**9) >= 0.95 * float(baseline.get("net_profit") or 10**9),
            "pm130_quality_at_least_current": bool(pm130_best.get("meets_current_pm130_quality")),
        }
        passed = all(conditions.values())
        return {
            "best_candidate_label": best["label"],
            "strict_conditions": conditions,
            "strict_conditions_passed": passed,
            "beats_v2_82": passed,
            "adoption_recommendation": "adopt_candidate_for_next_review" if passed else "reject",
            "reason": "all strict Phase 9-F adoption gates passed" if passed else "one or more strict Phase 9-F adoption gates failed",
        }

    def _monthly_win_rate(self, trades: pd.DataFrame) -> float | None:
        sells = self._sell_trades(trades)
        if sells.empty or "exit_date" not in sells.columns:
            return None
        months = pd.to_datetime(sells["exit_date"], errors="coerce").dt.strftime("%Y-%m")
        monthly = _numeric(sells.get("net_profit")).groupby(months).sum()
        monthly = monthly[monthly.index.notna()]
        return float((monthly > 0).mean()) if not monthly.empty else None

    def _average_capital_utilization(self, daily: pd.DataFrame) -> float | None:
        if daily.empty or not {"positions_value", "total_assets"}.issubset(daily.columns):
            return None
        total = _numeric(daily.get("total_assets")).replace(0, pd.NA)
        utilization = (_numeric(daily.get("positions_value")) / total).dropna()
        return float(utilization.mean()) if not utilization.empty else None

    def _cagr(self, final_assets: Any, initial: Any) -> float | None:
        final = float(final_assets or 0.0)
        start = float(initial or 0.0)
        if final <= 0 or start <= 0:
            return None
        start_date, end_date = self.period.split("_to_")
        years = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25, 1e-9)
        return float((final / start) ** (1 / years) - 1)

    def _profile_table(self, groups: dict[str, list[dict[str, Any]]], columns: list[str]) -> str:
        rows = []
        for label, items in groups.items():
            for item in items:
                rows.append({"label": label, **item})
        return self._table(rows, ["label", *columns])

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
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
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, list):
            return ", ".join(str(item) for item in value[:10])
        return str(value).replace("\n", " ")


def build_phase9f_pm_ai_v3_backtest_candidate_audit(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3BacktestCandidateAudit(root).build_report()
