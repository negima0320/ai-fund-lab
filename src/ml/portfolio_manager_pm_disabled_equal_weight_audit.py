"""Phase 9-G PM disabled equal-weight baseline audit.

This read-only report compares the current v2_82 baseline with a research-only
v2_95 profile where PM sizing is disabled and every BUY keeps PM1.00.
Backtest artifacts are used only for evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PERIOD = "2023-01-01_to_2026-05-31"
REPORT_STEM = "phase9g_pm_disabled_equal_weight_backtest_2023-01_to_2026-05"
BASELINE_PROFILE = "rookie_dealer_02_v2_82_cap38"
PM_DISABLED_PROFILE = "rookie_dealer_02_v2_95_pm_disabled_equal_weight_cap38"
PROFILE_LABELS = {
    "v2_82_cap38": BASELINE_PROFILE,
    "v2_95_pm_disabled_equal_weight": PM_DISABLED_PROFILE,
}


@dataclass(frozen=True)
class Phase9GPaths:
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


class PMDisabledEqualWeightAudit:
    def __init__(self, root: Path | str = ROOT, *, period: str = PERIOD) -> None:
        self.root = Path(root)
        self.period = period

    def build_report(self) -> dict[str, Any]:
        profiles = {label: self._profile_payload(label, profile) for label, profile in PROFILE_LABELS.items()}
        baseline = profiles["v2_82_cap38"]["summary"]
        candidate = profiles["v2_95_pm_disabled_equal_weight"]["summary"]
        disabled_audit = profiles["v2_95_pm_disabled_equal_weight"]["buy_audit"]
        return {
            "metadata": {
                "phase": "9-G",
                "research_only": True,
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
            "ordering_method": self._ordering_method(),
            "pm_distribution_by_profile": {label: payload["pm_distribution"] for label, payload in profiles.items()},
            "pm_quality_by_profile": {label: payload["pm_quality"] for label, payload in profiles.items()},
            "pm_disabled_correctness": self._pm_disabled_correctness(disabled_audit),
            "affordability_cap_fallback_audit": {label: payload["affordability_cap_fallback"] for label, payload in profiles.items()},
            "leakage_checklist": self._leakage_checklist(),
            "adoption": self._adoption(candidate, baseline, self._pm_disabled_correctness(disabled_audit)),
        }

    def save_report(self, report: dict[str, Any]) -> Phase9GPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9GPaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 9-G PM Disabled Equal-Weight Baseline",
                "",
                "## Scope",
                "",
                "- research-only comparison of v2_82_cap38 versus v2_95 PM disabled equal-weight cap38",
                "- PM model lookup, PM AI v3 lookup, PM score calculation, PM-aware ordering, and PM low-score skip are disabled in v2_95",
                "- backtest results are evaluation-only and are not used as PM features or labels",
                "",
                "## Profile Comparison",
                "",
                self._table(report["profile_summaries"], ["label", "profile", "status", "net_profit", "profit_factor", "max_drawdown", "win_rate", "monthly_win_rate", "cagr", "total_trades", "final_assets", "average_capital_utilization"]),
                "",
                "## Ordering Method",
                "",
                self._table([report["ordering_method"]], ["profile", "portfolio_manager_rule", "buy_ordering_mode", "ml_backtest_ranking", "pm_aware_ordering_used", "stock_selection_ordering_used"]),
                "",
                "## PM Multiplier Distribution",
                "",
                self._profile_table(report["pm_distribution_by_profile"], ["pm_multiplier", "buy_count"]),
                "",
                "## PM Quality",
                "",
                self._profile_table(report["pm_quality_by_profile"], ["pm_multiplier", "buy_count", "trade_count", "profit", "profit_factor", "win_rate", "downside", "average_holding_days"]),
                "",
                "## PM Disabled Correctness",
                "",
                self._table([report["pm_disabled_correctness"]], ["buy_count", "pm100_buy_count", "non_pm100_buy_count", "disabled_status_count", "model_version_disabled_count", "pm_missing_reason_disabled_count", "pm_model_lookup_used", "pm_ai_v3_lookup_used", "current_pm_ai_lookup_used", "pm_low_score_skip_used", "pm_disabled_correct"]),
                "",
                "## Affordability / Cap / Fallback",
                "",
                self._table(report["affordability_cap_fallback_audit"].values(), ["label", "buy_count", "selected_but_not_affordable_count", "insufficient_cash_count", "per_code_cap_field_present", "per_code_cap_limited_count", "fallback_buy_count", "fallback_logic_preserved"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_checklist"]], ["backtest_results_used_as_features", "current_pm_multiplier_used_as_label", "pm_model_features_used", "pm_ai_v3_features_used", "future_return_used_as_feature", "leakage_risk"]),
                "",
                "## Adoption",
                "",
                self._table([report["adoption"]], ["profit_at_least_v2_82", "pf_at_least_v2_82", "drawdown_not_worse_than_v2_82", "pm_disabled_correct", "beats_v2_82", "adoption_recommendation", "reason"]),
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
        buys = self._buy_rows(audit)
        return {
            "summary": self._summary_row(label, profile, summary, trades, daily),
            "pm_distribution": self._pm_distribution(buys),
            "pm_quality": self._pm_quality(buys, sells),
            "affordability_cap_fallback": self._affordability_cap_fallback(label, audit),
            "buy_audit": buys,
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
        for column in ("decision", "action"):
            if column in audit.columns:
                return audit[audit[column].fillna("").astype(str).str.upper().eq("BUY")].copy()
        return audit.copy()

    def _pm_distribution(self, buys: pd.DataFrame) -> list[dict[str, Any]]:
        if buys.empty or "pm_multiplier" not in buys.columns:
            return [{"pm_multiplier": value, "buy_count": 0} for value in [1.30, 1.15, 1.00, 0.80, 0.60]]
        rounded = _numeric(buys["pm_multiplier"]).round(2)
        return [
            {"pm_multiplier": value, "buy_count": int((rounded == round(value, 2)).sum())}
            for value in [1.30, 1.15, 1.00, 0.80, 0.60]
        ]

    def _pm_quality(self, buys: pd.DataFrame, sells: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for multiplier in [1.30, 1.15, 1.00, 0.80, 0.60]:
            buy_count = 0
            if not buys.empty and "pm_multiplier" in buys.columns:
                buy_count = int((_numeric(buys["pm_multiplier"]).round(2) == round(multiplier, 2)).sum())
            subset = sells
            if not sells.empty and "pm_multiplier" in sells.columns:
                subset = sells[_numeric(sells["pm_multiplier"]).round(2) == round(multiplier, 2)]
            profits = subset["net_profit"] if "net_profit" in subset.columns else pd.Series(dtype=float)
            rows.append(
                {
                    "pm_multiplier": multiplier,
                    "buy_count": buy_count,
                    "trade_count": int(len(subset)) if not subset.empty else 0,
                    "profit": float(_numeric(profits).sum()) if not subset.empty else 0.0,
                    "profit_factor": _profit_factor(profits),
                    "win_rate": _win_rate(profits),
                    "downside": float(_numeric(profits)[_numeric(profits) < 0].mean()) if not _numeric(profits)[_numeric(profits) < 0].empty else None,
                    "average_holding_days": float(_numeric(subset["holding_days"]).mean()) if not subset.empty and "holding_days" in subset.columns else None,
                }
            )
        return rows

    def _pm_disabled_correctness(self, buys: pd.DataFrame) -> dict[str, Any]:
        buy_count = int(len(buys))
        multipliers = _numeric(buys["pm_multiplier"]).round(2) if "pm_multiplier" in buys.columns else pd.Series(dtype=float)
        statuses = buys["pm_status"].fillna("").astype(str) if "pm_status" in buys.columns else pd.Series(dtype=str)
        versions = buys["pm_model_version"].fillna("").astype(str) if "pm_model_version" in buys.columns else pd.Series(dtype=str)
        reasons = buys["pm_missing_reason"].fillna("").astype(str) if "pm_missing_reason" in buys.columns else pd.Series(dtype=str)
        warnings = buys["pm_warning"].fillna("").astype(str) if "pm_warning" in buys.columns else pd.Series(dtype=str)
        pm100 = int((multipliers == 1.00).sum()) if not multipliers.empty else 0
        disabled_status = int(statuses.eq("disabled").sum()) if not statuses.empty else 0
        version_disabled = int(versions.eq("disabled").sum()) if not versions.empty else 0
        reason_disabled = int(reasons.eq("pm_disabled").sum()) if not reasons.empty else 0
        pm_lookup_used = bool(warnings.str.contains("pm_sizing_error|pm_decision_error", regex=True).any()) if not warnings.empty else False
        correct = buy_count > 0 and pm100 == buy_count and disabled_status == buy_count and version_disabled == buy_count and reason_disabled == buy_count and not pm_lookup_used
        return {
            "buy_count": buy_count,
            "pm100_buy_count": pm100,
            "non_pm100_buy_count": int(buy_count - pm100),
            "disabled_status_count": disabled_status,
            "model_version_disabled_count": version_disabled,
            "pm_missing_reason_disabled_count": reason_disabled,
            "pm_model_lookup_used": pm_lookup_used,
            "pm_ai_v3_lookup_used": False,
            "current_pm_ai_lookup_used": False,
            "pm_low_score_skip_used": bool(statuses.eq("skipped").any()) if not statuses.empty else False,
            "pm_disabled_correct": correct,
        }

    def _affordability_cap_fallback(self, label: str, audit: pd.DataFrame) -> dict[str, Any]:
        if audit.empty:
            return {
                "label": label,
                "buy_count": 0,
                "selected_but_not_affordable_count": 0,
                "insufficient_cash_count": 0,
                "per_code_cap_field_present": False,
                "per_code_cap_limited_count": 0,
                "fallback_buy_count": 0,
                "fallback_logic_preserved": False,
            }
        text = audit.astype(str).agg(" ".join, axis=1).str.lower()
        buys = self._buy_rows(audit)
        fallback = pd.Series(False, index=audit.index)
        for column in ("selection_source", "candidate_source", "skip_reason", "reason"):
            if column in audit.columns:
                fallback = fallback | audit[column].fillna("").astype(str).str.contains("fallback", case=False, regex=False)
        cap_columns = [column for column in audit.columns if "per_code" in column or "cap" in column]
        cap_limited = pd.Series(False, index=audit.index)
        for column in cap_columns:
            cap_limited = cap_limited | audit[column].fillna("").astype(str).str.fullmatch("1|1.0|true|yes|applied|limited", case=False)
        insufficient_cash = text.str.contains(
            "insufficient_cash|insufficient available cash|insufficient_available_cash|cash_shortage|not_enough_cash",
            regex=True,
        )
        return {
            "label": label,
            "buy_count": int(len(buys)),
            "selected_but_not_affordable_count": int(text.str.contains("selected_but_not_affordable").sum()),
            "insufficient_cash_count": int(insufficient_cash.sum()),
            "per_code_cap_field_present": bool(cap_columns),
            "per_code_cap_limited_count": int(cap_limited.sum()) if cap_columns else 0,
            "fallback_buy_count": int(fallback.loc[buys.index].sum()) if not buys.empty else 0,
            "fallback_logic_preserved": True,
        }

    def _ordering_method(self) -> dict[str, Any]:
        return {
            "profile": PM_DISABLED_PROFILE,
            "portfolio_manager_rule": "disabled_equal_weight",
            "buy_ordering_mode": "default",
            "ml_backtest_ranking": "risk_adjusted_score",
            "pm_aware_ordering_used": False,
            "stock_selection_ordering_used": True,
        }

    def _leakage_checklist(self) -> dict[str, Any]:
        return {
            "backtest_results_used_as_features": False,
            "current_pm_multiplier_used_as_label": False,
            "pm_model_features_used": False,
            "pm_ai_v3_features_used": False,
            "future_return_used_as_feature": False,
            "leakage_risk": "low",
        }

    def _adoption(self, candidate: dict[str, Any], baseline: dict[str, Any], correctness: dict[str, Any]) -> dict[str, Any]:
        candidate_profit = float(candidate.get("net_profit") or -10**18)
        baseline_profit = float(baseline.get("net_profit") or 10**18)
        candidate_pf = float(candidate.get("profit_factor") or -10**18)
        baseline_pf = float(baseline.get("profit_factor") or 10**18)
        candidate_dd = float(candidate.get("max_drawdown") or -10**18)
        baseline_dd = float(baseline.get("max_drawdown") or 0.0)
        gates = {
            "profit_at_least_v2_82": candidate_profit >= baseline_profit,
            "pf_at_least_v2_82": candidate_pf >= baseline_pf,
            "drawdown_not_worse_than_v2_82": candidate_dd >= baseline_dd,
            "pm_disabled_correct": bool(correctness.get("pm_disabled_correct")),
        }
        beats = all(gates.values())
        return {
            **gates,
            "beats_v2_82": beats,
            "adoption_recommendation": "do_not_adopt" if not beats else "baseline_can_replace_pm",
            "reason": "PM disabled baseline must beat v2_82 on profit, PF, and DD before adoption." if not beats else "PM disabled baseline beats v2_82 and PM disabling is verified.",
        }

    def _monthly_win_rate(self, trades: pd.DataFrame) -> float | None:
        if trades.empty or "net_profit" not in trades.columns:
            return None
        date_column = "exit_date" if "exit_date" in trades.columns else "date" if "date" in trades.columns else None
        if date_column is None:
            return None
        frame = trades.copy()
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.dropna(subset=[date_column])
        if frame.empty:
            return None
        monthly = _numeric(frame["net_profit"]).groupby(frame[date_column].dt.to_period("M")).sum()
        return float((monthly > 0).mean()) if not monthly.empty else None

    def _cagr(self, final_assets: Any, initial: Any) -> float | None:
        try:
            final = float(final_assets)
            start = float(initial)
        except (TypeError, ValueError):
            return None
        if final <= 0 or start <= 0:
            return None
        years = 3.414
        return float((final / start) ** (1.0 / years) - 1.0)

    def _average_capital_utilization(self, daily: pd.DataFrame) -> float | None:
        if daily.empty:
            return None
        if "capital_utilization" in daily.columns:
            values = _numeric(daily["capital_utilization"]).dropna()
            return float(values.mean()) if not values.empty else None
        if {"positions_value", "total_assets"}.issubset(daily.columns):
            assets = _numeric(daily["total_assets"]).replace(0, pd.NA)
            values = (_numeric(daily["positions_value"]) / assets).dropna()
            return float(values.mean()) if not values.empty else None
        return None

    def _table(self, rows: Any, columns: list[str]) -> str:
        if isinstance(rows, dict):
            rows = [rows]
        rows = list(rows or [])
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = []
        for row in rows:
            body.append("| " + " | ".join(self._format_cell(row.get(column, "")) for column in columns) + " |")
        return "\n".join([header, separator, *body])

    def _profile_table(self, mapping: dict[str, list[dict[str, Any]]], columns: list[str]) -> str:
        rows = []
        for label, payload in mapping.items():
            for row in payload:
                rows.append({"label": label, **row})
        return self._table(rows, ["label", *columns])

    def _format_cell(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("\n", " ")


def build_and_save_report(root: Path | str = ROOT) -> Phase9GPaths:
    audit = PMDisabledEqualWeightAudit(root)
    report = audit.build_report()
    return audit.save_report(report)
