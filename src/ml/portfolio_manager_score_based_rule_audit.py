"""Phase 10-A score-based PM rule backtest audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_score_based_rule import RULE_DEFINITIONS, feature_leakage_audit


ROOT = Path(__file__).resolve().parents[2]
PERIOD = "2023-01-01_to_2026-05-31"
REPORT_STEM = "phase10a_score_based_pm_rule_backtest_2023-01_to_2026-05"
REFERENCE_PROFILE = "rookie_dealer_02_v2_82_cap38"
PM_DISABLED_PROFILE = "rookie_dealer_02_v2_95_pm_disabled_equal_weight_cap38"
RULE_PROFILES = {
    "v2_96_rule_a": "rookie_dealer_02_v2_96_score_based_pm_rule_a",
    "v2_96b_rule_b": "rookie_dealer_02_v2_96b_score_based_pm_rule_b",
    "v2_96c_rule_c": "rookie_dealer_02_v2_96c_score_based_pm_rule_c",
}
PROFILE_LABELS = {"v2_82_reference": REFERENCE_PROFILE, "v2_95_pm_disabled": PM_DISABLED_PROFILE, **RULE_PROFILES}
CLEANUP_MANIFEST = Path("reports/ml/phase10a_phase9_cleanup_manifest.json")


@dataclass(frozen=True)
class Phase10APaths:
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


class ScoreBasedPMRuleAudit:
    def __init__(self, root: Path | str = ROOT, *, period: str = PERIOD) -> None:
        self.root = Path(root)
        self.period = period

    def build_report(self) -> dict[str, Any]:
        profiles = {label: self._profile_payload(label, profile) for label, profile in PROFILE_LABELS.items()}
        baseline = profiles["v2_95_pm_disabled"]["summary"]
        reference = profiles["v2_82_reference"]["summary"]
        rule_summaries = [profiles[label]["summary"] for label in RULE_PROFILES]
        leakage = self._leakage_checklist()
        return {
            "metadata": {
                "phase": "10-A",
                "research_only": True,
                "training_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_model_regenerated_historical_predictions": False,
                "pm_ai_v3_retrained": False,
                "pm_ai_v3_inference_used": False,
                "current_pm_ai_inference_used": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
            },
            "period": self.period,
            "profiles": PROFILE_LABELS,
            "rule_definitions": RULE_DEFINITIONS,
            "profile_summaries": [payload["summary"] for payload in profiles.values()],
            "comparison_vs_pm_disabled": self._compare_against(rule_summaries, baseline, "v2_95_pm_disabled"),
            "comparison_vs_v2_82_reference": self._compare_against(rule_summaries, reference, "v2_82_reference"),
            "pm_distribution_by_profile": {label: payload["pm_distribution"] for label, payload in profiles.items()},
            "pm_quality_by_profile": {label: payload["pm_quality"] for label, payload in profiles.items()},
            "score_distribution_by_profile": {label: payload["score_distribution"] for label, payload in profiles.items()},
            "leakage_checklist": leakage,
            "adoption": self._adoption(rule_summaries, baseline, leakage),
            "phase9_cleanup": self._cleanup_manifest(),
        }

    def save_report(self, report: dict[str, Any]) -> Phase10APaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase10APaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Portfolio Manager Phase 10-A Score-Based PM Rule Backtest",
                "",
                "## Scope",
                "",
                "- research-only score-based PM rules using Stock Selection prediction-time scores",
                "- no PM AI model, no PM AI v3 model, no retraining, no current PM inference",
                "- v2_82 is reference only; main comparison is against v2_95 PM-disabled equal-weight baseline",
                "",
                "## Rule Definitions",
                "",
                self._rule_table(),
                "",
                "## Profile Comparison",
                "",
                self._table(report["profile_summaries"], ["label", "profile", "status", "net_profit", "profit_factor", "max_drawdown", "win_rate", "monthly_win_rate", "cagr", "total_trades", "final_assets", "average_capital_utilization"]),
                "",
                "## Comparison vs PM Disabled",
                "",
                self._table(report["comparison_vs_pm_disabled"], ["label", "net_profit_delta", "profit_factor_delta", "max_drawdown_delta", "win_rate_delta", "improves_net_profit", "improves_profit_factor", "drawdown_not_worse", "candidate_gate_passed"]),
                "",
                "## Reference Gap vs v2_82",
                "",
                self._table(report["comparison_vs_v2_82_reference"], ["label", "net_profit_delta", "profit_factor_delta", "max_drawdown_delta", "win_rate_delta"]),
                "",
                "## PM Multiplier Distribution",
                "",
                self._profile_table(report["pm_distribution_by_profile"], ["pm_multiplier", "buy_count"]),
                "",
                "## PM Multiplier Quality",
                "",
                self._profile_table(report["pm_quality_by_profile"], ["pm_multiplier", "buy_count", "trade_count", "profit", "profit_factor", "win_rate", "downside", "average_holding_days"]),
                "",
                "## Score Distribution",
                "",
                self._profile_table(report["score_distribution_by_profile"], ["score_count", "score_mean", "score_p10", "score_p25", "score_p50", "score_p75", "score_p90"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_checklist"]], ["feature_columns", "forbidden_feature_count", "forbidden_feature_columns", "pm_ai_model_used", "pm_ai_v3_model_used", "current_pm_multiplier_used", "backtest_results_used_as_features", "leakage_risk"]),
                "",
                "## Phase 9 Cleanup",
                "",
                self._table([report["phase9_cleanup"]], ["dry_run_reported", "deleted", "deleted_path_count", "bytes_before", "bytes_after", "bytes_saved"]),
                "",
                "## Adoption",
                "",
                self._table([report["adoption"]], ["best_profile_label", "candidate_gate_passed", "strong_gate_passed", "adoption_recommendation", "next_phase_recommendation"]),
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
            "score_distribution": [self._score_distribution(buys)],
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
            losses = _numeric(profits)
            rows.append(
                {
                    "pm_multiplier": multiplier,
                    "buy_count": buy_count,
                    "trade_count": int(len(subset)) if not subset.empty else 0,
                    "profit": float(losses.sum()) if not subset.empty else 0.0,
                    "profit_factor": _profit_factor(profits),
                    "win_rate": _win_rate(profits),
                    "downside": float(losses[losses < 0].mean()) if not losses[losses < 0].empty else None,
                    "average_holding_days": float(_numeric(subset["holding_days"]).mean()) if not subset.empty and "holding_days" in subset.columns else None,
                }
            )
        return rows

    def _score_distribution(self, buys: pd.DataFrame) -> dict[str, Any]:
        values = _numeric(buys["score_based_pm_score"]) if "score_based_pm_score" in buys.columns else pd.Series(dtype=float)
        values = values.dropna()
        if values.empty:
            return {"score_count": 0, "score_mean": None, "score_p10": None, "score_p25": None, "score_p50": None, "score_p75": None, "score_p90": None}
        return {
            "score_count": int(len(values)),
            "score_mean": float(values.mean()),
            "score_p10": float(values.quantile(0.10)),
            "score_p25": float(values.quantile(0.25)),
            "score_p50": float(values.quantile(0.50)),
            "score_p75": float(values.quantile(0.75)),
            "score_p90": float(values.quantile(0.90)),
        }

    def _compare_against(self, rows: list[dict[str, Any]], baseline: dict[str, Any], baseline_label: str) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            net_delta = self._delta(row.get("net_profit"), baseline.get("net_profit"))
            pf_delta = self._delta(row.get("profit_factor"), baseline.get("profit_factor"))
            dd_delta = self._delta(row.get("max_drawdown"), baseline.get("max_drawdown"))
            win_delta = self._delta(row.get("win_rate"), baseline.get("win_rate"))
            out.append(
                {
                    "label": row.get("label"),
                    "baseline_label": baseline_label,
                    "net_profit_delta": net_delta,
                    "profit_factor_delta": pf_delta,
                    "max_drawdown_delta": dd_delta,
                    "win_rate_delta": win_delta,
                    "improves_net_profit": net_delta is not None and net_delta > 0,
                    "improves_profit_factor": pf_delta is not None and pf_delta > 0,
                    "drawdown_not_worse": dd_delta is not None and dd_delta >= 0,
                    "candidate_gate_passed": bool(net_delta is not None and net_delta > 0 and pf_delta is not None and pf_delta > 0 and dd_delta is not None and dd_delta >= 0),
                }
            )
        return out

    def _leakage_checklist(self) -> dict[str, Any]:
        features = ["risk_adjusted_score", "expected_return", "stock_selection_rank_score", "candidate_strength"]
        audit = feature_leakage_audit(features)
        return {
            **audit,
            "pm_ai_model_used": False,
            "pm_ai_v3_model_used": False,
            "current_pm_multiplier_used": False,
            "backtest_results_used_as_features": False,
            "cash_or_portfolio_used_for_score": False,
            "leakage_risk": audit["leakage_risk"],
        }

    def _adoption(self, rows: list[dict[str, Any]], baseline: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        comparisons = self._compare_against(rows, baseline, "v2_95_pm_disabled")
        best = max(rows, key=lambda row: float(row.get("net_profit") or -10**18), default={})
        best_cmp = next((row for row in comparisons if row.get("label") == best.get("label")), {})
        strong = bool(
            float(best.get("profit_factor") or -10**18) >= 2.0
            and float(best.get("max_drawdown") or -10**18) >= -0.10
            and float(best.get("net_profit") or -10**18) >= 2_500_000
            and float(best.get("win_rate") or -10**18) >= 0.50
            and float(best.get("monthly_win_rate") or -10**18) >= 0.70
        )
        candidate = bool(best_cmp.get("candidate_gate_passed") and leakage.get("leakage_risk") == "low")
        return {
            "best_profile_label": best.get("label"),
            "candidate_gate_passed": candidate,
            "strong_gate_passed": strong,
            "adoption_recommendation": "strong_candidate" if strong else "candidate_for_review" if candidate else "do_not_adopt",
            "next_phase_recommendation": "Phase 10-B detailed robustness audit" if candidate else "Reject score-based PM rules or redesign thresholds",
        }

    def _cleanup_manifest(self) -> dict[str, Any]:
        manifest = _read_json(self.root / CLEANUP_MANIFEST)
        return manifest or {
            "dry_run_reported": False,
            "deleted": False,
            "deleted_path_count": 0,
            "bytes_before": 0,
            "bytes_after": 0,
            "bytes_saved": 0,
            "paths": [],
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

    def _delta(self, value: Any, baseline: Any) -> float | None:
        try:
            return float(value) - float(baseline)
        except (TypeError, ValueError):
            return None

    def _rule_table(self) -> str:
        rows = []
        for rule, payload in RULE_DEFINITIONS.items():
            rows.append({"rule": rule, "name": payload.get("name"), "features": ",".join(payload.get("features", [])), "description": payload.get("description")})
        return self._table(rows, ["rule", "name", "features", "description"])

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
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        return str(value).replace("\n", " ")


def build_and_save_report(root: Path | str = ROOT) -> Phase10APaths:
    audit = ScoreBasedPMRuleAudit(root)
    report = audit.build_report()
    return audit.save_report(report)
