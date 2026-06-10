"""Phase 10-B score-based PM Rule C threshold/bucket robustness audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_score_based_rule import ALLOWED_SCORE_FEATURES, THRESHOLD_VARIANTS, WEIGHT_VARIANTS, feature_leakage_audit
from ml.portfolio_manager_score_based_rule_audit import _numeric, _profit_factor, _read_csv, _read_json, _win_rate


ROOT = Path(__file__).resolve().parents[2]
PERIOD = "2023-01-01_to_2026-05-31"
REPORT_STEM = "phase10b_score_based_pm_rule_threshold_audit_2023-01_to_2026-05"
REFERENCE_PROFILE = "rookie_dealer_02_v2_82_cap38"
PM_DISABLED_PROFILE = "rookie_dealer_02_v2_95_pm_disabled_equal_weight_cap38"
RULE_C_PROFILE = "rookie_dealer_02_v2_96c_score_based_pm_rule_c"
OPT_PROFILES = {
    "v2_97_opt1_conservative_high": "rookie_dealer_02_v2_97_score_based_pm_rule_c_opt1",
    "v2_97b_opt2_no_060": "rookie_dealer_02_v2_97b_score_based_pm_rule_c_opt2",
    "v2_97c_opt3_no_115": "rookie_dealer_02_v2_97c_score_based_pm_rule_c_opt3",
    "v2_97d_opt4_inverted_low_check": "rookie_dealer_02_v2_97d_score_based_pm_rule_c_opt4",
    "v2_97e_opt5_strength_heavy": "rookie_dealer_02_v2_97e_score_based_pm_rule_c_opt5",
}
PROFILE_LABELS = {
    "v2_82_reference": REFERENCE_PROFILE,
    "v2_95_pm_disabled": PM_DISABLED_PROFILE,
    "v2_96c_rule_c": RULE_C_PROFILE,
    **OPT_PROFILES,
}


@dataclass(frozen=True)
class Phase10BPaths:
    markdown: Path
    json: Path


class ScoreBasedPMThresholdAudit:
    def __init__(self, root: Path | str = ROOT, *, period: str = PERIOD) -> None:
        self.root = Path(root)
        self.period = period

    def build_report(self) -> dict[str, Any]:
        profiles = {label: self._profile_payload(label, profile) for label, profile in PROFILE_LABELS.items()}
        baseline = profiles["v2_95_pm_disabled"]["summary"]
        rule_c = profiles["v2_96c_rule_c"]["summary"]
        opt_summaries = [profiles[label]["summary"] for label in OPT_PROFILES]
        leakage = self._leakage_checklist()
        return {
            "metadata": {
                "phase": "10-B",
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
            "tested_threshold_candidates": THRESHOLD_VARIANTS,
            "tested_weight_candidates": WEIGHT_VARIANTS,
            "backtested_candidates": self._backtested_candidates(),
            "profile_summaries": [payload["summary"] for payload in profiles.values()],
            "rule_c_detail": profiles["v2_96c_rule_c"],
            "comparison_vs_pm_disabled": self._compare_against(opt_summaries, baseline, "v2_95_pm_disabled"),
            "comparison_vs_rule_c": self._compare_against(opt_summaries, rule_c, "v2_96c_rule_c"),
            "comparison_vs_v2_82_reference": self._compare_against(opt_summaries, profiles["v2_82_reference"]["summary"], "v2_82_reference"),
            "pm_distribution_by_profile": {label: payload["pm_distribution"] for label, payload in profiles.items()},
            "pm_quality_by_profile": {label: payload["pm_quality"] for label, payload in profiles.items()},
            "score_distribution_by_profile": {label: [payload["score_distribution"]] for label, payload in profiles.items()},
            "monthly_contribution_by_profile": {label: payload["monthly_contribution"] for label, payload in profiles.items()},
            "yearly_contribution_by_profile": {label: payload["yearly_contribution"] for label, payload in profiles.items()},
            "bucket_monotonicity_by_profile": {label: payload["bucket_monotonicity"] for label, payload in profiles.items()},
            "pm060_analysis": self._pm060_analysis(profiles["v2_96c_rule_c"]),
            "best_candidate": self._best_candidate(opt_summaries, baseline, rule_c, leakage),
            "leakage_checklist": leakage,
            "adoption": self._adoption(opt_summaries, baseline, rule_c, leakage),
        }

    def save_report(self, report: dict[str, Any]) -> Phase10BPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase10BPaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Portfolio Manager Phase 10-B Rule C Threshold / Bucket Robustness Audit",
                "",
                "## Scope",
                "",
                "- research-only threshold/bucket audit for score-based Rule C",
                "- no training, no PM AI model, no PM AI v3 model, no current PM inference",
                "- only four Stock Selection prediction-time scores are allowed",
                "",
                "## Backtested Candidates",
                "",
                self._table(report["backtested_candidates"], ["label", "profile", "threshold_variant", "weight_variant"]),
                "",
                "## Profile Comparison",
                "",
                self._table(report["profile_summaries"], ["label", "profile", "status", "net_profit", "profit_factor", "max_drawdown", "win_rate", "monthly_win_rate", "cagr", "total_trades", "final_assets", "average_capital_utilization"]),
                "",
                "## Comparison vs PM Disabled",
                "",
                self._table(report["comparison_vs_pm_disabled"], ["label", "net_profit_delta", "profit_factor_delta", "max_drawdown_delta", "win_rate_delta", "candidate_gate_passed"]),
                "",
                "## Comparison vs Rule C",
                "",
                self._table(report["comparison_vs_rule_c"], ["label", "net_profit_delta", "profit_factor_delta", "max_drawdown_delta", "win_rate_delta", "improved_item_count"]),
                "",
                "## PM Quality",
                "",
                self._profile_table(report["pm_quality_by_profile"], ["pm_multiplier", "buy_count", "trade_count", "profit", "profit_factor", "win_rate", "downside", "average_holding_days"]),
                "",
                "## Score Distribution",
                "",
                self._profile_table(report["score_distribution_by_profile"], ["score_count", "score_mean", "score_p10", "score_p25", "score_p50", "score_p75", "score_p90"]),
                "",
                "## Bucket Monotonicity",
                "",
                self._table(report["bucket_monotonicity_by_profile"].values(), ["label", "profit_monotonic", "profit_factor_monotonic", "pm130_profit", "pm060_profit", "pm060_quality_problem"]),
                "",
                "## Rule C PM0.60 Analysis",
                "",
                self._table([report["pm060_analysis"]], ["pm060_buy_count", "pm060_trade_count", "pm060_profit", "pm060_profit_factor", "pm060_win_rate", "sample_size_warning", "interpretation"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_checklist"]], ["feature_columns", "forbidden_feature_count", "pm_ai_model_used", "pm_ai_v3_model_used", "current_pm_multiplier_used", "backtest_results_used_as_features", "leakage_risk"]),
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
        pm_quality = self._pm_quality(buys, sells)
        return {
            "summary": self._summary_row(label, profile, summary, trades, daily),
            "pm_distribution": self._pm_distribution(buys),
            "pm_quality": pm_quality,
            "score_distribution": self._score_distribution(buys),
            "monthly_contribution": self._period_contribution(sells, "M"),
            "yearly_contribution": self._period_contribution(sells, "Y"),
            "year2026_contribution": self._year_contribution(sells, 2026),
            "bucket_monotonicity": self._bucket_monotonicity(label, pm_quality),
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
        return [{"pm_multiplier": value, "buy_count": int((rounded == round(value, 2)).sum())} for value in [1.30, 1.15, 1.00, 0.80, 0.60]]

    def _pm_quality(self, buys: pd.DataFrame, sells: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for multiplier in [1.30, 1.15, 1.00, 0.80, 0.60]:
            buy_count = int((_numeric(buys["pm_multiplier"]).round(2) == round(multiplier, 2)).sum()) if not buys.empty and "pm_multiplier" in buys.columns else 0
            subset = sells
            if not sells.empty and "pm_multiplier" in sells.columns:
                subset = sells[_numeric(sells["pm_multiplier"]).round(2) == round(multiplier, 2)]
            profits = subset["net_profit"] if "net_profit" in subset.columns else pd.Series(dtype=float)
            profit_values = _numeric(profits)
            rows.append(
                {
                    "pm_multiplier": multiplier,
                    "buy_count": buy_count,
                    "trade_count": int(len(subset)) if not subset.empty else 0,
                    "profit": float(profit_values.sum()) if not subset.empty else 0.0,
                    "profit_factor": _profit_factor(profits),
                    "win_rate": _win_rate(profits),
                    "downside": float(profit_values[profit_values < 0].mean()) if not profit_values[profit_values < 0].empty else None,
                    "average_holding_days": float(_numeric(subset["holding_days"]).mean()) if not subset.empty and "holding_days" in subset.columns else None,
                }
            )
        return rows

    def _score_distribution(self, buys: pd.DataFrame) -> dict[str, Any]:
        score_column = "pm_rule_score" if "pm_rule_score" in buys.columns else "score_based_pm_score" if "score_based_pm_score" in buys.columns else "pm_score"
        values = _numeric(buys[score_column]) if score_column in buys.columns else pd.Series(dtype=float)
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

    def _period_contribution(self, sells: pd.DataFrame, freq: str) -> list[dict[str, Any]]:
        if sells.empty or "net_profit" not in sells.columns:
            return []
        date_column = "exit_date" if "exit_date" in sells.columns else "date" if "date" in sells.columns else None
        if date_column is None:
            return []
        frame = sells.copy()
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.dropna(subset=[date_column])
        if frame.empty:
            return []
        grouped = _numeric(frame["net_profit"]).groupby(frame[date_column].dt.to_period(freq)).agg(["count", "sum", "mean"])
        return [{"period": str(index), "trade_count": int(row["count"]), "profit": float(row["sum"]), "average_profit": float(row["mean"])} for index, row in grouped.iterrows()]

    def _year_contribution(self, sells: pd.DataFrame, year: int) -> dict[str, Any]:
        rows = self._period_contribution(sells, "Y")
        return next((row for row in rows if str(year) in row["period"]), {"period": str(year), "trade_count": 0, "profit": 0.0, "average_profit": None})

    def _bucket_monotonicity(self, label: str, pm_quality: list[dict[str, Any]]) -> dict[str, Any]:
        by_pm = {float(row["pm_multiplier"]): row for row in pm_quality}
        profits = [float(by_pm[pm].get("profit") or 0.0) for pm in [1.30, 1.15, 1.00, 0.80, 0.60]]
        pfs = [by_pm[pm].get("profit_factor") for pm in [1.30, 1.15, 1.00, 0.80, 0.60]]
        pfs_numeric = [float(value) for value in pfs if value is not None]
        return {
            "label": label,
            "profit_monotonic": all(left >= right for left, right in zip(profits, profits[1:])),
            "profit_factor_monotonic": all(left >= right for left, right in zip(pfs_numeric, pfs_numeric[1:])) if len(pfs_numeric) >= 2 else False,
            "pm130_profit": by_pm[1.30].get("profit"),
            "pm060_profit": by_pm[0.60].get("profit"),
            "pm060_quality_problem": float(by_pm[0.60].get("profit") or 0.0) > float(by_pm[1.30].get("profit") or 0.0) or (by_pm[0.60].get("profit_factor") or 0) > (by_pm[1.30].get("profit_factor") or 0),
        }

    def _pm060_analysis(self, rule_c: dict[str, Any]) -> dict[str, Any]:
        row = next((item for item in rule_c["pm_quality"] if float(item["pm_multiplier"]) == 0.60), {})
        trade_count = int(row.get("trade_count") or 0)
        return {
            "pm060_buy_count": row.get("buy_count"),
            "pm060_trade_count": trade_count,
            "pm060_profit": row.get("profit"),
            "pm060_profit_factor": row.get("profit_factor"),
            "pm060_win_rate": row.get("win_rate"),
            "sample_size_warning": trade_count < 30,
            "interpretation": "PM0.60 looks good but sample size is small; do not treat the low bucket as reliably bad without threshold redesign.",
        }

    def _compare_against(self, rows: list[dict[str, Any]], baseline: dict[str, Any], baseline_label: str) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            deltas = {
                "net_profit_delta": self._delta(row.get("net_profit"), baseline.get("net_profit")),
                "profit_factor_delta": self._delta(row.get("profit_factor"), baseline.get("profit_factor")),
                "max_drawdown_delta": self._delta(row.get("max_drawdown"), baseline.get("max_drawdown")),
                "win_rate_delta": self._delta(row.get("win_rate"), baseline.get("win_rate")),
            }
            improved = [
                deltas["net_profit_delta"] is not None and deltas["net_profit_delta"] > 0,
                deltas["profit_factor_delta"] is not None and deltas["profit_factor_delta"] > 0,
                deltas["max_drawdown_delta"] is not None and deltas["max_drawdown_delta"] >= 0,
                deltas["win_rate_delta"] is not None and deltas["win_rate_delta"] > 0,
            ]
            out.append(
                {
                    "label": row.get("label"),
                    "baseline_label": baseline_label,
                    **deltas,
                    "candidate_gate_passed": bool(improved[0] and improved[1] and improved[2]),
                    "improved_item_count": int(sum(improved)),
                }
            )
        return out

    def _best_candidate(self, rows: list[dict[str, Any]], baseline: dict[str, Any], rule_c: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        comparisons = self._compare_against(rows, baseline, "v2_95_pm_disabled")
        rule_c_comparisons = self._compare_against(rows, rule_c, "v2_96c_rule_c")
        candidates = []
        for row in rows:
            base_cmp = next((item for item in comparisons if item["label"] == row["label"]), {})
            rule_cmp = next((item for item in rule_c_comparisons if item["label"] == row["label"]), {})
            score = float(row.get("net_profit") or -10**18)
            candidates.append((bool(base_cmp.get("candidate_gate_passed")), int(rule_cmp.get("improved_item_count") or 0), score, row, base_cmp, rule_cmp))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        if not candidates:
            return {}
        _base_pass, _rule_items, _score, row, base_cmp, rule_cmp = candidates[0]
        return {"summary": row, "vs_pm_disabled": base_cmp, "vs_rule_c": rule_cmp, "leakage_risk": leakage.get("leakage_risk")}

    def _adoption(self, rows: list[dict[str, Any]], baseline: dict[str, Any], rule_c: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        best = self._best_candidate(rows, baseline, rule_c, leakage)
        summary = best.get("summary", {})
        base_cmp = best.get("vs_pm_disabled", {})
        rule_cmp = best.get("vs_rule_c", {})
        strong = bool(
            float(summary.get("profit_factor") or -10**18) >= 1.8
            and float(summary.get("max_drawdown") or -10**18) >= -0.12
            and float(summary.get("net_profit") or -10**18) >= 1_500_000
            and float(summary.get("win_rate") or -10**18) >= 0.48
            and float(summary.get("monthly_win_rate") or -10**18) >= 0.60
        )
        candidate = bool(base_cmp.get("candidate_gate_passed") and int(rule_cmp.get("improved_item_count") or 0) >= 2 and leakage.get("leakage_risk") == "low")
        return {
            "best_profile_label": summary.get("label"),
            "candidate_gate_passed": candidate,
            "strong_gate_passed": strong,
            "adoption_recommendation": "strong_candidate" if strong else "candidate_for_review" if candidate else "do_not_adopt",
            "next_phase_recommendation": "Phase 10-C focused validation" if candidate else "Keep v2_96c as reference candidate and redesign buckets",
        }

    def _leakage_checklist(self) -> dict[str, Any]:
        audit = feature_leakage_audit(sorted(ALLOWED_SCORE_FEATURES))
        return {
            **audit,
            "pm_ai_model_used": False,
            "pm_ai_v3_model_used": False,
            "current_pm_multiplier_used": False,
            "backtest_results_used_as_features": False,
            "cash_or_portfolio_used_for_score": False,
            "leakage_risk": audit["leakage_risk"],
        }

    def _backtested_candidates(self) -> list[dict[str, str]]:
        return [
            {"label": "v2_97_opt1_conservative_high", "profile": OPT_PROFILES["v2_97_opt1_conservative_high"], "threshold_variant": "conservative_high", "weight_variant": "original"},
            {"label": "v2_97b_opt2_no_060", "profile": OPT_PROFILES["v2_97b_opt2_no_060"], "threshold_variant": "no_060", "weight_variant": "original"},
            {"label": "v2_97c_opt3_no_115", "profile": OPT_PROFILES["v2_97c_opt3_no_115"], "threshold_variant": "no_115", "weight_variant": "original"},
            {"label": "v2_97d_opt4_inverted_low_check", "profile": OPT_PROFILES["v2_97d_opt4_inverted_low_check"], "threshold_variant": "inverted_low_check", "weight_variant": "original"},
            {"label": "v2_97e_opt5_strength_heavy", "profile": OPT_PROFILES["v2_97e_opt5_strength_heavy"], "threshold_variant": "original", "weight_variant": "strength_heavy"},
        ]

    def _monthly_win_rate(self, trades: pd.DataFrame) -> float | None:
        monthly = self._period_contribution(self._sell_trades(trades), "M")
        return float(sum(1 for row in monthly if row["profit"] > 0) / len(monthly)) if monthly else None

    def _cagr(self, final_assets: Any, initial: Any) -> float | None:
        try:
            final = float(final_assets)
            start = float(initial)
        except (TypeError, ValueError):
            return None
        if final <= 0 or start <= 0:
            return None
        return float((final / start) ** (1.0 / 3.414) - 1.0)

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

    def _table(self, rows: Any, columns: list[str]) -> str:
        if isinstance(rows, dict):
            rows = [rows]
        rows = list(rows or [])
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(self._format_cell(row.get(column, "")) for column in columns) + " |" for row in rows]
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
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value).replace("\n", " ")


def build_and_save_report(root: Path | str = ROOT) -> Phase10BPaths:
    audit = ScoreBasedPMThresholdAudit(root)
    report = audit.build_report()
    return audit.save_report(report)
