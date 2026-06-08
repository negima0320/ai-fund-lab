"""Phase 8-D PM AI v2 calibration redesign audit.

This read-only audit tests alternative mappings from PM AI v2 probabilities to
PM multipliers. It does not add profiles, run backtests, or overwrite models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase8b_pm_candidate_integration_audit import (
    CANDIDATE_DATASET,
    CANDIDATE_PM_DIR,
    PERIOD,
    Phase8BPMCandidateIntegrationAudit,
    _profit_factor,
    _profit_series,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase8d_pm_ai_v2_calibration_audit_2023-01_to_2026-05"
V282_FINAL = Path("reports/final/v2_82_cap38/core_2023-01_to_2026-05")
V290_LOG = Path("logs/backtests/rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38/2023-01-01_to_2026-05-31")
PHASE8C_JSON = Path("reports/ml/phase8c_pm_ai_v2_backtest_2023-01_to_2026-05.json")
PM_BUCKETS = [1.30, 1.15, 1.00, 0.80, 0.60]


@dataclass(frozen=True)
class Phase8DReportPaths:
    markdown: Path
    json: Path


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


def _quantiles(series: pd.Series | None) -> dict[str, float | None]:
    values = _numeric(series).dropna()
    if values.empty:
        return {key: None for key in ["min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    qs = {
        "min": values.min(),
        "p01": values.quantile(0.01),
        "p05": values.quantile(0.05),
        "p10": values.quantile(0.10),
        "p25": values.quantile(0.25),
        "p50": values.quantile(0.50),
        "p75": values.quantile(0.75),
        "p90": values.quantile(0.90),
        "p95": values.quantile(0.95),
        "p99": values.quantile(0.99),
        "max": values.max(),
    }
    return {key: float(value) for key, value in qs.items()}


def _win_rate(profits: pd.Series | None) -> float | None:
    values = _numeric(profits).dropna()
    if values.empty:
        return None
    return float((values > 0).mean())


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def assign_calibration(frame: pd.DataFrame, rule: str) -> pd.Series:
    """Assign PM multipliers using one of the Phase 8-D virtual rules."""

    high = _numeric(frame.get("candidate_high_conviction_proba"))
    avoid = _numeric(frame.get("candidate_avoid_proba"))
    score = _numeric(frame.get("candidate_pm_score"))
    out = pd.Series(1.00, index=frame.index, dtype=float)

    if frame.empty:
        return out

    if rule == "Rule A":
        high_rank = high.rank(pct=True, method="first")
        out.loc[high_rank >= 0.90] = 1.30
        out.loc[(high_rank >= 0.80) & (high_rank < 0.90)] = 1.15
        out.loc[avoid.rank(pct=True, method="first") >= 0.80] = 0.80
    elif rule == "Rule B":
        high_rank = high.rank(pct=True, method="first")
        out.loc[high_rank >= 0.85] = 1.30
        out.loc[(high_rank >= 0.70) & (high_rank < 0.85)] = 1.15
        out.loc[avoid.rank(pct=True, method="first") >= 0.80] = 0.80
    elif rule == "Rule C":
        rank = score.rank(pct=True, method="first")
        out.loc[rank >= 0.90] = 1.30
        out.loc[(rank >= 0.75) & (rank < 0.90)] = 1.15
        out.loc[rank <= 0.25] = 0.80
    elif rule == "Rule D":
        high_rank = high.rank(pct=True, method="first")
        avoid_rank = avoid.rank(pct=True, method="first")
        out.loc[(high_rank >= 0.85) & (avoid_rank <= 0.60)] = 1.30
        out.loc[(high_rank >= 0.70) & out.eq(1.00)] = 1.15
        out.loc[avoid_rank >= 0.80] = 0.80
    elif rule == "Rule E":
        rank = score.rank(pct=True, method="first")
        out.loc[rank >= 0.757] = 1.30
        out.loc[(rank >= 0.657) & (rank < 0.757)] = 1.15
        out.loc[rank <= 0.327] = 0.80
    elif rule == "Rule F":
        rank = score.rank(pct=True, method="first")
        out.loc[rank >= 0.95] = 1.30
        out.loc[(rank >= 0.85) & (rank < 0.95)] = 1.15
        out.loc[rank <= 0.30] = 0.80
    else:
        raise ValueError(f"Unknown calibration rule: {rule}")
    return out


class Phase8DPMCalibrationAudit:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)

    def build_report(self) -> dict[str, Any]:
        scored_v282 = self._scored_v282_trades()
        v290_trades = self._load_v290_trades()
        phase8c = _read_json(self.root / PHASE8C_JSON)
        calibration_rows = self._calibration_rows(scored_v282)
        recommended = self._recommend_rule(calibration_rows)
        return {
            "metadata": {
                "phase": "8-D",
                "audit_only": True,
                "backtest_executed": False,
                "profile_added": False,
                "current_model_overwritten": False,
                "live_order_placement": False,
                "period": PERIOD,
            },
            "sources": self._sources(),
            "phase8c_snapshot": self._phase8c_snapshot(phase8c),
            "score_distribution": self._score_distribution(scored_v282, v290_trades),
            "current_thresholds": self._current_thresholds(),
            "calibration_candidates": calibration_rows,
            "pm130_reproducibility": self._pm130_reproducibility(scored_v282, calibration_rows),
            "pm080_overuse_check": self._pm080_overuse(scored_v282, calibration_rows),
            "verdict": self._verdict(calibration_rows, recommended),
        }

    def save_report(self, report: dict[str, Any]) -> Phase8DReportPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase8DReportPaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 8-D PM AI v2 Calibration Redesign Audit",
                "",
                "## Scope",
                "",
                "- audit only",
                "- no backtest, no profile addition, no current model overwrite",
                "",
                "## Score Distribution",
                "",
                self._table(report["score_distribution"], ["source", "metric", "min", "p10", "p25", "p50", "p75", "p90", "p95", "max"]),
                "",
                "## Current Thresholds",
                "",
                self._table([report["current_thresholds"]], ["pm130", "pm115", "pm100", "pm080", "threshold_too_strict"]),
                "",
                "## Calibration Candidates",
                "",
                self._table(report["calibration_candidates"], ["rule", "estimated_profit", "estimated_profit_delta_vs_v2_90", "estimated_profit_delta_vs_v2_82", "estimated_pf", "estimated_capital_utilization_direction", "estimated_dd_direction", "pm130_count", "pm130_profit_approximation", "pm080_count", "pm080_profit_approximation", "current_pm130_recall", "current_pm130_precision"]),
                "",
                "## PM 1.30 Reproducibility",
                "",
                self._table(report["pm130_reproducibility"], ["rule", "current_pm130_recall_by_rule", "current_pm130_precision_by_rule", "recovered_pm130_profit", "missed_pm130_profit"]),
                "",
                "## PM 0.80 Overuse Check",
                "",
                self._table(report["pm080_overuse_check"], ["rule", "pm080_count_by_rule", "pm080_profit_by_rule", "pm080_overuse_risk"]),
                "",
                "## Verdict",
                "",
                self._table([report["verdict"]], ["calibration_rule_recommended", "pm_ai_v2_calibration_feasible", "pm_ai_v2_needs_retraining", "pm_ai_v2_needs_label_redesign", "ready_for_phase8e_backtest", "reason"]),
                "",
            ]
        )

    def _sources(self) -> dict[str, str]:
        return {
            "phase8c_report": str(self.root / PHASE8C_JSON),
            "v290_trades": str(self.root / V290_LOG / "trades.csv"),
            "v290_purchase_audit": str(self.root / V290_LOG / "purchase_audit.csv"),
            "v282_trades": str(self.root / V282_FINAL / "trades.csv"),
            "v282_purchase_audit": str(self.root / V282_FINAL / "purchase_audit.csv"),
            "candidate_model_metadata": str(self.root / CANDIDATE_PM_DIR / "model_metadata.json"),
            "candidate_feature_columns": str(self.root / CANDIDATE_PM_DIR / "feature_columns.json"),
            "candidate_preprocess": str(self.root / CANDIDATE_PM_DIR / "preprocess.json"),
            "candidate_dataset_for_scoring": str(self.root / CANDIDATE_DATASET),
        }

    def _scored_v282_trades(self) -> pd.DataFrame:
        audit = Phase8BPMCandidateIntegrationAudit(self.root)
        trades = audit._load_trades()  # Existing read-only scorer used by Phase 8-B.
        scored = audit._score_candidate_pm(trades)
        compared = audit._compare_current_and_candidate(scored)
        return compared[compared.get("candidate_prediction_available", pd.Series(False, index=compared.index)).astype(bool)].copy()

    def _load_v290_trades(self) -> pd.DataFrame:
        trades = _read_csv(self.root / V290_LOG / "trades.csv")
        if trades.empty:
            return trades
        if "action" in trades.columns:
            trades = trades[trades["action"].fillna("").astype(str).eq("SELL")].copy()
        return trades

    def _phase8c_snapshot(self, phase8c: dict[str, Any]) -> dict[str, Any]:
        basic = {row.get("label"): row for row in phase8c.get("basic_comparison", []) if isinstance(row, dict)}
        return {
            "v2_82_net_profit": basic.get("v2_82_cap38", {}).get("net_profit"),
            "v2_90_net_profit": basic.get("v2_90_pm_ai_v2_api_only_cap38", {}).get("net_profit"),
            "v2_82_profit_factor": basic.get("v2_82_cap38", {}).get("profit_factor"),
            "v2_90_profit_factor": basic.get("v2_90_pm_ai_v2_api_only_cap38", {}).get("profit_factor"),
            "v2_82_capital_utilization": basic.get("v2_82_cap38", {}).get("average_capital_utilization"),
            "v2_90_capital_utilization": basic.get("v2_90_pm_ai_v2_api_only_cap38", {}).get("average_capital_utilization"),
        }

    def _score_distribution(self, scored_v282: pd.DataFrame, v290: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        sources = [
            ("v2_82_scored_by_pm_v2", scored_v282),
            ("v2_90_actual", v290.rename(columns={"pm_high_conviction_proba": "candidate_high_conviction_proba", "pm_avoid_proba": "candidate_avoid_proba", "pm_score": "candidate_pm_score"})),
        ]
        for source, frame in sources:
            for metric in ["candidate_high_conviction_proba", "candidate_avoid_proba", "candidate_pm_score"]:
                row = {"source": source, "metric": metric}
                row.update(_quantiles(frame.get(metric)))
                rows.append(row)
        return rows

    def _current_thresholds(self) -> dict[str, Any]:
        return {
            "pm130": "high_conviction_proba - avoid_proba >= 0.40",
            "pm115": ">= 0.20",
            "pm100": ">= 0.00",
            "pm080": ">= -0.20",
            "threshold_too_strict": True,
        }

    def _calibration_rows(self, scored: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        base_profit_v282 = 3_777_545.04
        base_profit_v290 = 791_720.04
        for rule in ["Rule A", "Rule B", "Rule C", "Rule D", "Rule E", "Rule F"]:
            row = self._rule_result(scored, rule, base_profit_v282, base_profit_v290)
            rows.append(row)
        return rows

    def _rule_result(self, scored: pd.DataFrame, rule: str, base_v282: float, base_v290: float) -> dict[str, Any]:
        out = scored.copy()
        out["virtual_multiplier"] = assign_calibration(out, rule)
        current_multiplier = _numeric(out.get("current_pm_multiplier")).replace(0, pd.NA)
        out["virtual_profit"] = _profit_series(out).reindex(out.index, fill_value=0.0) * (out["virtual_multiplier"] / current_multiplier).fillna(1.0)
        profits = _numeric(out["virtual_profit"])
        pm130 = out[out["virtual_multiplier"].round(2).eq(1.30)]
        pm080 = out[out["virtual_multiplier"].round(2).eq(0.80)]
        current_pm130 = out[_numeric(out.get("current_pm_multiplier")).round(2).eq(1.30)]
        recovered = current_pm130[current_pm130["virtual_multiplier"].round(2).eq(1.30)]
        precision = _safe_div(len(recovered), len(pm130)) if len(pm130) else 0.0
        recall = _safe_div(len(recovered), len(current_pm130)) if len(current_pm130) else 0.0
        estimated = float(profits.sum())
        return {
            "rule": rule,
            "estimated_profit": estimated,
            "estimated_profit_delta_vs_v2_90": estimated - base_v290,
            "estimated_profit_delta_vs_v2_82": estimated - base_v282,
            "estimated_pf": _profit_factor(profits),
            "estimated_capital_utilization_direction": "up_vs_v2_90" if pm130.shape[0] > 0 else "flat_or_down",
            "estimated_dd_direction": "moderate_upside_risk" if pm130.shape[0] >= 60 else "limited",
            "pm130_count": int(len(pm130)),
            "pm130_profit_approximation": float(_numeric(pm130["virtual_profit"]).sum()) if not pm130.empty else 0.0,
            "pm080_count": int(len(pm080)),
            "pm080_profit_approximation": float(_numeric(pm080["virtual_profit"]).sum()) if not pm080.empty else 0.0,
            "current_pm130_recall": recall,
            "current_pm130_precision": precision,
            "recovered_pm130_profit": float(_profit_series(recovered).sum()) if not recovered.empty else 0.0,
            "missed_pm130_profit": float(_profit_series(current_pm130[~current_pm130.index.isin(recovered.index)]).sum()) if not current_pm130.empty else 0.0,
        }

    def _pm130_reproducibility(self, scored: pd.DataFrame, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "rule": row["rule"],
                "current_pm130_recall_by_rule": row["current_pm130_recall"],
                "current_pm130_precision_by_rule": row["current_pm130_precision"],
                "recovered_pm130_profit": row["recovered_pm130_profit"],
                "missed_pm130_profit": row["missed_pm130_profit"],
            }
            for row in rows
        ]

    def _pm080_overuse(self, scored: pd.DataFrame, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        current_pm080_count = int(_numeric(scored.get("current_pm_multiplier")).round(2).eq(0.80).sum()) if not scored.empty else 0
        return [
            {
                "rule": row["rule"],
                "pm080_count_by_rule": row["pm080_count"],
                "pm080_profit_by_rule": row["pm080_profit_approximation"],
                "pm080_overuse_risk": row["pm080_count"] > current_pm080_count * 1.5,
            }
            for row in rows
        ]

    def _recommend_rule(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = [
            row
            for row in rows
            if row["pm130_count"] >= 60
            and row["estimated_profit_delta_vs_v2_90"] > 500_000
            and not (row["pm080_count"] > 250)
        ]
        if not candidates:
            return {}
        return max(candidates, key=lambda row: (row["estimated_profit"], row["current_pm130_recall"]))

    def _verdict(self, rows: list[dict[str, Any]], recommended: dict[str, Any]) -> dict[str, Any]:
        feasible = bool(recommended)
        return {
            "calibration_rule_recommended": recommended.get("rule") if recommended else "none",
            "pm_ai_v2_calibration_feasible": feasible,
            "pm_ai_v2_needs_retraining": not feasible,
            "pm_ai_v2_needs_label_redesign": not feasible,
            "ready_for_phase8e_backtest": feasible,
            "reason": "Use virtual calibration backtest next." if feasible else "Candidate scores are too compressed or fail PM1.30 recovery under tested rules.",
        }

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = []
        for row in rows:
            body.append("| " + " | ".join(self._format(row.get(column)) for column in columns) + " |")
        return "\n".join([header, sep, *body])

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")


def build_phase8d_report(root: Path | str = ROOT) -> dict[str, Any]:
    return Phase8DPMCalibrationAudit(root).build_report()

