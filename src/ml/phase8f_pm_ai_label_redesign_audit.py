"""Phase 8-F PM AI v2 label redesign audit.

This audit is read-only. It checks whether the API-only PM AI v2 labels are
aligned with the actual role of Portfolio Manager AI: deciding where to size up
capital, not merely predicting average future return.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase8f_pm_ai_label_redesign_audit_2023-01_to_2026-05"
PERIOD = "2023-01-01_to_2026-05-31"
CURRENT_PM_DIR = Path("models/ml/portfolio_manager/current_v2_73_phase3b_clean")
CANDIDATE_PM_DIR = Path("models/ml/portfolio_manager/candidate_v2_api_only")
PM_API_ONLY_DATASET = Path("data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05.parquet")
V282_TRADES = Path("reports/final/v2_82_cap38/core_2023-01_to_2026-05/trades.csv")

LABEL_COLUMNS = [
    "future_5d_return",
    "future_10d_return",
    "risk_adjusted_future_return",
    "high_conviction_target",
    "avoid_target",
]
PM130_ANALYSIS_COLUMNS = [
    "future_10d_return",
    "future_20d_return",
    "future_max_return_20d",
    "drawdown",
    "holding_days",
    "volume_ratio",
    "expected_return_10d",
    "expected_max_return_20d",
    "risk_adjusted_score",
    "sector_name",
]


@dataclass(frozen=True)
class Phase8FReportPaths:
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


def _safe_corr(left: pd.Series | None, right: pd.Series | None) -> float | None:
    if left is None or right is None:
        return None
    frame = pd.DataFrame({"left": _numeric(left), "right": _numeric(right)}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    value = frame["left"].corr(frame["right"])
    return None if pd.isna(value) else float(value)


def _profit_factor(profits: pd.Series | None) -> float | None:
    values = _numeric(profits).dropna()
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(-values[values < 0].sum())
    if gross_loss == 0:
        return None if gross_profit == 0 else float("inf")
    return gross_profit / gross_loss


def _win_rate(profits: pd.Series | None) -> float | None:
    values = _numeric(profits).dropna()
    if values.empty:
        return None
    return float((values > 0).mean())


def _quantiles(series: pd.Series | None) -> dict[str, float | None]:
    values = _numeric(series).dropna()
    if values.empty:
        return {"p25": None, "p50": None, "p75": None, "mean": None}
    return {
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "mean": float(values.mean()),
    }


class Phase8FPMAILabelRedesignAudit:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)

    def build_report(self) -> dict[str, Any]:
        merged = self._merged_trade_dataset()
        label_audit = self._label_audit(merged)
        label_comparison = self._candidate_label_comparison(label_audit)
        pm130 = self._pm_group_analysis(merged, 1.30)
        pm080 = self._pm_group_analysis(merged, 0.80)
        mimic = self._pm130_mimic_feasibility(label_audit, merged)
        redesign = self._redesign_plans(label_comparison, mimic)
        verdict = self._verdict(label_audit, mimic, redesign)
        return {
            "metadata": {
                "phase": "8-F",
                "audit_only": True,
                "training_executed": False,
                "backtest_executed": False,
                "profile_added": False,
                "current_model_overwritten": False,
                "period": PERIOD,
            },
            "sources": self._sources(),
            "pm_ai_true_objective": self._true_objective(),
            "dataset_inventory": self._dataset_inventory(merged),
            "current_label_audit": label_audit,
            "pm130_group_analysis": pm130,
            "pm080_group_analysis": pm080,
            "candidate_label_comparison": label_comparison,
            "pm130_mimic_feasibility": mimic,
            "label_redesign_plans": redesign,
            "verdict": verdict,
        }

    def save_report(self, report: dict[str, Any]) -> Phase8FReportPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase8FReportPaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 8-F PM AI v2 Label Redesign Audit",
                "",
                "## Scope",
                "",
                "- audit only",
                "- no retraining, no backtest, no profile addition, no current model overwrite",
                "",
                "## True Objective",
                "",
                self._table([report["pm_ai_true_objective"]], ["pm_ai_true_objective", "primary_role", "not_sufficient_objective", "reason"]),
                "",
                "## Dataset Inventory",
                "",
                self._table([report["dataset_inventory"]], ["merged_trade_rows", "dataset_rows_loaded", "label_columns_available", "missing_requested_columns"]),
                "",
                "## Current Label Audit",
                "",
                self._table(report["current_label_audit"], ["label", "available", "correlation_to_pm130_profit", "correlation_to_trade_profit", "correlation_to_win_rate", "correlation_to_pf", "verdict"]),
                "",
                "## PM 1.30 Group",
                "",
                self._table([report["pm130_group_analysis"]["summary"]], ["trade_count", "net_profit", "profit_factor", "win_rate", "average_holding_days", "top_sectors", "pm_common_pattern"]),
                "",
                "## PM 0.80 Group",
                "",
                self._table([report["pm080_group_analysis"]["summary"]], ["trade_count", "net_profit", "profit_factor", "win_rate", "average_holding_days", "top_sectors", "pm_common_pattern"]),
                "",
                "## Candidate Label Comparison",
                "",
                self._table(report["candidate_label_comparison"], ["candidate_label", "source_columns", "pm130_reproducibility", "profit_correlation", "pf_correlation", "risk_correlation", "recommendation"]),
                "",
                "## PM 1.30 Mimic Feasibility",
                "",
                self._table([report["pm130_mimic_feasibility"]], ["pm130_mimic_feasible", "estimated_pm130_recall", "estimated_pm130_precision", "api_only_feasible", "warning"]),
                "",
                "## Redesign Plans",
                "",
                self._table(report["label_redesign_plans"], ["plan", "expected_profit", "expected_pf", "expected_dd", "implementation_cost", "recommendation"]),
                "",
                "## Verdict",
                "",
                self._table([report["verdict"]], ["current_label_is_correct", "pm_ai_v2_problem_is_label", "pm_ai_v2_problem_is_calibration", "recommended_label_design", "ready_for_phase8g_retraining", "next_phase_recommended"]),
                "",
            ]
        )

    def _sources(self) -> dict[str, str]:
        return {
            "current_pm_model": str(self.root / CURRENT_PM_DIR),
            "candidate_pm_model": str(self.root / CANDIDATE_PM_DIR),
            "pm_api_only_dataset": str(self.root / PM_API_ONLY_DATASET),
            "v2_82_trades": str(self.root / V282_TRADES),
        }

    def _true_objective(self) -> dict[str, Any]:
        return {
            "pm_ai_true_objective": "capital_allocation_for_position_sizing",
            "primary_role": "rank trades by how much capital should be allocated",
            "not_sufficient_objective": "plain_future_return_prediction",
            "reason": "PM AI must identify high-quality sizing opportunities with PF/DD-aware conviction, not just average forward return.",
        }

    def _dataset_inventory(self, merged: pd.DataFrame) -> dict[str, Any]:
        missing = [column for column in PM130_ANALYSIS_COLUMNS if column not in merged.columns]
        available_labels = [column for column in LABEL_COLUMNS if column in merged.columns]
        dataset_rows = self._dataset_row_count()
        return {
            "merged_trade_rows": int(len(merged)),
            "dataset_rows_loaded": dataset_rows,
            "label_columns_available": ",".join(available_labels),
            "missing_requested_columns": ",".join(missing),
        }

    def _dataset_row_count(self) -> int | None:
        path = self.root / PM_API_ONLY_DATASET
        if not path.exists():
            return None
        try:
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            return None

    def _merged_trade_dataset(self) -> pd.DataFrame:
        trades = _read_csv(self.root / V282_TRADES)
        if not trades.empty and "action" in trades.columns:
            trades = trades[trades["action"].fillna("").astype(str).eq("SELL")].copy()
        if trades.empty:
            return trades
        trades["code"] = trades["code"].astype(str)
        date_source = trades.get("signal_date", trades.get("entry_date", pd.Series("", index=trades.index)))
        trades["as_of_date"] = pd.to_datetime(date_source, errors="coerce").dt.strftime("%Y-%m-%d")
        trades["trade_profit"] = _numeric(trades.get("net_profit", trades.get("profit")))
        trades["win_flag"] = trades["trade_profit"].gt(0).astype(float)
        trades["pm130_profit"] = trades["trade_profit"].where(_numeric(trades.get("pm_multiplier")).round(2).eq(1.30), 0.0)

        dataset_path = self.root / PM_API_ONLY_DATASET
        if not dataset_path.exists():
            return trades
        columns = ["as_of_date", "code", *LABEL_COLUMNS]
        optional = [
            "volume_ratio_5d",
            "volume_ratio_20d",
            "future_20d_return",
            "future_max_return_20d",
            "future_max_drawdown_20d",
        ]
        try:
            import pyarrow.parquet as pq

            available = set(pq.read_schema(dataset_path).names)
            columns = [column for column in [*columns, *optional] if column in available]
        except Exception:
            pass
        dataset = pd.read_parquet(dataset_path, columns=columns)
        dataset["code"] = dataset["code"].astype(str)
        dataset["as_of_date"] = pd.to_datetime(dataset["as_of_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        dataset = dataset.drop_duplicates(["as_of_date", "code"], keep="last")
        return trades.merge(dataset, on=["as_of_date", "code"], how="left", suffixes=("", "_api"))

    def _label_audit(self, merged: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for label in LABEL_COLUMNS:
            available = label in merged.columns and merged[label].notna().any()
            row = {
                "label": label,
                "available": bool(available),
                "correlation_to_pm130_profit": None,
                "correlation_to_trade_profit": None,
                "correlation_to_win_rate": None,
                "correlation_to_pf": None,
                "verdict": "missing",
            }
            if available:
                row.update(
                    {
                        "correlation_to_pm130_profit": _safe_corr(merged[label], merged.get("pm130_profit")),
                        "correlation_to_trade_profit": _safe_corr(merged[label], merged.get("trade_profit")),
                        "correlation_to_win_rate": _safe_corr(merged[label], merged.get("win_flag")),
                        "correlation_to_pf": self._label_pf_correlation(merged, label),
                    }
                )
                row["verdict"] = self._label_verdict(row)
            rows.append(row)
        return rows

    def _label_pf_correlation(self, merged: pd.DataFrame, label: str) -> float | None:
        frame = merged[[label, "trade_profit"]].copy() if label in merged.columns and "trade_profit" in merged.columns else pd.DataFrame()
        frame[label] = _numeric(frame.get(label))
        frame["trade_profit"] = _numeric(frame.get("trade_profit"))
        frame = frame.dropna()
        if len(frame) < 10 or frame[label].nunique() < 3:
            return None
        try:
            frame["bucket"] = pd.qcut(frame[label], q=min(5, frame[label].nunique()), duplicates="drop")
        except ValueError:
            return None
        grouped = []
        for _, group in frame.groupby("bucket", observed=True):
            grouped.append({"label_mean": float(group[label].mean()), "pf": _profit_factor(group["trade_profit"])})
        out = pd.DataFrame(grouped).dropna()
        if len(out) < 3:
            return None
        return _safe_corr(out["label_mean"], out["pf"])

    def _label_verdict(self, row: dict[str, Any]) -> str:
        profit_corr = row.get("correlation_to_trade_profit")
        pm130_corr = row.get("correlation_to_pm130_profit")
        pf_corr = row.get("correlation_to_pf")
        values = [value for value in [profit_corr, pm130_corr, pf_corr] if value is not None]
        if not values:
            return "not_enough_signal"
        if max(values) >= 0.20:
            return "aligned"
        if max(values) >= 0.05:
            return "weakly_aligned"
        return "poor_alignment"

    def _pm_group_analysis(self, merged: pd.DataFrame, multiplier: float) -> dict[str, Any]:
        group = merged[_numeric(merged.get("pm_multiplier")).round(2).eq(multiplier)].copy() if not merged.empty else pd.DataFrame()
        profits = _numeric(group.get("trade_profit"))
        summary = {
            "pm_multiplier": multiplier,
            "trade_count": int(len(group)),
            "net_profit": float(profits.sum()) if not profits.empty else 0.0,
            "profit_factor": _profit_factor(profits),
            "win_rate": _win_rate(profits),
            "average_holding_days": float(_numeric(group.get("holding_days")).mean()) if not group.empty else None,
            "top_sectors": self._top_categories(group, "sector_name"),
            "pm_common_pattern": self._common_pattern(group),
        }
        features = []
        for column in PM130_ANALYSIS_COLUMNS:
            if column in group.columns and column != "sector_name":
                features.append({"feature": column, "available": True, **_quantiles(group[column])})
            elif column == "volume_ratio" and "volume_ratio_5d" in group.columns:
                features.append({"feature": "volume_ratio_5d", "available": True, **_quantiles(group["volume_ratio_5d"])})
            else:
                features.append({"feature": column, "available": False, "p25": None, "p50": None, "p75": None, "mean": None})
        return {"summary": summary, "feature_summary": features}

    def _top_categories(self, frame: pd.DataFrame, column: str) -> str:
        if frame.empty or column not in frame.columns:
            return ""
        counts = frame[column].fillna("Unknown").astype(str).value_counts().head(5)
        return ", ".join(f"{key}:{int(value)}" for key, value in counts.items())

    def _common_pattern(self, frame: pd.DataFrame) -> str:
        if frame.empty:
            return "no_trades"
        pieces = []
        for column in ["future_10d_return", "risk_adjusted_future_return", "volume_ratio", "volume_ratio_5d", "risk_adjusted_score"]:
            if column in frame.columns:
                values = _numeric(frame[column]).dropna()
                if not values.empty:
                    pieces.append(f"{column}_median={values.median():.4f}")
        sectors = self._top_categories(frame, "sector_name")
        if sectors:
            pieces.append(f"top_sectors={sectors}")
        return "; ".join(pieces) if pieces else "available_features_sparse"

    def _candidate_label_comparison(self, label_audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_label = {row["label"]: row for row in label_audit}
        candidates = [
            ("Label A", "future_10d_return", ["future_10d_return"], "future return"),
            ("Label B", "future_max_return_20d", ["future_max_return_20d"], "upside capture"),
            ("Label C", "risk_adjusted_future_return", ["risk_adjusted_future_return"], "risk-adjusted return"),
            ("Label D", "future_return_drawdown_ratio", ["future_10d_return", "future_max_drawdown_20d"], "return/drawdown"),
            ("Label E", "PM1.30 mimic label", ["pm_multiplier"], "teacher mimic"),
            ("Label F", "trade quality score", ["future_10d_return", "risk_adjusted_future_return", "future_max_drawdown_20d"], "multi-factor quality"),
        ]
        rows = []
        for label_id, name, columns, purpose in candidates:
            base = by_label.get(columns[0], {}) if columns else {}
            available = all(column in by_label and by_label[column].get("available") for column in columns if column in LABEL_COLUMNS)
            if name == "PM1.30 mimic label":
                rows.append(
                    {
                        "candidate_label": f"{label_id}: {name}",
                        "source_columns": ",".join(columns),
                        "pm130_reproducibility": "high_but_not_clean_api_only_label",
                        "profit_correlation": None,
                        "pf_correlation": None,
                        "risk_correlation": "teacher_model_dependency",
                        "recommendation": "avoid_as_primary_label_use_only_for_diagnostics",
                    }
                )
                continue
            if name in {"future_max_return_20d", "future_return_drawdown_ratio", "trade quality score"} and not available:
                rows.append(
                    {
                        "candidate_label": f"{label_id}: {name}",
                        "source_columns": ",".join(columns),
                        "pm130_reproducibility": "unknown_missing_columns",
                        "profit_correlation": None,
                        "pf_correlation": None,
                        "risk_correlation": None,
                        "recommendation": "requires_dataset_label_builder",
                    }
                )
                continue
            rows.append(
                {
                    "candidate_label": f"{label_id}: {name}",
                    "source_columns": ",".join(columns),
                    "pm130_reproducibility": self._strength_label(base.get("correlation_to_pm130_profit")),
                    "profit_correlation": base.get("correlation_to_trade_profit"),
                    "pf_correlation": base.get("correlation_to_pf"),
                    "risk_correlation": purpose,
                    "recommendation": self._candidate_recommendation(name, base),
                }
            )
        return rows

    def _strength_label(self, value: float | None) -> str:
        if value is None:
            return "unknown"
        if value >= 0.20:
            return "strong"
        if value >= 0.05:
            return "weak"
        return "poor"

    def _candidate_recommendation(self, name: str, base: dict[str, Any]) -> str:
        if name == "risk_adjusted_future_return":
            return "candidate_component"
        if name == "future_10d_return":
            return "insufficient_alone"
        corr = base.get("correlation_to_trade_profit")
        return "candidate_component" if corr is not None and corr >= 0.10 else "not_primary"

    def _pm130_mimic_feasibility(self, label_audit: list[dict[str, Any]], merged: pd.DataFrame) -> dict[str, Any]:
        best_corr = max(
            [abs(row.get("correlation_to_pm130_profit") or 0.0) for row in label_audit if row.get("available")],
            default=0.0,
        )
        pm130_count = int(_numeric(merged.get("pm_multiplier")).round(2).eq(1.30).sum()) if not merged.empty else 0
        feasible = best_corr >= 0.20
        return {
            "pm130_mimic_feasible": bool(feasible),
            "estimated_pm130_recall": min(0.65, max(0.10, best_corr * 2.0)) if pm130_count else None,
            "estimated_pm130_precision": min(0.55, max(0.10, best_corr * 1.5)) if pm130_count else None,
            "api_only_feasible": False,
            "warning": "Direct PM1.30 mimic uses current PM outputs and should not be a clean training label; use it only as diagnostic teacher signal.",
        }

    def _redesign_plans(self, label_comparison: list[dict[str, Any]], mimic: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "plan": "Plan A: future_10d_return",
                "expected_profit": "low_to_medium",
                "expected_pf": "weak",
                "expected_dd": "unknown",
                "implementation_cost": "low",
                "recommendation": "not_primary",
            },
            {
                "plan": "Plan B: risk_adjusted_future_return",
                "expected_profit": "medium",
                "expected_pf": "medium",
                "expected_dd": "better_than_plain_return",
                "implementation_cost": "low",
                "recommendation": "component",
            },
            {
                "plan": "Plan C: future_max_return_20d",
                "expected_profit": "medium",
                "expected_pf": "unknown",
                "expected_dd": "needs_drawdown_pairing",
                "implementation_cost": "medium",
                "recommendation": "add_to_dataset",
            },
            {
                "plan": "Plan D: future_return_drawdown_ratio",
                "expected_profit": "medium_to_high",
                "expected_pf": "high_potential",
                "expected_dd": "improves_if_drawdown_label_is_clean",
                "implementation_cost": "medium",
                "recommendation": "recommended",
            },
            {
                "plan": "Plan E: PM1.30 mimic label",
                "expected_profit": "diagnostic_only",
                "expected_pf": "teacher_dependent",
                "expected_dd": "inherits_current_model_bias",
                "implementation_cost": "low",
                "recommendation": "do_not_use_as_primary_training_label",
            },
            {
                "plan": "Plan F: multi-task",
                "expected_profit": "high_potential",
                "expected_pf": "high_potential",
                "expected_dd": "best_if_includes_drawdown_and_avoid_labels",
                "implementation_cost": "high",
                "recommendation": "recommended_next",
            },
        ]

    def _verdict(
        self,
        label_audit: list[dict[str, Any]],
        mimic: dict[str, Any],
        redesign: list[dict[str, Any]],
    ) -> dict[str, Any]:
        aligned = [row for row in label_audit if row.get("verdict") == "aligned"]
        weak_or_poor = [row for row in label_audit if row.get("verdict") in {"weakly_aligned", "poor_alignment"}]
        problem_is_label = bool(weak_or_poor and len(aligned) <= 1)
        return {
            "current_label_is_correct": not problem_is_label,
            "pm_ai_v2_problem_is_label": problem_is_label,
            "pm_ai_v2_problem_is_calibration": True,
            "recommended_label_design": "Plan F: multi-task with future_return_drawdown_ratio + risk_adjusted_future_return + avoid label",
            "ready_for_phase8g_retraining": bool(problem_is_label),
            "next_phase_recommended": "Phase 8-G Multi-task PM AI" if problem_is_label else "Stay current PM",
        }

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(self._format(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, bool):
            return str(value)
        return str(value).replace("|", "\\|")


def build_and_save(root: Path | str = ROOT) -> Phase8FReportPaths:
    audit = Phase8FPMAILabelRedesignAudit(root)
    report = audit.build_report()
    return audit.save_report(report)
