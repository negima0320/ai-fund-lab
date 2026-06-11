"""Phase 13-C2 Horizon-Aware Valuation Training Dataset Audit.

This 2025-only audit designs and evaluates candidate training targets for a
future horizon-aware valuation model. It does not train models, regenerate
predictions, run strategy backtests, modify profiles, or overwrite models.
Future columns are used only for target construction and evaluation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase13a_horizon_reality_audit import ROOT
from ml.phase13c_horizon_aware_valuation_prototype import REQUIRED_MEAN_TRADE_RETURN_20D
from ml.phase13d_hold_exit_dataset_audit import CANDIDATE_SETS, Phase13DHoldExitDatasetAudit


REPORT_STEM = "phase13c2_horizon_aware_valuation_dataset_audit_2025"
TARGETS = [
    "target_a_high_return_20d",
    "target_b_high_peak_low_decay",
    "target_c_return_retention",
    "target_d_high_edge_low_downside",
    "target_e_composite_success",
]
DETECTABILITY_SCORES = [
    "candidate_strength",
    "opportunity_proba",
    "opportunity_downside_score",
    "downside_safe_score",
    "score_v5_return_retention_candidate",
]
TOP_K = [1, 3, 5, 10]
REQUIRED_REPORT_KEYS = [
    "recommended_training_target",
    "recommended_training_task",
    "recommended_label_horizon",
    "target_positive_rate",
    "expected_positive_mean_return_20d",
    "target_meets_required_trade_return",
    "dataset_ready_for_training",
    "ready_for_phase13c3",
    "ready_for_phase13e3",
    "ready_for_strategy_backtest",
    "recommended_next_phase",
    "leakage_risk",
    "blocking_issues",
]


@dataclass(frozen=True)
class Phase13C2Paths:
    markdown: Path
    json: Path


class Phase13C2HorizonAwareValuationDatasetAudit:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)
        self.phase13d = Phase13DHoldExitDatasetAudit(root)

    def run(self) -> Phase13C2Paths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data, source_info = self.phase13d.phase13a.load_comparison_dataset()
        data = self.prepare_dataset(data)
        leakage = self.leakage_checklist()
        leakage["blocking_issues"] = self.blocking_issues(data)
        leakage["leakage_risk"] = "low" if not leakage["blocking_issues"] else "medium"
        universe = self.primary_universe(data)
        distribution = self.target_distribution_audit(universe)
        detectability = self.existing_score_detectability_audit(universe)
        oos = self.strict_oos_design()
        readiness = self.dataset_readiness_scores(distribution, detectability, oos)
        recommendation = self.recommendation(distribution, detectability, readiness, leakage)
        return {
            "metadata": self.metadata(),
            "input_summary": self.input_summary(data, source_info, leakage),
            "target_definitions": self.target_definitions(),
            "target_distribution_audit": distribution,
            "existing_score_detectability_audit": detectability,
            "strict_oos_design_feasibility": oos,
            "dataset_readiness_score_definition": self.dataset_readiness_score_definition(),
            "dataset_readiness_scores": readiness,
            "final_recommendation": recommendation,
            "leakage_checklist": leakage,
            **{key: recommendation.get(key) for key in REQUIRED_REPORT_KEYS},
        }

    def prepare_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for column in result.columns:
            if column not in {"date", "code"}:
                result[column] = _numeric(result[column])
        for column in ["opportunity_downside_score", "downside_safe_score", "candidate_strength"]:
            if column in result.columns:
                result[f"{column}_pct"] = result.groupby("date")[column].rank(method="average", pct=True)
        result["score_v5_return_retention_candidate"] = (
            result.get("opportunity_downside_score_pct", 0.0) * 0.5
            + result.get("downside_safe_score_pct", 0.0) * 0.3
            + result.get("candidate_strength_pct", 0.0) * 0.2
        )
        ret = _numeric(result.get("future_return_20d"))
        peak = _numeric(result.get("future_max_return_20d"))
        dd = _numeric(result.get("future_max_drawdown_20d"))
        retention = ret / peak.where(peak > 0)
        result["target_a_high_return_20d"] = ret.ge(REQUIRED_MEAN_TRADE_RETURN_20D).astype(float)
        result["target_b_high_peak_low_decay"] = ((peak >= 0.10) & (ret >= 0.03)).astype(float)
        result["target_c_return_retention"] = ((retention >= 0.35) & (peak >= 0.08)).fillna(False).astype(float)
        result["target_d_high_edge_low_downside"] = ((ret >= 0.04) & (dd > -0.08)).astype(float)
        result["target_e_composite_success"] = ((ret >= 0.04) & (peak >= 0.08) & (dd > -0.10)).astype(float)
        return result

    def primary_universe(self, data: pd.DataFrame) -> pd.DataFrame:
        method = next(method for method in CANDIDATE_SETS if method.name == "candidate_strength_top50")
        return self.phase13d.method_frame(data, method)

    def target_distribution_audit(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for target in TARGETS:
            positives = data[_numeric(data[target]).fillna(0).astype(bool)]
            daily_counts = positives.groupby("date")["code"].count()
            all_dates = data["date"].drop_duplicates()
            daily_counts = daily_counts.reindex(all_dates, fill_value=0)
            rows.append(
                {
                    "target": target,
                    "positive_count": int(len(positives)),
                    "positive_rate": self.mean(data, target),
                    "mean_future_return_20d_when_positive": self.mean(positives, "future_return_20d"),
                    "mean_future_max_return_20d_when_positive": self.mean(positives, "future_max_return_20d"),
                    "mean_future_max_drawdown_20d_when_positive": self.mean(positives, "future_max_drawdown_20d"),
                    "downside_bad_rate_when_positive": self.mean(positives, "downside_bad_20d"),
                    "daily_positive_availability_rate": _safe_float((daily_counts > 0).mean()) if len(daily_counts) else None,
                    "avg_positive_candidates_per_day": _safe_float(daily_counts.mean()) if len(daily_counts) else None,
                    "min_positive_candidates_per_day": int(daily_counts.min()) if len(daily_counts) else 0,
                    "p50_positive_candidates_per_day": _safe_float(daily_counts.quantile(0.50)) if len(daily_counts) else None,
                    "p90_positive_candidates_per_day": _safe_float(daily_counts.quantile(0.90)) if len(daily_counts) else None,
                }
            )
        return rows

    def existing_score_detectability_audit(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for target in TARGETS:
            base_positive = self.mean(data, target) or 0.0
            positive_count = int(_numeric(data[target]).fillna(0).sum())
            for score in DETECTABILITY_SCORES:
                for top_k in TOP_K:
                    selected = self.top_n_by_day(data, score, top_k)
                    precision = self.mean(selected, target)
                    selected_positive = int(_numeric(selected[target]).fillna(0).sum())
                    rows.append(
                        {
                            "target": target,
                            "score": score,
                            "top_k": top_k,
                            "precision": precision,
                            "lift": _safe_float(precision / base_positive) if precision is not None and base_positive else None,
                            "recall": _safe_float(selected_positive / positive_count) if positive_count else None,
                            "base_positive_rate": base_positive,
                        }
                    )
        return rows

    def strict_oos_design(self) -> dict[str, Any]:
        forbidden = [
            "future_return_*",
            "future_max_return_*",
            "future_max_drawdown_*",
            "top_decile_*",
            "downside_bad_*",
            "backtest/trade/cash/portfolio/selected/bought/current_pm columns",
        ]
        return {
            "recommended_train_period": "2023-01-04 to 2023-12-31",
            "recommended_validation_period": "2024-01-01 to 2024-12-31",
            "recommended_test_period": "2025-01-01 to 2025-12-31",
            "strict_oos_possible": True,
            "feature_set_source": "Phase11-A allowed prediction-time features plus existing score columns; future columns excluded from features.",
            "forbidden_features": forbidden,
            "target_columns": TARGETS,
        }

    def dataset_readiness_scores(self, distribution: list[dict[str, Any]], detectability: list[dict[str, Any]], oos: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for dist in distribution:
            target = dist["target"]
            best_detection = max(
                (row for row in detectability if row["target"] == target and row["top_k"] == 5),
                key=lambda row: self.value(row, "precision"),
            )
            positive_score = self.score_closeness(self.value(dist, "positive_rate"), 0.15, 0.40)
            availability_score = min(100.0, self.value(dist, "daily_positive_availability_rate") * 100.0)
            return_score = min(100.0, self.value(dist, "mean_future_return_20d_when_positive") / REQUIRED_MEAN_TRADE_RETURN_20D * 100.0)
            downside_score = max(0.0, 100.0 - self.value(dist, "downside_bad_rate_when_positive") * 200.0)
            detect_score = min(100.0, self.value(best_detection, "lift") / 2.0 * 100.0) if best_detection.get("lift") is not None else 0.0
            oos_score = 100.0 if oos["strict_oos_possible"] else 0.0
            total = (
                positive_score * 0.15
                + availability_score * 0.20
                + return_score * 0.25
                + downside_score * 0.15
                + detect_score * 0.15
                + oos_score * 0.10
            )
            rows.append(
                {
                    "target": target,
                    "target_positive_rate_score": _safe_float(positive_score),
                    "daily_availability_score": _safe_float(availability_score),
                    "mean_return_when_positive_score": _safe_float(return_score),
                    "downside_control_score": _safe_float(downside_score),
                    "existing_score_detectability_score": _safe_float(detect_score),
                    "strict_oos_feasibility_score": _safe_float(oos_score),
                    "dataset_readiness_score": _safe_float(total),
                    "best_existing_score_top5": best_detection["score"],
                    "best_existing_score_top5_precision": best_detection["precision"],
                    "best_existing_score_top5_lift": best_detection["lift"],
                }
            )
        return rows

    def recommendation(
        self,
        distribution: list[dict[str, Any]],
        detectability: list[dict[str, Any]],
        readiness: list[dict[str, Any]],
        leakage: dict[str, Any],
    ) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return self.blocked_recommendation(leakage)
        best = max(readiness, key=lambda row: self.value(row, "dataset_readiness_score"))
        dist = next(row for row in distribution if row["target"] == best["target"])
        ready = self.value(best, "dataset_readiness_score") >= 70 and self.value(dist, "daily_positive_availability_rate") >= 0.80
        target_meets = self.value(dist, "mean_future_return_20d_when_positive") >= REQUIRED_MEAN_TRADE_RETURN_20D
        next_phase = "Phase13-C3 Horizon-Aware Valuation Model Prototype" if ready and target_meets else "Phase13-C3b Dataset Redesign"
        return {
            "recommended_training_target": best["target"],
            "recommended_training_task": "binary_classification" if ready else "ranking_classification",
            "recommended_label_horizon": "20d",
            "target_positive_rate": dist["positive_rate"],
            "expected_positive_mean_return_20d": dist["mean_future_return_20d_when_positive"],
            "target_meets_required_trade_return": target_meets,
            "dataset_ready_for_training": ready,
            "ready_for_phase13c3": ready and target_meets,
            "ready_for_phase13e3": False,
            "ready_for_strategy_backtest": False,
            "recommended_next_phase": next_phase,
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
            "reason": (
                f"target={best['target']} readiness={self.value(best, 'dataset_readiness_score'):.2f}; "
                f"positive_rate={self.value(dist, 'positive_rate'):.4f}; "
                f"mean_return_positive={self.value(dist, 'mean_future_return_20d_when_positive'):.4f}; "
                f"best_top5_score={best['best_existing_score_top5']} precision={self.value(best, 'best_existing_score_top5_precision'):.4f}."
            ),
        }

    def blocked_recommendation(self, leakage: dict[str, Any]) -> dict[str, Any]:
        return {
            "recommended_training_target": None,
            "recommended_training_task": "insufficient_evidence",
            "recommended_label_horizon": "20d",
            "target_positive_rate": None,
            "expected_positive_mean_return_20d": None,
            "target_meets_required_trade_return": False,
            "dataset_ready_for_training": False,
            "ready_for_phase13c3": False,
            "ready_for_phase13e3": False,
            "ready_for_strategy_backtest": False,
            "recommended_next_phase": "Phase13-Halt Reassess System Thesis",
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def target_definitions(self) -> dict[str, str]:
        return {
            "target_a_high_return_20d": "future_return_20d >= 0.0556",
            "target_b_high_peak_low_decay": "future_max_return_20d >= 0.10 and future_return_20d >= 0.03",
            "target_c_return_retention": "future_return_20d / future_max_return_20d >= 0.35 and future_max_return_20d >= 0.08",
            "target_d_high_edge_low_downside": "future_return_20d >= 0.04 and future_max_drawdown_20d > -0.08",
            "target_e_composite_success": "future_return_20d >= 0.04 and future_max_return_20d >= 0.08 and future_max_drawdown_20d > -0.10",
        }

    def dataset_readiness_score_definition(self) -> dict[str, str]:
        return {
            "target_positive_rate": "score closeness to 15%-40% usable positive-rate band",
            "daily_availability": "daily_positive_availability_rate * 100",
            "mean_return_when_positive": "mean_future_return_20d_when_positive / required_mean_trade_return_20d",
            "downside_control": "100 - downside_bad_rate_when_positive * 200",
            "existing_score_detectability": "best existing top5 lift / 2",
            "strict_oos_feasibility": "100 if train 2023 / validation 2024 / test 2025 is feasible",
            "total": "0.15 positive + 0.20 availability + 0.25 return + 0.15 downside + 0.15 detectability + 0.10 strict_oos",
        }

    def input_summary(self, data: pd.DataFrame, source_info: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_files": source_info["source_files"],
            "row_count": int(len(data)),
            "date_min": data["date"].min().date().isoformat() if not data.empty else None,
            "date_max": data["date"].max().date().isoformat() if not data.empty else None,
            "unique_code_count": int(data["code"].nunique()) if not data.empty else 0,
            "available_score_columns": source_info["available_score_columns"],
            "available_future_columns": source_info["available_future_columns"],
            "missing_columns": source_info["missing_columns"],
            "primary_candidate_universe": "candidate_strength_top50",
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def score_closeness(self, value: float, low: float, high: float) -> float:
        if low <= value <= high:
            return 100.0
        if value < low:
            return max(0.0, value / low * 100.0)
        return max(0.0, 100.0 - (value - high) / high * 100.0)

    def top_n_by_day(self, data: pd.DataFrame, score: str, n: int) -> pd.DataFrame:
        return (
            data.sort_values(["date", score, "turnover_value", "code"], ascending=[True, False, False, True])
            .groupby("date", sort=False, group_keys=False)
            .head(n)
            .copy()
        )

    def mean(self, frame: pd.DataFrame, column: str) -> float | None:
        if column not in frame.columns:
            return None
        values = _numeric(frame[column]).dropna()
        return _safe_float(values.mean()) if not values.empty else None

    def value(self, row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        try:
            if value is None:
                return 0.0
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if math.isnan(result) else result

    def blocking_issues(self, data: pd.DataFrame) -> list[str]:
        issues = []
        required = ["future_return_20d", "future_max_return_20d", "future_max_drawdown_20d", "candidate_strength", "opportunity_downside_score", "downside_safe_score"]
        for column in required:
            if column not in data.columns:
                issues.append(f"missing_required_column:{column}")
        if data.empty:
            issues.append("empty_2025_dataset")
        return issues

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_as_features": [],
            "future_columns_used_only_for_targets_and_evaluation": self.phase13d.phase13a.expected_future_columns(),
            "new_model_trained": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "strategy_backtest_executed": False,
            "full_backtest_executed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
            "period": {"start": "2025-01-01", "end": "2025-12-31"},
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "13-C2",
            "scope": "2025-only horizon-aware valuation target dataset audit",
            "new_model_trained": False,
            "strategy_backtest_executed": False,
            "full_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase13C2Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase13C2Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Phase 13-C2 Horizon-Aware Valuation Training Dataset Audit",
                "",
                "## Input Summary",
                "",
                self.table([report["input_summary"]], ["row_count", "date_min", "date_max", "unique_code_count", "available_score_columns", "available_future_columns", "missing_columns", "primary_candidate_universe", "leakage_risk", "blocking_issues"]),
                "",
                "## Target Definitions",
                "",
                self.table([{"target": key, "definition": value} for key, value in report["target_definitions"].items()], ["target", "definition"]),
                "",
                "## Target Distribution Audit",
                "",
                self.table(report["target_distribution_audit"], ["target", "positive_count", "positive_rate", "mean_future_return_20d_when_positive", "mean_future_max_return_20d_when_positive", "mean_future_max_drawdown_20d_when_positive", "downside_bad_rate_when_positive", "daily_positive_availability_rate", "avg_positive_candidates_per_day", "min_positive_candidates_per_day", "p50_positive_candidates_per_day", "p90_positive_candidates_per_day"]),
                "",
                "## Existing Score Detectability Audit",
                "",
                self.table(report["existing_score_detectability_audit"], ["target", "score", "top_k", "precision", "lift", "recall", "base_positive_rate"]),
                "",
                "## Strict OOS Design Feasibility",
                "",
                self.table([report["strict_oos_design_feasibility"]], ["recommended_train_period", "recommended_validation_period", "recommended_test_period", "strict_oos_possible", "feature_set_source", "forbidden_features", "target_columns"]),
                "",
                "## Dataset Readiness Scores",
                "",
                self.table(report["dataset_readiness_scores"], ["target", "target_positive_rate_score", "daily_availability_score", "mean_return_when_positive_score", "downside_control_score", "existing_score_detectability_score", "strict_oos_feasibility_score", "dataset_readiness_score", "best_existing_score_top5", "best_existing_score_top5_precision", "best_existing_score_top5_lift"]),
                "",
                "## Final Recommendation",
                "",
                self.table([report["final_recommendation"]], REQUIRED_REPORT_KEYS + ["reason"]),
                "",
                "## Leakage Checklist",
                "",
                self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_targets_and_evaluation", "new_model_trained", "existing_model_overwritten", "profile_changed", "strategy_backtest_executed", "full_backtest_executed", "historical_predictions_regenerated", "jquants_api_called", "openai_api_called", "leakage_risk", "blocking_issues"]),
                "",
            ]
        )

    def table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(self.format_value(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def format_value(self, value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return f"{value:.4f}"
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, list):
            return ", ".join(map(str, value))
        if value is None:
            return ""
        return str(value)
