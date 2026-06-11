"""Phase 13-C Horizon-Aware Valuation Prototype.

This is a 2025-only read-only audit of existing prediction-time scores and
horizon-aware score combinations. It does not train models or run strategy
backtests. Future columns are used only for evaluation and theoretical
candidate-quality analysis, never to build scores.
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
from ml.phase13d_hold_exit_dataset_audit import CANDIDATE_SETS, Phase13DHoldExitDatasetAudit


REPORT_STEM = "phase13c_horizon_aware_valuation_prototype_2025"
REQUIRED_MEAN_TRADE_RETURN_20D = 0.0556
PHASE13R_ENTRY_MEAN_FUTURE_RETURN_20D = 0.027111155539625197
SCORE_COLUMNS_USED = [
    "opportunity_proba",
    "opportunity_downside_score",
    "candidate_strength",
    "downside_safe_score",
]
SCORE_NAMES = [
    "score_v1_opportunity_only",
    "score_v2_opportunity_downside",
    "score_v3_candidate_strength_downside_safe",
    "score_v4_balanced_edge",
    "score_v5_return_retention_candidate",
]
BUCKETS = [1, 3, 5, 10]
REQUIRED_REPORT_KEYS = [
    "recommended_valuation_score",
    "recommended_candidate_bucket",
    "expected_mean_trade_return_20d",
    "required_mean_trade_return_20d",
    "meets_required_trade_return",
    "candidate_quality_improved_vs_phase13r",
    "ready_for_phase13e3",
    "ready_for_phase13f",
    "ready_for_strategy_backtest",
    "recommended_next_phase",
    "leakage_risk",
    "blocking_issues",
]


@dataclass(frozen=True)
class Phase13CPaths:
    markdown: Path
    json: Path


class Phase13CHorizonAwareValuationPrototype:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)
        self.phase13d = Phase13DHoldExitDatasetAudit(root)

    def run(self) -> Phase13CPaths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data, source_info = self.phase13d.phase13a.load_comparison_dataset()
        data = self.prepare_scores(data)
        leakage = self.leakage_checklist()
        leakage["blocking_issues"] = self.blocking_issues(data)
        leakage["leakage_risk"] = "low" if not leakage["blocking_issues"] else "medium"
        candidate_sets = self.candidate_set_baselines(data)
        quality = self.candidate_quality(data)
        tradeoff = self.edge_downside_tradeoff(quality)
        retention = self.profit_retention_potential(data)
        recommendation = self.recommendation(quality, leakage)
        return {
            "metadata": self.metadata(),
            "input_artifact_summary": self.input_artifact_summary(data, source_info, leakage),
            "score_definitions": self.score_definitions(),
            "candidate_set_baselines": candidate_sets,
            "candidate_quality_comparison": quality,
            "edge_downside_tradeoff": tradeoff,
            "profit_retention_potential": retention,
            "final_recommendation": recommendation,
            "leakage_checklist": leakage,
            **{key: recommendation.get(key) for key in REQUIRED_REPORT_KEYS},
        }

    def prepare_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for column in result.columns:
            if column not in {"date", "code"}:
                result[column] = _numeric(result[column])
        for column in SCORE_COLUMNS_USED:
            if column in result.columns:
                result[f"{column}_pct"] = result.groupby("date")[column].rank(method="average", pct=True)
        result["score_v1_opportunity_only"] = result["opportunity_proba"]
        result["score_v2_opportunity_downside"] = result["opportunity_downside_score"]
        result["score_v3_candidate_strength_downside_safe"] = result.get("candidate_strength_pct", 0.0) + result.get("downside_safe_score_pct", 0.0)
        result["score_v4_balanced_edge"] = (
            result.get("opportunity_proba_pct", 0.0) * 0.4
            + result.get("candidate_strength_pct", 0.0) * 0.3
            + result.get("downside_safe_score_pct", 0.0) * 0.3
        )
        result["score_v5_return_retention_candidate"] = (
            result.get("opportunity_downside_score_pct", 0.0) * 0.5
            + result.get("downside_safe_score_pct", 0.0) * 0.3
            + result.get("candidate_strength_pct", 0.0) * 0.2
        )
        return result

    def candidate_set_baselines(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        wanted = {
            "stock_selection_rank_score_top50",
            "candidate_strength_top50",
            "valuation_first_top50",
            "opportunity_downside_top50",
            "candidate_strength_top5",
            "valuation_first_top5",
            "opportunity_downside_top5",
        }
        for method in [method for method in CANDIDATE_SETS if method.name in wanted]:
            frame = self.phase13d.method_frame(data, method)
            rows.append(self.metrics_row(method.name, method.score_column or "", method.top_n, frame))
        return rows

    def candidate_quality(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        universe = self.primary_universe(data)
        for score in SCORE_NAMES:
            for bucket in BUCKETS:
                frame = self.top_n_by_day(universe, score, bucket)
                rows.append(self.metrics_row(score, score, bucket, frame))
        return rows

    def metrics_row(self, name: str, score_column: str, bucket: int | None, frame: pd.DataFrame) -> dict[str, Any]:
        mean_ret = self.mean(frame, "future_return_20d")
        mean_peak = self.mean(frame, "future_max_return_20d")
        mean_dd = self.mean(frame, "future_max_drawdown_20d")
        decay = self.mean_decay(frame)
        return {
            "score_name": name,
            "score_column": score_column,
            "bucket": f"top{bucket}" if bucket else None,
            "sample_count": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if not frame.empty else 0,
            "mean_future_return_20d": mean_ret,
            "mean_future_max_return_20d": mean_peak,
            "mean_future_max_drawdown_20d": mean_dd,
            "top_decile_rate_20d": self.mean(frame, "top_decile_20d"),
            "downside_bad_rate_20d": self.mean(frame, "downside_bad_20d"),
            "theoretical_annual_return_hold20d": _safe_float(mean_ret * 252 / 20) if mean_ret is not None else None,
            "profit_retention_potential": _safe_float(mean_ret / mean_peak) if mean_ret is not None and mean_peak else None,
            "peak_to_final_decay_rate": decay,
            "meets_required_trade_return": bool(mean_ret is not None and mean_ret >= REQUIRED_MEAN_TRADE_RETURN_20D),
        }

    def edge_downside_tradeoff(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tradeoff = []
        for row in rows:
            mean_ret = row.get("mean_future_return_20d")
            mean_dd = row.get("mean_future_max_drawdown_20d")
            ratio = _safe_float(mean_ret / abs(mean_dd)) if mean_ret is not None and mean_dd not in {None, 0} else None
            tradeoff.append(
                {
                    "score_name": row["score_name"],
                    "bucket": row["bucket"],
                    "mean_future_return_20d": mean_ret,
                    "mean_future_max_return_20d": row.get("mean_future_max_return_20d"),
                    "mean_future_max_drawdown_20d": mean_dd,
                    "downside_bad_rate_20d": row.get("downside_bad_rate_20d"),
                    "edge_to_downside_ratio": ratio,
                }
            )
        return tradeoff

    def profit_retention_potential(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        universe = self.primary_universe(data)
        for score in SCORE_NAMES:
            for bucket in BUCKETS:
                frame = self.top_n_by_day(universe, score, bucket)
                ret = _numeric(frame.get("future_return_20d"))
                peak = _numeric(frame.get("future_max_return_20d"))
                retention = (ret / peak.replace(0, pd.NA)).replace([float("inf"), float("-inf")], pd.NA)
                decay = peak - ret
                rows.append(
                    {
                        "score_name": score,
                        "bucket": f"top{bucket}",
                        "sample_count": int(len(frame)),
                        "future_return_over_future_max_return_mean": _safe_float(retention.dropna().mean()) if not retention.dropna().empty else None,
                        "future_max_minus_future_return_mean": _safe_float(decay.dropna().mean()) if not decay.dropna().empty else None,
                        "peak_to_final_decay_rate": self.mean_decay(frame),
                    }
                )
        return rows

    def recommendation(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {
                "recommended_valuation_score": None,
                "recommended_candidate_bucket": None,
                "expected_mean_trade_return_20d": None,
                "required_mean_trade_return_20d": REQUIRED_MEAN_TRADE_RETURN_20D,
                "meets_required_trade_return": False,
                "candidate_quality_improved_vs_phase13r": False,
                "ready_for_phase13e3": False,
                "ready_for_phase13f": False,
                "ready_for_strategy_backtest": False,
                "recommended_next_phase": "Phase13-Halt Reassess System Thesis",
                "leakage_risk": leakage["leakage_risk"],
                "blocking_issues": leakage["blocking_issues"],
            }
        best = max(rows, key=self.recommendation_score)
        improved = self.value(best, "mean_future_return_20d") > PHASE13R_ENTRY_MEAN_FUTURE_RETURN_20D
        meets = bool(best.get("meets_required_trade_return"))
        close = self.value(best, "mean_future_return_20d") >= REQUIRED_MEAN_TRADE_RETURN_20D * 0.80
        if meets or close:
            next_phase = "Phase13-E3 Entry/Exit Interaction Audit"
        elif improved:
            next_phase = "Phase13-C2 Horizon-Aware Valuation Training Dataset Audit"
        else:
            next_phase = "Phase13-Halt Reassess System Thesis"
        return {
            "recommended_valuation_score": best["score_name"],
            "recommended_candidate_bucket": best["bucket"],
            "expected_mean_trade_return_20d": best["mean_future_return_20d"],
            "required_mean_trade_return_20d": REQUIRED_MEAN_TRADE_RETURN_20D,
            "meets_required_trade_return": meets,
            "candidate_quality_improved_vs_phase13r": improved,
            "ready_for_phase13e3": next_phase == "Phase13-E3 Entry/Exit Interaction Audit",
            "ready_for_phase13f": False,
            "ready_for_strategy_backtest": False,
            "recommended_next_phase": next_phase,
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
            "reason": (
                f"best={best['score_name']} {best['bucket']} mean20={self.value(best, 'mean_future_return_20d'):.4f}; "
                f"required={REQUIRED_MEAN_TRADE_RETURN_20D:.4f}; improved_vs_phase13r={improved}."
            ),
        }

    def recommendation_score(self, row: dict[str, Any]) -> float:
        ret = self.value(row, "mean_future_return_20d")
        peak = self.value(row, "mean_future_max_return_20d")
        downside = self.value(row, "downside_bad_rate_20d")
        retention = self.value(row, "profit_retention_potential")
        sample_penalty = 0.0 if self.value(row, "sample_count") >= 300 else 0.02
        return ret * 0.45 + peak * 0.20 + retention * 0.20 - downside * 0.10 - sample_penalty

    def primary_universe(self, data: pd.DataFrame) -> pd.DataFrame:
        method = next(method for method in CANDIDATE_SETS if method.name == "candidate_strength_top50")
        return self.phase13d.method_frame(data, method)

    def top_n_by_day(self, data: pd.DataFrame, score: str, n: int) -> pd.DataFrame:
        return (
            data.sort_values(["date", score, "turnover_value", "code"], ascending=[True, False, False, True])
            .groupby("date", sort=False, group_keys=False)
            .head(n)
            .copy()
        )

    def mean_decay(self, frame: pd.DataFrame) -> float | None:
        if "future_max_return_20d" not in frame.columns or "future_return_20d" not in frame.columns:
            return None
        decay = _numeric(frame["future_max_return_20d"]) - _numeric(frame["future_return_20d"])
        return _safe_float(decay.dropna().mean()) if not decay.dropna().empty else None

    def mean(self, frame: pd.DataFrame, column: str) -> float | None:
        if column not in frame.columns:
            return None
        values = _numeric(frame[column]).dropna()
        return _safe_float(values.mean()) if not values.empty else None

    def score_definitions(self) -> dict[str, str]:
        return {
            "score_v1_opportunity_only": "opportunity_proba",
            "score_v2_opportunity_downside": "opportunity_downside_score",
            "score_v3_candidate_strength_downside_safe": "same-day percentile(candidate_strength) + same-day percentile(downside_safe_score)",
            "score_v4_balanced_edge": "0.4 * percentile(opportunity_proba) + 0.3 * percentile(candidate_strength) + 0.3 * percentile(downside_safe_score)",
            "score_v5_return_retention_candidate": "0.5 * percentile(opportunity_downside_score) + 0.3 * percentile(downside_safe_score) + 0.2 * percentile(candidate_strength)",
            "future_columns_used_in_scores": "none",
        }

    def input_artifact_summary(self, data: pd.DataFrame, source_info: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_files": source_info["source_files"],
            "row_count": int(len(data)),
            "date_min": data["date"].min().date().isoformat() if not data.empty else None,
            "date_max": data["date"].max().date().isoformat() if not data.empty else None,
            "unique_code_count": int(data["code"].nunique()) if not data.empty else 0,
            "available_score_columns": source_info["available_score_columns"],
            "available_future_columns": source_info["available_future_columns"],
            "missing_columns": source_info["missing_columns"],
            "candidate_sets": [
                "candidate_strength_top50",
                "candidate_strength_top5",
                "valuation_first_top50",
                "valuation_first_top5",
                "opportunity_downside_top50",
                "opportunity_downside_top5",
                "stock_selection_rank_score_top50",
            ],
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def blocking_issues(self, data: pd.DataFrame) -> list[str]:
        issues = []
        for column in ["date", "code", *SCORE_COLUMNS_USED, "future_return_20d", "future_max_return_20d", "future_max_drawdown_20d"]:
            if column not in data.columns:
                issues.append(f"missing_required_column:{column}")
        if data.empty:
            issues.append("empty_2025_dataset")
        return issues

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_as_scores": [],
            "future_columns_used_only_for_evaluation": self.phase13d.phase13a.expected_future_columns(),
            "future_columns_used_as_features": [],
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
            "phase": "13-C",
            "scope": "2025-only horizon-aware valuation score audit",
            "new_model_trained": False,
            "strategy_backtest_executed": False,
            "full_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase13CPaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase13CPaths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Phase 13-C Horizon-Aware Valuation Prototype",
                "",
                "## Input Summary",
                "",
                self.table([report["input_artifact_summary"]], ["row_count", "date_min", "date_max", "unique_code_count", "available_score_columns", "available_future_columns", "missing_columns", "candidate_sets", "leakage_risk", "blocking_issues"]),
                "",
                "## Score Definitions",
                "",
                self.table([{"score_name": key, "definition": value} for key, value in report["score_definitions"].items()], ["score_name", "definition"]),
                "",
                "## Candidate Set Baselines",
                "",
                self.table(report["candidate_set_baselines"], ["score_name", "bucket", "sample_count", "mean_future_return_20d", "mean_future_max_return_20d", "mean_future_max_drawdown_20d", "top_decile_rate_20d", "downside_bad_rate_20d", "theoretical_annual_return_hold20d", "profit_retention_potential"]),
                "",
                "## Candidate Quality Comparison",
                "",
                self.table(report["candidate_quality_comparison"], ["score_name", "bucket", "sample_count", "mean_future_return_20d", "mean_future_max_return_20d", "mean_future_max_drawdown_20d", "top_decile_rate_20d", "downside_bad_rate_20d", "theoretical_annual_return_hold20d", "profit_retention_potential", "meets_required_trade_return"]),
                "",
                "## Edge / Downside Tradeoff",
                "",
                self.table(report["edge_downside_tradeoff"], ["score_name", "bucket", "mean_future_return_20d", "mean_future_max_return_20d", "mean_future_max_drawdown_20d", "downside_bad_rate_20d", "edge_to_downside_ratio"]),
                "",
                "## Profit Retention Potential",
                "",
                self.table(report["profit_retention_potential"], ["score_name", "bucket", "future_return_over_future_max_return_mean", "future_max_minus_future_return_mean", "peak_to_final_decay_rate"]),
                "",
                "## Final Recommendation",
                "",
                self.table([report["final_recommendation"]], REQUIRED_REPORT_KEYS + ["reason"]),
                "",
                "## Leakage Checklist",
                "",
                self.table([report["leakage_checklist"]], ["future_columns_used_as_scores", "future_columns_used_only_for_evaluation", "new_model_trained", "existing_model_overwritten", "profile_changed", "strategy_backtest_executed", "full_backtest_executed", "historical_predictions_regenerated", "jquants_api_called", "openai_api_called", "leakage_risk", "blocking_issues"]),
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

    def value(self, row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        try:
            if value is None:
                return 0.0
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if math.isnan(result) else result

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
