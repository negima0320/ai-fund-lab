"""Phase 12-A dynamic capital allocation quality audit.

This module reads the Phase 11-A dataset and Phase 11-B3 research models,
generates 2025-only predictions, and audits allocation quality. It does not run
a strategy backtest, overwrite models, change profiles, or regenerate
historical predictions.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11b_valuation_engine_prototype import CLASSIFICATION_TARGET, DATASET_PATH, Phase11BValuationEnginePrototype
from ml.phase11b3_expected_downside_model import DOWNSIDE_BAD_THRESHOLD, DOWNSIDE_TARGET, MODEL_DIR as PHASE11B3_MODEL_DIR
from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase11i_strict_oos import TEST_END, TEST_START


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase12a_dynamic_capital_allocation_2025"
ARTIFACT_PATH = Path("data/ml/valuation_engine/phase12a_dynamic_capital_allocation_2025.parquet")
ROUND_LOT = 100
DAILY_BUDGET = 900_000.0
MAX_POSITIONS = 5

EVAL_COLUMNS = [
    "future_return_20d",
    "future_max_return_20d",
    "future_max_drawdown_20d",
    "opportunity_value_20d",
    "opportunity_top_decile_20d",
    DOWNSIDE_TARGET,
]


@dataclass(frozen=True)
class Phase12AOptions:
    daily_buy_budget: float = DAILY_BUDGET
    max_positions: int = MAX_POSITIONS
    round_lot: int = ROUND_LOT
    save_artifact: bool = True


@dataclass(frozen=True)
class Phase12APaths:
    markdown: Path
    json: Path
    artifact: Path | None


class Phase12ADynamicCapitalAllocation:
    def __init__(self, root: Path | str = ROOT, *, options: Phase12AOptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase12AOptions()

    def run(self) -> Phase12APaths:
        report, allocation_rows = self.build_report()
        return self.save_outputs(report, allocation_rows)

    def build_report(self) -> tuple[dict[str, Any], pd.DataFrame]:
        dataset = self.load_dataset()
        feature_columns = self.load_feature_columns()
        leakage = self.leakage_checklist(feature_columns)
        if leakage["blocking_issues"]:
            report = {
                "metadata": self.metadata(),
                "dataset_summary": self.dataset_summary(dataset, pd.DataFrame()),
                "feature_policy": {"feature_columns": feature_columns, "feature_count": len(feature_columns)},
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], leakage),
            }
            return report, pd.DataFrame()

        test = self.test_2025(dataset)
        scored = self.score_2025(test, feature_columns)
        allocation_rows = self.build_allocation_rows(scored)
        rule_quality = self.rule_quality(allocation_rows)
        report = {
            "metadata": self.metadata(),
            "dataset_summary": self.dataset_summary(dataset, test),
            "feature_policy": {"feature_columns": feature_columns, "feature_count": len(feature_columns)},
            "score_definitions": self.score_definitions(),
            "allocation_weight_definition": self.allocation_weight_definition(),
            "rule_quality": rule_quality,
            "summary": self.summary(rule_quality),
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(rule_quality, leakage),
        }
        return report, allocation_rows

    def load_dataset(self) -> pd.DataFrame:
        data = pd.read_parquet(self.root / DATASET_PATH)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        data[DOWNSIDE_TARGET] = (_numeric(data["future_max_drawdown_20d"]) <= DOWNSIDE_BAD_THRESHOLD).astype(int)
        return data.dropna(subset=["date", "code", CLASSIFICATION_TARGET, DOWNSIDE_TARGET, "future_max_drawdown_20d"]).reset_index(drop=True)

    def load_feature_columns(self) -> list[str]:
        path = self.root / PHASE11B3_MODEL_DIR / "feature_columns.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def load_models(self) -> tuple[Any, Any]:
        import joblib

        model_dir = self.root / PHASE11B3_MODEL_DIR
        opportunity = joblib.load(model_dir / "opportunity_top_decile_20d_classifier.joblib")
        downside = joblib.load(model_dir / "downside_bad_20d_classifier.joblib")
        return opportunity, downside

    def test_2025(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset[(dataset["date"] >= TEST_START) & (dataset["date"] <= TEST_END)].copy().reset_index(drop=True)

    def score_2025(self, test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        data = test.copy()
        for column in feature_columns:
            if pd.api.types.is_bool_dtype(data[column].dtype):
                data[column] = data[column].astype(int)
            else:
                data[column] = _numeric(data[column])
        data[feature_columns] = data[feature_columns].replace([float("inf"), float("-inf")], pd.NA)
        opportunity_model, downside_model = self.load_models()
        data["opportunity_proba"] = opportunity_model.predict_proba(data[feature_columns])[:, 1]
        data["downside_bad_proba"] = downside_model.predict_proba(data[feature_columns])[:, 1]
        data["opportunity_rank_percentile"] = data.groupby("date")["opportunity_proba"].rank(method="average", pct=True)
        data["downside_rank_percentile"] = data.groupby("date")["downside_bad_proba"].rank(method="average", pct=True)
        data["confidence"] = ((data["opportunity_proba"] - 0.5).abs() + (data["downside_bad_proba"] - 0.5).abs()).clip(0, 1)
        data["score_a"] = data["opportunity_proba"] * (1.0 - data["downside_bad_proba"])
        data["score_b"] = data["opportunity_proba"] - data["downside_bad_proba"]
        data["score_c"] = data["opportunity_rank_percentile"] * (1.0 - data["downside_bad_proba"])
        data["score_d"] = data["opportunity_proba"] * (1.0 - data["downside_rank_percentile"])
        data["score_e"] = 0.7 * data["opportunity_rank_percentile"] + 0.3 * (1.0 - data["downside_rank_percentile"])
        baseline = data["stock_selection_rank_score"] if "stock_selection_rank_score" in data.columns else data["risk_adjusted_score"]
        data["baseline_rank_score"] = _numeric(baseline).fillna(-10**18)
        return data.dropna(subset=["date", "code", "close"]).sort_values(["date", "code"]).reset_index(drop=True)

    def build_allocation_rows(self, scored: pd.DataFrame) -> pd.DataFrame:
        frames = [
            self.top5_rule(scored, "baseline_equal_weight_top5", "baseline_rank_score"),
            self.top5_rule(scored, "opportunity_only_top5", "opportunity_proba"),
            self.top5_rule(scored, "downside_safe_top5", "downside_bad_proba", ascending=True),
        ]
        for rule_name, score_column in [
            ("score_a_weighted", "score_a"),
            ("score_b_weighted", "score_b"),
            ("score_c_weighted", "score_c"),
            ("score_d_weighted", "score_d"),
            ("score_e_weighted", "score_e"),
        ]:
            frames.append(self.weighted_rule(scored, rule_name, score_column))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def top5_rule(self, scored: pd.DataFrame, rule_name: str, score_column: str, *, ascending: bool = False) -> pd.DataFrame:
        top = (
            scored.sort_values(["date", score_column, "turnover_value", "code"], ascending=[True, ascending, False, True])
            .groupby("date", group_keys=False)
            .head(self.options.max_positions)
            .copy()
        )
        top["allocation_rule"] = rule_name
        top["allocation_score"] = top[score_column]
        top["allocation_score_percentile"] = 1.0
        top["allocation_weight"] = 1.0
        top["allocation_bucket"] = "top5_equal"
        return top

    def weighted_rule(self, scored: pd.DataFrame, rule_name: str, score_column: str) -> pd.DataFrame:
        data = scored.copy()
        data["allocation_rule"] = rule_name
        data["allocation_score"] = data[score_column]
        data["allocation_score_percentile"] = data.groupby("date")[score_column].rank(method="average", pct=True)
        data["allocation_weight"] = data["allocation_score_percentile"].map(self.weight_from_percentile)
        data["allocation_bucket"] = data["allocation_score_percentile"].map(self.bucket_from_percentile)
        return data[data["allocation_weight"] > 0].copy()

    def weight_from_percentile(self, percentile: float) -> float:
        if percentile >= 0.95:
            return 1.0
        if percentile >= 0.90:
            return 0.70
        if percentile >= 0.80:
            return 0.40
        if percentile >= 0.70:
            return 0.20
        return 0.0

    def bucket_from_percentile(self, percentile: float) -> str:
        if percentile >= 0.95:
            return "p95_1.00"
        if percentile >= 0.90:
            return "p90_0.70"
        if percentile >= 0.80:
            return "p80_0.40"
        if percentile >= 0.70:
            return "p70_0.20"
        return "zero"

    def rule_quality(self, allocation_rows: pd.DataFrame) -> list[dict[str, Any]]:
        if allocation_rows.empty:
            return []
        rows = []
        for rule_name, group in allocation_rows.groupby("allocation_rule", sort=True):
            daily_weight = group.groupby("date")["allocation_weight"].sum()
            rows.append(
                {
                    "allocation_rule": rule_name,
                    "allocated_rows": int(len(group)),
                    "candidate_days": int(group["date"].nunique()),
                    "average_allocated_candidates_per_day": _safe_float(len(group) / group["date"].nunique()) if group["date"].nunique() else None,
                    "average_weight": self.mean(group, "allocation_weight"),
                    "weight_distribution": dict(sorted(Counter(group["allocation_bucket"]).items())),
                    "budget_usage_proxy": _safe_float((daily_weight.clip(upper=self.options.max_positions) / self.options.max_positions).mean()) if not daily_weight.empty else None,
                    **self.weighted_quality(group),
                    **self.unweighted_quality(group),
                }
            )
        return rows

    def weighted_quality(self, frame: pd.DataFrame) -> dict[str, Any]:
        weight = _numeric(frame["allocation_weight"])
        return {
            "weighted_future_return_20d": self.weighted_mean(frame, "future_return_20d", weight),
            "weighted_future_max_return_20d": self.weighted_mean(frame, "future_max_return_20d", weight),
            "weighted_future_max_drawdown_20d": self.weighted_mean(frame, "future_max_drawdown_20d", weight),
            "weighted_opportunity_value_20d": self.weighted_mean(frame, "opportunity_value_20d", weight),
            "weighted_opportunity_top_decile_rate": self.weighted_mean(frame, "opportunity_top_decile_20d", weight),
            "weighted_downside_bad_rate": self.weighted_mean(frame, DOWNSIDE_TARGET, weight),
        }

    def unweighted_quality(self, frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "future_return_20d_mean": self.mean(frame, "future_return_20d"),
            "future_max_return_20d_mean": self.mean(frame, "future_max_return_20d"),
            "future_max_drawdown_20d_mean": self.mean(frame, "future_max_drawdown_20d"),
            "opportunity_value_20d_mean": self.mean(frame, "opportunity_value_20d"),
            "opportunity_top_decile_20d_rate": self.mean(frame, "opportunity_top_decile_20d"),
            "downside_bad_rate": self.mean(frame, DOWNSIDE_TARGET),
            "avg_opportunity_proba": self.mean(frame, "opportunity_proba"),
            "avg_downside_bad_proba": self.mean(frame, "downside_bad_proba"),
            "avg_confidence": self.mean(frame, "confidence"),
        }

    def weighted_mean(self, frame: pd.DataFrame, column: str, weight: pd.Series) -> float | None:
        if column not in frame.columns or frame.empty:
            return None
        values = _numeric(frame[column])
        valid = pd.DataFrame({"value": values, "weight": weight}).dropna()
        total_weight = float(valid["weight"].sum()) if not valid.empty else 0.0
        if total_weight <= 0:
            return None
        return _safe_float((valid["value"] * valid["weight"]).sum() / total_weight)

    def mean(self, frame: pd.DataFrame, column: str) -> float | None:
        values = _numeric(frame[column]) if column in frame.columns else pd.Series(dtype=float)
        return _safe_float(values.mean()) if not values.empty else None

    def summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = [
            row
            for row in rows
            if (row.get("weighted_opportunity_top_decile_rate") or 0.0) >= 0.20
            and (row.get("weighted_downside_bad_rate") or 1.0) <= 0.25
        ]
        ideal = [
            row
            for row in rows
            if (row.get("weighted_opportunity_top_decile_rate") or 0.0) >= 0.24
            and (row.get("weighted_downside_bad_rate") or 1.0) <= 0.20
        ]
        best = self.best_rule(rows)
        return {
            "best_allocation_rule": best.get("allocation_rule") if best else None,
            "best_rule_reason": self.best_rule_reason(best),
            "opportunity_downside_tradeoff_summary": self.tradeoff_summary(rows),
            "capital_utilization_proxy_summary": self.capital_summary(rows),
            "minimum_line_passed_rules": [row["allocation_rule"] for row in eligible],
            "ideal_line_passed_rules": [row["allocation_rule"] for row in ideal],
            "ready_for_phase12b": bool(eligible),
            "recommended_next_phase": "Phase12-B limited allocation strategy check" if eligible else "Phase12-A2 allocation score refinement",
        }

    def best_rule(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        def score(row: dict[str, Any]) -> tuple[float, float, float]:
            top = row.get("weighted_opportunity_top_decile_rate") or 0.0
            downside = row.get("weighted_downside_bad_rate") or 1.0
            opp_value = row.get("weighted_opportunity_value_20d") or -10**9
            penalty = max(0.0, downside - 0.25) * 2.0 + max(0.0, 0.20 - top)
            return (top - downside - penalty, opp_value, row.get("budget_usage_proxy") or 0.0)
        return max(rows, key=score)

    def best_rule_reason(self, row: dict[str, Any] | None) -> str:
        if not row:
            return "No allocation rows were produced."
        return (
            f"{row['allocation_rule']} balances weighted top-decile rate "
            f"{row.get('weighted_opportunity_top_decile_rate'):.4f} and weighted downside bad rate "
            f"{row.get('weighted_downside_bad_rate'):.4f} with budget usage proxy "
            f"{row.get('budget_usage_proxy'):.4f}."
        )

    def tradeoff_summary(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No rules evaluated."
        compact = [
            f"{row['allocation_rule']}: top={row.get('weighted_opportunity_top_decile_rate'):.4f}, downside={row.get('weighted_downside_bad_rate'):.4f}"
            for row in rows
        ]
        return "; ".join(compact)

    def capital_summary(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No rules evaluated."
        compact = [f"{row['allocation_rule']}: usage_proxy={row.get('budget_usage_proxy'):.4f}" for row in rows]
        return "; ".join(compact)

    def score_definitions(self) -> dict[str, str]:
        return {
            "score_a": "opportunity_proba * (1 - downside_bad_proba)",
            "score_b": "opportunity_proba - downside_bad_proba",
            "score_c": "opportunity_rank_percentile * (1 - downside_bad_proba)",
            "score_d": "opportunity_proba * (1 - downside_rank_percentile)",
            "score_e": "0.7 * opportunity_rank_percentile + 0.3 * (1 - downside_rank_percentile)",
        }

    def allocation_weight_definition(self) -> dict[str, float]:
        return {"p95": 1.0, "p90": 0.70, "p80": 0.40, "p70": 0.20, "else": 0.0}

    def leakage_checklist(self, feature_columns: list[str]) -> dict[str, Any]:
        helper = Phase11BValuationEnginePrototype(self.root)
        future = [column for column in feature_columns if column.startswith("future_") or column.startswith("opportunity_value") or column in {CLASSIFICATION_TARGET, DOWNSIDE_TARGET}]
        forbidden = [column for column in feature_columns if helper.is_forbidden_column(column)]
        blocking = []
        if future:
            blocking.append("future columns used as features")
        if forbidden:
            blocking.append("forbidden columns used as features")
        return {
            "future_columns_used_as_features": future,
            "future_columns_used_only_for_evaluation": EVAL_COLUMNS,
            "backtest_columns_used_as_features": [column for column in feature_columns if "backtest" in column.lower()],
            "trade_result_columns_used_as_features": [column for column in feature_columns if any(token in column.lower() for token in ["trade", "profit", "loss"])],
            "cash_or_portfolio_columns_used_as_model_features": [column for column in feature_columns if any(token in column.lower() for token in ["cash", "portfolio", "position"])],
            "selected_or_bought_used_as_features": any(any(token in column.lower() for token in ["selected", "bought", "affordable"]) for column in feature_columns),
            "current_pm_multiplier_used": any(any(token in column.lower() for token in ["pm_multiplier", "current_pm"]) for column in feature_columns),
            "historical_predictions_regenerated": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "leakage_risk": "high" if blocking else "low",
            "blocking_issues": blocking,
        }

    def recommendation(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"ready_for_phase12b": False, "recommended_next_phase": "Fix leakage blockers"}
        summary = self.summary(rows)
        return {
            "ready_for_phase12b": summary["ready_for_phase12b"],
            "recommended_next_phase": summary["recommended_next_phase"],
            "reason": "Proceed only if weighted top-decile rate >= 0.20 and weighted downside bad rate <= 0.25.",
        }

    def dataset_summary(self, dataset: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(dataset)),
            "test_rows": int(len(test)),
            "test_date_range": {
                "min": test["date"].min().date().isoformat() if not test.empty else None,
                "max": test["date"].max().date().isoformat() if not test.empty else None,
            },
            "test_candidate_days": int(test["date"].nunique()) if not test.empty else 0,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "12-A",
            "scope": "2025 allocation quality audit only",
            "model_source": str(self.root / PHASE11B3_MODEL_DIR),
            "strategy_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
        }

    def save_outputs(self, report: dict[str, Any], allocation_rows: pd.DataFrame) -> Phase12APaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        artifact_path = self.root / ARTIFACT_PATH if self.options.save_artifact and not allocation_rows.empty else None
        if artifact_path:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            allocation_rows.to_parquet(artifact_path, index=False)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12APaths(markdown=markdown_path, json=json_path, artifact=artifact_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-A Dynamic Capital Allocation Research",
            "",
            "## Summary",
            "",
            self.table([report["summary"]], ["best_allocation_rule", "best_rule_reason", "ready_for_phase12b", "recommended_next_phase"]),
            "",
            "## Rule Quality",
            "",
            self.table(
                report.get("rule_quality", []),
                [
                    "allocation_rule",
                    "allocated_rows",
                    "candidate_days",
                    "average_allocated_candidates_per_day",
                    "average_weight",
                    "budget_usage_proxy",
                    "weighted_future_return_20d",
                    "weighted_future_max_return_20d",
                    "weighted_future_max_drawdown_20d",
                    "weighted_opportunity_value_20d",
                    "weighted_opportunity_top_decile_rate",
                    "weighted_downside_bad_rate",
                    "opportunity_top_decile_20d_rate",
                    "downside_bad_rate",
                    "weight_distribution",
                ],
            ),
            "",
            "## Leakage Checklist",
            "",
            self.table(
                [report["leakage_checklist"]],
                [
                    "future_columns_used_as_features",
                    "future_columns_used_only_for_evaluation",
                    "backtest_columns_used_as_features",
                    "trade_result_columns_used_as_features",
                    "cash_or_portfolio_columns_used_as_model_features",
                    "selected_or_bought_used_as_features",
                    "current_pm_multiplier_used",
                    "historical_predictions_regenerated",
                    "existing_model_overwritten",
                    "profile_changed",
                    "full_backtest_executed",
                    "leakage_risk",
                    "blocking_issues",
                ],
            ),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["ready_for_phase12b", "recommended_next_phase", "reason"]),
            "",
        ]
        return "\n".join(lines)

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
