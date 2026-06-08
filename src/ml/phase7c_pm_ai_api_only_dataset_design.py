"""Phase 7-C PM AI API-only dataset design audit.

This module only designs a safer PM AI retraining dataset. It does not rebuild
the dataset, train models, run backtests, add profiles, or touch current model
directories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase7b_prime_pm_ai_leakage_fix import (
    PM_DATASET,
    PM_MODEL_DIR,
    ROOT,
    classify_pm_column_prime,
)


REPORT_STEM = "phase7c_pm_ai_api_only_dataset_design_2021_to_2026"
STOCK_DATASET = Path("data/ml/datasets/ml_dataset.parquet")
EXIT_AI_V2_DATASET = Path("data/ml/exit_ai_v2/exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet")

CANDIDATE_LIST_REBUILD_COLUMNS = {
    "selected_count_in_day",
    "candidate_count_in_day",
    "rank_in_day",
    "score_rank_in_day",
    "candidate_rank",
    "score_rank",
}

FORBIDDEN_REBUILD_COLUMNS = {
    "max_positions_remaining_before",
}

SAFE_STOCK_PREDICTION_COLUMNS = {
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "risk_adjusted_score",
}

LABEL_CANDIDATES = [
    "future_5d_return",
    "future_10d_return",
    "future_max_return_20d",
    "future_max_drawdown_20d",
    "risk_adjusted_future_return",
    "high_conviction_target",
    "avoid_target",
    "ideal_weight_bucket",
]


@dataclass(frozen=True)
class Phase7CPaths:
    markdown: Path
    json: Path


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value[:10]) + (", ..." if len(value) > 10 else "")
    return str(value).replace("\n", " ")


def is_candidate_list_feature(column: str) -> bool:
    lower = column.lower()
    return (
        column in CANDIDATE_LIST_REBUILD_COLUMNS
        or lower.startswith("day_avg_")
        or lower.startswith("day_max_")
        or lower.endswith("_percentile_in_day")
        or lower.endswith("_gap_to_best")
        or "candidate_strength" in lower
    )


def classify_rebuild_column(column: str) -> str:
    if column in FORBIDDEN_REBUILD_COLUMNS:
        return "backtest_position_state"
    if is_candidate_list_feature(column):
        return "forbidden_candidate_list_dependent"
    category = classify_pm_column_prime(column)
    if category == "model_prediction_feature" and column in SAFE_STOCK_PREDICTION_COLUMNS:
        return "stock_selection_walk_forward_prediction"
    if category == "candidate_list_dependent":
        return "forbidden_candidate_list_dependent"
    return category


class Phase7CPMAIAPIOnlyDatasetDesignAudit:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)

    def build_report(self) -> dict[str, Any]:
        pm_dataset = _read_parquet(self._root(PM_DATASET))
        stock_dataset = _read_parquet(self._root(STOCK_DATASET))
        exit_dataset = _read_parquet(self._root(EXIT_AI_V2_DATASET))
        metadata = _read_json(self._root(PM_MODEL_DIR) / "model_metadata.json") or {}
        column_audit = self._pm_column_audit(pm_dataset)
        feature_design = self._feature_design(pm_dataset, stock_dataset, column_audit)
        label_design = self._label_design(stock_dataset)
        data_availability = self._data_availability(stock_dataset, exit_dataset)
        plans = self._plan_comparison(feature_design, label_design, data_availability)
        final = self._final_judgement(feature_design, label_design, data_availability, plans)
        return {
            "metadata": {
                "phase": "7-C",
                "design_audit_only": True,
                "model_retraining_executed": False,
                "dataset_generated": False,
                "full_backtest_executed": False,
                "profile_added": False,
                "current_model_overwritten": False,
                "full_pytest_executed": False,
                "pm_model_profile": metadata.get("model_profile", ""),
            },
            "input_paths": {
                "pm_dataset": str(self._root(PM_DATASET)),
                "pm_model": str(self._root(PM_MODEL_DIR)),
                "stock_dataset": str(self._root(STOCK_DATASET)),
                "exit_ai_v2_dataset": str(self._root(EXIT_AI_V2_DATASET)),
            },
            "pm_dataset_column_audit": column_audit,
            "api_only_dataset_design": self._dataset_design_notes(),
            "feature_set_design": feature_design,
            "label_design": label_design,
            "data_availability_2021_to_2026": data_availability,
            "retraining_plan_comparison": plans,
            "final_judgement": final,
        }

    def save_report(self, result: dict[str, Any]) -> Phase7CPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path = report_dir / f"{REPORT_STEM}.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase7CPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# AI Retraining Phase 7-C PM AI API-Only Dataset Design",
            "",
            "## Scope",
            "",
            "- design audit only",
            "- no retraining, no dataset generation, no backtest, no profile addition, no current model overwrite, no full pytest",
            "",
            "## PM Dataset Column Audit",
            "",
            self._table(
                [result["pm_dataset_column_audit"]],
                ["total_columns", "safe_columns_count", "stock_prediction_columns_count", "forbidden_columns_count", "unknown_columns_count"],
            ),
            "",
            "## Feature Set Design",
            "",
            self._table(
                [result["feature_set_design"]],
                [
                    "recommended_feature_set",
                    "recommended_feature_count",
                    "candidate_feature_removal_required",
                    "stock_selection_prediction_allowed",
                    "forbidden_columns_to_remove",
                ],
            ),
            "",
            "## Label Design",
            "",
            self._table(
                result["label_design"]["labels"],
                ["label", "api_only", "leakage_risk", "retraining_suitability", "source"],
            ),
            "",
            "## 2021 Data Availability",
            "",
            self._table(
                result["data_availability_2021_to_2026"]["sources"],
                ["source", "exists", "date_range", "usable_rows", "usable_codes", "missing_rate", "walk_forward_availability", "blocking_issues"],
            ),
            "",
            "## Plan Comparison",
            "",
            self._table(
                result["retraining_plan_comparison"],
                ["plan", "benefit", "risk", "cost", "comparison_difficulty", "recommended_order"],
            ),
            "",
            "## Final Judgement",
            "",
            self._table(
                [result["final_judgement"]],
                [
                    "api_only_pm_dataset_feasible",
                    "candidate_feature_removal_required",
                    "recommended_label_set",
                    "recommended_feature_set",
                    "recommended_retraining_plan",
                    "ready_for_phase7d",
                ],
            ),
            "",
        ]
        return "\n".join(lines)

    def _root(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def _pm_column_audit(self, dataset: pd.DataFrame) -> dict[str, Any]:
        rows = []
        by_category: dict[str, list[str]] = {}
        for column in dataset.columns:
            category = classify_rebuild_column(str(column))
            by_category.setdefault(category, []).append(str(column))
            rows.append({"column": str(column), "category": category})
        safe_categories = {"api_price_feature", "api_financial_feature", "api_market_feature"}
        safe = [row["column"] for row in rows if row["category"] in safe_categories]
        stock_prediction = [row["column"] for row in rows if row["category"] == "stock_selection_walk_forward_prediction"]
        forbidden = [
            row["column"]
            for row in rows
            if row["category"].startswith("backtest_")
            or row["category"] in {"future_label", "target_label", "identifier", "forbidden_candidate_list_dependent"}
        ]
        unknown = by_category.get("unknown", [])
        return {
            "total_columns": int(len(dataset.columns)),
            "columns_by_category": {key: sorted(value) for key, value in sorted(by_category.items())},
            "safe_columns": sorted(safe),
            "safe_columns_count": len(safe),
            "stock_prediction_columns": sorted(stock_prediction),
            "stock_prediction_columns_count": len(stock_prediction),
            "forbidden_columns": sorted(forbidden),
            "forbidden_columns_count": len(forbidden),
            "unknown_columns": sorted(unknown),
            "unknown_columns_count": len(unknown),
        }

    def _feature_design(self, pm_dataset: pd.DataFrame, stock_dataset: pd.DataFrame, column_audit: dict[str, Any]) -> dict[str, Any]:
        stock_columns = set(stock_dataset.columns)
        safe_pm = [c for c in column_audit["safe_columns"] if c in stock_columns or c in pm_dataset.columns]
        stock_predictions = [c for c in SAFE_STOCK_PREDICTION_COLUMNS if c in pm_dataset.columns or c in stock_columns]
        recommended = sorted(set(safe_pm + stock_predictions) - set(column_audit["forbidden_columns"]))
        return {
            "recommended_feature_set": "api_price_financial_market_plus_stock_walk_forward_predictions_no_candidate_list_features",
            "recommended_features": recommended,
            "recommended_feature_count": len(recommended),
            "candidate_feature_removal_required": True,
            "stock_selection_prediction_allowed": "walk_forward_generated_only",
            "forbidden_columns_to_remove": column_audit["forbidden_columns"],
            "candidate_list_features_removed": sorted(c for c in column_audit["forbidden_columns"] if is_candidate_list_feature(c)),
            "expected_benefit": "removes same-day candidate-list dependency while preserving API-origin price, market, financial, volume, technical, and stock prediction signals",
            "expected_risk": "PM model may lose some contextual ranking signal; fair comparison requires rebuilding predictions walk-forward",
        }

    def _label_design(self, stock_dataset: pd.DataFrame) -> dict[str, Any]:
        labels = []
        columns = set(stock_dataset.columns)
        for label in LABEL_CANDIDATES:
            exists = label in columns
            if label in {"future_5d_return", "future_10d_return", "future_max_return_20d"}:
                api_only = exists
                risk = "low" if exists else "low_design_missing_column"
                suitability = "high" if exists else "requires_builder"
                source = "mechanical future return from API-origin price series"
            elif label == "future_max_drawdown_20d":
                api_only = True
                risk = "low" if exists else "low_design_requires_price_window_builder"
                suitability = "medium_high"
                source = "mechanical future drawdown from API-origin price series"
            elif label == "risk_adjusted_future_return":
                api_only = True
                risk = "low_design"
                suitability = "high"
                source = "future return minus future drawdown / volatility penalty from API-origin price series"
            else:
                api_only = True
                risk = "low_if_derived_from_future_return_without_backtest_selection"
                suitability = "high" if label != "ideal_weight_bucket" else "medium_high"
                source = "derived label from API-only future return distribution"
            labels.append(
                {
                    "label": label,
                    "exists_in_stock_dataset": exists,
                    "api_only": api_only,
                    "leakage_risk": risk,
                    "retraining_suitability": suitability,
                    "source": source,
                }
            )
        recommended = ["future_5d_return", "future_10d_return", "risk_adjusted_future_return", "high_conviction_target", "avoid_target"]
        return {
            "labels": labels,
            "recommended_label_set": recommended,
            "label_policy": "labels must be generated mechanically from API-origin future prices after as_of_date; no trades.csv or backtest outcomes",
        }

    def _data_availability(self, stock_dataset: pd.DataFrame, exit_dataset: pd.DataFrame) -> dict[str, Any]:
        stock = self._source_availability("stock_dataset", STOCK_DATASET, stock_dataset, "date")
        exit_ai = self._source_availability("exit_ai_v2_dataset", EXIT_AI_V2_DATASET, exit_dataset, "as_of_date")
        pm = self._source_availability("current_pm_dataset_reference", PM_DATASET, _read_parquet(self._root(PM_DATASET)), "signal_date")
        sources = [stock, exit_ai, pm]
        blocking = []
        if not stock["exists"]:
            blocking.append("stock dataset missing")
        if stock["date_from"] and stock["date_from"] > "2021-06-01":
            blocking.append("stock dataset does not start at 2021-06")
        return {
            "sources": sources,
            "available_from": min([s["date_from"] for s in sources if s["date_from"]] or [""]),
            "available_to": max([s["date_to"] for s in sources if s["date_to"]] or [""]),
            "usable_rows": stock["usable_rows"],
            "usable_codes": stock["usable_codes"],
            "missing_rate": stock["missing_rate"],
            "walk_forward_availability": "required_for_stock_prediction_features; current predictions cover PM period, builder must generate 2021-forward folds",
            "blocking_issues": blocking,
            "retraining_from_2021_feasible": not blocking and bool(stock["usable_rows"]),
        }

    def _source_availability(self, name: str, rel_path: Path, frame: pd.DataFrame, date_column: str) -> dict[str, Any]:
        date_from = None
        date_to = None
        if date_column in frame.columns and not frame.empty:
            dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
            if not dates.empty:
                date_from = str(dates.min().date())
                date_to = str(dates.max().date())
        missing_rate = None
        if not frame.empty:
            sample = frame.head(100_000)
            missing_rate = float(sample.isna().mean().mean())
        codes = int(frame["code"].astype(str).nunique()) if "code" in frame.columns and not frame.empty else 0
        return {
            "source": name,
            "path": str(self._root(rel_path)),
            "exists": self._root(rel_path).exists(),
            "date_from": date_from,
            "date_to": date_to,
            "date_range": f"{date_from} to {date_to}" if date_from else "",
            "usable_rows": int(len(frame)),
            "usable_codes": codes,
            "missing_rate": missing_rate,
            "walk_forward_availability": "required" if name == "stock_dataset" else "reference",
            "blocking_issues": "" if self._root(rel_path).exists() else "missing",
        }

    def _dataset_design_notes(self) -> dict[str, Any]:
        return {
            "candidate_dataset": "PM API-only dataset rebuilt from stock_dataset universe",
            "allowed_sources": [
                "API-origin price, volume, technical, financial, and market features",
                "Stock Selection AI walk-forward predictions generated only from past-trained folds",
                "future-return labels mechanically generated from API-origin prices",
            ],
            "forbidden_sources": [
                "trades.csv",
                "actual_* / realized_* / profit_* / cash_* / portfolio_* / position_*",
                "decision / exit_reason / skip_reason",
                "selected_count_in_day",
                "candidate_count_in_day / rank_in_day / same-day percentiles / gap-to-best / day aggregates",
            ],
            "join_keys": ["date/as_of_date", "code"],
        }

    def _plan_comparison(self, feature_design: dict[str, Any], label_design: dict[str, Any], data: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "plan": "Plan A: PM AI API-only dataset rebuild",
                "benefit": "safe retraining baseline",
                "risk": "medium: may retain candidate-context if not explicitly banned",
                "cost": "medium",
                "comparison_difficulty": "medium",
                "recommended_order": 2,
            },
            {
                "plan": "Plan B: PM AI API-only rebuild + complete candidate-list feature removal",
                "benefit": "best leakage hygiene and live reproducibility",
                "risk": "medium: removes currently important features",
                "cost": "medium",
                "comparison_difficulty": "medium",
                "recommended_order": 1,
            },
            {
                "plan": "Plan C: PM AI rebuild + Stock Selection prediction only",
                "benefit": "keeps core AI signal without same-day universe aggregates",
                "risk": "medium_low if walk-forward predictions are available",
                "cost": "medium_high",
                "comparison_difficulty": "high",
                "recommended_order": 3,
            },
            {
                "plan": "Plan D: PM AI rebuild + Market Regime features",
                "benefit": "may encode Phase 6 bear/bull behavior cleanly",
                "risk": "medium: extra feature design and ablation required",
                "cost": "medium",
                "comparison_difficulty": "medium_high",
                "recommended_order": 4,
            },
        ]

    def _final_judgement(
        self,
        feature_design: dict[str, Any],
        label_design: dict[str, Any],
        data: dict[str, Any],
        plans: list[dict[str, Any]],
    ) -> dict[str, Any]:
        feasible = bool(data["retraining_from_2021_feasible"] and feature_design["recommended_feature_count"] > 0)
        return {
            "api_only_pm_dataset_feasible": feasible,
            "candidate_feature_removal_required": True,
            "recommended_label_set": label_design["recommended_label_set"],
            "recommended_feature_set": feature_design["recommended_feature_set"],
            "recommended_retraining_plan": plans[1]["plan"],
            "ready_for_phase7d": feasible,
            "reason": "Stock dataset provides 2021-forward API-origin features; rebuild must remove candidate-list dependent columns and regenerate labels from future price only.",
        }

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)


def build_and_save_phase7c_report(root: Path | str = ROOT) -> Phase7CPaths:
    audit = Phase7CPMAIAPIOnlyDatasetDesignAudit(root)
    return audit.save_report(audit.build_report())
