"""Phase 5-D Exit AI v2 training design audit.

This module designs training and comparison policy only. It does not train a
model, generate predictions, run a backtest, or modify current model paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase5d_exit_ai_v2_training_design_2021-06_to_2026-05"
DATASET_PATH = ROOT / "data" / "ml" / "exit_ai_v2" / "exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet"
EXISTING_EXIT_MODEL = ROOT / "models" / "ml" / "exit" / "current_v2_66"
PROPOSED_MODEL_DIR = ROOT / "models" / "ml" / "exit_ai_v2" / "candidate_v2_api_only"

KEY_COLUMNS = {"code", "as_of_date", "split"}
LABEL_COLUMNS = {
    "future_return_3d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "avoid_loss_5d",
    "miss_profit_5d",
    "exit_quality_score",
    "exit_quality_score_risk_adjusted",
    "future_max_drawdown_5d",
    "future_max_drawdown_10d",
    "future_max_return_5d",
    "future_max_return_10d",
}
FORBIDDEN_COLUMNS = {
    "trade_id",
    "actual_exit_date",
    "actual_sell_price",
    "realized_profit",
    "realized_return",
    "win",
    "loss",
    "holding_days",
    "remaining_days_to_actual_exit",
    "exit_reason",
    "selected_count_in_day",
    "portfolio_cash",
    "total_assets",
    "market_value",
    "profile_id",
}
HIGH_MISSING_REFERENCE = {
    "BPS": 0.6327,
    "FOP_growth": 0.4008,
    "FSales_growth": 0.3786,
    "FEPS_growth": 0.3611,
    "OP_growth": 0.3154,
    "Sales_growth": 0.2775,
    "NP_growth": 0.2767,
}
FINANCIAL_FEATURES = {
    "EPS",
    "BPS",
    "EqAR",
    "Sales_growth",
    "OP_growth",
    "NP_growth",
    "FEPS_growth",
    "FSales_growth",
    "FOP_growth",
    "PayoutRatioAnn",
}


@dataclass(frozen=True)
class Phase5DPaths:
    markdown: Path
    json: Path


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _is_label_like(column: str) -> bool:
    lower = column.lower()
    return (
        column in LABEL_COLUMNS
        or lower.startswith(("future_return_", "future_max_", "avoid_loss_", "miss_profit_"))
        or "exit_quality_score" in lower
        or "target" in lower
        or "label" in lower
    )


def _is_forbidden(column: str) -> bool:
    lower = column.lower()
    return column in FORBIDDEN_COLUMNS or "backtest" in lower or "v2_" in lower


def _feature_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if column not in KEY_COLUMNS and not _is_label_like(column) and not _is_forbidden(column)
    ]


def _stats(series: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p10": float(clean.quantile(0.10)),
        "p90": float(clean.quantile(0.90)),
    }


class Phase5DExitAIV2TrainingDesignAudit:
    def __init__(self, root: Path | str = ROOT, *, dataset_path: Path | None = None) -> None:
        self.root = Path(root)
        self.dataset_path = self._root_path(dataset_path or DATASET_PATH)
        self.existing_exit_model = self._root_path(EXISTING_EXIT_MODEL)
        self.proposed_model_dir = self._root_path(PROPOSED_MODEL_DIR)

    def build_report(self) -> dict[str, Any]:
        dataset = _read_parquet(self.dataset_path)
        features = _feature_columns(list(dataset.columns))
        missing_rates = self._missing_rates(dataset, features)
        feature_sets = self._feature_set_design(features, missing_rates)
        recommended_feature_set = "feature_set_drop_missing_30pct"
        tasks = self._task_design(dataset)
        imputation = self._imputation_design(dataset, feature_sets[recommended_feature_set]["features"], missing_rates)
        split = self._split_design(dataset)
        leakage = self._leakage_audit(dataset, feature_sets[recommended_feature_set]["features"], split)
        return {
            "metadata": {
                "phase": "5-D",
                "training_design_only": True,
                "model_training_executed": False,
                "full_backtest_executed": False,
                "full_pytest_executed": False,
                "profile_added": False,
                "current_model_overwritten": False,
            },
            "input_paths": {
                "dataset": str(self.dataset_path),
                "existing_exit_model_reference": str(self.existing_exit_model),
                "proposed_exit_ai_v2_model_dir": str(self.proposed_model_dir),
            },
            "dataset_inventory": self._dataset_inventory(dataset, features),
            "task_candidates": tasks,
            "recommended_task": "ranking-style exit_quality_score top decile",
            "feature_set_design": feature_sets,
            "recommended_feature_set": recommended_feature_set,
            "imputation_design": imputation,
            "split_design": split,
            "evaluation_design": self._evaluation_design(),
            "fair_comparison_policy": self._fair_comparison_policy(),
            "leakage_audit": leakage,
            "recommended_next_phase": self._recommended_next_phase(leakage),
        }

    def save_report(self, result: dict[str, Any]) -> Phase5DPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase5DPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        feature_sets = [
            {"name": name, **{k: v for k, v in payload.items() if k != "features"}}
            for name, payload in result["feature_set_design"].items()
        ]
        lines = [
            "# AI Retraining Phase 5-D Exit AI v2 Training Design",
            "",
            "## Scope",
            "",
            "- training design/audit only",
            "- no model retraining, no full backtest, no profile creation",
            "- backtest outcomes are not training labels",
            "",
            "## Dataset",
            "",
            self._table([result["dataset_inventory"]], ["rows", "columns", "feature_count", "train_rows", "validation_rows", "test_rows"]),
            "",
            "## Task Candidates",
            "",
            self._table(result["task_candidates"], ["candidate", "task", "target", "recommended", "reason"]),
            "",
            f"Recommended task: `{result['recommended_task']}`",
            "",
            "## Feature Sets",
            "",
            self._table(feature_sets, ["name", "feature_count", "dropped_features", "expected_risk", "expected_benefit", "recommended"]),
            "",
            f"Recommended feature set: `{result['recommended_feature_set']}`",
            "",
            "## Imputation",
            "",
            self._table([result["imputation_design"]], ["strategy", "leakage_safe_imputation", "missing_indicator_recommended", "columns_to_drop"]),
            "",
            "## Leakage Audit",
            "",
            self._table([result["leakage_audit"]], ["forbidden_columns_found", "label_like_columns_in_features", "split_overlap", "imputer_full_period_fit_risk", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommended Next Phase",
            "",
            f"`{result['recommended_next_phase']}`",
            "",
        ]
        return "\n".join(lines)

    def _dataset_inventory(self, dataset: pd.DataFrame, features: list[str]) -> dict[str, Any]:
        split_counts = dataset["split"].value_counts().to_dict() if "split" in dataset.columns else {}
        return {
            "rows": int(len(dataset)),
            "columns": int(len(dataset.columns)),
            "feature_count": int(len(features)),
            "label_columns_present": sorted(set(dataset.columns) & LABEL_COLUMNS),
            "train_rows": int(split_counts.get("train", 0)),
            "validation_rows": int(split_counts.get("validation", 0)),
            "test_rows": int(split_counts.get("test", 0)),
            "exit_quality_score": _stats(dataset["exit_quality_score"]) if "exit_quality_score" in dataset.columns else {},
            "avoid_loss_5d_positive_rate": float(dataset["avoid_loss_5d"].mean()) if "avoid_loss_5d" in dataset.columns and not dataset.empty else None,
            "miss_profit_5d_positive_rate": float(dataset["miss_profit_5d"].mean()) if "miss_profit_5d" in dataset.columns and not dataset.empty else None,
        }

    def _task_design(self, dataset: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {"candidate": "A", "task": "regression", "target": "exit_quality_score", "recommended": False, "reason": "Good scalar baseline, but direct regression can over-focus on average error."},
            {"candidate": "B", "task": "regression", "target": "future_return_5d", "recommended": False, "reason": "Useful diagnostic target, inverse of exit quality, less directly action-framed."},
            {"candidate": "C", "task": "binary classification", "target": "avoid_loss_5d", "positive_rate": self._rate(dataset, "avoid_loss_5d"), "recommended": False, "reason": "Captures loss avoidance but ignores early-exit missed profit."},
            {"candidate": "D", "task": "binary classification", "target": "miss_profit_5d", "positive_rate": self._rate(dataset, "miss_profit_5d"), "recommended": False, "reason": "Good guardrail for early exits, not sufficient as the primary exit trigger."},
            {"candidate": "E", "task": "multi-output", "target": "avoid_loss_5d + miss_profit_5d", "recommended": False, "reason": "Interpretable but more complex for first trainer prototype."},
            {"candidate": "F", "task": "ranking-style", "target": "exit_quality_score top decile", "recommended": True, "reason": "Matches trading usage: act only on strongest exit-risk signals and measure decile lift/calibration."},
        ]

    def _feature_set_design(self, features: list[str], missing_rates: dict[str, float]) -> dict[str, Any]:
        drop_30 = [column for column in features if missing_rates.get(column, 0.0) >= 0.30]
        technical = [
            column
            for column in features
            if column not in FINANCIAL_FEATURES and not column.endswith("_growth") and column not in {"EPS", "BPS", "EqAR"}
        ]
        no_fin_high_missing = [
            column
            for column in features
            if column not in {"BPS", "FOP_growth", "FSales_growth", "FEPS_growth", "OP_growth"}
        ]
        return {
            "feature_set_all_41": {
                "features": features,
                "feature_count": len(features),
                "dropped_features": [],
                "expected_risk": "higher imputation variance from BPS/growth gaps",
                "expected_benefit": "max information retention for first baseline",
                "recommended": False,
            },
            "feature_set_drop_missing_30pct": {
                "features": [column for column in features if column not in set(drop_30)],
                "feature_count": len(features) - len(drop_30),
                "dropped_features": drop_30,
                "expected_risk": "may remove useful financial regime information",
                "expected_benefit": "cleaner first trainer with lower missingness sensitivity",
                "recommended": True,
            },
            "feature_set_price_technical_only": {
                "features": technical,
                "feature_count": len(technical),
                "dropped_features": [column for column in features if column not in set(technical)],
                "expected_risk": "ignores financial context",
                "expected_benefit": "lowest leakage/missingness risk and easiest production parity",
                "recommended": False,
            },
            "feature_set_no_financial_high_missing": {
                "features": no_fin_high_missing,
                "feature_count": len(no_fin_high_missing),
                "dropped_features": [column for column in features if column not in set(no_fin_high_missing)],
                "expected_risk": "partial financial feature selection may be unstable",
                "expected_benefit": "keeps lower-missing financial signals while dropping worst offenders",
                "recommended": False,
            },
        }

    def _imputation_design(self, dataset: pd.DataFrame, feature_columns: list[str], missing_rates: dict[str, float]) -> dict[str, Any]:
        categorical = [column for column in feature_columns if column in dataset.columns and not pd.api.types.is_numeric_dtype(dataset[column])]
        numeric = [column for column in feature_columns if column in dataset.columns and pd.api.types.is_numeric_dtype(dataset[column])]
        missing_indicator = [column for column in feature_columns if missing_rates.get(column, 0.0) >= 0.05]
        return {
            "strategy": "fit imputer inside each train fold only; apply fitted imputer to validation/test",
            "numeric_strategy": "median",
            "categorical_strategy": "mode",
            "columns_to_impute": sorted([column for column in feature_columns if missing_rates.get(column, 0.0) > 0.0]),
            "numeric_columns_to_impute": sorted([column for column in numeric if missing_rates.get(column, 0.0) > 0.0]),
            "categorical_columns_to_impute": sorted([column for column in categorical if missing_rates.get(column, 0.0) > 0.0]),
            "columns_to_drop": sorted([column for column in feature_columns if missing_rates.get(column, 0.0) >= 0.80]),
            "label_columns_imputed": False,
            "missing_indicator_recommended": bool(missing_indicator),
            "missing_indicator_columns": sorted(missing_indicator),
            "leakage_safe_imputation": True,
            "full_period_fit_forbidden": True,
        }

    def _split_design(self, dataset: pd.DataFrame) -> dict[str, Any]:
        static = []
        if "split" in dataset.columns and "as_of_date" in dataset.columns:
            dates = pd.to_datetime(dataset["as_of_date"], errors="coerce")
            for split_name, group in dataset.assign(_date=dates).groupby("split", dropna=False):
                static.append(
                    {
                        "split": str(split_name),
                        "rows": int(len(group)),
                        "start": str(group["_date"].min().date()) if not group.empty else None,
                        "end": str(group["_date"].max().date()) if not group.empty else None,
                    }
                )
        return {
            "static_split_plan": static,
            "walk_forward_plan": {
                "train_window": "expanding from 2021-06-01 or rolling 24 months",
                "validation_window": "latest 1-3 months before prediction month",
                "test_window": "next prediction month",
                "purge_gap": "at least 20 business days when labels use 20d horizon",
            },
            "retrain_cadence": "monthly for walk-forward prediction artifacts",
            "validation_window": "2024 static validation, plus rolling validation in walk-forward",
            "test_window": "2025-01-06 to 2026-03-30 static test",
            "comparison_policy": "static split for model selection, walk-forward predictions for trading integration comparison",
        }

    def _evaluation_design(self) -> dict[str, list[dict[str, str]]]:
        return {
            "regression": [
                {"metric": "MAE", "purpose": "average target error"},
                {"metric": "RMSE", "purpose": "large error penalty"},
                {"metric": "Spearman rank correlation", "purpose": "exit-risk ordering quality"},
                {"metric": "top decile lift", "purpose": "quality of strongest exit-risk bucket"},
                {"metric": "bottom decile lift", "purpose": "quality of safest hold bucket"},
                {"metric": "calibration by decile", "purpose": "monotonicity and action threshold design"},
            ],
            "binary": [
                {"metric": "AUC", "purpose": "threshold-free discrimination"},
                {"metric": "PR-AUC", "purpose": "positive-class sensitivity"},
                {"metric": "precision@top10%", "purpose": "top action bucket precision"},
                {"metric": "recall@top10%", "purpose": "coverage of true exit/hold events"},
                {"metric": "calibration", "purpose": "probability reliability"},
                {"metric": "confusion matrix by threshold", "purpose": "operating point selection"},
            ],
            "trading_integration": [
                {"metric": "v2_78 baseline vs v2_78 + Exit AI v2", "purpose": "isolate Exit AI change"},
                {"metric": "net_profit / PF / DD / win_rate", "purpose": "headline strategy effect"},
                {"metric": "monthly_win_rate", "purpose": "stability"},
                {"metric": "early_exit_rate", "purpose": "avoid Phase 4 early-sell regression"},
                {"metric": "post_exit_return_5d/10d/20d", "purpose": "missed profit after exits"},
                {"metric": "high_pm_early_exit_rate", "purpose": "guard high-PM positions"},
            ],
        }

    def _fair_comparison_policy(self) -> dict[str, Any]:
        return {
            "existing_model": str(self.existing_exit_model),
            "proposed_model_dir": str(self.proposed_model_dir),
            "current_model_overwrite_forbidden": True,
            "buy_pm_logic": "same v2_78 buy/PM logic",
            "changed_component": "Exit AI only",
            "historical_prediction_policy": "do not regenerate past predictions with current model",
            "walk_forward_policy": "train only on data available before each prediction month",
            "backtest_policy": "not in Phase 5-D; future integration test only after trainer prototype",
        }

    def _leakage_audit(self, dataset: pd.DataFrame, feature_columns: list[str], split: dict[str, Any]) -> dict[str, Any]:
        columns = list(dataset.columns)
        forbidden = [column for column in columns if _is_forbidden(column)]
        label_like = [column for column in feature_columns if _is_label_like(column)]
        selected_count = "selected_count_in_day" in columns
        split_overlap = self._split_overlap(split["static_split_plan"])
        blocking = []
        if forbidden:
            blocking.append("Forbidden backtest/profile columns are present in the dataset.")
        if label_like:
            blocking.append("Label-like columns are present in feature columns.")
        if split_overlap:
            blocking.append("Static split date ranges overlap.")
        if selected_count:
            blocking.append("selected_count_in_day is present.")
        return {
            "forbidden_columns_found": forbidden,
            "label_like_columns_in_features": label_like,
            "future_return_as_feature_found": [column for column in feature_columns if column.startswith("future_return_")],
            "target_or_label_as_feature_found": [column for column in feature_columns if "target" in column.lower() or "label" in column.lower()],
            "selected_count_in_day_found": selected_count,
            "backtest_result_columns_found": [column for column in columns if "backtest" in column.lower()],
            "split_overlap": split_overlap,
            "imputer_full_period_fit_risk": "blocked by design; imputer must fit on train fold only",
            "leakage_risk": "high" if blocking else "low",
            "blocking_issues": blocking,
        }

    def _recommended_next_phase(self, leakage: dict[str, Any]) -> str:
        if leakage["blocking_issues"]:
            return "Retraining deferred"
        return "Phase 5-E Exit AI v2 Trainer Prototype"

    def _missing_rates(self, dataset: pd.DataFrame, features: list[str]) -> dict[str, float]:
        if dataset.empty or not features:
            return {}
        return {column: float(rate) for column, rate in dataset[features].isna().mean().items()}

    def _rate(self, dataset: pd.DataFrame, column: str) -> float | None:
        if column not in dataset.columns or dataset.empty:
            return None
        return float(dataset[column].mean())

    def _split_overlap(self, rows: list[dict[str, Any]]) -> bool:
        ranges = []
        for row in rows:
            if row.get("start") and row.get("end"):
                ranges.append((pd.Timestamp(row["start"]), pd.Timestamp(row["end"]), row["split"]))
        for index, left in enumerate(ranges):
            for right in ranges[index + 1 :]:
                if left[0] <= right[1] and right[0] <= left[1]:
                    return True
        return False

    def _root_path(self, path: Path) -> Path:
        if path.is_absolute():
            try:
                return self.root / path.relative_to(ROOT)
            except ValueError:
                return path
        return self.root / path

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value[:8])
                    if len(row.get(column, [])) > 8:
                        value += ", ..."
                values.append(str(value).replace("\n", " "))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)


def build_report(root: Path | str = ROOT) -> dict[str, Any]:
    return Phase5DExitAIV2TrainingDesignAudit(root).build_report()


def save_report(result: dict[str, Any], root: Path | str = ROOT) -> Phase5DPaths:
    return Phase5DExitAIV2TrainingDesignAudit(root).save_report(result)


def run(root: Path | str = ROOT) -> Phase5DPaths:
    audit = Phase5DExitAIV2TrainingDesignAudit(root)
    return audit.save_report(audit.build_report())
