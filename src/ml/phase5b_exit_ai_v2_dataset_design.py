"""Phase 5-B Exit AI v2 API-only dataset design audit.

This module does not build a training dataset. It designs and audits the schema
for a future Exit AI v2 dataset whose features and labels are derived only from
API-origin market/financial/price data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase5b_exit_ai_v2_dataset_design_2021-06_to_2026-05"

BASE_DATASET = ROOT / "data" / "ml" / "datasets" / "ml_dataset.parquet"
EXISTING_EXIT_DATASET = ROOT / "data" / "ml" / "exit_datasets" / "exit_dataset_v2_66_2023-01_to_2026-05.parquet"
RAW_PRICE_DIR = ROOT / "data" / "raw"

FORBIDDEN_COLUMNS = {
    "trade_id",
    "actual_exit_date",
    "actual_sell_price",
    "realized_profit",
    "realized_return",
    "profit",
    "net_profit",
    "win",
    "loss",
    "result",
    "holding_days",
    "remaining_days_to_actual_exit",
    "exit_reason",
    "profile_id",
    "portfolio_cash",
    "cash",
    "market_value",
    "total_assets",
    "selected_count_in_day",
    "backtest_profit",
    "backtest_result",
    "actual_net_profit",
    "actual_holding_days",
    "decision",
    "skip_reason",
}

BASE_REQUIRED_COLUMNS = {"date", "code", "close"}
LABEL_COLUMNS = [
    "future_return_3d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "should_hold_5d",
    "should_exit_5d",
    "avoid_loss_5d",
    "miss_profit_5d",
    "exit_quality_score",
    "future_max_drawdown_5d",
    "future_max_drawdown_10d",
    "future_max_return_5d",
    "future_max_return_10d",
]


@dataclass(frozen=True)
class Phase5BPaths:
    markdown: Path
    json: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _date_range(frame: pd.DataFrame) -> dict[str, str | None]:
    for column in ["date", "signal_date", "as_of_date"]:
        if column in frame.columns and not frame.empty:
            values = frame[column].dropna().astype(str)
            if not values.empty:
                return {"start": str(values.min()), "end": str(values.max())}
    return {"start": None, "end": None}


def _safe_feature_columns(columns: list[str]) -> list[str]:
    blocked = set(FORBIDDEN_COLUMNS) | set(LABEL_COLUMNS)
    return [column for column in columns if column not in blocked]


class Phase5BExitAIV2DatasetDesignAudit:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        base_dataset: Path | None = None,
        existing_exit_dataset: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.base_dataset = self._root_path(base_dataset or BASE_DATASET)
        self.existing_exit_dataset = self._root_path(existing_exit_dataset or EXISTING_EXIT_DATASET)
        self.raw_price_dir = self._root_path(RAW_PRICE_DIR)

    def build_report(self) -> dict[str, Any]:
        base = _read_parquet(self.base_dataset)
        existing = _read_parquet(self.existing_exit_dataset)
        schema = self._schema_design(base)
        labels = self._label_design()
        feasibility = self._sample_feasibility(base)
        split = self._split_design()
        leakage = self._leakage_audit(schema, labels)
        existing_comparison = self._existing_dataset_comparison(existing, leakage)
        evaluation = self._evaluation_design()
        return {
            "metadata": {
                "phase": "5-B",
                "audit_only": True,
                "dataset_generated": False,
                "model_retraining_executed": False,
                "full_backtest_executed": False,
                "full_pytest_executed": False,
                "api_only_design": True,
            },
            "input_paths": {
                "base_stock_selection_dataset": str(self.base_dataset),
                "existing_exit_dataset_reference_only": str(self.existing_exit_dataset),
                "raw_price_cache": str(self.raw_price_dir / "prices_YYYY-MM-DD.json"),
            },
            "data_policy": self._data_policy(),
            "dataset_schema_design": schema,
            "label_design": labels,
            "sample_generation_feasibility": feasibility,
            "time_split_design": split,
            "leakage_audit": leakage,
            "existing_exit_dataset_comparison": existing_comparison,
            "evaluation_design": evaluation,
            "recommended_next_phase": self._recommended_next_phase(leakage, feasibility),
        }

    def save_report(self, result: dict[str, Any]) -> Phase5BPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase5BPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# AI Retraining Phase 5-B Exit AI v2 API-Only Dataset Design",
            "",
            "## Scope",
            "",
            "- audit/design only",
            "- no dataset generation, no model retraining, no profile creation, no full backtest",
            "- existing Exit dataset is reference-only and is not approved for retraining",
            "- backtest outcomes are not teacher labels",
            "",
            "## Schema",
            "",
            self._table(result["dataset_schema_design"]["schema_fields"], ["column", "role", "source", "allowed"]),
            "",
            "## Label Design",
            "",
            self._table(result["label_design"]["label_candidates"], ["label", "task", "definition", "recommended"]),
            "",
            f"Recommended label: `{result['label_design']['recommended_label']}`",
            "",
            "## Sample Feasibility",
            "",
            self._table([result["sample_generation_feasibility"]], ["candidate_rows_before_label_horizon_filtering", "rows_after_3d_label_available", "rows_after_5d_label_available", "rows_after_10d_label_available", "rows_after_20d_label_available", "unique_codes", "date_range", "missing_price_count", "missing_feature_count", "rows_excluded_due_to_horizon_shortage", "rows_excluded_due_to_missing_features"]),
            "",
            "## Leakage Audit",
            "",
            self._table([result["leakage_audit"]], ["forbidden_columns_found", "safe_feature_columns_count", "label_columns", "leakage_risk", "blocking_issues"]),
            "",
            "## Existing Dataset Comparison",
            "",
            self._table([result["existing_exit_dataset_comparison"]], ["existing_exit_dataset_rows", "existing_forbidden_columns_found", "existing_dataset_retraining_allowed", "new_dataset_design_retraining_allowed"]),
            "",
            "## Evaluation Design",
            "",
            self._table(result["evaluation_design"]["model_level_metrics"], ["metric", "purpose"]),
            "",
            self._table(result["evaluation_design"]["backtest_level_metrics"], ["metric", "purpose"]),
            "",
            "## Recommended Next Phase",
            "",
            f"`{result['recommended_next_phase']}`",
            "",
        ]
        return "\n".join(lines)

    def _root_path(self, path: Path) -> Path:
        if path.is_absolute():
            try:
                return self.root / path.relative_to(ROOT)
            except ValueError:
                return path
        return self.root / path

    def _data_policy(self) -> dict[str, Any]:
        return {
            "allowed_sources": [
                "API-origin price series",
                "API-origin financial data",
                "API-derived features",
                "mechanical future-return labels from API price series",
                "walk-forward artifacts only if produced without future feature leakage",
            ],
            "forbidden_sources": [
                "trades.csv as teacher labels",
                "backtest_summary.json as teacher labels",
                "summary.csv / portfolio history as teacher labels",
                "realized P/L or win/loss as teacher labels",
                "v2_xx profile trading outcomes as labels",
                "selected-only backtest universe as training universe",
                "current model regenerated historical predictions",
                "selected_count_in_day",
            ],
        }

    def _schema_design(self, base: pd.DataFrame) -> dict[str, Any]:
        columns = list(base.columns)
        safe_features = _safe_feature_columns(columns)
        fields = [
            {"column": "code", "role": "key", "source": "API-derived base dataset", "allowed": True},
            {"column": "as_of_date", "role": "time key", "source": "API-derived base date", "allowed": True},
        ]
        for column in safe_features:
            if column in {"date", "code"}:
                continue
            fields.append({"column": column, "role": "feature", "source": "API-derived feature dataset", "allowed": True})
        for column in LABEL_COLUMNS:
            fields.append({"column": column, "role": "label", "source": "mechanical future price path", "allowed": True})
        forbidden_present = sorted(set(columns) & FORBIDDEN_COLUMNS)
        for column in forbidden_present:
            fields.append({"column": column, "role": "forbidden", "source": "backtest/outcome-like column", "allowed": False})
        return {
            "row_definition": "one code/as_of_date observation from the API-derived all-stock dataset, with features known at as_of_date and labels mechanically computed from future API price path",
            "base_dataset_rows": int(len(base)),
            "base_dataset_columns": columns,
            "schema_fields": fields,
            "safe_feature_columns": safe_features,
            "forbidden_columns_present_in_base": forbidden_present,
            "stock_selection_scores_policy": "exclude by default; may be separately audited only as walk-forward prediction artifacts, never current-model regenerated predictions",
        }

    def _label_design(self) -> dict[str, Any]:
        candidates = [
            {"label": "should_exit_5d", "task": "binary", "definition": "future_return_5d < negative_threshold", "recommended": False},
            {"label": "should_hold_5d", "task": "binary", "definition": "future_return_5d > positive_threshold", "recommended": False},
            {"label": "future_return_5d", "task": "regression", "definition": "close(t+5 business days) / close(t) - 1", "recommended": False},
            {"label": "avoid_loss_5d + miss_profit_5d", "task": "multi-label", "definition": "avoid_loss_5d if future loss/drawdown exceeds threshold; miss_profit_5d if future return/upside exceeds threshold", "recommended": False},
            {"label": "exit_quality_score", "task": "ranking/regression", "definition": "risk-adjusted score combining avoid-loss benefit and missed-profit penalty from API price path", "recommended": True},
        ]
        return {
            "label_candidates": candidates,
            "basic_labels": LABEL_COLUMNS,
            "recommended_label": "exit_quality_score",
            "recommended_reason": "A scalar/ranking label can separate beneficial exits from early exits while retaining avoid_loss_5d and miss_profit_5d as interpretable components.",
            "threshold_defaults": {
                "should_hold_5d": "future_return_5d > +2%",
                "should_exit_5d": "future_return_5d < -2%",
                "avoid_loss_5d": "future_return_5d <= -3% or future_max_drawdown_5d <= -3%",
                "miss_profit_5d": "future_return_5d >= +3% or future_max_return_5d >= +5%",
            },
        }

    def _sample_feasibility(self, base: pd.DataFrame) -> dict[str, Any]:
        if base.empty:
            return {
                "candidate_rows_before_label_horizon_filtering": 0,
                "rows_after_3d_label_available": 0,
                "rows_after_5d_label_available": 0,
                "rows_after_10d_label_available": 0,
                "rows_after_20d_label_available": 0,
                "unique_codes": 0,
                "date_range": "",
                "missing_price_count": 0,
                "missing_feature_count": 0,
                "rows_excluded_due_to_horizon_shortage": 0,
                "rows_excluded_due_to_missing_features": 0,
            }
        frame = base.copy()
        date_col = "date" if "date" in frame.columns else "as_of_date"
        frame[date_col] = frame[date_col].astype(str)
        dates = sorted(frame[date_col].dropna().unique())
        date_rank = {date: index for index, date in enumerate(dates)}
        ranks = frame[date_col].map(date_rank)
        total_dates = len(dates)
        counts = {}
        for horizon in [3, 5, 10, 20]:
            counts[horizon] = int((ranks <= total_dates - horizon - 1).sum()) if total_dates > horizon else 0
        safe_features = _safe_feature_columns(list(frame.columns))
        feature_cols = [column for column in safe_features if column not in {"date", "code", "as_of_date"}]
        missing_feature_rows = int(frame[feature_cols].isna().any(axis=1).sum()) if feature_cols else 0
        missing_price = int(frame["close"].isna().sum()) if "close" in frame.columns else int(len(frame))
        date_range = _date_range(frame)
        return {
            "candidate_rows_before_label_horizon_filtering": int(len(frame)),
            "rows_after_3d_label_available": counts[3],
            "rows_after_5d_label_available": counts[5],
            "rows_after_10d_label_available": counts[10],
            "rows_after_20d_label_available": counts[20],
            "unique_codes": int(frame["code"].nunique()) if "code" in frame.columns else 0,
            "date_range": f"{date_range['start']} to {date_range['end']}",
            "missing_price_count": missing_price,
            "missing_feature_count": missing_feature_rows,
            "rows_excluded_due_to_horizon_shortage": int(len(frame) - counts[20]),
            "rows_excluded_due_to_missing_features": missing_feature_rows,
        }

    def _split_design(self) -> dict[str, Any]:
        return {
            "fixed_split": {
                "train": "2021-06-01 to 2023-12-31",
                "validation": "2024-01-01 to 2024-12-31",
                "test": "2025-01-01 to 2026-05-31 minus horizon tail",
            },
            "walk_forward": {
                "train_window": "expanding or 24-month rolling",
                "validation_window": "1 to 3 months before prediction month",
                "prediction_window": "1 month",
                "retrain_cadence": "monthly",
            },
            "fair_comparison": [
                "Do not regenerate historical predictions with current model.",
                "Compare v2_78 against v2_78_exit_v2 with only Exit AI swapped.",
                "Ensure every feature is known as of as_of_date.",
            ],
        }

    def _leakage_audit(self, schema: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
        base_columns = set(schema.get("base_dataset_columns", []))
        forbidden = sorted(base_columns & FORBIDDEN_COLUMNS)
        label_columns = labels.get("basic_labels", [])
        blocking = []
        if forbidden:
            blocking.append("Base API dataset contains forbidden columns; remove before any builder implementation.")
        if "selected_count_in_day" in base_columns:
            blocking.append("selected_count_in_day is forbidden.")
        return {
            "forbidden_columns_found": forbidden,
            "safe_feature_columns_count": len(schema.get("safe_feature_columns", [])),
            "label_columns": label_columns,
            "leakage_risk": "low" if not blocking else "high",
            "blocking_issues": blocking,
        }

    def _existing_dataset_comparison(self, existing: pd.DataFrame, leakage: dict[str, Any]) -> dict[str, Any]:
        columns = list(existing.columns)
        forbidden = sorted(set(columns) & FORBIDDEN_COLUMNS)
        return {
            "existing_exit_dataset_rows": int(len(existing)),
            "existing_exit_dataset_columns": columns,
            "existing_forbidden_columns_found": forbidden,
            "existing_dataset_retraining_allowed": False,
            "existing_dataset_reason": "Existing v2_66 exit dataset includes backtest/trade path columns and is audit/reference only.",
            "new_dataset_design_retraining_allowed": not bool(leakage.get("blocking_issues")),
            "new_dataset_reason": "New design uses API-derived all-stock rows and mechanical future price labels, with forbidden columns blocked.",
        }

    def _evaluation_design(self) -> dict[str, Any]:
        return {
            "model_level_metrics": [
                {"metric": "AUC / PR-AUC", "purpose": "binary avoid_loss_5d and miss_profit_5d quality"},
                {"metric": "MAE", "purpose": "future_return / exit_quality_score regression error"},
                {"metric": "rank correlation", "purpose": "ranking quality for exit timing"},
                {"metric": "calibration", "purpose": "probability reliability"},
                {"metric": "top decile lift", "purpose": "actionable high-confidence exit signals"},
                {"metric": "false early-exit risk", "purpose": "penalize exits followed by positive future returns"},
            ],
            "backtest_level_metrics": [
                {"metric": "net_profit / PF / DD / win_rate", "purpose": "v2_78 vs v2_78_exit_v2 headline comparison"},
                {"metric": "monthly_win_rate", "purpose": "stability check"},
                {"metric": "early_exit_rate", "purpose": "avoid repeating Phase 4 early-sell issue"},
                {"metric": "post_exit_return_5d/10d/20d", "purpose": "measure missed profit after exits"},
                {"metric": "high PM early exit rate", "purpose": "verify high-PM positions are not sold too early"},
            ],
        }

    def _recommended_next_phase(self, leakage: dict[str, Any], feasibility: dict[str, Any]) -> str:
        if leakage.get("blocking_issues"):
            return "Retraining deferred"
        if feasibility.get("rows_after_20d_label_available", 0) <= 0:
            return "Phase 5-C Exit AI v2 Label Experiment"
        return "Phase 5-C Exit AI v2 Dataset Builder"

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
    return Phase5BExitAIV2DatasetDesignAudit(root).build_report()


def save_report(result: dict[str, Any], root: Path | str = ROOT) -> Phase5BPaths:
    return Phase5BExitAIV2DatasetDesignAudit(root).save_report(result)


def run(root: Path | str = ROOT) -> Phase5BPaths:
    audit = Phase5BExitAIV2DatasetDesignAudit(root)
    return audit.save_report(audit.build_report())

