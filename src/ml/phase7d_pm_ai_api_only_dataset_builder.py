"""Phase 7-D PM AI API-only dataset builder.

The builder creates a retraining-ready PM AI dataset from API-origin features
and future-return labels only. It never uses backtest trades, realized P/L,
portfolio state, selected_count_in_day, candidate-list aggregate features, or
current model directories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase7c_pm_ai_api_only_dataset_design import classify_rebuild_column, is_candidate_list_feature


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase7d_pm_ai_api_only_dataset_builder_2021_to_2026"
BASE_DATASET = Path("data/ml/datasets/ml_dataset.parquet")
WALK_FORWARD_DIR = Path("data/ml/walk_forward_predictions")
SAMPLE_OUTPUT = Path("data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05_sample.parquet")
FULL_OUTPUT = Path("data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05.parquet")

KEY_COLUMNS = ["as_of_date", "code", "split"]
LABEL_COLUMNS = [
    "future_5d_return",
    "future_10d_return",
    "risk_adjusted_future_return",
    "high_conviction_target",
    "avoid_target",
]
STOCK_PREDICTION_COLUMNS = [
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "risk_adjusted_score",
]
SAFE_FEATURE_CANDIDATES = [
    "BPS",
    "EPS",
    "EqAR",
    "FEPS_growth",
    "FOP_growth",
    "FSales_growth",
    "NP_growth",
    "OP_growth",
    "PayoutRatioAnn",
    "Sales_growth",
    "body_ratio",
    "close",
    "close_position",
    "daily_range_ratio",
    "days_to_earnings",
    "gap_up_ratio",
    "is_near_earnings",
    "lower_shadow_ratio",
    "ma10_gap",
    "ma25_gap",
    "ma25_slope",
    "ma5_gap",
    "ma5_slope",
    "ma75_gap",
    "relative_return_10d",
    "relative_return_20d",
    "relative_return_5d",
    "return_10d",
    "return_1d",
    "return_20d",
    "return_3d",
    "return_5d",
    "topix_return_10d",
    "topix_return_20d",
    "topix_return_5d",
    "turnover_ratio_20d",
    "turnover_ratio_5d",
    "turnover_value",
    "upper_shadow_ratio",
    "volume",
    "volume_ratio_20d",
    "volume_ratio_5d",
] + STOCK_PREDICTION_COLUMNS

FORBIDDEN_EXACT_COLUMNS = {
    "selected_count_in_day",
    "candidate_count_in_day",
    "rank_in_day",
    "score_rank_in_day",
    "candidate_rank",
    "score_rank",
    "max_positions_remaining_before",
    "cash_before",
    "cash_after",
    "decision",
    "exit_reason",
    "skip_reason",
}
FORBIDDEN_PREFIXES = ("actual_", "realized_", "profit_", "cash_", "portfolio_", "position_")


@dataclass(frozen=True)
class BuildOptions:
    dry_run: bool = True
    sample_rows: int | None = None
    write_full: bool = False


@dataclass(frozen=True)
class Phase7DPaths:
    markdown: Path
    json: Path
    dataset: Path | None


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


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


def _is_forbidden_column(column: str) -> bool:
    lower = column.lower()
    return (
        column in FORBIDDEN_EXACT_COLUMNS
        or lower.startswith(FORBIDDEN_PREFIXES)
        or is_candidate_list_feature(column)
        or lower.startswith("day_avg_")
        or lower.startswith("day_max_")
        or lower.endswith("_percentile_in_day")
        or lower.endswith("_gap_to_best")
    )


def _is_label_like_feature(column: str) -> bool:
    lower = column.lower()
    return (
        column in LABEL_COLUMNS
        or lower.startswith("future_")
        or lower.endswith("_target")
        or "target" in lower
        or "label" in lower
    )


class Phase7DPMAIAPIOnlyDatasetBuilder:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        base_dataset: Path | None = None,
        sample_output: Path | None = None,
        full_output: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.base_dataset = self._root(base_dataset or BASE_DATASET)
        self.walk_forward_dir = self._root(WALK_FORWARD_DIR)
        self.sample_output = self._root(sample_output or SAMPLE_OUTPUT)
        self.full_output = self._root(full_output or FULL_OUTPUT)

    def build(self, options: BuildOptions | None = None) -> dict[str, Any]:
        options = options or BuildOptions()
        base = _read_parquet(self.base_dataset)
        result = self._build_from_frame(base, options)
        dataset = result.pop("_dataset", pd.DataFrame())
        if result["metadata"]["dataset_write_requested"] and not result["leakage_audit"]["blocking_issues"]:
            output = self.full_output if options.write_full else self.sample_output
            output.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(output, index=False)
            result["metadata"]["dataset_written"] = True
            result["output_paths"]["dataset"] = str(output)
            result["row_counts"]["rows_written"] = int(len(dataset))
        return result

    def save_report(self, result: dict[str, Any]) -> Phase7DPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        dataset_path = Path(result["output_paths"]["dataset"]) if result["output_paths"].get("dataset") else None
        return Phase7DPaths(markdown=md_path, json=json_path, dataset=dataset_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# AI Retraining Phase 7-D PM AI API-Only Dataset Builder",
            "",
            "## Scope",
            "",
            "- API-only dataset builder",
            "- no retraining, no backtest, no profile addition, no current model overwrite, no full pytest",
            "",
            "## Outputs",
            "",
            self._table([result["output_paths"]], ["dataset", "markdown", "json"]),
            "",
            "## Row Counts",
            "",
            self._table(
                [result["row_counts"]],
                ["total_rows", "final_rows", "dropped_rows", "final_feature_count", "final_label_count", "rows_written"],
            ),
            "",
            "## Split Counts",
            "",
            self._table(result["split"]["row_counts"], ["split", "rows", "start", "end"]),
            "",
            "## Leakage Audit",
            "",
            self._table(
                [result["leakage_audit"]],
                [
                    "forbidden_columns_found",
                    "candidate_list_dependent_columns_found",
                    "selected_count_in_day_found",
                    "leakage_risk",
                    "blocking_issues",
                    "ready_for_phase7e",
                ],
            ),
            "",
            "## Label Distribution",
            "",
            self._table(
                [result["label_distribution"]],
                [
                    "future_5d_return_mean",
                    "future_10d_return_mean",
                    "risk_adjusted_future_return_mean",
                    "high_conviction_positive_rate",
                    "avoid_positive_rate",
                ],
            ),
            "",
            "## Feature Missing Rates Top 30",
            "",
            self._table(result["feature_missing_rates_top30"], ["feature", "missing_rate"]),
            "",
            "## Dropped High Missing Features",
            "",
            self._table(result["dropped_high_missing_features"], ["feature", "missing_rate", "reason"]),
            "",
        ]
        return "\n".join(lines)

    def _build_from_frame(self, base: pd.DataFrame, options: BuildOptions) -> dict[str, Any]:
        metadata = {
            "phase": "7-D",
            "api_only_builder": True,
            "model_retraining_executed": False,
            "dataset_write_requested": bool(options.sample_rows or options.write_full),
            "dataset_written": False,
            "dry_run": bool(options.dry_run and not options.sample_rows and not options.write_full),
            "sample_rows": options.sample_rows,
            "write_full": options.write_full,
        }
        output_paths = {
            "dataset": None,
            "markdown": str(self.root / "reports" / "ml" / f"{REPORT_STEM}.md"),
            "json": str(self.root / "reports" / "ml" / f"{REPORT_STEM}.json"),
        }
        if base.empty or not {"date", "code", "close"}.issubset(base.columns):
            return self._empty_report(metadata, output_paths, list(base.columns))

        frame = base.copy()
        frame["as_of_date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["code"] = frame["code"].astype(str)
        frame = frame.dropna(subset=["as_of_date", "code", "close"]).sort_values(["code", "as_of_date"]).reset_index(drop=True)
        frame = self._join_walk_forward_predictions_if_available(frame)
        labels = self._add_labels(frame)
        labels = self._add_split(labels)
        feature_columns = self._feature_columns(labels)
        feature_columns, dropped_high_missing = self._drop_high_missing_features(labels, feature_columns)
        final = self._finalize_dataset(labels, feature_columns)
        if options.sample_rows and len(final) > options.sample_rows:
            final = final.head(options.sample_rows).copy()
        leakage = self._leakage_audit(list(base.columns), feature_columns)
        missing_rates = self._feature_missing_rates(final, feature_columns)
        row_counts = {
            "total_rows": int(len(frame)),
            "final_rows": int(len(final)),
            "dropped_rows": int(len(frame) - len(final)),
            "final_feature_count": int(len(feature_columns)),
            "final_label_count": int(len(LABEL_COLUMNS)),
            "rows_written": 0,
        }
        result = {
            "metadata": metadata,
            "input_paths": {
                "base_dataset": str(self.base_dataset),
                "walk_forward_predictions": str(self.walk_forward_dir),
                "phase7c_design_report": str(self.root / "reports" / "ml" / "phase7c_pm_ai_api_only_dataset_design_2021_to_2026.json"),
            },
            "output_paths": output_paths,
            "data_policy": self._data_policy(),
            "schema": {
                "key_columns": KEY_COLUMNS,
                "feature_columns": feature_columns,
                "label_columns": LABEL_COLUMNS,
            },
            "row_counts": row_counts,
            "date_range": self._date_range(final),
            "split": self._split_report(final),
            "feature_missing_rates_top30": missing_rates[:30],
            "dropped_high_missing_features": dropped_high_missing,
            "label_distribution": self._label_distribution(final),
            "leakage_audit": leakage,
            "recommended_next_phase": "Phase 7-E PM AI API-only Trainer Design" if not leakage["blocking_issues"] else "Retraining deferred",
            "_dataset": final,
        }
        return result

    def _join_walk_forward_predictions_if_available(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing_predictions = [column for column in STOCK_PREDICTION_COLUMNS if column not in frame.columns]
        if not missing_predictions or not self.walk_forward_dir.exists():
            return frame
        files = sorted(self.walk_forward_dir.glob("*.parquet"))
        if not files:
            return frame
        pieces = []
        needed = {"code", *STOCK_PREDICTION_COLUMNS}
        for path in files:
            piece = _read_parquet(path)
            date_column = "date" if "date" in piece.columns else "prediction_date" if "prediction_date" in piece.columns else None
            if date_column is None or not needed.issubset(piece.columns):
                continue
            use_cols = [date_column, "code", *STOCK_PREDICTION_COLUMNS]
            pieces.append(piece[use_cols].rename(columns={date_column: "as_of_date"}))
        if not pieces:
            return frame
        predictions = pd.concat(pieces, ignore_index=True)
        predictions["as_of_date"] = pd.to_datetime(predictions["as_of_date"], errors="coerce")
        predictions["code"] = predictions["code"].astype(str)
        return frame.merge(predictions.drop_duplicates(["as_of_date", "code"]), on=["as_of_date", "code"], how="left")

    def _add_labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        close = pd.to_numeric(result["close"], errors="coerce")
        grouped_close = result.assign(_close=close).groupby("code", sort=False)["_close"]
        for horizon in [5, 10]:
            column = f"future_{horizon}d_return"
            if column not in result.columns:
                result[column] = grouped_close.shift(-horizon) / close - 1.0
            result[column] = pd.to_numeric(result[column], errors="coerce")
        future_returns_10d = pd.concat(
            [(grouped_close.shift(-step) / close - 1.0).rename(str(step)) for step in range(1, 11)],
            axis=1,
        )
        future_max_drawdown_10d = future_returns_10d.min(axis=1, skipna=False)
        result["risk_adjusted_future_return"] = result["future_10d_return"] - future_max_drawdown_10d.clip(upper=0).abs()
        result = self._add_split(result)
        train = result.loc[result["split"] == "train", "future_10d_return"].dropna()
        top_threshold = float(train.quantile(0.90)) if not train.empty else None
        bottom_threshold = float(train.quantile(0.10)) if not train.empty else None
        result["high_conviction_target"] = result["future_10d_return"] >= top_threshold if top_threshold is not None else False
        avoid_by_10d = result["future_10d_return"] <= bottom_threshold if bottom_threshold is not None else False
        result["avoid_target"] = avoid_by_10d | (result["future_5d_return"] <= -0.03)
        return result

    def _add_split(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        dates = pd.to_datetime(result["as_of_date"], errors="coerce")
        result["split"] = "test"
        result.loc[(dates >= pd.Timestamp("2021-06-01")) & (dates <= pd.Timestamp("2023-12-31")), "split"] = "train"
        result.loc[(dates >= pd.Timestamp("2024-01-01")) & (dates <= pd.Timestamp("2024-12-31")), "split"] = "validation"
        result.loc[(dates >= pd.Timestamp("2025-01-01")) & (dates <= pd.Timestamp("2026-05-31")), "split"] = "test"
        result["as_of_date"] = dates
        return result

    def _feature_columns(self, frame: pd.DataFrame) -> list[str]:
        features = []
        for column in SAFE_FEATURE_CANDIDATES:
            if column not in frame.columns:
                continue
            if _is_forbidden_column(column) or _is_label_like_feature(column):
                continue
            if classify_rebuild_column(column).startswith("backtest_"):
                continue
            features.append(column)
        return features

    def _finalize_dataset(self, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        required = ["as_of_date", "code", "split", *LABEL_COLUMNS]
        dataset = frame.dropna(subset=required).copy()
        keep = KEY_COLUMNS + feature_columns + LABEL_COLUMNS
        dataset = dataset[[column for column in keep if column in dataset.columns]].copy()
        dataset["as_of_date"] = pd.to_datetime(dataset["as_of_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return dataset

    def _leakage_audit(self, base_columns: list[str], feature_columns: list[str]) -> dict[str, Any]:
        forbidden_in_features = [column for column in feature_columns if _is_forbidden_column(column)]
        candidate_in_features = [column for column in feature_columns if is_candidate_list_feature(column)]
        selected_found = "selected_count_in_day" in feature_columns or "selected_count_in_day" in base_columns
        label_like = [column for column in feature_columns if _is_label_like_feature(column)]
        blocking = []
        if forbidden_in_features:
            blocking.append("Forbidden columns remain in feature schema.")
        if candidate_in_features:
            blocking.append("Candidate-list dependent columns remain in feature schema.")
        if selected_found:
            blocking.append("selected_count_in_day found.")
        if label_like:
            blocking.append("Future/target/label-like columns remain in feature schema.")
        return {
            "forbidden_columns_found": sorted(set(forbidden_in_features)),
            "candidate_list_dependent_columns_found": sorted(set(candidate_in_features)),
            "selected_count_in_day_found": bool(selected_found),
            "label_like_feature_columns_found": sorted(set(label_like)),
            "leakage_risk": "high" if blocking else "low",
            "blocking_issues": blocking,
            "ready_for_phase7e": not blocking,
        }

    def _feature_missing_rates(self, dataset: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
        if dataset.empty or not feature_columns:
            return []
        rates = dataset[feature_columns].isna().mean().sort_values(ascending=False)
        return [{"feature": column, "missing_rate": float(rate)} for column, rate in rates.items()]

    def _drop_high_missing_features(self, frame: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
        if frame.empty or not feature_columns:
            return feature_columns, []
        label_ready = frame.dropna(subset=LABEL_COLUMNS)
        if label_ready.empty:
            return feature_columns, []
        rates = label_ready[feature_columns].isna().mean()
        dropped = [
            {
                "feature": column,
                "missing_rate": float(rate),
                "reason": "missing_rate >= 0.80",
            }
            for column, rate in rates.items()
            if float(rate) >= 0.80
        ]
        dropped_names = {row["feature"] for row in dropped}
        retained = [column for column in feature_columns if column not in dropped_names]
        return retained, sorted(dropped, key=lambda row: row["missing_rate"], reverse=True)

    def _label_distribution(self, dataset: pd.DataFrame) -> dict[str, Any]:
        def mean(column: str) -> float | None:
            if column not in dataset.columns or dataset.empty:
                return None
            return float(pd.to_numeric(dataset[column], errors="coerce").mean())

        return {
            "future_5d_return_mean": mean("future_5d_return"),
            "future_10d_return_mean": mean("future_10d_return"),
            "risk_adjusted_future_return_mean": mean("risk_adjusted_future_return"),
            "high_conviction_positive_rate": mean("high_conviction_target"),
            "avoid_positive_rate": mean("avoid_target"),
            "label_rows": int(len(dataset)),
        }

    def _split_report(self, dataset: pd.DataFrame) -> dict[str, Any]:
        rows = []
        for split, group in dataset.groupby("split", dropna=False):
            dates = pd.to_datetime(group["as_of_date"], errors="coerce")
            rows.append(
                {
                    "split": str(split),
                    "rows": int(len(group)),
                    "start": str(dates.min().date()) if not dates.empty else None,
                    "end": str(dates.max().date()) if not dates.empty else None,
                }
            )
        return {"row_counts": rows}

    def _date_range(self, dataset: pd.DataFrame) -> dict[str, str | None]:
        if dataset.empty:
            return {"from": None, "to": None}
        dates = pd.to_datetime(dataset["as_of_date"], errors="coerce").dropna()
        return {"from": str(dates.min().date()), "to": str(dates.max().date())}

    def _empty_report(self, metadata: dict[str, Any], output_paths: dict[str, Any], columns: list[str]) -> dict[str, Any]:
        return {
            "metadata": metadata,
            "input_paths": {"base_dataset": str(self.base_dataset), "walk_forward_predictions": str(self.walk_forward_dir)},
            "output_paths": output_paths,
            "data_policy": self._data_policy(),
            "schema": {"key_columns": KEY_COLUMNS, "feature_columns": [], "label_columns": LABEL_COLUMNS},
            "row_counts": {"total_rows": 0, "final_rows": 0, "dropped_rows": 0, "final_feature_count": 0, "final_label_count": len(LABEL_COLUMNS), "rows_written": 0},
            "date_range": {"from": None, "to": None},
            "split": {"row_counts": []},
            "feature_missing_rates_top30": [],
            "dropped_high_missing_features": [],
            "label_distribution": {},
            "leakage_audit": {
                "forbidden_columns_found": [],
                "candidate_list_dependent_columns_found": [],
                "selected_count_in_day_found": "selected_count_in_day" in columns,
                "label_like_feature_columns_found": [],
                "leakage_risk": "high",
                "blocking_issues": ["Base dataset is missing or lacks required date/code/close columns."],
                "ready_for_phase7e": False,
            },
            "recommended_next_phase": "Retraining deferred",
            "_dataset": pd.DataFrame(),
        }

    def _data_policy(self) -> dict[str, list[str]]:
        return {
            "allowed_sources": [
                "API-origin price, volume, technical, financial, and market features",
                "Stock Selection AI walk-forward predictions only",
                "future return labels mechanically generated from API-origin prices",
            ],
            "forbidden_sources": [
                "trades.csv / realized profit / win-loss / portfolio history",
                "selected_count_in_day",
                "candidate_count_in_day / rank_in_day / score_rank_in_day",
                "day_avg_* / day_max_* / *_percentile_in_day / *_gap_to_best",
                "max_positions_remaining_before / cash_before / cash_after",
                "decision / exit_reason / skip_reason",
            ],
        }

    def _root(self, path: Path) -> Path:
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
            lines.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)


def build_report(root: Path | str = ROOT, options: BuildOptions | None = None) -> dict[str, Any]:
    return Phase7DPMAIAPIOnlyDatasetBuilder(root).build(options)


def run(root: Path | str = ROOT, options: BuildOptions | None = None) -> Phase7DPaths:
    builder = Phase7DPMAIAPIOnlyDatasetBuilder(root)
    result = builder.build(options)
    return builder.save_report(result)
