"""Phase 5-C Exit AI v2 API-only dataset builder.

This builder creates labels only from API-derived price paths. It does not use
backtest trades, realized P/L, profile outcomes, or current-model regenerated
historical predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase5c_exit_ai_v2_dataset_builder_2021-06_to_2026-05"
BASE_DATASET = ROOT / "data" / "ml" / "datasets" / "ml_dataset.parquet"
OUTPUT_DATASET = ROOT / "data" / "ml" / "exit_ai_v2" / "exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet"

KEY_COLUMNS = ["code", "as_of_date"]
LABEL_COLUMNS = [
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
]

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

FUTURE_LABEL_SOURCE_COLUMNS = {
    "future_5d_return",
    "future_10d_return",
    "upside_10d",
    "bad_entry_10d",
    "future_max_return_10d",
    "future_max_return_20d",
    "future_swing_success_20d",
}

HORIZONS = [3, 5, 10, 20]


@dataclass(frozen=True)
class Phase5CPaths:
    markdown: Path
    json: Path
    dataset: Path | None


@dataclass(frozen=True)
class BuildOptions:
    dry_run: bool = True
    sample_rows: int | None = None
    write_full: bool = False


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _is_forbidden_column(column: str) -> bool:
    lower = column.lower()
    return column in FORBIDDEN_COLUMNS or "backtest" in lower or "v2_" in lower


def _is_label_like_feature(column: str) -> bool:
    lower = column.lower()
    if column in FUTURE_LABEL_SOURCE_COLUMNS or column in LABEL_COLUMNS:
        return True
    prefixes = ("future_return_", "future_max_", "avoid_loss_", "miss_profit_")
    return lower.startswith(prefixes) or "exit_quality_score" in lower or "target" in lower or "label" in lower


def _feature_columns(columns: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    features: list[str] = []
    dropped: list[dict[str, str]] = []
    for column in columns:
        if column in {"date", "as_of_date", "code"}:
            continue
        if _is_forbidden_column(column):
            dropped.append({"column": column, "reason": "forbidden backtest/profile/portfolio column"})
            continue
        if _is_label_like_feature(column):
            dropped.append({"column": column, "reason": "future/target/label-like column excluded from features"})
            continue
        features.append(column)
    return features, dropped


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


class Phase5CExitAIV2DatasetBuilder:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        base_dataset: Path | None = None,
        output_dataset: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.base_dataset = self._root_path(base_dataset or BASE_DATASET)
        self.output_dataset = self._root_path(output_dataset or OUTPUT_DATASET)

    def build(self, options: BuildOptions | None = None) -> dict[str, Any]:
        options = options or BuildOptions()
        base = _read_parquet(self.base_dataset)
        result = self._build_from_frame(base, options)
        if result["metadata"]["dataset_write_requested"] and not result["leakage_audit"]["blocking_issues"]:
            dataset = result.pop("_dataset")
            self.output_dataset.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(self.output_dataset, index=False)
            result["output_paths"]["dataset"] = str(self.output_dataset)
            result["metadata"]["dataset_written"] = True
            result["metadata"]["dataset_rows_written"] = int(len(dataset))
        else:
            result.pop("_dataset", None)
        return result

    def save_report(self, result: dict[str, Any]) -> Phase5CPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        dataset_path = Path(result["output_paths"]["dataset"]) if result["output_paths"].get("dataset") else None
        return Phase5CPaths(markdown=md_path, json=json_path, dataset=dataset_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# AI Retraining Phase 5-C Exit AI v2 API-Only Dataset Builder",
            "",
            "## Scope",
            "",
            "- API-only dataset builder",
            "- no model retraining, no profile creation, no full backtest",
            "- backtest outcomes are not teacher labels",
            "",
            "## Outputs",
            "",
            self._table([result["output_paths"]], ["dataset", "markdown", "json"]),
            "",
            "## Row Counts",
            "",
            self._table([result["row_counts"]], ["rows_before_filter", "rows_after_horizon_filter", "rows_after_missing_feature_filter", "horizon_missing_count", "label_missing_count", "feature_missing_count", "final_feature_count"]),
            "",
            "## Leakage Audit",
            "",
            self._table([result["leakage_audit"]], ["forbidden_columns_found", "label_like_feature_columns_found", "leakage_risk", "blocking_issues"]),
            "",
            "## Label Distribution",
            "",
            self._table([result["label_distribution"]["overall"]], ["future_return_5d_mean", "future_return_5d_median", "future_return_5d_p10", "future_return_5d_p90", "avoid_loss_5d_positive_rate", "miss_profit_5d_positive_rate", "exit_quality_score_mean", "exit_quality_score_median"]),
            "",
            "## Split Counts",
            "",
            self._table(result["split"]["row_counts"], ["split", "rows", "start", "end"]),
            "",
        ]
        return "\n".join(lines)

    def _build_from_frame(self, base: pd.DataFrame, options: BuildOptions) -> dict[str, Any]:
        metadata = {
            "phase": "5-C",
            "api_only_builder": True,
            "model_retraining_executed": False,
            "full_backtest_executed": False,
            "full_pytest_executed": False,
            "dataset_write_requested": bool(options.write_full or options.sample_rows),
            "dataset_written": False,
            "dry_run": bool(options.dry_run and not options.write_full and not options.sample_rows),
            "sample_rows": options.sample_rows,
            "write_full": options.write_full,
        }
        output_paths = {
            "dataset": None,
            "markdown": str(self.root / "reports" / "ml" / f"{REPORT_STEM}.md"),
            "json": str(self.root / "reports" / "ml" / f"{REPORT_STEM}.json"),
        }
        if base.empty or not {"date", "code", "close"}.issubset(base.columns):
            empty = self._empty_report(metadata, output_paths, list(base.columns))
            empty["_dataset"] = pd.DataFrame()
            return empty

        frame = base.copy()
        frame["as_of_date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["as_of_date", "code", "close"]).sort_values(["code", "as_of_date"]).reset_index(drop=True)
        feature_candidates, dropped = _feature_columns(list(base.columns))
        labels = self._add_labels(frame)
        dataset, missing_report = self._apply_missing_policy(labels, feature_candidates)
        dataset = self._add_split(dataset)
        leakage = self._leakage_audit(list(base.columns), missing_report["feature_columns"])
        distribution = self._label_distribution(dataset)
        split = self._split_report(dataset)
        if options.sample_rows and len(dataset) > options.sample_rows:
            dataset = dataset.head(options.sample_rows).copy()
        selected_columns = KEY_COLUMNS + [column for column in dataset.columns if column not in set(KEY_COLUMNS)]
        dataset = dataset[selected_columns]

        row_counts = {
            "rows_before_filter": int(len(frame)),
            "rows_after_horizon_filter": int(missing_report["rows_after_horizon_filter"]),
            "rows_after_missing_feature_filter": int(missing_report["rows_after_missing_feature_filter"]),
            "horizon_missing_count": int(missing_report["horizon_missing_count"]),
            "label_missing_count": int(missing_report["label_missing_count"]),
            "feature_missing_count": int(missing_report["feature_missing_count"]),
            "final_feature_count": int(missing_report["final_feature_count"]),
        }
        result = {
            "metadata": metadata,
            "input_paths": {"base_dataset": str(self.base_dataset)},
            "output_paths": output_paths,
            "data_policy": self._data_policy(),
            "schema": {
                "key_columns": KEY_COLUMNS,
                "label_columns": LABEL_COLUMNS,
                "feature_columns": missing_report["feature_columns"],
            },
            "row_counts": row_counts,
            "missing_policy": missing_report,
            "leakage_audit": leakage,
            "label_distribution": distribution,
            "split": split,
            "dropped_feature_columns": dropped + missing_report["dropped_high_missing_features"],
            "recommended_next_phase": "Phase 5-D Exit AI v2 Dataset Audit" if not leakage["blocking_issues"] else "Retraining deferred",
            "_dataset": dataset,
        }
        return result

    def _add_labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        close = pd.to_numeric(result["close"], errors="coerce")
        grouped_close = result.assign(_close=close).groupby("code", sort=False)["_close"]
        for horizon in HORIZONS:
            future_close = grouped_close.shift(-horizon)
            result[f"future_return_{horizon}d"] = future_close / close - 1.0
        for horizon in [5, 10]:
            future_returns = pd.concat(
                [(grouped_close.shift(-step) / close - 1.0).rename(str(step)) for step in range(1, horizon + 1)],
                axis=1,
            )
            result[f"future_max_drawdown_{horizon}d"] = future_returns.min(axis=1, skipna=False)
            result[f"future_max_return_{horizon}d"] = future_returns.max(axis=1, skipna=False)
        result["avoid_loss_5d"] = result["future_return_5d"] <= -0.03
        result["miss_profit_5d"] = result["future_return_5d"] >= 0.03
        result["exit_quality_score"] = -result["future_return_5d"]
        drawdown_penalty = result["future_max_drawdown_5d"].clip(upper=0).abs()
        result["exit_quality_score_risk_adjusted"] = -result["future_return_5d"] + drawdown_penalty
        return result

    def _apply_missing_policy(self, frame: pd.DataFrame, feature_candidates: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
        required_labels = [column for column in LABEL_COLUMNS if column in frame.columns]
        horizon_missing_mask = frame[[f"future_return_{horizon}d" for horizon in HORIZONS]].isna().any(axis=1)
        label_missing_mask = frame[required_labels].isna().any(axis=1)
        after_horizon = frame.loc[~horizon_missing_mask & ~label_missing_mask].copy()
        feature_missing_rates = after_horizon[feature_candidates].isna().mean().sort_values(ascending=False) if feature_candidates else pd.Series(dtype=float)
        high_missing = [column for column, rate in feature_missing_rates.items() if rate >= 0.80]
        retained_features = [column for column in feature_candidates if column not in set(high_missing)]
        feature_missing_mask = after_horizon[retained_features].isna().any(axis=1) if retained_features else pd.Series(False, index=after_horizon.index)
        keep_columns = KEY_COLUMNS + retained_features + LABEL_COLUMNS + ["split"]
        dataset = after_horizon[[column for column in keep_columns if column in after_horizon.columns]].copy()
        top_missing = [
            {"column": column, "missing_rate": float(rate)}
            for column, rate in feature_missing_rates.head(50).items()
        ]
        return dataset, {
            "rows_after_horizon_filter": int(len(after_horizon)),
            "rows_after_missing_feature_filter": int(len(after_horizon)),
            "horizon_missing_count": int(horizon_missing_mask.sum()),
            "label_missing_count": int(label_missing_mask.sum()),
            "feature_missing_count": int(feature_missing_mask.sum()),
            "final_feature_count": int(len(retained_features)),
            "feature_columns": retained_features,
            "dropped_high_missing_features": [
                {"column": column, "reason": "missing_rate >= 0.80", "missing_rate": float(feature_missing_rates[column])}
                for column in high_missing
            ],
            "missing_rate_by_feature_top50": top_missing,
            "impute_plan": {
                "label_missing": "drop rows",
                "horizon_missing": "drop rows",
                "feature_missing": "do not drop rows in builder; median/mode imputation should be fitted inside training folds",
                "missing_rate_80pct_plus": "exclude feature candidate",
                "missing_rate_30pct_plus": "report as caution",
            },
        }

    def _add_split(self, dataset: pd.DataFrame) -> pd.DataFrame:
        result = dataset.copy()
        dates = pd.to_datetime(result["as_of_date"], errors="coerce")
        result["split"] = "test"
        result.loc[dates <= pd.Timestamp("2023-12-31"), "split"] = "train"
        result.loc[(dates >= pd.Timestamp("2024-01-01")) & (dates <= pd.Timestamp("2024-12-31")), "split"] = "validation"
        result.loc[dates >= pd.Timestamp("2025-01-01"), "split"] = "test"
        result["as_of_date"] = dates.dt.strftime("%Y-%m-%d")
        return result

    def _leakage_audit(self, base_columns: list[str], feature_columns: list[str]) -> dict[str, Any]:
        forbidden = [column for column in base_columns if _is_forbidden_column(column)]
        label_like = [column for column in feature_columns if _is_label_like_feature(column)]
        blocking = []
        if forbidden:
            blocking.append("Forbidden backtest/profile/portfolio columns are present in the base dataset.")
        if label_like:
            blocking.append("Future/target/label-like columns remain in feature columns.")
        return {
            "forbidden_columns_found": forbidden,
            "label_like_feature_columns_found": label_like,
            "leakage_risk": "high" if blocking else "low",
            "blocking_issues": blocking,
        }

    def _label_distribution(self, dataset: pd.DataFrame) -> dict[str, Any]:
        overall = {}
        for column in ["future_return_5d", "future_return_10d", "exit_quality_score"]:
            stats = _stats(dataset[column]) if column in dataset.columns else {"mean": None, "median": None, "p10": None, "p90": None}
            for key, value in stats.items():
                overall[f"{column}_{key}"] = value
        overall["avoid_loss_5d_positive_rate"] = float(dataset["avoid_loss_5d"].mean()) if "avoid_loss_5d" in dataset.columns and not dataset.empty else None
        overall["miss_profit_5d_positive_rate"] = float(dataset["miss_profit_5d"].mean()) if "miss_profit_5d" in dataset.columns and not dataset.empty else None
        by_year = []
        if not dataset.empty:
            year_values = pd.to_datetime(dataset["as_of_date"], errors="coerce").dt.year
            for year, group in dataset.groupby(year_values):
                if pd.isna(year):
                    continue
                row = {"year": int(year), "rows": int(len(group))}
                row.update({f"future_return_5d_{k}": v for k, v in _stats(group["future_return_5d"]).items()})
                row["avoid_loss_5d_positive_rate"] = float(group["avoid_loss_5d"].mean())
                row["miss_profit_5d_positive_rate"] = float(group["miss_profit_5d"].mean())
                by_year.append(row)
        return {
            "overall": overall,
            "by_year": by_year,
            "by_price_band": self._price_band_distribution(dataset),
            "by_scale_category": self._category_distribution(dataset, "scale_category"),
        }

    def _price_band_distribution(self, dataset: pd.DataFrame) -> list[dict[str, Any]]:
        if "close" not in dataset.columns or dataset.empty:
            return []
        bands = pd.cut(pd.to_numeric(dataset["close"], errors="coerce"), bins=[0, 500, 1000, 3000, 10000, float("inf")], labels=["0-500", "500-1000", "1000-3000", "3000-10000", "10000+"])
        rows = []
        for band, group in dataset.groupby(bands, observed=True):
            rows.append({"price_band": str(band), "rows": int(len(group)), "future_return_5d_mean": _stats(group["future_return_5d"])["mean"]})
        return rows

    def _category_distribution(self, dataset: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        if column not in dataset.columns or dataset.empty:
            return []
        rows = []
        for value, group in dataset.groupby(column, dropna=False):
            rows.append({column: str(value), "rows": int(len(group)), "future_return_5d_mean": _stats(group["future_return_5d"])["mean"]})
        return rows

    def _split_report(self, dataset: pd.DataFrame) -> dict[str, Any]:
        rows = []
        distributions = []
        for split, group in dataset.groupby("split", dropna=False):
            dates = pd.to_datetime(group["as_of_date"], errors="coerce")
            rows.append({"split": str(split), "rows": int(len(group)), "start": str(dates.min().date()) if not dates.empty else None, "end": str(dates.max().date()) if not dates.empty else None})
            stats = _stats(group["future_return_5d"])
            distributions.append({"split": str(split), "rows": int(len(group)), **{f"future_return_5d_{key}": value for key, value in stats.items()}})
        return {"row_counts": rows, "label_distributions": distributions}

    def _empty_report(self, metadata: dict[str, Any], output_paths: dict[str, Any], columns: list[str]) -> dict[str, Any]:
        return {
            "metadata": metadata,
            "input_paths": {"base_dataset": str(self.base_dataset)},
            "output_paths": output_paths,
            "data_policy": self._data_policy(),
            "schema": {"key_columns": KEY_COLUMNS, "label_columns": LABEL_COLUMNS, "feature_columns": []},
            "row_counts": {"rows_before_filter": 0, "rows_after_horizon_filter": 0, "rows_after_missing_feature_filter": 0, "horizon_missing_count": 0, "label_missing_count": 0, "feature_missing_count": 0, "final_feature_count": 0},
            "missing_policy": {"base_columns": columns, "missing_rate_by_feature_top50": [], "dropped_high_missing_features": []},
            "leakage_audit": {"forbidden_columns_found": [], "label_like_feature_columns_found": [], "leakage_risk": "high", "blocking_issues": ["Base dataset is missing or lacks required date/code/close columns."]},
            "label_distribution": {"overall": {}, "by_year": [], "by_price_band": [], "by_scale_category": []},
            "split": {"row_counts": [], "label_distributions": []},
            "dropped_feature_columns": [],
            "recommended_next_phase": "Retraining deferred",
        }

    def _data_policy(self) -> dict[str, list[str]]:
        return {
            "allowed_sources": [
                "data/ml/datasets/ml_dataset.parquet",
                "data/raw/prices_YYYY-MM-DD.json",
                "data/ml/features/",
                "data/ml/labels/",
                "mechanical future return labels from API-origin price series",
            ],
            "forbidden_sources": [
                "trades.csv teacher labels",
                "backtest_summary.json teacher labels",
                "summary.csv / portfolio history teacher labels",
                "realized P/L or win/loss",
                "v2_75-v2_79 trading outcomes",
                "selected-only backtest universe",
                "selected_count_in_day",
                "current model regenerated historical predictions",
            ],
        }

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


def build_report(root: Path | str = ROOT, options: BuildOptions | None = None) -> dict[str, Any]:
    return Phase5CExitAIV2DatasetBuilder(root).build(options)


def save_report(result: dict[str, Any], root: Path | str = ROOT) -> Phase5CPaths:
    return Phase5CExitAIV2DatasetBuilder(root).save_report(result)


def run(root: Path | str = ROOT, options: BuildOptions | None = None) -> Phase5CPaths:
    builder = Phase5CExitAIV2DatasetBuilder(root)
    result = builder.build(options)
    return builder.save_report(result)
