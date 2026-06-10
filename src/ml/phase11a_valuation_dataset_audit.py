"""Phase 11-A Valuation Engine dataset audit.

This audit is intentionally read-only. It builds a valuation research dataset
from prediction-time ML artifacts and API-origin market data, then checks label
quality, feature availability, feature/label relationships, and leakage risk.
It must not read backtest outcomes, trades, portfolio state, or current PM
multiplier data as features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
START_DATE = "2023-01-01"
END_DATE = "2026-05-31"
REPORT_STEM = "phase11a_valuation_dataset_audit_2023-01_to_2026-05"
DATASET_PATH = Path("data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet")

LABEL_COLUMNS = [
    "future_return_20d",
    "future_max_return_20d",
    "future_max_drawdown_20d",
    "opportunity_value_20d",
    "opportunity_top_decile_20d",
]

CORE_LABEL_COLUMNS = [
    "future_return_20d",
    "future_max_return_20d",
    "future_max_drawdown_20d",
    "opportunity_value_20d",
]

STOCK_SELECTION_BASE_FEATURES = [
    "risk_adjusted_score",
    "expected_return",
    "stock_selection_rank_score",
    "candidate_strength",
]

FEATURE_CANDIDATES = {
    "market": [
        "topix_return_5d",
        "topix_return_10d",
        "topix_return_20d",
        "relative_return_5d",
        "relative_return_10d",
        "relative_return_20d",
    ],
    "price": [
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "ma5_gap",
        "ma10_gap",
        "ma25_gap",
        "ma75_gap",
        "ma5_slope",
        "ma25_slope",
        "body_ratio",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "close_position",
        "gap_up_ratio",
        "daily_range_ratio",
    ],
    "volume_liquidity": [
        "volume",
        "turnover_value",
        "volume_ratio_5d",
        "volume_ratio_20d",
        "turnover_ratio_5d",
        "turnover_ratio_20d",
    ],
    "financial": [
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
    ],
    "stock_selection": STOCK_SELECTION_BASE_FEATURES,
}

FORBIDDEN_TOKENS = {
    "backtest",
    "trade",
    "trades",
    "profit",
    "loss",
    "pnl",
    "cash",
    "portfolio",
    "position",
    "selected",
    "bought",
    "affordable",
    "skip",
    "exit",
    "final_assets",
    "actual",
    "outcome",
    "pm_multiplier",
    "current_pm",
}
FUTURE_PREFIXES = ("future_",)
ALWAYS_FORBIDDEN_COLUMNS = {"selected_count_in_day", "current_pm_multiplier", "pm_multiplier"}
ALLOWED_NAME_EXCEPTIONS = {"close_position"}


@dataclass(frozen=True)
class Phase11APaths:
    markdown: Path
    json: Path
    dataset: Path | None


def _date_text_from_path(path: Path, prefix: str) -> str:
    return path.stem.removeprefix(prefix)


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(result) else result


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _safe_corr(left: pd.Series | None, right: pd.Series | None, *, method: str = "pearson") -> float | None:
    if left is None or right is None:
        return None
    frame = pd.DataFrame({"left": _numeric(left), "right": _numeric(right)}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    value = frame["left"].corr(frame["right"], method=method)
    return None if pd.isna(value) else float(value)


def _summary_stats(series: pd.Series | None) -> dict[str, Any]:
    values = _numeric(series).dropna()
    total = 0 if series is None else int(len(series))
    if values.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "missing_rate": 1.0 if total else None,
        }
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "missing_rate": float(1.0 - values.count() / total) if total else None,
    }


class Phase11AValuationDatasetAudit:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        save_dataset: bool = True,
        use_cached_dataset: bool = True,
    ) -> None:
        self.root = Path(root)
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.save_dataset_enabled = save_dataset
        self.use_cached_dataset = use_cached_dataset

    def build_report(self) -> dict[str, Any]:
        dataset, build_info = self.build_dataset()
        feature_columns = self.feature_columns(dataset)
        leakage = self.leakage_checklist(feature_columns, dataset)
        label_summary = self.label_summary(dataset)
        feature_availability = self.feature_availability(dataset, feature_columns)
        correlations = self.correlation_audit(dataset, feature_columns)
        deciles = self.decile_quality_audit(dataset, feature_columns)
        stock_selection = self.stock_selection_score_audit(dataset)
        recommendation = self.recommendation(dataset, feature_availability, correlations, leakage)
        return {
            "metadata": {
                "phase": "11-A",
                "audit_only": True,
                "training_executed": False,
                "backtest_executed": False,
                "historical_predictions_regenerated": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
                "start_date": self.start_date.strftime("%Y-%m-%d"),
                "end_date": self.end_date.strftime("%Y-%m-%d"),
            },
            "sources": self.sources(),
            "build_info": build_info,
            "dataset_summary": self.dataset_summary(dataset, feature_columns),
            "label_summary": label_summary,
            "feature_availability": feature_availability,
            "correlation_audit": correlations,
            "decile_quality_audit": deciles,
            "stock_selection_score_audit": stock_selection,
            "leakage_checklist": leakage,
            "recommendation": recommendation,
        }

    def save_report(self, report: dict[str, Any], dataset: pd.DataFrame | None = None) -> Phase11APaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        dataset_path = None
        if self.save_dataset_enabled and dataset is not None and not dataset.empty:
            dataset_path = self.root / DATASET_PATH
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(dataset_path, index=False)
        return Phase11APaths(markdown=md_path, json=json_path, dataset=dataset_path)

    def run(self) -> Phase11APaths:
        dataset, build_info = self.build_dataset()
        feature_columns = self.feature_columns(dataset)
        report = {
            "metadata": {
                "phase": "11-A",
                "audit_only": True,
                "training_executed": False,
                "backtest_executed": False,
                "historical_predictions_regenerated": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
                "start_date": self.start_date.strftime("%Y-%m-%d"),
                "end_date": self.end_date.strftime("%Y-%m-%d"),
            },
            "sources": self.sources(),
            "build_info": build_info,
            "dataset_summary": self.dataset_summary(dataset, feature_columns),
            "label_summary": self.label_summary(dataset),
            "feature_availability": self.feature_availability(dataset, feature_columns),
            "correlation_audit": self.correlation_audit(dataset, feature_columns),
            "decile_quality_audit": self.decile_quality_audit(dataset, feature_columns),
            "stock_selection_score_audit": self.stock_selection_score_audit(dataset),
            "leakage_checklist": self.leakage_checklist(feature_columns, dataset),
        }
        report["recommendation"] = self.recommendation(
            dataset,
            report["feature_availability"],
            report["correlation_audit"],
            report["leakage_checklist"],
        )
        return self.save_report(report, dataset)

    def sources(self) -> dict[str, str]:
        return {
            "features": str(self.root / "data/ml/features/features_YYYY-MM-DD.parquet"),
            "labels": str(self.root / "data/ml/labels/labels_YYYY-MM-DD.parquet"),
            "walk_forward_predictions": str(self.root / "data/ml/walk_forward_predictions/predictions_YYYY-MM-DD.parquet"),
            "raw_prices": str(self.root / "data/raw/prices_YYYY-MM-DD.json"),
            "backtest_logs_used": "false",
            "trades_csv_used": "false",
        }

    def build_dataset(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        cached = self._load_cached_dataset()
        if cached is not None:
            return cached, {
                "feature_prediction_rows": int(len(cached)),
                "label_rows": int(len(cached)),
                "joined_rows": int(len(cached)),
                "actual_date_range": {
                    "min": cached["date"].min().strftime("%Y-%m-%d"),
                    "max": cached["date"].max().strftime("%Y-%m-%d"),
                }
                if not cached.empty
                else None,
                "coverage": 1.0 if not cached.empty else 0.0,
                "cached_dataset_used": True,
                "cached_dataset_path": str(self.root / DATASET_PATH),
                "note": "Loaded cached Phase 11-A valuation dataset; no backtest outputs are read.",
            }
        base = self._load_daily_feature_prediction_frames()
        labels = self._build_opportunity_labels(base)
        if base.empty or labels.empty:
            return pd.DataFrame(), {
                "feature_prediction_rows": int(len(base)),
                "label_rows": int(len(labels)),
                "joined_rows": 0,
                "actual_date_range": None,
                "coverage": 0.0,
            }
        dataset = base.merge(labels, on=["date", "code"], how="inner")
        dataset = self._add_opportunity_top_decile(dataset)
        dataset = dataset.sort_values(["date", "code"]).reset_index(drop=True)
        date_range = None
        if not dataset.empty:
            date_range = {
                "min": dataset["date"].min().strftime("%Y-%m-%d"),
                "max": dataset["date"].max().strftime("%Y-%m-%d"),
            }
        return dataset, {
            "feature_prediction_rows": int(len(base)),
            "label_rows": int(len(labels)),
            "joined_rows": int(len(dataset)),
            "actual_date_range": date_range,
            "coverage": float(len(dataset) / len(base)) if len(base) else 0.0,
            "note": "Labels use API-origin raw prices and existing label artifacts; no backtest outputs are read.",
        }

    def _load_cached_dataset(self) -> pd.DataFrame | None:
        path = self.root / DATASET_PATH
        if not self.use_cached_dataset or not path.exists():
            return None
        data = pd.read_parquet(path)
        if data.empty or not {"date", "code", *LABEL_COLUMNS}.issubset(data.columns):
            return None
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        data = data[(data["date"] >= self.start_date) & (data["date"] <= self.end_date)].copy()
        return data.sort_values(["date", "code"]).reset_index(drop=True)

    def feature_columns(self, dataset: pd.DataFrame) -> list[str]:
        if dataset.empty:
            return []
        wanted = []
        for columns in FEATURE_CANDIDATES.values():
            wanted.extend(columns)
        for base in STOCK_SELECTION_BASE_FEATURES:
            wanted.extend(
                [
                    f"{base}_rank_in_day",
                    f"{base}_percentile_in_day",
                    f"{base}_gap_to_best",
                ]
            )
        return [column for column in dict.fromkeys(wanted) if column in dataset.columns and self._feature_allowed(column)]

    def label_summary(self, dataset: pd.DataFrame) -> list[dict[str, Any]]:
        return [{"label": label, **_summary_stats(dataset.get(label))} for label in LABEL_COLUMNS]

    def feature_availability(self, dataset: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
        rows = []
        all_candidates = []
        for group, columns in FEATURE_CANDIDATES.items():
            all_candidates.extend((group, column) for column in columns)
        for base in STOCK_SELECTION_BASE_FEATURES:
            all_candidates.extend(("stock_selection_relative", f"{base}_{suffix}") for suffix in ["rank_in_day", "percentile_in_day", "gap_to_best"])

        for group, column in all_candidates:
            exists = column in dataset.columns
            series = dataset[column] if exists else None
            numeric = _numeric(series) if exists else pd.Series(dtype=float)
            missing_rate = float(series.isna().mean()) if exists and len(series) else None
            allowed = exists and column in feature_columns
            exclude_reason = ""
            if not exists:
                exclude_reason = "missing"
            elif not self._feature_allowed(column):
                exclude_reason = "forbidden_or_future"
            elif missing_rate is not None and missing_rate >= 0.80:
                exclude_reason = "high_missing_rate_review"
            rows.append(
                {
                    "group": group,
                    "feature": column,
                    "exists": bool(exists),
                    "allowed": bool(allowed),
                    "exclude_reason": exclude_reason,
                    "missing_rate": missing_rate,
                    "dtype": str(series.dtype) if exists else None,
                    "count": int(numeric.count()) if exists else 0,
                    "mean": float(numeric.mean()) if not numeric.dropna().empty else None,
                    "min": float(numeric.min()) if not numeric.dropna().empty else None,
                    "max": float(numeric.max()) if not numeric.dropna().empty else None,
                    "p10": float(numeric.quantile(0.10)) if not numeric.dropna().empty else None,
                    "p90": float(numeric.quantile(0.90)) if not numeric.dropna().empty else None,
                }
            )
        return rows

    def correlation_audit(self, dataset: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
        rows = []
        for feature in feature_columns:
            if _numeric(dataset.get(feature)).dropna().nunique() < 2:
                continue
            for label in CORE_LABEL_COLUMNS:
                rows.append(
                    {
                        "feature": feature,
                        "label": label,
                        "pearson": _safe_corr(dataset.get(feature), dataset.get(label), method="pearson"),
                        "spearman": _safe_corr(dataset.get(feature), dataset.get(label), method="spearman"),
                    }
                )
        return rows

    def decile_quality_audit(self, dataset: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
        rows = []
        labels = [*CORE_LABEL_COLUMNS, "opportunity_top_decile_20d"]
        for feature in feature_columns:
            values = _numeric(dataset.get(feature))
            if values.dropna().nunique() < 2:
                continue
            frame = dataset[["date", feature, *labels]].copy()
            frame[feature] = values
            frame = frame.dropna(subset=[feature])
            if frame.empty:
                continue
            try:
                frame["decile"] = pd.qcut(frame[feature], q=10, labels=False, duplicates="drop") + 1
            except ValueError:
                continue
            for decile, group in frame.groupby("decile", dropna=True):
                row = {"feature": feature, "bucket_type": "global_decile", "decile": int(decile), "count": int(len(group))}
                for label in labels:
                    row[f"{label}_mean"] = float(_numeric(group[label]).mean()) if label in group else None
                rows.append(row)
        return rows

    def stock_selection_score_audit(self, dataset: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        labels = ["opportunity_value_20d", "opportunity_top_decile_20d"]
        for base in STOCK_SELECTION_BASE_FEATURES:
            for variant in [base, f"{base}_rank_in_day", f"{base}_percentile_in_day", f"{base}_gap_to_best"]:
                if variant not in dataset.columns:
                    continue
                row = {"base_score": base, "variant": variant, "count": int(_numeric(dataset[variant]).count())}
                for label in labels:
                    row[f"{label}_pearson"] = _safe_corr(dataset[variant], dataset.get(label), method="pearson")
                    row[f"{label}_spearman"] = _safe_corr(dataset[variant], dataset.get(label), method="spearman")
                rows.append(row)
        return rows

    def leakage_checklist(self, feature_columns: list[str], dataset: pd.DataFrame) -> dict[str, Any]:
        forbidden = [column for column in feature_columns if self._is_forbidden_name(column)]
        future_in_features = [column for column in feature_columns if column.startswith(FUTURE_PREFIXES)]
        backtest_columns = [column for column in feature_columns if any(token in column.lower() for token in ["backtest", "result"])]
        trade_columns = [column for column in feature_columns if any(token in column.lower() for token in ["trade", "profit", "loss", "pnl", "actual", "outcome"])]
        cash_portfolio = [
            column
            for column in feature_columns
            if column.lower() not in ALLOWED_NAME_EXCEPTIONS
            and any(token in column.lower() for token in ["cash", "portfolio", "position"])
        ]
        selected_bought = [column for column in feature_columns if any(token in column.lower() for token in ["selected", "bought", "affordable"])]
        blocking = []
        if forbidden:
            blocking.append("forbidden feature column selected")
        if future_in_features:
            blocking.append("future column selected as feature")
        if "selected_count_in_day" in feature_columns:
            blocking.append("selected_count_in_day selected as feature")
        if "pm_multiplier" in feature_columns or "current_pm_multiplier" in feature_columns:
            blocking.append("current PM multiplier selected as feature")
        leakage_risk = "high" if blocking else "low"
        return {
            "feature_count": int(len(feature_columns)),
            "forbidden_columns_found": forbidden,
            "future_columns_in_features": future_in_features,
            "backtest_columns_in_features": backtest_columns,
            "trade_result_columns_in_features": trade_columns,
            "cash_or_portfolio_columns_in_features": cash_portfolio,
            "current_pm_multiplier_used": "pm_multiplier" in feature_columns or "current_pm_multiplier" in feature_columns,
            "selected_or_bought_used": bool(selected_bought),
            "selected_count_in_day_used": "selected_count_in_day" in feature_columns,
            "historical_predictions_regenerated": False,
            "leakage_risk": leakage_risk,
            "blocking_issues": blocking,
            "all_dataset_columns_matching_forbidden_tokens": [column for column in dataset.columns if self._is_forbidden_name(column)],
        }

    def recommendation(
        self,
        dataset: pd.DataFrame,
        feature_availability: list[dict[str, Any]],
        correlations: list[dict[str, Any]],
        leakage: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_groups = sorted(
            {
                row["group"]
                for row in feature_availability
                if row.get("allowed") and row.get("missing_rate") is not None and float(row["missing_rate"]) < 0.80
            }
        )
        excluded_groups = sorted({row["group"] for row in feature_availability if row.get("exclude_reason")})
        corr_frame = pd.DataFrame(correlations)
        top_features: list[dict[str, Any]] = []
        if not corr_frame.empty:
            corr_frame["abs_spearman"] = pd.to_numeric(corr_frame["spearman"], errors="coerce").abs()
            top_features = corr_frame.sort_values("abs_spearman", ascending=False).head(20).drop(columns=["abs_spearman"]).to_dict("records")
        blocking = leakage.get("blocking_issues") or []
        return {
            "ready_for_phase11b": not blocking and not dataset.empty,
            "recommended_primary_label": "opportunity_value_20d",
            "recommended_secondary_labels": ["future_max_return_20d", "future_max_drawdown_20d", "opportunity_top_decile_20d"],
            "recommended_feature_groups": allowed_groups,
            "excluded_feature_groups": excluded_groups,
            "prototype_model_task_candidates": [
                {"task": "regression", "target": "opportunity_value_20d", "recommendation": "primary"},
                {"task": "classification", "target": "opportunity_top_decile_20d", "recommendation": "secondary"},
                {"task": "multi_output", "target": "expected_upside / expected_downside", "recommendation": "research"},
            ],
            "top_correlation_features": top_features,
            "known_risks": [
                "opportunity_value_20d is still price-derived and may favor short-term momentum unless valuation features add independent signal",
                "financial statement missingness should be reviewed before prototype training",
                "candidate_strength is derived from prediction-time Stock Selection outputs when no native column exists",
            ],
            "next_phase_command": "PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 scripts/ml/audit_phase11a_valuation_dataset.py",
        }

    def dataset_summary(self, dataset: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
        if dataset.empty:
            return {
                "rows": 0,
                "unique_codes": 0,
                "date_range": None,
                "candidate_days": 0,
                "feature_count": int(len(feature_columns)),
                "label_count": int(len(LABEL_COLUMNS)),
                "coverage": 0.0,
            }
        label_complete = dataset[LABEL_COLUMNS].notna().all(axis=1)
        return {
            "rows": int(len(dataset)),
            "unique_codes": int(dataset["code"].nunique()),
            "date_range": {
                "min": dataset["date"].min().strftime("%Y-%m-%d"),
                "max": dataset["date"].max().strftime("%Y-%m-%d"),
            },
            "candidate_days": int(dataset["date"].nunique()),
            "feature_count": int(len(feature_columns)),
            "label_count": int(len(LABEL_COLUMNS)),
            "coverage": float(label_complete.mean()) if len(dataset) else 0.0,
        }

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Phase 11-A Valuation Engine Dataset Audit",
                "",
                "## Scope",
                "",
                "- audit only",
                "- no training, no backtest, no profile change",
                "- no backtest result, trade result, cash, portfolio, selected/bought/affordable, or current PM multiplier features",
                "",
                "## Dataset Summary",
                "",
                self._table([report["dataset_summary"]], ["rows", "unique_codes", "date_range", "candidate_days", "feature_count", "label_count", "coverage"]),
                "",
                "## Label Summary",
                "",
                self._table(report["label_summary"], ["label", "count", "mean", "median", "p10", "p25", "p75", "p90", "missing_rate"]),
                "",
                "## Feature Availability",
                "",
                self._table(report["feature_availability"], ["group", "feature", "exists", "allowed", "exclude_reason", "missing_rate", "dtype", "mean", "min", "max", "p10", "p90"]),
                "",
                "## Correlation Audit",
                "",
                self._table(report["correlation_audit"][:120], ["feature", "label", "pearson", "spearman"]),
                "",
                "## Decile Quality Audit",
                "",
                self._table(report["decile_quality_audit"][:160], ["feature", "bucket_type", "decile", "count", "future_return_20d_mean", "future_max_return_20d_mean", "future_max_drawdown_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_mean"]),
                "",
                "## Stock Selection Score Audit",
                "",
                self._table(report["stock_selection_score_audit"], ["base_score", "variant", "count", "opportunity_value_20d_pearson", "opportunity_value_20d_spearman", "opportunity_top_decile_20d_pearson", "opportunity_top_decile_20d_spearman"]),
                "",
                "## Leakage Checklist",
                "",
                self._table([report["leakage_checklist"]], ["forbidden_columns_found", "future_columns_in_features", "backtest_columns_in_features", "trade_result_columns_in_features", "cash_or_portfolio_columns_in_features", "current_pm_multiplier_used", "selected_or_bought_used", "selected_count_in_day_used", "historical_predictions_regenerated", "leakage_risk", "blocking_issues"]),
                "",
                "## Recommendation",
                "",
                self._table([report["recommendation"]], ["ready_for_phase11b", "recommended_primary_label", "recommended_secondary_labels", "recommended_feature_groups", "excluded_feature_groups", "prototype_model_task_candidates", "known_risks", "next_phase_command"]),
                "",
            ]
        )

    def _load_daily_feature_prediction_frames(self) -> pd.DataFrame:
        frames = []
        feature_root = self.root / "data" / "ml" / "features"
        pred_root = self.root / "data" / "ml" / "walk_forward_predictions"
        for pred_path in sorted(pred_root.glob("predictions_*.parquet")):
            date_text = _date_text_from_path(pred_path, "predictions_")
            date = pd.Timestamp(date_text)
            if date < self.start_date or date > self.end_date:
                continue
            feature_path = feature_root / f"features_{date_text}.parquet"
            if not feature_path.exists():
                continue
            features = pd.read_parquet(feature_path)
            predictions = pd.read_parquet(pred_path)
            joined = self._join_features_predictions(features, predictions)
            if not joined.empty:
                frames.append(joined)
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        data = self._derive_stock_selection_scores(data)
        data = self._add_same_day_relative_features(data)
        return data.sort_values(["date", "code"]).reset_index(drop=True)

    def _join_features_predictions(self, features: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        left = features.copy()
        right = predictions.copy()
        for frame in [left, right]:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["code"] = frame["code"].astype("string")
        keep_prediction_columns = [
            "date",
            "code",
            "expected_return_5d",
            "expected_return_10d",
            "upside_probability_10d",
            "bad_entry_probability_10d",
            "expected_max_return_10d",
            "expected_max_return_20d",
            "swing_success_probability_20d",
            "entry_risk_label",
            "ml_score",
        ]
        available = [column for column in keep_prediction_columns if column in right.columns]
        return left.merge(right[available], on=["date", "code"], how="inner", suffixes=("", "_prediction"))

    def _derive_stock_selection_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        if "risk_adjusted_score" not in result.columns:
            result["risk_adjusted_score"] = _numeric(result.get("expected_return_10d")) - 0.5 * _numeric(result.get("bad_entry_probability_10d"))
        if "expected_return" not in result.columns:
            result["expected_return"] = _numeric(result.get("expected_return_10d"))
        if "stock_selection_rank_score" not in result.columns:
            result["stock_selection_rank_score"] = _numeric(result.get("ml_score"))
        if "candidate_strength" not in result.columns:
            result["candidate_strength"] = (
                _numeric(result.get("expected_max_return_20d"))
                + _numeric(result.get("swing_success_probability_20d"))
                - _numeric(result.get("bad_entry_probability_10d"))
            )
        return result

    def _add_same_day_relative_features(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        grouped = result.groupby("date", group_keys=False)
        for column in STOCK_SELECTION_BASE_FEATURES:
            if column not in result.columns:
                continue
            values = _numeric(result[column])
            result[column] = values
            rank = grouped[column].rank(method="min", ascending=False)
            count = grouped[column].transform("count")
            result[f"{column}_rank_in_day"] = rank
            result[f"{column}_percentile_in_day"] = 1.0 - ((rank - 1.0) / (count - 1.0)).where(count > 1, 0.0)
            result[f"{column}_gap_to_best"] = grouped[column].transform("max") - result[column]
        return result

    def _build_opportunity_labels(self, base: pd.DataFrame) -> pd.DataFrame:
        label_root = self.root / "data" / "ml" / "labels"
        existing = self._load_existing_labels(label_root)
        price_labels = self._labels_from_raw_prices(base)
        if price_labels.empty:
            return existing
        if existing.empty:
            return price_labels
        labels = price_labels.merge(
            existing[["date", "code", "future_max_return_20d"]].rename(columns={"future_max_return_20d": "existing_future_max_return_20d"}),
            on=["date", "code"],
            how="left",
        )
        labels["future_max_return_20d"] = labels["future_max_return_20d"].combine_first(labels["existing_future_max_return_20d"])
        labels = labels.drop(columns=["existing_future_max_return_20d"])
        if not base.empty:
            keys = base[["date", "code"]].drop_duplicates()
            labels = keys.merge(labels, on=["date", "code"], how="inner")
        return labels

    def _load_existing_labels(self, label_root: Path) -> pd.DataFrame:
        frames = []
        for path in sorted(label_root.glob("labels_*.parquet")):
            date_text = _date_text_from_path(path, "labels_")
            date = pd.Timestamp(date_text)
            if date < self.start_date or date > self.end_date:
                continue
            df = pd.read_parquet(path)
            if {"date", "code", "future_max_return_20d"}.issubset(df.columns):
                part = df[["date", "code", "future_max_return_20d"]].copy()
                part["date"] = pd.to_datetime(part["date"], errors="coerce")
                part["code"] = part["code"].astype("string")
                frames.append(part)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _labels_from_raw_prices(self, base: pd.DataFrame) -> pd.DataFrame:
        prices = self._load_raw_prices()
        if prices.empty or base.empty:
            return pd.DataFrame()
        wanted: dict[str, set[pd.Timestamp]] = {}
        for row in base[["date", "code"]].drop_duplicates().itertuples(index=False):
            wanted.setdefault(str(row.code), set()).add(pd.Timestamp(row.date))
        rows = []
        for code, group in prices.groupby("code", sort=False):
            wanted_dates = wanted.get(str(code))
            if not wanted_dates:
                continue
            group = group.sort_values("date").reset_index(drop=True)
            dates = group["date"].tolist()
            opens = _numeric(group["open"]).tolist()
            closes = _numeric(group["close"]).tolist()
            highs = _numeric(group["high"]).tolist()
            lows = _numeric(group["low"]).tolist()
            for idx in range(0, len(group) - 20):
                date = pd.Timestamp(dates[idx])
                if date not in wanted_dates:
                    continue
                entry_price = opens[idx + 1]
                if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
                    continue
                window_highs = pd.Series(highs[idx + 1 : idx + 21], dtype="float64").dropna()
                window_lows = pd.Series(lows[idx + 1 : idx + 21], dtype="float64").dropna()
                future_close = closes[idx + 20]
                if pd.isna(future_close) or window_highs.empty or window_lows.empty:
                    continue
                future_return = float(future_close / entry_price - 1.0)
                future_max_return = float(window_highs.max() / entry_price - 1.0)
                future_max_drawdown = float(window_lows.min() / entry_price - 1.0)
                rows.append(
                    {
                        "date": date,
                        "code": str(code),
                        "future_return_20d": future_return,
                        "future_max_return_20d": future_max_return,
                        "future_max_drawdown_20d": future_max_drawdown,
                        "opportunity_value_20d": future_max_return - abs(future_max_drawdown),
                    }
                )
        labels = pd.DataFrame(rows)
        if labels.empty:
            return labels
        labels["code"] = labels["code"].astype("string")
        return labels

    def _load_raw_prices(self) -> pd.DataFrame:
        frames = []
        raw_root = self.root / "data" / "raw"
        price_start = self.start_date
        price_end = self.end_date + pd.Timedelta(days=70)
        for path in sorted(raw_root.glob("prices_*.json")):
            date_text = _date_text_from_path(path, "prices_")
            try:
                date = pd.Timestamp(date_text)
            except ValueError:
                continue
            if date < price_start or date > price_end:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("prices") if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                continue
            frame = pd.DataFrame(records)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        for column in ["open", "high", "low", "close"]:
            data[column] = _numeric(data[column])
        return data.dropna(subset=["date", "code", "open", "high", "low", "close"]).sort_values(["code", "date"]).reset_index(drop=True)

    def _add_opportunity_top_decile(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data = dataset.copy()
        data["opportunity_top_decile_20d"] = 0
        if data.empty or "opportunity_value_20d" not in data.columns:
            return data
        rank = data.groupby("date")["opportunity_value_20d"].rank(method="first", ascending=False)
        count = data.groupby("date")["opportunity_value_20d"].transform("count")
        threshold = (count * 0.10).clip(lower=1)
        data["opportunity_top_decile_20d"] = (rank <= threshold).astype(int)
        return data

    def _feature_allowed(self, column: str) -> bool:
        return not column.startswith(FUTURE_PREFIXES) and not self._is_forbidden_name(column)

    def _is_forbidden_name(self, column: str) -> bool:
        lowered = column.lower()
        if lowered in ALLOWED_NAME_EXCEPTIONS:
            return False
        if lowered in ALWAYS_FORBIDDEN_COLUMNS:
            return True
        return any(token in lowered for token in FORBIDDEN_TOKENS)

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in rows:
            values = [self._format_cell(row.get(column)) for column in columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def _format_cell(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False, default=str)
        if value is None:
            return ""
        return str(value).replace("\n", " ")
