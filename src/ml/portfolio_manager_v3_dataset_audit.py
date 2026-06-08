"""Phase 9-C PM AI v3 dataset quality and label audit.

Read-only audit for the Phase 9-B clean dataset. It does not train a model,
run a backtest, regenerate historical predictions, or overwrite current PM/Exit
model artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_v3_dataset_builder import (
    CONDITIONAL_RELATIVE_FEATURES,
    FORBIDDEN_TOKENS,
    KEY_COLUMNS,
    LABEL_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase9c_pm_ai_v3_dataset_audit_2023-01_to_2026-05"
DATASET_PATH = Path("data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_2023-01_to_2026-05.parquet")
MARKET_REGIME_PATH = Path("data/ml/portfolio_manager_v3/pm_v3_market_regime_daily_2023-01_to_2026-05.parquet")
METADATA_COLUMNS = ["data_source", "relative_feature_timing"]
CORRELATION_LABELS = [
    "future_10d_return",
    "downside_penalized_return_10d",
    "risk_adjusted_future_return_10d",
    "relative_future_utility_percentile_in_day",
    "top_decile_future_utility_in_day",
    "bottom_decile_future_utility_in_day",
]


@dataclass(frozen=True)
class Phase9CAuditPaths:
    markdown: Path
    json: Path


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if getattr(series, "dtype", None) == bool:
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _quantiles(series: pd.Series | None) -> dict[str, float | None]:
    values = _numeric(series).dropna()
    if values.empty:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "p10": None, "p25": None, "p75": None, "p90": None}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
    }


def _safe_mean(series: pd.Series | None) -> float | None:
    values = _numeric(series).dropna()
    return None if values.empty else float(values.mean())


class PMAIV3DatasetQualityAudit:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        dataset_path: Path | None = None,
        market_regime_path: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.dataset_path = self._root(dataset_path or DATASET_PATH)
        self.market_regime_path = self._root(market_regime_path or MARKET_REGIME_PATH)

    def build_report(self) -> dict[str, Any]:
        dataset = _read_parquet(self.dataset_path)
        market = _read_parquet(self.market_regime_path)
        feature_columns = self._feature_columns(dataset)
        label_columns = [column for column in LABEL_COLUMNS if column in dataset.columns]
        quality = self._basic_quality(dataset, feature_columns, label_columns)
        labels = self._label_audit(dataset, label_columns)
        corr = self._correlation_audit(dataset, feature_columns)
        market_audit = self._market_regime_audit(dataset, market)
        ranking = self._same_day_ranking_audit(dataset)
        sizing = self._sizing_label_candidates(dataset)
        leakage = self._leakage_audit(feature_columns, corr)
        verdict = self._verdict(quality, labels, corr, market_audit, ranking, sizing, leakage)
        return {
            "metadata": {
                "phase": "9-C",
                "audit_only": True,
                "training_executed": False,
                "backtest_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_model_regenerated_historical_predictions": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
            },
            "input_paths": {
                "dataset": str(self.dataset_path),
                "market_regime": str(self.market_regime_path),
            },
            "basic_quality": quality,
            "label_audit": labels,
            "correlation_audit": corr,
            "market_regime_audit": market_audit,
            "same_day_ranking_audit": ranking,
            "sizing_label_candidate_audit": sizing,
            "leakage_audit": leakage,
            "verdict": verdict,
            "feature_columns": feature_columns,
            "label_columns": label_columns,
        }

    def save_report(self, report: dict[str, Any]) -> Phase9CAuditPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9CAuditPaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 9-C PM AI v3 Dataset Audit",
                "",
                "## Scope",
                "",
                "- dataset quality / label audit only",
                "- no training, no backtest, no API refetch, no current artifact overwrite",
                "",
                "## Basic Quality",
                "",
                self._table([report["basic_quality"]["summary"]], ["row_count", "date_min", "date_max", "code_count", "feature_count", "label_count", "duplicate_key_count", "infinite_value_count", "constant_feature_count", "high_missing_feature_count", "date_coverage_gap_count"]),
                "",
                "## Label Null Rate",
                "",
                self._table(report["basic_quality"]["label_null_rate"], ["column", "null_rate"]),
                "",
                "## Feature Missing Top 20",
                "",
                self._table(report["basic_quality"]["feature_null_rate_top20"], ["column", "null_rate"]),
                "",
                "## Label Audit",
                "",
                self._table(report["label_audit"], ["label", "mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90", "null_rate", "positive_rate", "top_decile_positive_count", "bottom_decile_positive_count"]),
                "",
                "## Correlation Audit",
                "",
                self._table(report["correlation_audit"]["label_summaries"], ["label", "top_positive", "top_negative", "near_zero_count", "suspicious_high_correlation_features"]),
                "",
                "## Market Regime Audit",
                "",
                self._table(report["market_regime_audit"]["by_regime"], ["market_regime_class_prototype", "row_count", "future_10d_return_mean", "downside_penalized_return_10d_mean", "top_decile_rate", "bottom_decile_rate"]),
                "",
                self._table(report["market_regime_audit"]["attack_score_quantiles"], ["attack_score_bucket", "row_count", "future_10d_return_mean", "downside_penalized_return_10d_mean", "top_decile_rate", "bottom_decile_rate"]),
                "",
                "## Same-Day Ranking Audit",
                "",
                self._table([report["same_day_ranking_audit"]], ["rank_bounds_valid", "expected_return_rank_corr_to_future_rank", "risk_adjusted_rank_corr_to_future_rank", "bad_entry_rank_corr_to_bottom_decile", "candidate_strength_corr_to_future_utility", "gap_to_best_corr_to_future_utility"]),
                "",
                "## Sizing Label Candidate Distribution",
                "",
                self._table(report["sizing_label_candidate_audit"]["distribution"], ["label", "count", "rate"]),
                "",
                "## Leakage Audit",
                "",
                self._table([report["leakage_audit"]], ["forbidden_feature_count", "forbidden_feature_columns", "label_columns_in_features", "suspicious_high_correlation_features", "leakage_risk"]),
                "",
                "## Verdict",
                "",
                self._table([report["verdict"]], ["dataset_is_trainable", "recommended_target_label", "recommended_layer2_label", "recommended_layer3_label", "market_regime_usefulness", "relative_feature_usefulness", "next_phase_recommendation"]),
                "",
            ]
        )

    def _root(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def _feature_columns(self, dataset: pd.DataFrame) -> list[str]:
        excluded = set(KEY_COLUMNS + LABEL_COLUMNS + METADATA_COLUMNS)
        return [column for column in dataset.columns if column not in excluded]

    def _basic_quality(self, dataset: pd.DataFrame, feature_columns: list[str], label_columns: list[str]) -> dict[str, Any]:
        if dataset.empty:
            return {"summary": self._empty_summary(), "candidate_count_distribution": {}, "label_null_rate": [], "feature_null_rate_top20": [], "constant_features": [], "high_missing_features": [], "date_coverage_gaps": []}
        dates = pd.to_datetime(dataset["prediction_date"], errors="coerce")
        candidate_counts = dataset.groupby("prediction_date")["code"].count()
        duplicate_key_count = int(dataset.duplicated(["prediction_date", "code"]).sum())
        numeric = dataset[feature_columns + label_columns].apply(pd.to_numeric, errors="coerce")
        infinite_value_count = int((numeric == float("inf")).sum().sum() + (numeric == float("-inf")).sum().sum())
        constant_features = [column for column in feature_columns if dataset[column].nunique(dropna=True) <= 1]
        feature_null = self._null_rates(dataset, feature_columns)
        high_missing = [row["column"] for row in feature_null if (row["null_rate"] or 0) >= 0.50]
        date_gaps = self._date_coverage_gaps(dates)
        return {
            "summary": {
                "row_count": int(len(dataset)),
                "date_min": dates.min().strftime("%Y-%m-%d") if not dates.dropna().empty else None,
                "date_max": dates.max().strftime("%Y-%m-%d") if not dates.dropna().empty else None,
                "code_count": int(dataset["code"].nunique()),
                "feature_count": len(feature_columns),
                "label_count": len(label_columns),
                "duplicate_key_count": duplicate_key_count,
                "infinite_value_count": infinite_value_count,
                "constant_feature_count": len(constant_features),
                "high_missing_feature_count": len(high_missing),
                "date_coverage_gap_count": len(date_gaps),
            },
            "candidate_count_distribution": {
                "min": int(candidate_counts.min()),
                "mean": float(candidate_counts.mean()),
                "median": float(candidate_counts.median()),
                "max": int(candidate_counts.max()),
            },
            "label_null_rate": self._null_rates(dataset, label_columns),
            "feature_null_rate_top20": feature_null[:20],
            "constant_features": constant_features,
            "high_missing_features": high_missing,
            "date_coverage_gaps": date_gaps[:20],
        }

    def _label_audit(self, dataset: pd.DataFrame, label_columns: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        top = dataset.get("top_decile_future_utility_in_day", pd.Series(False, index=dataset.index)).astype(bool)
        bottom = dataset.get("bottom_decile_future_utility_in_day", pd.Series(False, index=dataset.index)).astype(bool)
        for label in label_columns:
            series = _numeric(dataset.get(label))
            stats = _quantiles(series)
            rows.append(
                {
                    "label": label,
                    **stats,
                    "null_rate": float(dataset[label].isna().mean()),
                    "positive_rate": float(series.gt(0).mean()) if not series.dropna().empty else None,
                    "top_decile_positive_count": int(series[top].gt(0).sum()) if label in dataset else 0,
                    "bottom_decile_positive_count": int(series[bottom].gt(0).sum()) if label in dataset else 0,
                }
            )
        return rows

    def _correlation_audit(self, dataset: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
        numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(dataset[column]) or dataset[column].dtype == bool]
        summaries: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        suspicious_all: set[str] = set()
        for label in [column for column in CORRELATION_LABELS if column in dataset.columns]:
            label_series = _numeric(dataset[label])
            corrs: list[dict[str, Any]] = []
            for feature in numeric_features:
                values = _numeric(dataset[feature])
                frame = pd.DataFrame({"x": values, "y": label_series}).dropna()
                if len(frame) < 5 or frame["x"].nunique() <= 1 or frame["y"].nunique() <= 1:
                    continue
                corr = frame["x"].corr(frame["y"])
                if pd.notna(corr):
                    corrs.append({"feature": feature, "correlation": float(corr)})
            ordered = sorted(corrs, key=lambda row: row["correlation"], reverse=True)
            negative = sorted(corrs, key=lambda row: row["correlation"])
            near_zero = [row["feature"] for row in corrs if abs(row["correlation"]) < 0.01]
            suspicious = [row["feature"] for row in corrs if abs(row["correlation"]) >= 0.98]
            suspicious_all.update(suspicious)
            details[label] = {
                "top_positive_20": ordered[:20],
                "top_negative_20": negative[:20],
                "near_zero_features": near_zero,
                "suspicious_high_correlation_features": suspicious,
            }
            summaries.append(
                {
                    "label": label,
                    "top_positive": self._corr_text(ordered[:3]),
                    "top_negative": self._corr_text(negative[:3]),
                    "near_zero_count": len(near_zero),
                    "suspicious_high_correlation_features": suspicious,
                }
            )
        return {"label_summaries": summaries, "details": details, "suspicious_high_correlation_features": sorted(suspicious_all)}

    def _market_regime_audit(self, dataset: pd.DataFrame, market: pd.DataFrame) -> dict[str, Any]:
        if dataset.empty:
            return {"by_regime": [], "attack_score_quantiles": [], "market_regime_usefulness": "unknown"}
        data = dataset.copy()
        if "date" in market.columns:
            market = market.copy()
            market["prediction_date"] = pd.to_datetime(market["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            cols = [c for c in ["prediction_date", "market_regime_class_prototype", "market_attack_score_prototype"] if c in market.columns]
            data = data.drop(
                columns=[c for c in ["market_regime_class_prototype", "market_attack_score_prototype"] if c in data.columns],
                errors="ignore",
            )
            data = data.merge(market[cols].drop_duplicates("prediction_date"), on="prediction_date", how="left")
        by_regime = []
        for regime, group in data.groupby("market_regime_class_prototype", dropna=False):
            by_regime.append(self._market_group_row(str(regime), group))
        score = _numeric(data.get("market_attack_score_prototype"))
        data["_attack_bucket"] = pd.qcut(score.rank(method="first"), q=5, labels=["q1_low", "q2", "q3", "q4", "q5_high"], duplicates="drop")
        bucket_rows = [
            self._market_group_row(str(bucket), group, key="attack_score_bucket")
            for bucket, group in data.groupby("_attack_bucket", dropna=True, observed=False)
        ]
        top_rates = [row["top_decile_rate"] for row in bucket_rows if row["top_decile_rate"] is not None]
        spread = (max(top_rates) - min(top_rates)) if top_rates else 0.0
        return {"by_regime": by_regime, "attack_score_quantiles": bucket_rows, "market_regime_usefulness": "medium" if spread >= 0.03 else "low"}

    def _same_day_ranking_audit(self, dataset: pd.DataFrame) -> dict[str, Any]:
        if dataset.empty:
            return {}
        data = dataset.copy()
        counts = _numeric(data.get("candidate_count_in_day"))
        ranks = _numeric(data.get("relative_future_utility_rank_in_day"))
        rank_bounds_valid = bool(ranks.dropna().ge(1).all() and (ranks.dropna() <= counts[ranks.dropna().index]).all())
        data["_expected_rank"] = data.groupby("prediction_date")["expected_return_10d"].rank(method="first", ascending=False)
        data["_risk_rank"] = data.groupby("prediction_date")["risk_adjusted_score"].rank(method="first", ascending=False)
        data["_bad_rank"] = data.groupby("prediction_date")["bad_entry_probability_10d"].rank(method="first", ascending=False)
        future_rank = _numeric(data.get("relative_future_utility_rank_in_day"))
        return {
            "rank_bounds_valid": rank_bounds_valid,
            "expected_return_rank_corr_to_future_rank": self._corr(data["_expected_rank"], future_rank),
            "risk_adjusted_rank_corr_to_future_rank": self._corr(data["_risk_rank"], future_rank),
            "bad_entry_rank_corr_to_bottom_decile": self._corr(data["_bad_rank"], _numeric(data.get("bottom_decile_future_utility_in_day"))),
            "candidate_strength_corr_to_future_utility": self._corr(data.get("candidate_strength"), data.get("downside_penalized_return_10d")),
            "gap_to_best_corr_to_future_utility": self._corr(data.get("gap_to_best"), data.get("downside_penalized_return_10d")),
            "relative_feature_usefulness": "medium" if abs(self._corr(data.get("candidate_strength"), data.get("downside_penalized_return_10d")) or 0.0) >= 0.03 else "low",
        }

    def _sizing_label_candidates(self, dataset: pd.DataFrame) -> dict[str, Any]:
        if dataset.empty:
            return {"distribution": [], "columns": []}
        utility = _numeric(dataset.get("downside_penalized_return_10d"))
        percentile = _numeric(dataset.get("relative_future_utility_percentile_in_day"))
        mae = _numeric(dataset.get("max_adverse_excursion_10d"))
        bottom = dataset.get("bottom_decile_future_utility_in_day", pd.Series(False, index=dataset.index)).astype(bool)
        labels = pd.Series("pm_v3_size_label_100_candidate", index=dataset.index)
        labels.loc[(percentile >= 0.90) & (mae >= -0.03) & utility.gt(0)] = "pm_v3_size_label_130_candidate"
        labels.loc[(percentile >= 0.70) & labels.eq("pm_v3_size_label_100_candidate")] = "pm_v3_size_label_115_candidate"
        labels.loc[(bottom | (mae <= -0.06))] = "pm_v3_size_label_060_candidate"
        labels.loc[(percentile <= 0.30) & labels.eq("pm_v3_size_label_100_candidate")] = "pm_v3_size_label_080_candidate"
        counts = labels.value_counts().sort_index()
        return {
            "rule": "1.30 if top utility and MAE >= -3%; 1.15 upper utility; 1.00 middle; 0.80 lower; 0.60 bottom/large downside",
            "distribution": [{"label": str(label), "count": int(count), "rate": float(count / len(labels))} for label, count in counts.items()],
            "columns": sorted(counts.index.astype(str).tolist()),
        }

    def _leakage_audit(self, feature_columns: list[str], corr: dict[str, Any]) -> dict[str, Any]:
        forbidden = [column for column in feature_columns if self._has_forbidden_token(column)]
        label_like = [column for column in feature_columns if self._is_label_like(column)]
        suspicious = corr.get("suspicious_high_correlation_features", [])
        risk = "high" if forbidden or label_like else "medium" if suspicious else "low"
        return {
            "forbidden_feature_count": len(forbidden),
            "forbidden_feature_columns": forbidden,
            "suspicious_high_correlation_features": suspicious,
            "label_columns_in_features": label_like,
            "leakage_risk": risk,
        }

    def _verdict(self, quality: dict[str, Any], labels: list[dict[str, Any]], corr: dict[str, Any], market: dict[str, Any], ranking: dict[str, Any], sizing: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        summary = quality["summary"]
        blocking = bool(leakage["forbidden_feature_count"] or leakage["label_columns_in_features"] or summary["duplicate_key_count"] or summary["infinite_value_count"])
        label_null = {row["label"]: row["null_rate"] for row in labels}
        target_null = label_null.get("downside_penalized_return_10d")
        target_ok = (1.0 if target_null is None else target_null) < 0.10
        return {
            "dataset_is_trainable": bool(not blocking and target_ok),
            "recommended_target_label": "downside_penalized_return_10d",
            "recommended_layer2_label": "relative_future_utility_percentile_in_day",
            "recommended_layer3_label": "downside_penalized_return_10d plus pm_v3_size_label_* candidate buckets",
            "features_to_drop": quality.get("high_missing_features", []),
            "features_to_keep": "all non-forbidden non-label features except high-missing features after trainer-side imputation/drop policy",
            "market_regime_usefulness": market.get("market_regime_usefulness"),
            "relative_feature_usefulness": ranking.get("relative_feature_usefulness"),
            "next_phase_recommendation": "Phase 9-D: PM AI v3 Trainer Prototype" if not blocking and target_ok else "Phase 9-B2: Dataset Builder Fix",
        }

    def _market_group_row(self, name: str, group: pd.DataFrame, key: str = "market_regime_class_prototype") -> dict[str, Any]:
        return {
            key: name,
            "row_count": int(len(group)),
            "future_10d_return_mean": _safe_mean(group.get("future_10d_return")),
            "downside_penalized_return_10d_mean": _safe_mean(group.get("downside_penalized_return_10d")),
            "top_decile_rate": _safe_mean(group.get("top_decile_future_utility_in_day")),
            "bottom_decile_rate": _safe_mean(group.get("bottom_decile_future_utility_in_day")),
        }

    def _null_rates(self, dataset: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
        rows = [{"column": column, "null_rate": float(dataset[column].isna().mean())} for column in columns if column in dataset.columns]
        return sorted(rows, key=lambda row: (-row["null_rate"], row["column"]))

    def _date_coverage_gaps(self, dates: pd.Series) -> list[dict[str, Any]]:
        unique = sorted(pd.to_datetime(dates.dropna().unique()))
        gaps = []
        for prev, cur in zip(unique, unique[1:]):
            days = int((cur - prev).days)
            if days > 5:
                gaps.append({"previous_date": prev.strftime("%Y-%m-%d"), "next_date": cur.strftime("%Y-%m-%d"), "calendar_gap_days": days})
        return gaps

    def _corr(self, left: pd.Series | None, right: pd.Series | None) -> float | None:
        frame = pd.DataFrame({"left": _numeric(left), "right": _numeric(right)}).dropna()
        if len(frame) < 5 or frame["left"].nunique() <= 1 or frame["right"].nunique() <= 1:
            return None
        value = frame["left"].corr(frame["right"])
        return None if pd.isna(value) else float(value)

    def _corr_text(self, rows: list[dict[str, Any]]) -> str:
        return ", ".join(f"{row['feature']}:{row['correlation']:.3f}" for row in rows)

    def _empty_summary(self) -> dict[str, Any]:
        return {"row_count": 0, "date_min": None, "date_max": None, "code_count": 0, "feature_count": 0, "label_count": 0, "duplicate_key_count": 0, "infinite_value_count": 0, "constant_feature_count": 0, "high_missing_feature_count": 0, "date_coverage_gap_count": 0}

    def _has_forbidden_token(self, column: str) -> bool:
        lowered = column.lower()
        return any(token in lowered for token in FORBIDDEN_TOKENS)

    def _is_label_like(self, column: str) -> bool:
        lowered = column.lower()
        return lowered.startswith("future_") or "label" in lowered or "target" in lowered or column in LABEL_COLUMNS

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(self._format(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, list):
            return ", ".join(str(item) for item in value[:12])
        if isinstance(value, dict):
            return ", ".join(f"{key}:{val}" for key, val in value.items())
        return str(value).replace("\n", " ")


def build_phase9c_pm_ai_v3_dataset_audit(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3DatasetQualityAudit(root).build_report()
