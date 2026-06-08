"""Phase 9-D2B PM AI v3 mapping stability audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.portfolio_manager_v3_dataset_builder import FORBIDDEN_TOKENS, LABEL_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase9d2b_mapping_stability_audit_2023-01_to_2026-05"
DATASET_PATH = Path("data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_pm_sizing_universe_2023-01_to_2026-05.parquet")
MODEL_DIR = Path("models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe")
PREDICTION_COLUMNS = ["pm_v3_rank_score_pred", "pm_v3_downside_utility_pred", "pm_v3_top_utility_proba"]
MAPPINGS = [
    "mapping_a_rank_score_only",
    "mapping_b_downside_utility_only",
    "mapping_c_rank_plus_downside_blend",
    "mapping_d_conservative_high_conviction",
    "mapping_e_classifier_gate",
]


@dataclass(frozen=True)
class Phase9D2BPaths:
    markdown: Path
    json: Path


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if getattr(series, "dtype", None) == bool:
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _mean(series: pd.Series | None) -> float | None:
    values = _numeric(series).dropna()
    return float(values.mean()) if not values.empty else None


class PMAIV3MappingStabilityAudit:
    def __init__(self, root: Path | str = ROOT, *, dataset_path: Path | None = None, model_dir: Path | None = None) -> None:
        self.root = Path(root)
        self.dataset_path = self._root(dataset_path or DATASET_PATH)
        self.model_dir = self._root(model_dir or MODEL_DIR)

    def build_report(self) -> dict[str, Any]:
        feature_columns = self._load_feature_columns()
        dataset = self._load_dataset(feature_columns)
        scored = self._attach_predictions(dataset, feature_columns)
        mappings = self._build_mappings(scored)
        yearly = self._group_audit(scored, mappings, "year", self._year_groups(scored))
        regime = self._group_audit(scored, mappings, "market_regime", self._regime_groups(scored))
        rolling = self._group_audit(scored, mappings, "half_year", self._half_year_groups(scored))
        stability = self._stability_scores(yearly)
        leakage = self._leakage(feature_columns)
        conclusion = self._conclusion(stability, yearly, leakage)
        return {
            "metadata": {
                "phase": "9-D2B",
                "audit_only": True,
                "training_executed": False,
                "mapping_adjustment_executed": False,
                "strategy_backtest_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
            },
            "input_paths": {"dataset": str(self.dataset_path), "model_dir": str(self.model_dir)},
            "feature_columns": feature_columns,
            "prediction_columns": PREDICTION_COLUMNS,
            "mapping_names": MAPPINGS,
            "dataset_summary": self._dataset_summary(scored),
            "yearly_results": yearly,
            "market_regime_results": regime,
            "rolling_results": rolling,
            "stability_score": stability,
            "conclusion": conclusion,
            "leakage_checklist": leakage,
        }

    def save_report(self, report: dict[str, Any]) -> Phase9D2BPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9D2BPaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# PM AI v3 Phase 9-D2B Mapping Stability Audit",
                "",
                "## Scope",
                "",
                "- audit only; no retraining, no remapping, no strategy integration/backtest",
                "- future returns are used only as labels/evaluation metrics",
                "",
                "## Dataset",
                "",
                self._table([report["dataset_summary"]], ["row_count", "date_min", "date_max", "code_count"]),
                "",
                "## Yearly Results",
                "",
                self._table(report["yearly_results"], ["mapping", "group", "pm130_count", "pm130_downside_mean", "overall_downside_mean", "delta", "pm130_gt_pm115", "pm130_gt_pm100", "pm130_gt_pm080", "pm060_downside_mean"]),
                "",
                "## Market Regime Results",
                "",
                self._table(report["market_regime_results"], ["mapping", "group", "pm130_count", "pm130_downside_mean", "overall_downside_mean", "delta", "pm130_gt_pm115", "pm130_gt_pm100", "pm130_gt_pm080", "pm060_downside_mean"]),
                "",
                "## Rolling Results",
                "",
                self._table(report["rolling_results"], ["mapping", "group", "pm130_count", "pm130_downside_mean", "overall_downside_mean", "delta"]),
                "",
                "## Stability Score",
                "",
                self._table(report["stability_score"], ["mapping", "yearly_positive_count", "yearly_negative_count", "average_delta", "worst_year_delta", "best_year_delta", "consistency_score"]),
                "",
                "## Conclusion",
                "",
                self._table([report["conclusion"]], ["best_mapping_by_stability", "best_mapping_by_performance", "mapping_d_is_stable", "phase9e2_integration_audit_worth_testing"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_checklist"]], ["forbidden_feature_count", "forbidden_feature_columns", "label_columns_in_features", "future_columns_in_features", "leakage_risk"]),
                "",
            ]
        )

    def _root(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def _load_feature_columns(self) -> list[str]:
        path = self.model_dir / "feature_columns.json"
        return [str(column) for column in json.loads(path.read_text(encoding="utf-8"))]

    def _load_dataset(self, feature_columns: list[str]) -> pd.DataFrame:
        columns = [
            "prediction_date",
            "code",
            "market_regime_key",
            "downside_penalized_return_10d",
            "relative_future_utility_percentile_in_day",
            *feature_columns,
        ]
        available = set(pd.read_parquet(self.dataset_path).columns)
        return pd.read_parquet(self.dataset_path, columns=[column for column in columns if column in available])

    def _load_models(self) -> dict[str, Any]:
        return {
            "rank": joblib.load(self.model_dir / "model_a_candidate_ranking_regressor.joblib"),
            "downside": joblib.load(self.model_dir / "model_b_downside_utility_regressor.joblib"),
            "top": joblib.load(self.model_dir / "model_c_top_utility_classifier.joblib"),
        }

    def _attach_predictions(self, dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        if dataset.empty:
            return dataset.copy()
        out = dataset.copy()
        models = self._load_models()
        x = out[features]
        out["pm_v3_rank_score_pred"] = models["rank"].predict(x)
        out["pm_v3_downside_utility_pred"] = models["downside"].predict(x)
        out["pm_v3_top_utility_proba"] = models["top"].predict_proba(x)[:, 1] if hasattr(models["top"], "predict_proba") else models["top"].predict(x)
        out["prediction_date"] = pd.to_datetime(out["prediction_date"], errors="coerce")
        out["year"] = out["prediction_date"].dt.year.astype("Int64").astype(str)
        out["half_year"] = out["prediction_date"].dt.year.astype("Int64").astype(str) + "H" + (((out["prediction_date"].dt.month - 1) // 6) + 1).astype("Int64").astype(str)
        out["market_regime_key"] = out.get("market_regime_key", pd.Series("unknown", index=out.index)).fillna("unknown").astype(str)
        return out

    def _build_mappings(self, scored: pd.DataFrame) -> dict[str, pd.Series]:
        rank_pct = _numeric(scored.get("pm_v3_rank_score_pred")).rank(method="first", pct=True)
        utility_pct = _numeric(scored.get("pm_v3_downside_utility_pred")).rank(method="first", pct=True)
        top_pct = _numeric(scored.get("pm_v3_top_utility_proba")).rank(method="first", pct=True)
        blend = 0.5 * rank_pct + 0.5 * utility_pct
        return {
            "mapping_a_rank_score_only": self._quantile_mapping(rank_pct),
            "mapping_b_downside_utility_only": self._quantile_mapping(utility_pct),
            "mapping_c_rank_plus_downside_blend": self._quantile_mapping(blend),
            "mapping_d_conservative_high_conviction": self._mapping_d(rank_pct, utility_pct, top_pct),
            "mapping_e_classifier_gate": self._mapping_e(blend, top_pct),
        }

    def _quantile_mapping(self, pct: pd.Series) -> pd.Series:
        out = pd.Series(1.00, index=pct.index)
        out.loc[pct >= 0.90] = 1.30
        out.loc[(pct >= 0.75) & (pct < 0.90)] = 1.15
        out.loc[pct <= 0.25] = 0.80
        out.loc[pct <= 0.10] = 0.60
        return out

    def _mapping_d(self, rank_pct: pd.Series, utility_pct: pd.Series, top_pct: pd.Series) -> pd.Series:
        out = pd.Series(1.00, index=rank_pct.index)
        out.loc[(rank_pct >= 0.80) | (utility_pct >= 0.75)] = 1.15
        out.loc[(rank_pct >= 0.90) & (utility_pct >= 0.75) & (top_pct >= 0.60)] = 1.30
        out.loc[utility_pct <= 0.25] = 0.80
        out.loc[(utility_pct <= 0.10) | (top_pct <= 0.10)] = 0.60
        return out

    def _mapping_e(self, blend: pd.Series, top_pct: pd.Series) -> pd.Series:
        out = self._quantile_mapping(blend)
        out.loc[(out == 1.30) & (top_pct < 0.60)] = 1.15
        out.loc[(out == 1.15) & (top_pct < 0.40)] = 1.00
        out.loc[(blend <= 0.10) | (top_pct <= 0.10)] = 0.60
        return out

    def _year_groups(self, scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {year: group for year, group in scored.groupby("year") if year in {"2023", "2024", "2025", "2026"}}

    def _regime_groups(self, scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {regime: group for regime, group in scored.groupby("market_regime_key") if regime in {"attack", "neutral", "defensive"}}

    def _half_year_groups(self, scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {name: group for name, group in scored.groupby("half_year")}

    def _group_audit(self, scored: pd.DataFrame, mappings: dict[str, pd.Series], group_type: str, groups: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for mapping, multipliers in mappings.items():
            for group_name, group in groups.items():
                local_mult = multipliers.loc[group.index]
                rows.append({"mapping": mapping, "group_type": group_type, "group": group_name, **self._quality(group, local_mult)})
        return rows

    def _quality(self, group: pd.DataFrame, multipliers: pd.Series) -> dict[str, Any]:
        work = group.copy()
        work["_multiplier"] = multipliers
        overall = _mean(work.get("downside_penalized_return_10d"))
        by = {}
        for mult in [1.30, 1.15, 1.00, 0.80, 0.60]:
            subset = work[_numeric(work["_multiplier"]).round(2).eq(mult)]
            by[mult] = {
                "count": int(len(subset)),
                "downside": _mean(subset.get("downside_penalized_return_10d")),
            }
        pm130 = by[1.30]
        return {
            "row_count": int(len(work)),
            "pm130_count": pm130["count"],
            "pm130_downside_mean": pm130["downside"],
            "overall_downside_mean": overall,
            "delta": None if pm130["downside"] is None or overall is None else pm130["downside"] - overall,
            "pm130_gt_pm115": self._gt(pm130["downside"], by[1.15]["downside"]),
            "pm130_gt_pm100": self._gt(pm130["downside"], by[1.00]["downside"]),
            "pm130_gt_pm080": self._gt(pm130["downside"], by[0.80]["downside"]),
            "pm060_count": by[0.60]["count"],
            "pm060_downside_mean": by[0.60]["downside"],
        }

    def _gt(self, left: float | None, right: float | None) -> bool | None:
        return None if left is None or right is None else bool(left > right)

    def _stability_scores(self, yearly: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for mapping in MAPPINGS:
            deltas = [row.get("delta") for row in yearly if row["mapping"] == mapping and row.get("delta") is not None]
            positives = [delta for delta in deltas if delta > 0]
            negatives = [delta for delta in deltas if delta <= 0]
            consistency = len(positives) / len(deltas) if deltas else 0.0
            rows.append(
                {
                    "mapping": mapping,
                    "yearly_positive_count": len(positives),
                    "yearly_negative_count": len(negatives),
                    "average_delta": float(pd.Series(deltas).mean()) if deltas else None,
                    "worst_year_delta": float(min(deltas)) if deltas else None,
                    "best_year_delta": float(max(deltas)) if deltas else None,
                    "consistency_score": consistency,
                }
            )
        return rows

    def _conclusion(self, stability: list[dict[str, Any]], yearly: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        best_stability = max(stability, key=lambda row: (row.get("consistency_score") or 0.0, row.get("average_delta") or -10**9))
        best_perf = max(stability, key=lambda row: row.get("average_delta") or -10**9)
        d = next(row for row in stability if row["mapping"] == "mapping_d_conservative_high_conviction")
        d_year_rows = [row for row in yearly if row["mapping"] == "mapping_d_conservative_high_conviction"]
        d_beats_mid = all(bool(row.get("pm130_gt_pm115")) and bool(row.get("pm130_gt_pm100")) and bool(row.get("pm130_gt_pm080")) for row in d_year_rows if row.get("pm130_count", 0) > 0)
        stable = bool((d.get("consistency_score") or 0.0) >= 0.75 and (d.get("average_delta") or -1.0) > 0 and d_beats_mid)
        return {
            "best_mapping_by_stability": best_stability["mapping"],
            "best_mapping_by_performance": best_perf["mapping"],
            "mapping_d_is_stable": stable,
            "phase9e2_integration_audit_worth_testing": bool(stable and leakage["leakage_risk"] == "low"),
            "reason": "mapping_d has positive yearly delta and beats mid buckets consistently" if stable else "mapping_d stability gates were not fully satisfied",
        }

    def _leakage(self, feature_columns: list[str]) -> dict[str, Any]:
        forbidden = [column for column in feature_columns if any(token in column.lower() for token in FORBIDDEN_TOKENS)]
        labels = [column for column in feature_columns if column in LABEL_COLUMNS or "label" in column.lower() or "target" in column.lower()]
        future = [column for column in feature_columns if column.lower().startswith("future_")]
        return {
            "forbidden_feature_count": len(forbidden),
            "forbidden_feature_columns": forbidden,
            "label_columns_in_features": labels,
            "future_columns_in_features": future,
            "leakage_risk": "high" if forbidden or labels or future else "low",
            "backtest_artifacts_used_as_features": False,
            "current_pm_multiplier_used_as_label": False,
        }

    def _dataset_summary(self, scored: pd.DataFrame) -> dict[str, Any]:
        dates = pd.to_datetime(scored["prediction_date"], errors="coerce").dropna()
        return {
            "row_count": int(len(scored)),
            "date_min": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
            "date_max": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
            "code_count": int(scored["code"].nunique()) if "code" in scored.columns else 0,
        }

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
            return ", ".join(str(item) for item in value[:8])
        return str(value).replace("\n", " ")


def build_phase9d2b_mapping_stability_audit(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3MappingStabilityAudit(root).build_report()
