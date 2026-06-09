"""Phase 9-E2 PM AI v3 integration audit on the PM sizing universe.

This is a read-only audit. It attaches predictions from the Phase 9-D2 PM AI
v3 candidate models to the Phase 9-B3 PM sizing universe dataset and compares
candidate multiplier mappings. It does not run a strategy backtest, retrain
models, or overwrite current PM/Exit/profile artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.portfolio_manager_v3_dataset_builder import FORBIDDEN_TOKENS, LABEL_COLUMNS
from ml.portfolio_manager_v3_mapping_stability_audit import _mean, _numeric


ROOT = Path(__file__).resolve().parents[2]
PERIOD = "2023-01-01_to_2026-05-31"
REPORT_STEM = "phase9e2_pm_ai_v3_integration_pm_sizing_universe_2023-01_to_2026-05"
DATASET_PATH = Path("data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_pm_sizing_universe_2023-01_to_2026-05.parquet")
MODEL_DIR = Path("models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe")
PHASE9D3_REPORT = Path("reports/ml/phase9d3_mapping_threshold_audit_2023-01_to_2026-05.json")
V293_PROFILES = {
    "v2_93_a": "rookie_dealer_02_v2_93_pm_ai_v3_candidate",
    "v2_93_b": "rookie_dealer_02_v2_93b_pm_ai_v3_candidate_conservative",
    "v2_93_c": "rookie_dealer_02_v2_93c_pm_ai_v3_candidate_half_pm130",
}
PREDICTION_COLUMNS = [
    "pm_v3_rank_score_pred",
    "pm_v3_downside_utility_pred",
    "pm_v3_top_utility_proba",
    "pm_v3_score_blend",
]
EVALUATION_LABEL_COLUMNS = [
    "future_10d_return",
    "downside_penalized_return_10d",
    "relative_future_utility_percentile_in_day",
    "top_decile_future_utility_in_day",
    "bottom_decile_future_utility_in_day",
    "max_adverse_excursion_10d",
]
MAPPING_CONFIGS = {
    "e_139_classifier_gate_recommended": {
        "type": "classifier_gate_threshold",
        "classifier_gate_threshold": 0.80,
        "rank_threshold": 0.75,
        "downside_threshold": 0.80,
    },
    "e_140_classifier_gate_more_pm130": {
        "type": "classifier_gate_threshold",
        "classifier_gate_threshold": 0.80,
        "rank_threshold": 0.75,
        "downside_threshold": 0.75,
    },
    "e_120_classifier_gate_wider": {
        "type": "classifier_gate_threshold",
        "classifier_gate_threshold": 0.75,
        "rank_threshold": 0.75,
        "downside_threshold": 0.75,
    },
    "mapping_a_rank_score_only": {"type": "rank_score_only"},
    "mapping_c_rank_plus_downside_blend": {"type": "rank_plus_downside_blend"},
}


@dataclass(frozen=True)
class Phase9E2Paths:
    markdown: Path
    json: Path


def normalize_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(4) if digits else text


def normalize_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


class PMAIV3IntegrationAuditPMSizingUniverse:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        dataset_path: Path | None = None,
        model_dir: Path | None = None,
        phase9d3_report_path: Path | None = None,
        period: str = PERIOD,
        profiles: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.dataset_path = self._root(dataset_path or DATASET_PATH)
        self.model_dir = self._root(model_dir or MODEL_DIR)
        self.phase9d3_report_path = self._root(phase9d3_report_path or PHASE9D3_REPORT)
        self.period = period
        self.profiles = profiles or V293_PROFILES

    def build_report(self) -> dict[str, Any]:
        feature_columns = self._load_feature_columns()
        dataset = self._load_dataset(feature_columns)
        scored = self._attach_predictions(dataset, feature_columns)
        mappings = self._build_mappings(scored)
        overall = self._group_audit(scored, mappings, "all", {"all": scored})
        yearly = self._group_audit(scored, mappings, "year", self._year_groups(scored))
        half_year = self._group_audit(scored, mappings, "half_year", self._half_year_groups(scored))
        stability = self._stability_scores(yearly, overall)
        coverage = self._coverage_audit(scored)
        leakage = self._leakage(feature_columns)
        best = self._best_mapping(stability, overall, coverage, leakage)
        return {
            "metadata": {
                "phase": "9-E2",
                "audit_only": True,
                "prediction_attachment_only": True,
                "training_executed": False,
                "strategy_backtest_executed": False,
                "strategy_integration_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_model_regenerated_historical_predictions": False,
                "old_candidate_phase9d_model_used": False,
                "old_top10_fixed_dataset_used": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
            },
            "input_paths": {
                "dataset": str(self.dataset_path),
                "model_dir": str(self.model_dir),
                "phase9d3_report": str(self.phase9d3_report_path),
                "coverage_target_profiles": self.profiles,
            },
            "phase9d3_recommended_config": self._phase9d3_recommendation(),
            "dataset_summary": self._dataset_summary(scored),
            "feature_columns": feature_columns,
            "prediction_columns": PREDICTION_COLUMNS,
            "prediction_summary": self._prediction_summary(scored),
            "evaluation_label_columns": EVALUATION_LABEL_COLUMNS,
            "mapping_candidates": [{"mapping": name, **config} for name, config in MAPPING_CONFIGS.items()],
            "coverage_audit": coverage,
            "mapping_quality_overall": overall,
            "mapping_quality_yearly": yearly,
            "mapping_quality_half_year": half_year,
            "mapping_stability": stability,
            "best_mapping": best,
            "leakage_checklist": leakage,
        }

    def save_report(self, report: dict[str, Any]) -> Phase9E2Paths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9E2Paths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        best = report["best_mapping"]
        return "\n".join(
            [
                "# PM AI v3 Phase 9-E2 Integration Audit on PM Sizing Universe",
                "",
                "## Scope",
                "",
                "- prediction attachment and mapping audit on PM sizing universe only",
                "- no retraining, no strategy backtest, no current PM/Exit/v2_82 overwrite",
                "- Phase 9-F backtest artifacts are used only as coverage target keys",
                "",
                "## Coverage",
                "",
                self._table([report["coverage_audit"]], ["target_key_count", "matched_key_count", "coverage_rate", "missing_key_count", "top_missing_reason", "sample_missing_keys"]),
                "",
                "## Prediction Columns",
                "",
                ", ".join(report["prediction_columns"]),
                "",
                "## Mapping Candidates",
                "",
                self._table(report["mapping_candidates"], ["mapping", "type", "classifier_gate_threshold", "rank_threshold", "downside_threshold"]),
                "",
                "## Overall Mapping Quality",
                "",
                self._table(report["mapping_quality_overall"], ["mapping", "group", "pm130_count", "pm115_count", "pm100_count", "pm080_count", "pm060_count", "pm130_downside_mean", "overall_downside_mean", "delta", "pm130_gt_pm115", "pm130_gt_pm100", "pm130_gt_pm080", "pm060_downside_mean"]),
                "",
                "## Yearly Stability",
                "",
                self._table(report["mapping_stability"], ["mapping", "pm130_count", "pm130_2026_count", "average_delta", "worst_year_delta", "best_year_delta", "yearly_positive_count", "consistency_score"]),
                "",
                "## 2026 Firing Audit",
                "",
                self._table([row for row in report["mapping_quality_yearly"] if row.get("group") == "2026"], ["mapping", "pm130_count", "pm115_count", "pm100_count", "pm080_count", "pm060_count", "delta"]),
                "",
                "## Best Mapping",
                "",
                self._table([best], ["best_mapping_name", "best_mapping_pm130_count", "best_mapping_avg_delta", "best_mapping_consistency_score", "best_mapping_2026_pm130_count", "confidence_level", "phase9f2_backtest_worth_testing"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_checklist"]], ["forbidden_feature_count", "forbidden_feature_columns", "label_columns_in_features", "future_columns_in_features", "leakage_risk", "backtest_artifacts_used_for_coverage_target_only", "backtest_artifacts_used_as_features"]),
                "",
            ]
        )

    def _root(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def _load_feature_columns(self) -> list[str]:
        return [str(column) for column in json.loads((self.model_dir / "feature_columns.json").read_text(encoding="utf-8"))]

    def _load_dataset(self, feature_columns: list[str]) -> pd.DataFrame:
        requested = [
            "prediction_date",
            "code",
            *EVALUATION_LABEL_COLUMNS,
            *feature_columns,
        ]
        available = set(pd.read_parquet(self.dataset_path).columns)
        return pd.read_parquet(self.dataset_path, columns=[column for column in requested if column in available])

    def _load_models(self) -> dict[str, Any]:
        return {
            "rank": joblib.load(self.model_dir / "model_a_candidate_ranking_regressor.joblib"),
            "downside": joblib.load(self.model_dir / "model_b_downside_utility_regressor.joblib"),
            "top": joblib.load(self.model_dir / "model_c_top_utility_classifier.joblib"),
        }

    def _attach_predictions(self, dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        out = dataset.copy()
        models = self._load_models()
        x = out[features]
        out["pm_v3_rank_score_pred"] = models["rank"].predict(x)
        out["pm_v3_downside_utility_pred"] = models["downside"].predict(x)
        top = models["top"]
        out["pm_v3_top_utility_proba"] = top.predict_proba(x)[:, 1] if hasattr(top, "predict_proba") else top.predict(x)
        out["_rank_pct"] = _numeric(out["pm_v3_rank_score_pred"]).rank(method="first", pct=True)
        out["_downside_pct"] = _numeric(out["pm_v3_downside_utility_pred"]).rank(method="first", pct=True)
        out["_top_pct"] = _numeric(out["pm_v3_top_utility_proba"]).rank(method="first", pct=True)
        out["_blend_pct"] = 0.5 * out["_rank_pct"] + 0.5 * out["_downside_pct"]
        out["pm_v3_score_blend"] = 0.5 * out["pm_v3_rank_score_pred"] + 0.5 * out["pm_v3_downside_utility_pred"]
        out["prediction_date"] = pd.to_datetime(out["prediction_date"], errors="coerce")
        out["year"] = out["prediction_date"].dt.year.astype("Int64").astype(str)
        out["half_year"] = out["prediction_date"].dt.year.astype("Int64").astype(str) + "H" + (((out["prediction_date"].dt.month - 1) // 6) + 1).astype("Int64").astype(str)
        return out

    def _build_mappings(self, scored: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "e_139_classifier_gate_recommended": self._mapping_e_threshold(scored, 0.80, 0.75, 0.80),
            "e_140_classifier_gate_more_pm130": self._mapping_e_threshold(scored, 0.80, 0.75, 0.75),
            "e_120_classifier_gate_wider": self._mapping_e_threshold(scored, 0.75, 0.75, 0.75),
            "mapping_a_rank_score_only": self._quantile_mapping(scored["_rank_pct"]),
            "mapping_c_rank_plus_downside_blend": self._quantile_mapping(scored["_blend_pct"]),
        }

    def _mapping_e_threshold(self, scored: pd.DataFrame, gate: float, rank_t: float, down_t: float) -> pd.Series:
        out = pd.Series(1.00, index=scored.index)
        out.loc[(scored["_blend_pct"] >= 0.75) | (scored["_rank_pct"] >= rank_t)] = 1.15
        out.loc[(scored["_rank_pct"] >= rank_t) & (scored["_downside_pct"] >= down_t) & (scored["_top_pct"] >= gate)] = 1.30
        out.loc[scored["_blend_pct"] <= 0.25] = 0.80
        out.loc[(scored["_blend_pct"] <= 0.10) | (scored["_top_pct"] <= 0.10)] = 0.60
        return out

    def _quantile_mapping(self, pct: pd.Series) -> pd.Series:
        out = pd.Series(1.00, index=pct.index)
        out.loc[pct >= 0.90] = 1.30
        out.loc[(pct >= 0.75) & (pct < 0.90)] = 1.15
        out.loc[pct <= 0.25] = 0.80
        out.loc[pct <= 0.10] = 0.60
        return out

    def _group_audit(self, scored: pd.DataFrame, mappings: dict[str, pd.Series], group_type: str, groups: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for mapping, multipliers in mappings.items():
            for group_name, group in groups.items():
                rows.append({"mapping": mapping, "group_type": group_type, "group": group_name, **self._quality(group, multipliers.loc[group.index])})
        return rows

    def _quality(self, group: pd.DataFrame, multipliers: pd.Series) -> dict[str, Any]:
        work = group.copy()
        work["_multiplier"] = multipliers
        overall = _mean(work.get("downside_penalized_return_10d"))
        by = {}
        for mult in [1.30, 1.15, 1.00, 0.80, 0.60]:
            subset = work[_numeric(work["_multiplier"]).round(2).eq(mult)]
            by[mult] = {"count": int(len(subset)), "downside": _mean(subset.get("downside_penalized_return_10d"))}
        pm130 = by[1.30]
        return {
            "row_count": int(len(work)),
            "pm130_count": pm130["count"],
            "pm115_count": by[1.15]["count"],
            "pm100_count": by[1.00]["count"],
            "pm080_count": by[0.80]["count"],
            "pm060_count": by[0.60]["count"],
            "pm130_downside_mean": pm130["downside"],
            "overall_downside_mean": overall,
            "delta": None if pm130["downside"] is None or overall is None else pm130["downside"] - overall,
            "pm130_gt_pm115": self._gt(pm130["downside"], by[1.15]["downside"]),
            "pm130_gt_pm100": self._gt(pm130["downside"], by[1.00]["downside"]),
            "pm130_gt_pm080": self._gt(pm130["downside"], by[0.80]["downside"]),
            "pm060_downside_mean": by[0.60]["downside"],
        }

    def _year_groups(self, scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {year: group for year, group in scored.groupby("year") if year in {"2023", "2024", "2025", "2026"}}

    def _half_year_groups(self, scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {name: group for name, group in scored.groupby("half_year")}

    def _stability_scores(self, yearly: list[dict[str, Any]], overall: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for mapping in MAPPING_CONFIGS:
            year_rows = [row for row in yearly if row["mapping"] == mapping]
            deltas = [row.get("delta") for row in year_rows if row.get("delta") is not None]
            positives = [delta for delta in deltas if delta > 0]
            overall_row = next((row for row in overall if row["mapping"] == mapping), {})
            pm130_2026 = next((row.get("pm130_count", 0) for row in year_rows if row["group"] == "2026"), 0)
            rows.append(
                {
                    "mapping": mapping,
                    "pm130_count": int(overall_row.get("pm130_count", 0)),
                    "pm130_2026_count": int(pm130_2026),
                    "average_delta": float(pd.Series(deltas).mean()) if deltas else None,
                    "worst_year_delta": float(min(deltas)) if deltas else None,
                    "best_year_delta": float(max(deltas)) if deltas else None,
                    "yearly_positive_count": len(positives),
                    "yearly_negative_count": len(deltas) - len(positives),
                    "consistency_score": len(positives) / len(deltas) if deltas else 0.0,
                    "overall_delta": overall_row.get("delta"),
                    "pm130_downside_mean": overall_row.get("pm130_downside_mean"),
                    "pm060_downside_mean": overall_row.get("pm060_downside_mean"),
                }
            )
        return rows

    def _coverage_audit(self, scored: pd.DataFrame) -> dict[str, Any]:
        dataset_keys = self._dataset_keys(scored)
        targets = self._coverage_target_keys()
        if targets.empty:
            return {
                "target_key_count": 0,
                "matched_key_count": 0,
                "coverage_rate": None,
                "missing_key_count": 0,
                "top_missing_reason": "coverage_target_logs_missing",
                "sample_missing_keys": [],
            }
        target_keys = targets["lookup_key"].dropna()
        matched_mask = target_keys.isin(dataset_keys)
        matched = target_keys[matched_mask]
        missing = target_keys[~matched_mask]
        return {
            "target_key_count": int(len(target_keys)),
            "unique_target_key_count": int(target_keys.nunique()),
            "matched_key_count": int(len(matched)),
            "unique_matched_key_count": int(matched.nunique()),
            "coverage_rate": float(len(matched) / len(target_keys)) if len(target_keys) else None,
            "missing_key_count": int(len(missing)),
            "unique_missing_key_count": int(missing.nunique()),
            "top_missing_reason": "missing_prediction_date_code_in_pm_sizing_universe" if len(missing) else "",
            "sample_missing_keys": [str(key) for key in missing.head(10)],
            "backtest_artifacts_used_for_coverage_target_only": True,
        }

    def _coverage_target_keys(self) -> pd.DataFrame:
        frames = []
        for label, profile in self.profiles.items():
            path = self.root / "logs" / "backtests" / profile / self.period / "purchase_audit.csv"
            frame = _read_csv(path)
            if frame.empty:
                continue
            rows = frame.copy()
            pm_marker = rows.get("pm_model_version", pd.Series("", index=rows.index)).fillna("").astype(str).str.contains("pm_ai_v3")
            status_marker = rows.get("pm_status", pd.Series("", index=rows.index)).notna()
            missing_marker = rows.get("pm_missing_reason", pd.Series("", index=rows.index)).fillna("").astype(str).str.contains("pm_v3")
            rows = rows[pm_marker | status_marker | missing_marker].copy()
            if rows.empty:
                continue
            rows["profile_label"] = label
            rows["profile"] = profile
            rows["prediction_date"] = rows.get("signal_date", pd.Series("", index=rows.index)).map(normalize_date)
            rows["normalized_code"] = rows.get("code", pd.Series("", index=rows.index)).map(normalize_code)
            rows["lookup_key"] = rows["prediction_date"] + "|" + rows["normalized_code"]
            frames.append(rows[["profile_label", "profile", "lookup_key"]])
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=["lookup_key"])

    def _dataset_keys(self, scored: pd.DataFrame) -> set[str]:
        dates = pd.to_datetime(scored["prediction_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        codes = scored["code"].map(normalize_code)
        return set((dates + "|" + codes).dropna())

    def _best_mapping(
        self,
        stability: list[dict[str, Any]],
        overall: list[dict[str, Any]],
        coverage: dict[str, Any],
        leakage: dict[str, Any],
    ) -> dict[str, Any]:
        overall_by_mapping = {row["mapping"]: row for row in overall}
        candidates = []
        for row in stability:
            quality = overall_by_mapping.get(row["mapping"], {})
            gates = [
                (coverage.get("coverage_rate") or 0.0) >= 0.95,
                leakage["leakage_risk"] == "low",
                (quality.get("delta") or -10**9) > 0,
                bool(quality.get("pm130_gt_pm115")),
                bool(quality.get("pm130_gt_pm100")),
                bool(quality.get("pm130_gt_pm080")),
                (row.get("consistency_score") or 0.0) >= 0.75,
                (row.get("pm130_2026_count") or 0) > 0,
            ]
            if all(gates):
                candidates.append({**row, **quality})
        if not candidates:
            return {
                "best_mapping_name": None,
                "best_mapping_config": {},
                "best_mapping_pm130_count": 0,
                "best_mapping_avg_delta": None,
                "best_mapping_consistency_score": 0.0,
                "best_mapping_2026_pm130_count": 0,
                "confidence_level": "none",
                "phase9f2_backtest_worth_testing": False,
                "reason": "minimum gates were not satisfied",
            }
        best = sorted(
            candidates,
            key=lambda row: (
                row.get("consistency_score") or 0.0,
                row.get("average_delta") or -10**9,
                row.get("pm130_2026_count") or 0,
                row.get("pm130_count") or 0,
            ),
            reverse=True,
        )[0]
        confidence = "low" if int(best.get("pm130_2026_count") or 0) < 10 else "medium"
        return {
            "best_mapping_name": best["mapping"],
            "best_mapping_config": MAPPING_CONFIGS.get(best["mapping"], {}),
            "best_mapping_pm130_count": int(best.get("pm130_count", 0)),
            "best_mapping_avg_delta": best.get("average_delta"),
            "best_mapping_consistency_score": best.get("consistency_score"),
            "best_mapping_2026_pm130_count": int(best.get("pm130_2026_count", 0)),
            "confidence_level": confidence,
            "phase9f2_backtest_worth_testing": True,
            "reason": "2026 PM1.30 count is below 10, so confidence is low" if confidence == "low" else "minimum gates were satisfied",
        }

    def _leakage(self, features: list[str]) -> dict[str, Any]:
        forbidden = [f for f in features if any(token in f.lower() for token in FORBIDDEN_TOKENS)]
        labels = [f for f in features if f in LABEL_COLUMNS or "label" in f.lower() or "target" in f.lower()]
        future = [f for f in features if f.lower().startswith("future_")]
        return {
            "feature_columns": features,
            "prediction_columns": PREDICTION_COLUMNS,
            "evaluation_label_columns": EVALUATION_LABEL_COLUMNS,
            "forbidden_feature_count": len(forbidden),
            "forbidden_feature_columns": forbidden,
            "label_columns_in_features": labels,
            "future_columns_in_features": future,
            "leakage_risk": "high" if forbidden or labels or future else "low",
            "current_pm_ai_overwritten": False,
            "current_exit_ai_overwritten": False,
            "v2_82_profile_overwritten": False,
            "backtest_artifacts_used_for_coverage_target_only": True,
            "backtest_artifacts_used_as_features": False,
            "current_pm_multiplier_used_as_label": False,
        }

    def _phase9d3_recommendation(self) -> dict[str, Any]:
        payload = _read_json(self.phase9d3_report_path)
        return payload.get("recommended_threshold_config", {})

    def _dataset_summary(self, scored: pd.DataFrame) -> dict[str, Any]:
        dates = pd.to_datetime(scored["prediction_date"], errors="coerce").dropna()
        return {
            "row_count": int(len(scored)),
            "date_min": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
            "date_max": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
            "code_count": int(scored["code"].nunique()) if "code" in scored else 0,
        }

    def _prediction_summary(self, scored: pd.DataFrame) -> dict[str, Any]:
        return {
            column: {"mean": _mean(scored.get(column)), "min": self._min(scored.get(column)), "max": self._max(scored.get(column))}
            for column in PREDICTION_COLUMNS
            if column in scored.columns
        }

    def _min(self, series: pd.Series | None) -> float | None:
        values = _numeric(series).dropna()
        return float(values.min()) if not values.empty else None

    def _max(self, series: pd.Series | None) -> float | None:
        values = _numeric(series).dropna()
        return float(values.max()) if not values.empty else None

    def _gt(self, left: float | None, right: float | None) -> bool | None:
        return None if left is None or right is None else bool(left > right)

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
            return ", ".join(str(item) for item in value[:10])
        if isinstance(value, dict):
            return ", ".join(f"{key}={val}" for key, val in value.items())
        return str(value).replace("\n", " ")


def build_phase9e2_pm_ai_v3_integration_pm_sizing_universe(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3IntegrationAuditPMSizingUniverse(root).build_report()
