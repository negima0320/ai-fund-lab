"""Phase 9-E PM AI v3 candidate integration audit.

Read-only audit that attaches Phase 9-D candidate model predictions to the
Phase 9-B dataset and compares prototype multiplier mappings. It does not
integrate with strategy code, run a backtest, or overwrite current artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.portfolio_manager_v3_dataset_builder import FORBIDDEN_TOKENS, LABEL_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase9e_pm_ai_v3_integration_audit_2023-01_to_2026-05"
DATASET_PATH = Path("data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_2023-01_to_2026-05.parquet")
MODEL_DIR = Path("models/ml/portfolio_manager_v3/candidate_phase9d")
V282_PURCHASE_AUDIT = Path("reports/final/v2_82_cap38/core_2023-01_to_2026-05/purchase_audit.csv")
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
MAPPING_NAMES = [
    "mapping_a_rank_score_only",
    "mapping_b_downside_utility_only",
    "mapping_c_rank_plus_downside_blend",
    "mapping_d_conservative_high_conviction",
    "mapping_e_classifier_gate",
]


@dataclass(frozen=True)
class Phase9EPaths:
    markdown: Path
    json: Path


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if getattr(series, "dtype", None) == bool:
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _mean(series: pd.Series | None) -> float | None:
    values = _numeric(series).dropna()
    return None if values.empty else float(values.mean())


class PMAIV3CandidateIntegrationAudit:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        dataset_path: Path | None = None,
        model_dir: Path | None = None,
        v282_purchase_audit: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.dataset_path = self._root(dataset_path or DATASET_PATH)
        self.model_dir = self._root(model_dir or MODEL_DIR)
        self.v282_purchase_audit = self._root(v282_purchase_audit or V282_PURCHASE_AUDIT)

    def build_report(self) -> dict[str, Any]:
        dataset = _read_parquet(self.dataset_path)
        feature_columns = self._load_feature_columns()
        scored = self._attach_predictions(dataset, feature_columns)
        mappings = self._build_mappings(scored)
        mapping_quality = self._mapping_quality(scored, mappings)
        current_pm = self._current_pm_comparison(scored, mappings)
        leakage = self._leakage_guard(feature_columns)
        verdict = self._verdict(mapping_quality, leakage, current_pm)
        return {
            "metadata": {
                "phase": "9-E",
                "audit_only": True,
                "strategy_integration_executed": False,
                "backtest_executed": False,
                "training_executed": False,
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
                "model_dir": str(self.model_dir),
                "v2_82_purchase_audit": str(self.v282_purchase_audit),
            },
            "feature_columns": feature_columns,
            "prediction_columns": PREDICTION_COLUMNS,
            "evaluation_label_columns": EVALUATION_LABEL_COLUMNS,
            "prediction_summary": self._prediction_summary(scored),
            "mapping_candidates": list(mappings.keys()),
            "mapping_quality": mapping_quality,
            "current_pm_comparison": current_pm,
            "leakage_guard": leakage,
            "verdict": verdict,
        }

    def save_report(self, report: dict[str, Any]) -> Phase9EPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9EPaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        verdict = report["verdict"]
        best = report["mapping_quality"].get("best_mapping", {})
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 9-E PM AI v3 Candidate Integration Audit",
                "",
                "## Scope",
                "",
                "- prediction attachment and mapping audit only",
                "- no strategy integration, no backtest, no current PM/Exit/profile overwrite",
                "",
                "## Prediction Columns",
                "",
                ", ".join(report["prediction_columns"]),
                "",
                "## Mapping Summary",
                "",
                self._table(report["mapping_quality"]["summary_rows"], ["mapping", "split", "overall_downside_mean", "pm130_count", "pm130_downside_mean", "pm130_vs_overall_delta", "pm130_better_than_115_100_080", "pm060_downside_mean"]),
                "",
                "## Best Mapping",
                "",
                self._table([best], ["mapping", "pm130_count", "pm130_downside_mean", "overall_downside_mean", "test_pm130_downside_mean", "test_overall_downside_mean"]),
                "",
                "## Current PM Comparison",
                "",
                self._table([report["current_pm_comparison"]["summary"]], ["joined_rows", "current_pm130_count", "current_pm130_downside_mean", "best_pm130_overlap_count", "best_pm130_overlap_rate", "current_pm130_but_v3_low_downside_mean", "v3_pm130_but_current_low_downside_mean"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_guard"]], ["forbidden_feature_count", "forbidden_feature_columns", "label_columns_in_features", "leakage_risk"]),
                "",
                "## Verdict",
                "",
                self._table([verdict], ["pm_v3_mapping_viable", "best_mapping_name", "best_mapping_pm130_count", "best_mapping_pm130_actual_downside_mean", "overall_actual_downside_mean", "current_pm130_actual_downside_mean", "phase9f_backtest_worth_testing", "next_phase_recommendation"]),
                "",
            ]
        )

    def _root(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def _load_feature_columns(self) -> list[str]:
        path = self.model_dir / "feature_columns.json"
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [str(column) for column in payload]

    def _load_models(self) -> dict[str, Any]:
        files = {
            "rank": "model_a_candidate_ranking_regressor.joblib",
            "downside": "model_b_downside_utility_regressor.joblib",
            "top": "model_c_top_utility_classifier.joblib",
        }
        return {name: joblib.load(self.model_dir / filename) for name, filename in files.items()}

    def _attach_predictions(self, dataset: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        if dataset.empty or not feature_columns:
            return dataset.copy()
        models = self._load_models()
        out = dataset.copy()
        x = out[feature_columns]
        out["pm_v3_rank_score_pred"] = models["rank"].predict(x)
        out["pm_v3_downside_utility_pred"] = models["downside"].predict(x)
        if hasattr(models["top"], "predict_proba"):
            out["pm_v3_top_utility_proba"] = models["top"].predict_proba(x)[:, 1]
        else:
            out["pm_v3_top_utility_proba"] = models["top"].predict(x)
        out["pm_v3_score_blend"] = 0.5 * self._rank_pct(out["pm_v3_rank_score_pred"]) + 0.5 * self._rank_pct(out["pm_v3_downside_utility_pred"])
        return out

    def _build_mappings(self, scored: pd.DataFrame) -> dict[str, pd.Series]:
        if scored.empty:
            return {}
        return {
            "mapping_a_rank_score_only": self._quantile_mapping(scored["pm_v3_rank_score_pred"]),
            "mapping_b_downside_utility_only": self._quantile_mapping(scored["pm_v3_downside_utility_pred"]),
            "mapping_c_rank_plus_downside_blend": self._quantile_mapping(scored["pm_v3_score_blend"]),
            "mapping_d_conservative_high_conviction": self._mapping_d(scored),
            "mapping_e_classifier_gate": self._mapping_e(scored),
        }

    def _quantile_mapping(self, score: pd.Series) -> pd.Series:
        pct = self._rank_pct(score)
        out = pd.Series(1.00, index=score.index)
        out.loc[pct >= 0.90] = 1.30
        out.loc[(pct >= 0.75) & (pct < 0.90)] = 1.15
        out.loc[pct <= 0.25] = 0.80
        out.loc[pct <= 0.10] = 0.60
        return out

    def _mapping_d(self, scored: pd.DataFrame) -> pd.Series:
        rank_pct = self._rank_pct(scored["pm_v3_rank_score_pred"])
        down_pct = self._rank_pct(scored["pm_v3_downside_utility_pred"])
        top_pct = self._rank_pct(scored["pm_v3_top_utility_proba"])
        out = pd.Series(1.00, index=scored.index)
        out.loc[(rank_pct >= 0.80) | (down_pct >= 0.75)] = 1.15
        out.loc[(rank_pct >= 0.80) & (down_pct >= 0.80)] = 1.30
        out.loc[down_pct <= 0.25] = 0.80
        out.loc[(down_pct <= 0.10) | (top_pct <= 0.10)] = 0.60
        return out

    def _mapping_e(self, scored: pd.DataFrame) -> pd.Series:
        blend = self._rank_pct(scored["pm_v3_score_blend"])
        top = self._rank_pct(scored["pm_v3_top_utility_proba"])
        out = self._quantile_mapping(scored["pm_v3_score_blend"])
        out.loc[(out == 1.30) & (top < 0.60)] = 1.15
        out.loc[(out == 1.15) & (top < 0.40)] = 1.00
        out.loc[(blend <= 0.10) | (top <= 0.10)] = 0.60
        return out

    def _mapping_quality(self, scored: pd.DataFrame, mappings: dict[str, pd.Series]) -> dict[str, Any]:
        rows = []
        details = {}
        for name, multipliers in mappings.items():
            scored_name = scored.copy()
            scored_name["_mapping_multiplier"] = multipliers
            detail_rows = []
            for split, group in self._split_groups(scored_name).items():
                overall = _mean(group.get("downside_penalized_return_10d"))
                by_mult = []
                for multiplier, mgroup in group.groupby("_mapping_multiplier"):
                    by_mult.append(self._multiplier_row(float(multiplier), mgroup, overall))
                pm130 = next((row for row in by_mult if row["multiplier"] == 1.30), {})
                pm060 = next((row for row in by_mult if row["multiplier"] == 0.60), {})
                rows.append(
                    {
                        "mapping": name,
                        "split": split,
                        "overall_downside_mean": overall,
                        "pm130_count": pm130.get("row_count", 0),
                        "pm130_downside_mean": pm130.get("actual_downside_penalized_return_10d_mean"),
                        "pm130_vs_overall_delta": None if pm130.get("actual_downside_penalized_return_10d_mean") is None or overall is None else pm130["actual_downside_penalized_return_10d_mean"] - overall,
                        "pm130_better_than_115_100_080": self._pm130_beats_mid(by_mult),
                        "pm060_downside_mean": pm060.get("actual_downside_penalized_return_10d_mean"),
                    }
                )
                detail_rows.append({"split": split, "by_multiplier": by_mult})
            details[name] = detail_rows
        best = self._best_mapping(rows)
        return {"summary_rows": rows, "details": details, "best_mapping": best}

    def _multiplier_row(self, multiplier: float, group: pd.DataFrame, overall: float | None) -> dict[str, Any]:
        return {
            "multiplier": multiplier,
            "row_count": int(len(group)),
            "rate": float(len(group) / len(group.index.unique())) if len(group) else 0.0,
            "actual_future_10d_return_mean": _mean(group.get("future_10d_return")),
            "actual_downside_penalized_return_10d_mean": _mean(group.get("downside_penalized_return_10d")),
            "actual_relative_future_utility_percentile_mean": _mean(group.get("relative_future_utility_percentile_in_day")),
            "top_decile_future_utility_rate": _mean(group.get("top_decile_future_utility_in_day")),
            "bottom_decile_future_utility_rate": _mean(group.get("bottom_decile_future_utility_in_day")),
            "max_adverse_excursion_10d_mean": _mean(group.get("max_adverse_excursion_10d")),
            "vs_overall_downside_delta": None if overall is None else (_mean(group.get("downside_penalized_return_10d")) or 0.0) - overall,
        }

    def _best_mapping(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        test_rows = [row for row in rows if row["split"] == "test" and row.get("pm130_count", 0) >= 5 and row.get("pm130_downside_mean") is not None]
        if not test_rows:
            return {}
        best_test = max(test_rows, key=lambda row: row.get("pm130_vs_overall_delta") or -10**9)
        all_row = next((row for row in rows if row["mapping"] == best_test["mapping"] and row["split"] == "all"), {})
        return {
            "mapping": best_test["mapping"],
            "pm130_count": all_row.get("pm130_count"),
            "pm130_downside_mean": all_row.get("pm130_downside_mean"),
            "overall_downside_mean": all_row.get("overall_downside_mean"),
            "test_pm130_count": best_test.get("pm130_count"),
            "test_pm130_downside_mean": best_test.get("pm130_downside_mean"),
            "test_overall_downside_mean": best_test.get("overall_downside_mean"),
        }

    def _current_pm_comparison(self, scored: pd.DataFrame, mappings: dict[str, pd.Series]) -> dict[str, Any]:
        purchase = _read_csv(self.v282_purchase_audit)
        if purchase.empty or scored.empty or not mappings:
            return {"summary": {"joined_rows": 0}, "distribution": []}
        purchase = purchase.copy()
        purchase["prediction_date"] = pd.to_datetime(purchase["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        purchase["code"] = purchase["code"].astype(str)
        current = purchase[["prediction_date", "code", "pm_multiplier"]].dropna(subset=["pm_multiplier"]).drop_duplicates(["prediction_date", "code"])
        best_name = self._best_mapping(self._mapping_quality(scored, mappings)["summary_rows"]).get("mapping", next(iter(mappings)))
        scored_with_mapping = scored.copy()
        scored_with_mapping["_v3_multiplier"] = mappings[best_name]
        joined = scored_with_mapping.merge(current, on=["prediction_date", "code"], how="inner")
        current130 = joined[_numeric(joined.get("pm_multiplier")).round(2).eq(1.30)]
        v3130 = joined[_numeric(joined.get("_v3_multiplier")).round(2).eq(1.30)]
        overlap = current130.index.intersection(v3130.index)
        current_low = joined[_numeric(joined.get("pm_multiplier")).round(2).ge(1.30) & _numeric(joined.get("_v3_multiplier")).le(0.80)]
        v3_low = joined[_numeric(joined.get("_v3_multiplier")).round(2).ge(1.30) & _numeric(joined.get("pm_multiplier")).le(0.80)]
        return {
            "summary": {
                "joined_rows": int(len(joined)),
                "current_pm130_count": int(len(current130)),
                "current_pm130_downside_mean": _mean(current130.get("downside_penalized_return_10d")),
                "best_mapping_name": best_name,
                "best_pm130_count": int(len(v3130)),
                "best_pm130_downside_mean": _mean(v3130.get("downside_penalized_return_10d")),
                "best_pm130_overlap_count": int(len(overlap)),
                "best_pm130_overlap_rate": float(len(overlap) / len(v3130)) if len(v3130) else None,
                "current_pm130_but_v3_low_downside_mean": _mean(current_low.get("downside_penalized_return_10d")),
                "v3_pm130_but_current_low_downside_mean": _mean(v3_low.get("downside_penalized_return_10d")),
            },
            "distribution": self._distribution(joined.get("pm_multiplier")),
        }

    def _leakage_guard(self, feature_columns: list[str]) -> dict[str, Any]:
        forbidden = [column for column in feature_columns if self._has_forbidden_token(column)]
        label_like = [column for column in feature_columns if column in LABEL_COLUMNS or column.startswith("future_") or "label" in column or "target" in column]
        return {
            "forbidden_feature_count": len(forbidden),
            "forbidden_feature_columns": forbidden,
            "label_columns_in_features": label_like,
            "leakage_risk": "high" if forbidden or label_like else "low",
        }

    def _verdict(self, quality: dict[str, Any], leakage: dict[str, Any], current_pm: dict[str, Any]) -> dict[str, Any]:
        best = quality.get("best_mapping", {})
        rows = quality.get("summary_rows", [])
        best_name = best.get("mapping")
        test = next((row for row in rows if row["mapping"] == best_name and row["split"] == "test"), {})
        all_row = next((row for row in rows if row["mapping"] == best_name and row["split"] == "all"), {})
        conditions = [
            leakage["leakage_risk"] == "low",
            (all_row.get("pm130_vs_overall_delta") or -10**9) > 0,
            bool(all_row.get("pm130_better_than_115_100_080")),
            (test.get("pm130_vs_overall_delta") or -10**9) > 0,
            (test.get("pm130_count") or 0) >= 5,
        ]
        viable = all(conditions)
        return {
            "pm_v3_mapping_viable": viable,
            "best_mapping_name": best_name,
            "best_mapping_pm130_count": best.get("pm130_count"),
            "best_mapping_pm130_actual_downside_mean": best.get("pm130_downside_mean"),
            "overall_actual_downside_mean": best.get("overall_downside_mean"),
            "current_pm130_actual_downside_mean": current_pm.get("summary", {}).get("current_pm130_downside_mean"),
            "phase9f_backtest_worth_testing": viable,
            "next_phase_recommendation": "Phase 9-F PM AI v3 Backtest Candidate" if viable else "Phase 9-D2 mapping/model redesign",
            "strict_conditions_passed": conditions,
        }

    def _prediction_summary(self, scored: pd.DataFrame) -> dict[str, Any]:
        return {column: {"mean": _mean(scored.get(column)), "min": float(_numeric(scored.get(column)).min()), "max": float(_numeric(scored.get(column)).max())} for column in PREDICTION_COLUMNS if column in scored.columns}

    def _split_groups(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        dates = pd.to_datetime(frame["prediction_date"], errors="coerce")
        return {
            "all": frame,
            "train": frame[(dates >= "2023-01-04") & (dates <= "2024-12-31")],
            "validation": frame[(dates >= "2025-01-01") & (dates <= "2025-12-31")],
            "test": frame[(dates >= "2026-01-01") & (dates <= "2026-04-27")],
        }

    def _pm130_beats_mid(self, rows: list[dict[str, Any]]) -> bool:
        by = {row["multiplier"]: row.get("actual_downside_penalized_return_10d_mean") for row in rows}
        pm130 = by.get(1.30)
        if pm130 is None:
            return False
        return all(pm130 > by.get(mult, -10**9) for mult in [1.15, 1.00, 0.80] if by.get(mult) is not None)

    def _rank_pct(self, score: pd.Series) -> pd.Series:
        return _numeric(score).rank(method="first", pct=True)

    def _distribution(self, series: pd.Series | None) -> list[dict[str, Any]]:
        values = _numeric(series).dropna().round(2)
        counts = values.value_counts().sort_index()
        total = int(counts.sum())
        return [{"value": float(key), "count": int(count), "rate": float(count / total) if total else 0.0} for key, count in counts.items()]

    def _has_forbidden_token(self, column: str) -> bool:
        lowered = column.lower()
        return any(token in lowered for token in FORBIDDEN_TOKENS)

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
        return str(value).replace("\n", " ")


def build_phase9e_pm_ai_v3_integration_audit(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3CandidateIntegrationAudit(root).build_report()
