"""Phase 9-D3 PM AI v3 mapping E threshold optimization audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.portfolio_manager_v3_dataset_builder import FORBIDDEN_TOKENS, LABEL_COLUMNS
from ml.portfolio_manager_v3_mapping_stability_audit import DATASET_PATH, MODEL_DIR, ROOT, _mean, _numeric


REPORT_STEM = "phase9d3_mapping_threshold_audit_2023-01_to_2026-05"
RANK_THRESHOLDS = [0.95, 0.90, 0.85, 0.80, 0.75]
DOWNSIDE_THRESHOLDS = [0.90, 0.85, 0.80, 0.75]
CLASSIFIER_GATE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


@dataclass(frozen=True)
class Phase9D3Paths:
    markdown: Path
    json: Path


class PMAIV3MappingThresholdAudit:
    def __init__(self, root: Path | str = ROOT, *, dataset_path: Path | None = None, model_dir: Path | None = None) -> None:
        self.root = Path(root)
        self.dataset_path = self._root(dataset_path or DATASET_PATH)
        self.model_dir = self._root(model_dir or MODEL_DIR)

    def build_report(self) -> dict[str, Any]:
        feature_columns = self._load_feature_columns()
        scored = self._attach_predictions(self._load_dataset(feature_columns), feature_columns)
        rows, yearly = self._grid_search(scored)
        stability = self._stability(rows, yearly)
        top = self._top_candidates(stability)
        recommendation = top[0] if top else {}
        leakage = self._leakage(feature_columns)
        return {
            "metadata": {
                "phase": "9-D3",
                "audit_only": True,
                "training_executed": False,
                "strategy_backtest_executed": False,
                "mapping_adjustment_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
            },
            "input_paths": {"dataset": str(self.dataset_path), "model_dir": str(self.model_dir)},
            "threshold_grid": {
                "classifier_gate_thresholds": CLASSIFIER_GATE_THRESHOLDS,
                "rank_thresholds": RANK_THRESHOLDS,
                "downside_thresholds": DOWNSIDE_THRESHOLDS,
                "candidate_count": len(rows),
            },
            "grid_results": rows,
            "yearly_results": yearly,
            "stability_scores": stability,
            "top10_configs": top[:10],
            "recommended_threshold_config": recommendation,
            "phase9e2_integration_audit_worth_testing": bool(recommendation and leakage["leakage_risk"] == "low"),
            "leakage_checklist": leakage,
        }

    def save_report(self, report: dict[str, Any]) -> Phase9D3Paths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9D3Paths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# PM AI v3 Phase 9-D3 Mapping Threshold Audit",
                "",
                "## Scope",
                "",
                "- mapping_e_classifier_gate threshold audit only",
                "- no retraining, no strategy integration, no backtest",
                "",
                "## Grid",
                "",
                self._table([report["threshold_grid"]], ["candidate_count", "classifier_gate_thresholds", "rank_thresholds", "downside_thresholds"]),
                "",
                "## Top 10 Configs",
                "",
                self._table(report["top10_configs"], ["config_id", "classifier_gate_threshold", "rank_threshold", "downside_threshold", "pm130_count", "average_delta", "worst_year_delta", "consistency_score", "pm130_2026_count"]),
                "",
                "## Recommended",
                "",
                self._table([report["recommended_threshold_config"]], ["config_id", "classifier_gate_threshold", "rank_threshold", "downside_threshold", "pm130_count", "average_delta", "consistency_score", "pm130_2026_count"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_checklist"]], ["forbidden_feature_count", "label_columns_in_features", "future_columns_in_features", "leakage_risk"]),
                "",
            ]
        )

    def _root(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def _load_feature_columns(self) -> list[str]:
        return [str(column) for column in json.loads((self.model_dir / "feature_columns.json").read_text(encoding="utf-8"))]

    def _load_dataset(self, feature_columns: list[str]) -> pd.DataFrame:
        cols = ["prediction_date", "code", "downside_penalized_return_10d", *feature_columns]
        available = set(pd.read_parquet(self.dataset_path).columns)
        return pd.read_parquet(self.dataset_path, columns=[c for c in cols if c in available])

    def _attach_predictions(self, dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        out = dataset.copy()
        x = out[features]
        rank = joblib.load(self.model_dir / "model_a_candidate_ranking_regressor.joblib")
        downside = joblib.load(self.model_dir / "model_b_downside_utility_regressor.joblib")
        top = joblib.load(self.model_dir / "model_c_top_utility_classifier.joblib")
        out["_rank_pct"] = pd.Series(rank.predict(x), index=out.index).rank(method="first", pct=True)
        out["_downside_pct"] = pd.Series(downside.predict(x), index=out.index).rank(method="first", pct=True)
        top_pred = top.predict_proba(x)[:, 1] if hasattr(top, "predict_proba") else top.predict(x)
        out["_top_pct"] = pd.Series(top_pred, index=out.index).rank(method="first", pct=True)
        out["_blend_pct"] = 0.5 * out["_rank_pct"] + 0.5 * out["_downside_pct"]
        out["prediction_date"] = pd.to_datetime(out["prediction_date"], errors="coerce")
        out["year"] = out["prediction_date"].dt.year.astype("Int64").astype(str)
        return out

    def _grid_search(self, scored: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        yearly: list[dict[str, Any]] = []
        config_id = 0
        for gate in CLASSIFIER_GATE_THRESHOLDS:
            for rank_t in RANK_THRESHOLDS:
                for down_t in DOWNSIDE_THRESHOLDS:
                    config_id += 1
                    multipliers = self._mapping_e_threshold(scored, gate, rank_t, down_t)
                    cfg = {
                        "config_id": f"e_{config_id:03d}",
                        "classifier_gate_threshold": gate,
                        "rank_threshold": rank_t,
                        "downside_threshold": down_t,
                    }
                    rows.append({**cfg, **self._quality(scored, multipliers)})
                    for year, group in scored.groupby("year"):
                        if year not in {"2023", "2024", "2025", "2026"}:
                            continue
                        yearly.append({**cfg, "year": year, **self._quality(group, multipliers.loc[group.index])})
        return rows, yearly

    def _mapping_e_threshold(self, scored: pd.DataFrame, gate: float, rank_t: float, down_t: float) -> pd.Series:
        out = pd.Series(1.00, index=scored.index)
        blend = scored["_blend_pct"]
        top = scored["_top_pct"]
        down = scored["_downside_pct"]
        out.loc[(blend >= 0.75) | (rank_t <= scored["_rank_pct"])] = 1.15
        out.loc[(scored["_rank_pct"] >= rank_t) & (down >= down_t) & (top >= gate)] = 1.30
        out.loc[blend <= 0.25] = 0.80
        out.loc[(blend <= 0.10) | (top <= 0.10)] = 0.60
        return out

    def _quality(self, frame: pd.DataFrame, multipliers: pd.Series) -> dict[str, Any]:
        work = frame.copy()
        work["_multiplier"] = multipliers
        overall = _mean(work.get("downside_penalized_return_10d"))
        by = {}
        for mult in [1.30, 1.15, 1.00, 0.80, 0.60]:
            subset = work[_numeric(work["_multiplier"]).round(2).eq(mult)]
            by[mult] = {"count": int(len(subset)), "downside": _mean(subset.get("downside_penalized_return_10d"))}
        pm130 = by[1.30]
        return {
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

    def _stability(self, rows: list[dict[str, Any]], yearly: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        by_cfg = {row["config_id"]: row for row in rows}
        for config_id, overall in by_cfg.items():
            year_rows = [row for row in yearly if row["config_id"] == config_id]
            deltas = [row["delta"] for row in year_rows if row.get("delta") is not None]
            positives = [delta for delta in deltas if delta > 0]
            config = {
                key: overall[key]
                for key in ["config_id", "classifier_gate_threshold", "rank_threshold", "downside_threshold", "pm130_count"]
            }
            pm130_2026 = next((row.get("pm130_count", 0) for row in year_rows if row["year"] == "2026"), 0)
            out.append(
                {
                    **config,
                    "pm130_2026_count": int(pm130_2026),
                    "average_delta": float(pd.Series(deltas).mean()) if deltas else None,
                    "worst_year_delta": float(min(deltas)) if deltas else None,
                    "best_year_delta": float(max(deltas)) if deltas else None,
                    "yearly_positive_count": len(positives),
                    "consistency_score": len(positives) / len(deltas) if deltas else 0.0,
                    "overall_delta": overall.get("delta"),
                    "pm130_downside_mean": overall.get("pm130_downside_mean"),
                    "pm060_downside_mean": overall.get("pm060_downside_mean"),
                }
            )
        return out

    def _top_candidates(self, stability: list[dict[str, Any]]) -> list[dict[str, Any]]:
        eligible = [
            row
            for row in stability
            if row.get("pm130_count", 0) > 100
            and row.get("pm130_2026_count", 0) > 0
            and (row.get("consistency_score") or 0) >= 0.75
            and (row.get("average_delta") or -1) > 0
        ]
        return sorted(eligible, key=lambda row: (row.get("consistency_score") or 0, row.get("average_delta") or -1, row.get("pm130_2026_count") or 0), reverse=True)

    def _leakage(self, features: list[str]) -> dict[str, Any]:
        forbidden = [f for f in features if any(token in f.lower() for token in FORBIDDEN_TOKENS)]
        labels = [f for f in features if f in LABEL_COLUMNS or "label" in f.lower() or "target" in f.lower()]
        future = [f for f in features if f.lower().startswith("future_")]
        return {
            "forbidden_feature_count": len(forbidden),
            "forbidden_feature_columns": forbidden,
            "label_columns_in_features": labels,
            "future_columns_in_features": future,
            "leakage_risk": "high" if forbidden or labels or future else "low",
            "backtest_artifacts_used_as_features": False,
            "current_pm_multiplier_used_as_label": False,
        }

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
        return str(value).replace("\n", " ")


def build_phase9d3_mapping_threshold_audit(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3MappingThresholdAudit(root).build_report()
