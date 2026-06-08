"""Phase 9-D2 PM AI v3 trainer for the PM sizing universe dataset."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ml.portfolio_manager_v3_trainer import (
    MODEL_TARGETS,
    ROOT,
    PMAIV3TrainerPrototype,
    Phase9DTrainOptions,
    _numeric,
)


REPORT_STEM = "phase9d2_pm_ai_v3_trainer_pm_sizing_universe_2023-01_to_2026-05"
DATASET_PATH = Path("data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_pm_sizing_universe_2023-01_to_2026-05.parquet")
MODEL_DIR = Path("models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe")


@dataclass(frozen=True)
class Phase9D2TrainPaths:
    model_dir: Path
    markdown: Path
    json: Path


class PMAIV3PMSizingUniverseTrainer(PMAIV3TrainerPrototype):
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        dataset_path: Path | None = None,
        model_dir: Path | None = None,
        max_iter: int = 40,
    ) -> None:
        super().__init__(root, dataset_path=dataset_path or DATASET_PATH, model_dir=model_dir or MODEL_DIR)
        self.max_iter = int(max_iter)

    def run(self, options: Phase9DTrainOptions | None = None) -> dict[str, Any]:
        report = super().run(options or Phase9DTrainOptions(save_models=True, include_market_comparison=True))
        dataset = pd.read_parquet(self.dataset_path) if self.dataset_path.exists() else pd.DataFrame()
        report["metadata"].update(
            {
                "phase": "9-D2",
                "trainer_prototype": False,
                "pm_sizing_universe_retrain": True,
                "strategy_backtest_executed": False,
                "pm_ai_v3_retrained": bool(report["metadata"].get("training_executed")),
                "old_candidate_phase9d_overwritten": False,
                "backtest_artifacts_used_as_features": False,
                "current_pm_multiplier_used_as_label": False,
            }
        )
        report["dataset_summary"] = self._dataset_summary(dataset)
        report["model_output"]["candidate_phase9d_overwritten"] = False
        report["model_output"]["model_dir"] = str(self.model_dir)
        report["leakage_checklist"].update(
            {
                "feature_columns": report["feature_plan"].get("feature_columns", []),
                "label_columns": list(MODEL_TARGETS[target]["target"] for target in MODEL_TARGETS),
                "dropped_features": report["feature_plan"].get("dropped_features", []),
                "future_columns_in_features": [
                    column for column in report["feature_plan"].get("feature_columns", []) if str(column).lower().startswith("future_")
                ],
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
                "backtest_artifacts_used_as_features": False,
                "current_pm_multiplier_used_as_label": False,
            }
        )
        report["verdict"] = self._phase9d2_verdict(report)
        return report

    def save_report(self, report: dict[str, Any]) -> Phase9D2TrainPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9D2TrainPaths(model_dir=self.model_dir, markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        base = super().format_markdown(report).replace("Phase 9-D PM AI v3 Trainer Prototype", "Phase 9-D2 PM AI v3 PM Sizing Universe Trainer")
        mapping_rows = report.get("multiplier_prototype", {}).get("mapping_summary_rows", [])
        dataset = report.get("dataset_summary", {})
        return "\n".join(
            [
                base,
                "## PM Sizing Universe Dataset",
                "",
                self._table([dataset], ["row_count", "date_min", "date_max", "code_count"]),
                "",
                "## Mapping Prototype Candidates",
                "",
                self._table(mapping_rows, ["mapping", "split", "pm130_count", "pm130_downside_mean", "overall_downside_mean", "pm130_vs_overall_delta", "pm130_better_than_mid", "pm060_downside_mean"]),
                "",
            ]
        )

    def _train_one(
        self,
        dataset: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
        features: list[str],
        target: str,
        kind: str,
    ) -> tuple[dict[str, Any], Any, dict[str, pd.Series]]:
        train = splits["train"].dropna(subset=[target]).copy()
        valid = splits["validation"].dropna(subset=[target]).copy()
        test = splits["test"].dropna(subset=[target]).copy()
        if train.empty or valid.empty or test.empty:
            return {"training_status": "missing_split_rows"}, None, {}
        x_train = train[features]
        y_train = _numeric(train[target])
        if kind == "classification":
            y_train = y_train.astype(int)
            model = HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=0.06, early_stopping=True, random_state=42)
        else:
            model = HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=0.06, early_stopping=True, random_state=42)
        model.fit(x_train, y_train)
        metrics = {
            split: self._evaluate_split(frame.dropna(subset=[target]), features, target, model, kind)
            for split, frame in {"train": train, "validation": valid, "test": test}.items()
        }
        preds = {
            split: pd.Series(self._predict(model, frame[features], kind), index=frame.index)
            for split, frame in {"train": train, "validation": valid, "test": test}.items()
        }
        return metrics, model, preds

    def _save_models(
        self,
        models: dict[str, Any],
        features: list[str],
        feature_plan: dict[str, Any],
        splits: dict[str, pd.DataFrame],
        leakage: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, str]:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for model_name, model in models.items():
            if model is None:
                continue
            path = self.model_dir / MODEL_TARGETS[model_name]["filename"]
            joblib.dump(model, path)
            paths[model_name] = str(path)
        (self.model_dir / "feature_columns.json").write_text(json.dumps(features, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata = {
            "phase": "9-D2",
            "dataset": str(self.dataset_path),
            "model_dir": str(self.model_dir),
            "feature_count": len(features),
            "dropped_features": feature_plan["dropped_features"],
            "split_rows": {split: int(len(frame)) for split, frame in splits.items()},
            "leakage_checklist": leakage,
            "current_pm_ai_overwritten": False,
            "current_exit_ai_overwritten": False,
            "v2_82_profile_overwritten": False,
            "old_candidate_phase9d_overwritten": False,
        }
        (self.model_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        (self.model_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        paths["feature_columns"] = str(self.model_dir / "feature_columns.json")
        paths["training_metadata"] = str(self.model_dir / "training_metadata.json")
        paths["metrics"] = str(self.model_dir / "metrics.json")
        return paths

    def _multiplier_prototype(self, dataset: pd.DataFrame, splits: dict[str, pd.DataFrame], predictions: dict[str, dict[str, pd.Series]]) -> dict[str, Any]:
        if not predictions:
            return {"distribution": [], "mapping_summary_rows": []}
        rows: list[dict[str, Any]] = []
        details: dict[str, list[dict[str, Any]]] = {}
        for split_name in ["validation", "test"]:
            frame = splits.get(split_name, pd.DataFrame()).copy()
            if frame.empty:
                continue
            rank = predictions.get("model_a_candidate_ranking_regressor", {}).get(split_name)
            utility = predictions.get("model_b_downside_utility_regressor", {}).get(split_name)
            top = predictions.get("model_c_top_utility_classifier", {}).get(split_name)
            if rank is None or utility is None or top is None:
                continue
            work = frame.copy()
            work["_rank_pred"] = rank
            work["_utility_pred"] = utility
            work["_top_pred"] = top
            mappings = self._mapping_candidates(work)
            overall = float(_numeric(work.get("downside_penalized_return_10d")).mean())
            for name, multiplier in mappings.items():
                work_name = work.copy()
                work_name["_mapping_multiplier"] = multiplier
                by_mult = [self._multiplier_row(mult, group, overall) for mult, group in work_name.groupby("_mapping_multiplier")]
                pm130 = next((row for row in by_mult if row["multiplier"] == 1.30), {})
                pm060 = next((row for row in by_mult if row["multiplier"] == 0.60), {})
                rows.append(
                    {
                        "mapping": name,
                        "split": split_name,
                        "pm130_count": pm130.get("count", 0),
                        "pm130_downside_mean": pm130.get("actual_downside_mean"),
                        "overall_downside_mean": overall,
                        "pm130_vs_overall_delta": None if pm130.get("actual_downside_mean") is None else pm130["actual_downside_mean"] - overall,
                        "pm130_better_than_mid": self._pm130_beats_mid(by_mult),
                        "pm060_downside_mean": pm060.get("actual_downside_mean"),
                    }
                )
                details[f"{name}:{split_name}"] = by_mult
        best = self._best_mapping(rows)
        test = splits.get("test", pd.DataFrame())
        top10_downside = best.get("test_pm130_downside_mean")
        overall = best.get("test_overall_downside_mean")
        return {
            "mapping_summary_rows": rows,
            "mapping_details": details,
            "best_mapping": best,
            "distribution": details.get(f"{best.get('mapping')}:test", []),
            "predicted_top10_actual_downside_penalized_return_10d": top10_downside,
            "overall_actual_downside_penalized_return_10d": overall if overall is not None else (float(_numeric(test.get("downside_penalized_return_10d")).mean()) if not test.empty else None),
            "prototype_for_research_only": True,
        }

    def _mapping_candidates(self, work: pd.DataFrame) -> dict[str, pd.Series]:
        rank_pct = _numeric(work["_rank_pred"]).rank(method="first", pct=True)
        utility_pct = _numeric(work["_utility_pred"]).rank(method="first", pct=True)
        top_pct = _numeric(work["_top_pred"]).rank(method="first", pct=True)
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

    def _multiplier_row(self, multiplier: float, group: pd.DataFrame, overall: float | None) -> dict[str, Any]:
        downside = _numeric(group.get("downside_penalized_return_10d"))
        rank_pct = _numeric(group.get("relative_future_utility_percentile_in_day"))
        return {
            "multiplier": float(multiplier),
            "count": int(len(group)),
            "rate": float(len(group) / max(1, len(group.index.unique()))),
            "actual_downside_mean": float(downside.mean()) if not downside.dropna().empty else None,
            "actual_rank_percentile_mean": float(rank_pct.mean()) if not rank_pct.dropna().empty else None,
            "vs_overall_downside_delta": None if overall is None or downside.dropna().empty else float(downside.mean()) - overall,
        }

    def _pm130_beats_mid(self, rows: list[dict[str, Any]]) -> bool:
        by = {row["multiplier"]: row.get("actual_downside_mean") for row in rows}
        pm130 = by.get(1.30)
        if pm130 is None:
            return False
        return all(pm130 > by.get(mult, -10**9) for mult in [1.15, 1.00, 0.80] if by.get(mult) is not None)

    def _best_mapping(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        test_rows = [row for row in rows if row["split"] == "test" and row.get("pm130_count", 0) > 0 and row.get("pm130_downside_mean") is not None]
        if not test_rows:
            return {}
        best = max(test_rows, key=lambda row: row.get("pm130_vs_overall_delta") or -10**9)
        return {
            "mapping": best["mapping"],
            "test_pm130_count": best.get("pm130_count"),
            "test_pm130_downside_mean": best.get("pm130_downside_mean"),
            "test_overall_downside_mean": best.get("overall_downside_mean"),
            "test_pm130_vs_overall_delta": best.get("pm130_vs_overall_delta"),
        }

    def _dataset_summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        if dataset.empty:
            return {"row_count": 0}
        dates = pd.to_datetime(dataset["prediction_date"], errors="coerce").dropna()
        return {
            "row_count": int(len(dataset)),
            "date_min": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
            "date_max": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
            "code_count": int(dataset["code"].nunique()) if "code" in dataset.columns else 0,
        }

    def _phase9d2_verdict(self, report: dict[str, Any]) -> dict[str, Any]:
        leakage = report.get("leakage_checklist", {})
        best = report.get("multiplier_prototype", {}).get("best_mapping", {})
        top10 = best.get("test_pm130_downside_mean")
        overall = best.get("test_overall_downside_mean")
        worth = bool(leakage.get("leakage_risk") == "low" and top10 is not None and overall is not None and top10 > overall)
        return {
            "phase9e2_integration_audit_worth_testing": worth,
            "recommended_next_phase": "Phase 9-E2: PM AI v3 Candidate Integration Audit on PM sizing universe" if worth else "Phase 9-D3: trainer/label refinement",
            "reason": "best mapping PM1.30 downside utility beats test overall" if worth else "best mapping PM1.30 utility does not clear test overall or leakage blocked",
        }


def train_phase9d2_pm_ai_v3_pm_sizing_universe(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3PMSizingUniverseTrainer(root).run()
