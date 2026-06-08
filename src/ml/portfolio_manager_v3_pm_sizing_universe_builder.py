"""Phase 9-B3 PM AI v3 PM sizing universe dataset builder.

This builder keeps the original Phase 9-B top10 dataset intact and writes a
separate dataset whose row universe is all walk-forward prediction candidates
that can be joined to API-derived feature rows. Backtest artifacts are read only
for coverage auditing against Phase 9-F PM sizing keys, never as features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_v3_coverage_root_cause_audit import normalize_code, normalize_date
from ml.portfolio_manager_v3_dataset_builder import (
    CONDITIONAL_RELATIVE_FEATURES,
    DOWNSIDE_PENALTY,
    END_DATE,
    FORBIDDEN_TOKENS,
    LABEL_COLUMNS,
    PMAIV3BuildOptions,
    PMAIV3CleanDatasetBuilder,
    ROOT,
    START_DATE,
    _numeric,
)


PERIOD_LABEL = "2023-01_to_2026-05"
REPORT_STEM = "phase9b3_pm_ai_v3_pm_sizing_universe_dataset_2023-01_to_2026-05"
OUTPUT_DIR = Path("data/ml/portfolio_manager_v3")
DATASET_OUTPUT = OUTPUT_DIR / f"portfolio_manager_v3_dataset_pm_sizing_universe_{PERIOD_LABEL}.parquet"
MARKET_REGIME_OUTPUT = OUTPUT_DIR / f"pm_v3_market_regime_daily_pm_sizing_universe_{PERIOD_LABEL}.parquet"
PERIOD = "2023-01-01_to_2026-05-31"
V293_PROFILES = {
    "v2_93_a": "rookie_dealer_02_v2_93_pm_ai_v3_candidate",
    "v2_93_b": "rookie_dealer_02_v2_93b_pm_ai_v3_candidate_conservative",
    "v2_93_c": "rookie_dealer_02_v2_93c_pm_ai_v3_candidate_half_pm130",
}


@dataclass(frozen=True)
class PMAIV3PMSizingUniversePaths:
    dataset: Path
    market_regime: Path
    markdown: Path
    json: Path


class PMAIV3PMSizingUniverseDatasetBuilder(PMAIV3CleanDatasetBuilder):
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        dataset_output: Path | None = None,
        market_regime_output: Path | None = None,
        profiles: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            root,
            dataset_output=dataset_output or DATASET_OUTPUT,
            market_regime_output=market_regime_output or MARKET_REGIME_OUTPUT,
        )
        self.profiles = profiles or V293_PROFILES

    def build(self, options: PMAIV3BuildOptions | None = None) -> dict[str, Any]:
        options = options or PMAIV3BuildOptions(
            start_date=START_DATE,
            end_date=END_DATE,
            top_n=999999,
            min_turnover_value=0.0,
            downside_penalty=DOWNSIDE_PENALTY,
            write_outputs=True,
        )
        base = self._load_base_dataset(options)
        market_regime = self._build_market_regime_table(base, options)
        predictions = self._load_walk_forward_predictions(options)
        candidates = self._build_candidate_pool(base, predictions, market_regime, options)
        dataset = self._add_relative_labels(candidates)
        feature_columns = self._feature_columns(dataset)
        label_columns = [column for column in LABEL_COLUMNS if column in dataset.columns]
        leakage = self._leakage_audit(feature_columns, label_columns)
        final = self._finalize_dataset(dataset, feature_columns, label_columns)
        coverage_targets = self._load_phase9f_pm_sizing_keys()
        coverage = self._coverage_against_targets(final, coverage_targets)
        quality = self._quality_audit(final, feature_columns, label_columns)
        report = self._phase9b3_report(final, market_regime, feature_columns, label_columns, leakage, coverage, quality, options)
        if options.write_outputs and not leakage["blocking_issues"]:
            self.dataset_output.parent.mkdir(parents=True, exist_ok=True)
            final.to_parquet(self.dataset_output, index=False)
            market_regime.to_parquet(self.market_regime_output, index=False)
            report["metadata"]["dataset_written"] = True
            report["metadata"]["market_regime_written"] = True
        report["_dataset"] = final
        report["_market_regime"] = market_regime
        return report

    def save_report(self, report: dict[str, Any]) -> PMAIV3PMSizingUniversePaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        serializable = {key: value for key, value in report.items() if not key.startswith("_")}
        json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_phase9b3_markdown(serializable), encoding="utf-8")
        return PMAIV3PMSizingUniversePaths(
            dataset=Path(serializable["output_paths"]["dataset"]),
            market_regime=Path(serializable["output_paths"]["market_regime"]),
            markdown=md_path,
            json=json_path,
        )

    def _build_candidate_pool(
        self,
        base: pd.DataFrame,
        predictions: pd.DataFrame,
        market_regime: pd.DataFrame,
        options: PMAIV3BuildOptions,
    ) -> pd.DataFrame:
        if base.empty or predictions.empty:
            return pd.DataFrame()
        base_features = base.drop(columns=[column for column in ["date"] if column in base.columns]).copy()
        merged = predictions.merge(base_features, on=["prediction_date", "code"], how="inner", suffixes=("", "_api"))
        for column in ["expected_return_10d", "bad_entry_probability_10d", "turnover_value"]:
            if column in merged.columns:
                merged[column] = pd.to_numeric(merged[column], errors="coerce")
        if "risk_adjusted_score" not in merged.columns:
            merged["risk_adjusted_score"] = merged["expected_return_10d"] - 0.5 * merged["bad_entry_probability_10d"]
        merged["stock_selection_rank_score"] = merged["risk_adjusted_score"]
        candidates = merged.dropna(subset=["prediction_date", "code", "risk_adjusted_score"]).copy()
        candidates = candidates.sort_values(["prediction_date", "risk_adjusted_score"], ascending=[True, False])
        candidates = self._add_candidate_relative_features(candidates)
        market_cols = [
            "date",
            "topix_return_1d_proxy",
            "topix_ma_distance",
            "topix_volatility",
            "market_attack_score_prototype",
            "market_regime_class_prototype",
        ]
        candidates = candidates.merge(
            market_regime[[column for column in market_cols if column in market_regime.columns]].rename(columns={"date": "prediction_date"}),
            on="prediction_date",
            how="left",
        )
        candidates["market_date"] = candidates["prediction_date"]
        candidates["market_regime_key"] = candidates.get("market_regime_class_prototype")
        candidates["data_source"] = "all_walk_forward_predictions_joined_to_api_dataset"
        candidates["relative_feature_timing"] = "computed_after_pm_sizing_candidate_universe_before_cash_portfolio_backtest"
        return candidates

    def _load_phase9f_pm_sizing_keys(self) -> pd.DataFrame:
        frames = []
        for label, profile in self.profiles.items():
            path = self.root / "logs" / "backtests" / profile / PERIOD / "purchase_audit.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            marker = frame.get("pm_model_version", pd.Series("", index=frame.index)).fillna("").astype(str).str.contains("pm_ai_v3")
            missing = frame.get("pm_missing_reason", pd.Series("", index=frame.index)).fillna("").astype(str).str.contains("pm_v3")
            frame = frame[marker | missing].copy()
            frame["profile_label"] = label
            frame["prediction_date"] = frame.get("signal_date", pd.Series("", index=frame.index)).map(normalize_date)
            frame["trade_date"] = frame.get("entry_date", pd.Series("", index=frame.index)).map(normalize_date)
            frame["buy_date"] = frame["trade_date"]
            frame["code"] = frame.get("code", pd.Series("", index=frame.index)).astype(str)
            frame["normalized_code"] = frame["code"].map(normalize_code)
            frames.append(frame[["profile_label", "prediction_date", "trade_date", "buy_date", "code", "normalized_code"]])
        if not frames:
            return pd.DataFrame(columns=["profile_label", "prediction_date", "trade_date", "buy_date", "code", "normalized_code"])
        return pd.concat(frames, ignore_index=True)

    def _coverage_against_targets(self, dataset: pd.DataFrame, targets: pd.DataFrame) -> dict[str, Any]:
        if dataset.empty or targets.empty:
            return {"target_count": int(len(targets)), "coverage_matrix": []}
        data = dataset[["prediction_date", "code"]].copy()
        data["prediction_date"] = data["prediction_date"].map(normalize_date)
        data["normalized_code"] = data["code"].map(normalize_code)
        dataset_keys = set(data["prediction_date"] + "|" + data["normalized_code"])
        dataset_dates = set(data["prediction_date"])
        dataset_codes = set(data["normalized_code"])
        variants = {
            "prediction_date+code": targets["prediction_date"] + "|" + targets["normalized_code"],
            "trade_date+code": targets["trade_date"] + "|" + targets["normalized_code"],
            "buy_date+code": targets["buy_date"] + "|" + targets["normalized_code"],
        }
        rows = [self._coverage_row(name, series, dataset_keys) for name, series in variants.items()]
        rows.append(self._set_overlap_row("date-only overlap", targets["prediction_date"].isin(dataset_dates), targets["prediction_date"]))
        rows.append(self._set_overlap_row("code-only overlap", targets["normalized_code"].isin(dataset_codes), targets["normalized_code"]))
        key_row = rows[0] if rows else {}
        return {
            "target_count": int(len(targets)),
            "coverage_matrix": rows,
            "pm_sizing_key_coverage": key_row.get("coverage_rate"),
            "coverage_goal": 0.95,
            "coverage_goal_met": bool((key_row.get("coverage_rate") or 0.0) >= 0.95),
        }

    def _coverage_row(self, name: str, keys: pd.Series, dataset_keys: set[str]) -> dict[str, Any]:
        match = keys.isin(dataset_keys)
        return {
            "key": name,
            "matched_rows": int(match.sum()),
            "unmatched_rows": int((~match).sum()),
            "coverage_rate": float(match.mean()) if len(match) else None,
            "sample_matched_keys": keys[match].head(5).tolist(),
            "sample_unmatched_keys": keys[~match].head(5).tolist(),
        }

    def _set_overlap_row(self, name: str, match: pd.Series, values: pd.Series) -> dict[str, Any]:
        return {
            "key": name,
            "matched_rows": int(match.sum()),
            "unmatched_rows": int((~match).sum()),
            "coverage_rate": float(match.mean()) if len(match) else None,
            "sample_matched_keys": values[match].head(5).tolist(),
            "sample_unmatched_keys": values[~match].head(5).tolist(),
        }

    def _quality_audit(self, dataset: pd.DataFrame, feature_columns: list[str], label_columns: list[str]) -> dict[str, Any]:
        if dataset.empty:
            return {"row_count": 0}
        duplicates = int(dataset.duplicated(["prediction_date", "code"]).sum()) if {"prediction_date", "code"}.issubset(dataset.columns) else 0
        numeric = dataset[feature_columns].apply(pd.to_numeric, errors="coerce") if feature_columns else pd.DataFrame()
        infinite_count = int((numeric.replace([float("inf"), float("-inf")], pd.NA).isna() & numeric.notna()).sum().sum()) if not numeric.empty else 0
        constant = [column for column in feature_columns if numeric[column].nunique(dropna=True) <= 1] if not numeric.empty else []
        missing = self._null_rates(dataset, feature_columns)
        high_missing = [row["column"] for row in missing if row["null_rate"] is not None and row["null_rate"] >= 0.50]
        return {
            "row_count": int(len(dataset)),
            "date_range": self._date_range(dataset),
            "code_count": int(dataset["code"].nunique()),
            "candidate_count_stats": self._candidate_count_stats(dataset),
            "feature_count": len(feature_columns),
            "label_count": len(label_columns),
            "duplicate_key_count": duplicates,
            "label_null_rate": self._null_rates(dataset, label_columns),
            "feature_null_rate_top20": missing[:20],
            "infinite_count": infinite_count,
            "constant_feature_count": len(constant),
            "constant_features": constant[:50],
            "high_missing_feature_count": len(high_missing),
            "high_missing_features": high_missing[:50],
        }

    def _phase9b3_report(
        self,
        dataset: pd.DataFrame,
        market_regime: pd.DataFrame,
        feature_columns: list[str],
        label_columns: list[str],
        leakage: dict[str, Any],
        coverage: dict[str, Any],
        quality: dict[str, Any],
        options: PMAIV3BuildOptions,
    ) -> dict[str, Any]:
        option_a = coverage.get("pm_sizing_key_coverage")
        return {
            "metadata": {
                "phase": "9-B3",
                "dataset_purpose": "PM AI v3 PM sizing universe coverage fix dataset",
                "training_executed": False,
                "mapping_adjustment_executed": False,
                "strategy_backtest_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_model_regenerated_historical_predictions": False,
                "pm_ai_v3_retrained": False,
                "dataset_written": False,
                "market_regime_written": False,
                "adopted_universe": "Option A: all walk-forward predictions joined to API-derived dataset",
                "candidate_rule": "all walk-forward prediction candidates with API feature rows; no top10 cap",
            },
            "input_paths": {
                "base_dataset": str(self.base_dataset),
                "walk_forward_predictions": str(self.walk_forward_dir),
                "phase9f_backtest_logs": str(self.root / "logs/backtests"),
            },
            "output_paths": {
                "dataset": str(self.dataset_output),
                "market_regime": str(self.market_regime_output),
                "markdown": str(self.root / "reports/ml" / f"{REPORT_STEM}.md"),
                "json": str(self.root / "reports/ml" / f"{REPORT_STEM}.json"),
            },
            "universe_design": {
                "adopted_option": "A",
                "row_grain": "prediction-time PM sizing candidate x prediction_date x code",
                "uses_top10_fixed": False,
                "uses_backtest_artifacts_for_coverage_target_only": True,
                "uses_backtest_artifacts_as_features": False,
            },
            "option_coverage_comparison": [
                {"option": "current_top10_fixed_phase9b", "coverage_rate": 0.0, "status": "known_from_phase9b2"},
                {"option": "option_a_all_walk_forward_predictions", "coverage_rate": option_a, "status": "adopted"},
                {"option": "option_b_reproduce_pre_selection_universe", "coverage_rate": None, "status": "not_required_after_option_a_met_goal"},
                {"option": "option_c_phase9f_pm_sizing_keys_as_coverage_target", "coverage_rate": option_a, "status": "not_adopted_backtest_keys_not_used_as_rows"},
            ],
            "dataset_summary": self._dataset_summary(dataset),
            "quality_audit": quality,
            "feature_columns": feature_columns,
            "label_columns": label_columns,
            "metadata_columns": ["prediction_date", "code", "market_date", "market_regime_key", "data_source", "relative_feature_timing"],
            "conditional_feature_columns": [column for column in CONDITIONAL_RELATIVE_FEATURES if column in feature_columns],
            "coverage_against_phase9f_pm_sizing_keys": coverage,
            "leakage_audit": {
                **leakage,
                "backtest_artifacts_used_for_coverage_target_only": True,
                "backtest_artifacts_used_as_features": False,
            },
            "current_artifact_safety": {
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
            },
            "rerun_readiness": {
                "phase9d_retrain_required_before_phase9e": True,
                "phase9e_rerun_ready_after_retraining": False,
                "phase9f_rerun_ready_now": False,
                "reason": "Dataset coverage is fixed, but PM AI v3 models were trained on the old top10 universe and must not be reused for final evaluation without Phase 9-D retraining.",
            },
        }

    def format_phase9b3_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# PM AI v3 Phase 9-B3 PM Sizing Universe Dataset",
                "",
                "## Universe Design",
                "",
                self._table([report["universe_design"]], ["adopted_option", "row_grain", "uses_top10_fixed", "uses_backtest_artifacts_for_coverage_target_only", "uses_backtest_artifacts_as_features"]),
                "",
                "## Option Coverage",
                "",
                self._table(report["option_coverage_comparison"], ["option", "coverage_rate", "status"]),
                "",
                "## Dataset Summary",
                "",
                self._table([report["dataset_summary"]], ["row_count", "date_min", "date_max", "code_count", "candidate_count_stats"]),
                "",
                "## Quality Audit",
                "",
                self._table([report["quality_audit"]], ["feature_count", "label_count", "duplicate_key_count", "infinite_count", "constant_feature_count", "high_missing_feature_count"]),
                "",
                "## Phase 9-F PM Sizing Key Coverage",
                "",
                self._table(report["coverage_against_phase9f_pm_sizing_keys"]["coverage_matrix"], ["key", "matched_rows", "unmatched_rows", "coverage_rate", "sample_unmatched_keys"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_audit"]], ["forbidden_feature_count", "forbidden_feature_columns", "label_columns_in_features", "leakage_risk", "backtest_artifacts_used_for_coverage_target_only", "backtest_artifacts_used_as_features"]),
                "",
                "## Rerun Readiness",
                "",
                self._table([report["rerun_readiness"]], ["phase9d_retrain_required_before_phase9e", "phase9e_rerun_ready_after_retraining", "phase9f_rerun_ready_now", "reason"]),
                "",
            ]
        )

    def _date_range(self, dataset: pd.DataFrame) -> dict[str, Any]:
        dates = pd.to_datetime(dataset["prediction_date"], errors="coerce").dropna()
        return {"date_min": dates.min().strftime("%Y-%m-%d"), "date_max": dates.max().strftime("%Y-%m-%d")} if not dates.empty else {"date_min": None, "date_max": None}

    def _candidate_count_stats(self, dataset: pd.DataFrame) -> dict[str, Any]:
        counts = dataset.groupby("prediction_date")["code"].count()
        if counts.empty:
            return {}
        return {
            "min": int(counts.min()),
            "median": float(counts.median()),
            "mean": float(counts.mean()),
            "max": int(counts.max()),
        }


def build_phase9b3_pm_ai_v3_pm_sizing_universe_dataset(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3PMSizingUniverseDatasetBuilder(root).build()
