"""Phase 9-B PM AI v3 clean dataset builder.

The builder creates a separate PM AI v3 research dataset. It uses only
J-Quants/API-derived feature rows, historical walk-forward predictions, same-day
candidate-relative features computed before cash/portfolio decisions, and
future-price labels. It does not read backtest trades or overwrite current PM,
Exit, or profile artifacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase9b_pm_ai_v3_dataset_builder_2023-01_to_2026-05"
PERIOD_LABEL = "2023-01_to_2026-05"
START_DATE = "2023-01-01"
END_DATE = "2026-05-31"

BASE_DATASET = Path("data/ml/datasets/ml_dataset.parquet")
WALK_FORWARD_DIR = Path("data/ml/walk_forward_predictions")
OUTPUT_DIR = Path("data/ml/portfolio_manager_v3")
DATASET_OUTPUT = OUTPUT_DIR / f"portfolio_manager_v3_dataset_{PERIOD_LABEL}.parquet"
MARKET_REGIME_OUTPUT = OUTPUT_DIR / f"pm_v3_market_regime_daily_{PERIOD_LABEL}.parquet"

CURRENT_PM_DIR = Path("models/ml/portfolio_manager/current_v2_73_phase3b_clean")
CURRENT_EXIT_DIR = Path("models/ml/exit/current_v2_66")
V282_PROFILE = Path("config/profiles/rookie_dealer_02_v2_82_cap38.yaml")

TOP_N_CANDIDATES = 10
MIN_TURNOVER_VALUE = 50_000_000
DOWNSIDE_PENALTY = 1.0

KEY_COLUMNS = ["prediction_date", "code", "market_date", "market_regime_key"]

STOCK_PREDICTION_FEATURES = [
    "expected_return_5d",
    "expected_return_10d",
    "expected_max_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "upside_probability_10d",
    "bad_entry_probability_10d",
    "ml_score",
    "risk_adjusted_score",
    "stock_selection_rank_score",
]

MARKET_FEATURES = [
    "topix_return_5d",
    "topix_return_10d",
    "topix_return_20d",
    "topix_return_1d_proxy",
    "topix_ma_distance",
    "topix_volatility",
    "market_attack_score_prototype",
]

PRICE_VOLUME_FEATURES = [
    "close",
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
    "gap_up_ratio",
    "daily_range_ratio",
    "volume",
    "turnover_value",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "turnover_ratio_5d",
    "turnover_ratio_20d",
]

FINANCIAL_FEATURES = [
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
]

EVENT_FEATURES = [
    "days_to_earnings",
    "days_after_earnings",
    "is_near_earnings",
]

CONDITIONAL_RELATIVE_FEATURES = [
    "candidate_count_in_day",
    "rank_in_day",
    "percentile_in_day",
    "gap_to_best",
    "candidate_strength",
]

LABEL_COLUMNS = [
    "future_5d_return",
    "future_10d_return",
    "max_favorable_excursion_10d",
    "max_adverse_excursion_10d",
    "downside_penalized_return_10d",
    "risk_adjusted_future_return_10d",
    "relative_future_utility_rank_in_day",
    "relative_future_utility_percentile_in_day",
    "top_decile_future_utility_in_day",
    "bottom_decile_future_utility_in_day",
]

FORBIDDEN_TOKENS = [
    "selected",
    "bought",
    "affordable",
    "cash",
    "portfolio",
    "position",
    "profit",
    "loss",
    "pnl",
    "result",
    "backtest",
    "exit",
    "skip",
    "filled",
    "actual",
    "final_assets",
    "trade_result",
    "realized",
]

ALLOWED_LABEL_TOKENS = ["future_return"]


@dataclass(frozen=True)
class PMAIV3DatasetPaths:
    dataset: Path
    market_regime: Path
    markdown: Path
    json: Path


@dataclass(frozen=True)
class PMAIV3BuildOptions:
    start_date: str = START_DATE
    end_date: str = END_DATE
    top_n: int = TOP_N_CANDIDATES
    min_turnover_value: float = MIN_TURNOVER_VALUE
    downside_penalty: float = DOWNSIDE_PENALTY
    write_outputs: bool = True


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
    return pd.to_numeric(series, errors="coerce")


def _safe_mean(series: pd.Series | None) -> float | None:
    values = _numeric(series).dropna()
    return None if values.empty else float(values.mean())


class PMAIV3CleanDatasetBuilder:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        base_dataset: Path | None = None,
        walk_forward_dir: Path | None = None,
        dataset_output: Path | None = None,
        market_regime_output: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.base_dataset = self._root(base_dataset or BASE_DATASET)
        self.walk_forward_dir = self._root(walk_forward_dir or WALK_FORWARD_DIR)
        self.dataset_output = self._root(dataset_output or DATASET_OUTPUT)
        self.market_regime_output = self._root(market_regime_output or MARKET_REGIME_OUTPUT)

    def build(self, options: PMAIV3BuildOptions | None = None) -> dict[str, Any]:
        options = options or PMAIV3BuildOptions()
        base = self._load_base_dataset(options)
        market_regime = self._build_market_regime_table(base, options)
        predictions = self._load_walk_forward_predictions(options)
        candidates = self._build_candidate_pool(base, predictions, market_regime, options)
        dataset = self._add_relative_labels(candidates)
        feature_columns = self._feature_columns(dataset)
        label_columns = [column for column in LABEL_COLUMNS if column in dataset.columns]
        leakage = self._leakage_audit(feature_columns, label_columns)
        final = self._finalize_dataset(dataset, feature_columns, label_columns)
        report = self._report(final, market_regime, feature_columns, label_columns, leakage, options)
        if options.write_outputs and not leakage["blocking_issues"]:
            self.dataset_output.parent.mkdir(parents=True, exist_ok=True)
            final.to_parquet(self.dataset_output, index=False)
            market_regime.to_parquet(self.market_regime_output, index=False)
            report["output_paths"]["dataset"] = str(self.dataset_output)
            report["output_paths"]["market_regime"] = str(self.market_regime_output)
            report["metadata"]["dataset_written"] = True
            report["metadata"]["market_regime_written"] = True
        report["_dataset"] = final
        report["_market_regime"] = market_regime
        return report

    def save_report(self, report: dict[str, Any]) -> PMAIV3DatasetPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        serializable = {key: value for key, value in report.items() if not key.startswith("_")}
        json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(serializable), encoding="utf-8")
        return PMAIV3DatasetPaths(
            dataset=Path(serializable["output_paths"]["dataset"]),
            market_regime=Path(serializable["output_paths"]["market_regime"]),
            markdown=md_path,
            json=json_path,
        )

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 9-B PM AI v3 Clean Dataset Builder",
                "",
                "## Purpose",
                "",
                "Build a separate PM AI v3 research dataset from API-derived features, historical walk-forward predictions, prediction-time relative features, and future-price labels.",
                "",
                "## Inputs",
                "",
                self._table([report["input_paths"]], ["base_dataset", "walk_forward_predictions"]),
                "",
                "## Outputs",
                "",
                self._table([report["output_paths"]], ["dataset", "market_regime", "markdown", "json"]),
                "",
                "## Dataset Summary",
                "",
                self._table([report["dataset_summary"]], ["row_count", "date_min", "date_max", "code_count", "candidate_count_stats"]),
                "",
                "## Feature Classification",
                "",
                self._table(report["feature_classification"], ["category", "count", "columns", "rule"]),
                "",
                "## Label Classification",
                "",
                self._table(report["label_classification"], ["label", "definition", "source", "feature_allowed"]),
                "",
                "## Leakage Checklist",
                "",
                self._table([report["leakage_audit"]], ["forbidden_feature_count", "forbidden_feature_columns", "label_columns_in_features", "leakage_risk", "blocking_issues"]),
                "",
                "## Null Rates",
                "",
                "### Labels",
                "",
                self._table(report["label_null_rate"], ["column", "null_rate"]),
                "",
                "### Features Top Missing",
                "",
                self._table(report["feature_null_rate_top"], ["column", "null_rate"]),
                "",
                "## Relative Feature Timing",
                "",
                report["relative_feature_timing"],
                "",
                "## Phase 9-C Model Plan",
                "",
                self._table(report["phase9c_model_plan"], ["model", "purpose", "target", "notes"]),
                "",
                "## Current Artifact Safety",
                "",
                self._table([report["current_artifact_safety"]], ["current_pm_ai_overwritten", "current_exit_ai_overwritten", "v2_82_profile_overwritten", "current_v282_maintained"]),
                "",
            ]
        )

    def _root(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def _load_base_dataset(self, options: PMAIV3BuildOptions) -> pd.DataFrame:
        base = _read_parquet(self.base_dataset)
        if base.empty:
            return base
        out = base.copy()
        out["prediction_date"] = pd.to_datetime(out["date"], errors="coerce")
        out["code"] = out["code"].astype(str)
        mask = (out["prediction_date"] >= pd.Timestamp(options.start_date)) & (
            out["prediction_date"] <= pd.Timestamp(options.end_date)
        )
        return out.loc[mask].copy()

    def _load_walk_forward_predictions(self, options: PMAIV3BuildOptions) -> pd.DataFrame:
        if not self.walk_forward_dir.exists():
            return pd.DataFrame()
        pieces: list[pd.DataFrame] = []
        for path in sorted(self.walk_forward_dir.glob("predictions_*.parquet")):
            date_text = path.stem.replace("predictions_", "")
            date = pd.to_datetime(date_text, errors="coerce")
            if pd.isna(date) or date < pd.Timestamp(options.start_date) or date > pd.Timestamp(options.end_date):
                continue
            frame = _read_parquet(path)
            if frame.empty:
                continue
            date_column = "date" if "date" in frame.columns else "prediction_date" if "prediction_date" in frame.columns else None
            if date_column is None:
                frame["prediction_date"] = date
            else:
                frame["prediction_date"] = pd.to_datetime(frame[date_column], errors="coerce")
            if "code" not in frame.columns:
                continue
            frame["code"] = frame["code"].astype(str)
            keep = [
                column
                for column in [
                    "prediction_date",
                    "code",
                    "expected_return_5d",
                    "expected_return_10d",
                    "expected_max_return_10d",
                    "expected_max_return_20d",
                    "swing_success_probability_20d",
                    "upside_probability_10d",
                    "bad_entry_probability_10d",
                    "entry_risk_label",
                    "ml_score",
                ]
                if column in frame.columns
            ]
            pieces.append(frame[keep].copy())
        if not pieces:
            return pd.DataFrame()
        predictions = pd.concat(pieces, ignore_index=True)
        predictions = predictions.dropna(subset=["prediction_date", "code"])
        return predictions.drop_duplicates(["prediction_date", "code"])

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
        liquid = merged[_numeric(merged.get("turnover_value")).ge(options.min_turnover_value)].copy()
        liquid = liquid.dropna(subset=["risk_adjusted_score"])
        liquid = liquid.sort_values(["prediction_date", "risk_adjusted_score"], ascending=[True, False])
        candidates = liquid.groupby("prediction_date", group_keys=False).head(options.top_n).copy()
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
        candidates["data_source"] = "walk_forward_predictions_joined_to_api_dataset"
        candidates["relative_feature_timing"] = "computed_after_candidate_pool_before_cash_portfolio_backtest"
        return candidates

    def _add_candidate_relative_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        grouped = out.groupby("prediction_date", group_keys=False)
        out["candidate_count_in_day"] = grouped["code"].transform("count")
        out["rank_in_day"] = grouped["risk_adjusted_score"].rank(method="first", ascending=False)
        count = _numeric(out["candidate_count_in_day"])
        rank = _numeric(out["rank_in_day"])
        out["percentile_in_day"] = 1.0 - ((rank - 1.0) / (count - 1.0))
        out.loc[count.le(1), "percentile_in_day"] = 1.0
        best = grouped["risk_adjusted_score"].transform("max")
        mean = grouped["risk_adjusted_score"].transform("mean")
        out["gap_to_best"] = best - out["risk_adjusted_score"]
        out["candidate_strength"] = out["risk_adjusted_score"] - mean
        return out

    def _build_market_regime_table(self, base: pd.DataFrame, options: PMAIV3BuildOptions) -> pd.DataFrame:
        if base.empty:
            return pd.DataFrame(columns=["date"])
        daily = (
            base.sort_values("prediction_date")
            .groupby("prediction_date", as_index=False)
            .agg(
                topix_return_5d=("topix_return_5d", "first"),
                topix_return_10d=("topix_return_10d", "first"),
                topix_return_20d=("topix_return_20d", "first"),
                market_turnover=("turnover_value", "sum"),
            )
        )
        daily = daily.rename(columns={"prediction_date": "date"})
        daily["topix_return_1d_proxy"] = _numeric(daily.get("topix_return_5d")).diff()
        daily["topix_ma_distance"] = _numeric(daily.get("topix_return_20d"))
        daily["topix_volatility"] = _numeric(daily.get("topix_return_1d_proxy")).rolling(20, min_periods=5).std()
        daily["market_volume_trend"] = _numeric(daily.get("market_turnover")) / _numeric(daily.get("market_turnover")).rolling(20, min_periods=5).mean()
        trend = _numeric(daily.get("topix_return_20d")).fillna(0.0)
        short = _numeric(daily.get("topix_return_5d")).fillna(0.0)
        vol = _numeric(daily.get("topix_volatility")).fillna(_numeric(daily.get("topix_volatility")).median()).fillna(0.0)
        daily["market_attack_score_prototype"] = trend + 0.5 * short - 0.5 * vol
        score = _numeric(daily.get("market_attack_score_prototype"))
        daily["market_regime_class_prototype"] = "neutral"
        daily.loc[score.ge(0.03), "market_regime_class_prototype"] = "attack"
        daily.loc[score.le(-0.03), "market_regime_class_prototype"] = "defensive"
        mask = (daily["date"] >= pd.Timestamp(options.start_date)) & (daily["date"] <= pd.Timestamp(options.end_date))
        return daily.loc[mask].copy()

    def _add_relative_labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        future_10 = _numeric(out.get("future_10d_return"))
        mfe = _numeric(out.get("future_max_return_10d"))
        if mfe.empty or "future_max_return_10d" not in out.columns:
            mfe = future_10
        out["max_favorable_excursion_10d"] = mfe
        out["max_adverse_excursion_10d"] = self._max_adverse_excursion_from_base(out)
        mae_penalty = _numeric(out["max_adverse_excursion_10d"]).clip(upper=0).abs()
        out["downside_penalized_return_10d"] = future_10 - mae_penalty * DOWNSIDE_PENALTY
        out["risk_adjusted_future_return_10d"] = out["downside_penalized_return_10d"]
        utility = _numeric(out["downside_penalized_return_10d"])
        grouped = out.assign(_utility=utility).groupby("prediction_date", group_keys=False)
        out["relative_future_utility_rank_in_day"] = grouped["_utility"].rank(method="first", ascending=False)
        valid_count = grouped["_utility"].transform("count")
        rank = _numeric(out["relative_future_utility_rank_in_day"])
        out["relative_future_utility_percentile_in_day"] = 1.0 - ((rank - 1.0) / (valid_count - 1.0))
        out.loc[valid_count.le(1), "relative_future_utility_percentile_in_day"] = 1.0
        top_cutoff = (valid_count * 0.10).clip(lower=1).apply(lambda value: math.ceil(value) if pd.notna(value) else pd.NA)
        bottom_start = valid_count - top_cutoff + 1
        out["top_decile_future_utility_in_day"] = rank.le(top_cutoff)
        out["bottom_decile_future_utility_in_day"] = rank.ge(bottom_start)
        return out.drop(columns=["_utility"], errors="ignore")

    def _max_adverse_excursion_from_base(self, frame: pd.DataFrame) -> pd.Series:
        if frame.empty or "close" not in frame.columns:
            return pd.Series(dtype=float)
        # The base dataset does not retain intraday high/low for Phase 9-B, so
        # prototype MAE uses future close-to-current-close returns.
        close = _numeric(frame.get("close"))
        full = _read_parquet(self.base_dataset)
        if full.empty or "close" not in full.columns:
            return _numeric(frame.get("future_10d_return")).clip(upper=0)
        full = full.copy()
        full["prediction_date"] = pd.to_datetime(full["date"], errors="coerce")
        full["code"] = full["code"].astype(str)
        full = full.sort_values(["code", "prediction_date"])
        full_close = _numeric(full["close"])
        grouped = full.assign(_close=full_close).groupby("code", sort=False)["_close"]
        future_returns = pd.concat(
            [(grouped.shift(-step) / full_close - 1.0).rename(str(step)) for step in range(1, 11)],
            axis=1,
        )
        full["max_adverse_excursion_10d"] = future_returns.min(axis=1, skipna=False)
        lookup = full[["prediction_date", "code", "max_adverse_excursion_10d"]].drop_duplicates(["prediction_date", "code"])
        joined = frame[["prediction_date", "code"]].merge(lookup, on=["prediction_date", "code"], how="left")
        return _numeric(joined["max_adverse_excursion_10d"])

    def _feature_columns(self, dataset: pd.DataFrame) -> list[str]:
        candidates = [
            *STOCK_PREDICTION_FEATURES,
            *MARKET_FEATURES,
            *PRICE_VOLUME_FEATURES,
            *FINANCIAL_FEATURES,
            *EVENT_FEATURES,
            *CONDITIONAL_RELATIVE_FEATURES,
        ]
        features: list[str] = []
        for column in candidates:
            if column not in dataset.columns:
                continue
            if self._is_label_like(column) or self._has_forbidden_token(column):
                continue
            features.append(column)
        return list(dict.fromkeys(features))

    def _finalize_dataset(self, dataset: pd.DataFrame, feature_columns: list[str], label_columns: list[str]) -> pd.DataFrame:
        if dataset.empty:
            return dataset
        keep = [*KEY_COLUMNS, *feature_columns, *label_columns, "data_source", "relative_feature_timing"]
        final = dataset[[column for column in keep if column in dataset.columns]].copy()
        final["prediction_date"] = pd.to_datetime(final["prediction_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        final["market_date"] = pd.to_datetime(final["market_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return final.dropna(subset=["prediction_date", "code"]).copy()

    def _leakage_audit(self, feature_columns: list[str], label_columns: list[str]) -> dict[str, Any]:
        forbidden = [column for column in feature_columns if self._has_forbidden_token(column)]
        label_in_features = [column for column in feature_columns if column in label_columns or self._is_label_like(column)]
        blocking = []
        if forbidden:
            blocking.append("forbidden feature token found")
        if label_in_features:
            blocking.append("label-like column found in features")
        return {
            "forbidden_feature_count": len(forbidden),
            "forbidden_feature_columns": forbidden,
            "label_columns_in_features": label_in_features,
            "conditional_feature_columns": [column for column in CONDITIONAL_RELATIVE_FEATURES if column in feature_columns],
            "future_return_label_exception": ALLOWED_LABEL_TOKENS,
            "current_pm_ai_overwritten": False,
            "current_exit_ai_overwritten": False,
            "v2_82_profile_overwritten": False,
            "blocking_issues": blocking,
            "leakage_risk": "low" if not blocking else "high",
        }

    def _report(
        self,
        dataset: pd.DataFrame,
        market_regime: pd.DataFrame,
        feature_columns: list[str],
        label_columns: list[str],
        leakage: dict[str, Any],
        options: PMAIV3BuildOptions,
    ) -> dict[str, Any]:
        return {
            "metadata": {
                "phase": "9-B",
                "dataset_purpose": "PM AI v3 clean research dataset",
                "training_executed": False,
                "backtest_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_model_regenerated_historical_predictions": False,
                "dataset_written": False,
                "market_regime_written": False,
                "downside_penalty": options.downside_penalty,
                "candidate_rule": f"top {options.top_n} by risk_adjusted_score after turnover_value >= {options.min_turnover_value:.0f}",
            },
            "input_paths": {
                "base_dataset": str(self.base_dataset),
                "walk_forward_predictions": str(self.walk_forward_dir),
            },
            "output_paths": {
                "dataset": str(self.dataset_output),
                "market_regime": str(self.market_regime_output),
                "markdown": str(self.root / "reports/ml" / f"{REPORT_STEM}.md"),
                "json": str(self.root / "reports/ml" / f"{REPORT_STEM}.json"),
            },
            "dataset_summary": self._dataset_summary(dataset),
            "market_regime_summary": self._market_regime_summary(market_regime),
            "feature_columns": feature_columns,
            "label_columns": label_columns,
            "forbidden_feature_columns": leakage["forbidden_feature_columns"],
            "conditional_feature_columns": leakage["conditional_feature_columns"],
            "feature_classification": self._feature_classification(feature_columns),
            "label_classification": self._label_classification(),
            "leakage_audit": leakage,
            "label_null_rate": self._null_rates(dataset, label_columns),
            "feature_null_rate_top": self._null_rates(dataset, feature_columns)[:30],
            "relative_feature_timing": "Relative features are computed after each day's Stock Selection walk-forward candidate pool is finalized and before any cash, portfolio, selected/bought/affordable, backtest, or exit decision.",
            "phase9c_model_plan": self._phase9c_model_plan(),
            "current_artifact_safety": {
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
                "current_v282_maintained": True,
            },
        }

    def _dataset_summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        if dataset.empty:
            return {
                "row_count": 0,
                "date_min": None,
                "date_max": None,
                "code_count": 0,
                "candidate_count_stats": {},
            }
        dates = pd.to_datetime(dataset["prediction_date"], errors="coerce")
        counts = dataset.groupby("prediction_date")["code"].count()
        return {
            "row_count": int(len(dataset)),
            "date_min": dates.min().strftime("%Y-%m-%d") if not dates.dropna().empty else None,
            "date_max": dates.max().strftime("%Y-%m-%d") if not dates.dropna().empty else None,
            "code_count": int(dataset["code"].nunique()),
            "candidate_count_stats": {
                "min": int(counts.min()) if not counts.empty else 0,
                "mean": float(counts.mean()) if not counts.empty else 0.0,
                "median": float(counts.median()) if not counts.empty else 0.0,
                "max": int(counts.max()) if not counts.empty else 0,
            },
        }

    def _market_regime_summary(self, market_regime: pd.DataFrame) -> dict[str, Any]:
        if market_regime.empty:
            return {"row_count": 0, "class_distribution": {}}
        counts = market_regime.get("market_regime_class_prototype", pd.Series(dtype=str)).value_counts().to_dict()
        return {"row_count": int(len(market_regime)), "class_distribution": {str(k): int(v) for k, v in counts.items()}}

    def _feature_classification(self, feature_columns: list[str]) -> list[dict[str, Any]]:
        groups = [
            ("stock_selection_prediction", STOCK_PREDICTION_FEATURES, "historical walk-forward prediction score columns"),
            ("market", MARKET_FEATURES, "TOPIX/market prototype features"),
            ("price_volume", PRICE_VOLUME_FEATURES, "J-Quants price, technical, volume, liquidity features"),
            ("financial", FINANCIAL_FEATURES, "J-Quants financial statement features"),
            ("event", EVENT_FEATURES, "earnings/event distance features when available"),
            ("conditional_relative", CONDITIONAL_RELATIVE_FEATURES, "prediction-time same-day candidate-relative features"),
        ]
        feature_set = set(feature_columns)
        return [
            {
                "category": name,
                "count": len([column for column in columns if column in feature_set]),
                "columns": [column for column in columns if column in feature_set],
                "rule": rule,
            }
            for name, columns, rule in groups
        ]

    def _label_classification(self) -> list[dict[str, Any]]:
        definitions = {
            "future_5d_return": "5 business-day future return from J-Quants price",
            "future_10d_return": "10 business-day future return from J-Quants price",
            "max_favorable_excursion_10d": "maximum favorable 10d future close/high-return proxy",
            "max_adverse_excursion_10d": "minimum 10d future close-return proxy",
            "downside_penalized_return_10d": f"future_10d_return - abs(min(0, MAE_10d)) * {DOWNSIDE_PENALTY}",
            "risk_adjusted_future_return_10d": "downside-penalized future utility",
            "relative_future_utility_rank_in_day": "same-day rank by future utility label",
            "relative_future_utility_percentile_in_day": "same-day percentile by future utility label",
            "top_decile_future_utility_in_day": "same-day top decile by future utility label",
            "bottom_decile_future_utility_in_day": "same-day bottom decile by future utility label",
        }
        return [
            {
                "label": column,
                "definition": definitions[column],
                "source": "future J-Quants price label",
                "feature_allowed": False,
            }
            for column in LABEL_COLUMNS
        ]

    def _null_rates(self, dataset: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
        if dataset.empty:
            return [{"column": column, "null_rate": None} for column in columns]
        rows = []
        for column in columns:
            if column in dataset.columns:
                rows.append({"column": column, "null_rate": float(dataset[column].isna().mean())})
        return sorted(rows, key=lambda row: (-1.0 if row["null_rate"] is None else -row["null_rate"], row["column"]))

    def _phase9c_model_plan(self) -> list[dict[str, Any]]:
        return [
            {
                "model": "Market Regime prototype model",
                "purpose": "attack/neutral/defensive daily context",
                "target": "market_regime_class_prototype or future market utility label",
                "notes": "do not use strategy PnL-defined good days",
            },
            {
                "model": "Candidate Ranking model",
                "purpose": "rank same-day candidates",
                "target": "relative_future_utility_rank_in_day or top_decile_future_utility_in_day",
                "notes": "ranking objective first; classifier only as baseline",
            },
            {
                "model": "Position Sizing model",
                "purpose": "map regime and candidate quality to PM multiplier buckets",
                "target": "future utility / downside bucket labels",
                "notes": "do not imitate current PM multiplier as the sole target",
            },
        ]

    def _has_forbidden_token(self, column: str) -> bool:
        lower = column.lower()
        return any(token in lower for token in FORBIDDEN_TOKENS)

    def _is_label_like(self, column: str) -> bool:
        lower = column.lower()
        return lower.startswith("future_") or "target" in lower or "label" in lower or column in LABEL_COLUMNS

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
        if isinstance(value, dict):
            return ", ".join(f"{key}:{val}" for key, val in value.items())
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value).replace("\n", " ")


def build_phase9b_pm_ai_v3_dataset(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3CleanDatasetBuilder(root).build()
