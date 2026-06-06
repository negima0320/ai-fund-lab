from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROFILE = "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue"
PERIOD_LABEL = "2023-01_to_2026-05"

ML_FEATURE_COLUMNS = [
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "risk_adjusted_score",
]

PRICE_FEATURE_COLUMNS = [
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
    "close_position",
    "gap_up_ratio",
    "daily_range_ratio",
    "volume",
    "turnover_value",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "turnover_ratio_5d",
    "turnover_ratio_20d",
]

TOPIX_FEATURE_COLUMNS = [
    "topix_return_5d",
    "topix_return_10d",
    "topix_return_20d",
    "relative_return_5d",
    "relative_return_10d",
    "relative_return_20d",
]

FINANCIAL_FEATURE_COLUMNS = [
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

EARNINGS_FEATURE_COLUMNS = ["days_to_earnings", "is_near_earnings"]

RELATIVE_FEATURE_COLUMNS = [
    "rank_in_day",
    "score_rank_in_day",
    "risk_adjusted_score_percentile_in_day",
    "expected_return_percentile_in_day",
    "bad_entry_percentile_in_day",
    "score_gap_to_best",
    "expected_return_gap_to_best",
    "bad_entry_gap_to_best",
]

DAY_FEATURE_COLUMNS = [
    "candidate_count_in_day",
    "day_avg_risk_adjusted_score",
    "day_max_risk_adjusted_score",
    "day_avg_expected_return_10d",
    "day_avg_bad_entry_probability",
]

CAPITAL_STATE_FEATURE_COLUMNS = [
    "current_capital_utilization",
    "current_positions_count",
    "cash_before_ratio",
]

FEATURE_COLUMNS = (
    ML_FEATURE_COLUMNS
    + PRICE_FEATURE_COLUMNS
    + TOPIX_FEATURE_COLUMNS
    + FINANCIAL_FEATURE_COLUMNS
    + EARNINGS_FEATURE_COLUMNS
    + RELATIVE_FEATURE_COLUMNS
    + DAY_FEATURE_COLUMNS
    + CAPITAL_STATE_FEATURE_COLUMNS
)

LABEL_COLUMNS = [
    "realized_return",
    "positive_trade",
    "high_conviction_target",
    "avoid_target",
    "ideal_weight_bucket",
    "ideal_cash_reserve_bucket",
    "future_5d_return",
    "future_10d_return",
]

AUDIT_COLUMNS = [
    "decision",
    "actual_buy_amount",
    "actual_shares",
    "actual_net_profit",
    "actual_holding_days",
    "skip_reason",
    "exit_reason",
]

CLEAN_ML_FEATURE_COLUMNS = ML_FEATURE_COLUMNS

CLEAN_PRICE_FEATURE_COLUMNS = PRICE_FEATURE_COLUMNS

CLEAN_TOPIX_FEATURE_COLUMNS = TOPIX_FEATURE_COLUMNS

CLEAN_FINANCIAL_FEATURE_COLUMNS = FINANCIAL_FEATURE_COLUMNS

CLEAN_EARNINGS_FEATURE_COLUMNS = EARNINGS_FEATURE_COLUMNS

CLEAN_RELATIVE_FEATURE_COLUMNS = [
    "rank_in_day",
    "score_rank_in_day",
    "risk_adjusted_score_percentile_in_day",
    "expected_return_percentile_in_day",
    "expected_max_return_percentile_in_day",
    "swing_success_percentile_in_day",
    "bad_entry_percentile_in_day",
    "score_gap_to_best",
    "expected_return_gap_to_best",
    "expected_max_return_gap_to_best",
    "swing_success_gap_to_best",
    "bad_entry_gap_to_best",
    "candidate_count_in_day",
]

CLEAN_DAY_FEATURE_COLUMNS = [
    "day_avg_risk_adjusted_score",
    "day_max_risk_adjusted_score",
    "day_avg_expected_return_10d",
    "day_avg_expected_max_return_20d",
    "day_avg_swing_success_probability_20d",
    "day_avg_bad_entry_probability",
    "day_candidate_strength",
    "day_risk_level",
]

CLEAN_FEATURE_COLUMNS = (
    CLEAN_ML_FEATURE_COLUMNS
    + CLEAN_PRICE_FEATURE_COLUMNS
    + CLEAN_TOPIX_FEATURE_COLUMNS
    + CLEAN_FINANCIAL_FEATURE_COLUMNS
    + CLEAN_EARNINGS_FEATURE_COLUMNS
    + CLEAN_RELATIVE_FEATURE_COLUMNS
    + CLEAN_DAY_FEATURE_COLUMNS
)

CLEAN_FORBIDDEN_FEATURE_COLUMNS = [
    "decision",
    "skip_reason",
    "exit_reason",
    "actual_buy_amount",
    "actual_shares",
    "actual_net_profit",
    "actual_return",
    "actual_holding_days",
    "cash_before",
    "cash_after",
    "cash_before_ratio",
    "current_capital_utilization",
    "current_positions_count",
    "daily_buy_limit_remaining_before",
    "daily_buy_limit_remaining_after",
    "max_positions_remaining_before",
    "max_positions_remaining_after",
    "final_amount",
    "final_shares",
    "final_price",
    "scaled_amount",
    "scaled_shares",
    "planned_amount",
    "planned_shares",
    "affordable_amount",
    "affordable_shares_100lot",
    "is_affordable_100lot",
    "affordability_ratio",
    "target_exposure",
]


@dataclass(frozen=True)
class PortfolioManagerDatasetPaths:
    dataset: Path
    markdown: Path
    json: Path


class PortfolioManagerDatasetBuilder:
    def __init__(
        self,
        root: str | Path = ".",
        profile: str = PROFILE,
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        initial_cash: float = 1_000_000.0,
        max_positions: int = 10,
    ) -> None:
        self.root = Path(root)
        self.profile = profile
        self.start_date = start_date
        self.end_date = end_date
        self.period_key = f"{start_date}_to_{end_date}"
        self.period_label = f"{start_date[:7]}_to_{end_date[:7]}"
        self.initial_cash = float(initial_cash)
        self.max_positions = int(max_positions)
        self.feature_root = self.root / "data" / "ml" / "features"
        self.label_root = self.root / "data" / "ml" / "labels"
        self.prediction_root = self.root / "data" / "ml" / "walk_forward_predictions"
        self.output_root = self.root / "data" / "ml" / "portfolio_manager"
        self.report_root = self.root / "reports" / "ml"

    def build_dataset(self) -> pd.DataFrame:
        audit = self._load_purchase_audit()
        if audit.empty:
            return audit
        dataset = self._join_features(audit)
        dataset = self._join_predictions(dataset)
        dataset = self._join_labels(dataset)
        dataset = self._join_trade_results(dataset)
        dataset = self._add_candidate_relative_features(dataset)
        dataset = self._add_day_features(dataset)
        dataset = self._add_capital_state_features(dataset)
        dataset = self._add_label_columns(dataset)
        dataset = self._finalize_columns(dataset)
        return dataset

    def save(self, dataset: pd.DataFrame) -> PortfolioManagerDatasetPaths:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.report_root.mkdir(parents=True, exist_ok=True)
        dataset_path = self.output_root / f"portfolio_manager_dataset_v2_73_{self.period_label}.parquet"
        markdown = self.report_root / f"portfolio_manager_phase2_dataset_summary_{self.period_label}.md"
        json_path = self.report_root / f"portfolio_manager_phase2_dataset_summary_{self.period_label}.json"
        dataset.to_parquet(dataset_path, index=False)
        summary = self.summary(dataset)
        markdown.write_text(self.format_markdown(summary), encoding="utf-8")
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerDatasetPaths(dataset=dataset_path, markdown=markdown, json=json_path)

    def summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        feature_rates = {
            column: float(dataset[column].notna().mean())
            for column in FEATURE_COLUMNS
            if column in dataset.columns
        }
        label_distribution = {
            "positive_trade_rate": self._rate(dataset, "positive_trade"),
            "high_conviction_target_rate": self._rate(dataset, "high_conviction_target"),
            "avoid_target_rate": self._rate(dataset, "avoid_target"),
            "ideal_weight_bucket": self._value_counts(dataset, "ideal_weight_bucket"),
            "ideal_cash_reserve_bucket": self._value_counts(dataset, "ideal_cash_reserve_bucket"),
        }
        prediction_columns = ["expected_return_10d", "bad_entry_probability_10d", "risk_adjusted_score"]
        prediction_joined = dataset[prediction_columns].notna().all(axis=1) if not dataset.empty else pd.Series(dtype=bool)
        leakage = self.leakage_audit()
        return {
            "profile": self.profile,
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "rows": int(len(dataset)),
            "unique_dates": int(dataset["signal_date"].nunique()) if "signal_date" in dataset.columns else 0,
            "unique_codes": int(dataset["code"].nunique()) if "code" in dataset.columns else 0,
            "feature_count": int(len(FEATURE_COLUMNS)),
            "feature_columns": FEATURE_COLUMNS,
            "label_columns": LABEL_COLUMNS,
            "audit_columns": AUDIT_COLUMNS,
            "prediction_join_rate": float(prediction_joined.mean()) if len(prediction_joined) else None,
            "prediction_source_distribution": self._value_counts(dataset, "prediction_source"),
            "feature_non_null_rates": feature_rates,
            "feature_non_null_summary": self._non_null_summary(feature_rates),
            "label_distribution": label_distribution,
            "decision_distribution": self._value_counts(dataset, "decision"),
            "leakage_audit": leakage,
            "quality_assessment": self._quality_assessment(dataset, feature_rates, prediction_joined),
        }

    def format_markdown(self, summary: dict[str, Any]) -> str:
        low_features = [
            {"feature": name, "non_null_rate": rate}
            for name, rate in sorted(summary["feature_non_null_rates"].items(), key=lambda item: item[1])
            if rate < 0.50
        ][:20]
        lines = [
            "# Portfolio Manager AI Phase 2 Dataset Summary",
            "",
            f"- profile: `{summary['profile']}`",
            f"- period: {summary['period']['start_date']} to {summary['period']['end_date']}",
            "- unit: one candidate per signal_date/code",
            "- feature policy: J-Quants/features plus historical walk-forward prediction snapshots only.",
            "- prediction parquet is preferred when available.",
            "- if prediction parquet is unavailable, purchase_audit is used only as an already-joined historical prediction snapshot fallback.",
            "- purchase_audit/trades decision and result columns are audit/label columns only, never training features.",
            "",
            "## Dataset",
            "",
            self._table(
                [summary],
                ["rows", "unique_dates", "unique_codes", "feature_count", "prediction_join_rate"],
            ),
            "",
            "## Feature Non-Null Summary",
            "",
            self._table([summary["feature_non_null_summary"]], ["mean", "min", "max", "below_50pct", "above_90pct"]),
            "",
            "## Low Coverage Features",
            "",
            self._table(low_features, ["feature", "non_null_rate"]),
            "",
            "## Label Distribution",
            "",
            self._table(
                [
                    {
                        "positive_trade_rate": summary["label_distribution"]["positive_trade_rate"],
                        "high_conviction_target_rate": summary["label_distribution"]["high_conviction_target_rate"],
                        "avoid_target_rate": summary["label_distribution"]["avoid_target_rate"],
                    }
                ],
                ["positive_trade_rate", "high_conviction_target_rate", "avoid_target_rate"],
            ),
            "",
            "### ideal_weight_bucket",
            "",
            self._table(summary["label_distribution"]["ideal_weight_bucket"], ["value", "count", "rate"]),
            "",
            "### ideal_cash_reserve_bucket",
            "",
            self._table(summary["label_distribution"]["ideal_cash_reserve_bucket"], ["value", "count", "rate"]),
            "",
            "## Prediction Source",
            "",
            self._table(summary["prediction_source_distribution"], ["value", "count", "rate"]),
            "",
            "## Leakage Audit",
            "",
        ]
        for item in summary["leakage_audit"]:
            lines.append(f"- {item}")
        lines.extend(["", "## Quality Assessment", ""])
        for item in summary["quality_assessment"]:
            lines.append(f"- {item}")
        lines.append("")
        return "\n".join(lines)

    def leakage_audit(self) -> list[str]:
        forbidden = set(AUDIT_COLUMNS + ["actual_return", "actual_net_profit_rate", "exit_date"])
        leaked = sorted(forbidden.intersection(FEATURE_COLUMNS))
        lines = [
            "Feature columns exclude decision, skip_reason, exit_reason, actual buy amount, shares, holding days, and realized PnL.",
            "Future labels are derived after feature assembly and are not included in FEATURE_COLUMNS.",
            "Historical ML prediction values are loaded from walk-forward prediction parquet when available; no current model inference is run.",
            "If standalone prediction parquet is unavailable, purchase_audit is used only as a carrier of already-joined walk-forward prediction columns, not decision/result columns.",
            "J-Quants feature parquet is keyed by signal_date/code and was generated by FeatureBuilder with target-date-only data.",
        ]
        if leaked:
            lines.append(f"FAIL: forbidden audit/result columns found in feature list: {', '.join(leaked)}")
        else:
            lines.append("PASS: no audit/result columns are present in FEATURE_COLUMNS.")
        return lines

    def _load_purchase_audit(self) -> pd.DataFrame:
        path = self.root / "logs" / "backtests" / self.profile / self.period_key / "purchase_audit.csv"
        if not path.exists():
            return pd.DataFrame()
        data = pd.read_csv(path)
        data["code"] = data["code"].astype(str)
        data["signal_date"] = pd.to_datetime(data["signal_date"], errors="coerce")
        if "entry_date" in data.columns:
            data["entry_date"] = pd.to_datetime(data["entry_date"], errors="coerce")
        for column in [
            "candidate_rank",
            "score_rank",
            "risk_adjusted_score",
            "expected_return_10d",
            "bad_entry_probability_10d",
            "cash_before",
            "cash_after",
            "daily_buy_limit_remaining_before",
            "daily_buy_limit_remaining_after",
            "max_positions_remaining_before",
            "final_amount",
            "final_shares",
        ]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data[(data["signal_date"] >= pd.Timestamp(self.start_date)) & (data["signal_date"] <= pd.Timestamp(self.end_date))]
        return data.reset_index(drop=True)

    def _join_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        rows = []
        feature_columns = ["date", "code", *PRICE_FEATURE_COLUMNS, *TOPIX_FEATURE_COLUMNS, *FINANCIAL_FEATURE_COLUMNS, *EARNINGS_FEATURE_COLUMNS]
        for date, group in dataset.groupby("signal_date", dropna=False):
            if pd.isna(date):
                rows.append(group)
                continue
            date_text = pd.Timestamp(date).strftime("%Y-%m-%d")
            path = self.feature_root / f"features_{date_text}.parquet"
            if not path.exists():
                rows.append(group)
                continue
            features = pd.read_parquet(path)
            features["code"] = features["code"].astype(str)
            available = [column for column in feature_columns if column in features.columns]
            features = features[available].drop_duplicates(subset=["code"], keep="last")
            merged = group.merge(features.drop(columns=["date"], errors="ignore"), on="code", how="left", suffixes=("", "_feature"))
            rows.append(merged)
        return pd.concat(rows, ignore_index=True) if rows else dataset

    def _join_predictions(self, dataset: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for date, group in dataset.groupby("signal_date", dropna=False):
            group = group.copy()
            if pd.isna(date):
                rows.append(group)
                continue
            date_text = pd.Timestamp(date).strftime("%Y-%m-%d")
            path = self.prediction_root / f"predictions_{date_text}.parquet"
            if path.exists():
                predictions = pd.read_parquet(path)
                predictions["code"] = predictions["code"].astype(str)
                if "risk_adjusted_score" not in predictions.columns:
                    predictions["risk_adjusted_score"] = (
                        pd.to_numeric(predictions.get("expected_return_10d"), errors="coerce")
                        - 0.5 * pd.to_numeric(predictions.get("bad_entry_probability_10d"), errors="coerce")
                    )
                cols = ["code", *[column for column in ML_FEATURE_COLUMNS if column in predictions.columns]]
                group = group.merge(predictions[cols].drop_duplicates("code"), on="code", how="left", suffixes=("", "_prediction"))
                for column in ML_FEATURE_COLUMNS:
                    pred_col = f"{column}_prediction"
                    if pred_col in group.columns:
                        if column not in group.columns:
                            group[column] = pd.NA
                        group[column] = pd.to_numeric(group[pred_col], errors="coerce").fillna(
                            pd.to_numeric(group[column], errors="coerce")
                        )
                        group = group.drop(columns=[pred_col])
                core = ["expected_return_10d", "bad_entry_probability_10d", "risk_adjusted_score"]
                group["prediction_source"] = "prediction_parquet"
                group.loc[~group[core].notna().all(axis=1), "prediction_source"] = pd.NA
            if "risk_adjusted_score" not in group.columns or group["risk_adjusted_score"].isna().any():
                group["risk_adjusted_score"] = pd.to_numeric(group.get("risk_adjusted_score"), errors="coerce").fillna(
                    pd.to_numeric(group.get("expected_return_10d"), errors="coerce")
                    - 0.5 * pd.to_numeric(group.get("bad_entry_probability_10d"), errors="coerce")
                )
            if "prediction_source" not in group.columns:
                group["prediction_source"] = pd.NA
            core_joined = group[["expected_return_10d", "bad_entry_probability_10d", "risk_adjusted_score"]].notna().all(axis=1)
            group.loc[group["prediction_source"].isna() & core_joined, "prediction_source"] = "purchase_audit_prediction_snapshot"
            group["prediction_joined"] = core_joined
            rows.append(group)
        return pd.concat(rows, ignore_index=True) if rows else dataset

    def _join_labels(self, dataset: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for date, group in dataset.groupby("signal_date", dropna=False):
            if pd.isna(date):
                rows.append(group)
                continue
            date_text = pd.Timestamp(date).strftime("%Y-%m-%d")
            path = self.label_root / f"labels_{date_text}.parquet"
            if not path.exists():
                rows.append(group)
                continue
            labels = pd.read_parquet(path)
            labels["code"] = labels["code"].astype(str)
            label_cols = [column for column in ["code", "future_5d_return", "future_10d_return"] if column in labels.columns]
            rows.append(group.merge(labels[label_cols].drop_duplicates("code"), on="code", how="left"))
        return pd.concat(rows, ignore_index=True) if rows else dataset

    def _join_trade_results(self, dataset: pd.DataFrame) -> pd.DataFrame:
        path = self.root / "logs" / "backtests" / self.profile / self.period_key / "trades.csv"
        dataset["actual_net_profit"] = pd.NA
        dataset["actual_holding_days"] = pd.NA
        dataset["exit_reason"] = pd.NA
        if not path.exists():
            return dataset
        trades = pd.read_csv(path)
        if "action" in trades.columns:
            trades = trades[trades["action"].astype(str).eq("SELL")].copy()
        trades["code"] = trades["code"].astype(str)
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in trades.columns:
                trades[column] = pd.to_datetime(trades[column], errors="coerce")
        for column in ["net_profit", "holding_days"]:
            if column in trades.columns:
                trades[column] = pd.to_numeric(trades[column], errors="coerce")
        keys = ["signal_date", "entry_date", "code"]
        result = trades[keys + ["net_profit", "holding_days", "exit_reason"]].drop_duplicates(keys, keep="last")
        merged = dataset.merge(result, on=keys, how="left", suffixes=("", "_trade"))
        merged["actual_net_profit"] = pd.to_numeric(merged["net_profit"], errors="coerce")
        merged["actual_holding_days"] = pd.to_numeric(merged["holding_days"], errors="coerce")
        if "exit_reason_trade" in merged.columns:
            merged["exit_reason"] = merged["exit_reason_trade"]
        return merged.drop(columns=[column for column in ["net_profit", "holding_days", "exit_reason_trade"] if column in merged.columns])

    def _add_candidate_relative_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data = dataset.copy()
        data["rank_in_day"] = pd.to_numeric(data.get("candidate_rank"), errors="coerce")
        data["score_rank_in_day"] = pd.to_numeric(data.get("score_rank"), errors="coerce")
        grouped = data.groupby("signal_date", dropna=False)
        data["risk_adjusted_score_percentile_in_day"] = grouped["risk_adjusted_score"].rank(pct=True)
        data["expected_return_percentile_in_day"] = grouped["expected_return_10d"].rank(pct=True)
        # Lower bad_entry is better, so descending=False means low risk receives low percentile.
        data["bad_entry_percentile_in_day"] = grouped["bad_entry_probability_10d"].rank(pct=True)
        data["score_gap_to_best"] = grouped["risk_adjusted_score"].transform("max") - data["risk_adjusted_score"]
        data["expected_return_gap_to_best"] = grouped["expected_return_10d"].transform("max") - data["expected_return_10d"]
        data["bad_entry_gap_to_best"] = data["bad_entry_probability_10d"] - grouped["bad_entry_probability_10d"].transform("min")
        return data

    def _add_day_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data = dataset.copy()
        grouped = data.groupby("signal_date", dropna=False)
        data["candidate_count_in_day"] = grouped["code"].transform("count")
        data["day_avg_risk_adjusted_score"] = grouped["risk_adjusted_score"].transform("mean")
        data["day_max_risk_adjusted_score"] = grouped["risk_adjusted_score"].transform("max")
        data["day_avg_expected_return_10d"] = grouped["expected_return_10d"].transform("mean")
        data["day_avg_bad_entry_probability"] = grouped["bad_entry_probability_10d"].transform("mean")
        return data

    def _add_capital_state_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data = dataset.copy()
        cash = pd.to_numeric(data.get("cash_before"), errors="coerce")
        data["cash_before_ratio"] = cash / self.initial_cash
        data["current_capital_utilization"] = (1.0 - data["cash_before_ratio"]).clip(lower=0.0)
        remaining = pd.to_numeric(data.get("max_positions_remaining_before"), errors="coerce")
        data["current_positions_count"] = self.max_positions - remaining
        return data

    def _add_label_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data = dataset.copy()
        data["realized_return"] = pd.to_numeric(data.get("future_10d_return"), errors="coerce")
        data["positive_trade"] = data["realized_return"] > 0
        grouped = data.groupby("signal_date", dropna=False)["realized_return"]
        top_threshold = grouped.transform(lambda series: series.quantile(0.80))
        bottom_threshold = grouped.transform(lambda series: series.quantile(0.20))
        data["high_conviction_target"] = data["realized_return"] >= top_threshold
        data["avoid_target"] = (data["realized_return"] <= bottom_threshold) | (data["realized_return"] <= -0.05)
        data["ideal_weight_bucket"] = "normal"
        data.loc[data["high_conviction_target"], "ideal_weight_bucket"] = "strong"
        data.loc[data["avoid_target"], "ideal_weight_bucket"] = "weak"
        day_avg = data.groupby("signal_date", dropna=False)["realized_return"].transform("mean")
        data["ideal_cash_reserve_bucket"] = "normal"
        data.loc[day_avg > 0.01, "ideal_cash_reserve_bucket"] = "aggressive"
        data.loc[day_avg < 0.0, "ideal_cash_reserve_bucket"] = "defensive"
        data["actual_buy_amount"] = pd.to_numeric(data.get("final_amount"), errors="coerce")
        data["actual_shares"] = pd.to_numeric(data.get("final_shares"), errors="coerce")
        return data

    def _finalize_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data = dataset.copy()
        data["signal_date"] = pd.to_datetime(data["signal_date"], errors="coerce")
        data["code"] = data["code"].astype(str)
        for column in FEATURE_COLUMNS + LABEL_COLUMNS:
            if column not in data.columns:
                data[column] = pd.NA
        for column in AUDIT_COLUMNS:
            if column not in data.columns:
                data[column] = pd.NA
        ordered = ["signal_date", "code", *FEATURE_COLUMNS, *LABEL_COLUMNS, *AUDIT_COLUMNS]
        extra = [column for column in data.columns if column not in ordered]
        return data[ordered + extra]

    def _quality_assessment(self, dataset: pd.DataFrame, feature_rates: dict[str, float], prediction_joined: pd.Series) -> list[str]:
        lines = []
        join_rate = float(prediction_joined.mean()) if len(prediction_joined) else 0.0
        lines.append(f"prediction join rate={join_rate:.4f}.")
        low = [name for name, rate in feature_rates.items() if rate < 0.50]
        if low:
            lines.append(f"{len(low)} features are below 50% non-null coverage: {', '.join(low[:10])}.")
        else:
            lines.append("all tracked features have at least 50% non-null coverage.")
        if join_rate >= 0.95 and dataset["realized_return"].notna().mean() >= 0.95:
            lines.append("Dataset quality is sufficient to proceed to Phase 3 model experiments.")
        else:
            lines.append("Dataset is usable for inspection, but low prediction/label coverage should be reviewed before training.")
        return lines

    def _non_null_summary(self, rates: dict[str, float]) -> dict[str, Any]:
        values = list(rates.values())
        if not values:
            return {"mean": None, "min": None, "max": None, "below_50pct": 0, "above_90pct": 0}
        return {
            "mean": float(sum(values) / len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "below_50pct": int(sum(value < 0.50 for value in values)),
            "above_90pct": int(sum(value >= 0.90 for value in values)),
        }

    def _rate(self, dataset: pd.DataFrame, column: str) -> float | None:
        if dataset.empty or column not in dataset.columns:
            return None
        return float(dataset[column].fillna(False).astype(bool).mean())

    def _value_counts(self, dataset: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        if dataset.empty or column not in dataset.columns:
            return []
        counts = dataset[column].fillna("<NA>").astype(str).value_counts()
        total = int(counts.sum())
        return [{"value": value, "count": int(count), "rate": float(count / total) if total else None} for value, count in counts.items()]

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("|" + "|".join(self._format(row.get(column)) for column in columns) + "|")
        return "\n".join(lines)

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("\n", " ")


class CleanPortfolioManagerDatasetBuilder(PortfolioManagerDatasetBuilder):
    """Build a clean Portfolio Manager dataset with only J-Quants and ML-prediction features."""

    def save(self, dataset: pd.DataFrame) -> PortfolioManagerDatasetPaths:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.report_root.mkdir(parents=True, exist_ok=True)
        dataset_path = self.output_root / f"portfolio_manager_dataset_v2_73_clean_{self.period_label}.parquet"
        markdown = self.report_root / f"portfolio_manager_phase3b_clean_dataset_summary_{self.period_label}.md"
        json_path = self.report_root / f"portfolio_manager_phase3b_clean_dataset_summary_{self.period_label}.json"
        dataset.to_parquet(dataset_path, index=False)
        summary = self.summary(dataset)
        markdown.write_text(self.format_markdown(summary), encoding="utf-8")
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerDatasetPaths(dataset=dataset_path, markdown=markdown, json=json_path)

    def build_dataset(self) -> pd.DataFrame:
        audit = self._load_purchase_audit()
        if audit.empty:
            return audit
        dataset = self._join_features(audit)
        dataset = self._join_predictions(dataset)
        dataset = self._join_labels(dataset)
        dataset = self._join_trade_results(dataset)
        dataset = self._add_candidate_relative_features(dataset)
        dataset = self._add_day_features(dataset)
        dataset = self._add_label_columns(dataset)
        dataset = self._finalize_columns(dataset)
        return dataset

    def summary(self, dataset: pd.DataFrame) -> dict[str, Any]:
        feature_rates = {
            column: float(dataset[column].notna().mean())
            for column in CLEAN_FEATURE_COLUMNS
            if column in dataset.columns
        }
        prediction_columns = ["expected_return_10d", "bad_entry_probability_10d", "risk_adjusted_score"]
        prediction_joined = dataset[prediction_columns].notna().all(axis=1) if not dataset.empty else pd.Series(dtype=bool)
        forbidden_in_features = sorted(set(CLEAN_FORBIDDEN_FEATURE_COLUMNS).intersection(CLEAN_FEATURE_COLUMNS))
        label_distribution = {
            "positive_trade_rate": self._rate(dataset, "positive_trade"),
            "high_conviction_target_rate": self._rate(dataset, "high_conviction_target"),
            "avoid_target_rate": self._rate(dataset, "avoid_target"),
            "ideal_weight_bucket": self._value_counts(dataset, "ideal_weight_bucket"),
            "ideal_cash_reserve_bucket": self._value_counts(dataset, "ideal_cash_reserve_bucket"),
        }
        return {
            "profile": self.profile,
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "rows": int(len(dataset)),
            "unique_dates": int(dataset["signal_date"].nunique()) if "signal_date" in dataset.columns else 0,
            "unique_codes": int(dataset["code"].nunique()) if "code" in dataset.columns else 0,
            "feature_count": int(len(CLEAN_FEATURE_COLUMNS)),
            "feature_columns": CLEAN_FEATURE_COLUMNS,
            "label_columns": LABEL_COLUMNS,
            "audit_columns": AUDIT_COLUMNS,
            "forbidden_feature_columns": CLEAN_FORBIDDEN_FEATURE_COLUMNS,
            "forbidden_columns_in_features": forbidden_in_features,
            "prediction_join_rate": float(prediction_joined.mean()) if len(prediction_joined) else None,
            "expected_max_return_20d_non_null_rate": float(dataset["expected_max_return_20d"].notna().mean())
            if "expected_max_return_20d" in dataset.columns and len(dataset)
            else None,
            "swing_success_probability_20d_non_null_rate": float(dataset["swing_success_probability_20d"].notna().mean())
            if "swing_success_probability_20d" in dataset.columns and len(dataset)
            else None,
            "prediction_source_distribution": self._value_counts(dataset, "prediction_source"),
            "feature_non_null_rates": feature_rates,
            "feature_non_null_summary": self._non_null_summary(feature_rates),
            "label_distribution": label_distribution,
            "decision_distribution": self._value_counts(dataset, "decision"),
            "leakage_audit": self.leakage_audit(),
            "quality_assessment": self._quality_assessment(dataset, feature_rates, prediction_joined),
        }

    def format_markdown(self, summary: dict[str, Any]) -> str:
        low_features = [
            {"feature": name, "non_null_rate": rate}
            for name, rate in sorted(summary["feature_non_null_rates"].items(), key=lambda item: item[1])
            if rate < 0.50
        ][:20]
        lines = [
            "# Portfolio Manager AI Phase 3-B Clean Dataset Summary",
            "",
            f"- profile: `{summary['profile']}`",
            f"- period: {summary['period']['start_date']} to {summary['period']['end_date']}",
            "- unit: one candidate per signal_date/code",
            "- feature policy: J-Quants/features plus walk-forward ML prediction parquet only.",
            "- purchase_audit/trades are used only for candidate reconstruction, labels, result matching, and audit columns.",
            "",
            "## Dataset",
            "",
            self._table(
                [summary],
                [
                    "rows",
                    "unique_dates",
                    "unique_codes",
                    "feature_count",
                    "prediction_join_rate",
                    "expected_max_return_20d_non_null_rate",
                    "swing_success_probability_20d_non_null_rate",
                ],
            ),
            "",
            "## Feature Columns",
            "",
            self._table([{"feature": column} for column in summary["feature_columns"]], ["feature"]),
            "",
            "## Label Columns",
            "",
            self._table([{"label": column} for column in summary["label_columns"]], ["label"]),
            "",
            "## Audit Columns",
            "",
            self._table([{"audit": column} for column in summary["audit_columns"]], ["audit"]),
            "",
            "## Forbidden Feature Check",
            "",
            self._table([{"forbidden_columns_in_features": ", ".join(summary["forbidden_columns_in_features"]) or "none"}], ["forbidden_columns_in_features"]),
            "",
            "## Feature Non-Null Summary",
            "",
            self._table([summary["feature_non_null_summary"]], ["mean", "min", "max", "below_50pct", "above_90pct"]),
            "",
            "## Low Coverage Features",
            "",
            self._table(low_features, ["feature", "non_null_rate"]),
            "",
            "## Label Distribution",
            "",
            self._table(
                [
                    {
                        "positive_trade_rate": summary["label_distribution"]["positive_trade_rate"],
                        "high_conviction_target_rate": summary["label_distribution"]["high_conviction_target_rate"],
                        "avoid_target_rate": summary["label_distribution"]["avoid_target_rate"],
                    }
                ],
                ["positive_trade_rate", "high_conviction_target_rate", "avoid_target_rate"],
            ),
            "",
            "### ideal_weight_bucket",
            "",
            self._table(summary["label_distribution"]["ideal_weight_bucket"], ["value", "count", "rate"]),
            "",
            "### ideal_cash_reserve_bucket",
            "",
            self._table(summary["label_distribution"]["ideal_cash_reserve_bucket"], ["value", "count", "rate"]),
            "",
            "## Prediction Source",
            "",
            self._table(summary["prediction_source_distribution"], ["value", "count", "rate"]),
            "",
            "## Leakage Audit",
            "",
        ]
        for item in summary["leakage_audit"]:
            lines.append(f"- {item}")
        lines.extend(["", "## Quality Assessment", ""])
        for item in summary["quality_assessment"]:
            lines.append(f"- {item}")
        lines.append("")
        return "\n".join(lines)

    def leakage_audit(self) -> list[str]:
        forbidden = set(CLEAN_FORBIDDEN_FEATURE_COLUMNS + LABEL_COLUMNS)
        leaked = sorted(forbidden.intersection(CLEAN_FEATURE_COLUMNS))
        lines = [
            "Clean feature columns are restricted to J-Quants feature parquet and walk-forward ML prediction parquet values.",
            "Backtest execution state, purchase amounts, cash state, decisions, and realized PnL are not included in clean feature_columns.",
            "purchase_audit/trades remain available only as labels/audit/result matching columns.",
            "No current model inference is used to create historical predictions.",
        ]
        if leaked:
            lines.append(f"FAIL: forbidden columns found in clean feature list: {', '.join(leaked)}")
        else:
            lines.append("PASS: no forbidden audit/result/label columns are present in clean feature list.")
        return lines

    def _join_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        rows = []
        feature_columns = [
            "date",
            "code",
            *CLEAN_PRICE_FEATURE_COLUMNS,
            *CLEAN_TOPIX_FEATURE_COLUMNS,
            *CLEAN_FINANCIAL_FEATURE_COLUMNS,
            *CLEAN_EARNINGS_FEATURE_COLUMNS,
        ]
        for date, group in dataset.groupby("signal_date", dropna=False):
            if pd.isna(date):
                rows.append(group)
                continue
            date_text = pd.Timestamp(date).strftime("%Y-%m-%d")
            path = self.feature_root / f"features_{date_text}.parquet"
            if not path.exists():
                rows.append(group)
                continue
            features = pd.read_parquet(path)
            features["code"] = features["code"].astype(str)
            available = [column for column in feature_columns if column in features.columns]
            features = features[available].drop_duplicates(subset=["code"], keep="last")
            rows.append(group.merge(features.drop(columns=["date"], errors="ignore"), on="code", how="left"))
        return pd.concat(rows, ignore_index=True) if rows else dataset

    def _join_predictions(self, dataset: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for date, group in dataset.groupby("signal_date", dropna=False):
            group = group.copy()
            group["prediction_source"] = pd.NA
            group["prediction_joined"] = False
            if pd.isna(date):
                rows.append(group)
                continue
            date_text = pd.Timestamp(date).strftime("%Y-%m-%d")
            path = self.prediction_root / f"predictions_{date_text}.parquet"
            if not path.exists():
                rows.append(group)
                continue
            predictions = pd.read_parquet(path)
            predictions["code"] = predictions["code"].astype(str)
            if "risk_adjusted_score" not in predictions.columns:
                predictions["risk_adjusted_score"] = (
                    pd.to_numeric(predictions.get("expected_return_10d"), errors="coerce")
                    - 0.5 * pd.to_numeric(predictions.get("bad_entry_probability_10d"), errors="coerce")
                )
            cols = ["code", *[column for column in CLEAN_ML_FEATURE_COLUMNS if column in predictions.columns]]
            group = group.drop(columns=[column for column in CLEAN_ML_FEATURE_COLUMNS if column in group.columns], errors="ignore")
            group = group.merge(predictions[cols].drop_duplicates("code"), on="code", how="left")
            core = ["expected_return_10d", "bad_entry_probability_10d", "risk_adjusted_score"]
            joined = group[core].notna().all(axis=1)
            group.loc[joined, "prediction_source"] = "prediction_parquet"
            group["prediction_joined"] = joined
            rows.append(group)
        return pd.concat(rows, ignore_index=True) if rows else dataset

    def _add_candidate_relative_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data = dataset.copy()
        data["rank_in_day"] = pd.to_numeric(data.get("candidate_rank"), errors="coerce")
        data["score_rank_in_day"] = pd.to_numeric(data.get("score_rank"), errors="coerce")
        grouped = data.groupby("signal_date", dropna=False)
        data["risk_adjusted_score_percentile_in_day"] = grouped["risk_adjusted_score"].rank(pct=True)
        data["expected_return_percentile_in_day"] = grouped["expected_return_10d"].rank(pct=True)
        data["expected_max_return_percentile_in_day"] = grouped["expected_max_return_20d"].rank(pct=True)
        data["swing_success_percentile_in_day"] = grouped["swing_success_probability_20d"].rank(pct=True)
        data["bad_entry_percentile_in_day"] = grouped["bad_entry_probability_10d"].rank(pct=True)
        data["score_gap_to_best"] = grouped["risk_adjusted_score"].transform("max") - data["risk_adjusted_score"]
        data["expected_return_gap_to_best"] = grouped["expected_return_10d"].transform("max") - data["expected_return_10d"]
        data["expected_max_return_gap_to_best"] = grouped["expected_max_return_20d"].transform("max") - data["expected_max_return_20d"]
        data["swing_success_gap_to_best"] = grouped["swing_success_probability_20d"].transform("max") - data["swing_success_probability_20d"]
        data["bad_entry_gap_to_best"] = data["bad_entry_probability_10d"] - grouped["bad_entry_probability_10d"].transform("min")
        data["candidate_count_in_day"] = grouped["code"].transform("count")
        return data

    def _add_day_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data = dataset.copy()
        grouped = data.groupby("signal_date", dropna=False)
        data["day_avg_risk_adjusted_score"] = grouped["risk_adjusted_score"].transform("mean")
        data["day_max_risk_adjusted_score"] = grouped["risk_adjusted_score"].transform("max")
        data["day_avg_expected_return_10d"] = grouped["expected_return_10d"].transform("mean")
        data["day_avg_expected_max_return_20d"] = grouped["expected_max_return_20d"].transform("mean")
        data["day_avg_swing_success_probability_20d"] = grouped["swing_success_probability_20d"].transform("mean")
        data["day_avg_bad_entry_probability"] = grouped["bad_entry_probability_10d"].transform("mean")
        data["day_candidate_strength"] = data["day_avg_risk_adjusted_score"] + 0.5 * data["day_avg_expected_return_10d"]
        data["day_risk_level"] = data["day_avg_bad_entry_probability"]
        return data

    def _finalize_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data = dataset.copy()
        data["signal_date"] = pd.to_datetime(data["signal_date"], errors="coerce")
        data["code"] = data["code"].astype(str)
        for column in CLEAN_FEATURE_COLUMNS + LABEL_COLUMNS:
            if column not in data.columns:
                data[column] = pd.NA
        for column in AUDIT_COLUMNS:
            if column not in data.columns:
                data[column] = pd.NA
        ordered = ["signal_date", "code", *CLEAN_FEATURE_COLUMNS, *LABEL_COLUMNS, *AUDIT_COLUMNS]
        extra = [column for column in data.columns if column not in ordered]
        return data[ordered + extra]
