"""Phase 9-A PM AI re-architecture audit.

This audit is read-only. It documents how the current Portfolio Manager AI is
used in v2_82 and proposes a clean redesign as market regime + candidate
ranking + position sizing. It must not train models, run backtests, or overwrite
current model/profile artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase9a_pm_ai_rearchitecture_audit_2023-01_to_2026-05"
PERIOD = "2023-01-01_to_2026-05-31"
PROFILE = "rookie_dealer_02_v2_82_cap38"

V282_FINAL_RUN = Path("reports/final/v2_82_cap38/core_2023-01_to_2026-05")
V282_LOG_RUN = Path(f"logs/backtests/{PROFILE}/{PERIOD}")
CURRENT_PM_DIR = Path("models/ml/portfolio_manager/current_v2_73_phase3b_clean")
CURRENT_EXIT_DIR = Path("models/ml/exit/current_v2_66")
V282_PROFILE = Path(f"config/profiles/{PROFILE}.yaml")
CURRENT_PM_DATASET = Path("data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet")
WALK_FORWARD_PREDICTIONS = Path("data/ml/walk_forward_predictions")

CORE_SCORE_COLUMNS = [
    "expected_return_10d",
    "risk_adjusted_score",
    "bad_entry_probability_10d",
]
VOLATILITY_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "daily_range_ratio",
    "gap_up_ratio",
    "ma5_gap",
    "ma25_gap",
]
VOLUME_COLUMNS = [
    "volume",
    "turnover_value",
    "volume_ratio",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "turnover_ratio_5d",
    "turnover_ratio_20d",
]
FINANCIAL_COLUMNS = [
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
EARNINGS_COLUMNS = ["days_to_earnings", "is_near_earnings"]

ALLOWED_FEATURE_CANDIDATES = [
    "topix_return_5d",
    "topix_return_10d",
    "topix_return_20d",
    "topix_ma_distance",
    "topix_volatility_20d",
    "relative_return_5d",
    "relative_return_10d",
    "relative_return_20d",
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "risk_adjusted_score",
    "ml_score",
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
    "days_to_earnings",
    "is_near_earnings",
]

CONDITIONAL_RELATIVE_FEATURES = [
    "candidate_count_in_day",
    "rank_in_day",
    "score_rank_in_day",
    "percentile_in_day",
    "gap_to_best",
    "candidate_strength",
    "risk_adjusted_score_percentile_in_day",
    "expected_return_percentile_in_day",
    "bad_entry_percentile_in_day",
    "score_gap_to_best",
    "expected_return_gap_to_best",
    "day_candidate_strength",
]

FORBIDDEN_FEATURE_CANDIDATES = [
    "selected_count_in_day",
    "bought_count_in_day",
    "affordable_count_in_day",
    "cash_before",
    "cash_after",
    "available_cash",
    "daily_buy_limit_remaining_before",
    "daily_buy_limit_remaining_after",
    "max_positions_remaining_before",
    "position_state",
    "portfolio_state",
    "actual_profit",
    "realized_profit",
    "actual_net_profit",
    "profit",
    "net_profit",
    "profit_rate",
    "trade_result",
    "result",
    "backtest_outcome",
    "exit_reason",
    "exit_ai_triggered",
    "exit_ai_probability",
    "skip_reason",
    "reject_reason",
    "filled",
    "unfilled",
    "order_status",
    "final_assets",
    "audit_result",
    "decision",
    "actual_buy_amount",
    "actual_shares",
]

FORBIDDEN_TOKENS = [
    "selected",
    "bought",
    "affordable",
    "cash",
    "profit",
    "backtest",
    "result",
    "exit",
    "skip",
    "final_assets",
    "position_state",
    "portfolio_state",
    "filled",
    "actual_",
]


@dataclass(frozen=True)
class Phase9AReportPaths:
    markdown: Path
    json: Path


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
    return pd.to_numeric(series, errors="coerce")


def _safe_mean(series: pd.Series | None) -> float | None:
    values = _numeric(series).dropna()
    return None if values.empty else float(values.mean())


def _safe_sum(series: pd.Series | None) -> float:
    values = _numeric(series).dropna()
    return float(values.sum()) if not values.empty else 0.0


def _profit_factor(series: pd.Series | None) -> float | None:
    values = _numeric(series).dropna()
    if values.empty:
        return None
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(-values[values < 0].sum())
    if gross_loss == 0:
        return None if gross_profit == 0 else float("inf")
    return gross_profit / gross_loss


def _win_rate(series: pd.Series | None) -> float | None:
    values = _numeric(series).dropna()
    if values.empty:
        return None
    return float(values.gt(0).mean())


class Phase9APMAIRearchitectureAudit:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)

    def build_report(self) -> dict[str, Any]:
        purchase = self._load_purchase_audit()
        trades = self._load_trades()
        dataset = self._load_pm_dataset_subset()
        enriched_purchase = self._enrich_purchase_rows(purchase, dataset)
        role = self._current_pm_role_audit(enriched_purchase, trades)
        feature_classification = self._feature_classification()
        leakage = self._leakage_checklist(feature_classification)
        architecture = self._architecture_design()
        labels = self._label_design()
        phase8 = self._phase8_failure_summary()
        phase9b = self._phase9b_dataset_design()
        verdict = self._verdict(leakage)
        return {
            "metadata": {
                "phase": "9-A",
                "profile": PROFILE,
                "period": PERIOD,
                "audit_only": True,
                "training_executed": False,
                "backtest_executed": False,
                "long_running_retraining_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
                "uses_walk_forward_predictions_for_historical_backtest": True,
                "current_model_regenerated_historical_predictions": False,
            },
            "sources": self._sources(),
            "current_pm_ai_role": role,
            "phase8_failure_summary": phase8,
            "feature_classification": feature_classification,
            "three_layer_architecture": architecture,
            "label_design": labels,
            "leakage_risk_checklist": leakage,
            "phase9b_dataset_design": phase9b,
            "verdict": verdict,
        }

    def save_report(self, report: dict[str, Any]) -> Phase9AReportPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9AReportPaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        role = report["current_pm_ai_role"]
        features = report["feature_classification"]
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 9-A Re-Architecture Audit",
                "",
                "## Scope",
                "",
                "- audit and data design only",
                "- no retraining, no backtest, no API refetch, no live order",
                "- current PM AI, current Exit AI, and v2_82 profile are not overwritten",
                "",
                "## Current PM AI Role",
                "",
                self._table([role["summary"]], ["profile", "purchase_rows", "buy_rows_with_pm", "trade_rows", "pm_multiplier_distribution", "interpretation"]),
                "",
                "### PM Multiplier Feature Audit",
                "",
                self._table(
                    role["pm_multiplier_feature_summary"],
                    [
                        "pm_multiplier",
                        "trade_count",
                        "buy_decision_count",
                        "avg_expected_return_10d",
                        "avg_risk_adjusted_score",
                        "avg_bad_entry_probability_10d",
                        "avg_candidate_count_in_day",
                        "avg_rank_in_day",
                        "avg_percentile_in_day",
                        "top_market_regime",
                    ],
                ),
                "",
                "### PM Multiplier Outcome Reference",
                "",
                "This section is separated from feature design and must not be used as PM AI training features.",
                "",
                self._table(role["outcome_reference_by_multiplier"], ["pm_multiplier", "trade_count", "net_profit", "profit_factor", "win_rate"]),
                "",
                "## Phase 8 Failure Summary",
                "",
                self._table(report["phase8_failure_summary"], ["attempt", "result", "failure_reason", "decision"]),
                "",
                "## Feature Classification",
                "",
                self._table([features["allowed"]], ["category", "feature_count", "examples", "rule"]),
                "",
                self._table([features["conditional"]], ["category", "feature_count", "features", "condition"]),
                "",
                self._table([features["forbidden"]], ["category", "feature_count", "features", "rule"]),
                "",
                "## Three-Layer Architecture",
                "",
                self._table(report["three_layer_architecture"], ["layer", "purpose", "input_candidates", "outputs", "training_constraint"]),
                "",
                "## Label Design",
                "",
                self._table(report["label_design"], ["label", "definition", "source", "allowed_as_feature", "target_layer", "leakage_risk"]),
                "",
                "## Leakage Risk Checklist",
                "",
                self._table([report["leakage_risk_checklist"]], ["forbidden_feature_candidate_count", "forbidden_tokens_in_allowed_features", "conditional_relative_features_classified", "current_artifacts_overwritten", "overall_leakage_risk"]),
                "",
                "## Phase 9-B Dataset Design",
                "",
                self._table(report["phase9b_dataset_design"], ["component", "design", "must_include", "must_exclude"]),
                "",
                "## Verdict",
                "",
                self._table([report["verdict"]], ["keep_current_v282", "replace_current_pm_now", "phase9b_ready", "reason"]),
                "",
            ]
        )

    def _sources(self) -> dict[str, Any]:
        run = self._v282_run_dir()
        prediction_files = list((self.root / WALK_FORWARD_PREDICTIONS).glob("predictions_*.parquet"))
        return {
            "v2_82_run_dir": str(run),
            "trades": str(run / "trades.csv"),
            "purchase_audit": str(run / "purchase_audit.csv"),
            "current_pm_dataset_reference": str(self.root / CURRENT_PM_DATASET),
            "walk_forward_prediction_dir": str(self.root / WALK_FORWARD_PREDICTIONS),
            "walk_forward_prediction_file_count": len(prediction_files),
            "current_pm_model": str(self.root / CURRENT_PM_DIR),
            "current_exit_model": str(self.root / CURRENT_EXIT_DIR),
            "v2_82_profile": str(self.root / V282_PROFILE),
        }

    def _v282_run_dir(self) -> Path:
        final_run = self.root / V282_FINAL_RUN
        if (final_run / "purchase_audit.csv").exists():
            return final_run
        return self.root / V282_LOG_RUN

    def _load_purchase_audit(self) -> pd.DataFrame:
        frame = _read_csv(self._v282_run_dir() / "purchase_audit.csv")
        return self._normalize(frame)

    def _load_trades(self) -> pd.DataFrame:
        frame = _read_csv(self._v282_run_dir() / "trades.csv")
        return self._normalize(frame)

    def _normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        for column in ["signal_date", "entry_date", "exit_date", "trade_date"]:
            if column in out.columns:
                out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
        if "code" in out.columns:
            out["code"] = out["code"].astype(str)
        return out

    def _load_pm_dataset_subset(self) -> pd.DataFrame:
        path = self.root / CURRENT_PM_DATASET
        if not path.exists():
            return pd.DataFrame()
        wanted = [
            "signal_date",
            "code",
            *VOLATILITY_COLUMNS,
            *VOLUME_COLUMNS,
            *FINANCIAL_COLUMNS,
            *EARNINGS_COLUMNS,
            "rank_in_day",
            "score_rank_in_day",
            "risk_adjusted_score_percentile_in_day",
            "expected_return_percentile_in_day",
            "bad_entry_percentile_in_day",
            "score_gap_to_best",
            "expected_return_gap_to_best",
            "bad_entry_gap_to_best",
            "candidate_count_in_day",
            "day_candidate_strength",
            "day_risk_level",
        ]
        try:
            frame = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
        frame = frame[[column for column in wanted if column in frame.columns]]
        return self._normalize(frame)

    def _enrich_purchase_rows(self, purchase: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
        if purchase.empty:
            return purchase
        out = purchase.copy()
        date_col = "signal_date" if "signal_date" in out.columns else "entry_date"
        grouped = out.groupby(date_col, dropna=False)
        out["candidate_count_in_day_audit"] = grouped["code"].transform("count") if "code" in out.columns else None
        if "candidate_rank" in out.columns:
            out["rank_in_day_audit"] = _numeric(out["candidate_rank"])
        elif "score_rank" in out.columns:
            out["rank_in_day_audit"] = _numeric(out["score_rank"])
        else:
            out["rank_in_day_audit"] = None
        count = _numeric(out.get("candidate_count_in_day_audit"))
        rank = _numeric(out.get("rank_in_day_audit"))
        out["percentile_in_day_audit"] = 1.0 - ((rank - 1.0) / (count - 1.0))
        out.loc[count.le(1), "percentile_in_day_audit"] = 1.0
        if dataset.empty or "signal_date" not in out.columns:
            return out
        join_columns = [column for column in dataset.columns if column not in out.columns or column in {"signal_date", "code"}]
        dataset = dataset[join_columns].drop_duplicates(["signal_date", "code"])
        return out.merge(dataset, on=["signal_date", "code"], how="left", suffixes=("", "_dataset"))

    def _current_pm_role_audit(self, purchase: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
        buy_rows = self._buy_pm_rows(purchase)
        trade_rows = trades.copy()
        if not trade_rows.empty and "action" in trade_rows.columns:
            trade_rows = trade_rows[trade_rows["action"].fillna("").astype(str).str.upper().eq("SELL")]
        distribution = self._value_counts(_numeric(buy_rows.get("pm_multiplier")).round(2))
        multipliers = sorted(set(distribution.keys()) | {"0.6", "0.8", "1.0", "1.15", "1.3"}, key=float, reverse=True)
        trade_counts = self._value_counts(_numeric(trade_rows.get("pm_multiplier")).round(2))
        return {
            "summary": {
                "profile": PROFILE,
                "purchase_rows": int(len(purchase)),
                "buy_rows_with_pm": int(len(buy_rows)),
                "trade_rows": int(len(trade_rows)),
                "pm_multiplier_distribution": distribution,
                "interpretation": "current PM AI is a sizing and ordering signal; Phase 8 showed count calibration alone does not recover PM1.30 quality.",
            },
            "pm_multiplier_feature_summary": [self._feature_summary_for_multiplier(buy_rows, value, trade_counts) for value in multipliers],
            "outcome_reference_by_multiplier": [self._outcome_summary_for_multiplier(trade_rows, value) for value in multipliers],
            "feature_groups_used_for_audit_only": {
                "core_scores": CORE_SCORE_COLUMNS,
                "volatility": VOLATILITY_COLUMNS,
                "volume": VOLUME_COLUMNS,
                "financial": FINANCIAL_COLUMNS,
                "earnings": EARNINGS_COLUMNS,
                "relative": ["candidate_count_in_day_audit", "rank_in_day_audit", "percentile_in_day_audit"],
            },
        }

    def _buy_pm_rows(self, purchase: pd.DataFrame) -> pd.DataFrame:
        if purchase.empty or "pm_multiplier" not in purchase.columns:
            return pd.DataFrame()
        out = purchase[_numeric(purchase.get("pm_multiplier")).notna()].copy()
        if "decision" in out.columns:
            out = out[out["decision"].fillna("").astype(str).str.upper().eq("BUY")]
        return out

    def _feature_summary_for_multiplier(self, rows: pd.DataFrame, multiplier: str, trade_counts: dict[str, int]) -> dict[str, Any]:
        if rows.empty:
            group = rows
        else:
            group = rows[_numeric(rows.get("pm_multiplier")).round(2).eq(float(multiplier))]
        result: dict[str, Any] = {
            "pm_multiplier": multiplier,
            "buy_decision_count": int(len(group)),
            "trade_count": int(trade_counts.get(multiplier, 0)),
            "avg_expected_return_10d": _safe_mean(group.get("expected_return_10d")),
            "avg_risk_adjusted_score": _safe_mean(group.get("risk_adjusted_score")),
            "avg_bad_entry_probability_10d": _safe_mean(group.get("bad_entry_probability_10d")),
            "avg_candidate_count_in_day": _safe_mean(group.get("candidate_count_in_day_audit", group.get("candidate_count_in_day"))),
            "avg_rank_in_day": _safe_mean(group.get("rank_in_day_audit", group.get("rank_in_day"))),
            "avg_percentile_in_day": _safe_mean(group.get("percentile_in_day_audit")),
            "top_market_regime": self._top_value(group.get("market_regime", group.get("day_risk_level"))),
            "volatility_feature_means": self._column_means(group, VOLATILITY_COLUMNS),
            "volume_feature_means": self._column_means(group, VOLUME_COLUMNS),
            "financial_feature_means": self._column_means(group, FINANCIAL_COLUMNS),
            "earnings_feature_means": self._column_means(group, EARNINGS_COLUMNS),
        }
        return result

    def _outcome_summary_for_multiplier(self, trades: pd.DataFrame, multiplier: str) -> dict[str, Any]:
        if trades.empty or "pm_multiplier" not in trades.columns:
            group = pd.DataFrame()
        else:
            group = trades[_numeric(trades.get("pm_multiplier")).round(2).eq(float(multiplier))]
        profit = group.get("net_profit", group.get("profit"))
        return {
            "pm_multiplier": multiplier,
            "trade_count": int(len(group)),
            "net_profit": _safe_sum(profit),
            "profit_factor": _profit_factor(profit),
            "win_rate": _win_rate(profit),
            "learning_feature_use": "forbidden_reference_only",
        }

    def _feature_classification(self) -> dict[str, Any]:
        return {
            "allowed": {
                "category": "A_allowed_features",
                "feature_count": len(ALLOWED_FEATURE_CANDIDATES),
                "features": ALLOWED_FEATURE_CANDIDATES,
                "examples": ",".join(ALLOWED_FEATURE_CANDIDATES[:12]),
                "rule": "J-Quants/API-derived market, price, volume, financial, earnings, and prediction-time Stock Selection scores only.",
            },
            "conditional": {
                "category": "B_conditionally_allowed_relative_features",
                "feature_count": len(CONDITIONAL_RELATIVE_FEATURES),
                "features": CONDITIONAL_RELATIVE_FEATURES,
                "condition": "Allowed only after the same-day Stock Selection candidate pool is finalized and before cash, portfolio, backtest, selected/bought/affordable, or exit decisions.",
            },
            "forbidden": {
                "category": "C_forbidden_features",
                "feature_count": len(FORBIDDEN_FEATURE_CANDIDATES),
                "features": FORBIDDEN_FEATURE_CANDIDATES,
                "rule": "Never use backtest/trade/cash/portfolio/position/exit/skip/fill/profit/result/audit columns as PM AI training features.",
            },
        }

    def _architecture_design(self) -> list[dict[str, Any]]:
        return [
            {
                "layer": "Layer 1 Market Regime Model",
                "purpose": "classify the prediction date as attack, neutral, or defensive",
                "input_candidates": "TOPIX returns, TOPIX MA distance, TOPIX volatility, market/sector breadth if API-derived, market volume trend",
                "outputs": "market_attack_score, market_regime_class",
                "training_constraint": "no profit-defined attack days; labels must come from market/index behavior, not backtest PnL",
            },
            {
                "layer": "Layer 2 Candidate Ranking Model",
                "purpose": "rank same-day Stock Selection candidates before capital decisions",
                "input_candidates": "Stock Selection scores, expected_return_10d, risk_adjusted_score, bad_entry_probability_10d, financial, liquidity, volatility, earnings distance, conditional relative features",
                "outputs": "candidate_rank_score, same_day_rank, same_day_percentile",
                "training_constraint": "candidate set relative features are allowed only if computed before cash/portfolio/backtest decisions",
            },
            {
                "layer": "Layer 3 Position Sizing Model",
                "purpose": "map market state and candidate quality into PM multipliers",
                "input_candidates": "market_attack_score, market_regime_class, candidate_rank_score, same_day_percentile, liquidity risk, volatility risk, financial quality, event risk",
                "outputs": "PM1.30, PM1.15, PM1.00, PM0.80, PM0.60",
                "training_constraint": "no realized trade result, cash, position state, exit reason, or affordability feature",
            },
        ]

    def _label_design(self) -> list[dict[str, Any]]:
        labels = [
            ("future_5d_return", "close/open forward return over 5 business days", "future J-Quants price", False, "Layer 2/3", "low_label_only"),
            ("future_10d_return", "close/open forward return over 10 business days", "future J-Quants price", False, "Layer 2/3", "low_label_only"),
            ("max_adverse_excursion_10d", "worst forward drawdown within 10 business days", "future J-Quants price", False, "Layer 3 risk sizing", "low_label_only"),
            ("max_favorable_excursion_10d", "best forward upside within 10 business days", "future J-Quants price", False, "Layer 2 ranking", "low_label_only"),
            ("risk_adjusted_future_return", "future return penalized by adverse excursion or volatility", "future J-Quants price", False, "Layer 2/3", "low_label_only"),
            ("downside_penalized_return", "future return minus downside penalty", "future J-Quants price", False, "Layer 3", "low_label_only"),
            ("top_decile_within_day", "same-day candidate top decile by future utility", "future price label plus prediction-time candidate pool", False, "Layer 2", "low_if_label_only"),
            ("avoid_bottom_decile_within_day", "same-day candidate bottom decile by future utility", "future price label plus prediction-time candidate pool", False, "Layer 2/3", "low_if_label_only"),
            ("relative_future_utility_rank_in_day", "same-day rank of future utility among candidates", "future price label plus prediction-time candidate pool", False, "Layer 2", "low_if_label_only"),
        ]
        return [
            {
                "label": label,
                "definition": definition,
                "source": source,
                "allowed_as_feature": allowed,
                "target_layer": layer,
                "leakage_risk": risk,
            }
            for label, definition, source, allowed, layer, risk in labels
        ]

    def _phase8_failure_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "attempt": "v2_90 PM AI v2 raw",
                "result": "PM1.30 disappeared and capital utilization collapsed",
                "failure_reason": "clean single-name classifier was too conservative as direct replacement",
                "decision": "rejected",
            },
            {
                "attempt": "v2_91 PM AI v2 calibrated",
                "result": "PM1.30 count recovered but PM1.30 quality did not",
                "failure_reason": "calibration fixed distribution but not sizing alpha",
                "decision": "rejected",
            },
            {
                "attempt": "v2_92 rule-based relative allocator",
                "result": "operationally valid but far below v2_82",
                "failure_reason": "simple Stock Selection relative rank is insufficient",
                "decision": "rejected",
            },
        ]

    def _leakage_checklist(self, feature_classification: dict[str, Any]) -> dict[str, Any]:
        allowed = feature_classification["allowed"]["features"]
        conditional = feature_classification["conditional"]["features"]
        forbidden_hits = sorted({feature for feature in allowed if self._contains_forbidden_token(feature)})
        relative_ok = all(feature in conditional for feature in ["candidate_count_in_day", "rank_in_day", "percentile_in_day", "gap_to_best", "candidate_strength"])
        return {
            "jquants_api_derived_only_for_features": True,
            "stock_selection_scores_prediction_time_only": True,
            "future_price_labels_allowed_as_labels_only": True,
            "forbidden_feature_candidate_count": len(forbidden_hits),
            "forbidden_tokens_in_allowed_features": forbidden_hits,
            "conditional_relative_features_classified": relative_ok,
            "selected_bought_affordable_forbidden": True,
            "cash_portfolio_position_forbidden": True,
            "profit_result_backtest_exit_skip_forbidden": True,
            "historical_backtest_uses_walk_forward_predictions": True,
            "current_artifacts_overwritten": False,
            "overall_leakage_risk": "low_for_phase9a_design",
        }

    def _phase9b_dataset_design(self) -> list[dict[str, Any]]:
        return [
            {
                "component": "dataset row grain",
                "design": "one row per Stock Selection candidate per prediction date",
                "must_include": "date, code, API-derived features, Stock Selection prediction scores, conditional same-day relative features",
                "must_exclude": "whether candidate was selected, bought, affordable, filled, exited, or profitable",
            },
            {
                "component": "market regime table",
                "design": "daily TOPIX and market breadth feature table joined by prediction date",
                "must_include": "TOPIX returns, MA distance, volatility, breadth/volume if available from API/cache",
                "must_exclude": "strategy PnL by date or profitable-day labels",
            },
            {
                "component": "ranking labels",
                "design": "future utility labels computed from J-Quants future prices for all candidates",
                "must_include": "future returns, MFE/MAE, downside-penalized utility, same-day future utility rank",
                "must_exclude": "actual trade result, exit reason, realized trade PnL",
            },
            {
                "component": "sizing labels",
                "design": "map future utility, downside, and market regime into label buckets for multiplier learning",
                "must_include": "utility quantiles and avoid labels derived only from future prices",
                "must_exclude": "current PM multiplier imitation as the only objective",
            },
            {
                "component": "validation",
                "design": "time split and later walk-forward backtest profile without overwriting current artifacts",
                "must_include": "leakage audit, feature list audit, distribution checks, PM multiplier quality by bucket",
                "must_exclude": "current model historical prediction regeneration",
            },
        ]

    def _verdict(self, leakage: dict[str, Any]) -> dict[str, Any]:
        clean = leakage["forbidden_feature_candidate_count"] == 0 and leakage["conditional_relative_features_classified"]
        return {
            "keep_current_v282": True,
            "replace_current_pm_now": False,
            "phase9b_ready": bool(clean),
            "reason": "v2_82 remains the production candidate; Phase 9-A only defines a clean PM AI v3 dataset/architecture path.",
        }

    def _column_means(self, frame: pd.DataFrame, columns: list[str]) -> dict[str, float | None]:
        return {column: _safe_mean(frame.get(column)) for column in columns if column in frame.columns}

    def _top_value(self, series: pd.Series | None) -> str | None:
        if series is None:
            return None
        values = series.dropna().astype(str)
        values = values[values.ne("")]
        if values.empty:
            return None
        return str(values.value_counts().idxmax())

    def _value_counts(self, series: pd.Series) -> dict[str, int]:
        values = series.dropna()
        counts = values.value_counts().sort_index()
        return {str(key): int(value) for key, value in counts.items()}

    def _contains_forbidden_token(self, feature: str) -> bool:
        lowered = feature.lower()
        return any(token in lowered for token in FORBIDDEN_TOKENS)

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(self._format(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if value == float("inf"):
                return "inf"
            return f"{value:.6g}"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value)
        if isinstance(value, dict):
            return ", ".join(f"{key}:{val}" for key, val in value.items())
        return str(value).replace("\n", " ")


def build_phase9a_report(root: Path | str = ROOT) -> dict[str, Any]:
    return Phase9APMAIRearchitectureAudit(root).build_report()
