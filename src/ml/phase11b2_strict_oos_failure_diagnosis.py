"""Phase 11-B2 strict OOS failure diagnosis.

This audit reads the Phase 11-A dataset and the Phase 11-I research-only model
to diagnose why strict OOS rank lift did not translate into strategy results.
It does not train a production model, run a full backtest, change profiles, or
overwrite any existing model artifacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11b_valuation_engine_prototype import CLASSIFICATION_TARGET, DATASET_PATH, Phase11BValuationEnginePrototype
from ml.phase11e_exit_dd_guard import FUTURE_EVAL_COLUMNS, _numeric, _safe_float
from ml.phase11i_strict_oos import MODEL_DIR as PHASE11I_MODEL_DIR
from ml.phase11i_strict_oos import TEST_END, TEST_START, TRAIN_END, TRAIN_START


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase11b2_strict_oos_failure_diagnosis_2025"
ROUND_LOT = 100
DOWNSIDE_BAD_THRESHOLD = -0.10
FAILURE_RETURN_THRESHOLD = -0.10
FAILURE_DRAWDOWN_THRESHOLD = -0.15

DRIFT_FEATURES = [
    "risk_adjusted_score",
    "expected_return",
    "stock_selection_rank_score",
    "candidate_strength",
    "relative_return_20d",
    "ma25_gap",
    "ma75_gap",
    "daily_range_ratio",
    "turnover_value",
    "EPS",
    "Sales_growth",
    "topix_return_20d",
]

FILTER_FEATURES = [
    "daily_range_ratio",
    "ma75_gap",
    "relative_return_20d",
    "turnover_value",
]


@dataclass(frozen=True)
class Phase11B2Options:
    initial_cash: float = 1_000_000.0
    daily_buy_budget: float = 900_000.0
    max_positions: int = 5
    round_lot: int = ROUND_LOT
    holding_days: int = 20
    stop_loss_rate: float = -0.08
    opportunity_drop_threshold: float = 0.15
    opportunity_rank_floor: float = 0.50
    cost_rate: float = 0.002


@dataclass(frozen=True)
class Phase11B2Paths:
    markdown: Path
    json: Path


class Phase11B2StrictOOSFailureDiagnosis:
    def __init__(self, root: Path | str = ROOT, *, options: Phase11B2Options | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase11B2Options()

    def run(self) -> Phase11B2Paths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        dataset = self.load_dataset()
        feature_columns = self.load_feature_columns()
        leakage = self.leakage_checklist(feature_columns)
        if leakage["blocking_issues"]:
            return {
                "metadata": self.metadata(),
                "dataset_summary": self.dataset_summary(dataset, pd.DataFrame(), pd.DataFrame()),
                "feature_policy": {"feature_columns": feature_columns, "feature_count": len(feature_columns)},
                "leakage_checklist": leakage,
                "diagnosis_summary": self.diagnosis_summary({}, [], [], [], leakage),
            }

        train, test = self.split_dataset(dataset)
        scored = self.score_2025(test, feature_columns)
        baseline_top5 = self.daily_top(scored, "baseline_rank_score", 5)
        valuation_top5 = self.daily_top(scored, "opportunity_top_decile_proba", 5)
        trades = self.simulate_e4(scored)
        prediction_deciles = self.prediction_quality_by_decile(scored)
        top_candidate_diagnostics = [
            self.top_candidate_summary("baseline_top5", baseline_top5),
            self.top_candidate_summary("strict_oos_valuation_top5", valuation_top5),
        ]
        exit_diagnostics = self.exit_trigger_diagnostics(trades)
        drift = self.feature_drift_audit(train, test)
        failure_samples = self.candidate_failure_samples(valuation_top5)
        filter_audit = self.downside_filter_audit(scored, valuation_top5)
        summary = self.diagnosis_summary(top_candidate_diagnostics[1], exit_diagnostics, drift, filter_audit, leakage)
        return {
            "metadata": self.metadata(),
            "dataset_summary": self.dataset_summary(dataset, train, test),
            "feature_policy": {"feature_columns": feature_columns, "feature_count": len(feature_columns)},
            "prediction_quality_by_decile": prediction_deciles,
            "top_candidate_diagnostics": top_candidate_diagnostics,
            "exit_trigger_diagnostics": exit_diagnostics,
            "feature_drift_audit": drift,
            "candidate_failure_samples": failure_samples,
            "downside_filter_audit": filter_audit,
            "diagnosis_summary": summary,
            "leakage_checklist": leakage,
        }

    def load_dataset(self) -> pd.DataFrame:
        data = pd.read_parquet(self.root / DATASET_PATH)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        return data.dropna(subset=["date", "code", CLASSIFICATION_TARGET]).reset_index(drop=True)

    def load_feature_columns(self) -> list[str]:
        path = self.root / PHASE11I_MODEL_DIR / "feature_columns.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def load_classifier(self) -> Any:
        import joblib

        path = self.root / PHASE11I_MODEL_DIR / "opportunity_top_decile_20d_classifier.joblib"
        return joblib.load(path)

    def split_dataset(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = dataset[(dataset["date"] >= TRAIN_START) & (dataset["date"] <= TRAIN_END)].copy()
        test = dataset[(dataset["date"] >= TEST_START) & (dataset["date"] <= TEST_END)].copy()
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def score_2025(self, test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        data = test.copy()
        for column in feature_columns:
            if pd.api.types.is_bool_dtype(data[column].dtype):
                data[column] = data[column].astype(int)
            else:
                data[column] = _numeric(data[column])
        data[feature_columns] = data[feature_columns].replace([float("inf"), float("-inf")], pd.NA)
        classifier = self.load_classifier()
        data["opportunity_top_decile_proba"] = classifier.predict_proba(data[feature_columns])[:, 1]
        data["opportunity_score_proba_rank"] = data.groupby("date")["opportunity_top_decile_proba"].rank(method="average", pct=True)
        baseline = data["stock_selection_rank_score"] if "stock_selection_rank_score" in data.columns else data["risk_adjusted_score"]
        data["baseline_rank_score"] = _numeric(baseline).fillna(-10**18)
        return data.dropna(subset=["date", "code", "close"]).sort_values(["date", "code"]).reset_index(drop=True)

    def prediction_quality_by_decile(self, scored: pd.DataFrame) -> list[dict[str, Any]]:
        data = scored.copy()
        data["prediction_decile"] = pd.qcut(data["opportunity_top_decile_proba"].rank(method="first"), 10, labels=False, duplicates="drop") + 1
        rows = []
        for decile, group in data.groupby("prediction_decile", sort=True):
            rows.append({"prediction_decile": int(decile), "count": int(len(group)), **self.quality_metrics(group)})
        return rows

    def daily_top(self, scored: pd.DataFrame, rank_column: str, n: int) -> pd.DataFrame:
        return (
            scored.sort_values(["date", rank_column, "turnover_value", "code"], ascending=[True, False, False, True])
            .groupby("date", group_keys=False)
            .head(n)
            .reset_index(drop=True)
        )

    def top_candidate_summary(self, label: str, frame: pd.DataFrame) -> dict[str, Any]:
        lot_cost = _numeric(frame["close"]) * self.options.round_lot if "close" in frame.columns else pd.Series(dtype=float)
        return {
            "candidate_set": label,
            "rows": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if "date" in frame.columns else 0,
            "avg_close": _safe_float(_numeric(frame["close"]).mean()) if "close" in frame.columns else None,
            "avg_lot_cost": _safe_float(lot_cost.mean()),
            "avg_turnover_value": _safe_float(_numeric(frame["turnover_value"]).mean()) if "turnover_value" in frame.columns else None,
            "avg_opportunity_top_decile_proba": _safe_float(_numeric(frame["opportunity_top_decile_proba"]).mean()) if "opportunity_top_decile_proba" in frame.columns else None,
            "avg_opportunity_score_proba_rank": _safe_float(_numeric(frame["opportunity_score_proba_rank"]).mean()) if "opportunity_score_proba_rank" in frame.columns else None,
            **self.quality_metrics(frame),
        }

    def quality_metrics(self, frame: pd.DataFrame) -> dict[str, Any]:
        drawdown = _numeric(frame["future_max_drawdown_20d"]) if "future_max_drawdown_20d" in frame.columns else pd.Series(dtype=float)
        return {
            "future_return_20d_mean": self.mean(frame, "future_return_20d"),
            "future_max_return_20d_mean": self.mean(frame, "future_max_return_20d"),
            "future_max_drawdown_20d_mean": self.mean(frame, "future_max_drawdown_20d"),
            "opportunity_value_20d_mean": self.mean(frame, "opportunity_value_20d"),
            "opportunity_top_decile_20d_rate": self.mean(frame, "opportunity_top_decile_20d"),
            "downside_bad_rate": _safe_float((drawdown <= DOWNSIDE_BAD_THRESHOLD).mean()) if not drawdown.empty else None,
        }

    def mean(self, frame: pd.DataFrame, column: str) -> float | None:
        values = _numeric(frame[column]) if column in frame.columns else pd.Series(dtype=float)
        return _safe_float(values.mean()) if not values.empty else None

    def simulate_e4(self, scored: pd.DataFrame) -> pd.DataFrame:
        cash = self.options.initial_cash
        positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        dates = list(pd.Series(scored["date"].dropna().unique()).sort_values())
        by_date = {date: group.set_index("code", drop=False) for date, group in scored.groupby("date", sort=True)}

        for current_date in dates:
            current = by_date[current_date]
            still_open = []
            for position in positions:
                current_row = current.loc[position["code"]] if position["code"] in current.index else None
                reason = self.exit_reason(position, current_date, current_row)
                if reason:
                    exit_close = float(current_row["close"]) if current_row is not None else float(position["last_close"])
                    trade = self.close_position(position, current_date, exit_close, reason, current_row)
                    cash += trade["exit_cash_flow"]
                    trades.append(trade)
                else:
                    if current_row is not None:
                        position["last_close"] = float(current_row["close"])
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            ranked = current.reset_index(drop=True).sort_values(["opportunity_top_decile_proba", "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions * 4)
            raw_amount = min(cash, self.options.daily_buy_budget) / max(1, min(self.options.max_positions, len(ranked)))
            bought = 0
            for _, row in ranked.iterrows():
                if bought >= slots:
                    break
                lot_cost = float(row["close"]) * self.options.round_lot
                lots = int(raw_amount // (lot_cost * (1.0 + self.options.cost_rate))) if lot_cost > 0 else 0
                buy_amount = lots * lot_cost
                buy_cost = buy_amount * self.options.cost_rate
                if lots <= 0 or buy_amount + buy_cost > cash:
                    continue
                cash -= buy_amount + buy_cost
                bought += 1
                positions.append(
                    {
                        "entry_date": current_date,
                        "due_date": current_date + pd.offsets.BDay(self.options.holding_days),
                        "code": str(row["code"]),
                        "buy_amount": buy_amount,
                        "buy_cost": buy_cost,
                        "lot_count": lots,
                        "entry_close": float(row["close"]),
                        "last_close": float(row["close"]),
                        "entry_proba": _safe_float(row.get("opportunity_top_decile_proba")),
                        "entry_rank": _safe_float(row.get("opportunity_score_proba_rank")),
                        **{column: _safe_float(row.get(column)) for column in FUTURE_EVAL_COLUMNS},
                    }
                )
        if dates:
            last_date = dates[-1]
            for position in positions:
                trades.append(self.close_position(position, last_date, float(position["last_close"]), "forced_end_of_period", None))
        return pd.DataFrame(trades)

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None) -> str | None:
        if current_row is not None:
            observed_return = float(current_row["close"]) / float(position["entry_close"]) - 1.0
            if observed_return <= self.options.stop_loss_rate:
                return "stop_loss"
            current_proba = _safe_float(current_row.get("opportunity_top_decile_proba"))
            current_rank = _safe_float(current_row.get("opportunity_score_proba_rank"))
            entry_proba = _safe_float(position.get("entry_proba"))
            if current_rank is not None and current_rank < self.options.opportunity_rank_floor:
                return "opportunity_rank_below_floor"
            if current_proba is not None and entry_proba is not None and current_proba <= entry_proba - self.options.opportunity_drop_threshold:
                return "opportunity_proba_drop"
        if current_date >= position["due_date"]:
            return "time_exit_20d"
        return None

    def close_position(self, position: dict[str, Any], exit_date: pd.Timestamp, exit_close: float, reason: str, current_row: pd.Series | None) -> dict[str, Any]:
        exit_amount = float(position["lot_count"]) * self.options.round_lot * exit_close
        sell_cost = exit_amount * self.options.cost_rate
        exit_cash_flow = exit_amount - sell_cost
        profit = exit_cash_flow - float(position["buy_amount"]) - float(position["buy_cost"])
        exit_proba = _safe_float(current_row.get("opportunity_top_decile_proba")) if current_row is not None else None
        exit_rank = _safe_float(current_row.get("opportunity_score_proba_rank")) if current_row is not None else None
        entry_proba = _safe_float(position.get("entry_proba"))
        return {
            "entry_date": position["entry_date"],
            "exit_date": exit_date,
            "code": position["code"],
            "exit_reason": reason,
            "holding_days": len(pd.bdate_range(position["entry_date"], exit_date)) - 1,
            "realized_return": profit / float(position["buy_amount"]) if position["buy_amount"] else None,
            "realized_profit": profit,
            "exit_cash_flow": exit_cash_flow,
            "entry_proba": entry_proba,
            "exit_proba": exit_proba,
            "proba_drop": entry_proba - exit_proba if entry_proba is not None and exit_proba is not None else None,
            "entry_rank": _safe_float(position.get("entry_rank")),
            "exit_rank": exit_rank,
            "cost_paid": float(position["buy_cost"]) + sell_cost,
            **{column: position.get(column) for column in FUTURE_EVAL_COLUMNS},
        }

    def exit_trigger_diagnostics(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty:
            return []
        rows = []
        for reason, group in trades.groupby("exit_reason", sort=True):
            rows.append(
                {
                    "exit_reason": str(reason),
                    "count": int(len(group)),
                    "avg_holding_days": self.mean(group, "holding_days"),
                    "avg_realized_return": self.mean(group, "realized_return"),
                    "avg_entry_proba": self.mean(group, "entry_proba"),
                    "avg_exit_proba": self.mean(group, "exit_proba"),
                    "avg_proba_drop": self.mean(group, "proba_drop"),
                    "avg_entry_rank": self.mean(group, "entry_rank"),
                    "avg_exit_rank": self.mean(group, "exit_rank"),
                }
            )
        return rows

    def feature_drift_audit(self, train: pd.DataFrame, test: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for column in DRIFT_FEATURES:
            if column not in train.columns or column not in test.columns:
                continue
            left = _numeric(train[column]).dropna()
            right = _numeric(test[column]).dropna()
            if left.empty or right.empty:
                continue
            pooled = math.sqrt((float(left.var()) + float(right.var())) / 2.0) if len(left) > 1 and len(right) > 1 else 0.0
            smd = (float(right.mean()) - float(left.mean())) / pooled if pooled else None
            rows.append(
                {
                    "feature": column,
                    "train_mean": _safe_float(left.mean()),
                    "test_mean": _safe_float(right.mean()),
                    "train_p10": _safe_float(left.quantile(0.10)),
                    "test_p10": _safe_float(right.quantile(0.10)),
                    "train_p90": _safe_float(left.quantile(0.90)),
                    "test_p90": _safe_float(right.quantile(0.90)),
                    "standardized_mean_diff": _safe_float(smd),
                }
            )
        return rows

    def candidate_failure_samples(self, valuation_top5: pd.DataFrame) -> list[dict[str, Any]]:
        data = valuation_top5.copy()
        mask = (_numeric(data["future_return_20d"]) <= FAILURE_RETURN_THRESHOLD) | (_numeric(data["future_max_drawdown_20d"]) <= FAILURE_DRAWDOWN_THRESHOLD)
        columns = [
            "date",
            "code",
            "close",
            "opportunity_top_decile_proba",
            "opportunity_score_proba_rank",
            "future_return_20d",
            "future_max_return_20d",
            "future_max_drawdown_20d",
            "opportunity_value_20d",
            "risk_adjusted_score",
            "expected_return",
            "stock_selection_rank_score",
            "candidate_strength",
            "relative_return_20d",
            "ma25_gap",
            "ma75_gap",
            "daily_range_ratio",
            "turnover_value",
        ]
        sample = data.loc[mask, [column for column in columns if column in data.columns]].copy()
        sample["_sort_loss"] = _numeric(sample.get("future_return_20d"))
        sample["_sort_dd"] = _numeric(sample.get("future_max_drawdown_20d"))
        sample = sample.sort_values(["_sort_loss", "_sort_dd"], ascending=[True, True]).head(20).drop(columns=["_sort_loss", "_sort_dd"], errors="ignore")
        return json.loads(sample.to_json(orient="records", date_format="iso"))

    def downside_filter_audit(self, scored: pd.DataFrame, valuation_top5: pd.DataFrame) -> list[dict[str, Any]]:
        thresholds = self.filter_thresholds(scored)
        audits = [self.filter_quality("no_filter", valuation_top5)]
        filters = {
            "daily_range_ratio_p90_or_lower": valuation_top5[_numeric(valuation_top5.get("daily_range_ratio")) <= thresholds.get("daily_range_ratio_p90", float("inf"))],
            "ma75_gap_p90_or_lower": valuation_top5[_numeric(valuation_top5.get("ma75_gap")) <= thresholds.get("ma75_gap_p90", float("inf"))],
            "relative_return_20d_p90_or_lower": valuation_top5[_numeric(valuation_top5.get("relative_return_20d")) <= thresholds.get("relative_return_20d_p90", float("inf"))],
            "turnover_value_p10_or_higher": valuation_top5[_numeric(valuation_top5.get("turnover_value")) >= thresholds.get("turnover_value_p10", float("-inf"))],
        }
        combo = valuation_top5.copy()
        for column in ["daily_range_ratio", "ma75_gap", "relative_return_20d"]:
            if column in combo.columns:
                combo = combo[_numeric(combo[column]) <= thresholds.get(f"{column}_p90", float("inf"))]
        if "turnover_value" in combo.columns:
            combo = combo[_numeric(combo["turnover_value"]) >= thresholds.get("turnover_value_p10", float("-inf"))]
        filters["combined_simple_downside_filter"] = combo
        for label, frame in filters.items():
            audits.append(self.filter_quality(label, frame))
        return audits

    def filter_thresholds(self, scored: pd.DataFrame) -> dict[str, float]:
        thresholds: dict[str, float] = {}
        for column in FILTER_FEATURES:
            if column not in scored.columns:
                continue
            values = _numeric(scored[column]).dropna()
            if values.empty:
                continue
            if column == "turnover_value":
                thresholds[f"{column}_p10"] = float(values.quantile(0.10))
            else:
                thresholds[f"{column}_p90"] = float(values.quantile(0.90))
        return thresholds

    def filter_quality(self, label: str, frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "filter": label,
            "rows": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if "date" in frame.columns else 0,
            "avg_candidates_per_day": _safe_float(len(frame) / frame["date"].nunique()) if "date" in frame.columns and frame["date"].nunique() else None,
            **self.quality_metrics(frame),
        }

    def diagnosis_summary(
        self,
        valuation_top5: dict[str, Any],
        exit_rows: list[dict[str, Any]],
        drift_rows: list[dict[str, Any]],
        filter_rows: list[dict[str, Any]],
        leakage: dict[str, Any],
    ) -> dict[str, Any]:
        downside_bad_rate = _safe_float(valuation_top5.get("downside_bad_rate")) if valuation_top5 else None
        downside_problem = bool(downside_bad_rate is not None and downside_bad_rate >= 0.30)
        exit_counts = {row["exit_reason"]: int(row["count"]) for row in exit_rows}
        opportunity_exits = exit_counts.get("opportunity_proba_drop", 0) + exit_counts.get("opportunity_rank_below_floor", 0)
        total_exits = sum(exit_counts.values())
        exit_overreaction = bool(total_exits and opportunity_exits / total_exits >= 0.60)
        drifted = [row for row in drift_rows if abs(_safe_float(row.get("standardized_mean_diff")) or 0.0) >= 0.50]
        feature_drift = bool(drifted)
        best_filter = self.best_downside_filter(filter_rows)

        reasons = []
        if downside_problem:
            reasons.append("valuation_top5_contains_high_downside_candidates")
        if exit_overreaction:
            reasons.append("opportunity_exit_is_dominant_and_may_be_unstable")
        if feature_drift:
            reasons.append("major_feature_distribution_drift_between_2023_train_and_2025_test")
        if not reasons:
            reasons.append("classification_rank_lift_does_not_translate_to_trade_path_under_current_exit_and_cost_assumptions")

        if downside_problem:
            recommended = "Phase11-B3 expected_downside model prototype"
        elif feature_drift:
            recommended = "Phase11-B3 feature drift robust model"
        elif exit_overreaction:
            recommended = "Phase11-E2 exit threshold recalibration"
        else:
            recommended = "Phase11-I2 strict OOS strategy simplification"

        return {
            "main_failure_reason": ", ".join(reasons),
            "downside_problem_detected": downside_problem,
            "exit_overreaction_detected": exit_overreaction,
            "feature_drift_detected": feature_drift,
            "drifted_features": [row["feature"] for row in drifted],
            "best_simple_filter": best_filter,
            "recommended_model_improvement": self.recommended_model_improvement(downside_problem, feature_drift, exit_overreaction),
            "recommended_next_phase": recommended if not leakage["blocking_issues"] else "Fix leakage blockers",
        }

    def best_downside_filter(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        baseline = next((row for row in rows if row["filter"] == "no_filter"), None)
        candidates = [row for row in rows if row["filter"] != "no_filter" and row.get("rows", 0) > 0]
        if not candidates:
            return None
        candidates.sort(key=lambda row: ((baseline.get("downside_bad_rate") or 0.0) - (row.get("downside_bad_rate") or 0.0), row.get("opportunity_value_20d_mean") or -10**9), reverse=True)
        best = candidates[0]
        return {
            "filter": best["filter"],
            "downside_bad_rate_delta": _safe_float((best.get("downside_bad_rate") or 0.0) - ((baseline or {}).get("downside_bad_rate") or 0.0)),
            "opportunity_value_delta": _safe_float((best.get("opportunity_value_20d_mean") or 0.0) - ((baseline or {}).get("opportunity_value_20d_mean") or 0.0)),
            "rows": best.get("rows"),
        }

    def recommended_model_improvement(self, downside_problem: bool, feature_drift: bool, exit_overreaction: bool) -> str:
        if downside_problem:
            return "Add expected_downside / drawdown-risk target before re-running strict OOS strategy variants."
        if feature_drift:
            return "Audit drift-stable features and calibrate the classifier on 2024 before strategy checks."
        if exit_overreaction:
            return "Recalibrate Opportunity Exit on strict OOS validation before changing the classifier."
        return "Simplify strict OOS strategy first, then test whether classification-only rank lift is economically usable."

    def leakage_checklist(self, feature_columns: list[str]) -> dict[str, Any]:
        helper = Phase11BValuationEnginePrototype(self.root)
        future = [column for column in feature_columns if column.startswith("future_") or column.startswith("opportunity_value") or column == CLASSIFICATION_TARGET]
        forbidden = [column for column in feature_columns if helper.is_forbidden_column(column)]
        blocking = []
        if future:
            blocking.append("future columns used as features")
        if forbidden:
            blocking.append("forbidden columns used as features")
        return {
            "future_columns_used_as_features": future,
            "future_columns_used_only_for_evaluation": FUTURE_EVAL_COLUMNS,
            "backtest_columns_used_as_features": [column for column in feature_columns if "backtest" in column.lower()],
            "trade_result_columns_used_as_features": [column for column in feature_columns if any(token in column.lower() for token in ["trade", "profit", "loss"])],
            "cash_or_portfolio_columns_used_as_model_features": [column for column in feature_columns if any(token in column.lower() for token in ["cash", "portfolio", "position"])],
            "selected_or_bought_used_as_features": any(any(token in column.lower() for token in ["selected", "bought", "affordable"]) for column in feature_columns),
            "current_pm_multiplier_used": any(any(token in column.lower() for token in ["pm_multiplier", "current_pm"]) for column in feature_columns),
            "historical_predictions_regenerated": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "leakage_risk": "high" if blocking else "low",
            "blocking_issues": blocking,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "11-B2",
            "scope": "2025 diagnosis only",
            "research_only": True,
            "model_source": str(self.root / PHASE11I_MODEL_DIR),
            "existing_model_overwritten": False,
            "phase11b_model_overwritten": False,
            "phase11i_model_overwritten": False,
            "profile_added": False,
            "profile_modified": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
            "historical_predictions_regenerated": False,
            "full_backtest_executed": False,
        }

    def dataset_summary(self, dataset: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(dataset)),
            "train_2023_rows": int(len(train)),
            "test_2025_rows": int(len(test)),
            "test_date_range": {"min": self.date_value(test, "min"), "max": self.date_value(test, "max")},
        }

    def date_value(self, frame: pd.DataFrame, mode: str) -> str | None:
        if frame.empty:
            return None
        value = frame["date"].min() if mode == "min" else frame["date"].max()
        return value.date().isoformat()

    def save_outputs(self, report: dict[str, Any]) -> Phase11B2Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase11B2Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 11-B2 Strict OOS Failure Diagnosis",
            "",
            "## Diagnosis Summary",
            "",
            self.table([report["diagnosis_summary"]], ["main_failure_reason", "downside_problem_detected", "exit_overreaction_detected", "feature_drift_detected", "recommended_model_improvement", "recommended_next_phase"]),
            "",
            "## Prediction Quality By Decile",
            "",
            self.table(report.get("prediction_quality_by_decile", []), ["prediction_decile", "count", "future_return_20d_mean", "future_max_return_20d_mean", "future_max_drawdown_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_rate", "downside_bad_rate"]),
            "",
            "## Top Candidate Diagnostics",
            "",
            self.table(report.get("top_candidate_diagnostics", []), ["candidate_set", "rows", "candidate_days", "avg_close", "avg_lot_cost", "avg_turnover_value", "avg_opportunity_top_decile_proba", "avg_opportunity_score_proba_rank", "future_return_20d_mean", "future_max_return_20d_mean", "future_max_drawdown_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_rate", "downside_bad_rate"]),
            "",
            "## Exit Trigger Diagnostics",
            "",
            self.table(report.get("exit_trigger_diagnostics", []), ["exit_reason", "count", "avg_holding_days", "avg_realized_return", "avg_entry_proba", "avg_exit_proba", "avg_proba_drop", "avg_entry_rank", "avg_exit_rank"]),
            "",
            "## Feature Drift Audit",
            "",
            self.table(report.get("feature_drift_audit", []), ["feature", "train_mean", "test_mean", "train_p10", "test_p10", "train_p90", "test_p90", "standardized_mean_diff"]),
            "",
            "## Downside Filter Audit",
            "",
            self.table(report.get("downside_filter_audit", []), ["filter", "rows", "candidate_days", "avg_candidates_per_day", "future_return_20d_mean", "future_max_return_20d_mean", "future_max_drawdown_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_rate", "downside_bad_rate"]),
            "",
            "## Candidate Failure Samples",
            "",
            self.table(report.get("candidate_failure_samples", []), ["date", "code", "close", "opportunity_top_decile_proba", "opportunity_score_proba_rank", "future_return_20d", "future_max_return_20d", "future_max_drawdown_20d", "opportunity_value_20d", "risk_adjusted_score", "expected_return", "stock_selection_rank_score", "candidate_strength", "relative_return_20d", "ma25_gap", "ma75_gap", "daily_range_ratio", "turnover_value"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_evaluation", "backtest_columns_used_as_features", "trade_result_columns_used_as_features", "cash_or_portfolio_columns_used_as_model_features", "selected_or_bought_used_as_features", "current_pm_multiplier_used", "historical_predictions_regenerated", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "leakage_risk", "blocking_issues"]),
            "",
        ]
        return "\n".join(lines)

    def table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(self.format_value(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def format_value(self, value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return f"{value:.4f}"
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, list):
            return ", ".join(map(str, value))
        if value is None:
            return ""
        return str(value)
