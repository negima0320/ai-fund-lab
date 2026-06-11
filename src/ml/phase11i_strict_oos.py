"""Phase 11-I strict walk-forward OOS prototype.

Research-only strict split:
- train: 2023
- validation: 2024
- test: 2025

This module saves a separate research model directory and never overwrites the
existing Phase 11-B candidate model, profiles, or current production models.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.phase11b_valuation_engine_prototype import (
    CLASSIFICATION_TARGET,
    DATASET_PATH,
    Phase11BValuationEnginePrototype,
    TARGET_COLUMNS,
)
from ml.phase11e_exit_dd_guard import FUTURE_EVAL_COLUMNS, _numeric, _safe_float


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path("models/ml/valuation_engine/research_phase11i_strict_oos")
REPORT_STEM = "phase11i_strict_walk_forward_oos_2025"

TRAIN_START = "2023-01-04"
TRAIN_END = "2023-12-31"
VALIDATION_START = "2024-01-01"
VALIDATION_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"
ROUND_LOT = 100


@dataclass(frozen=True)
class Phase11IOptions:
    max_train_rows: int = 250_000
    random_state: int = 42
    max_iter: int = 80
    learning_rate: float = 0.06
    save_model: bool = True
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
class StrategySpec:
    name: str
    rank_column: str
    exit_guard: bool
    cooldown_days: int = 0
    minimum_holding_guard_days: int = 0


@dataclass(frozen=True)
class Phase11IPaths:
    markdown: Path
    json: Path
    model_dir: Path | None


STRATEGIES = [
    StrategySpec("baseline_equal_allocation", rank_column="baseline_rank_score", exit_guard=False),
    StrategySpec("strict_oos_valuation_top5_no_guard", rank_column="opportunity_top_decile_proba", exit_guard=False),
    StrategySpec("strict_oos_E4", rank_column="opportunity_top_decile_proba", exit_guard=True),
    StrategySpec("strict_oos_H2_cooldown_10d", rank_column="opportunity_top_decile_proba", exit_guard=True, cooldown_days=10),
    StrategySpec("strict_oos_H3_min_hold_3d", rank_column="opportunity_top_decile_proba", exit_guard=True, minimum_holding_guard_days=3),
]


class Phase11IStrictOOS:
    def __init__(self, root: Path | str = ROOT, *, options: Phase11IOptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase11IOptions()

    def run(self) -> Phase11IPaths:
        report, model_bundle = self.build_report_and_model()
        return self.save_outputs(report, model_bundle)

    def build_report_and_model(self) -> tuple[dict[str, Any], dict[str, Any]]:
        dataset = self.load_dataset()
        feature_columns = Phase11BValuationEnginePrototype(self.root).extract_feature_columns(dataset)
        leakage = self.leakage_checklist(feature_columns)
        if leakage["blocking_issues"]:
            report = {
                "metadata": self.metadata(),
                "split": self.split_definition(),
                "feature_policy": {"feature_columns": feature_columns, "feature_count": len(feature_columns)},
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], leakage),
            }
            return report, {}

        train, validation, test = self.split_dataset(dataset)
        train_prepared, validation_prepared, test_prepared = self.prepare_frames(train, validation, test, feature_columns)
        classifier = self.train_classifier(train_prepared, feature_columns)
        validation_predictions = self.predict(classifier, validation_prepared, feature_columns)
        test_predictions = self.predict(classifier, test_prepared, feature_columns)
        test_scored = self.attach_predictions(test_prepared, test_predictions)
        strategy_results, buy_quality = self.strategy_check(test_scored)
        model_quality = self.model_quality(test_prepared, test_predictions)
        validation_quality = self.model_quality(validation_prepared, validation_predictions)
        report = {
            "metadata": self.metadata(),
            "split": self.split_definition(),
            "dataset_summary": self.dataset_summary(dataset, train, validation, test),
            "feature_policy": {"feature_columns": feature_columns, "feature_count": len(feature_columns), "target_columns_excluded": sorted(TARGET_COLUMNS)},
            "model_config": {
                "model": "HistGradientBoostingClassifier",
                "max_iter": self.options.max_iter,
                "learning_rate": self.options.learning_rate,
                "max_train_rows": self.options.max_train_rows,
                "random_state": self.options.random_state,
            },
            "validation_model_quality": validation_quality,
            "test_model_quality": model_quality,
            "phase11b_reference_2025": {
                "AUC": 0.6478,
                "PR_AUC": 0.1600,
                "precision_at_top10": 0.1998,
                "base_positive_rate": 0.0997,
            },
            "strategy_results": strategy_results,
            "buy_quality": buy_quality,
            "strict_oos_judgement": self.strict_oos_judgement(strategy_results),
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(strategy_results, leakage),
        }
        model_bundle = {"classifier": classifier, "feature_columns": feature_columns, "report_metadata": report["metadata"]}
        return report, model_bundle

    def load_dataset(self) -> pd.DataFrame:
        data = pd.read_parquet(self.root / DATASET_PATH)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        return data.dropna(subset=["date", "code", CLASSIFICATION_TARGET]).reset_index(drop=True)

    def split_dataset(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = dataset[(dataset["date"] >= TRAIN_START) & (dataset["date"] <= TRAIN_END)].copy()
        validation = dataset[(dataset["date"] >= VALIDATION_START) & (dataset["date"] <= VALIDATION_END)].copy()
        test = dataset[(dataset["date"] >= TEST_START) & (dataset["date"] <= TEST_END)].copy()
        if self.options.max_train_rows and len(train) > self.options.max_train_rows:
            train = train.sample(n=self.options.max_train_rows, random_state=self.options.random_state).sort_values(["date", "code"])
        return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)

    def prepare_frames(self, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        frames = []
        for frame in [train, validation, test]:
            data = frame.copy()
            for column in feature_columns:
                if pd.api.types.is_bool_dtype(data[column].dtype):
                    data[column] = data[column].astype(int)
                else:
                    data[column] = _numeric(data[column])
            data[feature_columns] = data[feature_columns].replace([float("inf"), float("-inf")], pd.NA)
            frames.append(data)
        return frames[0], frames[1], frames[2]

    def train_classifier(self, train: pd.DataFrame, feature_columns: list[str]) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier

        classifier = HistGradientBoostingClassifier(
            max_iter=self.options.max_iter,
            learning_rate=self.options.learning_rate,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            early_stopping=True,
            random_state=self.options.random_state,
        )
        classifier.fit(train[feature_columns], train[CLASSIFICATION_TARGET].astype(int))
        return classifier

    def predict(self, classifier: Any, frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
        return pd.Series(classifier.predict_proba(frame[feature_columns])[:, 1], index=frame.index, name="opportunity_top_decile_proba")

    def attach_predictions(self, test: pd.DataFrame, proba: pd.Series) -> pd.DataFrame:
        data = test.copy()
        data["opportunity_top_decile_proba"] = proba
        data["opportunity_score_proba_rank"] = data.groupby("date")["opportunity_top_decile_proba"].rank(method="average", pct=True)
        baseline_rank = data["stock_selection_rank_score"] if "stock_selection_rank_score" in data.columns else pd.Series(dtype=float)
        if baseline_rank.isna().all():
            baseline_rank = data["risk_adjusted_score"]
        data["baseline_rank_score"] = _numeric(baseline_rank).fillna(-10**18)
        return data.dropna(subset=["date", "code", "close"]).sort_values(["date", "code"]).reset_index(drop=True)

    def model_quality(self, frame: pd.DataFrame, proba: pd.Series) -> dict[str, Any]:
        from sklearn.metrics import average_precision_score, roc_auc_score

        actual = frame[CLASSIFICATION_TARGET].astype(int)
        positive_rate = float(actual.mean()) if len(actual) else None
        top_n = max(1, int(len(proba) * 0.10))
        top_index = proba.sort_values(ascending=False).head(top_n).index
        deciles = self.prediction_deciles(frame, proba)
        return {
            "target": CLASSIFICATION_TARGET,
            "AUC": _safe_float(roc_auc_score(actual, proba)) if actual.nunique() > 1 else None,
            "PR_AUC": _safe_float(average_precision_score(actual, proba)) if actual.nunique() > 1 else None,
            "precision_at_top10": _safe_float(actual.loc[top_index].mean()) if top_n else None,
            "base_positive_rate": _safe_float(positive_rate),
            "prediction_decile_actual_positive_rate": deciles,
        }

    def prediction_deciles(self, frame: pd.DataFrame, proba: pd.Series) -> list[dict[str, Any]]:
        data = pd.DataFrame({"proba": proba, "actual": frame[CLASSIFICATION_TARGET].astype(int)})
        data["decile"] = pd.qcut(data["proba"].rank(method="first"), 10, labels=False, duplicates="drop") + 1
        return [
            {"prediction_decile": int(decile), "count": int(len(group)), "actual_positive_rate": _safe_float(group["actual"].mean()), "proba_mean": _safe_float(group["proba"].mean())}
            for decile, group in data.groupby("decile", sort=True)
        ]

    def strategy_check(self, data: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results = []
        quality = []
        for spec in STRATEGIES:
            trades, daily = self.simulate(data, spec)
            results.append(self.metrics(spec.name, trades, daily))
            quality.append(self.buy_quality(spec.name, trades))
        return results, quality

    def simulate(self, data: pd.DataFrame, spec: StrategySpec) -> tuple[pd.DataFrame, pd.DataFrame]:
        cash = self.options.initial_cash
        positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        daily_rows: list[dict[str, Any]] = []
        last_exit_by_code: dict[str, pd.Timestamp] = {}
        dates = list(pd.Series(data["date"].dropna().unique()).sort_values())
        by_date = {date: group.set_index("code", drop=False) for date, group in data.groupby("date", sort=True)}

        for current_date in dates:
            current = by_date[current_date]
            current_rank_frame = current.reset_index(drop=True)
            still_open = []
            for position in positions:
                current_row = current.loc[position["code"]] if position["code"] in current.index else None
                reason = self.exit_reason(position, current_date, current_row, spec)
                if reason:
                    exit_close = float(current_row["close"]) if current_row is not None else float(position["last_close"])
                    trade = self.close_position(position, current_date, exit_close, reason, spec.name)
                    cash += trade["exit_cash_flow"]
                    trades.append(trade)
                    last_exit_by_code[position["code"]] = current_date
                else:
                    if current_row is not None:
                        position["last_close"] = float(current_row["close"])
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            ranked = current_rank_frame.sort_values([spec.rank_column, "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions * 4)
            selected = []
            for _, row in ranked.iterrows():
                if len(selected) >= slots:
                    break
                code = str(row["code"])
                if spec.cooldown_days and self.in_cooldown(code, current_date, last_exit_by_code, spec.cooldown_days):
                    continue
                selected.append(row)
            selected_frame = pd.DataFrame(selected)
            raw_amount = min(cash, self.options.daily_buy_budget) / max(1, min(self.options.max_positions, len(ranked)))
            bought_today = 0
            for _, row in selected_frame.iterrows():
                lot_cost = float(row["close"]) * self.options.round_lot
                lots = int(raw_amount // (lot_cost * (1.0 + self.options.cost_rate))) if lot_cost > 0 else 0
                buy_amount = lots * lot_cost
                buy_cost = buy_amount * self.options.cost_rate
                cash_out = buy_amount + buy_cost
                if lots <= 0 or cash_out > cash:
                    continue
                cash -= cash_out
                bought_today += 1
                positions.append(
                    {
                        "entry_date": current_date,
                        "due_date": current_date + pd.offsets.BDay(self.options.holding_days),
                        "strategy": spec.name,
                        "code": str(row["code"]),
                        "buy_amount": buy_amount,
                        "buy_cost": buy_cost,
                        "lot_count": lots,
                        "entry_close": float(row["close"]),
                        "last_close": float(row["close"]),
                        "entry_opportunity_top_decile_proba": _safe_float(row.get("opportunity_top_decile_proba")),
                        **{column: _safe_float(row.get(column)) for column in FUTURE_EVAL_COLUMNS},
                    }
                )

            marked_value = sum(float(position["lot_count"]) * self.options.round_lot * float(position["last_close"]) for position in positions)
            daily_rows.append(
                {
                    "strategy": spec.name,
                    "date": current_date,
                    "cash": cash,
                    "open_position_count": len(positions),
                    "bought_today": bought_today,
                    "marked_position_value": marked_value,
                    "total_assets": cash + marked_value,
                    "capital_utilization": marked_value / self.options.initial_cash if self.options.initial_cash else None,
                }
            )
        if dates:
            last_date = dates[-1]
            for position in positions:
                trade = self.close_position(position, last_date, float(position["last_close"]), "forced_end_of_period", position.get("strategy", "forced_end_of_period"))
                cash += trade["exit_cash_flow"]
                trades.append(trade)
            if daily_rows:
                daily_rows[-1]["total_assets"] = cash
                daily_rows[-1]["marked_position_value"] = 0.0
                daily_rows[-1]["capital_utilization"] = 0.0
        return pd.DataFrame(trades), pd.DataFrame(daily_rows)

    def in_cooldown(self, code: str, current_date: pd.Timestamp, last_exit_by_code: dict[str, pd.Timestamp], cooldown_days: int) -> bool:
        if code not in last_exit_by_code:
            return False
        return len(pd.bdate_range(last_exit_by_code[code], current_date)) - 1 <= cooldown_days

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None, spec: StrategySpec) -> str | None:
        holding_days = len(pd.bdate_range(position["entry_date"], current_date)) - 1
        if current_row is not None and spec.exit_guard:
            observed_return = float(current_row["close"]) / float(position["entry_close"]) - 1.0
            if observed_return <= self.options.stop_loss_rate:
                return "stop_loss"
            if holding_days >= spec.minimum_holding_guard_days:
                current_proba = _safe_float(current_row.get("opportunity_top_decile_proba"))
                current_rank = _safe_float(current_row.get("opportunity_score_proba_rank"))
                entry_proba = _safe_float(position.get("entry_opportunity_top_decile_proba"))
                if current_rank is not None and current_rank < self.options.opportunity_rank_floor:
                    return "opportunity_rank_below_floor"
                if current_proba is not None and entry_proba is not None and current_proba <= entry_proba - self.options.opportunity_drop_threshold:
                    return "opportunity_proba_drop"
        if current_date >= position["due_date"]:
            return "time_exit_20d"
        return None

    def close_position(self, position: dict[str, Any], exit_date: pd.Timestamp, exit_close: float, reason: str, strategy: str) -> dict[str, Any]:
        exit_amount = float(position["lot_count"]) * self.options.round_lot * exit_close
        sell_cost = exit_amount * self.options.cost_rate
        exit_cash_flow = exit_amount - sell_cost
        total_cost = float(position["buy_cost"]) + sell_cost
        profit = exit_cash_flow - float(position["buy_amount"]) - float(position["buy_cost"])
        holding_days = len(pd.bdate_range(position["entry_date"], exit_date)) - 1
        return {
            "strategy": strategy,
            "entry_date": position["entry_date"],
            "exit_date": exit_date,
            "code": position["code"],
            "buy_amount": position["buy_amount"],
            "exit_amount": exit_amount,
            "exit_cash_flow": exit_cash_flow,
            "realized_profit": profit,
            "realized_return": profit / float(position["buy_amount"]) if position["buy_amount"] else None,
            "holding_days": holding_days,
            "exit_reason": reason,
            "cost_paid": total_cost,
            **{column: position.get(column) for column in FUTURE_EVAL_COLUMNS},
        }

    def metrics(self, strategy: str, trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
        profits = _numeric(trades["realized_profit"]) if not trades.empty else pd.Series(dtype=float)
        gross_profit = float(profits[profits > 0].sum()) if not profits.empty else 0.0
        gross_loss = abs(float(profits[profits < 0].sum())) if not profits.empty else 0.0
        equity = _numeric(daily["total_assets"]) if not daily.empty else pd.Series([self.options.initial_cash])
        drawdown = equity / equity.cummax() - 1.0
        reentries = self.reentry_counts(trades)
        return {
            "strategy": strategy,
            "net_profit": _safe_float(profits.sum()) if not profits.empty else 0.0,
            "PF": _safe_float(gross_profit / gross_loss) if gross_loss else (None if gross_profit == 0 else float("inf")),
            "DD": _safe_float(drawdown.min()) if not drawdown.empty else 0.0,
            "win_rate": _safe_float((profits > 0).mean()) if not profits.empty else None,
            "total_trades": int(len(trades)),
            "final_assets": _safe_float(self.options.initial_cash + profits.sum()) if not profits.empty else self.options.initial_cash,
            "capital_utilization": _safe_float(_numeric(daily["capital_utilization"]).mean()) if not daily.empty else None,
            "average_holding_days": _safe_float(_numeric(trades["holding_days"]).mean()) if not trades.empty else None,
            "median_holding_days": _safe_float(_numeric(trades["holding_days"]).median()) if not trades.empty else None,
            "same_code_reentry_count": reentries["same_code_reentry_count"],
            "reentry_within_5_days_count": reentries["reentry_within_5_days_count"],
            "exit_reason_counts": dict(Counter(trades["exit_reason"])) if "exit_reason" in trades.columns else {},
            "cost_paid": _safe_float(_numeric(trades["cost_paid"]).sum()) if "cost_paid" in trades.columns else 0.0,
        }

    def buy_quality(self, strategy: str, trades: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {"strategy": strategy, "buy_count": int(len(trades))}
        for column in FUTURE_EVAL_COLUMNS:
            values = _numeric(trades[column]) if column in trades.columns else pd.Series(dtype=float)
            key = "opportunity_top_decile_20d_rate" if column == "opportunity_top_decile_20d" else f"{column}_mean"
            row[key] = _safe_float(values.mean()) if not values.empty else None
        return row

    def reentry_counts(self, trades: pd.DataFrame) -> dict[str, int]:
        if trades.empty:
            return {"same_code_reentry_count": 0, "reentry_within_5_days_count": 0}
        same_code_reentry = 0
        reentry_5d = 0
        for _, group in trades.sort_values(["code", "entry_date"]).groupby("code", sort=False):
            previous_exit = None
            for _, row in group.iterrows():
                entry = pd.Timestamp(row["entry_date"])
                if previous_exit is not None:
                    same_code_reentry += 1
                    if len(pd.bdate_range(previous_exit, entry)) - 1 <= 5:
                        reentry_5d += 1
                previous_exit = pd.Timestamp(row["exit_date"])
        return {"same_code_reentry_count": same_code_reentry, "reentry_within_5_days_count": reentry_5d}

    def strict_oos_judgement(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_strategy = {row["strategy"]: row for row in rows}
        baseline = by_strategy.get("baseline_equal_allocation", {})
        candidates = [by_strategy.get(name, {}) for name in ["strict_oos_E4", "strict_oos_H2_cooldown_10d", "strict_oos_H3_min_hold_3d"]]
        passed = []
        better = []
        for row in candidates:
            if not row:
                continue
            name = row.get("strategy")
            beats_baseline = (_safe_float(row.get("net_profit")) or -10**18) > (_safe_float(baseline.get("net_profit")) or 10**18)
            base_pass = beats_baseline and (_safe_float(row.get("PF")) or 0.0) >= 1.5 and (_safe_float(row.get("DD")) or -1.0) >= -0.12 and (_safe_float(row.get("net_profit")) or 0.0) > 0
            better_pass = base_pass and (_safe_float(row.get("PF")) or 0.0) >= 1.8 and (_safe_float(row.get("DD")) or -1.0) >= -0.10
            if name in {"strict_oos_H2_cooldown_10d", "strict_oos_H3_min_hold_3d"}:
                e4 = by_strategy.get("strict_oos_E4", {})
                better_pass = better_pass and int(row.get("reentry_within_5_days_count") or 0) < int(e4.get("reentry_within_5_days_count") or 10**9)
            if base_pass:
                passed.append(name)
            if better_pass:
                better.append(name)
        return {"passed_strategies": passed, "better_passed_strategies": better, "all_passed": bool(passed), "better_conditions_passed": bool(better)}

    def dataset_summary(self, dataset: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(dataset)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "train_date_range": self.date_range(train),
            "validation_date_range": self.date_range(validation),
            "test_date_range": self.date_range(test),
        }

    def date_range(self, frame: pd.DataFrame) -> dict[str, Any]:
        return {"min": frame["date"].min().date().isoformat() if not frame.empty else None, "max": frame["date"].max().date().isoformat() if not frame.empty else None}

    def split_definition(self) -> dict[str, Any]:
        return {
            "train": {"start": TRAIN_START, "end": TRAIN_END},
            "validation": {"start": VALIDATION_START, "end": VALIDATION_END},
            "test": {"start": TEST_START, "end": TEST_END},
            "train_validation_test_overlap": False,
            "strict_model_oos": True,
        }

    def leakage_checklist(self, feature_columns: list[str]) -> dict[str, Any]:
        future_columns = [column for column in feature_columns if column.startswith("future_") or column.startswith("opportunity_value") or column == CLASSIFICATION_TARGET]
        forbidden = [column for column in feature_columns if Phase11BValuationEnginePrototype(self.root).is_forbidden_column(column)]
        blocking = []
        if future_columns:
            blocking.append("future columns used as features")
        if forbidden:
            blocking.append("forbidden columns used as features")
        return {
            "future_columns_used_as_features": future_columns,
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
            "train_validation_test_overlap": False,
            "strict_model_oos": True,
            "leakage_risk": "high" if blocking else "low",
            "blocking_issues": blocking,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "11-I",
            "research_only": True,
            "classification_only": True,
            "existing_model_overwritten": False,
            "candidate_phase11b_overwritten": False,
            "profile_added": False,
            "profile_modified": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
            "full_backtest_executed": False,
            "model_dir": str(self.root / MODEL_DIR),
        }

    def recommendation(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"strict_oos_effective": False, "recommended_next_phase": "Fix Phase11-I leakage blockers"}
        judgement = self.strict_oos_judgement(rows) if rows else {"all_passed": False, "better_conditions_passed": False}
        return {
            "strict_oos_effective": bool(judgement.get("all_passed")),
            "better_conditions_passed": bool(judgement.get("better_conditions_passed")),
            "recommended_next_phase": "Phase11-J strict OOS refinement or limited integration design" if judgement.get("all_passed") else "Phase11-B2 valuation model improvement",
            "reason": "Strict OOS pass requires a strict-oos valuation strategy to beat baseline with PF >= 1.5, DD >= -12%, and positive net profit under 0.2% cost.",
        }

    def save_outputs(self, report: dict[str, Any], model_bundle: dict[str, Any]) -> Phase11IPaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        model_dir = self.root / MODEL_DIR if self.options.save_model and model_bundle else None
        if model_dir:
            self.save_model_bundle(model_dir, model_bundle, report)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase11IPaths(markdown=markdown_path, json=json_path, model_dir=model_dir)

    def save_model_bundle(self, model_dir: Path, model_bundle: dict[str, Any], report: dict[str, Any]) -> None:
        import joblib

        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_bundle["classifier"], model_dir / "opportunity_top_decile_20d_classifier.joblib")
        (model_dir / "feature_columns.json").write_text(json.dumps(model_bundle["feature_columns"], ensure_ascii=False, indent=2), encoding="utf-8")
        metadata = {
            "phase": "11-I",
            "research_only": True,
            "strict_model_oos": True,
            "split": report["split"],
            "feature_count": len(model_bundle["feature_columns"]),
            "classification_target": CLASSIFICATION_TARGET,
            "existing_model_overwritten": False,
        }
        (model_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 11-I Strict Walk-Forward OOS Prototype",
            "",
            "## Scope",
            "",
            "- train 2023, validation 2024, test 2025",
            "- classification only",
            "- research-only model directory; existing candidate_phase11b is not overwritten",
            "",
            "## Model Quality 2025",
            "",
            self.table([report.get("test_model_quality", {})], ["AUC", "PR_AUC", "precision_at_top10", "base_positive_rate"]),
            "",
            "## Strategy Results 2025",
            "",
            self.table(report.get("strategy_results", []), ["strategy", "net_profit", "PF", "DD", "win_rate", "total_trades", "final_assets", "capital_utilization", "average_holding_days", "median_holding_days", "same_code_reentry_count", "reentry_within_5_days_count", "exit_reason_counts", "cost_paid"]),
            "",
            "## BUY Quality",
            "",
            self.table(report.get("buy_quality", []), ["strategy", "buy_count", "future_return_20d_mean", "future_max_return_20d_mean", "future_max_drawdown_20d_mean", "opportunity_value_20d_mean", "opportunity_top_decile_20d_rate"]),
            "",
            "## Strict OOS Judgement",
            "",
            self.table([report.get("strict_oos_judgement", {})], ["passed_strategies", "better_passed_strategies", "all_passed", "better_conditions_passed"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_evaluation", "backtest_columns_used_as_features", "trade_result_columns_used_as_features", "cash_or_portfolio_columns_used_as_model_features", "selected_or_bought_used_as_features", "current_pm_multiplier_used", "historical_predictions_regenerated", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "train_validation_test_overlap", "strict_model_oos", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["strict_oos_effective", "better_conditions_passed", "recommended_next_phase", "reason"]),
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
