"""Phase 11-H cooldown / minimum-holding guard and strict OOS design.

This is a 2024/2025-only lightweight check. It does not execute a full-period
backtest, retrain walk-forward models, regenerate historical predictions, or
change profiles. Future columns are retained only for reporting quality.
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

from ml.phase11e_exit_dd_guard import FUTURE_EVAL_COLUMNS, _numeric, _safe_float


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = Path("data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet")
MODEL_DIR = Path("models/ml/valuation_engine/candidate_phase11b")
REPORT_STEM = "phase11h_cooldown_minhold_guard_2024_2025"

YEARS = [2024, 2025]
ROUND_LOT = 100
BASELINE_RANK_COLUMNS = ["stock_selection_rank_score", "risk_adjusted_score", "expected_return", "candidate_strength"]


@dataclass(frozen=True)
class Phase11HOptions:
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
class GuardVariant:
    name: str
    cooldown_days: int = 0
    minimum_holding_guard_days: int = 0


@dataclass(frozen=True)
class Phase11HPaths:
    markdown: Path
    json: Path


VARIANTS = [
    GuardVariant("H0_baseline_E4", cooldown_days=0, minimum_holding_guard_days=0),
    GuardVariant("H1_cooldown_5d", cooldown_days=5, minimum_holding_guard_days=0),
    GuardVariant("H2_cooldown_10d", cooldown_days=10, minimum_holding_guard_days=0),
    GuardVariant("H3_min_hold_3d", cooldown_days=0, minimum_holding_guard_days=3),
    GuardVariant("H4_cooldown_5d_min_hold_3d", cooldown_days=5, minimum_holding_guard_days=3),
]


class Phase11HCooldownMinHoldGuard:
    def __init__(self, root: Path | str = ROOT, *, options: Phase11HOptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase11HOptions()

    def run(self) -> Phase11HPaths:
        report = self.build_report()
        return self.save_report(report)

    def build_report(self) -> dict[str, Any]:
        data = self.load_scored_frame()
        leakage = self.leakage_checklist()
        if leakage["blocking_issues"]:
            return {
                "metadata": self.metadata(),
                "conditions": self.conditions(),
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], leakage),
            }

        variant_results = []
        buy_quality = []
        for year in YEARS:
            year_data = data[data["date"].dt.year == year].copy()
            for variant in VARIANTS:
                trades, daily = self.simulate(year_data, variant)
                variant_results.append(self.metrics(year, variant.name, trades, daily))
                buy_quality.append(self.buy_quality(year, variant.name, trades))

        report = {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "variant_results": variant_results,
            "buy_quality": buy_quality,
            "guard_effectiveness": self.guard_effectiveness(variant_results),
            "strict_walk_forward_oos_design": self.strict_oos_design(),
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(variant_results, leakage),
        }
        return report

    def load_scored_frame(self) -> pd.DataFrame:
        import joblib

        model_dir = self.root / MODEL_DIR
        feature_columns = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
        classifier = joblib.load(model_dir / "opportunity_top_decile_20d_classifier.joblib")
        columns = sorted(set(["date", "code", "close", "turnover_value", *BASELINE_RANK_COLUMNS, *feature_columns, *FUTURE_EVAL_COLUMNS]))
        data = pd.read_parquet(self.root / DATASET_PATH, columns=columns)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        data = data[data["date"].dt.year.isin(YEARS)].copy()
        for column in columns:
            if column not in {"date", "code"}:
                data[column] = _numeric(data[column])
        proba = np.asarray(classifier.predict_proba(data[feature_columns]))[:, 1]
        data["opportunity_top_decile_proba"] = proba
        data["opportunity_score_proba_rank"] = data.groupby("date")["opportunity_top_decile_proba"].rank(method="average", pct=True)
        return data.dropna(subset=["date", "code", "close"]).sort_values(["date", "code"]).reset_index(drop=True)

    def simulate(self, data: pd.DataFrame, variant: GuardVariant) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                reason = self.exit_reason(position, current_date, current_row, variant)
                if reason:
                    exit_close = float(current_row["close"]) if current_row is not None else float(position["last_close"])
                    trade = self.close_position(position, current_date, exit_close, reason, variant.name)
                    cash += trade["exit_cash_flow"]
                    trades.append(trade)
                    last_exit_by_code[position["code"]] = current_date
                else:
                    if current_row is not None:
                        position["last_close"] = float(current_row["close"])
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            ranked = current_rank_frame.sort_values(["opportunity_top_decile_proba", "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions * 4)
            selected = []
            for _, row in ranked.iterrows():
                if len(selected) >= slots:
                    break
                code = str(row["code"])
                if self.in_cooldown(code, current_date, last_exit_by_code, variant.cooldown_days):
                    continue
                selected.append(row)
            selected_frame = pd.DataFrame(selected)
            top5_count = min(self.options.max_positions, len(ranked))
            raw_amount = min(cash, self.options.daily_buy_budget) / max(1, top5_count)
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
                    "variant": variant.name,
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
                trade = self.close_position(position, last_date, float(position["last_close"]), "forced_end_of_period", variant.name)
                cash += trade["exit_cash_flow"]
                trades.append(trade)
            if daily_rows:
                daily_rows[-1]["total_assets"] = cash
                daily_rows[-1]["marked_position_value"] = 0.0
                daily_rows[-1]["capital_utilization"] = 0.0
        return pd.DataFrame(trades), pd.DataFrame(daily_rows)

    def in_cooldown(self, code: str, current_date: pd.Timestamp, last_exit_by_code: dict[str, pd.Timestamp], cooldown_days: int) -> bool:
        if cooldown_days <= 0 or code not in last_exit_by_code:
            return False
        elapsed = len(pd.bdate_range(last_exit_by_code[code], current_date)) - 1
        return elapsed <= cooldown_days

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None, variant: GuardVariant) -> str | None:
        holding_days = len(pd.bdate_range(position["entry_date"], current_date)) - 1
        if current_row is not None:
            observed_return = float(current_row["close"]) / float(position["entry_close"]) - 1.0
            if observed_return <= self.options.stop_loss_rate:
                return "stop_loss"
            opportunity_allowed = holding_days >= variant.minimum_holding_guard_days
            if opportunity_allowed:
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

    def close_position(self, position: dict[str, Any], exit_date: pd.Timestamp, exit_close: float, reason: str, variant: str) -> dict[str, Any]:
        exit_amount = float(position["lot_count"]) * self.options.round_lot * exit_close
        sell_cost = exit_amount * self.options.cost_rate
        exit_cash_flow = exit_amount - sell_cost
        total_cost = float(position["buy_cost"]) + sell_cost
        profit = exit_cash_flow - float(position["buy_amount"]) - float(position["buy_cost"])
        holding_days = len(pd.bdate_range(position["entry_date"], exit_date)) - 1
        return {
            "variant": variant,
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

    def metrics(self, year: int, variant: str, trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
        profits = _numeric(trades["realized_profit"]) if not trades.empty else pd.Series(dtype=float)
        gross_profit = float(profits[profits > 0].sum()) if not profits.empty else 0.0
        gross_loss = abs(float(profits[profits < 0].sum())) if not profits.empty else 0.0
        equity = _numeric(daily["total_assets"]) if not daily.empty else pd.Series([self.options.initial_cash])
        drawdown = equity / equity.cummax() - 1.0
        reentries = self.reentry_counts(trades)
        return {
            "year": year,
            "variant": variant,
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

    def buy_quality(self, year: int, variant: str, trades: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {"year": year, "variant": variant, "buy_count": int(len(trades))}
        for column in FUTURE_EVAL_COLUMNS:
            values = _numeric(trades[column]) if column in trades.columns else pd.Series(dtype=float)
            row[f"{column}_mean"] = _safe_float(values.mean()) if not values.empty else None
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

    def guard_effectiveness(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_key = {(row["year"], row["variant"]): row for row in rows}
        out = []
        for year in YEARS:
            h0 = by_key.get((year, "H0_baseline_E4"), {})
            for variant in VARIANTS:
                row = by_key.get((year, variant.name), {})
                out.append(
                    {
                        "year": year,
                        "variant": variant.name,
                        "reentry_within_5_days_delta_vs_h0": self.delta(row.get("reentry_within_5_days_count"), h0.get("reentry_within_5_days_count")),
                        "net_profit_delta_vs_h0": self.delta(row.get("net_profit"), h0.get("net_profit")),
                        "PF_delta_vs_h0": self.delta(row.get("PF"), h0.get("PF")),
                        "DD_delta_vs_h0": self.delta(row.get("DD"), h0.get("DD")),
                        "passes_basic_thresholds": self.passes_basic(row),
                    }
                )
        return out

    def passes_basic(self, row: dict[str, Any]) -> bool:
        return (
            (_safe_float(row.get("PF")) or 0.0) >= 1.8
            and (_safe_float(row.get("DD")) or -1.0) >= -0.10
            and (_safe_float(row.get("net_profit")) or 0.0) > 0
        )

    def delta(self, value: Any, base: Any) -> float | None:
        value_f = _safe_float(value)
        base_f = _safe_float(base)
        if value_f is None or base_f is None:
            return None
        return value_f - base_f

    def strict_oos_design(self) -> dict[str, Any]:
        return {
            "why_2024_is_not_strict_oos": "Phase11-B candidate model was trained through 2024-12-31, so 2024 strategy results overlap the model train window.",
            "required_retraining_splits": {
                "recommended_first_split": {"train": "2023-01-04 to 2023-12-31", "validation": "2024-01-01 to 2024-12-31", "test": "2025-01-01 to 2025-12-31"},
                "later_split": {"train": "2023-2024", "validation": "2025", "test": "2026 available range"},
            },
            "required_artifacts": [
                "versioned Phase11 valuation dataset snapshot",
                "candidate_phase11 strict-oos model directory, separate from candidate_phase11b",
                "feature_columns.json per split",
                "year-specific in-memory or saved prediction artifact with explicit as-of policy",
                "Phase11-H/I comparison reports",
            ],
            "estimated_cost_risk": "medium: retraining is lightweight, but strict walk-forward prediction management and artifact lineage need careful controls.",
            "recommended_first_strict_oos_year": 2025,
            "minimal_safe_execution_plan": [
                "Do not overwrite candidate_phase11b or current production models.",
                "Train a new research-only model using 2023 only.",
                "Use 2024 only for validation/threshold choice.",
                "Evaluate 2025 once as strict test.",
                "Save all split metadata and leakage checklist before any wider backtest.",
            ],
            "phase11h_retraining_executed": False,
            "phase11h_walk_forward_prediction_regenerated": False,
        }

    def dataset_summary(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for year, group in data.groupby(data["date"].dt.year):
            rows.append(
                {
                    "year": int(year),
                    "rows": int(len(group)),
                    "unique_codes": int(group["code"].nunique()),
                    "candidate_days": int(group["date"].nunique()),
                    "date_min": group["date"].min().date().isoformat(),
                    "date_max": group["date"].max().date().isoformat(),
                }
            )
        return rows

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "11-H",
            "years": YEARS,
            "limited_years_only": True,
            "full_backtest_executed": False,
            "profile_added": False,
            "profile_modified": False,
            "current_model_overwritten": False,
            "historical_predictions_regenerated": False,
            "historical_predictions_saved": False,
            "walk_forward_retraining_executed": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
            "dataset_path": str(self.root / DATASET_PATH),
            "model_dir": str(self.root / MODEL_DIR),
        }

    def conditions(self) -> dict[str, Any]:
        return {
            "base_strategy": "E4 with cost 0.2%",
            "daily_buy_budget": self.options.daily_buy_budget,
            "max_positions": self.options.max_positions,
            "holding_days": self.options.holding_days,
            "stop_loss": self.options.stop_loss_rate,
            "opportunity_drop_threshold": self.options.opportunity_drop_threshold,
            "opportunity_rank_floor": self.options.opportunity_rank_floor,
            "cost_rate": self.options.cost_rate,
            "variants": [variant.__dict__ for variant in VARIANTS],
        }

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_as_features": [],
            "future_columns_used_only_for_evaluation": FUTURE_EVAL_COLUMNS,
            "backtest_columns_used_as_features": [],
            "trade_result_columns_used_as_features": [],
            "cash_or_portfolio_columns_used_as_model_features": [],
            "selected_or_bought_used_as_features": False,
            "current_pm_multiplier_used": False,
            "historical_predictions_regenerated": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "walk_forward_retraining_executed": False,
            "decision_columns": ["opportunity_top_decile_proba", "opportunity_score_proba_rank", "close", "turnover_value"],
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def recommendation(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"cooldown_minhold_passed": False, "recommended_next_phase": "Fix Phase11-H leakage blockers"}
        effectiveness = self.guard_effectiveness(rows)
        candidates = []
        for variant in [v.name for v in VARIANTS if v.name != "H0_baseline_E4"]:
            yearly = [row for row in effectiveness if row["variant"] == variant]
            if len(yearly) == len(YEARS) and all(row["passes_basic_thresholds"] for row in yearly) and all((row["reentry_within_5_days_delta_vs_h0"] or 0) < 0 for row in yearly):
                candidates.append(variant)
        return {
            "cooldown_minhold_passed": bool(candidates),
            "passing_variants": candidates,
            "recommended_next_phase": "Phase11-I strict walk-forward OOS prototype" if candidates else "Phase11-H2 guard threshold adjustment",
            "reason": "Pass requires PF >= 1.8, DD >= -10%, positive net profit, and lower 5-day reentry than H0 in both 2024 and 2025.",
        }

    def save_report(self, report: dict[str, Any]) -> Phase11HPaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase11HPaths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 11-H Cooldown / Minimum Holding Guard 2024-2025",
            "",
            "## Scope",
            "",
            "- 2024 and 2025 only",
            "- base strategy: E4 with 0.2% one-way cost",
            "- no full backtest, no profile change, no walk-forward retraining",
            "",
            "## Conditions",
            "",
            self.table([report["conditions"]], ["base_strategy", "daily_buy_budget", "max_positions", "holding_days", "stop_loss", "cost_rate"]),
            "",
            "## Variant Results",
            "",
            self.table(report.get("variant_results", []), ["year", "variant", "net_profit", "PF", "DD", "win_rate", "total_trades", "final_assets", "capital_utilization", "average_holding_days", "median_holding_days", "same_code_reentry_count", "reentry_within_5_days_count", "cost_paid", "exit_reason_counts"]),
            "",
            "## Guard Effectiveness",
            "",
            self.table(report.get("guard_effectiveness", []), ["year", "variant", "reentry_within_5_days_delta_vs_h0", "net_profit_delta_vs_h0", "PF_delta_vs_h0", "DD_delta_vs_h0", "passes_basic_thresholds"]),
            "",
            "## Strict Walk-Forward OOS Design",
            "",
            self.table([report.get("strict_walk_forward_oos_design", {})], ["why_2024_is_not_strict_oos", "recommended_first_strict_oos_year", "estimated_cost_risk", "phase11h_retraining_executed", "phase11h_walk_forward_prediction_regenerated"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_evaluation", "backtest_columns_used_as_features", "trade_result_columns_used_as_features", "cash_or_portfolio_columns_used_as_model_features", "selected_or_bought_used_as_features", "current_pm_multiplier_used", "historical_predictions_regenerated", "profile_changed", "full_backtest_executed", "walk_forward_retraining_executed", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["cooldown_minhold_passed", "passing_variants", "recommended_next_phase", "reason"]),
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
