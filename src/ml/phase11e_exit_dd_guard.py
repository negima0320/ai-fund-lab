"""Phase 11-E limited exit / DD guard integration.

This is a 2025-only lightweight exit experiment on top of the Phase 11-D
Valuation top5 candidate. It uses observed close/proba snapshots from the
existing Phase 11-A/C artifacts for sequential exit decisions. It does not use
future lows for stop-loss decisions and does not run a full-period strategy
backtest or modify profiles.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SIMULATION_PATH = Path("data/ml/valuation_engine/phase11c_allocation_simulation_2025.parquet")
DATASET_PATH = Path("data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet")
PHASE11D_REPORT_PATH = Path("reports/ml/phase11d_combined_backtest_2025.json")
REPORT_STEM = "phase11e_exit_dd_guard_2025"

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
BASE_RULE = "equal_weight_top5"
ROUND_LOT = 100
FUTURE_EVAL_COLUMNS = [
    "future_return_20d",
    "future_max_return_20d",
    "future_max_drawdown_20d",
    "opportunity_value_20d",
    "opportunity_top_decile_20d",
]


@dataclass(frozen=True)
class Phase11EOptions:
    initial_cash: float = 1_000_000.0
    daily_buy_budget: float = 900_000.0
    max_positions: int = 5
    round_lot: int = ROUND_LOT
    holding_days: int = 20
    opportunity_drop_threshold: float = 0.15
    opportunity_rank_floor: float = 0.50


@dataclass(frozen=True)
class ExitVariant:
    name: str
    stop_loss_rate: float | None = None
    opportunity_exit: bool = False


@dataclass(frozen=True)
class Phase11EPaths:
    markdown: Path
    json: Path


VARIANTS = [
    ExitVariant("E0_no_guard"),
    ExitVariant("E1_stop_loss_8pct", stop_loss_rate=-0.08),
    ExitVariant("E2_stop_loss_5pct", stop_loss_rate=-0.05),
    ExitVariant("E3_opportunity_disappeared", opportunity_exit=True),
    ExitVariant("E4_stop_loss_8pct_plus_opportunity", stop_loss_rate=-0.08, opportunity_exit=True),
]


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


class Phase11EExitDDGuard:
    def __init__(self, root: Path | str = ROOT, *, options: Phase11EOptions | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase11EOptions()

    def run(self) -> Phase11EPaths:
        report = self.build_report()
        return self.save_report(report)

    def build_report(self) -> dict[str, Any]:
        data = self.load_frame()
        leakage = self.leakage_checklist()
        phase11d_reference = self.load_phase11d_reference()
        if leakage["blocking_issues"]:
            return {
                "metadata": self.metadata(),
                "conditions": self.conditions(),
                "phase11d_candidate_reference": phase11d_reference,
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], leakage),
            }

        variant_results = []
        trades_by_variant: dict[str, pd.DataFrame] = {}
        for variant in VARIANTS:
            trades, daily = self.simulate_variant(data, variant)
            trades_by_variant[variant.name] = trades
            variant_results.append(self.metrics(variant.name, trades, daily))

        deltas = self.delta_tables(variant_results, phase11d_reference)
        report = {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "phase11d_candidate_reference": phase11d_reference,
            "variant_results": variant_results,
            "delta_vs_phase11d_candidate": deltas["phase11d"],
            "delta_vs_e0_path_no_guard": deltas["e0"],
            "exit_reason_counts": [self.exit_reason_counts(name, trades) for name, trades in trades_by_variant.items()],
            "buy_quality_by_variant": [self.buy_quality(name, trades) for name, trades in trades_by_variant.items()],
            "skipped_variants": [],
            "leakage_checklist": leakage,
            "recommendation": self.recommendation(variant_results, leakage),
        }
        return report

    def load_frame(self) -> pd.DataFrame:
        simulation = pd.read_parquet(self.root / SIMULATION_PATH)
        simulation["date"] = pd.to_datetime(simulation["date"], errors="coerce")
        simulation["code"] = simulation["code"].astype("string")
        simulation = simulation[
            (simulation["rule"] == BASE_RULE)
            & (simulation["date"] >= pd.Timestamp(START_DATE))
            & (simulation["date"] <= pd.Timestamp(END_DATE))
        ].copy()

        keep_from_sim = [
            "date",
            "code",
            "opportunity_top_decile_proba",
            "opportunity_score_proba_rank",
            *FUTURE_EVAL_COLUMNS,
        ]
        dataset = pd.read_parquet(self.root / DATASET_PATH, columns=["date", "code", "close", "turnover_value"])
        dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
        dataset["code"] = dataset["code"].astype("string")
        dataset = dataset[
            (dataset["date"] >= pd.Timestamp(START_DATE))
            & (dataset["date"] <= pd.Timestamp(END_DATE))
        ].drop_duplicates(["date", "code"], keep="last")
        data = simulation[keep_from_sim].merge(dataset, on=["date", "code"], how="left", validate="many_to_one")
        for column in ["close", "turnover_value", "opportunity_top_decile_proba", "opportunity_score_proba_rank", *FUTURE_EVAL_COLUMNS]:
            data[column] = _numeric(data[column])
        return data.dropna(subset=["date", "code", "close"]).sort_values(["date", "code"]).reset_index(drop=True)

    def load_phase11d_reference(self) -> dict[str, Any]:
        path = self.root / PHASE11D_REPORT_PATH
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("strategy_results", []):
            if row.get("strategy") == "candidate_valuation_top5":
                return {"variant": "phase11d_candidate_reference", **row}
        return {}

    def simulate_variant(self, data: pd.DataFrame, variant: ExitVariant) -> tuple[pd.DataFrame, pd.DataFrame]:
        cash = self.options.initial_cash
        positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        daily_rows: list[dict[str, Any]] = []
        dates = list(pd.Series(data["date"].dropna().unique()).sort_values())
        frame_by_date = {date: group.set_index("code", drop=False) for date, group in data.groupby("date", sort=True)}

        for current_date in dates:
            current = frame_by_date[current_date]
            current_rank_frame = current.reset_index(drop=True)
            still_open = []
            for position in positions:
                current_row = current.loc[position["code"]] if position["code"] in current.index else None
                exit_reason = self.exit_reason(position, current_date, current_row, variant)
                if exit_reason:
                    exit_close = float(current_row["close"]) if current_row is not None else float(position["last_close"])
                    trade = self.close_position(position, current_date, exit_close, exit_reason, variant.name)
                    cash += trade["exit_amount"]
                    trades.append(trade)
                else:
                    if current_row is not None:
                        position["last_close"] = float(current_row["close"])
                        position["last_date"] = current_date
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            ranked = current_rank_frame.sort_values(["opportunity_top_decile_proba", "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions)
            selected = ranked.head(slots) if slots else ranked.iloc[0:0]
            raw_amount = min(cash, self.options.daily_buy_budget) / max(1, min(self.options.max_positions, len(ranked)))
            bought_today = 0
            for _, row in selected.iterrows():
                lot_cost = float(row["close"]) * self.options.round_lot
                lots = int(raw_amount // lot_cost) if lot_cost > 0 else 0
                buy_amount = lots * lot_cost
                if lots <= 0 or buy_amount > cash:
                    continue
                cash -= buy_amount
                bought_today += 1
                positions.append(
                    {
                        "entry_date": current_date,
                        "due_date": current_date + pd.offsets.BDay(self.options.holding_days),
                        "code": str(row["code"]),
                        "buy_amount": buy_amount,
                        "lot_count": lots,
                        "entry_close": float(row["close"]),
                        "last_close": float(row["close"]),
                        "last_date": current_date,
                        "entry_opportunity_top_decile_proba": _safe_float(row.get("opportunity_top_decile_proba")),
                        "entry_opportunity_score_proba_rank": _safe_float(row.get("opportunity_score_proba_rank")),
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
                cash += trade["exit_amount"]
                trades.append(trade)
            if daily_rows:
                daily_rows[-1]["total_assets"] = cash
                daily_rows[-1]["marked_position_value"] = 0.0
                daily_rows[-1]["capital_utilization"] = 0.0

        return pd.DataFrame(trades), pd.DataFrame(daily_rows)

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None, variant: ExitVariant) -> str | None:
        if current_row is not None and variant.stop_loss_rate is not None:
            observed_return = float(current_row["close"]) / float(position["entry_close"]) - 1.0
            if observed_return <= variant.stop_loss_rate:
                return "stop_loss"
        if current_row is not None and variant.opportunity_exit:
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

    def close_position(self, position: dict[str, Any], exit_date: pd.Timestamp, exit_close: float, exit_reason: str, variant_name: str) -> dict[str, Any]:
        exit_amount = float(position["lot_count"]) * self.options.round_lot * exit_close
        profit = exit_amount - float(position["buy_amount"])
        holding_days = len(pd.bdate_range(position["entry_date"], exit_date)) - 1
        return {
            "variant": variant_name,
            "entry_date": position["entry_date"],
            "exit_date": exit_date,
            "code": position["code"],
            "buy_amount": position["buy_amount"],
            "exit_amount": exit_amount,
            "entry_close": position["entry_close"],
            "exit_close": exit_close,
            "realized_profit": profit,
            "realized_return": profit / float(position["buy_amount"]) if position["buy_amount"] else None,
            "holding_days": holding_days,
            "exit_reason": exit_reason,
            **{column: position.get(column) for column in FUTURE_EVAL_COLUMNS},
        }

    def metrics(self, variant: str, trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
        profits = _numeric(trades["realized_profit"]) if not trades.empty else pd.Series(dtype=float)
        gross_profit = float(profits[profits > 0].sum()) if not profits.empty else 0.0
        gross_loss = abs(float(profits[profits < 0].sum())) if not profits.empty else 0.0
        equity = _numeric(daily["total_assets"]) if not daily.empty else pd.Series([self.options.initial_cash])
        drawdown = equity / equity.cummax() - 1.0
        return {
            "variant": variant,
            "net_profit": _safe_float(profits.sum()) if not profits.empty else 0.0,
            "PF": _safe_float(gross_profit / gross_loss) if gross_loss else (None if gross_profit == 0 else float("inf")),
            "DD": _safe_float(drawdown.min()) if not drawdown.empty else 0.0,
            "win_rate": _safe_float((profits > 0).mean()) if not profits.empty else None,
            "total_trades": int(len(trades)),
            "final_assets": _safe_float(self.options.initial_cash + profits.sum()) if not profits.empty else self.options.initial_cash,
            "capital_utilization": _safe_float(_numeric(daily["capital_utilization"]).mean()) if not daily.empty else None,
            "average_holding_days": _safe_float(_numeric(trades["holding_days"]).mean()) if not trades.empty else None,
            "exit_reason_counts": dict(Counter(trades["exit_reason"])) if "exit_reason" in trades.columns else {},
        }

    def delta_tables(self, rows: list[dict[str, Any]], phase11d_reference: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        e0 = next((row for row in rows if row["variant"] == "E0_no_guard"), {})
        return {
            "phase11d": [self.delta_row(row, phase11d_reference, "phase11d_candidate_reference") for row in rows],
            "e0": [self.delta_row(row, e0, "E0_no_guard") for row in rows],
        }

    def delta_row(self, row: dict[str, Any], base: dict[str, Any], baseline_name: str) -> dict[str, Any]:
        return {
            "variant": row.get("variant"),
            "baseline": baseline_name,
            "net_profit_delta": self.delta(row.get("net_profit"), base.get("net_profit")),
            "PF_delta": self.delta(row.get("PF"), base.get("PF")),
            "DD_delta": self.delta(row.get("DD"), base.get("DD")),
            "win_rate_delta": self.delta(row.get("win_rate"), base.get("win_rate")),
        }

    def delta(self, value: Any, base: Any) -> float | None:
        value_f = _safe_float(value)
        base_f = _safe_float(base)
        if value_f is None or base_f is None:
            return None
        return value_f - base_f

    def exit_reason_counts(self, variant: str, trades: pd.DataFrame) -> dict[str, Any]:
        return {"variant": variant, **dict(Counter(trades["exit_reason"]))} if "exit_reason" in trades.columns else {"variant": variant}

    def buy_quality(self, variant: str, trades: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {"variant": variant, "buy_count": int(len(trades))}
        for column in FUTURE_EVAL_COLUMNS:
            values = _numeric(trades[column]) if column in trades.columns else pd.Series(dtype=float)
            row[f"{column}_mean"] = _safe_float(values.mean()) if not values.empty else None
        return row

    def dataset_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(data)),
            "unique_codes": int(data["code"].nunique()),
            "candidate_days": int(data["date"].nunique()),
            "date_range": {
                "min": data["date"].min().date().isoformat() if not data.empty else None,
                "max": data["date"].max().date().isoformat() if not data.empty else None,
            },
            "e3_status": "implemented",
            "e3_definition": f"exit when current rank < {self.options.opportunity_rank_floor} or proba drops by >= {self.options.opportunity_drop_threshold}",
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "11-E",
            "limited_2025_only": True,
            "full_backtest_executed": False,
            "profile_added": False,
            "profile_modified": False,
            "current_model_overwritten": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
            "stop_loss_uses_future_low": False,
            "simulation_path": str(self.root / SIMULATION_PATH),
            "dataset_path": str(self.root / DATASET_PATH),
        }

    def conditions(self) -> dict[str, Any]:
        return {
            "period": {"start": START_DATE, "end": END_DATE},
            "initial_cash": self.options.initial_cash,
            "daily_buy_budget": self.options.daily_buy_budget,
            "max_positions": self.options.max_positions,
            "round_lot": self.options.round_lot,
            "holding_days": self.options.holding_days,
            "rank_column": "opportunity_top_decile_proba",
            "candidate_threshold": "top5",
            "variants": [variant.__dict__ for variant in VARIANTS],
        }

    def leakage_checklist(self) -> dict[str, Any]:
        decision_columns = ["opportunity_top_decile_proba", "opportunity_score_proba_rank", "close", "turnover_value"]
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
            "stop_loss_uses_future_low": False,
            "decision_columns": decision_columns,
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def recommendation(self, rows: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"dd_improved_variant_found": False, "recommended_next_phase": "Fix Phase11-E leakage blockers"}
        candidates = [
            row
            for row in rows
            if row.get("variant") != "E0_no_guard"
            and (_safe_float(row.get("DD")) or -1.0) >= -0.10
            and (_safe_float(row.get("PF")) or 0.0) >= 1.5
            and (_safe_float(row.get("net_profit")) or 0.0) >= 100_000
        ]
        best = max(candidates, key=lambda row: (_safe_float(row.get("DD")) or -1.0, _safe_float(row.get("PF")) or 0.0), default=None)
        return {
            "dd_improved_variant_found": best is not None,
            "best_variant": best.get("variant") if best else None,
            "ready_for_next_phase": best is not None,
            "recommended_next_phase": "Phase11-F limited robustness check" if best else "Phase11-E2 risk model or allocation-side DD control",
            "reason": "Pass requires DD within -10%, PF >= 1.5, and net_profit >= 100,000 in the 2025-only research check.",
        }

    def save_report(self, report: dict[str, Any]) -> Phase11EPaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase11EPaths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 11-E Limited Exit / DD Guard 2025",
            "",
            "## Scope",
            "",
            "- 2025 only",
            "- variants: no guard, -8% stop, -5% stop, opportunity exit, stop + opportunity exit",
            "- stop-loss decisions use observed close snapshots only, not future lows",
            "- no full-period backtest, no profile change, no model overwrite",
            "",
            "## Conditions",
            "",
            self.table([report["conditions"]], ["initial_cash", "daily_buy_budget", "max_positions", "round_lot", "holding_days", "rank_column", "candidate_threshold"]),
            "",
            "## Phase 11-D Candidate Reference",
            "",
            self.table([report.get("phase11d_candidate_reference", {})], ["net_profit", "PF", "DD", "win_rate", "total_trades", "final_assets", "capital_utilization"]),
            "",
            "## Variant Results",
            "",
            self.table(report.get("variant_results", []), ["variant", "net_profit", "PF", "DD", "win_rate", "total_trades", "final_assets", "capital_utilization", "average_holding_days", "exit_reason_counts"]),
            "",
            "## Delta vs Phase 11-D Candidate",
            "",
            self.table(report.get("delta_vs_phase11d_candidate", []), ["variant", "net_profit_delta", "PF_delta", "DD_delta", "win_rate_delta"]),
            "",
            "## Exit Reason Counts",
            "",
            self.table(report.get("exit_reason_counts", []), ["variant", "time_exit_20d", "stop_loss", "opportunity_rank_below_floor", "opportunity_proba_drop", "forced_end_of_period"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_evaluation", "backtest_columns_used_as_features", "trade_result_columns_used_as_features", "cash_or_portfolio_columns_used_as_model_features", "selected_or_bought_used_as_features", "current_pm_multiplier_used", "historical_predictions_regenerated", "profile_changed", "full_backtest_executed", "stop_loss_uses_future_low", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["dd_improved_variant_found", "best_variant", "ready_for_next_phase", "recommended_next_phase", "reason"]),
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
