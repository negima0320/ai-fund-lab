"""Phase 12-B3 exit / hold decision audit.

This 2025-only audit replays S3a dynamic raw weight and S2 opportunity E4
trades, then inspects post-exit and pre-exit price paths. It is an audit, not a
new strategy backtest or variant search. Future label columns are used only as
trade quality context.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase12a_dynamic_capital_allocation import EVAL_COLUMNS
from ml.phase12b2_allocation_execution_adjustment import Phase12B2Options, StrategySpec
from ml.phase12b_limited_allocation_strategy_check import ARTIFACT_PATH, BASELINE_RANK_COLUMNS, END_DATE, ROUND_LOT, START_DATE


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase12b3_exit_hold_audit_2025"
FUTURE_EVAL_COLUMNS = EVAL_COLUMNS
AUDIT_STRATEGIES = [
    StrategySpec("S2_opportunity_top5_E4", rank_column="opportunity_proba", exit_guard=True),
    StrategySpec("S3a_dynamic_raw_weight", rank_column="opportunity_proba", exit_guard=True, allocation_mode="dynamic_raw"),
]
OPPORTUNITY_EXIT_REASONS = {"opportunity_proba_drop", "opportunity_rank_below_floor"}


@dataclass(frozen=True)
class Phase12B3Options(Phase12B2Options):
    pass


@dataclass(frozen=True)
class Phase12B3Paths:
    markdown: Path
    json: Path


class Phase12B3ExitHoldAudit:
    def __init__(self, root: Path | str = ROOT, *, options: Phase12B3Options | None = None) -> None:
        self.root = Path(root)
        self.options = options or Phase12B3Options()

    def run(self) -> Phase12B3Paths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data = self.load_frame()
        leakage = self.leakage_checklist()
        if leakage["blocking_issues"]:
            return {
                "metadata": self.metadata(),
                "conditions": self.conditions(),
                "dataset_summary": self.dataset_summary(data),
                "leakage_checklist": leakage,
                "recommendation": self.recommendation([], [], [], leakage),
            }

        trades_by_strategy: dict[str, pd.DataFrame] = {}
        for spec in AUDIT_STRATEGIES:
            trades_by_strategy[spec.name] = self.simulate(data, spec)

        audited = [self.audit_trades(name, trades, data) for name, trades in trades_by_strategy.items()]
        all_audits = pd.concat(audited, ignore_index=True) if audited else pd.DataFrame()
        early_exit = self.early_exit_audit(all_audits)
        late_exit = self.late_exit_audit(all_audits)
        opportunity_exit = self.opportunity_exit_quality(all_audits)
        hold_candidate = self.hold_candidate_audit(all_audits)
        exit_reason_ranking = self.exit_reason_ranking(all_audits)
        recommendation = self.recommendation(early_exit, late_exit, opportunity_exit, leakage)
        return {
            "metadata": self.metadata(),
            "conditions": self.conditions(),
            "dataset_summary": self.dataset_summary(data),
            "trade_summary": self.trade_summary(all_audits),
            "early_exit_audit": early_exit,
            "late_exit_audit": late_exit,
            "opportunity_exit_quality": opportunity_exit,
            "hold_candidate_audit": hold_candidate,
            "exit_reason_ranking": exit_reason_ranking,
            "leakage_checklist": leakage,
            "recommendation": recommendation,
        }

    def load_frame(self) -> pd.DataFrame:
        data = pd.read_parquet(self.root / ARTIFACT_PATH)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        data = data[(data["date"] >= START_DATE) & (data["date"] <= END_DATE)].copy()
        columns = [
            "date",
            "code",
            "close",
            "turnover_value",
            *BASELINE_RANK_COLUMNS,
            "opportunity_proba",
            "downside_bad_proba",
            "opportunity_rank_percentile",
            "downside_rank_percentile",
            "confidence",
            *FUTURE_EVAL_COLUMNS,
        ]
        available = [column for column in columns if column in data.columns]
        data = data[available].drop_duplicates(["date", "code"])
        for column in data.columns:
            if column not in {"date", "code"}:
                data[column] = _numeric(data[column])
        data["baseline_rank_score"] = self.baseline_rank(data)
        data["opportunity_top_decile_proba"] = data["opportunity_proba"]
        data["opportunity_score_proba_rank"] = data["opportunity_rank_percentile"]
        data["a3_3_allocation_weight"] = self.a3_3_weight(data["downside_rank_percentile"])
        return data.dropna(subset=["date", "code", "close", "opportunity_proba", "downside_rank_percentile"]).sort_values(["date", "code"]).reset_index(drop=True)

    def baseline_rank(self, data: pd.DataFrame) -> pd.Series:
        for column in BASELINE_RANK_COLUMNS:
            if column in data.columns and not data[column].isna().all():
                return _numeric(data[column]).fillna(-10**18)
        return pd.Series(-10**18, index=data.index, dtype=float)

    def a3_3_weight(self, downside_rank: pd.Series) -> pd.Series:
        rank = _numeric(downside_rank)
        return rank.map(lambda value: 1.0 if value <= 0.40 else 0.6 if value <= 0.70 else 0.3 if value <= 0.85 else 0.0)

    def simulate(self, data: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
        cash = self.options.initial_cash
        positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
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
                else:
                    if current_row is not None:
                        position["last_close"] = float(current_row["close"])
                    still_open.append(position)
            positions = still_open

            slots = max(0, self.options.max_positions - len(positions))
            ranked = current_rank_frame.sort_values([spec.rank_column, "turnover_value", "code"], ascending=[False, False, True]).head(self.options.max_positions)
            if spec.allocation_mode != "equal":
                ranked = ranked.copy()
                ranked = ranked[_numeric(ranked["a3_3_allocation_weight"]) > 0].copy()
            selected = ranked.head(slots) if slots else ranked.iloc[0:0]
            target_amounts = self.target_amounts(selected, spec, cash)
            for index, row in selected.iterrows():
                target_amount = target_amounts.get(index, 0.0)
                lot_cost = float(row["close"]) * self.options.round_lot
                lots = int(target_amount // (lot_cost * (1.0 + self.options.cost_rate))) if lot_cost > 0 else 0
                buy_amount = lots * lot_cost
                buy_cost = buy_amount * self.options.cost_rate
                cash_out = buy_amount + buy_cost
                if lots <= 0 or cash_out > cash:
                    continue
                cash -= cash_out
                positions.append(
                    {
                        "entry_date": current_date,
                        "due_date": current_date + pd.offsets.BDay(self.options.holding_days),
                        "strategy": spec.name,
                        "code": str(row["code"]),
                        "buy_amount": buy_amount,
                        "buy_cost": buy_cost,
                        "target_buy_amount": target_amount,
                        "lot_count": lots,
                        "entry_close": float(row["close"]),
                        "last_close": float(row["close"]),
                        "entry_opportunity_top_decile_proba": _safe_float(row.get("opportunity_top_decile_proba")),
                        "allocation_weight": _safe_float(row.get("a3_3_allocation_weight")) if spec.allocation_mode != "equal" else 1.0,
                        **{column: _safe_float(row.get(column)) for column in FUTURE_EVAL_COLUMNS},
                    }
                )

        if dates:
            last_date = dates[-1]
            for position in positions:
                trade = self.close_position(position, last_date, float(position["last_close"]), "forced_end_of_period", position.get("strategy", "forced_end_of_period"))
                cash += trade["exit_cash_flow"]
                trades.append(trade)
        return pd.DataFrame(trades)

    def target_amounts(self, selected: pd.DataFrame, spec: StrategySpec, cash: float) -> dict[Any, float]:
        if selected.empty:
            return {}
        available_budget = min(cash, self.options.daily_buy_budget)
        if spec.allocation_mode == "dynamic_raw":
            weights = _numeric(selected["a3_3_allocation_weight"]).clip(lower=0)
            return {index: available_budget * float(weight) / self.options.max_positions for index, weight in weights.items()}
        amount = available_budget / max(1, min(self.options.max_positions, len(selected)))
        return {index: amount for index in selected.index}

    def exit_reason(self, position: dict[str, Any], current_date: pd.Timestamp, current_row: pd.Series | None, spec: StrategySpec) -> str | None:
        if current_row is not None and spec.exit_guard:
            observed_return = float(current_row["close"]) / float(position["entry_close"]) - 1.0
            if observed_return <= self.options.stop_loss_rate:
                return "stop_loss"
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
            "entry_close": position["entry_close"],
            "exit_close": exit_close,
            "buy_amount": position["buy_amount"],
            "target_buy_amount": position.get("target_buy_amount"),
            "exit_amount": exit_amount,
            "exit_cash_flow": exit_cash_flow,
            "realized_profit": profit,
            "realized_return": profit / float(position["buy_amount"]) if position["buy_amount"] else None,
            "holding_days": holding_days,
            "exit_reason": reason,
            "cost_paid": total_cost,
            "allocation_weight": position.get("allocation_weight", 1.0),
            **{column: position.get(column) for column in FUTURE_EVAL_COLUMNS},
        }

    def audit_trades(self, strategy: str, trades: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        audited = trades.copy()
        path_by_code = {code: group.sort_values("date")[["date", "close"]] for code, group in data.groupby("code", sort=False)}
        post_rows = []
        for _, trade in audited.iterrows():
            path = path_by_code.get(str(trade["code"]), pd.DataFrame())
            post = self.post_exit_metrics(trade, path)
            pre = self.pre_exit_metrics(trade, path)
            post_rows.append({**post, **pre})
        return pd.concat([audited.reset_index(drop=True), pd.DataFrame(post_rows)], axis=1)

    def post_exit_metrics(self, trade: pd.Series, path: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {}
        exit_date = pd.Timestamp(trade["exit_date"])
        exit_close = _safe_float(trade.get("exit_close"))
        for horizon in [5, 10, 20]:
            end_date = exit_date + pd.offsets.BDay(horizon)
            window = path[(path["date"] > exit_date) & (path["date"] <= end_date)]
            if exit_close is None or window.empty:
                out[f"post_exit_max_return_{horizon}d"] = None
            else:
                out[f"post_exit_max_return_{horizon}d"] = _safe_float(_numeric(window["close"]).max() / exit_close - 1.0)
        return out

    def pre_exit_metrics(self, trade: pd.Series, path: pd.DataFrame) -> dict[str, Any]:
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date"])
        entry_close = _safe_float(trade.get("entry_close"))
        window = path[(path["date"] >= entry_date) & (path["date"] <= exit_date)]
        if entry_close is None or window.empty:
            return {"max_profit_before_exit": None, "profit_decay_before_exit": None}
        max_profit = _safe_float(_numeric(window["close"]).max() / entry_close - 1.0)
        realized = _safe_float(trade.get("realized_return"))
        return {
            "max_profit_before_exit": max_profit,
            "profit_decay_before_exit": _safe_float(max_profit - realized) if max_profit is not None and realized is not None else None,
        }

    def early_exit_audit(self, audited: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for strategy, group in audited.groupby("strategy", sort=True):
            for horizon in [5, 10, 20]:
                column = f"post_exit_max_return_{horizon}d"
                values = _numeric(group[column])
                rows.append(
                    {
                        "strategy": strategy,
                        "horizon": f"{horizon}d",
                        "trade_count": int(len(group)),
                        "avg_post_exit_return": _safe_float(values.mean()),
                        "p90_post_exit_return": _safe_float(values.quantile(0.90)),
                        "count_post_exit_10pct_plus": int((values >= 0.10).sum()),
                        "count_post_exit_20pct_plus": int((values >= 0.20).sum()),
                    }
                )
        return rows

    def late_exit_audit(self, audited: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        candidates = audited[(audited["exit_reason"] == "stop_loss") | (_numeric(audited["realized_return"]) <= -0.08)].copy()
        for strategy, group in candidates.groupby("strategy", sort=True):
            rows.append(
                {
                    "strategy": strategy,
                    "late_exit_trade_count": int(len(group)),
                    "avg_realized_return": _safe_float(_numeric(group["realized_return"]).mean()),
                    "avg_max_profit_before_exit": _safe_float(_numeric(group["max_profit_before_exit"]).mean()),
                    "avg_profit_decay_before_exit": _safe_float(_numeric(group["profit_decay_before_exit"]).mean()),
                    "p90_profit_decay_before_exit": _safe_float(_numeric(group["profit_decay_before_exit"]).quantile(0.90)),
                }
            )
        return rows

    def opportunity_exit_quality(self, audited: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        data = audited[audited["exit_reason"].isin(OPPORTUNITY_EXIT_REASONS)]
        for strategy, group in data.groupby("strategy", sort=True):
            post20 = _numeric(group["post_exit_max_return_20d"])
            rows.append(
                {
                    "strategy": strategy,
                    "opportunity_exit_count": int(len(group)),
                    "avg_realized_return": _safe_float(_numeric(group["realized_return"]).mean()),
                    "avg_post_exit_20d_return": _safe_float(post20.mean()),
                    "p90_post_exit_20d_return": _safe_float(post20.quantile(0.90)),
                    "count_post_exit_10pct_plus": int((post20 >= 0.10).sum()),
                    "opportunity_exit_effective": bool(_safe_float(post20.mean()) is not None and (_safe_float(post20.mean()) or 0.0) <= 0.05),
                }
            )
        return rows

    def hold_candidate_audit(self, audited: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        data = audited[audited["exit_reason"] == "time_exit_20d"]
        for strategy, group in data.groupby("strategy", sort=True):
            post20 = _numeric(group["post_exit_max_return_20d"])
            rows.append(
                {
                    "strategy": strategy,
                    "time_exit_trade_count": int(len(group)),
                    "avg_extra_return_after_20d": _safe_float(post20.mean()),
                    "p90_extra_return_after_20d": _safe_float(post20.quantile(0.90)),
                    "count_extra_10pct_plus": int((post20 >= 0.10).sum()),
                }
            )
        return rows

    def exit_reason_ranking(self, audited: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for (strategy, reason), group in audited.groupby(["strategy", "exit_reason"], sort=True):
            post20 = _numeric(group["post_exit_max_return_20d"])
            realized = _numeric(group["realized_return"])
            rows.append(
                {
                    "strategy": strategy,
                    "exit_reason": reason,
                    "count": int(len(group)),
                    "avg_realized_return": _safe_float(realized.mean()),
                    "win_rate": _safe_float((realized > 0).mean()),
                    "avg_post_exit_return": _safe_float(post20.mean()),
                    "p90_post_exit_return": _safe_float(post20.quantile(0.90)),
                    "count_post_exit_10pct_plus": int((post20 >= 0.10).sum()),
                }
            )
        return rows

    def trade_summary(self, audited: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for strategy, group in audited.groupby("strategy", sort=True):
            rows.append(
                {
                    "strategy": strategy,
                    "trade_count": int(len(group)),
                    "avg_realized_return": _safe_float(_numeric(group["realized_return"]).mean()),
                    "win_rate": _safe_float((_numeric(group["realized_return"]) > 0).mean()),
                    "exit_reason_counts": dict(Counter(group["exit_reason"])),
                }
            )
        return rows

    def recommendation(self, early: list[dict[str, Any]], late: list[dict[str, Any]], opportunity: list[dict[str, Any]], leakage: dict[str, Any]) -> dict[str, Any]:
        if leakage["blocking_issues"]:
            return {"recommended_next_phase": "Fix Phase12-B3 leakage blockers"}
        target_early = [row for row in early if row["strategy"] == "S3a_dynamic_raw_weight" and row["horizon"] == "20d"]
        target_late = [row for row in late if row["strategy"] == "S3a_dynamic_raw_weight"]
        target_opp = [row for row in opportunity if row["strategy"] == "S3a_dynamic_raw_weight"]
        early_detected = bool(target_early and ((target_early[0].get("avg_post_exit_return") or 0.0) >= 0.05 or int(target_early[0].get("count_post_exit_10pct_plus") or 0) >= 5))
        late_detected = bool(target_late and ((target_late[0].get("avg_profit_decay_before_exit") or 0.0) >= 0.08))
        opp_effective = bool(target_opp and target_opp[0].get("opportunity_exit_effective"))
        if early_detected and not late_detected:
            main = "early_exit"
            improvement = "Relax Opportunity Exit or test hold-extension after high-quality dynamic entries."
            next_phase = "Phase12-B4 hold_extension_test"
        elif late_detected:
            main = "late_exit"
            improvement = "Prototype trailing exit or faster decay guard before stop-loss."
            next_phase = "Phase12-B4 trailing_exit_prototype"
        elif not opp_effective:
            main = "opportunity_exit_quality_unclear"
            improvement = "Recalibrate Opportunity Exit thresholds before increasing allocation."
            next_phase = "Phase12-B4 opportunity_exit_recalibration"
        else:
            main = "no_major_exit_problem_detected"
            improvement = "Proceed carefully to combine dynamic allocation with existing exit logic."
            next_phase = "Phase12-C dynamic allocation + improved exit"
        return {
            "main_exit_problem": main,
            "early_exit_detected": early_detected,
            "late_exit_detected": late_detected,
            "opportunity_exit_effective": opp_effective,
            "recommended_exit_improvement": improvement,
            "recommended_next_phase": next_phase,
        }

    def dataset_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(data)),
            "unique_codes": int(data["code"].nunique()) if not data.empty else 0,
            "candidate_days": int(data["date"].nunique()) if not data.empty else 0,
            "date_range": {
                "min": data["date"].min().date().isoformat() if not data.empty else None,
                "max": data["date"].max().date().isoformat() if not data.empty else None,
            },
            "source_artifact": str(self.root / ARTIFACT_PATH),
        }

    def conditions(self) -> dict[str, Any]:
        return {
            "period": {"start": START_DATE, "end": END_DATE},
            "audit_strategies": [spec.__dict__ for spec in AUDIT_STRATEGIES],
            "post_exit_horizons": ["5d", "10d", "20d"],
            "primary_strategy": "S3a_dynamic_raw_weight",
            "comparison_strategy": "S2_opportunity_top5_E4",
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "12-B3",
            "scope": "2025 exit / hold decision audit only",
            "full_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_added": False,
            "profile_modified": False,
            "historical_predictions_regenerated": False,
            "openai_api_called": False,
            "jquants_api_refetched": False,
        }

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_only_for_audit": FUTURE_EVAL_COLUMNS,
            "future_columns_used_as_features": [],
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase12B3Paths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase12B3Paths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Phase 12-B3 Exit / Hold Decision Audit",
            "",
            "## Recommendation",
            "",
            self.table([report["recommendation"]], ["main_exit_problem", "early_exit_detected", "late_exit_detected", "opportunity_exit_effective", "recommended_exit_improvement", "recommended_next_phase"]),
            "",
            "## Trade Summary",
            "",
            self.table(report.get("trade_summary", []), ["strategy", "trade_count", "avg_realized_return", "win_rate", "exit_reason_counts"]),
            "",
            "## Early Exit Audit",
            "",
            self.table(report.get("early_exit_audit", []), ["strategy", "horizon", "trade_count", "avg_post_exit_return", "p90_post_exit_return", "count_post_exit_10pct_plus", "count_post_exit_20pct_plus"]),
            "",
            "## Late Exit Audit",
            "",
            self.table(report.get("late_exit_audit", []), ["strategy", "late_exit_trade_count", "avg_realized_return", "avg_max_profit_before_exit", "avg_profit_decay_before_exit", "p90_profit_decay_before_exit"]),
            "",
            "## Opportunity Exit Quality",
            "",
            self.table(report.get("opportunity_exit_quality", []), ["strategy", "opportunity_exit_count", "avg_realized_return", "avg_post_exit_20d_return", "p90_post_exit_20d_return", "count_post_exit_10pct_plus", "opportunity_exit_effective"]),
            "",
            "## Hold Candidate Audit",
            "",
            self.table(report.get("hold_candidate_audit", []), ["strategy", "time_exit_trade_count", "avg_extra_return_after_20d", "p90_extra_return_after_20d", "count_extra_10pct_plus"]),
            "",
            "## Exit Reason Ranking",
            "",
            self.table(report.get("exit_reason_ranking", []), ["strategy", "exit_reason", "count", "avg_realized_return", "win_rate", "avg_post_exit_return", "p90_post_exit_return", "count_post_exit_10pct_plus"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_only_for_audit", "future_columns_used_as_features", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "leakage_risk", "blocking_issues"]),
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
