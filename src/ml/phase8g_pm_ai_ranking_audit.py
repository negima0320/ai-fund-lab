"""Phase 8-G PM AI ranking / relative allocation audit.

This audit is read-only. It checks whether Portfolio Manager AI should be
treated as a same-day candidate ranking and capital allocation problem rather
than a standalone single-name classifier.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase8g_pm_ai_ranking_audit_2023-01_to_2026-05"
PERIOD = "2023-01-01_to_2026-05-31"

RUN_SOURCES = {
    "v2_82_current_pm_cap38": {
        "trades": Path("reports/final/v2_82_cap38/core_2023-01_to_2026-05/trades.csv"),
        "purchase_audit": Path("reports/final/v2_82_cap38/core_2023-01_to_2026-05/purchase_audit.csv"),
    },
    "v2_90_pm_ai_v2_api_only_cap38": {
        "trades": Path("logs/backtests/rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38/2023-01-01_to_2026-05-31/trades.csv"),
        "purchase_audit": Path("logs/backtests/rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38/2023-01-01_to_2026-05-31/purchase_audit.csv"),
    },
    "v2_91_pm_ai_v2_calibrated_rule_e_cap38": {
        "trades": Path("logs/backtests/rookie_dealer_02_v2_91_pm_ai_v2_calibrated_rule_e_cap38/2023-01-01_to_2026-05-31/trades.csv"),
        "purchase_audit": Path("logs/backtests/rookie_dealer_02_v2_91_pm_ai_v2_calibrated_rule_e_cap38/2023-01-01_to_2026-05-31/purchase_audit.csv"),
    },
}
PM_API_ONLY_DATASET = Path("data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05.parquet")
CURRENT_PM_DIR = Path("models/ml/portfolio_manager/current_v2_73_phase3b_clean")
CANDIDATE_PM_DIR = Path("models/ml/portfolio_manager/candidate_v2_api_only")

RELATIVE_SCORE_COLUMNS = [
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "risk_adjusted_score",
    "entry_score",
    "total_score",
    "volume_ratio",
]
LOWER_IS_BETTER_COLUMNS = {"bad_entry_probability_10d"}


@dataclass(frozen=True)
class Phase8GReportPaths:
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


def _safe_sum(series: pd.Series | None) -> float:
    values = _numeric(series).dropna()
    return float(values.sum()) if not values.empty else 0.0


def _safe_mean(series: pd.Series | None) -> float | None:
    values = _numeric(series).dropna()
    return None if values.empty else float(values.mean())


def _profit_factor(profits: pd.Series | None) -> float | None:
    values = _numeric(profits).dropna()
    if values.empty:
        return None
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(-values[values < 0].sum())
    if gross_loss == 0:
        return None if gross_profit == 0 else float("inf")
    return gross_profit / gross_loss


def _quantiles(series: pd.Series | None) -> dict[str, float | None]:
    values = _numeric(series).dropna()
    if values.empty:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "mean": None}
    return {
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "mean": float(values.mean()),
    }


class Phase8GPMAIRankingAudit:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)

    def build_report(self) -> dict[str, Any]:
        payloads = self._load_payloads()
        daily_summary = self._daily_candidate_summary(payloads)
        v282 = payloads["v2_82_current_pm_cap38"]
        pm130_ranks = self._relative_rank_summary(v282["purchase_audit"], 1.30)
        pm080_ranks = self._relative_rank_summary(v282["purchase_audit"], 0.80)
        v2_failure = self._pm_ai_v2_failure_ranking(payloads)
        label_candidates = self._ranking_label_candidates()
        rule_candidates = self._rule_based_allocator_candidates(pm130_ranks, pm080_ranks)
        api_only = self._relative_feature_api_only_assessment()
        verdict = self._verdict(v2_failure, api_only, rule_candidates)
        return {
            "metadata": {
                "phase": "8-G",
                "audit_only": True,
                "training_executed": False,
                "backtest_executed": False,
                "profile_added": False,
                "current_model_overwritten": False,
                "live_order_executed": False,
                "period": PERIOD,
            },
            "sources": self._sources(),
            "problem_definition": self._problem_definition(),
            "daily_candidate_summary": daily_summary,
            "current_pm130_relative_rank_audit": pm130_ranks,
            "current_pm080_relative_rank_audit": pm080_ranks,
            "pm_ai_v2_failure_ranking_analysis": v2_failure,
            "ranking_label_candidates": label_candidates,
            "rule_based_relative_allocator_candidates": rule_candidates,
            "relative_feature_api_only_assessment": api_only,
            "verdict": verdict,
        }

    def save_report(self, report: dict[str, Any]) -> Phase8GReportPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase8GReportPaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        daily_overview = report["daily_candidate_summary"]["overview"]
        pm130 = report["current_pm130_relative_rank_audit"]["score_summaries"]
        pm080 = report["current_pm080_relative_rank_audit"]["score_summaries"]
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 8-G PM AI Ranking / Relative Allocation Audit",
                "",
                "## Scope",
                "",
                "- audit only",
                "- no retraining, no backtest, no profile addition, no current model overwrite, no live order",
                "",
                "## Problem Definition",
                "",
                self._table([report["problem_definition"]], ["pm_problem_type_recommended", "is_single_name_classifier", "is_candidate_ranking_problem", "is_capital_allocation_problem", "is_risk_management_problem", "reason"]),
                "",
                "## Daily Candidate Summary",
                "",
                self._table(daily_overview, ["run", "days", "avg_candidate_count", "avg_selected_count", "avg_bought_count", "avg_affordable_count", "total_planned_buy_amount", "total_actual_buy_amount"]),
                "",
                "## Current PM 1.30 Relative Rank",
                "",
                self._table(pm130, ["score_column", "available", "target_count", "rank_p50", "percentile_p50", "gap_to_best_p50", "relative_top_candidate_rate"]),
                "",
                "## Current PM 0.80 Relative Rank",
                "",
                self._table(pm080, ["score_column", "available", "target_count", "rank_p50", "percentile_p50", "gap_to_best_p50", "relative_low_candidate_rate"]),
                "",
                "## PM AI v2 Failure From Ranking View",
                "",
                self._table(report["pm_ai_v2_failure_ranking_analysis"]["run_summaries"], ["run", "pm130_trade_count", "pm130_profit", "pm130_pf", "relative_quality", "wrong_pm130_reason"]),
                "",
                "## Ranking Label Candidates",
                "",
                self._table(report["ranking_label_candidates"], ["label", "definition", "api_only_feasibility", "leakage_risk", "pm130_reproducibility", "expected_implementation_cost", "expected_backtest_value"]),
                "",
                "## Rule-Based Relative Allocator Candidates",
                "",
                self._table(report["rule_based_relative_allocator_candidates"], ["rule", "definition", "expected_profit_direction", "expected_risk_direction", "candidate_for_phase8h_backtest"]),
                "",
                "## API-Only Relative Features",
                "",
                self._table([report["relative_feature_api_only_assessment"]], ["relative_features_api_only_feasible", "allowed_relative_features", "forbidden_relative_features", "reason"]),
                "",
                "## Verdict",
                "",
                self._table([report["verdict"]], ["pm_ai_v2_failure_reason", "ranking_problem_supported", "relative_allocation_worth_testing", "best_next_approach"]),
                "",
            ]
        )

    def _sources(self) -> dict[str, Any]:
        return {
            "runs": {run: {kind: str(self.root / path) for kind, path in paths.items()} for run, paths in RUN_SOURCES.items()},
            "pm_api_only_dataset": str(self.root / PM_API_ONLY_DATASET),
            "current_pm_model": str(self.root / CURRENT_PM_DIR),
            "candidate_pm_model": str(self.root / CANDIDATE_PM_DIR),
        }

    def _load_payloads(self) -> dict[str, dict[str, pd.DataFrame]]:
        payloads: dict[str, dict[str, pd.DataFrame]] = {}
        for run, paths in RUN_SOURCES.items():
            trades = _read_csv(self.root / paths["trades"])
            purchase = _read_csv(self.root / paths["purchase_audit"])
            payloads[run] = {
                "trades": self._normalize_frame(trades),
                "purchase_audit": self._normalize_frame(purchase),
            }
        return payloads

    def _normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        for column in ["code", "decision", "candidate_source", "skip_reason", "action"]:
            if column in out.columns:
                out[column] = out[column].fillna("").astype(str)
        for column in ["entry_date", "signal_date", "trade_date", "exit_date"]:
            if column in out.columns:
                out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
        return out

    def _problem_definition(self) -> dict[str, Any]:
        return {
            "pm_problem_type_recommended": "candidate_ranking_and_relative_capital_allocation",
            "is_single_name_classifier": False,
            "is_candidate_ranking_problem": True,
            "is_capital_allocation_problem": True,
            "is_risk_management_problem": True,
            "reason": "v2_91 restored PM1.30 count but not PM1.30 profit quality, so the missing signal is likely same-day relative priority and sizing, not standalone conviction probability.",
        }

    def _daily_candidate_summary(self, payloads: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Any]:
        daily_rows: list[dict[str, Any]] = []
        overview_rows: list[dict[str, Any]] = []
        for run, payload in payloads.items():
            purchase = payload["purchase_audit"]
            if purchase.empty:
                overview_rows.append(self._empty_daily_overview(run))
                continue
            date_column = self._candidate_date_column(purchase)
            if date_column is None:
                overview_rows.append(self._empty_daily_overview(run))
                continue
            for date_value, group in purchase.groupby(date_column, dropna=True):
                if not date_value or str(date_value) == "NaT":
                    continue
                final_amount = _numeric(group.get("final_amount"))
                planned_amount = _numeric(group.get("planned_amount"))
                decision = group.get("decision", pd.Series("", index=group.index)).fillna("").astype(str).str.upper()
                candidate_source = group.get("candidate_source", pd.Series("", index=group.index)).fillna("").astype(str)
                pm_dist = self._value_counts_dict(_numeric(group.get("pm_multiplier")).round(2))
                daily_rows.append(
                    {
                        "run": run,
                        "date": str(date_value),
                        "candidate_count": int(len(group)),
                        "selected_count": int(candidate_source.str.lower().eq("selected").sum()) if "candidate_source" in group.columns else int(len(group)),
                        "bought_count": int(decision.eq("BUY").sum()),
                        "affordable_count": int(final_amount.gt(0).sum()),
                        "pm_multiplier_distribution": pm_dist,
                        "total_planned_buy_amount": float(planned_amount.sum()) if not planned_amount.empty else 0.0,
                        "total_actual_buy_amount": float(final_amount.sum()) if not final_amount.empty else 0.0,
                    }
                )
            run_rows = [row for row in daily_rows if row["run"] == run]
            overview_rows.append(self._daily_overview(run, run_rows))
        return {
            "reconstruction_source": "purchase_audit.csv grouped by entry_date/trade_date/signal_date",
            "affordable_count_definition": "rows with positive final_amount; this is a realized affordability proxy, not a training feature",
            "overview": overview_rows,
            "rows": daily_rows,
        }

    def _candidate_date_column(self, frame: pd.DataFrame) -> str | None:
        for column in ["entry_date", "trade_date", "signal_date"]:
            if column in frame.columns:
                return column
        return None

    def _daily_overview(self, run: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return self._empty_daily_overview(run)
        frame = pd.DataFrame(rows)
        return {
            "run": run,
            "days": int(len(frame)),
            "avg_candidate_count": float(frame["candidate_count"].mean()),
            "avg_selected_count": float(frame["selected_count"].mean()),
            "avg_bought_count": float(frame["bought_count"].mean()),
            "avg_affordable_count": float(frame["affordable_count"].mean()),
            "total_planned_buy_amount": float(frame["total_planned_buy_amount"].sum()),
            "total_actual_buy_amount": float(frame["total_actual_buy_amount"].sum()),
        }

    def _empty_daily_overview(self, run: str) -> dict[str, Any]:
        return {
            "run": run,
            "days": 0,
            "avg_candidate_count": None,
            "avg_selected_count": None,
            "avg_bought_count": None,
            "avg_affordable_count": None,
            "total_planned_buy_amount": 0.0,
            "total_actual_buy_amount": 0.0,
        }

    def _relative_rank_summary(self, purchase: pd.DataFrame, target_multiplier: float) -> dict[str, Any]:
        details = self._relative_rank_details(purchase, target_multiplier)
        summaries = []
        for column in RELATIVE_SCORE_COLUMNS:
            column_details = [row for row in details if row["score_column"] == column]
            if not column_details:
                summaries.append(
                    {
                        "score_column": column,
                        "available": column in purchase.columns if not purchase.empty else False,
                        "target_count": 0,
                        "rank_p50": None,
                        "percentile_p50": None,
                        "gap_to_best_p50": None,
                        "relative_top_candidate_rate": None,
                        "relative_low_candidate_rate": None,
                    }
                )
                continue
            detail_frame = pd.DataFrame(column_details)
            top_rate = float(detail_frame["is_relative_top_candidate"].mean()) if "is_relative_top_candidate" in detail_frame else None
            low_rate = float(detail_frame["is_relative_low_candidate"].mean()) if "is_relative_low_candidate" in detail_frame else None
            summaries.append(
                {
                    "score_column": column,
                    "available": True,
                    "target_count": int(len(detail_frame)),
                    "rank_p50": self._median(detail_frame.get("rank")),
                    "percentile_p50": self._median(detail_frame.get("relative_percentile")),
                    "gap_to_best_p50": self._median(detail_frame.get("gap_to_best")),
                    "rank_distribution": _quantiles(detail_frame.get("rank")),
                    "percentile_distribution": _quantiles(detail_frame.get("relative_percentile")),
                    "gap_to_best_distribution": _quantiles(detail_frame.get("gap_to_best")),
                    "relative_top_candidate_rate": top_rate,
                    "relative_low_candidate_rate": low_rate,
                }
            )
        return {
            "target_pm_multiplier": target_multiplier,
            "score_columns_requested": RELATIVE_SCORE_COLUMNS,
            "available_score_columns": [column for column in RELATIVE_SCORE_COLUMNS if column in purchase.columns] if not purchase.empty else [],
            "missing_score_columns": [column for column in RELATIVE_SCORE_COLUMNS if column not in purchase.columns] if not purchase.empty else RELATIVE_SCORE_COLUMNS,
            "score_summaries": summaries,
            "detail_rows_sample": details[:200],
        }

    def _relative_rank_details(self, purchase: pd.DataFrame, target_multiplier: float) -> list[dict[str, Any]]:
        if purchase.empty or "pm_multiplier" not in purchase.columns:
            return []
        date_column = self._candidate_date_column(purchase)
        if date_column is None:
            return []
        details: list[dict[str, Any]] = []
        target_mask = _numeric(purchase.get("pm_multiplier")).round(2).eq(round(target_multiplier, 2))
        decision = purchase.get("decision", pd.Series("", index=purchase.index)).fillna("").astype(str).str.upper()
        if "decision" in purchase.columns:
            target_mask &= decision.eq("BUY")
        for date_value, group in purchase.groupby(date_column, dropna=True):
            if len(group) <= 1:
                continue
            target_indices = group.index.intersection(purchase.index[target_mask])
            if target_indices.empty:
                continue
            for column in RELATIVE_SCORE_COLUMNS:
                if column not in group.columns:
                    continue
                values = _numeric(group[column])
                valid = values.dropna()
                if valid.empty:
                    continue
                lower_is_better = column in LOWER_IS_BETTER_COLUMNS
                ranks = values.rank(method="min", ascending=lower_is_better)
                best_value = float(valid.min() if lower_is_better else valid.max())
                valid_count = int(valid.count())
                top_cutoff = max(1, math.ceil(valid_count * 0.10))
                low_cutoff = max(1, math.ceil(valid_count * 0.25))
                for idx in target_indices:
                    value = values.loc[idx]
                    rank = ranks.loc[idx]
                    if pd.isna(value) or pd.isna(rank):
                        continue
                    relative_percentile = 1.0 if valid_count <= 1 else 1.0 - ((float(rank) - 1.0) / float(valid_count - 1))
                    gap_to_best = (float(value) - best_value) if lower_is_better else (best_value - float(value))
                    row = group.loc[idx]
                    details.append(
                        {
                            "date": str(date_value),
                            "code": str(row.get("code", "")),
                            "pm_multiplier": float(target_multiplier),
                            "score_column": column,
                            "score_value": float(value),
                            "candidate_count": int(len(group)),
                            "valid_score_count": valid_count,
                            "rank": float(rank),
                            "relative_percentile": float(relative_percentile),
                            "gap_to_best": float(gap_to_best),
                            "is_relative_top_candidate": bool(float(rank) <= top_cutoff),
                            "is_relative_low_candidate": bool(float(rank) >= max(1, valid_count - low_cutoff + 1)),
                        }
                    )
        return details

    def _pm_ai_v2_failure_ranking(self, payloads: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Any]:
        run_summaries = []
        for run in [
            "v2_82_current_pm_cap38",
            "v2_90_pm_ai_v2_api_only_cap38",
            "v2_91_pm_ai_v2_calibrated_rule_e_cap38",
        ]:
            trades = payloads[run]["trades"]
            purchase = payloads[run]["purchase_audit"]
            pm130_profit = self._pm_profit_summary(trades, 1.30)
            rank_summary = self._relative_rank_summary(purchase, 1.30)
            quality = self._relative_quality_label(rank_summary)
            wrong_reason = self._wrong_pm130_reason(run, pm130_profit, quality)
            run_summaries.append(
                {
                    "run": run,
                    "pm130_trade_count": pm130_profit["trade_count"],
                    "pm130_profit": pm130_profit["net_profit"],
                    "pm130_pf": pm130_profit["profit_factor"],
                    "relative_quality": quality,
                    "wrong_pm130_reason": wrong_reason,
                    "available_score_columns": ",".join(rank_summary["available_score_columns"]),
                }
            )
        return {
            "run_summaries": run_summaries,
            "v2_90_pm130_relative_quality": next(row for row in run_summaries if row["run"].startswith("v2_90"))["relative_quality"],
            "v2_91_pm130_relative_quality": next(row for row in run_summaries if row["run"].startswith("v2_91"))["relative_quality"],
            "wrong_pm130_reason": "; ".join(row["wrong_pm130_reason"] for row in run_summaries if row["wrong_pm130_reason"]),
        }

    def _pm_profit_summary(self, trades: pd.DataFrame, multiplier: float) -> dict[str, Any]:
        if trades.empty or "pm_multiplier" not in trades.columns:
            return {"trade_count": 0, "net_profit": 0.0, "profit_factor": None}
        frame = trades.copy()
        if "action" in frame.columns:
            frame = frame[frame["action"].fillna("").astype(str).str.upper().eq("SELL")]
        group = frame[_numeric(frame.get("pm_multiplier")).round(2).eq(round(multiplier, 2))]
        profit = group.get("net_profit", group.get("profit"))
        return {
            "trade_count": int(len(group)),
            "net_profit": _safe_sum(profit),
            "profit_factor": _profit_factor(profit),
        }

    def _relative_quality_label(self, rank_summary: dict[str, Any]) -> str:
        summaries = [row for row in rank_summary["score_summaries"] if row.get("available") and row.get("target_count", 0) > 0]
        if not summaries:
            return "no_pm130_or_no_score_columns"
        top_rates = [row.get("relative_top_candidate_rate") for row in summaries if row.get("relative_top_candidate_rate") is not None]
        median_rate = float(pd.Series(top_rates).median()) if top_rates else 0.0
        if median_rate >= 0.35:
            return "strong_relative_top_alignment"
        if median_rate >= 0.15:
            return "mixed_relative_alignment"
        return "weak_relative_alignment"

    def _wrong_pm130_reason(self, run: str, profit: dict[str, Any], quality: str) -> str:
        if profit["trade_count"] == 0:
            return "PM1.30 was not generated, so capital allocation became too conservative."
        if run.startswith("v2_91") and (profit["net_profit"] or 0.0) < 500000:
            return "PM1.30 count recovered but profit quality did not; calibrated scores did not recover same-day relative winners."
        if quality == "weak_relative_alignment":
            return "PM1.30 names are not consistently top-ranked within their same-day candidate set."
        return ""

    def _ranking_label_candidates(self) -> list[dict[str, Any]]:
        return [
            {
                "label": "Label A",
                "definition": "same-day candidate top 10% by future_10d_return",
                "api_only_feasibility": "conditional_yes",
                "leakage_risk": "low_if_future_return_is_label_only",
                "pm130_reproducibility": "medium",
                "expected_implementation_cost": "medium",
                "expected_backtest_value": "medium_to_high",
            },
            {
                "label": "Label B",
                "definition": "same-day candidate top 10% by risk_adjusted_future_return",
                "api_only_feasibility": "conditional_yes",
                "leakage_risk": "low_if_label_is_out_of_feature_set",
                "pm130_reproducibility": "high_potential",
                "expected_implementation_cost": "medium",
                "expected_backtest_value": "high",
            },
            {
                "label": "Label C",
                "definition": "same-day candidate top 10% by future_return_drawdown_ratio",
                "api_only_feasibility": "conditional_yes",
                "leakage_risk": "low_if_drawdown_is_label_only",
                "pm130_reproducibility": "high_potential",
                "expected_implementation_cost": "medium_high",
                "expected_backtest_value": "high",
            },
            {
                "label": "Label D",
                "definition": "same-day candidate top 10% by API-derived trade_quality_score",
                "api_only_feasibility": "conditional_yes",
                "leakage_risk": "medium_until_score_formula_is_frozen",
                "pm130_reproducibility": "high_potential",
                "expected_implementation_cost": "medium_high",
                "expected_backtest_value": "high",
            },
            {
                "label": "Label E",
                "definition": "top-k allocation label for candidates that deserve larger sizing",
                "api_only_feasibility": "conditional_yes",
                "leakage_risk": "low_if_utility_is_api_future_label_not_backtest_result",
                "pm130_reproducibility": "high_potential",
                "expected_implementation_cost": "high",
                "expected_backtest_value": "high",
            },
            {
                "label": "Label F",
                "definition": "pairwise same-day candidate ranking label",
                "api_only_feasibility": "conditional_yes",
                "leakage_risk": "low_if_pairs_use_future_labels_only",
                "pm130_reproducibility": "high_potential",
                "expected_implementation_cost": "high",
                "expected_backtest_value": "high_but_expensive",
            },
        ]

    def _rule_based_allocator_candidates(
        self,
        pm130_ranks: dict[str, Any],
        pm080_ranks: dict[str, Any],
    ) -> list[dict[str, Any]]:
        available = set(pm130_ranks.get("available_score_columns", []))
        rows = [
            ("risk_adjusted_score_top10", "risk_adjusted_score same-day top 10% -> PM 1.30; top 25% -> PM 1.15", "up_if_current_pm130_aligns", "medium", "risk_adjusted_score" in available),
            ("expected_return_10d_top10", "expected_return_10d same-day top 10% -> PM 1.30", "medium_up", "medium", "expected_return_10d" in available),
            ("expected_max_return_20d_top10", "expected_max_return_20d same-day top 10% -> PM 1.30", "unknown_until_column_available", "medium", "expected_max_return_20d" in available),
            ("bad_entry_bottom_allocator", "bad_entry_probability_10d high percentile -> PM 0.80", "protective", "low_to_medium", "bad_entry_probability_10d" in available),
            ("relative_rank_blend", "blend risk_adjusted_score/expected_return/bad_entry rank percentiles into PM multiplier", "up", "medium", bool(available.intersection({"risk_adjusted_score", "expected_return_10d", "bad_entry_probability_10d"}))),
        ]
        return [
            {
                "rule": rule,
                "definition": definition,
                "expected_profit_direction": profit_direction,
                "expected_risk_direction": risk_direction,
                "candidate_for_phase8h_backtest": bool(candidate),
            }
            for rule, definition, profit_direction, risk_direction, candidate in rows
        ]

    def _relative_feature_api_only_assessment(self) -> dict[str, Any]:
        return {
            "relative_features_api_only_feasible": True,
            "candidate_count_in_day_is_forbidden": False,
            "rank_in_day_operationally_reproducible": True,
            "percentile_in_day_api_only_allowed": True,
            "gap_to_best_api_only_allowed": True,
            "selected_count_in_day_forbidden": True,
            "allowed_relative_features": "candidate_count_in_day,rank_in_day,score_rank_in_day,percentile_in_day,gap_to_best,candidate_strength",
            "forbidden_relative_features": "selected_count_in_day,bought_count_in_day,affordable_count_in_day,cash_after,portfolio_state,position_state,backtest_outcome",
            "reason": "Relative features are API-only only when computed from the prediction-time candidate pool before cash/portfolio decisions; selected/bought/affordable counts remain policy/backtest dependent.",
        }

    def _verdict(
        self,
        v2_failure: dict[str, Any],
        api_only: dict[str, Any],
        rule_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        phase8h_candidates = [row for row in rule_candidates if row["candidate_for_phase8h_backtest"]]
        return {
            "pm_ai_v2_failure_reason": "single_name_classifier_and_calibration_do_not_restore_same_day_relative_winner_quality",
            "ranking_problem_supported": True,
            "relative_allocation_worth_testing": bool(api_only["relative_features_api_only_feasible"] and phase8h_candidates),
            "best_next_approach": "Phase 8-H Rule-Based Relative Allocator Backtest",
            "next_alternative": "Phase 8-H PM AI Ranking Dataset Design",
        }

    def _value_counts_dict(self, series: pd.Series) -> dict[str, int]:
        values = series.dropna()
        counts = values.value_counts().sort_index()
        return {str(key): int(value) for key, value in counts.items()}

    def _median(self, series: pd.Series | None) -> float | None:
        values = _numeric(series).dropna()
        return None if values.empty else float(values.median())

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(self._format(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if math.isinf(value):
                return "inf"
            return f"{value:.4f}"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, dict):
            return ", ".join(f"{key}:{val}" for key, val in value.items())
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        return str(value).replace("|", "\\|")


def build_and_save(root: Path | str = ROOT) -> Phase8GReportPaths:
    audit = Phase8GPMAIRankingAudit(root)
    report = audit.build_report()
    return audit.save_report(report)
