from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_phase3d_detail_audit import PERIOD, PortfolioManagerPhase3DDetailAudit
from ml.portfolio_manager_phase3h_capital_utilization import PHASE3H_PROFILE
from ml.portfolio_manager_phase3j_affordability import PortfolioManagerPhase3JAffordabilityAudit


TARGET_PROFILE = PHASE3H_PROFILE


@dataclass(frozen=True)
class PortfolioManagerPhase3KPaths:
    markdown: Path
    json: Path


class PortfolioManagerPhase3KCandidateRankingAudit:
    """Audit candidate ordering and fallback path from existing logs only."""

    def __init__(
        self,
        root: str | Path = ".",
        target_profile: str = TARGET_PROFILE,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.target_profile = target_profile
        self.period = period
        self.detail = PortfolioManagerPhase3DDetailAudit(root=self.root, period=period)
        self.affordability = PortfolioManagerPhase3JAffordabilityAudit(root=self.root, target_profile=target_profile, period=period)
        self.report_dir = self.root / "reports" / "ml"

    def build(self) -> dict[str, Any]:
        payload = self.affordability._profile_payload(self.target_profile)
        audit = self._candidate_log(payload)
        unaffordable = audit[audit["is_selected_but_not_affordable"]].copy()
        candidate_paths = self._candidate_path_rows(audit, unaffordable)
        fallback_quality = self._fallback_candidate_quality(audit, unaffordable)
        rank_quality = self._rank_quality(audit)
        result = {
            "purpose": "Portfolio Manager AI Phase 3-K buy candidate ranking / fallback path audit",
            "period": self.period,
            "constraints": {
                "api_refetch": False,
                "openai_api": False,
                "historical_predictions_source": "data/ml/walk_forward_predictions/",
                "current_model_historical_regeneration": False,
                "selected_count_in_day_used": False,
                "trading_logic_changed": False,
                "new_profile_added": False,
                "full_backtest_executed": False,
                "live_order_placement": False,
            },
            "profile": self.target_profile,
            "ranking_basis": self._ranking_basis(),
            "candidate_path_classification": self._path_classification(candidate_paths),
            "candidate_path_detail_sample": candidate_paths[:120],
            "fallback_candidate_quality": fallback_quality,
            "rank_quality": rank_quality,
            "fallback_decision_flags": self._decision_flags(candidate_paths, fallback_quality, audit),
            "audit_notes": [
                {
                    "item": "candidate_log_scope",
                    "value": "purchase_audit.csv records selected / skip / buy attempts, not the full scored universe.",
                },
                {
                    "item": "fallback_quality_scope",
                    "value": "Affordable alternatives are inferred from same-day BUY/SCALED_BUY audit rows after unaffordable skips.",
                },
            ],
        }
        return result

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase3KPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase3k_candidate_ranking_audit_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerPhase3KPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        table = self.detail._table
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 3-K Candidate Ranking / Fallback Path Audit",
                "",
                "## Scope",
                "",
                f"- period: `{result['period']}`",
                f"- target: `{result['profile']}`",
                "- no API refetch",
                "- no OpenAI API",
                "- no current model historical regeneration",
                "- `selected_count_in_day` used: `False`",
                "- no trading logic changes",
                "- no full backtest executed",
                "",
                "## Current Ranking Basis",
                "",
                table([result["ranking_basis"]], list(result["ranking_basis"].keys())),
                "",
                "## Candidate Path Classification",
                "",
                table(result["candidate_path_classification"], ["classification", "count"]),
                "",
                "## Candidate Path Detail Sample",
                "",
                table(
                    result["candidate_path_detail_sample"],
                    [
                        "date",
                        "classification",
                        "candidate_order",
                        "code",
                        "expected_return_10d",
                        "risk_adjusted_score",
                        "pm_score",
                        "pm_multiplier",
                        "planned_buy_amount",
                        "affordable",
                        "skip_reason",
                        "actually_bought",
                        "was_next_affordable_candidate_available",
                    ],
                ),
                "",
                "## Fallback Candidate Quality",
                "",
                table([result["fallback_candidate_quality"]], list(result["fallback_candidate_quality"].keys())),
                "",
                "## Expected Return / Rank Quality",
                "",
                table(
                    result["rank_quality"],
                    [
                        "rank_group",
                        "candidate_count",
                        "average_expected_return",
                        "average_pm_score",
                        "average_hypothetical_return_5d",
                        "average_hypothetical_return_10d",
                    ],
                ),
                "",
                "## Fallback Decision Flags",
                "",
                table(result["fallback_decision_flags"], ["flag", "value", "detail"]),
                "",
                "## Audit Notes",
                "",
                table(result["audit_notes"], ["item", "value"]),
                "",
            ]
        )

    def _ranking_basis(self) -> dict[str, Any]:
        return {
            "selected_sort_function": "_sort_selected_candidates",
            "selected_sort_columns": "daily_score_rank ASC, risk_adjusted_score DESC, code ASC",
            "ml_backtest_ranking": "risk_adjusted_score",
            "sort_direction": "rank ascending; score descending",
            "pm_ai_timing": "PM AI sizing is applied after selected candidate ordering, during buy sizing.",
            "affordability_timing": "cash / daily_buy_limit / round lot / PM sizing / per-code cap are checked after ordering.",
            "fallback_replace_function": "_find_affordable_fallback_candidate",
            "surplus_fallback_function": "_find_surplus_affordable_fallback_candidate",
            "fallback_sort_columns": "regular_priority ASC, risk_adjusted_score DESC, rank ASC, round_lot_amount ASC, code ASC",
            "skip_advances_loop": "The selected loop continues after SKIP; replace fallback is attempted only when enabled and skip reason is eligible.",
            "selected_count_in_day_used": False,
        }

    def _candidate_log(self, payload: dict[str, Any]) -> pd.DataFrame:
        audit = payload["audit"].copy()
        if audit.empty:
            return audit
        audit["signal_date"] = pd.to_datetime(audit.get("signal_date"), errors="coerce").dt.strftime("%Y-%m-%d")
        for column in [
            "candidate_rank",
            "score_rank",
            "fallback_rank",
            "risk_adjusted_score",
            "expected_return_10d",
            "pm_score",
            "pm_multiplier",
            "planned_amount",
            "pm_target_amount",
            "final_amount",
        ]:
            if column in audit.columns:
                audit[column] = pd.to_numeric(audit[column], errors="coerce")
        audit["candidate_order"] = self._first_available(audit, ["candidate_rank", "score_rank", "fallback_rank"])
        audit["candidate_order"] = audit["candidate_order"].fillna(999999)
        for column in ["decision", "skip_reason", "reject_reason", "candidate_source"]:
            if column not in audit.columns:
                audit[column] = ""
            audit[column] = audit[column].fillna("").astype(str)
        reason = (audit["skip_reason"] + "|" + audit["reject_reason"]).str.lower()
        audit["is_selected_but_not_affordable"] = reason.str.contains("selected_but_not_affordable", regex=False)
        audit["actually_bought"] = audit["decision"].str.upper().isin(["BUY", "SCALED_BUY"])
        audit["affordable"] = audit["actually_bought"]
        audit["planned_buy_amount"] = self._first_available(audit, ["pm_target_amount", "planned_amount"])
        audit = self.affordability._attach_hypothetical_returns(audit)
        return audit.sort_values(["signal_date", "candidate_order", "code"]).reset_index(drop=True)

    def _candidate_path_rows(self, audit: pd.DataFrame, unaffordable: pd.DataFrame) -> list[dict[str, Any]]:
        if audit.empty or unaffordable.empty:
            return []
        rows: list[dict[str, Any]] = []
        for date, day_unaffordable in unaffordable.groupby("signal_date"):
            day = audit[audit["signal_date"].eq(date)].sort_values(["candidate_order", "code"])
            if day.empty:
                continue
            first_order = float(day["candidate_order"].min())
            top = day[day["candidate_order"].eq(first_order)]
            top_unaffordable = bool(top["is_selected_but_not_affordable"].any())
            buys_after_top = day[(day["candidate_order"] > first_order) & day["actually_bought"]]
            buys_any = day[day["actually_bought"]]
            low_pm_skips = day["skip_reason"].str.contains("pm_low_score_skip", case=False, na=False)
            all_unaffordable = bool(day["is_selected_but_not_affordable"].all())
            if top_unaffordable and not buys_any.empty and not buys_after_top.empty:
                classification = "top_candidate_unaffordable_but_next_candidate_bought"
            elif top_unaffordable and buys_any.empty and len(day) <= len(day_unaffordable):
                classification = "top_candidate_unaffordable_and_no_buy"
            elif all_unaffordable:
                classification = "all_candidates_unaffordable"
            elif bool(low_pm_skips.all()) and len(day) > 0:
                classification = "all_candidates_low_pm_skipped"
            elif top_unaffordable and buys_any.empty and len(day) > len(day_unaffordable):
                classification = "top_candidate_unaffordable_but_affordable_candidate_exists_not_bought"
            else:
                classification = "candidate_log_insufficient"
            was_next = bool(not buys_after_top.empty)
            for _, row in day.iterrows():
                rows.append(
                    {
                        "date": date,
                        "classification": classification,
                        "candidate_order": self._json_value(row.get("candidate_order")),
                        "code": str(row.get("code") or ""),
                        "expected_return_10d": self._json_value(row.get("expected_return_10d")),
                        "risk_adjusted_score": self._json_value(row.get("risk_adjusted_score")),
                        "pm_score": self._json_value(row.get("pm_score")),
                        "pm_multiplier": self._json_value(row.get("pm_multiplier")),
                        "planned_buy_amount": self._json_value(row.get("planned_buy_amount")),
                        "affordable": bool(row.get("affordable")),
                        "skip_reason": str(row.get("skip_reason") or row.get("reject_reason") or ""),
                        "actually_bought": bool(row.get("actually_bought")),
                        "was_next_affordable_candidate_available": was_next,
                    }
                )
        return rows

    def _path_classification(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        days = {(row["date"], row["classification"]) for row in rows}
        counts: dict[str, int] = {}
        for _, classification in days:
            counts[classification] = counts.get(classification, 0) + 1
        return [{"classification": key, "count": value} for key, value in sorted(counts.items(), key=lambda item: item[0])]

    def _fallback_candidate_quality(self, audit: pd.DataFrame, unaffordable: pd.DataFrame) -> dict[str, Any]:
        if audit.empty or unaffordable.empty:
            return self._empty_quality()
        days = set(unaffordable["signal_date"].dropna().astype(str))
        alternatives = audit[audit["signal_date"].isin(days) & audit["actually_bought"]].copy()
        if alternatives.empty:
            return self._empty_quality()
        order = pd.to_numeric(alternatives["candidate_order"], errors="coerce").dropna()
        return {
            "candidate_count": int(len(alternatives)),
            "average_candidate_order": self._mean(alternatives.get("candidate_order")),
            "median_candidate_order": float(order.median()) if not order.empty else None,
            "p75_candidate_order": float(order.quantile(0.75)) if not order.empty else None,
            "p90_candidate_order": float(order.quantile(0.90)) if not order.empty else None,
            "average_expected_return": self._mean(alternatives.get("expected_return_10d")),
            "average_pm_score": self._mean(alternatives.get("pm_score")),
            "average_pm_multiplier": self._mean(alternatives.get("pm_multiplier")),
            "hypothetical_return_5d": self._mean(alternatives.get("hypothetical_return_5d")),
            "hypothetical_return_10d": self._mean(alternatives.get("hypothetical_return_10d")),
        }

    def _rank_quality(self, audit: pd.DataFrame) -> list[dict[str, Any]]:
        if audit.empty:
            return []
        data = audit.copy()
        data["rank_group"] = pd.to_numeric(data["candidate_order"], errors="coerce").apply(self._rank_group)
        rows = []
        order = ["rank 1", "rank 2-3", "rank 4-5", "rank 6-10", "rank 11+"]
        for group_name, group in data.groupby("rank_group", dropna=False):
            rows.append(
                {
                    "rank_group": str(group_name),
                    "candidate_count": int(len(group)),
                    "average_expected_return": self._mean(group.get("expected_return_10d")),
                    "average_pm_score": self._mean(group.get("pm_score")),
                    "average_hypothetical_return_5d": self._mean(group.get("hypothetical_return_5d")),
                    "average_hypothetical_return_10d": self._mean(group.get("hypothetical_return_10d")),
                }
            )
        return sorted(rows, key=lambda row: order.index(row["rank_group"]) if row["rank_group"] in order else 99)

    def _decision_flags(
        self,
        path_rows: list[dict[str, Any]],
        fallback_quality: dict[str, Any],
        audit: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        classifications = {row["classification"] for row in path_rows}
        fallback_enabled = bool(audit.get("candidate_source", pd.Series(dtype=str)).astype(str).eq("fallback").any())
        quality_return = fallback_quality.get("hypothetical_return_5d")
        quality_count = int(fallback_quality.get("candidate_count") or 0)
        return [
            {
                "flag": "current_logic_already_falls_back",
                "value": fallback_enabled,
                "detail": "candidate_source=fallback found in purchase_audit" if fallback_enabled else "No fallback candidate_source found in current log.",
            },
            {
                "flag": "blocked_candidate_stops_buy_loop",
                "value": "top_candidate_unaffordable_and_no_buy" in classifications,
                "detail": ",".join(sorted(classifications)),
            },
            {
                "flag": "affordable_alternative_is_high_quality",
                "value": bool(quality_count and quality_return is not None and quality_return > 0),
                "detail": f"count={quality_count}, avg_5d={quality_return}",
            },
            {
                "flag": "fallback_to_next_affordable_candidate_recommended",
                "value": bool(
                    ("top_candidate_unaffordable_and_no_buy" in classifications or "candidate_log_insufficient" in classifications)
                    and quality_count
                    and quality_return is not None
                    and quality_return > 0
                ),
                "detail": "Recommend only if full candidate logs confirm alternatives not already being bought.",
            },
            {
                "flag": "ranking_log_insufficient",
                "value": True,
                "detail": "purchase_audit is not a complete scored-candidate log; it records selected/attempted candidates only.",
            },
        ]

    def _rank_group(self, value: Any) -> str:
        try:
            rank = float(value)
        except Exception:
            return "rank 11+"
        if rank <= 1:
            return "rank 1"
        if rank <= 3:
            return "rank 2-3"
        if rank <= 5:
            return "rank 4-5"
        if rank <= 10:
            return "rank 6-10"
        return "rank 11+"

    def _empty_quality(self) -> dict[str, Any]:
        return {
            "candidate_count": 0,
            "average_candidate_order": None,
            "median_candidate_order": None,
            "p75_candidate_order": None,
            "p90_candidate_order": None,
            "average_expected_return": None,
            "average_pm_score": None,
            "average_pm_multiplier": None,
            "hypothetical_return_5d": None,
            "hypothetical_return_10d": None,
        }

    def _first_available(self, df: pd.DataFrame, columns: list[str]) -> pd.Series:
        result = pd.Series(pd.NA, index=df.index, dtype="Float64")
        for column in columns:
            if column in df.columns:
                values = pd.to_numeric(df[column], errors="coerce")
                result = result.where(result.notna(), values)
        return result

    def _mean(self, series: pd.Series | None) -> float | None:
        if series is None:
            return None
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    def _json_value(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value
