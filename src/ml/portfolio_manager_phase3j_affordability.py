from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_phase3d_detail_audit import (
    BASELINE_PROFILE,
    PERIOD,
    PHASE3D_PROFILE,
    PortfolioManagerPhase3DDetailAudit,
)
from ml.portfolio_manager_phase3e import PHASE3E_PROFILE
from ml.portfolio_manager_phase3g import PHASE3G_PROFILE
from ml.portfolio_manager_phase3h_capital_utilization import PHASE3H_PROFILE
from ml.portfolio_manager_phase3i_candidate_pool import POOL_X2_PROFILE, POOL_X3_PROFILE


TARGET_PROFILE = PHASE3H_PROFILE
ROUND_LOT_SIZE = 100
PRICE_CACHE_DIR = Path("data/cache/jquants/prices")


@dataclass(frozen=True)
class PortfolioManagerPhase3JPaths:
    markdown: Path
    json: Path


class PortfolioManagerPhase3JAffordabilityAudit:
    """Audit why selected candidates could not become affordable orders.

    This class reads existing backtest logs only. It does not refetch APIs,
    regenerate predictions, or change trading behavior.
    """

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
        self.report_dir = self.root / "reports" / "ml"
        self.price_cache_dir = self.root / PRICE_CACHE_DIR

    def build(self) -> dict[str, Any]:
        target_payload = self._profile_payload(self.target_profile)
        unaffordable = self._selected_but_not_affordable_rows(target_payload)
        enriched = self._enrich_unaffordable_rows(unaffordable, target_payload)
        alternatives = self._fallback_possibility(enriched, target_payload)
        profile_comparison = self._profile_comparison()
        result = {
            "purpose": "Portfolio Manager AI Phase 3-J affordability / fallback / round-lot audit",
            "period": self.period,
            "constraints": {
                "api_refetch": False,
                "openai_api": False,
                "historical_predictions_source": "data/ml/walk_forward_predictions/",
                "current_model_historical_regeneration": False,
                "selected_count_in_day_used": False,
                "trading_logic_changed": False,
                "new_profile_added": False,
                "live_order_placement": False,
            },
            "profiles": {
                "target": self.target_profile,
                "v2_75": PHASE3D_PROFILE,
                "v2_76": PHASE3E_PROFILE,
                "v2_77_cap_020": PHASE3G_PROFILE,
                "v2_77_cap_030": PHASE3H_PROFILE,
                "pool_x2": POOL_X2_PROFILE,
                "pool_x3": POOL_X3_PROFILE,
            },
            "selected_but_not_affordable_count": int(len(enriched)),
            "selected_but_not_affordable_detail_sample": self._records(enriched.head(80)),
            "reason_summary": self._reason_summary(enriched),
            "round_lot_price_distribution": self._round_lot_distribution(enriched),
            "pm_score_band_summary": self._pm_score_band_summary(enriched),
            "missed_opportunity_summary": self._missed_opportunity_summary(enriched),
            "top_missed_opportunities": self._top_missed_opportunities(enriched),
            "fallback_possibility": alternatives,
            "profile_comparison": profile_comparison,
            "improvement_candidates": self._improvement_candidates(enriched, alternatives, profile_comparison),
            "audit_notes": [
                {
                    "item": "scope",
                    "value": "Existing logs only; no full backtest rerun; no model retraining.",
                },
                {
                    "item": "hypothetical_returns",
                    "value": "Computed from cached close prices when available; missing prices remain null.",
                },
                {
                    "item": "per_code_cap_columns",
                    "value": "If purchase_audit lacks explicit cap columns, cap-related reason is inferred conservatively.",
                },
            ],
        }
        return result

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase3JPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase3j_affordability_audit_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerPhase3JPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        table = self.detail._table
        return "\n".join(
            [
                "# Portfolio Manager AI Phase 3-J Affordability / Fallback / Round Lot Audit",
                "",
                "## Scope",
                "",
                f"- period: `{result['period']}`",
                f"- target: `{result['profiles']['target']}`",
                "- no API refetch",
                "- no OpenAI API",
                "- no current model historical regeneration",
                "- `selected_count_in_day` used: `False`",
                "- no trading logic changes",
                "- no new profile added",
                "",
                "## Selected But Not Affordable Summary",
                "",
                f"- total count: `{result['selected_but_not_affordable_count']}`",
                "",
                table(
                    result["reason_summary"],
                    [
                        "dominant_blocking_reason",
                        "count",
                        "average_shortage_amount",
                        "average_capital_utilization",
                        "average_cash",
                        "average_total_assets",
                        "average_pm_score",
                        "average_pm_multiplier",
                        "average_minimum_lot_amount",
                        "average_planned_buy_amount",
                        "average_allowed_additional_buy",
                    ],
                ),
                "",
                "## Round Lot / Price Distribution",
                "",
                table([result["round_lot_price_distribution"]], list(result["round_lot_price_distribution"].keys())),
                "",
                "## PM Score Band Summary",
                "",
                table(
                    result["pm_score_band_summary"],
                    [
                        "pm_score_band",
                        "count",
                        "average_pm_score",
                        "average_pm_multiplier",
                        "average_minimum_lot_amount",
                        "average_shortage_amount",
                        "average_capital_utilization",
                        "average_cash",
                        "reason_breakdown",
                    ],
                ),
                "",
                "## Missed Opportunity Summary",
                "",
                table(
                    result["missed_opportunity_summary"],
                    [
                        "group",
                        "value",
                        "count",
                        "average_hypothetical_return_3d",
                        "average_hypothetical_return_5d",
                        "average_hypothetical_return_10d",
                    ],
                ),
                "",
                "## Top Missed Opportunities",
                "",
                table(
                    result["top_missed_opportunities"],
                    [
                        "signal_date",
                        "code",
                        "pm_score",
                        "pm_multiplier",
                        "dominant_blocking_reason",
                        "minimum_lot_amount",
                        "shortage_amount",
                        "hypothetical_return_5d",
                        "hypothetical_return_10d",
                    ],
                ),
                "",
                "## Fallback / Replacement Possibility",
                "",
                table([result["fallback_possibility"]], list(result["fallback_possibility"].keys())),
                "",
                "## Profile Comparison",
                "",
                table(
                    result["profile_comparison"],
                    [
                        "profile_label",
                        "profile",
                        "selected_but_not_affordable",
                        "insufficient_available_cash",
                        "daily_buy_limit_scaled_below_round_lot",
                        "per_code_exposure_cap_scaled_below_round_lot",
                        "pm_low_score_skip",
                        "average_capital_utilization",
                        "net_profit",
                        "profit_factor",
                        "max_drawdown",
                        "win_rate",
                    ],
                ),
                "",
                "## Improvement Candidates",
                "",
                table(result["improvement_candidates"], ["priority", "candidate", "signal", "reason"]),
                "",
                "## Audit Notes",
                "",
                table(result["audit_notes"], ["item", "value"]),
                "",
            ]
        )

    def _profile_payload(self, profile: str) -> dict[str, Any]:
        backtest_dir = self.root / "logs" / "backtests" / profile / self.period
        summary_raw = self.detail._read_json(backtest_dir / "backtest_summary.json")
        trades_raw = self.detail._read_csv(backtest_dir / "trades.csv")
        audit = self.detail._read_csv(backtest_dir / "purchase_audit.csv")
        trades = self.detail._sell_trades_with_pm(trades_raw, audit)
        daily = self._daily_frame(backtest_dir / "summary.csv", summary_raw)
        return {
            "profile": profile,
            "summary": self.detail._summary_row(profile, summary_raw, trades),
            "summary_raw": summary_raw,
            "audit": audit,
            "trades": trades,
            "daily": daily,
        }

    def _daily_frame(self, path: Path, summary_raw: dict[str, Any]) -> pd.DataFrame:
        if path.exists():
            daily = pd.read_csv(path)
        else:
            daily = pd.DataFrame(summary_raw.get("daily_asset_curve") or [])
        if daily.empty:
            return daily
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "market_value" not in daily.columns and "positions_value" in daily.columns:
            daily["market_value"] = daily["positions_value"]
        for column in ["cash", "market_value", "total_assets", "open_positions_count"]:
            if column in daily.columns:
                daily[column] = pd.to_numeric(daily[column], errors="coerce")
        daily["capital_utilization"] = self._safe_div(daily.get("market_value"), daily.get("total_assets"))
        return daily

    def _selected_but_not_affordable_rows(self, payload: dict[str, Any]) -> pd.DataFrame:
        audit = payload["audit"].copy()
        if audit.empty:
            return audit
        for column in ["skip_reason", "reject_reason", "decision"]:
            if column not in audit.columns:
                audit[column] = ""
            audit[column] = audit[column].fillna("").astype(str)
        reason_text = (audit["skip_reason"] + "|" + audit["reject_reason"]).str.lower()
        mask = reason_text.str.contains("selected_but_not_affordable", regex=False)
        return audit.loc[mask].copy()

    def _enrich_unaffordable_rows(self, rows: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        if rows.empty:
            return rows.copy()
        df = rows.copy()
        df["signal_date"] = pd.to_datetime(df.get("signal_date"), errors="coerce").dt.strftime("%Y-%m-%d")
        numeric_columns = [
            "pm_score",
            "pm_multiplier",
            "planned_amount",
            "pm_base_planned_amount",
            "pm_target_amount",
            "pm_cash_capped_target_amount",
            "allocation_limit",
            "cash_before",
            "daily_buy_limit_remaining_before",
            "final_amount",
            "planned_shares",
            "pm_base_planned_shares",
            "scaled_shares",
            "final_shares",
            "risk_adjusted_score",
            "expected_return_10d",
            "bad_entry_probability_10d",
        ]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        df["planned_buy_amount_before_constraints"] = self._first_available(
            df, ["pm_base_planned_amount", "planned_amount", "pm_target_amount"]
        )
        df["planned_buy_amount_after_pm"] = self._first_available(
            df, ["pm_target_amount", "planned_amount", "pm_base_planned_amount"]
        )
        df["planned_buy_amount_after_per_code_cap"] = self._first_available(
            df, ["allocation_limit", "pm_cash_capped_target_amount", "planned_buy_amount_after_pm"]
        )
        df["available_cash"] = df.get("cash_before")
        df["daily_buy_remaining"] = df.get("daily_buy_limit_remaining_before")
        df["allowed_additional_buy"] = self._row_min(
            df,
            [
                "available_cash",
                "daily_buy_remaining",
                "planned_buy_amount_after_per_code_cap",
            ],
        )
        df["close_price"] = self._infer_price(df)
        df["entry_price"] = df["close_price"]
        df["round_lot_size"] = ROUND_LOT_SIZE
        df["minimum_lot_amount"] = df["close_price"] * ROUND_LOT_SIZE
        df["shortage_amount"] = (df["minimum_lot_amount"] - df["allowed_additional_buy"]).clip(lower=0)

        daily = payload["daily"]
        if not daily.empty:
            join_cols = [
                c for c in ["date", "cash", "market_value", "total_assets", "capital_utilization", "open_positions_count"]
                if c in daily.columns
            ]
            df = df.merge(daily[join_cols], left_on="signal_date", right_on="date", how="left", suffixes=("", "_daily"))
        df["total_assets"] = self._first_available(df, ["total_assets", "cash_before"])
        df["market_value"] = self._first_available(df, ["market_value"])
        df["capital_utilization"] = self._first_available(df, ["capital_utilization"])
        df["holding_count"] = self._first_available(df, ["open_positions_count"])
        df["current_code_exposure"] = pd.NA
        df["per_code_cap_limit"] = df.get("allocation_limit") if "allocation_limit" in df.columns else pd.NA
        df["dominant_blocking_reason"] = df.apply(self._blocking_reason, axis=1)
        df["pm_score_band"] = df["pm_score"].apply(self._pm_score_band)
        df = self._attach_hypothetical_returns(df)
        return df

    def _blocking_reason(self, row: pd.Series) -> str:
        skip = str(row.get("skip_reason") or "").lower()
        reject = str(row.get("reject_reason") or "").lower()
        scale = str(row.get("scale_reason") or "").lower()
        resize = str(row.get("pm_resize_reason") or "").lower()
        text = "|".join([skip, reject, scale, resize])
        minimum = self._num(row.get("minimum_lot_amount"))
        cash = self._num(row.get("available_cash"))
        daily = self._num(row.get("daily_buy_remaining"))
        allowed = self._num(row.get("allowed_additional_buy"))
        pm_target = self._num(row.get("planned_buy_amount_after_pm"))
        per_code = self._num(row.get("planned_buy_amount_after_per_code_cap"))

        if "duplicate" in text or "already" in text:
            return "already_holding_or_duplicate"
        if "daily_buy_limit" in text and "round_lot" in text:
            return "below_round_lot_after_per_code_cap" if per_code < minimum else "daily_buy_limit_shortage"
        if "per_code" in text and "round_lot" in text:
            return "below_round_lot_after_per_code_cap"
        if "cash" in text and cash < minimum:
            return "cash_shortage"
        if daily < minimum:
            return "daily_buy_limit_shortage"
        if cash < minimum:
            return "cash_shortage"
        if allowed < minimum and per_code < minimum:
            return "per_code_cap_shortage"
        if pm_target < minimum:
            return "below_round_lot_after_pm_sizing"
        if allowed < minimum:
            return "selected_but_not_affordable_due_to_price"
        return "unknown"

    def _reason_summary(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        metrics = {
            "shortage_amount": "average_shortage_amount",
            "capital_utilization": "average_capital_utilization",
            "available_cash": "average_cash",
            "total_assets": "average_total_assets",
            "pm_score": "average_pm_score",
            "pm_multiplier": "average_pm_multiplier",
            "minimum_lot_amount": "average_minimum_lot_amount",
            "planned_buy_amount_after_pm": "average_planned_buy_amount",
            "allowed_additional_buy": "average_allowed_additional_buy",
        }
        rows = []
        for reason, group in df.groupby("dominant_blocking_reason", dropna=False):
            row = {"dominant_blocking_reason": str(reason), "count": int(len(group))}
            for column, out in metrics.items():
                row[out] = self._mean(group.get(column))
            rows.append(row)
        return sorted(rows, key=lambda row: row["count"], reverse=True)

    def _round_lot_distribution(self, df: pd.DataFrame) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for column in ["minimum_lot_amount", "close_price", "shortage_amount"]:
            series = pd.to_numeric(df.get(column), errors="coerce") if column in df else pd.Series(dtype=float)
            quantiles = series.dropna().quantile([0.10, 0.25, 0.50, 0.75, 0.90]).to_dict() if not series.dropna().empty else {}
            for q, value in quantiles.items():
                result[f"{column}_p{int(q * 100)}"] = float(value)
        if df.empty:
            result.update(
                {
                    "high_price_candidate_count": 0,
                    "minimum_lot_amount_above_cash_count": 0,
                    "minimum_lot_amount_above_daily_remaining_count": 0,
                    "minimum_lot_amount_above_allowed_additional_buy_count": 0,
                }
            )
            return result
        minimum = pd.to_numeric(df["minimum_lot_amount"], errors="coerce")
        result["high_price_candidate_count"] = int((minimum >= 500_000).sum())
        result["minimum_lot_amount_above_cash_count"] = int((minimum > pd.to_numeric(df["available_cash"], errors="coerce")).sum())
        result["minimum_lot_amount_above_daily_remaining_count"] = int(
            (minimum > pd.to_numeric(df["daily_buy_remaining"], errors="coerce")).sum()
        )
        result["minimum_lot_amount_above_allowed_additional_buy_count"] = int(
            (minimum > pd.to_numeric(df["allowed_additional_buy"], errors="coerce")).sum()
        )
        return result

    def _pm_score_band_summary(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        rows = []
        for band, group in df.groupby("pm_score_band", dropna=False):
            rows.append(
                {
                    "pm_score_band": str(band),
                    "count": int(len(group)),
                    "average_pm_score": self._mean(group.get("pm_score")),
                    "average_pm_multiplier": self._mean(group.get("pm_multiplier")),
                    "average_minimum_lot_amount": self._mean(group.get("minimum_lot_amount")),
                    "average_shortage_amount": self._mean(group.get("shortage_amount")),
                    "average_capital_utilization": self._mean(group.get("capital_utilization")),
                    "average_cash": self._mean(group.get("available_cash")),
                    "reason_breakdown": self._value_counts(group.get("dominant_blocking_reason")),
                }
            )
        order = ["< -0.20", "-0.20 to 0", "0 to 0.20", "0.20 to 0.40", ">= 0.40", "unknown"]
        return sorted(rows, key=lambda row: order.index(row["pm_score_band"]) if row["pm_score_band"] in order else 99)

    def _missed_opportunity_summary(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for label, column in [("reason", "dominant_blocking_reason"), ("pm_score_band", "pm_score_band")]:
            if df.empty or column not in df:
                continue
            for value, group in df.groupby(column, dropna=False):
                rows.append(
                    {
                        "group": label,
                        "value": str(value),
                        "count": int(len(group)),
                        "average_hypothetical_return_3d": self._mean(group.get("hypothetical_return_3d")),
                        "average_hypothetical_return_5d": self._mean(group.get("hypothetical_return_5d")),
                        "average_hypothetical_return_10d": self._mean(group.get("hypothetical_return_10d")),
                    }
                )
        return rows

    def _top_missed_opportunities(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty or "hypothetical_return_5d" not in df:
            return []
        columns = [
            "signal_date",
            "code",
            "pm_score",
            "pm_multiplier",
            "dominant_blocking_reason",
            "minimum_lot_amount",
            "shortage_amount",
            "hypothetical_return_5d",
            "hypothetical_return_10d",
        ]
        present = [c for c in columns if c in df.columns]
        return self._records(df.sort_values("hypothetical_return_5d", ascending=False).head(20)[present])

    def _fallback_possibility(self, unaffordable: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
        audit = payload["audit"].copy()
        if audit.empty or unaffordable.empty:
            return {
                "days_with_unaffordable_candidate": 0,
                "days_with_affordable_alternative_candidate": 0,
                "affordable_alternative_count": 0,
                "affordable_alternative_average_pm_score": None,
                "affordable_alternative_average_pm_multiplier": None,
                "affordable_alternative_hypothetical_return_5d": None,
            }
        audit["signal_date"] = pd.to_datetime(audit.get("signal_date"), errors="coerce").dt.strftime("%Y-%m-%d")
        days = set(unaffordable["signal_date"].dropna().astype(str))
        candidates = audit[audit["signal_date"].isin(days)].copy()
        for column in ["decision", "skip_reason", "reject_reason"]:
            if column not in candidates:
                candidates[column] = ""
            candidates[column] = candidates[column].fillna("").astype(str)
        bought = candidates[candidates["decision"].str.upper().isin(["BUY", "SCALED_BUY"])].copy()
        bought = self._attach_hypothetical_returns(self._normalize_for_hypothesis(bought))
        return {
            "days_with_unaffordable_candidate": int(len(days)),
            "days_with_affordable_alternative_candidate": int(bought["signal_date"].nunique()) if not bought.empty else 0,
            "affordable_alternative_count": int(len(bought)),
            "affordable_alternative_average_pm_score": self._mean(pd.to_numeric(bought.get("pm_score"), errors="coerce")),
            "affordable_alternative_average_pm_multiplier": self._mean(pd.to_numeric(bought.get("pm_multiplier"), errors="coerce")),
            "affordable_alternative_hypothetical_return_5d": self._mean(bought.get("hypothetical_return_5d")),
        }

    def _profile_comparison(self) -> list[dict[str, Any]]:
        profiles = {
            "v2_75": PHASE3D_PROFILE,
            "v2_76": PHASE3E_PROFILE,
            "v2_77_cap_020": PHASE3G_PROFILE,
            "v2_77_cap_030": PHASE3H_PROFILE,
            "pool_x2": POOL_X2_PROFILE,
            "pool_x3": POOL_X3_PROFILE,
        }
        rows = []
        for label, profile in profiles.items():
            payload = self._profile_payload(profile)
            audit = payload["audit"]
            skip_counts = self._skip_counts(audit)
            util = self._daily_distribution(payload["daily"])
            rows.append(
                {
                    "profile_label": label,
                    "profile": profile,
                    "selected_but_not_affordable": skip_counts.get("selected_but_not_affordable", 0),
                    "insufficient_available_cash": skip_counts.get("insufficient_available_cash", 0),
                    "daily_buy_limit_scaled_below_round_lot": skip_counts.get("daily_buy_limit_scaled_below_round_lot", 0),
                    "per_code_exposure_cap_scaled_below_round_lot": skip_counts.get(
                        "per_code_exposure_cap_scaled_below_round_lot", 0
                    ),
                    "pm_low_score_skip": skip_counts.get("pm_low_score_skip", 0),
                    "average_capital_utilization": util.get("average_capital_utilization"),
                    "net_profit": payload["summary"]["net_profit"],
                    "profit_factor": payload["summary"]["profit_factor"],
                    "max_drawdown": payload["summary"]["max_drawdown"],
                    "win_rate": payload["summary"]["win_rate"],
                }
            )
        return rows

    def _improvement_candidates(
        self,
        unaffordable: pd.DataFrame,
        fallback: dict[str, Any],
        profile_comparison: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        high_pm = unaffordable[pd.to_numeric(unaffordable.get("pm_score"), errors="coerce") >= 0.20] if not unaffordable.empty else pd.DataFrame()
        high_pm_rate = len(high_pm) / len(unaffordable) if len(unaffordable) else 0
        avg_missed_5d = self._mean(unaffordable.get("hypothetical_return_5d")) if not unaffordable.empty else None
        affordable_days = int(fallback.get("days_with_affordable_alternative_candidate") or 0)
        unaffordable_days = int(fallback.get("days_with_unaffordable_candidate") or 0)
        fallback_rate = affordable_days / unaffordable_days if unaffordable_days else 0
        round_lot_counts = self._value_counts(unaffordable.get("dominant_blocking_reason")) if not unaffordable.empty else {}
        rows = []
        if fallback_rate >= 0.25:
            rows.append(
                {
                    "priority": 1,
                    "candidate": "fallback_to_next_affordable_candidate",
                    "signal": f"affordable_alternative_day_rate={fallback_rate:.2%}",
                    "reason": "Some blocked days still had affordable alternatives in the log.",
                }
            )
        if high_pm_rate >= 0.30 and (avg_missed_5d is None or avg_missed_5d > 0):
            rows.append(
                {
                    "priority": 2,
                    "candidate": "allow_smaller_position_for_high_pm",
                    "signal": f"high_pm_unaffordable_rate={high_pm_rate:.2%}, avg_missed_5d={avg_missed_5d}",
                    "reason": "A meaningful share of unaffordable candidates had high PM scores.",
                }
            )
        if round_lot_counts.get("below_round_lot_after_per_code_cap", 0) or round_lot_counts.get("per_code_cap_shortage", 0):
            rows.append(
                {
                    "priority": 3,
                    "candidate": "loosen_per_code_cap_only_for_high_pm",
                    "signal": str(round_lot_counts),
                    "reason": "Per-code cap appears to shrink some candidates below round lot.",
                }
            )
        rows.append(
            {
                "priority": 9,
                "candidate": "candidate_pool_expansion_rejected",
                "signal": "Phase 3-I x2/x3 did not improve no_candidates or utilization enough.",
                "reason": "Candidate count alone is not the active bottleneck.",
            }
        )
        if not rows:
            rows.append(
                {
                    "priority": 1,
                    "candidate": "keep_current_v2_77_cap030",
                    "signal": "No strong positive missed-opportunity signal.",
                    "reason": "Unblocking may dilute quality if missed candidates are weak.",
                }
            )
        return sorted(rows, key=lambda row: row["priority"])

    def _attach_hypothetical_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        out = df.copy()
        if "signal_date" not in out or "code" not in out:
            return out
        needed_dates = sorted(set(out["signal_date"].dropna().astype(str)))
        price_map = self._load_price_map_around(needed_dates, extra_days=20)
        for horizon in [3, 5, 10]:
            out[f"hypothetical_return_{horizon}d"] = out.apply(
                lambda row: self._future_return(price_map, str(row.get("signal_date")), str(row.get("code")), horizon),
                axis=1,
            )
        return out

    def _load_price_map_around(self, dates: list[str], extra_days: int) -> dict[str, dict[str, float]]:
        if not dates or not self.price_cache_dir.exists():
            return {}
        files = sorted(self.price_cache_dir.glob("*.json"))
        if not files:
            return {}
        start = pd.to_datetime(min(dates), errors="coerce") - pd.Timedelta(days=5)
        end = pd.to_datetime(max(dates), errors="coerce") + pd.Timedelta(days=extra_days * 2 + 10)
        result: dict[str, dict[str, float]] = {}
        for path in files:
            date = path.stem
            dt = pd.to_datetime(date, errors="coerce")
            if pd.isna(dt) or dt < start or dt > end:
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            records = raw.get("prices") or raw.get("records") if isinstance(raw, dict) else raw
            if not isinstance(records, list):
                continue
            daily: dict[str, float] = {}
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                code = str(rec.get("code") or rec.get("Code") or "")
                close = rec.get("close", rec.get("Close", rec.get("C", rec.get("AdjustmentClose"))))
                value = self._num(close)
                if code and pd.notna(value):
                    daily[code] = float(value)
            if daily:
                result[date] = daily
        return result

    def _future_return(self, price_map: dict[str, dict[str, float]], date: str, code: str, horizon: int) -> float | None:
        available_dates = [d for d in sorted(price_map) if d >= date and code in price_map[d]]
        if len(available_dates) <= horizon:
            return None
        start_date = available_dates[0]
        future_date = available_dates[horizon]
        start_price = price_map[start_date].get(code)
        future_price = price_map[future_date].get(code)
        if not start_price or not future_price:
            return None
        return float(future_price / start_price - 1)

    def _normalize_for_hypothesis(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal_date"] = pd.to_datetime(out.get("signal_date"), errors="coerce").dt.strftime("%Y-%m-%d")
        return out

    def _infer_price(self, df: pd.DataFrame) -> pd.Series:
        shares = self._first_available(df, ["pm_base_planned_shares", "planned_shares", "scaled_shares", "final_shares"])
        amount = self._first_available(df, ["pm_base_planned_amount", "planned_amount", "scaled_amount", "final_amount"])
        price = amount / shares.replace(0, pd.NA)
        return pd.to_numeric(price, errors="coerce")

    def _skip_counts(self, audit: pd.DataFrame) -> dict[str, int]:
        if audit.empty:
            return {}
        parts = []
        for column in ["skip_reason", "reject_reason"]:
            if column in audit.columns:
                parts.append(audit[column].fillna("").astype(str))
        if not parts:
            return {}
        reason = parts[0]
        for part in parts[1:]:
            reason = reason.mask(reason == "", part)
        reason = reason[reason != ""]
        return {str(k): int(v) for k, v in reason.value_counts().to_dict().items()}

    def _daily_distribution(self, daily: pd.DataFrame) -> dict[str, Any]:
        if daily.empty or "capital_utilization" not in daily:
            return {"average_capital_utilization": None}
        util = pd.to_numeric(daily["capital_utilization"], errors="coerce").dropna()
        return {"average_capital_utilization": float(util.mean()) if not util.empty else None}

    def _pm_score_band(self, value: Any) -> str:
        score = self._num(value)
        if pd.isna(score):
            return "unknown"
        if score < -0.20:
            return "< -0.20"
        if score < 0:
            return "-0.20 to 0"
        if score < 0.20:
            return "0 to 0.20"
        if score < 0.40:
            return "0.20 to 0.40"
        return ">= 0.40"

    def _records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(df.where(pd.notna(df), None).to_json(orient="records"))

    def _value_counts(self, series: pd.Series | None) -> dict[str, int]:
        if series is None:
            return {}
        return {str(k): int(v) for k, v in series.fillna("unknown").value_counts().to_dict().items()}

    def _first_available(self, df: pd.DataFrame, columns: list[str]) -> pd.Series:
        result = pd.Series(pd.NA, index=df.index, dtype="Float64")
        for column in columns:
            if column in df.columns:
                values = pd.to_numeric(df[column], errors="coerce")
                result = result.where(result.notna(), values)
        return result

    def _row_min(self, df: pd.DataFrame, columns: list[str]) -> pd.Series:
        present = [pd.to_numeric(df[c], errors="coerce") for c in columns if c in df.columns]
        if not present:
            return pd.Series(pd.NA, index=df.index, dtype="Float64")
        return pd.concat(present, axis=1).min(axis=1, skipna=True)

    def _safe_div(self, numerator: pd.Series | None, denominator: pd.Series | None) -> pd.Series:
        if numerator is None or denominator is None:
            return pd.Series(dtype=float)
        return pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(denominator, errors="coerce").replace(0, pd.NA)

    def _mean(self, series: pd.Series | None) -> float | None:
        if series is None:
            return None
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    def _num(self, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")
