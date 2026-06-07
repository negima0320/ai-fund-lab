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


PHASE3H_PROFILE = "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030"
LOW_UTILIZATION_THRESHOLD = 0.50


@dataclass(frozen=True)
class PortfolioManagerPhase3HPaths:
    markdown: Path
    json: Path


class PortfolioManagerPhase3HCapitalUtilizationAudit:
    def __init__(
        self,
        root: str | Path = ".",
        baseline_profile: str = BASELINE_PROFILE,
        phase3d_profile: str = PHASE3D_PROFILE,
        phase3e_profile: str = PHASE3E_PROFILE,
        phase3g_profile: str = PHASE3G_PROFILE,
        phase3h_profile: str = PHASE3H_PROFILE,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.baseline_profile = baseline_profile
        self.phase3d_profile = phase3d_profile
        self.phase3e_profile = phase3e_profile
        self.phase3g_profile = phase3g_profile
        self.phase3h_profile = phase3h_profile
        self.period = period
        self.detail = PortfolioManagerPhase3DDetailAudit(
            root=self.root,
            baseline_profile=baseline_profile,
            phase3d_profile=phase3d_profile,
            period=period,
        )
        self.report_dir = self.root / "reports" / "ml"

    def build(self) -> dict[str, Any]:
        profiles = {
            "v2_73": self.baseline_profile,
            "v2_75": self.phase3d_profile,
            "v2_76": self.phase3e_profile,
            "v2_77_cap_020": self.phase3g_profile,
            "v2_77_cap_030": self.phase3h_profile,
        }
        payloads = {label: self._profile_payload(profile) for label, profile in profiles.items()}
        target = payloads["v2_77_cap_030"]
        result = {
            "purpose": "Portfolio Manager AI Phase 3-H v2_77 cap 0.30 capital utilization audit",
            "period": self.period,
            "constraints": {
                "api_refetch": False,
                "openai_api": False,
                "historical_predictions_source": "data/ml/walk_forward_predictions/",
                "current_model_historical_regeneration": False,
                "selected_count_in_day_used": False,
                "trading_logic_changed": False,
                "live_order_placement": False,
            },
            "profiles": profiles,
            "summary_comparison": [payloads[label]["summary"] for label in profiles],
            "capital_utilization_distribution": {
                label: self._daily_distribution(payload["daily"])
                for label, payload in payloads.items()
            },
            "monthly_capital_utilization": {
                label: self._monthly_rows(payload)
                for label, payload in payloads.items()
            },
            "target_low_utilization_days": self._low_utilization_days(target),
            "target_skip_reason_utilization": self._skip_reason_utilization(target),
            "bottleneck_flags": self._bottleneck_flags(target),
            "next_action_candidates": [],
            "audit_notes": self._audit_notes(target),
        }
        result["next_action_candidates"] = self._next_action_candidates(result)
        return result

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase3HPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase3h_capital_utilization_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerPhase3HPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        table = self.detail._table
        monthly_target = result["monthly_capital_utilization"]["v2_77_cap_030"]
        lines = [
            "# Portfolio Manager AI Phase 3-H Capital Utilization Audit",
            "",
            "## Scope",
            "",
            f"- period: `{result['period']}`",
            f"- baseline / v2_73: `{result['profiles']['v2_73']}`",
            f"- v2_75: `{result['profiles']['v2_75']}`",
            f"- v2_76: `{result['profiles']['v2_76']}`",
            f"- v2_77 cap 0.20: `{result['profiles']['v2_77_cap_020']}`",
            f"- v2_77 cap 0.30: `{result['profiles']['v2_77_cap_030']}`",
            "- no API refetch",
            "- no OpenAI API",
            "- no current model historical regeneration",
            "- `selected_count_in_day` used: `False`",
            "- no trading logic changes",
            "",
            "## Summary Comparison",
            "",
            table(result["summary_comparison"], ["profile", "net_profit", "profit_factor", "max_drawdown", "win_rate", "total_trades"]),
            "",
            "## Daily Capital Utilization Distribution",
            "",
            table(
                [{"profile": label, **row} for label, row in result["capital_utilization_distribution"].items()],
                [
                    "profile",
                    "average_capital_utilization",
                    "median_capital_utilization",
                    "min_capital_utilization",
                    "max_capital_utilization",
                    "p10_capital_utilization",
                    "p25_capital_utilization",
                    "p75_capital_utilization",
                    "p90_capital_utilization",
                    "days_below_30pct",
                    "days_below_50pct",
                    "days_below_70pct",
                    "days_above_80pct",
                    "cash_idle_days",
                    "average_cash",
                    "median_cash",
                    "average_market_value",
                    "median_market_value",
                    "average_total_assets",
                    "median_total_assets",
                ],
            ),
            "",
            "## v2_77 cap 0.30 Monthly Capital Utilization",
            "",
            table(
                monthly_target,
                [
                    "month",
                    "average_capital_utilization",
                    "median_capital_utilization",
                    "max_capital_utilization",
                    "average_holding_count",
                    "max_holding_count",
                    "monthly_profit",
                    "monthly_win",
                    "buy_count",
                    "sell_count",
                    "total_buy_amount",
                    "total_sell_amount",
                    "pm_low_score_skip_count",
                    "per_code_exposure_cap_scaled_below_round_lot_count",
                    "selected_but_not_affordable_count",
                    "insufficient_available_cash_count",
                    "daily_buy_limit_scaled_below_round_lot_count",
                    "target_exposure_limit_count",
                ],
            ),
            "",
            "## v2_77 cap 0.30 Low Utilization Days",
            "",
            table(
                result["target_low_utilization_days"][:80],
                [
                    "date",
                    "capital_utilization",
                    "cash",
                    "market_value",
                    "total_assets",
                    "holding_count",
                    "candidate_count",
                    "buy_count",
                    "sell_count",
                    "dominant_reason",
                    "skip_reason_counts",
                ],
            ),
            "",
            "## Skip Reason vs Capital Utilization",
            "",
            table(
                result["target_skip_reason_utilization"],
                [
                    "skip_reason",
                    "count",
                    "average_capital_utilization_on_day",
                    "median_capital_utilization_on_day",
                    "average_cash_on_day",
                    "average_market_value_on_day",
                    "average_total_assets_on_day",
                    "average_holding_count_on_day",
                ],
            ),
            "",
            "## Bottleneck Flags",
            "",
            table(result["bottleneck_flags"], ["flag", "value", "detail"]),
            "",
            "## Next Action Candidates",
            "",
            table(result["next_action_candidates"], ["priority", "candidate", "reason"]),
            "",
            "## Audit Notes",
            "",
            table(result["audit_notes"], ["item", "value"]),
            "",
        ]
        return "\n".join(lines)

    def _profile_payload(self, profile: str) -> dict[str, Any]:
        backtest_dir = self.root / "logs" / "backtests" / profile / self.period
        summary_raw = self.detail._read_json(backtest_dir / "backtest_summary.json")
        trades_raw = self.detail._read_csv(backtest_dir / "trades.csv")
        audit = self.detail._read_csv(backtest_dir / "purchase_audit.csv")
        trades = self.detail._sell_trades_with_pm(trades_raw, audit)
        daily = self._daily_frame(backtest_dir / "summary.csv", summary_raw, trades)
        return {
            "profile": profile,
            "summary": self.detail._summary_row(profile, summary_raw, trades),
            "summary_raw": summary_raw,
            "trades": trades,
            "trades_raw": trades_raw,
            "audit": audit,
            "daily": daily,
        }

    def _daily_frame(self, summary_csv: Path, summary_raw: dict[str, Any], trades: pd.DataFrame) -> pd.DataFrame:
        if summary_csv.exists():
            daily = pd.read_csv(summary_csv)
        else:
            daily = pd.DataFrame(summary_raw.get("daily_asset_curve") or [])
        if daily.empty:
            return daily
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for column in ["cash", "positions_value", "market_value", "total_assets", "daily_profit", "open_positions_count"]:
            if column in daily.columns:
                daily[column] = pd.to_numeric(daily[column], errors="coerce")
        if "market_value" not in daily.columns:
            daily["market_value"] = pd.to_numeric(daily.get("positions_value"), errors="coerce").fillna(0.0)
        daily["summary_market_value"] = pd.to_numeric(daily.get("market_value"), errors="coerce").fillna(0.0)
        if "holding_count" not in daily.columns:
            daily["holding_count"] = pd.to_numeric(daily.get("open_positions_count"), errors="coerce").fillna(0).astype(int)
        committed_values = []
        holding_counts = []
        if not trades.empty and {"entry_date", "exit_date"}.issubset(trades.columns):
            trade_dates = trades.copy()
            trade_dates["entry_date"] = pd.to_datetime(trade_dates["entry_date"], errors="coerce")
            trade_dates["exit_date"] = pd.to_datetime(trade_dates["exit_date"], errors="coerce")
            for date in pd.to_datetime(daily["date"], errors="coerce"):
                open_trades = trade_dates[(trade_dates["entry_date"] <= date) & (trade_dates["exit_date"] >= date)]
                amount = pd.to_numeric(open_trades.get("final_amount"), errors="coerce").fillna(
                    pd.to_numeric(open_trades.get("entry_price"), errors="coerce").fillna(0.0)
                    * pd.to_numeric(open_trades.get("shares"), errors="coerce").fillna(0.0)
                )
                committed_values.append(float(amount.sum()))
                holding_counts.append(int(len(open_trades)))
        else:
            committed_values = [0.0] * len(daily)
            holding_counts = [0] * len(daily)
        daily["committed_market_value"] = committed_values
        daily["holding_count"] = holding_counts
        total_assets = pd.to_numeric(daily.get("total_assets"), errors="coerce").replace(0, pd.NA)
        daily["capital_utilization"] = pd.to_numeric(daily["committed_market_value"], errors="coerce").fillna(0.0) / total_assets
        if "cash" not in daily.columns:
            daily["cash"] = total_assets.fillna(0.0) - pd.to_numeric(daily["market_value"], errors="coerce").fillna(0.0)
        return daily

    def _daily_distribution(self, daily: pd.DataFrame) -> dict[str, Any]:
        if daily.empty or "capital_utilization" not in daily.columns:
            return {}
        util = pd.to_numeric(daily["capital_utilization"], errors="coerce")
        cash = pd.to_numeric(daily.get("cash"), errors="coerce")
        market = pd.to_numeric(daily.get("market_value"), errors="coerce")
        assets = pd.to_numeric(daily.get("total_assets"), errors="coerce")
        return {
            "average_capital_utilization": self._mean(util),
            "median_capital_utilization": self._median(util),
            "min_capital_utilization": self._min(util),
            "max_capital_utilization": self._max(util),
            "p10_capital_utilization": self._quantile(util, 0.10),
            "p25_capital_utilization": self._quantile(util, 0.25),
            "p75_capital_utilization": self._quantile(util, 0.75),
            "p90_capital_utilization": self._quantile(util, 0.90),
            "days_below_30pct": int(util.fillna(0).lt(0.30).sum()),
            "days_below_50pct": int(util.fillna(0).lt(0.50).sum()),
            "days_below_70pct": int(util.fillna(0).lt(0.70).sum()),
            "days_above_80pct": int(util.fillna(0).ge(0.80).sum()),
            "cash_idle_days": int(util.fillna(0).lt(0.10).sum()),
            "average_cash": self._mean(cash),
            "median_cash": self._median(cash),
            "average_market_value": self._mean(market),
            "median_market_value": self._median(market),
            "average_total_assets": self._mean(assets),
            "median_total_assets": self._median(assets),
        }

    def _monthly_rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        daily = payload["daily"].copy()
        if daily.empty:
            return []
        daily["month"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m")
        trades_raw = payload["trades_raw"].copy()
        trades = payload["trades"].copy()
        audit = payload["audit"].copy()
        rows = []
        for month, group in daily.dropna(subset=["month"]).groupby("month", sort=True):
            month_trades = self._rows_in_month(trades_raw, "entry_date", month)
            month_sells = self._rows_in_month(trades_raw, "exit_date", month)
            month_sell_trades = self._rows_in_month(trades, "exit_date", month)
            month_audit = self._rows_in_month(audit, "entry_date", month)
            skip_counts = self._skip_counts(month_audit)
            sell_profit = self.detail._profit(month_sell_trades).sum() if not month_sell_trades.empty else 0.0
            rows.append(
                {
                    "month": month,
                    "average_capital_utilization": self._mean(group["capital_utilization"]),
                    "median_capital_utilization": self._median(group["capital_utilization"]),
                    "max_capital_utilization": self._max(group["capital_utilization"]),
                    "average_holding_count": self._mean(group["holding_count"]),
                    "max_holding_count": self._max(group["holding_count"]),
                    "monthly_profit": float(sell_profit),
                    "monthly_win": bool(sell_profit > 0),
                    "buy_count": int(month_trades.get("action", pd.Series(dtype=str)).astype(str).eq("BUY").sum()) if not month_trades.empty else 0,
                    "sell_count": int(month_sells.get("action", pd.Series(dtype=str)).astype(str).eq("SELL").sum()) if not month_sells.empty else 0,
                    "total_buy_amount": self._amount_sum(month_trades, "BUY"),
                    "total_sell_amount": self._amount_sum(month_sells, "SELL"),
                    "pm_low_score_skip_count": skip_counts.get("pm_low_score_skip", 0),
                    "per_code_exposure_cap_scaled_below_round_lot_count": skip_counts.get("per_code_exposure_cap_scaled_below_round_lot", 0),
                    "selected_but_not_affordable_count": skip_counts.get("selected_but_not_affordable", 0),
                    "insufficient_available_cash_count": skip_counts.get("insufficient_available_cash", 0),
                    "daily_buy_limit_scaled_below_round_lot_count": skip_counts.get("daily_buy_limit_scaled_below_round_lot", 0),
                    "target_exposure_limit_count": skip_counts.get("target_exposure_limit", 0),
                }
            )
        return rows

    def _low_utilization_days(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        daily = payload["daily"]
        if daily.empty:
            return []
        low = daily[pd.to_numeric(daily["capital_utilization"], errors="coerce").fillna(0).lt(LOW_UTILIZATION_THRESHOLD)].copy()
        trades_raw = payload["trades_raw"]
        audit = payload["audit"]
        rows = []
        for _, day in low.iterrows():
            date = str(day["date"])
            day_audit = self._rows_on_date(audit, "entry_date", date)
            day_trades_entry = self._rows_on_date(trades_raw, "entry_date", date)
            day_trades_exit = self._rows_on_date(trades_raw, "exit_date", date)
            skip_counts = self._skip_counts(day_audit)
            buy_count = int(day_trades_entry.get("action", pd.Series(dtype=str)).astype(str).eq("BUY").sum()) if not day_trades_entry.empty else 0
            sell_count = int(day_trades_exit.get("action", pd.Series(dtype=str)).astype(str).eq("SELL").sum()) if not day_trades_exit.empty else 0
            rows.append(
                {
                    "date": date,
                    "capital_utilization": self._float(day.get("capital_utilization")),
                    "cash": self._float(day.get("cash")),
                    "market_value": self._float(day.get("market_value")),
                    "total_assets": self._float(day.get("total_assets")),
                    "holding_count": int(day.get("holding_count") or 0),
                    "candidate_count": int(len(day_audit)),
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "skip_reason_counts": "; ".join(f"{k}:{v}" for k, v in sorted(skip_counts.items())) if skip_counts else "",
                    "dominant_reason": self._dominant_reason(skip_counts, candidate_count=len(day_audit), buy_count=buy_count, sell_count=sell_count),
                }
            )
        return rows

    def _skip_reason_utilization(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        audit = payload["audit"]
        daily = payload["daily"]
        if audit.empty or daily.empty or "skip_reason" not in audit.columns:
            return []
        skipped = audit[audit.get("decision", pd.Series(dtype=str)).astype(str).eq("SKIP")].copy()
        if skipped.empty:
            return []
        skipped["entry_date"] = pd.to_datetime(skipped["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        merged = skipped.merge(
            daily[["date", "capital_utilization", "cash", "market_value", "total_assets", "holding_count"]],
            left_on="entry_date",
            right_on="date",
            how="left",
        )
        rows = []
        for reason, group in merged.groupby(merged["skip_reason"].fillna("<NA>").astype(str), sort=True):
            rows.append(
                {
                    "skip_reason": reason,
                    "count": int(len(group)),
                    "average_capital_utilization_on_day": self._mean(group["capital_utilization"]),
                    "median_capital_utilization_on_day": self._median(group["capital_utilization"]),
                    "average_cash_on_day": self._mean(group["cash"]),
                    "average_market_value_on_day": self._mean(group["market_value"]),
                    "average_total_assets_on_day": self._mean(group["total_assets"]),
                    "average_holding_count_on_day": self._mean(group["holding_count"]),
                }
            )
        return sorted(rows, key=lambda row: row["count"], reverse=True)

    def _bottleneck_flags(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        low_days = self._low_utilization_days(payload)
        low_count = len(low_days)
        reason_counts: dict[str, int] = {}
        for row in low_days:
            reason = str(row.get("dominant_reason") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        low_ratio = lambda key: reason_counts.get(key, 0) / low_count if low_count else 0.0
        all_skip_counts = self._skip_counts(payload["audit"])
        lot_count = all_skip_counts.get("daily_buy_limit_scaled_below_round_lot", 0) + all_skip_counts.get("per_code_exposure_cap_scaled_below_round_lot", 0)
        flags = [
            ("low_utilization_due_to_pm_skip", low_ratio("candidates_all_low_pm_skipped") > 0.30 or all_skip_counts.get("pm_low_score_skip", 0) > max(20, low_count * 0.5), reason_counts),
            ("low_utilization_due_to_per_code_cap", low_ratio("per_code_cap_blocked") > 0.20 or all_skip_counts.get("per_code_exposure_cap_scaled_below_round_lot", 0) > 0, reason_counts),
            ("low_utilization_due_to_lot_size", lot_count > 20, {"lot_related_skip_count": lot_count}),
            ("low_utilization_due_to_cash_shortage", low_ratio("insufficient_cash") > 0.20 or all_skip_counts.get("insufficient_available_cash", 0) > 20, all_skip_counts),
            ("low_utilization_due_to_candidate_shortage", low_ratio("no_candidates") + low_ratio("exit_only_day") > 0.30, reason_counts),
            ("low_utilization_due_to_target_exposure", low_ratio("target_exposure_limit") > 0.20 or all_skip_counts.get("target_exposure_limit", 0) > 0, all_skip_counts),
        ]
        return [{"flag": name, "value": bool(value), "detail": str(detail)} for name, value, detail in flags]

    def _next_action_candidates(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        flags = {row["flag"]: bool(row["value"]) for row in result["bottleneck_flags"]}
        rows = []
        priority = 1
        if flags.get("low_utilization_due_to_pm_skip"):
            rows.append({"priority": priority, "candidate": "per-code cap 0.30維持 + 0.8 multiplier帯/low-score skip閾値の再調整", "reason": "低稼働日にPM skipが多い可能性があるため"})
            priority += 1
        if flags.get("low_utilization_due_to_candidate_shortage"):
            rows.append({"priority": priority, "candidate": "per-code cap 0.30維持 + candidate pool拡大", "reason": "候補不足/exit-only日が稼働率を押し下げている可能性があるため"})
            priority += 1
        if flags.get("low_utilization_due_to_target_exposure"):
            rows.append({"priority": priority, "candidate": "per-code cap 0.30維持 + target portfolio exposure拡大", "reason": "target exposure系skipが稼働率上限になっている可能性があるため"})
            priority += 1
        if flags.get("low_utilization_due_to_cash_shortage"):
            rows.append({"priority": priority, "candidate": "per-code cap 0.30維持 + daily_buy_limitを総資産連動化", "reason": "cash/daily limit制約の影響を分離するため"})
            priority += 1
        if flags.get("low_utilization_due_to_per_code_cap"):
            rows.append({"priority": priority, "candidate": "per-code cap 0.30維持 + per-code cap発動時の次候補補充", "reason": "capで止まった資金を別候補へ回せるか確認するため"})
            priority += 1
        if not rows:
            rows.append({"priority": 1, "candidate": "資金稼働率は無理に上げず現状維持", "reason": "利益/PF/DDが良く、明確な単一ボトルネックが弱い場合は過剰最適化を避ける"})
        rows.append({"priority": priority + len(rows), "candidate": "full realistic backtest before adoption", "reason": "今回の監査はロジック変更なし。改善案は別profileで検証する"})
        return rows

    def _audit_notes(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        audit = payload["audit"]
        cap_cols = [column for column in audit.columns if column.startswith("pm_per_code_")] if not audit.empty else []
        return [
            {"item": "cap_audit_columns_available", "value": bool(cap_cols)},
            {"item": "cap_audit_columns", "value": ", ".join(cap_cols)},
            {"item": "selected_count_in_day_used", "value": False},
            {"item": "api_refetch", "value": False},
            {"item": "trading_logic_changed", "value": False},
        ]

    def _dominant_reason(self, skip_counts: dict[str, int], *, candidate_count: int, buy_count: int, sell_count: int) -> str:
        if skip_counts:
            top_reason, _ = max(skip_counts.items(), key=lambda item: item[1])
            mapping = {
                "pm_low_score_skip": "candidates_all_low_pm_skipped",
                "per_code_exposure_cap": "per_code_cap_blocked",
                "per_code_exposure_cap_scaled_below_round_lot": "below_round_lot_after_scaling",
                "insufficient_available_cash": "insufficient_cash",
                "selected_but_not_affordable": "selected_but_not_affordable",
                "daily_buy_limit_scaled_below_round_lot": "below_round_lot_after_scaling",
                "target_exposure_limit": "target_exposure_limit",
                "max_positions_limit": "max_positions_limit",
            }
            return mapping.get(top_reason, top_reason or "unknown")
        if candidate_count == 0 and buy_count == 0 and sell_count > 0:
            return "exit_only_day"
        if candidate_count == 0:
            return "no_candidates"
        return "unknown"

    def _skip_counts(self, audit: pd.DataFrame) -> dict[str, int]:
        if audit.empty or "skip_reason" not in audit.columns:
            return {}
        skipped = audit[audit.get("decision", pd.Series(dtype=str)).astype(str).eq("SKIP")].copy()
        return {str(key): int(value) for key, value in skipped["skip_reason"].fillna("<NA>").astype(str).value_counts().items()}

    def _rows_on_date(self, df: pd.DataFrame, column: str, date: str) -> pd.DataFrame:
        if df.empty or column not in df.columns:
            return pd.DataFrame()
        dates = pd.to_datetime(df[column], errors="coerce").dt.strftime("%Y-%m-%d")
        return df[dates.eq(date)].copy()

    def _rows_in_month(self, df: pd.DataFrame, column: str, month: str) -> pd.DataFrame:
        if df.empty or column not in df.columns:
            return pd.DataFrame()
        months = pd.to_datetime(df[column], errors="coerce").dt.strftime("%Y-%m")
        return df[months.eq(month)].copy()

    def _amount_sum(self, trades: pd.DataFrame, action: str) -> float:
        if trades.empty or "action" not in trades.columns:
            return 0.0
        subset = trades[trades["action"].astype(str).eq(action)]
        if subset.empty:
            return 0.0
        if action == "SELL":
            amount = pd.to_numeric(subset.get("exit_price"), errors="coerce").fillna(0.0) * pd.to_numeric(subset.get("shares"), errors="coerce").fillna(0.0)
        else:
            amount = pd.to_numeric(subset.get("entry_price"), errors="coerce").fillna(0.0) * pd.to_numeric(subset.get("shares"), errors="coerce").fillna(0.0)
        return float(amount.sum())

    def _float(self, value: Any) -> float:
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _mean(self, series: pd.Series) -> float | None:
        data = pd.to_numeric(series, errors="coerce").dropna()
        return float(data.mean()) if not data.empty else None

    def _median(self, series: pd.Series) -> float | None:
        data = pd.to_numeric(series, errors="coerce").dropna()
        return float(data.median()) if not data.empty else None

    def _min(self, series: pd.Series) -> float | None:
        data = pd.to_numeric(series, errors="coerce").dropna()
        return float(data.min()) if not data.empty else None

    def _max(self, series: pd.Series) -> float | None:
        data = pd.to_numeric(series, errors="coerce").dropna()
        return float(data.max()) if not data.empty else None

    def _quantile(self, series: pd.Series, quantile: float) -> float | None:
        data = pd.to_numeric(series, errors="coerce").dropna()
        return float(data.quantile(quantile)) if not data.empty else None
