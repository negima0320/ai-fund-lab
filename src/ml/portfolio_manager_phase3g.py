from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_phase3d_detail_audit import (
    BASELINE_PROFILE,
    FOCUS_CODE,
    PERIOD,
    PHASE3D_PROFILE,
    PortfolioManagerPhase3DDetailAudit,
)
from ml.portfolio_manager_phase3e import PHASE3E_PROFILE
from ml.portfolio_manager_phase3f_drawdown import PortfolioManagerPhase3FDrawdownAudit


PHASE3G_PROFILE = "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap"
SECONDARY_FOCUS_CODE = "62540"
CAP_RATE_PROFILES = {
    "0.15": "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_015",
    "0.20": PHASE3G_PROFILE,
    "0.25": "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_025",
    "0.30": "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030",
}
DD_FOCUS_START = "2025-09-26"
DD_FOCUS_TROUGH = "2025-09-29"
DD_FOCUS_RECOVERY = "2025-10-24"


@dataclass(frozen=True)
class PortfolioManagerPhase3GPaths:
    markdown: Path
    json: Path


class PortfolioManagerPhase3GReporter:
    def __init__(
        self,
        root: str | Path = ".",
        baseline_profile: str = BASELINE_PROFILE,
        phase3d_profile: str = PHASE3D_PROFILE,
        phase3e_profile: str = PHASE3E_PROFILE,
        phase3g_profile: str = PHASE3G_PROFILE,
        cap_rate_profiles: dict[str, str] | None = None,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.baseline_profile = baseline_profile
        self.phase3d_profile = phase3d_profile
        self.phase3e_profile = phase3e_profile
        self.phase3g_profile = phase3g_profile
        self.cap_rate_profiles = cap_rate_profiles or CAP_RATE_PROFILES
        self.period = period
        self.audit = PortfolioManagerPhase3DDetailAudit(root=self.root, baseline_profile=baseline_profile, phase3d_profile=phase3d_profile, period=period)
        self.drawdown_audit = PortfolioManagerPhase3FDrawdownAudit(
            root=self.root,
            baseline_profile=baseline_profile,
            phase3d_profile=phase3d_profile,
            phase3e_profile=phase3e_profile,
            period=period,
        )
        self.report_dir = self.root / "reports" / "ml"

    def build(self) -> dict[str, Any]:
        profiles = {
            "v2_73": self.baseline_profile,
            "v2_75": self.phase3d_profile,
            "v2_76": self.phase3e_profile,
            "v2_77": self.phase3g_profile,
        }
        payloads = {label: self.audit._profile_payload(profile) for label, profile in profiles.items()}
        cap_payloads = {rate: self.audit._profile_payload(profile) for rate, profile in self.cap_rate_profiles.items()}
        phase3f = self.drawdown_audit.build()
        return {
            "purpose": "Portfolio Manager AI Phase 3-G per-code exposure cap validation",
            "period": self.period,
            "constraints": {
                "api_refetch": False,
                "openai_api": False,
                "historical_predictions_source": "data/ml/walk_forward_predictions/",
                "current_model_historical_regeneration": False,
                "selected_count_in_day_used": False,
                "trading_logic_changed_for_existing_profiles": False,
            },
            "profiles": profiles,
            "summary_comparison": [payloads[label]["summary"] for label in ["v2_73", "v2_75", "v2_76", "v2_77"]],
            "cap_rate_sweep": [self._cap_rate_row(rate, payload) for rate, payload in cap_payloads.items()],
            "skip_reason_comparison": self._skip_reason_comparison(payloads),
            "code_concentration": {label: payload["code_summary"] for label, payload in payloads.items()},
            "focus_code_dependency": {
                "v2_75": self.audit._focus_code_dependency(payloads["v2_73"], payloads["v2_75"], FOCUS_CODE),
                "v2_76": self.audit._focus_code_dependency(payloads["v2_73"], payloads["v2_76"], FOCUS_CODE),
                "v2_77": self.audit._focus_code_dependency(payloads["v2_73"], payloads["v2_77"], FOCUS_CODE),
            },
            "secondary_focus_code_dependency": {
                "v2_75": self.audit._focus_code_dependency(payloads["v2_73"], payloads["v2_75"], SECONDARY_FOCUS_CODE),
                "v2_76": self.audit._focus_code_dependency(payloads["v2_73"], payloads["v2_76"], SECONDARY_FOCUS_CODE),
                "v2_77": self.audit._focus_code_dependency(payloads["v2_73"], payloads["v2_77"], SECONDARY_FOCUS_CODE),
            },
            "pm_multiplier_summary": {
                "v2_76": self.audit._pm_group_summary(payloads["v2_76"]["trades"], "pm_multiplier"),
                "v2_77": self.audit._pm_group_summary(payloads["v2_77"]["trades"], "pm_multiplier"),
            },
            "pm_score_band_summary": {
                "v2_76": self.audit._pm_score_band_summary(payloads["v2_76"]["trades"]),
                "v2_77": self.audit._pm_score_band_summary(payloads["v2_77"]["trades"]),
            },
            "capital_utilization_comparison": {label: payload["capital_utilization"] for label, payload in payloads.items()},
            "phase3f_reference": {
                "v2_76_max_drawdown_window": phase3f["drawdown_windows"]["phase3e"],
                "root_cause_flags": phase3f["root_cause_flags"],
            },
            "drawdown_focus_comparison": self._drawdown_focus_comparison(payloads),
            "promotion_judgement": self._promotion_judgement(payloads),
        }

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase3GPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase3g_per_code_cap_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerPhase3GPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 3-G Per-Code Exposure Cap",
            "",
            "## Scope",
            "",
            f"- period: `{result['period']}`",
            f"- baseline / v2_73: `{result['profiles']['v2_73']}`",
            f"- defensive reference / v2_75: `{result['profiles']['v2_75']}`",
            f"- aggressive reference / v2_76: `{result['profiles']['v2_76']}`",
            f"- new profile / v2_77: `{result['profiles']['v2_77']}`",
            "- no API refetch",
            "- no OpenAI API",
            "- no current model historical regeneration",
            "- `selected_count_in_day` used: `False`",
            "",
            "## Summary Comparison",
            "",
            self.audit._table(result["summary_comparison"], ["profile", "net_profit", "profit_factor", "max_drawdown", "win_rate", "total_trades"]),
            "",
            "## Cap Rate Sweep",
            "",
            self.audit._table(
                result["cap_rate_sweep"],
                [
                    "cap_rate",
                    "profile",
                    "net_profit",
                    "profit_factor",
                    "max_drawdown",
                    "win_rate",
                    "total_trades",
                    "monthly_win_rate",
                    "winning_months",
                    "losing_months",
                    "max_consecutive_losing_months",
                    "average_capital_utilization",
                    "median_capital_utilization",
                    "max_capital_utilization",
                    "average_holding_count",
                    "max_holding_count",
                    "per_code_exposure_cap_audit_available",
                    "per_code_exposure_cap_skip_count",
                    "per_code_exposure_cap_reduction_count",
                    "selected_but_not_affordable",
                    "insufficient_available_cash",
                    "daily_buy_limit_scaled_below_round_lot",
                    "pm_low_score_skip_count",
                ],
            ),
            "",
            "## v2_76 DD Focus Period Comparison",
            "",
            self.audit._table(
                result["drawdown_focus_comparison"],
                [
                    "profile",
                    "period_start",
                    "period_trough",
                    "period_recovery",
                    "start_assets",
                    "trough_assets",
                    "recovery_assets",
                    "max_drawdown_in_period",
                    "focus_code_exposure",
                    "focus_code_realized_profit",
                    "secondary_focus_code",
                    "secondary_focus_code_exposure",
                    "secondary_focus_code_realized_profit",
                    "average_capital_utilization",
                    "max_capital_utilization",
                    "average_holding_count",
                    "max_holding_count",
                ],
            ),
            "",
            f"## {FOCUS_CODE} Dependency",
            "",
            self.audit._table(
                [{"profile": key, **value} for key, value in result["focus_code_dependency"].items()],
                ["profile", "phase3d_profit", "phase3d_contribution_rate", "phase3d_excluding_profit", "phase3d_excluding_profit_factor", "phase3d_excluding_max_drawdown"],
            ),
            "",
            f"## {SECONDARY_FOCUS_CODE} Dependency",
            "",
            self.audit._table(
                [{"profile": key, **value} for key, value in result["secondary_focus_code_dependency"].items()],
                ["profile", "phase3d_profit", "phase3d_contribution_rate", "phase3d_excluding_profit", "phase3d_excluding_profit_factor", "phase3d_excluding_max_drawdown"],
            ),
            "",
            "## Code Concentration",
            "",
            self.audit._table(
                [{"profile": key, **value} for key, value in result["code_concentration"].items()],
                ["profile", "total_profit", "top1_contribution_rate", "top3_contribution_rate", "top5_contribution_rate", "worst_code", "worst_code_profit"],
            ),
            "",
            "## PM Multiplier Summary",
            "",
            "### v2_76",
            "",
            self.audit._table(result["pm_multiplier_summary"]["v2_76"], ["group", "trade_count", "total_buy_amount", "net_profit", "profit_factor", "win_rate", "average_profit"]),
            "",
            "### v2_77",
            "",
            self.audit._table(result["pm_multiplier_summary"]["v2_77"], ["group", "trade_count", "total_buy_amount", "net_profit", "profit_factor", "win_rate", "average_profit"]),
            "",
            "## PM Score Band Summary",
            "",
            "### v2_76",
            "",
            self.audit._table(result["pm_score_band_summary"]["v2_76"], ["group", "trade_count", "total_buy_amount", "net_profit", "profit_factor", "win_rate", "average_profit", "return_on_buy_amount"]),
            "",
            "### v2_77",
            "",
            self.audit._table(result["pm_score_band_summary"]["v2_77"], ["group", "trade_count", "total_buy_amount", "net_profit", "profit_factor", "win_rate", "average_profit", "return_on_buy_amount"]),
            "",
            "## Skip Reason Comparison",
            "",
            self.audit._table(result["skip_reason_comparison"], ["skip_reason", "v2_73", "v2_75", "v2_76", "v2_77"]),
            "",
            "## Capital Utilization",
            "",
            self.audit._table(
                [{"profile": key, **value} for key, value in result["capital_utilization_comparison"].items()],
                ["profile", "average_capital_utilization", "median_capital_utilization", "max_capital_utilization", "average_holding_count", "max_positions_days", "cash_idle_days"],
            ),
            "",
            "## Promotion Judgement",
            "",
            self.audit._table(result["promotion_judgement"], ["criterion", "passed", "detail"]),
            "",
        ]
        return "\n".join(lines)

    def _cap_rate_row(self, rate: str, payload: dict[str, Any]) -> dict[str, Any]:
        summary = payload["summary"]
        capital = payload["capital_utilization"]
        monthly = payload["monthly"]
        audit = payload["audit"]
        skip_counts = self._skip_counts(audit)
        cap_columns_available = (
            not audit.empty
            and "pm_per_code_cap_reason" in audit.columns
            and "pm_per_code_cap_reduced" in audit.columns
        )
        cap_reason = audit["pm_per_code_cap_reason"].fillna("").astype(str) if cap_columns_available else pd.Series(dtype=str)
        cap_reduced = self._truthy(audit["pm_per_code_cap_reduced"]) if cap_columns_available else pd.Series(dtype=bool)
        return {
            "cap_rate": rate,
            "profile": summary["profile"],
            **summary,
            "monthly_win_rate": self.audit._monthly_win_rate(monthly),
            "winning_months": int(sum(1 for row in monthly if float(row.get("monthly_profit") or 0) > 0)),
            "losing_months": int(sum(1 for row in monthly if float(row.get("monthly_profit") or 0) < 0)),
            "max_consecutive_losing_months": self._max_consecutive_losing_months(monthly),
            **capital,
            "max_holding_count": self._max_holding_count(payload),
            "per_code_exposure_cap_audit_available": cap_columns_available,
            "per_code_exposure_cap_skip_count": int(cap_reason.str.contains("per_code_exposure_cap", na=False).sum()) if cap_columns_available else None,
            "per_code_exposure_cap_reduction_count": int(cap_reduced.sum()) if cap_columns_available else None,
            "selected_but_not_affordable": skip_counts.get("selected_but_not_affordable", 0),
            "insufficient_available_cash": skip_counts.get("insufficient_available_cash", 0),
            "daily_buy_limit_scaled_below_round_lot": skip_counts.get("daily_buy_limit_scaled_below_round_lot", 0),
            "pm_low_score_skip_count": skip_counts.get("pm_low_score_skip", 0),
        }

    def _drawdown_focus_comparison(self, payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for label in ["v2_76", "v2_77"]:
            payload = payloads[label]
            rows.append(self._focus_period_row(label, payload))
        return rows

    def _focus_period_row(self, label: str, payload: dict[str, Any]) -> dict[str, Any]:
        curve = pd.DataFrame(payload.get("summary_raw", {}).get("daily_asset_curve") or [])
        # _profile_payload does not expose raw summary, so load directly.
        raw_summary = self.audit._read_json(self.root / "logs" / "backtests" / payload["profile"] / self.period / "backtest_summary.json")
        curve = pd.DataFrame(raw_summary.get("daily_asset_curve") or [])
        if curve.empty:
            return {"profile": label}
        curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
        start = pd.Timestamp(DD_FOCUS_START)
        trough = pd.Timestamp(DD_FOCUS_TROUGH)
        recovery = pd.Timestamp(DD_FOCUS_RECOVERY)
        period_curve = curve[(curve["date"] >= start) & (curve["date"] <= recovery)].copy()
        start_assets = self._asset_at_or_after(curve, start)
        trough_assets = self._asset_at_or_after(curve, trough)
        recovery_assets = self._asset_at_or_after(curve, recovery)
        max_dd = self._max_drawdown_between(curve, start, recovery)
        trades = payload["trades"].copy()
        if not trades.empty:
            trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
            trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
        exposure = self._code_exposure_between(trades, FOCUS_CODE, start, trough)
        focus_trades = trades[(trades["code"].astype(str).eq(FOCUS_CODE)) & (trades["exit_date"] >= start) & (trades["exit_date"] <= recovery)] if not trades.empty else pd.DataFrame()
        secondary_exposure = self._code_exposure_between(trades, SECONDARY_FOCUS_CODE, start, trough)
        secondary_trades = (
            trades[
                (trades["code"].astype(str).eq(SECONDARY_FOCUS_CODE))
                & (trades["exit_date"] >= start)
                & (trades["exit_date"] <= recovery)
            ]
            if not trades.empty
            else pd.DataFrame()
        )
        return {
            "profile": label,
            "period_start": DD_FOCUS_START,
            "period_trough": DD_FOCUS_TROUGH,
            "period_recovery": DD_FOCUS_RECOVERY,
            "start_assets": start_assets,
            "trough_assets": trough_assets,
            "recovery_assets": recovery_assets,
            "max_drawdown_in_period": max_dd,
            "focus_code_exposure": exposure,
            "focus_code_realized_profit": float(self.audit._profit(focus_trades).sum()) if not focus_trades.empty else 0.0,
            "secondary_focus_code": SECONDARY_FOCUS_CODE,
            "secondary_focus_code_exposure": secondary_exposure,
            "secondary_focus_code_realized_profit": float(self.audit._profit(secondary_trades).sum()) if not secondary_trades.empty else 0.0,
            **self._capital_between(payload, start, recovery),
        }

    def _skip_reason_comparison(self, payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        counts = {label: self._skip_counts(payload["audit"]) for label, payload in payloads.items()}
        reasons = sorted(set().union(*(set(item) for item in counts.values())))
        return [{"skip_reason": reason, **{label: counts[label].get(reason, 0) for label in ["v2_73", "v2_75", "v2_76", "v2_77"]}} for reason in reasons]

    def _promotion_judgement(self, payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        v75 = payloads["v2_75"]["summary"]
        v76 = payloads["v2_76"]["summary"]
        v77 = payloads["v2_77"]["summary"]
        focus77 = self.audit._focus_code_dependency(payloads["v2_73"], payloads["v2_77"], FOCUS_CODE)
        return [
            {"criterion": "net_profit_above_v2_75", "passed": v77["net_profit"] > v75["net_profit"], "detail": f"{v77['net_profit']} vs {v75['net_profit']}"},
            {"criterion": "profit_factor_above_v2_75", "passed": v77["profit_factor"] > v75["profit_factor"], "detail": f"{v77['profit_factor']} vs {v75['profit_factor']}"},
            {"criterion": "drawdown_better_than_v2_76", "passed": v77["max_drawdown"] > v76["max_drawdown"], "detail": f"{v77['max_drawdown']} vs {v76['max_drawdown']}"},
            {"criterion": "drawdown_near_v2_75", "passed": v77["max_drawdown"] >= v75["max_drawdown"] - 0.03, "detail": f"{v77['max_drawdown']} vs {v75['max_drawdown']}"},
            {"criterion": "focus_code_dependency_not_worse", "passed": abs(focus77["phase3d_contribution_rate"]) < 0.35, "detail": str(focus77["phase3d_contribution_rate"])},
        ]

    def _skip_counts(self, audit: pd.DataFrame) -> dict[str, int]:
        return self.audit._skip_counts(audit)

    def _truthy(self, series: pd.Series | None) -> pd.Series:
        if series is None:
            return pd.Series(dtype=bool)
        return series.astype(str).str.lower().isin({"true", "1", "yes"})

    def _max_consecutive_losing_months(self, rows: list[dict[str, Any]]) -> int:
        best = current = 0
        for row in rows:
            if float(row.get("monthly_profit") or 0) < 0:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    def _max_holding_count(self, payload: dict[str, Any]) -> int:
        raw_summary = self.audit._read_json(self.root / "logs" / "backtests" / payload["profile"] / self.period / "backtest_summary.json")
        curve = pd.DataFrame(raw_summary.get("daily_asset_curve") or [])
        trades = payload["trades"]
        if curve.empty or trades.empty:
            return 0
        dates = pd.to_datetime(curve["date"], errors="coerce")
        entries = pd.to_datetime(trades["entry_date"], errors="coerce")
        exits = pd.to_datetime(trades["exit_date"], errors="coerce")
        return int(max(((entries <= date) & (exits >= date)).sum() for date in dates))

    def _asset_at_or_after(self, curve: pd.DataFrame, date: pd.Timestamp) -> float | None:
        data = curve[curve["date"] >= date].sort_values("date")
        return float(data.iloc[0]["total_assets"]) if not data.empty else None

    def _max_drawdown_between(self, curve: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
        data = curve[(curve["date"] >= start) & (curve["date"] <= end)].sort_values("date")
        if data.empty:
            return None
        peak = float(data.iloc[0]["total_assets"])
        max_dd = 0.0
        for value in pd.to_numeric(data["total_assets"], errors="coerce").dropna():
            peak = max(peak, float(value))
            max_dd = min(max_dd, float(value) / peak - 1.0 if peak else 0.0)
        return max_dd

    def _code_exposure_between(self, trades: pd.DataFrame, code: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
        if trades.empty:
            return 0.0
        data = trades[trades["code"].astype(str).eq(str(code))].copy()
        open_trades = data[(data["entry_date"] <= end) & (data["exit_date"] >= start)]
        if open_trades.empty:
            return 0.0
        amount = pd.to_numeric(open_trades.get("final_amount"), errors="coerce").fillna(
            pd.to_numeric(open_trades.get("entry_price"), errors="coerce").fillna(0.0)
            * pd.to_numeric(open_trades.get("shares"), errors="coerce").fillna(0.0)
        )
        return float(amount.max()) if len(amount) else 0.0

    def _capital_between(self, payload: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
        raw_summary = self.audit._read_json(self.root / "logs" / "backtests" / payload["profile"] / self.period / "backtest_summary.json")
        curve = pd.DataFrame(raw_summary.get("daily_asset_curve") or [])
        trades = payload["trades"].copy()
        if curve.empty or trades.empty:
            return {
                "average_capital_utilization": None,
                "max_capital_utilization": None,
                "average_holding_count": None,
                "max_holding_count": None,
            }
        curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
        curve = curve[(curve["date"] >= start) & (curve["date"] <= end)]
        trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
        trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
        utilizations = []
        counts = []
        for _, row in curve.iterrows():
            date = row["date"]
            open_trades = trades[(trades["entry_date"] <= date) & (trades["exit_date"] >= date)]
            amount = pd.to_numeric(open_trades.get("final_amount"), errors="coerce").fillna(0.0)
            total_assets = float(row.get("total_assets") or 0)
            utilizations.append(float(amount.sum()) / total_assets if total_assets else 0.0)
            counts.append(int(len(open_trades)))
        return {
            "average_capital_utilization": float(pd.Series(utilizations).mean()) if utilizations else None,
            "max_capital_utilization": float(pd.Series(utilizations).max()) if utilizations else None,
            "average_holding_count": float(pd.Series(counts).mean()) if counts else None,
            "max_holding_count": int(max(counts)) if counts else 0,
        }
