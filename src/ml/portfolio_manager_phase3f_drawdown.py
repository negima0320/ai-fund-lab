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


INITIAL_CAPITAL = 1_000_000.0


@dataclass(frozen=True)
class PortfolioManagerPhase3FPaths:
    markdown: Path
    json: Path


class PortfolioManagerPhase3FDrawdownAudit:
    def __init__(
        self,
        root: str | Path = ".",
        baseline_profile: str = BASELINE_PROFILE,
        phase3d_profile: str = PHASE3D_PROFILE,
        phase3e_profile: str = PHASE3E_PROFILE,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.baseline_profile = baseline_profile
        self.phase3d_profile = phase3d_profile
        self.phase3e_profile = phase3e_profile
        self.period = period
        self.detail = PortfolioManagerPhase3DDetailAudit(
            root=self.root,
            baseline_profile=baseline_profile,
            phase3d_profile=phase3d_profile,
            period=period,
        )
        self.report_dir = self.root / "reports" / "ml"

    def build(self) -> dict[str, Any]:
        baseline = self._profile_payload(self.baseline_profile)
        phase3d = self._profile_payload(self.phase3d_profile)
        phase3e = self._profile_payload(self.phase3e_profile)
        phase3d_dd = self._drawdown_window(phase3d["summary_raw"])
        phase3e_dd = self._drawdown_window(phase3e["summary_raw"])
        phase3e_period_trades = self._trades_in_period(phase3e["trades"], phase3e_dd["start_date"], phase3e_dd["trough_date"])
        phase3d_same_period_trades = self._trades_in_period(phase3d["trades"], phase3e_dd["start_date"], phase3e_dd["trough_date"])
        return {
            "purpose": "Portfolio Manager AI Phase 3-F v2_76 drawdown root cause audit",
            "period": self.period,
            "profiles": {
                "baseline": self.baseline_profile,
                "phase3d": self.phase3d_profile,
                "phase3e": self.phase3e_profile,
            },
            "constraints": {
                "api_refetch": False,
                "openai_api": False,
                "historical_predictions_source": "data/ml/walk_forward_predictions/",
                "current_model_historical_regeneration": False,
                "selected_count_in_day_used": False,
                "trading_logic_changed": False,
            },
            "drawdown_windows": {
                "phase3d": self._drawdown_summary(phase3d, phase3d_dd),
                "phase3e": self._drawdown_summary(phase3e, phase3e_dd),
            },
            "phase3e_drawdown_code_contribution": self._group_contribution(phase3e_period_trades, "code"),
            "phase3e_drawdown_pm_multiplier_contribution": self._group_contribution(phase3e_period_trades, "pm_multiplier"),
            "phase3e_drawdown_pm_score_band_contribution": self._score_band_contribution(phase3e_period_trades),
            "phase3e_drawdown_daily_detail": self._daily_detail(phase3e, phase3e_dd),
            "same_period_comparison": self._same_period_comparison(phase3d, phase3e, phase3e_dd),
            "root_cause_flags": self._root_cause_flags(phase3e, phase3e_dd, phase3e_period_trades),
            "guard_recommendations": [],
        }

    def finalize(self, result: dict[str, Any]) -> dict[str, Any]:
        result["guard_recommendations"] = self._guard_recommendations(result)
        return result

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase3FPaths:
        result = self.finalize(result)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase3f_drawdown_audit_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerPhase3FPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        table = self.detail._table
        lines = [
            "# Portfolio Manager AI Phase 3-F Drawdown Root Cause Audit",
            "",
            "## Scope",
            "",
            f"- period: `{result['period']}`",
            f"- baseline: `{result['profiles']['baseline']}`",
            f"- phase3d / v2_75: `{result['profiles']['phase3d']}`",
            f"- phase3e / v2_76: `{result['profiles']['phase3e']}`",
            "- no API refetch",
            "- no OpenAI API",
            "- no current model historical regeneration",
            "- no trading logic changes",
            "- `selected_count_in_day` used: `False`",
            "",
            "## Max Drawdown Windows",
            "",
            table(
                [
                    {"profile": "v2_75", **result["drawdown_windows"]["phase3d"]},
                    {"profile": "v2_76", **result["drawdown_windows"]["phase3e"]},
                ],
                [
                    "profile",
                    "max_drawdown",
                    "start_date",
                    "trough_date",
                    "recovery_date",
                    "duration_days",
                    "start_assets",
                    "trough_assets",
                    "recovery_assets",
                    "drawdown_amount",
                    "period_profit",
                    "buy_count",
                    "sell_count",
                    "average_capital_utilization",
                    "max_capital_utilization",
                    "average_holding_count",
                    "max_holding_count",
                ],
            ),
            "",
            "## v2_76 Drawdown Code Contribution",
            "",
            table(
                result["phase3e_drawdown_code_contribution"][:20],
                ["group", "trade_count", "net_profit", "gross_profit", "gross_loss", "win_rate", "profit_factor", "average_profit", "max_single_loss", "total_buy_amount", "return_on_buy_amount"],
            ),
            "",
            "## v2_76 Drawdown PM Multiplier Contribution",
            "",
            table(
                result["phase3e_drawdown_pm_multiplier_contribution"],
                ["group", "trade_count", "net_profit", "gross_profit", "gross_loss", "win_rate", "profit_factor", "average_profit", "max_single_loss", "total_buy_amount", "return_on_buy_amount"],
            ),
            "",
            "## v2_76 Drawdown PM Score Band Contribution",
            "",
            table(
                result["phase3e_drawdown_pm_score_band_contribution"],
                ["group", "trade_count", "net_profit", "gross_profit", "gross_loss", "win_rate", "profit_factor", "average_profit", "max_single_loss", "total_buy_amount", "return_on_buy_amount"],
            ),
            "",
            "## Same Period Comparison",
            "",
            table(
                [result["same_period_comparison"]],
                [
                    "period_start",
                    "period_end",
                    "phase3d_period_profit",
                    "phase3e_period_profit",
                    "delta",
                    "phase3d_max_drawdown_in_period",
                    "phase3e_max_drawdown_in_period",
                    "phase3d_average_capital_utilization",
                    "phase3e_average_capital_utilization",
                    "phase3d_average_holding_count",
                    "phase3e_average_holding_count",
                    "phase3d_trade_count",
                    "phase3e_trade_count",
                ],
            ),
            "",
            "## v2_76 Drawdown Daily Detail",
            "",
            table(
                result["phase3e_drawdown_daily_detail"],
                [
                    "date",
                    "total_assets",
                    "cash",
                    "market_value",
                    "capital_utilization",
                    "holding_count",
                    "daily_profit",
                    "cumulative_profit",
                    "new_buy_count",
                    "sell_count",
                    "buy_amount",
                    "sell_amount",
                    "realized_profit",
                    "top_losing_code",
                    "top_losing_amount",
                ],
            ),
            "",
            "## Root Cause Flags",
            "",
            table(result["root_cause_flags"], ["flag", "value", "detail"]),
            "",
            "## Guard Recommendations",
            "",
            table(result["guard_recommendations"], ["candidate", "priority", "reason"]),
            "",
            "## Notes",
            "",
            "- `cash`, `market_value`, and capital utilization are reconstructed from closed trades and the asset curve.",
            "- Unrealized profit is not directly available in the persisted daily curve, so daily detail focuses on realized exits and exposure approximation.",
        ]
        return "\n".join(lines)

    def _profile_payload(self, profile: str) -> dict[str, Any]:
        payload = self.detail._profile_payload(profile)
        backtest_dir = self.root / "logs" / "backtests" / profile / self.period
        payload["summary_raw"] = self._read_json(backtest_dir / "backtest_summary.json")
        return payload

    def _drawdown_window(self, summary: dict[str, Any]) -> dict[str, Any]:
        curve = pd.DataFrame(summary.get("daily_asset_curve") or [])
        if curve.empty:
            return self._empty_drawdown_window()
        curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
        curve["total_assets"] = pd.to_numeric(curve["total_assets"], errors="coerce")
        curve = curve.dropna(subset=["date", "total_assets"]).sort_values("date").reset_index(drop=True)
        if curve.empty:
            return self._empty_drawdown_window()
        running_peak = curve["total_assets"].cummax()
        drawdown = curve["total_assets"] / running_peak - 1.0
        trough_idx = int(drawdown.idxmin())
        peak_value = float(running_peak.iloc[trough_idx])
        start_candidates = curve.loc[:trough_idx]
        peak_rows = start_candidates[start_candidates["total_assets"].eq(peak_value)]
        start_idx = int(peak_rows.index[-1]) if not peak_rows.empty else 0
        recovery_idx = None
        for idx in range(trough_idx + 1, len(curve)):
            if float(curve.loc[idx, "total_assets"]) >= peak_value:
                recovery_idx = idx
                break
        recovery_assets = float(curve.loc[recovery_idx, "total_assets"]) if recovery_idx is not None else None
        recovery_date = curve.loc[recovery_idx, "date"].strftime("%Y-%m-%d") if recovery_idx is not None else None
        start_assets = float(curve.loc[start_idx, "total_assets"])
        trough_assets = float(curve.loc[trough_idx, "total_assets"])
        return {
            "start_idx": start_idx,
            "trough_idx": trough_idx,
            "recovery_idx": recovery_idx,
            "start_date": curve.loc[start_idx, "date"].strftime("%Y-%m-%d"),
            "trough_date": curve.loc[trough_idx, "date"].strftime("%Y-%m-%d"),
            "recovery_date": recovery_date,
            "duration_days": int((recovery_idx if recovery_idx is not None else trough_idx) - start_idx + 1),
            "start_assets": start_assets,
            "trough_assets": trough_assets,
            "recovery_assets": recovery_assets,
            "drawdown_amount": trough_assets - start_assets,
            "drawdown_rate": trough_assets / start_assets - 1.0 if start_assets else 0.0,
            "curve": curve,
        }

    def _empty_drawdown_window(self) -> dict[str, Any]:
        return {
            "start_idx": None,
            "trough_idx": None,
            "recovery_idx": None,
            "start_date": None,
            "trough_date": None,
            "recovery_date": None,
            "duration_days": 0,
            "start_assets": None,
            "trough_assets": None,
            "recovery_assets": None,
            "drawdown_amount": None,
            "drawdown_rate": None,
            "curve": pd.DataFrame(),
        }

    def _drawdown_summary(self, payload: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
        start = window["start_date"]
        trough = window["trough_date"]
        trades = self._trades_in_period(payload["trades"], start, trough)
        daily = self._daily_detail(payload, window)
        return {
            "max_drawdown": float(payload["summary"].get("max_drawdown") or window.get("drawdown_rate") or 0.0),
            "start_date": start,
            "trough_date": trough,
            "recovery_date": window["recovery_date"],
            "duration_days": window["duration_days"],
            "start_assets": window["start_assets"],
            "trough_assets": window["trough_assets"],
            "recovery_assets": window["recovery_assets"],
            "drawdown_amount": window["drawdown_amount"],
            "drawdown_rate": window["drawdown_rate"],
            "period_profit": float(self._profit(trades).sum()),
            "buy_count": int(sum(row["new_buy_count"] for row in daily)),
            "sell_count": int(sum(row["sell_count"] for row in daily)),
            "average_capital_utilization": self._mean(daily, "capital_utilization"),
            "max_capital_utilization": self._max(daily, "capital_utilization"),
            "average_holding_count": self._mean(daily, "holding_count"),
            "max_holding_count": self._max(daily, "holding_count"),
        }

    def _trades_in_period(self, trades: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
        if trades.empty or not start or not end:
            return pd.DataFrame(columns=trades.columns)
        data = trades.copy()
        data["exit_date"] = pd.to_datetime(data["exit_date"], errors="coerce")
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return data[(data["exit_date"] >= start_ts) & (data["exit_date"] <= end_ts)].copy()

    def _daily_detail(self, payload: dict[str, Any], window: dict[str, Any]) -> list[dict[str, Any]]:
        curve = window["curve"]
        if curve.empty:
            return []
        start_idx = int(window["start_idx"])
        end_idx = int(window["trough_idx"])
        dates = curve.loc[start_idx:end_idx, ["date", "total_assets"]].copy()
        trades = payload["trades"].copy()
        if trades.empty:
            return []
        trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
        trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
        trades["entry_notional"] = self._buy_amount(trades)
        rows: list[dict[str, Any]] = []
        prev_assets: float | None = None
        for _, row in dates.iterrows():
            date = row["date"]
            total_assets = float(row["total_assets"])
            open_trades = trades[(trades["entry_date"] <= date) & (trades["exit_date"] >= date)]
            buys = trades[trades["entry_date"].eq(date)]
            sells = trades[trades["exit_date"].eq(date)]
            market_value = float(open_trades["entry_notional"].sum()) if not open_trades.empty else 0.0
            realized = float(self._profit(sells).sum()) if not sells.empty else 0.0
            losing = pd.DataFrame()
            if not sells.empty:
                sell_profits = sells.assign(_profit=self._profit(sells))
                losing = sell_profits[sell_profits["_profit"] < 0].sort_values("_profit").head(1)
            daily_profit = 0.0 if prev_assets is None else total_assets - prev_assets
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "total_assets": total_assets,
                    "cash": total_assets - market_value,
                    "market_value": market_value,
                    "capital_utilization": market_value / total_assets if total_assets else 0.0,
                    "holding_count": int(len(open_trades)),
                    "daily_profit": daily_profit,
                    "cumulative_profit": total_assets - INITIAL_CAPITAL,
                    "new_buy_count": int(len(buys)),
                    "sell_count": int(len(sells)),
                    "buy_amount": float(buys["entry_notional"].sum()) if not buys.empty else 0.0,
                    "sell_amount": float((pd.to_numeric(sells.get("exit_price"), errors="coerce").fillna(0.0) * pd.to_numeric(sells.get("shares"), errors="coerce").fillna(0.0)).sum()) if not sells.empty else 0.0,
                    "realized_profit": realized,
                    "unrealized_profit": None,
                    "top_losing_code": str(losing.iloc[0]["code"]) if not losing.empty else "",
                    "top_losing_amount": float(losing.iloc[0]["_profit"]) if not losing.empty else 0.0,
                }
            )
            prev_assets = total_assets
        return rows

    def _same_period_comparison(self, phase3d: dict[str, Any], phase3e: dict[str, Any], phase3e_window: dict[str, Any]) -> dict[str, Any]:
        start = phase3e_window["start_date"]
        trough = phase3e_window["trough_date"]
        d_trades = self._trades_in_period(phase3d["trades"], start, trough)
        e_trades = self._trades_in_period(phase3e["trades"], start, trough)
        d_window = self._window_for_period(phase3d["summary_raw"], start, trough)
        e_window = self._window_for_period(phase3e["summary_raw"], start, trough)
        d_daily = self._daily_detail(phase3d, d_window)
        e_daily = self._daily_detail(phase3e, e_window)
        d_profit = float(self._profit(d_trades).sum())
        e_profit = float(self._profit(e_trades).sum())
        return {
            "period_start": start,
            "period_end": trough,
            "phase3d_period_profit": d_profit,
            "phase3e_period_profit": e_profit,
            "delta": e_profit - d_profit,
            "phase3d_max_drawdown_in_period": self._period_drawdown(phase3d["summary_raw"], start, trough),
            "phase3e_max_drawdown_in_period": self._period_drawdown(phase3e["summary_raw"], start, trough),
            "phase3d_average_capital_utilization": self._mean(d_daily, "capital_utilization"),
            "phase3e_average_capital_utilization": self._mean(e_daily, "capital_utilization"),
            "phase3d_average_holding_count": self._mean(d_daily, "holding_count"),
            "phase3e_average_holding_count": self._mean(e_daily, "holding_count"),
            "phase3d_trade_count": int(len(d_trades)),
            "phase3e_trade_count": int(len(e_trades)),
        }

    def _window_for_period(self, summary: dict[str, Any], start: str | None, end: str | None) -> dict[str, Any]:
        curve = pd.DataFrame(summary.get("daily_asset_curve") or [])
        if curve.empty or not start or not end:
            return self._empty_drawdown_window()
        curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
        curve["total_assets"] = pd.to_numeric(curve["total_assets"], errors="coerce")
        subset = curve[(curve["date"] >= pd.Timestamp(start)) & (curve["date"] <= pd.Timestamp(end))].copy().reset_index(drop=True)
        if subset.empty:
            return self._empty_drawdown_window()
        return {
            "start_idx": 0,
            "trough_idx": len(subset) - 1,
            "recovery_idx": None,
            "start_date": subset.iloc[0]["date"].strftime("%Y-%m-%d"),
            "trough_date": subset.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "recovery_date": None,
            "duration_days": int(len(subset)),
            "start_assets": float(subset.iloc[0]["total_assets"]),
            "trough_assets": float(subset.iloc[-1]["total_assets"]),
            "recovery_assets": None,
            "drawdown_amount": float(subset.iloc[-1]["total_assets"] - subset.iloc[0]["total_assets"]),
            "drawdown_rate": float(subset.iloc[-1]["total_assets"] / subset.iloc[0]["total_assets"] - 1.0) if float(subset.iloc[0]["total_assets"]) else 0.0,
            "curve": subset,
        }

    def _period_drawdown(self, summary: dict[str, Any], start: str | None, end: str | None) -> float:
        curve = pd.DataFrame(summary.get("daily_asset_curve") or [])
        if curve.empty or not start or not end:
            return 0.0
        curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
        curve["total_assets"] = pd.to_numeric(curve["total_assets"], errors="coerce")
        subset = curve[(curve["date"] >= pd.Timestamp(start)) & (curve["date"] <= pd.Timestamp(end))].copy()
        if subset.empty:
            return 0.0
        running_peak = subset["total_assets"].cummax()
        return float((subset["total_assets"] / running_peak - 1.0).min())

    def _group_contribution(self, trades: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        if trades.empty or column not in trades.columns:
            return []
        data = trades.copy()
        if column == "pm_multiplier":
            data[column] = pd.to_numeric(data[column], errors="coerce")
        rows = [self._contribution_row(str(group), frame) for group, frame in data.dropna(subset=[column]).groupby(column)]
        return sorted(rows, key=lambda row: row["net_profit"])

    def _score_band_contribution(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty or "pm_score" not in trades.columns:
            return []
        data = trades.copy()
        score = pd.to_numeric(data["pm_score"], errors="coerce")
        data["pm_score_band"] = pd.cut(
            score,
            bins=[-999, -0.20, 0.0, 0.20, 0.40, 999],
            labels=["< -0.20", "-0.20 to 0", "0 to 0.20", "0.20 to 0.40", ">= 0.40"],
            include_lowest=True,
        )
        rows = [self._contribution_row(str(group), frame) for group, frame in data.dropna(subset=["pm_score_band"]).groupby("pm_score_band", observed=False)]
        return rows

    def _contribution_row(self, group: str, trades: pd.DataFrame) -> dict[str, Any]:
        profit = self._profit(trades)
        buy_amount = self._buy_amount(trades)
        gross_profit = float(profit[profit > 0].sum())
        gross_loss = float(profit[profit < 0].sum())
        return {
            "group": group,
            "code": group,
            "trade_count": int(len(trades)),
            "net_profit": float(profit.sum()),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "win_rate": float(profit.gt(0).mean()) if len(profit) else 0.0,
            "profit_factor": gross_profit / abs(gross_loss) if gross_loss else (float("inf") if gross_profit else 0.0),
            "average_profit": float(profit.mean()) if len(profit) else 0.0,
            "max_single_loss": float(profit.min()) if len(profit) else 0.0,
            "total_buy_amount": float(buy_amount.sum()),
            "return_on_buy_amount": float(profit.sum() / buy_amount.sum()) if float(buy_amount.sum()) else 0.0,
        }

    def _root_cause_flags(self, phase3e: dict[str, Any], window: dict[str, Any], trades: pd.DataFrame) -> list[dict[str, Any]]:
        code_rows = self._group_contribution(trades, "code")
        multiplier_rows = self._group_contribution(trades, "pm_multiplier")
        daily = self._daily_detail(phase3e, window)
        total_loss = abs(float(sum(row["net_profit"] for row in code_rows if row["net_profit"] < 0)))
        worst_code_loss = abs(float(min((row["net_profit"] for row in code_rows), default=0.0)))
        high_multiplier_loss = abs(float(sum(row["net_profit"] for row in multiplier_rows if row["group"] == "1.3" and row["net_profit"] < 0)))
        avg_util = self._mean(daily, "capital_utilization") or 0.0
        max_util = self._max(daily, "capital_utilization") or 0.0
        avg_hold = self._mean(daily, "holding_count") or 0.0
        max_hold = self._max(daily, "holding_count") or 0.0
        holding_avg = float(pd.to_numeric(trades.get("holding_days"), errors="coerce").dropna().mean()) if not trades.empty and "holding_days" in trades.columns else 0.0
        loss_trades = trades.assign(_profit=self._profit(trades))
        large_loss = loss_trades[loss_trades["_profit"].lt(0)].copy()
        large_loss_holding = float(pd.to_numeric(large_loss.get("holding_days"), errors="coerce").dropna().mean()) if not large_loss.empty and "holding_days" in large_loss.columns else 0.0
        losing_codes = int(sum(1 for row in code_rows if row["net_profit"] < 0))
        return [
            {
                "flag": "specific_code_concentration",
                "value": bool(total_loss and worst_code_loss / total_loss >= 0.30),
                "detail": f"worst_code_loss_share={worst_code_loss / total_loss if total_loss else 0:.4f}",
            },
            {
                "flag": "high_multiplier_concentration",
                "value": bool(total_loss and high_multiplier_loss / total_loss >= 0.50),
                "detail": f"multiplier_1.30_loss_share={high_multiplier_loss / total_loss if total_loss else 0:.4f}",
            },
            {
                "flag": "capital_utilization_spike",
                "value": bool(max_util - avg_util >= 0.20),
                "detail": f"avg={avg_util:.4f} max={max_util:.4f}",
            },
            {
                "flag": "holding_count_spike",
                "value": bool(max_hold - avg_hold >= 2.0),
                "detail": f"avg={avg_hold:.4f} max={max_hold:.4f}",
            },
            {
                "flag": "market_regime_like_drop",
                "value": bool(losing_codes >= 5 and not (total_loss and worst_code_loss / total_loss >= 0.30)),
                "detail": f"losing_codes={losing_codes}",
            },
            {
                "flag": "exit_delay_suspected",
                "value": bool(large_loss_holding > holding_avg + 1.0),
                "detail": f"loss_holding_avg={large_loss_holding:.4f} all_holding_avg={holding_avg:.4f}",
            },
        ]

    def _guard_recommendations(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        flags = {row["flag"]: bool(row["value"]) for row in result["root_cause_flags"]}
        rows = []
        if flags.get("high_multiplier_concentration"):
            rows.append({"candidate": "v2_76 + pm_multiplier 1.30 cap to 1.20", "priority": "high", "reason": "high multiplier losses dominate DD period"})
        if flags.get("capital_utilization_spike"):
            rows.append({"candidate": "v2_76 + high multiplier daily buy cap", "priority": "high", "reason": "capital utilization spikes during DD"})
        if flags.get("specific_code_concentration"):
            rows.append({"candidate": "v2_76 + per-code exposure cap", "priority": "high", "reason": "single code explains a large share of DD-period losses"})
        if flags.get("holding_count_spike"):
            rows.append({"candidate": "v2_76 + holding_count cap", "priority": "medium", "reason": "holding count spikes during DD"})
        if flags.get("market_regime_like_drop"):
            rows.append({"candidate": "v2_76 + DD-period market risk guard", "priority": "medium", "reason": "losses are broad rather than single-name concentrated"})
        if flags.get("exit_delay_suspected"):
            rows.append({"candidate": "v2_76 + Exit AI strengthening", "priority": "medium", "reason": "large loss trades appear held longer than average"})
        rows.append({"candidate": "v2_75維持", "priority": "baseline", "reason": "v2_75 has much better max DD and remains the defensive reference"})
        return rows

    def _profit(self, trades: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(trades.get("net_profit"), errors="coerce").fillna(0.0)

    def _buy_amount(self, trades: pd.DataFrame) -> pd.Series:
        if trades.empty:
            return pd.Series(dtype=float)
        if "final_amount" in trades.columns:
            amount = pd.to_numeric(trades["final_amount"], errors="coerce")
        else:
            amount = pd.Series([pd.NA] * len(trades), index=trades.index)
        fallback = pd.to_numeric(trades.get("entry_price"), errors="coerce").fillna(0.0) * pd.to_numeric(trades.get("shares"), errors="coerce").fillna(0.0)
        return amount.fillna(fallback).fillna(0.0)

    def _mean(self, rows: list[dict[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(sum(values) / len(values)) if values else None

    def _max(self, rows: list[dict[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return max(values) if values else None

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
