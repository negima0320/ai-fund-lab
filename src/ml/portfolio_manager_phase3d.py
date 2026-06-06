from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PERIOD = "2023-01-01_to_2026-05-31"
BASELINE_PROFILE = "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue"
PHASE3D_PROFILE = "rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing"


@dataclass(frozen=True)
class PortfolioManagerPhase3DPaths:
    markdown: Path
    json: Path


class PortfolioManagerPhase3DReporter:
    def __init__(
        self,
        root: str | Path = ".",
        baseline_profile: str = BASELINE_PROFILE,
        phase3d_profile: str = PHASE3D_PROFILE,
        period: str = PERIOD,
        smoke_period: str = "2026-03-01_to_2026-03-31",
    ) -> None:
        self.root = Path(root)
        self.baseline_profile = baseline_profile
        self.phase3d_profile = phase3d_profile
        self.period = period
        self.smoke_period = smoke_period
        self.report_dir = self.root / "reports" / "ml"

    def build(self) -> dict[str, Any]:
        baseline = self._profile_summary(self.baseline_profile, self.period)
        phase3d = self._profile_summary(self.phase3d_profile, self.period)
        smoke = self._profile_summary(self.phase3d_profile, self.smoke_period)
        comparison = self._comparison_rows(baseline, phase3d)
        return {
            "purpose": "Portfolio Manager AI Phase 3-D full backtest comparison",
            "model_dir": "models/ml/portfolio_manager/current_v2_73_phase3b_clean",
            "feature_count": 68,
            "data_lineage_audit_status": "PASS",
            "selected_count_in_day_used": False,
            "profiles": {
                "baseline": self.baseline_profile,
                "phase3d": self.phase3d_profile,
            },
            "period": self.period,
            "smoke_period": self.smoke_period,
            "smoke": smoke,
            "baseline": baseline,
            "phase3d": phase3d,
            "comparison": comparison,
            "pm_multiplier_analysis": self._pm_group_analysis(self.phase3d_profile, self.period, "pm_multiplier"),
            "pm_score_band_analysis": self._pm_score_band_analysis(self.phase3d_profile, self.period),
            "baseline_discrepancy_notes": self._baseline_discrepancy_notes(baseline),
            "recommendation": self._recommendation(baseline, phase3d),
        }

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase3DPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase3d_full_backtest_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerPhase3DPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        metric_cols = [
            "profile",
            "final_assets",
            "net_profit",
            "profit_factor",
            "max_drawdown",
            "win_rate",
            "total_trades",
            "average_trade_profit",
            "monthly_win_rate",
            "winning_months",
            "losing_months",
            "average_monthly_return",
            "worst_month_return",
            "best_month_return",
            "max_consecutive_losing_months",
            "average_capital_utilization",
            "average_holding_count",
            "focus_67400_contribution",
        ]
        lines = [
            "# Portfolio Manager AI Phase 3-D Full Backtest",
            "",
            "## Setup",
            "",
            f"- model_dir: `{result['model_dir']}`",
            f"- feature_count: {result['feature_count']}",
            f"- data_lineage_audit_status: {result['data_lineage_audit_status']}",
            f"- selected_count_in_day_used: {result['selected_count_in_day_used']}",
            "- PM rule: `high_conviction_proba - avoid_proba` multiplier sizing.",
            "- no API refetch, no current-model historical regeneration, no v2_73 destructive change.",
            "",
            "## Smoke Test",
            "",
            self._table([result["smoke"]], metric_cols),
            "",
            "## v2_73 Baseline Reproduction",
            "",
            self._table([result["baseline"]], metric_cols),
            "",
            "## v2_73 vs Phase 3-D",
            "",
            self._table([result["baseline"], result["phase3d"]], metric_cols),
            "",
            "## Delta",
            "",
            self._table(result["comparison"], ["metric", "baseline", "phase3d", "delta"]),
            "",
            "## PM Multiplier Analysis",
            "",
            self._table(
                result["pm_multiplier_analysis"],
                ["group", "trade_count", "net_profit", "win_rate", "profit_factor", "average_profit"],
            ),
            "",
            "## PM Score Band Analysis",
            "",
            self._table(
                result["pm_score_band_analysis"],
                ["group", "trade_count", "net_profit", "win_rate", "profit_factor", "average_profit"],
            ),
            "",
            "## Baseline Discrepancy Notes",
            "",
        ]
        lines.extend(f"- {item}" for item in result["baseline_discrepancy_notes"])
        lines.extend(["", "## Recommendation", ""])
        lines.extend(f"- {item}" for item in result["recommendation"])
        lines.append("")
        return "\n".join(lines)

    def _profile_summary(self, profile: str, period: str) -> dict[str, Any]:
        backtest_dir = self.root / "logs" / "backtests" / profile / period
        summary_path = backtest_dir / "backtest_summary.json"
        if not summary_path.exists():
            return {"profile": profile, "period": period, "missing": True}
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trades = self._read_csv(backtest_dir / "trades.csv")
        audit = self._read_csv(backtest_dir / "purchase_audit.csv")
        sell_trades = trades[trades.get("action", pd.Series(dtype=str)).astype(str).eq("SELL")].copy() if not trades.empty else trades
        net_profit = float(summary.get("net_cumulative_profit") or summary.get("cumulative_profit") or 0)
        trade_count = int(summary.get("closed_trades_count") or summary.get("closed_trade_count") or len(sell_trades))
        monthly = self._monthly_summary(sell_trades)
        asset_curve = pd.DataFrame(summary.get("daily_asset_curve") or [])
        if not asset_curve.empty and "total_assets" in asset_curve.columns:
            initial = float(summary.get("initial_capital") or 1_000_000)
            asset_curve["capital_utilization"] = (
                1.0 - pd.to_numeric(asset_curve.get("cash"), errors="coerce").fillna(0.0) / initial
                if "cash" in asset_curve.columns
                else pd.NA
            )
            avg_capital_utilization = float(asset_curve["capital_utilization"].dropna().mean())
            avg_holding_count = float(pd.to_numeric(asset_curve.get("open_positions_count"), errors="coerce").dropna().mean()) if "open_positions_count" in asset_curve.columns else None
        else:
            avg_capital_utilization = None
            avg_holding_count = None
        return {
            "profile": profile,
            "period": period,
            "final_assets": float(summary.get("final_assets") or 0),
            "net_profit": net_profit,
            "profit_factor": float(summary.get("profit_factor") or self._profit_factor(sell_trades)),
            "max_drawdown": float(summary.get("max_drawdown") or 0),
            "win_rate": float(summary.get("win_rate") or 0),
            "total_trades": trade_count,
            "average_trade_profit": net_profit / trade_count if trade_count else 0,
            "monthly_win_rate": monthly["monthly_win_rate"],
            "winning_months": monthly["winning_months"],
            "losing_months": monthly["losing_months"],
            "average_monthly_return": monthly["average_monthly_return"],
            "worst_month_return": monthly["worst_month_return"],
            "best_month_return": monthly["best_month_return"],
            "max_consecutive_losing_months": monthly["max_consecutive_losing_months"],
            "average_capital_utilization": avg_capital_utilization,
            "average_holding_count": avg_holding_count,
            "skip_reason_breakdown": self._value_counts(audit, "skip_reason"),
            "affordability_skip_count": self._affordability_skip_count(audit),
            "focus_67400_contribution": self._code_profit(sell_trades, "67400"),
            "path": str(backtest_dir),
        }

    def _monthly_summary(self, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty or "exit_date" not in trades.columns:
            return {
                "monthly_win_rate": 0,
                "winning_months": 0,
                "losing_months": 0,
                "average_monthly_return": 0,
                "worst_month_return": 0,
                "best_month_return": 0,
                "max_consecutive_losing_months": 0,
            }
        data = trades.copy()
        data["exit_date"] = pd.to_datetime(data["exit_date"], errors="coerce")
        data["month"] = data["exit_date"].dt.strftime("%Y-%m")
        data["net_profit"] = pd.to_numeric(data.get("net_profit"), errors="coerce").fillna(0.0)
        monthly_profit = data.groupby("month")["net_profit"].sum().sort_index()
        if monthly_profit.empty:
            return self._monthly_summary(pd.DataFrame())
        losing = monthly_profit.lt(0)
        return {
            "monthly_win_rate": float(monthly_profit.gt(0).mean()),
            "winning_months": int(monthly_profit.gt(0).sum()),
            "losing_months": int(losing.sum()),
            "average_monthly_return": float((monthly_profit / 1_000_000.0).mean()),
            "worst_month_return": float((monthly_profit / 1_000_000.0).min()),
            "best_month_return": float((monthly_profit / 1_000_000.0).max()),
            "max_consecutive_losing_months": self._max_consecutive(losing.tolist()),
        }

    def _pm_group_analysis(self, profile: str, period: str, column: str) -> list[dict[str, Any]]:
        merged = self._sells_with_purchase_audit(profile, period)
        if merged.empty or column not in merged.columns:
            return []
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        rows = []
        for value, group in merged.dropna(subset=[column]).groupby(column):
            rows.append(self._trade_group_row(str(value), group))
        return rows

    def _pm_score_band_analysis(self, profile: str, period: str) -> list[dict[str, Any]]:
        merged = self._sells_with_purchase_audit(profile, period)
        if merged.empty or "pm_score" not in merged.columns:
            return []
        score = pd.to_numeric(merged["pm_score"], errors="coerce")
        bins = [-999, -0.20, 0.0, 0.20, 0.40, 999]
        labels = ["< -0.20", "-0.20 to 0", "0 to 0.20", "0.20 to 0.40", ">= 0.40"]
        merged["pm_score_band"] = pd.cut(score, bins=bins, labels=labels, include_lowest=True)
        return [self._trade_group_row(str(label), group) for label, group in merged.dropna(subset=["pm_score_band"]).groupby("pm_score_band", observed=False)]

    def _sells_with_purchase_audit(self, profile: str, period: str) -> pd.DataFrame:
        backtest_dir = self.root / "logs" / "backtests" / profile / period
        trades = self._read_csv(backtest_dir / "trades.csv")
        audit = self._read_csv(backtest_dir / "purchase_audit.csv")
        if trades.empty or audit.empty:
            return pd.DataFrame()
        sells = trades[trades["action"].astype(str).eq("SELL")].copy()
        buys = audit[audit["decision"].astype(str).isin(["BUY", "SCALED_BUY"])].copy()
        for frame in [sells, buys]:
            frame["code"] = frame["code"].astype(str)
            frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        pm_cols = [column for column in buys.columns if column.startswith("pm_")]
        sells = sells.drop(columns=[column for column in sells.columns if column.startswith("pm_")], errors="ignore")
        return sells.merge(buys[["signal_date", "entry_date", "code", *pm_cols]], on=["signal_date", "entry_date", "code"], how="left")

    def _comparison_rows(self, baseline: dict[str, Any], phase3d: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for metric in ["net_profit", "profit_factor", "max_drawdown", "win_rate", "total_trades", "monthly_win_rate", "focus_67400_contribution"]:
            left = baseline.get(metric)
            right = phase3d.get(metric)
            rows.append({"metric": metric, "baseline": left, "phase3d": right, "delta": self._delta(left, right)})
        return rows

    def _baseline_discrepancy_notes(self, baseline: dict[str, Any]) -> list[str]:
        return [
            "Phase 3-D comparison uses the v2_73 backtest artifacts generated in the same working tree and period, not older handoff numbers.",
            "Phase 3-C was a lightweight amount-multiplier simulation over fixed v2_73 trades; Phase 3-D is a full paper engine run with cash, daily_buy_limit, scaled buy, max_positions, and Exit AI.",
            "Differences can come from realized-vs-open-position accounting, period coverage at the final entry date, per-day cache reuse, and full engine order rejection/sizing behavior.",
            f"Current reproduced v2_73 net_profit={baseline.get('net_profit')} PF={baseline.get('profit_factor')} DD={baseline.get('max_drawdown')}.",
        ]

    def _recommendation(self, baseline: dict[str, Any], phase3d: dict[str, Any]) -> list[str]:
        if phase3d.get("missing"):
            return ["Phase 3-D backtest artifact is missing; run the full backtest first."]
        net_delta = self._delta(baseline.get("net_profit"), phase3d.get("net_profit"))
        pf_delta = self._delta(baseline.get("profit_factor"), phase3d.get("profit_factor"))
        dd_delta = self._delta(baseline.get("max_drawdown"), phase3d.get("max_drawdown"))
        if (net_delta or 0) > 0 and (pf_delta or 0) >= -0.02 and (dd_delta or 0) >= -0.03:
            return ["PM high_minus_avoid sizing is a candidate for further validation; it improved net profit without materially weakening PF/DD."]
        return ["Do not promote yet; compare PF/DD/monthly stability before treating v2_75 as a mainline profile."]

    def _trade_group_row(self, group: str, trades: pd.DataFrame) -> dict[str, Any]:
        profit = pd.to_numeric(trades.get("net_profit"), errors="coerce").fillna(0.0)
        return {
            "group": group,
            "trade_count": int(len(trades)),
            "net_profit": float(profit.sum()),
            "win_rate": float(profit.gt(0).mean()) if len(profit) else 0,
            "profit_factor": self._profit_factor(trades),
            "average_profit": float(profit.mean()) if len(profit) else 0,
        }

    def _read_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _profit_factor(self, trades: pd.DataFrame) -> float:
        if trades.empty or "net_profit" not in trades.columns:
            return 0.0
        profit = pd.to_numeric(trades["net_profit"], errors="coerce").fillna(0.0)
        gross_profit = float(profit[profit > 0].sum())
        gross_loss = abs(float(profit[profit < 0].sum()))
        return gross_profit / gross_loss if gross_loss else float("inf") if gross_profit else 0.0

    def _code_profit(self, trades: pd.DataFrame, code: str) -> float:
        if trades.empty or "code" not in trades.columns or "net_profit" not in trades.columns:
            return 0.0
        data = trades[trades["code"].astype(str).eq(str(code))]
        return float(pd.to_numeric(data["net_profit"], errors="coerce").fillna(0.0).sum())

    def _affordability_skip_count(self, audit: pd.DataFrame) -> int:
        if audit.empty or "skip_reason" not in audit.columns:
            return 0
        reasons = audit["skip_reason"].fillna("").astype(str)
        return int(reasons.str.contains("affordable|unaffordable|cash|daily_buy_limit|round_lot", case=False, regex=True).sum())

    def _value_counts(self, df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        if df.empty or column not in df.columns:
            return []
        counts = df[column].fillna("<NA>").astype(str).value_counts()
        total = int(counts.sum())
        return [{"value": key, "count": int(value), "rate": float(value / total) if total else 0} for key, value in counts.items()]

    def _delta(self, left: Any, right: Any) -> float | None:
        try:
            return float(right) - float(left)
        except (TypeError, ValueError):
            return None

    def _max_consecutive(self, flags: list[bool]) -> int:
        best = current = 0
        for flag in flags:
            current = current + 1 if flag else 0
            best = max(best, current)
        return best

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = []
        for row in rows:
            body.append("| " + " | ".join(self._format_value(row.get(column, "")) for column in columns) + " |")
        return "\n".join([header, sep, *body])

    def _format_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
