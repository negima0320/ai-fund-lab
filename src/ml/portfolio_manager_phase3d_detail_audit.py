from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PERIOD = "2023-01-01_to_2026-05-31"
BASELINE_PROFILE = "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue"
PHASE3D_PROFILE = "rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing"
FOCUS_CODE = "67400"
INITIAL_CAPITAL = 1_000_000.0


@dataclass(frozen=True)
class DetailAuditPaths:
    markdown: Path
    json: Path


class PortfolioManagerPhase3DDetailAudit:
    def __init__(
        self,
        root: str | Path = ".",
        baseline_profile: str = BASELINE_PROFILE,
        phase3d_profile: str = PHASE3D_PROFILE,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.baseline_profile = baseline_profile
        self.phase3d_profile = phase3d_profile
        self.period = period
        self.report_dir = self.root / "reports" / "ml"

    def build(self) -> dict[str, Any]:
        baseline = self._profile_payload(self.baseline_profile)
        phase3d = self._profile_payload(self.phase3d_profile)
        return {
            "purpose": "Portfolio Manager AI Phase 3-D detail audit",
            "period": self.period,
            "baseline_profile": self.baseline_profile,
            "phase3d_profile": self.phase3d_profile,
            "pm_log_check": self._pm_log_check(phase3d),
            "baseline_summary": baseline["summary"],
            "phase3d_summary": phase3d["summary"],
            "monthly_comparison": self._monthly_comparison(baseline["monthly"], phase3d["monthly"]),
            "baseline_monthly": baseline["monthly"],
            "phase3d_monthly": phase3d["monthly"],
            "baseline_code_summary": baseline["code_summary"],
            "phase3d_code_summary": phase3d["code_summary"],
            "phase3d_top_codes": phase3d["top_codes"],
            "phase3d_bottom_codes": phase3d["bottom_codes"],
            "focus_code_dependency": self._focus_code_dependency(baseline, phase3d, FOCUS_CODE),
            "pm_multiplier_summary": self._pm_group_summary(phase3d["trades"], "pm_multiplier"),
            "pm_score_band_summary": self._pm_score_band_summary(phase3d["trades"]),
            "skip_reason_comparison": self._skip_reason_comparison(baseline["audit"], phase3d["audit"]),
            "capital_utilization_comparison": {
                "baseline": baseline["capital_utilization"],
                "phase3d": phase3d["capital_utilization"],
            },
            "promotion_judgement": self._promotion_judgement(baseline, phase3d),
        }

    def save(self, result: dict[str, Any]) -> DetailAuditPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase3d_detail_audit_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return DetailAuditPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 3-D Detail Audit",
            "",
            "## Scope",
            "",
            f"- period: `{result['period']}`",
            f"- baseline: `{result['baseline_profile']}`",
            f"- phase3d: `{result['phase3d_profile']}`",
            "- no API refetch",
            "- no current model historical regeneration",
            "- PM features are audited clean model outputs",
            "",
            "## PM Log Check",
            "",
            self._table(
                [result["pm_log_check"]],
                [
                    "all_trades_buy_rows",
                    "all_trades_buy_pm_non_null_rows",
                    "trades_csv_sell_rows",
                    "trades_csv_sell_pm_non_null_rows",
                    "audit_buy_rows",
                    "audit_trade_match_rate",
                    "pm_values_match",
                ],
            ),
            "",
            "## Summary",
            "",
            self._table([result["baseline_summary"], result["phase3d_summary"]], ["profile", "net_profit", "profit_factor", "max_drawdown", "win_rate", "total_trades"]),
            "",
            "## Monthly Comparison",
            "",
            self._table(result["monthly_comparison"], ["month", "baseline_profit", "phase3d_profit", "delta", "phase3d_cumulative_profit"]),
            "",
            "## Code Concentration",
            "",
            self._table([result["phase3d_code_summary"]], ["total_profit", "top1_contribution_rate", "top3_contribution_rate", "top5_contribution_rate", "worst_code", "worst_code_profit"]),
            "",
            "### Top 10 Codes",
            "",
            self._table(result["phase3d_top_codes"], ["code", "trade_count", "net_profit", "win_rate", "profit_factor", "average_profit"]),
            "",
            "### Bottom 10 Codes",
            "",
            self._table(result["phase3d_bottom_codes"], ["code", "trade_count", "net_profit", "win_rate", "profit_factor", "average_profit"]),
            "",
            f"## {FOCUS_CODE} Dependency",
            "",
            self._table([result["focus_code_dependency"]], ["baseline_profit", "phase3d_profit", "phase3d_contribution_rate", "phase3d_excluding_profit", "phase3d_excluding_profit_factor", "phase3d_excluding_max_drawdown", "excluding_still_beats_baseline_profit"]),
            "",
            "## PM Multiplier Summary",
            "",
            self._table(result["pm_multiplier_summary"], ["group", "trade_count", "total_buy_amount", "net_profit", "profit_factor", "win_rate", "average_profit", "average_holding_days"]),
            "",
            "## PM Score Band Summary",
            "",
            self._table(result["pm_score_band_summary"], ["group", "trade_count", "total_buy_amount", "net_profit", "profit_factor", "win_rate", "average_profit", "return_on_buy_amount"]),
            "",
            "## Skip Reason Comparison",
            "",
            self._table(result["skip_reason_comparison"], ["skip_reason", "baseline_count", "phase3d_count", "delta"]),
            "",
            "## Capital Utilization",
            "",
            self._table(
                [
                    {"profile": "baseline", **result["capital_utilization_comparison"]["baseline"]},
                    {"profile": "phase3d", **result["capital_utilization_comparison"]["phase3d"]},
                ],
                ["profile", "average_capital_utilization", "median_capital_utilization", "max_capital_utilization", "average_holding_count", "max_positions_days", "cash_idle_days"],
            ),
            "",
            "## Promotion Judgement",
            "",
            self._table(result["promotion_judgement"], ["criterion", "passed", "detail"]),
            "",
        ]
        return "\n".join(lines)

    def _profile_payload(self, profile: str) -> dict[str, Any]:
        backtest_dir = self.root / "logs" / "backtests" / profile / self.period
        summary = self._read_json(backtest_dir / "backtest_summary.json")
        trades_raw = self._read_csv(backtest_dir / "trades.csv")
        audit = self._read_csv(backtest_dir / "purchase_audit.csv")
        all_trades = pd.DataFrame(summary.get("all_trades") or [])
        trades = self._sell_trades_with_pm(trades_raw, audit)
        monthly = self._monthly_rows(trades)
        return {
            "profile": profile,
            "summary": self._summary_row(profile, summary, trades),
            "trades": trades,
            "all_trades": all_trades,
            "audit": audit,
            "monthly": monthly,
            "code_summary": self._code_concentration(trades),
            "top_codes": self._code_rows(trades, ascending=False)[:10],
            "bottom_codes": self._code_rows(trades, ascending=True)[:10],
            "capital_utilization": self._capital_utilization(summary, trades),
        }

    def _summary_row(self, profile: str, summary: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
        return {
            "profile": profile,
            "net_profit": float(summary.get("net_cumulative_profit") or self._profit(trades).sum()),
            "profit_factor": float(summary.get("profit_factor") or self._profit_factor(trades)),
            "max_drawdown": float(summary.get("max_drawdown") or self._monthly_drawdown(trades)),
            "win_rate": float(summary.get("win_rate") or self._win_rate(trades)),
            "total_trades": int(len(trades) or summary.get("closed_trades_count") or summary.get("closed_trade_count") or 0),
        }

    def _sell_trades_with_pm(self, trades: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        sells = trades[trades["action"].astype(str).eq("SELL")].copy()
        if sells.empty:
            return sells
        if audit.empty:
            return sells
        buys = audit[audit["decision"].astype(str).isin(["BUY", "SCALED_BUY"])].copy()
        if buys.empty:
            return sells
        for frame in (sells, buys):
            frame["code"] = frame["code"].astype(str)
            frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        pm_cols = [column for column in buys.columns if column.startswith("pm_")]
        audit_cols = ["signal_date", "entry_date", "code", "final_amount", *pm_cols]
        merged = sells.merge(
            buys[audit_cols],
            on=["signal_date", "entry_date", "code"],
            how="left",
            suffixes=("", "_audit"),
        )
        for column in pm_cols:
            audit_col = f"{column}_audit"
            if audit_col in merged.columns:
                if column in merged.columns:
                    merged[column] = merged[column].where(merged[column].notna() & (merged[column].astype(str) != ""), merged[audit_col])
                    merged = merged.drop(columns=[audit_col])
                else:
                    merged[column] = merged[audit_col]
                    merged = merged.drop(columns=[audit_col])
        if "final_amount" not in merged.columns and "final_amount_audit" in merged.columns:
            merged = merged.rename(columns={"final_amount_audit": "final_amount"})
        elif "final_amount_audit" in merged.columns:
            merged["final_amount"] = merged["final_amount"].where(merged["final_amount"].notna(), merged["final_amount_audit"])
            merged = merged.drop(columns=["final_amount_audit"])
        return merged

    def _pm_log_check(self, phase3d: dict[str, Any]) -> dict[str, Any]:
        trades = phase3d["trades"]
        all_trades = phase3d.get("all_trades", pd.DataFrame())
        audit = phase3d["audit"]
        buy_audit = audit[audit.get("decision", pd.Series(dtype=str)).astype(str).isin(["BUY", "SCALED_BUY"])].copy() if not audit.empty else audit
        pm_cols = ["pm_high_conviction_proba", "pm_avoid_proba", "pm_score", "pm_multiplier", "pm_model_version", "pm_feature_count"]
        sell_pm_non_null = int(trades[pm_cols].notna().all(axis=1).sum()) if not trades.empty and all(col in trades.columns for col in pm_cols) else 0
        buy_pm_non_null = 0
        buy_rows = 0
        if not all_trades.empty and "action" in all_trades.columns:
            buy_trades = all_trades[all_trades["action"].astype(str).eq("BUY")].copy()
            buy_rows = int(len(buy_trades))
            if buy_rows and all(col in buy_trades.columns for col in pm_cols):
                buy_pm_non_null = int(buy_trades[pm_cols].notna().all(axis=1).sum())
        match_rate = 0.0
        values_match = False
        if not trades.empty and not buy_audit.empty:
            matched = trades.dropna(subset=["pm_multiplier"]) if "pm_multiplier" in trades.columns else pd.DataFrame()
            match_rate = float(len(matched) / len(trades)) if len(trades) else 0.0
            values_match = bool(match_rate > 0.95)
        return {
            "all_trades_buy_rows": buy_rows,
            "all_trades_buy_pm_non_null_rows": buy_pm_non_null,
            "trades_csv_sell_rows": int(len(trades)),
            "trades_csv_sell_pm_non_null_rows": sell_pm_non_null,
            "audit_buy_rows": int(len(buy_audit)) if not buy_audit.empty else 0,
            "audit_trade_match_rate": match_rate,
            "pm_values_match": values_match,
        }

    def _monthly_rows(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty or "exit_date" not in trades.columns:
            return []
        data = trades.copy()
        data["exit_date"] = pd.to_datetime(data["exit_date"], errors="coerce")
        data["month"] = data["exit_date"].dt.strftime("%Y-%m")
        data["net_profit"] = self._profit(data)
        grouped = data.dropna(subset=["month"]).groupby("month")["net_profit"].sum().sort_index()
        rows = []
        cumulative = 0.0
        for month, profit in grouped.items():
            cumulative += float(profit)
            rows.append(
                {
                    "month": month,
                    "monthly_profit": float(profit),
                    "monthly_return": float(profit) / INITIAL_CAPITAL,
                    "monthly_win": bool(profit > 0),
                    "cumulative_profit": cumulative,
                }
            )
        return rows

    def _monthly_comparison(self, baseline: list[dict[str, Any]], phase3d: list[dict[str, Any]]) -> list[dict[str, Any]]:
        left = {row["month"]: row for row in baseline}
        right = {row["month"]: row for row in phase3d}
        rows = []
        for month in sorted(set(left) | set(right)):
            b = float(left.get(month, {}).get("monthly_profit") or 0)
            p = float(right.get(month, {}).get("monthly_profit") or 0)
            rows.append(
                {
                    "month": month,
                    "baseline_profit": b,
                    "phase3d_profit": p,
                    "delta": p - b,
                    "phase3d_cumulative_profit": float(right.get(month, {}).get("cumulative_profit") or 0),
                }
            )
        return rows

    def _code_rows(self, trades: pd.DataFrame, ascending: bool) -> list[dict[str, Any]]:
        if trades.empty:
            return []
        rows = []
        for code, group in trades.groupby(trades["code"].astype(str)):
            profit = self._profit(group)
            rows.append(
                {
                    "code": code,
                    "trade_count": int(len(group)),
                    "net_profit": float(profit.sum()),
                    "win_rate": float(profit.gt(0).mean()) if len(profit) else 0.0,
                    "profit_factor": self._profit_factor(group),
                    "average_profit": float(profit.mean()) if len(profit) else 0.0,
                }
            )
        return sorted(rows, key=lambda row: row["net_profit"], reverse=not ascending)

    def _code_concentration(self, trades: pd.DataFrame) -> dict[str, Any]:
        rows = self._code_rows(trades, ascending=False)
        total = float(sum(row["net_profit"] for row in rows))
        positive_total = total if total > 0 else 1.0
        worst = min(rows, key=lambda row: row["net_profit"], default={"code": "", "net_profit": 0.0})
        return {
            "total_profit": total,
            "top1_contribution_rate": float(sum(row["net_profit"] for row in rows[:1]) / positive_total) if rows else 0.0,
            "top3_contribution_rate": float(sum(row["net_profit"] for row in rows[:3]) / positive_total) if rows else 0.0,
            "top5_contribution_rate": float(sum(row["net_profit"] for row in rows[:5]) / positive_total) if rows else 0.0,
            "worst_code": worst["code"],
            "worst_code_profit": worst["net_profit"],
        }

    def _focus_code_dependency(self, baseline: dict[str, Any], phase3d: dict[str, Any], code: str) -> dict[str, Any]:
        baseline_profit = self._code_profit(baseline["trades"], code)
        phase_profit = self._code_profit(phase3d["trades"], code)
        total = float(phase3d["summary"]["net_profit"])
        excluding = phase3d["trades"][~phase3d["trades"]["code"].astype(str).eq(code)].copy() if not phase3d["trades"].empty else phase3d["trades"]
        excluding_profit = float(self._profit(excluding).sum())
        return {
            "baseline_profit": baseline_profit,
            "phase3d_profit": phase_profit,
            "phase3d_contribution_rate": phase_profit / total if total else 0.0,
            "phase3d_excluding_profit": excluding_profit,
            "phase3d_excluding_profit_factor": self._profit_factor(excluding),
            "phase3d_excluding_max_drawdown": self._monthly_drawdown(excluding),
            "excluding_still_beats_baseline_profit": excluding_profit > float(baseline["summary"]["net_profit"]),
        }

    def _pm_group_summary(self, trades: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        if trades.empty or column not in trades.columns:
            return []
        data = trades.copy()
        data[column] = pd.to_numeric(data[column], errors="coerce")
        rows = []
        for value, group in data.dropna(subset=[column]).groupby(column):
            rows.append(self._trade_group_row(str(value), group))
        return rows

    def _pm_score_band_summary(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
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
        return [self._trade_group_row(str(label), group) for label, group in data.dropna(subset=["pm_score_band"]).groupby("pm_score_band", observed=False)]

    def _trade_group_row(self, group: str, trades: pd.DataFrame) -> dict[str, Any]:
        profit = self._profit(trades)
        buy_amount = pd.to_numeric(trades.get("final_amount"), errors="coerce").fillna(0.0) if "final_amount" in trades.columns else pd.Series([0.0] * len(trades))
        return {
            "group": group,
            "trade_count": int(len(trades)),
            "total_buy_amount": float(buy_amount.sum()),
            "net_profit": float(profit.sum()),
            "profit_factor": self._profit_factor(trades),
            "win_rate": float(profit.gt(0).mean()) if len(profit) else 0.0,
            "average_profit": float(profit.mean()) if len(profit) else 0.0,
            "average_holding_days": float(pd.to_numeric(trades.get("holding_days"), errors="coerce").dropna().mean()) if "holding_days" in trades.columns else 0.0,
            "return_on_buy_amount": float(profit.sum() / buy_amount.sum()) if float(buy_amount.sum()) else 0.0,
        }

    def _skip_reason_comparison(self, baseline_audit: pd.DataFrame, phase_audit: pd.DataFrame) -> list[dict[str, Any]]:
        left = self._skip_counts(baseline_audit)
        right = self._skip_counts(phase_audit)
        rows = []
        for reason in sorted(set(left) | set(right)):
            rows.append(
                {
                    "skip_reason": reason,
                    "baseline_count": left.get(reason, 0),
                    "phase3d_count": right.get(reason, 0),
                    "delta": right.get(reason, 0) - left.get(reason, 0),
                }
            )
        return rows

    def _skip_counts(self, audit: pd.DataFrame) -> dict[str, int]:
        if audit.empty or "skip_reason" not in audit.columns:
            return {}
        skipped = audit[audit.get("decision", pd.Series(dtype=str)).astype(str).eq("SKIP")].copy()
        return {str(key): int(value) for key, value in skipped["skip_reason"].fillna("<NA>").astype(str).value_counts().items()}

    def _capital_utilization(self, summary: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
        curve = pd.DataFrame(summary.get("daily_asset_curve") or [])
        if curve.empty or trades.empty:
            return {
                "average_capital_utilization": None,
                "median_capital_utilization": None,
                "max_capital_utilization": None,
                "average_holding_count": None,
                "max_positions_days": 0,
                "cash_idle_days": 0,
            }
        curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
        exposures = []
        counts = []
        for date in curve["date"]:
            open_trades = trades[
                (pd.to_datetime(trades["entry_date"], errors="coerce") <= date)
                & (pd.to_datetime(trades["exit_date"], errors="coerce") >= date)
            ]
            amount = pd.to_numeric(open_trades.get("final_amount"), errors="coerce").fillna(
                pd.to_numeric(open_trades.get("entry_price"), errors="coerce").fillna(0.0)
                * pd.to_numeric(open_trades.get("shares"), errors="coerce").fillna(0.0)
            )
            exposures.append(float(amount.sum()))
            counts.append(int(len(open_trades)))
        total_assets = pd.to_numeric(curve.get("total_assets"), errors="coerce").replace(0, pd.NA)
        utilization = pd.Series(exposures) / total_assets.reset_index(drop=True)
        return {
            "average_capital_utilization": float(utilization.dropna().mean()) if not utilization.dropna().empty else None,
            "median_capital_utilization": float(utilization.dropna().median()) if not utilization.dropna().empty else None,
            "max_capital_utilization": float(utilization.dropna().max()) if not utilization.dropna().empty else None,
            "average_holding_count": float(pd.Series(counts).mean()) if counts else None,
            "max_positions_days": int(pd.Series(counts).ge(10).sum()) if counts else 0,
            "cash_idle_days": int(utilization.fillna(0).lt(0.1).sum()) if len(utilization) else 0,
        }

    def _promotion_judgement(self, baseline: dict[str, Any], phase3d: dict[str, Any]) -> list[dict[str, Any]]:
        b = baseline["summary"]
        p = phase3d["summary"]
        focus = self._focus_code_dependency(baseline, phase3d, FOCUS_CODE)
        pm_rows = self._pm_score_band_summary(phase3d["trades"])
        profits = [float(row["net_profit"]) for row in pm_rows]
        monotonic = all(left <= right for left, right in zip(profits, profits[1:])) if len(profits) >= 2 else False
        checks = [
            ("net_profit_above_baseline", p["net_profit"] > b["net_profit"], f"{p['net_profit']} vs {b['net_profit']}"),
            ("profit_factor_above_baseline", p["profit_factor"] > b["profit_factor"], f"{p['profit_factor']} vs {b['profit_factor']}"),
            ("drawdown_improved", p["max_drawdown"] > b["max_drawdown"], f"{p['max_drawdown']} vs {b['max_drawdown']}"),
            ("monthly_win_rate_not_worse", self._monthly_win_rate(phase3d["monthly"]) >= self._monthly_win_rate(baseline["monthly"]), ""),
            ("excluding_67400_still_beats_baseline", bool(focus["excluding_still_beats_baseline_profit"]), str(focus["phase3d_excluding_profit"])),
            ("top3_concentration_not_extreme", phase3d["code_summary"]["top3_contribution_rate"] < 0.75, str(phase3d["code_summary"]["top3_contribution_rate"])),
            ("pm_score_monotonic_profit", monotonic, str(profits)),
            ("pm_values_available_in_trades", self._pm_log_check(phase3d)["trades_csv_sell_pm_non_null_rows"] > 0, ""),
        ]
        return [{"criterion": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks]

    def _monthly_win_rate(self, rows: list[dict[str, Any]]) -> float:
        return float(sum(1 for row in rows if row.get("monthly_profit", 0) > 0) / len(rows)) if rows else 0.0

    def _code_profit(self, trades: pd.DataFrame, code: str) -> float:
        if trades.empty:
            return 0.0
        return float(self._profit(trades[trades["code"].astype(str).eq(str(code))]).sum())

    def _profit(self, trades: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(trades.get("net_profit"), errors="coerce").fillna(0.0)

    def _win_rate(self, trades: pd.DataFrame) -> float:
        profit = self._profit(trades)
        return float(profit.gt(0).mean()) if len(profit) else 0.0

    def _profit_factor(self, trades: pd.DataFrame) -> float:
        profit = self._profit(trades)
        gross_profit = float(profit[profit > 0].sum())
        gross_loss = abs(float(profit[profit < 0].sum()))
        if gross_loss:
            return gross_profit / gross_loss
        return float("inf") if gross_profit else 0.0

    def _monthly_drawdown(self, trades: pd.DataFrame) -> float:
        rows = self._monthly_rows(trades)
        equity = INITIAL_CAPITAL
        peak = equity
        max_dd = 0.0
        for row in rows:
            equity += float(row["monthly_profit"])
            peak = max(peak, equity)
            if peak:
                max_dd = min(max_dd, equity / peak - 1.0)
        return max_dd

    def _read_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            values = [self._format_value(row.get(column, "")) for column in columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def _format_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
