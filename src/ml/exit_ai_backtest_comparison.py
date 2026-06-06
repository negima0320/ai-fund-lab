from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROFILES = [
    "rookie_dealer_02_v2_65",
    "rookie_dealer_02_v2_66_ml_ranked",
    "rookie_dealer_02_v2_68_ml_ranked_exit_ai_050",
    "rookie_dealer_02_v2_69_ml_ranked_exit_ai_055",
    "rookie_dealer_02_v2_70_ml_ranked_exit_ai_060",
    "rookie_dealer_02_v2_67_ml_standalone",
]


@dataclass(frozen=True)
class ExitAIBacktestComparisonPaths:
    markdown: Path
    json: Path


class ExitAIBacktestComparison:
    def __init__(
        self,
        root: str | Path = ".",
        profiles: list[str] | None = None,
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
    ) -> None:
        self.root = Path(root)
        self.profiles = profiles or list(DEFAULT_PROFILES)
        self.start_date = start_date
        self.end_date = end_date
        self.period_key = f"{start_date}_to_{end_date}"

    def build(self) -> dict[str, Any]:
        summary = []
        exit_ai = []
        monthly = []
        for profile in self.profiles:
            summary_payload = self._load_summary(profile)
            trades = self._load_sell_trades(profile)
            summary.append(self._summary_row(profile, summary_payload, trades))
            exit_ai.append(self._exit_ai_row(profile, trades))
            monthly.extend(self._monthly_rows(profile, trades))
        return {
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "profiles": self.profiles,
            "summary": summary,
            "exit_ai": exit_ai,
            "monthly": monthly,
            "best_by_net_profit": max(summary, key=lambda row: row.get("net_profit") or -10**18) if summary else None,
            "best_by_profit_factor": max(summary, key=lambda row: row.get("profit_factor") or -10**18) if summary else None,
        }

    def save(self, result: dict[str, Any]) -> ExitAIBacktestComparisonPaths:
        out_dir = self.root / "reports" / "ml"
        out_dir.mkdir(parents=True, exist_ok=True)
        markdown = out_dir / "ml_exit_ai_backtest_comparison_2023-01_to_2026-05.md"
        json_path = out_dir / "ml_exit_ai_backtest_comparison_2023-01_to_2026-05.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return ExitAIBacktestComparisonPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# ML Exit AI Backtest Comparison",
            "",
            f"- period: {self.start_date} to {self.end_date}",
            "- source: logs/backtests outputs",
            "- note: v2_65/v2_66/v2_67 are unchanged baselines; v2_68-v2_70 are derived Exit AI profiles.",
            "",
            "## Profile Comparison",
            "",
            self._table(
                result["summary"],
                [
                    "profile",
                    "final_assets",
                    "net_profit",
                    "win_rate",
                    "profit_factor",
                    "max_drawdown",
                    "total_trades",
                    "average_holding_days",
                    "monthly_win_rate",
                    "losing_months",
                    "best_month",
                    "worst_month",
                ],
            ),
            "",
            "## Exit AI Logs",
            "",
            self._table(
                result["exit_ai"],
                [
                    "profile",
                    "exit_ai_trigger_count",
                    "exit_ai_signal_count",
                    "exit_ai_prediction_join_rate",
                    "triggered_net_profit",
                    "triggered_win_rate",
                    "triggered_average_profit",
                    "average_probability",
                    "p50_probability",
                    "p90_probability",
                ],
            ),
            "",
            "## Monthly Results",
            "",
            self._table(
                result["monthly"],
                ["profile", "month", "net_profit", "win_rate", "profit_factor", "trade_count"],
            ),
            "",
            "## Summary",
            "",
            f"- best_by_net_profit: `{(result.get('best_by_net_profit') or {}).get('profile')}`",
            f"- best_by_profit_factor: `{(result.get('best_by_profit_factor') or {}).get('profile')}`",
            "",
        ]
        return "\n".join(lines)

    def _load_summary(self, profile: str) -> dict[str, Any]:
        path = self.root / "logs" / "backtests" / profile / self.period_key / "backtest_summary.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _load_sell_trades(self, profile: str) -> pd.DataFrame:
        path = self.root / "logs" / "backtests" / profile / self.period_key / "trades.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if df.empty:
            return df
        if "action" in df.columns:
            df = df[df["action"].astype(str).eq("SELL")].copy()
        for column in ["entry_date", "exit_date"]:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")
        for column in [
            "net_profit",
            "profit",
            "holding_days",
            "exit_ai_probability",
            "exit_ai_threshold",
        ]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        if "net_profit" not in df.columns and "profit" in df.columns:
            df["net_profit"] = df["profit"]
        return df

    def _summary_row(self, profile: str, payload: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
        profits = pd.to_numeric(trades.get("net_profit"), errors="coerce").fillna(0.0) if not trades.empty else pd.Series(dtype=float)
        monthly = self._monthly_rows(profile, trades)
        monthly_wins = sum(1 for row in monthly if (row.get("net_profit") or 0) > 0)
        best_month = max(monthly, key=lambda row: row.get("net_profit") or -10**18, default={})
        worst_month = min(monthly, key=lambda row: row.get("net_profit") or 10**18, default={})
        return {
            "profile": profile,
            "final_assets": self._pick(payload, ["final_assets", "latest_total_assets", "total_assets"]),
            "net_profit": self._pick(payload, ["net_cumulative_profit", "cumulative_profit"]) or (float(profits.sum()) if not profits.empty else None),
            "win_rate": self._pick(payload, ["win_rate"]) or self._win_rate(profits),
            "profit_factor": self._pick(payload, ["profit_factor"]) or self._profit_factor(profits),
            "max_drawdown": self._pick(payload, ["max_drawdown"]),
            "total_trades": int(len(trades)),
            "average_holding_days": float(pd.to_numeric(trades.get("holding_days"), errors="coerce").mean()) if not trades.empty and "holding_days" in trades.columns else None,
            "monthly_win_rate": monthly_wins / len(monthly) if monthly else None,
            "losing_months": sum(1 for row in monthly if (row.get("net_profit") or 0) < 0),
            "best_month": f"{best_month.get('month')}:{best_month.get('net_profit')}" if best_month else None,
            "worst_month": f"{worst_month.get('month')}:{worst_month.get('net_profit')}" if worst_month else None,
        }

    def _exit_ai_row(self, profile: str, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty or "exit_ai_probability" not in trades.columns:
            return {
                "profile": profile,
                "exit_ai_trigger_count": 0,
                "exit_ai_signal_count": 0,
                "exit_ai_prediction_join_rate": None,
                "triggered_net_profit": None,
                "triggered_win_rate": None,
                "triggered_average_profit": None,
                "average_probability": None,
                "p50_probability": None,
                "p90_probability": None,
            }
        triggered = trades[trades.get("exit_ai_triggered").astype(str).str.lower().isin({"true", "1", "yes"})] if "exit_ai_triggered" in trades.columns else trades.iloc[0:0]
        signal = trades[trades.get("exit_ai_signal").astype(str).str.lower().isin({"true", "1", "yes"})] if "exit_ai_signal" in trades.columns else trades.iloc[0:0]
        probabilities = pd.to_numeric(trades["exit_ai_probability"], errors="coerce").dropna()
        joined = trades.get("exit_ai_prediction_joined")
        join_rate = None
        if joined is not None:
            join_rate = float(joined.astype(str).str.lower().isin({"true", "1", "yes"}).mean())
        triggered_profit = pd.to_numeric(triggered.get("net_profit"), errors="coerce").fillna(0.0) if not triggered.empty else pd.Series(dtype=float)
        return {
            "profile": profile,
            "exit_ai_trigger_count": int(len(triggered)),
            "exit_ai_signal_count": int(len(signal)),
            "exit_ai_prediction_join_rate": join_rate,
            "triggered_net_profit": float(triggered_profit.sum()) if not triggered_profit.empty else 0.0,
            "triggered_win_rate": self._win_rate(triggered_profit),
            "triggered_average_profit": float(triggered_profit.mean()) if not triggered_profit.empty else None,
            "average_probability": float(probabilities.mean()) if not probabilities.empty else None,
            "p50_probability": float(probabilities.quantile(0.50)) if not probabilities.empty else None,
            "p90_probability": float(probabilities.quantile(0.90)) if not probabilities.empty else None,
        }

    def _monthly_rows(self, profile: str, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty or "exit_date" not in trades.columns:
            return []
        df = trades.dropna(subset=["exit_date"]).copy()
        if df.empty:
            return []
        df["month"] = df["exit_date"].dt.to_period("M").astype(str)
        rows = []
        for month, group in df.groupby("month"):
            profits = pd.to_numeric(group["net_profit"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "profile": profile,
                    "month": str(month),
                    "net_profit": float(profits.sum()),
                    "win_rate": self._win_rate(profits),
                    "profit_factor": self._profit_factor(profits),
                    "trade_count": int(len(group)),
                }
            )
        return rows

    def _pick(self, payload: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if payload.get(key) is not None:
                return payload.get(key)
        return None

    def _win_rate(self, profits: pd.Series) -> float | None:
        if profits.empty:
            return None
        return float((profits > 0).mean())

    def _profit_factor(self, profits: pd.Series) -> float | None:
        if profits.empty:
            return None
        gross_profit = float(profits[profits > 0].sum())
        gross_loss = float(-profits[profits < 0].sum())
        return gross_profit / gross_loss if gross_loss else None

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(self._format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
