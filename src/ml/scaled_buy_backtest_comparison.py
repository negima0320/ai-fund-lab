from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROFILES = [
    "rookie_dealer_02_v2_66_ml_ranked",
    "rookie_dealer_02_v2_68_ml_ranked_exit_ai_050",
    "rookie_dealer_02_v2_70_ml_ranked_exit_ai_060",
    "rookie_dealer_02_v2_71_ml_ranked_exit_ai_050_scaled_buy",
]
SCALED_PROFILE = "rookie_dealer_02_v2_71_ml_ranked_exit_ai_050_scaled_buy"


@dataclass(frozen=True)
class ScaledBuyBacktestComparisonPaths:
    markdown: Path
    json: Path
    scaled_buy_trades_csv: Path


class ScaledBuyBacktestComparison:
    def __init__(
        self,
        root: str | Path = ".",
        profiles: list[str] | None = None,
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        scaled_profile: str = SCALED_PROFILE,
    ) -> None:
        self.root = Path(root)
        self.profiles = profiles or list(DEFAULT_PROFILES)
        self.start_date = start_date
        self.end_date = end_date
        self.period_key = f"{start_date}_to_{end_date}"
        self.scaled_profile = scaled_profile

    def build(self) -> dict[str, Any]:
        summary = []
        monthly = []
        for profile in self.profiles:
            trades = self._load_sell_trades(profile)
            summary_payload = self._load_summary(profile)
            summary.append(self._summary_row(profile, summary_payload, trades))
            monthly.extend(self._monthly_rows(profile, trades))
        scaled_buy_trades = self._scaled_buy_trades(self.scaled_profile)
        result = {
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "profiles": self.profiles,
            "summary": summary,
            "monthly": monthly,
            "scaled_buy_summary": self._scaled_buy_summary(scaled_buy_trades),
            "scaled_buy_symbols": self._scaled_buy_symbols(scaled_buy_trades),
            "focus_67400": self._focus_symbol(scaled_buy_trades, "67400"),
            "march_2026": self._march_summary(),
            "scaled_buy_trades": self._records(scaled_buy_trades),
            "best_by_net_profit": max(summary, key=lambda row: row.get("net_profit") or -10**18) if summary else None,
            "best_by_profit_factor": max(summary, key=lambda row: row.get("profit_factor") or -10**18) if summary else None,
        }
        result["diagnosis"] = self._diagnosis(result)
        result["scaled_buy_trades_df"] = scaled_buy_trades
        return result

    def save(self, result: dict[str, Any]) -> ScaledBuyBacktestComparisonPaths:
        out_dir = self.root / "reports" / "ml"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "scaled_buy_backtest_comparison_2023-01_to_2026-05"
        markdown = out_dir / f"{stem}.md"
        json_path = out_dir / f"{stem}.json"
        csv_path = out_dir / "scaled_buy_trades_2023-01_to_2026-05.csv"

        scaled_df = result.pop("scaled_buy_trades_df")
        scaled_df.to_csv(csv_path, index=False)
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        result["scaled_buy_trades_df"] = scaled_df
        return ScaledBuyBacktestComparisonPaths(markdown=markdown, json=json_path, scaled_buy_trades_csv=csv_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Scaled Buy Backtest Comparison",
            "",
            f"- period: {self.start_date} to {self.end_date}",
            "- source: existing/new `logs/backtests` outputs; no API fetch",
            "- note: v2_71 is a derived profile; existing profiles are unchanged.",
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
            "## Scaled Buy Summary",
            "",
            self._table(
                [result["scaled_buy_summary"]],
                [
                    "scaled_buy_trigger_count",
                    "scaled_buy_closed_count",
                    "scaled_buy_profit",
                    "scaled_buy_win_rate",
                    "average_original_amount",
                    "average_scaled_amount",
                ],
            ),
            "",
            "## 67400 Focus",
            "",
            self._table(
                [result["focus_67400"]],
                [
                    "bought",
                    "entry_date",
                    "exit_date",
                    "shares",
                    "original_planned_shares",
                    "amount",
                    "original_amount",
                    "net_profit",
                    "net_profit_rate",
                    "exit_reason",
                ],
            ),
            "",
            "## 2026-03 Net Profit",
            "",
            self._table(result["march_2026"], ["profile", "month", "net_profit", "win_rate", "profit_factor", "trade_count"]),
            "",
            "## Scaled Buy Symbols",
            "",
            self._table(result["scaled_buy_symbols"], ["code", "trigger_count", "closed_count", "net_profit"]),
            "",
            "## Diagnosis",
            "",
        ]
        lines.extend(f"- {item}" for item in result["diagnosis"])
        lines.append("")
        return "\n".join(lines)

    def _load_summary(self, profile: str) -> dict[str, Any]:
        path = self.root / "logs" / "backtests" / profile / self.period_key / "backtest_summary.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _load_all_trades(self, profile: str) -> pd.DataFrame:
        rows = self._load_summary(profile).get("all_trades") or []
        return pd.DataFrame(rows)

    def _load_sell_trades(self, profile: str) -> pd.DataFrame:
        path = self.root / "logs" / "backtests" / profile / self.period_key / "trades.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if df.empty:
            return df
        if "action" in df.columns:
            df = df[df["action"].astype(str).eq("SELL")].copy()
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")
        for column in ["net_profit", "profit", "holding_days", "net_profit_rate", "scaled_amount", "original_amount", "original_planned_shares"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        if "net_profit" not in df.columns and "profit" in df.columns:
            df["net_profit"] = df["profit"]
        return df

    def _scaled_buy_trades(self, profile: str) -> pd.DataFrame:
        sells = self._load_sell_trades(profile)
        all_trades = self._load_all_trades(profile)
        if sells.empty:
            return pd.DataFrame()
        scaled_sells = sells[self._truthy_series(sells.get("scaled_buy_triggered"))].copy() if "scaled_buy_triggered" in sells.columns else sells.iloc[0:0].copy()
        if not scaled_sells.empty:
            return scaled_sells.sort_values(["entry_date", "code"], na_position="last")

        if all_trades.empty or "scaled_buy_triggered" not in all_trades.columns:
            return scaled_sells
        buys = all_trades[
            all_trades.get("action", pd.Series(dtype=str)).astype(str).eq("BUY")
            & self._truthy_series(all_trades.get("scaled_buy_triggered"))
        ].copy()
        if buys.empty:
            return scaled_sells
        for column in ["entry_date"]:
            buys[column] = pd.to_datetime(buys[column], errors="coerce")
        merge_cols = [
            "code",
            "entry_date",
            "shares",
            "original_planned_shares",
            "scaled_shares",
            "original_amount",
            "scaled_amount",
            "scale_reason",
        ]
        buy_meta = buys[[column for column in merge_cols if column in buys.columns]].drop_duplicates(["code", "entry_date"])
        merged = sells.merge(buy_meta, on=["code", "entry_date"], how="inner", suffixes=("", "_buy"))
        merged["scaled_buy_triggered"] = True
        return merged.sort_values(["entry_date", "code"], na_position="last")

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

    def _march_summary(self) -> list[dict[str, Any]]:
        rows = []
        for profile in self.profiles:
            march = [row for row in self._monthly_rows(profile, self._load_sell_trades(profile)) if row.get("month") == "2026-03"]
            rows.extend(march)
        return rows

    def _scaled_buy_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        profits = pd.to_numeric(df.get("net_profit"), errors="coerce").fillna(0.0) if not df.empty else pd.Series(dtype=float)
        return {
            "scaled_buy_trigger_count": int(len(df)),
            "scaled_buy_closed_count": int(len(df)),
            "scaled_buy_profit": float(profits.sum()) if not profits.empty else 0.0,
            "scaled_buy_win_rate": self._win_rate(profits),
            "average_original_amount": self._mean(df, "original_amount"),
            "average_scaled_amount": self._mean(df, "scaled_amount"),
        }

    def _scaled_buy_symbols(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        rows = []
        for code, group in df.groupby(df["code"].astype(str)):
            profits = pd.to_numeric(group.get("net_profit"), errors="coerce").fillna(0.0)
            rows.append(
                {
                    "code": code,
                    "trigger_count": int(len(group)),
                    "closed_count": int(len(group)),
                    "net_profit": float(profits.sum()),
                }
            )
        return sorted(rows, key=lambda row: row["net_profit"], reverse=True)

    def _focus_symbol(self, df: pd.DataFrame, code: str) -> dict[str, Any]:
        if df.empty:
            return {"bought": False}
        matched = df[df["code"].astype(str).eq(str(code))].copy()
        if matched.empty:
            return {"bought": False}
        row = matched.sort_values("net_profit", ascending=False).iloc[0].to_dict()
        return {
            "bought": True,
            "entry_date": self._date_text(row.get("entry_date")),
            "exit_date": self._date_text(row.get("exit_date")),
            "shares": self._number(row.get("shares")),
            "original_planned_shares": self._number(row.get("original_planned_shares")),
            "amount": self._number(row.get("scaled_amount") or row.get("amount")),
            "original_amount": self._number(row.get("original_amount")),
            "net_profit": self._number(row.get("net_profit")),
            "net_profit_rate": self._number(row.get("net_profit_rate")),
            "exit_reason": row.get("exit_reason"),
        }

    def _diagnosis(self, result: dict[str, Any]) -> list[str]:
        summary = {row["profile"]: row for row in result["summary"]}
        v68 = summary.get("rookie_dealer_02_v2_68_ml_ranked_exit_ai_050", {})
        v71 = summary.get(self.scaled_profile, {})
        focus = result["focus_67400"]
        notes = []
        if focus.get("bought"):
            notes.append(f"67400 was bought by v2_71 with {focus.get('shares')} shares and net_profit {self._format(focus.get('net_profit'))}.")
        else:
            notes.append("67400 was not found among closed scaled-buy trades.")
        if v68 and v71:
            delta = self._number(v71.get("net_profit")) - self._number(v68.get("net_profit"))
            notes.append(f"v2_71 net_profit delta vs v2_68: {self._format(delta)}.")
        scaled = result["scaled_buy_summary"]
        notes.append(
            f"scaled buy triggered on {scaled.get('scaled_buy_trigger_count')} closed trades; total contribution {self._format(scaled.get('scaled_buy_profit'))}."
        )
        return notes

    def _records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        return df.where(pd.notna(df), None).to_dict("records")

    def _pick(self, payload: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if payload.get(key) is not None:
                return payload.get(key)
        return None

    def _mean(self, df: pd.DataFrame, column: str) -> float | None:
        if df.empty or column not in df.columns:
            return None
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(series.mean()) if not series.empty else None

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

    def _truthy_series(self, series: Any) -> pd.Series:
        if series is None:
            return pd.Series(dtype=bool)
        return series.astype(str).str.lower().isin({"true", "1", "yes"})

    def _number(self, value: Any) -> float:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return 0.0 if pd.isna(numeric) else float(numeric)

    def _date_text(self, value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)

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
