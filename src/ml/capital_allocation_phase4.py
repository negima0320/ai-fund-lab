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
    "rookie_dealer_02_v2_72_ml_ranked_exit_ai_scaled_buy_v2",
]
V2_72_PROFILE = "rookie_dealer_02_v2_72_ml_ranked_exit_ai_scaled_buy_v2"


@dataclass(frozen=True)
class CapitalAllocationPhase4Paths:
    markdown: Path
    json: Path
    purchase_audit_summary_md: Path


class CapitalAllocationPhase4Comparison:
    def __init__(
        self,
        root: str | Path = ".",
        profiles: list[str] | None = None,
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        focus_profile: str = V2_72_PROFILE,
    ) -> None:
        self.root = Path(root)
        self.profiles = profiles or list(DEFAULT_PROFILES)
        self.start_date = start_date
        self.end_date = end_date
        self.period_key = f"{start_date}_to_{end_date}"
        self.focus_profile = focus_profile
        self.report_dir = self.root / "reports" / "ml"

    def build(self) -> dict[str, Any]:
        summary = []
        for profile in self.profiles:
            trades = self._load_trades(profile)
            payload = self._load_summary(profile)
            daily = self._load_daily(profile)
            purchase_audit = self._load_purchase_audit(profile)
            summary.append(self._summary_row(profile, payload, trades, daily, purchase_audit))
        focus_trades = self._load_trades(self.focus_profile)
        purchase_audit = self._load_purchase_audit(self.focus_profile)
        scaled_trades = self._scaled_trades(focus_trades)
        result = {
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "profiles": self.profiles,
            "summary": summary,
            "focus_profile": self.focus_profile,
            "focus_67400": self._focus_symbol(focus_trades, "67400"),
            "scaled_buy_summary": self._scaled_summary(scaled_trades),
            "scaled_buy_without_67400": self._trade_stats(scaled_trades[~scaled_trades["code"].astype(str).eq("67400")]) if not scaled_trades.empty else self._trade_stats(scaled_trades),
            "purchase_audit_summary": self._purchase_audit_summary(purchase_audit),
            "purchase_audit_decisions": self._purchase_audit_decisions(purchase_audit),
            "diagnosis": self._diagnosis(summary, focus_trades, scaled_trades, purchase_audit),
        }
        return result

    def save(self, result: dict[str, Any]) -> CapitalAllocationPhase4Paths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "capital_allocation_phase4_v2_72_comparison_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        audit_md = self.report_dir / "v2_72_purchase_audit_summary_2023-01_to_2026-05.md"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        audit_md.write_text(self.format_purchase_audit_markdown(result), encoding="utf-8")
        return CapitalAllocationPhase4Paths(markdown=markdown, json=json_path, purchase_audit_summary_md=audit_md)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Capital Allocation Phase 4 v2_72 Comparison",
            "",
            f"- period: {self.start_date} to {self.end_date}",
            "- source: `logs/backtests` outputs; no API fetch and no live order",
            "- note: v2_72 is a derived profile; existing profiles are unchanged.",
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
                    "capital_utilization",
                    "scaled_buy_count",
                    "continue_candidate_buy_count",
                    "skipped_but_affordable_count",
                    "rejected_by_daily_limit_count",
                    "purchase_audit_rows",
                ],
            ),
            "",
            "## v2_72 67400 Focus",
            "",
            self._table(
                [result["focus_67400"]],
                ["bought", "entry_date", "exit_date", "shares", "amount", "net_profit", "net_profit_rate", "exit_reason"],
            ),
            "",
            "## v2_72 Scaled Buy",
            "",
            self._table([result["scaled_buy_summary"]], ["trade_count", "net_profit", "win_rate", "profit_factor", "best_trade", "worst_trade"]),
            "",
            "### v2_72 Scaled Buy Excluding 67400",
            "",
            self._table([result["scaled_buy_without_67400"]], ["trade_count", "net_profit", "win_rate", "profit_factor", "best_trade", "worst_trade"]),
            "",
            "## Purchase Audit",
            "",
            self._table([result["purchase_audit_summary"]], ["rows", "buy_count", "scaled_buy_count", "skip_count", "daily_limit_skip_count", "duplicate_skip_count"]),
            "",
            self._table(result["purchase_audit_decisions"], ["decision", "count"]),
            "",
            "## Diagnosis",
            "",
        ]
        lines.extend(f"- {item}" for item in result["diagnosis"])
        lines.append("")
        return "\n".join(lines)

    def format_purchase_audit_markdown(self, result: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# v2_72 Purchase Audit Summary",
                "",
                f"- profile: `{self.focus_profile}`",
                f"- period: `{self.period_key}`",
                "",
                "## Summary",
                "",
                self._table([result["purchase_audit_summary"]], ["rows", "buy_count", "scaled_buy_count", "skip_count", "daily_limit_skip_count", "duplicate_skip_count"]),
                "",
                "## Decisions",
                "",
                self._table(result["purchase_audit_decisions"], ["decision", "count"]),
                "",
            ]
        )

    def _backtest_dir(self, profile: str) -> Path:
        return self.root / "logs" / "backtests" / profile / self.period_key

    def _load_summary(self, profile: str) -> dict[str, Any]:
        path = self._backtest_dir(profile) / "backtest_summary.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _load_trades(self, profile: str) -> pd.DataFrame:
        path = self._backtest_dir(profile) / "trades.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df = self._merge_trade_metadata_from_summary(profile, df)
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")
        for column in ["net_profit", "profit", "holding_days", "net_profit_rate", "amount"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        if "net_profit" not in df.columns and "profit" in df.columns:
            df["net_profit"] = df["profit"]
        return df[df.get("action", pd.Series(dtype=str)).astype(str).eq("SELL")].copy() if "action" in df.columns else df

    def _merge_trade_metadata_from_summary(self, profile: str, df: pd.DataFrame) -> pd.DataFrame:
        metadata_columns = [
            "candidate_source",
            "fallback_rank",
            "raw_candidate_rank",
            "risk_adjusted_score",
            "expected_return_10d",
            "bad_entry_probability_10d",
        ]
        missing_columns = [column for column in metadata_columns if column not in df.columns]
        if not missing_columns or "trade_id" not in df.columns:
            return df
        summary = self._load_summary(profile)
        rows = summary.get("all_trades", []) if isinstance(summary, dict) else []
        if not isinstance(rows, list) or not rows:
            return df
        metadata_rows = [
            {"trade_id": row.get("trade_id"), **{column: row.get(column) for column in missing_columns}}
            for row in rows
            if isinstance(row, dict) and row.get("action") == "SELL" and row.get("trade_id")
        ]
        if not metadata_rows:
            return df
        metadata = pd.DataFrame(metadata_rows).drop_duplicates(subset=["trade_id"], keep="last")
        return df.merge(metadata, on="trade_id", how="left")

    def _load_daily(self, profile: str) -> pd.DataFrame:
        path = self._backtest_dir(profile) / "summary.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        for column in ["total_assets", "portfolio_value", "positions_value", "market_value", "cash"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def _load_purchase_audit(self, profile: str) -> pd.DataFrame:
        path = self._backtest_dir(profile) / "purchase_audit.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        for column in ["final_amount", "planned_amount", "scaled_amount", "daily_buy_limit_remaining_before", "daily_buy_limit_remaining_after"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def _summary_row(self, profile: str, payload: dict[str, Any], trades: pd.DataFrame, daily: pd.DataFrame, purchase_audit: pd.DataFrame) -> dict[str, Any]:
        if not payload and trades.empty and daily.empty and purchase_audit.empty:
            return {
                "profile": profile,
                "status": "missing_backtest",
                "final_assets": None,
                "net_profit": None,
                "win_rate": None,
                "profit_factor": None,
                "max_drawdown": None,
                "total_trades": 0,
                "average_holding_days": None,
                "monthly_win_rate": None,
                "losing_months": None,
                "best_month": None,
                "worst_month": None,
                "capital_utilization": None,
                "scaled_buy_count": 0,
                "continue_candidate_buy_count": 0,
                "skipped_but_affordable_count": 0,
                "rejected_by_daily_limit_count": 0,
                "purchase_audit_rows": 0,
            }
        stats = self._trade_stats(trades)
        monthly = self._monthly_rows(trades)
        best_month = max(monthly, key=lambda row: row.get("net_profit") or -10**18, default={})
        worst_month = min(monthly, key=lambda row: row.get("net_profit") or 10**18, default={})
        return {
            "profile": profile,
            "final_assets": self._first(payload, ["final_assets", "total_assets"]),
            "net_profit": self._first(payload, ["net_cumulative_profit", "cumulative_profit"]) or stats["net_profit"],
            "win_rate": self._first(payload, ["win_rate"]) or stats["win_rate"],
            "profit_factor": self._first(payload, ["profit_factor"]) or stats["profit_factor"],
            "max_drawdown": self._first(payload, ["max_drawdown"]),
            "total_trades": self._first(payload, ["total_trades", "closed_trade_count", "closed_trades_count"]) or stats["trade_count"],
            "average_holding_days": self._first(payload, ["average_holding_days"]) or (float(pd.to_numeric(trades.get("holding_days"), errors="coerce").mean()) if not trades.empty and "holding_days" in trades.columns else None),
            "monthly_win_rate": sum(1 for row in monthly if (row.get("net_profit") or 0) > 0) / len(monthly) if monthly else None,
            "losing_months": sum(1 for row in monthly if (row.get("net_profit") or 0) < 0),
            "best_month": f"{best_month.get('month')}:{best_month.get('net_profit')}" if best_month else None,
            "worst_month": f"{worst_month.get('month')}:{worst_month.get('net_profit')}" if worst_month else None,
            "capital_utilization": self._capital_utilization(daily),
            "scaled_buy_count": self._scaled_count(trades),
            "continue_candidate_buy_count": self._continue_candidate_buy_count(purchase_audit),
            "skipped_but_affordable_count": int((purchase_audit.get("skip_reason", pd.Series(dtype=str)).astype(str).eq("")).sum()) if not purchase_audit.empty else 0,
            "rejected_by_daily_limit_count": int(purchase_audit.get("skip_reason", pd.Series(dtype=str)).astype(str).str.contains("daily_buy_limit", na=False).sum()) if not purchase_audit.empty else 0,
            "purchase_audit_rows": int(len(purchase_audit)),
        }

    def _capital_utilization(self, daily: pd.DataFrame) -> float | None:
        if daily.empty or "total_assets" not in daily.columns:
            return None
        value_col = (
            "portfolio_value"
            if "portfolio_value" in daily.columns
            else "positions_value"
            if "positions_value" in daily.columns
            else "market_value"
            if "market_value" in daily.columns
            else None
        )
        if not value_col:
            return None
        ratio = pd.to_numeric(daily[value_col], errors="coerce") / pd.to_numeric(daily["total_assets"], errors="coerce")
        return float(ratio.mean()) if not ratio.dropna().empty else None

    def _scaled_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty or "scaled_buy_triggered" not in trades.columns:
            return trades.iloc[0:0].copy()
        series = trades["scaled_buy_triggered"].astype(str).str.lower().isin({"true", "1", "yes"})
        return trades[series].copy()

    def _scaled_count(self, trades: pd.DataFrame) -> int:
        return int(len(self._scaled_trades(trades)))

    def _continue_candidate_buy_count(self, purchase_audit: pd.DataFrame) -> int:
        if purchase_audit.empty or "decision" not in purchase_audit.columns or "candidate_rank" not in purchase_audit.columns:
            return 0
        decision = purchase_audit["decision"].astype(str)
        rank = pd.to_numeric(purchase_audit["candidate_rank"], errors="coerce")
        return int(decision.isin(["BUY", "SCALED_BUY"]).where(rank > 1, False).sum())

    def _trade_stats(self, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty or "net_profit" not in trades.columns:
            return {"trade_count": 0, "net_profit": 0.0, "win_rate": None, "profit_factor": None, "best_trade": None, "worst_trade": None}
        profits = pd.to_numeric(trades["net_profit"], errors="coerce").fillna(0.0)
        gross_profit = float(profits[profits > 0].sum())
        gross_loss = abs(float(profits[profits < 0].sum()))
        return {
            "trade_count": int(len(profits)),
            "net_profit": float(profits.sum()),
            "win_rate": float((profits > 0).mean()) if len(profits) else None,
            "profit_factor": gross_profit / gross_loss if gross_loss else (None if gross_profit == 0 else float("inf")),
            "best_trade": float(profits.max()) if len(profits) else None,
            "worst_trade": float(profits.min()) if len(profits) else None,
        }

    def _monthly_rows(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty or "exit_date" not in trades.columns:
            return []
        df = trades.dropna(subset=["exit_date"]).copy()
        if df.empty:
            return []
        df["month"] = df["exit_date"].dt.to_period("M").astype(str)
        rows = []
        for month, group in df.groupby("month"):
            profits = pd.to_numeric(group["net_profit"], errors="coerce").fillna(0.0)
            rows.append({"month": str(month), "net_profit": float(profits.sum()), "trade_count": int(len(group))})
        return rows

    def _focus_symbol(self, trades: pd.DataFrame, code: str) -> dict[str, Any]:
        if trades.empty:
            return {"bought": False}
        df = trades[trades["code"].astype(str).eq(code)].copy()
        if df.empty:
            return {"bought": False}
        row = df.sort_values("net_profit", ascending=False).iloc[0].to_dict()
        return {
            "bought": True,
            "entry_date": str(row.get("entry_date", ""))[:10],
            "exit_date": str(row.get("exit_date", ""))[:10],
            "shares": row.get("shares"),
            "amount": row.get("amount"),
            "net_profit": row.get("net_profit"),
            "net_profit_rate": row.get("net_profit_rate"),
            "exit_reason": row.get("exit_reason"),
        }

    def _scaled_summary(self, scaled_trades: pd.DataFrame) -> dict[str, Any]:
        return self._trade_stats(scaled_trades)

    def _purchase_audit_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"rows": 0, "buy_count": 0, "scaled_buy_count": 0, "skip_count": 0, "daily_limit_skip_count": 0, "duplicate_skip_count": 0}
        decision = df.get("decision", pd.Series(dtype=str)).astype(str)
        skip_reason = df.get("skip_reason", pd.Series(dtype=str)).astype(str)
        return {
            "rows": int(len(df)),
            "buy_count": int(decision.eq("BUY").sum()),
            "scaled_buy_count": int(decision.eq("SCALED_BUY").sum()),
            "skip_count": int(decision.eq("SKIP").sum()),
            "daily_limit_skip_count": int(skip_reason.str.contains("daily_buy_limit", na=False).sum()),
            "duplicate_skip_count": int(skip_reason.eq("duplicate_holding").sum()),
        }

    def _purchase_audit_decisions(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty or "decision" not in df.columns:
            return []
        return [{"decision": str(decision), "count": int(count)} for decision, count in df["decision"].astype(str).value_counts().items()]

    def _diagnosis(self, summary: list[dict[str, Any]], focus_trades: pd.DataFrame, scaled_trades: pd.DataFrame, purchase_audit: pd.DataFrame) -> list[str]:
        focus = self._focus_symbol(focus_trades, "67400")
        scaled_without = self._trade_stats(scaled_trades[~scaled_trades["code"].astype(str).eq("67400")]) if not scaled_trades.empty else self._trade_stats(scaled_trades)
        v72 = next((row for row in summary if row["profile"] == self.focus_profile), {})
        v71 = next((row for row in summary if "v2_71" in row["profile"]), {})
        return [
            f"67400 bought by v2_72: {focus.get('bought')} profit={focus.get('net_profit')}",
            f"v2_72 net_profit={v72.get('net_profit')} vs v2_71 net_profit={v71.get('net_profit')}",
            f"scaled-buy excluding 67400 net_profit={scaled_without.get('net_profit')}",
            f"purchase_audit rows={len(purchase_audit)}",
        ]

    def _first(self, payload: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if payload.get(key) is not None:
                return payload[key]
        return None

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("|" + "|".join(self._format(row.get(column)) for column in columns) + "|")
        return "\n".join(lines)

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("\n", " ")
