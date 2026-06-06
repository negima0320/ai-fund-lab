from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.capital_allocation_phase4 import CapitalAllocationPhase4Comparison


V2_66_PROFILE = "rookie_dealer_02_v2_66_ml_ranked"
V2_70_PROFILE = "rookie_dealer_02_v2_70_ml_ranked_exit_ai_060"
V2_73_PROFILE = "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue"
V2_74_PROFILE = "rookie_dealer_02_v2_74_ml_ranked_exit_ai_affordable_fallback"
DEFAULT_PROFILES = [V2_66_PROFILE, V2_70_PROFILE, V2_73_PROFILE, V2_74_PROFILE]


@dataclass(frozen=True)
class CapitalAllocationPhase7Paths:
    markdown: Path
    json: Path
    fallback_trades_csv: Path


class CapitalAllocationPhase7AffordableFallback(CapitalAllocationPhase4Comparison):
    def __init__(
        self,
        root: str | Path = ".",
        profiles: list[str] | None = None,
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        focus_profile: str = V2_74_PROFILE,
    ) -> None:
        super().__init__(
            root=root,
            profiles=profiles or list(DEFAULT_PROFILES),
            start_date=start_date,
            end_date=end_date,
            focus_profile=focus_profile,
        )

    def run_missing_backtests(self) -> list[dict[str, Any]]:
        rows = []
        for profile in self.profiles:
            backtest_dir = self._backtest_dir(profile)
            if (backtest_dir / "backtest_summary.json").exists():
                rows.append({"profile": profile, "status": "skipped_existing"})
                continue
            cmd = [
                sys.executable,
                "src/main.py",
                "--mode",
                "backtest",
                "--provider",
                "jquants",
                "--profile",
                profile,
                "--start-date",
                self.start_date,
                "--end-date",
                self.end_date,
                "--skip-price-fetch",
                "--summary-only",
                "--no-daily-logs",
                "--quiet",
                "--progress-interval",
                "100",
            ]
            completed = subprocess.run(cmd, cwd=self.root, check=False)
            rows.append({"profile": profile, "status": "ok" if completed.returncode == 0 else "failed", "returncode": completed.returncode})
            if completed.returncode != 0:
                break
        return rows

    def build(self) -> dict[str, Any]:
        rows = []
        for profile in self.profiles:
            trades = self._load_trades(profile)
            payload = self._load_summary(profile)
            daily = self._load_daily(profile)
            purchase_audit = self._load_purchase_audit(profile)
            row = {
                **self._summary_row(profile, payload, trades, daily, purchase_audit),
                "average_holding_count": self._average_holding_count(daily),
                **self._source_stats(trades, purchase_audit),
            }
            rows.append(row)
        focus_trades = self._load_trades(self.focus_profile)
        focus_audit = self._load_purchase_audit(self.focus_profile)
        fallback_trades = self._fallback_trades(focus_trades)
        selected_trades = self._selected_trades(focus_trades)
        v73 = next((row for row in rows if row.get("profile") == V2_73_PROFILE), {})
        v74 = next((row for row in rows if row.get("profile") == V2_74_PROFILE), {})
        return {
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "profiles": self.profiles,
            "focus_profile": self.focus_profile,
            "summary": rows,
            "v2_73_to_v2_74_delta": self._delta(v73, v74),
            "fallback_summary": self._trade_stats(fallback_trades),
            "selected_summary": self._trade_stats(selected_trades),
            "fallback_purchase_audit": self._purchase_audit_source_summary(focus_audit, "fallback"),
            "selected_purchase_audit": self._purchase_audit_source_summary(focus_audit, "selected"),
            "fallback_trades": fallback_trades.to_dict(orient="records") if not fallback_trades.empty else [],
            "diagnosis": self._diagnosis(rows, fallback_trades),
        }

    def save(self, result: dict[str, Any]) -> CapitalAllocationPhase7Paths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "capital_allocation_phase7_affordable_fallback_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        fallback_csv = self.report_dir / "v2_74_fallback_trades_2023-01_to_2026-05.csv"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        fallback_df = pd.DataFrame(result.get("fallback_trades") or [])
        fallback_df.to_csv(fallback_csv, index=False)
        return CapitalAllocationPhase7Paths(markdown=markdown, json=json_path, fallback_trades_csv=fallback_csv)

    def format_markdown(self, result: dict[str, Any]) -> str:
        columns = [
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
            "capital_utilization",
            "average_holding_count",
            "fallback_buy_count",
            "fallback_profit",
            "fallback_win_rate",
            "fallback_profit_factor",
            "selected_buy_count",
            "selected_profit",
            "selected_profit_factor",
        ]
        lines = [
            "# Capital Allocation Phase 7 Affordable Fallback",
            "",
            f"- period: {self.start_date} to {self.end_date}",
            f"- focus_profile: `{self.focus_profile}`",
            "- source: generated/existing backtest logs; no OpenAI API, no live orders.",
            "",
            "## Summary",
            "",
            self._table(result["summary"], columns),
            "",
            "## v2_73 to v2_74 Delta",
            "",
            self._table([result["v2_73_to_v2_74_delta"]], ["final_assets", "net_profit", "profit_factor", "max_drawdown", "capital_utilization", "average_holding_count", "total_trades"]),
            "",
            "## v2_74 Fallback Performance",
            "",
            self._table([result["fallback_summary"]], ["trade_count", "net_profit", "win_rate", "profit_factor", "best_trade", "worst_trade"]),
            "",
            "## v2_74 Selected Performance",
            "",
            self._table([result["selected_summary"]], ["trade_count", "net_profit", "win_rate", "profit_factor", "best_trade", "worst_trade"]),
            "",
            "## Purchase Audit Source Summary",
            "",
            self._table([result["selected_purchase_audit"], result["fallback_purchase_audit"]], ["candidate_source", "rows", "buy_count", "scaled_buy_count", "skip_count", "final_amount"]),
            "",
            "## Diagnosis",
            "",
        ]
        lines.extend(f"- {line}" for line in result["diagnosis"])
        lines.append("")
        return "\n".join(lines)

    def _load_purchase_audit(self, profile: str) -> pd.DataFrame:
        df = super()._load_purchase_audit(profile)
        if df.empty:
            return df
        for column in ["fallback_rank", "raw_candidate_rank", "candidate_rank", "score_rank", "final_shares"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        if "candidate_source" not in df.columns:
            df["candidate_source"] = "selected"
        df["candidate_source"] = df["candidate_source"].fillna("selected").replace("", "selected")
        return df

    def _average_holding_count(self, daily: pd.DataFrame) -> float | None:
        if daily.empty or "open_positions_count" not in daily.columns:
            return None
        values = pd.to_numeric(daily["open_positions_count"], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    def _source_stats(self, trades: pd.DataFrame, purchase_audit: pd.DataFrame) -> dict[str, Any]:
        fallback = self._fallback_trades(trades)
        selected = self._selected_trades(trades)
        fallback_stats = self._trade_stats(fallback)
        selected_stats = self._trade_stats(selected)
        fallback_audit = self._purchase_audit_source_summary(purchase_audit, "fallback")
        selected_audit = self._purchase_audit_source_summary(purchase_audit, "selected")
        return {
            "fallback_buy_count": int(fallback_audit["buy_count"]),
            "fallback_closed_trade_count": int(fallback_stats["trade_count"]),
            "fallback_profit": fallback_stats["net_profit"],
            "fallback_win_rate": fallback_stats["win_rate"],
            "fallback_profit_factor": fallback_stats["profit_factor"],
            "selected_buy_count": int(selected_audit["buy_count"]),
            "selected_closed_trade_count": int(selected_stats["trade_count"]),
            "selected_profit": selected_stats["net_profit"],
            "selected_win_rate": selected_stats["win_rate"],
            "selected_profit_factor": selected_stats["profit_factor"],
        }

    def _fallback_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        if "candidate_source" in trades.columns:
            mask = trades["candidate_source"].fillna("").astype(str).eq("fallback")
        elif "affordable_fallback_buy_selected" in trades.columns:
            mask = self._truthy_series(trades["affordable_fallback_buy_selected"])
        else:
            mask = pd.Series(False, index=trades.index)
        return trades[mask].copy()

    def _selected_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        if "candidate_source" in trades.columns:
            return trades[~trades["candidate_source"].fillna("").astype(str).eq("fallback")].copy()
        if "affordable_fallback_buy_selected" in trades.columns:
            return trades[~self._truthy_series(trades["affordable_fallback_buy_selected"])].copy()
        return trades.copy()

    def _purchase_audit_source_summary(self, purchase_audit: pd.DataFrame, source: str) -> dict[str, Any]:
        if purchase_audit.empty:
            return {"candidate_source": source, "rows": 0, "buy_count": 0, "scaled_buy_count": 0, "skip_count": 0, "final_amount": 0.0}
        df = purchase_audit.copy()
        if "candidate_source" not in df.columns:
            df["candidate_source"] = "selected"
        df["candidate_source"] = df["candidate_source"].fillna("selected").replace("", "selected")
        df = df[df["candidate_source"].astype(str).eq(source)]
        decision = df.get("decision", pd.Series(dtype=str)).astype(str)
        return {
            "candidate_source": source,
            "rows": int(len(df)),
            "buy_count": int(decision.isin(["BUY", "SCALED_BUY"]).sum()),
            "scaled_buy_count": int(decision.eq("SCALED_BUY").sum()),
            "skip_count": int(decision.eq("SKIP").sum()),
            "final_amount": float(pd.to_numeric(df.get("final_amount", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not df.empty else 0.0,
        }

    def _delta(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        keys = ["final_assets", "net_profit", "profit_factor", "max_drawdown", "capital_utilization", "average_holding_count", "total_trades"]
        return {key: self._number(after.get(key)) - self._number(before.get(key)) for key in keys}

    def _diagnosis(self, rows: list[dict[str, Any]], fallback_trades: pd.DataFrame) -> list[str]:
        v73 = next((row for row in rows if row.get("profile") == V2_73_PROFILE), {})
        v74 = next((row for row in rows if row.get("profile") == V2_74_PROFILE), {})
        lines = []
        if v73 and v74:
            lines.append(
                f"capital_utilization {self._format(v73.get('capital_utilization'))} -> {self._format(v74.get('capital_utilization'))}; "
                f"average_holding_count {self._format(v73.get('average_holding_count'))} -> {self._format(v74.get('average_holding_count'))}."
            )
            lines.append(
                f"net_profit delta={self._format(self._number(v74.get('net_profit')) - self._number(v73.get('net_profit')))}, "
                f"max_drawdown delta={self._format(self._number(v74.get('max_drawdown')) - self._number(v73.get('max_drawdown')))}."
            )
        fallback_stats = self._trade_stats(fallback_trades)
        if fallback_stats["trade_count"]:
            lines.append(
                f"fallback closed trades={fallback_stats['trade_count']} net_profit={self._format(fallback_stats['net_profit'])} "
                f"PF={self._format(fallback_stats['profit_factor'])} win_rate={self._format(fallback_stats['win_rate'])}."
            )
        else:
            lines.append("fallback closed trades are zero; affordability fallback did not execute in this run.")
        return lines

    def _truthy_series(self, series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})

    def _number(self, value: Any) -> float:
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0
