from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.capital_allocation_phase4 import CapitalAllocationPhase4Comparison


BASE_PROFILE = "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue"
PERIOD = "2023-01-01_to_2026-05-31"


CONDITIONS = [
    {"condition": "fixed_500000", "mode": "fixed", "limit": 500_000},
    {"condition": "fixed_900000", "mode": "fixed", "limit": 900_000, "profile": BASE_PROFILE},
    {"condition": "fixed_1200000", "mode": "fixed", "limit": 1_200_000},
    {"condition": "fixed_1500000", "mode": "fixed", "limit": 1_500_000},
    {"condition": "fixed_2000000", "mode": "fixed", "limit": 2_000_000},
    {"condition": "asset_ratio_050", "mode": "asset_ratio", "ratio": 0.50},
    {"condition": "asset_ratio_070", "mode": "asset_ratio", "ratio": 0.70},
    {"condition": "asset_ratio_090", "mode": "asset_ratio", "ratio": 0.90},
    {"condition": "asset_ratio_100", "mode": "asset_ratio", "ratio": 1.00},
    {"condition": "unlimited", "mode": "unlimited"},
]


@dataclass(frozen=True)
class DailyBuyLimitSensitivityPaths:
    markdown: Path
    json: Path
    summary_csv: Path


class DailyBuyLimitSensitivity:
    def __init__(
        self,
        root: str | Path = ".",
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        conditions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.start_date = start_date
        self.end_date = end_date
        self.period_key = f"{start_date}_to_{end_date}"
        self.conditions = conditions or CONDITIONS
        self.report_dir = self.root / "reports" / "ml"
        self.profile_dir = self.root / "config" / "profiles"

    def profile_id_for(self, condition: dict[str, Any]) -> str:
        if condition.get("profile"):
            return str(condition["profile"])
        return f"rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue_{condition['condition']}"

    def ensure_profiles(self) -> list[str]:
        base_path = self.profile_dir / f"{BASE_PROFILE}.yaml"
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        profile_ids = []
        for condition in self.conditions:
            profile_id = self.profile_id_for(condition)
            profile_ids.append(profile_id)
            if profile_id == BASE_PROFILE:
                continue
            profile = dict(base)
            profile["profile_id"] = profile_id
            profile["profile_name"] = f"{base.get('profile_name', BASE_PROFILE)} {condition['condition']}"
            profile["description"] = f"Daily buy limit sensitivity: {condition['condition']}"
            scaled_buy = dict(profile.get("scaled_buy") or {})
            scaled_buy["enabled"] = True
            mode = condition["mode"]
            if mode == "fixed":
                scaled_buy["limit_mode"] = "fixed"
                scaled_buy["daily_buy_limit"] = int(condition["limit"])
                profile.setdefault("risk_margin", {})["max_daily_buy_amount"] = int(condition["limit"])
            elif mode == "asset_ratio":
                scaled_buy["limit_mode"] = "asset_ratio"
                scaled_buy["daily_buy_limit_ratio"] = float(condition["ratio"])
            elif mode == "unlimited":
                scaled_buy["limit_mode"] = "unlimited"
            profile["scaled_buy"] = scaled_buy
            profile["purchase_audit"] = {"enabled": True, "note": f"daily_buy_limit_sensitivity:{condition['condition']}"}
            path = self.profile_dir / f"{profile_id}.yaml"
            path.write_text(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return profile_ids

    def run_backtests(self, profile_ids: list[str]) -> list[dict[str, Any]]:
        rows = []
        for profile_id in profile_ids:
            backtest_dir = self.root / "logs" / "backtests" / profile_id / self.period_key
            if (backtest_dir / "backtest_summary.json").exists():
                rows.append({"profile": profile_id, "status": "skipped_existing"})
                continue
            cmd = [
                sys.executable,
                "src/main.py",
                "--mode",
                "backtest",
                "--provider",
                "jquants",
                "--profile",
                profile_id,
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
            rows.append({"profile": profile_id, "status": "ok" if completed.returncode == 0 else "failed", "returncode": completed.returncode})
            if completed.returncode != 0:
                break
        return rows

    def build(self, profile_ids: list[str]) -> dict[str, Any]:
        comparison = CapitalAllocationPhase4Comparison(
            root=self.root,
            profiles=profile_ids,
            start_date=self.start_date,
            end_date=self.end_date,
            focus_profile=BASE_PROFILE,
        )
        result = comparison.build()
        rows = []
        for condition in self.conditions:
            profile_id = self.profile_id_for(condition)
            summary = next((row for row in result["summary"] if row["profile"] == profile_id), None)
            if not summary:
                continue
            trades = comparison._load_trades(profile_id)
            row = {
                **condition,
                **summary,
                "profile": profile_id,
                **self._concentration_metrics(trades),
            }
            rows.append(row)
        return {
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "base_profile": BASE_PROFILE,
            "summary": rows,
            "best_net_profit": max(rows, key=lambda row: row.get("net_profit") or -10**18) if rows else None,
            "best_profit_factor": max(rows, key=lambda row: row.get("profit_factor") or -10**18) if rows else None,
            "best_drawdown": max(rows, key=lambda row: row.get("max_drawdown") or -10**18) if rows else None,
        }

    def save(self, result: dict[str, Any]) -> DailyBuyLimitSensitivityPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "capital_allocation_phase6_daily_buy_limit_sensitivity_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        csv_path = self.report_dir / f"{stem}_summary.csv"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        self._write_csv(csv_path, result["summary"])
        return DailyBuyLimitSensitivityPaths(markdown=markdown, json=json_path, summary_csv=csv_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        columns = [
            "condition",
            "net_profit",
            "profit_factor",
            "max_drawdown",
            "win_rate",
            "total_trades",
            "capital_utilization",
            "scaled_buy_count",
            "purchase_audit_rows",
            "max_position_amount",
            "max_loss",
            "top1_trade_contribution",
            "top3_trade_contribution",
            "focus_67400_contribution",
        ]
        lines = [
            "# Capital Allocation Phase 6 Daily Buy Limit Sensitivity",
            "",
            f"- period: {self.start_date} to {self.end_date}",
            f"- base_profile: `{BASE_PROFILE}`",
            "- source: generated backtest logs; no API fetch and no live orders",
            "",
            "## Summary",
            "",
            self._table(result["summary"], columns),
            "",
            "## Best Conditions",
            "",
            self._table(
                [
                    {"metric": "net_profit", **(result.get("best_net_profit") or {})},
                    {"metric": "profit_factor", **(result.get("best_profit_factor") or {})},
                    {"metric": "drawdown", **(result.get("best_drawdown") or {})},
                ],
                ["metric", *columns[:5]],
            ),
            "",
        ]
        return "\n".join(lines)

    def _concentration_metrics(self, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty or "net_profit" not in trades.columns:
            return {
                "max_position_amount": None,
                "max_loss": None,
                "top1_trade_contribution": None,
                "top3_trade_contribution": None,
                "focus_67400_contribution": None,
            }
        profits = pd.to_numeric(trades["net_profit"], errors="coerce").fillna(0.0)
        total = float(profits.sum())
        sorted_profit = profits.sort_values(ascending=False)
        focus = float(profits[trades["code"].astype(str).eq("67400")].sum()) if "code" in trades.columns else 0.0
        if "amount" in trades.columns:
            max_amount = float(pd.to_numeric(trades["amount"], errors="coerce").max())
        elif {"entry_price", "shares"}.issubset(trades.columns):
            max_amount = float((pd.to_numeric(trades["entry_price"], errors="coerce") * pd.to_numeric(trades["shares"], errors="coerce")).max())
        else:
            max_amount = None
        return {
            "max_position_amount": max_amount,
            "max_loss": float(profits.min()),
            "top1_trade_contribution": float(sorted_profit.head(1).sum() / total) if total else None,
            "top3_trade_contribution": float(sorted_profit.head(3).sum() / total) if total else None,
            "focus_67400_contribution": float(focus / total) if total else None,
        }

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

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
