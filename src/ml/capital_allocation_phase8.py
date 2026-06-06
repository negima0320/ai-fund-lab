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

from ml.capital_allocation_phase7 import (
    CapitalAllocationPhase7AffordableFallback,
    V2_73_PROFILE,
    V2_74_PROFILE,
)


PERIOD = "2023-01-01_to_2026-05-31"

PRIORITY_CONDITIONS = [
    {
        "condition": "risk_adjusted_gte_005",
        "description": "fallback risk_adjusted_score >= 0.05",
        "policy": {"min_risk_adjusted_score": 0.05},
    },
    {
        "condition": "expected_002_bad_entry_lte_070",
        "description": "fallback expected_return_10d >= 0.02 and bad_entry_probability_10d <= 0.70",
        "policy": {"min_expected_return_10d": 0.02, "max_bad_entry_probability_10d": 0.70},
    },
    {
        "condition": "max_fallback_1_per_day",
        "description": "max_fallback_buys_per_day = 1",
        "policy": {"max_fallback_buys_per_day": 1},
    },
]


@dataclass(frozen=True)
class CapitalAllocationPhase8Paths:
    markdown: Path
    json: Path
    summary_csv: Path


class CapitalAllocationPhase8FallbackFilter(CapitalAllocationPhase7AffordableFallback):
    def __init__(
        self,
        root: str | Path = ".",
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        conditions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.conditions = conditions or list(PRIORITY_CONDITIONS)
        self.profile_dir = Path(root) / "config" / "profiles"
        profiles = [V2_73_PROFILE, V2_74_PROFILE, *[self.profile_id_for(condition) for condition in self.conditions]]
        super().__init__(
            root=root,
            profiles=profiles,
            start_date=start_date,
            end_date=end_date,
            focus_profile=V2_74_PROFILE,
        )

    def profile_id_for(self, condition: dict[str, Any]) -> str:
        return f"rookie_dealer_02_v2_74_ml_ranked_exit_ai_affordable_fallback_{condition['condition']}"

    def ensure_profiles(self) -> list[str]:
        base_path = self.profile_dir / f"{V2_74_PROFILE}.yaml"
        base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        profile_ids = [V2_73_PROFILE, V2_74_PROFILE]
        for condition in self.conditions:
            profile_id = self.profile_id_for(condition)
            profile_ids.append(profile_id)
            profile = dict(base)
            profile["profile_id"] = profile_id
            profile["profile_name"] = f"{base.get('profile_name', V2_74_PROFILE)} {condition['condition']}"
            profile["description"] = f"Capital Allocation Phase 8 fallback filter: {condition['description']}"
            policy = dict(profile.get("affordable_fallback_buy") or {})
            policy.update(condition.get("policy") or {})
            profile["affordable_fallback_buy"] = policy
            profile["purchase_audit"] = {
                "enabled": True,
                "note": f"capital_allocation_phase8:{condition['condition']}",
            }
            path = self.profile_dir / f"{profile_id}.yaml"
            path.write_text(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return profile_ids

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
        labels = {V2_73_PROFILE: "baseline_v2_73", V2_74_PROFILE: "baseline_v2_74"}
        labels.update({self.profile_id_for(condition): condition["condition"] for condition in self.conditions})
        descriptions = {self.profile_id_for(condition): condition["description"] for condition in self.conditions}
        for profile in self.profiles:
            trades = self._load_trades(profile)
            payload = self._load_summary(profile)
            daily = self._load_daily(profile)
            purchase_audit = self._load_purchase_audit(profile)
            fallback_trades = self._fallback_trades(trades)
            row = {
                "condition": labels.get(profile, profile),
                "description": descriptions.get(profile, ""),
                **self._summary_row(profile, payload, trades, daily, purchase_audit),
                "average_holding_count": self._average_holding_count(daily),
                **self._source_stats(trades, purchase_audit),
                **self._concentration_metrics(trades),
                **self._fallback_quality(fallback_trades),
            }
            rows.append(row)
        baselines = {
            "v2_73": next((row for row in rows if row.get("profile") == V2_73_PROFILE), {}),
            "v2_74": next((row for row in rows if row.get("profile") == V2_74_PROFILE), {}),
        }
        return {
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "profiles": self.profiles,
            "conditions": self.conditions,
            "summary": rows,
            "best_net_profit": max(rows, key=lambda row: row.get("net_profit") or -10**18) if rows else None,
            "best_profit_factor": max(rows, key=lambda row: row.get("profit_factor") or -10**18) if rows else None,
            "best_drawdown": max(rows, key=lambda row: row.get("max_drawdown") or -10**18) if rows else None,
            "baselines": baselines,
            "diagnosis": self._phase8_diagnosis(rows, baselines),
        }

    def save(self, result: dict[str, Any]) -> CapitalAllocationPhase8Paths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "capital_allocation_phase8_fallback_filter_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        csv_path = self.report_dir / f"{stem}_summary.csv"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        self._write_csv(csv_path, result["summary"])
        return CapitalAllocationPhase8Paths(markdown=markdown, json=json_path, summary_csv=csv_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        columns = [
            "condition",
            "net_profit",
            "profit_factor",
            "max_drawdown",
            "win_rate",
            "total_trades",
            "capital_utilization",
            "average_holding_count",
            "fallback_buy_count",
            "fallback_profit",
            "fallback_profit_factor",
            "fallback_win_rate",
            "losing_months",
            "monthly_win_rate",
            "focus_67400_contribution",
            "top3_trade_contribution",
        ]
        lines = [
            "# Capital Allocation Phase 8 Fallback Filter",
            "",
            f"- period: {self.start_date} to {self.end_date}",
            f"- base_profile: `{V2_74_PROFILE}`",
            "- scope: priority 3-condition screening only; existing profiles are unchanged.",
            "- source: generated/existing backtest logs; no API fetch and no live orders.",
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
            "## Diagnosis",
            "",
        ]
        lines.extend(f"- {item}" for item in result["diagnosis"])
        lines.append("")
        return "\n".join(lines)

    def _fallback_quality(self, fallback_trades: pd.DataFrame) -> dict[str, Any]:
        stats = self._trade_stats(fallback_trades)
        return {
            "fallback_quality_trade_count": stats["trade_count"],
            "fallback_quality_profit": stats["net_profit"],
            "fallback_quality_pf": stats["profit_factor"],
        }

    def _concentration_metrics(self, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty or "net_profit" not in trades.columns:
            return {
                "focus_67400_contribution": None,
                "top1_trade_contribution": None,
                "top3_trade_contribution": None,
            }
        profits = pd.to_numeric(trades["net_profit"], errors="coerce").fillna(0.0)
        total = float(profits.sum())
        sorted_profit = profits.sort_values(ascending=False)
        focus = float(profits[trades["code"].astype(str).eq("67400")].sum()) if "code" in trades.columns else 0.0
        return {
            "focus_67400_contribution": float(focus / total) if total else None,
            "top1_trade_contribution": float(sorted_profit.head(1).sum() / total) if total else None,
            "top3_trade_contribution": float(sorted_profit.head(3).sum() / total) if total else None,
        }

    def _phase8_diagnosis(self, rows: list[dict[str, Any]], baselines: dict[str, dict[str, Any]]) -> list[str]:
        v73 = baselines.get("v2_73") or {}
        v74 = baselines.get("v2_74") or {}
        candidates = [row for row in rows if row.get("profile") not in {V2_73_PROFILE, V2_74_PROFILE}]
        best_pf = max(candidates, key=lambda row: row.get("profit_factor") or -10**18, default={})
        best_dd = max(candidates, key=lambda row: row.get("max_drawdown") or -10**18, default={})
        best_profit = max(candidates, key=lambda row: row.get("net_profit") or -10**18, default={})
        lines = [
            f"v2_73 net_profit={self._format(v73.get('net_profit'))} PF={self._format(v73.get('profit_factor'))} DD={self._format(v73.get('max_drawdown'))}.",
            f"v2_74 net_profit={self._format(v74.get('net_profit'))} PF={self._format(v74.get('profit_factor'))} DD={self._format(v74.get('max_drawdown'))}.",
        ]
        if best_profit:
            lines.append(f"best net_profit condition is {best_profit.get('condition')} net_profit={self._format(best_profit.get('net_profit'))}.")
        if best_pf:
            lines.append(f"PF-focused condition is {best_pf.get('condition')} PF={self._format(best_pf.get('profit_factor'))}.")
        if best_dd:
            lines.append(f"DD-focused condition is {best_dd.get('condition')} DD={self._format(best_dd.get('max_drawdown'))}.")
        if candidates and v73:
            beats_v73 = [row.get("condition") for row in candidates if (row.get("net_profit") or -10**18) > (v73.get("net_profit") or -10**18)]
            lines.append(f"conditions with net_profit above v2_73: {', '.join(beats_v73) if beats_v73 else 'none'}.")
        return lines

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
