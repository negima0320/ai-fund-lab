from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.position_sizing_phase1 import DEFAULT_PROFILES
from ml.position_sizing_phase1 import ML_COLUMNS
from ml.position_sizing_phase1 import PositionSizingPhase1Simulation


@dataclass(frozen=True)
class PositionSizingPhase2Paths:
    markdown: Path
    json: Path
    summary_csv: Path


class PositionSizingPhase2SoftRules(PositionSizingPhase1Simulation):
    def _multipliers(self, trades: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
        if trades.empty:
            return [("baseline", "multiplier = 1.0", pd.Series(dtype=float))]
        bad = pd.to_numeric(trades.get("bad_entry_probability_10d"), errors="coerce")
        expected = pd.to_numeric(trades.get("expected_return_10d"), errors="coerce")
        baseline = pd.Series(1.0, index=trades.index)

        bad_soft = pd.Series(1.0, index=trades.index)
        bad_soft = bad_soft.mask(bad < 0.40, 1.2)
        bad_soft = bad_soft.mask(bad > 0.70, 0.7)
        bad_soft = bad_soft.fillna(1.0)

        bad_very_soft = pd.Series(1.0, index=trades.index)
        bad_very_soft = bad_very_soft.mask(bad < 0.40, 1.1)
        bad_very_soft = bad_very_soft.mask(bad > 0.70, 0.8)
        bad_very_soft = bad_very_soft.fillna(1.0)

        expected_soft = pd.Series(1.0, index=trades.index)
        expected_soft = expected_soft.mask(expected >= 0.05, 1.3)
        expected_soft = expected_soft.mask((expected >= 0.03) & (expected < 0.05), 1.15)
        expected_soft = expected_soft.mask(expected < 0.01, 0.8)
        expected_soft = expected_soft.fillna(1.0)

        combined_soft = pd.Series(1.0, index=trades.index)
        combined_soft = combined_soft.mask((bad < 0.40) & (expected >= 0.03), 1.2)
        combined_soft = combined_soft.mask((bad > 0.70) | (expected < 0.01), 0.8)
        combined_soft = combined_soft.fillna(1.0)

        return [
            ("baseline", "multiplier = 1.0", baseline),
            ("bad_entry_defensive_soft", "bad_entry <0.40=1.2, 0.40-0.70=1.0, >0.70=0.7", bad_soft),
            ("bad_entry_defensive_very_soft", "bad_entry <0.40=1.1, 0.40-0.70=1.0, >0.70=0.8", bad_very_soft),
            ("expected_return_soft", "expected >=0.05=1.3, 0.03-0.05=1.15, 0.01-0.03=1.0, <0.01=0.8", expected_soft),
            ("combined_soft", "bad<0.40 and expected>=0.03=1.2, bad>0.70 or expected<0.01=0.8, otherwise=1.0", combined_soft),
        ]

    def save(self, result: dict[str, Any]) -> PositionSizingPhase2Paths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "position_sizing_phase2_soft_rules_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        csv_path = self.report_dir / f"{stem}_summary.csv"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps({k: v for k, v in result.items() if k != "trades"}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        self._write_csv(csv_path, result.get("summary") or [])
        return PositionSizingPhase2Paths(markdown=markdown, json=json_path, summary_csv=csv_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        columns = [
            "profile",
            "sizing_rule",
            "adjusted_net_profit",
            "profit_delta",
            "profit_factor",
            "max_drawdown",
            "win_rate",
            "monthly_win_rate",
            "losing_months",
            "worst_trade",
            "best_trade",
            "focus_67400_contribution",
            "top3_trade_contribution",
            "average_multiplier",
        ]
        lines = [
            "# Position Sizing Phase 2 Soft Rules",
            "",
            f"- period: {self.start_date} to {self.end_date}",
            "- method: post-trade simulation; symbols/timing are fixed and only net_profit is multiplied.",
            "- source: existing backtest logs plus purchase_audit/walk-forward predictions; no backtest rerun, no API fetch, no live orders.",
            "",
            "## ML Join Summary",
            "",
            self._table(result.get("join_summary", []), ["profile", "trade_count", "ml_joined_count", "ml_join_rate", "purchase_audit_join_count", "prediction_fallback_join_count"]),
            "",
            "## Soft Sizing Rule Summary",
            "",
            self._table(result["summary"], columns),
            "",
            "## Best Rules",
            "",
            self._table(
                [
                    {"metric": "net_profit", **(result.get("best_by_net_profit") or {})},
                    {"metric": "profit_factor", **(result.get("best_by_profit_factor") or {})},
                    {"metric": "drawdown", **(result.get("best_by_drawdown") or {})},
                ],
                ["metric", *columns[:6]],
            ),
            "",
            "## Diagnosis",
            "",
        ]
        lines.extend(f"- {item}" for item in result.get("diagnosis", []))
        lines.append("")
        return "\n".join(lines)

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


def build_default_phase2(root: str | Path = ".") -> PositionSizingPhase2SoftRules:
    return PositionSizingPhase2SoftRules(root=root, profiles=list(DEFAULT_PROFILES))
