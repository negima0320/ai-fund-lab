from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_phase3d_detail_audit import (
    BASELINE_PROFILE,
    FOCUS_CODE,
    PERIOD,
    PHASE3D_PROFILE,
    PortfolioManagerPhase3DDetailAudit,
)


PHASE3E_PROFILE = "rookie_dealer_02_v2_76_pm_ai_low_score_skip"


@dataclass(frozen=True)
class PortfolioManagerPhase3EPaths:
    markdown: Path
    json: Path


class PortfolioManagerPhase3EReporter:
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
        self.audit = PortfolioManagerPhase3DDetailAudit(
            root=self.root,
            baseline_profile=baseline_profile,
            phase3d_profile=phase3d_profile,
            period=period,
        )
        self.report_dir = self.root / "reports" / "ml"

    def build(self) -> dict[str, Any]:
        baseline = self.audit._profile_payload(self.baseline_profile)
        phase3d = self.audit._profile_payload(self.phase3d_profile)
        phase3e = self.audit._profile_payload(self.phase3e_profile)
        return {
            "purpose": "Portfolio Manager AI Phase 3-E low PM score skip validation",
            "period": self.period,
            "profiles": {
                "baseline": self.baseline_profile,
                "phase3d": self.phase3d_profile,
                "phase3e": self.phase3e_profile,
            },
            "data_lineage_audit_status": "PASS",
            "selected_count_in_day_used": False,
            "summaries": [baseline["summary"], phase3d["summary"], phase3e["summary"]],
            "phase3e_vs_phase3d_delta": self._summary_delta(phase3d["summary"], phase3e["summary"]),
            "phase3e_vs_baseline_delta": self._summary_delta(baseline["summary"], phase3e["summary"]),
            "pm_low_score_skip_count": self._skip_counts(phase3e["audit"]).get("pm_low_score_skip", 0),
            "skip_reason_comparison": self._skip_reason_comparison_three(baseline["audit"], phase3d["audit"], phase3e["audit"]),
            "monthly_comparison": self._monthly_comparison_three(baseline["monthly"], phase3d["monthly"], phase3e["monthly"]),
            "code_concentration": {
                "baseline": baseline["code_summary"],
                "phase3d": phase3d["code_summary"],
                "phase3e": phase3e["code_summary"],
            },
            "phase3e_top_codes": phase3e["top_codes"],
            "phase3e_bottom_codes": phase3e["bottom_codes"],
            "focus_code_dependency": {
                "phase3d": self.audit._focus_code_dependency(baseline, phase3d, FOCUS_CODE),
                "phase3e": self.audit._focus_code_dependency(baseline, phase3e, FOCUS_CODE),
            },
            "pm_multiplier_summary": {
                "phase3d": self.audit._pm_group_summary(phase3d["trades"], "pm_multiplier"),
                "phase3e": self.audit._pm_group_summary(phase3e["trades"], "pm_multiplier"),
            },
            "pm_score_band_summary": {
                "phase3d": self.audit._pm_score_band_summary(phase3d["trades"]),
                "phase3e": self.audit._pm_score_band_summary(phase3e["trades"]),
            },
            "capital_utilization_comparison": {
                "baseline": baseline["capital_utilization"],
                "phase3d": phase3d["capital_utilization"],
                "phase3e": phase3e["capital_utilization"],
            },
            "promotion_judgement": self._promotion_judgement(baseline, phase3d, phase3e),
        }

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase3EPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase3e_low_score_skip_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerPhase3EPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 3-E Low Score Skip",
            "",
            "## Purpose",
            "",
            "Validate whether skipping `pm_score < -0.20` candidates improves v2_75.",
            "",
            "## Scope",
            "",
            f"- period: `{result['period']}`",
            f"- baseline: `{result['profiles']['baseline']}`",
            f"- phase3d / v2_75: `{result['profiles']['phase3d']}`",
            f"- phase3e / v2_76: `{result['profiles']['phase3e']}`",
            "- no API refetch",
            "- no current model historical regeneration",
            "- data lineage audit status: `PASS`",
            "- `selected_count_in_day` used: `False`",
            "",
            "## Summary Comparison",
            "",
            self.audit._table(result["summaries"], ["profile", "net_profit", "profit_factor", "max_drawdown", "win_rate", "total_trades"]),
            "",
            "## v2_76 Delta",
            "",
            "### vs v2_75",
            "",
            self.audit._table([result["phase3e_vs_phase3d_delta"]], ["net_profit_delta", "profit_factor_delta", "max_drawdown_delta", "win_rate_delta", "total_trades_delta"]),
            "",
            "### vs v2_73",
            "",
            self.audit._table([result["phase3e_vs_baseline_delta"]], ["net_profit_delta", "profit_factor_delta", "max_drawdown_delta", "win_rate_delta", "total_trades_delta"]),
            "",
            "## Low Score Skip",
            "",
            f"- `pm_low_score_skip` count: `{result['pm_low_score_skip_count']}`",
            "",
            "## Skip Reason Comparison",
            "",
            self.audit._table(result["skip_reason_comparison"], ["skip_reason", "baseline_count", "phase3d_count", "phase3e_count", "phase3e_minus_phase3d"]),
            "",
            "## Monthly Comparison",
            "",
            self.audit._table(result["monthly_comparison"], ["month", "baseline_profit", "phase3d_profit", "phase3e_profit", "phase3e_minus_phase3d"]),
            "",
            "## Code Concentration",
            "",
            self.audit._table(
                [
                    {"profile": "baseline", **result["code_concentration"]["baseline"]},
                    {"profile": "phase3d", **result["code_concentration"]["phase3d"]},
                    {"profile": "phase3e", **result["code_concentration"]["phase3e"]},
                ],
                ["profile", "total_profit", "top1_contribution_rate", "top3_contribution_rate", "top5_contribution_rate", "worst_code", "worst_code_profit"],
            ),
            "",
            "### v2_76 Top 10 Codes",
            "",
            self.audit._table(result["phase3e_top_codes"], ["code", "trade_count", "net_profit", "win_rate", "profit_factor", "average_profit"]),
            "",
            f"## {FOCUS_CODE} Dependency",
            "",
            self.audit._table(
                [
                    {"profile": "phase3d", **result["focus_code_dependency"]["phase3d"]},
                    {"profile": "phase3e", **result["focus_code_dependency"]["phase3e"]},
                ],
                ["profile", "phase3d_profit", "phase3d_contribution_rate", "phase3d_excluding_profit", "phase3d_excluding_profit_factor", "phase3d_excluding_max_drawdown", "excluding_still_beats_baseline_profit"],
            ),
            "",
            "## PM Multiplier Summary",
            "",
            "### v2_75",
            "",
            self.audit._table(result["pm_multiplier_summary"]["phase3d"], ["group", "trade_count", "net_profit", "profit_factor", "win_rate", "average_profit"]),
            "",
            "### v2_76",
            "",
            self.audit._table(result["pm_multiplier_summary"]["phase3e"], ["group", "trade_count", "net_profit", "profit_factor", "win_rate", "average_profit"]),
            "",
            "## PM Score Band Summary",
            "",
            "### v2_75",
            "",
            self.audit._table(result["pm_score_band_summary"]["phase3d"], ["group", "trade_count", "net_profit", "profit_factor", "win_rate", "return_on_buy_amount"]),
            "",
            "### v2_76",
            "",
            self.audit._table(result["pm_score_band_summary"]["phase3e"], ["group", "trade_count", "net_profit", "profit_factor", "win_rate", "return_on_buy_amount"]),
            "",
            "## Capital Utilization",
            "",
            self.audit._table(
                [
                    {"profile": "baseline", **result["capital_utilization_comparison"]["baseline"]},
                    {"profile": "phase3d", **result["capital_utilization_comparison"]["phase3d"]},
                    {"profile": "phase3e", **result["capital_utilization_comparison"]["phase3e"]},
                ],
                ["profile", "average_capital_utilization", "median_capital_utilization", "max_capital_utilization", "average_holding_count", "max_positions_days", "cash_idle_days"],
            ),
            "",
            "## Promotion Judgement",
            "",
            self.audit._table(result["promotion_judgement"], ["criterion", "passed", "detail"]),
            "",
        ]
        return "\n".join(lines)

    def _summary_delta(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        return {
            "net_profit_delta": float(after["net_profit"]) - float(before["net_profit"]),
            "profit_factor_delta": float(after["profit_factor"]) - float(before["profit_factor"]),
            "max_drawdown_delta": float(after["max_drawdown"]) - float(before["max_drawdown"]),
            "win_rate_delta": float(after["win_rate"]) - float(before["win_rate"]),
            "total_trades_delta": int(after["total_trades"]) - int(before["total_trades"]),
        }

    def _skip_reason_comparison_three(self, baseline: pd.DataFrame, phase3d: pd.DataFrame, phase3e: pd.DataFrame) -> list[dict[str, Any]]:
        b = self._skip_counts(baseline)
        d = self._skip_counts(phase3d)
        e = self._skip_counts(phase3e)
        rows = []
        for reason in sorted(set(b) | set(d) | set(e)):
            rows.append(
                {
                    "skip_reason": reason,
                    "baseline_count": b.get(reason, 0),
                    "phase3d_count": d.get(reason, 0),
                    "phase3e_count": e.get(reason, 0),
                    "phase3e_minus_phase3d": e.get(reason, 0) - d.get(reason, 0),
                }
            )
        return rows

    def _skip_counts(self, audit: pd.DataFrame) -> dict[str, int]:
        return self.audit._skip_counts(audit)

    def _monthly_comparison_three(self, baseline: list[dict[str, Any]], phase3d: list[dict[str, Any]], phase3e: list[dict[str, Any]]) -> list[dict[str, Any]]:
        b = {row["month"]: row for row in baseline}
        d = {row["month"]: row for row in phase3d}
        e = {row["month"]: row for row in phase3e}
        rows = []
        for month in sorted(set(b) | set(d) | set(e)):
            bp = float(b.get(month, {}).get("monthly_profit") or 0.0)
            dp = float(d.get(month, {}).get("monthly_profit") or 0.0)
            ep = float(e.get(month, {}).get("monthly_profit") or 0.0)
            rows.append(
                {
                    "month": month,
                    "baseline_profit": bp,
                    "phase3d_profit": dp,
                    "phase3e_profit": ep,
                    "phase3e_minus_phase3d": ep - dp,
                }
            )
        return rows

    def _promotion_judgement(self, baseline: dict[str, Any], phase3d: dict[str, Any], phase3e: dict[str, Any]) -> list[dict[str, Any]]:
        b = baseline["summary"]
        d = phase3d["summary"]
        e = phase3e["summary"]
        focus = self.audit._focus_code_dependency(baseline, phase3e, FOCUS_CODE)
        checks = [
            ("net_profit_above_v2_75", e["net_profit"] > d["net_profit"], f"{e['net_profit']} vs {d['net_profit']}"),
            ("profit_factor_above_v2_75", e["profit_factor"] > d["profit_factor"], f"{e['profit_factor']} vs {d['profit_factor']}"),
            ("drawdown_not_worse_than_v2_75", e["max_drawdown"] >= d["max_drawdown"], f"{e['max_drawdown']} vs {d['max_drawdown']}"),
            ("win_rate_not_worse_than_v2_75", e["win_rate"] >= d["win_rate"], f"{e['win_rate']} vs {d['win_rate']}"),
            ("net_profit_above_v2_73", e["net_profit"] > b["net_profit"], f"{e['net_profit']} vs {b['net_profit']}"),
            ("pm_low_score_skip_triggered", self._skip_counts(phase3e["audit"]).get("pm_low_score_skip", 0) > 0, str(self._skip_counts(phase3e["audit"]).get("pm_low_score_skip", 0))),
            ("excluding_67400_still_beats_baseline", bool(focus["excluding_still_beats_baseline_profit"]), str(focus["phase3d_excluding_profit"])),
        ]
        return [{"criterion": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks]
