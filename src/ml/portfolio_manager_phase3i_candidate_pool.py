from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profile_loader import load_profile, load_yaml_config

from ml.portfolio_manager_phase3d_detail_audit import PERIOD
from ml.portfolio_manager_phase3h_capital_utilization import PortfolioManagerPhase3HCapitalUtilizationAudit


CURRENT_PROFILE = "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030"
POOL_X2_PROFILE = "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030_pool_x2"
POOL_X3_PROFILE = "rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030_pool_x3"


@dataclass(frozen=True)
class PortfolioManagerPhase3IPaths:
    markdown: Path
    json: Path


class PortfolioManagerPhase3ICandidatePoolAudit:
    def __init__(
        self,
        root: str | Path = ".",
        current_profile: str = CURRENT_PROFILE,
        pool_x2_profile: str = POOL_X2_PROFILE,
        pool_x3_profile: str = POOL_X3_PROFILE,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.current_profile = current_profile
        self.pool_x2_profile = pool_x2_profile
        self.pool_x3_profile = pool_x3_profile
        self.period = period
        self.capital_audit = PortfolioManagerPhase3HCapitalUtilizationAudit(root=self.root, period=period)
        self.report_dir = self.root / "reports" / "ml"

    def build(self) -> dict[str, Any]:
        profiles = {
            "current": self.current_profile,
            "candidate_pool_x2": self.pool_x2_profile,
            "candidate_pool_x3": self.pool_x3_profile,
        }
        payloads = {label: self.capital_audit._profile_payload(profile) for label, profile in profiles.items()}
        settings = {label: self._candidate_pool_settings(profile) for label, profile in profiles.items()}
        comparison = [self._comparison_row(label, payloads[label], settings[label]) for label in profiles]
        current = comparison[0]
        for row in comparison:
            row["no_candidates_improvement_vs_current"] = current["no_candidates_count"] - row["no_candidates_count"]
            row["capital_utilization_delta_vs_current"] = row["average_capital_utilization"] - current["average_capital_utilization"]
        return {
            "purpose": "Portfolio Manager AI Phase 3-I candidate pool expansion audit",
            "period": self.period,
            "constraints": {
                "api_refetch": False,
                "openai_api": False,
                "historical_predictions_source": "data/ml/walk_forward_predictions/",
                "current_model_historical_regeneration": False,
                "selected_count_in_day_used": False,
                "trading_logic_changed_for_existing_profiles": False,
                "live_order_placement": False,
            },
            "profiles": profiles,
            "candidate_pool_settings": settings,
            "comparison": comparison,
            "low_utilization_reason_counts": {
                label: self._low_reason_counts(payload)
                for label, payload in payloads.items()
            },
            "judgement": self._judgement(comparison),
            "next_actions": self._next_actions(comparison),
        }

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase3IPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase3i_candidate_pool_expansion_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerPhase3IPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        table = self.capital_audit.detail._table
        lines = [
            "# Portfolio Manager AI Phase 3-I Candidate Pool Expansion Audit",
            "",
            "## Scope",
            "",
            f"- period: `{result['period']}`",
            f"- current: `{result['profiles']['current']}`",
            f"- candidate_pool_x2: `{result['profiles']['candidate_pool_x2']}`",
            f"- candidate_pool_x3: `{result['profiles']['candidate_pool_x3']}`",
            "- no API refetch",
            "- no OpenAI API",
            "- no current model historical regeneration",
            "- `selected_count_in_day` used: `False`",
            "- no existing profile behavior changes",
            "",
            "## Candidate Pool Settings",
            "",
            table(
                [{"variant": label, **settings} for label, settings in result["candidate_pool_settings"].items()],
                ["variant", "profile", "max_selected", "multiplier_vs_current"],
            ),
            "",
            "## Comparison",
            "",
            table(
                result["comparison"],
                [
                    "variant",
                    "profile",
                    "max_selected",
                    "net_profit",
                    "profit_factor",
                    "max_drawdown",
                    "win_rate",
                    "total_trades",
                    "monthly_win_rate",
                    "average_capital_utilization",
                    "median_capital_utilization",
                    "days_below_50pct",
                    "cash_idle_days",
                    "average_holding_count",
                    "no_candidates_count",
                    "exit_only_day_count",
                    "pm_low_score_skip_count",
                    "selected_but_not_affordable_count",
                    "no_candidates_improvement_vs_current",
                    "capital_utilization_delta_vs_current",
                ],
            ),
            "",
            "## Low Utilization Reason Counts",
            "",
            table(
                [
                    {"variant": variant, "reason": reason, "count": count}
                    for variant, counts in result["low_utilization_reason_counts"].items()
                    for reason, count in counts.items()
                ],
                ["variant", "reason", "count"],
            ),
            "",
            "## Judgement",
            "",
            table(result["judgement"], ["criterion", "variant", "passed", "detail"]),
            "",
            "## Next Actions",
            "",
            table(result["next_actions"], ["priority", "action", "reason"]),
            "",
        ]
        return "\n".join(lines)

    def _candidate_pool_settings(self, profile: str) -> dict[str, Any]:
        config = self._load_profile_config(profile)
        max_selected = int(config.get("selection", {}).get("max_selected") or 0)
        current = int(self._load_profile_config(self.current_profile).get("selection", {}).get("max_selected") or max_selected or 1)
        return {
            "profile": profile,
            "max_selected": max_selected,
            "multiplier_vs_current": float(max_selected / current) if current else None,
        }

    def _load_profile_config(self, profile: str) -> dict[str, Any]:
        local_path = self.root / "config" / "profiles" / f"{profile}.yaml"
        if local_path.exists():
            return load_yaml_config(local_path)
        return load_profile(profile)

    def _comparison_row(self, label: str, payload: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
        summary = payload["summary"]
        distribution = self.capital_audit._daily_distribution(payload["daily"])
        low_counts = self._low_reason_counts(payload)
        skip_counts = self.capital_audit._skip_counts(payload["audit"])
        monthly_rows = self.capital_audit._monthly_rows(payload)
        return {
            "variant": label,
            "profile": summary["profile"],
            "max_selected": settings["max_selected"],
            "net_profit": summary["net_profit"],
            "profit_factor": summary["profit_factor"],
            "max_drawdown": summary["max_drawdown"],
            "win_rate": summary["win_rate"],
            "total_trades": summary["total_trades"],
            "monthly_win_rate": self._monthly_win_rate(monthly_rows),
            "average_capital_utilization": distribution.get("average_capital_utilization"),
            "median_capital_utilization": distribution.get("median_capital_utilization"),
            "days_below_50pct": distribution.get("days_below_50pct"),
            "cash_idle_days": distribution.get("cash_idle_days"),
            "average_holding_count": self._average_monthly_value(monthly_rows, "average_holding_count"),
            "no_candidates_count": low_counts.get("no_candidates", 0),
            "exit_only_day_count": low_counts.get("exit_only_day", 0),
            "pm_low_score_skip_count": skip_counts.get("pm_low_score_skip", 0),
            "selected_but_not_affordable_count": skip_counts.get("selected_but_not_affordable", 0),
        }

    def _low_reason_counts(self, payload: dict[str, Any]) -> dict[str, int]:
        rows = self.capital_audit._low_utilization_days(payload)
        counts: dict[str, int] = {}
        for row in rows:
            reason = str(row.get("dominant_reason") or "unknown")
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def _judgement(self, comparison: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for row in comparison[1:]:
            rows.extend(
                [
                    {
                        "criterion": "capital_utilization_60_to_65pct",
                        "variant": row["variant"],
                        "passed": 0.60 <= float(row.get("average_capital_utilization") or 0) <= 0.65,
                        "detail": str(row.get("average_capital_utilization")),
                    },
                    {
                        "criterion": "profit_factor_at_least_2_3",
                        "variant": row["variant"],
                        "passed": float(row.get("profit_factor") or 0) >= 2.3,
                        "detail": str(row.get("profit_factor")),
                    },
                    {
                        "criterion": "drawdown_under_10pct",
                        "variant": row["variant"],
                        "passed": float(row.get("max_drawdown") or 0) >= -0.10,
                        "detail": str(row.get("max_drawdown")),
                    },
                    {
                        "criterion": "net_profit_over_3m",
                        "variant": row["variant"],
                        "passed": float(row.get("net_profit") or 0) >= 3_000_000,
                        "detail": str(row.get("net_profit")),
                    },
                ]
            )
        return rows

    def _next_actions(self, comparison: list[dict[str, Any]]) -> list[dict[str, Any]]:
        current = comparison[0] if comparison else {}
        current_profit = float(current.get("net_profit") or 0)
        current_util = float(current.get("average_capital_utilization") or 0)
        candidates = comparison[1:]
        qualified = [
            row for row in candidates
            if float(row.get("profit_factor") or 0) >= 2.3 and float(row.get("max_drawdown") or 0) >= -0.10
            and (
                float(row.get("net_profit") or 0) > current_profit
                or float(row.get("average_capital_utilization") or 0) >= current_util + 0.02
            )
        ]
        if qualified:
            best = max(qualified, key=lambda row: float(row.get("net_profit") or 0))
            return [
                {
                    "priority": 1,
                    "action": f"{best['variant']}を本命候補として詳細監査",
                    "reason": "PF/DD条件を満たし、候補プール拡大で利益または稼働率の改善余地があるため",
                }
            ]
        return [
            {
                "priority": 1,
                "action": "candidate pool拡大は保留し、別の稼働率改善案へ進む",
                "reason": "候補プール拡大でcurrent比の利益改善または十分な資金稼働率改善が確認できないため",
            }
        ]

    def _monthly_win_rate(self, rows: list[dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return float(sum(1 for row in rows if row.get("monthly_win")) / len(rows))

    def _average_monthly_value(self, rows: list[dict[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(sum(values) / len(values)) if values else None
