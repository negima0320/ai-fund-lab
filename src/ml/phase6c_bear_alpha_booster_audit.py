"""Phase 6-C Bear alpha booster audit.

Read-only audit that checks whether v2_78 Bear-regime winners contain a
repeatable alpha signal worth testing as a future booster rule. This module
does not change trading logic, add profiles, execute backtests, or train
models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ml.phase6b_bear_market_winner_audit import (
    BASE_PROFILE,
    PERIOD,
    ROOT,
    Phase6BBearMarketWinnerAudit,
    _mean,
    _profit_factor,
    _win_rate,
)


REPORT_STEM = "phase6c_bear_alpha_booster_audit_2023-01_to_2026-05"


@dataclass(frozen=True)
class Phase6CReportPaths:
    markdown: Path
    json: Path


class Phase6CBearAlphaBoosterAudit(Phase6BBearMarketWinnerAudit):
    def build_report(self) -> dict[str, Any]:
        trades = self._load_trades()
        purchase = self._load_purchase_audit()
        listed = self._load_listed_info()
        regime = self._load_regime()
        enriched = self._enrich_trades(trades, purchase, listed, regime)
        bear = enriched[enriched["regime"].eq("Bear")].copy() if not enriched.empty else pd.DataFrame()
        bull = enriched[enriched["regime"].eq("Bull")].copy() if not enriched.empty else pd.DataFrame()
        bear = self._add_audit_flags(bear)

        condition_rows = self._condition_audit(bear)
        booster_rows = self._booster_audit(bear)
        pm080 = self._pm080_analysis(bear)
        common = self._common_points(bear, condition_rows, pm080)
        verdict = self._phase6c_verdict(bear, bull, condition_rows, booster_rows, pm080)

        return {
            "metadata": {
                "phase": "6-C",
                "audit_only": True,
                "logic_changed": False,
                "profile_added": False,
                "full_backtest_executed": False,
                "full_pytest_executed": False,
                "api_refetch": False,
                "openai_used": False,
                "selected_count_in_day_used": False,
                "live_order_placement": False,
                "profile": self.profile,
                "period": self.period,
            },
            "sources": {
                "trades_csv": str(self._log_dir() / "trades.csv"),
                "purchase_audit_csv": str(self._log_dir() / "purchase_audit.csv"),
                "topix_cache": "data/cache/jquants/topix_prices/*.json",
                "listed_info_cache": "data/cache/jquants/listed_info/*.json",
                "regime_note": "TOPIX rows are de-duplicated by date before MA25/MA75 Bear classification.",
                "liquidity_note": "volume_ratio from trades.csv is used as the liquidity proxy.",
                "market_cap_note": "market_cap numeric value is unavailable in existing logs/cache; scale_category is used as market_cap_band when available.",
            },
            "coverage": {
                "total_trades": int(len(enriched)),
                "bear_trades": int(len(bear)),
                "bull_trades": int(len(bull)),
                "bear_winner_trades": int((pd.to_numeric(bear.get("profit"), errors="coerce") > 0).sum()) if not bear.empty else 0,
            },
            "bear_alpha_condition_audit": condition_rows,
            "booster_virtual_audit": booster_rows,
            "pm_080_analysis": pm080,
            "bear_alpha_common_points": common,
            "verdict": verdict,
        }

    def save_report(self, result: dict[str, Any]) -> Phase6CReportPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase6CReportPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 6-C Bear Alpha Booster Audit",
            "",
            "## Scope",
            "",
            "- audit only",
            "- no logic change, no new profile, no backtest execution",
            "- existing v2_78 logs and local cache only",
            "",
            "## Coverage",
            "",
            self._table([result["coverage"]], ["total_trades", "bear_trades", "bull_trades", "bear_winner_trades"]),
            "",
            "## Bear Alpha Conditions",
            "",
            self._table(result["bear_alpha_condition_audit"], ["condition", "trade_count", "net_profit", "profit_factor", "win_rate", "avg_profit_per_trade", "avg_return", "avg_buy_amount"]),
            "",
            "## Booster Virtual Audit",
            "",
            self._table(result["booster_virtual_audit"], ["rule", "matched_trade_count", "actual_bear_profit", "virtual_profit", "profit_delta", "profit_delta_rate", "virtual_profit_factor", "virtual_win_rate", "note"]),
            "",
            "## PM 0.80 Analysis",
            "",
            self._table([result["pm_080_analysis"]["summary"]], ["trade_count", "net_profit", "profit_factor", "win_rate", "avg_return", "avg_volume_ratio", "avg_holding_days", "judgement"]),
            "",
            self._table(result["pm_080_analysis"]["by_sector"], ["bucket", "trade_count", "net_profit", "profit_factor", "win_rate", "avg_return"]),
            "",
            "## Common Points",
            "",
            self._table([result["bear_alpha_common_points"]], ["common_sector", "common_pm_range", "common_holding_days", "common_liquidity", "common_return_pattern"]),
            "",
            "## Verdict",
            "",
            self._table([result["verdict"]], ["bear_alpha_exists", "bear_alpha_already_captured_by_pm", "bear_alpha_booster_worth_testing", "booster_rule_recommended", "expected_profit_direction", "expected_dd_direction", "next_phase_recommended"]),
            "",
        ]
        return "\n".join(lines)

    def _add_audit_flags(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        out["profit"] = pd.to_numeric(out.get("profit"), errors="coerce")
        out["return"] = pd.to_numeric(out.get("return"), errors="coerce")
        out["pm_multiplier_numeric"] = pd.to_numeric(out.get("pm_multiplier"), errors="coerce")
        out["volume_ratio_numeric"] = pd.to_numeric(out.get("volume_ratio"), errors="coerce")
        out["holding_days_numeric"] = pd.to_numeric(out.get("holding_days"), errors="coerce")
        out["buy_amount_numeric"] = pd.to_numeric(out.get("buy_amount"), errors="coerce")
        out["volume_ratio_gte_2"] = out["volume_ratio_numeric"].ge(2.0)
        out["pm_130"] = out["pm_multiplier_numeric"].round(2).eq(1.30)
        out["pm_gte_115"] = out["pm_multiplier_numeric"].ge(1.15)
        out["holding_gte_3"] = out["holding_days_numeric"].ge(3)
        return out

    def _condition_audit(self, bear: pd.DataFrame) -> list[dict[str, Any]]:
        conditions: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
            ("A_Bear", lambda frame: pd.Series(True, index=frame.index)),
            ("B_Bear_volume_ratio_gte_2", lambda frame: frame["volume_ratio_gte_2"]),
            ("C_Bear_PM_1_30", lambda frame: frame["pm_130"]),
            ("D_Bear_PM_1_30_volume_ratio_gte_2", lambda frame: frame["pm_130"] & frame["volume_ratio_gte_2"]),
            ("E_Bear_PM_gte_1_15", lambda frame: frame["pm_gte_115"]),
            ("F_Bear_holding_days_gte_3", lambda frame: frame["holding_gte_3"]),
            ("G_Bear_PM_1_30_volume_ratio_gte_2_holding_days_gte_3", lambda frame: frame["pm_130"] & frame["volume_ratio_gte_2"] & frame["holding_gte_3"]),
        ]
        rows = []
        for name, masker in conditions:
            subset = bear[masker(bear)].copy() if not bear.empty else pd.DataFrame()
            rows.append(self._stats_row(name, subset))
        return rows

    def _stats_row(self, label: str, frame: pd.DataFrame) -> dict[str, Any]:
        profits = pd.to_numeric(frame.get("profit"), errors="coerce") if not frame.empty else pd.Series(dtype=float)
        return {
            "condition": label,
            "trade_count": int(len(frame)),
            "net_profit": float(profits.sum()) if not profits.empty else 0.0,
            "profit_factor": _profit_factor(profits),
            "win_rate": _win_rate(profits),
            "avg_profit_per_trade": float(profits.mean()) if not profits.empty else None,
            "avg_return": _mean(frame.get("return")) if not frame.empty else None,
            "avg_buy_amount": _mean(frame.get("buy_amount_numeric")) if not frame.empty else None,
        }

    def _booster_audit(self, bear: pd.DataFrame) -> list[dict[str, Any]]:
        if bear.empty:
            return []
        return [
            self._scaled_rule_row(bear, "Rule_A_Bear_PM_1_30_buy_amount_plus_25pct", bear["pm_130"], 0.25),
            self._scaled_rule_row(bear, "Rule_B_Bear_PM_1_30_volume_ratio_gte_2_buy_amount_plus_25pct", bear["pm_130"] & bear["volume_ratio_gte_2"], 0.25),
            self._scaled_rule_row(bear, "Rule_C_Bear_PM_gte_1_15_buy_amount_plus_50pct", bear["pm_gte_115"], 0.50),
            self._hold_only_rule_row(bear, "Rule_D_keep_only_Bear_PM_1_30", bear["pm_130"]),
            self._hold_only_rule_row(bear, "Rule_E_keep_only_Bear_volume_ratio_gte_2", bear["volume_ratio_gte_2"]),
        ]

    def _scaled_rule_row(self, bear: pd.DataFrame, rule: str, mask: pd.Series, boost: float) -> dict[str, Any]:
        actual = pd.to_numeric(bear["profit"], errors="coerce").fillna(0.0)
        virtual = actual.copy()
        virtual.loc[mask] = virtual.loc[mask] * (1.0 + boost)
        return self._virtual_row(
            rule=rule,
            matched_trade_count=int(mask.sum()),
            actual=actual,
            virtual=virtual,
            note=f"Linear profit approximation: matched Bear trades scaled by +{boost:.0%}.",
        )

    def _hold_only_rule_row(self, bear: pd.DataFrame, rule: str, mask: pd.Series) -> dict[str, Any]:
        actual = pd.to_numeric(bear["profit"], errors="coerce").fillna(0.0)
        virtual = actual.where(mask, 0.0)
        return self._virtual_row(
            rule=rule,
            matched_trade_count=int(mask.sum()),
            actual=actual,
            virtual=virtual,
            note="Bear-only lightweight approximation: unmatched Bear trades are removed.",
        )

    def _virtual_row(self, *, rule: str, matched_trade_count: int, actual: pd.Series, virtual: pd.Series, note: str) -> dict[str, Any]:
        actual_profit = float(actual.sum())
        virtual_profit = float(virtual.sum())
        delta = virtual_profit - actual_profit
        return {
            "rule": rule,
            "matched_trade_count": matched_trade_count,
            "actual_bear_profit": actual_profit,
            "virtual_profit": virtual_profit,
            "profit_delta": delta,
            "profit_delta_rate": (delta / abs(actual_profit)) if actual_profit else None,
            "virtual_profit_factor": _profit_factor(virtual),
            "virtual_win_rate": _win_rate(virtual),
            "note": note,
        }

    def _pm080_analysis(self, bear: pd.DataFrame) -> dict[str, Any]:
        pm080 = bear[bear["pm_multiplier_numeric"].round(2).eq(0.80)].copy() if not bear.empty else pd.DataFrame()
        profits = pd.to_numeric(pm080.get("profit"), errors="coerce") if not pm080.empty else pd.Series(dtype=float)
        summary = {
            "trade_count": int(len(pm080)),
            "net_profit": float(profits.sum()) if not profits.empty else 0.0,
            "profit_factor": _profit_factor(profits),
            "win_rate": _win_rate(profits),
            "avg_return": _mean(pm080.get("return")) if not pm080.empty else None,
            "avg_volume_ratio": _mean(pm080.get("volume_ratio_numeric")) if not pm080.empty else None,
            "avg_holding_days": _mean(pm080.get("holding_days_numeric")) if not pm080.empty else None,
            "judgement": self._pm080_judgement(pm080),
        }
        detail_columns = ["code", "sector", "buy_date", "sell_date", "profit", "return", "volume_ratio", "holding_days", "pm_score", "buy_amount"]
        details = pm080.sort_values("profit", ascending=False)[detail_columns].to_dict("records") if not pm080.empty else []
        return {
            "summary": summary,
            "by_sector": self._bucket_stats_extended(pm080, "sector"),
            "by_volume_liquidity": self._bucket_stats_extended(pm080, "liquidity"),
            "by_holding_days_bucket": self._bucket_stats_extended(pm080, "holding_days_bucket"),
            "details": details,
        }

    def _pm080_judgement(self, pm080: pd.DataFrame) -> str:
        if pm080.empty:
            return "unknown"
        profit = float(pd.to_numeric(pm080.get("profit"), errors="coerce").sum())
        win_rate = _win_rate(pd.to_numeric(pm080.get("profit"), errors="coerce")) or 0.0
        high_liquidity_rate = float(pm080["volume_ratio_gte_2"].mean()) if "volume_ratio_gte_2" in pm080.columns else 0.0
        if profit > 0 and win_rate >= 0.6 and high_liquidity_rate >= 0.6:
            return "stock_selection_ai_or_rebound_pattern_likely; pm_ai_may_be_underestimating"
        if profit > 0:
            return "profitable_but_cause_unclear"
        return "no_positive_pm080_alpha_detected"

    def _bucket_stats_extended(self, frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        if frame.empty or column not in frame.columns:
            return []
        rows = []
        for bucket, group in frame.groupby(column, dropna=False):
            profits = pd.to_numeric(group.get("profit"), errors="coerce")
            rows.append(
                {
                    "bucket": str(bucket),
                    "trade_count": int(len(group)),
                    "net_profit": float(profits.sum()) if not profits.empty else 0.0,
                    "profit_factor": _profit_factor(profits),
                    "win_rate": _win_rate(profits),
                    "avg_return": _mean(group.get("return")),
                }
            )
        return sorted(rows, key=lambda row: (row["net_profit"], row["trade_count"]), reverse=True)

    def _common_points(self, bear: pd.DataFrame, condition_rows: list[dict[str, Any]], pm080: dict[str, Any]) -> dict[str, Any]:
        winners = bear[pd.to_numeric(bear.get("profit"), errors="coerce").gt(0)].copy() if not bear.empty else pd.DataFrame()
        return {
            "common_sector": self._top_bucket(winners, "sector"),
            "common_pm_range": self._top_bucket(winners, "pm_multiplier"),
            "common_holding_days": self._top_bucket(winners, "holding_days_bucket"),
            "common_liquidity": self._top_bucket(winners, "liquidity"),
            "common_return_pattern": self._return_pattern(winners),
            "strongest_condition": self._best_condition(condition_rows),
            "pm080_judgement": pm080.get("summary", {}).get("judgement"),
        }

    def _return_pattern(self, frame: pd.DataFrame) -> str:
        if frame.empty:
            return "unknown"
        avg_return = _mean(frame.get("return"))
        avg_holding = _mean(frame.get("holding_days_numeric"))
        if avg_return is not None and avg_return > 0.03 and avg_holding is not None and avg_holding <= 5:
            return "short_holding_rebound_or_breakout"
        if avg_return is not None and avg_return > 0:
            return "positive_short_term_return"
        return "unknown"

    def _best_condition(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        candidates = [row for row in rows if row.get("trade_count", 0) >= 3]
        if not candidates:
            candidates = rows
        best = max(candidates, key=lambda row: (row.get("avg_profit_per_trade") or 0.0, row.get("net_profit") or 0.0))
        return str(best.get("condition", ""))

    def _phase6c_verdict(
        self,
        bear: pd.DataFrame,
        bull: pd.DataFrame,
        condition_rows: list[dict[str, Any]],
        booster_rows: list[dict[str, Any]],
        pm080: dict[str, Any],
    ) -> dict[str, Any]:
        bear_pf = _profit_factor(pd.to_numeric(bear.get("profit"), errors="coerce")) or 0.0
        bull_pf = _profit_factor(pd.to_numeric(bull.get("profit"), errors="coerce")) or 0.0
        bear_win = _win_rate(pd.to_numeric(bear.get("profit"), errors="coerce")) or 0.0
        bear_alpha_exists = len(bear) >= 20 and bear_pf > max(2.0, bull_pf * 1.2) and bear_win >= 0.6
        high_pm_profit = 0.0
        bear_profit = float(pd.to_numeric(bear.get("profit"), errors="coerce").sum()) if not bear.empty else 0.0
        if not bear.empty:
            high_pm_profit = float(pd.to_numeric(bear[bear["pm_gte_115"]].get("profit"), errors="coerce").sum())
        captured_by_pm = bool(bear_profit > 0 and high_pm_profit / bear_profit >= 0.35)
        positive_boosters = [row for row in booster_rows if (row.get("profit_delta") or 0.0) > 0]
        best_booster = max(positive_boosters, key=lambda row: row.get("profit_delta") or 0.0) if positive_boosters else {}
        hold_only_rows = [row for row in booster_rows if str(row.get("rule", "")).startswith(("Rule_D", "Rule_E"))]
        hold_only_improves = any((row.get("profit_delta") or 0.0) > 0 for row in hold_only_rows)
        pm080_profitable = (pm080.get("summary", {}).get("net_profit") or 0.0) > 0
        worth_testing = bool(bear_alpha_exists and positive_boosters and not hold_only_improves)
        recommended_rule = str(best_booster.get("rule", "")) if worth_testing else "No booster implementation yet"
        return {
            "bear_alpha_exists": bool(bear_alpha_exists),
            "bear_alpha_already_captured_by_pm": bool(captured_by_pm),
            "pm_080_profit_suggests_stock_selection_or_rebound_alpha": bool(pm080_profitable),
            "bear_alpha_booster_worth_testing": worth_testing,
            "booster_rule_recommended": recommended_rule,
            "expected_profit_direction": "positive_for_small_scaled_high_pm_rules" if worth_testing else "uncertain",
            "expected_dd_direction": "may_worsen_due_to_concentration" if worth_testing else "unknown",
            "next_phase_recommended": "Phase 6-D Bear Booster Design Audit" if worth_testing else "Keep v2_78; do not implement Bear booster yet",
        }


def run_phase6c_bear_alpha_booster_audit(root: Path | str = ROOT) -> Phase6CReportPaths:
    audit = Phase6CBearAlphaBoosterAudit(root, profile=BASE_PROFILE, period=PERIOD)
    return audit.save_report(audit.build_report())
