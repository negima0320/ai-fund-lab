"""Phase 6-D Bear booster design audit.

Read-only design audit for Bear-regime position-size booster candidates. The
audit estimates profit, concentration, drawdown, and utilization impact from
existing v2_78 trades only. It does not implement a Bear mode or execute a
backtest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase6b_bear_market_winner_audit import BASE_PROFILE, PERIOD, ROOT, _mean, _profit_factor, _win_rate
from ml.phase6c_bear_alpha_booster_audit import Phase6CBearAlphaBoosterAudit


REPORT_STEM = "phase6d_bear_booster_design_audit_2023-01_to_2026-05"


@dataclass(frozen=True)
class Phase6DReportPaths:
    markdown: Path
    json: Path


class Phase6DBearBoosterDesignAudit(Phase6CBearAlphaBoosterAudit):
    def build_report(self) -> dict[str, Any]:
        trades = self._load_trades()
        purchase = self._load_purchase_audit()
        listed = self._load_listed_info()
        regime = self._load_regime()
        enriched = self._enrich_trades(trades, purchase, listed, regime)
        bear = enriched[enriched["regime"].eq("Bear")].copy() if not enriched.empty else pd.DataFrame()
        bull = enriched[enriched["regime"].eq("Bull")].copy() if not enriched.empty else pd.DataFrame()
        bear = self._add_audit_flags(bear)

        rules = self._design_rule_audit(bear)
        pm080 = self._pm080_protection_audit(bear)
        dd = self._dd_risk_audit(bear, rules)
        verdict = self._phase6d_verdict(bear, bull, rules, pm080)

        return {
            "metadata": {
                "phase": "6-D",
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
                "approximation_note": "Profit impact is linearly scaled from realized trade profit; no backtest, cash path, or DD path is recomputed.",
            },
            "coverage": {
                "total_trades": int(len(enriched)),
                "bear_trades": int(len(bear)),
                "bull_trades": int(len(bull)),
                "bear_profit": self._sum_profit(bear),
                "bear_profit_factor": _profit_factor(pd.to_numeric(bear.get("profit"), errors="coerce")) if not bear.empty else None,
                "bear_win_rate": _win_rate(pd.to_numeric(bear.get("profit"), errors="coerce")) if not bear.empty else None,
            },
            "booster_rule_design_audit": rules,
            "pm_080_protection_audit": pm080,
            "dd_risk_audit": dd,
            "verdict": verdict,
        }

    def save_report(self, result: dict[str, Any]) -> Phase6DReportPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase6DReportPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 6-D Bear Booster Design Audit",
            "",
            "## Scope",
            "",
            "- audit only",
            "- no logic change, no new profile, no backtest execution",
            "- existing v2_78 logs and local cache only",
            "",
            "## Coverage",
            "",
            self._table([result["coverage"]], ["total_trades", "bear_trades", "bull_trades", "bear_profit", "bear_profit_factor", "bear_win_rate"]),
            "",
            "## Booster Rule Design Audit",
            "",
            self._table(
                result["booster_rule_design_audit"],
                [
                    "rule",
                    "matched_trades",
                    "profit_delta",
                    "profit_delta_pct",
                    "pf_approximation",
                    "dd_risk_score",
                    "concentration_risk_score",
                    "capital_utilization_impact",
                    "risk_label",
                ],
            ),
            "",
            "## PM 0.80 Protection Audit",
            "",
            self._table([result["pm_080_protection_audit"]["summary"]], ["trade_count", "profit", "profit_contribution", "profit_if_removed", "profit_delta_if_removed", "profit_factor", "win_rate", "recommendation"]),
            "",
            self._table(result["pm_080_protection_audit"]["by_sector"], ["bucket", "trade_count", "net_profit", "profit_factor", "win_rate", "avg_return"]),
            "",
            self._table(result["pm_080_protection_audit"]["by_holding_days_bucket"], ["bucket", "trade_count", "net_profit", "profit_factor", "win_rate", "avg_return"]),
            "",
            "## DD Risk Audit",
            "",
            self._table(result["dd_risk_audit"], ["rule", "matched_trades", "max_single_code_incremental_profit", "single_code_risk_score", "pm_concentration_risk_score", "negative_trade_scaled_loss_delta", "dd_risk_score"]),
            "",
            "## Verdict",
            "",
            self._table([result["verdict"]], ["safe_booster_exists", "booster_rule_recommended", "expected_profit_direction", "expected_dd_direction", "expected_utilization_direction", "ready_for_phase6e", "keep_v278_as_primary"]),
            "",
        ]
        return "\n".join(lines)

    def _design_rule_audit(self, bear: pd.DataFrame) -> list[dict[str, Any]]:
        if bear.empty:
            return []
        rules: list[tuple[str, pd.Series, float]] = [
            ("Rule_A_Bear_PM_gte_1_15_buy_amount_plus_25pct", bear["pm_gte_115"], 0.25),
            ("Rule_B_Bear_PM_gte_1_15_buy_amount_plus_50pct", bear["pm_gte_115"], 0.50),
            ("Rule_C_Bear_PM_gte_1_15_buy_amount_plus_75pct", bear["pm_gte_115"], 0.75),
            ("Rule_D_Bear_PM_gte_1_30_buy_amount_plus_25pct", bear["pm_multiplier_numeric"].ge(1.30), 0.25),
            ("Rule_E_Bear_PM_gte_1_30_buy_amount_plus_50pct", bear["pm_multiplier_numeric"].ge(1.30), 0.50),
            ("Rule_F_Bear_PM_gte_1_15_holding_days_gte_3_buy_amount_plus_50pct", bear["pm_gte_115"] & bear["holding_gte_3"], 0.50),
            ("Rule_G_Bear_PM_gte_1_15_volume_ratio_gte_2_buy_amount_plus_50pct", bear["pm_gte_115"] & bear["volume_ratio_gte_2"], 0.50),
        ]
        return [self._booster_rule_row(bear, rule, mask, boost) for rule, mask, boost in rules]

    def _booster_rule_row(self, bear: pd.DataFrame, rule: str, mask: pd.Series, boost: float) -> dict[str, Any]:
        actual = pd.to_numeric(bear["profit"], errors="coerce").fillna(0.0)
        virtual = actual.copy()
        virtual.loc[mask] = virtual.loc[mask] * (1.0 + boost)
        delta = float(virtual.sum() - actual.sum())
        base_profit = float(actual.sum())
        utilization_impact = self._capital_utilization_impact(bear, mask, boost)
        concentration = self._concentration_score(bear, mask, boost)
        dd_score = self._dd_risk_score(bear, mask, boost, concentration)
        return {
            "rule": rule,
            "matched_trades": int(mask.sum()),
            "boost_rate": boost,
            "actual_bear_profit": base_profit,
            "virtual_profit": float(virtual.sum()),
            "profit_delta": delta,
            "profit_delta_pct": delta / abs(base_profit) if base_profit else None,
            "pf_approximation": _profit_factor(virtual),
            "win_rate_approximation": _win_rate(virtual),
            "dd_risk_score": dd_score,
            "concentration_risk_score": concentration,
            "capital_utilization_impact": utilization_impact,
            "risk_label": self._risk_label(dd_score, concentration, utilization_impact),
        }

    def _capital_utilization_impact(self, bear: pd.DataFrame, mask: pd.Series, boost: float) -> float | None:
        amounts = pd.to_numeric(bear.get("buy_amount_numeric"), errors="coerce").fillna(0.0)
        total_amount = float(amounts.sum())
        if total_amount <= 0:
            return None
        incremental = float((amounts[mask] * boost).sum())
        return incremental / total_amount

    def _concentration_score(self, bear: pd.DataFrame, mask: pd.Series, boost: float) -> float:
        selected = bear[mask].copy()
        if selected.empty:
            return 0.0
        amounts = pd.to_numeric(selected.get("buy_amount_numeric"), errors="coerce").fillna(0.0) * boost
        total_incremental = float(amounts.sum())
        if total_incremental <= 0:
            return 0.0
        by_code = selected.assign(_incremental_amount=amounts).groupby("code")["_incremental_amount"].sum()
        max_code_share = float(by_code.max() / total_incremental) if not by_code.empty else 0.0
        matched_share = float(len(selected) / max(len(bear), 1))
        score = 100.0 * (0.70 * max_code_share + 0.30 * matched_share)
        return min(100.0, score)

    def _dd_risk_score(self, bear: pd.DataFrame, mask: pd.Series, boost: float, concentration_score: float) -> float:
        selected = bear[mask].copy()
        if selected.empty:
            return 0.0
        profits = pd.to_numeric(selected.get("profit"), errors="coerce").fillna(0.0)
        loss_delta = abs(float((profits[profits < 0] * boost).sum()))
        selected_profit_abs = abs(float(profits.sum()))
        loss_ratio = loss_delta / selected_profit_abs if selected_profit_abs else 0.0
        boost_score = min(100.0, boost * 100.0)
        score = 0.45 * boost_score + 0.35 * concentration_score + 0.20 * min(100.0, loss_ratio * 100.0)
        return min(100.0, score)

    def _risk_label(self, dd_score: float, concentration_score: float, utilization_impact: float | None) -> str:
        util = utilization_impact or 0.0
        if dd_score < 35 and concentration_score < 35 and util < 0.15:
            return "low"
        if dd_score < 55 and concentration_score < 55 and util < 0.30:
            return "medium"
        return "high"

    def _pm080_protection_audit(self, bear: pd.DataFrame) -> dict[str, Any]:
        if bear.empty:
            empty = {"trade_count": 0, "profit": 0.0, "profit_contribution": None, "profit_if_removed": 0.0, "profit_delta_if_removed": 0.0, "profit_factor": None, "win_rate": None, "recommendation": "unknown"}
            return {"summary": empty, "by_sector": [], "by_holding_days_bucket": [], "details": []}
        pm080 = bear[bear["pm_multiplier_numeric"].round(2).eq(0.80)].copy()
        bear_profit = self._sum_profit(bear)
        pm_profit = self._sum_profit(pm080)
        profit_if_removed = bear_profit - pm_profit
        profits = pd.to_numeric(pm080.get("profit"), errors="coerce") if not pm080.empty else pd.Series(dtype=float)
        summary = {
            "trade_count": int(len(pm080)),
            "profit": pm_profit,
            "profit_contribution": pm_profit / bear_profit if bear_profit else None,
            "profit_if_removed": profit_if_removed,
            "profit_delta_if_removed": -pm_profit,
            "profit_factor": _profit_factor(profits),
            "win_rate": _win_rate(profits),
            "avg_return": _mean(pm080.get("return")) if not pm080.empty else None,
            "avg_holding_days": _mean(pm080.get("holding_days_numeric")) if not pm080.empty else None,
            "recommendation": "keep_pm_080; do_not_filter_low_pm_in_bear" if pm_profit > 0 else "no_pm080_protection_needed",
        }
        detail_columns = ["code", "sector", "buy_date", "sell_date", "profit", "return", "volume_ratio", "holding_days", "pm_score", "buy_amount"]
        return {
            "summary": summary,
            "by_sector": self._bucket_stats_extended(pm080, "sector"),
            "by_holding_days_bucket": self._bucket_stats_extended(pm080, "holding_days_bucket"),
            "details": pm080.sort_values("profit", ascending=False)[detail_columns].to_dict("records") if not pm080.empty else [],
        }

    def _dd_risk_audit(self, bear: pd.DataFrame, rule_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if bear.empty:
            return []
        rows = []
        for rule in rule_rows:
            name = str(rule["rule"])
            boost = float(rule["boost_rate"])
            mask = self._mask_for_rule_name(bear, name)
            selected = bear[mask].copy()
            profits = pd.to_numeric(selected.get("profit"), errors="coerce").fillna(0.0)
            incremental_profit = profits * boost
            by_code = selected.assign(_incremental_profit=incremental_profit).groupby("code")["_incremental_profit"].sum() if not selected.empty else pd.Series(dtype=float)
            negative_delta = float((incremental_profit[incremental_profit < 0]).sum()) if not incremental_profit.empty else 0.0
            rows.append(
                {
                    "rule": name,
                    "matched_trades": int(len(selected)),
                    "max_single_code_incremental_profit": float(by_code.abs().max()) if not by_code.empty else 0.0,
                    "single_code_risk_score": self._single_code_score(by_code),
                    "pm_concentration_risk_score": self._pm_concentration_score(selected),
                    "negative_trade_scaled_loss_delta": negative_delta,
                    "dd_risk_score": rule["dd_risk_score"],
                }
            )
        return rows

    def _mask_for_rule_name(self, bear: pd.DataFrame, name: str) -> pd.Series:
        if "PM_gte_1_30" in name:
            mask = bear["pm_multiplier_numeric"].ge(1.30)
        else:
            mask = bear["pm_gte_115"]
        if "holding_days_gte_3" in name:
            mask = mask & bear["holding_gte_3"]
        if "volume_ratio_gte_2" in name:
            mask = mask & bear["volume_ratio_gte_2"]
        return mask

    def _single_code_score(self, by_code: pd.Series) -> float:
        if by_code.empty:
            return 0.0
        total = float(by_code.abs().sum())
        if total <= 0:
            return 0.0
        return min(100.0, float(by_code.abs().max() / total) * 100.0)

    def _pm_concentration_score(self, selected: pd.DataFrame) -> float:
        if selected.empty:
            return 0.0
        counts = selected["pm_multiplier_numeric"].round(2).value_counts(normalize=True)
        if counts.empty:
            return 0.0
        return min(100.0, float(counts.max()) * 100.0)

    def _phase6d_verdict(self, bear: pd.DataFrame, bull: pd.DataFrame, rules: list[dict[str, Any]], pm080: dict[str, Any]) -> dict[str, Any]:
        safe_candidates = [
            row
            for row in rules
            if (row.get("profit_delta") or 0.0) > 0
            and (row.get("dd_risk_score") or 100.0) < 45
            and (row.get("concentration_risk_score") or 100.0) < 45
            and (row.get("capital_utilization_impact") or 1.0) < 0.25
        ]
        if safe_candidates:
            recommended = max(safe_candidates, key=lambda row: row.get("profit_delta") or 0.0)
        else:
            recommended = {}
        pm080_profit = pm080.get("summary", {}).get("profit") or 0.0
        return {
            "safe_booster_exists": bool(safe_candidates),
            "booster_rule_recommended": recommended.get("rule", "No implementation yet"),
            "expected_profit_direction": "positive" if recommended else "uncertain",
            "expected_dd_direction": "slightly_worse_or_uncertain" if recommended else "unknown",
            "expected_utilization_direction": "slightly_higher" if recommended else "unchanged",
            "pm_080_should_be_kept": bool(pm080_profit > 0),
            "ready_for_phase6e": bool(recommended),
            "keep_v278_as_primary": True,
        }

    def _sum_profit(self, frame: pd.DataFrame) -> float:
        if frame.empty:
            return 0.0
        return float(pd.to_numeric(frame.get("profit"), errors="coerce").fillna(0.0).sum())


def run_phase6d_bear_booster_design_audit(root: Path | str = ROOT) -> Phase6DReportPaths:
    audit = Phase6DBearBoosterDesignAudit(root, profile=BASE_PROFILE, period=PERIOD)
    return audit.save_report(audit.build_report())
