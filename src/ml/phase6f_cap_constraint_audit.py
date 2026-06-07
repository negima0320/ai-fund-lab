"""Phase 6-F cap constraint audit.

Read-only audit for per-code exposure cap bottlenecks. It estimates whether the
current 30% cap prevents profitable PM/Bear sizing without changing profiles or
running a backtest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase6b_bear_market_winner_audit import BASE_PROFILE, PERIOD, ROOT, _profit_factor, _win_rate
from ml.phase6c_bear_alpha_booster_audit import Phase6CBearAlphaBoosterAudit


REPORT_STEM = "phase6f_cap_constraint_audit_2023-01_to_2026-05"
BOOSTER_PROFILE = "rookie_dealer_02_v2_81_bear_pm115_booster_50"
CAP_RATES = [0.35, 0.40, 0.50]


@dataclass(frozen=True)
class Phase6FReportPaths:
    markdown: Path
    json: Path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _truthy(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})


class Phase6FCapConstraintAudit(Phase6CBearAlphaBoosterAudit):
    def build_report(self) -> dict[str, Any]:
        trades = self._load_trades()
        purchase = self._load_purchase_audit()
        listed = self._load_listed_info()
        regime = self._load_regime()
        enriched = self._enrich_trades(trades, purchase, listed, regime)
        purchase = self._enrich_purchase(purchase, enriched)
        cap_rows = self._cap_rate_audit(purchase)
        booster_rows = self._booster_cap_audit(purchase)
        pm_rows = self._pm_high_score_audit(purchase)
        virtual_rows = self._virtual_rule_audit(purchase)
        dd_rows = self._dd_risk_audit(virtual_rows)
        verdict = self._verdict(cap_rows, booster_rows, virtual_rows)
        return {
            "metadata": {
                "phase": "6-F",
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
                "current_per_code_cap": 0.30,
            },
            "sources": {
                "trades_csv": str(self._log_dir() / "trades.csv"),
                "purchase_audit_csv": str(self._log_dir() / "purchase_audit.csv"),
                "optional_v2_81_purchase_audit_csv": str(self.root / "logs" / "backtests" / BOOSTER_PROFILE / self.period / "purchase_audit.csv"),
                "approximation_note": "Additional profit is estimated from realized return on executed amount; no cash path, same-day replacement, or DD path is recomputed.",
            },
            "coverage": self._coverage(purchase),
            "cap_rate_audit": cap_rows,
            "bear_booster_cap_relation": booster_rows,
            "pm_high_score_cap_audit": pm_rows,
            "virtual_comparison": virtual_rows,
            "dd_risk_audit": dd_rows,
            "verdict": verdict,
        }

    def save_report(self, result: dict[str, Any]) -> Phase6FReportPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase6FReportPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 6-F Cap Constraint Audit",
            "",
            "## Scope",
            "",
            "- audit only",
            "- no logic change, no new profile, no backtest execution",
            "- existing v2_78 logs only; v2_81 logs are optional for booster blockage diagnostics",
            "",
            "## Coverage",
            "",
            self._table([result["coverage"]], ["purchase_rows", "cap_hit_count", "cap_reduction_amount", "cap_prevented_buy_amount"]),
            "",
            "## Cap Rate Audit",
            "",
            self._table(result["cap_rate_audit"], ["cap_rate", "cap_hit_count", "affected_trades", "cap_reduction_amount", "cap_prevented_buy_amount", "newly_allowed_amount", "profit_approximation"]),
            "",
            "## Bear Booster Relation",
            "",
            self._table(result["bear_booster_cap_relation"], ["cap_rate", "booster_blocked_by_cap", "booster_allowed_amount", "booster_still_blocked_amount", "booster_profit_approximation"]),
            "",
            "## PM High Score Cap Audit",
            "",
            self._table(result["pm_high_score_cap_audit"], ["pm_bucket", "trade_count", "reduction_amount", "estimated_profit"]),
            "",
            "## Virtual Comparison",
            "",
            self._table(result["virtual_comparison"], ["rule", "profit_delta", "pf_approximation", "dd_approximation", "concentration_risk"]),
            "",
            "## DD Risk",
            "",
            self._table(result["dd_risk_audit"], ["rule", "concentration_risk", "single_code_concentration", "dd_approximation"]),
            "",
            "## Verdict",
            "",
            self._table([result["verdict"]], ["cap_is_current_bottleneck", "cap_relaxation_worth_testing", "safest_cap_candidate", "expected_profit_direction", "expected_dd_direction", "ready_for_phase6g"]),
            "",
        ]
        return "\n".join(lines)

    def _log_dir_for_profile(self, profile: str) -> Path:
        return self.root / "logs" / "backtests" / profile / self.period

    def _enrich_purchase(self, purchase: pd.DataFrame, enriched_trades: pd.DataFrame) -> pd.DataFrame:
        if purchase.empty:
            return purchase
        out = purchase.copy()
        out["code"] = out["code"].map(lambda value: str(value).removesuffix(".0"))
        if "entry_date" in out.columns:
            out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if not enriched_trades.empty:
            trade_cols = ["code", "buy_date", "profit", "return", "regime"]
            joined = enriched_trades[trade_cols].rename(columns={"buy_date": "entry_date"})
            out = out.merge(joined.drop_duplicates(["code", "entry_date"], keep="last"), on=["code", "entry_date"], how="left")
        for column in [
            "pm_per_code_cap_original_amount",
            "pm_per_code_cap_amount",
            "pm_per_code_max_exposure",
            "pm_per_code_current_exposure",
            "final_amount",
            "pm_multiplier",
            "profit",
            "return",
        ]:
            if column in out.columns:
                out[column] = pd.to_numeric(out[column], errors="coerce")
        out["cap_reduction_amount"] = (out.get("pm_per_code_cap_original_amount", 0) - out.get("pm_per_code_cap_amount", 0)).clip(lower=0)
        out["cap_hit"] = _truthy(out, "pm_per_code_cap_reduced") | _truthy(out, "pm_per_code_cap_skip") | out["cap_reduction_amount"].gt(0)
        out["executed_return"] = self._executed_return(out)
        out["regime"] = out.get("regime", pd.Series("", index=out.index)).fillna("")
        return out

    def _executed_return(self, frame: pd.DataFrame) -> pd.Series:
        rate = _num(frame, "return")
        if not rate.dropna().empty:
            return rate.fillna(0.0)
        profit = _num(frame, "profit").fillna(0.0)
        amount = _num(frame, "final_amount").replace(0, pd.NA)
        return (profit / amount).fillna(0.0)

    def _coverage(self, purchase: pd.DataFrame) -> dict[str, Any]:
        if purchase.empty:
            return {"purchase_rows": 0, "cap_hit_count": 0, "cap_reduction_amount": 0.0, "cap_prevented_buy_amount": 0.0}
        reduction = _num(purchase, "cap_reduction_amount").fillna(0.0)
        return {
            "purchase_rows": int(len(purchase)),
            "cap_hit_count": int(purchase["cap_hit"].sum()),
            "cap_reduction_amount": float(reduction.sum()),
            "cap_prevented_buy_amount": float(reduction.sum()),
        }

    def _cap_rate_audit(self, purchase: pd.DataFrame) -> list[dict[str, Any]]:
        return [self._cap_rate_row(purchase, rate) for rate in CAP_RATES]

    def _cap_rate_row(self, purchase: pd.DataFrame, cap_rate: float) -> dict[str, Any]:
        affected = self._affected_for_cap(purchase, cap_rate)
        newly_allowed = _num(affected, f"new_allowed_{cap_rate}").fillna(0.0)
        profit_delta = newly_allowed * _num(affected, "executed_return").fillna(0.0)
        reduction = _num(affected, "cap_reduction_amount").fillna(0.0)
        return {
            "cap_rate": cap_rate,
            "cap_hit_count": int(purchase["cap_hit"].sum()) if not purchase.empty and "cap_hit" in purchase.columns else 0,
            "affected_trades": int(len(affected)),
            "cap_reduction_amount": float(reduction.sum()),
            "cap_prevented_buy_amount": float(reduction.sum()),
            "newly_allowed_amount": float(newly_allowed.sum()),
            "profit_approximation": float(profit_delta.sum()),
        }

    def _affected_for_cap(self, purchase: pd.DataFrame, cap_rate: float) -> pd.DataFrame:
        if purchase.empty:
            return purchase.copy()
        rows = purchase[purchase["cap_hit"]].copy()
        if rows.empty:
            return rows
        current_rate = _num(rows, "pm_per_code_cap_rate").replace(0, pd.NA).fillna(0.30)
        max_exposure = _num(rows, "pm_per_code_max_exposure").fillna(0.0)
        total_assets_proxy = max_exposure / current_rate
        current_exposure = _num(rows, "pm_per_code_current_exposure").fillna(0.0)
        original = _num(rows, "pm_per_code_cap_original_amount").fillna(0.0)
        actual = _num(rows, "pm_per_code_cap_amount").fillna(0.0)
        new_allowed_total = (total_assets_proxy * cap_rate - current_exposure).clip(lower=0)
        new_amount = pd.concat([original, new_allowed_total], axis=1).min(axis=1)
        rows[f"new_allowed_{cap_rate}"] = (new_amount - actual).clip(lower=0)
        rows = rows[rows[f"new_allowed_{cap_rate}"].gt(0)].copy()
        return rows

    def _booster_cap_audit(self, purchase: pd.DataFrame) -> list[dict[str, Any]]:
        booster = self._load_booster_purchase_rows()
        if booster.empty:
            booster = purchase[(purchase.get("regime", pd.Series("", index=purchase.index)).eq("Bear")) & _num(purchase, "pm_multiplier").ge(1.15)].copy()
            booster["booster_incremental_amount"] = _num(booster, "pm_per_code_cap_original_amount").fillna(0.0) * 0.5
        else:
            booster["booster_incremental_amount"] = (
                _num(booster, "bear_pm_booster_after_amount").fillna(0.0) - _num(booster, "bear_pm_booster_before_amount").fillna(0.0)
            ).clip(lower=0)
            booster["executed_return"] = self._executed_return(booster)
        rows = []
        for rate in CAP_RATES:
            allowed = self._booster_allowed_for_cap(booster, rate)
            blocked = float(_num(booster, "booster_incremental_amount").fillna(0.0).sum())
            allowed_amount = float(allowed.sum()) if not allowed.empty else 0.0
            profit = float((allowed * _num(booster, "executed_return").fillna(0.0)).sum()) if not allowed.empty else 0.0
            rows.append(
                {
                    "cap_rate": rate,
                    "booster_blocked_by_cap": blocked,
                    "booster_allowed_amount": allowed_amount,
                    "booster_still_blocked_amount": max(0.0, blocked - allowed_amount),
                    "booster_profit_approximation": profit,
                }
            )
        return rows

    def _load_booster_purchase_rows(self) -> pd.DataFrame:
        path = self._log_dir_for_profile(BOOSTER_PROFILE) / "purchase_audit.csv"
        frame = _read_csv(path)
        if frame.empty:
            return frame
        frame["code"] = frame["code"].map(lambda value: str(value).removesuffix(".0"))
        if "entry_date" in frame.columns:
            frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame[_truthy(frame, "bear_pm_booster_applied")].copy()
        trades = Phase6CBearAlphaBoosterAudit(self.root, profile=BOOSTER_PROFILE, period=self.period)._load_trades()
        if not trades.empty:
            trades["buy_date"] = trades.get("entry_date")
            trades["profit"] = pd.to_numeric(trades.get("net_profit"), errors="coerce")
            trades["return"] = pd.to_numeric(trades.get("net_profit_rate"), errors="coerce")
            frame = frame.merge(trades[["code", "buy_date", "profit", "return"]].rename(columns={"buy_date": "entry_date"}), on=["code", "entry_date"], how="left")
        return frame

    def _booster_allowed_for_cap(self, booster: pd.DataFrame, cap_rate: float) -> pd.Series:
        if booster.empty:
            return pd.Series(dtype=float)
        current_rate = _num(booster, "pm_per_code_cap_rate").replace(0, pd.NA).fillna(0.30)
        total_assets_proxy = _num(booster, "pm_per_code_max_exposure").fillna(0.0) / current_rate
        current_exposure = _num(booster, "pm_per_code_current_exposure").fillna(0.0)
        actual = _num(booster, "pm_per_code_cap_amount").fillna(_num(booster, "final_amount").fillna(0.0))
        original = _num(booster, "pm_per_code_cap_original_amount").fillna(actual)
        cap_room_new = (total_assets_proxy * cap_rate - current_exposure - actual).clip(lower=0)
        requested_extra = _num(booster, "booster_incremental_amount").fillna((original - actual).clip(lower=0))
        return pd.concat([cap_room_new, requested_extra], axis=1).min(axis=1).clip(lower=0)

    def _pm_high_score_audit(self, purchase: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for label, threshold in [("PM>=1.15", 1.15), ("PM>=1.30", 1.30)]:
            subset = purchase[purchase["cap_hit"] & _num(purchase, "pm_multiplier").ge(threshold)].copy() if not purchase.empty else pd.DataFrame()
            reduction = _num(subset, "cap_reduction_amount").fillna(0.0)
            rows.append(
                {
                    "pm_bucket": label,
                    "trade_count": int(len(subset)),
                    "reduction_amount": float(reduction.sum()),
                    "estimated_profit": float((reduction * _num(subset, "executed_return").fillna(0.0)).sum()) if not subset.empty else 0.0,
                }
            )
        return rows

    def _virtual_rule_audit(self, purchase: pd.DataFrame) -> list[dict[str, Any]]:
        rows = [self._virtual_cap_rule(purchase, f"Rule_{name}_cap_{int(rate*100)}pct", rate) for name, rate in [("A", 0.35), ("B", 0.40), ("C", 0.50)]]
        booster = {row["cap_rate"]: row for row in self._booster_cap_audit(purchase)}
        for name, rate in [("D", 0.35), ("E", 0.40), ("F", 0.50)]:
            base = self._virtual_cap_rule(purchase, f"Rule_{name}_bear_pm115_booster_cap_{int(rate*100)}pct", rate)
            base["profit_delta"] += float(booster.get(rate, {}).get("booster_profit_approximation") or 0.0)
            base["pf_approximation"] = self._pf_with_delta(purchase, base["profit_delta"])
            base["concentration_risk"] = self._risk_label_for_value(self._concentration_from_amount(purchase, rate) + 10)
            rows.append(base)
        return rows

    def _virtual_cap_rule(self, purchase: pd.DataFrame, rule: str, cap_rate: float) -> dict[str, Any]:
        affected = self._affected_for_cap(purchase, cap_rate)
        allowed = _num(affected, f"new_allowed_{cap_rate}").fillna(0.0)
        profit_delta = float((allowed * _num(affected, "executed_return").fillna(0.0)).sum()) if not affected.empty else 0.0
        return {
            "rule": rule,
            "cap_rate": cap_rate,
            "profit_delta": profit_delta,
            "pf_approximation": self._pf_with_delta(purchase, profit_delta),
            "dd_approximation": self._risk_label_for_value(self._concentration_from_amount(affected, cap_rate)),
            "concentration_risk": self._risk_label_for_value(self._concentration_from_amount(affected, cap_rate)),
        }

    def _pf_with_delta(self, purchase: pd.DataFrame, delta: float) -> float | None:
        profits = _num(purchase, "profit").dropna()
        if profits.empty:
            return None
        adjusted = pd.concat([profits, pd.Series([delta])], ignore_index=True)
        return _profit_factor(adjusted)

    def _concentration_from_amount(self, frame: pd.DataFrame, cap_rate: float) -> float:
        if frame.empty:
            return 0.0
        amount_col = f"new_allowed_{cap_rate}"
        amount = _num(frame, amount_col).fillna(0.0) if amount_col in frame.columns else _num(frame, "cap_reduction_amount").fillna(0.0)
        total = float(amount.sum())
        if total <= 0 or "code" not in frame.columns:
            return 0.0
        by_code = frame.assign(_amount=amount).groupby("code")["_amount"].sum()
        return float(by_code.max() / total) * 100.0 if not by_code.empty else 0.0

    def _risk_label_for_value(self, value: float) -> str:
        if value < 20:
            return "low"
        if value < 40:
            return "medium"
        return "high"

    def _dd_risk_audit(self, virtual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "rule": row["rule"],
                "concentration_risk": row["concentration_risk"],
                "single_code_concentration": row.get("dd_approximation"),
                "dd_approximation": row["dd_approximation"],
            }
            for row in virtual_rows
        ]

    def _verdict(self, cap_rows: list[dict[str, Any]], booster_rows: list[dict[str, Any]], virtual_rows: list[dict[str, Any]]) -> dict[str, Any]:
        total_allowed = max((row.get("newly_allowed_amount") or 0.0 for row in cap_rows), default=0.0)
        best = max(virtual_rows, key=lambda row: row.get("profit_delta") or 0.0) if virtual_rows else {}
        safe_candidates = [row for row in virtual_rows if row.get("profit_delta", 0.0) > 0 and row.get("concentration_risk") in {"low", "medium"}]
        safest = safe_candidates[0]["rule"] if safe_candidates else "No cap relaxation yet"
        return {
            "cap_is_current_bottleneck": total_allowed > 0,
            "cap_relaxation_worth_testing": bool(best and (best.get("profit_delta") or 0.0) > 0),
            "safest_cap_candidate": safest,
            "best_profit_candidate": best.get("rule", ""),
            "expected_profit_direction": "positive" if best and (best.get("profit_delta") or 0.0) > 0 else "uncertain",
            "expected_dd_direction": "worse_or_uncertain",
            "ready_for_phase6g": bool(safe_candidates),
        }

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows_"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(self._format_cell(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)


def run_phase6f_cap_constraint_audit(root: Path | str = ROOT) -> Phase6FReportPaths:
    audit = Phase6FCapConstraintAudit(root, profile=BASE_PROFILE, period=PERIOD)
    return audit.save_report(audit.build_report())
