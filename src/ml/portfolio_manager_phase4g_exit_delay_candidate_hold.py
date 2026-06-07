"""Phase 4-G clean exit-delay / candidate-presence hold audit.

The audit is read-only. It estimates whether a clean rule can reproduce the
beneficial v2_79 path divergence without adopting v2_79's side effect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PERIOD = "2023-01-01_to_2026-05-31"
REPORT_STEM = "portfolio_manager_phase4g_exit_delay_candidate_hold_audit_2023-01_to_2026-05"
PRIMARY = "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
REFERENCE = "rookie_dealer_02_v2_79_high_pm_min_hold_5d"
TOP_DELTA_CODES = ["96970", "34960", "70120", "56310", "58050", "58380", "31970", "91010"]


@dataclass(frozen=True)
class Phase4GPaths:
    markdown: Path
    json: Path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _to_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _to_int(value: Any) -> int | None:
    value = _to_float(value)
    return None if value is None else int(value)


def _to_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _profit_factor(values: list[float]) -> float | None:
    profits = [float(value) for value in values if value is not None]
    if not profits:
        return None
    gross_profit = sum(value for value in profits if value > 0)
    gross_loss = abs(sum(value for value in profits if value < 0))
    return None if gross_loss == 0 else gross_profit / gross_loss


def _win_rate(values: list[float]) -> float | None:
    profits = [float(value) for value in values if value is not None]
    return None if not profits else sum(1 for value in profits if value > 0) / len(profits)


def _net_profit_column(row: dict[str, Any]) -> float:
    return _to_float(row.get("net_profit")) or _to_float(row.get("profit")) or 0.0


def _exit_ai_rows(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "exit_reason" not in trades.columns:
        return pd.DataFrame()
    reason = trades["exit_reason"].fillna("").astype(str).str.lower()
    exit_ai = trades.get("exit_ai_triggered", pd.Series(False, index=trades.index)).map(_truthy)
    hard = reason.str.contains("stop|損切|take_profit|利確|max_holding|最大保有|forced|強制|hard", regex=True)
    rows = trades[(exit_ai | reason.str.contains("exit ai|avoid_loss", regex=True)) & ~hard].copy()
    if rows.empty:
        return rows
    rows["code"] = rows.get("code", "").fillna("").astype(str)
    rows["sell_date"] = rows.get("exit_date", "").fillna("").astype(str)
    rows["buy_date"] = rows.get("entry_date", "").fillna("").astype(str)
    return rows.sort_values(["sell_date", "code", "buy_date"], kind="stable")


class PriceCache:
    def __init__(self, root: Path) -> None:
        self.raw_dir = root / "data" / "raw"
        self._dates: list[str] | None = None
        self._cache: dict[str, dict[str, dict[str, Any]]] = {}

    def trading_dates(self) -> list[str]:
        if self._dates is None:
            self._dates = sorted(path.stem.replace("prices_", "") for path in self.raw_dir.glob("prices_*.json"))
        return self._dates

    def next_trading_date(self, date: str) -> str | None:
        for candidate in self.trading_dates():
            if candidate > date:
                return candidate
        return None

    def price(self, date: str, code: str) -> dict[str, Any] | None:
        if date not in self._cache:
            payload = _read_json(self.raw_dir / f"prices_{date}.json")
            rows = payload.get("prices") if isinstance(payload.get("prices"), list) else []
            self._cache[date] = {str(row.get("code")): row for row in rows if isinstance(row, dict)}
        return self._cache[date].get(str(code))

    def close(self, date: str, code: str) -> float | None:
        row = self.price(date, code)
        return _to_float(row.get("close")) if row else None


class PortfolioManagerPhase4GExitDelayCandidateHoldAudit:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        primary: str = PRIMARY,
        reference: str = REFERENCE,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.primary = primary
        self.reference = reference
        self.period = period
        self.prices = PriceCache(self.root)

    def build_report(self) -> dict[str, Any]:
        primary = self._load_profile_logs(self.primary)
        reference = self._load_profile_logs(self.reference)
        exit_rows = _exit_ai_rows(primary["trades"])
        delay = self._exit_delay_audit(exit_rows)
        presence = self._candidate_presence_audit(exit_rows, primary["purchase_audit"])
        rules = self._candidate_presence_rule_audit(delay["rows"], presence["rows"])
        check_71570 = self._check_71570(delay["rows"], presence["rows"], rules)
        reproducibility = self._v279_reproducibility(delay["rows"], rules, primary["trades"], reference["trades"])
        judgement = self._judgement(delay, rules, reproducibility, presence["log_reliability"])
        return {
            "metadata": {
                "phase": "4-G",
                "audit_only": True,
                "full_backtest_executed": False,
                "full_pytest_executed": False,
                "uses_existing_logs_only": True,
                "primary": self.primary,
                "comparison_reference": self.reference,
                "period": self.period,
            },
            "input_files": {
                "primary": primary["input_files"],
                "comparison_reference": reference["input_files"],
                "prices": str(self.root / "data" / "raw" / "prices_YYYY-MM-DD.json"),
            },
            "exit_delay_1d_audit": delay,
            "candidate_presence_audit": presence,
            "candidate_presence_hold_virtual_audit": rules,
            "check_71570": check_71570,
            "v279_delta_reproducibility": reproducibility,
            "clean_v280_candidate_judgement": judgement,
            "next_actions": self._next_actions(judgement),
        }

    def save_report(self, result: dict[str, Any]) -> Phase4GPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase4GPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 4-G Exit Delay / Candidate Presence Hold Audit",
            "",
            "## Scope",
            "",
            "- audit_only: `true`",
            "- source: existing primary/reference logs, `purchase_audit.csv`, `trades.csv`, and cached `data/raw/prices_YYYY-MM-DD.json`.",
            "- no profile creation, no backtest, no API fetch, no prediction regeneration.",
            "",
            "## Exit AI 1-Day Delay",
            "",
            self._table([result["exit_delay_1d_audit"]["summary"]], ["trade_count", "actual_net_profit", "virtual_delay_1d_net_profit", "profit_delta", "actual_pf", "virtual_pf", "actual_win_rate", "virtual_win_rate", "average_profit_delta", "positive_delta_count", "negative_delta_count"]),
            "",
            "### By PM Multiplier",
            "",
            self._table(result["exit_delay_1d_audit"]["by_pm_multiplier"], ["bucket", "trade_count", "actual_net_profit", "virtual_net_profit", "profit_delta", "virtual_pf", "virtual_win_rate"]),
            "",
            "### By Holding Days",
            "",
            self._table(result["exit_delay_1d_audit"]["by_holding_days"], ["bucket", "trade_count", "actual_net_profit", "virtual_net_profit", "profit_delta", "virtual_pf", "virtual_win_rate"]),
            "",
            "## Candidate Presence Hold Rules",
            "",
            self._table(result["candidate_presence_hold_virtual_audit"]["rules"], ["rule", "eligible_trade_count", "held_trade_count", "actual_net_profit", "virtual_net_profit", "profit_delta", "virtual_pf", "virtual_win_rate", "positive_delta_count", "negative_delta_count", "average_delta", "median_delta"]),
            "",
            "## 71570 Reproduction Check",
            "",
            self._table([result["check_71570"]], ["actual_sell_date", "actual_net_profit", "virtual_hold_sell_date", "virtual_net_profit", "same_day_candidate_present", "next_day_candidate_present", "pm_score", "pm_multiplier", "rule_matched"]),
            "",
            "## v2_79 Top Delta Reproducibility",
            "",
            self._table([result["v279_delta_reproducibility"]], ["explained_count", "unexplained_count", "explained_profit_delta", "unexplained_profit_delta"]),
            "",
            "## Clean v2_80 Judgement",
            "",
            self._table([result["clean_v280_candidate_judgement"]], ["exit_delay_1d_recommended", "candidate_presence_hold_recommended", "pm_multiplier_presence_hold_recommended", "clean_v280_worth_implementing", "v279_side_effect_reproducible_by_clean_rule", "log_reliability"]),
            "",
            "## Next Actions",
            "",
        ]
        lines.extend(f"- {item}" for item in result.get("next_actions", []))
        lines.append("")
        return "\n".join(lines)

    def _load_profile_logs(self, profile: str) -> dict[str, Any]:
        base = self.root / "logs" / "backtests" / profile / self.period
        files = {
            "backtest_summary": base / "backtest_summary.json",
            "summary": base / "summary.csv",
            "trades": base / "trades.csv",
            "purchase_audit": base / "purchase_audit.csv",
        }
        return {
            "input_files": {key: str(path) for key, path in files.items()},
            "summary_json": _read_json(files["backtest_summary"]),
            "daily": _read_csv(files["summary"]),
            "trades": _read_csv(files["trades"]),
            "purchase_audit": _read_csv(files["purchase_audit"]),
        }

    def _exit_delay_audit(self, exit_rows: pd.DataFrame) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for row in exit_rows.to_dict("records"):
            sell_date = _to_str(row.get("sell_date"))
            code = _to_str(row.get("code"))
            next_date = self.prices.next_trading_date(sell_date)
            next_close = self.prices.close(next_date, code) if next_date else None
            actual_exit = _to_float(row.get("actual_exit_price")) or _to_float(row.get("exit_price"))
            shares = _to_float(row.get("shares")) or 0.0
            actual_net = _net_profit_column(row)
            if next_date and next_close is not None and actual_exit is not None:
                delta = (next_close - actual_exit) * shares
                virtual_net = actual_net + delta
                missing = False
            else:
                delta = None
                virtual_net = actual_net
                missing = True
            rows.append(
                {
                    "code": code,
                    "buy_date": _to_str(row.get("buy_date")),
                    "actual_sell_date": sell_date,
                    "virtual_sell_date": next_date,
                    "actual_exit_price": actual_exit,
                    "virtual_exit_close": next_close,
                    "shares": shares,
                    "actual_net_profit": actual_net,
                    "virtual_delay_1d_net_profit": virtual_net,
                    "profit_delta": delta,
                    "pm_score": _to_float(row.get("pm_score")),
                    "pm_multiplier": _to_float(row.get("pm_multiplier")),
                    "holding_days": _to_int(row.get("holding_days")),
                    "realized_profit_sign": "positive" if actual_net > 0 else "negative",
                    "price_data_missing": missing,
                }
            )
        valid = [row for row in rows if row.get("profit_delta") is not None]
        return {
            "summary": self._virtual_summary(valid, "virtual_delay_1d_net_profit"),
            "by_pm_multiplier": self._group_summary(valid, "pm_multiplier_bucket"),
            "by_holding_days": self._group_summary(valid, "holding_days_bucket"),
            "by_realized_profit_sign": self._group_summary(valid, "realized_profit_sign"),
            "rows": [self._decorate_buckets(row) for row in rows],
            "price_method": "virtual net profit = actual net profit + shares * (next close - actual exit price); fees/tax are not re-estimated.",
        }

    def _decorate_buckets(self, row: dict[str, Any]) -> dict[str, Any]:
        pm = row.get("pm_multiplier")
        days = row.get("holding_days")
        row["pm_multiplier_bucket"] = self._pm_bucket(pm)
        row["holding_days_bucket"] = self._holding_bucket(days)
        return row

    def _pm_bucket(self, value: float | None) -> str:
        if value is None:
            return "missing"
        if value <= 0.8:
            return "0.8"
        if value <= 1.0:
            return "1.0"
        if value <= 1.15:
            return "1.15"
        return "1.3"

    def _holding_bucket(self, value: int | None) -> str:
        if value is None:
            return "missing"
        if value <= 1:
            return "1d"
        if value <= 3:
            return "2-3d"
        if value <= 5:
            return "4-5d"
        return "6d+"

    def _virtual_summary(self, rows: list[dict[str, Any]], virtual_key: str) -> dict[str, Any]:
        actual = [float(row["actual_net_profit"]) for row in rows]
        virtual = [float(row[virtual_key]) for row in rows]
        deltas = [float(row["profit_delta"]) for row in rows if row.get("profit_delta") is not None]
        return {
            "trade_count": len(rows),
            "actual_net_profit": round(sum(actual), 2),
            "virtual_delay_1d_net_profit": round(sum(virtual), 2),
            "virtual_net_profit": round(sum(virtual), 2),
            "profit_delta": round(sum(deltas), 2),
            "actual_pf": _profit_factor(actual),
            "virtual_pf": _profit_factor(virtual),
            "actual_win_rate": _win_rate(actual),
            "virtual_win_rate": _win_rate(virtual),
            "average_profit_delta": None if not deltas else sum(deltas) / len(deltas),
            "average_delta": None if not deltas else sum(deltas) / len(deltas),
            "median_delta": None if not deltas else median(deltas),
            "positive_delta_count": sum(1 for delta in deltas if delta > 0),
            "negative_delta_count": sum(1 for delta in deltas if delta < 0),
        }

    def _group_summary(self, rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            decorated = self._decorate_buckets(dict(row))
            buckets.setdefault(_to_str(decorated.get(group_key)), []).append(decorated)
        result = []
        for bucket, items in sorted(buckets.items()):
            summary = self._virtual_summary(items, "virtual_delay_1d_net_profit")
            result.append({"bucket": bucket, **summary})
        return result

    def _candidate_presence_audit(self, exit_rows: pd.DataFrame, audit: pd.DataFrame) -> dict[str, Any]:
        audit_rows = audit.copy()
        if audit_rows.empty:
            return {"log_reliability": "low_missing_purchase_audit", "rows": []}
        audit_rows["code"] = audit_rows.get("code", "").fillna("").astype(str)
        audit_rows["signal_date"] = audit_rows.get("signal_date", "").fillna("").astype(str)
        audit_rows["entry_date"] = audit_rows.get("entry_date", "").fillna("").astype(str)
        rows = []
        for trade in exit_rows.to_dict("records"):
            code = _to_str(trade.get("code"))
            sell_date = _to_str(trade.get("sell_date"))
            next_date = self.prices.next_trading_date(sell_date)
            same = self._candidate_row(audit_rows, code, sell_date)
            nxt = self._candidate_row(audit_rows, code, next_date or "")
            rows.append(
                {
                    "code": code,
                    "buy_date": _to_str(trade.get("buy_date")),
                    "sell_date": sell_date,
                    "next_trading_date": next_date,
                    "same_day_candidate_present": same is not None,
                    "next_day_candidate_present": nxt is not None,
                    "same_day_pm_score": _to_float((same or {}).get("pm_score")),
                    "next_day_pm_score": _to_float((nxt or {}).get("pm_score")),
                    "same_day_pm_multiplier": _to_float((same or {}).get("pm_multiplier")),
                    "next_day_pm_multiplier": _to_float((nxt or {}).get("pm_multiplier")),
                    "same_day_candidate_order": self._candidate_order(same),
                    "next_day_candidate_order": self._candidate_order(nxt),
                }
            )
        return {
            "log_reliability": "medium_partial_purchase_audit_candidates",
            "definition": "candidate presence is reconstructed from purchase_audit rows where signal_date or entry_date equals the checked date; this is not a complete scored universe.",
            "rows": rows,
        }

    def _candidate_row(self, audit: pd.DataFrame, code: str, date: str) -> dict[str, Any] | None:
        if not date:
            return None
        rows = audit[(audit["code"].eq(str(code))) & (audit["signal_date"].eq(date) | audit["entry_date"].eq(date))]
        if rows.empty:
            return None
        return rows.sort_values(["signal_date", "entry_date"], kind="stable").iloc[0].to_dict()

    def _candidate_order(self, row: dict[str, Any] | None) -> float | None:
        if not row:
            return None
        for key in ["pm_aware_candidate_order", "candidate_rank", "score_rank"]:
            value = _to_float(row.get(key))
            if value is not None:
                return value
        return None

    def _candidate_presence_rule_audit(self, delay_rows: list[dict[str, Any]], presence_rows: list[dict[str, Any]]) -> dict[str, Any]:
        presence_by_key = {(row["code"], row["sell_date"]): row for row in presence_rows}
        rules = []
        for rule in ["A_same_day_pm_score_gte_0", "B_next_day_pm_score_gte_0", "C_same_or_next_pm_multiplier_gte_1_0", "D_trade_pm_multiplier_gte_1_15"]:
            held_rows = []
            eligible = [row for row in delay_rows if not row.get("price_data_missing")]
            for row in eligible:
                presence = presence_by_key.get((row["code"], row["actual_sell_date"]), {})
                if self._rule_matches(rule, row, presence):
                    held_rows.append(row)
            actual = sum(float(row["actual_net_profit"]) for row in held_rows)
            virtual = sum(float(row["virtual_delay_1d_net_profit"]) for row in held_rows)
            deltas = [float(row["profit_delta"]) for row in held_rows if row.get("profit_delta") is not None]
            summary = self._virtual_summary(held_rows, "virtual_delay_1d_net_profit")
            rules.append(
                {
                    "rule": rule,
                    "eligible_trade_count": len(eligible),
                    "held_trade_count": len(held_rows),
                    "actual_net_profit": round(actual, 2),
                    "virtual_net_profit": round(virtual, 2),
                    "profit_delta": round(sum(deltas), 2),
                    "virtual_pf": summary["virtual_pf"],
                    "virtual_win_rate": summary["virtual_win_rate"],
                    "positive_delta_count": summary["positive_delta_count"],
                    "negative_delta_count": summary["negative_delta_count"],
                    "average_delta": summary["average_delta"],
                    "median_delta": summary["median_delta"],
                }
            )
        return {"rules": rules}

    def _rule_matches(self, rule: str, row: dict[str, Any], presence: dict[str, Any]) -> bool:
        if rule == "A_same_day_pm_score_gte_0":
            value = _to_float(presence.get("same_day_pm_score"))
            return bool(presence.get("same_day_candidate_present")) and value is not None and value >= 0
        if rule == "B_next_day_pm_score_gte_0":
            value = _to_float(presence.get("next_day_pm_score"))
            return bool(presence.get("next_day_candidate_present")) and value is not None and value >= 0
        if rule == "C_same_or_next_pm_multiplier_gte_1_0":
            values = [presence.get("same_day_pm_multiplier"), presence.get("next_day_pm_multiplier")]
            return any(value is not None and value >= 1.0 for value in (_to_float(value) for value in values))
        if rule == "D_trade_pm_multiplier_gte_1_15":
            value = _to_float(row.get("pm_multiplier"))
            return value is not None and value >= 1.15
        return False

    def _check_71570(self, delay_rows: list[dict[str, Any]], presence_rows: list[dict[str, Any]], rules: dict[str, Any]) -> dict[str, Any]:
        row = next((item for item in delay_rows if item.get("code") == "71570" and item.get("actual_sell_date") == "2023-01-24"), {})
        if not row:
            row = next((item for item in delay_rows if item.get("code") == "71570"), {})
        presence = next((item for item in presence_rows if item.get("code") == "71570" and item.get("sell_date") == "2023-01-24"), {})
        matched = []
        for rule in rules.get("rules", []):
            if row and self._rule_matches(rule["rule"], row, presence):
                matched.append(rule["rule"])
        return {
            "actual_sell_date": row.get("actual_sell_date"),
            "actual_net_profit": row.get("actual_net_profit"),
            "virtual_hold_sell_date": row.get("virtual_sell_date"),
            "virtual_net_profit": row.get("virtual_delay_1d_net_profit"),
            "same_day_candidate_present": presence.get("same_day_candidate_present"),
            "next_day_candidate_present": presence.get("next_day_candidate_present"),
            "pm_score": row.get("pm_score"),
            "pm_multiplier": row.get("pm_multiplier"),
            "rule_matched": ",".join(matched),
        }

    def _v279_reproducibility(
        self,
        delay_rows: list[dict[str, Any]],
        rules: dict[str, Any],
        primary_trades: pd.DataFrame,
        reference_trades: pd.DataFrame,
    ) -> dict[str, Any]:
        delay_by_code = {row["code"]: row for row in delay_rows if row.get("code") in TOP_DELTA_CODES}
        rule_names = [row["rule"] for row in rules.get("rules", []) if row.get("profit_delta", 0) > 0]
        primary_profit = self._profit_by_code(primary_trades)
        reference_profit = self._profit_by_code(reference_trades)
        explained = []
        unexplained = []
        for code in TOP_DELTA_CODES:
            delta = reference_profit.get(code, 0.0) - primary_profit.get(code, 0.0)
            delay_row = delay_by_code.get(code)
            is_explained = bool(delay_row and delay_row.get("profit_delta") and delay_row.get("profit_delta") > 0 and rule_names)
            (explained if is_explained else unexplained).append({"code": code, "profit_delta": delta})
        return {
            "target_codes": TOP_DELTA_CODES,
            "explained_count": len(explained),
            "unexplained_count": len(unexplained),
            "explained_profit_delta": round(sum(row["profit_delta"] for row in explained), 2),
            "unexplained_profit_delta": round(sum(row["profit_delta"] for row in unexplained), 2),
            "explained_rows": explained,
            "unexplained_rows": unexplained,
        }

    def _profit_by_code(self, trades: pd.DataFrame) -> dict[str, float]:
        if trades.empty or "code" not in trades.columns:
            return {}
        df = trades.copy()
        df["code"] = df["code"].astype(str)
        df["net_profit"] = pd.to_numeric(df.get("net_profit", 0), errors="coerce").fillna(0.0)
        return {code: float(group["net_profit"].sum()) for code, group in df.groupby("code")}

    def _judgement(self, delay: dict[str, Any], rules: dict[str, Any], reproducibility: dict[str, Any], reliability: str) -> dict[str, Any]:
        delay_delta = float(delay["summary"].get("profit_delta") or 0.0)
        best_rule = max(rules.get("rules", []), key=lambda row: float(row.get("profit_delta") or 0.0), default={})
        best_rule_delta = float(best_rule.get("profit_delta") or 0.0)
        return {
            "exit_delay_1d_recommended": delay_delta > 0,
            "candidate_presence_hold_recommended": best_rule_delta > 0 and str(best_rule.get("rule", "")).startswith(("A_", "B_")),
            "pm_multiplier_presence_hold_recommended": best_rule_delta > 0 and str(best_rule.get("rule", "")).startswith(("C_", "D_")),
            "clean_v280_worth_implementing": delay_delta > 0 or best_rule_delta > 0,
            "v279_side_effect_reproducible_by_clean_rule": int(reproducibility.get("explained_count") or 0) > 0,
            "best_rule": best_rule.get("rule"),
            "best_rule_profit_delta": best_rule_delta,
            "log_reliability": reliability,
        }

    def _next_actions(self, judgement: dict[str, Any]) -> list[str]:
        actions = ["Keep v2_78 as the main profile and keep v2_79 on hold."]
        if judgement.get("exit_delay_1d_recommended"):
            actions.append("Consider clean v2_80: Exit AI 1-day delay.")
        if judgement.get("candidate_presence_hold_recommended"):
            actions.append("Consider clean v2_80: Candidate Presence Hold.")
        if judgement.get("pm_multiplier_presence_hold_recommended"):
            actions.append("Consider clean v2_80: high-PM candidate presence hold.")
        if not judgement.get("clean_v280_worth_implementing"):
            actions.append("Put Exit AI delay improvements on hold and move to Market Regime Audit.")
        return actions

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = round(value, 6)
                values.append(str(value).replace("\n", " "))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)


def build_report(root: Path | str = ROOT) -> dict[str, Any]:
    return PortfolioManagerPhase4GExitDelayCandidateHoldAudit(root).build_report()


def save_report(result: dict[str, Any], root: Path | str = ROOT) -> Phase4GPaths:
    return PortfolioManagerPhase4GExitDelayCandidateHoldAudit(root).save_report(result)


def run(root: Path | str = ROOT) -> Phase4GPaths:
    audit = PortfolioManagerPhase4GExitDelayCandidateHoldAudit(root)
    return audit.save_report(audit.build_report())
