"""Phase 4-F side-effect audit for v2_78 vs v2_79.

This module is intentionally read-only. It compares existing backtest logs,
loaded profile configuration, and static references in ``paper_trade.py`` to
explain why v2_79 changed despite the high-PM minimum-hold guard not blocking
any exits.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config_version import load_config  # noqa: E402
from profile_loader import PROFILE_ALIASES, get_profile_path, load_profile  # noqa: E402


PERIOD = "2023-01-01_to_2026-05-31"
REPORT_STEM = "portfolio_manager_phase4f_side_effect_audit_2023-01_to_2026-05"
V278 = "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
V279 = "rookie_dealer_02_v2_79_high_pm_min_hold_5d"
HIGH_PM_THRESHOLD = 1.15

AUDIT_KEYS = [
    "portfolio_manager_ai_sizing.buy_ordering_mode",
    "portfolio_manager_ai_sizing.pm_order_weight",
    "portfolio_manager_ai_sizing.fallback_to_next_affordable_selected",
    "portfolio_manager_ai_sizing.fallback_min_pm_score",
    "portfolio_manager_ai_sizing.fallback_min_pm_multiplier",
    "portfolio_manager_ai_sizing.per_code_exposure_cap_rate",
    "portfolio_manager_ai_sizing.low_score_skip_enabled",
    "portfolio_manager_ai_sizing.low_score_skip_threshold",
    "selection.max_selected",
    "portfolio.max_positions",
    "trading.max_positions",
    "ml_exit_ai.enabled",
    "ml_exit_ai.threshold",
    "portfolio_manager_ai_sizing.high_pm_min_hold_enabled",
    "portfolio_manager_ai_sizing.high_pm_min_hold_days",
    "portfolio_manager_ai_sizing.high_pm_min_hold_min_multiplier",
]

PAPER_TRADE_KEYS = [
    "buy_ordering_mode",
    "pm_order_weight",
    "fallback_to_next_affordable_selected",
    "fallback_min_pm_score",
    "fallback_min_pm_multiplier",
    "per_code_exposure_cap_rate",
    "low_score_skip_enabled",
    "low_score_skip_threshold",
    "daily_buy_limit",
    "max_selected",
    "max_positions",
    "high_pm_min_hold_enabled",
    "high_pm_min_hold_days",
    "high_pm_min_hold_min_multiplier",
]


@dataclass(frozen=True)
class Phase4FAuditPaths:
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
    number = _to_float(value)
    return None if number is None else int(number)


def _to_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _get_path(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key, value in payload.items():
        if str(key).startswith("_"):
            continue
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.update(_flatten(value, dotted))
        else:
            rows[dotted] = value
    return rows


def _profile_dir(root: Path, profile: str) -> Path:
    return root / "logs" / "backtests" / profile / PERIOD


def _buy_rows(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty or "decision" not in audit.columns:
        return pd.DataFrame()
    rows = audit[
        audit["decision"].fillna("").astype(str).isin(["BUY", "SCALED_BUY"])
        & (pd.to_numeric(audit.get("final_shares", 0), errors="coerce").fillna(0) > 0)
    ].copy()
    if rows.empty:
        return rows
    rows["buy_date"] = rows.get("entry_date", "").fillna("").astype(str)
    rows["code"] = rows.get("code", "").fillna("").astype(str)
    rows = rows.sort_values(["buy_date", "code", "trade_id"], kind="stable")
    rows["instance"] = rows.groupby(["buy_date", "code"]).cumcount()
    return rows


def _sell_rows(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "action" not in trades.columns:
        return pd.DataFrame()
    rows = trades[trades["action"].fillna("").astype(str).eq("SELL")].copy()
    if rows.empty:
        return rows
    rows["buy_date"] = rows.get("entry_date", "").fillna("").astype(str)
    rows["sell_date"] = rows.get("exit_date", "").fillna("").astype(str)
    rows["code"] = rows.get("code", "").fillna("").astype(str)
    if "trade_id" not in rows.columns:
        rows["trade_id"] = ""
    rows = rows.sort_values(["buy_date", "code", "sell_date", "trade_id"], kind="stable")
    rows["instance"] = rows.groupby(["buy_date", "code"]).cumcount()
    return rows


def _records(df: pd.DataFrame, limit: int = 200) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return json.loads(df.head(limit).where(pd.notna(df), None).to_json(orient="records", force_ascii=False))


def _keyed(df: pd.DataFrame, columns: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    if df.empty:
        return {}
    return {tuple(row.get(column) for column in columns): row for row in df.to_dict("records")}


def _opposite_skip_lookup(audit: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if audit.empty:
        return {}
    rows = audit.copy()
    rows["buy_date"] = rows.get("entry_date", "").fillna("").astype(str)
    rows["code"] = rows.get("code", "").fillna("").astype(str)
    decision = rows.get("decision", pd.Series(index=rows.index, dtype=str)).fillna("").astype(str)
    rows = rows[~decision.isin(["BUY", "SCALED_BUY"])]
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows.to_dict("records"):
        lookup.setdefault((_to_str(row.get("buy_date")), _to_str(row.get("code"))), row)
    return lookup


def _buy_amount(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    for key in ["final_amount", "scaled_amount", "planned_amount"]:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _candidate_order(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    for key in ["pm_aware_candidate_order", "original_candidate_order", "candidate_rank", "score_rank"]:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _skip_reason(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    for key in ["skip_reason", "scale_reason", "reject_reason", "allocation_reason"]:
        value = _to_str(row.get(key))
        if value:
            return value
    return ""


def _profit_factor(values: pd.Series) -> float | None:
    profits = pd.to_numeric(values, errors="coerce").dropna()
    if profits.empty:
        return None
    gross_profit = float(profits[profits > 0].sum())
    gross_loss = abs(float(profits[profits < 0].sum()))
    return None if gross_loss == 0 else gross_profit / gross_loss


class PortfolioManagerPhase4FSideEffectAudit:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        v278: str = V278,
        v279: str = V279,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.v278 = v278
        self.v279 = v279
        self.period = period

    def build_report(self) -> dict[str, Any]:
        data = self._load_existing_logs()
        profile_diff = self._profile_diff()
        code_refs = self._paper_trade_reference_audit()
        daily_divergence = self._daily_divergence(data)
        buy_root = self._buy_root_cause(data)
        sell_root = self._sell_root_cause(data)
        minimum_hold = self._minimum_hold_confirmation(data, profile_diff)
        side_effect = self._side_effect_judgement(profile_diff, code_refs, daily_divergence, buy_root, sell_root, minimum_hold)
        return {
            "metadata": {
                "phase": "4-F",
                "audit_only": True,
                "full_backtest_executed": False,
                "full_pytest_executed": False,
                "uses_existing_logs_only": True,
                "profiles": {"v2_78": self.v278, "v2_79": self.v279},
                "period": self.period,
            },
            "input_files": data["input_files"],
            "profile_diff": profile_diff,
            "paper_trade_reference_audit": code_refs,
            "daily_path_divergence": daily_divergence,
            "buy_root_cause": buy_root,
            "sell_root_cause": sell_root,
            "minimum_hold_confirmation": minimum_hold,
            "side_effect_judgement": side_effect,
            "next_actions": self._next_actions(side_effect),
        }

    def save_report(self, result: dict[str, Any]) -> Phase4FAuditPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase4FAuditPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 4-F Side Effect Root Cause Audit",
            "",
            "## Scope",
            "",
            "- audit_only: `true`",
            "- source: existing `backtest_summary.json`, `summary.csv`, `trades.csv`, `purchase_audit.csv`, profile YAML/effective config, and static `src/paper_trade.py` references.",
            "- prohibited actions respected by design: no backtest, no API fetch, no OpenAI call, no prediction regeneration, no live order placement.",
            "",
            "## Profile Diff",
            "",
            self._table(result["profile_diff"].get("focused_key_differences", []), ["key", "v2_78", "v2_79", "difference_type"]),
            "",
            "## Minimum Hold References",
            "",
            self._table(result["paper_trade_reference_audit"].get("high_pm_min_hold_references", []), ["line", "function", "category", "text"]),
            "",
            "## Daily Path Divergence",
            "",
            self._table([result["daily_path_divergence"]], ["first_divergence_date", "divergence_type", "v2_78_cash", "v2_79_cash", "v2_78_holdings", "v2_79_holdings", "likely_cause", "reliability"]),
            "",
            "## BUY Root Cause",
            "",
            self._table(result["buy_root_cause"].get("summary", []), ["metric", "value"]),
            "",
            self._table(result["buy_root_cause"].get("rows", [])[:80], ["buy_date", "code", "only_side", "pm_score", "pm_multiplier", "buy_amount", "skip_reason_on_opposite_side", "cash_before_buy", "daily_buy_remaining", "per_code_cap_remaining", "candidate_order", "buy_priority_score", "likely_cause"]),
            "",
            "## SELL Root Cause",
            "",
            self._table(result["sell_root_cause"].get("summary", []), ["metric", "value"]),
            "",
            self._table(result["sell_root_cause"].get("rows", [])[:80], ["code", "buy_date", "sell_date_v278", "sell_date_v279", "exit_reason_v278", "exit_reason_v279", "holding_days_v278", "holding_days_v279", "realized_profit_v278", "realized_profit_v279", "high_pm_min_hold_applied", "high_pm_min_hold_blocked_exit_count", "likely_cause"]),
            "",
            "## Minimum Hold Confirmation",
            "",
            self._table([result["minimum_hold_confirmation"]], ["high_pm_min_hold_enabled", "high_pm_min_hold_days", "high_pm_target_position_count", "high_pm_exit_ai_signal_under_min_hold_count", "blocked_exit_count", "blocked_exit_count_consistent", "bug_candidate"]),
            "",
            "## Side Effect Judgement",
            "",
            self._table([result["side_effect_judgement"]], ["v279_improvement_explained", "minimum_hold_directly_effective", "unintended_buy_side_effect_suspected", "unintended_sizing_side_effect_suspected", "log_comparison_reliable", "v279_safe_to_adopt", "should_create_clean_v280_from_discovered_effect"]),
            "",
            "## Next Actions",
            "",
        ]
        lines.extend(f"- {item}" for item in result.get("next_actions", []))
        lines.append("")
        return "\n".join(lines)

    def _load_existing_logs(self) -> dict[str, Any]:
        result: dict[str, Any] = {"profiles": {}, "input_files": {}}
        for label, profile in {"v2_78": self.v278, "v2_79": self.v279}.items():
            base = self.root / "logs" / "backtests" / profile / self.period
            files = {
                "backtest_summary": base / "backtest_summary.json",
                "summary": base / "summary.csv",
                "trades": base / "trades.csv",
                "purchase_audit": base / "purchase_audit.csv",
            }
            result["input_files"][label] = {key: str(path) for key, path in files.items()}
            result["profiles"][label] = {
                "base_dir": str(base),
                "summary_json": _read_json(files["backtest_summary"]),
                "daily": _read_csv(files["summary"]),
                "trades": _read_csv(files["trades"]),
                "purchase_audit": _read_csv(files["purchase_audit"]),
            }
        return result

    def _profile_diff(self) -> dict[str, Any]:
        raw_78 = load_config(get_profile_path(self.v278))
        raw_79 = load_config(get_profile_path(self.v279))
        effective_78 = load_profile(self.v278)
        effective_79 = load_profile(self.v279)
        raw_flat = _flatten(raw_78)
        raw_flat_79 = _flatten(raw_79)
        effective_flat = _flatten(effective_78)
        effective_flat_79 = _flatten(effective_79)
        raw_diffs = self._diff_flat(raw_flat, raw_flat_79)
        effective_diffs = self._diff_flat(effective_flat, effective_flat_79)
        focused = []
        for key in AUDIT_KEYS:
            left = _get_path(effective_78, key)
            right = _get_path(effective_79, key)
            raw_left = _get_path(raw_78, key)
            raw_right = _get_path(raw_79, key)
            if left != right or raw_left != raw_right:
                focused.append(
                    {
                        "key": key,
                        "v2_78": left,
                        "v2_79": right,
                        "raw_v2_78": raw_left,
                        "raw_v2_79": raw_right,
                        "difference_type": "explicit_profile_diff" if raw_left != raw_right else "inherited_or_default_diff",
                    }
                )
        aliases = {
            alias: target
            for alias, target in PROFILE_ALIASES.items()
            if target in {self.v278, self.v279} or alias in {"rookie_dealer_02_v2_78", "rookie_dealer_02_v2.78", "rookie_dealer_02_v2_79", "rookie_dealer_02_v2.79"}
        }
        paper_trade_values = [
            {"key": key, "v2_78": _get_path(effective_78, f"portfolio_manager_ai_sizing.{key}") or _get_path(effective_78, key), "v2_79": _get_path(effective_79, f"portfolio_manager_ai_sizing.{key}") or _get_path(effective_79, key)}
            for key in PAPER_TRADE_KEYS
        ]
        return {
            "differing_keys": sorted({item["key"] for item in effective_diffs}),
            "raw_profile_differences": raw_diffs,
            "effective_config_differences": effective_diffs,
            "focused_key_differences": focused,
            "profile_loader_alias_differences": aliases,
            "paper_trade_referenced_setting_values": paper_trade_values,
        }

    def _diff_flat(self, left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                rows.append({"key": key, "v2_78": left.get(key), "v2_79": right.get(key)})
        return rows

    def _paper_trade_reference_audit(self) -> dict[str, Any]:
        path = self.root / "src" / "paper_trade.py"
        text = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        refs = []
        current_func = ""
        for index, line in enumerate(text, start=1):
            stripped = line.strip()
            if stripped.startswith("def "):
                current_func = stripped.split("(", 1)[0].replace("def ", "")
            if "high_pm_min_hold" in line:
                category = self._reference_category(current_func, line)
                refs.append({"line": index, "function": current_func, "category": category, "text": stripped})
        key_refs = []
        for key in PAPER_TRADE_KEYS:
            locations = [index for index, line in enumerate(text, start=1) if key in line]
            key_refs.append({"key": key, "reference_count": len(locations), "lines": locations[:20]})
        categories = {row["category"] for row in refs}
        return {
            "paper_trade_path": str(path),
            "high_pm_min_hold_references": refs,
            "referenced_setting_keys": key_refs,
            "high_pm_min_hold_affects_buy_side": any(cat in categories for cat in ["buy", "sizing", "affordability", "purchase_audit"]),
            "high_pm_min_hold_affects_exit_side": "exit" in categories,
            "static_conclusion": "high_pm_min_hold references are confined to policy extraction, trade-field logging, and exit guard paths unless a BUY-side category appears.",
        }

    def _reference_category(self, func: str, line: str) -> str:
        haystack = f"{func} {line}".lower()
        if "purchase_audit" in haystack or "trade_fields" in haystack:
            return "purchase_audit"
        if "apply_high_pm_min_hold_exit_guard" in haystack or "exit" in haystack:
            return "exit"
        if "sizing" in haystack or "buy" in haystack:
            return "sizing"
        return "policy"

    def _daily_divergence(self, data: dict[str, Any]) -> dict[str, Any]:
        left = data["profiles"]["v2_78"]["daily"]
        right = data["profiles"]["v2_79"]["daily"]
        left_buys = self._daily_codes(_buy_rows(data["profiles"]["v2_78"]["purchase_audit"]), "buy_date")
        right_buys = self._daily_codes(_buy_rows(data["profiles"]["v2_79"]["purchase_audit"]), "buy_date")
        left_sells = self._daily_codes(_sell_rows(data["profiles"]["v2_78"]["trades"]), "sell_date")
        right_sells = self._daily_codes(_sell_rows(data["profiles"]["v2_79"]["trades"]), "sell_date")
        dates = sorted(set(left.get("date", pd.Series(dtype=str)).astype(str)) | set(right.get("date", pd.Series(dtype=str)).astype(str)) | set(left_buys) | set(right_buys) | set(left_sells) | set(right_sells))
        left_by_date = {str(row.get("date")): row for row in left.to_dict("records")} if not left.empty else {}
        right_by_date = {str(row.get("date")): row for row in right.to_dict("records")} if not right.empty else {}
        for date in dates:
            lrow = left_by_date.get(date, {})
            rrow = right_by_date.get(date, {})
            lb, rb = left_buys.get(date, []), right_buys.get(date, [])
            ls, rs = left_sells.get(date, []), right_sells.get(date, [])
            divergence_type = ""
            if lb != rb:
                divergence_type = "buy_difference"
            elif ls != rs:
                divergence_type = "sell_difference"
            elif _to_float(lrow.get("cash")) != _to_float(rrow.get("cash")):
                divergence_type = "cash_difference"
            elif _to_int(lrow.get("open_positions_count")) != _to_int(rrow.get("open_positions_count")):
                divergence_type = "holding_difference"
            if divergence_type:
                return {
                    "first_divergence_date": date,
                    "divergence_type": divergence_type,
                    "v2_78_cash": _to_float(lrow.get("cash")),
                    "v2_79_cash": _to_float(rrow.get("cash")),
                    "v2_78_holdings": _to_int(lrow.get("open_positions_count")),
                    "v2_79_holdings": _to_int(rrow.get("open_positions_count")),
                    "v2_78_buys": lb,
                    "v2_79_buys": rb,
                    "v2_78_sells": ls,
                    "v2_79_sells": rs,
                    "likely_cause": self._daily_likely_cause(divergence_type),
                    "reliability": "medium_summary_csv_only",
                    "note": "No per-day portfolio JSON was found for the target profile directories; path divergence is reconstructed from summary.csv, trades.csv, and purchase_audit.csv.",
                }
        return {"first_divergence_date": None, "divergence_type": "none", "reliability": "medium_summary_csv_only"}

    def _daily_codes(self, rows: pd.DataFrame, date_column: str) -> dict[str, list[str]]:
        if rows.empty or date_column not in rows.columns:
            return {}
        result: dict[str, list[str]] = {}
        for date, group in rows.groupby(date_column, sort=True):
            result[str(date)] = sorted(str(code) for code in group.get("code", pd.Series(dtype=str)).dropna())
        return result

    def _daily_likely_cause(self, divergence_type: str) -> str:
        return {
            "buy_difference": "buy_universe_or_affordability_path_divergence",
            "sell_difference": "sell_timing_path_divergence",
            "cash_difference": "position_size_or_realized_profit_path_divergence",
            "holding_difference": "holding_path_divergence",
        }.get(divergence_type, "unknown")

    def _buy_root_cause(self, data: dict[str, Any]) -> dict[str, Any]:
        left = _keyed(_buy_rows(data["profiles"]["v2_78"]["purchase_audit"]), ["buy_date", "code", "instance"])
        right = _keyed(_buy_rows(data["profiles"]["v2_79"]["purchase_audit"]), ["buy_date", "code", "instance"])
        left_skips = _opposite_skip_lookup(data["profiles"]["v2_78"]["purchase_audit"])
        right_skips = _opposite_skip_lookup(data["profiles"]["v2_79"]["purchase_audit"])
        rows = []
        counts = {"only_v2_78_buy_count": 0, "only_v2_79_buy_count": 0, "common_buy_count": 0}
        for key in sorted(set(left) | set(right)):
            lrow = left.get(key)
            rrow = right.get(key)
            if lrow and rrow:
                counts["common_buy_count"] += 1
                continue
            side = "v2_78" if lrow else "v2_79"
            counts[f"only_{side}_buy_count"] += 1
            row = lrow or rrow or {}
            opposite = right_skips.get((key[0], key[1])) if lrow else left_skips.get((key[0], key[1]))
            rows.append(
                {
                    "buy_date": key[0],
                    "code": key[1],
                    "only_side": side,
                    "pm_score": _to_float(row.get("pm_score")),
                    "pm_multiplier": _to_float(row.get("pm_multiplier")),
                    "buy_amount": _buy_amount(row),
                    "skip_reason_on_opposite_side": _skip_reason(opposite),
                    "cash_before_buy": _to_float(row.get("cash_before")),
                    "daily_buy_remaining": _to_float(row.get("daily_buy_limit_remaining_before")),
                    "per_code_cap_remaining": _to_float(row.get("pm_per_code_allowed_additional_buy")),
                    "candidate_order": _candidate_order(row),
                    "buy_priority_score": _to_float(row.get("buy_priority_score")),
                    "likely_cause": self._buy_likely_cause(row, opposite),
                }
            )
        return {
            "summary": [{"metric": key, "value": value} for key, value in counts.items()],
            "rows": rows,
            "sample_limit_note": "Markdown shows first 80 rows; JSON contains all rows.",
        }

    def _buy_likely_cause(self, row: dict[str, Any], opposite: dict[str, Any] | None) -> str:
        reason = _skip_reason(opposite)
        if not opposite:
            return "missing_log_data"
        if reason == "fallback_quality_filter" or _truthy(opposite.get("skipped_by_fallback_quality_filter")):
            return "fallback_quality_filter_difference"
        if reason in {"selected_but_not_affordable", "insufficient_available_cash", "target_exposure_limit", "round_lot_unaffordable"}:
            return "cash_path_divergence"
        if reason in {"duplicate_holding", "max_positions_limit"}:
            return "holding_path_divergence"
        if "per_code" in reason:
            return "per_code_cap_path_divergence"
        if _buy_amount(row) != _buy_amount(opposite):
            return "sizing_difference"
        return "unknown"

    def _sell_root_cause(self, data: dict[str, Any]) -> dict[str, Any]:
        left = _keyed(_sell_rows(data["profiles"]["v2_78"]["trades"]), ["buy_date", "code", "instance"])
        right = _keyed(_sell_rows(data["profiles"]["v2_79"]["trades"]), ["buy_date", "code", "instance"])
        rows = []
        changed_sell_date = 0
        for key in sorted(set(left) | set(right)):
            lrow = left.get(key)
            rrow = right.get(key)
            if lrow and rrow and _to_str(lrow.get("sell_date")) == _to_str(rrow.get("sell_date")):
                continue
            if lrow and rrow:
                changed_sell_date += 1
            row = rrow or lrow or {}
            blocked = _to_int(row.get("high_pm_min_hold_blocked_exit_count")) or 0
            rows.append(
                {
                    "code": key[1],
                    "buy_date": key[0],
                    "sell_date_v278": _to_str((lrow or {}).get("sell_date")),
                    "sell_date_v279": _to_str((rrow or {}).get("sell_date")),
                    "exit_reason_v278": _to_str((lrow or {}).get("exit_reason")),
                    "exit_reason_v279": _to_str((rrow or {}).get("exit_reason")),
                    "holding_days_v278": _to_int((lrow or {}).get("holding_days")),
                    "holding_days_v279": _to_int((rrow or {}).get("holding_days")),
                    "realized_profit_v278": _to_float((lrow or {}).get("net_profit")),
                    "realized_profit_v279": _to_float((rrow or {}).get("net_profit")),
                    "high_pm_min_hold_applied": _truthy(row.get("high_pm_min_hold_applied")),
                    "high_pm_min_hold_blocked_exit_count": blocked,
                    "likely_cause": self._sell_likely_cause(lrow, rrow, blocked),
                }
            )
        summary = [
            {"metric": "sell_date_changed_count", "value": changed_sell_date},
            {"metric": "sell_difference_row_count", "value": len(rows)},
        ]
        return {"summary": summary, "rows": rows}

    def _sell_likely_cause(self, left: dict[str, Any] | None, right: dict[str, Any] | None, blocked: int) -> str:
        if blocked > 0:
            return "minimum_hold_direct_effect"
        if left is None or right is None:
            return "buy_universe_or_holding_path_divergence"
        if _to_float(left.get("shares")) != _to_float(right.get("shares")):
            return "position_size_change"
        if _to_str(left.get("exit_reason")) != _to_str(right.get("exit_reason")):
            return "exit_reason_path_divergence"
        return "sell_timing_change"

    def _minimum_hold_confirmation(self, data: dict[str, Any], profile_diff: dict[str, Any]) -> dict[str, Any]:
        config = load_profile(self.v279)
        policy = config.get("portfolio_manager_ai_sizing", {})
        trades = _sell_rows(data["profiles"]["v2_79"]["trades"])
        pm = pd.to_numeric(trades.get("pm_multiplier", pd.Series(dtype=float)), errors="coerce") if not trades.empty else pd.Series(dtype=float)
        high_pm = trades[pm >= HIGH_PM_THRESHOLD].copy() if not trades.empty else pd.DataFrame()
        blocked = 0
        if not trades.empty and "high_pm_min_hold_blocked_exit_count" in trades.columns:
            blocked = int(pd.to_numeric(trades["high_pm_min_hold_blocked_exit_count"], errors="coerce").fillna(0).sum())
        elif not trades.empty and "high_pm_min_hold_blocked_exit" in trades.columns:
            blocked = int(trades["high_pm_min_hold_blocked_exit"].map(_truthy).sum())
        under_min_exit_ai = 0
        if not high_pm.empty:
            days = int(policy.get("high_pm_min_hold_days") or 0)
            under_days = pd.to_numeric(high_pm.get("holding_days", pd.Series(dtype=float)), errors="coerce") < days
            exit_ai = high_pm.get("exit_ai_triggered", pd.Series(False, index=high_pm.index)).map(_truthy)
            under_min_exit_ai = int((under_days & exit_ai).sum())
        bug_candidate = under_min_exit_ai > 0 and blocked == 0
        return {
            "high_pm_min_hold_enabled": bool(policy.get("high_pm_min_hold_enabled", False)),
            "high_pm_min_hold_days": int(policy.get("high_pm_min_hold_days") or 0),
            "high_pm_min_hold_min_multiplier": float(policy.get("high_pm_min_hold_min_multiplier") or HIGH_PM_THRESHOLD),
            "high_pm_target_position_count": int(len(high_pm)),
            "high_pm_exit_ai_signal_under_min_hold_count": under_min_exit_ai,
            "blocked_exit_count": blocked,
            "blocked_exit_count_consistent": blocked == 0 and under_min_exit_ai == 0,
            "bug_candidate": bug_candidate,
            "profile_differences_include_only_min_hold_in_pm_policy": self._pm_policy_diff_only_min_hold(profile_diff),
        }

    def _pm_policy_diff_only_min_hold(self, profile_diff: dict[str, Any]) -> bool:
        allowed = {
            "portfolio_manager_ai_sizing.high_pm_min_hold_enabled",
            "portfolio_manager_ai_sizing.high_pm_min_hold_days",
            "portfolio_manager_ai_sizing.high_pm_min_hold_min_multiplier",
            "portfolio_manager_ai_sizing.note",
        }
        pm_diffs = [row["key"] for row in profile_diff.get("effective_config_differences", []) if row["key"].startswith("portfolio_manager_ai_sizing.")]
        return all(key in allowed for key in pm_diffs)

    def _side_effect_judgement(
        self,
        profile_diff: dict[str, Any],
        code_refs: dict[str, Any],
        daily: dict[str, Any],
        buy_root: dict[str, Any],
        sell_root: dict[str, Any],
        minimum_hold: dict[str, Any],
    ) -> dict[str, Any]:
        only_buy_count = sum(int(row["value"]) for row in buy_root.get("summary", []) if str(row["metric"]).startswith("only_"))
        sizing_rows = sum(1 for row in sell_root.get("rows", []) if row.get("likely_cause") == "position_size_change")
        direct = int(minimum_hold.get("blocked_exit_count") or 0) > 0
        buy_side_ref = bool(code_refs.get("high_pm_min_hold_affects_buy_side"))
        reliable = daily.get("reliability") in {"high_daily_json", "medium_summary_csv_only"}
        improvement_explained = bool(only_buy_count or sizing_rows or daily.get("divergence_type") not in {"none", None})
        return {
            "v279_improvement_explained": improvement_explained,
            "minimum_hold_directly_effective": direct,
            "unintended_buy_side_effect_suspected": buy_side_ref or (only_buy_count > 0 and not direct),
            "unintended_sizing_side_effect_suspected": sizing_rows > 0 and not direct,
            "log_comparison_reliable": reliable,
            "v279_safe_to_adopt": False if not direct else reliable and not buy_side_ref,
            "should_create_clean_v280_from_discovered_effect": improvement_explained and not direct,
            "reason": "v2_79 should remain on hold until the non-minimum-hold divergence is explained by the generated audit.",
        }

    def _next_actions(self, side_effect: dict[str, Any]) -> list[str]:
        actions = ["Keep v2_78 as the main candidate until v2_79 improvement is explained."]
        if side_effect.get("should_create_clean_v280_from_discovered_effect"):
            actions.append("If the audit identifies beneficial buy-universe or sizing behavior, implement it explicitly as a clean v2_80 instead of adopting v2_79 as a minimum-hold change.")
        actions.append("Do not run a full backtest or full pytest for this audit step.")
        actions.append("If minimum-hold target positions existed but no under-5d Exit AI signal existed, blocked_exit_count=0 is expected.")
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
    return PortfolioManagerPhase4FSideEffectAudit(root).build_report()


def save_report(result: dict[str, Any], root: Path | str = ROOT) -> Phase4FAuditPaths:
    return PortfolioManagerPhase4FSideEffectAudit(root).save_report(result)


def run(root: Path | str = ROOT) -> Phase4FAuditPaths:
    audit = PortfolioManagerPhase4FSideEffectAudit(root)
    return audit.save_report(audit.build_report())
