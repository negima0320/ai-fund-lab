#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


PERIOD = "2023-01-01_to_2026-05-31"
REPORT_STEM = "portfolio_manager_phase4d_v278_vs_v279_diff_audit_2023-01_to_2026-05"
V278 = "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
V279 = "rookie_dealer_02_v2_79_high_pm_min_hold_5d"
AFFORDABILITY_REASONS = {"selected_but_not_affordable", "insufficient_available_cash"}


def _profile_dir(root: Path, profile: str) -> Path:
    return root / "logs" / "backtests" / profile / PERIOD


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _to_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _to_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _profit_factor(values: pd.Series) -> float | None:
    profits = pd.to_numeric(values, errors="coerce").dropna()
    if profits.empty:
        return None
    gross_profit = float(profits[profits > 0].sum())
    gross_loss = abs(float(profits[profits < 0].sum()))
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def _win_rate(values: pd.Series) -> float | None:
    profits = pd.to_numeric(values, errors="coerce").dropna()
    if profits.empty:
        return None
    return float((profits > 0).mean())


def _buy_audit_rows(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty or "decision" not in audit.columns:
        return pd.DataFrame()
    buys = audit[
        audit["decision"].fillna("").astype(str).isin(["BUY", "SCALED_BUY"])
        & (pd.to_numeric(audit.get("final_shares", 0), errors="coerce").fillna(0) > 0)
    ].copy()
    if buys.empty:
        return buys
    buys["buy_date"] = buys.get("entry_date", "").fillna("").astype(str)
    buys["code"] = buys.get("code", "").fillna("").astype(str)
    if "trade_id" not in buys.columns:
        buys["trade_id"] = ""
    buys = buys.sort_values(["buy_date", "code", "trade_id"], kind="stable")
    buys["instance"] = buys.groupby(["buy_date", "code"]).cumcount()
    return buys


def _sell_trade_rows(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "action" not in trades.columns:
        return pd.DataFrame()
    sells = trades[trades["action"].fillna("").astype(str).eq("SELL")].copy()
    if sells.empty:
        return sells
    sells["buy_date"] = sells.get("entry_date", "").fillna("").astype(str)
    sells["code"] = sells.get("code", "").fillna("").astype(str)
    if "trade_id" not in sells.columns:
        sells["trade_id"] = ""
    if "exit_date" not in sells.columns:
        sells["exit_date"] = ""
    sells = sells.sort_values(["buy_date", "code", "exit_date", "trade_id"], kind="stable")
    sells["instance"] = sells.groupby(["buy_date", "code"]).cumcount()
    return sells


def _keyed(df: pd.DataFrame, columns: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    if df.empty:
        return {}
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in df.to_dict("records"):
        result[tuple(row.get(column) for column in columns)] = row
    return result


def _skip_lookup(audit: pd.DataFrame) -> dict[tuple[str, str], str]:
    if audit.empty:
        return {}
    rows = audit.copy()
    rows["buy_date"] = rows.get("entry_date", "").fillna("").astype(str)
    rows["code"] = rows.get("code", "").fillna("").astype(str)
    decision = rows.get("decision", pd.Series(index=rows.index, dtype=str)).fillna("").astype(str)
    rows = rows[~decision.isin(["BUY", "SCALED_BUY"])]
    lookup: dict[tuple[str, str], str] = {}
    for row in rows.to_dict("records"):
        reason = _to_str(row.get("skip_reason")) or _to_str(row.get("scale_reason")) or _to_str(row.get("reject_reason"))
        if not reason:
            continue
        lookup.setdefault((_to_str(row.get("buy_date")), _to_str(row.get("code"))), reason)
    return lookup


def _buy_amount(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    for column in ["final_amount", "scaled_amount", "planned_amount"]:
        value = _to_float(row.get(column))
        if value is not None:
            return value
    return None


def _sell_buy_amount(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    entry = _to_float(row.get("entry_price"))
    shares = _to_float(row.get("shares"))
    if entry is None or shares is None:
        return None
    return entry * shares


def _rank(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    value = _to_float(row.get("candidate_rank"))
    if value is not None:
        return value
    return _to_float(row.get("score_rank"))


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return value
    return None


def _format_reason(base: str, details: list[str]) -> str:
    return base if not details else f"{base}: " + ", ".join(details)


def _build_buy_comparison(v278_audit: pd.DataFrame, v279_audit: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
    left = _keyed(_buy_audit_rows(v278_audit), ["buy_date", "code", "instance"])
    right = _keyed(_buy_audit_rows(v279_audit), ["buy_date", "code", "instance"])
    v278_skips = _skip_lookup(v278_audit)
    v279_skips = _skip_lookup(v279_audit)
    rows: list[dict[str, Any]] = []
    counts = {
        "only_v2_78_buy_count": 0,
        "only_v2_79_buy_count": 0,
        "common_buy_count": 0,
    }
    for key in sorted(set(left) | set(right)):
        lrow = left.get(key)
        rrow = right.get(key)
        bought_78 = lrow is not None
        bought_79 = rrow is not None
        if bought_78 and bought_79:
            counts["common_buy_count"] += 1
            details: list[str] = []
            if _buy_amount(lrow) != _buy_amount(rrow):
                details.append("buy_amount_changed")
            if _rank(lrow) != _rank(rrow):
                details.append("rank_changed")
            reason = _format_reason("common_buy", details)
        elif bought_78:
            counts["only_v2_78_buy_count"] += 1
            opposite = v279_skips.get((key[0], key[1]), "")
            reason = _format_reason("only_v2_78_buy", [f"v2_79={opposite}"] if opposite else [])
        else:
            counts["only_v2_79_buy_count"] += 1
            opposite = v278_skips.get((key[0], key[1]), "")
            reason = _format_reason("only_v2_79_buy", [f"v2_78={opposite}"] if opposite else [])

        source = rrow or lrow or {}
        rows.append(
            {
                "buy_date": key[0],
                "code": key[1],
                "v2_78_bought": bought_78,
                "v2_79_bought": bought_79,
                "pm_score": _first_not_none(_to_float((rrow or {}).get("pm_score")), _to_float((lrow or {}).get("pm_score"))),
                "pm_multiplier": _first_not_none(
                    _to_float((rrow or {}).get("pm_multiplier")),
                    _to_float((lrow or {}).get("pm_multiplier")),
                ),
                "buy_amount_v278": _buy_amount(lrow),
                "buy_amount_v279": _buy_amount(rrow),
                "buy_amount_delta": (_buy_amount(rrow) or 0.0) - (_buy_amount(lrow) or 0.0),
                "rank_v278": _rank(lrow),
                "rank_v279": _rank(rrow),
                "rank": _first_not_none(_rank(rrow), _rank(lrow)),
                "reason": reason,
                "name": _to_str(source.get("name")),
            }
        )
    return rows, counts


def _exit_category(row: dict[str, Any]) -> str:
    reason = _to_str(row.get("exit_reason")).lower()
    exit_ai_triggered = str(row.get("exit_ai_triggered")).strip().lower() in {"1", "true", "yes"}
    if exit_ai_triggered or "exit ai" in reason or "avoid_loss" in reason:
        return "exit_ai"
    if "stop_loss" in reason or "損切" in reason:
        return "stop_loss"
    if "take_profit" in reason or "利確" in reason:
        return "take_profit"
    if "max_holding" in reason or "最大保有" in reason:
        return "max_holding"
    if "forced" in reason or "force" in reason or "強制" in reason:
        return "forced_exit"
    return "other"


def _exit_reason_counts(trades: pd.DataFrame) -> dict[str, int]:
    counts = {key: 0 for key in ["exit_ai", "stop_loss", "take_profit", "max_holding", "forced_exit", "other"]}
    for row in _sell_trade_rows(trades).to_dict("records"):
        counts[_exit_category(row)] += 1
    return counts


def _build_sell_comparison(v278_trades: pd.DataFrame, v279_trades: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    left = _keyed(_sell_trade_rows(v278_trades), ["buy_date", "code", "instance"])
    right = _keyed(_sell_trade_rows(v279_trades), ["buy_date", "code", "instance"])
    rows: list[dict[str, Any]] = []
    common_day_deltas: list[float] = []
    common_profit_deltas: list[float] = []
    sell_date_changed = 0
    changed_trade_count = 0
    for key in sorted(set(left) | set(right)):
        lrow = left.get(key)
        rrow = right.get(key)
        profit_78 = _to_float((lrow or {}).get("net_profit"))
        profit_79 = _to_float((rrow or {}).get("net_profit"))
        holding_78 = _to_float((lrow or {}).get("holding_days"))
        holding_79 = _to_float((rrow or {}).get("holding_days"))
        profit_delta = (profit_79 or 0.0) - (profit_78 or 0.0)
        common = lrow is not None and rrow is not None
        if common:
            common_profit_deltas.append(profit_delta)
            if holding_78 is not None and holding_79 is not None:
                common_day_deltas.append(holding_79 - holding_78)
            if _to_str(lrow.get("exit_date")) != _to_str(rrow.get("exit_date")):
                sell_date_changed += 1
            if (
                _to_str(lrow.get("exit_date")) != _to_str(rrow.get("exit_date"))
                or abs(profit_delta) > 1e-9
                or _sell_buy_amount(lrow) != _sell_buy_amount(rrow)
            ):
                changed_trade_count += 1
        else:
            changed_trade_count += 1
        rows.append(
            {
                "code": key[1],
                "buy_date": key[0],
                "sell_date_v278": _to_str((lrow or {}).get("exit_date")),
                "sell_date_v279": _to_str((rrow or {}).get("exit_date")),
                "holding_days_v278": holding_78,
                "holding_days_v279": holding_79,
                "holding_day_delta": (holding_79 or 0.0) - (holding_78 or 0.0),
                "realized_profit_v278": profit_78,
                "realized_profit_v279": profit_79,
                "profit_delta": profit_delta,
                "buy_amount_v278": _sell_buy_amount(lrow),
                "buy_amount_v279": _sell_buy_amount(rrow),
                "pm_multiplier_v278": _to_float((lrow or {}).get("pm_multiplier")),
                "pm_multiplier_v279": _to_float((rrow or {}).get("pm_multiplier")),
                "exit_reason_v278": _to_str((lrow or {}).get("exit_reason")),
                "exit_reason_v279": _to_str((rrow or {}).get("exit_reason")),
                "match_status": "common" if common else ("only_v2_78_sell" if lrow is not None else "only_v2_79_sell"),
            }
        )
    summary = {
        "sell_date_changed_count": sell_date_changed,
        "changed_trade_count": changed_trade_count,
        "average_holding_day_delta": float(pd.Series(common_day_deltas).mean()) if common_day_deltas else None,
        "average_profit_delta": float(pd.Series(common_profit_deltas).mean()) if common_profit_deltas else None,
    }
    return rows, summary


def _high_pm_stats(trades: pd.DataFrame) -> dict[str, Any]:
    sells = _sell_trade_rows(trades)
    if sells.empty or "pm_multiplier" not in sells.columns:
        return {
            "trade_count": 0,
            "net_profit": 0.0,
            "profit_factor": None,
            "win_rate": None,
            "average_holding_days": None,
            "average_buy_amount": None,
        }
    high_pm = sells[pd.to_numeric(sells["pm_multiplier"], errors="coerce") >= 1.15].copy()
    profits = pd.to_numeric(high_pm.get("net_profit"), errors="coerce")
    holding = pd.to_numeric(high_pm.get("holding_days"), errors="coerce")
    buy_amount = pd.to_numeric(high_pm.get("entry_price"), errors="coerce") * pd.to_numeric(high_pm.get("shares"), errors="coerce")
    return {
        "trade_count": int(profits.notna().sum()),
        "net_profit": float(profits.sum()) if not profits.empty else 0.0,
        "profit_factor": _profit_factor(profits),
        "win_rate": _win_rate(profits),
        "average_holding_days": float(holding.dropna().mean()) if not holding.dropna().empty else None,
        "average_buy_amount": float(buy_amount.dropna().mean()) if not buy_amount.dropna().empty else None,
    }


def _is_high_pm(row: dict[str, Any] | None) -> bool:
    value = _to_float((row or {}).get("pm_multiplier"))
    return value is not None and value >= 1.15


def _build_profit_contribution(
    v278_trades: pd.DataFrame,
    v279_trades: pd.DataFrame,
    v278_audit: pd.DataFrame,
    v279_audit: pd.DataFrame,
    total_profit_delta: float,
) -> dict[str, Any]:
    left = _keyed(_sell_trade_rows(v278_trades), ["buy_date", "code", "instance"])
    right = _keyed(_sell_trade_rows(v279_trades), ["buy_date", "code", "instance"])
    v278_skips = _skip_lookup(v278_audit)
    v279_skips = _skip_lookup(v279_audit)
    buckets = {
        "buy_universe_change": 0.0,
        "sell_timing_change": 0.0,
        "position_size_change": 0.0,
        "affordability_change": 0.0,
        "high_pm_change": 0.0,
        "unknown": 0.0,
    }
    details = {key: 0 for key in buckets}
    for key in sorted(set(left) | set(right)):
        lrow = left.get(key)
        rrow = right.get(key)
        profit_78 = _to_float((lrow or {}).get("net_profit")) or 0.0
        profit_79 = _to_float((rrow or {}).get("net_profit")) or 0.0
        delta = profit_79 - profit_78
        if lrow is None or rrow is None:
            opposite_reason = v278_skips.get((key[0], key[1]), "") if rrow is not None else v279_skips.get((key[0], key[1]), "")
            bucket = "affordability_change" if opposite_reason in AFFORDABILITY_REASONS else "buy_universe_change"
        elif _to_str(lrow.get("exit_date")) != _to_str(rrow.get("exit_date")):
            bucket = "sell_timing_change"
        elif _sell_buy_amount(lrow) != _sell_buy_amount(rrow):
            bucket = "position_size_change"
        elif _is_high_pm(lrow) or _is_high_pm(rrow):
            bucket = "high_pm_change"
        else:
            bucket = "unknown"
        buckets[bucket] += delta
        if abs(delta) > 1e-9:
            details[bucket] += 1

    explained = sum(buckets.values())
    residual = total_profit_delta - explained
    if abs(residual) > 0.01:
        buckets["unknown"] += residual
    dominant = max(buckets.items(), key=lambda item: abs(item[1]))[0] if buckets else "unknown"
    return {
        "method": "Matched closed SELL trades by buy_date+code+instance. Only-side trades are buy_universe or affordability if the opposite audit skipped the same buy_date+code for affordability. Common trades are classified by sell-date change, then size change, then high-PM residual, then unknown.",
        "total_profit_delta": total_profit_delta,
        "buckets": buckets,
        "changed_trade_counts_by_bucket": details,
        "dominant_bucket": dominant,
    }


def _profile_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "net_profit": summary.get("net_cumulative_profit"),
        "profit_factor": summary.get("profit_factor"),
        "max_drawdown": summary.get("max_drawdown"),
        "win_rate": summary.get("win_rate"),
        "trades": summary.get("closed_trades_count"),
    }


def _top_rows(rows: list[dict[str, Any]], key: str, limit: int = 20) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: abs(float(row.get(key) or 0.0)), reverse=True)[:limit]


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_report(root: Path = ROOT) -> dict[str, Any]:
    v278_dir = _profile_dir(root, V278)
    v279_dir = _profile_dir(root, V279)
    v278_summary = _read_summary(v278_dir / "backtest_summary.json")
    v279_summary = _read_summary(v279_dir / "backtest_summary.json")
    v278_trades = _read_csv(v278_dir / "trades.csv")
    v279_trades = _read_csv(v279_dir / "trades.csv")
    v278_audit = _read_csv(v278_dir / "purchase_audit.csv")
    v279_audit = _read_csv(v279_dir / "purchase_audit.csv")

    buy_rows, buy_summary = _build_buy_comparison(v278_audit, v279_audit)
    sell_rows, sell_summary = _build_sell_comparison(v278_trades, v279_trades)
    total_delta = float((v279_summary.get("net_cumulative_profit") or 0.0) - (v278_summary.get("net_cumulative_profit") or 0.0))
    contribution = _build_profit_contribution(v278_trades, v279_trades, v278_audit, v279_audit, total_delta)
    high_pm = {
        "v2_78": _high_pm_stats(v278_trades),
        "v2_79": _high_pm_stats(v279_trades),
    }
    high_pm["delta"] = {
        key: (high_pm["v2_79"].get(key) or 0.0) - (high_pm["v2_78"].get(key) or 0.0)
        for key in ["trade_count", "net_profit", "average_holding_days", "average_buy_amount"]
    }
    blocked_exit_count = int(
        pd.to_numeric(
            _sell_trade_rows(v279_trades).get("high_pm_min_hold_blocked_exit_count", pd.Series(dtype=float)),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    return {
        "purpose": "Portfolio Manager AI Phase 4-D v2_78 vs v2_79 difference audit",
        "period": PERIOD,
        "constraints": {
            "audit_only": True,
            "logic_changed": False,
            "new_profile_added": False,
            "full_backtest_run": False,
            "full_pytest_run": False,
        },
        "profiles": {
            "v2_78": V278,
            "v2_79": V279,
        },
        "profile_summary": {
            "v2_78": _profile_summary(v278_summary),
            "v2_79": _profile_summary(v279_summary),
            "delta": {
                "net_profit": total_delta,
                "profit_factor": (v279_summary.get("profit_factor") or 0.0) - (v278_summary.get("profit_factor") or 0.0),
                "max_drawdown": (v279_summary.get("max_drawdown") or 0.0) - (v278_summary.get("max_drawdown") or 0.0),
                "win_rate": (v279_summary.get("win_rate") or 0.0) - (v278_summary.get("win_rate") or 0.0),
                "trades": (v279_summary.get("closed_trades_count") or 0) - (v278_summary.get("closed_trades_count") or 0),
            },
        },
        "buy_summary": buy_summary,
        "buy_comparison": buy_rows,
        "sell_summary": sell_summary,
        "sell_comparison": sell_rows,
        "exit_reason_counts": {
            "v2_78": _exit_reason_counts(v278_trades),
            "v2_79": _exit_reason_counts(v279_trades),
        },
        "high_pm_summary": high_pm,
        "minimum_hold_effect": {
            "high_pm_min_hold_blocked_exit_count": blocked_exit_count,
            "minimum_hold_activated": blocked_exit_count > 0,
        },
        "profit_contribution": contribution,
        "top_profit_delta_trades": _top_rows(sell_rows, "profit_delta", 30),
    }


def format_markdown(result: dict[str, Any]) -> str:
    profile_rows = [
        {"profile": profile, **values}
        for profile, values in result["profile_summary"].items()
        if profile in {"v2_78", "v2_79"}
    ]
    contribution_rows = [
        {
            "bucket": bucket,
            "profit_delta": value,
            "changed_trade_count": result["profit_contribution"]["changed_trade_counts_by_bucket"].get(bucket, 0),
        }
        for bucket, value in result["profit_contribution"]["buckets"].items()
    ]
    high_pm_rows = [
        {"profile": profile, **values}
        for profile, values in result["high_pm_summary"].items()
        if profile in {"v2_78", "v2_79"}
    ]
    exit_rows = [
        {"profile": profile, **counts}
        for profile, counts in result["exit_reason_counts"].items()
    ]
    buy_only_rows = [
        row for row in result["buy_comparison"] if row["reason"].startswith("only_")
    ][:30]
    top_delta_rows = result["top_profit_delta_trades"][:30]
    lines = [
        "# Portfolio Manager AI Phase 4-D v2_78 vs v2_79 Difference Audit",
        "",
        "## Scope",
        "",
        f"- period: `{result['period']}`",
        "- audit only; no trading logic change; no new profile",
        "- reads existing backtest logs only",
        "",
        "## Profile Summary",
        "",
        _table(profile_rows, ["profile", "net_profit", "profit_factor", "max_drawdown", "win_rate", "trades"]),
        "",
        "## Buy Summary",
        "",
        _table([result["buy_summary"]], ["only_v2_78_buy_count", "only_v2_79_buy_count", "common_buy_count"]),
        "",
        "## Sell Summary",
        "",
        _table(
            [result["sell_summary"]],
            ["sell_date_changed_count", "changed_trade_count", "average_holding_day_delta", "average_profit_delta"],
        ),
        "",
        "## Exit Reason Counts",
        "",
        _table(exit_rows, ["profile", "exit_ai", "stop_loss", "take_profit", "max_holding", "forced_exit", "other"]),
        "",
        "## High PM Summary",
        "",
        _table(
            high_pm_rows,
            ["profile", "trade_count", "net_profit", "profit_factor", "win_rate", "average_holding_days", "average_buy_amount"],
        ),
        "",
        "## Minimum Hold Effect",
        "",
        _table([result["minimum_hold_effect"]], ["high_pm_min_hold_blocked_exit_count", "minimum_hold_activated"]),
        "",
        "## Profit Contribution",
        "",
        f"- method: {result['profit_contribution']['method']}",
        f"- dominant_bucket: `{result['profit_contribution']['dominant_bucket']}`",
        "",
        _table(contribution_rows, ["bucket", "profit_delta", "changed_trade_count"]),
        "",
        "## Buy Diff Sample",
        "",
        _table(
            buy_only_rows,
            [
                "buy_date",
                "code",
                "v2_78_bought",
                "v2_79_bought",
                "pm_score",
                "pm_multiplier",
                "buy_amount_v278",
                "buy_amount_v279",
                "rank",
                "reason",
            ],
        ),
        "",
        "## Top Profit Delta Trades",
        "",
        _table(
            top_delta_rows,
            [
                "code",
                "buy_date",
                "sell_date_v278",
                "sell_date_v279",
                "holding_days_v278",
                "holding_days_v279",
                "realized_profit_v278",
                "realized_profit_v279",
                "profit_delta",
                "match_status",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def save_report(result: dict[str, Any], root: Path = ROOT) -> tuple[Path, Path]:
    report_dir = root / "reports" / "ml"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = report_dir / f"{REPORT_STEM}.md"
    json_path = report_dir / f"{REPORT_STEM}.json"
    markdown.write_text(format_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return markdown, json_path


def main() -> int:
    result = build_report(ROOT)
    markdown, json_path = save_report(result, ROOT)
    print(f"markdown={markdown}")
    print(f"json={json_path}")
    print(
        "summary="
        f"profit_delta={result['profile_summary']['delta']['net_profit']} "
        f"dominant={result['profit_contribution']['dominant_bucket']} "
        f"blocked_exit_count={result['minimum_hold_effect']['high_pm_min_hold_blocked_exit_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
