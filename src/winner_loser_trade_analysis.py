"""Winner/loser trade analysis from existing backtest artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def build_winner_loser_trade_analysis(
    root: Path,
    profile_id: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    log_dir = root / "logs" / "backtests" / profile_id / f"{start_date}_to_{end_date}"
    trades_path = log_dir / "trades.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"trades.csv not found: {trades_path}")

    rows = _read_csv_rows(trades_path)
    trades = [_normalize_trade(row) for row in rows if str(row.get("action") or "").upper() == "SELL"]
    trades = [row for row in trades if row.get("gross_profit") is not None]
    winners = [row for row in trades if float(row.get("gross_profit") or 0) > 0]
    losers = [row for row in trades if float(row.get("gross_profit") or 0) <= 0]

    axes = _axis_definitions()
    axis_analysis = {
        axis_name: _bucket_stats(trades, bucket_func)
        for axis_name, bucket_func in axes.items()
    }
    winner_common = {
        axis_name: _bucket_stats(winners, bucket_func)
        for axis_name, bucket_func in axes.items()
    }
    loser_common = {
        axis_name: _bucket_stats(losers, bucket_func)
        for axis_name, bucket_func in axes.items()
    }
    diff = _difference_rankings(winners, losers, axes)
    contribution = _profit_contribution(trades)
    summary = _summary_stats(trades, winners, losers)
    rule_candidates = _rule_candidates(diff, axis_analysis)

    return {
        "profile_id": profile_id,
        "period": {"start_date": start_date, "end_date": end_date},
        "source": {
            "trades_csv": str(trades_path.relative_to(root)),
            "primary_metric": "gross_profit / gross_profit_rate",
            "note": "run-experiments/backtest/J-Quants access not executed; existing closed SELL trades only",
        },
        "summary": summary,
        "profit_contribution": contribution,
        "winner_common_features": winner_common,
        "loser_common_features": loser_common,
        "all_trade_bucket_analysis": axis_analysis,
        "difference_analysis": diff,
        "rule_candidate_proposal": rule_candidates,
        "top_winners": _top_trades(trades, reverse=True, limit=20),
        "top_losers": _top_trades(trades, reverse=False, limit=20),
        "data_quality": _data_quality(trades, axes),
    }


def render_winner_loser_trade_analysis_markdown(analysis: dict[str, Any]) -> str:
    summary = analysis.get("summary", {})
    contribution = analysis.get("profit_contribution", {})
    diff = analysis.get("difference_analysis", {})
    rules = analysis.get("rule_candidate_proposal", {})
    lines = [
        "# Winner / Loser Trade Analysis",
        "",
        f"- profile_id: {analysis.get('profile_id')}",
        f"- period: {analysis.get('period', {}).get('start_date')} to {analysis.get('period', {}).get('end_date')}",
        f"- source: {analysis.get('source', {}).get('trades_csv')}",
        f"- primary_metric: {analysis.get('source', {}).get('primary_metric')}",
        f"- note: {analysis.get('source', {}).get('note')}",
        "",
        "## Closed Trade Summary",
        "",
        *_summary_lines(summary),
        "",
        "## Profit Contribution Analysis",
        "",
        *_profit_contribution_lines(contribution),
        "",
        "## Winner Common Feature Analysis",
        "",
        *_axis_section_lines(analysis.get("winner_common_features", {}), limit=10),
        "",
        "## Loser Common Feature Analysis",
        "",
        *_axis_section_lines(analysis.get("loser_common_features", {}), limit=10),
        "",
        "## Winner / Loser Difference Ranking",
        "",
        "### Win-Heavy Conditions",
        "",
        *_diff_table_lines(diff.get("win_heavy_conditions", [])[:20]),
        "",
        "### Loss-Heavy Conditions",
        "",
        *_diff_table_lines(diff.get("loss_heavy_conditions", [])[:20]),
        "",
        "## Rule Candidate Proposal",
        "",
        "### Keep Or Boost Candidates",
        "",
        *_rule_table_lines(rules.get("keep_or_boost_candidates", [])),
        "",
        "### Exclude Or Penalize Candidates",
        "",
        *_rule_table_lines(rules.get("exclude_or_penalize_candidates", [])),
        "",
        "## Next A/B Profile Ideas",
        "",
        *_next_ab_lines(rules.get("next_ab_profile_ideas", [])),
        "",
        "## Top Winners",
        "",
        *_trade_table_lines(analysis.get("top_winners", [])),
        "",
        "## Top Losers",
        "",
        *_trade_table_lines(analysis.get("top_losers", [])),
        "",
        "## Data Quality",
        "",
        *_generic_lines(analysis.get("data_quality", {})),
    ]
    return "\n".join(lines) + "\n"


def write_winner_loser_trade_analysis(
    root: Path,
    profile_id: str,
    start_date: str,
    end_date: str,
) -> tuple[Path, Path]:
    analysis = build_winner_loser_trade_analysis(root, profile_id, start_date, end_date)
    out_dir = root / "reports" / profile_id / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "winner_loser_trade_analysis.json"
    md_path = out_dir / "winner_loser_trade_analysis.md"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_winner_loser_trade_analysis_markdown(analysis), encoding="utf-8")
    return md_path, json_path


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _normalize_trade(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for key in [
        "entry_price",
        "exit_price",
        "shares",
        "profit",
        "profit_rate",
        "gross_profit",
        "gross_profit_rate",
        "net_profit",
        "net_profit_rate",
        "rsi",
        "volume_ratio",
        "holding_days",
        "total_score",
        "technical_score",
        "relative_strength_score",
        "relative_strength_5d",
        "relative_strength_10d",
        "relative_strength_20d",
        "market_context_score",
    ]:
        normalized[key] = _number(row.get(key))
    entry_price = normalized.get("entry_price")
    normalized["round_lot_amount"] = round(entry_price * 100, 2) if entry_price is not None else None
    normalized["entry_year"] = str(row.get("entry_date") or "")[:4] or "unknown"
    normalized["entry_month"] = str(row.get("entry_date") or "")[:7] or "unknown"
    normalized["is_win"] = float(normalized.get("gross_profit") or 0) > 0
    return normalized


def _summary_stats(trades: list[dict[str, Any]], winners: list[dict[str, Any]], losers: list[dict[str, Any]]) -> dict[str, Any]:
    profits = [_num(row.get("gross_profit")) for row in trades]
    rates = [_num(row.get("gross_profit_rate")) for row in trades]
    gross_profit_total = round(sum(value for value in profits if value > 0), 2)
    gross_loss_total = round(sum(value for value in profits if value <= 0), 2)
    net_profit_total = round(sum(_num(row.get("net_profit")) for row in trades), 2)
    return {
        "total_trades": len(trades),
        "win_count": len(winners),
        "loss_count": len(losers),
        "win_rate": _ratio(len(winners), len(trades)),
        "gross_profit_total": gross_profit_total,
        "gross_loss_total": gross_loss_total,
        "net_profit_total": net_profit_total,
        "profit_factor": _profit_factor(gross_profit_total, gross_loss_total),
        "average_profit": _average(profits),
        "median_profit": _median(profits),
        "average_profit_rate": _average(rates),
        "median_profit_rate": _median(rates),
    }


def _profit_contribution(trades: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sorted([row for row in trades if _num(row.get("gross_profit")) > 0], key=lambda row: _num(row.get("gross_profit")), reverse=True)
    losses = sorted([row for row in trades if _num(row.get("gross_profit")) < 0], key=lambda row: _num(row.get("gross_profit")))
    gross_profit_total = sum(_num(row.get("gross_profit")) for row in wins)
    out: dict[str, Any] = {
        "gross_profit_total": round(gross_profit_total, 2),
        "top_profit_sums": {},
        "top_profit_share": {},
        "top_loss_sums": {},
        "profit_coverage_trade_count": {},
    }
    for n in [10, 30, 50, 100]:
        profit_sum = round(sum(_num(row.get("gross_profit")) for row in wins[:n]), 2)
        loss_sum = round(sum(_num(row.get("gross_profit")) for row in losses[:n]), 2)
        out["top_profit_sums"][f"top_{n}"] = profit_sum
        out["top_profit_share"][f"top_{n}"] = _ratio(profit_sum, gross_profit_total)
        out["top_loss_sums"][f"bottom_{n}"] = loss_sum
    for threshold in [0.5, 0.7, 0.9]:
        out["profit_coverage_trade_count"][f"{int(threshold * 100)}pct"] = _coverage_count(wins, gross_profit_total * threshold)
    return out


def _bucket_stats(trades: list[dict[str, Any]], bucket_func: Any) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trades:
        grouped[str(bucket_func(row))].append(row)
    rows = []
    for bucket, items in grouped.items():
        profits = [_num(row.get("gross_profit")) for row in items]
        rates = [_num(row.get("gross_profit_rate")) for row in items]
        wins = [row for row in items if _num(row.get("gross_profit")) > 0]
        losses = [row for row in items if _num(row.get("gross_profit")) <= 0]
        gross_profit = sum(value for value in profits if value > 0)
        gross_loss = sum(value for value in profits if value <= 0)
        rows.append(
            {
                "bucket": bucket,
                "count": len(items),
                "win_count": len(wins),
                "loss_count": len(losses),
                "win_rate": _ratio(len(wins), len(items)),
                "gross_profit_total": round(gross_profit, 2),
                "gross_loss_total": round(gross_loss, 2),
                "net_profit_total": round(sum(_num(row.get("net_profit")) for row in items), 2),
                "average_gross_profit": _average(profits),
                "average_gross_profit_rate": _average(rates),
                "profit_factor": _profit_factor(gross_profit, gross_loss),
            }
        )
    return sorted(rows, key=lambda item: (item["count"], item["gross_profit_total"]), reverse=True)


def _difference_rankings(
    winners: list[dict[str, Any]],
    losers: list[dict[str, Any]],
    axes: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    win_total = len(winners) or 1
    loss_total = len(losers) or 1
    rows = []
    for axis_name, bucket_func in axes.items():
        win_groups = _group_raw(winners, bucket_func)
        loss_groups = _group_raw(losers, bucket_func)
        for bucket in sorted(set(win_groups) | set(loss_groups)):
            win_count = len(win_groups.get(bucket, []))
            loss_count = len(loss_groups.get(bucket, []))
            matched = win_count + loss_count
            if matched < 5:
                continue
            win_share = win_count / win_total
            loss_share = loss_count / loss_total
            all_items = win_groups.get(bucket, []) + loss_groups.get(bucket, [])
            rows.append(
                {
                    "axis": axis_name,
                    "bucket": bucket,
                    "win_count": win_count,
                    "loss_count": loss_count,
                    "matched_trade_count": matched,
                    "win_share": round(win_share, 4),
                    "loss_share": round(loss_share, 4),
                    "share_diff": round(win_share - loss_share, 4),
                    "total_gross_profit": round(sum(_num(row.get("gross_profit")) for row in all_items), 2),
                    "average_gross_profit_rate": _average([_num(row.get("gross_profit_rate")) for row in all_items]),
                    "win_rate": _ratio(win_count, matched),
                }
            )
    return {
        "win_heavy_conditions": sorted(rows, key=lambda item: (item["share_diff"], item["total_gross_profit"]), reverse=True),
        "loss_heavy_conditions": sorted(rows, key=lambda item: (item["share_diff"], -item["total_gross_profit"])),
    }


def _rule_candidates(diff: dict[str, list[dict[str, Any]]], axis_analysis: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    actionable_axes = _actionable_rule_axes()
    keep = []
    for row in diff.get("win_heavy_conditions", []):
        if len(keep) >= 8:
            break
        if row["axis"] in actionable_axes and row["matched_trade_count"] >= 10 and row["total_gross_profit"] > 0:
            keep.append(_candidate_rule("keep_or_boost", row))

    exclude = []
    for row in diff.get("loss_heavy_conditions", []):
        if len(exclude) >= 8:
            break
        if row["axis"] in actionable_axes and row["matched_trade_count"] >= 10 and row["total_gross_profit"] < 0:
            exclude.append(_candidate_rule("exclude_or_penalize", row))

    next_ab = []
    for index, row in enumerate(exclude[:2], start=1):
        next_ab.append(
            {
                "profile_idea": f"v2_26_loss_penalty_{index}",
                "condition": row["condition"],
                "change": "除外ではなくscore penaltyから検証",
                "reason": f"loss_shareがwin_shareを上回り、total_gross_profit={row['total_gross_profit']}",
            }
        )
    if keep:
        row = keep[0]
        next_ab.append(
            {
                "profile_idea": "v2_26_winner_boost_1",
                "condition": row["condition"],
                "change": "小幅加点または同点時優先から検証",
                "reason": row["reason"],
            }
        )
    return {
        "keep_or_boost_candidates": keep,
        "exclude_or_penalize_candidates": exclude,
        "next_ab_profile_ideas": next_ab[:3],
        "note": "候補はentry-timeで利用可能な軸のみから作成。profit_rate/holding_daysなど結果ラベルはランキング表示のみでルール候補から除外。",
    }


def _candidate_rule(kind: str, row: dict[str, Any]) -> dict[str, Any]:
    action = "boost" if kind == "keep_or_boost" else "penalize"
    return {
        "rule_name": f"{action}_{row['axis']}_{row['bucket']}".replace(" ", "_").replace("/", "_"),
        "condition": f"{row['axis']} == {row['bucket']}",
        "matched_trade_count": row["matched_trade_count"],
        "win_count": row["win_count"],
        "loss_count": row["loss_count"],
        "win_rate": row["win_rate"],
        "total_gross_profit": row["total_gross_profit"],
        "share_diff": row["share_diff"],
        "priority": "High" if abs(row["share_diff"]) >= 0.05 and row["matched_trade_count"] >= 20 else "Medium",
        "reason": (
            f"win_share={row['win_share']}, loss_share={row['loss_share']}, "
            f"total_gross_profit={row['total_gross_profit']}"
        ),
    }


def _axis_definitions() -> dict[str, Any]:
    return {
        "sector_name": lambda row: row.get("sector_name") or "unknown",
        "rsi_bucket": lambda row: _rsi_bucket(row.get("rsi")),
        "volume_ratio_bucket": lambda row: _volume_bucket(row.get("volume_ratio")),
        "round_lot_amount_bucket": lambda row: _round_lot_bucket(row.get("round_lot_amount")),
        "entry_price_bucket": lambda row: _entry_price_bucket(row.get("entry_price")),
        "holding_days_bucket": lambda row: _holding_bucket(row.get("holding_days")),
        "profit_rate_bucket": lambda row: _profit_rate_bucket(row.get("gross_profit_rate")),
        "entry_month": lambda row: row.get("entry_month") or "unknown",
        "entry_year": lambda row: row.get("entry_year") or "unknown",
        "market_regime": lambda row: row.get("market_regime") or "unknown",
        "relative_strength_score_bucket": lambda row: _score_component_bucket(row.get("relative_strength_score")),
        "relative_strength_5d_bucket": lambda row: _return_bucket(row.get("relative_strength_5d")),
        "relative_strength_10d_bucket": lambda row: _return_bucket(row.get("relative_strength_10d")),
        "relative_strength_20d_bucket": lambda row: _return_bucket(row.get("relative_strength_20d")),
        "market_context_score_bucket": lambda row: _score_component_bucket(row.get("market_context_score")),
        "total_score_bucket": lambda row: _total_score_bucket(row.get("total_score")),
    }


def _actionable_rule_axes() -> set[str]:
    return {
        "sector_name",
        "rsi_bucket",
        "volume_ratio_bucket",
        "round_lot_amount_bucket",
        "entry_price_bucket",
        "entry_month",
        "entry_year",
        "market_regime",
        "relative_strength_score_bucket",
        "relative_strength_5d_bucket",
        "relative_strength_10d_bucket",
        "relative_strength_20d_bucket",
        "market_context_score_bucket",
        "total_score_bucket",
    }


def _group_raw(trades: list[dict[str, Any]], bucket_func: Any) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trades:
        grouped[str(bucket_func(row))].append(row)
    return grouped


def _top_trades(trades: list[dict[str, Any]], *, reverse: bool, limit: int) -> list[dict[str, Any]]:
    rows = sorted(trades, key=lambda row: _num(row.get("gross_profit")), reverse=reverse)[:limit]
    keys = [
        "entry_date",
        "exit_date",
        "code",
        "name",
        "sector_name",
        "entry_price",
        "exit_price",
        "shares",
        "gross_profit",
        "gross_profit_rate",
        "holding_days",
        "exit_reason",
        "total_score",
        "rsi",
        "volume_ratio",
        "market_regime",
    ]
    return [{key: row.get(key) for key in keys} for row in rows]


def _data_quality(trades: list[dict[str, Any]], axes: dict[str, Any]) -> dict[str, Any]:
    missing = {}
    for axis_name, bucket_func in axes.items():
        missing[axis_name] = sum(1 for row in trades if str(bucket_func(row)) == "unknown")
    return {
        "closed_sell_trade_count": len(trades),
        "unknown_bucket_counts": missing,
        "future_data_note": "entry-time trade columns only; post-exit profit is used solely as outcome label",
    }


def _coverage_count(rows: list[dict[str, Any]], target: float) -> int | None:
    if target <= 0:
        return None
    total = 0.0
    for index, row in enumerate(rows, start=1):
        total += _num(row.get("gross_profit"))
        if total >= target:
            return index
    return None


def _rsi_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 40:
        return "<40"
    if value < 50:
        return "40-50"
    if value < 60:
        return "50-60"
    if value < 70:
        return "60-70"
    return "70+"


def _volume_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 2:
        return "<2"
    if value < 3:
        return "2-3"
    if value < 5:
        return "3-5"
    if value < 8:
        return "5-8"
    return "8+"


def _round_lot_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value <= 300000:
        return "<=300k"
    if value <= 400000:
        return "300k-400k"
    if value <= 500000:
        return "400k-500k"
    if value <= 700000:
        return "500k-700k"
    return "700k+"


def _entry_price_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value <= 1000:
        return "<=1000"
    if value <= 2000:
        return "1000-2000"
    if value <= 3000:
        return "2000-3000"
    if value <= 5000:
        return "3000-5000"
    return "5000+"


def _holding_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value <= 1:
        return "1 day"
    if value == 2:
        return "2 days"
    if value == 3:
        return "3 days"
    if value <= 5:
        return "4-5 days"
    if value <= 10:
        return "6-10 days"
    return "11+ days"


def _profit_rate_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value <= -0.10:
        return "<= -10%"
    if value <= -0.05:
        return "-10% to -5%"
    if value <= -0.03:
        return "-5% to -3%"
    if value <= -0.01:
        return "-3% to -1%"
    if value <= 0:
        return "-1% to 0%"
    if value < 0.01:
        return "0% to 1%"
    if value < 0.03:
        return "1% to 3%"
    if value < 0.05:
        return "3% to 5%"
    if value < 0.10:
        return "5% to 10%"
    return "10%+"


def _return_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < -0.05:
        return "<-5%"
    if value < 0:
        return "-5% to 0%"
    if value < 0.03:
        return "0% to 3%"
    if value < 0.05:
        return "3% to 5%"
    return "5%+"


def _score_component_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 0:
        return "<0"
    if value == 0:
        return "0"
    if value <= 3:
        return "1-3"
    if value <= 6:
        return "4-6"
    return "7+"


def _total_score_bucket(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "unknown"
    if value < 45:
        return "<45"
    if value < 50:
        return "45-49"
    if value < 55:
        return "50-54"
    if value < 60:
        return "55-59"
    if value < 65:
        return "60-64"
    return "65+"


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> float:
    number = _number(value)
    return float(number or 0.0)


def _average(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return round(sum(values) / len(values), 6) if values else None


def _median(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return round(statistics.median(values), 6) if values else None


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _profit_factor(gross_profit: float, gross_loss: float) -> float | None:
    if gross_loss == 0:
        return None
    return round(gross_profit / abs(gross_loss), 6)


def _summary_lines(summary: dict[str, Any]) -> list[str]:
    return [f"- {key}: {_fmt(value)}" for key, value in summary.items()]


def _profit_contribution_lines(contribution: dict[str, Any]) -> list[str]:
    lines = [f"- gross_profit_total: {_fmt(contribution.get('gross_profit_total'))}"]
    for key in ["top_profit_sums", "top_profit_share", "top_loss_sums", "profit_coverage_trade_count"]:
        lines.append(f"- {key}: {json.dumps(contribution.get(key, {}), ensure_ascii=False)}")
    return lines


def _axis_section_lines(axis_data: dict[str, list[dict[str, Any]]], limit: int) -> list[str]:
    lines = []
    for axis_name, rows in axis_data.items():
        lines.extend([f"### {axis_name}", "", *_bucket_table_lines(rows[:limit]), ""])
    return lines


def _bucket_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- no data"]
    lines = ["| bucket | count | win_rate | gross_profit | gross_loss | avg_rate | PF |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row.get('bucket')} | {row.get('count')} | {_fmt(row.get('win_rate'))} | "
            f"{_fmt(row.get('gross_profit_total'))} | {_fmt(row.get('gross_loss_total'))} | "
            f"{_fmt(row.get('average_gross_profit_rate'))} | {_fmt(row.get('profit_factor'))} |"
        )
    return lines


def _diff_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- no data"]
    lines = ["| axis | bucket | trades | win | loss | share_diff | total_profit | win_rate |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row.get('axis')} | {row.get('bucket')} | {row.get('matched_trade_count')} | "
            f"{row.get('win_count')} | {row.get('loss_count')} | {_fmt(row.get('share_diff'))} | "
            f"{_fmt(row.get('total_gross_profit'))} | {_fmt(row.get('win_rate'))} |"
        )
    return lines


def _rule_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- no candidate"]
    lines = ["| rule_name | condition | trades | win_rate | total_profit | priority | reason |", "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for row in rows:
        lines.append(
            f"| {row.get('rule_name')} | {row.get('condition')} | {row.get('matched_trade_count')} | "
            f"{_fmt(row.get('win_rate'))} | {_fmt(row.get('total_gross_profit'))} | {row.get('priority')} | {row.get('reason')} |"
        )
    return lines


def _next_ab_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- no candidate"]
    return [f"- {row.get('profile_idea')}: {row.get('condition')} / {row.get('change')} / {row.get('reason')}" for row in rows]


def _trade_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- no data"]
    lines = ["| entry | exit | code | name | sector | gross_profit | gross_rate | days | score | rsi | volume | reason |", "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(
            f"| {row.get('entry_date')} | {row.get('exit_date')} | {row.get('code')} | {row.get('name')} | "
            f"{row.get('sector_name')} | {_fmt(row.get('gross_profit'))} | {_fmt(row.get('gross_profit_rate'))} | "
            f"{_fmt(row.get('holding_days'))} | {_fmt(row.get('total_score'))} | {_fmt(row.get('rsi'))} | "
            f"{_fmt(row.get('volume_ratio'))} | {row.get('exit_reason')} |"
        )
    return lines


def _generic_lines(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [f"- {key}: {json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else item}" for key, item in value.items()]
    return [f"- {value}"]


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate winner/loser trade analysis from existing artifacts.")
    parser.add_argument("--profile-id", default="rookie_dealer_02_v2_26")
    parser.add_argument("--start-date", default="2021-06-01")
    parser.add_argument("--end-date", default="2026-05-29")
    args = parser.parse_args()
    md_path, json_path = write_winner_loser_trade_analysis(ROOT, args.profile_id, args.start_date, args.end_date)
    print(f"markdown: {md_path.relative_to(ROOT)}")
    print(f"json: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
