"""Market regime analysis from existing backtest artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from market_regime import classification_definition, classify_market_context


ROOT = Path(__file__).resolve().parents[1]
INITIAL_CAPITAL_DEFAULT = 1_000_000.0


def build_market_regime_analysis(root: Path, profile_id: str, start_date: str, end_date: str) -> dict[str, Any]:
    log_dir = root / "logs" / "backtests" / profile_id / f"{start_date}_to_{end_date}"
    trades_path = log_dir / "trades.csv"
    summary_path = log_dir / "summary.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"trades.csv not found: {trades_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    trades = [_normalize_trade(row) for row in _read_csv_rows(trades_path) if str(row.get("action") or "").upper() == "SELL"]
    summary_rows = [_normalize_summary(row) for row in _read_csv_rows(summary_path)]
    contexts = _load_market_contexts(root, start_date, end_date)
    regime_by_date = {date: classify_market_context(context) for date, context in contexts.items()}

    for trade in trades:
        trade["classified_regime"] = regime_by_date.get(str(trade.get("entry_date") or ""), _fallback_trade_regime(trade))
    for row in summary_rows:
        row["classified_regime"] = regime_by_date.get(str(row.get("date") or ""), "unknown")

    regime_order = ["strong_bull", "bull", "range", "bear", "strong_bear", "unknown"]
    trade_stats = _trade_stats_by_regime(trades, regime_order)
    capital_stats = _capital_stats_by_regime(summary_rows, regime_order)
    simulations = _exposure_simulations(summary_rows, regime_by_date)
    recommendations = _recommendations(trade_stats, capital_stats, simulations)

    return {
        "profile_id": profile_id,
        "period": {"start_date": start_date, "end_date": end_date},
        "source": {
            "trades_csv": str(trades_path.relative_to(root)),
            "summary_csv": str(summary_path.relative_to(root)),
            "market_context_dir": "data/processed/market_context_YYYY-MM-DD.json",
            "note": "run-experiments/backtest/J-Quants access not executed; existing artifacts only",
        },
        "classification_definition": _classification_definition(),
        "market_context_coverage": {
            "summary_day_count": len(summary_rows),
            "market_context_file_count_in_period": len(contexts),
            "classified_day_count": sum(1 for row in summary_rows if row.get("classified_regime") != "unknown"),
            "unknown_day_count": sum(1 for row in summary_rows if row.get("classified_regime") == "unknown"),
        },
        "regime_day_count": dict(_count_by(row.get("classified_regime") for row in summary_rows)),
        "trade_performance_by_regime": trade_stats,
        "capital_utilization_by_regime": capital_stats,
        "exposure_improvement_simulation": simulations,
        "improvement_candidates": recommendations,
    }


def render_market_regime_analysis_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Market Regime Analysis",
        "",
        f"- profile_id: {analysis.get('profile_id')}",
        f"- period: {analysis.get('period', {}).get('start_date')} to {analysis.get('period', {}).get('end_date')}",
        f"- source: {analysis.get('source', {}).get('trades_csv')}",
        f"- note: {analysis.get('source', {}).get('note')}",
        "",
        "## 1. 市場局面分類",
        "",
        *_classification_lines(analysis.get("classification_definition", {})),
        "",
        "### Coverage",
        "",
        *_generic_lines(analysis.get("market_context_coverage", {})),
        "",
        "### Regime Day Count",
        "",
        *_generic_lines(analysis.get("regime_day_count", {})),
        "",
        "## 2. 局面別トレード成績",
        "",
        *_trade_regime_table(analysis.get("trade_performance_by_regime", {})),
        "",
        "## 3. 局面別資金利用率",
        "",
        *_capital_regime_table(analysis.get("capital_utilization_by_regime", {})),
        "",
        "## 4. 仮想エクスポージャー改善シミュレーション",
        "",
        *_simulation_lines(analysis.get("exposure_improvement_simulation", {})),
        "",
        "## 5. 次の改善候補提案",
        "",
        *_recommendation_lines(analysis.get("improvement_candidates", {})),
    ]
    return "\n".join(lines) + "\n"


def write_market_regime_analysis(root: Path, profile_id: str, start_date: str, end_date: str) -> tuple[Path, Path]:
    analysis = build_market_regime_analysis(root, profile_id, start_date, end_date)
    out_dir = root / "reports" / profile_id / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "market_regime_analysis.json"
    md_path = out_dir / "market_regime_analysis.md"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_market_regime_analysis_markdown(analysis), encoding="utf-8")
    return md_path, json_path


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _normalize_trade(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ["gross_profit", "gross_profit_rate", "net_profit", "net_profit_rate", "holding_days"]:
        out[key] = _number(row.get(key))
    return out


def _normalize_summary(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ["cash", "positions_value", "total_assets", "daily_profit", "max_drawdown", "open_positions_count"]:
        out[key] = _number(row.get(key))
    total_assets = out.get("total_assets") or 0.0
    out["cash_ratio"] = (out.get("cash") or 0.0) / total_assets if total_assets else None
    out["market_exposure"] = (out.get("positions_value") or 0.0) / total_assets if total_assets else None
    return out


def _load_market_contexts(root: Path, start_date: str, end_date: str) -> dict[str, dict[str, Any]]:
    contexts = {}
    for path in sorted((root / "data" / "processed").glob("market_context_*.json")):
        date = path.stem.replace("market_context_", "")
        if date < start_date or date > end_date:
            continue
        try:
            contexts[date] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return contexts


def _classify_market_regime(context: dict[str, Any]) -> str:
    return classify_market_context(context)


def _fallback_trade_regime(trade: dict[str, Any]) -> str:
    base = str(trade.get("market_regime") or "")
    return {"risk_on": "bull", "neutral": "range", "risk_off": "bear"}.get(base, "unknown")


def _classification_definition() -> dict[str, Any]:
    definition = classification_definition()
    definition["data_limitations"] = "TOPIX/Nikkei values were mostly null in existing market_context files; no new data was fetched."
    return definition


def _trade_stats_by_regime(trades: list[dict[str, Any]], regime_order: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade.get("classified_regime") or "unknown")].append(trade)
    return {regime: _trade_stats(grouped.get(regime, [])) for regime in regime_order if grouped.get(regime)}


def _trade_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    profits = [_num(row.get("gross_profit")) for row in trades]
    gross_profit = sum(value for value in profits if value > 0)
    gross_loss = sum(value for value in profits if value <= 0)
    wins = sum(1 for value in profits if value > 0)
    return {
        "trade_count": len(trades),
        "win_rate": _ratio(wins, len(trades)),
        "profit_factor": _profit_factor(gross_profit, gross_loss),
        "avg_profit_rate": _average([_num(row.get("gross_profit_rate")) for row in trades]),
        "total_profit": round(sum(profits), 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "max_drawdown": _max_trade_drawdown(profits),
    }


def _capital_stats_by_regime(summary_rows: list[dict[str, Any]], regime_order: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row.get("classified_regime") or "unknown")].append(row)
    return {regime: _capital_stats(grouped.get(regime, [])) for regime in regime_order if grouped.get(regime)}


def _capital_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "day_count": len(rows),
        "average_cash_ratio": _average([_num_or_none(row.get("cash_ratio")) for row in rows]),
        "average_market_exposure": _average([_num_or_none(row.get("market_exposure")) for row in rows]),
        "average_position_count": _average([_num_or_none(row.get("open_positions_count")) for row in rows]),
        "total_daily_profit": round(sum(_num(row.get("daily_profit")) for row in rows), 2),
    }


def _exposure_simulations(summary_rows: list[dict[str, Any]], regime_by_date: dict[str, str]) -> dict[str, Any]:
    cases = {
        "case_a": {
            "description": "strong_bull target_exposure=0.95, others current exposure",
            "targets": {"strong_bull": 0.95},
        },
        "case_b": {
            "description": "strong_bull=1.00, range=0.70, bear/strong_bear=0.40",
            "targets": {"strong_bull": 1.00, "range": 0.70, "bear": 0.40, "strong_bear": 0.40},
        },
        "case_c": {
            "description": "strong_bull=1.00, strong_bear=0.00, others current exposure",
            "targets": {"strong_bull": 1.00, "strong_bear": 0.00},
        },
    }
    baseline = _simulate_case(summary_rows, regime_by_date, {})
    return {
        "method": (
            "既存summary.csvの日次損益を market_exposure に比例して拡大/縮小する概算。"
            "market_exposure=0の日は追加投資機会を推定できないため損益0倍扱い。"
        ),
        "baseline": baseline,
        **{name: _simulation_with_delta(summary_rows, regime_by_date, payload, baseline) for name, payload in cases.items()},
    }


def _simulation_with_delta(
    summary_rows: list[dict[str, Any]],
    regime_by_date: dict[str, str],
    payload: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    result = _simulate_case(summary_rows, regime_by_date, payload["targets"])
    result["description"] = payload["description"]
    result["target_exposure_by_regime"] = payload["targets"]
    result["estimated_profit_improvement"] = round(result["estimated_total_profit"] - baseline["estimated_total_profit"], 2)
    result["estimated_dd_change"] = round(result["estimated_max_drawdown"] - baseline["estimated_max_drawdown"], 6)
    result["estimated_exposure_change"] = round(result["average_simulated_exposure"] - baseline["average_simulated_exposure"], 6)
    return result


def _simulate_case(summary_rows: list[dict[str, Any]], regime_by_date: dict[str, str], targets: dict[str, float]) -> dict[str, Any]:
    equity = INITIAL_CAPITAL_DEFAULT
    peak = equity
    max_dd = 0.0
    total_profit = 0.0
    simulated_exposures = []
    scaling_by_regime: dict[str, list[float]] = defaultdict(list)
    for row in summary_rows:
        regime = regime_by_date.get(str(row.get("date") or ""), str(row.get("classified_regime") or "unknown"))
        exposure = _num(row.get("market_exposure"))
        target = targets.get(regime, exposure)
        if exposure <= 0:
            scale = 0.0 if target != exposure else 1.0
            simulated_exposure = exposure
        else:
            scale = target / exposure
            simulated_exposure = target
        daily_profit = _num(row.get("daily_profit")) * scale
        total_profit += daily_profit
        equity += daily_profit
        peak = max(peak, equity)
        if peak:
            max_dd = min(max_dd, (equity - peak) / peak)
        simulated_exposures.append(simulated_exposure)
        scaling_by_regime[regime].append(scale)
    return {
        "estimated_total_profit": round(total_profit, 2),
        "estimated_final_assets": round(INITIAL_CAPITAL_DEFAULT + total_profit, 2),
        "estimated_max_drawdown": round(max_dd, 6),
        "average_simulated_exposure": _average(simulated_exposures),
        "average_scaling_factor_by_regime": {
            regime: _average(values)
            for regime, values in sorted(scaling_by_regime.items())
            if values
        },
    }


def _recommendations(
    trade_stats: dict[str, Any],
    capital_stats: dict[str, Any],
    simulations: dict[str, Any],
) -> dict[str, Any]:
    ranked_regimes = sorted(
        [
            {
                "regime": regime,
                "trade_count": stats.get("trade_count", 0),
                "profit_factor": stats.get("profit_factor"),
                "total_profit": stats.get("total_profit"),
                "average_market_exposure": (capital_stats.get(regime, {}) or {}).get("average_market_exposure"),
            }
            for regime, stats in trade_stats.items()
        ],
        key=lambda row: (_num(row.get("total_profit")), _num(row.get("profit_factor"))),
        reverse=True,
    )
    best_case_name, best_case = max(
        ((name, payload) for name, payload in simulations.items() if name.startswith("case_")),
        key=lambda item: _num(item[1].get("estimated_profit_improvement")),
    )
    return {
        "regime_profit_ranking": ranked_regimes,
        "best_simulation_case": best_case_name,
        "best_simulation_summary": best_case,
        "next_profile_candidates": [
            {
                "profile_idea": "v2_26_dynamic_exposure_strong_bull_95",
                "rule": "strong_bull の target_exposure を0.95へ引き上げ、他局面は現状維持",
                "reason": "Case Aの概算で、強い上昇局面だけ資金投入を増やす単純な検証",
            },
            {
                "profile_idea": "v2_26_dynamic_exposure_bull_bear_control",
                "rule": "strong_bull=1.0 / range=0.7 / bear=0.4 / strong_bear=0.0",
                "reason": "Case B/Cを組み合わせ、資金投入とDD抑制を同時に検証",
            },
        ],
        "recommended_target_exposure": {
            "strong_bull": 0.95,
            "bull": "current_or_0.90",
            "range": 0.70,
            "bear": 0.40,
            "strong_bear": 0.0,
        },
        "recommended_cash_ratio": {
            "strong_bull": 0.05,
            "range": 0.30,
            "bear": 0.60,
            "strong_bear": 1.00,
        },
        "dynamic_exposure_effectiveness": (
            "既存成果物ベースの概算では、銘柄選定微調整よりも、局面ごとの投入率変更をA/Bする価値があります。"
            "ただしmarket_exposure=0の日の機会創出は推定不能なので、実際のbacktestで検証が必要です。"
        ),
    }


def _max_trade_drawdown(profits: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for profit in profits:
        equity += profit
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return round(max_dd, 2)


def _count_by(values: Any) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value or "unknown")] += 1
    return dict(counts)


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> float:
    return float(_number(value) or 0.0)


def _num_or_none(value: Any) -> float | None:
    return _number(value)


def _average(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _profit_factor(gross_profit: float, gross_loss: float) -> float | None:
    if gross_loss == 0:
        return None
    return round(gross_profit / abs(gross_loss), 6)


def _classification_lines(definition: dict[str, Any]) -> list[str]:
    return [f"- {key}: {value}" for key, value in definition.items()]


def _generic_lines(payload: dict[str, Any]) -> list[str]:
    return [f"- {key}: {value}" for key, value in payload.items()]


def _trade_regime_table(payload: dict[str, Any]) -> list[str]:
    lines = [
        "| regime | trade_count | win_rate | profit_factor | avg_profit_rate | total_profit | max_drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for regime, row in payload.items():
        lines.append(
            f"| {regime} | {row.get('trade_count')} | {_fmt(row.get('win_rate'))} | {_fmt(row.get('profit_factor'))} | "
            f"{_fmt(row.get('avg_profit_rate'))} | {_fmt(row.get('total_profit'))} | {_fmt(row.get('max_drawdown'))} |"
        )
    return lines


def _capital_regime_table(payload: dict[str, Any]) -> list[str]:
    lines = [
        "| regime | day_count | average_cash_ratio | average_market_exposure | average_position_count | total_daily_profit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for regime, row in payload.items():
        lines.append(
            f"| {regime} | {row.get('day_count')} | {_fmt(row.get('average_cash_ratio'))} | "
            f"{_fmt(row.get('average_market_exposure'))} | {_fmt(row.get('average_position_count'))} | {_fmt(row.get('total_daily_profit'))} |"
        )
    return lines


def _simulation_lines(payload: dict[str, Any]) -> list[str]:
    lines = [f"- method: {payload.get('method')}", ""]
    for name, row in payload.items():
        if name == "method":
            continue
        lines.extend([f"### {name}", "", *_generic_lines(row), ""])
    return lines


def _recommendation_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        f"- best_simulation_case: {payload.get('best_simulation_case')}",
        f"- dynamic_exposure_effectiveness: {payload.get('dynamic_exposure_effectiveness')}",
        f"- recommended_target_exposure: {json.dumps(payload.get('recommended_target_exposure', {}), ensure_ascii=False)}",
        f"- recommended_cash_ratio: {json.dumps(payload.get('recommended_cash_ratio', {}), ensure_ascii=False)}",
        "",
        "### Next Profile Candidates",
        "",
    ]
    for row in payload.get("next_profile_candidates", []):
        lines.append(f"- {row.get('profile_idea')}: {row.get('rule')} / {row.get('reason')}")
    lines.extend(["", "### Regime Profit Ranking", "", *_trade_regime_table({row["regime"]: row for row in payload.get("regime_profit_ranking", [])})])
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate market regime analysis from existing artifacts.")
    parser.add_argument("--profile-id", default="rookie_dealer_02_v2_26")
    parser.add_argument("--start-date", default="2021-06-01")
    parser.add_argument("--end-date", default="2026-05-29")
    args = parser.parse_args()
    md_path, json_path = write_market_regime_analysis(ROOT, args.profile_id, args.start_date, args.end_date)
    print(f"markdown: {md_path.relative_to(ROOT)}")
    print(f"json: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
