"""Phase 6-A market regime audit for the v2_78 baseline.

This module is read-only. It uses existing backtest logs, local TOPIX cache,
and local market_context files to audit where the current baseline is strong
or weak by market regime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.data_loader import JQuantsDataLoader


ROOT = Path(__file__).resolve().parents[2]
PERIOD = "2023-01-01_to_2026-05-31"
BASE_PROFILE = "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
REPORT_STEM = "phase6a_market_regime_audit_2023-01_to_2026-05"
REGIME_ORDER = ["Bull", "Neutral", "Bear", "Unknown"]
PM_BUCKETS = [1.30, 1.15, 1.00, 0.80]


@dataclass(frozen=True)
class Phase6AReportPaths:
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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_float(value: Any) -> float | None:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _sum(series: pd.Series | None) -> float:
    if series is None:
        return 0.0
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.sum()) if not values.empty else 0.0


def _mean(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _profit_factor(profits: pd.Series | None) -> float | None:
    if profits is None:
        return None
    values = pd.to_numeric(profits, errors="coerce").dropna()
    if values.empty:
        return None
    gross_profit = float(values[values > 0].sum())
    gross_loss = abs(float(values[values < 0].sum()))
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def _win_rate(profits: pd.Series | None) -> float | None:
    if profits is None:
        return None
    values = pd.to_numeric(profits, errors="coerce").dropna()
    if values.empty:
        return None
    return float((values > 0).mean())


def _drawdown_from_assets(assets: pd.Series | None) -> float | None:
    if assets is None:
        return None
    values = pd.to_numeric(assets, errors="coerce").dropna()
    if values.empty:
        return None
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    return float(drawdown.min())


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class Phase6AMarketRegimeAudit:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        profile: str = BASE_PROFILE,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.profile = profile
        self.period = period
        self.start_date, self.end_date = period.split("_to_")

    def build_report(self) -> dict[str, Any]:
        summary = self._load_summary()
        trades = self._load_trades()
        purchases = self._load_purchase_audit()
        topix = self._load_topix()
        breadth = self._load_market_breadth(summary)
        regime = self._build_regime_frame(topix, breadth)
        summary_regime = self._attach_summary_regime(summary, regime)
        trades_regime = self._attach_trade_regime(trades, regime)
        regime_summary = self._regime_summary(summary_regime, trades_regime, purchases)
        yearly = self._yearly_summary(summary_regime, trades_regime)
        dd = self._drawdown_analysis(summary_regime)
        high_pm = self._high_pm_summary(trades_regime)
        virtual = self._virtual_strategy_audit(trades_regime)
        contribution = self._contribution_summary(regime_summary)
        verdict = self._verdict(regime_summary, virtual, dd)
        return {
            "metadata": {
                "phase": "6-A",
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
                "summary_csv": str(self._log_dir() / "summary.csv"),
                "trades_csv": str(self._log_dir() / "trades.csv"),
                "purchase_audit_csv": str(self._log_dir() / "purchase_audit.csv"),
                "topix_cache": "data/cache/jquants/topix_prices/*.json",
                "market_breadth": "data/processed/market_context_YYYY-MM-DD.json",
            },
            "regime_definition": {
                "Bull": "TOPIX close > MA75 and TOPIX MA25 > MA75",
                "Bear": "TOPIX close < MA75 and TOPIX MA25 < MA75",
                "Neutral": "all other TOPIX rows",
                "Unknown": "TOPIX row unavailable",
                "notes": [
                    "TOPIX 20d return and TOPIX drawdown are diagnostic fields, not classification thresholds.",
                    "market breadth is auxiliary only and is not required for classification.",
                ],
            },
            "coverage": self._coverage(summary, trades, regime),
            "regime_by_day": regime_summary,
            "bull_bear_contribution": contribution,
            "drawdown_analysis": dd,
            "high_pm_by_regime": high_pm,
            "virtual_strategy_audit": virtual,
            "yearly_summary": yearly,
            "verdict": verdict,
        }

    def save_report(self, result: dict[str, Any]) -> Phase6AReportPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase6AReportPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 6-A Market Regime Audit",
            "",
            "## Scope",
            "",
            "- audit only",
            "- no logic change, no new profile, no backtest execution",
            "- existing logs and local market cache only",
            "",
            "## Regime Definition",
            "",
            self._table([result["regime_definition"]], ["Bull", "Neutral", "Bear", "Unknown"]),
            "",
            "## Coverage",
            "",
            self._table([result["coverage"]], ["summary_days", "trades", "topix_days", "classified_summary_days", "classified_trade_count", "market_breadth_days"]),
            "",
            "## Regime Summary",
            "",
            self._table(result["regime_by_day"], ["regime", "days", "trades", "net_profit", "profit_factor", "max_drawdown", "win_rate", "average_holding_days", "average_capital_utilization", "average_position_count", "average_buy_amount", "monthly_win_rate"]),
            "",
            "## Bull / Bear Contribution",
            "",
            self._table(result["bull_bear_contribution"], ["regime", "profit_contribution", "pf_contribution", "trade_contribution", "monthly_win_rate"]),
            "",
            "## Drawdown Analysis",
            "",
            self._table([result["drawdown_analysis"]], ["max_drawdown", "dd_start", "dd_trough", "dd_recovery", "regime_at_dd"]),
            "",
            "## High PM By Regime",
            "",
            self._table(result["high_pm_by_regime"], ["regime", "pm_multiplier", "trade_count", "profit", "profit_factor", "win_rate"]),
            "",
            "## Virtual Strategy Audit",
            "",
            self._table(result["virtual_strategy_audit"], ["rule", "trades", "net_profit", "profit_delta_vs_actual", "profit_factor", "win_rate", "note"]),
            "",
            "## Yearly Summary",
            "",
            self._table(result["yearly_summary"], ["year", "profit", "return", "profit_factor", "max_drawdown", "win_rate"]),
            "",
            "## Verdict",
            "",
            self._table([result["verdict"]], ["regime_effect_detected", "bull_market_advantage", "bear_market_weakness", "regime_filter_worth_implementing", "recommended_next_phase"]),
            "",
        ]
        return "\n".join(lines)

    def _log_dir(self) -> Path:
        return self.root / "logs" / "backtests" / self.profile / self.period

    def _load_summary(self) -> pd.DataFrame:
        frame = _read_csv(self._log_dir() / "summary.csv")
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return frame

    def _load_trades(self) -> pd.DataFrame:
        frame = _read_csv(self._log_dir() / "trades.csv")
        if frame.empty:
            return frame
        frame = frame.copy()
        if "action" in frame.columns:
            frame = frame[frame["action"].fillna("").astype(str).eq("SELL")].copy()
        for column in ["entry_date", "exit_date"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")
        return frame

    def _load_purchase_audit(self) -> pd.DataFrame:
        frame = _read_csv(self._log_dir() / "purchase_audit.csv")
        if frame.empty:
            return frame
        frame = frame.copy()
        for column in ["entry_date", "signal_date"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")
        return frame

    def _load_topix(self) -> pd.DataFrame:
        lookback_start = (pd.Timestamp(self.start_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        loader = JQuantsDataLoader(self.root / "data" / "cache" / "jquants")
        topix = loader.load_topix(lookback_start, self.end_date)
        if topix.empty:
            return topix
        topix = topix.copy().dropna(subset=["date", "close"]).sort_values("date")
        topix["date"] = pd.to_datetime(topix["date"], errors="coerce")
        topix["close"] = pd.to_numeric(topix["close"], errors="coerce")
        topix["ma25"] = topix["close"].rolling(25, min_periods=1).mean()
        topix["ma75"] = topix["close"].rolling(75, min_periods=1).mean()
        topix["return_20d"] = topix["close"] / topix["close"].shift(20) - 1.0
        topix["drawdown"] = topix["close"] / topix["close"].cummax() - 1.0
        topix["regime"] = topix.apply(self._classify_regime, axis=1)
        topix["date"] = topix["date"].dt.strftime("%Y-%m-%d")
        return topix[topix["date"].between(self.start_date, self.end_date)].reset_index(drop=True)

    def _load_market_breadth(self, summary: pd.DataFrame) -> pd.DataFrame:
        dates = summary.get("date", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if not summary.empty else []
        rows = []
        for day in dates:
            payload = _read_json(self.root / "data" / "processed" / f"market_context_{day}.json")
            if not payload:
                continue
            rows.append(
                {
                    "date": day,
                    "advance_ratio": _to_float(payload.get("advance_ratio")),
                    "average_change_rate": _to_float(payload.get("average_change_rate")),
                    "market_regime_raw": payload.get("market_regime"),
                }
            )
        return pd.DataFrame(rows)

    def _classify_regime(self, row: pd.Series) -> str:
        close = _to_float(row.get("close"))
        ma25 = _to_float(row.get("ma25"))
        ma75 = _to_float(row.get("ma75"))
        if close is None or ma25 is None or ma75 is None:
            return "Unknown"
        if close > ma75 and ma25 > ma75:
            return "Bull"
        if close < ma75 and ma25 < ma75:
            return "Bear"
        return "Neutral"

    def _build_regime_frame(self, topix: pd.DataFrame, breadth: pd.DataFrame) -> pd.DataFrame:
        if topix.empty:
            return pd.DataFrame(columns=["date", "regime"])
        columns = ["date", "close", "ma25", "ma75", "return_20d", "drawdown", "regime"]
        regime = topix[columns].copy()
        if not breadth.empty:
            regime = regime.merge(breadth, on="date", how="left")
        return regime

    def _attach_summary_regime(self, summary: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
        if summary.empty:
            return summary
        merged = summary.merge(regime, on="date", how="left", suffixes=("", "_topix"))
        merged["regime"] = merged["regime"].fillna("Unknown")
        return merged

    def _attach_trade_regime(self, trades: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        out = trades.copy()
        regime_map = regime.set_index("date")["regime"].to_dict() if not regime.empty and "date" in regime.columns else {}
        out["regime"] = out.get("entry_date", pd.Series("", index=out.index)).map(regime_map).fillna("Unknown")
        return out

    def _coverage(self, summary: pd.DataFrame, trades: pd.DataFrame, regime: pd.DataFrame) -> dict[str, Any]:
        summary_days = len(summary)
        classified_summary_days = 0
        if not summary.empty and not regime.empty:
            classified_summary_days = int(summary["date"].isin(set(regime["date"])).sum())
        classified_trade_count = int(trades.get("entry_date", pd.Series(dtype=str)).isin(set(regime.get("date", pd.Series(dtype=str)))).sum()) if not trades.empty else 0
        return {
            "summary_days": summary_days,
            "trades": int(len(trades)),
            "topix_days": int(len(regime)),
            "classified_summary_days": classified_summary_days,
            "classified_trade_count": classified_trade_count,
            "market_breadth_days": int(regime.get("advance_ratio", pd.Series(dtype=float)).notna().sum()) if not regime.empty else 0,
        }

    def _regime_summary(self, summary: pd.DataFrame, trades: pd.DataFrame, purchases: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for regime in REGIME_ORDER:
            day_rows = summary[summary.get("regime", pd.Series(dtype=str)).eq(regime)] if not summary.empty else pd.DataFrame()
            trade_rows = trades[trades.get("regime", pd.Series(dtype=str)).eq(regime)] if not trades.empty else pd.DataFrame()
            buy_amount = self._buy_amount_by_regime(purchases, regime, summary)
            profits = pd.to_numeric(trade_rows.get("net_profit"), errors="coerce") if not trade_rows.empty else pd.Series(dtype=float)
            rows.append(
                {
                    "regime": regime,
                    "days": int(len(day_rows)),
                    "trades": int(len(trade_rows)),
                    "net_profit": float(profits.sum()) if not profits.empty else 0.0,
                    "profit_factor": _profit_factor(profits),
                    "max_drawdown": _drawdown_from_assets(day_rows.get("total_assets")) if not day_rows.empty else None,
                    "win_rate": _win_rate(profits),
                    "average_holding_days": _mean(trade_rows.get("holding_days")) if not trade_rows.empty else None,
                    "average_capital_utilization": _mean(pd.to_numeric(day_rows.get("positions_value"), errors="coerce") / pd.to_numeric(day_rows.get("total_assets"), errors="coerce").replace(0, pd.NA)) if not day_rows.empty else None,
                    "average_position_count": _mean(day_rows.get("open_positions_count")) if not day_rows.empty else None,
                    "average_buy_amount": buy_amount,
                    "monthly_win_rate": self._monthly_win_rate(trade_rows),
                }
            )
        return rows

    def _buy_amount_by_regime(self, purchases: pd.DataFrame, regime: str, summary: pd.DataFrame) -> float | None:
        if purchases.empty:
            return None
        date_column = "entry_date" if "entry_date" in purchases.columns else "signal_date"
        if date_column not in purchases.columns:
            return None
        regime_map = summary.set_index("date")["regime"].to_dict() if not summary.empty and "regime" in summary.columns else {}
        rows = purchases.copy()
        rows["regime"] = rows[date_column].map(regime_map).fillna("Unknown")
        rows = rows[rows["regime"].eq(regime)]
        amount_columns = ["buy_amount", "pm_resized_amount", "scaled_amount", "planned_amount", "round_lot_amount"]
        for column in amount_columns:
            if column in rows.columns:
                values = pd.to_numeric(rows[column], errors="coerce").dropna()
                if not values.empty:
                    return float(values.mean())
        return None

    def _monthly_win_rate(self, trades: pd.DataFrame) -> float | None:
        if trades.empty or "exit_date" not in trades.columns:
            return None
        months = pd.to_datetime(trades["exit_date"], errors="coerce").dt.strftime("%Y-%m")
        monthly = pd.to_numeric(trades.get("net_profit"), errors="coerce").groupby(months).sum()
        monthly = monthly[monthly.index.notna()]
        if monthly.empty:
            return None
        return float((monthly > 0).mean())

    def _contribution_summary(self, regime_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total_profit = sum(float(row.get("net_profit") or 0.0) for row in regime_summary)
        total_trades = sum(int(row.get("trades") or 0) for row in regime_summary)
        weighted_pf = sum((row.get("profit_factor") or 0.0) * int(row.get("trades") or 0) for row in regime_summary)
        return [
            {
                "regime": row["regime"],
                "profit_contribution": (float(row.get("net_profit") or 0.0) / total_profit) if total_profit else None,
                "pf_contribution": (((row.get("profit_factor") or 0.0) * int(row.get("trades") or 0)) / weighted_pf) if weighted_pf else None,
                "trade_contribution": (int(row.get("trades") or 0) / total_trades) if total_trades else None,
                "monthly_win_rate": row.get("monthly_win_rate"),
            }
            for row in regime_summary
        ]

    def _drawdown_analysis(self, summary: pd.DataFrame) -> dict[str, Any]:
        if summary.empty or "total_assets" not in summary.columns:
            return {"max_drawdown": None, "dd_start": "", "dd_trough": "", "dd_recovery": "", "regime_at_dd": "", "regime_around_dd": []}
        rows = summary.copy()
        assets = pd.to_numeric(rows["total_assets"], errors="coerce")
        running_max = assets.cummax()
        drawdown = assets / running_max - 1.0
        trough_idx = drawdown.idxmin()
        peak_value = running_max.loc[trough_idx]
        before = rows.loc[:trough_idx].copy()
        start_idx = assets.loc[:trough_idx][assets.loc[:trough_idx].eq(peak_value)].index[-1]
        after = rows.loc[trough_idx:].copy()
        recovered = after[pd.to_numeric(after["total_assets"], errors="coerce").ge(peak_value)]
        around = rows.loc[max(0, trough_idx - 5) : trough_idx + 5, ["date", "regime", "total_assets"]].to_dict("records")
        return {
            "max_drawdown": float(drawdown.loc[trough_idx]),
            "dd_start": str(rows.loc[start_idx, "date"]),
            "dd_trough": str(rows.loc[trough_idx, "date"]),
            "dd_recovery": "" if recovered.empty else str(recovered.iloc[0]["date"]),
            "regime_at_dd": str(rows.loc[trough_idx, "regime"]),
            "regime_around_dd": around,
        }

    def _high_pm_summary(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        if trades.empty or "pm_multiplier" not in trades.columns:
            return rows
        pm = pd.to_numeric(trades["pm_multiplier"], errors="coerce")
        for regime in REGIME_ORDER:
            for bucket in PM_BUCKETS:
                selected = trades[trades["regime"].eq(regime) & pm.eq(bucket)]
                profits = pd.to_numeric(selected.get("net_profit"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
                rows.append(
                    {
                        "regime": regime,
                        "pm_multiplier": bucket,
                        "trade_count": int(len(selected)),
                        "profit": float(profits.sum()) if not profits.empty else 0.0,
                        "profit_factor": _profit_factor(profits),
                        "win_rate": _win_rate(profits),
                    }
                )
        return rows

    def _virtual_strategy_audit(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        actual = self._scaled_trade_stats(trades, "Actual", "Actual trade log; no virtual filter")
        cases = [
            ("Rule A Bull only", {"Bull": 1.0}, "Keep only trades entered in Bull regime"),
            ("Rule B Bull + Neutral only", {"Bull": 1.0, "Neutral": 1.0}, "Skip Bear entries"),
            ("Rule C Bear buy amount 50%", {"Bull": 1.0, "Neutral": 1.0, "Bear": 0.5, "Unknown": 1.0}, "Scale Bear trade P/L by 50%"),
            ("Rule D Bear new buys stopped", {"Bull": 1.0, "Neutral": 1.0, "Unknown": 1.0}, "Drop Bear trades"),
        ]
        rows = [actual]
        actual_profit = float(actual.get("net_profit") or 0.0)
        for name, weights, note in cases:
            row = self._scaled_trade_stats(trades, name, note, weights)
            row["profit_delta_vs_actual"] = float(row.get("net_profit") or 0.0) - actual_profit
            rows.append(row)
        rows[0]["profit_delta_vs_actual"] = 0.0
        return rows

    def _scaled_trade_stats(self, trades: pd.DataFrame, name: str, note: str, weights: dict[str, float] | None = None) -> dict[str, Any]:
        if trades.empty:
            return {"rule": name, "trades": 0, "net_profit": 0.0, "profit_factor": None, "win_rate": None, "note": note}
        rows = trades.copy()
        if weights is None:
            rows["_weight"] = 1.0
        else:
            rows["_weight"] = rows["regime"].map(weights).fillna(0.0)
        rows = rows[rows["_weight"].gt(0)].copy()
        profits = pd.to_numeric(rows.get("net_profit"), errors="coerce").fillna(0.0) * rows["_weight"]
        return {
            "rule": name,
            "trades": int(len(rows)),
            "net_profit": float(profits.sum()),
            "profit_factor": _profit_factor(profits),
            "win_rate": _win_rate(profits),
            "note": note,
        }

    def _yearly_summary(self, summary: pd.DataFrame, trades: pd.DataFrame) -> list[dict[str, Any]]:
        years = [2023, 2024, 2025, 2026]
        rows = []
        previous_assets = 1_000_000.0
        for year in years:
            daily = summary[pd.to_datetime(summary.get("date"), errors="coerce").dt.year.eq(year)] if not summary.empty else pd.DataFrame()
            yearly_trades = trades[pd.to_datetime(trades.get("exit_date"), errors="coerce").dt.year.eq(year)] if not trades.empty else pd.DataFrame()
            end_assets = float(pd.to_numeric(daily["total_assets"], errors="coerce").iloc[-1]) if not daily.empty else previous_assets
            profit = end_assets - previous_assets
            profits = pd.to_numeric(yearly_trades.get("net_profit"), errors="coerce") if not yearly_trades.empty else pd.Series(dtype=float)
            rows.append(
                {
                    "year": year,
                    "profit": profit,
                    "return": profit / previous_assets if previous_assets else None,
                    "profit_factor": _profit_factor(profits),
                    "max_drawdown": _drawdown_from_assets(daily.get("total_assets")) if not daily.empty else None,
                    "win_rate": _win_rate(profits),
                }
            )
            previous_assets = end_assets
        return rows

    def _verdict(self, regime_summary: list[dict[str, Any]], virtual: list[dict[str, Any]], dd: dict[str, Any]) -> dict[str, Any]:
        by_regime = {row["regime"]: row for row in regime_summary}
        bull_pf = by_regime.get("Bull", {}).get("profit_factor") or 0.0
        bear_pf = by_regime.get("Bear", {}).get("profit_factor") or 0.0
        bull_profit = by_regime.get("Bull", {}).get("net_profit") or 0.0
        bear_profit = by_regime.get("Bear", {}).get("net_profit") or 0.0
        best_virtual = max(virtual[1:], key=lambda row: float(row.get("profit_delta_vs_actual") or -10**18)) if len(virtual) > 1 else {}
        regime_effect = abs(float(bull_profit) - float(bear_profit)) > 100_000 or abs(float(bull_pf) - float(bear_pf)) > 0.3
        filter_worth = bool((best_virtual.get("profit_delta_vs_actual") or 0.0) > 0)
        return {
            "regime_effect_detected": regime_effect,
            "bull_market_advantage": bull_profit > bear_profit and bull_pf >= bear_pf,
            "bear_market_weakness": bear_profit < 0 or (bear_pf > 0 and bear_pf < bull_pf),
            "regime_filter_worth_implementing": filter_worth,
            "recommended_next_phase": "Phase 6-B Market Regime Filter Design" if filter_worth else "Keep v2_78 and continue regime diagnostics",
            "best_virtual_rule": best_virtual.get("rule"),
            "dd_regime": dd.get("regime_at_dd"),
        }

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows_"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(self._format_cell(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format_cell(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)


def run_phase6a_market_regime_audit(root: Path | str = ROOT) -> Phase6AReportPaths:
    audit = Phase6AMarketRegimeAudit(root)
    return audit.save_report(audit.build_report())
