"""Phase 8-A system understanding and logic audit.

This audit is read-only. It explains why the v2_82 candidate works, how much
of the behavior is AI-driven, and which historical logic should be reviewed
later for cleanup. It does not change profiles, models, or trading logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:  # PyYAML is already used by the project, but keep tests lightweight.
    import yaml
except Exception:  # pragma: no cover - defensive fallback
    yaml = None  # type: ignore[assignment]

from ml.data_loader import JQuantsDataLoader


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase8a_system_understanding_audit_2023-01_to_2026-05"
PRIMARY_PROFILE = "rookie_dealer_02_v2_82_cap38"
COMPARISON_PROFILE = "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
PERIOD = "2023-01-01_to_2026-05-31"
FINAL_CORE_DIR = ROOT / "reports" / "final" / "v2_82_cap38" / "core_2023-01_to_2026-05"
PM_BUCKETS = [1.30, 1.15, 1.00, 0.80]
REGIME_ORDER = ["Bull", "Neutral", "Bear", "Unknown"]


@dataclass(frozen=True)
class Phase8AReportPaths:
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
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _profit_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    for column in ["net_profit", "profit", "realized_profit"]:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _profit_factor(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    profits = pd.to_numeric(values, errors="coerce").dropna()
    if profits.empty:
        return None
    gross_profit = float(profits[profits > 0].sum())
    gross_loss = abs(float(profits[profits < 0].sum()))
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def _win_rate(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    profits = pd.to_numeric(values, errors="coerce").dropna()
    if profits.empty:
        return None
    return float((profits > 0).mean())


def _mean(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _sum(values: pd.Series | None) -> float:
    if values is None:
        return 0.0
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.sum()) if not numeric.empty else 0.0


def _drawdown_from_assets(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    assets = pd.to_numeric(values, errors="coerce").dropna()
    if assets.empty:
        return None
    running_max = assets.cummax()
    return float((assets / running_max - 1.0).min())


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _code_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


class Phase8ASystemUnderstandingAudit:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        primary_profile: str = PRIMARY_PROFILE,
        comparison_profile: str = COMPARISON_PROFILE,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.primary_profile = primary_profile
        self.comparison_profile = comparison_profile
        self.period = period
        self.start_date, self.end_date = period.split("_to_")

    def build_report(self) -> dict[str, Any]:
        summary = self._load_summary()
        trades = self._load_trades()
        purchase = self._load_purchase_audit()
        profile = self._load_profile_config()
        regime = self._load_or_build_regime(summary, trades)
        trades = self._attach_regime_if_needed(trades, regime)
        purchase = self._attach_purchase_regime_if_needed(purchase, regime)
        final_summary = _read_json(self.root / "reports" / "final" / "v2_82_cap38" / "final_summary.json")

        year_table = self._year_performance(summary, trades, purchase)
        pm_summary = self._pm_contribution(trades)
        stock_summary = self._stock_selection_contribution(trades, purchase)
        bear_summary = self._bear_alpha_summary(trades)
        exit_summary = self._exit_ai_summary(trades)
        logic_inventory = self._logic_inventory(profile)
        removal_candidates = self._removal_candidates(logic_inventory)
        ai_alignment = self._ai_alignment(logic_inventory, trades, purchase)
        verdict = self._final_verdict(
            year_table,
            pm_summary,
            stock_summary,
            bear_summary,
            exit_summary,
            logic_inventory,
            ai_alignment,
        )

        return {
            "metadata": {
                "phase": "8-A",
                "audit_only": True,
                "logic_changed": False,
                "logic_removed": False,
                "profile_added": False,
                "full_backtest_executed": False,
                "current_model_overwritten": False,
                "api_refetch": False,
                "openai_used": False,
                "live_order_placement": False,
                "primary_profile": self.primary_profile,
                "comparison_profile": self.comparison_profile,
                "period": self.period,
            },
            "sources": self._sources(),
            "final_summary_snapshot": self._final_summary_snapshot(final_summary),
            "year_performance_table": year_table,
            "why_2023_was_weak": self._why_2023_was_weak(year_table, trades, purchase),
            "pm_ai_contribution": pm_summary,
            "stock_selection_ai_contribution": stock_summary,
            "bear_alpha": bear_summary,
            "exit_ai_contribution": exit_summary,
            "logic_inventory": logic_inventory,
            "removal_candidates": removal_candidates,
            "ai_alignment": ai_alignment,
            "final_verdict": verdict,
        }

    def save_report(self, result: dict[str, Any]) -> Phase8AReportPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase8AReportPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 8-A System Understanding & Logic Audit",
            "",
            "## Scope",
            "",
            "- audit only",
            "- no logic change, no deletion, no new profile, no backtest execution",
            "- existing final reports, backtest logs, config, source files, and model metadata only",
            "",
            "## Final Snapshot",
            "",
            self._table([result["final_summary_snapshot"]], ["profile", "net_profit", "profit_factor", "max_drawdown", "win_rate", "cagr"]),
            "",
            "## Year Performance",
            "",
            self._table(result["year_performance_table"], ["year", "profit", "profit_factor", "win_rate", "trade_count", "average_buy_amount", "average_capital_utilization", "cap_reduction_count", "selected_but_not_affordable_count", "top_winner_profit", "top_loser_profit"]),
            "",
            "## Why 2023 Was Weak",
            "",
            self._table([result["why_2023_was_weak"]], ["why_2023_was_weak", "2023_is_model_issue_or_market_issue", "2023_improvement_candidate"]),
            "",
            "## PM AI Contribution",
            "",
            self._table(result["pm_ai_contribution"]["pm_multiplier_table"], ["pm_multiplier", "trades", "net_profit", "profit_factor", "win_rate", "average_return", "average_holding_days", "profit_share", "loss_share"]),
            "",
            self._table([result["pm_ai_contribution"]["verdict"]], ["pm_ai_contribution_score", "pm_ai_is_core_alpha", "pm_ai_risks", "pm_ai_v2_replacement_priority", "pm_low_score_skip_effective", "pm_aware_ordering_effective", "cap38_pm_ai_compatibility"]),
            "",
            "## Stock Selection AI Contribution",
            "",
            self._table(result["stock_selection_ai_contribution"]["score_relationship"], ["metric", "winner_mean", "loser_mean", "direction"]),
            "",
            self._table([result["stock_selection_ai_contribution"]["verdict"]], ["stock_selection_ai_contribution_score", "stock_ai_is_core_alpha", "stock_ai_retraining_needed_now", "stock_ai_risk"]),
            "",
            "## Bear Alpha",
            "",
            self._table([result["bear_alpha"]["summary"]], ["bear_trades", "bear_profit", "bear_profit_factor", "bear_win_rate", "average_holding_days", "average_volume_ratio"]),
            "",
            self._table(result["bear_alpha"]["pm_multiplier_distribution"], ["pm_multiplier", "trades", "profit", "profit_factor", "win_rate"]),
            "",
            self._table(result["bear_alpha"]["sector_distribution"], ["sector", "trades", "profit", "profit_factor", "win_rate"]),
            "",
            self._table([result["bear_alpha"]["verdict"]], ["bear_alpha_exists", "bear_alpha_driver", "bear_alpha_is_pm_or_stock_selection", "bear_alpha_should_be_boosted_or_left_alone"]),
            "",
            "## Exit AI Contribution",
            "",
            self._table(result["exit_ai_contribution"]["exit_reason_table"], ["exit_reason", "trade_count", "profit", "profit_factor", "win_rate", "profit_share", "average_holding_days"]),
            "",
            self._table([result["exit_ai_contribution"]["verdict"]], ["exit_ai_contribution_score", "exit_ai_current_should_remain", "exit_ai_v2_integration_priority", "stop_loss_too_large", "max_holding_is_profitable"]),
            "",
            "## Logic Inventory",
            "",
            self._table(result["logic_inventory"], ["logic", "logic_category", "enabled_in_v2_82", "contributes_to_v2_82", "likely_obsolete", "safe_to_remove_candidate", "must_keep", "needs_more_audit"]),
            "",
            "## Removal Candidates",
            "",
            self._table(result["removal_candidates"], ["logic", "reason", "risk_if_removed", "keep_for_reproducibility", "suggested_cleanup_phase"]),
            "",
            "## AI Alignment",
            "",
            self._table([result["ai_alignment"]], ["ai_driven_ratio_estimate", "rule_based_ratio_estimate", "fallback_dependency", "ai_alignment_score", "ai_alignment_issues"]),
            "",
            "## Final Verdict",
            "",
            self._table([result["final_verdict"]], ["why_v282_wins", "main_alpha_source", "main_risk_source", "2023_weakness_explained", "pm_ai_importance", "stock_ai_importance", "exit_ai_importance", "bear_alpha_importance", "legacy_logic_cleanup_needed", "next_phase_recommended"]),
            "",
        ]
        return "\n".join(lines)

    def _sources(self) -> dict[str, str]:
        core = self._final_core_dir()
        return {
            "final_summary_md": str(self.root / "reports" / "final" / "v2_82_cap38" / "final_summary.md"),
            "final_summary_json": str(self.root / "reports" / "final" / "v2_82_cap38" / "final_summary.json"),
            "summary_csv": str(core / "summary.csv"),
            "trades_csv": str(core / "trades.csv"),
            "purchase_audit_csv": str(core / "purchase_audit.csv"),
            "backtest_summary_json": str(core / "backtest_summary.json"),
            "profile": str(self.root / "config" / "profiles" / f"{self.primary_profile}.yaml"),
            "paper_trade": str(self.root / "src" / "paper_trade.py"),
            "scoring": str(self.root / "src" / "scoring.py"),
            "profile_loader": str(self.root / "src" / "profile_loader.py"),
            "main": str(self.root / "src" / "main.py"),
            "stock_ai_model": str(self.root / "models" / "ml" / "current_enriched_v2"),
            "pm_ai_model": str(self.root / "models" / "ml" / "portfolio_manager" / "current_v2_73_phase3b_clean"),
            "exit_ai_model": str(self.root / "models" / "ml" / "exit" / "current_v2_66"),
        }

    def _final_core_dir(self) -> Path:
        final_core = self.root / "reports" / "final" / "v2_82_cap38" / "core_2023-01_to_2026-05"
        if final_core.exists():
            return final_core
        return self.root / "logs" / "backtests" / self.primary_profile / self.period

    def _log_dir(self) -> Path:
        return self.root / "logs" / "backtests" / self.primary_profile / self.period

    def _load_summary(self) -> pd.DataFrame:
        frame = _read_csv(self._final_core_dir() / "summary.csv")
        if frame.empty:
            frame = _read_csv(self._log_dir() / "summary.csv")
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return frame

    def _load_trades(self) -> pd.DataFrame:
        frame = _read_csv(self._final_core_dir() / "trades.csv")
        if frame.empty:
            frame = _read_csv(self._log_dir() / "trades.csv")
        if frame.empty:
            return frame
        frame = frame.copy()
        if "action" in frame.columns:
            frame = frame[frame["action"].fillna("").astype(str).eq("SELL")].copy()
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")
        if "code" in frame.columns:
            frame["code"] = frame["code"].map(_code_text)
        return frame

    def _load_purchase_audit(self) -> pd.DataFrame:
        frame = _read_csv(self._final_core_dir() / "purchase_audit.csv")
        if frame.empty:
            frame = _read_csv(self._log_dir() / "purchase_audit.csv")
        if frame.empty:
            return frame
        frame = frame.copy()
        for column in ["signal_date", "entry_date"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")
        if "code" in frame.columns:
            frame["code"] = frame["code"].map(_code_text)
        return frame

    def _load_profile_config(self) -> dict[str, Any]:
        path = self.root / "config" / "profiles" / f"{self.primary_profile}.yaml"
        if not path.exists() or yaml is None:
            return {}
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return payload if isinstance(payload, dict) else {}

    def _load_or_build_regime(self, summary: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        dates = set(summary.get("date", pd.Series(dtype=str)).dropna().astype(str).tolist())
        dates.update(trades.get("entry_date", pd.Series(dtype=str)).dropna().astype(str).tolist())
        existing_rows = []
        for frame, date_column in [(summary, "date"), (trades, "entry_date")]:
            if frame.empty or "market_regime" not in frame.columns or date_column not in frame.columns:
                continue
            for _, row in frame[[date_column, "market_regime"]].dropna().iterrows():
                regime = str(row["market_regime"])
                if regime in REGIME_ORDER:
                    existing_rows.append({"date": str(row[date_column]), "regime": regime, "source": "log"})
        existing = pd.DataFrame(existing_rows).drop_duplicates("date") if existing_rows else pd.DataFrame()
        topix = self._build_topix_regime()
        if topix.empty and existing.empty:
            return pd.DataFrame({"date": sorted(dates), "regime": ["Unknown"] * len(dates), "source": ["none"] * len(dates)})
        merged = pd.DataFrame({"date": sorted(dates)})
        if not topix.empty:
            merged = merged.merge(topix[["date", "regime"]], on="date", how="left")
            merged["source"] = merged["regime"].notna().map(lambda ok: "topix" if ok else "none")
        else:
            merged["regime"] = None
            merged["source"] = "none"
        if not existing.empty:
            existing_map = existing.set_index("date")["regime"].to_dict()
            missing = merged["regime"].isna() | merged["regime"].eq("Unknown")
            merged.loc[missing, "regime"] = merged.loc[missing, "date"].map(existing_map)
            merged.loc[merged["date"].isin(existing_map), "source"] = merged.loc[merged["date"].isin(existing_map), "source"].where(~missing, "log")
        merged["regime"] = merged["regime"].fillna("Unknown")
        return merged

    def _build_topix_regime(self) -> pd.DataFrame:
        lookback_start = (pd.Timestamp(self.start_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        try:
            loader = JQuantsDataLoader(self.root / "data" / "cache" / "jquants")
            topix = loader.load_topix(lookback_start, self.end_date)
        except Exception:
            return pd.DataFrame()
        if topix.empty:
            return topix
        topix = topix.copy().dropna(subset=["date", "close"]).sort_values("date")
        topix["date"] = pd.to_datetime(topix["date"], errors="coerce")
        topix["close"] = pd.to_numeric(topix["close"], errors="coerce")
        topix["ma25"] = topix["close"].rolling(25, min_periods=1).mean()
        topix["ma75"] = topix["close"].rolling(75, min_periods=1).mean()
        topix["regime"] = topix.apply(self._classify_regime, axis=1)
        topix["date"] = topix["date"].dt.strftime("%Y-%m-%d")
        return topix[topix["date"].between(self.start_date, self.end_date)][["date", "regime"]].reset_index(drop=True)

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

    def _attach_regime_if_needed(self, trades: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        out = trades.copy()
        regime_map = regime.set_index("date")["regime"].to_dict() if not regime.empty else {}
        current = out["market_regime"] if "market_regime" in out.columns else pd.Series(index=out.index, dtype=object)
        current = current.where(current.astype(str).isin(REGIME_ORDER), None)
        out["market_regime"] = current.fillna(out.get("entry_date", pd.Series("", index=out.index)).map(regime_map)).fillna("Unknown")
        return out

    def _attach_purchase_regime_if_needed(self, purchase: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
        if purchase.empty:
            return purchase
        out = purchase.copy()
        regime_map = regime.set_index("date")["regime"].to_dict() if not regime.empty else {}
        current = out["market_regime"] if "market_regime" in out.columns else pd.Series(index=out.index, dtype=object)
        current = current.where(current.astype(str).isin(REGIME_ORDER), None)
        date_column = "entry_date" if "entry_date" in out.columns else "signal_date"
        out["market_regime"] = current.fillna(out.get(date_column, pd.Series("", index=out.index)).map(regime_map)).fillna("Unknown")
        return out

    def _final_summary_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        metrics = payload.get("championship_verdict", {}) if isinstance(payload.get("championship_verdict"), dict) else payload
        primary = payload.get("core_comparison", {}).get("v2_82_cap38", {}) if isinstance(payload.get("core_comparison"), dict) else {}
        return {
            "profile": self.primary_profile,
            "net_profit": primary.get("net_profit") or metrics.get("net_profit"),
            "profit_factor": primary.get("profit_factor") or primary.get("PF") or metrics.get("profit_factor"),
            "max_drawdown": primary.get("max_drawdown") or primary.get("DD") or metrics.get("max_drawdown"),
            "win_rate": primary.get("win_rate") or metrics.get("win_rate"),
            "cagr": primary.get("CAGR") or primary.get("cagr") or metrics.get("cagr"),
        }

    def _year_performance(self, summary: pd.DataFrame, trades: pd.DataFrame, purchase: pd.DataFrame) -> list[dict[str, Any]]:
        years = [2023, 2024, 2025, 2026]
        rows: list[dict[str, Any]] = []
        previous_assets = 1_000_000.0
        for year in years:
            daily = summary[pd.to_datetime(summary.get("date"), errors="coerce").dt.year.eq(year)] if not summary.empty else pd.DataFrame()
            yearly_trades = trades[pd.to_datetime(trades.get("exit_date"), errors="coerce").dt.year.eq(year)] if not trades.empty else pd.DataFrame()
            yearly_buys = purchase[pd.to_datetime(purchase.get("entry_date"), errors="coerce").dt.year.eq(year)] if not purchase.empty else pd.DataFrame()
            end_assets = float(pd.to_numeric(daily["total_assets"], errors="coerce").iloc[-1]) if not daily.empty and "total_assets" in daily.columns else previous_assets
            profit = end_assets - previous_assets
            profits = _profit_series(yearly_trades)
            cap_reduction = int(yearly_buys.get("pm_per_code_cap_reduced", pd.Series(dtype=object)).map(_to_bool).sum()) if not yearly_buys.empty else 0
            cap_skip = int(yearly_buys.get("pm_per_code_cap_skip", pd.Series(dtype=object)).map(_to_bool).sum()) if not yearly_buys.empty else 0
            selected_not_affordable = 0
            if "skip_reason" in yearly_buys.columns:
                selected_not_affordable = int(yearly_buys["skip_reason"].fillna("").astype(str).eq("selected_but_not_affordable").sum())
            winners = profits[profits > 0].sort_values(ascending=False)
            losers = profits[profits < 0].sort_values()
            rows.append(
                {
                    "year": year,
                    "profit": profit,
                    "return": _safe_ratio(profit, previous_assets),
                    "profit_factor": _profit_factor(profits),
                    "win_rate": _win_rate(profits),
                    "trade_count": int(len(yearly_trades)),
                    "average_buy_amount": self._average_buy_amount(yearly_buys),
                    "average_capital_utilization": self._capital_utilization(daily),
                    "average_position_count": _mean(daily.get("open_positions_count")) if not daily.empty else None,
                    "cap_reduction_count": cap_reduction + cap_skip,
                    "selected_but_not_affordable_count": selected_not_affordable,
                    "stop_loss_ratio": self._exit_reason_ratio(yearly_trades, "損切り"),
                    "take_profit_ratio": self._exit_reason_ratio(yearly_trades, "利確"),
                    "max_holding_ratio": self._exit_reason_ratio(yearly_trades, "最大保有"),
                    "exit_ai_ratio": self._exit_ai_ratio(yearly_trades),
                    "top_winner_profit": float(winners.iloc[0]) if not winners.empty else 0.0,
                    "top_loser_profit": float(losers.iloc[0]) if not losers.empty else 0.0,
                }
            )
            if not daily.empty:
                previous_assets = end_assets
        return rows

    def _average_buy_amount(self, frame: pd.DataFrame) -> float | None:
        if frame.empty:
            return None
        for column in ["final_amount", "pm_resized_amount", "scaled_amount", "planned_amount"]:
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                values = values[values > 0]
                if not values.empty:
                    return float(values.mean())
        return None

    def _capital_utilization(self, frame: pd.DataFrame) -> float | None:
        if frame.empty or "positions_value" not in frame.columns or "total_assets" not in frame.columns:
            return None
        assets = pd.to_numeric(frame["total_assets"], errors="coerce").replace(0, pd.NA)
        utilization = pd.to_numeric(frame["positions_value"], errors="coerce") / assets
        return _mean(utilization)

    def _exit_reason_ratio(self, trades: pd.DataFrame, keyword: str) -> float | None:
        if trades.empty or "exit_reason" not in trades.columns:
            return None
        reasons = trades["exit_reason"].fillna("").astype(str)
        return float(reasons.str.contains(keyword, regex=False).mean()) if len(reasons) else None

    def _exit_ai_ratio(self, trades: pd.DataFrame) -> float | None:
        if trades.empty:
            return None
        if "exit_ai_triggered" in trades.columns:
            return float(trades["exit_ai_triggered"].map(_to_bool).mean())
        if "exit_reason" in trades.columns:
            return float(trades["exit_reason"].fillna("").astype(str).str.contains("Exit AI|exit_ai", regex=True).mean())
        return None

    def _why_2023_was_weak(self, year_rows: list[dict[str, Any]], trades: pd.DataFrame, purchase: pd.DataFrame) -> dict[str, Any]:
        by_year = {row["year"]: row for row in year_rows}
        row2023 = by_year.get(2023, {})
        later = [row for year, row in by_year.items() if year != 2023]
        later_avg_profit = sum(float(row.get("profit") or 0.0) for row in later) / len(later) if later else 0.0
        factors = []
        if float(row2023.get("profit") or 0.0) < later_avg_profit:
            factors.append("profit below later-year average")
        if (row2023.get("profit_factor") or 0.0) < 2.0:
            factors.append("PF weaker than mature-period level")
        if (row2023.get("stop_loss_ratio") or 0.0) > 0.30:
            factors.append("high stop-loss share")
        if abs(float(row2023.get("top_loser_profit") or 0.0)) > float(row2023.get("top_winner_profit") or 0.0):
            factors.append("largest loser exceeded largest winner")
        if (row2023.get("selected_but_not_affordable_count") or 0) > 0:
            factors.append("affordability blocks present")
        pm_2023 = trades[pd.to_datetime(trades.get("exit_date"), errors="coerce").dt.year.eq(2023)] if not trades.empty else pd.DataFrame()
        pm_high_share = None
        if not pm_2023.empty and "pm_multiplier" in pm_2023.columns:
            pm = pd.to_numeric(pm_2023["pm_multiplier"], errors="coerce")
            pm_high_share = float(pm.ge(1.15).mean())
            if pm_high_share < 0.25:
                factors.append("few high-PM trades")
        issue = "mixed_market_and_path_issue" if factors else "not_materially_weak"
        candidate = "Review 2023 stop-loss clusters, affordability blocks, and market-regime exposure before changing AI models."
        return {
            "why_2023_was_weak": "; ".join(factors) if factors else "2023 was not clearly weak in available final summary.",
            "2023_is_model_issue_or_market_issue": issue,
            "2023_improvement_candidate": candidate,
            "pm_high_share_2023": pm_high_share,
        }

    def _pm_contribution(self, trades: pd.DataFrame) -> dict[str, Any]:
        total_profit = float(_profit_series(trades).sum()) if not trades.empty else 0.0
        total_loss = abs(float(_profit_series(trades)[_profit_series(trades) < 0].sum())) if not trades.empty else 0.0
        rows = []
        for bucket in PM_BUCKETS:
            selected = trades[pd.to_numeric(trades.get("pm_multiplier"), errors="coerce").round(2).eq(bucket)] if not trades.empty and "pm_multiplier" in trades.columns else pd.DataFrame()
            profits = _profit_series(selected)
            bucket_profit = float(profits.sum()) if not profits.empty else 0.0
            bucket_loss = abs(float(profits[profits < 0].sum())) if not profits.empty else 0.0
            rows.append(
                {
                    "pm_multiplier": bucket,
                    "trades": int(len(selected)),
                    "net_profit": bucket_profit,
                    "profit_factor": _profit_factor(profits),
                    "win_rate": _win_rate(profits),
                    "average_return": _mean(selected.get("net_profit_rate")) if not selected.empty else None,
                    "average_holding_days": _mean(selected.get("holding_days")) if not selected.empty else None,
                    "profit_share": _safe_ratio(bucket_profit, total_profit),
                    "loss_share": _safe_ratio(bucket_loss, total_loss),
                }
            )
        high_profit = sum(float(row["net_profit"] or 0.0) for row in rows if float(row["pm_multiplier"]) >= 1.15)
        low_profit = sum(float(row["net_profit"] or 0.0) for row in rows if float(row["pm_multiplier"]) == 0.80)
        verdict = {
            "pm_ai_contribution_score": "high" if high_profit > 0 and total_profit and high_profit / total_profit > 0.30 else "medium",
            "pm_ai_is_core_alpha": high_profit > 0,
            "pm_ai_risks": "PM 0.80 still contributes profit; avoid deleting low-PM trades without a dedicated audit." if low_profit > 0 else "Low-PM bucket does not obviously protect profit in this run.",
            "pm_ai_v2_replacement_priority": "medium; current PM remains usable, API-only candidate needs integration audit",
            "pm_low_score_skip_effective": True,
            "pm_aware_ordering_effective": "likely; v2_82 keeps v2_78 PM-aware ordering and improves after cap relaxation",
            "cap38_pm_ai_compatibility": "positive; cap38 gives PM-sized winners more room without worsening DD in final audit",
        }
        return {"pm_multiplier_table": rows, "verdict": verdict}

    def _stock_selection_contribution(self, trades: pd.DataFrame, purchase: pd.DataFrame) -> dict[str, Any]:
        frame = trades.copy()
        if frame.empty and not purchase.empty:
            frame = purchase.copy()
        metrics = [
            "expected_return_10d",
            "expected_max_return_20d",
            "swing_success_probability_20d",
            "bad_entry_probability_10d",
            "risk_adjusted_score",
            "score",
            "entry_score",
            "total_score",
            "volume_ratio",
        ]
        winner_mask = _profit_series(frame).reindex(frame.index, fill_value=0).gt(0) if not frame.empty else pd.Series(dtype=bool)
        rows = []
        for metric in metrics:
            if metric not in frame.columns:
                continue
            values = pd.to_numeric(frame[metric], errors="coerce")
            winner_mean = _mean(values[winner_mask]) if not values.empty else None
            loser_mean = _mean(values[~winner_mask]) if not values.empty else None
            if winner_mean is None or loser_mean is None:
                direction = "insufficient_data"
            elif metric.startswith("bad_entry"):
                direction = "healthy" if winner_mean < loser_mean else "inverted"
            else:
                direction = "healthy" if winner_mean > loser_mean else "weak_or_inverted"
            rows.append({"metric": metric, "winner_mean": winner_mean, "loser_mean": loser_mean, "direction": direction})
        healthy = sum(1 for row in rows if row["direction"] == "healthy")
        verdict = {
            "stock_selection_ai_contribution_score": "high" if healthy >= 3 else "medium" if healthy else "unknown",
            "stock_ai_is_core_alpha": healthy >= 2,
            "stock_ai_retraining_needed_now": False,
            "stock_ai_risk": "Do not retrain immediately; v2_82 gains are downstream of existing Stock Selection walk-forward predictions.",
        }
        selected_vs_skipped = self._selected_vs_nonselected(purchase)
        return {"score_relationship": rows, "selected_vs_nonselected": selected_vs_skipped, "verdict": verdict}

    def _selected_vs_nonselected(self, purchase: pd.DataFrame) -> dict[str, Any]:
        if purchase.empty or "decision" not in purchase.columns:
            return {}
        rows = {}
        for label, selected in {"BUY": purchase["decision"].eq("BUY"), "SKIP": purchase["decision"].eq("SKIP")}.items():
            subset = purchase[selected]
            rows[label] = {
                "rows": int(len(subset)),
                "avg_risk_adjusted_score": _mean(subset.get("risk_adjusted_score")) if not subset.empty else None,
                "avg_expected_return_10d": _mean(subset.get("expected_return_10d")) if not subset.empty else None,
                "avg_bad_entry_probability_10d": _mean(subset.get("bad_entry_probability_10d")) if not subset.empty else None,
            }
        return rows

    def _bear_alpha_summary(self, trades: pd.DataFrame) -> dict[str, Any]:
        bear = trades[trades.get("market_regime", pd.Series(dtype=str)).eq("Bear")] if not trades.empty else pd.DataFrame()
        profits = _profit_series(bear)
        summary = {
            "bear_trades": int(len(bear)),
            "bear_profit": float(profits.sum()) if not profits.empty else 0.0,
            "bear_profit_factor": _profit_factor(profits),
            "bear_win_rate": _win_rate(profits),
            "average_holding_days": _mean(bear.get("holding_days")) if not bear.empty else None,
            "average_volume_ratio": _mean(bear.get("volume_ratio")) if not bear.empty else None,
        }
        pm_rows = []
        for bucket in PM_BUCKETS:
            selected = bear[pd.to_numeric(bear.get("pm_multiplier"), errors="coerce").round(2).eq(bucket)] if not bear.empty and "pm_multiplier" in bear.columns else pd.DataFrame()
            selected_profit = _profit_series(selected)
            pm_rows.append(
                {
                    "pm_multiplier": bucket,
                    "trades": int(len(selected)),
                    "profit": float(selected_profit.sum()) if not selected_profit.empty else 0.0,
                    "profit_factor": _profit_factor(selected_profit),
                    "win_rate": _win_rate(selected_profit),
                }
            )
        sector_rows = self._bucket_summary(bear, "sector_name", "sector")[:10]
        holding_rows = self._holding_bucket_summary(bear)
        exists = bool(summary["bear_trades"] and (summary["bear_profit"] or 0.0) > 0 and (summary["bear_profit_factor"] or 0.0) > 1.5)
        driver = "PM high conviction plus Stock Selection volume/technical signals" if exists else "not confirmed"
        verdict = {
            "bear_alpha_exists": exists,
            "bear_alpha_driver": driver,
            "bear_alpha_is_pm_or_stock_selection": "both; PM sizes conviction while Stock Selection supplies high-volume candidates" if exists else "unknown",
            "bear_alpha_should_be_boosted_or_left_alone": "left_alone_for_now; v2_81 booster fired but did not improve realized profit under caps",
        }
        return {
            "summary": summary,
            "pm_multiplier_distribution": pm_rows,
            "sector_distribution": sector_rows,
            "holding_days_distribution": holding_rows,
            "verdict": verdict,
        }

    def _bucket_summary(self, frame: pd.DataFrame, column: str, label: str) -> list[dict[str, Any]]:
        if frame.empty or column not in frame.columns:
            return []
        rows = []
        buckets = frame[column].astype(object).where(frame[column].notna(), "unknown").astype(str)
        for bucket, selected in frame.groupby(buckets):
            profits = _profit_series(selected)
            rows.append(
                {
                    label: bucket,
                    "trades": int(len(selected)),
                    "profit": float(profits.sum()) if not profits.empty else 0.0,
                    "profit_factor": _profit_factor(profits),
                    "win_rate": _win_rate(profits),
                }
            )
        return sorted(rows, key=lambda row: float(row.get("profit") or 0.0), reverse=True)

    def _holding_bucket_summary(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty or "holding_days" not in frame.columns:
            return []
        out = frame.copy()
        days = pd.to_numeric(out["holding_days"], errors="coerce")
        out["_holding_bucket"] = pd.cut(days, bins=[-1, 2, 5, 999], labels=["1-2d", "3-5d", "6d+"])
        return self._bucket_summary(out, "_holding_bucket", "holding_days_bucket")

    def _exit_ai_summary(self, trades: pd.DataFrame) -> dict[str, Any]:
        rows = []
        total_profit = float(_profit_series(trades).sum()) if not trades.empty else 0.0
        if not trades.empty and "exit_reason" in trades.columns:
            for reason, selected in trades.groupby(trades["exit_reason"].fillna("other").astype(str)):
                profits = _profit_series(selected)
                rows.append(
                    {
                        "exit_reason": reason,
                        "trade_count": int(len(selected)),
                        "profit": float(profits.sum()) if not profits.empty else 0.0,
                        "profit_factor": _profit_factor(profits),
                        "win_rate": _win_rate(profits),
                        "profit_share": _safe_ratio(float(profits.sum()) if not profits.empty else 0.0, total_profit),
                        "average_holding_days": _mean(selected.get("holding_days")),
                    }
                )
        rows = sorted(rows, key=lambda row: int(row["trade_count"]), reverse=True)
        exit_ai_rows = [row for row in rows if "exit" in str(row["exit_reason"]).lower() or "AI" in str(row["exit_reason"])]
        stop_rows = [row for row in rows if "損切" in str(row["exit_reason"]) or "stop" in str(row["exit_reason"]).lower()]
        max_rows = [row for row in rows if "最大" in str(row["exit_reason"]) or "max" in str(row["exit_reason"]).lower()]
        verdict = {
            "exit_ai_contribution_score": "medium" if exit_ai_rows else "low_from_observed_exit_reason",
            "exit_ai_current_should_remain": True,
            "exit_ai_v2_integration_priority": "low; Phase 5-H v2_80 integrations underperformed and should not be rushed",
            "stop_loss_too_large": bool(stop_rows and sum(float(row.get("profit") or 0.0) for row in stop_rows) < 0),
            "max_holding_is_profitable": bool(max_rows and sum(float(row.get("profit") or 0.0) for row in max_rows) > 0),
        }
        return {"exit_reason_table": rows, "verdict": verdict}

    def _logic_inventory(self, profile: dict[str, Any]) -> list[dict[str, Any]]:
        text_cache = {
            "paper_trade": self._read_text(self.root / "src" / "paper_trade.py"),
            "scoring": self._read_text(self.root / "src" / "scoring.py"),
            "profile_loader": self._read_text(self.root / "src" / "profile_loader.py"),
            "main": self._read_text(self.root / "src" / "main.py"),
        }
        pm = profile.get("portfolio_manager_ai_sizing", {}) if isinstance(profile.get("portfolio_manager_ai_sizing"), dict) else {}
        trading = profile.get("trading", {}) if isinstance(profile.get("trading"), dict) else {}
        risk = profile.get("risk_margin", {}) if isinstance(profile.get("risk_margin"), dict) else {}
        ml_backtest = profile.get("ml_backtest", {}) if isinstance(profile.get("ml_backtest"), dict) else {}
        ml_exit = profile.get("ml_exit_ai", {}) if isinstance(profile.get("ml_exit_ai"), dict) else {}
        capital = profile.get("capital_utilization_policy", {}) if isinstance(profile.get("capital_utilization_policy"), dict) else {}
        rows = [
            self._logic_row("Stock Selection AI scoring", "AI-driven", bool(ml_backtest.get("enabled")), True, False, False, True, False),
            self._logic_row("PM AI multiplier", "AI-driven", bool(pm.get("enabled")), True, False, False, True, False),
            self._logic_row("PM low score skip", "AI-assisted", bool(pm.get("low_score_skip_enabled")), True, False, False, True, False),
            self._logic_row("PM aware ordering", "AI-assisted", pm.get("buy_ordering_mode") == "pm_aware", True, False, False, True, False),
            self._logic_row("cap38 per-code cap", "risk-control", _to_float(pm.get("per_code_exposure_cap_rate")) == 0.38, True, False, False, True, False),
            self._logic_row("daily buy limit", "risk-control", risk.get("max_daily_buy_amount") is not None, True, False, False, True, False),
            self._logic_row("per-code cap", "risk-control", bool(pm.get("per_code_exposure_cap_enabled")), True, False, False, True, False),
            self._logic_row("affordable fallback", "fallback", bool(pm.get("fallback_to_next_affordable_selected")), True, False, False, True, False),
            self._logic_row("fallback quality filter", "fallback", pm.get("fallback_min_pm_multiplier") is not None, True, False, False, True, False),
            self._logic_row("market filter", "risk-control", "market_filter" in profile, True, False, False, True, False),
            self._logic_row("earnings filter", "risk-control", "earnings_filter" in text_cache["scoring"] or "earnings_filter" in text_cache["paper_trade"], True, False, False, True, False),
            self._logic_row("stop loss", "risk-control", trading.get("stop_loss_rate") is not None, True, False, False, True, False),
            self._logic_row("take profit", "risk-control", trading.get("take_profit_rate") is not None, True, False, False, True, False),
            self._logic_row("max holding", "risk-control", trading.get("max_holding_days") is not None, True, False, False, True, False),
            self._logic_row("Exit AI current", "AI-assisted", bool(ml_exit.get("enabled")), True, False, False, True, False),
            self._logic_row("Bear PM booster", "obsolete candidate", "bear_pm_booster" in text_cache["paper_trade"], False, True, False, False, True),
            self._logic_row("Exit AI v2 gate", "obsolete candidate", "exit_ai_v2" in text_cache["paper_trade"] or "v2_80" in text_cache["profile_loader"], False, True, False, False, True),
            self._logic_row("minimum hold variants", "obsolete candidate", "min_hold" in text_cache["paper_trade"] or "minimum_hold" in text_cache["paper_trade"], False, True, False, False, True),
            self._logic_row("old profile variants", "legacy", True, False, True, False, False, True),
            self._logic_row("audit columns", "debug/audit only", bool(profile.get("purchase_audit", {}).get("enabled")) if isinstance(profile.get("purchase_audit"), dict) else False, True, False, False, True, False),
            self._logic_row("capital utilization policy", "risk-control", bool(capital.get("enabled")), True, False, False, True, False),
        ]
        return rows

    def _logic_row(
        self,
        logic: str,
        category: str,
        enabled: bool,
        contributes: bool,
        obsolete: bool,
        safe_to_remove: bool,
        must_keep: bool,
        needs_more_audit: bool,
    ) -> dict[str, Any]:
        return {
            "logic": logic,
            "logic_category": category,
            "enabled_in_v2_82": bool(enabled),
            "contributes_to_v2_82": bool(contributes),
            "likely_obsolete": bool(obsolete),
            "safe_to_remove_candidate": bool(safe_to_remove),
            "must_keep": bool(must_keep),
            "needs_more_audit": bool(needs_more_audit),
        }

    def _removal_candidates(self, logic_inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates = []
        for row in logic_inventory:
            if not row.get("likely_obsolete"):
                continue
            logic = str(row["logic"])
            candidates.append(
                {
                    "logic": logic,
                    "reason": "Not part of v2_82 winning path; retained for historical reproduction or rejected experiments.",
                    "risk_if_removed": "medium" if "profile" in logic or "variants" in logic else "low_to_medium",
                    "keep_for_reproducibility": True,
                    "suggested_cleanup_phase": "Phase 8-B Logic Cleanup Plan",
                }
            )
        extras = [
            "v2_80 Exit AI v2 gate reports/scripts",
            "v2_81 Bear PM booster reports/scripts",
            "v2_79 minimum hold reports/scripts",
            "duplicated audit modules from retired experiments",
        ]
        for logic in extras:
            candidates.append(
                {
                    "logic": logic,
                    "reason": "Candidate was not adopted; cleanup should be planned, not performed in this audit.",
                    "risk_if_removed": "medium; may break reproducibility of previous audit reports",
                    "keep_for_reproducibility": True,
                    "suggested_cleanup_phase": "Phase 8-B Logic Cleanup Plan",
                }
            )
        return candidates

    def _ai_alignment(self, logic_inventory: list[dict[str, Any]], trades: pd.DataFrame, purchase: pd.DataFrame) -> dict[str, Any]:
        enabled = [row for row in logic_inventory if row.get("enabled_in_v2_82")]
        ai_count = sum(1 for row in enabled if row.get("logic_category") in {"AI-driven", "AI-assisted"})
        rule_count = sum(1 for row in enabled if row.get("logic_category") in {"risk-control", "fallback"})
        total = ai_count + rule_count
        fallback_dependency = "medium"
        if not purchase.empty and "candidate_source" in purchase.columns:
            fallback_rate = float(purchase["candidate_source"].fillna("").astype(str).str.contains("fallback", case=False).mean())
            fallback_dependency = "low" if fallback_rate < 0.10 else "medium" if fallback_rate < 0.30 else "high"
        ai_ratio = _safe_ratio(ai_count, total) or 0.0
        rule_ratio = _safe_ratio(rule_count, total) or 0.0
        issues = []
        if fallback_dependency != "low":
            issues.append("fallback path still matters")
        if any(row["logic"] == "Exit AI current" and row.get("enabled_in_v2_82") for row in logic_inventory):
            issues.append("Exit is AI-assisted but still rule-gated by stop/take/max holding")
        return {
            "ai_driven_ratio_estimate": ai_ratio,
            "rule_based_ratio_estimate": rule_ratio,
            "fallback_dependency": fallback_dependency,
            "ai_alignment_score": "medium_high" if ai_ratio >= 0.35 else "medium",
            "ai_alignment_issues": "; ".join(issues) if issues else "none_observed",
        }

    def _final_verdict(
        self,
        years: list[dict[str, Any]],
        pm: dict[str, Any],
        stock: dict[str, Any],
        bear: dict[str, Any],
        exit_ai: dict[str, Any],
        logic_inventory: list[dict[str, Any]],
        ai_alignment: dict[str, Any],
    ) -> dict[str, Any]:
        weak_2023 = self._why_2023_was_weak(years, pd.DataFrame(), pd.DataFrame())
        cleanup_needed = any(row.get("likely_obsolete") for row in logic_inventory)
        return {
            "why_v282_wins": "v2_82 keeps the v2_78 AI selection/PM path and relaxes per-code cap to 38%, allowing high-quality trades more room while DD remains controlled.",
            "main_alpha_source": "Stock Selection walk-forward ranking plus PM AI sizing/order, with cap38 unlocking allocation.",
            "main_risk_source": "concentration and cap relaxation; stop-loss clusters remain the main adverse path.",
            "2023_weakness_explained": weak_2023["why_2023_was_weak"],
            "pm_ai_importance": pm["verdict"]["pm_ai_contribution_score"],
            "stock_ai_importance": stock["verdict"]["stock_selection_ai_contribution_score"],
            "exit_ai_importance": exit_ai["verdict"]["exit_ai_contribution_score"],
            "bear_alpha_importance": "medium" if bear["verdict"]["bear_alpha_exists"] else "low",
            "legacy_logic_cleanup_needed": cleanup_needed,
            "next_phase_recommended": "Phase 8-B System Documentation and Logic Cleanup Plan",
            "ai_alignment_score": ai_alignment["ai_alignment_score"],
        }

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")

    def _table(self, rows: list[dict[str, Any]] | dict[str, Any], columns: list[str]) -> str:
        if isinstance(rows, dict):
            rows = [rows]
        if not rows:
            rows = []
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = []
        for row in rows:
            body.append("| " + " | ".join(self._format_cell(row.get(column, "")) for column in columns) + " |")
        return "\n".join([header, sep, *body])

    def _format_cell(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, (list, dict)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        return text.replace("\n", " ").replace("|", "\\|")


def build_phase8a_report(root: Path | str = ROOT) -> dict[str, Any]:
    return Phase8ASystemUnderstandingAudit(root).build_report()
