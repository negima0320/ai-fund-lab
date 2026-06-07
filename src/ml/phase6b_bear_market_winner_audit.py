"""Phase 6-B Bear market winner audit.

Read-only audit that explains what v2_78 bought during TOPIX Bear regimes.
No backtest, profile creation, or model execution is performed here.
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
REPORT_STEM = "phase6b_bear_market_winner_audit_2023-01_to_2026-05"
REGIME_ORDER = ["Bull", "Neutral", "Bear", "Unknown"]


@dataclass(frozen=True)
class Phase6BReportPaths:
    markdown: Path
    json: Path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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


def _code_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _bucket_holding_days(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "unknown"
    days = int(numeric)
    if days <= 2:
        return "1-2d"
    if days <= 4:
        return "3-4d"
    return "5d+"


def _bucket_liquidity(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if float(numeric) >= 2.0:
        return "volume_ratio>=2"
    if float(numeric) >= 1.0:
        return "volume_ratio>=1"
    return "volume_ratio<1"


class Phase6BBearMarketWinnerAudit:
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
        trades = self._load_trades()
        purchase = self._load_purchase_audit()
        listed = self._load_listed_info()
        regime = self._load_regime()
        enriched = self._enrich_trades(trades, purchase, listed, regime)
        bear = enriched[enriched["regime"].eq("Bear")].copy() if not enriched.empty else pd.DataFrame()
        bull = enriched[enriched["regime"].eq("Bull")].copy() if not enriched.empty else pd.DataFrame()
        bear_winners = bear[pd.to_numeric(bear.get("profit"), errors="coerce").gt(0)].copy() if not bear.empty else pd.DataFrame()
        return {
            "metadata": {
                "phase": "6-B",
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
                "market_cap_note": "market_cap numeric value is unavailable in existing logs/cache; scale_category is used as market_cap_band.",
                "liquidity_note": "volume_ratio from trades.csv is used as the liquidity proxy.",
            },
            "coverage": self._coverage(enriched),
            "bear_trade_details": self._bear_trade_details(bear),
            "bear_top50_by_profit": self._top_trades(bear_winners, "profit"),
            "bear_top50_by_return": self._top_trades(bear_winners, "return"),
            "bull_vs_bear_comparison": self._bull_vs_bear_comparison(bull, bear),
            "bear_winner_patterns": self._bear_winner_patterns(bear_winners),
            "pattern_hypotheses": self._pattern_hypotheses(bull, bear, bear_winners),
            "verdict": self._verdict(bull, bear, bear_winners),
        }

    def save_report(self, result: dict[str, Any]) -> Phase6BReportPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase6BReportPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 6-B Bear Market Winner Audit",
            "",
            "## Scope",
            "",
            "- audit only",
            "- no logic change, no new profile, no backtest execution",
            "- existing v2_78 logs and local cache only",
            "",
            "## Coverage",
            "",
            self._table([result["coverage"]], ["total_trades", "classified_trades", "bear_trades", "bear_winner_trades", "bull_trades"]),
            "",
            "## Bear Trades",
            "",
            self._table(result["bear_trade_details"][:50], ["code", "sector", "buy_date", "sell_date", "profit", "return", "holding_days", "pm_score", "pm_multiplier", "buy_amount", "market_cap_band", "liquidity"]),
            "",
            "## Bear Winners Top50 Profit",
            "",
            self._table(result["bear_top50_by_profit"], ["code", "sector", "buy_date", "sell_date", "profit", "return", "holding_days", "pm_score", "pm_multiplier", "buy_amount", "market_cap_band", "liquidity"]),
            "",
            "## Bear Winners Top50 Return",
            "",
            self._table(result["bear_top50_by_return"], ["code", "sector", "buy_date", "sell_date", "profit", "return", "holding_days", "pm_score", "pm_multiplier", "buy_amount", "market_cap_band", "liquidity"]),
            "",
            "## Bull vs Bear",
            "",
            self._table(result["bull_vs_bear_comparison"], ["regime", "trade_count", "profit", "profit_factor", "win_rate", "avg_pm_score", "avg_pm_multiplier", "avg_holding_days", "avg_buy_amount", "top_sector", "top_market_cap_band", "top_liquidity"]),
            "",
            "## Bear Winner Patterns",
            "",
            self._table(result["bear_winner_patterns"]["sector"], ["bucket", "trade_count", "profit", "profit_factor", "win_rate"]),
            "",
            self._table(result["bear_winner_patterns"]["pm_multiplier"], ["bucket", "trade_count", "profit", "profit_factor", "win_rate"]),
            "",
            self._table(result["bear_winner_patterns"]["market_cap_band"], ["bucket", "trade_count", "profit", "profit_factor", "win_rate"]),
            "",
            self._table(result["bear_winner_patterns"]["holding_days_bucket"], ["bucket", "trade_count", "profit", "profit_factor", "win_rate"]),
            "",
            "## Hypotheses",
            "",
            self._table([result["pattern_hypotheses"]], ["bear_winner_common_patterns", "bear_specific_patterns", "bull_only_patterns"]),
            "",
            "## Verdict",
            "",
            self._table([result["verdict"]], ["bear_mode_worth_implementing", "bear_specific_signal_exists", "pm_ai_already_capturing_bear_alpha", "next_phase_recommended"]),
            "",
        ]
        return "\n".join(lines)

    def _log_dir(self) -> Path:
        return self.root / "logs" / "backtests" / self.profile / self.period

    def _load_trades(self) -> pd.DataFrame:
        frame = _read_csv(self._log_dir() / "trades.csv")
        if frame.empty:
            return frame
        frame = frame.copy()
        if "action" in frame.columns:
            frame = frame[frame["action"].fillna("").astype(str).eq("SELL")].copy()
        frame["code"] = frame["code"].map(_code_text)
        for column in ["entry_date", "exit_date"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")
        return frame

    def _load_purchase_audit(self) -> pd.DataFrame:
        frame = _read_csv(self._log_dir() / "purchase_audit.csv")
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["code"] = frame["code"].map(_code_text)
        for column in ["entry_date", "signal_date"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")
        return frame

    def _load_listed_info(self) -> pd.DataFrame:
        loader = JQuantsDataLoader(self.root / "data" / "cache" / "jquants")
        frame = loader.load_listed_info(self.end_date)
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["code"] = frame["code"].map(_code_text)
        return frame.drop_duplicates("code", keep="last")

    def _load_regime(self) -> pd.DataFrame:
        lookback_start = (pd.Timestamp(self.start_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        loader = JQuantsDataLoader(self.root / "data" / "cache" / "jquants")
        topix = loader.load_topix(lookback_start, self.end_date)
        if topix.empty:
            return pd.DataFrame(columns=["date", "regime"])
        topix = topix.copy().dropna(subset=["date", "close"]).sort_values("date")
        topix = topix.drop_duplicates("date", keep="last")
        topix["date"] = pd.to_datetime(topix["date"], errors="coerce")
        topix["close"] = pd.to_numeric(topix["close"], errors="coerce")
        topix["ma25"] = topix["close"].rolling(25, min_periods=1).mean()
        topix["ma75"] = topix["close"].rolling(75, min_periods=1).mean()
        topix["regime"] = topix.apply(self._classify_topix_row, axis=1)
        topix["date"] = topix["date"].dt.strftime("%Y-%m-%d")
        return topix[topix["date"].between(self.start_date, self.end_date)][["date", "regime"]].reset_index(drop=True)

    def _classify_topix_row(self, row: pd.Series) -> str:
        close = pd.to_numeric(pd.Series([row.get("close")]), errors="coerce").iloc[0]
        ma25 = pd.to_numeric(pd.Series([row.get("ma25")]), errors="coerce").iloc[0]
        ma75 = pd.to_numeric(pd.Series([row.get("ma75")]), errors="coerce").iloc[0]
        if pd.isna(close) or pd.isna(ma25) or pd.isna(ma75):
            return "Unknown"
        if close > ma75 and ma25 > ma75:
            return "Bull"
        if close < ma75 and ma25 < ma75:
            return "Bear"
        return "Neutral"

    def _enrich_trades(self, trades: pd.DataFrame, purchase: pd.DataFrame, listed: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        out = trades.copy()
        regime_map = regime.set_index("date")["regime"].to_dict() if not regime.empty else {}
        out["regime"] = out.get("entry_date", pd.Series("", index=out.index)).map(regime_map).fillna("Unknown")
        out["profit"] = pd.to_numeric(out.get("net_profit"), errors="coerce")
        out["return"] = pd.to_numeric(out.get("net_profit_rate"), errors="coerce")
        out["sector"] = out.get("sector_name", "")
        out["buy_date"] = out.get("entry_date", "")
        out["sell_date"] = out.get("exit_date", "")
        out["liquidity"] = out.get("volume_ratio", pd.Series(index=out.index, dtype=float)).map(_bucket_liquidity)
        out["holding_days_bucket"] = out.get("holding_days", pd.Series(index=out.index, dtype=float)).map(_bucket_holding_days)
        out["buy_amount"] = self._buy_amount(out, purchase)
        if not listed.empty and {"code", "scale_category"}.issubset(listed.columns):
            out = out.merge(listed[["code", "scale_category"]], on="code", how="left")
            out["market_cap_band"] = out["scale_category"].fillna("unknown")
        else:
            out["market_cap_band"] = "unknown"
        return out

    def _buy_amount(self, trades: pd.DataFrame, purchase: pd.DataFrame) -> pd.Series:
        fallback = self._numeric_trade_column(trades, "pm_resized_amount")
        if fallback.isna().all():
            fallback = self._numeric_trade_column(trades, "scaled_amount")
        if fallback.isna().all():
            fallback = self._numeric_trade_column(trades, "entry_price") * self._numeric_trade_column(trades, "shares")
        if purchase.empty:
            return fallback
        amount_column = next((col for col in ["final_amount", "pm_resized_amount", "scaled_amount", "planned_amount"] if col in purchase.columns), None)
        if amount_column is None:
            return fallback
        joined = trades[["code", "entry_date"]].merge(
            purchase[["code", "entry_date", amount_column]].drop_duplicates(["code", "entry_date"], keep="last"),
            on=["code", "entry_date"],
            how="left",
        )
        amount = pd.to_numeric(joined[amount_column], errors="coerce")
        return amount.fillna(fallback.reset_index(drop=True)).set_axis(trades.index)

    def _numeric_trade_column(self, trades: pd.DataFrame, column: str) -> pd.Series:
        if column not in trades.columns:
            return pd.Series(pd.NA, index=trades.index, dtype="Float64")
        return pd.to_numeric(trades[column], errors="coerce")

    def _coverage(self, trades: pd.DataFrame) -> dict[str, Any]:
        return {
            "total_trades": int(len(trades)),
            "classified_trades": int(trades["regime"].ne("Unknown").sum()) if not trades.empty and "regime" in trades.columns else 0,
            "bear_trades": int(trades["regime"].eq("Bear").sum()) if not trades.empty and "regime" in trades.columns else 0,
            "bear_winner_trades": int((trades["regime"].eq("Bear") & pd.to_numeric(trades.get("profit"), errors="coerce").gt(0)).sum()) if not trades.empty and "regime" in trades.columns else 0,
            "bull_trades": int(trades["regime"].eq("Bull").sum()) if not trades.empty and "regime" in trades.columns else 0,
        }

    def _bear_trade_details(self, bear: pd.DataFrame) -> list[dict[str, Any]]:
        columns = ["code", "sector", "buy_date", "sell_date", "profit", "return", "holding_days", "pm_score", "pm_multiplier", "buy_amount", "market_cap_band", "liquidity"]
        if bear.empty:
            return []
        return bear.sort_values("profit", ascending=False)[columns].to_dict("records")

    def _top_trades(self, frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        columns = ["code", "sector", "buy_date", "sell_date", "profit", "return", "holding_days", "pm_score", "pm_multiplier", "buy_amount", "market_cap_band", "liquidity"]
        if frame.empty or column not in frame.columns:
            return []
        return frame.sort_values(column, ascending=False).head(50)[columns].to_dict("records")

    def _bull_vs_bear_comparison(self, bull: pd.DataFrame, bear: pd.DataFrame) -> list[dict[str, Any]]:
        return [self._comparison_row("Bull", bull), self._comparison_row("Bear", bear)]

    def _comparison_row(self, label: str, frame: pd.DataFrame) -> dict[str, Any]:
        profits = pd.to_numeric(frame.get("profit"), errors="coerce") if not frame.empty else pd.Series(dtype=float)
        return {
            "regime": label,
            "trade_count": int(len(frame)),
            "profit": float(profits.sum()) if not profits.empty else 0.0,
            "profit_factor": _profit_factor(profits),
            "win_rate": _win_rate(profits),
            "avg_pm_score": _mean(frame.get("pm_score")) if not frame.empty else None,
            "avg_pm_multiplier": _mean(frame.get("pm_multiplier")) if not frame.empty else None,
            "avg_holding_days": _mean(frame.get("holding_days")) if not frame.empty else None,
            "avg_buy_amount": _mean(frame.get("buy_amount")) if not frame.empty else None,
            "top_sector": self._top_bucket(frame, "sector"),
            "top_market_cap_band": self._top_bucket(frame, "market_cap_band"),
            "top_liquidity": self._top_bucket(frame, "liquidity"),
        }

    def _top_bucket(self, frame: pd.DataFrame, column: str) -> str:
        if frame.empty or column not in frame.columns:
            return ""
        counts = frame[column].fillna("unknown").astype(str).value_counts()
        return "" if counts.empty else str(counts.index[0])

    def _bear_winner_patterns(self, bear_winners: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        return {
            "sector": self._bucket_stats(bear_winners, "sector"),
            "pm_multiplier": self._bucket_stats(bear_winners, "pm_multiplier"),
            "market_cap_band": self._bucket_stats(bear_winners, "market_cap_band"),
            "holding_days_bucket": self._bucket_stats(bear_winners, "holding_days_bucket"),
        }

    def _bucket_stats(self, frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        if frame.empty or column not in frame.columns:
            return []
        rows = []
        for bucket, group in frame.groupby(column, dropna=False):
            profits = pd.to_numeric(group.get("profit"), errors="coerce")
            rows.append(
                {
                    "bucket": str(bucket),
                    "trade_count": int(len(group)),
                    "profit": float(profits.sum()) if not profits.empty else 0.0,
                    "profit_factor": _profit_factor(profits),
                    "win_rate": _win_rate(profits),
                }
            )
        return sorted(rows, key=lambda row: (row["profit"], row["trade_count"]), reverse=True)

    def _pattern_hypotheses(self, bull: pd.DataFrame, bear: pd.DataFrame, bear_winners: pd.DataFrame) -> dict[str, Any]:
        bear_top_sector = self._top_bucket(bear_winners, "sector")
        bear_top_pm = self._top_bucket(bear_winners, "pm_multiplier")
        bear_top_cap = self._top_bucket(bear_winners, "market_cap_band")
        bear_specific = []
        if bear_top_sector and bear_top_sector != self._top_bucket(bull, "sector"):
            bear_specific.append(f"sector={bear_top_sector}")
        if bear_top_pm and bear_top_pm != self._top_bucket(bull, "pm_multiplier"):
            bear_specific.append(f"pm_multiplier={bear_top_pm}")
        if bear_top_cap and bear_top_cap != self._top_bucket(bull, "market_cap_band"):
            bear_specific.append(f"market_cap_band={bear_top_cap}")
        common = [item for item in [f"sector={bear_top_sector}" if bear_top_sector else "", f"pm_multiplier={bear_top_pm}" if bear_top_pm else "", f"market_cap_band={bear_top_cap}" if bear_top_cap else ""] if item]
        return {
            "bear_winner_common_patterns": common,
            "bear_specific_patterns": bear_specific,
            "bull_only_patterns": [
                item
                for item in [
                    f"sector={self._top_bucket(bull, 'sector')}" if self._top_bucket(bull, "sector") else "",
                    f"market_cap_band={self._top_bucket(bull, 'market_cap_band')}" if self._top_bucket(bull, "market_cap_band") else "",
                ]
                if item and item not in common
            ],
        }

    def _verdict(self, bull: pd.DataFrame, bear: pd.DataFrame, bear_winners: pd.DataFrame) -> dict[str, Any]:
        bull_pf = _profit_factor(pd.to_numeric(bull.get("profit"), errors="coerce")) or 0.0
        bear_pf = _profit_factor(pd.to_numeric(bear.get("profit"), errors="coerce")) or 0.0
        bear_profit = float(pd.to_numeric(bear.get("profit"), errors="coerce").sum()) if not bear.empty else 0.0
        high_pm_profit = 0.0
        if not bear.empty and "pm_multiplier" in bear.columns:
            high_pm = bear[pd.to_numeric(bear["pm_multiplier"], errors="coerce").ge(1.15)]
            high_pm_profit = float(pd.to_numeric(high_pm.get("profit"), errors="coerce").sum()) if not high_pm.empty else 0.0
        pm_capturing = bear_profit > 0 and high_pm_profit / bear_profit >= 0.35 if bear_profit else False
        bear_signal = bear_pf > bull_pf * 1.25 and len(bear_winners) >= 10
        return {
            "bear_mode_worth_implementing": bool(bear_signal and not pm_capturing),
            "bear_specific_signal_exists": bool(bear_signal),
            "pm_ai_already_capturing_bear_alpha": bool(pm_capturing),
            "next_phase_recommended": "Phase 6-C Bear Signal Design Audit" if bear_signal and not pm_capturing else "Keep v2_78; audit Bear winners before adding a Bear mode",
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


def run_phase6b_bear_market_winner_audit(root: Path | str = ROOT) -> Phase6BReportPaths:
    audit = Phase6BBearMarketWinnerAudit(root)
    return audit.save_report(audit.build_report())
