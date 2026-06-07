from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_phase3d_detail_audit import PERIOD
from ml.portfolio_manager_phase4a_exit_quality import (
    EARLY_EXIT_RETURN_THRESHOLD,
    GOOD_EXIT_RETURN_THRESHOLD,
    PortfolioManagerPhase4AExitQualityAudit,
)


TARGET_PROFILE = "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
TARGET_LABEL = "v2_78_w025"
MIN_HOLD_DAYS = (3, 5, 7)
HIGH_PM_MULTIPLIERS = (1.15, 1.30)


@dataclass(frozen=True)
class PortfolioManagerPhase4BPaths:
    markdown: Path
    json: Path


class PortfolioManagerPhase4BHighPMMinHoldAudit:
    def __init__(
        self,
        root: str | Path = ".",
        profile: str = TARGET_PROFILE,
        label: str = TARGET_LABEL,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.profile = profile
        self.label = label
        self.period = period
        self.report_dir = self.root / "reports" / "ml"
        self.phase4a = PortfolioManagerPhase4AExitQualityAudit(root=self.root, period=period)

    def build(self) -> dict[str, Any]:
        price_frame = self.phase4a._load_price_frame()
        trades = self.phase4a._profile_sell_trades(self.profile)
        audited = self.phase4a._audit_trades(self.label, self.profile, trades, price_frame)
        virtual_rows = self._virtual_min_hold_rows(audited, price_frame)
        result = {
            "purpose": "Portfolio Manager AI Phase 4-B high PM minimum hold audit",
            "period": self.period,
            "constraints": {
                "api_refetch": False,
                "openai_api": False,
                "historical_predictions_source": "existing backtest logs and data/raw/prices_YYYY-MM-DD.json",
                "current_model_historical_regeneration": False,
                "selected_count_in_day_used": False,
                "trading_logic_changed": False,
                "new_profile_added": False,
                "full_backtest_executed": False,
                "exit_ai_logic_changed": False,
                "live_order_placement": False,
            },
            "profile": self.profile,
            "label": self.label,
            "price_source": {
                "path": "data/raw/prices_YYYY-MM-DD.json",
                "available_price_days": int(price_frame["date"].nunique()) if not price_frame.empty else 0,
                "first_price_date": self.phase4a._date_min(price_frame),
                "last_price_date": self.phase4a._date_max(price_frame),
            },
            "pm_multiplier_exit_quality": self._pm_multiplier_exit_quality(audited),
            "high_pm_actual_summary": self._actual_high_pm_summary(audited),
            "minimum_hold_simulation": self._minimum_hold_summary(virtual_rows),
            "recommendations": [],
            "sample_virtual_trades": self._records(pd.DataFrame(virtual_rows).head(80)),
        }
        result["recommendations"] = self._recommendations(result)
        return result

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase4BPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase4b_high_pm_min_hold_audit_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerPhase4BPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 4-B High PM Minimum Hold Audit",
            "",
            "## Scope",
            "",
            f"- period: `{result['period']}`",
            f"- profile: `{result['profile']}`",
            "- audit only; no profile changes / no full backtest",
            "- no API refetch / no OpenAI API / no current model historical regeneration",
            "- `selected_count_in_day` used: `False`",
            "",
            "## Price Source",
            "",
            self._table(
                [result["price_source"]],
                ["path", "available_price_days", "first_price_date", "last_price_date"],
            ),
            "",
            "## PM Multiplier Exit Quality",
            "",
            self._table(
                result["pm_multiplier_exit_quality"],
                [
                    "pm_multiplier",
                    "trade_count",
                    "realized_net_profit",
                    "average_holding_days",
                    "median_holding_days",
                    "average_post_exit_return_5d",
                    "average_post_exit_return_10d",
                    "average_post_exit_return_20d",
                    "early_exit_rate",
                    "good_exit_rate",
                ],
            ),
            "",
            "## High PM Actual Baseline",
            "",
            self._table(
                [result["high_pm_actual_summary"]],
                [
                    "trade_count",
                    "actual_net_profit",
                    "actual_profit_factor",
                    "actual_win_rate",
                    "actual_average_return",
                    "average_holding_days",
                    "early_exit_rate",
                    "good_exit_rate",
                ],
            ),
            "",
            "## High PM Minimum Hold Simulation",
            "",
            self._table(
                result["minimum_hold_simulation"],
                [
                    "minimum_hold_days",
                    "eligible_high_pm_trades",
                    "changed_trade_count",
                    "actual_net_profit",
                    "virtual_net_profit",
                    "profit_delta",
                    "virtual_profit_factor",
                    "virtual_win_rate",
                    "virtual_average_return",
                    "virtual_dd_estimated",
                    "price_missing_count",
                ],
            ),
            "",
            "## Recommendation Flags",
            "",
            self._table(result["recommendations"], ["flag", "value", "detail"]),
            "",
            "## Sample Virtual Trades",
            "",
            self._table(
                result["sample_virtual_trades"],
                [
                    "code",
                    "buy_date",
                    "actual_sell_date",
                    "minimum_hold_days",
                    "virtual_sell_date",
                    "holding_days",
                    "actual_profit",
                    "virtual_profit",
                    "profit_delta",
                    "pm_multiplier",
                ],
            ),
            "",
        ]
        return "\n".join(lines)

    def _pm_multiplier_exit_quality(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for multiplier in [0.8, 1.0, 1.15, 1.3]:
            group = trades[pd.to_numeric(trades.get("pm_multiplier"), errors="coerce").round(2).eq(multiplier)]
            rows.append(
                {
                    "pm_multiplier": multiplier,
                    "trade_count": int(len(group)),
                    "realized_net_profit": self._sum(group, "realized_profit"),
                    "average_holding_days": self._mean(group, "holding_days"),
                    "median_holding_days": self._median(group, "holding_days"),
                    "average_post_exit_return_5d": self._mean(group, "post_exit_return_5d"),
                    "average_post_exit_return_10d": self._mean(group, "post_exit_return_10d"),
                    "average_post_exit_return_20d": self._mean(group, "post_exit_return_20d"),
                    "early_exit_rate": self._label_rate(group, "early_exit"),
                    "good_exit_rate": self._label_rate(group, "good_exit"),
                }
            )
        return rows

    def _actual_high_pm_summary(self, trades: pd.DataFrame) -> dict[str, Any]:
        high_pm = self._high_pm(trades)
        return {
            "trade_count": int(len(high_pm)),
            "actual_net_profit": self._sum(high_pm, "realized_profit"),
            "actual_profit_factor": self._profit_factor(high_pm, "realized_profit"),
            "actual_win_rate": self._win_rate(high_pm, "realized_profit"),
            "actual_average_return": self._mean(high_pm, "realized_return"),
            "average_holding_days": self._mean(high_pm, "holding_days"),
            "early_exit_rate": self._label_rate(high_pm, "early_exit"),
            "good_exit_rate": self._label_rate(high_pm, "good_exit"),
        }

    def _virtual_min_hold_rows(self, trades: pd.DataFrame, prices: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty:
            return []
        by_code = {code: group.sort_values("date").reset_index(drop=True) for code, group in prices.groupby("code")}
        rows: list[dict[str, Any]] = []
        for trade in self._high_pm(trades).to_dict("records"):
            price_history = by_code.get(str(trade.get("code")), pd.DataFrame())
            for minimum_hold in MIN_HOLD_DAYS:
                rows.append(self._virtual_trade_row(trade, price_history, minimum_hold))
        return rows

    def _virtual_trade_row(self, trade: dict[str, Any], price_history: pd.DataFrame, minimum_hold: int) -> dict[str, Any]:
        buy_date = pd.to_datetime(trade.get("buy_date"), errors="coerce")
        holding_days = self._float_or_none(trade.get("holding_days")) or 0.0
        buy_price = self._float_or_none(trade.get("buy_price"))
        actual_sell_price = self._float_or_none(trade.get("sell_price"))
        actual_profit = float(trade.get("realized_profit") or 0.0)
        shares = float(trade.get("shares") or 0.0)
        changed = bool(holding_days < minimum_hold)
        virtual_sell_date = trade.get("sell_date")
        virtual_sell_price = actual_sell_price
        price_missing = False
        if changed:
            future = price_history[price_history["date"] > buy_date].reset_index(drop=True) if not price_history.empty and pd.notna(buy_date) else pd.DataFrame()
            if len(future) >= minimum_hold:
                virtual_sell_date = future.loc[minimum_hold - 1, "date"]
                virtual_sell_price = self._float_or_none(future.loc[minimum_hold - 1, "close"])
            else:
                price_missing = True
                changed = False
        if changed:
            virtual_profit = (virtual_sell_price - buy_price) * shares if virtual_sell_price is not None and buy_price is not None else None
            virtual_return = (virtual_sell_price / buy_price - 1.0) if virtual_sell_price is not None and buy_price not in (None, 0) else None
        else:
            virtual_profit = actual_profit
            virtual_return = self._float_or_none(trade.get("realized_return"))
        return {
            "code": str(trade.get("code")),
            "buy_date": self._date_str(trade.get("buy_date")),
            "actual_sell_date": self._date_str(trade.get("sell_date")),
            "minimum_hold_days": minimum_hold,
            "virtual_sell_date": self._date_str(virtual_sell_date),
            "holding_days": holding_days,
            "actual_profit": actual_profit,
            "actual_return": self._float_or_none(trade.get("realized_return")),
            "virtual_profit": virtual_profit,
            "virtual_return": virtual_return,
            "profit_delta": (virtual_profit - actual_profit) if virtual_profit is not None else None,
            "pm_multiplier": self._float_or_none(trade.get("pm_multiplier")),
            "changed_by_min_hold": changed,
            "price_missing": price_missing,
        }

    def _minimum_hold_summary(self, virtual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frame = pd.DataFrame(virtual_rows)
        rows = []
        for minimum_hold in MIN_HOLD_DAYS:
            group = frame[frame["minimum_hold_days"].eq(minimum_hold)] if not frame.empty else pd.DataFrame()
            rows.append(
                {
                    "minimum_hold_days": minimum_hold,
                    "eligible_high_pm_trades": int(len(group)),
                    "changed_trade_count": int(group.get("changed_by_min_hold", pd.Series(dtype=bool)).fillna(False).sum()) if not group.empty else 0,
                    "actual_net_profit": self._sum(group, "actual_profit"),
                    "virtual_net_profit": self._sum(group, "virtual_profit"),
                    "profit_delta": self._sum(group, "profit_delta"),
                    "virtual_profit_factor": self._profit_factor(group, "virtual_profit"),
                    "virtual_win_rate": self._win_rate(group, "virtual_profit"),
                    "virtual_average_return": self._mean(group, "virtual_return"),
                    "virtual_dd_estimated": None,
                    "price_missing_count": int(group.get("price_missing", pd.Series(dtype=bool)).fillna(False).sum()) if not group.empty else 0,
                }
            )
        return rows

    def _recommendations(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        actual = result["high_pm_actual_summary"]
        actual_pf = actual.get("actual_profit_factor")
        rows = []
        for summary in result["minimum_hold_simulation"]:
            days = int(summary["minimum_hold_days"])
            delta = float(summary.get("profit_delta") or 0.0)
            virtual_pf = summary.get("virtual_profit_factor")
            changed = int(summary.get("changed_trade_count") or 0)
            recommended = changed > 0 and delta > 0 and (virtual_pf is None or actual_pf is None or virtual_pf >= actual_pf * 0.95)
            rows.append(
                {
                    "flag": f"high_pm_min_hold_{days}d_recommended",
                    "value": bool(recommended),
                    "detail": (
                        f"changed_trades={changed}, profit_delta={delta:.2f}, "
                        f"actual_pf={actual_pf}, virtual_pf={virtual_pf}; DD is not estimated in this lightweight audit"
                    ),
                }
            )
        rows.append(
            {
                "flag": "dd_improvement_estimated",
                "value": False,
                "detail": "This audit does not replay portfolio equity, so DD impact is a follow-up backtest item.",
            }
        )
        return rows

    def _high_pm(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty or "pm_multiplier" not in trades.columns:
            return pd.DataFrame()
        multipliers = pd.to_numeric(trades["pm_multiplier"], errors="coerce").round(2)
        return trades[multipliers.isin(HIGH_PM_MULTIPLIERS)].copy()

    def _records(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        return frame.where(pd.notna(frame), None).to_dict("records")

    def _table(self, rows: list[dict[str, Any]] | dict[str, Any], columns: list[str]) -> str:
        if isinstance(rows, dict):
            rows = [rows]
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(self._fmt(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _fmt(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return "None"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _profit_factor(self, frame: pd.DataFrame, profit_column: str) -> float | None:
        if frame.empty or profit_column not in frame.columns:
            return None
        values = pd.to_numeric(frame[profit_column], errors="coerce").dropna()
        gross_profit = float(values[values > 0].sum())
        gross_loss = float(-values[values < 0].sum())
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return gross_profit / gross_loss

    def _win_rate(self, frame: pd.DataFrame, profit_column: str) -> float | None:
        if frame.empty or profit_column not in frame.columns:
            return None
        values = pd.to_numeric(frame[profit_column], errors="coerce").dropna()
        return float((values > 0).mean()) if not values.empty else None

    def _label_rate(self, frame: pd.DataFrame, label: str) -> float:
        return self.phase4a._label_rate(frame, label)

    def _sum(self, frame: pd.DataFrame, column: str) -> float:
        return self.phase4a._sum(frame, column)

    def _mean(self, frame: pd.DataFrame, column: str) -> float | None:
        return self.phase4a._mean(frame, column)

    def _median(self, frame: pd.DataFrame, column: str) -> float | None:
        return self.phase4a._median(frame, column)

    def _float_or_none(self, value: Any) -> float | None:
        return self.phase4a._float_or_none(value)

    def _date_str(self, value: Any) -> str | None:
        timestamp = pd.to_datetime(value, errors="coerce")
        return timestamp.strftime("%Y-%m-%d") if pd.notna(timestamp) else None
