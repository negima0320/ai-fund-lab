from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_phase3d_detail_audit import (
    PERIOD,
    PHASE3D_PROFILE,
    PortfolioManagerPhase3DDetailAudit,
)
from ml.portfolio_manager_phase3e import PHASE3E_PROFILE
from ml.portfolio_manager_phase3h_capital_utilization import PHASE3H_PROFILE


POST_EXIT_HORIZONS = (1, 3, 5, 10, 20)
HYPOTHETICAL_HOLD_HORIZONS = (3, 5, 10, 20)
EARLY_EXIT_RETURN_THRESHOLD = 0.05
GOOD_EXIT_RETURN_THRESHOLD = 0.01
LOSS_CUT_SUCCESS_THRESHOLD = -0.03
LATE_LOSS_CUT_REALIZED_THRESHOLD = -0.03
LATE_LOSS_CUT_FLAT_THRESHOLD = 0.02


@dataclass(frozen=True)
class PortfolioManagerPhase4APaths:
    markdown: Path
    json: Path


class PortfolioManagerPhase4AExitQualityAudit:
    def __init__(
        self,
        root: str | Path = ".",
        v2_75_profile: str = PHASE3D_PROFILE,
        v2_76_profile: str = PHASE3E_PROFILE,
        v2_77_cap_030_profile: str = PHASE3H_PROFILE,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.period = period
        self.profiles = {
            "v2_75": v2_75_profile,
            "v2_76": v2_76_profile,
            "v2_77_cap_030": v2_77_cap_030_profile,
        }
        self.detail = PortfolioManagerPhase3DDetailAudit(root=self.root, period=period)
        self.report_dir = self.root / "reports" / "ml"
        self.price_dir = self.root / "data" / "raw"

    def build(self) -> dict[str, Any]:
        price_frame = self._load_price_frame()
        trade_rows = []
        profile_trades: dict[str, pd.DataFrame] = {}
        for label, profile in self.profiles.items():
            trades = self._profile_sell_trades(profile)
            audited = self._audit_trades(label, profile, trades, price_frame)
            profile_trades[label] = audited
            trade_rows.extend(self._records(audited))

        trades_frame = pd.DataFrame(trade_rows)
        result = {
            "purpose": "Portfolio Manager AI Phase 4-A Exit AI quality audit",
            "period": self.period,
            "constraints": {
                "api_refetch": False,
                "openai_api": False,
                "historical_predictions_source": "data/ml/walk_forward_predictions/",
                "current_model_historical_regeneration": False,
                "selected_count_in_day_used": False,
                "trading_logic_changed": False,
                "live_order_placement": False,
                "exit_ai_logic_changed": False,
            },
            "profiles": self.profiles,
            "price_source": {
                "path": "data/raw/prices_YYYY-MM-DD.json",
                "available_price_days": int(price_frame["date"].nunique()) if not price_frame.empty else 0,
                "first_price_date": self._date_min(price_frame),
                "last_price_date": self._date_max(price_frame),
            },
            "exit_trades": trade_rows,
            "profile_summary": self._profile_summary(trades_frame),
            "profit_loss_summary": self._profit_loss_summary(trades_frame),
            "pm_multiplier_summary": self._pm_multiplier_summary(trades_frame),
            "holding_days_summary": self._holding_days_summary(trades_frame),
            "hypothetical_hold_summary": self._hypothetical_hold_summary(trades_frame),
            "exit_ai_judgement": {},
        }
        result["exit_ai_judgement"] = self._exit_ai_judgement(result)
        return result

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase4APaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase4a_exit_quality_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return PortfolioManagerPhase4APaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        table = self.detail._table
        lines = [
            "# Portfolio Manager AI Phase 4-A Exit AI Quality Audit",
            "",
            "## Scope",
            "",
            f"- period: `{result['period']}`",
            f"- v2_75: `{result['profiles']['v2_75']}`",
            f"- v2_76: `{result['profiles']['v2_76']}`",
            f"- v2_77 cap 0.30: `{result['profiles']['v2_77_cap_030']}`",
            "- no API refetch",
            "- no OpenAI API",
            "- no current model historical regeneration",
            "- `selected_count_in_day` used: `False`",
            "- no trading logic changes",
            "- no Exit AI logic changes",
            "",
            "## Price Source",
            "",
            table([result["price_source"]], ["path", "available_price_days", "first_price_date", "last_price_date"]),
            "",
            "## Profile Exit Quality Summary",
            "",
            table(
                result["profile_summary"],
                [
                    "profile",
                    "trade_count",
                    "realized_net_profit",
                    "average_realized_return",
                    "median_realized_return",
                    "average_post_exit_return_1d",
                    "average_post_exit_return_3d",
                    "average_post_exit_return_5d",
                    "average_post_exit_return_10d",
                    "average_post_exit_return_20d",
                    "early_exit_count",
                    "early_exit_rate",
                    "good_exit_count",
                    "good_exit_rate",
                    "loss_cut_success_count",
                    "loss_cut_success_rate",
                    "average_missed_profit_20d",
                    "total_estimated_missed_profit_20d",
                ],
            ),
            "",
            "## Profit vs Loss Trade Audit",
            "",
            table(
                result["profit_loss_summary"],
                [
                    "profile",
                    "trade_result_group",
                    "trade_count",
                    "realized_net_profit",
                    "average_realized_return",
                    "average_post_exit_return_5d",
                    "average_post_exit_return_10d",
                    "average_post_exit_return_20d",
                    "early_exit_rate",
                    "good_exit_rate",
                    "loss_cut_success_rate",
                    "late_exit_rate",
                ],
            ),
            "",
            "## PM Multiplier Audit",
            "",
            table(
                result["pm_multiplier_summary"],
                [
                    "profile",
                    "pm_multiplier",
                    "trade_count",
                    "realized_net_profit",
                    "average_holding_days",
                    "average_post_exit_return_5d",
                    "average_post_exit_return_10d",
                    "average_post_exit_return_20d",
                    "early_exit_rate",
                    "good_exit_rate",
                ],
            ),
            "",
            "## Holding Days Audit",
            "",
            table(
                result["holding_days_summary"],
                [
                    "profile",
                    "holding_days_group",
                    "trade_count",
                    "realized_net_profit",
                    "average_realized_return",
                    "average_post_exit_return_5d",
                    "average_post_exit_return_10d",
                    "average_post_exit_return_20d",
                    "early_exit_rate",
                    "good_exit_rate",
                    "loss_cut_success_rate",
                    "late_exit_rate",
                ],
            ),
            "",
            "## Hypothetical Hold Continuation",
            "",
            table(
                result["hypothetical_hold_summary"],
                [
                    "profile",
                    "actual_realized_profit",
                    "hypothetical_profit_hold_3d",
                    "hypothetical_profit_hold_5d",
                    "hypothetical_profit_hold_10d",
                    "hypothetical_profit_hold_20d",
                    "actual_minus_hold_3d",
                    "actual_minus_hold_5d",
                    "actual_minus_hold_10d",
                    "actual_minus_hold_20d",
                ],
            ),
            "",
            "## Exit AI Provisional Judgement",
            "",
            table(result["exit_ai_judgement"], ["profile", "flag", "value", "detail"]),
            "",
            "## Sample Exit Trades",
            "",
            table(
                result["exit_trades"][:80],
                [
                    "profile",
                    "code",
                    "buy_date",
                    "sell_date",
                    "holding_days",
                    "buy_price",
                    "sell_price",
                    "realized_profit",
                    "realized_return",
                    "post_exit_return_1d",
                    "post_exit_return_3d",
                    "post_exit_return_5d",
                    "post_exit_return_10d",
                    "post_exit_return_20d",
                    "max_post_exit_return_20d",
                    "min_post_exit_return_20d",
                    "exit_quality_label",
                    "pm_multiplier",
                ],
            ),
            "",
        ]
        return "\n".join(lines)

    def _profile_sell_trades(self, profile: str) -> pd.DataFrame:
        backtest_dir = self.root / "logs" / "backtests" / profile / self.period
        trades_raw = self.detail._read_csv(backtest_dir / "trades.csv")
        audit = self.detail._read_csv(backtest_dir / "purchase_audit.csv")
        return self.detail._sell_trades_with_pm(trades_raw, audit)

    def _load_price_frame(self) -> pd.DataFrame:
        rows = []
        for path in sorted(self.price_dir.glob("prices_*.json")):
            payload = self.detail._read_json(path)
            for row in payload.get("prices") or []:
                rows.append(
                    {
                        "date": str(row.get("date") or payload.get("date") or path.stem.replace("prices_", "")),
                        "code": str(row.get("code")),
                        "close": row.get("close"),
                    }
                )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["date", "code", "close"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["code"] = frame["code"].astype(str)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        return frame.dropna(subset=["date", "code", "close"]).sort_values(["code", "date"]).reset_index(drop=True)

    def _audit_trades(self, label: str, profile: str, trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()
        data = trades.copy()
        data["profile"] = label
        data["profile_name"] = profile
        data["code"] = data["code"].astype(str)
        data["buy_date"] = pd.to_datetime(data.get("entry_date"), errors="coerce")
        data["sell_date"] = pd.to_datetime(data.get("exit_date"), errors="coerce")
        data["buy_price"] = pd.to_numeric(data.get("entry_price"), errors="coerce")
        data["sell_price"] = pd.to_numeric(data.get("exit_price"), errors="coerce")
        data["shares"] = pd.to_numeric(data.get("shares"), errors="coerce").fillna(0.0)
        data["realized_profit"] = self._numeric_series(data, "net_profit").fillna(self._numeric_series(data, "profit")).fillna(0.0)
        data["realized_return"] = self._numeric_series(data, "net_profit_rate").fillna(self._numeric_series(data, "profit_rate"))
        data["holding_days"] = pd.to_numeric(data.get("holding_days"), errors="coerce")
        data["pm_multiplier"] = pd.to_numeric(data.get("pm_multiplier"), errors="coerce")

        by_code = {code: group.sort_values("date").reset_index(drop=True) for code, group in prices.groupby("code")}
        audited_rows = []
        for row in data.to_dict("records"):
            price_history = by_code.get(str(row.get("code")), pd.DataFrame())
            audited_rows.append(self._audit_trade_row(row, price_history))
        return pd.DataFrame(audited_rows)

    def _audit_trade_row(self, row: dict[str, Any], price_history: pd.DataFrame) -> dict[str, Any]:
        sell_date = row.get("sell_date")
        sell_price = self._float_or_none(row.get("sell_price"))
        buy_price = self._float_or_none(row.get("buy_price"))
        shares = float(row.get("shares") or 0.0)
        future = price_history[price_history["date"] > sell_date].reset_index(drop=True) if not price_history.empty and pd.notna(sell_date) else pd.DataFrame()
        closes = [float(value) for value in future["close"].tolist()] if not future.empty else []

        for horizon in POST_EXIT_HORIZONS:
            row[f"post_exit_return_{horizon}d"] = self._return_at(closes, horizon, sell_price)
        window = closes[:20]
        returns_20d = [(close / sell_price - 1.0) for close in window] if sell_price else []
        row["max_post_exit_return_20d"] = max(returns_20d) if returns_20d else None
        row["min_post_exit_return_20d"] = min(returns_20d) if returns_20d else None
        for horizon in HYPOTHETICAL_HOLD_HORIZONS:
            close = closes[horizon - 1] if len(closes) >= horizon else None
            row[f"hypothetical_profit_hold_{horizon}d"] = (close - buy_price) * shares if close is not None and buy_price is not None else None
            row[f"actual_minus_hold_{horizon}d"] = (
                float(row.get("realized_profit") or 0.0) - row[f"hypothetical_profit_hold_{horizon}d"]
                if row[f"hypothetical_profit_hold_{horizon}d"] is not None
                else None
            )
        row["exit_quality_label"] = self._exit_quality_label(row)
        return row

    def _exit_quality_label(self, row: dict[str, Any]) -> str:
        realized_return = self._float_or_none(row.get("realized_return"))
        post_20d = self._float_or_none(row.get("post_exit_return_20d"))
        max_20d = self._float_or_none(row.get("max_post_exit_return_20d"))
        min_20d = self._float_or_none(row.get("min_post_exit_return_20d"))
        if max_20d is not None and max_20d >= EARLY_EXIT_RETURN_THRESHOLD:
            return "early_exit"
        if realized_return is not None and realized_return < 0 and min_20d is not None and min_20d <= LOSS_CUT_SUCCESS_THRESHOLD:
            return "loss_cut_success"
        if (
            realized_return is not None
            and realized_return <= LATE_LOSS_CUT_REALIZED_THRESHOLD
            and post_20d is not None
            and abs(post_20d) <= LATE_LOSS_CUT_FLAT_THRESHOLD
        ):
            return "late_exit"
        if post_20d is not None and post_20d <= GOOD_EXIT_RETURN_THRESHOLD:
            return "good_exit"
        return "uncertain"

    def _profile_summary(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        return [self._summary_row(profile, group) for profile, group in self._group(trades, "profile")]

    def _profit_loss_summary(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty:
            return []
        data = trades.copy()
        data["trade_result_group"] = data["realized_profit"].apply(lambda value: "profit_trade" if float(value or 0) > 0 else "loss_trade")
        rows = []
        for (profile, group_name), group in data.groupby(["profile", "trade_result_group"]):
            rows.append({"profile": profile, "trade_result_group": group_name, **self._compact_summary(group)})
        return rows

    def _pm_multiplier_summary(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty or "pm_multiplier" not in trades.columns:
            return []
        rows = []
        data = trades.copy()
        data["pm_multiplier"] = pd.to_numeric(data["pm_multiplier"], errors="coerce").round(2)
        for profile in sorted(data["profile"].dropna().unique()):
            profile_data = data[data["profile"].eq(profile)]
            for multiplier in [0.8, 1.0, 1.15, 1.3]:
                group = profile_data[profile_data["pm_multiplier"].eq(multiplier)]
                rows.append(
                    {
                        "profile": profile,
                        "pm_multiplier": multiplier,
                        "trade_count": int(len(group)),
                        "realized_net_profit": self._sum(group, "realized_profit"),
                        "average_holding_days": self._mean(group, "holding_days"),
                        "average_post_exit_return_5d": self._mean(group, "post_exit_return_5d"),
                        "average_post_exit_return_10d": self._mean(group, "post_exit_return_10d"),
                        "average_post_exit_return_20d": self._mean(group, "post_exit_return_20d"),
                        "early_exit_rate": self._label_rate(group, "early_exit"),
                        "good_exit_rate": self._label_rate(group, "good_exit"),
                    }
                )
        return rows

    def _holding_days_summary(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty:
            return []
        data = trades.copy()
        data["holding_days_group"] = data["holding_days"].apply(self._holding_group)
        order = {"1d": 0, "2-3d": 1, "4-5d": 2, "6-10d": 3, "11d+": 4, "unknown": 5}
        rows = []
        for (profile, group_name), group in data.groupby(["profile", "holding_days_group"]):
            rows.append({"profile": profile, "holding_days_group": group_name, **self._compact_summary(group)})
        return sorted(rows, key=lambda row: (row["profile"], order.get(row["holding_days_group"], 99)))

    def _hypothetical_hold_summary(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for profile, group in self._group(trades, "profile"):
            row = {"profile": profile, "actual_realized_profit": self._sum(group, "realized_profit")}
            for horizon in HYPOTHETICAL_HOLD_HORIZONS:
                row[f"hypothetical_profit_hold_{horizon}d"] = self._sum(group, f"hypothetical_profit_hold_{horizon}d")
            for horizon in HYPOTHETICAL_HOLD_HORIZONS:
                row[f"actual_minus_hold_{horizon}d"] = self._sum(group, f"actual_minus_hold_{horizon}d")
            rows.append(row)
        return rows

    def _exit_ai_judgement(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        summaries = {row["profile"]: row for row in result["profile_summary"]}
        holds = {row["profile"]: row for row in result["hypothetical_hold_summary"]}
        pm_rows = pd.DataFrame(result["pm_multiplier_summary"])
        holding_rows = pd.DataFrame(result["holding_days_summary"])
        judgements = []
        for profile, summary in summaries.items():
            hold = holds.get(profile, {})
            actual_minus_5d = float(hold.get("actual_minus_hold_5d") or 0.0)
            actual_minus_10d = float(hold.get("actual_minus_hold_10d") or 0.0)
            post_10d = float(summary.get("average_post_exit_return_10d") or 0.0)
            post_20d = float(summary.get("average_post_exit_return_20d") or 0.0)
            loss_group = next((row for row in result["profit_loss_summary"] if row["profile"] == profile and row["trade_result_group"] == "loss_trade"), {})
            high_pm = pm_rows[(pm_rows["profile"].eq(profile)) & (pm_rows["pm_multiplier"].isin([1.15, 1.3]))] if not pm_rows.empty else pd.DataFrame()
            short_holding = holding_rows[(holding_rows["profile"].eq(profile)) & (holding_rows["holding_days_group"].isin(["1d", "2-3d"]))] if not holding_rows.empty else pd.DataFrame()
            checks = [
                (
                    "exit_ai_likely_effective",
                    actual_minus_5d > 0 or actual_minus_10d > 0 or float(summary.get("loss_cut_success_rate") or 0.0) >= 0.20,
                    f"actual_minus_hold_5d={actual_minus_5d:.2f}, actual_minus_hold_10d={actual_minus_10d:.2f}, loss_cut_success_rate={summary.get('loss_cut_success_rate')}",
                ),
                (
                    "early_exit_problem_suspected",
                    post_10d >= 0.03 or post_20d >= 0.04,
                    f"average_post_exit_return_10d={post_10d:.4f}, average_post_exit_return_20d={post_20d:.4f}",
                ),
                (
                    "late_loss_cut_problem_suspected",
                    float(loss_group.get("average_realized_return") or 0.0) <= -0.03
                    and float(loss_group.get("average_post_exit_return_10d") or 0.0) > -0.01,
                    f"loss_avg_realized_return={loss_group.get('average_realized_return')}, loss_avg_post_10d={loss_group.get('average_post_exit_return_10d')}",
                ),
                (
                    "high_pm_early_exit_suspected",
                    self._weighted_rate(high_pm, "early_exit_rate", "trade_count") >= 0.30,
                    f"pm_1.15_1.30_weighted_early_exit_rate={self._weighted_rate(high_pm, 'early_exit_rate', 'trade_count'):.4f}",
                ),
                (
                    "short_holding_too_aggressive",
                    self._weighted_rate(short_holding, "early_exit_rate", "trade_count") >= 0.30,
                    f"holding_1_to_3d_weighted_early_exit_rate={self._weighted_rate(short_holding, 'early_exit_rate', 'trade_count'):.4f}",
                ),
            ]
            judgements.extend({"profile": profile, "flag": flag, "value": bool(value), "detail": detail} for flag, value, detail in checks)
        return judgements

    def _summary_row(self, profile: str, group: pd.DataFrame) -> dict[str, Any]:
        row = {"profile": profile, **self._compact_summary(group)}
        row["average_missed_profit_20d"] = self._mean(group, "actual_minus_hold_20d") * -1.0
        row["total_estimated_missed_profit_20d"] = self._sum(group, "actual_minus_hold_20d") * -1.0
        return row

    def _compact_summary(self, group: pd.DataFrame) -> dict[str, Any]:
        return {
            "trade_count": int(len(group)),
            "realized_net_profit": self._sum(group, "realized_profit"),
            "average_realized_return": self._mean(group, "realized_return"),
            "median_realized_return": self._median(group, "realized_return"),
            "average_post_exit_return_1d": self._mean(group, "post_exit_return_1d"),
            "average_post_exit_return_3d": self._mean(group, "post_exit_return_3d"),
            "average_post_exit_return_5d": self._mean(group, "post_exit_return_5d"),
            "average_post_exit_return_10d": self._mean(group, "post_exit_return_10d"),
            "average_post_exit_return_20d": self._mean(group, "post_exit_return_20d"),
            "early_exit_count": self._label_count(group, "early_exit"),
            "early_exit_rate": self._label_rate(group, "early_exit"),
            "good_exit_count": self._label_count(group, "good_exit"),
            "good_exit_rate": self._label_rate(group, "good_exit"),
            "loss_cut_success_count": self._label_count(group, "loss_cut_success"),
            "loss_cut_success_rate": self._label_rate(group, "loss_cut_success"),
            "late_exit_count": self._label_count(group, "late_exit"),
            "late_exit_rate": self._label_rate(group, "late_exit"),
        }

    def _records(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        output_columns = [
            "profile",
            "profile_name",
            "code",
            "buy_date",
            "sell_date",
            "holding_days",
            "buy_price",
            "sell_price",
            "shares",
            "realized_profit",
            "realized_return",
            "post_exit_return_1d",
            "post_exit_return_3d",
            "post_exit_return_5d",
            "post_exit_return_10d",
            "post_exit_return_20d",
            "max_post_exit_return_20d",
            "min_post_exit_return_20d",
            "exit_quality_label",
            "pm_multiplier",
            "hypothetical_profit_hold_3d",
            "hypothetical_profit_hold_5d",
            "hypothetical_profit_hold_10d",
            "hypothetical_profit_hold_20d",
            "actual_minus_hold_3d",
            "actual_minus_hold_5d",
            "actual_minus_hold_10d",
            "actual_minus_hold_20d",
        ]
        data = frame[[column for column in output_columns if column in frame.columns]].copy()
        for column in ["buy_date", "sell_date"]:
            data[column] = pd.to_datetime(data[column], errors="coerce").dt.strftime("%Y-%m-%d")
        return data.where(pd.notna(data), None).to_dict("records")

    def _return_at(self, closes: list[float], horizon: int, sell_price: float | None) -> float | None:
        if sell_price is None or sell_price == 0 or len(closes) < horizon:
            return None
        return float(closes[horizon - 1] / sell_price - 1.0)

    def _holding_group(self, value: Any) -> str:
        days = self._float_or_none(value)
        if days is None:
            return "unknown"
        if days <= 1:
            return "1d"
        if days <= 3:
            return "2-3d"
        if days <= 5:
            return "4-5d"
        if days <= 10:
            return "6-10d"
        return "11d+"

    def _group(self, frame: pd.DataFrame, column: str) -> list[tuple[str, pd.DataFrame]]:
        if frame.empty or column not in frame.columns:
            return []
        return [(str(key), group.copy()) for key, group in frame.groupby(column)]

    def _label_count(self, frame: pd.DataFrame, label: str) -> int:
        if frame.empty or "exit_quality_label" not in frame.columns:
            return 0
        return int(frame["exit_quality_label"].astype(str).eq(label).sum())

    def _label_rate(self, frame: pd.DataFrame, label: str) -> float:
        return float(self._label_count(frame, label) / len(frame)) if len(frame) else 0.0

    def _sum(self, frame: pd.DataFrame, column: str) -> float:
        if frame.empty or column not in frame.columns:
            return 0.0
        return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())

    def _mean(self, frame: pd.DataFrame, column: str) -> float | None:
        if frame.empty or column not in frame.columns:
            return None
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    def _median(self, frame: pd.DataFrame, column: str) -> float | None:
        if frame.empty or column not in frame.columns:
            return None
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(values.median()) if not values.empty else None

    def _weighted_rate(self, frame: pd.DataFrame, rate_column: str, weight_column: str) -> float:
        if frame.empty or rate_column not in frame.columns or weight_column not in frame.columns:
            return 0.0
        weights = pd.to_numeric(frame[weight_column], errors="coerce").fillna(0.0)
        rates = pd.to_numeric(frame[rate_column], errors="coerce").fillna(0.0)
        total = float(weights.sum())
        return float((rates * weights).sum() / total) if total else 0.0

    def _numeric_series(self, frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Float64")
        return pd.to_numeric(frame[column], errors="coerce")

    def _float_or_none(self, value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(number):
            return None
        return number

    def _date_min(self, frame: pd.DataFrame) -> str | None:
        if frame.empty:
            return None
        value = frame["date"].min()
        return value.strftime("%Y-%m-%d") if pd.notna(value) else None

    def _date_max(self, frame: pd.DataFrame) -> str | None:
        if frame.empty:
            return None
        value = frame["date"].max()
        return value.strftime("%Y-%m-%d") if pd.notna(value) else None
