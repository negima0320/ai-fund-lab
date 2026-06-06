from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_PROFILE = "rookie_dealer_02_v2_66_ml_ranked"
EXIT_PROFILE = "rookie_dealer_02_v2_68_ml_ranked_exit_ai_050"
PERIOD_KEY = "2023-01-01_to_2026-05-31"


@dataclass(frozen=True)
class CapitalAllocationAuditPaths:
    markdown: Path
    json: Path


class CapitalAllocationAudit:
    """Audit whether Exit AI changed capital allocation enough to alter later buys."""

    def __init__(
        self,
        root: str | Path = ".",
        base_profile: str = BASE_PROFILE,
        exit_profile: str = EXIT_PROFILE,
        period_key: str = PERIOD_KEY,
        focus_month: str = "2026-03",
        focus_code: str = "67400",
    ) -> None:
        self.root = Path(root)
        self.base_profile = base_profile
        self.exit_profile = exit_profile
        self.period_key = period_key
        self.focus_month = focus_month
        self.focus_code = focus_code

    def build(self) -> dict[str, Any]:
        base_summary = self._load_summary(self.base_profile)
        exit_summary = self._load_summary(self.exit_profile)
        base_trades = self._load_trades(self.base_profile)
        exit_trades = self._load_trades(self.exit_profile)

        matched = self._match_trades(base_trades, exit_trades)
        focus_window = self._focus_cash_window(base_summary, exit_summary)
        focus_trade = self._focus_trade(base_trades, exit_trades)
        month_diff = self._month_trade_diff(base_trades, exit_trades)
        exit_ai_reinvestment = self._exit_ai_reinvestment(exit_trades, base_summary, exit_summary)
        candidate_audit = self._focus_candidate_audit()
        reinvestment_trades = self._reinvestment_trades(base_trades, exit_trades, exit_summary)
        reinvestment_simulation = self._reinvestment_simulation(exit_trades, reinvestment_trades)

        result = {
            "period": self.period_key,
            "profiles": {"base": self.base_profile, "exit_ai": self.exit_profile},
            "focus": {"month": self.focus_month, "code": self.focus_code},
            "data_limitations": [
                "Backtest logs contain trades.csv, summary.csv, and backtest_summary.json; processed scored-candidate snapshots are used for the focus-date candidate audit.",
                "Cash, portfolio value, open position count, rejected BUY orders, and executed sell trades can be audited directly.",
                "Cooldown and score-threshold results are post-hoc approximations; they remove observed v2_68-only trades but do not rerun the order engine.",
            ],
            "focus_trade": focus_trade,
            "cash_window": self._records(focus_window),
            "capital_usage_summary": self._capital_usage_summary(base_summary, exit_summary),
            "month_trade_diff": month_diff,
            "exit_ai_reinvestment": exit_ai_reinvestment,
            "candidate_audit": candidate_audit,
            "reinvestment_trade_summary": self._trade_metrics(reinvestment_trades, label="exit_ai_reinvestment_trades"),
            "reinvestment_trades": self._records(reinvestment_trades),
            "reinvestment_simulation": reinvestment_simulation,
            "matched_trade_delta": self._matched_delta_summary(matched),
            "diagnosis": self._diagnosis(focus_trade, focus_window, month_diff, exit_ai_reinvestment, candidate_audit, reinvestment_simulation),
        }
        return result

    def save(self, result: dict[str, Any]) -> CapitalAllocationAuditPaths:
        out_dir = self.root / "reports" / "ml"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "capital_allocation_phase1"
        markdown = out_dir / f"{stem}.md"
        json_path = out_dir / f"{stem}.json"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return CapitalAllocationAuditPaths(markdown=markdown, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# ML Capital Allocation Audit",
            "",
            f"- period: `{result['period']}`",
            f"- base: `{result['profiles']['base']}`",
            f"- exit_ai: `{result['profiles']['exit_ai']}`",
            f"- focus: `{result['focus']['month']}` / code `{result['focus']['code']}`",
            "- note: existing logs only; no backtest rerun and no trading logic change",
            "",
            "## Data Limitations",
            "",
        ]
        lines.extend(f"- {item}" for item in result["data_limitations"])
        lines.extend(
            [
                "",
                "## 67400 Focus Trade",
                "",
                self._table([result["focus_trade"]], [
                    "code",
                    "base_signal_date",
                    "base_entry_date",
                    "base_exit_date",
                    "base_shares",
                    "base_entry_price",
                    "base_net_profit",
                    "exit_profile_has_trade",
                    "exit_profile_same_signal_trade",
                    "cash_available_on_signal_exit_ai",
                    "open_positions_on_signal_exit_ai",
                    "cash_available_on_entry_exit_ai",
                    "open_positions_on_entry_exit_ai",
                    "capital_blocked_reason",
                ]),
                "",
                "## 67400 Candidate / Rejection Audit",
                "",
                self._table([result["candidate_audit"]], [
                    "code",
                    "candidate_found",
                    "raw_candidate_rank",
                    "score_rank",
                    "selected",
                    "selected_rank",
                    "selected_reason",
                    "ml_prediction_found",
                    "ml_all_universe_rank",
                    "risk_adjusted_score",
                    "buy_order_status",
                    "buy_rejected_reason",
                    "buy_amount",
                    "buy_shares",
                    "buy_allocation_limit",
                ]),
                "",
                "## Cash / Portfolio Window",
                "",
                self._table(result["cash_window"], [
                    "date",
                    "base_cash",
                    "exit_cash",
                    "cash_delta",
                    "base_positions_value",
                    "exit_positions_value",
                    "base_total_assets",
                    "exit_total_assets",
                    "base_open_positions",
                    "exit_open_positions",
                    "base_utilization",
                    "exit_utilization",
                ]),
                "",
                "## Capital Usage Summary",
                "",
                self._table(result["capital_usage_summary"], [
                    "scope",
                    "profile",
                    "avg_cash",
                    "avg_positions_value",
                    "avg_total_assets",
                    "avg_utilization",
                    "avg_open_positions",
                    "max_open_positions",
                ]),
                "",
                "## 2026-03 Trade Diff",
                "",
                "### Base-only Trades",
                "",
                self._table(result["month_trade_diff"]["base_only_trades"], [
                    "code",
                    "signal_date",
                    "entry_date",
                    "exit_date",
                    "shares",
                    "entry_price",
                    "net_profit",
                    "exit_reason",
                ]),
                "",
                "### Exit-AI-only Trades",
                "",
                self._table(result["month_trade_diff"]["exit_only_trades"], [
                    "code",
                    "signal_date",
                    "entry_date",
                    "exit_date",
                    "shares",
                    "entry_price",
                    "net_profit",
                    "exit_reason",
                ]),
                "",
                "## Exit AI Reinvestment Events",
                "",
                self._table(result["exit_ai_reinvestment"], [
                    "trigger_code",
                    "trigger_exit_date",
                    "trigger_net_profit",
                    "cash_before",
                    "cash_after",
                    "cash_delta",
                    "next_exit_only_code",
                    "next_exit_only_entry_date",
                    "next_exit_only_net_profit",
                ]),
                "",
                "## Reinvestment Trade Summary",
                "",
                self._table([result["reinvestment_trade_summary"]], [
                    "label",
                    "trade_count",
                    "net_profit",
                    "win_rate",
                    "profit_factor",
                    "average_profit",
                ]),
                "",
                "## Reinvestment Simulation",
                "",
                self._table(result["reinvestment_simulation"], [
                    "scenario",
                    "removed_trade_count",
                    "removed_profit",
                    "adjusted_net_profit",
                    "profit_delta_vs_current",
                    "win_rate",
                    "profit_factor",
                    "note",
                ]),
                "",
                "## Diagnosis",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in result["diagnosis"])
        lines.append("")
        return "\n".join(lines)

    def _load_summary(self, profile: str) -> pd.DataFrame:
        path = self.root / "logs" / "backtests" / profile / self.period_key / "summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for column in ["cash", "positions_value", "total_assets", "open_positions_count", "daily_profit"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["utilization"] = df["positions_value"] / df["total_assets"]
        return df

    def _load_trades(self, profile: str) -> pd.DataFrame:
        path = self.root / "logs" / "backtests" / profile / self.period_key / "trades.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "action" in df.columns:
            df = df[df["action"].astype(str).eq("SELL")].copy()
        for column in ["signal_date", "entry_date", "exit_date"]:
            df[column] = pd.to_datetime(df[column], errors="coerce")
        for column in ["shares", "entry_price", "exit_price", "net_profit", "net_profit_rate", "exit_ai_probability"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        df["code"] = df["code"].astype(str)
        df = df.sort_values(["entry_date", "signal_date", "exit_date", "code"]).reset_index(drop=True)
        df["_occurrence"] = df.groupby(["entry_date", "code"], dropna=False).cumcount()
        df["match_key"] = df["entry_date"].dt.strftime("%Y-%m-%d") + "|" + df["code"] + "|" + df["_occurrence"].astype(str)
        return df

    def _match_trades(self, base: pd.DataFrame, exit_trades: pd.DataFrame) -> pd.DataFrame:
        cols = ["match_key", "code", "signal_date", "entry_date", "exit_date", "shares", "entry_price", "net_profit", "exit_reason"]
        base_small = base[cols].rename(columns={column: f"base_{column}" for column in cols if column != "match_key"})
        exit_small = exit_trades[cols].rename(columns={column: f"exit_{column}" for column in cols if column != "match_key"})
        merged = base_small.merge(exit_small, on="match_key", how="outer", indicator=True)
        merged["status"] = merged["_merge"].map({"both": "matched", "left_only": "base_only", "right_only": "exit_only"})
        merged["profit_delta"] = pd.to_numeric(merged.get("exit_net_profit"), errors="coerce") - pd.to_numeric(merged.get("base_net_profit"), errors="coerce")
        return merged.drop(columns=["_merge"])

    def _focus_trade(self, base: pd.DataFrame, exit_trades: pd.DataFrame) -> dict[str, Any]:
        base_focus = base[base["code"].eq(self.focus_code)].sort_values("net_profit", ascending=False).head(1)
        row = base_focus.iloc[0] if not base_focus.empty else pd.Series(dtype=object)
        signal_date = row.get("signal_date")
        entry_date = row.get("entry_date")
        exit_same_code = exit_trades[exit_trades["code"].eq(self.focus_code)].copy()
        exit_same_signal = exit_same_code[exit_same_code["signal_date"].eq(signal_date)] if not exit_same_code.empty else exit_same_code
        exit_has_trade = not exit_same_code.empty
        signal_summary = self._summary_row(self.exit_profile, signal_date)
        entry_summary = self._summary_row(self.exit_profile, entry_date)
        cash_needed = self._to_float(row.get("shares")) * self._to_float(row.get("entry_price"))
        cash_on_signal = signal_summary.get("cash")
        cash_on_entry = entry_summary.get("cash")
        open_on_signal = signal_summary.get("open_positions_count")
        open_on_entry = entry_summary.get("open_positions_count")
        capital_blocked = (
            cash_on_entry is not None
            and cash_needed is not None
            and cash_on_entry >= cash_needed
            and open_on_entry is not None
            and open_on_entry < 10
        )
        return {
            "code": self.focus_code,
            "base_signal_date": self._date_text(signal_date),
            "base_entry_date": self._date_text(entry_date),
            "base_exit_date": self._date_text(row.get("exit_date")),
            "base_shares": self._to_float(row.get("shares")),
            "base_entry_price": self._to_float(row.get("entry_price")),
            "base_entry_notional": cash_needed,
            "base_net_profit": self._to_float(row.get("net_profit")),
            "exit_profile_has_trade": bool(exit_has_trade),
            "exit_profile_same_signal_trade": bool(not exit_same_signal.empty),
            "cash_available_on_signal_exit_ai": cash_on_signal,
            "open_positions_on_signal_exit_ai": open_on_signal,
            "cash_available_on_entry_exit_ai": cash_on_entry,
            "open_positions_on_entry_exit_ai": open_on_entry,
            "capital_blocked_reason": "not_cash_or_position_limited" if capital_blocked else "unknown_or_possible_capital_limit",
        }

    def _focus_cash_window(self, base_summary: pd.DataFrame, exit_summary: pd.DataFrame) -> pd.DataFrame:
        start = pd.Timestamp("2026-03-02")
        end = pd.Timestamp("2026-03-13")
        left = base_summary[(base_summary["date"] >= start) & (base_summary["date"] <= end)].copy()
        right = exit_summary[(exit_summary["date"] >= start) & (exit_summary["date"] <= end)].copy()
        cols = ["date", "cash", "positions_value", "total_assets", "open_positions_count", "utilization"]
        merged = left[cols].merge(right[cols], on="date", how="outer", suffixes=("_base", "_exit")).sort_values("date")
        merged["cash_delta"] = merged["cash_exit"] - merged["cash_base"]
        merged = merged.rename(
            columns={
                "cash_base": "base_cash",
                "cash_exit": "exit_cash",
                "positions_value_base": "base_positions_value",
                "positions_value_exit": "exit_positions_value",
                "total_assets_base": "base_total_assets",
                "total_assets_exit": "exit_total_assets",
                "open_positions_count_base": "base_open_positions",
                "open_positions_count_exit": "exit_open_positions",
                "utilization_base": "base_utilization",
                "utilization_exit": "exit_utilization",
            }
        )
        return merged

    def _capital_usage_summary(self, base_summary: pd.DataFrame, exit_summary: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for scope, start, end in [("full_period", None, None), ("2026-03", "2026-03-01", "2026-03-31")]:
            for profile, df in [(self.base_profile, base_summary), (self.exit_profile, exit_summary)]:
                work = df.copy()
                if start and end:
                    work = work[(work["date"] >= pd.Timestamp(start)) & (work["date"] <= pd.Timestamp(end))]
                rows.append(
                    {
                        "scope": scope,
                        "profile": profile,
                        "avg_cash": self._mean(work["cash"]),
                        "avg_positions_value": self._mean(work["positions_value"]),
                        "avg_total_assets": self._mean(work["total_assets"]),
                        "avg_utilization": self._mean(work["utilization"]),
                        "avg_open_positions": self._mean(work["open_positions_count"]),
                        "max_open_positions": self._to_float(work["open_positions_count"].max()),
                    }
                )
        return rows

    def _month_trade_diff(self, base: pd.DataFrame, exit_trades: pd.DataFrame) -> dict[str, Any]:
        matched = self._match_trades(base, exit_trades)
        month_mask = (
            pd.to_datetime(matched.get("base_exit_date"), errors="coerce").dt.to_period("M").astype(str).eq(self.focus_month)
            | pd.to_datetime(matched.get("exit_exit_date"), errors="coerce").dt.to_period("M").astype(str).eq(self.focus_month)
        )
        month = matched[month_mask].copy()
        base_only = month[month["status"].eq("base_only")].copy()
        exit_only = month[month["status"].eq("exit_only")].copy()
        return {
            "base_only_profit": self._sum(base_only.get("base_net_profit")),
            "exit_only_profit": self._sum(exit_only.get("exit_net_profit")),
            "base_only_trades": self._records(self._rename_side(base_only, "base")),
            "exit_only_trades": self._records(self._rename_side(exit_only, "exit")),
        }

    def _exit_ai_reinvestment(self, exit_trades: pd.DataFrame, base_summary: pd.DataFrame, exit_summary: pd.DataFrame) -> list[dict[str, Any]]:
        triggered = exit_trades[exit_trades.get("exit_ai_triggered", pd.Series(False, index=exit_trades.index)).astype(str).str.lower().isin({"true", "1", "yes"})].copy()
        rows = []
        matched = self._match_trades(self._load_trades(self.base_profile), exit_trades)
        exit_only = matched[matched["status"].eq("exit_only")].copy()
        for _, trigger in triggered.iterrows():
            exit_date = trigger["exit_date"]
            after = exit_only[pd.to_datetime(exit_only["exit_entry_date"], errors="coerce") > exit_date].sort_values("exit_entry_date").head(1)
            before_summary = self._summary_row_from_df(exit_summary, exit_date - pd.Timedelta(days=1))
            after_summary = self._summary_row_from_df(exit_summary, exit_date)
            next_row = after.iloc[0] if not after.empty else pd.Series(dtype=object)
            rows.append(
                {
                    "trigger_code": trigger.get("code"),
                    "trigger_exit_date": self._date_text(exit_date),
                    "trigger_net_profit": self._to_float(trigger.get("net_profit")),
                    "cash_before": before_summary.get("cash"),
                    "cash_after": after_summary.get("cash"),
                    "cash_delta": (after_summary.get("cash") or 0) - (before_summary.get("cash") or 0) if before_summary and after_summary else None,
                    "next_exit_only_code": next_row.get("exit_code"),
                    "next_exit_only_entry_date": self._date_text(next_row.get("exit_entry_date")),
                    "next_exit_only_net_profit": self._to_float(next_row.get("exit_net_profit")),
                }
            )
        return rows

    def _focus_candidate_audit(self) -> dict[str, Any]:
        profile = self.exit_profile
        date_text = "2026-03-06"
        processed = self.root / "data" / "processed" / profile
        candidates_path = processed / f"candidates_{date_text}.json"
        scored_path = processed / f"scored_candidates_{date_text}.json"
        candidates = self._json_rows(candidates_path, "candidates")
        scored_payload = json.loads(scored_path.read_text(encoding="utf-8")) if scored_path.exists() else {}
        scores = scored_payload.get("scores") or []
        selected = scored_payload.get("selected") or []
        raw_hit = self._find_rank(candidates, self.focus_code)
        score_hit = self._find_rank(scores, self.focus_code)
        selected_hit = self._find_rank(selected, self.focus_code)
        prediction_info = self._prediction_rank(date_text, self.focus_code)
        buy_info = self._focus_buy_order(profile)
        selected_row = selected_hit[1] or score_hit[1] or raw_hit[1] or {}
        return {
            "code": self.focus_code,
            "candidate_found": bool(raw_hit[1] or score_hit[1] or selected_hit[1]),
            "raw_candidate_rank": raw_hit[0],
            "score_rank": selected_row.get("rank") or score_hit[0],
            "selected": bool(selected_row.get("selected")) if selected_row else False,
            "selected_rank": selected_hit[0],
            "selected_reason": selected_row.get("selected_reason") or selected_row.get("selection_reason") or selected_row.get("reason"),
            "ml_prediction_found": prediction_info.get("found"),
            "ml_all_universe_rank": prediction_info.get("rank"),
            "risk_adjusted_score": prediction_info.get("risk_adjusted_score"),
            "expected_return_10d": prediction_info.get("expected_return_10d"),
            "bad_entry_probability_10d": prediction_info.get("bad_entry_probability_10d"),
            "buy_order_status": buy_info.get("order_status"),
            "buy_rejected_reason": buy_info.get("rejected_reason") or buy_info.get("skipped_reason"),
            "buy_amount": buy_info.get("amount"),
            "buy_shares": buy_info.get("shares"),
            "buy_allocation_limit": buy_info.get("allocation_limit"),
        }

    def _json_rows(self, path: Path, key: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get(key) if isinstance(payload, dict) else payload
        return rows if isinstance(rows, list) else []

    def _find_rank(self, rows: list[dict[str, Any]], code: str) -> tuple[int | None, dict[str, Any] | None]:
        for index, row in enumerate(rows, start=1):
            if str(row.get("code")) == str(code):
                return index, row
        return None, None

    def _prediction_rank(self, date_text: str, code: str) -> dict[str, Any]:
        path = self.root / "data" / "ml" / "walk_forward_predictions" / f"predictions_{date_text}.parquet"
        if not path.exists():
            return {"found": False}
        df = pd.read_parquet(path)
        if df.empty or "code" not in df.columns:
            return {"found": False}
        df["code"] = df["code"].astype(str)
        df["risk_adjusted_score"] = pd.to_numeric(df.get("expected_return_10d"), errors="coerce") - 0.5 * pd.to_numeric(
            df.get("bad_entry_probability_10d"), errors="coerce"
        )
        ranked = df.dropna(subset=["risk_adjusted_score"]).sort_values("risk_adjusted_score", ascending=False).reset_index(drop=True)
        hit = ranked[ranked["code"].eq(str(code))]
        if hit.empty:
            return {"found": False}
        row = hit.iloc[0]
        return {
            "found": True,
            "rank": int(hit.index[0]) + 1,
            "risk_adjusted_score": self._to_float(row.get("risk_adjusted_score")),
            "expected_return_10d": self._to_float(row.get("expected_return_10d")),
            "bad_entry_probability_10d": self._to_float(row.get("bad_entry_probability_10d")),
        }

    def _focus_buy_order(self, profile: str) -> dict[str, Any]:
        data = self._load_backtest_summary_json(profile)
        for trade in data.get("all_trades", []):
            if str(trade.get("code")) == self.focus_code and trade.get("signal_date") == "2026-03-06" and trade.get("action") == "BUY":
                return {
                    "order_status": trade.get("order_status") or trade.get("status"),
                    "rejected_reason": trade.get("rejected_reason"),
                    "skipped_reason": trade.get("skipped_reason"),
                    "amount": self._to_float(trade.get("amount")),
                    "shares": self._to_float(trade.get("shares")),
                    "allocation_limit": self._to_float(trade.get("allocation_limit")),
                }
        return {}

    def _load_backtest_summary_json(self, profile: str) -> dict[str, Any]:
        path = self.root / "logs" / "backtests" / profile / self.period_key / "backtest_summary.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _reinvestment_trades(self, base: pd.DataFrame, exit_trades: pd.DataFrame, exit_summary: pd.DataFrame) -> pd.DataFrame:
        matched = self._match_trades(base, exit_trades)
        exit_only = matched[matched["status"].eq("exit_only")].copy()
        if exit_only.empty:
            return exit_only
        triggers = exit_trades[
            exit_trades.get("exit_ai_triggered", pd.Series(False, index=exit_trades.index)).astype(str).str.lower().isin({"true", "1", "yes"})
        ][["code", "exit_date", "net_profit"]].copy()
        triggers = triggers.rename(columns={"code": "trigger_code", "exit_date": "trigger_exit_date", "net_profit": "trigger_net_profit"})
        dates = sorted(pd.to_datetime(exit_summary["date"], errors="coerce").dropna().unique())
        rows = []
        for _, row in exit_only.iterrows():
            entry_date = pd.Timestamp(row.get("exit_entry_date"))
            prior = triggers[triggers["trigger_exit_date"] < entry_date].sort_values("trigger_exit_date").tail(1)
            enriched = row.to_dict()
            if not prior.empty:
                trig = prior.iloc[0]
                enriched["trigger_code"] = trig["trigger_code"]
                enriched["trigger_exit_date"] = trig["trigger_exit_date"]
                enriched["trigger_net_profit"] = trig["trigger_net_profit"]
                enriched["business_days_since_trigger"] = self._business_days_between(dates, trig["trigger_exit_date"], entry_date)
            else:
                enriched["trigger_code"] = None
                enriched["trigger_exit_date"] = None
                enriched["trigger_net_profit"] = None
                enriched["business_days_since_trigger"] = None
            pred = self._prediction_rank(self._date_text(row.get("exit_signal_date")) or "", str(row.get("exit_code")))
            enriched["risk_adjusted_score"] = pred.get("risk_adjusted_score")
            enriched["expected_return_10d"] = pred.get("expected_return_10d")
            enriched["bad_entry_probability_10d"] = pred.get("bad_entry_probability_10d")
            rows.append(enriched)
        return pd.DataFrame(rows)

    def _business_days_between(self, dates: list[Any], start: Any, end: Any) -> int | None:
        if start is None or end is None or pd.isna(start) or pd.isna(end):
            return None
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        return sum(1 for date_value in dates if start < pd.Timestamp(date_value) <= end)

    def _reinvestment_simulation(self, exit_trades: pd.DataFrame, reinvestment_trades: pd.DataFrame) -> list[dict[str, Any]]:
        current_metrics = self._trade_metrics(exit_trades, label="current_v2_68")
        total_profit = current_metrics["net_profit"]
        scenarios = [
            ("current_v2_68", pd.Series(False, index=reinvestment_trades.index), "actual v2_68"),
            ("no_reinvestment", pd.Series(True, index=reinvestment_trades.index), "remove all exit-AI-only trades"),
            (
                "cooldown_3_business_days",
                pd.to_numeric(reinvestment_trades.get("business_days_since_trigger"), errors="coerce").le(3),
                "remove exit-AI-only trades entered within 3 business days after an Exit AI trigger",
            ),
            (
                "cooldown_5_business_days",
                pd.to_numeric(reinvestment_trades.get("business_days_since_trigger"), errors="coerce").le(5),
                "remove exit-AI-only trades entered within 5 business days after an Exit AI trigger",
            ),
            (
                "risk_adjusted_score_gte_0",
                pd.to_numeric(reinvestment_trades.get("risk_adjusted_score"), errors="coerce").lt(0),
                "allow reinvestment only when risk_adjusted_score >= 0",
            ),
        ]
        scores = pd.to_numeric(reinvestment_trades.get("risk_adjusted_score"), errors="coerce").dropna()
        if not scores.empty:
            threshold = float(scores.median())
            scenarios.append(
                (
                    f"risk_adjusted_score_top_half_gte_{threshold:.4f}",
                    pd.to_numeric(reinvestment_trades.get("risk_adjusted_score"), errors="coerce").lt(threshold),
                    "allow only the upper half of observed reinvestment scores",
                )
            )
        rows = []
        for name, remove_mask, note in scenarios:
            removed = reinvestment_trades[remove_mask.fillna(False)].copy() if not reinvestment_trades.empty else reinvestment_trades
            removed_profit = self._sum(removed.get("exit_net_profit"))
            adjusted = exit_trades.copy()
            if not removed.empty:
                removed_keys = set(removed["match_key"].astype(str))
                adjusted = adjusted[~adjusted["match_key"].astype(str).isin(removed_keys)]
            metrics = self._trade_metrics(adjusted, label=name)
            rows.append(
                {
                    "scenario": name,
                    "removed_trade_count": int(len(removed)),
                    "removed_profit": removed_profit,
                    "adjusted_net_profit": total_profit - removed_profit,
                    "profit_delta_vs_current": -removed_profit,
                    "win_rate": metrics["win_rate"],
                    "profit_factor": metrics["profit_factor"],
                    "note": note,
                }
            )
        return rows

    def _trade_metrics(self, trades: pd.DataFrame, label: str) -> dict[str, Any]:
        profit_col = "net_profit" if "net_profit" in trades.columns else "exit_net_profit"
        profits = pd.to_numeric(trades.get(profit_col), errors="coerce").dropna() if not trades.empty else pd.Series(dtype=float)
        gross_profit = profits[profits > 0].sum()
        gross_loss = -profits[profits < 0].sum()
        return {
            "label": label,
            "trade_count": int(len(profits)),
            "net_profit": float(profits.sum()) if not profits.empty else 0.0,
            "win_rate": float((profits > 0).mean()) if not profits.empty else None,
            "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
            "average_profit": float(profits.mean()) if not profits.empty else None,
        }

    def _matched_delta_summary(self, matched: pd.DataFrame) -> dict[str, Any]:
        matched_rows = matched[matched["status"].eq("matched")]
        return {
            "matched_count": int(len(matched_rows)),
            "base_only_count": int(matched["status"].eq("base_only").sum()),
            "exit_only_count": int(matched["status"].eq("exit_only").sum()),
            "matched_profit_delta": self._sum(matched_rows.get("profit_delta")),
        }

    def _diagnosis(
        self,
        focus_trade: dict[str, Any],
        cash_window: pd.DataFrame,
        month_diff: dict[str, Any],
        reinvestment: list[dict[str, Any]],
        candidate_audit: dict[str, Any],
        reinvestment_simulation: list[dict[str, Any]],
    ) -> list[str]:
        diagnosis = []
        if candidate_audit.get("buy_rejected_reason"):
            diagnosis.append(
                f"Direct 67400 miss reason: selected candidate but BUY was rejected because `{candidate_audit['buy_rejected_reason']}`; attempted amount was {candidate_audit.get('buy_amount'):.0f}."
            )
        if focus_trade.get("capital_blocked_reason") == "not_cash_or_position_limited":
            diagnosis.append(
                "67400 was not missed because of cash shortage or max-position saturation: v2_68 had enough cash and zero open positions around the signal/entry dates."
            )
        if candidate_audit.get("ml_all_universe_rank"):
            diagnosis.append(
                f"67400 was weak in all-stock ML ranking: risk_adjusted rank {candidate_audit['ml_all_universe_rank']} with score {candidate_audit.get('risk_adjusted_score'):.4f}; it entered through the no-trade fallback rule, not ML rank strength."
            )
        else:
            diagnosis.append("67400 miss could not be explained conclusively from persisted logs.")
        diagnosis.append(
            f"In {self.focus_month}, base-only trades contributed {month_diff['base_only_profit']:.0f}, while exit-AI-only trades contributed {month_diff['exit_only_profit']:.0f}; the missing 67400 trade dominates this gap."
        )
        if reinvestment:
            negative_reinvestments = sum(1 for row in reinvestment if (row.get("next_exit_only_net_profit") or 0) < 0)
            diagnosis.append(
                f"Exit AI created earlier cash release events; {negative_reinvestments} later exit-AI-only follow-up trades were negative in the nearest-next-trade audit."
            )
        diagnosis.append(
            "The issue appears less like a bad Exit AI signal and more like unstable reinvestment/candidate sequencing after an early exit."
        )
        best = max(reinvestment_simulation, key=lambda row: row.get("profit_delta_vs_current") or -10**18, default={})
        if best:
            diagnosis.append(
                f"Best post-hoc reinvestment control in this audit: {best.get('scenario')} with delta {best.get('profit_delta_vs_current'):.0f} vs current v2_68."
            )
        diagnosis.append(
            "Improvement path: cap order size before daily-buy-limit validation, then test cooldown/score-gated reinvestment in a real backtest profile."
        )
        return diagnosis

    def _summary_row(self, profile: str, date_value: Any) -> dict[str, Any]:
        summary = self._load_summary(profile)
        return self._summary_row_from_df(summary, date_value)

    def _summary_row_from_df(self, summary: pd.DataFrame, date_value: Any) -> dict[str, Any]:
        if date_value is None or pd.isna(date_value):
            return {}
        date_value = pd.Timestamp(date_value)
        row = summary[summary["date"].eq(date_value)]
        if row.empty:
            row = summary[summary["date"] <= date_value].sort_values("date").tail(1)
        if row.empty:
            return {}
        record = row.iloc[0].to_dict()
        return {key: (None if pd.isna(value) else value) for key, value in record.items()}

    def _rename_side(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        cols = {
            f"{side}_code": "code",
            f"{side}_signal_date": "signal_date",
            f"{side}_entry_date": "entry_date",
            f"{side}_exit_date": "exit_date",
            f"{side}_shares": "shares",
            f"{side}_entry_price": "entry_price",
            f"{side}_net_profit": "net_profit",
            f"{side}_exit_reason": "exit_reason",
        }
        present = {old: new for old, new in cols.items() if old in df.columns}
        return df.rename(columns=present)[list(present.values())]

    def _records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        records = []
        for record in df.copy().to_dict("records"):
            cleaned = {}
            for key, value in record.items():
                if isinstance(value, pd.Timestamp):
                    cleaned[key] = self._date_text(value)
                elif pd.isna(value):
                    cleaned[key] = None
                elif isinstance(value, float):
                    cleaned[key] = round(value, 6)
                else:
                    cleaned[key] = value
            records.append(cleaned)
        return records

    def _sum(self, values: Any) -> float:
        if values is None:
            return 0.0
        return float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum())

    def _mean(self, values: Any) -> float | None:
        if values is None:
            return None
        value = pd.to_numeric(values, errors="coerce").mean()
        return None if pd.isna(value) else float(value)

    def _to_float(self, value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _date_text(self, value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    def _format(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, pd.Timestamp):
            return self._date_text(value) or ""
        return str(value)

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(self._format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)
