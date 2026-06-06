from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROFILE = "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue"
PERIOD = "2023-01-01_to_2026-05-31"

ML_COLUMNS = [
    "risk_adjusted_score",
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "turnover_value",
]


@dataclass(frozen=True)
class PortfolioManagerPhase1Paths:
    markdown: Path
    json: Path
    daily_allocations_csv: Path
    trade_allocations_csv: Path


class PortfolioManagerPhase1Simulation:
    def __init__(
        self,
        root: str | Path = ".",
        profile: str = PROFILE,
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        initial_cash: float = 1_000_000.0,
    ) -> None:
        self.root = Path(root)
        self.profile = profile
        self.start_date = start_date
        self.end_date = end_date
        self.period_key = f"{start_date}_to_{end_date}"
        self.initial_cash = float(initial_cash)
        self.report_dir = self.root / "reports" / "ml"
        self.prediction_dir = self.root / "data" / "ml" / "walk_forward_predictions"

    def build(self) -> dict[str, Any]:
        candidates = self.build_candidate_set()
        trades = candidates[(candidates["decision"].isin(["BUY", "SCALED_BUY"])) & candidates["actual_net_profit"].notna()].copy()
        summary: list[dict[str, Any]] = []
        daily_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        for rule_id, rule_name in self.rules():
            rule_trades, rule_daily = self._simulate_rule(trades, candidates, rule_id, rule_name)
            summary.append(self._summary(rule_id, rule_name, rule_trades))
            daily_rows.extend(rule_daily.to_dict(orient="records"))
            trade_rows.extend(rule_trades.to_dict(orient="records"))
        return {
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "profile": self.profile,
            "candidate_rows": int(len(candidates)),
            "closed_trade_rows": int(len(trades)),
            "summary": summary,
            "daily_allocations": daily_rows,
            "trade_allocations": trade_rows,
            "best_by_net_profit": self._best(summary, "adjusted_net_profit"),
            "best_by_profit_factor": self._best(summary, "profit_factor"),
            "best_by_drawdown": self._best(summary, "max_drawdown"),
            "diagnosis": self._diagnosis(summary),
        }

    def build_candidate_set(self) -> pd.DataFrame:
        backtest_dir = self.root / "logs" / "backtests" / self.profile / self.period_key
        audit_path = backtest_dir / "purchase_audit.csv"
        trades_path = backtest_dir / "trades.csv"
        if not audit_path.exists():
            return pd.DataFrame()
        audit = pd.read_csv(audit_path)
        audit["code"] = audit["code"].astype(str)
        for column in ["signal_date", "entry_date"]:
            if column in audit.columns:
                audit[column] = pd.to_datetime(audit[column], errors="coerce")
        for column in [
            "risk_adjusted_score",
            "expected_return_10d",
            "bad_entry_probability_10d",
            "final_amount",
            "cash_before",
            "cash_after",
            "daily_buy_limit_remaining_before",
            "daily_buy_limit_remaining_after",
            "max_positions_remaining_before",
            "candidate_rank",
            "score_rank",
        ]:
            if column in audit.columns:
                audit[column] = pd.to_numeric(audit[column], errors="coerce")
        if "candidate_source" not in audit.columns:
            audit["candidate_source"] = "selected"
        audit["candidate_source"] = audit["candidate_source"].fillna("selected").replace("", "selected")
        audit = self._join_predictions(audit)
        if "risk_adjusted_score" not in audit.columns or audit["risk_adjusted_score"].isna().any():
            audit["risk_adjusted_score"] = pd.to_numeric(audit.get("risk_adjusted_score"), errors="coerce").fillna(
                pd.to_numeric(audit.get("expected_return_10d"), errors="coerce")
                - 0.5 * pd.to_numeric(audit.get("bad_entry_probability_10d"), errors="coerce")
            )
        audit = self._join_trade_results(audit, trades_path)
        audit["actual_buy_amount"] = pd.to_numeric(audit.get("final_amount"), errors="coerce").fillna(0.0)
        amount = audit["actual_buy_amount"].where(audit["actual_buy_amount"] > 0)
        audit["actual_return"] = pd.to_numeric(audit.get("actual_net_profit"), errors="coerce") / amount
        return audit

    def rules(self) -> list[tuple[str, str]]:
        return [
            ("baseline", "current v2_73 actual sizing"),
            ("equal_weight_daily", "equal weight among actually bought candidates per signal date"),
            ("risk_adjusted_weight", "relative risk_adjusted_score weight per signal date"),
            ("expected_return_weight", "relative expected_return_10d weight per signal date"),
            ("bad_entry_defensive_weight", "lower bad_entry_probability_10d receives higher weight"),
            ("balanced_conviction_weight", "expected_return + 0.3*expected_max + 0.05*swing - 0.5*bad_entry"),
            ("cash_reserve_dynamic", "dynamic reserve by average daily risk_adjusted_score"),
            ("head_and_tail_rule", "swing middle capture; high bad_entry/overheat proxy is reduced"),
        ]

    def _simulate_rule(
        self,
        trades: pd.DataFrame,
        candidates: pd.DataFrame,
        rule_id: str,
        rule_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if trades.empty:
            return trades.copy(), pd.DataFrame()
        trade_rows = []
        daily_rows = []
        candidate_strength = candidates.groupby("signal_date", dropna=False)["risk_adjusted_score"].mean()
        for signal_date, group in trades.groupby("signal_date", dropna=False):
            group = group.copy()
            baseline_budget = float(pd.to_numeric(group["actual_buy_amount"], errors="coerce").fillna(0.0).sum())
            reserve_rate = self._cash_reserve_rate(rule_id, candidate_strength.get(signal_date))
            invest_budget = baseline_budget * (1.0 - reserve_rate)
            weights = self.weights_for_rule(group, rule_id)
            target_amount = weights * invest_budget
            original_amount = pd.to_numeric(group["actual_buy_amount"], errors="coerce").replace(0, pd.NA)
            multiplier = (target_amount / original_amount).fillna(0.0).astype(float)
            adjusted_net = pd.to_numeric(group["actual_net_profit"], errors="coerce").fillna(0.0) * multiplier
            group["portfolio_rule"] = rule_id
            group["portfolio_rule_name"] = rule_name
            group["allocation_weight"] = weights
            group["cash_reserve_rate"] = reserve_rate
            group["baseline_daily_invested_amount"] = baseline_budget
            group["simulated_daily_invested_amount"] = float(target_amount.sum())
            group["target_amount"] = target_amount
            group["amount_multiplier"] = multiplier
            group["adjusted_net_profit"] = adjusted_net
            trade_rows.append(group)
            day_candidates = candidates[candidates["signal_date"].eq(signal_date)]
            daily_rows.append(
                {
                    "signal_date": str(pd.Timestamp(signal_date).date()) if pd.notna(signal_date) else "",
                    "portfolio_rule": rule_id,
                    "candidate_count": int(len(day_candidates)),
                    "buy_count": int(len(group)),
                    "baseline_invested_amount": baseline_budget,
                    "simulated_invested_amount": float(target_amount.sum()),
                    "cash_reserve_rate": reserve_rate,
                    "average_risk_adjusted_score": self._mean(day_candidates, "risk_adjusted_score"),
                    "average_expected_return_10d": self._mean(day_candidates, "expected_return_10d"),
                    "average_bad_entry_probability_10d": self._mean(day_candidates, "bad_entry_probability_10d"),
                }
            )
        result = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
        return result, pd.DataFrame(daily_rows)

    def weights_for_rule(self, group: pd.DataFrame, rule_id: str) -> pd.Series:
        index = group.index
        if group.empty:
            return pd.Series(dtype=float)
        if rule_id == "baseline":
            amounts = pd.to_numeric(group.get("actual_buy_amount"), errors="coerce").fillna(0.0)
            return self._normalize(amounts, index)
        if rule_id == "equal_weight_daily":
            return pd.Series(1.0 / len(group), index=index)
        if rule_id == "risk_adjusted_weight":
            return self._normalize(self._shift_positive(group.get("risk_adjusted_score")), index)
        if rule_id == "expected_return_weight":
            return self._normalize(self._shift_positive(group.get("expected_return_10d")), index)
        if rule_id == "bad_entry_defensive_weight":
            bad = pd.to_numeric(group.get("bad_entry_probability_10d"), errors="coerce")
            return self._normalize((1.0 - bad).clip(lower=0.05), index)
        if rule_id == "balanced_conviction_weight":
            score = (
                pd.to_numeric(group.get("expected_return_10d"), errors="coerce").fillna(0.0)
                + 0.3 * pd.to_numeric(group.get("expected_max_return_20d"), errors="coerce").fillna(0.0)
                + 0.05 * pd.to_numeric(group.get("swing_success_probability_20d"), errors="coerce").fillna(0.0)
                - 0.5 * pd.to_numeric(group.get("bad_entry_probability_10d"), errors="coerce").fillna(0.5)
            )
            return self._normalize(self._shift_positive(score), index)
        if rule_id == "cash_reserve_dynamic":
            return self._normalize(self._shift_positive(group.get("risk_adjusted_score")), index)
        if rule_id == "head_and_tail_rule":
            score = (
                pd.to_numeric(group.get("expected_return_10d"), errors="coerce").fillna(0.0)
                + 0.2 * pd.to_numeric(group.get("expected_max_return_20d"), errors="coerce").fillna(0.0)
                + 0.08 * pd.to_numeric(group.get("swing_success_probability_20d"), errors="coerce").fillna(0.0)
                - 0.6 * pd.to_numeric(group.get("bad_entry_probability_10d"), errors="coerce").fillna(0.5)
            )
            bad = pd.to_numeric(group.get("bad_entry_probability_10d"), errors="coerce")
            expected = pd.to_numeric(group.get("expected_return_10d"), errors="coerce")
            swing = pd.to_numeric(group.get("swing_success_probability_20d"), errors="coerce")
            raw = self._shift_positive(score)
            raw = raw.mask((expected > 0.01) & (bad > 0.55), raw * 0.5)
            raw = raw.mask((swing > 0.20) & (bad < 0.35), raw * 1.2)
            return self._normalize(raw, index)
        return pd.Series(1.0 / len(group), index=index)

    def _summary(self, rule_id: str, rule_name: str, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty:
            return {"portfolio_rule": rule_id, "rule": rule_name, "trade_count": 0}
        adjusted = pd.to_numeric(trades["adjusted_net_profit"], errors="coerce").fillna(0.0)
        baseline = pd.to_numeric(trades["actual_net_profit"], errors="coerce").fillna(0.0)
        gross_profit = float(adjusted[adjusted > 0].sum())
        gross_loss = abs(float(adjusted[adjusted < 0].sum()))
        months = self._monthly_rows(trades)
        return {
            "portfolio_rule": rule_id,
            "rule": rule_name,
            "trade_count": int(len(trades)),
            "adjusted_net_profit": float(adjusted.sum()),
            "baseline_net_profit": float(baseline.sum()),
            "profit_delta": float(adjusted.sum() - baseline.sum()),
            "profit_factor": gross_profit / gross_loss if gross_loss else (None if gross_profit == 0 else float("inf")),
            "max_drawdown": self._max_drawdown(trades),
            "win_rate": float((adjusted > 0).mean()) if len(adjusted) else None,
            "monthly_win_rate": sum(1 for row in months if (row.get("adjusted_net_profit") or 0) > 0) / len(months) if months else None,
            "losing_months": sum(1 for row in months if (row.get("adjusted_net_profit") or 0) < 0),
            "average_daily_invested_amount": float(trades.groupby("signal_date")["simulated_daily_invested_amount"].first().mean()),
            "average_cash_reserve_rate": float(trades.groupby("signal_date")["cash_reserve_rate"].first().mean()),
            "average_holding_count": None,
            "top1_trade_contribution": self._top_contribution(adjusted, 1),
            "top3_trade_contribution": self._top_contribution(adjusted, 3),
            "focus_67400_contribution": self._focus_contribution(trades, adjusted, "67400"),
            "worst_trade": float(adjusted.min()) if len(adjusted) else None,
            "best_trade": float(adjusted.max()) if len(adjusted) else None,
        }

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase1Paths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase1_simulation_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        daily_csv = self.report_dir / f"{stem.replace('simulation', 'daily_allocations')}.csv"
        trade_csv = self.report_dir / f"{stem.replace('simulation', 'trade_allocations')}.csv"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps({k: v for k, v in result.items() if k not in {"daily_allocations", "trade_allocations"}}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        self._write_csv(daily_csv, result.get("daily_allocations") or [])
        self._write_csv(trade_csv, result.get("trade_allocations") or [])
        return PortfolioManagerPhase1Paths(markdown=markdown, json=json_path, daily_allocations_csv=daily_csv, trade_allocations_csv=trade_csv)

    def format_markdown(self, result: dict[str, Any]) -> str:
        columns = [
            "portfolio_rule",
            "adjusted_net_profit",
            "profit_delta",
            "profit_factor",
            "max_drawdown",
            "win_rate",
            "monthly_win_rate",
            "losing_months",
            "average_daily_invested_amount",
            "average_cash_reserve_rate",
            "top1_trade_contribution",
            "top3_trade_contribution",
            "focus_67400_contribution",
            "worst_trade",
            "best_trade",
        ]
        lines = [
            "# Portfolio Manager AI Phase 1 Simulation",
            "",
            f"- profile: `{self.profile}`",
            f"- period: {self.start_date} to {self.end_date}",
            "- method: post-trade amount multiplier simulation; trade dates and symbols are fixed.",
            "- limitation: this is not a full realistic cash/position/round-lot backtest.",
            "- source: existing `trades.csv`, `purchase_audit.csv`, and walk-forward predictions; no API fetch and no live orders.",
            "",
            "## Summary",
            "",
            self._table(result["summary"], columns),
            "",
            "## Best Rules",
            "",
            self._table(
                [
                    {"metric": "net_profit", **(result.get("best_by_net_profit") or {})},
                    {"metric": "profit_factor", **(result.get("best_by_profit_factor") or {})},
                    {"metric": "drawdown", **(result.get("best_by_drawdown") or {})},
                ],
                ["metric", *columns[:6]],
            ),
            "",
            "## Diagnosis",
            "",
        ]
        lines.extend(f"- {line}" for line in result.get("diagnosis", []))
        lines.append("")
        return "\n".join(lines)

    def _join_predictions(self, audit: pd.DataFrame) -> pd.DataFrame:
        if audit.empty:
            return audit
        cache: dict[str, pd.DataFrame] = {}
        for idx, row in audit.iterrows():
            date = row.get("signal_date")
            if pd.isna(date):
                continue
            date_key = pd.Timestamp(date).strftime("%Y-%m-%d")
            if date_key not in cache:
                path = self.prediction_dir / f"predictions_{date_key}.parquet"
                if path.exists():
                    data = pd.read_parquet(path)
                    data["code"] = data["code"].astype(str)
                    data["risk_adjusted_score"] = pd.to_numeric(data.get("expected_return_10d"), errors="coerce") - 0.5 * pd.to_numeric(data.get("bad_entry_probability_10d"), errors="coerce")
                    cache[date_key] = data.set_index("code")
                else:
                    cache[date_key] = pd.DataFrame()
            prediction = cache[date_key]
            code = str(row.get("code") or "")
            if prediction.empty or code not in prediction.index:
                continue
            pred = prediction.loc[code]
            for column in ML_COLUMNS:
                if column not in audit.columns:
                    audit[column] = pd.NA
                if pd.isna(audit.at[idx, column]):
                    audit.at[idx, column] = pred.get(column)
        for column in ML_COLUMNS:
            if column in audit.columns:
                audit[column] = pd.to_numeric(audit[column], errors="coerce")
        return audit

    def _join_trade_results(self, audit: pd.DataFrame, trades_path: Path) -> pd.DataFrame:
        audit["actual_net_profit"] = pd.NA
        if not trades_path.exists() or audit.empty:
            return audit
        trades = pd.read_csv(trades_path)
        if "action" in trades.columns:
            trades = trades[trades["action"].astype(str).eq("SELL")].copy()
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in trades.columns:
                trades[column] = pd.to_datetime(trades[column], errors="coerce")
        for column in ["net_profit", "net_profit_rate", "holding_days"]:
            if column in trades.columns:
                trades[column] = pd.to_numeric(trades[column], errors="coerce")
        keys = ["signal_date", "entry_date", "code"]
        trades["code"] = trades["code"].astype(str)
        result_columns = keys + ["trade_id", "exit_date", "holding_days", "net_profit", "net_profit_rate", "exit_reason"]
        result = trades[result_columns].drop_duplicates(subset=keys, keep="last")
        joined = audit.merge(result, on=keys, how="left", suffixes=("", "_trade"))
        joined["actual_net_profit"] = pd.to_numeric(joined["net_profit"], errors="coerce")
        joined["actual_net_profit_rate"] = pd.to_numeric(joined["net_profit_rate"], errors="coerce")
        return joined.drop(columns=[column for column in ["net_profit", "net_profit_rate"] if column in joined.columns])

    def _cash_reserve_rate(self, rule_id: str, average_risk_adjusted_score: Any) -> float:
        if rule_id != "cash_reserve_dynamic":
            return 0.0
        try:
            value = float(average_risk_adjusted_score)
        except (TypeError, ValueError):
            return 0.30
        if value >= -0.15:
            return 0.10
        if value < -0.25:
            return 0.50
        return 0.30

    def _normalize(self, values: pd.Series, index: pd.Index) -> pd.Series:
        values = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
        total = float(values.sum())
        if total <= 0:
            return pd.Series(1.0 / len(index), index=index)
        return values / total

    def _shift_positive(self, values: Any) -> pd.Series:
        series = pd.to_numeric(pd.Series(values), errors="coerce")
        if series.dropna().empty:
            return pd.Series(0.0, index=series.index)
        return (series - float(series.min()) + 1e-6).clip(lower=0.0)

    def _mean(self, df: pd.DataFrame, column: str) -> float | None:
        if df.empty or column not in df.columns:
            return None
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    def _monthly_rows(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty or "exit_date" not in trades.columns:
            return []
        data = trades.dropna(subset=["exit_date"]).copy()
        if data.empty:
            return []
        data["month"] = data["exit_date"].dt.to_period("M").astype(str)
        return [{"month": str(month), "adjusted_net_profit": float(group["adjusted_net_profit"].sum())} for month, group in data.groupby("month")]

    def _max_drawdown(self, trades: pd.DataFrame) -> float | None:
        if trades.empty:
            return None
        data = trades.dropna(subset=["exit_date"]).sort_values("exit_date")
        if data.empty:
            return None
        equity = self.initial_cash + pd.to_numeric(data["adjusted_net_profit"], errors="coerce").fillna(0.0).cumsum()
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        return float(drawdown.min()) if not drawdown.dropna().empty else None

    def _top_contribution(self, adjusted: pd.Series, n: int) -> float | None:
        total = float(adjusted.sum())
        if total == 0:
            return None
        return float(adjusted.sort_values(ascending=False).head(n).sum() / total)

    def _focus_contribution(self, trades: pd.DataFrame, adjusted: pd.Series, code: str) -> float | None:
        total = float(adjusted.sum())
        if total == 0:
            return None
        return float(adjusted[trades["code"].astype(str).eq(code)].sum() / total)

    def _best(self, rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        candidates = [row for row in rows if row.get(key) is not None and row.get("portfolio_rule") != "baseline"]
        return max(candidates, key=lambda row: row.get(key) or -10**18, default=None)

    def _diagnosis(self, rows: list[dict[str, Any]]) -> list[str]:
        baseline = next((row for row in rows if row.get("portfolio_rule") == "baseline"), {})
        best_profit = self._best(rows, "adjusted_net_profit") or {}
        best_pf = self._best(rows, "profit_factor") or {}
        best_dd = self._best(rows, "max_drawdown") or {}
        return [
            f"baseline adjusted_net_profit={self._format(baseline.get('adjusted_net_profit'))} PF={self._format(baseline.get('profit_factor'))} DD={self._format(baseline.get('max_drawdown'))}.",
            f"best net profit rule={best_profit.get('portfolio_rule')} adjusted_net_profit={self._format(best_profit.get('adjusted_net_profit'))}.",
            f"PF-focused rule={best_pf.get('portfolio_rule')} PF={self._format(best_pf.get('profit_factor'))}.",
            f"DD-focused rule={best_dd.get('portfolio_rule')} DD={self._format(best_dd.get('max_drawdown'))}.",
            "This is a trade amount multiplier simulation, not a full realistic portfolio backtest.",
        ]

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("|" + "|".join(self._format(row.get(column)) for column in columns) + "|")
        return "\n".join(lines)

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("\n", " ")
