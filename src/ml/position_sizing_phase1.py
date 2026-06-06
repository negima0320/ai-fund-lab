from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


V2_73_PROFILE = "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue"
V2_74_PROFILE = "rookie_dealer_02_v2_74_ml_ranked_exit_ai_affordable_fallback"
DEFAULT_PROFILES = [V2_73_PROFILE, V2_74_PROFILE]

ML_COLUMNS = ["risk_adjusted_score", "expected_return_10d", "bad_entry_probability_10d"]


@dataclass(frozen=True)
class PositionSizingPhase1Paths:
    markdown: Path
    json: Path
    trades_csv: Path


class PositionSizingPhase1Simulation:
    def __init__(
        self,
        root: str | Path = ".",
        profiles: list[str] | None = None,
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        initial_cash: float = 1_000_000.0,
    ) -> None:
        self.root = Path(root)
        self.profiles = profiles or list(DEFAULT_PROFILES)
        self.start_date = start_date
        self.end_date = end_date
        self.period_key = f"{start_date}_to_{end_date}"
        self.initial_cash = float(initial_cash)
        self.report_dir = self.root / "reports" / "ml"
        self.prediction_dir = self.root / "data" / "ml" / "walk_forward_predictions"

    def build(self) -> dict[str, Any]:
        summary: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        profile_join: list[dict[str, Any]] = []
        for profile in self.profiles:
            trades = self._load_enriched_trades(profile)
            profile_join.append(self._join_summary(profile, trades))
            for rule_id, rule_name, multipliers in self._multipliers(trades):
                result = self._simulate(profile, trades, rule_id, rule_name, multipliers)
                summary.append(result)
                per_trade = trades.copy()
                per_trade["profile"] = profile
                per_trade["sizing_rule"] = rule_id
                per_trade["sizing_rule_name"] = rule_name
                per_trade["position_size_multiplier"] = multipliers
                per_trade["adjusted_net_profit"] = pd.to_numeric(per_trade["net_profit"], errors="coerce").fillna(0.0) * multipliers
                trade_rows.extend(per_trade[self._trade_output_columns(per_trade)].to_dict(orient="records"))
        return {
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "profiles": self.profiles,
            "summary": summary,
            "join_summary": profile_join,
            "best_by_net_profit": self._best(summary, "adjusted_net_profit"),
            "best_by_profit_factor": self._best(summary, "profit_factor"),
            "best_by_drawdown": self._best(summary, "max_drawdown"),
            "trades": trade_rows,
            "diagnosis": self._diagnosis(summary),
        }

    def save(self, result: dict[str, Any]) -> PositionSizingPhase1Paths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "position_sizing_phase1_simulation_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        trades_csv = self.report_dir / "position_sizing_phase1_trades_2023-01_to_2026-05.csv"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps({k: v for k, v in result.items() if k != "trades"}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        self._write_csv(trades_csv, result.get("trades") or [])
        return PositionSizingPhase1Paths(markdown=markdown, json=json_path, trades_csv=trades_csv)

    def format_markdown(self, result: dict[str, Any]) -> str:
        columns = [
            "profile",
            "sizing_rule",
            "adjusted_net_profit",
            "profit_delta",
            "win_rate",
            "profit_factor",
            "max_drawdown",
            "worst_trade",
            "best_trade",
            "top1_trade_contribution",
            "top3_trade_contribution",
            "focus_67400_contribution",
            "monthly_win_rate",
            "losing_months",
            "average_multiplier",
        ]
        lines = [
            "# Position Sizing Phase 1 Simulation",
            "",
            f"- period: {self.start_date} to {self.end_date}",
            "- method: post-trade simulation; symbols/timing are fixed and only net_profit is multiplied.",
            "- source: existing backtest logs plus purchase_audit/walk-forward predictions; no backtest rerun, no API fetch, no live orders.",
            "",
            "## ML Join Summary",
            "",
            self._table(result.get("join_summary", []), ["profile", "trade_count", "ml_joined_count", "ml_join_rate", "purchase_audit_join_count", "prediction_fallback_join_count"]),
            "",
            "## Sizing Rule Summary",
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
                ["metric", *columns[:7]],
            ),
            "",
            "## Diagnosis",
            "",
        ]
        lines.extend(f"- {item}" for item in result.get("diagnosis", []))
        lines.append("")
        return "\n".join(lines)

    def _load_enriched_trades(self, profile: str) -> pd.DataFrame:
        backtest_dir = self.root / "logs" / "backtests" / profile / self.period_key
        trades_path = backtest_dir / "trades.csv"
        if not trades_path.exists():
            return pd.DataFrame()
        trades = pd.read_csv(trades_path)
        if "action" in trades.columns:
            trades = trades[trades["action"].astype(str).eq("SELL")].copy()
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in trades.columns:
                trades[column] = pd.to_datetime(trades[column], errors="coerce")
        for column in ["net_profit", "gross_profit", "net_profit_rate", "holding_days"]:
            if column in trades.columns:
                trades[column] = pd.to_numeric(trades[column], errors="coerce")
        trades["profile"] = profile
        trades["ml_join_source"] = ""
        trades = self._join_purchase_audit(trades, backtest_dir / "purchase_audit.csv")
        missing = trades[ML_COLUMNS].isna().any(axis=1)
        if bool(missing.any()):
            trades = self._join_predictions(trades)
        if "risk_adjusted_score" not in trades.columns or trades["risk_adjusted_score"].isna().any():
            expected = pd.to_numeric(trades.get("expected_return_10d"), errors="coerce")
            bad = pd.to_numeric(trades.get("bad_entry_probability_10d"), errors="coerce")
            trades["risk_adjusted_score"] = pd.to_numeric(trades.get("risk_adjusted_score"), errors="coerce").fillna(expected - 0.5 * bad)
        return trades.reset_index(drop=True)

    def _join_purchase_audit(self, trades: pd.DataFrame, audit_path: Path) -> pd.DataFrame:
        if trades.empty or not audit_path.exists():
            return trades
        audit = pd.read_csv(audit_path)
        if "decision" in audit.columns:
            audit = audit[audit["decision"].astype(str).isin(["BUY", "SCALED_BUY"])].copy()
        for column in ["signal_date", "entry_date"]:
            if column in audit.columns:
                audit[column] = pd.to_datetime(audit[column], errors="coerce")
        for column in ML_COLUMNS:
            if column in audit.columns:
                audit[column] = pd.to_numeric(audit[column], errors="coerce")
        keys = ["signal_date", "entry_date", "code"]
        if not set(keys).issubset(trades.columns) or not set(keys).issubset(audit.columns):
            return trades
        audit = audit.drop_duplicates(subset=keys, keep="last")
        joined = trades.merge(audit[keys + ML_COLUMNS], on=keys, how="left", suffixes=("", "_audit"))
        for column in ML_COLUMNS:
            audit_column = f"{column}_audit"
            if audit_column in joined.columns:
                if column not in joined.columns:
                    joined[column] = pd.NA
                joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(pd.to_numeric(joined[audit_column], errors="coerce"))
                joined = joined.drop(columns=[audit_column])
        joined_mask = joined[ML_COLUMNS].notna().all(axis=1)
        joined.loc[joined_mask, "ml_join_source"] = "purchase_audit"
        return joined

    def _join_predictions(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        cache: dict[str, pd.DataFrame] = {}
        for idx, row in trades.iterrows():
            if all(pd.notna(row.get(column)) for column in ML_COLUMNS):
                continue
            date = row.get("signal_date")
            if pd.isna(date):
                continue
            date_key = pd.Timestamp(date).strftime("%Y-%m-%d")
            if date_key not in cache:
                path = self.prediction_dir / f"predictions_{date_key}.parquet"
                if path.exists():
                    data = pd.read_parquet(path)
                    data["code"] = data["code"].astype(str)
                    if "risk_adjusted_score" not in data.columns:
                        data["risk_adjusted_score"] = pd.to_numeric(data.get("expected_return_10d"), errors="coerce") - 0.5 * pd.to_numeric(data.get("bad_entry_probability_10d"), errors="coerce")
                    cache[date_key] = data.set_index("code")
                else:
                    cache[date_key] = pd.DataFrame()
            data = cache[date_key]
            code = str(row.get("code") or "")
            if data.empty or code not in data.index:
                continue
            prediction = data.loc[code]
            for column in ML_COLUMNS:
                if pd.isna(trades.at[idx, column]):
                    trades.at[idx, column] = prediction.get(column)
            trades.at[idx, "ml_join_source"] = "prediction_fallback"
        return trades

    def _multipliers(self, trades: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
        if trades.empty:
            empty = pd.Series(dtype=float)
            return [("baseline", "multiplier=1.0", empty)]
        risk = pd.to_numeric(trades.get("risk_adjusted_score"), errors="coerce")
        expected = pd.to_numeric(trades.get("expected_return_10d"), errors="coerce")
        bad = pd.to_numeric(trades.get("bad_entry_probability_10d"), errors="coerce")
        baseline = pd.Series(1.0, index=trades.index)

        percentile = risk.rank(pct=True, method="average")
        decile_boost = pd.Series(1.0, index=trades.index)
        decile_boost = decile_boost.mask(percentile >= 0.90, 2.0)
        decile_boost = decile_boost.mask((percentile >= 0.70) & (percentile < 0.90), 1.5)
        decile_boost = decile_boost.mask(percentile < 0.30, 0.5)
        decile_boost = decile_boost.fillna(1.0)

        simple = pd.Series(1.0, index=trades.index)
        simple = simple.mask(risk > 0.10, 1.5)
        simple = simple.mask(risk < 0.00, 0.5)
        simple = simple.fillna(1.0)

        defensive = pd.Series(1.0, index=trades.index)
        defensive = defensive.mask(bad < 0.40, 1.5)
        defensive = defensive.mask(bad > 0.70, 0.5)
        defensive = defensive.fillna(1.0)

        expected_boost = pd.Series(1.0, index=trades.index)
        expected_boost = expected_boost.mask(expected >= 0.05, 2.0)
        expected_boost = expected_boost.mask((expected >= 0.03) & (expected < 0.05), 1.5)
        expected_boost = expected_boost.mask(expected < 0.01, 0.5)
        expected_boost = expected_boost.fillna(1.0)

        combined = pd.Series(1.0, index=trades.index)
        high = (risk >= 0.05) & (expected >= 0.03) & (bad <= 0.70)
        low = (risk < 0.00) | (expected < 0.01) | (bad > 0.80)
        combined = combined.mask(high, 2.0)
        combined = combined.mask(low, 0.5)
        combined = combined.fillna(1.0)

        return [
            ("baseline", "multiplier = 1.0", baseline),
            ("score_decile_boost", "risk_adjusted top10%=2.0, top10-30%=1.5, middle=1.0, bottom30%=0.5", decile_boost),
            ("score_simple_boost", "risk_adjusted >0.10=1.5, 0-0.10=1.0, <0=0.5", simple),
            ("bad_entry_defensive", "bad_entry <0.40=1.5, 0.40-0.70=1.0, >0.70=0.5", defensive),
            ("expected_return_boost", "expected >=0.05=2.0, 0.03-0.05=1.5, 0.01-0.03=1.0, <0.01=0.5", expected_boost),
            ("combined_conviction", "high conviction=2.0, medium=1.0, low=0.5", combined),
        ]

    def _simulate(self, profile: str, trades: pd.DataFrame, rule_id: str, rule_name: str, multipliers: pd.Series) -> dict[str, Any]:
        if trades.empty:
            return {"profile": profile, "sizing_rule": rule_id, "rule": rule_name, "trade_count": 0}
        net = pd.to_numeric(trades["net_profit"], errors="coerce").fillna(0.0)
        adjusted = net * multipliers
        gross_profit = float(adjusted[adjusted > 0].sum())
        gross_loss = abs(float(adjusted[adjusted < 0].sum()))
        original_total = float(net.sum())
        months = self._monthly_rows(trades, adjusted)
        return {
            "profile": profile,
            "sizing_rule": rule_id,
            "rule": rule_name,
            "trade_count": int(len(trades)),
            "original_net_profit": original_total,
            "adjusted_net_profit": float(adjusted.sum()),
            "profit_delta": float(adjusted.sum() - original_total),
            "win_rate": float((adjusted > 0).mean()) if len(adjusted) else None,
            "profit_factor": gross_profit / gross_loss if gross_loss else (None if gross_profit == 0 else float("inf")),
            "max_drawdown": self._max_drawdown(trades, adjusted),
            "worst_trade": float(adjusted.min()) if len(adjusted) else None,
            "best_trade": float(adjusted.max()) if len(adjusted) else None,
            "top1_trade_contribution": self._top_contribution(adjusted, 1),
            "top3_trade_contribution": self._top_contribution(adjusted, 3),
            "focus_67400_contribution": self._focus_contribution(trades, adjusted, "67400"),
            "monthly_win_rate": sum(1 for row in months if (row.get("adjusted_net_profit") or 0) > 0) / len(months) if months else None,
            "losing_months": sum(1 for row in months if (row.get("adjusted_net_profit") or 0) < 0),
            "average_multiplier": float(multipliers.mean()) if len(multipliers) else None,
            "max_multiplier": float(multipliers.max()) if len(multipliers) else None,
            "min_multiplier": float(multipliers.min()) if len(multipliers) else None,
            "ml_join_rate": float(trades[ML_COLUMNS].notna().all(axis=1).mean()) if len(trades) else None,
        }

    def _max_drawdown(self, trades: pd.DataFrame, adjusted: pd.Series) -> float | None:
        if trades.empty:
            return None
        data = pd.DataFrame({"exit_date": trades.get("exit_date"), "adjusted": adjusted}).sort_values("exit_date")
        equity = self.initial_cash + data["adjusted"].cumsum()
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        return float(drawdown.min()) if not drawdown.dropna().empty else None

    def _monthly_rows(self, trades: pd.DataFrame, adjusted: pd.Series) -> list[dict[str, Any]]:
        if "exit_date" not in trades.columns:
            return []
        data = trades.copy()
        data["adjusted_net_profit"] = adjusted
        data = data.dropna(subset=["exit_date"])
        if data.empty:
            return []
        data["month"] = data["exit_date"].dt.to_period("M").astype(str)
        return [
            {"month": str(month), "adjusted_net_profit": float(group["adjusted_net_profit"].sum())}
            for month, group in data.groupby("month")
        ]

    def _top_contribution(self, adjusted: pd.Series, n: int) -> float | None:
        total = float(adjusted.sum())
        if total == 0:
            return None
        return float(adjusted.sort_values(ascending=False).head(n).sum() / total)

    def _focus_contribution(self, trades: pd.DataFrame, adjusted: pd.Series, code: str) -> float | None:
        total = float(adjusted.sum())
        if total == 0 or "code" not in trades.columns:
            return None
        return float(adjusted[trades["code"].astype(str).eq(code)].sum() / total)

    def _join_summary(self, profile: str, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty:
            return {"profile": profile, "trade_count": 0, "ml_joined_count": 0, "ml_join_rate": None, "purchase_audit_join_count": 0, "prediction_fallback_join_count": 0}
        joined = trades[ML_COLUMNS].notna().all(axis=1)
        source = trades.get("ml_join_source", pd.Series(dtype=str)).astype(str)
        return {
            "profile": profile,
            "trade_count": int(len(trades)),
            "ml_joined_count": int(joined.sum()),
            "ml_join_rate": float(joined.mean()),
            "purchase_audit_join_count": int(source.eq("purchase_audit").sum()),
            "prediction_fallback_join_count": int(source.eq("prediction_fallback").sum()),
        }

    def _best(self, rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        candidates = [row for row in rows if row.get(key) is not None and row.get("sizing_rule") != "baseline"]
        return max(candidates, key=lambda row: row.get(key) or -10**18, default=None)

    def _diagnosis(self, rows: list[dict[str, Any]]) -> list[str]:
        lines = []
        for profile in self.profiles:
            baseline = next((row for row in rows if row.get("profile") == profile and row.get("sizing_rule") == "baseline"), {})
            candidates = [row for row in rows if row.get("profile") == profile and row.get("sizing_rule") != "baseline"]
            best_profit = max(candidates, key=lambda row: row.get("adjusted_net_profit") or -10**18, default={})
            best_pf = max(candidates, key=lambda row: row.get("profit_factor") or -10**18, default={})
            best_dd = max(candidates, key=lambda row: row.get("max_drawdown") or -10**18, default={})
            lines.append(
                f"{profile}: baseline net_profit={self._format(baseline.get('adjusted_net_profit'))} "
                f"PF={self._format(baseline.get('profit_factor'))} DD={self._format(baseline.get('max_drawdown'))}."
            )
            if best_profit:
                lines.append(f"{profile}: best net profit rule={best_profit.get('sizing_rule')} adjusted_net_profit={self._format(best_profit.get('adjusted_net_profit'))}.")
            if best_pf:
                lines.append(f"{profile}: PF-focused rule={best_pf.get('sizing_rule')} PF={self._format(best_pf.get('profit_factor'))}.")
            if best_dd:
                lines.append(f"{profile}: DD-focused rule={best_dd.get('sizing_rule')} DD={self._format(best_dd.get('max_drawdown'))}.")
        return lines

    def _trade_output_columns(self, df: pd.DataFrame) -> list[str]:
        columns = [
            "profile",
            "sizing_rule",
            "sizing_rule_name",
            "position_size_multiplier",
            "adjusted_net_profit",
            "trade_id",
            "signal_date",
            "entry_date",
            "exit_date",
            "code",
            "name",
            "net_profit",
            "net_profit_rate",
            "risk_adjusted_score",
            "expected_return_10d",
            "bad_entry_probability_10d",
            "ml_join_source",
        ]
        return [column for column in columns if column in df.columns]

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
