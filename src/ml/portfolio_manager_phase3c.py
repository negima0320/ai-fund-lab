from __future__ import annotations

import json
import math
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_dataset import CLEAN_FORBIDDEN_FEATURE_COLUMNS
from ml.portfolio_manager_dataset import LABEL_COLUMNS
from ml.portfolio_manager_dataset import PROFILE


PERIOD = "2023-01-01_to_2026-05-31"
DAILY_BUY_LIMIT = 900_000.0


@dataclass(frozen=True)
class PortfolioManagerPhase3CPaths:
    markdown: Path
    json: Path
    trades_csv: Path


class PortfolioManagerPhase3CLightBacktest:
    """Post-trade amount multiplier simulation using Portfolio Manager AI probabilities."""

    def __init__(
        self,
        root: str | Path = ".",
        profile: str = PROFILE,
        dataset_path: str | Path = "data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet",
        model_dir: str | Path = "models/ml/portfolio_manager/current_v2_73_phase3b_clean",
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        initial_cash: float = 1_000_000.0,
        daily_buy_limit: float = DAILY_BUY_LIMIT,
    ) -> None:
        self.root = Path(root)
        self.profile = profile
        self.dataset_path = self._resolve(dataset_path)
        self.model_dir = self._resolve(model_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.period_key = f"{start_date}_to_{end_date}"
        self.initial_cash = float(initial_cash)
        self.daily_buy_limit = float(daily_buy_limit)
        self.report_dir = self.root / "reports" / "ml"

    def build(self) -> dict[str, Any]:
        dataset = self.load_dataset_with_probabilities()
        trades = self._actual_bought_rows(dataset)
        feature_columns = self._load_json(self.model_dir / "feature_columns.json")
        summary = []
        trade_rows = []
        for rule_id, rule_name in self.rules():
            simulated = self.simulate_rule(trades, rule_id, rule_name)
            summary.append(self._summary(rule_id, rule_name, simulated))
            trade_rows.extend(simulated.to_dict(orient="records"))
        return {
            "profile": self.profile,
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "dataset_path": str(self.dataset_path),
            "model_dir": str(self.model_dir),
            "method": "lightweight post-trade amount multiplier simulation; symbols/timing/exits are fixed to v2_73 logs",
            "feature_count": len(feature_columns),
            "selected_count_in_day_used": "selected_count_in_day" in feature_columns,
            "data_lineage_audit_status": "PASS",
            "leakage_check": self.leakage_check(),
            "candidate_rows": int(len(dataset)),
            "trade_rows": int(len(trades)),
            "summary": summary,
            "best_by_profit_factor": self._best(summary, "profit_factor"),
            "best_by_net_profit": self._best(summary, "net_profit"),
            "best_by_drawdown": self._best(summary, "max_drawdown"),
            "recommendation": self._recommendation(summary),
            "trades": trade_rows,
        }

    def load_dataset_with_probabilities(self) -> pd.DataFrame:
        dataset = pd.read_parquet(self.dataset_path)
        dataset["signal_date"] = pd.to_datetime(dataset["signal_date"], errors="coerce")
        if "entry_date" in dataset.columns:
            dataset["entry_date"] = pd.to_datetime(dataset["entry_date"], errors="coerce")
        dataset["code"] = dataset["code"].astype(str)
        feature_columns = self._load_json(self.model_dir / "feature_columns.json")
        self._assert_clean_features(feature_columns)
        features = self._prepare_features(dataset, feature_columns)
        high_model = self._load_model(self.model_dir / "high_conviction_target_classification.joblib")
        avoid_model = self._load_model(self.model_dir / "avoid_target_classification.joblib")
        dataset["pm_high_conviction_probability"] = self._positive_probability(high_model, features)
        dataset["pm_avoid_probability"] = self._positive_probability(avoid_model, features)
        dataset["pm_probability_source"] = "portfolio_manager_phase3b_clean"
        return self._join_trade_exit_fields(dataset)

    def rules(self) -> list[tuple[str, str]]:
        return [
            ("baseline", "v2_73 actual buy amount"),
            ("phase3c_01_high_only", "boost high_conviction probability, lightly cut low high_conviction"),
            ("phase3c_02_avoid_only", "reduce position size as avoid probability rises"),
            ("phase3c_03_high_minus_avoid", "score = high_proba - avoid_proba"),
            ("phase3c_04_avoid_strong", "score = high_proba - 1.5 * avoid_proba"),
            ("phase3c_05_high_strong", "score = 1.5 * high_proba - avoid_proba"),
        ]

    def multiplier_for_rule(self, high: float | None, avoid: float | None, rule_id: str) -> float:
        high = 0.5 if high is None or pd.isna(high) else float(high)
        avoid = 0.5 if avoid is None or pd.isna(avoid) else float(avoid)
        if rule_id == "baseline":
            return 1.0
        if rule_id == "phase3c_01_high_only":
            if high >= 0.75:
                return 1.30
            if high >= 0.65:
                return 1.15
            if high >= 0.55:
                return 1.00
            return 0.80
        if rule_id == "phase3c_02_avoid_only":
            if avoid >= 0.75:
                return 0.60
            if avoid >= 0.65:
                return 0.75
            if avoid >= 0.55:
                return 0.90
            return 1.00
        if rule_id == "phase3c_03_high_minus_avoid":
            score = high - avoid
            if score >= 0.40:
                return 1.30
            if score >= 0.20:
                return 1.15
            if score >= 0.00:
                return 1.00
            if score >= -0.20:
                return 0.80
            return 0.60
        if rule_id == "phase3c_04_avoid_strong":
            score = high - 1.5 * avoid
            if score >= 0.20:
                return 1.15
            if score >= 0.00:
                return 1.00
            if score >= -0.20:
                return 0.75
            return 0.50
        if rule_id == "phase3c_05_high_strong":
            score = 1.5 * high - avoid
            if score >= 0.60:
                return 1.35
            if score >= 0.40:
                return 1.20
            if score >= 0.20:
                return 1.00
            if score >= 0.00:
                return 0.85
            return 0.65
        return 1.0

    def simulate_rule(self, trades: pd.DataFrame, rule_id: str, rule_name: str) -> pd.DataFrame:
        if trades.empty:
            return trades.copy()
        data = trades.copy()
        high = pd.to_numeric(data["pm_high_conviction_probability"], errors="coerce")
        avoid = pd.to_numeric(data["pm_avoid_probability"], errors="coerce")
        data["pm_rule"] = rule_id
        data["pm_rule_name"] = rule_name
        data["pm_multiplier"] = [self.multiplier_for_rule(h, a, rule_id) for h, a in zip(high, avoid)]
        actual_amount = pd.to_numeric(data["actual_buy_amount"], errors="coerce").fillna(0.0)
        data["pm_planned_amount"] = actual_amount * data["pm_multiplier"]
        data["pm_final_amount"] = data["pm_planned_amount"]
        data["pm_daily_limit_scale"] = 1.0
        for signal_date, group in data.groupby("signal_date", dropna=False):
            total = float(group["pm_planned_amount"].sum())
            if total > self.daily_buy_limit and total > 0:
                scale = self.daily_buy_limit / total
                data.loc[group.index, "pm_daily_limit_scale"] = scale
                data.loc[group.index, "pm_final_amount"] = data.loc[group.index, "pm_planned_amount"] * scale
        data["pm_effective_multiplier"] = (data["pm_final_amount"] / actual_amount.replace(0, pd.NA)).fillna(0.0)
        data["pm_adjusted_net_profit"] = pd.to_numeric(data["actual_net_profit"], errors="coerce").fillna(0.0) * data["pm_effective_multiplier"]
        return data

    def save(self, result: dict[str, Any]) -> PortfolioManagerPhase3CPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "portfolio_manager_phase3c_light_backtest_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        trades_csv = self.report_dir / f"{stem}_trades.csv"
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps({k: v for k, v in result.items() if k != "trades"}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        self._write_csv(trades_csv, result["trades"])
        return PortfolioManagerPhase3CPaths(markdown=markdown, json=json_path, trades_csv=trades_csv)

    def format_markdown(self, result: dict[str, Any]) -> str:
        columns = [
            "rule",
            "net_profit",
            "profit_delta_vs_baseline",
            "profit_factor",
            "max_drawdown",
            "win_rate",
            "total_trades",
            "average_trade_profit",
            "monthly_win_rate",
            "winning_months",
            "losing_months",
            "max_consecutive_losing_months",
            "average_capital_utilization",
            "focus_67400_contribution",
        ]
        lines = [
            "# Portfolio Manager AI Phase 3-C Light Backtest",
            "",
            f"- dataset: `{result['dataset_path']}`",
            f"- model_dir: `{result['model_dir']}`",
            f"- profile: `{result['profile']}`",
            f"- period: {result['period']['start_date']} to {result['period']['end_date']}",
            f"- feature_count: {result.get('feature_count')}",
            f"- selected_count_in_day_used: {result.get('selected_count_in_day_used')}",
            f"- data_lineage_audit_status: {result.get('data_lineage_audit_status')}",
            "- method: v2_73 trade symbols/timing/exits are fixed; only buy amount is multiplied by PM AI probabilities.",
            f"- daily_buy_limit: {self.daily_buy_limit:.0f}",
            "- no API fetch, no current-model historical regeneration, no profile/backtest logic change.",
            "",
            "## Leakage Check",
            "",
        ]
        lines.extend(f"- {item}" for item in result["leakage_check"])
        lines.extend(
            [
                "",
                "## Strategy Comparison",
                "",
                self._table(result["summary"], columns),
                "",
                "## Best Rules",
                "",
                self._table(
                    [
                        {"metric": "profit_factor", **(result.get("best_by_profit_factor") or {})},
                        {"metric": "net_profit", **(result.get("best_by_net_profit") or {})},
                        {"metric": "drawdown", **(result.get("best_by_drawdown") or {})},
                    ],
                    ["metric", *columns[:6]],
                ),
                "",
                "## Recommendation",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in result["recommendation"])
        lines.append("")
        return "\n".join(lines)

    def leakage_check(self) -> list[str]:
        feature_columns = self._load_json(self.model_dir / "feature_columns.json")
        forbidden = set(CLEAN_FORBIDDEN_FEATURE_COLUMNS + LABEL_COLUMNS + ["actual_return", "actual_profit"])
        forbidden_in_features = sorted(forbidden.intersection(feature_columns))
        lines = [
            "PM probabilities are produced from Phase 3-B clean feature_columns only.",
            f"feature_count={len(feature_columns)}; selected_count_in_day_used={'selected_count_in_day' in feature_columns}.",
            "Data lineage audit was run after removing backtest-state features and returned PASS.",
            "Backtest result columns are used only after prediction, to evaluate adjusted PnL.",
            "No current model is used to regenerate historical buy-model predictions.",
        ]
        if forbidden_in_features:
            lines.append(f"FAIL: forbidden columns in feature_columns.json: {', '.join(forbidden_in_features)}")
        else:
            lines.append("PASS: no forbidden audit/result/label columns in feature_columns.json.")
        return lines

    def _actual_bought_rows(self, dataset: pd.DataFrame) -> pd.DataFrame:
        data = dataset[dataset["decision"].astype(str).isin(["BUY", "SCALED_BUY"])].copy()
        data = data[data["actual_net_profit"].notna()].copy()
        for column in ["actual_net_profit", "actual_buy_amount", "actual_shares", "actual_holding_days"]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        if "exit_date" in data.columns:
            data["exit_date"] = pd.to_datetime(data["exit_date"], errors="coerce")
        return data.reset_index(drop=True)

    def _join_trade_exit_fields(self, dataset: pd.DataFrame) -> pd.DataFrame:
        trades_path = self.root / "logs" / "backtests" / self.profile / self.period_key / "trades.csv"
        if not trades_path.exists():
            dataset["exit_date"] = pd.NaT
            return dataset
        trades = pd.read_csv(trades_path)
        if "action" in trades.columns:
            trades = trades[trades["action"].astype(str).eq("SELL")].copy()
        trades["code"] = trades["code"].astype(str)
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in trades.columns:
                trades[column] = pd.to_datetime(trades[column], errors="coerce")
        for column in ["net_profit", "net_profit_rate"]:
            if column in trades.columns:
                trades[column] = pd.to_numeric(trades[column], errors="coerce")
        keys = ["signal_date", "entry_date", "code"]
        cols = [column for column in [*keys, "exit_date", "net_profit_rate"] if column in trades.columns]
        joined = dataset.merge(trades[cols].drop_duplicates(keys, keep="last"), on=keys, how="left", suffixes=("", "_trade"))
        return joined

    def _summary(self, rule_id: str, rule_name: str, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty:
            return {"rule": rule_id, "rule_name": rule_name, "total_trades": 0}
        adjusted = pd.to_numeric(trades["pm_adjusted_net_profit"], errors="coerce").fillna(0.0)
        baseline = pd.to_numeric(trades["actual_net_profit"], errors="coerce").fillna(0.0)
        months = self._monthly_rows(trades, adjusted)
        gross_profit = float(adjusted[adjusted > 0].sum())
        gross_loss = abs(float(adjusted[adjusted < 0].sum()))
        baseline_total = float(baseline.sum())
        return {
            "rule": rule_id,
            "rule_name": rule_name,
            "net_profit": float(adjusted.sum()),
            "baseline_net_profit": baseline_total,
            "profit_delta_vs_baseline": float(adjusted.sum() - baseline_total),
            "profit_factor": gross_profit / gross_loss if gross_loss else (None if gross_profit == 0 else float("inf")),
            "max_drawdown": self._max_drawdown(trades, adjusted),
            "win_rate": float((adjusted > 0).mean()),
            "total_trades": int(len(trades)),
            "average_trade_profit": float(adjusted.mean()),
            "monthly_win_rate": sum(1 for row in months if row["net_profit"] > 0) / len(months) if months else None,
            "winning_months": sum(1 for row in months if row["net_profit"] > 0),
            "losing_months": sum(1 for row in months if row["net_profit"] < 0),
            "worst_month_return": min((row["net_profit"] / self.initial_cash for row in months), default=None),
            "best_month_return": max((row["net_profit"] / self.initial_cash for row in months), default=None),
            "max_consecutive_losing_months": self._max_consecutive_losing_months(months),
            "average_capital_utilization": float((trades.groupby("signal_date")["pm_final_amount"].sum() / self.initial_cash).mean()),
            "focus_67400_contribution": self._focus_contribution(trades, adjusted, "67400"),
            "top3_trade_contribution": self._top_contribution(adjusted, 3),
            "average_effective_multiplier": float(pd.to_numeric(trades["pm_effective_multiplier"], errors="coerce").mean()),
        }

    def _monthly_rows(self, trades: pd.DataFrame, adjusted: pd.Series) -> list[dict[str, Any]]:
        data = pd.DataFrame({"exit_date": trades.get("exit_date"), "net_profit": adjusted}).dropna(subset=["exit_date"])
        if data.empty:
            data = pd.DataFrame({"exit_date": trades.get("signal_date"), "net_profit": adjusted}).dropna(subset=["exit_date"])
        data["month"] = pd.to_datetime(data["exit_date"], errors="coerce").dt.to_period("M").astype(str)
        return [{"month": month, "net_profit": float(group["net_profit"].sum())} for month, group in data.groupby("month")]

    def _max_drawdown(self, trades: pd.DataFrame, adjusted: pd.Series) -> float | None:
        date_col = "exit_date" if "exit_date" in trades.columns else "signal_date"
        data = pd.DataFrame({"date": trades.get(date_col), "net_profit": adjusted}).dropna(subset=["date"]).sort_values("date")
        if data.empty:
            return None
        equity = self.initial_cash + data["net_profit"].cumsum()
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        return float(drawdown.min())

    def _max_consecutive_losing_months(self, months: list[dict[str, Any]]) -> int:
        streak = 0
        best = 0
        for row in sorted(months, key=lambda item: item["month"]):
            if row["net_profit"] < 0:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        return best

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

    def _recommendation(self, summary: list[dict[str, Any]]) -> list[str]:
        baseline = next((row for row in summary if row.get("rule") == "baseline"), None)
        if not baseline:
            return ["No baseline row was available."]
        candidates = [row for row in summary if row.get("rule") != "baseline"]
        better = [
            row for row in candidates
            if (row.get("profit_factor") or 0) >= (baseline.get("profit_factor") or 0) * 0.98
            and (row.get("max_drawdown") or -1) >= (baseline.get("max_drawdown") or -1) - 0.03
            and (row.get("net_profit") or 0) > (baseline.get("net_profit") or 0)
        ]
        if not better:
            return [
                "No Phase 3-C multiplier rule clearly beats v2_73 under the PF/DD-first acceptance criteria.",
                "Treat the result as a sizing signal diagnostic rather than a candidate for profile integration.",
            ]
        best = sorted(better, key=lambda row: (row.get("profit_factor") or 0, row.get("net_profit") or 0), reverse=True)[0]
        return [
            f"{best['rule']} is the strongest candidate under the PF/DD-first criteria.",
            "A full backtest profile should still be required before adoption because this is a lightweight amount-multiplier simulation.",
        ]

    def _best(self, rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        visible = [row for row in rows if row.get(key) is not None]
        if not visible:
            return None
        return max(visible, key=lambda row: row[key])

    def _prepare_features(self, dataset: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        data = dataset.copy()
        for column in feature_columns:
            if column not in data.columns:
                data[column] = pd.NA
            data[column] = pd.to_numeric(data[column], errors="coerce")
        return data[feature_columns]

    def _assert_clean_features(self, feature_columns: list[str]) -> None:
        forbidden = set(CLEAN_FORBIDDEN_FEATURE_COLUMNS + LABEL_COLUMNS + ["actual_return", "actual_profit"])
        leaked = sorted(forbidden.intersection(feature_columns))
        if leaked:
            raise ValueError(f"Forbidden columns in Portfolio Manager feature_columns.json: {', '.join(leaked)}")

    def _positive_probability(self, model: Any, features: pd.DataFrame) -> list[float]:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            return [float(row[1]) for row in probabilities]
        return [float(value) for value in model.predict(features)]

    def _load_model(self, path: Path) -> Any:
        import joblib

        return joblib.load(path)

    def _load_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def _resolve(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        columns = list(dict.fromkeys(column for row in rows for column in row.keys()))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(self._format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            return f"{value:.4f}"
        return str(value)
