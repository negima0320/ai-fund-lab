from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import ML_DATA_ROOT, ML_MODELS_ROOT, ML_REPORTS_ROOT


THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TAX_RATE = 0.20315
INITIAL_CAPITAL = 1_000_000


@dataclass(frozen=True)
class ExitAvoidLossSimulationPaths:
    markdown: Path
    json: Path
    trades_csv: Path


class ExitAvoidLossSimulator:
    """Post-hoc threshold simulation for avoid_loss_5d Exit AI probabilities."""

    def __init__(
        self,
        root: str | Path = ".",
        dataset_path: str | Path | None = None,
        model_dir: str | Path | None = None,
        trades_path: str | Path | None = None,
        report_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.dataset_path = Path(dataset_path) if dataset_path else self._rooted(ML_DATA_ROOT) / "exit_datasets" / "exit_dataset_v2_66_2023-01_to_2026-05.parquet"
        self.model_dir = Path(model_dir) if model_dir else self._rooted(ML_MODELS_ROOT) / "exit" / "current_v2_66"
        self.trades_path = Path(trades_path) if trades_path else self.root / "logs" / "backtests" / "rookie_dealer_02_v2_66_ml_ranked" / "2023-01-01_to_2026-05-31" / "trades.csv"
        self.report_root = Path(report_root) if report_root else self._rooted(ML_REPORTS_ROOT)

    def build(self, thresholds: list[float] | None = None) -> dict[str, Any]:
        thresholds = thresholds or THRESHOLDS
        dataset = self._load_dataset_with_probabilities()
        trades = self._load_trades()
        baseline = self._baseline_trades(dataset, trades)
        simulations = []
        detail_frames = []
        for threshold in thresholds:
            simulated = self._simulate_threshold(dataset, baseline, threshold)
            simulations.append(self._summarize_threshold(simulated, baseline, threshold))
            detail_frames.append(simulated)
        details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
        return {
            "profile": "rookie_dealer_02_v2_66_ml_ranked",
            "period": {"start_date": "2023-01-01", "end_date": "2026-05-31"},
            "dataset_path": str(self.dataset_path),
            "model_dir": str(self.model_dir),
            "trades_path": str(self.trades_path),
            "thresholds": thresholds,
            "baseline": self._baseline_metrics(baseline),
            "results": simulations,
            "best_by_profit_delta": max(simulations, key=lambda row: row.get("profit_delta", -10**18)) if simulations else None,
            "best_by_profit_factor": max(simulations, key=lambda row: row.get("profit_factor") or -10**18) if simulations else None,
            "details": details,
        }

    def save(self, result: dict[str, Any]) -> ExitAvoidLossSimulationPaths:
        self.report_root.mkdir(parents=True, exist_ok=True)
        stem = "exit_avoid_loss_simulation_v2_66_2023-01_to_2026-05"
        markdown = self.report_root / f"{stem}.md"
        json_path = self.report_root / f"{stem}.json"
        trades_csv = self.report_root / f"{stem.replace('simulation', 'simulation_trades')}.csv"
        details = result.pop("details")
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        details.to_csv(trades_csv, index=False)
        result["details"] = details
        return ExitAvoidLossSimulationPaths(markdown=markdown, json=json_path, trades_csv=trades_csv)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Exit Avoid Loss Simulation v2_66",
            "",
            f"- dataset_path: `{result.get('dataset_path')}`",
            f"- model_dir: `{result.get('model_dir')}`",
            "- rule: sell at current close when `avoid_loss_5d_probability >= threshold`; otherwise keep actual exit",
            "- note: post-hoc analysis only; no backtest rerun and no trading logic change",
            "",
            "## Baseline",
            "",
            self._table([result["baseline"]], ["rule", "total_profit", "win_rate", "profit_factor", "max_drawdown", "average_holding_days", "trade_count"]),
            "",
            "## Threshold Results",
            "",
            self._table(
                result["results"],
                [
                    "threshold",
                    "total_profit",
                    "profit_delta",
                    "win_rate",
                    "profit_factor",
                    "max_drawdown",
                    "exit_changed_count",
                    "improved_trade_count",
                    "worsened_trade_count",
                    "avoided_large_loss_count",
                    "missed_profit_count",
                    "average_holding_days",
                    "precision",
                    "recall",
                    "worst20_profit_delta",
                    "best20_profit_delta",
                ],
            ),
            "",
            "## Monthly Impact",
            "",
        ]
        for row in result["results"]:
            lines.extend(
                [
                    f"### threshold {row['threshold']}",
                    "",
                    self._table(row["monthly_impact"], ["month", "baseline_profit", "simulated_profit", "profit_delta"]),
                    "",
                ]
            )
        best = result.get("best_by_profit_delta") or {}
        lines.extend(
            [
                "## Summary",
                "",
                f"- best_by_profit_delta: `{best.get('threshold')}` ({best.get('profit_delta')})",
                "- Interpret carefully: the current validation window is small, and this simulation only changes exits for trades that already happened.",
                "",
            ]
        )
        return "\n".join(lines)

    def _rooted(self, path: Path) -> Path:
        root = self.root.resolve()
        try:
            return root / path.resolve().relative_to(root)
        except ValueError:
            return path

    def _load_dataset_with_probabilities(self) -> pd.DataFrame:
        df = pd.read_parquet(self.dataset_path)
        feature_columns = json.loads((self.model_dir / "feature_columns.json").read_text(encoding="utf-8"))
        model = self._load_model(self.model_dir / "avoid_loss_5d_classification.joblib")
        features = df.copy()
        for column in feature_columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")
        probabilities = self._predict_probability(model, features[feature_columns])
        df["avoid_loss_5d_probability"] = probabilities
        return df

    def _load_model(self, path: Path) -> Any:
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("joblib is required to load Exit AI models. Install requirements.txt first.") from exc
        return joblib.load(path)

    def _predict_probability(self, model: Any, features: pd.DataFrame) -> list[float]:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            return [float(row[1]) for row in probabilities]
        return [float(value) for value in model.predict(features)]

    def _load_trades(self) -> pd.DataFrame:
        df = pd.read_csv(self.trades_path)
        if "action" in df.columns:
            df = df[df["action"].astype(str).eq("SELL")].copy()
        for column in ["entry_date", "exit_date"]:
            df[column] = pd.to_datetime(df[column], errors="coerce")
        for column in ["entry_price", "exit_price", "shares", "net_profit", "holding_days"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["code"] = df["code"].astype(str)
        if "trade_id" not in df.columns:
            df["trade_id"] = pd.NA
        df["trade_id"] = df["trade_id"].fillna(
            df["entry_date"].dt.strftime("%Y-%m-%d") + "_" + df["exit_date"].dt.strftime("%Y-%m-%d") + "_" + df["code"]
        )
        return df.dropna(subset=["entry_price", "exit_price", "shares", "net_profit"]).reset_index(drop=True)

    def _baseline_trades(self, dataset: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        trades_by_id = trades.set_index("trade_id").to_dict("index")
        rows = []
        for _, group in dataset.sort_values(["current_date"]).groupby("trade_id", sort=False):
            last = group.iloc[-1]
            actual = trades_by_id.get(str(last["trade_id"]), {})
            shares = self._to_float(actual.get("shares"), 1.0)
            baseline_profit = self._to_float(actual.get("net_profit"), None)
            if baseline_profit is None:
                baseline_profit = self._net_profit(float(last["entry_price"]), float(last["current_close"]), shares)
            rows.append(
                {
                    "trade_id": last["trade_id"],
                    "code": str(last["code"]),
                    "entry_date": last["entry_date"],
                    "actual_exit_date": self._date_text(actual.get("exit_date")) or last["actual_exit_date"],
                    "entry_price": float(last["entry_price"]),
                    "shares": shares,
                    "actual_exit_price": self._to_float(actual.get("exit_price"), float(last["current_close"])),
                    "actual_holding_days": int(self._to_float(actual.get("holding_days"), int(last["holding_days"]))),
                    "baseline_profit": baseline_profit,
                    "avoid_loss_5d": bool(group["avoid_loss_5d"].fillna(False).any()),
                }
            )
        return pd.DataFrame(rows)

    def _simulate_threshold(self, dataset: pd.DataFrame, baseline: pd.DataFrame, threshold: float) -> pd.DataFrame:
        baseline_by_trade = baseline.set_index("trade_id").to_dict("index")
        rows = []
        for trade_id, group in dataset.sort_values(["current_date"]).groupby("trade_id", sort=False):
            base = baseline_by_trade[str(trade_id)]
            trigger = group[group["avoid_loss_5d_probability"].ge(threshold)].head(1)
            if trigger.empty:
                exit_row = group.iloc[-1]
                changed = False
                simulated_profit = float(base["baseline_profit"])
                simulated_exit_date = base["actual_exit_date"]
                simulated_exit_price = float(base["actual_exit_price"])
                simulated_holding_days = int(base["actual_holding_days"])
            else:
                exit_row = trigger.iloc[0]
                changed = True
                simulated_profit = self._net_profit(float(exit_row["entry_price"]), float(exit_row["current_close"]), float(base["shares"]))
                simulated_exit_date = exit_row["current_date"]
                simulated_exit_price = float(exit_row["current_close"])
                simulated_holding_days = int(exit_row["holding_days"])
            baseline_profit = float(base["baseline_profit"])
            rows.append(
                {
                    "threshold": threshold,
                    "trade_id": trade_id,
                    "code": str(exit_row["code"]),
                    "entry_date": exit_row["entry_date"],
                    "baseline_exit_date": base["actual_exit_date"],
                    "simulated_exit_date": simulated_exit_date,
                    "entry_price": float(exit_row["entry_price"]),
                    "baseline_exit_price": float(base["actual_exit_price"]),
                    "simulated_exit_price": simulated_exit_price,
                    "baseline_profit": baseline_profit,
                    "simulated_profit": simulated_profit,
                    "profit_delta": simulated_profit - baseline_profit,
                    "baseline_holding_days": int(base["actual_holding_days"]),
                    "simulated_holding_days": simulated_holding_days,
                    "exit_changed": changed,
                    "avoid_loss_5d_probability": float(exit_row["avoid_loss_5d_probability"]),
                    "avoid_loss_5d": bool(base["avoid_loss_5d"]),
                    "improved": simulated_profit > baseline_profit,
                    "worsened": simulated_profit < baseline_profit,
                }
            )
        return pd.DataFrame(rows)

    def _summarize_threshold(self, simulated: pd.DataFrame, baseline: pd.DataFrame, threshold: float) -> dict[str, Any]:
        metrics = self._metrics(simulated, "simulated_profit", "simulated_holding_days")
        baseline_profit = float(baseline["baseline_profit"].sum())
        changed = simulated[simulated["exit_changed"]]
        actual_positive = simulated["avoid_loss_5d"].astype(bool)
        predicted_positive = simulated["exit_changed"].astype(bool)
        worst20 = set(baseline.sort_values("baseline_profit").head(20)["trade_id"])
        best20 = set(baseline.sort_values("baseline_profit", ascending=False).head(20)["trade_id"])
        monthly = self._monthly_impact(simulated)
        metrics.update(
            {
                "threshold": threshold,
                "profit_delta": float(metrics["total_profit"] - baseline_profit),
                "exit_changed_count": int(predicted_positive.sum()),
                "improved_trade_count": int((simulated["simulated_profit"] > simulated["baseline_profit"]).sum()),
                "worsened_trade_count": int((simulated["simulated_profit"] < simulated["baseline_profit"]).sum()),
                "avoided_large_loss_count": int(((simulated["baseline_profit"] < 0) & (simulated["simulated_profit"] > simulated["baseline_profit"])).sum()),
                "missed_profit_count": int(((simulated["baseline_profit"] > 0) & (simulated["simulated_profit"] < simulated["baseline_profit"])).sum()),
                "precision": self._precision(actual_positive, predicted_positive),
                "recall": self._recall(actual_positive, predicted_positive),
                "worst20_profit_delta": float(simulated[simulated["trade_id"].isin(worst20)]["profit_delta"].sum()),
                "best20_profit_delta": float(simulated[simulated["trade_id"].isin(best20)]["profit_delta"].sum()),
                "monthly_impact": monthly,
            }
        )
        return metrics

    def _baseline_metrics(self, baseline: pd.DataFrame) -> dict[str, Any]:
        metrics = self._metrics(baseline, "baseline_profit", "actual_holding_days")
        metrics["rule"] = "Baseline actual exit"
        return metrics

    def _metrics(self, frame: pd.DataFrame, profit_col: str, holding_col: str) -> dict[str, Any]:
        profits = pd.to_numeric(frame[profit_col], errors="coerce").fillna(0.0)
        wins = profits > 0
        gross_profit = float(profits[wins].sum())
        gross_loss = float(-profits[profits < 0].sum())
        equity = INITIAL_CAPITAL + profits.cumsum()
        drawdown = (equity - equity.cummax()) / equity.cummax()
        return {
            "rule": "",
            "total_profit": float(profits.sum()),
            "win_rate": float(wins.mean()) if len(wins) else None,
            "profit_factor": gross_profit / gross_loss if gross_loss else None,
            "max_drawdown": float(drawdown.min()) if len(drawdown) else None,
            "average_holding_days": float(pd.to_numeric(frame[holding_col], errors="coerce").mean()),
            "trade_count": int(len(frame)),
        }

    def _monthly_impact(self, simulated: pd.DataFrame) -> list[dict[str, Any]]:
        df = simulated.copy()
        df["month"] = pd.to_datetime(df["simulated_exit_date"], errors="coerce").dt.to_period("M").astype(str)
        grouped = df.groupby("month", dropna=True).agg(
            baseline_profit=("baseline_profit", "sum"),
            simulated_profit=("simulated_profit", "sum"),
            profit_delta=("profit_delta", "sum"),
        )
        return [
            {
                "month": str(index),
                "baseline_profit": float(row["baseline_profit"]),
                "simulated_profit": float(row["simulated_profit"]),
                "profit_delta": float(row["profit_delta"]),
            }
            for index, row in grouped.reset_index().set_index("month").iterrows()
        ]

    def _net_profit(self, entry_price: float, exit_price: float, shares: float) -> float:
        gross = (exit_price - entry_price) * shares
        tax = gross * TAX_RATE if gross > 0 else 0.0
        return float(gross - tax)

    def _to_float(self, value: Any, default: float | None = 0.0) -> float | None:
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _date_text(self, value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    def _precision(self, actual: pd.Series, predicted: pd.Series) -> float:
        true_positive = int((actual & predicted).sum())
        predicted_positive = int(predicted.sum())
        return true_positive / predicted_positive if predicted_positive else 0.0

    def _recall(self, actual: pd.Series, predicted: pd.Series) -> float:
        true_positive = int((actual & predicted).sum())
        actual_positive = int(actual.sum())
        return true_positive / actual_positive if actual_positive else 0.0

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
