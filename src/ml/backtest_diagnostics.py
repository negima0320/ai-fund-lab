from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROFILES = [
    "rookie_dealer_02_v2_65",
    "rookie_dealer_02_v2_66_ml_ranked",
    "rookie_dealer_02_v2_67_ml_standalone",
]


@dataclass(frozen=True)
class BacktestDiagnosticsPaths:
    markdown: Path
    json: Path
    monthly_csv: Path
    code_csv: Path


class MLBacktestDiagnostics:
    """Build diagnostics from existing backtest logs without rerunning trades."""

    def __init__(
        self,
        root: str | Path = ".",
        profiles: list[str] | None = None,
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        predictions_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.profiles = profiles or list(DEFAULT_PROFILES)
        self.start_date = start_date
        self.end_date = end_date
        self.period_key = f"{start_date}_to_{end_date}"
        self.predictions_root = Path(predictions_root) if predictions_root else self.root / "data" / "ml" / "walk_forward_predictions"

    def build(self) -> dict[str, Any]:
        summaries = {profile: self._load_summary(profile) for profile in self.profiles}
        trades = {profile: self._load_sell_trades(profile) for profile in self.profiles}
        all_trades = {profile: self._load_all_trades(profile, summaries[profile]) for profile in self.profiles}

        monthly = pd.concat([self._period_stats(df, profile, "M") for profile, df in trades.items()], ignore_index=True)
        yearly = pd.concat([self._period_stats(df, profile, "Y") for profile, df in trades.items()], ignore_index=True)
        code_stats = pd.concat([self._code_stats(df, profile) for profile, df in trades.items()], ignore_index=True)

        diff = self._monthly_diff(monthly, "rookie_dealer_02_v2_66_ml_ranked", "rookie_dealer_02_v2_65")
        ml_ranked = self._ml_ranked_analysis(
            all_trades.get("rookie_dealer_02_v2_66_ml_ranked", pd.DataFrame()),
            trades.get("rookie_dealer_02_v2_66_ml_ranked", pd.DataFrame()),
        )
        candidate_coverage = self._candidate_prediction_coverage("rookie_dealer_02_v2_66_ml_ranked")
        standalone = self._standalone_analysis(
            summaries.get("rookie_dealer_02_v2_67_ml_standalone", {}),
            trades.get("rookie_dealer_02_v2_67_ml_standalone", pd.DataFrame()),
        )
        summary_rows = [self._summary_row(profile, summaries[profile]) for profile in self.profiles]
        return {
            "period": {"start_date": self.start_date, "end_date": self.end_date},
            "profiles": self.profiles,
            "summary": summary_rows,
            "monthly": monthly.to_dict("records"),
            "yearly": yearly.to_dict("records"),
            "monthly_diff_v66_vs_v65": diff,
            "code_stats": code_stats.to_dict("records"),
            "ml_ranked_analysis": {**ml_ranked, "candidate_prediction_coverage": candidate_coverage},
            "standalone_analysis": standalone,
        }

    def save(self, result: dict[str, Any]) -> BacktestDiagnosticsPaths:
        out_dir = self.root / "reports" / "ml"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "ml_backtest_diagnostics_2023-01_to_2026-05"
        markdown = out_dir / f"{stem}.md"
        json_path = out_dir / f"{stem}.json"
        monthly_csv = out_dir / f"{stem}_monthly.csv"
        code_csv = out_dir / f"{stem}_code.csv"

        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame(result["monthly"]).to_csv(monthly_csv, index=False)
        pd.DataFrame(result["code_stats"]).to_csv(code_csv, index=False)
        return BacktestDiagnosticsPaths(markdown, json_path, monthly_csv, code_csv)

    def format_markdown(self, result: dict[str, Any]) -> str:
        diff = result["monthly_diff_v66_vs_v65"]
        ml = result["ml_ranked_analysis"]
        standalone = result["standalone_analysis"]
        lines = [
            "# ML Backtest Diagnostics",
            "",
            f"- period: {self.start_date} to {self.end_date}",
            "- source: existing logs/backtests outputs only",
            "- note: no backtest rerun, no API refetch, no trading logic change",
            "",
            "## Summary",
            "",
            self._table(
                result["summary"],
                ["profile", "final_assets", "net_profit", "win_rate", "profit_factor", "max_drawdown", "total_trades", "average_holding_days"],
            ),
            "",
            "## Yearly Comparison",
            "",
            self._table(
                result["yearly"],
                ["profile", "period", "net_profit", "win_rate", "profit_factor", "max_drawdown", "trade_count"],
            ),
            "",
            "## v2_66 vs v2_65 Monthly Difference",
            "",
            f"- improved_months: {diff['improved_months']}",
            f"- worsened_months: {diff['worsened_months']}",
            "",
            "### Top Improved Months",
            "",
            self._table(diff["top_improved_months"], ["period", "net_profit_diff", "v66_net_profit", "v65_net_profit"]),
            "",
            "### Top Worsened Months",
            "",
            self._table(diff["top_worsened_months"], ["period", "net_profit_diff", "v66_net_profit", "v65_net_profit"]),
            "",
            "## ML Ranked Contribution",
            "",
            f"- buy_join_success_rate: {ml.get('buy_join_success_rate')}",
            f"- candidate_join_success_rate: {ml.get('candidate_prediction_coverage', {}).get('join_success_rate')}",
            f"- candidate_missing_predictions: {ml.get('candidate_prediction_coverage', {}).get('missing_prediction_count')}",
            "",
            "### Risk Adjusted Score Bands",
            "",
            self._table(
                ml.get("risk_adjusted_score_bands", []),
                ["risk_adjusted_score_band", "trade_count", "net_profit", "win_rate", "profit_factor", "average_profit"],
            ),
            "",
            "### v2_66 Code Concentration",
            "",
            "#### Top Profit Codes",
            "",
            self._table(
                self._top_code_rows(result["code_stats"], "rookie_dealer_02_v2_66_ml_ranked", top=True),
                ["code", "trade_count", "total_profit", "win_rate", "average_profit", "worst_trade", "best_trade"],
            ),
            "",
            "#### Worst Profit Codes",
            "",
            self._table(
                self._top_code_rows(result["code_stats"], "rookie_dealer_02_v2_66_ml_ranked", top=False),
                ["code", "trade_count", "total_profit", "win_rate", "average_profit", "worst_trade", "best_trade"],
            ),
            "",
            "## ML Standalone Diagnostics",
            "",
            f"- skip_counts: {standalone.get('skip_counts')}",
            f"- holding_days_distribution: {standalone.get('holding_days_distribution')}",
            f"- realistic_reference: {standalone.get('realistic_reference')}",
            "",
            "## Interpretation",
            "",
            "- v2_66_ml_ranked appears to improve mainly by changing the order of already-eligible v2_65 candidates, not by replacing the strategy universe.",
            "- v2_67_ml_standalone is lower return than v2_66 but has stronger win-rate and drawdown characteristics, so it is more useful as a risk-controlled AI-only benchmark than as a direct replacement.",
            "- v2_66 is a reasonable next main-profile candidate, but monthly weak periods should be monitored before live use.",
            "",
        ]
        return "\n".join(lines)

    def _top_code_rows(self, rows: list[dict[str, Any]], profile: str, top: bool) -> list[dict[str, Any]]:
        filtered = [row for row in rows if row.get("profile") == profile]
        return sorted(filtered, key=lambda row: float(row.get("total_profit") or 0), reverse=top)[:10]

    def _load_summary(self, profile: str) -> dict[str, Any]:
        path = self.root / "logs" / "backtests" / profile / self.period_key / "backtest_summary.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_sell_trades(self, profile: str) -> pd.DataFrame:
        path = self.root / "logs" / "backtests" / profile / self.period_key / "trades.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if df.empty:
            return df
        if "action" in df.columns:
            df = df[df["action"].astype(str).eq("SELL")].copy()
        return self._normalize_trade_frame(df)

    def _load_all_trades(self, profile: str, summary: dict[str, Any]) -> pd.DataFrame:
        rows = summary.get("all_trades") or []
        df = pd.DataFrame(rows)
        return self._normalize_trade_frame(df) if not df.empty else df

    def _normalize_trade_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in output.columns:
                output[column] = pd.to_datetime(output[column], errors="coerce")
        for column in [
            "net_profit",
            "profit",
            "net_profit_rate",
            "profit_rate",
            "risk_adjusted_score",
            "expected_return_10d",
            "bad_entry_probability_10d",
            "expected_max_return_20d",
            "swing_success_probability_20d",
            "holding_days",
        ]:
            if column in output.columns:
                output[column] = pd.to_numeric(output[column], errors="coerce")
        if "net_profit" not in output.columns and "profit" in output.columns:
            output["net_profit"] = output["profit"]
        return output

    def _period_stats(self, df: pd.DataFrame, profile: str, freq: str) -> pd.DataFrame:
        if df.empty or "exit_date" not in df.columns:
            return pd.DataFrame()
        period = df["exit_date"].dt.to_period(freq).astype(str)
        rows = []
        for value, group in df.groupby(period):
            rows.append(
                {
                    "profile": profile,
                    "period": value,
                    "net_profit": float(group["net_profit"].sum()),
                    "monthly_return": float(group["net_profit"].sum() / 1_000_000),
                    "win_rate": self._win_rate(group),
                    "profit_factor": self._profit_factor(group),
                    "trade_count": int(len(group)),
                    "max_drawdown": self._simple_drawdown(group),
                }
            )
        return pd.DataFrame(rows)

    def _code_stats(self, df: pd.DataFrame, profile: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        rows = []
        for code, group in df.groupby(df["code"].astype(str)):
            rows.append(
                {
                    "profile": profile,
                    "code": code,
                    "trade_count": int(len(group)),
                    "total_profit": float(group["net_profit"].sum()),
                    "win_rate": self._win_rate(group),
                    "average_profit": float(group["net_profit"].mean()),
                    "worst_trade": float(group["net_profit"].min()),
                    "best_trade": float(group["net_profit"].max()),
                }
            )
        return pd.DataFrame(rows).sort_values(["profile", "total_profit"], ascending=[True, False])

    def _monthly_diff(self, monthly: pd.DataFrame, target: str, baseline: str) -> dict[str, Any]:
        if monthly.empty:
            return {"improved_months": 0, "worsened_months": 0, "top_improved_months": [], "top_worsened_months": []}
        left = monthly[monthly["profile"].eq(target)][["period", "net_profit"]].rename(columns={"net_profit": "v66_net_profit"})
        right = monthly[monthly["profile"].eq(baseline)][["period", "net_profit"]].rename(columns={"net_profit": "v65_net_profit"})
        diff = left.merge(right, on="period", how="outer").fillna(0.0)
        diff["net_profit_diff"] = diff["v66_net_profit"] - diff["v65_net_profit"]
        return {
            "improved_months": int((diff["net_profit_diff"] > 0).sum()),
            "worsened_months": int((diff["net_profit_diff"] < 0).sum()),
            "top_improved_months": diff.sort_values("net_profit_diff", ascending=False).head(10).to_dict("records"),
            "top_worsened_months": diff.sort_values("net_profit_diff", ascending=True).head(10).to_dict("records"),
        }

    def _ml_ranked_analysis(self, all_trades: pd.DataFrame, sell_trades: pd.DataFrame) -> dict[str, Any]:
        buys = all_trades[all_trades.get("action", pd.Series(dtype=str)).astype(str).eq("BUY")].copy() if not all_trades.empty else pd.DataFrame()
        if buys.empty:
            return {}
        joined = buys["risk_adjusted_score"].notna() if "risk_adjusted_score" in buys.columns else pd.Series(False, index=buys.index)
        if "risk_adjusted_score" not in sell_trades.columns:
            sell_trades = self._join_predictions(sell_trades)
        bands = self._score_band_stats(sell_trades, "risk_adjusted_score")
        return {
            "buy_count": int(len(buys)),
            "buy_joined_count": int(joined.sum()),
            "buy_join_success_rate": float(joined.mean()) if len(joined) else None,
            "risk_adjusted_score_bands": bands,
            "top_score_trade_stats": self._top_score_trade_stats(sell_trades),
        }

    def _candidate_prediction_coverage(self, profile: str) -> dict[str, Any]:
        scored_dir = self.root / "data" / "processed" / profile
        paths = sorted(scored_dir.glob("scored_candidates_*.json"))
        total = 0
        joined = 0
        selected_total = 0
        selected_joined = 0
        for path in paths:
            date_text = path.stem.replace("scored_candidates_", "")
            predictions = self._load_predictions_for_date(date_text)
            if predictions.empty:
                prediction_codes: set[str] = set()
            else:
                prediction_codes = set(predictions["code"].astype(str))
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload.get("scores", []):
                code = str(row.get("code") or "")
                total += 1
                is_joined = code in prediction_codes
                joined += int(is_joined)
                if row.get("selected"):
                    selected_total += 1
                    selected_joined += int(is_joined)
        return {
            "scored_candidate_count": total,
            "joined_prediction_count": joined,
            "missing_prediction_count": total - joined,
            "join_success_rate": float(joined / total) if total else None,
            "selected_candidate_count": selected_total,
            "selected_join_success_rate": float(selected_joined / selected_total) if selected_total else None,
        }

    def _standalone_analysis(self, summary: dict[str, Any], sell_trades: pd.DataFrame) -> dict[str, Any]:
        all_trades = pd.DataFrame(summary.get("all_trades") or [])
        skip_counts = {}
        if not all_trades.empty and "action" in all_trades.columns:
            skips = all_trades[all_trades["action"].astype(str).eq("SKIP_BUY")]
            if "skipped_reason" in skips.columns:
                skip_counts = skips["skipped_reason"].fillna("").replace("", "unknown").value_counts().to_dict()
        holding_dist = {}
        if not sell_trades.empty and "holding_days" in sell_trades.columns:
            holding_dist = sell_trades["holding_days"].dropna().astype(int).value_counts().sort_index().to_dict()
        realistic = self._load_realistic_reference()
        return {
            "skip_counts": skip_counts,
            "holding_days_distribution": holding_dist,
            "sell_trade_count": int(len(sell_trades)),
            "realistic_reference": realistic,
        }

    def _join_predictions(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty or "signal_date" not in trades.columns:
            return trades
        frames = []
        for date_value, group in trades.groupby(trades["signal_date"].dt.strftime("%Y-%m-%d")):
            predictions = self._load_predictions_for_date(str(date_value))
            if predictions.empty:
                frames.append(group)
                continue
            merged = group.copy()
            merged["code"] = merged["code"].astype(str)
            frames.append(merged.merge(predictions, on="code", how="left", suffixes=("", "_prediction")))
        return pd.concat(frames, ignore_index=True) if frames else trades

    def _load_predictions_for_date(self, date_text: str) -> pd.DataFrame:
        path = self.predictions_root / f"predictions_{date_text}.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if "code" in df.columns:
            df["code"] = df["code"].astype(str)
        if {"expected_return_10d", "bad_entry_probability_10d"}.issubset(df.columns):
            df["risk_adjusted_score"] = pd.to_numeric(df["expected_return_10d"], errors="coerce") - 0.5 * pd.to_numeric(df["bad_entry_probability_10d"], errors="coerce")
        return df

    def _score_band_stats(self, df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        if df.empty or column not in df.columns:
            return []
        data = df.dropna(subset=[column]).copy()
        if data.empty:
            return []
        data["risk_adjusted_score_band"] = pd.cut(
            data[column],
            bins=[-float("inf"), -0.30, -0.20, -0.10, 0.0, float("inf")],
            labels=["< -0.30", "-0.30 to -0.20", "-0.20 to -0.10", "-0.10 to 0", ">= 0"],
        )
        rows = []
        for band, group in data.groupby("risk_adjusted_score_band", observed=True):
            rows.append(
                {
                    "risk_adjusted_score_band": str(band),
                    "trade_count": int(len(group)),
                    "net_profit": float(group["net_profit"].sum()),
                    "win_rate": self._win_rate(group),
                    "profit_factor": self._profit_factor(group),
                    "average_profit": float(group["net_profit"].mean()),
                }
            )
        return rows

    def _top_score_trade_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty or "risk_adjusted_score" not in df.columns:
            return {}
        data = df.dropna(subset=["risk_adjusted_score"]).sort_values("risk_adjusted_score", ascending=False)
        top = data.head(max(1, int(len(data) * 0.2)))
        bottom = data.tail(max(1, int(len(data) * 0.2)))
        return {
            "top_20pct": {"trade_count": int(len(top)), "net_profit": float(top["net_profit"].sum()), "win_rate": self._win_rate(top)},
            "bottom_20pct": {"trade_count": int(len(bottom)), "net_profit": float(bottom["net_profit"].sum()), "win_rate": self._win_rate(bottom)},
        }

    def _load_realistic_reference(self) -> dict[str, Any] | None:
        path = self.root / "reports" / "ml" / "ml_realistic_portfolio_5y_enriched_v2.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("summary") or []
        return next((row for row in rows if row.get("config_id") == "risk_adjusted_return_top10_size200000_pos5_close_20d_turnover50000000"), None)

    def _summary_row(self, profile: str, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "profile": profile,
            "final_assets": summary.get("final_assets"),
            "net_profit": summary.get("net_cumulative_profit"),
            "win_rate": summary.get("win_rate"),
            "profit_factor": summary.get("profit_factor"),
            "max_drawdown": summary.get("max_drawdown"),
            "total_trades": summary.get("total_trades"),
            "average_holding_days": summary.get("average_holding_days"),
        }

    def _win_rate(self, df: pd.DataFrame) -> float | None:
        if df.empty or "net_profit" not in df.columns:
            return None
        return float((df["net_profit"] > 0).mean())

    def _profit_factor(self, df: pd.DataFrame) -> float | None:
        if df.empty or "net_profit" not in df.columns:
            return None
        gross_profit = float(df.loc[df["net_profit"] > 0, "net_profit"].sum())
        gross_loss = float(-df.loc[df["net_profit"] < 0, "net_profit"].sum())
        return gross_profit / gross_loss if gross_loss else None

    def _simple_drawdown(self, df: pd.DataFrame) -> float | None:
        if df.empty or "net_profit" not in df.columns:
            return None
        equity = 1_000_000 + df.sort_values("exit_date")["net_profit"].cumsum()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return float(drawdown.min()) if not drawdown.empty else None

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            values = [self._format_value(row.get(column)) for column in columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def _format_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
