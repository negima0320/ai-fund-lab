from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROFILE = "rookie_dealer_02_v2_71_ml_ranked_exit_ai_050_scaled_buy"
PERIOD = "2023-01-01_to_2026-05-31"


@dataclass(frozen=True)
class ScaledBuyAuditPaths:
    markdown: Path
    json: Path
    trades_csv: Path


class ScaledBuyAudit:
    """Audit scaled-buy concentration and side effects from existing logs only."""

    def __init__(
        self,
        root: str | Path = ".",
        profile: str = PROFILE,
        period_key: str = PERIOD,
        daily_buy_limit: float = 900_000,
    ) -> None:
        self.root = Path(root)
        self.profile = profile
        self.period_key = period_key
        self.daily_buy_limit = float(daily_buy_limit)
        self.backtest_dir = self.root / "logs" / "backtests" / profile / period_key
        self.report_dir = self.root / "reports" / "ml"

    def build(self) -> dict[str, Any]:
        trades = self._load_scaled_trades()
        all_trades = self._load_all_sell_trades()
        comparison = self._load_comparison()
        summary = self._load_summary()
        daily = self._load_daily_summary()
        enriched = self._enrich_trades(trades)

        total_stats = self._trade_stats(enriched)
        without_67400 = self._trade_stats(enriched[~enriched["code"].astype(str).eq("67400")])
        exclusion_rows = [
            {"case": "all_scaled_buy", **total_stats},
            {"case": "exclude_67400", **without_67400},
        ]
        for n in [1, 3, 5]:
            exclusion_rows.append({"case": f"exclude_top{n}_profit", **self._trade_stats(self._exclude_top_profit(enriched, n))})

        concentration = {
            "by_code": self._group_profit(enriched, "code"),
            "by_month": self._group_profit(enriched, "month"),
            "by_year": self._group_profit(enriched, "year"),
            "summary": self._concentration_summary(enriched),
        }
        comparisons = self._profile_comparisons(comparison, total_stats)
        dd = self._drawdown_audit(daily, enriched)
        size = self._size_summary(enriched)

        result = {
            "period": self.period_key,
            "profile": self.profile,
            "source": {
                "trades_csv": str(self.backtest_dir / "trades.csv"),
                "scaled_buy_trades_csv": str(self.report_dir / "scaled_buy_trades_2023-01_to_2026-05.csv"),
                "comparison_json": str(self.report_dir / "scaled_buy_backtest_comparison_2023-01_to_2026-05.json"),
            },
            "v2_71_summary": self._profile_summary_from_payload(summary, all_trades),
            "scaled_buy_stats": total_stats,
            "exclusion_sensitivity": exclusion_rows,
            "concentration": concentration,
            "profile_comparison": comparisons,
            "order_size": size,
            "drawdown": dd,
            "adoption_judgement": self._judgement(total_stats, concentration["summary"], comparisons, dd),
            "scaled_buy_trades": self._records(enriched),
            "scaled_buy_trades_df": enriched,
        }
        return result

    def save(self, result: dict[str, Any]) -> ScaledBuyAuditPaths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "scaled_buy_audit_v2_71_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        csv_path = self.report_dir / "scaled_buy_audit_trades_v2_71_2023-01_to_2026-05.csv"

        df = result.pop("scaled_buy_trades_df")
        df.to_csv(csv_path, index=False)
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        result["scaled_buy_trades_df"] = df
        return ScaledBuyAuditPaths(markdown=markdown, json=json_path, trades_csv=csv_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Scaled Buy Audit v2_71",
            "",
            f"- profile: `{result['profile']}`",
            f"- period: `{result['period']}`",
            "- source: existing backtest logs and generated scaled-buy comparison reports only",
            "- note: no backtest rerun, no API fetch, no trading logic change",
            "",
            "## Scaled Buy Performance",
            "",
            self._table([result["scaled_buy_stats"]], self._stats_columns()),
            "",
            "## Exclusion Sensitivity",
            "",
            self._table(result["exclusion_sensitivity"], ["case", *self._stats_columns()]),
            "",
            "## Concentration Summary",
            "",
            self._table([result["concentration"]["summary"]], ["scaled_profit_total", "67400_profit", "67400_contribution_rate", "top3_code_contribution_rate", "top5_code_contribution_rate"]),
            "",
            "### Code Contribution",
            "",
            self._table(result["concentration"]["by_code"], ["code", "trade_count", "net_profit", "win_rate", "profit_factor"]),
            "",
            "### Month Contribution",
            "",
            self._table(result["concentration"]["by_month"], ["month", "trade_count", "net_profit", "win_rate", "profit_factor"]),
            "",
            "### Year Contribution",
            "",
            self._table(result["concentration"]["by_year"], ["year", "trade_count", "net_profit", "win_rate", "profit_factor"]),
            "",
            "## Profile Comparison",
            "",
            self._table(result["profile_comparison"], ["case", "net_profit", "profit_delta_vs_v2_68", "trade_count", "notes"]),
            "",
            "## Order Size",
            "",
            self._table(
                [result["order_size"]],
                [
                    "average_original_amount",
                    "average_scaled_amount",
                    "average_scale_ratio",
                    "median_scale_ratio",
                    "average_daily_limit_utilization",
                    "max_daily_limit_utilization",
                    "average_share_reduction_rate",
                ],
            ),
            "",
            "## Drawdown Estimate",
            "",
            self._table(
                [
                    result["drawdown"]["baseline"],
                    result["drawdown"]["without_scaled_buy"],
                    result["drawdown"]["without_67400"],
                ],
                ["case", "final_assets", "max_drawdown", "max_drawdown_date"],
            ),
            "",
            "### Scaled Buy Months DD",
            "",
            self._table(result["drawdown"]["scaled_buy_monthly_dd"], ["month", "scaled_trade_count", "scaled_profit", "monthly_max_drawdown"]),
            "",
            "## Adoption Judgement",
            "",
        ]
        lines.extend(f"- {item}" for item in result["adoption_judgement"])
        lines.append("")
        return "\n".join(lines)

    def _load_scaled_trades(self) -> pd.DataFrame:
        path = self.report_dir / "scaled_buy_trades_2023-01_to_2026-05.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def _load_all_sell_trades(self) -> pd.DataFrame:
        path = self.backtest_dir / "trades.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def _load_comparison(self) -> dict[str, Any]:
        path = self.report_dir / "scaled_buy_backtest_comparison_2023-01_to_2026-05.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_summary(self) -> dict[str, Any]:
        path = self.backtest_dir / "backtest_summary.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _load_daily_summary(self) -> pd.DataFrame:
        path = self.backtest_dir / "summary.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for column in ["total_assets", "net_total_assets", "net_cumulative_profit"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def _enrich_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")
        numeric_columns = [
            "net_profit",
            "gross_profit",
            "profit",
            "net_profit_rate",
            "original_amount",
            "scaled_amount",
            "original_planned_shares",
            "scaled_shares",
            "shares",
        ]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        if "net_profit" not in df.columns and "profit" in df.columns:
            df["net_profit"] = df["profit"]
        df["month"] = df["exit_date"].dt.to_period("M").astype(str)
        df["year"] = df["exit_date"].dt.year.astype("Int64").astype(str)
        df["scale_ratio"] = self._safe_div(df.get("scaled_amount"), df.get("original_amount"))
        df["amount_reduction"] = pd.to_numeric(df.get("original_amount"), errors="coerce") - pd.to_numeric(df.get("scaled_amount"), errors="coerce")
        df["amount_reduction_rate"] = self._safe_div(df["amount_reduction"], df.get("original_amount"))
        df["share_reduction_rate"] = self._safe_div(
            pd.to_numeric(df.get("original_planned_shares"), errors="coerce") - pd.to_numeric(df.get("scaled_shares"), errors="coerce"),
            df.get("original_planned_shares"),
        )
        df["daily_limit_utilization"] = pd.to_numeric(df.get("scaled_amount"), errors="coerce") / self.daily_buy_limit
        return df

    def _trade_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        profits = pd.to_numeric(df.get("net_profit"), errors="coerce").fillna(0.0) if not df.empty else pd.Series(dtype=float)
        best = df.loc[profits.idxmax()].to_dict() if not profits.empty else {}
        worst = df.loc[profits.idxmin()].to_dict() if not profits.empty else {}
        return {
            "count": int(len(df)),
            "total_profit": float(profits.sum()) if not profits.empty else 0.0,
            "win_rate": self._win_rate(profits),
            "profit_factor": self._profit_factor(profits),
            "average_profit": float(profits.mean()) if not profits.empty else None,
            "median_profit": float(profits.median()) if not profits.empty else None,
            "best_trade": self._trade_label(best),
            "worst_trade": self._trade_label(worst),
        }

    def _exclude_top_profit(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        return df.sort_values("net_profit", ascending=False).iloc[n:].copy()

    def _group_profit(self, df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        rows = []
        if df.empty or column not in df.columns:
            return rows
        for key, group in df.groupby(df[column].astype(str), dropna=False):
            stats = self._trade_stats(group)
            rows.append(
                {
                    column: key,
                    "trade_count": stats["count"],
                    "net_profit": stats["total_profit"],
                    "win_rate": stats["win_rate"],
                    "profit_factor": stats["profit_factor"],
                }
            )
        return sorted(rows, key=lambda row: row["net_profit"], reverse=True)

    def _concentration_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        total = float(pd.to_numeric(df.get("net_profit"), errors="coerce").fillna(0.0).sum()) if not df.empty else 0.0
        code_rows = self._group_profit(df, "code")
        profit_67400 = next((row["net_profit"] for row in code_rows if str(row.get("code")) == "67400"), 0.0)
        top3 = sum(row["net_profit"] for row in code_rows[:3])
        top5 = sum(row["net_profit"] for row in code_rows[:5])
        return {
            "scaled_profit_total": total,
            "67400_profit": profit_67400,
            "67400_contribution_rate": profit_67400 / total if total else None,
            "top3_code_contribution_rate": top3 / total if total else None,
            "top5_code_contribution_rate": top5 / total if total else None,
        }

    def _profile_comparisons(self, comparison: dict[str, Any], scaled_stats: dict[str, Any]) -> list[dict[str, Any]]:
        summary = {row["profile"]: row for row in comparison.get("summary", [])}
        v68 = summary.get("rookie_dealer_02_v2_68_ml_ranked_exit_ai_050", {})
        v71 = summary.get(self.profile, {})
        v68_profit = self._number(v68.get("net_profit"))
        v71_profit = self._number(v71.get("net_profit"))
        scaled_profit = self._number(scaled_stats.get("total_profit"))
        return [
            {
                "case": "v2_68",
                "net_profit": v68_profit,
                "profit_delta_vs_v2_68": 0.0,
                "trade_count": v68.get("total_trades"),
                "notes": "baseline Exit AI 0.50 without scaled buy",
            },
            {
                "case": "v2_71",
                "net_profit": v71_profit,
                "profit_delta_vs_v2_68": v71_profit - v68_profit,
                "trade_count": v71.get("total_trades"),
                "notes": "actual scaled-buy backtest",
            },
            {
                "case": "v2_71_excluding_scaled_buy_trades",
                "net_profit": v71_profit - scaled_profit,
                "profit_delta_vs_v2_68": (v71_profit - scaled_profit) - v68_profit,
                "trade_count": int((v71.get("total_trades") or 0) - (scaled_stats.get("count") or 0)),
                "notes": "rough arithmetic exclusion; portfolio path not re-simulated",
            },
        ]

    def _size_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "average_original_amount": self._mean(df, "original_amount"),
            "average_scaled_amount": self._mean(df, "scaled_amount"),
            "average_scale_ratio": self._mean(df, "scale_ratio"),
            "median_scale_ratio": self._median(df, "scale_ratio"),
            "average_daily_limit_utilization": self._mean(df, "daily_limit_utilization"),
            "max_daily_limit_utilization": self._max(df, "daily_limit_utilization"),
            "average_share_reduction_rate": self._mean(df, "share_reduction_rate"),
        }

    def _drawdown_audit(self, daily: pd.DataFrame, scaled: pd.DataFrame) -> dict[str, Any]:
        baseline = self._equity_dd(daily, "baseline")
        without_scaled = self._adjusted_equity_dd(daily, scaled, "without_scaled_buy")
        without_67400 = self._adjusted_equity_dd(daily, scaled[~scaled["code"].astype(str).eq("67400")], "without_67400")
        return {
            "baseline": baseline,
            "without_scaled_buy": without_scaled,
            "without_67400": without_67400,
            "scaled_buy_monthly_dd": self._scaled_monthly_dd(daily, scaled),
            "note": "DD exclusions are path estimates by subtracting realized scaled-buy profits from total_assets after each exit date; no backtest rerun.",
        }

    def _adjusted_equity_dd(self, daily: pd.DataFrame, excluded: pd.DataFrame, case: str) -> dict[str, Any]:
        if daily.empty or excluded.empty:
            return self._equity_dd(daily, case)
        df = daily.copy()
        df["adjustment"] = 0.0
        for _, trade in excluded.iterrows():
            exit_date = trade.get("exit_date")
            profit = self._number(trade.get("net_profit"))
            if pd.isna(exit_date):
                continue
            df.loc[df["date"] >= exit_date, "adjustment"] += profit
        df["total_assets"] = pd.to_numeric(df.get("total_assets"), errors="coerce") - df["adjustment"]
        return self._equity_dd(df, case)

    def _equity_dd(self, daily: pd.DataFrame, case: str) -> dict[str, Any]:
        if daily.empty or "total_assets" not in daily.columns:
            return {"case": case, "final_assets": None, "max_drawdown": None, "max_drawdown_date": None}
        df = daily.dropna(subset=["date", "total_assets"]).copy()
        equity = pd.to_numeric(df["total_assets"], errors="coerce")
        peaks = equity.cummax()
        drawdowns = equity / peaks - 1
        idx = drawdowns.idxmin() if not drawdowns.empty else None
        return {
            "case": case,
            "final_assets": float(equity.iloc[-1]) if not equity.empty else None,
            "max_drawdown": float(drawdowns.loc[idx]) if idx is not None and not pd.isna(idx) else None,
            "max_drawdown_date": df.loc[idx, "date"].strftime("%Y-%m-%d") if idx is not None and not pd.isna(idx) else None,
        }

    def _scaled_monthly_dd(self, daily: pd.DataFrame, scaled: pd.DataFrame) -> list[dict[str, Any]]:
        if daily.empty or scaled.empty:
            return []
        months = sorted(scaled["month"].dropna().unique())
        rows = []
        for month in months:
            month_daily = daily[daily["date"].dt.to_period("M").astype(str).eq(str(month))]
            month_trades = scaled[scaled["month"].eq(month)]
            dd = self._equity_dd(month_daily, str(month))
            rows.append(
                {
                    "month": str(month),
                    "scaled_trade_count": int(len(month_trades)),
                    "scaled_profit": float(pd.to_numeric(month_trades["net_profit"], errors="coerce").fillna(0.0).sum()),
                    "monthly_max_drawdown": dd.get("max_drawdown"),
                }
            )
        return rows

    def _profile_summary_from_payload(self, payload: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
        profits = pd.to_numeric(trades.get("net_profit"), errors="coerce").fillna(0.0) if not trades.empty and "net_profit" in trades.columns else pd.Series(dtype=float)
        return {
            "final_assets": payload.get("final_assets"),
            "net_profit": payload.get("net_cumulative_profit") or float(profits.sum()),
            "win_rate": payload.get("win_rate") or self._win_rate(profits),
            "profit_factor": payload.get("profit_factor") or self._profit_factor(profits),
            "max_drawdown": payload.get("max_drawdown"),
            "total_trades": int(len(trades)),
        }

    def _judgement(self, stats: dict[str, Any], concentration: dict[str, Any], comparisons: list[dict[str, Any]], dd: dict[str, Any]) -> list[str]:
        rows = {row["case"]: row for row in comparisons}
        notes = [
            f"Scaled buy generated {self._format(stats.get('total_profit'))} across {stats.get('count')} closed trades.",
            f"67400 contribution rate is {self._format_pct(concentration.get('67400_contribution_rate'))}; top3 code contribution is {self._format_pct(concentration.get('top3_code_contribution_rate'))}.",
        ]
        if rows.get("v2_71"):
            notes.append(f"v2_71 remains ahead of v2_68 by {self._format(rows['v2_71'].get('profit_delta_vs_v2_68'))}.")
        if rows.get("v2_71_excluding_scaled_buy_trades"):
            delta = rows["v2_71_excluding_scaled_buy_trades"].get("profit_delta_vs_v2_68")
            notes.append(f"Removing scaled-buy trades arithmetically leaves delta vs v2_68 at {self._format(delta)}, so the edge is materially tied to scaled-buy executions.")
        baseline_dd = dd.get("baseline", {}).get("max_drawdown")
        no_scaled_dd = dd.get("without_scaled_buy", {}).get("max_drawdown")
        notes.append(f"Estimated DD baseline {self._format_pct(baseline_dd)} vs without scaled-buy {self._format_pct(no_scaled_dd)}.")
        notes.append("Adoption candidate: yes for v2_71, but monitor code/month concentration and keep scaled_buy profile-gated before generalizing.")
        return notes

    def _safe_div(self, numerator: Any, denominator: Any) -> pd.Series:
        n = pd.to_numeric(numerator, errors="coerce")
        d = pd.to_numeric(denominator, errors="coerce")
        return n.where(d.ne(0)) / d.where(d.ne(0))

    def _mean(self, df: pd.DataFrame, column: str) -> float | None:
        if df.empty or column not in df.columns:
            return None
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(series.mean()) if not series.empty else None

    def _median(self, df: pd.DataFrame, column: str) -> float | None:
        if df.empty or column not in df.columns:
            return None
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(series.median()) if not series.empty else None

    def _max(self, df: pd.DataFrame, column: str) -> float | None:
        if df.empty or column not in df.columns:
            return None
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(series.max()) if not series.empty else None

    def _number(self, value: Any) -> float:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return 0.0 if pd.isna(numeric) else float(numeric)

    def _win_rate(self, profits: pd.Series) -> float | None:
        if profits.empty:
            return None
        return float((profits > 0).mean())

    def _profit_factor(self, profits: pd.Series) -> float | None:
        if profits.empty:
            return None
        gross_profit = float(profits[profits > 0].sum())
        gross_loss = float(-profits[profits < 0].sum())
        return gross_profit / gross_loss if gross_loss else None

    def _trade_label(self, row: dict[str, Any]) -> str | None:
        if not row:
            return None
        entry = row.get("entry_date")
        entry_text = entry.strftime("%Y-%m-%d") if hasattr(entry, "strftime") else str(entry)
        return f"{entry_text} {row.get('code')} {self._format(row.get('net_profit'))}"

    def _records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        return df.where(pd.notna(df), None).to_dict("records")

    def _stats_columns(self) -> list[str]:
        return ["count", "total_profit", "win_rate", "profit_factor", "average_profit", "median_profit", "best_trade", "worst_trade"]

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(self._format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _format_pct(self, value: Any) -> str:
        if value is None:
            return ""
        numeric = self._number(value)
        return f"{numeric * 100:.2f}%"
