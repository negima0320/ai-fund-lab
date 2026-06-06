from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_PROFILE = "rookie_dealer_02_v2_66_ml_ranked"
EXIT_050_PROFILE = "rookie_dealer_02_v2_68_ml_ranked_exit_ai_050"
EXIT_060_PROFILE = "rookie_dealer_02_v2_70_ml_ranked_exit_ai_060"
PERIOD_KEY = "2023-01-01_to_2026-05-31"


@dataclass(frozen=True)
class ExitAITriggerAuditPaths:
    markdown: Path
    json: Path
    trigger_trades_csv: Path
    trade_delta_csv: Path


class ExitAITriggerAudit:
    """Audit actual Exit AI triggers from existing backtest logs only."""

    def __init__(
        self,
        root: str | Path = ".",
        base_profile: str = BASE_PROFILE,
        exit_050_profile: str = EXIT_050_PROFILE,
        exit_060_profile: str = EXIT_060_PROFILE,
        period_key: str = PERIOD_KEY,
        comparison_json: str | Path | None = None,
        exit_dataset_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.base_profile = base_profile
        self.exit_050_profile = exit_050_profile
        self.exit_060_profile = exit_060_profile
        self.period_key = period_key
        self.comparison_json = Path(comparison_json) if comparison_json else self.root / "reports" / "ml" / "ml_exit_ai_backtest_comparison_2023-01_to_2026-05.json"
        self.exit_dataset_path = Path(exit_dataset_path) if exit_dataset_path else self.root / "data" / "ml" / "exit_datasets" / "exit_dataset_v2_66_2023-01_to_2026-05.parquet"

    def build(self) -> dict[str, Any]:
        base = self._prepare_for_match(self._load_trades(self.base_profile), "v2_66")
        exit_050 = self._prepare_for_match(self._load_trades(self.exit_050_profile), "v2_68")
        exit_060 = self._prepare_for_match(self._load_trades(self.exit_060_profile), "v2_70")
        exit_features = self._load_exit_dataset_features()
        comparison = self._load_comparison()

        delta_050 = self._compare_profiles(base, exit_050, "v2_68")
        delta_060 = self._compare_profiles(base, exit_060, "v2_70")
        trigger_050 = self._trigger_trades(exit_050, delta_050, exit_features, "v2_68")
        trigger_060 = self._trigger_trades(exit_060, delta_060, exit_features, "v2_70")

        result = {
            "period": self.period_key,
            "source": {
                "base_profile": self.base_profile,
                "exit_050_profile": self.exit_050_profile,
                "exit_060_profile": self.exit_060_profile,
                "comparison_json": str(self.comparison_json),
                "exit_dataset_path": str(self.exit_dataset_path),
            },
            "match_audit": {
                "v2_68_vs_v2_66": self._match_audit(delta_050, len(base), len(exit_050)),
                "v2_70_vs_v2_66": self._match_audit(delta_060, len(base), len(exit_060)),
            },
            "v2_68_trigger_summary": self._trigger_summary(trigger_050),
            "v2_68_improvement_summary": self._improvement_summary(trigger_050),
            "v2_68_improvement_top10": self._records(trigger_050.sort_values("profit_delta", ascending=False).head(10)),
            "v2_68_worsening_top10": self._records(trigger_050.sort_values("profit_delta", ascending=True).head(10)),
            "march_2026_analysis": self._march_analysis(base, exit_050, delta_050, trigger_050),
            "drawdown_analysis": self._drawdown_analysis(base, exit_050, delta_050, trigger_050, comparison),
            "v2_70_comparison": self._v2_70_comparison(trigger_050, trigger_060, delta_050, delta_060),
            "monthly_delta": self._monthly_delta(delta_050),
            "trigger_trades_v2_68": self._records(trigger_050),
            "trade_delta_v2_66_vs_v2_68": self._records(delta_050),
        }
        result["diagnosis"] = self._diagnosis(result)
        result["trigger_trades_df"] = trigger_050
        result["trade_delta_df"] = delta_050
        return result

    def save(self, result: dict[str, Any]) -> ExitAITriggerAuditPaths:
        out_dir = self.root / "reports" / "ml"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "exit_ai_trigger_audit_2023-01_to_2026-05"
        markdown = out_dir / f"{stem}.md"
        json_path = out_dir / f"{stem}.json"
        trigger_csv = out_dir / "exit_ai_trigger_trades_v2_68_2023-01_to_2026-05.csv"
        delta_csv = out_dir / "exit_ai_trade_delta_v2_66_vs_v2_68_2023-01_to_2026-05.csv"

        trigger_df = result.pop("trigger_trades_df")
        delta_df = result.pop("trade_delta_df")
        trigger_df.to_csv(trigger_csv, index=False)
        delta_df.to_csv(delta_csv, index=False)
        markdown.write_text(self.format_markdown(result), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        result["trigger_trades_df"] = trigger_df
        result["trade_delta_df"] = delta_df
        return ExitAITriggerAuditPaths(markdown=markdown, json=json_path, trigger_trades_csv=trigger_csv, trade_delta_csv=delta_csv)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# Exit AI Trigger Audit",
            "",
            f"- period: `{result['period']}`",
            "- source: existing `logs/backtests` and existing Exit AI comparison report only",
            "- note: no backtest rerun, no trading logic change, no API fetch",
            "",
            "## Match Audit",
            "",
            self._table(
                [
                    {"pair": "v2_68_vs_v2_66", **result["match_audit"]["v2_68_vs_v2_66"]},
                    {"pair": "v2_70_vs_v2_66", **result["match_audit"]["v2_70_vs_v2_66"]},
                ],
                ["pair", "base_trades", "target_trades", "matched_count", "base_only_count", "target_only_count", "match_rate_vs_target"],
            ),
            "",
            "## v2_68 Trigger Summary",
            "",
            self._table(
                [result["v2_68_trigger_summary"]],
                [
                    "trigger_count",
                    "triggered_total_profit",
                    "triggered_win_rate",
                    "average_probability",
                    "average_profit_delta",
                    "positive_delta_count",
                    "negative_delta_count",
                ],
            ),
            "",
            "## v2_68 Improvement Summary",
            "",
            self._table(
                [result["v2_68_improvement_summary"]],
                ["improved_trade_count", "worsened_trade_count", "improvement_total", "worsening_total", "net_effect"],
            ),
            "",
            "## Improvement Top 10",
            "",
            self._table(result["v2_68_improvement_top10"], self._detail_columns()),
            "",
            "## Worsening Top 10",
            "",
            self._table(result["v2_68_worsening_top10"], self._detail_columns()),
            "",
            "## 2026-03 Analysis",
            "",
            self._table([result["march_2026_analysis"]["summary"]], [
                "v2_66_march_profit",
                "v2_68_march_profit",
                "profit_delta",
                "matched_delta",
                "triggered_delta",
                "v2_66_only_profit",
                "v2_68_only_profit",
            ]),
            "",
            "### 2026-03 Major Deltas",
            "",
            self._table(result["march_2026_analysis"]["major_deltas"], self._detail_columns()),
            "",
            "### 2026-03 v2_66-Only Trades",
            "",
            self._table(
                result["march_2026_analysis"]["v2_66_only_trades"],
                ["code", "v2_66_signal_date", "v2_66_entry_date", "v2_66_exit_date", "v2_66_exit_reason", "v2_66_net_profit", "v2_66_holding_days"],
            ),
            "",
            "### 2026-03 v2_68-Only Trades",
            "",
            self._table(
                result["march_2026_analysis"]["v2_68_only_trades"],
                ["code", "v2_68_signal_date", "v2_68_entry_date", "v2_68_exit_date", "v2_68_exit_reason", "v2_68_net_profit", "v2_68_holding_days"],
            ),
            "",
            "## Drawdown Analysis",
            "",
            self._table(
                [result["drawdown_analysis"]["summary"]],
                [
                    "v2_66_max_drawdown",
                    "v2_66_max_drawdown_date",
                    "v2_68_max_drawdown",
                    "v2_68_max_drawdown_date",
                    "drawdown_improvement",
                    "large_loss_mitigated_count",
                    "large_loss_mitigation_total",
                ],
            ),
            "",
            "### Loss Mitigation Trades",
            "",
            self._table(result["drawdown_analysis"]["loss_mitigation_trades"], self._detail_columns()),
            "",
            "## v2_70 Comparison",
            "",
            self._table(result["v2_70_comparison"]["sets"], ["set", "trade_count", "v2_68_delta_sum", "v2_70_delta_sum"]),
            "",
            "## Diagnosis",
            "",
        ]
        for item in result["diagnosis"]:
            lines.append(f"- {item}")
        lines.append("")
        return "\n".join(lines)

    def _load_trades(self, profile: str) -> pd.DataFrame:
        path = self.root / "logs" / "backtests" / profile / self.period_key / "trades.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "action" in df.columns:
            df = df[df["action"].astype(str).eq("SELL")].copy()
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")
        for column in [
            "holding_days",
            "net_profit",
            "net_profit_rate",
            "exit_ai_probability",
            "exit_ai_threshold",
            "expected_return_10d",
            "bad_entry_probability_10d",
            "risk_adjusted_score",
        ]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        df["code"] = df["code"].astype(str)
        if "net_profit" not in df.columns and "profit" in df.columns:
            df["net_profit"] = pd.to_numeric(df["profit"], errors="coerce")
        return df.reset_index(drop=True)

    def _prepare_for_match(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        df = df.copy()
        sort_cols = [column for column in ["entry_date", "signal_date", "exit_date", "code", "trade_id"] if column in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df["_entry_code_occurrence"] = df.groupby(["entry_date", "code"], dropna=False).cumcount()
        df["match_key"] = (
            df["entry_date"].dt.strftime("%Y-%m-%d").fillna("")
            + "|"
            + df["code"].astype(str)
            + "|"
            + df["_entry_code_occurrence"].astype(str)
        )
        rename = {}
        for column in [
            "trade_id",
            "signal_date",
            "entry_date",
            "exit_date",
            "exit_reason",
            "holding_days",
            "net_profit",
            "net_profit_rate",
            "exit_ai_triggered",
            "exit_ai_signal",
            "exit_ai_probability",
            "exit_ai_threshold",
        ]:
            if column in df.columns:
                rename[column] = f"{prefix}_{column}"
        return df.rename(columns=rename)

    def _compare_profiles(self, base: pd.DataFrame, target: pd.DataFrame, target_prefix: str) -> pd.DataFrame:
        keep_base = [
            "match_key",
            "code",
            "v2_66_trade_id",
            "v2_66_signal_date",
            "v2_66_entry_date",
            "v2_66_exit_date",
            "v2_66_exit_reason",
            "v2_66_holding_days",
            "v2_66_net_profit",
            "v2_66_net_profit_rate",
        ]
        keep_target = [
            "match_key",
            f"{target_prefix}_trade_id",
            f"{target_prefix}_signal_date",
            f"{target_prefix}_entry_date",
            f"{target_prefix}_exit_date",
            f"{target_prefix}_exit_reason",
            f"{target_prefix}_holding_days",
            f"{target_prefix}_net_profit",
            f"{target_prefix}_net_profit_rate",
            f"{target_prefix}_exit_ai_triggered",
            f"{target_prefix}_exit_ai_signal",
            f"{target_prefix}_exit_ai_probability",
            f"{target_prefix}_exit_ai_threshold",
        ]
        left = base[[column for column in keep_base if column in base.columns]].copy()
        right = target[[column for column in keep_target if column in target.columns]].copy()
        merged = left.merge(right, on="match_key", how="outer", indicator=True)
        if "code" not in merged.columns:
            merged["code"] = pd.NA
        target_code = target[["match_key", "code"]].drop_duplicates("match_key")
        merged = merged.merge(target_code.rename(columns={"code": "target_code"}), on="match_key", how="left")
        merged["code"] = merged["code"].fillna(merged["target_code"])
        merged.drop(columns=["target_code"], inplace=True)
        merged["match_status"] = merged["_merge"].map({"both": "matched", "left_only": "base_only", "right_only": "target_only"})
        merged.drop(columns=["_merge"], inplace=True)
        merged["profit_delta"] = pd.to_numeric(merged.get(f"{target_prefix}_net_profit"), errors="coerce") - pd.to_numeric(merged.get("v2_66_net_profit"), errors="coerce")
        merged["holding_days_delta"] = pd.to_numeric(merged.get(f"{target_prefix}_holding_days"), errors="coerce") - pd.to_numeric(merged.get("v2_66_holding_days"), errors="coerce")
        return merged

    def _load_exit_dataset_features(self) -> pd.DataFrame:
        columns = [
            "trade_id",
            "code",
            "entry_date",
            "current_date",
            "expected_return_10d",
            "bad_entry_probability_10d",
            "risk_adjusted_score",
        ]
        if not self.exit_dataset_path.exists():
            return pd.DataFrame(columns=columns)
        df = pd.read_parquet(self.exit_dataset_path, columns=[column for column in columns if column != "trade_id"] + ["trade_id"])
        for column in ["entry_date", "current_date"]:
            df[column] = pd.to_datetime(df[column], errors="coerce")
        df["code"] = df["code"].astype(str)
        return df.sort_values(["entry_date", "current_date", "code"]).drop_duplicates(["entry_date", "current_date", "code"], keep="last")

    def _trigger_trades(self, target: pd.DataFrame, delta: pd.DataFrame, exit_features: pd.DataFrame, prefix: str) -> pd.DataFrame:
        triggered_col = f"{prefix}_exit_ai_triggered"
        if triggered_col not in delta.columns:
            return delta.iloc[0:0].copy()
        triggered = delta[self._truthy(delta[triggered_col])].copy()
        if triggered.empty:
            return triggered
        triggered["avoid_loss_5d_probability"] = pd.to_numeric(triggered.get(f"{prefix}_exit_ai_probability"), errors="coerce")
        triggered["exit_ai_threshold"] = pd.to_numeric(triggered.get(f"{prefix}_exit_ai_threshold"), errors="coerce")
        triggered["signal_date"] = triggered.get(f"{prefix}_signal_date")
        triggered["entry_date"] = triggered.get(f"{prefix}_entry_date")
        triggered["exit_date"] = triggered.get(f"{prefix}_exit_date")
        triggered["exit_reason"] = triggered.get(f"{prefix}_exit_reason")
        triggered["holding_days"] = triggered.get(f"{prefix}_holding_days")
        triggered["net_profit"] = triggered.get(f"{prefix}_net_profit")
        triggered["net_profit_rate"] = triggered.get(f"{prefix}_net_profit_rate")
        if not exit_features.empty:
            triggered = triggered.merge(
                exit_features.rename(columns={"current_date": "exit_date"}),
                on=["entry_date", "exit_date", "code"],
                how="left",
                suffixes=("", "_feature"),
            )
        for column in ["expected_return_10d", "bad_entry_probability_10d", "risk_adjusted_score"]:
            if column not in triggered.columns:
                triggered[column] = pd.NA
        return triggered.sort_values(["exit_date", "code"]).reset_index(drop=True)

    def _trigger_summary(self, triggered: pd.DataFrame) -> dict[str, Any]:
        profits = pd.to_numeric(triggered.get("net_profit"), errors="coerce").fillna(0.0)
        deltas = pd.to_numeric(triggered.get("profit_delta"), errors="coerce").dropna()
        probabilities = pd.to_numeric(triggered.get("avoid_loss_5d_probability"), errors="coerce").dropna()
        return {
            "trigger_count": int(len(triggered)),
            "triggered_total_profit": float(profits.sum()) if not profits.empty else 0.0,
            "triggered_win_rate": self._win_rate(profits),
            "average_probability": float(probabilities.mean()) if not probabilities.empty else None,
            "average_profit_delta": float(deltas.mean()) if not deltas.empty else None,
            "positive_delta_count": int((deltas > 0).sum()) if not deltas.empty else 0,
            "negative_delta_count": int((deltas < 0).sum()) if not deltas.empty else 0,
        }

    def _improvement_summary(self, triggered: pd.DataFrame) -> dict[str, Any]:
        deltas = pd.to_numeric(triggered.get("profit_delta"), errors="coerce").dropna()
        improved = deltas[deltas > 0]
        worsened = deltas[deltas < 0]
        return {
            "improved_trade_count": int(len(improved)),
            "worsened_trade_count": int(len(worsened)),
            "improvement_total": float(improved.sum()) if not improved.empty else 0.0,
            "worsening_total": float(worsened.sum()) if not worsened.empty else 0.0,
            "net_effect": float(deltas.sum()) if not deltas.empty else 0.0,
        }

    def _march_analysis(self, base: pd.DataFrame, exit_050: pd.DataFrame, delta: pd.DataFrame, triggered: pd.DataFrame) -> dict[str, Any]:
        base_march = self._month_filter(base, "v2_66_exit_date", "2026-03")
        exit_march = self._month_filter(exit_050, "v2_68_exit_date", "2026-03")
        delta_march = delta[
            self._month_mask(delta.get("v2_66_exit_date"), "2026-03") | self._month_mask(delta.get("v2_68_exit_date"), "2026-03")
        ].copy()
        triggered_march = triggered[
            self._month_mask(triggered.get("v2_66_exit_date"), "2026-03") | self._month_mask(triggered.get("v2_68_exit_date"), "2026-03")
        ].copy()
        summary = {
            "v2_66_march_profit": self._sum(base_march.get("v2_66_net_profit")),
            "v2_68_march_profit": self._sum(exit_march.get("v2_68_net_profit")),
            "profit_delta": self._sum(exit_march.get("v2_68_net_profit")) - self._sum(base_march.get("v2_66_net_profit")),
            "matched_delta": self._sum(delta_march[delta_march["match_status"].eq("matched")].get("profit_delta")),
            "triggered_delta": self._sum(triggered_march.get("profit_delta")),
            "v2_66_only_profit": self._sum(delta_march[delta_march["match_status"].eq("base_only")].get("v2_66_net_profit")),
            "v2_68_only_profit": self._sum(delta_march[delta_march["match_status"].eq("target_only")].get("v2_68_net_profit")),
            "v2_66_trade_count": int(len(base_march)),
            "v2_68_trade_count": int(len(exit_march)),
            "triggered_trade_count": int(len(triggered_march)),
        }
        return {
            "summary": summary,
            "major_deltas": self._records(delta_march.dropna(subset=["profit_delta"]).reindex(delta_march["profit_delta"].abs().sort_values(ascending=False).index).head(20)),
            "v2_66_only_trades": self._records(delta_march[delta_march["match_status"].eq("base_only")].sort_values("v2_66_net_profit", ascending=False).head(20)),
            "v2_68_only_trades": self._records(delta_march[delta_march["match_status"].eq("target_only")].sort_values("v2_68_net_profit", ascending=True).head(20)),
            "triggered_trades": self._records(triggered_march),
        }

    def _load_comparison(self) -> dict[str, Any]:
        if not self.comparison_json.exists():
            return {}
        return json.loads(self.comparison_json.read_text(encoding="utf-8"))

    def _drawdown_analysis(
        self,
        base: pd.DataFrame,
        exit_050: pd.DataFrame,
        delta: pd.DataFrame,
        triggered: pd.DataFrame,
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        base_dd = self._drawdown(base, "v2_66_exit_date", "v2_66_net_profit")
        exit_dd = self._drawdown(exit_050, "v2_68_exit_date", "v2_68_net_profit")
        reported_base_dd = self._reported_max_drawdown(comparison, self.base_profile)
        reported_exit_dd = self._reported_max_drawdown(comparison, self.exit_050_profile)
        loss_mitigation = triggered[
            (pd.to_numeric(triggered.get("v2_66_net_profit"), errors="coerce") < -10_000)
            & (pd.to_numeric(triggered.get("profit_delta"), errors="coerce") > 0)
        ].copy()
        summary = {
            "v2_66_max_drawdown": reported_base_dd if reported_base_dd is not None else base_dd["max_drawdown"],
            "v2_66_max_drawdown_date": base_dd["max_drawdown_date"],
            "v2_68_max_drawdown": reported_exit_dd if reported_exit_dd is not None else exit_dd["max_drawdown"],
            "v2_68_max_drawdown_date": exit_dd["max_drawdown_date"],
            "drawdown_improvement": (
                (reported_exit_dd if reported_exit_dd is not None else exit_dd["max_drawdown"])
                - (reported_base_dd if reported_base_dd is not None else base_dd["max_drawdown"])
            )
            if (reported_base_dd is not None or base_dd["max_drawdown"] is not None) and (reported_exit_dd is not None or exit_dd["max_drawdown"] is not None)
            else None,
            "trade_cashflow_v2_66_max_drawdown": base_dd["max_drawdown"],
            "trade_cashflow_v2_68_max_drawdown": exit_dd["max_drawdown"],
            "large_loss_mitigated_count": int(len(loss_mitigation)),
            "large_loss_mitigation_total": self._sum(loss_mitigation.get("profit_delta")),
        }
        return {
            "summary": summary,
            "loss_mitigation_trades": self._records(loss_mitigation.sort_values("profit_delta", ascending=False).head(20)),
            "v2_66_drawdown_monthly": base_dd["monthly"],
            "v2_68_drawdown_monthly": exit_dd["monthly"],
        }

    def _v2_70_comparison(self, trigger_050: pd.DataFrame, trigger_060: pd.DataFrame, delta_050: pd.DataFrame, delta_060: pd.DataFrame) -> dict[str, Any]:
        keys_050 = set(trigger_050["match_key"].dropna().astype(str))
        keys_060 = set(trigger_060["match_key"].dropna().astype(str))
        rows = []
        for name, keys in [
            ("both", keys_050 & keys_060),
            ("v2_68_only", keys_050 - keys_060),
            ("v2_70_only", keys_060 - keys_050),
        ]:
            d68 = delta_050[delta_050["match_key"].astype(str).isin(keys)]
            d70 = delta_060[delta_060["match_key"].astype(str).isin(keys)]
            rows.append(
                {
                    "set": name,
                    "trade_count": int(len(keys)),
                    "v2_68_delta_sum": self._sum(d68.get("profit_delta")),
                    "v2_70_delta_sum": self._sum(d70.get("profit_delta")),
                }
            )
        return {"sets": rows}

    def _reported_max_drawdown(self, comparison: dict[str, Any], profile: str) -> float | None:
        for row in comparison.get("summary", []):
            if row.get("profile") == profile and row.get("max_drawdown") is not None:
                return float(row["max_drawdown"])
        return None

    def _monthly_delta(self, delta: pd.DataFrame) -> list[dict[str, Any]]:
        df = delta.dropna(subset=["v2_68_exit_date"]).copy()
        df["month"] = pd.to_datetime(df["v2_68_exit_date"], errors="coerce").dt.to_period("M").astype(str)
        rows = []
        for month, group in df.groupby("month"):
            rows.append(
                {
                    "month": str(month),
                    "profit_delta": self._sum(group.get("profit_delta")),
                    "matched_count": int(group["match_status"].eq("matched").sum()),
                    "target_only_count": int(group["match_status"].eq("target_only").sum()),
                }
            )
        return rows

    def _match_audit(self, delta: pd.DataFrame, base_count: int, target_count: int) -> dict[str, Any]:
        matched = int(delta["match_status"].eq("matched").sum())
        return {
            "base_trades": int(base_count),
            "target_trades": int(target_count),
            "matched_count": matched,
            "base_only_count": int(delta["match_status"].eq("base_only").sum()),
            "target_only_count": int(delta["match_status"].eq("target_only").sum()),
            "match_rate_vs_target": matched / target_count if target_count else None,
        }

    def _diagnosis(self, result: dict[str, Any]) -> list[str]:
        improvement = result["v2_68_improvement_summary"]
        march = result["march_2026_analysis"]["summary"]
        dd = result["drawdown_analysis"]["summary"]
        v70_sets = {row["set"]: row for row in result["v2_70_comparison"]["sets"]}
        diagnosis = [
            f"v2_68 Exit AI triggered {result['v2_68_trigger_summary']['trigger_count']} trades; net delta vs v2_66 on triggered trades was {improvement['net_effect']:.0f}.",
            f"2026-03 deterioration was {march['profit_delta']:.0f}; triggered-trade delta in that month was {march['triggered_delta']:.0f}, so both early exits and changed trade sequence should be reviewed.",
            f"Drawdown improved by {dd['drawdown_improvement']:.4f} with {dd['large_loss_mitigated_count']} large-loss mitigation trades totaling {dd['large_loss_mitigation_total']:.0f}.",
        ]
        both = v70_sets.get("both", {})
        only68 = v70_sets.get("v2_68_only", {})
        only70 = v70_sets.get("v2_70_only", {})
        diagnosis.append(
            "v2_70 is worth considering when it keeps most shared improvements while avoiding weak v2_68-only triggers: "
            f"both={both.get('trade_count')}, v2_68_only={only68.get('trade_count')}, v2_70_only={only70.get('trade_count')}."
        )
        return diagnosis

    def _drawdown(self, df: pd.DataFrame, date_col: str, profit_col: str, initial_assets: float = 1_000_000.0) -> dict[str, Any]:
        if df.empty or date_col not in df.columns or profit_col not in df.columns:
            return {"max_drawdown": None, "max_drawdown_date": None, "monthly": []}
        work = df.dropna(subset=[date_col]).sort_values(date_col).copy()
        work["profit"] = pd.to_numeric(work[profit_col], errors="coerce").fillna(0.0)
        work["equity"] = initial_assets + work["profit"].cumsum()
        work["peak"] = work["equity"].cummax()
        work["drawdown"] = work["equity"] / work["peak"] - 1.0
        min_idx = work["drawdown"].idxmin() if not work.empty else None
        work["month"] = pd.to_datetime(work[date_col], errors="coerce").dt.to_period("M").astype(str)
        monthly = []
        for month, group in work.groupby("month"):
            monthly.append({"month": str(month), "profit": float(group["profit"].sum()), "min_drawdown": float(group["drawdown"].min())})
        return {
            "max_drawdown": float(work.loc[min_idx, "drawdown"]) if min_idx is not None else None,
            "max_drawdown_date": self._date_text(work.loc[min_idx, date_col]) if min_idx is not None else None,
            "monthly": monthly,
        }

    def _month_filter(self, df: pd.DataFrame, date_col: str, month: str) -> pd.DataFrame:
        if date_col not in df.columns:
            return df.iloc[0:0]
        return df[self._month_mask(df[date_col], month)].copy()

    def _month_mask(self, values: Any, month: str) -> pd.Series:
        if values is None:
            return pd.Series(dtype=bool)
        return pd.to_datetime(values, errors="coerce").dt.to_period("M").astype(str).eq(month)

    def _detail_columns(self) -> list[str]:
        return [
            "signal_date",
            "entry_date",
            "exit_date",
            "code",
            "exit_reason",
            "holding_days",
            "net_profit",
            "net_profit_rate",
            "avoid_loss_5d_probability",
            "risk_adjusted_score",
            "expected_return_10d",
            "bad_entry_probability_10d",
            "v2_66_exit_date",
            "v2_68_exit_date",
            "v2_66_net_profit",
            "v2_68_net_profit",
            "profit_delta",
        ]

    def _records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        records = []
        for record in df.copy().to_dict("records"):
            cleaned = {}
            for key, value in record.items():
                if isinstance(value, pd.Timestamp):
                    cleaned[key] = self._date_text(value)
                elif pd.isna(value):
                    cleaned[key] = None
                else:
                    cleaned[key] = value
            records.append(cleaned)
        return records

    def _truthy(self, series: pd.Series) -> pd.Series:
        return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})

    def _sum(self, values: Any) -> float:
        if values is None:
            return 0.0
        series = pd.to_numeric(values, errors="coerce").fillna(0.0)
        return float(series.sum())

    def _win_rate(self, profits: pd.Series) -> float | None:
        if profits.empty:
            return None
        return float((profits > 0).mean())

    def _date_text(self, value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return pd.to_datetime(value).strftime("%Y-%m-%d")

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
