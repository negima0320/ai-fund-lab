"""Phase 13-A Horizon Reality Audit.

This is a 2025-only, read-only audit for horizon mismatch across Stock
Selection, Valuation, Downside, and Exit/Hold signals. It uses existing
artifacts only. It does not train models, regenerate predictions, run a full
backtest, change profiles, overwrite models, or call external APIs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase11e_exit_dd_guard import _numeric, _safe_float
from ml.phase12a_dynamic_capital_allocation import ARTIFACT_PATH
from ml.phase12b_limited_allocation_strategy_check import END_DATE, START_DATE


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase13a_horizon_reality_audit_2025"
LABEL_ROOT = Path("data/ml/labels")
HORIZONS = [5, 10, 20, 40]
BUCKETS = ["top5", "top10", "top50", "top100", "top_decile"]
SCORE_COLUMNS = [
    "stock_selection_rank_score",
    "candidate_strength",
    "opportunity_proba",
    "downside_safe_score",
    "opportunity_downside_score",
]
REQUIRED_REPORT_KEYS = [
    "recommended_candidate_generation_horizon",
    "recommended_valuation_horizon",
    "recommended_downside_horizon",
    "recommended_exit_hold_horizon",
    "stock_selection_action",
    "phase13b_recommendation",
    "ready_for_phase13b",
    "leakage_risk",
    "blocking_issues",
]


@dataclass(frozen=True)
class Phase13APaths:
    markdown: Path
    json: Path


class Phase13AHorizonRealityAudit:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)

    def run(self) -> Phase13APaths:
        report = self.build_report()
        return self.save_outputs(report)

    def build_report(self) -> dict[str, Any]:
        data, source_info = self.load_comparison_dataset()
        leakage = self.leakage_checklist()
        horizon_quality = self.horizon_score_quality(data)
        monotonicity = self.monotonicity_audit(data)
        stock_vs_valuation = self.stock_selection_vs_valuation(data)
        exit_hold = self.exit_hold_horizon_audit(data)
        recommendations = self.recommendations(horizon_quality, monotonicity, stock_vs_valuation, exit_hold, source_info, leakage)
        return {
            "metadata": self.metadata(),
            "input_artifact_summary": self.input_artifact_summary(data, source_info, leakage),
            "horizon_score_quality": horizon_quality,
            "score_monotonicity": monotonicity,
            "monotonicity_pass_by_score_and_horizon": self.monotonicity_pass_table(monotonicity),
            "best_horizon_by_score": self.best_horizon_by_score(monotonicity),
            "weak_horizon_by_score": self.weak_horizon_by_score(monotonicity),
            "stock_selection_vs_candidate_strength_vs_valuation": stock_vs_valuation,
            "exit_hold_horizon_audit": exit_hold,
            "final_recommendation": recommendations,
            "leakage_checklist": leakage,
            **{key: recommendations.get(key) for key in REQUIRED_REPORT_KEYS},
        }

    def load_comparison_dataset(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        artifact_path = self.root / ARTIFACT_PATH
        data = pd.read_parquet(artifact_path)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        data = data[(data["date"] >= START_DATE) & (data["date"] <= END_DATE)].copy()
        base_columns = [
            "date",
            "code",
            "close",
            "turnover_value",
            "stock_selection_rank_score",
            "risk_adjusted_score",
            "expected_return",
            "candidate_strength",
            "opportunity_proba",
            "downside_bad_proba",
            "opportunity_rank_percentile",
            "downside_rank_percentile",
            "future_return_20d",
            "future_max_return_20d",
            "future_max_drawdown_20d",
            "opportunity_value_20d",
            "opportunity_top_decile_20d",
            "downside_bad_20d",
        ]
        data = data[[column for column in base_columns if column in data.columns]].drop_duplicates(["date", "code"])

        labels = self.load_existing_labels()
        if not labels.empty:
            data = data.merge(labels, on=["date", "code"], how="left", suffixes=("", "_label"))
            rename_map = {
                "future_5d_return": "future_return_5d",
                "future_10d_return": "future_return_10d",
            }
            for old, new in rename_map.items():
                if old in data.columns and new not in data.columns:
                    data[new] = data[old]
            for column in ["future_5d_return", "future_10d_return", "future_max_return_10d", "future_max_return_20d"]:
                label_column = f"{column}_label"
                if label_column in data.columns:
                    data[column] = data[column].combine_first(data[label_column]) if column in data.columns else data[label_column]
                    data = data.drop(columns=[label_column])
            if "future_5d_return" in data.columns:
                data = data.drop(columns=["future_5d_return"])
            if "future_10d_return" in data.columns:
                data = data.drop(columns=["future_10d_return"])
            if "bad_entry_10d" in data.columns:
                data["downside_bad_10d"] = _numeric(data["bad_entry_10d"])
        for column in data.columns:
            if column not in {"date", "code"}:
                data[column] = _numeric(data[column])
        data["downside_safe_score"] = 1.0 - _numeric(data.get("downside_bad_proba")).fillna(1.0)
        data["opportunity_downside_score"] = _numeric(data.get("opportunity_proba")).fillna(0.0) * data["downside_safe_score"].clip(lower=0.0, upper=1.0)
        data = self.add_horizon_labels(data)
        available_future_columns = [column for column in self.expected_future_columns() if column in data.columns and not data[column].isna().all()]
        missing_columns = [column for column in self.expected_future_columns() if column not in data.columns or data[column].isna().all()]
        source_info = {
            "source_files": [str(artifact_path), str(self.root / LABEL_ROOT)],
            "available_score_columns": [column for column in SCORE_COLUMNS if column in data.columns],
            "available_future_columns": available_future_columns,
            "missing_columns": missing_columns,
            "label_files_loaded": int(labels["date"].nunique()) if not labels.empty and "date" in labels.columns else 0,
        }
        required = ["date", "code", "close", "turnover_value", "opportunity_proba", "downside_bad_proba"]
        data = data.dropna(subset=[column for column in required if column in data.columns])
        return data.sort_values(["date", "code"]).reset_index(drop=True), source_info

    def load_existing_labels(self) -> pd.DataFrame:
        label_dir = self.root / LABEL_ROOT
        if not label_dir.exists():
            return pd.DataFrame()
        frames = []
        for path in sorted(label_dir.glob("labels_2025-*.parquet")):
            frame = pd.read_parquet(path)
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["code"] = frame["code"].astype("string")
            frame = frame[(frame["date"] >= START_DATE) & (frame["date"] <= END_DATE)].copy()
            keep = [
                "date",
                "code",
                "future_5d_return",
                "future_10d_return",
                "bad_entry_10d",
                "future_max_return_10d",
                "future_max_return_20d",
                "future_swing_success_20d",
            ]
            frames.append(frame[[column for column in keep if column in frame.columns]])
        if not frames:
            return pd.DataFrame()
        labels = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "code"])
        for column in labels.columns:
            if column not in {"date", "code"}:
                labels[column] = _numeric(labels[column])
        return labels

    def add_horizon_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for horizon in HORIZONS:
            ret = f"future_return_{horizon}d"
            if ret in result.columns:
                values = _numeric(result[ret])
                cutoff = values.quantile(0.90) if values.notna().any() else math.nan
                result[f"top_decile_{horizon}d"] = values.ge(cutoff).astype(float) if not math.isnan(cutoff) else pd.NA
        if "downside_bad_20d" not in result.columns and "future_max_drawdown_20d" in result.columns:
            result["downside_bad_20d"] = _numeric(result["future_max_drawdown_20d"]).le(-0.10).astype(float)
        return result

    def expected_future_columns(self) -> list[str]:
        columns = []
        for horizon in HORIZONS:
            columns.extend(
                [
                    f"future_return_{horizon}d",
                    f"future_max_return_{horizon}d",
                    f"future_max_drawdown_{horizon}d",
                    f"top_decile_{horizon}d",
                    f"downside_bad_{horizon}d",
                ]
            )
        return columns

    def horizon_score_quality(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for score in SCORE_COLUMNS:
            if score not in data.columns:
                continue
            for bucket in BUCKETS:
                frame = self.bucket_frame(data, score, bucket)
                for horizon in HORIZONS:
                    rows.append(
                        {
                            "score": score,
                            "bucket": bucket,
                            "horizon": f"{horizon}d",
                            **self.horizon_metrics(frame, horizon),
                        }
                    )
        return rows

    def bucket_frame(self, data: pd.DataFrame, score: str, bucket: str) -> pd.DataFrame:
        if bucket.startswith("top") and bucket != "top_decile":
            n = int(bucket.removeprefix("top"))
            return self.top_n_by_day(data, score, n)
        if bucket == "top_decile":
            frame = data.copy()
            frame["_score_pct"] = frame.groupby("date")[score].rank(method="average", pct=True)
            return frame[frame["_score_pct"] >= 0.90].copy()
        raise ValueError(f"Unknown bucket: {bucket}")

    def top_n_by_day(self, data: pd.DataFrame, score: str, n: int) -> pd.DataFrame:
        return (
            data.sort_values(["date", score, "turnover_value", "code"], ascending=[True, False, False, True])
            .groupby("date", sort=False, group_keys=False)
            .head(n)
            .copy()
        )

    def horizon_metrics(self, frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
        return_col = f"future_return_{horizon}d"
        max_col = f"future_max_return_{horizon}d"
        dd_col = f"future_max_drawdown_{horizon}d"
        top_col = f"top_decile_{horizon}d"
        bad_col = f"downside_bad_{horizon}d"
        return {
            "sample_count": int(len(frame)),
            "candidate_days": int(frame["date"].nunique()) if not frame.empty else 0,
            "mean_future_return": self.mean(frame, return_col),
            "mean_future_max_return": self.mean(frame, max_col),
            "mean_future_max_drawdown": self.mean(frame, dd_col),
            "top_decile_rate": self.mean(frame, top_col),
            "downside_bad_rate": self.mean(frame, bad_col),
            "available_metrics": [column for column in [return_col, max_col, dd_col, top_col, bad_col] if column in frame.columns and not _numeric(frame[column]).dropna().empty],
        }

    def monotonicity_audit(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for score in SCORE_COLUMNS:
            if score not in data.columns:
                continue
            frame = data[["date", score, *[column for column in self.expected_future_columns() if column in data.columns]]].dropna(subset=[score]).copy()
            frame["score_percentile_in_day"] = frame.groupby("date")[score].rank(method="average", pct=True)
            frame["score_decile"] = (frame["score_percentile_in_day"] * 10).apply(math.ceil).clip(lower=1, upper=10).astype(int)
            for horizon in HORIZONS:
                decile_rows = []
                for decile, group in frame.groupby("score_decile", sort=True):
                    decile_rows.append({"decile": int(decile), **self.horizon_metrics(group.assign(code=""), horizon)})
                rows.append(
                    {
                        "score": score,
                        "horizon": f"{horizon}d",
                        "deciles": decile_rows,
                        "monotonicity_pass": self.monotonicity_pass(decile_rows),
                        "top_minus_bottom_return": self.top_minus_bottom(decile_rows, "mean_future_return"),
                        "top_minus_bottom_top_decile_rate": self.top_minus_bottom(decile_rows, "top_decile_rate"),
                    }
                )
        return rows

    def monotonicity_pass(self, decile_rows: list[dict[str, Any]]) -> bool:
        if len(decile_rows) < 2:
            return False
        return_delta = self.top_minus_bottom(decile_rows, "mean_future_return")
        rate_delta = self.top_minus_bottom(decile_rows, "top_decile_rate")
        return (return_delta is not None and return_delta > 0) or (rate_delta is not None and rate_delta > 0)

    def top_minus_bottom(self, decile_rows: list[dict[str, Any]], key: str) -> float | None:
        rows = [row for row in decile_rows if row.get(key) is not None]
        if len(rows) < 2:
            return None
        bottom = rows[0].get(key)
        top = rows[-1].get(key)
        if bottom is None or top is None:
            return None
        return _safe_float(float(top) - float(bottom))

    def monotonicity_pass_table(self, monotonicity: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "score": row["score"],
                "horizon": row["horizon"],
                "monotonicity_pass": row["monotonicity_pass"],
                "top_minus_bottom_return": row["top_minus_bottom_return"],
                "top_minus_bottom_top_decile_rate": row["top_minus_bottom_top_decile_rate"],
            }
            for row in monotonicity
        ]

    def best_horizon_by_score(self, monotonicity: list[dict[str, Any]]) -> dict[str, Any]:
        result = {}
        for score in SCORE_COLUMNS:
            rows = [row for row in monotonicity if row["score"] == score]
            if not rows:
                continue
            best = max(rows, key=lambda row: (row.get("top_minus_bottom_top_decile_rate") or -999, row.get("top_minus_bottom_return") or -999))
            result[score] = best["horizon"]
        return result

    def weak_horizon_by_score(self, monotonicity: list[dict[str, Any]]) -> dict[str, Any]:
        result = {}
        for score in SCORE_COLUMNS:
            rows = [row for row in monotonicity if row["score"] == score]
            if not rows:
                continue
            weak = min(rows, key=lambda row: (row.get("top_minus_bottom_top_decile_rate") or 0, row.get("top_minus_bottom_return") or 0))
            result[score] = weak["horizon"]
        return result

    def stock_selection_vs_valuation(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        specs = [
            ("candidate_universe_baseline", data),
            ("stock_selection_rank_score_top5", self.top_n_by_day(data, "stock_selection_rank_score", 5)),
            ("candidate_strength_top5", self.top_n_by_day(data, "candidate_strength", 5)),
            ("opportunity_top5", self.top_n_by_day(data, "opportunity_proba", 5)),
            ("opportunity_downside_top5", self.top_n_by_day(data, "opportunity_downside_score", 5)),
        ]
        rows = []
        for name, frame in specs:
            row = {
                "selection": name,
                "rows": int(len(frame)),
                "candidate_days": int(frame["date"].nunique()) if not frame.empty else 0,
            }
            for horizon in HORIZONS:
                metrics = self.horizon_metrics(frame, horizon)
                for key in ["mean_future_return", "mean_future_max_return", "mean_future_max_drawdown", "top_decile_rate", "downside_bad_rate"]:
                    row[f"{key}_{horizon}d"] = metrics[key]
            rows.append(row)
        return rows

    def exit_hold_horizon_audit(self, data: pd.DataFrame) -> dict[str, Any]:
        top = self.top_n_by_day(data, "opportunity_downside_score", 5)
        result: dict[str, Any] = {
            "scope": "opportunity_downside_score top5 by day",
            "sample_count": int(len(top)),
            "candidate_days": int(top["date"].nunique()) if not top.empty else 0,
        }
        if "future_return_5d" in top.columns and "future_return_20d" in top.columns:
            result["5d_fade_then_20d_recover_rate"] = self.rate((top["future_return_5d"] < 0) & (top["future_return_20d"] > 0))
        if "future_return_10d" in top.columns and "future_return_20d" in top.columns:
            result["10d_peakout_rate"] = self.rate((top["future_return_10d"] > 0.05) & (top["future_return_20d"] < top["future_return_10d"]))
        if "future_return_20d" in top.columns:
            result["20d_positive_rate"] = self.rate(top["future_return_20d"] > 0)
        if "future_return_40d" in top.columns:
            result["40d_positive_rate"] = self.rate(top["future_return_40d"] > 0)
        else:
            result["40d_status"] = "missing_in_existing_artifacts"
        if "future_max_return_20d" in top.columns and "future_return_20d" in top.columns:
            turned_loser = (_numeric(top["future_max_return_20d"]) >= 0.05) & (_numeric(top["future_return_20d"]) < 0)
            result["peak_5pct_to_final_loss_20d_count"] = int(turned_loser.sum())
            result["peak_5pct_to_final_loss_20d_rate"] = self.rate(turned_loser)
        if "future_max_drawdown_20d" in top.columns and "future_return_20d" in top.columns:
            result["20d_positive_with_large_drawdown_rate"] = self.rate((_numeric(top["future_return_20d"]) > 0) & (_numeric(top["future_max_drawdown_20d"]) <= -0.10))
        result["exit_hold_implication"] = self.exit_hold_implication(result)
        return result

    def exit_hold_implication(self, result: dict[str, Any]) -> str:
        if result.get("peak_5pct_to_final_loss_20d_rate", 0) and result.get("peak_5pct_to_final_loss_20d_rate", 0) > 0.05:
            return "Add profit-protection / break-even labels before broad strategy work."
        if result.get("5d_fade_then_20d_recover_rate", 0) and result.get("5d_fade_then_20d_recover_rate", 0) > 0.20:
            return "Avoid short-horizon exits for candidates whose 20d opportunity remains strong."
        return "20d hold/exit remains the best-supported horizon; 40d requires labels before conclusions."

    def recommendations(
        self,
        horizon_quality: list[dict[str, Any]],
        monotonicity: list[dict[str, Any]],
        stock_vs_valuation: list[dict[str, Any]],
        exit_hold: dict[str, Any],
        source_info: dict[str, Any],
        leakage: dict[str, Any],
    ) -> dict[str, Any]:
        best_candidate = self.best_bucket(horizon_quality, bucket="top50", prefer_scores=["opportunity_downside_score", "candidate_strength", "opportunity_proba"])
        best_valuation = self.best_bucket(horizon_quality, bucket="top5", prefer_scores=["opportunity_downside_score", "opportunity_proba"])
        best_downside = self.best_downside_horizon(horizon_quality)
        stock_action = self.stock_selection_action(stock_vs_valuation)
        missing_40d = any(column.endswith("_40d") for column in source_info.get("missing_columns", []))
        ready = leakage["leakage_risk"] == "low" and not leakage["blocking_issues"] and bool(best_candidate)
        return {
            "recommended_candidate_generation_horizon": best_candidate.get("horizon") if best_candidate else "insufficient_evidence",
            "recommended_valuation_horizon": best_valuation.get("horizon") if best_valuation else "insufficient_evidence",
            "recommended_downside_horizon": best_downside or "20d",
            "recommended_exit_hold_horizon": "20d_with_profit_protection_audit" if missing_40d else "20d_or_40d_followup",
            "stock_selection_action": stock_action,
            "phase13b_recommendation": "Phase13-B Candidate Generation Redesign with no prefilter vs candidate_strength top50/top100 vs valuation-first",
            "ready_for_phase13b": ready,
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
            "reason": self.recommendation_reason(best_candidate, best_valuation, best_downside, stock_action, exit_hold, missing_40d),
        }

    def best_bucket(self, rows: list[dict[str, Any]], *, bucket: str, prefer_scores: list[str]) -> dict[str, Any] | None:
        candidates = [row for row in rows if row["bucket"] == bucket and row["score"] in prefer_scores and row["top_decile_rate"] is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda row: (row.get("top_decile_rate") or -1, row.get("mean_future_return") or -1, -(row.get("downside_bad_rate") or 1)))

    def best_downside_horizon(self, rows: list[dict[str, Any]]) -> str | None:
        candidates = [row for row in rows if row["score"] == "opportunity_downside_score" and row["bucket"] == "top5" and row.get("downside_bad_rate") is not None]
        if not candidates:
            return None
        best = min(candidates, key=lambda row: row.get("downside_bad_rate") or 999)
        return best["horizon"]

    def stock_selection_action(self, stock_vs_valuation: list[dict[str, Any]]) -> str:
        rows = {row["selection"]: row for row in stock_vs_valuation}
        stock = rows.get("stock_selection_rank_score_top5", {})
        candidate_strength = rows.get("candidate_strength_top5", {})
        opportunity = rows.get("opportunity_top5", {})
        universe = rows.get("candidate_universe_baseline", {})
        stock_rate = stock.get("top_decile_rate_20d") or 0.0
        candidate_rate = candidate_strength.get("top_decile_rate_20d") or 0.0
        opportunity_rate = opportunity.get("top_decile_rate_20d") or 0.0
        universe_rate = universe.get("top_decile_rate_20d") or 0.0
        if opportunity_rate > candidate_rate and opportunity_rate > stock_rate:
            return "valuation_first"
        if candidate_rate > stock_rate and candidate_rate > universe_rate:
            return "replace_with_candidate_strength"
        if stock_rate < universe_rate:
            return "remove_prefilter"
        return "insufficient_evidence"

    def recommendation_reason(
        self,
        best_candidate: dict[str, Any] | None,
        best_valuation: dict[str, Any] | None,
        best_downside: str | None,
        stock_action: str,
        exit_hold: dict[str, Any],
        missing_40d: bool,
    ) -> str:
        return (
            f"Candidate horizon evidence points to {best_candidate.get('horizon') if best_candidate else 'insufficient evidence'}; "
            f"valuation top5 evidence points to {best_valuation.get('horizon') if best_valuation else 'insufficient evidence'}; "
            f"downside evidence points to {best_downside or '20d fallback'}; "
            f"stock_selection_action={stock_action}; "
            f"exit_hold={exit_hold.get('exit_hold_implication')}; "
            f"40d labels missing={missing_40d}."
        )

    def input_artifact_summary(self, data: pd.DataFrame, source_info: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_files": source_info["source_files"],
            "row_count": int(len(data)),
            "date_min": data["date"].min().date().isoformat() if not data.empty else None,
            "date_max": data["date"].max().date().isoformat() if not data.empty else None,
            "unique_code_count": int(data["code"].nunique()) if not data.empty else 0,
            "available_score_columns": source_info["available_score_columns"],
            "available_future_columns": source_info["available_future_columns"],
            "missing_columns": source_info["missing_columns"],
            "label_files_loaded": source_info["label_files_loaded"],
            "leakage_risk": leakage["leakage_risk"],
            "blocking_issues": leakage["blocking_issues"],
        }

    def mean(self, frame: pd.DataFrame, column: str) -> float | None:
        if column not in frame.columns:
            return None
        values = _numeric(frame[column]).dropna()
        return _safe_float(values.mean()) if not values.empty else None

    def rate(self, mask: pd.Series) -> float | None:
        return _safe_float(mask.mean()) if len(mask) else None

    def leakage_checklist(self) -> dict[str, Any]:
        return {
            "future_columns_used_as_features": [],
            "future_columns_used_only_for_evaluation": self.expected_future_columns(),
            "backtest_columns_used_as_features": [],
            "trade_result_columns_used_as_features": [],
            "cash_or_portfolio_columns_used_as_features": [],
            "selected_or_bought_used_as_features": False,
            "current_pm_multiplier_used": False,
            "new_model_trained": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "full_backtest_executed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
            "period": {"start": START_DATE, "end": END_DATE},
            "leakage_risk": "low",
            "blocking_issues": [],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "phase": "13-A",
            "scope": "2025-only horizon reality audit",
            "new_model_trained": False,
            "full_backtest_executed": False,
            "existing_model_overwritten": False,
            "profile_changed": False,
            "historical_predictions_regenerated": False,
            "jquants_api_called": False,
            "openai_api_called": False,
        }

    def save_outputs(self, report: dict[str, Any]) -> Phase13APaths:
        report_dir = self.root / "reports/ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        markdown_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(report), encoding="utf-8")
        return Phase13APaths(markdown=markdown_path, json=json_path)

    def to_markdown(self, report: dict[str, Any]) -> str:
        rec = report["final_recommendation"]
        lines = [
            "# Phase 13-A Horizon Reality Audit",
            "",
            "## Input Artifact Summary",
            "",
            self.table([report["input_artifact_summary"]], ["row_count", "date_min", "date_max", "unique_code_count", "available_score_columns", "available_future_columns", "missing_columns", "leakage_risk", "blocking_issues"]),
            "",
            "## Final Recommendation",
            "",
            self.table([rec], REQUIRED_REPORT_KEYS + ["reason"]),
            "",
            "## Stock Selection vs Candidate Strength vs Valuation",
            "",
            self.table(report["stock_selection_vs_candidate_strength_vs_valuation"], ["selection", "rows", "candidate_days", "mean_future_return_5d", "top_decile_rate_5d", "mean_future_return_10d", "top_decile_rate_10d", "mean_future_return_20d", "top_decile_rate_20d", "downside_bad_rate_20d"]),
            "",
            "## Horizon Score Quality",
            "",
            self.table(report["horizon_score_quality"], ["score", "bucket", "horizon", "sample_count", "candidate_days", "mean_future_return", "mean_future_max_return", "mean_future_max_drawdown", "top_decile_rate", "downside_bad_rate"]),
            "",
            "## Monotonicity Pass",
            "",
            self.table(report["monotonicity_pass_by_score_and_horizon"], ["score", "horizon", "monotonicity_pass", "top_minus_bottom_return", "top_minus_bottom_top_decile_rate"]),
            "",
            "## Exit / Hold Horizon Audit",
            "",
            self.table([report["exit_hold_horizon_audit"]], ["scope", "sample_count", "candidate_days", "5d_fade_then_20d_recover_rate", "10d_peakout_rate", "20d_positive_rate", "40d_status", "peak_5pct_to_final_loss_20d_count", "peak_5pct_to_final_loss_20d_rate", "20d_positive_with_large_drawdown_rate", "exit_hold_implication"]),
            "",
            "## Leakage Checklist",
            "",
            self.table([report["leakage_checklist"]], ["future_columns_used_as_features", "future_columns_used_only_for_evaluation", "new_model_trained", "existing_model_overwritten", "profile_changed", "full_backtest_executed", "historical_predictions_regenerated", "jquants_api_called", "openai_api_called", "leakage_risk", "blocking_issues"]),
            "",
        ]
        return "\n".join(lines)

    def table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(self.format_value(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def format_value(self, value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return f"{value:.4f}"
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, list):
            return ", ".join(map(str, value))
        if value is None:
            return ""
        return str(value)
