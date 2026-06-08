"""Phase 9-B2 / 9-F2 PM AI v3 coverage root cause audit.

Read-only audit for diagnosing why Phase 9-F PM v3 feature lookup coverage was
0%. Backtest artifacts are used only to inspect lookup coverage and are never
treated as PM AI v3 training features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_v3_dataset_builder import FORBIDDEN_TOKENS, LABEL_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
PERIOD = "2023-01-01_to_2026-05-31"
REPORT_STEM = "phase9b2_pm_ai_v3_coverage_root_cause_2023-01_to_2026-05"
DATASET_PATH = Path("data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_2023-01_to_2026-05.parquet")
FEATURE_COLUMNS_PATH = Path("models/ml/portfolio_manager_v3/candidate_phase9d/feature_columns.json")
PHASE9F_REPORT = Path("reports/ml/phase9f_pm_ai_v3_backtest_candidate_2023-01_to_2026-05.json")
V293_PROFILES = {
    "v2_93_a": "rookie_dealer_02_v2_93_pm_ai_v3_candidate",
    "v2_93_b": "rookie_dealer_02_v2_93b_pm_ai_v3_candidate_conservative",
    "v2_93_c": "rookie_dealer_02_v2_93c_pm_ai_v3_candidate_half_pm130",
}
V282_PROFILE = "rookie_dealer_02_v2_82_cap38"


@dataclass(frozen=True)
class Phase9B2Paths:
    markdown: Path
    json: Path


def normalize_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text
    return digits.zfill(4)


def normalize_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def previous_business_day(value: Any) -> str:
    text = normalize_date(value)
    if not text:
        return ""
    return (pd.Timestamp(text) - pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")


def next_business_day(value: Any) -> str:
    text = normalize_date(value)
    if not text:
        return ""
    return (pd.Timestamp(text) + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class PMAIV3CoverageRootCauseAudit:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        dataset_path: Path | str = DATASET_PATH,
        period: str = PERIOD,
        profiles: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.dataset_path = self._root(Path(dataset_path))
        self.period = period
        self.profiles = profiles or V293_PROFILES

    def build_report(self) -> dict[str, Any]:
        dataset = self._load_dataset_keys()
        backtest = self._load_backtest_pm_sizing_universe()
        coverage = self._coverage_matrix(dataset, backtest)
        dataset_audit = self._dataset_universe_audit(dataset)
        backtest_audit = self._backtest_universe_audit(backtest)
        diff = self._candidate_universe_diff(dataset, backtest)
        leakage = self._leakage_audit()
        cause = self._root_cause(dataset_audit, backtest_audit, coverage, diff)
        return {
            "metadata": {
                "phase": "9-B2/9-F2",
                "audit_only": True,
                "training_executed": False,
                "mapping_adjustment_executed": False,
                "strategy_backtest_executed": False,
                "openai_api_called": False,
                "jquants_api_refetched": False,
                "live_order_executed": False,
                "current_model_regenerated_historical_predictions": False,
                "current_pm_ai_overwritten": False,
                "current_exit_ai_overwritten": False,
                "v2_82_profile_overwritten": False,
            },
            "input_paths": {
                "dataset": str(self.dataset_path),
                "phase9f_report": str(self._root(PHASE9F_REPORT)),
                "backtest_profiles": self.profiles,
            },
            "dataset_universe": dataset_audit,
            "backtest_pm_sizing_universe": backtest_audit,
            "coverage_matrix": coverage,
            "candidate_universe_diff": diff,
            "root_cause": cause,
            "recommended_fix": self._recommended_fix(cause),
            "coverage_improvement_goal": {
                "minimum_pm_v3_feature_coverage": 0.95,
                "forbidden_feature_count": 0,
                "leakage_risk": "low",
                "current_pm_exit_v282_overwrite": False,
                "next_phase_if_not_met": "Phase 9-B3 Dataset Builder Fix",
                "phase9f_rerun_allowed_now": False,
            },
            "leakage_audit": leakage,
        }

    def save_report(self, report: dict[str, Any]) -> Phase9B2Paths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase9B2Paths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# PM AI v3 Phase 9-B2 / 9-F2 Coverage Root Cause Audit",
                "",
                "## Scope",
                "",
                "- read-only root cause audit for Phase 9-F PM v3 lookup coverage 0%",
                "- no training, no mapping adjustment, no strategy backtest rerun",
                "- backtest artifacts are used for coverage evaluation only, not as PM AI v3 features",
                "",
                "## Dataset Universe",
                "",
                self._table([report["dataset_universe"]], ["row_count", "date_count", "code_count", "date_min", "date_max", "candidate_count_per_day_min", "candidate_count_per_day_median", "candidate_count_per_day_max", "top10_fixed", "data_source_values"]),
                "",
                "## Backtest PM Sizing Universe",
                "",
                self._table([report["backtest_pm_sizing_universe"]], ["pm_sizing_call_count", "unique_date_count", "unique_code_count", "pm_v3_feature_found_count", "pm_v3_feature_coverage", "top_missing_reason"]),
                "",
                "## Coverage Matrix",
                "",
                self._table(report["coverage_matrix"], ["key", "matched_rows", "unmatched_rows", "coverage_rate", "sample_matched_keys", "sample_unmatched_keys"]),
                "",
                "## Candidate Universe Diff",
                "",
                self._table([report["candidate_universe_diff"]], ["dataset_only_dates_count", "backtest_only_dates_count", "common_dates_count", "dataset_only_codes_count", "backtest_only_codes_count", "common_codes_count", "same_date_code_overlap_rate", "universe_mismatch"]),
                "",
                "## Root Cause",
                "",
                self._table([report["root_cause"]], ["date_mismatch", "code_mismatch", "universe_mismatch", "top_missing_reason", "primary_root_cause", "phase9f_is_valid_pm_v3_performance_test"]),
                "",
                "## Recommended Fix",
                "",
                self._table([report["recommended_fix"]], ["next_phase", "direct_phase9f_rerun_allowed", "recommended_action"]),
                "",
                "## Leakage",
                "",
                self._table([report["leakage_audit"]], ["forbidden_feature_count", "forbidden_feature_columns", "leakage_risk", "backtest_artifacts_used_for_evaluation_only", "backtest_artifacts_used_as_features"]),
                "",
            ]
        )

    def _root(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def _load_dataset_keys(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            return pd.DataFrame(columns=["prediction_date", "code", "normalized_code"])
        columns = ["prediction_date", "code"]
        optional = ["data_source", "relative_feature_timing", "candidate_count_in_day", "rank_in_day"]
        available = set(pd.read_parquet(self.dataset_path).columns)
        frame = pd.read_parquet(self.dataset_path, columns=[*columns, *[c for c in optional if c in available]])
        frame["prediction_date"] = frame["prediction_date"].map(normalize_date)
        frame["date"] = frame["prediction_date"]
        frame["code"] = frame["code"].astype(str)
        frame["normalized_code"] = frame["code"].map(normalize_code)
        frame["pm_v3_lookup_key"] = frame["prediction_date"] + "|" + frame["normalized_code"]
        return frame

    def _load_backtest_pm_sizing_universe(self) -> pd.DataFrame:
        frames = []
        for label, profile in self.profiles.items():
            path = self.root / "logs" / "backtests" / profile / self.period / "purchase_audit.csv"
            frame = _read_csv(path)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["profile_label"] = label
            frame["profile"] = profile
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        rows = pd.concat(frames, ignore_index=True, sort=False)
        pm_marker = rows.get("pm_model_version", pd.Series("", index=rows.index)).fillna("").astype(str).str.contains("pm_ai_v3")
        status_marker = rows.get("pm_status", pd.Series("", index=rows.index)).notna()
        missing_marker = rows.get("pm_missing_reason", pd.Series("", index=rows.index)).fillna("").astype(str).str.contains("pm_v3")
        rows = rows[pm_marker | status_marker | missing_marker].copy()
        rows["prediction_date"] = rows.get("signal_date", pd.Series("", index=rows.index)).map(normalize_date)
        rows["trade_date"] = rows.get("entry_date", pd.Series("", index=rows.index)).map(normalize_date)
        rows["buy_date"] = rows["trade_date"]
        rows["date"] = rows["prediction_date"]
        rows["previous_business_day_trade_date"] = rows["trade_date"].map(previous_business_day)
        rows["next_business_day_prediction_date"] = rows["prediction_date"].map(next_business_day)
        rows["code"] = rows.get("code", pd.Series("", index=rows.index)).astype(str)
        rows["normalized_code"] = rows["code"].map(normalize_code)
        rows["pm_v3_lookup_key"] = rows["prediction_date"] + "|" + rows["normalized_code"]
        return rows

    def _dataset_universe_audit(self, dataset: pd.DataFrame) -> dict[str, Any]:
        if dataset.empty:
            return {"row_count": 0}
        counts = dataset.groupby("prediction_date")["normalized_code"].nunique()
        return {
            "row_count": int(len(dataset)),
            "date_count": int(dataset["prediction_date"].nunique()),
            "code_count": int(dataset["normalized_code"].nunique()),
            "date_min": str(dataset["prediction_date"].min()),
            "date_max": str(dataset["prediction_date"].max()),
            "candidate_count_per_day_min": int(counts.min()) if not counts.empty else 0,
            "candidate_count_per_day_median": float(counts.median()) if not counts.empty else None,
            "candidate_count_per_day_max": int(counts.max()) if not counts.empty else 0,
            "top10_fixed": bool((counts == 10).all()) if not counts.empty else False,
            "data_source_values": self._value_counts(dataset.get("data_source")),
            "relative_feature_timing_values": self._value_counts(dataset.get("relative_feature_timing")),
            "sample_keys": dataset["pm_v3_lookup_key"].head(10).tolist(),
        }

    def _backtest_universe_audit(self, rows: pd.DataFrame) -> dict[str, Any]:
        if rows.empty:
            return {"pm_sizing_call_count": 0}
        found = rows.get("pm_feature_found", pd.Series(False, index=rows.index)).map(_truthy)
        missing = rows.get("pm_missing_reason", pd.Series("", index=rows.index)).fillna("").astype(str)
        counts = missing[missing.ne("")].value_counts()
        return {
            "pm_sizing_call_count": int(len(rows)),
            "unique_date_count": int(rows["prediction_date"].nunique()),
            "unique_code_count": int(rows["normalized_code"].nunique()),
            "pm_v3_feature_found_count": int(found.sum()),
            "pm_v3_feature_coverage": float(found.mean()) if len(rows) else None,
            "missing_reason_distribution": {str(k): int(v) for k, v in counts.items()},
            "top_missing_reason": str(counts.index[0]) if not counts.empty else "",
            "candidate_source_distribution": self._value_counts(rows.get("candidate_source")),
            "decision_distribution": self._value_counts(rows.get("decision")),
            "sample_missing_keys": rows.loc[~found, "pm_v3_lookup_key"].head(10).tolist(),
        }

    def _coverage_matrix(self, dataset: pd.DataFrame, rows: pd.DataFrame) -> list[dict[str, Any]]:
        if dataset.empty or rows.empty:
            return []
        dataset_keys = set((dataset["prediction_date"] + "|" + dataset["normalized_code"]).dropna())
        dataset_dates = set(dataset["prediction_date"].dropna())
        dataset_codes = set(dataset["normalized_code"].dropna())
        variants = {
            "prediction_date+code": rows["prediction_date"] + "|" + rows["normalized_code"],
            "trade_date+code": rows["trade_date"] + "|" + rows["normalized_code"],
            "buy_date+code": rows["buy_date"] + "|" + rows["normalized_code"],
            "previous_business_day(trade_date)+code": rows["previous_business_day_trade_date"] + "|" + rows["normalized_code"],
            "next_business_day(prediction_date)+code": rows["next_business_day_prediction_date"] + "|" + rows["normalized_code"],
        }
        out = [self._coverage_row(name, series, dataset_keys) for name, series in variants.items()]
        date_match = rows["prediction_date"].isin(dataset_dates)
        code_match = rows["normalized_code"].isin(dataset_codes)
        out.append(self._set_overlap_row("date-only overlap", date_match, rows["prediction_date"]))
        out.append(self._set_overlap_row("code-only overlap", code_match, rows["normalized_code"]))
        trimmed = rows["normalized_code"].str.lstrip("0")
        dataset_trimmed = {code.lstrip("0") for code in dataset_codes}
        out.append(self._set_overlap_row("normalized_code_no_left_zero overlap", trimmed.isin(dataset_trimmed), trimmed))
        return out

    def _coverage_row(self, name: str, keys: pd.Series, dataset_keys: set[str]) -> dict[str, Any]:
        match = keys.isin(dataset_keys)
        return {
            "key": name,
            "matched_rows": int(match.sum()),
            "unmatched_rows": int((~match).sum()),
            "coverage_rate": float(match.mean()) if len(match) else None,
            "sample_matched_keys": keys[match].head(5).tolist(),
            "sample_unmatched_keys": keys[~match].head(5).tolist(),
        }

    def _set_overlap_row(self, name: str, match: pd.Series, values: pd.Series) -> dict[str, Any]:
        return {
            "key": name,
            "matched_rows": int(match.sum()),
            "unmatched_rows": int((~match).sum()),
            "coverage_rate": float(match.mean()) if len(match) else None,
            "sample_matched_keys": values[match].head(5).tolist(),
            "sample_unmatched_keys": values[~match].head(5).tolist(),
        }

    def _candidate_universe_diff(self, dataset: pd.DataFrame, rows: pd.DataFrame) -> dict[str, Any]:
        if dataset.empty or rows.empty:
            return {"universe_mismatch": True}
        dataset_dates = set(dataset["prediction_date"])
        bt_dates = set(rows["prediction_date"])
        dataset_codes = set(dataset["normalized_code"])
        bt_codes = set(rows["normalized_code"])
        same_date_rates = []
        for date in sorted(dataset_dates & bt_dates):
            d_codes = set(dataset.loc[dataset["prediction_date"].eq(date), "normalized_code"])
            b_codes = set(rows.loc[rows["prediction_date"].eq(date), "normalized_code"])
            if b_codes:
                same_date_rates.append(len(d_codes & b_codes) / len(b_codes))
        same_date_rate = float(sum(same_date_rates) / len(same_date_rates)) if same_date_rates else 0.0
        return {
            "dataset_only_dates_count": len(dataset_dates - bt_dates),
            "backtest_only_dates_count": len(bt_dates - dataset_dates),
            "common_dates_count": len(dataset_dates & bt_dates),
            "dataset_only_codes_count": len(dataset_codes - bt_codes),
            "backtest_only_codes_count": len(bt_codes - dataset_codes),
            "common_codes_count": len(dataset_codes & bt_codes),
            "same_date_code_overlap_rate": same_date_rate,
            "dataset_only_date_samples": sorted(dataset_dates - bt_dates)[:10],
            "backtest_only_date_samples": sorted(bt_dates - dataset_dates)[:10],
            "dataset_only_code_samples": sorted(dataset_codes - bt_codes)[:10],
            "backtest_only_code_samples": sorted(bt_codes - dataset_codes)[:10],
            "universe_mismatch": same_date_rate < 0.95,
        }

    def _root_cause(
        self,
        dataset_audit: dict[str, Any],
        backtest_audit: dict[str, Any],
        coverage: list[dict[str, Any]],
        diff: dict[str, Any],
    ) -> dict[str, Any]:
        by_key = {row["key"]: row for row in coverage}
        date_rate = float((by_key.get("date-only overlap") or {}).get("coverage_rate") or 0.0)
        code_rate = float((by_key.get("code-only overlap") or {}).get("coverage_rate") or 0.0)
        key_rate = float((by_key.get("prediction_date+code") or {}).get("coverage_rate") or 0.0)
        date_mismatch = date_rate < 0.95
        code_mismatch = code_rate < 0.95
        universe_mismatch = bool(diff.get("universe_mismatch")) or key_rate < 0.95
        if key_rate == 0.0 and date_rate >= 0.95 and code_rate > 0.0:
            primary = "same-date candidate universe mismatch: dataset top10 keys do not cover backtest PM sizing candidates"
        elif date_mismatch:
            primary = "date key mismatch between PM v3 dataset and backtest PM sizing rows"
        elif code_mismatch:
            primary = "code universe or normalization mismatch between PM v3 dataset and backtest PM sizing rows"
        else:
            primary = "lookup key construction mismatch"
        return {
            "current_coverage": backtest_audit.get("pm_v3_feature_coverage"),
            "date_mismatch": date_mismatch,
            "code_mismatch": code_mismatch,
            "universe_mismatch": universe_mismatch,
            "top_missing_reason": backtest_audit.get("top_missing_reason"),
            "dataset_top10_fixed": dataset_audit.get("top10_fixed"),
            "primary_root_cause": primary,
            "phase9f_is_valid_pm_v3_performance_test": False,
        }

    def _recommended_fix(self, cause: dict[str, Any]) -> dict[str, Any]:
        if cause.get("universe_mismatch"):
            action = (
                "Rebuild PM v3 dataset as the prediction-time PM sizing candidate universe used by v2_82/v2_93, "
                "including selected and fallback candidates before cash/portfolio/backtest decisions are used as features. "
                "Keep only J-Quants/API-derived features, walk-forward prediction scores, and prediction-time relative features."
            )
        elif cause.get("date_mismatch"):
            action = "Add explicit prediction_date_key and trade_date_key, then use the key that matches PM sizing call timing."
        elif cause.get("code_mismatch"):
            action = "Centralize normalized_code generation and use the same normalization in dataset builder and PM sizing lookup."
        else:
            action = "Audit PM sizing lookup construction and add coverage gates before any Phase 9-F rerun."
        return {
            "next_phase": "Phase 9-B3 Dataset Builder Fix",
            "direct_phase9f_rerun_allowed": False,
            "recommended_action": action,
        }

    def _leakage_audit(self) -> dict[str, Any]:
        path = self._root(FEATURE_COLUMNS_PATH)
        columns = []
        if path.exists():
            columns = [str(column) for column in json.loads(path.read_text(encoding="utf-8"))]
        forbidden = [column for column in columns if any(token in column.lower() for token in FORBIDDEN_TOKENS)]
        labels = [column for column in columns if column in LABEL_COLUMNS or column.lower().startswith("future_") or "label" in column.lower() or "target" in column.lower()]
        return {
            "feature_columns": columns,
            "forbidden_feature_count": len(forbidden),
            "forbidden_feature_columns": forbidden,
            "label_columns_in_features": labels,
            "leakage_risk": "high" if forbidden or labels else "low",
            "backtest_artifacts_used_for_evaluation_only": True,
            "backtest_artifacts_used_as_features": False,
        }

    def _value_counts(self, series: pd.Series | None) -> dict[str, int]:
        if series is None:
            return {}
        counts = series.fillna("").astype(str).value_counts()
        return {str(key): int(value) for key, value in counts.items() if str(key)}

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(self._format(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, list):
            return ", ".join(str(item) for item in value[:10])
        return str(value).replace("\n", " ")


def build_phase9b2_pm_ai_v3_coverage_root_cause_audit(root: Path | str = ROOT) -> dict[str, Any]:
    return PMAIV3CoverageRootCauseAudit(root).build_report()
