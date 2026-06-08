"""Phase 8-B PM AI candidate integration audit.

Read-only audit that scores v2_82 historical trades with the PM AI API-only
candidate model and compares the candidate decisions with the current PM AI
decisions recorded in the backtest logs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.phase7d_pm_ai_api_only_dataset_builder import is_candidate_list_feature
from ml.portfolio_manager_sizing import multiplier_from_high_minus_avoid


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase8b_pm_candidate_integration_audit_2023-01_to_2026-05"
PROFILE = "rookie_dealer_02_v2_82_cap38"
PERIOD = "2023-01-01_to_2026-05-31"
CURRENT_PM_DIR = Path("models/ml/portfolio_manager/current_v2_73_phase3b_clean")
CANDIDATE_PM_DIR = Path("models/ml/portfolio_manager/candidate_v2_api_only")
CANDIDATE_DATASET = Path("data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05.parquet")
FINAL_CORE_DIR = Path("reports/final/v2_82_cap38/core_2023-01_to_2026-05")
PM_BUCKETS = [1.30, 1.15, 1.00, 0.80, 0.60]


@dataclass(frozen=True)
class Phase8BReportPaths:
    markdown: Path
    json: Path


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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return []


def _to_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _profit_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    for column in ["net_profit", "profit"]:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _profit_factor(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    profits = pd.to_numeric(values, errors="coerce").dropna()
    if profits.empty:
        return None
    gross_profit = float(profits[profits > 0].sum())
    gross_loss = abs(float(profits[profits < 0].sum()))
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def _win_rate(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    profits = pd.to_numeric(values, errors="coerce").dropna()
    if profits.empty:
        return None
    return float((profits > 0).mean())


def _mean(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _code_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


class Phase8BPMCandidateIntegrationAudit:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        current_pm_dir: Path | str = CURRENT_PM_DIR,
        candidate_pm_dir: Path | str = CANDIDATE_PM_DIR,
        candidate_dataset: Path | str = CANDIDATE_DATASET,
        profile: str = PROFILE,
        period: str = PERIOD,
    ) -> None:
        self.root = Path(root)
        self.current_pm_dir = self._root(current_pm_dir)
        self.candidate_pm_dir = self._root(candidate_pm_dir)
        self.candidate_dataset = self._root(candidate_dataset)
        self.profile = profile
        self.period = period

    def build_report(self) -> dict[str, Any]:
        trades = self._load_trades()
        candidate_scored = self._score_candidate_pm(trades)
        compared = self._compare_current_and_candidate(candidate_scored)
        current_features = _read_json_list(self.current_pm_dir / "feature_columns.json")
        candidate_features = _read_json_list(self.candidate_pm_dir / "feature_columns.json")
        feature_diff = self._feature_diff(current_features, candidate_features)
        return {
            "metadata": {
                "phase": "8-B",
                "audit_only": True,
                "backtest_executed": False,
                "profile_added": False,
                "current_model_overwritten": False,
                "live_order_placement": False,
                "profile": self.profile,
                "period": self.period,
            },
            "sources": self._sources(),
            "coverage": self._coverage(trades, compared),
            "score_distribution": self._score_distribution(compared),
            "multiplier_distribution": self._multiplier_distribution(compared),
            "agreement": self._agreement_summary(compared),
            "candidate_changes": self._candidate_changes(compared),
            "profit_approximation": self._profit_approximation(compared),
            "pm130_analysis": self._pm130_analysis(compared),
            "pm080_analysis": self._pm080_analysis(compared),
            "feature_diff": feature_diff,
            "trust_verdict": self._trust_verdict(compared, feature_diff),
            "final_verdict": self._final_verdict(compared, feature_diff),
        }

    def save_report(self, report: dict[str, Any]) -> Phase8BReportPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.format_markdown(report), encoding="utf-8")
        return Phase8BReportPaths(markdown=md_path, json=json_path)

    def format_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# Portfolio Manager AI Phase 8-B Candidate Integration Audit",
            "",
            "## Scope",
            "",
            "- audit only",
            "- no backtest, no new profile, no current model overwrite, no live order",
            "",
            "## Coverage",
            "",
            self._table([report["coverage"]], ["trades", "prediction_available", "prediction_coverage", "missing_feature_rows"]),
            "",
            "## Score Distribution",
            "",
            self._table([report["score_distribution"]], ["current_score_mean", "candidate_score_mean", "current_high_conviction_rate", "candidate_high_conviction_rate", "current_avoid_rate", "candidate_avoid_rate"]),
            "",
            "## Multiplier Distribution",
            "",
            self._table(report["multiplier_distribution"], ["pm_multiplier", "current_trades", "candidate_trades", "current_profit", "candidate_scaled_profit"]),
            "",
            "## Agreement",
            "",
            self._table([report["agreement"]], ["agreement_rate", "disagreement_rate", "exact_agreement_count", "disagreement_count", "current_pm130_candidate_pm130_rate"]),
            "",
            "## Candidate Changes",
            "",
            self._table([report["candidate_changes"]], ["promoted_trades", "demoted_trades", "neutral_trades", "pm130_candidates_added", "pm130_candidates_removed", "pm080_removed_trades"]),
            "",
            "## Profit Approximation",
            "",
            self._table([report["profit_approximation"]], ["current_profit", "candidate_scaled_profit", "estimated_profit_delta", "estimated_pf_delta", "estimated_dd_direction", "estimated_utilization_direction"]),
            "",
            "## PM 1.30 Analysis",
            "",
            self._table([report["pm130_analysis"]], ["current_pm130_trades", "current_pm130_profit", "candidate_pm130_trades", "candidate_pm130_scaled_profit", "pm130_trade_delta", "quality_direction"]),
            "",
            "## PM 0.80 Analysis",
            "",
            self._table([report["pm080_analysis"]], ["current_pm080_trades", "current_pm080_profit", "candidate_pm080_trades", "candidate_pm080_scaled_profit", "pm080_removed_profit", "pm080_removed_trades"]),
            "",
            "## Feature Diff",
            "",
            self._table([report["feature_diff"]], ["current_feature_count", "candidate_feature_count", "candidate_list_dependent_removed_count", "api_only_explainability_improved"]),
            "",
            "## Trust Verdict",
            "",
            self._table([report["trust_verdict"]], ["candidate_pm_safe", "candidate_pm_better_than_current", "candidate_pm_worth_backtesting", "estimated_v283_direction"]),
            "",
            "## Final Verdict",
            "",
            self._table([report["final_verdict"]], ["recommendation", "reason", "next_phase_recommended"]),
            "",
        ]
        return "\n".join(lines)

    def _root(self, path: Path | str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    def _sources(self) -> dict[str, str]:
        return {
            "current_pm": str(self.current_pm_dir),
            "candidate_pm": str(self.candidate_pm_dir),
            "candidate_dataset": str(self.candidate_dataset),
            "trades_csv": str(self._core_dir() / "trades.csv"),
            "purchase_audit_csv": str(self._core_dir() / "purchase_audit.csv"),
        }

    def _core_dir(self) -> Path:
        final = self.root / FINAL_CORE_DIR
        if final.exists():
            return final
        return self.root / "logs" / "backtests" / self.profile / self.period

    def _load_trades(self) -> pd.DataFrame:
        trades = _read_csv(self._core_dir() / "trades.csv")
        if trades.empty:
            return trades
        trades = trades.copy()
        if "action" in trades.columns:
            trades = trades[trades["action"].fillna("").astype(str).eq("SELL")].copy()
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in trades.columns:
                trades[column] = pd.to_datetime(trades[column], errors="coerce").dt.strftime("%Y-%m-%d")
        if "code" in trades.columns:
            trades["code"] = trades["code"].map(_code_text)
        return trades.reset_index(drop=True)

    def _score_candidate_pm(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        feature_columns = _read_json_list(self.candidate_pm_dir / "feature_columns.json")
        if not feature_columns:
            out = trades.copy()
            out["candidate_prediction_available"] = False
            return out
        dataset = self._load_candidate_feature_rows(feature_columns, trades)
        joined = trades.merge(
            dataset,
            left_on=["signal_date", "code"],
            right_on=["as_of_date", "code"],
            how="left",
            suffixes=("", "_candidate_feature"),
        )
        available = joined["as_of_date"].notna() if "as_of_date" in joined.columns else pd.Series(False, index=joined.index)
        joined["candidate_prediction_available"] = available
        joined["candidate_feature_missing_count"] = joined[feature_columns].isna().sum(axis=1) if feature_columns else 0
        if not available.any():
            return joined
        high_model = self._load_joblib("high_conviction_target_classifier.joblib")
        avoid_model = self._load_joblib("avoid_target_classifier.joblib")
        x = self._transform_candidate_features(joined.loc[available, feature_columns], feature_columns, high_model)
        high_scores = self._predict_positive(high_model, x)
        avoid_scores = self._predict_positive(avoid_model, x)
        joined["candidate_high_conviction_proba"] = pd.NA
        joined["candidate_avoid_proba"] = pd.NA
        joined.loc[available, "candidate_high_conviction_proba"] = high_scores
        joined.loc[available, "candidate_avoid_proba"] = avoid_scores
        joined["candidate_pm_score"] = pd.to_numeric(joined["candidate_high_conviction_proba"], errors="coerce") - pd.to_numeric(joined["candidate_avoid_proba"], errors="coerce")
        joined["candidate_pm_multiplier"] = [
            multiplier_from_high_minus_avoid(high, avoid) if ok else None
            for ok, high, avoid in zip(available, joined["candidate_high_conviction_proba"], joined["candidate_avoid_proba"])
        ]
        return joined

    def _load_candidate_feature_rows(self, feature_columns: list[str], trades: pd.DataFrame) -> pd.DataFrame:
        columns = ["as_of_date", "code", *feature_columns]
        if not self.candidate_dataset.exists():
            return pd.DataFrame(columns=columns)
        dataset = pd.read_parquet(self.candidate_dataset, columns=columns)
        dataset["as_of_date"] = pd.to_datetime(dataset["as_of_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        dataset["code"] = dataset["code"].map(_code_text)
        keys = trades[["signal_date", "code"]].dropna().drop_duplicates()
        keys = keys.rename(columns={"signal_date": "as_of_date"})
        return dataset.merge(keys, on=["as_of_date", "code"], how="inner").drop_duplicates(["as_of_date", "code"], keep="last")

    def _load_joblib(self, filename: str) -> Any:
        import joblib

        return joblib.load(self.candidate_pm_dir / filename)

    def _transform_candidate_features(self, frame: pd.DataFrame, feature_columns: list[str], model: Any | None = None) -> pd.DataFrame:
        preprocess = _read_json(self.candidate_pm_dir / "preprocess.json")
        medians = preprocess.get("medians", {}) if isinstance(preprocess.get("medians"), dict) else {}
        missing_indicators = preprocess.get("missing_indicator_columns", [])
        missing_indicators = [str(column) for column in missing_indicators] if isinstance(missing_indicators, list) else []
        out = frame.copy()
        for column in feature_columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        for column in missing_indicators:
            if column in out.columns:
                out[f"{column}_missing"] = out[column].isna().astype(int)
        for column in feature_columns:
            if column in medians:
                out[column] = out[column].fillna(float(medians[column]))
        ordered_columns = list(feature_columns) + [f"{column}_missing" for column in missing_indicators if f"{column}_missing" in out.columns]
        model_columns = [str(column) for column in getattr(model, "feature_names_in_", [])]
        if model_columns:
            for column in model_columns:
                if column not in out.columns:
                    out[column] = 0
            ordered_columns = model_columns
        return out[ordered_columns]

    def _predict_positive(self, model: Any, frame: pd.DataFrame) -> list[float]:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(frame)
            return [float(value) for value in proba[:, 1]]
        values = model.predict(frame)
        return [float(value) for value in values]

    def _compare_current_and_candidate(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        out["current_pm_multiplier"] = pd.to_numeric(out.get("pm_multiplier"), errors="coerce")
        out["current_pm_score"] = pd.to_numeric(out.get("pm_score"), errors="coerce")
        out["current_high_conviction_proba"] = pd.to_numeric(out.get("pm_high_conviction_proba"), errors="coerce")
        out["current_avoid_proba"] = pd.to_numeric(out.get("pm_avoid_proba"), errors="coerce")
        out["candidate_pm_multiplier_num"] = pd.to_numeric(out.get("candidate_pm_multiplier"), errors="coerce")
        out["pm_multiplier_delta"] = out["candidate_pm_multiplier_num"] - out["current_pm_multiplier"]
        out["pm_change_type"] = "missing"
        available = out.get("candidate_prediction_available", pd.Series(False, index=out.index)).astype(bool)
        out.loc[available & out["pm_multiplier_delta"].gt(0), "pm_change_type"] = "promoted"
        out.loc[available & out["pm_multiplier_delta"].lt(0), "pm_change_type"] = "demoted"
        out.loc[available & out["pm_multiplier_delta"].eq(0), "pm_change_type"] = "neutral"
        out["profit"] = _profit_series(out).reindex(out.index, fill_value=0.0)
        ratio = out["candidate_pm_multiplier_num"] / out["current_pm_multiplier"].replace(0, pd.NA)
        out["candidate_scaled_profit"] = out["profit"] * ratio.fillna(1.0)
        return out

    def _coverage(self, trades: pd.DataFrame, compared: pd.DataFrame) -> dict[str, Any]:
        if compared.empty:
            return {"trades": int(len(trades)), "prediction_available": 0, "prediction_coverage": 0.0, "missing_feature_rows": int(len(trades))}
        available = compared.get("candidate_prediction_available", pd.Series(False, index=compared.index)).astype(bool)
        return {
            "trades": int(len(compared)),
            "prediction_available": int(available.sum()),
            "prediction_coverage": float(available.mean()) if len(available) else 0.0,
            "missing_feature_rows": int((~available).sum()),
        }

    def _score_distribution(self, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {}
        return {
            "current_score_mean": _mean(frame.get("current_pm_score")),
            "candidate_score_mean": _mean(frame.get("candidate_pm_score")),
            "current_high_conviction_rate": float(pd.to_numeric(frame.get("current_high_conviction_proba"), errors="coerce").ge(0.5).mean()),
            "candidate_high_conviction_rate": float(pd.to_numeric(frame.get("candidate_high_conviction_proba"), errors="coerce").ge(0.5).mean()),
            "current_avoid_rate": float(pd.to_numeric(frame.get("current_avoid_proba"), errors="coerce").ge(0.5).mean()),
            "candidate_avoid_rate": float(pd.to_numeric(frame.get("candidate_avoid_proba"), errors="coerce").ge(0.5).mean()),
        }

    def _multiplier_distribution(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for bucket in PM_BUCKETS:
            current = frame[pd.to_numeric(frame.get("current_pm_multiplier"), errors="coerce").round(2).eq(bucket)] if not frame.empty else pd.DataFrame()
            candidate = frame[pd.to_numeric(frame.get("candidate_pm_multiplier_num"), errors="coerce").round(2).eq(bucket)] if not frame.empty else pd.DataFrame()
            rows.append(
                {
                    "pm_multiplier": bucket,
                    "current_trades": int(len(current)),
                    "candidate_trades": int(len(candidate)),
                    "current_profit": float(_profit_series(current).sum()) if not current.empty else 0.0,
                    "candidate_scaled_profit": float(pd.to_numeric(candidate.get("candidate_scaled_profit"), errors="coerce").sum()) if not candidate.empty else 0.0,
                }
            )
        return rows

    def _agreement_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"agreement_rate": None, "disagreement_rate": None, "exact_agreement_count": 0, "disagreement_count": 0, "current_pm130_candidate_pm130_rate": None}
        available = frame[frame.get("candidate_prediction_available", pd.Series(False, index=frame.index)).astype(bool)]
        if available.empty:
            return {"agreement_rate": 0.0, "disagreement_rate": 0.0, "exact_agreement_count": 0, "disagreement_count": 0, "current_pm130_candidate_pm130_rate": None}
        same = available["current_pm_multiplier"].round(2).eq(available["candidate_pm_multiplier_num"].round(2))
        current_pm130 = available[available["current_pm_multiplier"].round(2).eq(1.30)]
        pm130_keep = float(current_pm130["candidate_pm_multiplier_num"].round(2).eq(1.30).mean()) if not current_pm130.empty else None
        return {
            "agreement_rate": float(same.mean()),
            "disagreement_rate": float((~same).mean()),
            "exact_agreement_count": int(same.sum()),
            "disagreement_count": int((~same).sum()),
            "current_pm130_candidate_pm130_rate": pm130_keep,
        }

    def _candidate_changes(self, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {}
        promoted = frame[frame["pm_change_type"].eq("promoted")]
        demoted = frame[frame["pm_change_type"].eq("demoted")]
        neutral = frame[frame["pm_change_type"].eq("neutral")]
        return {
            "promoted_trades": int(len(promoted)),
            "demoted_trades": int(len(demoted)),
            "neutral_trades": int(len(neutral)),
            "pm130_candidates_added": int((frame["candidate_pm_multiplier_num"].round(2).eq(1.30) & ~frame["current_pm_multiplier"].round(2).eq(1.30)).sum()),
            "pm130_candidates_removed": int((frame["current_pm_multiplier"].round(2).eq(1.30) & ~frame["candidate_pm_multiplier_num"].round(2).eq(1.30)).sum()),
            "pm080_removed_trades": int((frame["current_pm_multiplier"].round(2).eq(0.80) & ~frame["candidate_pm_multiplier_num"].round(2).eq(0.80)).sum()),
        }

    def _profit_approximation(self, frame: pd.DataFrame) -> dict[str, Any]:
        current = _profit_series(frame)
        candidate = pd.to_numeric(frame.get("candidate_scaled_profit"), errors="coerce").dropna() if not frame.empty else pd.Series(dtype=float)
        current_profit = float(current.sum()) if not current.empty else 0.0
        candidate_profit = float(candidate.sum()) if not candidate.empty else 0.0
        pf_delta = None
        current_pf = _profit_factor(current)
        candidate_pf = _profit_factor(candidate)
        if current_pf is not None and candidate_pf is not None:
            pf_delta = candidate_pf - current_pf
        avg_delta = _mean(frame.get("pm_multiplier_delta")) if not frame.empty else None
        return {
            "current_profit": current_profit,
            "candidate_scaled_profit": candidate_profit,
            "estimated_profit_delta": candidate_profit - current_profit,
            "current_pf": current_pf,
            "candidate_pf": candidate_pf,
            "estimated_pf_delta": pf_delta,
            "estimated_dd_direction": "worse_or_higher_concentration" if avg_delta and avg_delta > 0.05 else "flat_or_lower",
            "estimated_utilization_direction": "up" if avg_delta and avg_delta > 0 else "down_or_flat",
        }

    def _pm130_analysis(self, frame: pd.DataFrame) -> dict[str, Any]:
        current = frame[frame.get("current_pm_multiplier", pd.Series(dtype=float)).round(2).eq(1.30)] if not frame.empty else pd.DataFrame()
        candidate = frame[frame.get("candidate_pm_multiplier_num", pd.Series(dtype=float)).round(2).eq(1.30)] if not frame.empty else pd.DataFrame()
        current_profit = float(_profit_series(current).sum()) if not current.empty else 0.0
        candidate_profit = float(pd.to_numeric(candidate.get("candidate_scaled_profit"), errors="coerce").sum()) if not candidate.empty else 0.0
        quality_direction = "improves" if candidate_profit / max(len(candidate), 1) > current_profit / max(len(current), 1) else "unclear_or_worse"
        return {
            "current_pm130_trades": int(len(current)),
            "current_pm130_profit": current_profit,
            "candidate_pm130_trades": int(len(candidate)),
            "candidate_pm130_scaled_profit": candidate_profit,
            "pm130_trade_delta": int(len(candidate) - len(current)),
            "quality_direction": quality_direction,
        }

    def _pm080_analysis(self, frame: pd.DataFrame) -> dict[str, Any]:
        current = frame[frame.get("current_pm_multiplier", pd.Series(dtype=float)).round(2).eq(0.80)] if not frame.empty else pd.DataFrame()
        candidate = frame[frame.get("candidate_pm_multiplier_num", pd.Series(dtype=float)).round(2).eq(0.80)] if not frame.empty else pd.DataFrame()
        removed = frame[frame.get("current_pm_multiplier", pd.Series(dtype=float)).round(2).eq(0.80) & ~frame.get("candidate_pm_multiplier_num", pd.Series(dtype=float)).round(2).eq(0.80)] if not frame.empty else pd.DataFrame()
        return {
            "current_pm080_trades": int(len(current)),
            "current_pm080_profit": float(_profit_series(current).sum()) if not current.empty else 0.0,
            "candidate_pm080_trades": int(len(candidate)),
            "candidate_pm080_scaled_profit": float(pd.to_numeric(candidate.get("candidate_scaled_profit"), errors="coerce").sum()) if not candidate.empty else 0.0,
            "pm080_removed_profit": float(_profit_series(removed).sum()) if not removed.empty else 0.0,
            "pm080_removed_trades": int(len(removed)),
        }

    def _feature_diff(self, current_features: list[str], candidate_features: list[str]) -> dict[str, Any]:
        current_candidate_dependent = sorted(feature for feature in current_features if is_candidate_list_feature(feature))
        candidate_candidate_dependent = sorted(feature for feature in candidate_features if is_candidate_list_feature(feature))
        return {
            "current_feature_count": len(current_features),
            "candidate_feature_count": len(candidate_features),
            "current_only_features": sorted(set(current_features) - set(candidate_features)),
            "candidate_only_features": sorted(set(candidate_features) - set(current_features)),
            "current_candidate_list_dependent_features": current_candidate_dependent,
            "candidate_candidate_list_dependent_features": candidate_candidate_dependent,
            "candidate_list_dependent_removed_count": len(current_candidate_dependent) - len(candidate_candidate_dependent),
            "api_only_explainability_improved": not candidate_candidate_dependent and len(candidate_features) < len(current_features),
        }

    def _trust_verdict(self, frame: pd.DataFrame, feature_diff: dict[str, Any]) -> dict[str, Any]:
        coverage = self._coverage(frame, frame).get("prediction_coverage") or 0.0
        profit_delta = self._profit_approximation(frame).get("estimated_profit_delta") or 0.0
        pm080 = self._pm080_analysis(frame)
        safe = coverage >= 0.90 and not feature_diff.get("candidate_candidate_list_dependent_features")
        better = profit_delta > 0 and float(pm080.get("pm080_removed_profit") or 0.0) >= 0
        worth = safe and (better or abs(profit_delta) < 250_000)
        return {
            "candidate_pm_safe": safe,
            "candidate_pm_better_than_current": better,
            "candidate_pm_worth_backtesting": worth,
            "estimated_v283_direction": "positive" if better else "uncertain_or_negative",
        }

    def _final_verdict(self, frame: pd.DataFrame, feature_diff: dict[str, Any]) -> dict[str, Any]:
        trust = self._trust_verdict(frame, feature_diff)
        if trust["candidate_pm_better_than_current"] and trust["candidate_pm_worth_backtesting"]:
            recommendation = "Phase 8-C PM Candidate Backtest"
            reason = "Candidate is API-only, safe, and positive in lightweight profit approximation."
        elif trust["candidate_pm_safe"] and trust["candidate_pm_worth_backtesting"]:
            recommendation = "Phase 8-C PM Candidate Backtest"
            reason = "Candidate is safe and clean enough to test even though estimated edge is uncertain."
        elif trust["candidate_pm_safe"]:
            recommendation = "Stay current PM"
            reason = "Candidate is clean but does not clearly beat current PM on v2_82 trade approximation."
        else:
            recommendation = "Rebuild PM candidate"
            reason = "Candidate coverage or schema safety is not sufficient for integration."
        return {
            "recommendation": recommendation,
            "reason": reason,
            "next_phase_recommended": recommendation,
        }

    def _table(self, rows: list[dict[str, Any]] | dict[str, Any], columns: list[str]) -> str:
        if isinstance(rows, dict):
            rows = [rows]
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = []
        for row in rows:
            body.append("| " + " | ".join(self._format_cell(row.get(column, "")) for column in columns) + " |")
        return "\n".join([header, sep, *body])

    def _format_cell(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, (list, dict)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        return text.replace("\n", " ").replace("|", "\\|")


def build_phase8b_report(root: Path | str = ROOT) -> dict[str, Any]:
    return Phase8BPMCandidateIntegrationAudit(root).build_report()
