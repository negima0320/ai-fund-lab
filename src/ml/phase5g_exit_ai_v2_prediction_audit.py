"""Phase 5-G Exit AI v2 prediction / integration audit.

The audit is read-only. It scores existing v2_78 sell-log rows with the
candidate Exit AI v2 model and compares the scores against existing Exit AI
decisions without running a backtest or modifying any current model path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PERIOD = "2023-01-01_to_2026-05-31"
BASE_PROFILE = "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
REPORT_STEM = "phase5g_exit_ai_v2_prediction_audit_2023-01_to_2026-05"
MODEL_DIR = ROOT / "models" / "ml" / "exit_ai_v2" / "candidate_v2_api_only"
CURRENT_EXIT_MODEL_DIR = ROOT / "models" / "ml" / "exit" / "current_v2_66"
DATASET_PATH = ROOT / "data" / "ml" / "exit_ai_v2" / "exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet"


@dataclass(frozen=True)
class Phase5GPaths:
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _date_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.to_datetime(value, errors="coerce").strftime("%Y-%m-%d")


def _code_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _reason_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return None if clean.empty else float(clean.mean())


class Phase5GExitAIV2PredictionAudit:
    def __init__(
        self,
        root: Path | str = ROOT,
        *,
        profile: str = BASE_PROFILE,
        period: str = PERIOD,
        dataset_path: Path | None = None,
        model_dir: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.profile = profile
        self.period = period
        self.dataset_path = self._root_path(dataset_path or DATASET_PATH)
        self.model_dir = self._root_path(model_dir or MODEL_DIR)
        self.current_exit_model_dir = self._root_path(CURRENT_EXIT_MODEL_DIR)

    def build_report(self) -> dict[str, Any]:
        trades = self._load_trades()
        dataset = self._load_dataset()
        model_bundle = self._load_model_bundle()
        targets = self._build_targets(trades)
        scored = self._score_targets(targets, dataset, model_bundle)
        comparison = self._existing_exit_ai_comparison(scored)
        early = self._post_exit_return_audit(scored)
        high_pm = self._high_pm_audit(scored)
        rules = self._virtual_rule_audit(scored)
        leakage = self._leakage_audit(model_bundle, scored)
        return {
            "metadata": {
                "phase": "5-G",
                "audit_only": True,
                "full_backtest_executed": False,
                "profile_added": False,
                "full_pytest_executed": False,
                "current_model_overwritten": False,
                "base_profile": self.profile,
                "period": self.period,
            },
            "input_paths": {
                "trades": str(self._log_dir() / "trades.csv"),
                "dataset": str(self.dataset_path),
                "exit_ai_v2_model_dir": str(self.model_dir),
                "current_exit_model_dir_readonly": str(self.current_exit_model_dir),
            },
            "target_summary": self._target_summary(targets, scored),
            "prediction_summary": self._prediction_summary(scored),
            "existing_exit_ai_comparison": comparison,
            "post_exit_return_audit": early,
            "high_pm_early_exit_audit": high_pm,
            "virtual_rule_audit": rules,
            "leakage_integrity_audit": leakage,
            "recommended_next_phase": self._recommended_next_phase(scored, leakage, high_pm, rules),
            "scored_rows_sample": scored.head(50).to_dict("records") if not scored.empty else [],
        }

    def save_report(self, result: dict[str, Any]) -> Phase5GPaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase5GPaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# AI Retraining Phase 5-G Exit AI v2 Prediction / Integration Audit",
            "",
            "## Scope",
            "",
            "- audit only",
            "- no backtest, no profile creation, no current model overwrite",
            "- Exit AI v2 candidate model is read-only",
            "",
            "## Target Summary",
            "",
            self._table([result["target_summary"]], ["sell_rows", "exit_ai_sell_rows", "early_sell_suspect_rows", "loss_avoidance_rows", "high_pm_rows"]),
            "",
            "## Prediction Summary",
            "",
            self._table([result["prediction_summary"]], ["prediction_available_count", "prediction_missing_count", "coverage_rate", "top_decile_count", "average_score"]),
            "",
            "## Existing Exit AI Comparison",
            "",
            self._table([result["existing_exit_ai_comparison"]], ["agreement_count", "disagreement_count", "existing_exit_ai_only_count", "exit_ai_v2_only_count"]),
            "",
            "## Post Exit Return Audit",
            "",
            self._table(result["post_exit_return_audit"]["by_score_decile"], ["score_decile", "rows", "post_exit_return_5d_mean", "post_exit_return_10d_mean", "post_exit_return_20d_mean"]),
            "",
            "## High PM Early Exit Audit",
            "",
            self._table([result["high_pm_early_exit_audit"]], ["high_pm_exit_count", "high_pm_existing_exit_ai_count", "high_pm_v2_top_decile_count", "high_pm_v2_non_top_decile_post_exit_return_5d", "high_pm_v2_top_decile_post_exit_return_5d"]),
            "",
            "## Virtual Rules",
            "",
            self._table(result["virtual_rule_audit"]["rules"], ["rule", "candidate_count", "actual_profit", "post_exit_return_5d", "post_exit_return_10d", "post_exit_return_20d", "estimated_benefit_direction", "recommended"]),
            "",
            "## Leakage / Integrity",
            "",
            self._table([result["leakage_integrity_audit"]], ["prediction_uses_api_only_dataset_rows", "forbidden_feature_columns_found", "label_like_feature_columns_found", "selected_count_in_day_in_features", "model_loaded_from_candidate_path", "current_model_not_overwritten", "feature_schema_matches_training_metadata", "leakage_risk", "blocking_issues"]),
            "",
            "## Recommended Next Phase",
            "",
            f"`{result['recommended_next_phase']}`",
            "",
        ]
        return "\n".join(lines)

    def _log_dir(self) -> Path:
        return self.root / "logs" / "backtests" / self.profile / self.period

    def _load_trades(self) -> pd.DataFrame:
        return _read_csv(self._log_dir() / "trades.csv")

    def _load_dataset(self) -> pd.DataFrame:
        dataset = _read_parquet(self.dataset_path)
        if dataset.empty:
            return dataset
        dataset = dataset.copy()
        dataset["code"] = dataset["code"].map(_code_text)
        dataset["as_of_date"] = pd.to_datetime(dataset["as_of_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return dataset

    def _load_model_bundle(self) -> dict[str, Any]:
        import joblib

        metadata = _read_json(self.model_dir / "model_metadata.json")
        preprocess = _read_json(self.model_dir / "preprocess.json")
        model_path = self.model_dir / "exit_quality_top_decile_classifier.joblib"
        model = joblib.load(model_path) if model_path.exists() else None
        feature_columns = metadata.get("feature_columns") or _read_json(self.model_dir / "feature_columns.json")
        threshold = _to_float(metadata.get("train_top_decile_threshold"))
        return {
            "model": model,
            "model_path": model_path,
            "metadata": metadata,
            "preprocess": preprocess,
            "feature_columns": feature_columns if isinstance(feature_columns, list) else [],
            "threshold": threshold,
        }

    def _build_targets(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()
        rows = trades.copy()
        rows["code"] = rows["code"].map(_code_text)
        rows["buy_date"] = rows.get("entry_date", "").map(_date_text)
        rows["sell_date"] = rows.get("exit_date", "").map(_date_text)
        rows["as_of_date"] = rows["sell_date"]
        reason = rows.get("exit_reason", pd.Series("", index=rows.index)).map(_reason_text)
        rows["existing_exit_ai_triggered"] = rows.get("exit_ai_triggered", pd.Series(False, index=rows.index)).map(_truthy)
        rows["existing_exit_reason"] = rows.get("exit_reason", "")
        rows["is_exit_ai_sell"] = rows["existing_exit_ai_triggered"] | reason.str.contains("exit ai|avoid_loss", regex=True)
        rows["realized_profit"] = pd.to_numeric(rows.get("net_profit", rows.get("profit")), errors="coerce")
        rows["realized_profit_rate"] = pd.to_numeric(rows.get("net_profit_rate", rows.get("profit_rate")), errors="coerce")
        rows["is_loss_avoidance_case"] = reason.str.contains("損切|stop_loss|stop", regex=True) | rows["realized_profit"].lt(0)
        rows["pm_multiplier"] = pd.to_numeric(rows.get("pm_multiplier"), errors="coerce")
        rows["pm_score"] = pd.to_numeric(rows.get("pm_score"), errors="coerce")
        rows["is_high_pm"] = rows["pm_multiplier"].ge(1.15)
        keep = [
            "trade_id",
            "code",
            "buy_date",
            "sell_date",
            "as_of_date",
            "existing_exit_ai_triggered",
            "existing_exit_reason",
            "is_exit_ai_sell",
            "realized_profit",
            "realized_profit_rate",
            "is_loss_avoidance_case",
            "pm_score",
            "pm_multiplier",
            "is_high_pm",
        ]
        return rows[[column for column in keep if column in rows.columns]].copy()

    def _score_targets(self, targets: pd.DataFrame, dataset: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
        if targets.empty:
            return pd.DataFrame()
        merged = targets.merge(dataset, how="left", on=["code", "as_of_date"], suffixes=("", "_dataset"))
        feature_columns = bundle["feature_columns"]
        threshold = bundle["threshold"]
        merged["prediction_available"] = False
        merged["exit_ai_v2_score"] = pd.NA
        merged["exit_ai_v2_top_decile_flag"] = False
        merged["model_feature_missing_count"] = pd.NA
        if bundle["model"] is None or not feature_columns or merged.empty:
            return merged
        available = merged[feature_columns].notna().any(axis=1)
        if available.any():
            transformed = self._transform(merged.loc[available], bundle["preprocess"])
            scores = bundle["model"].predict_proba(transformed)[:, 1]
            merged.loc[available, "prediction_available"] = True
            merged.loc[available, "exit_ai_v2_score"] = scores
            merged.loc[available, "model_feature_missing_count"] = merged.loc[available, feature_columns].isna().sum(axis=1).astype(int)
        scored = pd.to_numeric(merged["exit_ai_v2_score"], errors="coerce")
        merged["exit_ai_v2_rank_percentile"] = scored.rank(pct=True, method="first")
        merged.loc[scored.notna(), "exit_ai_v2_top_decile_flag"] = merged.loc[scored.notna(), "exit_ai_v2_rank_percentile"].ge(0.90)
        merged["score_decile"] = pd.NA
        if scored.notna().sum() >= 10:
            merged.loc[scored.notna(), "score_decile"] = pd.qcut(scored[scored.notna()].rank(method="first"), 10, labels=False, duplicates="drop")
        merged["post_exit_return_5d"] = pd.to_numeric(merged.get("future_return_5d"), errors="coerce")
        merged["post_exit_return_10d"] = pd.to_numeric(merged.get("future_return_10d"), errors="coerce")
        merged["post_exit_return_20d"] = pd.to_numeric(merged.get("future_return_20d"), errors="coerce")
        merged["is_early_sell_suspect"] = merged[["post_exit_return_5d", "post_exit_return_10d", "post_exit_return_20d"]].gt(0).any(axis=1)
        return merged

    def _transform(self, frame: pd.DataFrame, preprocess: dict[str, Any]) -> pd.DataFrame:
        columns = preprocess.get("feature_columns", [])
        result = frame[columns].copy()
        transformed: dict[str, Any] = {}
        for column in preprocess.get("numeric_columns", []):
            transformed[column] = pd.to_numeric(result[column], errors="coerce").fillna(preprocess.get("medians", {}).get(column, 0.0))
        for column in preprocess.get("categorical_columns", []):
            filled = result[column].fillna(preprocess.get("modes", {}).get(column, "")).astype("category")
            transformed[column] = filled.cat.codes.astype(float)
        output = pd.DataFrame(transformed, index=frame.index)
        for column in preprocess.get("missing_indicator_columns", []):
            output[f"{column}_missing"] = result[column].isna().astype(int)
        return output

    def _target_summary(self, targets: pd.DataFrame, scored: pd.DataFrame) -> dict[str, Any]:
        return {
            "sell_rows": int(len(targets)),
            "exit_ai_sell_rows": int(targets.get("is_exit_ai_sell", pd.Series(dtype=bool)).sum()) if not targets.empty else 0,
            "early_sell_suspect_rows": int(scored.get("is_early_sell_suspect", pd.Series(dtype=bool)).sum()) if not scored.empty else 0,
            "loss_avoidance_rows": int(targets.get("is_loss_avoidance_case", pd.Series(dtype=bool)).sum()) if not targets.empty else 0,
            "high_pm_rows": int(targets.get("is_high_pm", pd.Series(dtype=bool)).sum()) if not targets.empty else 0,
        }

    def _prediction_summary(self, scored: pd.DataFrame) -> dict[str, Any]:
        if scored.empty:
            return {"prediction_available_count": 0, "prediction_missing_count": 0, "coverage_rate": 0.0, "top_decile_count": 0, "average_score": None}
        available = scored["prediction_available"].fillna(False)
        return {
            "prediction_available_count": int(available.sum()),
            "prediction_missing_count": int((~available).sum()),
            "coverage_rate": float(available.mean()),
            "top_decile_count": int(scored["exit_ai_v2_top_decile_flag"].fillna(False).sum()),
            "average_score": _mean(scored["exit_ai_v2_score"]),
        }

    def _existing_exit_ai_comparison(self, scored: pd.DataFrame) -> dict[str, Any]:
        if scored.empty:
            return {"agreement_count": 0, "disagreement_count": 0, "existing_exit_ai_only_count": 0, "exit_ai_v2_only_count": 0}
        existing = scored["is_exit_ai_sell"].fillna(False)
        v2 = scored["exit_ai_v2_top_decile_flag"].fillna(False)
        agree = existing.eq(v2)
        return {
            "agreement_count": int(agree.sum()),
            "disagreement_count": int((~agree).sum()),
            "existing_exit_ai_only_count": int((existing & ~v2).sum()),
            "exit_ai_v2_only_count": int((~existing & v2).sum()),
        }

    def _post_exit_return_audit(self, scored: pd.DataFrame) -> dict[str, Any]:
        rows = []
        if not scored.empty and "score_decile" in scored.columns:
            for decile, group in scored.dropna(subset=["score_decile"]).groupby("score_decile"):
                rows.append(
                    {
                        "score_decile": int(decile),
                        "rows": int(len(group)),
                        "post_exit_return_5d_mean": _mean(group["post_exit_return_5d"]),
                        "post_exit_return_10d_mean": _mean(group["post_exit_return_10d"]),
                        "post_exit_return_20d_mean": _mean(group["post_exit_return_20d"]),
                    }
                )
        top = scored[scored.get("exit_ai_v2_top_decile_flag", pd.Series(False, index=scored.index)).fillna(False)] if not scored.empty else pd.DataFrame()
        non_top = scored[~scored.get("exit_ai_v2_top_decile_flag", pd.Series(False, index=scored.index)).fillna(False)] if not scored.empty else pd.DataFrame()
        return {
            "by_score_decile": rows,
            "top_decile_vs_non_top": {
                "top_decile_post_exit_return_5d": _mean(top.get("post_exit_return_5d", pd.Series(dtype=float))),
                "non_top_post_exit_return_5d": _mean(non_top.get("post_exit_return_5d", pd.Series(dtype=float))),
                "top_decile_rows": int(len(top)),
                "non_top_rows": int(len(non_top)),
            },
        }

    def _high_pm_audit(self, scored: pd.DataFrame) -> dict[str, Any]:
        high = scored[scored.get("is_high_pm", pd.Series(False, index=scored.index)).fillna(False)] if not scored.empty else pd.DataFrame()
        top = high[high.get("exit_ai_v2_top_decile_flag", pd.Series(False, index=high.index)).fillna(False)] if not high.empty else pd.DataFrame()
        non_top = high[~high.get("exit_ai_v2_top_decile_flag", pd.Series(False, index=high.index)).fillna(False)] if not high.empty else pd.DataFrame()
        return {
            "high_pm_exit_count": int(len(high)),
            "high_pm_existing_exit_ai_count": int(high.get("is_exit_ai_sell", pd.Series(dtype=bool)).sum()) if not high.empty else 0,
            "high_pm_v2_top_decile_count": int(high.get("exit_ai_v2_top_decile_flag", pd.Series(dtype=bool)).sum()) if not high.empty else 0,
            "high_pm_v2_non_top_decile_post_exit_return_5d": _mean(non_top.get("post_exit_return_5d", pd.Series(dtype=float))),
            "high_pm_v2_top_decile_post_exit_return_5d": _mean(top.get("post_exit_return_5d", pd.Series(dtype=float))),
            "interpretation": "high PM non-top rows are suppression candidates; high PM top-decile rows are exit-continuation candidates",
        }

    def _virtual_rule_audit(self, scored: pd.DataFrame) -> dict[str, Any]:
        rules = [
            ("Rule A", scored["is_exit_ai_sell"].fillna(False) & ~scored["exit_ai_v2_top_decile_flag"].fillna(False) if not scored.empty else pd.Series(dtype=bool), "Suppress existing Exit AI sell when v2 is not top decile"),
            ("Rule B", scored["is_high_pm"].fillna(False) & scored["is_exit_ai_sell"].fillna(False) & ~scored["exit_ai_v2_top_decile_flag"].fillna(False) if not scored.empty else pd.Series(dtype=bool), "Suppress high-PM existing Exit AI sell when v2 is not top decile"),
            ("Rule C", scored["exit_ai_v2_top_decile_flag"].fillna(False) if not scored.empty else pd.Series(dtype=bool), "Exit-strengthening candidate when v2 is top decile"),
        ]
        rows = []
        for name, mask, reason in rules:
            group = scored[mask] if not scored.empty else pd.DataFrame()
            rows.append(
                {
                    "rule": name,
                    "candidate_count": int(len(group)),
                    "actual_profit": _mean(group.get("realized_profit", pd.Series(dtype=float))),
                    "post_exit_return_5d": _mean(group.get("post_exit_return_5d", pd.Series(dtype=float))),
                    "post_exit_return_10d": _mean(group.get("post_exit_return_10d", pd.Series(dtype=float))),
                    "post_exit_return_20d": _mean(group.get("post_exit_return_20d", pd.Series(dtype=float))),
                    "estimated_benefit_direction": self._benefit_direction(name, group),
                    "recommended": name == "Rule B",
                    "reason": reason,
                }
            )
        return {"rules": rows, "recommended_rule": "Rule B if high-PM non-top rows show positive post-exit returns; otherwise defer integration"}

    def _benefit_direction(self, rule: str, group: pd.DataFrame) -> str:
        mean_5d = _mean(group.get("post_exit_return_5d", pd.Series(dtype=float)))
        if mean_5d is None:
            return "unknown"
        if rule in {"Rule A", "Rule B"}:
            return "positive_if_post_exit_return_positive" if mean_5d > 0 else "negative_or_unclear"
        return "positive_if_post_exit_return_negative" if mean_5d < 0 else "negative_or_unclear"

    def _leakage_audit(self, bundle: dict[str, Any], scored: pd.DataFrame) -> dict[str, Any]:
        features = bundle.get("feature_columns", [])
        label_like = [column for column in features if "future_return" in column or "future_max" in column or "exit_quality" in column or "target" in column or "label" in column]
        forbidden = [column for column in features if column in {"trade_id", "realized_profit", "realized_return", "selected_count_in_day", "exit_reason", "holding_days"}]
        schema_matches = features == bundle.get("preprocess", {}).get("feature_columns", [])
        blocking = []
        if label_like:
            blocking.append("Label-like model features found.")
        if forbidden:
            blocking.append("Forbidden model features found.")
        if not schema_matches:
            blocking.append("Feature schema does not match preprocess metadata.")
        return {
            "prediction_uses_api_only_dataset_rows": True,
            "no_trades_fields_used_as_model_features": not bool(forbidden),
            "no_realized_profit_used_as_model_feature": "realized_profit" not in features,
            "no_future_return_labels_used_as_model_features": not bool(label_like),
            "selected_count_in_day_in_features": "selected_count_in_day" in features,
            "model_loaded_from_candidate_path": str(bundle.get("model_path", "")).startswith(str(self.model_dir)),
            "current_model_not_overwritten": True,
            "feature_schema_matches_training_metadata": schema_matches,
            "forbidden_feature_columns_found": forbidden,
            "label_like_feature_columns_found": label_like,
            "prediction_available_rows": int(scored.get("prediction_available", pd.Series(dtype=bool)).sum()) if not scored.empty else 0,
            "leakage_risk": "high" if blocking else "low",
            "blocking_issues": blocking,
        }

    def _recommended_next_phase(self, scored: pd.DataFrame, leakage: dict[str, Any], high_pm: dict[str, Any], rules: dict[str, Any]) -> str:
        if leakage.get("blocking_issues"):
            return "Need prediction coverage fixes"
        coverage = self._prediction_summary(scored).get("coverage_rate", 0.0)
        if coverage < 0.8:
            return "Need prediction coverage fixes"
        rule_b = next((row for row in rules["rules"] if row["rule"] == "Rule B"), {})
        if rule_b.get("candidate_count", 0) > 0:
            return "Phase 5-H Exit AI v2 Suppression Rule for high PM"
        if high_pm.get("high_pm_v2_top_decile_count", 0) > 0:
            return "Phase 5-H Exit AI v2 Conservative Gate"
        return "Do not integrate Exit AI v2 yet"

    def _root_path(self, path: Path) -> Path:
        if path.is_absolute():
            try:
                return self.root / path.relative_to(ROOT)
            except ValueError:
                return path
        return self.root / path

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value[:8])
                    if len(row.get(column, [])) > 8:
                        value += ", ..."
                values.append(str(value).replace("\n", " "))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)


def build_report(root: Path | str = ROOT) -> dict[str, Any]:
    return Phase5GExitAIV2PredictionAudit(root).build_report()


def save_report(result: dict[str, Any], root: Path | str = ROOT) -> Phase5GPaths:
    return Phase5GExitAIV2PredictionAudit(root).save_report(result)


def run(root: Path | str = ROOT) -> Phase5GPaths:
    audit = Phase5GExitAIV2PredictionAudit(root)
    return audit.save_report(audit.build_report())
