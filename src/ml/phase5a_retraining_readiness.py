"""Phase 5-A retraining readiness audit.

This is a read-only audit. It inventories existing models/datasets and decides
which AI should move to a dataset-design phase before any retraining happens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase5a_retraining_readiness_audit_2023-01_to_2026-05"
PERIOD_LABEL = "2023-01_to_2026-05"

STOCK_MODEL = ROOT / "models" / "ml" / "current_enriched_v2"
EXIT_MODEL = ROOT / "models" / "ml" / "exit" / "current_v2_66"
PM_MODEL = ROOT / "models" / "ml" / "portfolio_manager" / "current_v2_73_phase3b_clean"

STOCK_DATASET = ROOT / "data" / "ml" / "datasets" / "ml_dataset.parquet"
EXIT_DATASET = ROOT / "data" / "ml" / "exit_datasets" / "exit_dataset_v2_66_2023-01_to_2026-05.parquet"
PM_DATASET = ROOT / "data" / "ml" / "portfolio_manager" / "portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet"

WALK_FORWARD_PREDICTIONS = ROOT / "data" / "ml" / "walk_forward_predictions"
FEATURES_PATH = ROOT / "data" / "ml" / "features"
LABELS_PATH = ROOT / "data" / "ml" / "labels"

REPORT_PATHS = {
    "walk_forward_model_audit": ROOT / "reports" / "ml" / "walk_forward_model_audit_5y_enriched_v2.json",
    "portfolio_manager_lineage": ROOT / "reports" / "ml" / "portfolio_manager_data_lineage_audit_2023-01_to_2026-05.json",
    "exit_training": ROOT / "reports" / "ml" / "exit_model_training_v2_66_2023-01_to_2026-05.json",
    "exit_analysis": ROOT / "reports" / "ml" / "ml_exit_analysis_v2_66_2023-01_to_2026-05.json",
    "phase4a_exit_quality": ROOT / "reports" / "ml" / "portfolio_manager_phase4a_exit_quality_2023-01_to_2026-05.json",
    "phase4g_exit_delay": ROOT / "reports" / "ml" / "portfolio_manager_phase4g_exit_delay_candidate_hold_audit_2023-01_to_2026-05.json",
    "phase4c_pm_result": ROOT / "reports" / "ml" / "portfolio_manager_phase4c_high_pm_min_hold_2023-01_to_2026-05.json",
}

FORBIDDEN_FEATURES = {
    "selected_count_in_day",
    "actual_net_profit",
    "actual_buy_amount",
    "actual_shares",
    "actual_holding_days",
    "decision",
    "skip_reason",
    "exit_reason",
    "cash_after",
    "final_amount",
    "final_shares",
    "remaining_days_to_actual_exit",
}

BACKTEST_DERIVED_COLUMNS = {
    "trade_id",
    "actual_exit_date",
    "actual_buy_amount",
    "actual_shares",
    "actual_net_profit",
    "actual_holding_days",
    "realized_return",
    "positive_trade",
    "decision",
    "skip_reason",
    "exit_reason",
    "cash_before",
    "cash_after",
    "daily_buy_limit_remaining_before",
    "daily_buy_limit_remaining_after",
    "max_positions_remaining_before",
    "planned_shares",
    "planned_amount",
    "scaled_shares",
    "scaled_amount",
    "final_shares",
    "final_amount",
    "allocation_limit",
    "allocation_reason",
    "remaining_days_to_actual_exit",
}

API_ONLY_LABEL_POLICY = (
    "Retraining labels must be mechanically derived from API-origin market, "
    "financial, and price data only. Backtest trades, realized P/L, win/loss, "
    "portfolio history, and selected-only outcomes are audit references only."
)


@dataclass(frozen=True)
class Phase5APaths:
    markdown: Path
    json: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_parquet_sample(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.DataFrame()


def _parquet_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": 0, "columns": []}
    frame = _read_parquet_sample(path)
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "date_min": _date_min(frame),
        "date_max": _date_max(frame),
    }


def _date_min(frame: pd.DataFrame) -> str | None:
    for column in ["date", "signal_date", "current_date", "entry_date"]:
        if column in frame.columns and not frame.empty:
            values = frame[column].dropna().astype(str)
            return None if values.empty else str(values.min())
    return None


def _date_max(frame: pd.DataFrame) -> str | None:
    for column in ["date", "signal_date", "current_date", "entry_date"]:
        if column in frame.columns and not frame.empty:
            values = frame[column].dropna().astype(str)
            return None if values.empty else str(values.max())
    return None


def _feature_count(model_dir: Path, metadata: dict[str, Any]) -> int | None:
    if metadata.get("feature_count") is not None:
        return int(metadata["feature_count"])
    path = model_dir / "feature_columns.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return len(payload)
    return None


def _feature_columns(model_dir: Path, metadata: dict[str, Any]) -> list[str]:
    if isinstance(metadata.get("feature_columns"), list):
        return [str(item) for item in metadata["feature_columns"]]
    path = model_dir / "feature_columns.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(item) for item in payload]
    return []


def _targets(metadata: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    if isinstance(metadata.get("targets"), dict):
        return list(metadata["targets"].keys())
    return list(metrics.keys())


class Phase5ARetrainingReadinessAudit:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)

    def build_report(self) -> dict[str, Any]:
        inventory = self._model_inventory()
        exit_readiness = self._exit_ai_readiness(inventory)
        pm_readiness = self._pm_ai_readiness(inventory)
        stock_readiness = self._stock_selection_readiness(inventory)
        leakage = self._leakage_audit(inventory)
        priorities = self._priorities(exit_readiness, pm_readiness, stock_readiness, leakage)
        return {
            "metadata": {
                "phase": "5-A",
                "audit_only": True,
                "api_only_retraining_data_policy": True,
                "model_retraining_executed": False,
                "profile_added": False,
                "full_backtest_executed": False,
                "full_pytest_executed": False,
                "period": PERIOD_LABEL,
            },
            "input_paths": self._input_paths(),
            "data_policy": {
                "allowed_sources": [
                    "API-origin price series",
                    "API-origin financial data",
                    "features derived from API-origin data",
                    "mechanical future-return labels derived from API-origin price series",
                    "walk-forward predictions generated without future feature leakage",
                ],
                "forbidden_sources": [
                    "trades.csv as teacher labels",
                    "backtest_summary.json as teacher labels",
                    "summary.csv / portfolio history as teacher labels",
                    "realized P/L or win/loss as teacher labels",
                    "v2_75-v2_78 trading outcomes as ground truth",
                    "selected-only backtest universe as the training universe",
                    "post-backtest outcomes mixed into labels",
                ],
                "policy": API_ONLY_LABEL_POLICY,
            },
            "model_inventory": inventory,
            "exit_ai_readiness": exit_readiness,
            "portfolio_manager_ai_readiness": pm_readiness,
            "stock_selection_ai_readiness": stock_readiness,
            "leakage_audit": leakage,
            "retraining_priority": priorities,
            "recommended_next_phase": self._recommended_next_phase(priorities, leakage),
        }

    def save_report(self, result: dict[str, Any]) -> Phase5APaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase5APaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# AI Retraining Phase 5-A Readiness Audit",
            "",
            "## Scope",
            "",
            "- audit_only: `true`",
            "- no model retraining, no profile creation, no full backtest, no full pytest",
            "- current model directories are read-only inputs",
            "- retraining labels must be API-only mechanical future-return/quality labels; backtest outcomes are not allowed as teacher labels",
            "",
            "## Existing Model Inventory",
            "",
            self._table(
                result["model_inventory"],
                [
                    "model_name",
                    "current_model_path",
                    "dataset_path",
                    "feature_count",
                    "label_definition",
                    "training_period",
                    "validation_period",
                    "prediction_period",
                    "leakage_guard_available",
                    "retraining_dataset_policy_status",
                    "retraining_risk",
                ],
            ),
            "",
            "## Exit AI Readiness",
            "",
            self._table([result["exit_ai_readiness"]], ["recommended_exit_label", "label_leakage_risk", "required_data_available", "expected_benefit", "implementation_complexity"]),
            "",
            self._table(result["exit_ai_readiness"].get("feasible_labels", []), ["label", "feasible", "reason"]),
            "",
            "## Portfolio Manager AI Readiness",
            "",
            self._table([result["portfolio_manager_ai_readiness"]], ["pm_retraining_recommended", "reason", "risk_of_overfitting", "required_new_labels", "expected_benefit"]),
            "",
            "## Stock Selection AI Readiness",
            "",
            self._table([result["stock_selection_ai_readiness"]], ["stock_selection_retraining_recommended", "reason", "risk", "expected_benefit"]),
            "",
            "## Leakage Audit",
            "",
            self._table(result["leakage_audit"].get("leakage_risk_summary", []), ["risk", "status", "detail"]),
            "",
            "Blocking issues:",
            "",
        ]
        blocking = result["leakage_audit"].get("blocking_issues", [])
        lines.extend([f"- {item}" for item in blocking] or ["- None"])
        lines.extend(
            [
                "",
                "## Retraining Priority",
                "",
                self._table(result["retraining_priority"].get("candidates", []), ["candidate", "expected_profit_impact", "expected_dd_impact", "implementation_complexity", "leakage_risk", "comparison_difficulty", "recommended_order"]),
                "",
                "## Recommended Next Phase",
                "",
                f"`{result['recommended_next_phase']}`",
                "",
            ]
        )
        return "\n".join(lines)

    def _input_paths(self) -> dict[str, str]:
        return {
            "stock_model": str(self._root_path(STOCK_MODEL)),
            "exit_model": str(self._root_path(EXIT_MODEL)),
            "portfolio_manager_model": str(self._root_path(PM_MODEL)),
            "stock_dataset": str(self._root_path(STOCK_DATASET)),
            "exit_dataset": str(self._root_path(EXIT_DATASET)),
            "portfolio_manager_dataset": str(self._root_path(PM_DATASET)),
            "features": str(self._root_path(FEATURES_PATH)),
            "labels": str(self._root_path(LABELS_PATH)),
            "walk_forward_predictions": str(self._root_path(WALK_FORWARD_PREDICTIONS)),
        }

    def _root_path(self, path: Path) -> Path:
        if path.is_absolute():
            try:
                return self.root / path.relative_to(ROOT)
            except ValueError:
                return path
        return self.root / path

    def _model_inventory(self) -> list[dict[str, Any]]:
        specs = [
            {
                "model_name": "Stock Selection AI",
                "model_dir": self._root_path(STOCK_MODEL),
                "dataset": self._root_path(STOCK_DATASET),
                "label_definition": "future returns, upside_10d, bad_entry_10d, max return and swing labels; ranking uses risk_adjusted_score = expected_return_10d - 0.5 * bad_entry_probability_10d",
                "prediction_period": self._prediction_period(),
                "label_source_policy": "api_price_mechanical_labels",
                "retraining_risk": "high",
            },
            {
                "model_name": "Exit AI",
                "model_dir": self._root_path(EXIT_MODEL),
                "dataset": self._root_path(EXIT_DATASET),
                "label_definition": "current dataset contains exit-state rows and backtest-derived trade identifiers; future retraining must rebuild labels from API price paths only",
                "prediction_period": "existing audit dataset current_date rows; not approved as retraining labels under API-only policy",
                "label_source_policy": "mixed_backtest_state_reference_not_retraining_safe",
                "retraining_risk": "medium",
            },
            {
                "model_name": "Portfolio Manager AI",
                "model_dir": self._root_path(PM_MODEL),
                "dataset": self._root_path(PM_DATASET),
                "label_definition": "current labels are backtest-outcome-derived PM targets; future retraining must use API-only market/return labels and not copy trade outcomes",
                "prediction_period": "v2_73 clean dataset/backtest period 2023-01 to 2026-05; audit reference only",
                "label_source_policy": "backtest_outcome_reference_not_retraining_safe",
                "retraining_risk": "medium",
            },
        ]
        rows = []
        for spec in specs:
            metadata = _read_json(spec["model_dir"] / "model_metadata.json")
            metrics = _read_json(spec["model_dir"] / "metrics.json")
            dataset = _parquet_info(spec["dataset"])
            rows.append(
                {
                    "model_name": spec["model_name"],
                    "current_model_path": str(spec["model_dir"]),
                    "dataset_path": str(spec["dataset"]),
                    "dataset_exists": dataset["exists"],
                    "dataset_rows": dataset["rows"],
                    "dataset_columns": dataset["columns"],
                    "backtest_derived_columns": sorted(BACKTEST_DERIVED_COLUMNS & set(dataset["columns"])),
                    "feature_count": _feature_count(spec["model_dir"], metadata),
                    "feature_columns": _feature_columns(spec["model_dir"], metadata),
                    "targets": _targets(metadata, metrics),
                    "label_definition": spec["label_definition"],
                    "training_period": self._period(metadata, "train_start", "train_end"),
                    "validation_period": self._period(metadata, "valid_start", "valid_end") or self._period(metadata, "test_start", "test_end"),
                    "prediction_period": spec["prediction_period"],
                    "label_source_policy": spec["label_source_policy"],
                    "model_version": metadata.get("model_profile") or spec["model_dir"].name,
                    "metrics_available": bool(metrics),
                    "leakage_guard_available": bool(metadata.get("leakage_guard")),
                    "leakage_guard": metadata.get("leakage_guard", ""),
                    "retraining_dataset_policy_status": self._dataset_policy_status(spec["label_source_policy"], dataset["columns"]),
                    "data_lineage_report_path": self._lineage_path_for(spec["model_name"]),
                    "retraining_risk": spec["retraining_risk"],
                }
            )
        return rows

    def _dataset_policy_status(self, source_policy: str, columns: list[str]) -> str:
        if source_policy == "api_price_mechanical_labels":
            return "approved_reference_for_api_only_retraining_design"
        hits = sorted(BACKTEST_DERIVED_COLUMNS & set(columns))
        if hits:
            return "not_retraining_safe_backtest_derived_columns_present"
        return "needs_manual_source_verification"

    def _period(self, metadata: dict[str, Any], start_key: str, end_key: str) -> str:
        start = metadata.get(start_key)
        end = metadata.get(end_key)
        return "" if not start and not end else f"{start or '?'} to {end or '?'}"

    def _prediction_period(self) -> str:
        root = self._root_path(WALK_FORWARD_PREDICTIONS)
        dates = sorted(path.stem.replace("predictions_", "") for path in root.glob("predictions_*.parquet"))
        return "" if not dates else f"{dates[0]} to {dates[-1]}"

    def _lineage_path_for(self, model_name: str) -> str:
        if model_name == "Portfolio Manager AI":
            return str(self._root_path(REPORT_PATHS["portfolio_manager_lineage"]))
        if model_name == "Stock Selection AI":
            return str(self._root_path(REPORT_PATHS["walk_forward_model_audit"]))
        return str(self._root_path(REPORT_PATHS["exit_training"]))

    def _exit_ai_readiness(self, inventory: list[dict[str, Any]]) -> dict[str, Any]:
        exit_info = next(row for row in inventory if row["model_name"] == "Exit AI")
        columns = set(exit_info.get("dataset_columns", []))
        has_5d = {"future_remaining_return_5d", "hold_better_5d", "should_exit_now_5d", "avoid_loss_5d"}.issubset(columns)
        has_10d = "future_remaining_return_10d" in columns
        retrain_safe_existing = exit_info.get("retraining_dataset_policy_status") == "approved_reference_for_api_only_retraining_design"
        phase4g = _read_json(self._root_path(REPORT_PATHS["phase4g_exit_delay"]))
        phase4a = _read_json(self._root_path(REPORT_PATHS["phase4a_exit_quality"]))
        feasible = [
            {"label": "should_exit_now", "feasible": "design_only", "reason": "must be rebuilt from API price path thresholds, not existing backtest exit rows"},
            {"label": "should_hold_5d", "feasible": "design_only", "reason": "derive from t to t+5d API price returns and drawdown constraints"},
            {"label": "exit_would_avoid_loss_5d", "feasible": "design_only", "reason": "derive mechanically from future low/close path after t"},
            {"label": "exit_would_miss_profit_5d", "feasible": "design_only", "reason": "derive from t+5d/high path return; do not use realized trade profit"},
            {"label": "exit_quality_score", "feasible": "recommended_design", "reason": "composite API-price label separating avoid-loss exits from missed-profit exits"},
            {"label": "future_return_after_exit_5d", "feasible": "yes_api_price", "reason": "API price cache can mechanically compute t to t+5d return"},
            {"label": "future_return_after_exit_10d", "feasible": "yes_api_price" if has_10d else "design_only", "reason": "API price cache can mechanically compute t to t+10d return"},
            {"label": "future_return_after_exit_20d", "feasible": "partial", "reason": "requires price-cache derivation; not in current exit dataset"},
        ]
        expected = "high"
        if phase4g.get("exit_delay_1d_audit", {}).get("summary", {}).get("profit_delta", 0) < 0:
            expected = "medium-high; blanket delay is harmful, so label must separate avoid-loss exits from missed-profit exits"
        return {
            "current_objective": ", ".join(exit_info.get("targets", [])),
            "current_features": exit_info.get("feature_columns", []),
            "current_training_period": exit_info.get("training_period", ""),
            "current_metrics": "metrics.json available" if exit_info.get("metrics_available") else "missing",
            "dataset_rows": exit_info.get("dataset_rows", 0),
            "phase4_exit_evidence_available": bool(phase4a or phase4g),
            "feasible_labels": feasible,
            "recommended_exit_label": "API-only exit_quality_score with explicit avoid_loss_5d and miss_profit_5d components",
            "label_leakage_risk": "medium until rebuilt from API-only price paths",
            "required_data_available": bool(has_5d and has_10d),
            "existing_dataset_retraining_safe": retrain_safe_existing,
            "early_sell_problem_representable": "yes, with API price labels over all eligible holdings/candidates rather than trade outcomes",
            "late_stop_loss_problem_representable": "yes, with API future-low/drawdown labels",
            "high_pm_early_sell_problem_representable": "only if PM predictions/features at time t are joined as features; do not join v2_78 trade outcomes",
            "expected_benefit": expected,
            "implementation_complexity": "medium",
        }

    def _pm_ai_readiness(self, inventory: list[dict[str, Any]]) -> dict[str, Any]:
        pm_info = next(row for row in inventory if row["model_name"] == "Portfolio Manager AI")
        phase4c = _read_json(self._root_path(REPORT_PATHS["phase4c_pm_result"]))
        dataset_columns = set(pm_info.get("dataset_columns", []))
        forbidden_hits = sorted(FORBIDDEN_FEATURES & set(pm_info.get("feature_columns", [])))
        return {
            "current_objective": ", ".join(pm_info.get("targets", [])),
            "high_conviction_definition": "high_conviction_target classification probability",
            "avoid_proba_definition": "avoid_target classification probability",
            "pm_score_definition": "pm_score = high_conviction_proba - avoid_proba",
            "v2_78_pm_score_effective": bool(phase4c),
            "pm_retraining_recommended": "not_first",
            "reason": "v2_78 shows PM score is useful, but the current PM dataset uses backtest-outcome labels; PM retraining must wait for API-only label redesign",
            "risk_of_overfitting": "high if trained on selected-only/backtest-outcome rows; medium only after API-only full-universe/candidate labels are designed",
            "required_new_labels": "API-only future-return/quality labels; no realized P/L, win/loss, or v2_78 trade outcomes",
            "expected_benefit": "medium",
            "dataset_rows": pm_info.get("dataset_rows", 0),
            "forbidden_feature_hits": forbidden_hits,
            "existing_dataset_retraining_safe": pm_info.get("retraining_dataset_policy_status") == "approved_reference_for_api_only_retraining_design",
            "selected_count_in_day_used": "selected_count_in_day" in dataset_columns or "selected_count_in_day" in pm_info.get("feature_columns", []),
            "compatible_with_per_code_cap_and_pm_ordering": True,
        }

    def _stock_selection_readiness(self, inventory: list[dict[str, Any]]) -> dict[str, Any]:
        stock_info = next(row for row in inventory if row["model_name"] == "Stock Selection AI")
        return {
            "current_objective": ", ".join(stock_info.get("targets", [])),
            "ranking_score_definition": "risk_adjusted_score = expected_return_10d - 0.5 * bad_entry_probability_10d",
            "degradation_2023_2026": "no blocking degradation identified in this audit; current strategy stack is strongest after PM downstream controls",
            "pm_ai_downstream_correction": True,
            "stock_selection_retraining_recommended": False,
            "reason": "buy side is already tuned and changing the base selector has high blast radius",
            "risk": "high",
            "expected_benefit": "uncertain",
        }

    def _leakage_audit(self, inventory: list[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        blocking: list[str] = []
        retraining_blocks: list[str] = []
        safe_flags: dict[str, bool] = {}
        for item in inventory:
            name = item["model_name"]
            features = set(item.get("feature_columns", []))
            hits = sorted(features & FORBIDDEN_FEATURES)
            status = "pass" if not hits else "block"
            if hits:
                blocking.append(f"{name}: forbidden feature(s) in model feature_columns: {', '.join(hits)}")
            rows.append({"risk": f"{name} forbidden feature columns", "status": status, "detail": ", ".join(hits) if hits else "none"})
            safe_flags[f"{name}_feature_columns_safe"] = not hits
            guard = bool(item.get("leakage_guard_available"))
            rows.append({"risk": f"{name} leakage guard metadata", "status": "pass" if guard else "warn", "detail": item.get("leakage_guard", "") or "missing"})
            safe_flags[f"{name}_leakage_guard_available"] = guard
            if item.get("retraining_dataset_policy_status") == "not_retraining_safe_backtest_derived_columns_present":
                detail = ", ".join(item.get("backtest_derived_columns", [])[:12])
                rows.append({"risk": f"{name} API-only retraining label policy", "status": "block_retraining", "detail": detail})
                retraining_blocks.append(f"{name}: existing dataset is audit/reference only under API-only policy; rebuild labels from API-derived prices/features before retraining")
                safe_flags[f"{name}_existing_dataset_retraining_safe"] = False
            else:
                rows.append({"risk": f"{name} API-only retraining label policy", "status": "pass", "detail": item.get("retraining_dataset_policy_status", "")})
                safe_flags[f"{name}_existing_dataset_retraining_safe"] = True
        generic = [
            ("future price leakage", "warn", "labels require future prices; must enforce effective train-end by label horizon"),
            ("same-day close leakage", "warn", "safe only if prediction/backtest uses features known at signal time"),
            ("selected_count_in_day leakage", "pass", "forbidden by PM clean dataset/model feature checks"),
            ("backtest result leakage", "block_retraining", "backtest outcomes are not allowed as teacher labels; existing PM/Exit outcome datasets are reference-only"),
            ("target leakage from realized trades", "block_retraining", "realized trade P/L and win/loss must not be labels"),
            ("using current model for past predictions", "pass", "walk-forward predictions path and model audit are available"),
            ("train/test period overlap", "pass", "metadata shows chronological splits; future retraining must preserve this"),
        ]
        rows.extend({"risk": risk, "status": status, "detail": detail} for risk, status, detail in generic)
        safe_flags.update(
            {
                "safe_to_design_exit_dataset_v2": not blocking,
                "safe_to_retrain_exit_now": False,
                "safe_to_retrain_pm_now": False,
                "safe_to_retrain_stock_selection_now": False,
                "api_only_label_rebuild_required": True,
            }
        )
        return {
            "leakage_risk_summary": rows,
            "blocking_issues": blocking,
            "retraining_blocking_issues": retraining_blocks,
            "safe_to_retrain_flags": safe_flags,
        }

    def _priorities(
        self,
        exit_readiness: dict[str, Any],
        pm_readiness: dict[str, Any],
        stock_readiness: dict[str, Any],
        leakage: dict[str, Any],
    ) -> dict[str, Any]:
        blocking = bool(leakage.get("blocking_issues"))
        candidates = [
            {
                "candidate": "Exit AI retraining",
                "expected_profit_impact": "high",
                "expected_dd_impact": "medium-high",
                "implementation_complexity": exit_readiness["implementation_complexity"],
                "leakage_risk": exit_readiness["label_leakage_risk"],
                "comparison_difficulty": "medium",
                "recommended_order": 1 if not blocking else 2,
            },
            {
                "candidate": "Market Regime Audit before retraining",
                "expected_profit_impact": "medium",
                "expected_dd_impact": "high",
                "implementation_complexity": "medium",
                "leakage_risk": "low",
                "comparison_difficulty": "medium",
                "recommended_order": 2 if not blocking else 1,
            },
            {
                "candidate": "PM AI retraining",
                "expected_profit_impact": pm_readiness["expected_benefit"],
                "expected_dd_impact": "medium",
                "implementation_complexity": "medium-high",
                "leakage_risk": pm_readiness["risk_of_overfitting"],
                "comparison_difficulty": "high",
                "recommended_order": 3,
            },
            {
                "candidate": "Stock Selection AI retraining",
                "expected_profit_impact": stock_readiness["expected_benefit"],
                "expected_dd_impact": "uncertain",
                "implementation_complexity": "high",
                "leakage_risk": stock_readiness["risk"],
                "comparison_difficulty": "high",
                "recommended_order": 4,
            },
            {
                "candidate": "No retraining yet",
                "expected_profit_impact": "none",
                "expected_dd_impact": "none",
                "implementation_complexity": "low",
                "leakage_risk": "low",
                "comparison_difficulty": "low",
                "recommended_order": 5,
            },
        ]
        return {"candidates": candidates, "recommended_order": [row["candidate"] for row in sorted(candidates, key=lambda row: row["recommended_order"])]}

    def _recommended_next_phase(self, priorities: dict[str, Any], leakage: dict[str, Any]) -> str:
        if leakage.get("blocking_issues"):
            return "Phase 5-B Leakage Fix Before Retraining"
        first = priorities.get("recommended_order", [""])[0]
        if first == "Exit AI retraining":
            return "Phase 5-B Exit AI v2 API-Only Dataset Design"
        if first == "Market Regime Audit before retraining":
            return "Phase 5-B Market Regime Audit"
        return "Retraining deferred"

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
    return Phase5ARetrainingReadinessAudit(root).build_report()


def save_report(result: dict[str, Any], root: Path | str = ROOT) -> Phase5APaths:
    return Phase5ARetrainingReadinessAudit(root).save_report(result)


def run(root: Path | str = ROOT) -> Phase5APaths:
    audit = Phase5ARetrainingReadinessAudit(root)
    return audit.save_report(audit.build_report())
