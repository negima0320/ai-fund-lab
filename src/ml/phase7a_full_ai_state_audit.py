"""Phase 7-A full AI state and retraining readiness audit.

This module is intentionally read-only. It inventories the current AI/model
state, checks API-origin data availability from 2021 onward, and recommends a
safe retraining order before any training, profile change, or backtest happens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_STEM = "phase7a_full_ai_state_audit_2021_to_2026"
PERIOD_LABEL = "2021_to_2026"

STOCK_MODEL = Path("models/ml/current_enriched_v2")
PM_MODEL = Path("models/ml/portfolio_manager/current_v2_73_phase3b_clean")
EXIT_MODEL = Path("models/ml/exit/current_v2_66")
EXIT_V2_MODEL = Path("models/ml/exit_ai_v2/candidate_v2_api_only")

STOCK_DATASET = Path("data/ml/datasets/ml_dataset.parquet")
EXIT_DATASET = Path("data/ml/exit_datasets/exit_dataset_v2_66_2023-01_to_2026-05.parquet")
EXIT_V2_DATASET = Path("data/ml/exit_ai_v2/exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet")
PM_DATASET = Path("data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet")

FEATURES_DIR = Path("data/ml/features")
LABELS_DIR = Path("data/ml/labels")
WALK_FORWARD_DIR = Path("data/ml/walk_forward_predictions")
PRICE_CACHE_DIR = Path("data/cache/jquants/prices")
TOPIX_CACHE_DIR = Path("data/cache/jquants/topix_prices")
FINANCIAL_CACHE_DIR = Path("data/cache/jquants/financial_statements")
LISTED_INFO_DIR = Path("data/cache/jquants/listed_info")

FORBIDDEN_FEATURE_PATTERNS = (
    "selected_count_in_day",
    "actual_",
    "realized_",
    "win_loss",
    "portfolio_",
    "backtest",
    "profile_",
    "trade_id",
    "exit_reason",
    "skip_reason",
    "cash_",
    "final_",
)

FORBIDDEN_LABEL_COLUMNS = {
    "trades.csv",
    "realized_profit",
    "net_profit",
    "actual_net_profit",
    "win_loss",
    "positive_trade",
    "portfolio_value",
    "total_assets",
    "cash_after",
    "cash_before",
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


@dataclass(frozen=True)
class Phase7APaths:
    markdown: Path
    json: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_or_list(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _date_range(frame: pd.DataFrame) -> dict[str, str | None]:
    for column in ("date", "signal_date", "current_date", "as_of_date", "entry_date"):
        if column in frame.columns and not frame.empty:
            values = frame[column].dropna().astype(str)
            if not values.empty:
                return {"from": str(values.min()), "to": str(values.max()), "date_column": column}
    return {"from": None, "to": None, "date_column": None}


def _period(metadata: dict[str, Any], pairs: tuple[tuple[str, str], ...]) -> str:
    for start_key, end_key in pairs:
        start = metadata.get(start_key)
        end = metadata.get(end_key)
        if start or end:
            return f"{start or '?'} to {end or '?'}"
    return ""


def _feature_columns(model_dir: Path, metadata: dict[str, Any]) -> list[str]:
    value = metadata.get("feature_columns")
    if isinstance(value, list):
        return [str(item) for item in value]
    path = model_dir / "feature_columns.json"
    payload = _read_json_or_list(path)
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("feature_columns"), list):
        return [str(item) for item in payload["feature_columns"]]
    return []


def _feature_count(model_dir: Path, metadata: dict[str, Any]) -> int | None:
    if metadata.get("feature_count") is not None:
        try:
            return int(metadata["feature_count"])
        except Exception:
            pass
    columns = _feature_columns(model_dir, metadata)
    return len(columns) if columns else None


def _targets(metadata: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    for key in ("target", "target_label", "label", "model_name"):
        if metadata.get(key):
            return [str(metadata[key])]
    if isinstance(metadata.get("targets"), dict):
        return [str(item) for item in metadata["targets"].keys()]
    if isinstance(metadata.get("targets"), list):
        return [str(item) for item in metadata["targets"]]
    return [str(item) for item in metrics.keys()]


def _missing_rate(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    sample = frame.head(100_000)
    return float(sample.isna().mean().mean())


def _count_codes(frame: pd.DataFrame) -> int:
    if frame.empty or "code" not in frame.columns:
        return 0
    return int(frame["code"].astype(str).nunique())


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value[:8]) + (", ..." if len(value) > 8 else "")
    return str(value).replace("\n", " ")


class Phase7AFullAIStateAudit:
    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)

    def build_report(self) -> dict[str, Any]:
        inventory = self._ai_inventory()
        data = self._data_availability()
        pm = self._pm_dataset_audit(inventory)
        stock = self._stock_ai_audit(inventory, data)
        exit_ai = self._exit_ai_audit(inventory)
        leakage = self._leakage_audit(inventory)
        plans = self._retraining_plans(pm, stock, exit_ai, leakage, data)
        safe_design = self._safe_retraining_design(data)
        final = self._final_verdict(inventory, data, pm, stock, exit_ai, leakage, plans)
        return {
            "metadata": {
                "phase": "7-A",
                "audit_only": True,
                "model_retraining_executed": False,
                "full_backtest_executed": False,
                "profile_added": False,
                "current_model_overwritten": False,
                "full_pytest_executed": False,
                "jquants_api_refetch": False,
                "openai_used": False,
                "live_order": False,
                "period": PERIOD_LABEL,
                "current_main_candidate": "rookie_dealer_02_v2_82_cap38",
            },
            "data_policy": self._data_policy(),
            "input_paths": self._input_paths(),
            "ai_inventory": inventory,
            "data_availability_2021": data,
            "pm_ai_dataset_audit": pm,
            "stock_selection_ai_audit": stock,
            "exit_ai_audit": exit_ai,
            "leakage_audit": leakage,
            "retraining_plan_comparison": plans,
            "safe_retraining_design": safe_design,
            "final_verdict": final,
        }

    def save_report(self, result: dict[str, Any]) -> Phase7APaths:
        report_dir = self.root / "reports" / "ml"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{REPORT_STEM}.json"
        md_path = report_dir / f"{REPORT_STEM}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self.format_markdown(result), encoding="utf-8")
        return Phase7APaths(markdown=md_path, json=json_path)

    def format_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# AI Retraining Phase 7-A Full AI State Audit",
            "",
            "## Scope",
            "",
            "- audit only",
            "- no retraining, no full backtest, no profile addition, no current model overwrite, no full pytest",
            "- J-Quants API refetch and OpenAI API are not used",
            "- backtest outcomes are forbidden as teacher labels",
            "",
            "## AI Inventory",
            "",
            self._table(
                result["ai_inventory"],
                [
                    "model_name",
                    "current_model_path",
                    "model_exists",
                    "model_version",
                    "feature_count",
                    "training_dataset_path",
                    "training_dataset_exists",
                    "dataset_rows",
                    "dataset_date_range",
                    "target_label",
                    "api_only",
                    "backtest_derived_columns_present",
                    "leakage_risk",
                    "retraining_allowed",
                    "retraining_recommended",
                    "reason",
                ],
            ),
            "",
            "## 2021 Data Availability",
            "",
            self._table(
                result["data_availability_2021"].get("sources", []),
                ["source", "path", "exists", "available_from", "available_to", "usable_rows", "usable_codes", "missing_rate", "blocking_issues"],
            ),
            "",
            "## PM AI Dataset Audit",
            "",
            self._table([result["pm_ai_dataset_audit"]], ["pm_dataset_retraining_safe", "pm_dataset_needs_rebuild", "pm_retraining_priority", "reason"]),
            "",
            "## Stock Selection AI Audit",
            "",
            self._table([result["stock_selection_ai_audit"]], ["stock_ai_retraining_safe", "stock_ai_retraining_priority", "reason", "risk"]),
            "",
            "## Exit AI Audit",
            "",
            self._table([result["exit_ai_audit"]], ["exit_ai_current_state", "exit_ai_v2_candidate_state", "exit_ai_retraining_priority", "integration_failure_likely_cause", "reason"]),
            "",
            "## Leakage Audit",
            "",
            self._table(result["leakage_audit"].get("checks", []), ["check", "status", "detail"]),
            "",
            "## Retraining Plan Comparison",
            "",
            self._table(result["retraining_plan_comparison"], ["plan", "expected_benefit", "expected_risk", "leakage_risk", "implementation_cost", "comparison_difficulty", "recommended_order"]),
            "",
            "## Safe Retraining Design",
            "",
            self._table([result["safe_retraining_design"]], ["recommended_design", "walk_forward_training", "expanding_window", "static_split_viable", "reason"]),
            "",
            "## Final Verdict",
            "",
            self._table(
                [result["final_verdict"]],
                [
                    "all_ai_retraining_ready",
                    "recommended_first_retraining_target",
                    "recommended_retraining_order",
                    "retraining_should_start_now",
                    "blocking_issues",
                    "next_phase_recommended",
                ],
            ),
            "",
        ]
        return "\n".join(lines)

    def _root(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    def _input_paths(self) -> dict[str, str]:
        paths = {
            "stock_model": STOCK_MODEL,
            "portfolio_manager_model": PM_MODEL,
            "exit_model": EXIT_MODEL,
            "exit_ai_v2_candidate_model": EXIT_V2_MODEL,
            "stock_dataset": STOCK_DATASET,
            "portfolio_manager_dataset": PM_DATASET,
            "exit_dataset": EXIT_DATASET,
            "exit_ai_v2_dataset": EXIT_V2_DATASET,
            "features": FEATURES_DIR,
            "labels": LABELS_DIR,
            "walk_forward_predictions": WALK_FORWARD_DIR,
            "price_cache": PRICE_CACHE_DIR,
            "financial_cache": FINANCIAL_CACHE_DIR,
        }
        return {key: str(self._root(path)) for key, path in paths.items()}

    def _data_policy(self) -> dict[str, Any]:
        return {
            "allowed_teacher_sources": [
                "API-origin price series",
                "API-origin financial data",
                "API-origin market context",
                "features derived from API-origin data",
                "mechanical future-return labels from API-origin price series",
            ],
            "forbidden_teacher_sources": [
                "trades.csv",
                "backtest_summary.json",
                "portfolio_history / summary.csv",
                "realized_profit",
                "win/loss labels from executed trades",
                "v2_75/v2_76/v2_77/v2_78/v2_82 backtest outcomes",
                "selected-only backtest universe as the only training target universe",
                "current model regenerated past predictions",
                "selected_count_in_day",
            ],
        }

    def _ai_inventory(self) -> list[dict[str, Any]]:
        specs = [
            {
                "model_name": "Stock Selection AI",
                "model_path": STOCK_MODEL,
                "dataset_path": STOCK_DATASET,
                "target_label": "future-return / bad-entry / ranking labels",
                "label_source": "API-origin price-derived labels",
                "feature_source": "data/ml/features + API-origin derived feature store",
                "api_only": True,
                "base_retraining_recommended": "defer",
                "reason": "v2_82 buy-side stack is strong; stock selector retraining has high blast radius",
            },
            {
                "model_name": "Portfolio Manager AI",
                "model_path": PM_MODEL,
                "dataset_path": PM_DATASET,
                "target_label": "high_conviction_target / avoid_target",
                "label_source": "current PM dataset includes backtest/outcome-derived audit labels",
                "feature_source": "PM clean dataset feature columns",
                "api_only": False,
                "base_retraining_recommended": "rebuild_dataset_first",
                "reason": "current PM model is useful but dataset is not safe for direct retraining under API-only policy",
            },
            {
                "model_name": "Exit AI current v2_66",
                "model_path": EXIT_MODEL,
                "dataset_path": EXIT_DATASET,
                "target_label": "avoid_loss_5d_classification",
                "label_source": "existing exit-state dataset; reference only for retraining policy",
                "feature_source": "exit-state features and walk-forward predictions",
                "api_only": False,
                "base_retraining_recommended": "defer",
                "reason": "current Exit AI remains active; v2 candidate integration underperformed",
            },
            {
                "model_name": "Exit AI v2 candidate",
                "model_path": EXIT_V2_MODEL,
                "dataset_path": EXIT_V2_DATASET,
                "target_label": "exit_quality_top_decile",
                "label_source": "API-only future-return quality labels",
                "feature_source": "API-only Exit AI v2 dataset",
                "api_only": True,
                "base_retraining_recommended": "integration_redesign_first",
                "reason": "model trained safely, but Phase 5-H integration failed; likely integration/gating issue before retraining issue",
            },
            {
                "model_name": "Rule-based scoring / auxiliary scoring",
                "model_path": Path("config/profiles"),
                "dataset_path": Path("data/processed/common"),
                "target_label": "manual technical scoring rules",
                "label_source": "not trained model",
                "feature_source": "indicator cache / processed candidates",
                "api_only": True,
                "base_retraining_recommended": "not_applicable",
                "reason": "auxiliary scoring is rule/config based; audit for consistency rather than model retraining",
            },
        ]
        return [self._inventory_row(spec) for spec in specs]

    def _inventory_row(self, spec: dict[str, Any]) -> dict[str, Any]:
        model_path = self._root(spec["model_path"])
        dataset_path = self._root(spec["dataset_path"])
        metadata = _read_json(model_path / "model_metadata.json") if model_path.is_dir() else {}
        metrics = _read_json(model_path / "metrics.json") if model_path.is_dir() else {}
        dataset = self._dataset_info(dataset_path)
        features = _feature_columns(model_path, metadata) if model_path.is_dir() else []
        forbidden_features = self._forbidden_feature_hits(features)
        backtest_cols = sorted(BACKTEST_DERIVED_COLUMNS & set(dataset["columns"]))
        label_like_cols = sorted(FORBIDDEN_LABEL_COLUMNS & set(dataset["columns"]))
        backtest_present = bool(backtest_cols or label_like_cols)
        leakage_risk = self._leakage_risk(spec, forbidden_features, backtest_present)
        retraining_allowed = bool(spec["api_only"] and not forbidden_features and not backtest_present and dataset["exists"])
        return {
            "model_name": spec["model_name"],
            "current_model_path": str(model_path),
            "model_exists": model_path.exists(),
            "model_version": metadata.get("model_profile") or metadata.get("model_version") or metadata.get("version") or model_path.name,
            "feature_count": _feature_count(model_path, metadata) if model_path.is_dir() else None,
            "feature_columns": features,
            "training_dataset_path": str(dataset_path),
            "training_dataset_exists": dataset["exists"],
            "dataset_rows": dataset["rows"],
            "dataset_columns": dataset["columns"],
            "dataset_date_range": dataset["date_range"],
            "train_period": _period(metadata, (("train_start", "train_end"), ("training_start", "training_end"))),
            "validation_period": _period(metadata, (("valid_start", "valid_end"), ("validation_start", "validation_end"))),
            "test_period": _period(metadata, (("test_start", "test_end"),)),
            "target_label": ", ".join(_targets(metadata, metrics)) or spec["target_label"],
            "label_source": spec["label_source"],
            "feature_source": spec["feature_source"],
            "api_only": spec["api_only"],
            "backtest_derived_columns": backtest_cols,
            "label_like_columns": label_like_cols,
            "backtest_derived_columns_present": backtest_present,
            "forbidden_feature_hits": forbidden_features,
            "leakage_risk": leakage_risk,
            "retraining_allowed": retraining_allowed,
            "retraining_recommended": spec["base_retraining_recommended"],
            "reason": spec["reason"],
        }

    def _dataset_info(self, path: Path) -> dict[str, Any]:
        if path.is_dir():
            files = sorted(path.glob("*.parquet"))
            if not files:
                return {"exists": path.exists(), "rows": 0, "columns": [], "date_range": "", "missing_rate": None, "codes": 0}
            dates = self._dates_from_filenames(files)
            sample = _read_parquet(files[0])
            return {
                "exists": True,
                "rows": self._safe_row_count(files),
                "columns": list(sample.columns),
                "date_range": f"{dates[0]} to {dates[-1]}" if dates else "",
                "missing_rate": _missing_rate(sample),
                "codes": _count_codes(sample),
            }
        frame = _read_parquet(path)
        rng = _date_range(frame)
        return {
            "exists": path.exists(),
            "rows": int(len(frame)) if not frame.empty else 0,
            "columns": list(frame.columns) if not frame.empty else [],
            "date_range": f"{rng['from']} to {rng['to']}" if rng["from"] else "",
            "missing_rate": _missing_rate(frame),
            "codes": _count_codes(frame),
        }

    def _safe_row_count(self, files: list[Path]) -> int:
        total = 0
        for path in files[:2000]:
            frame = _read_parquet(path)
            total += int(len(frame))
        return total

    def _dates_from_filenames(self, files: list[Path]) -> list[str]:
        dates = []
        for path in files:
            text = path.stem
            for token in text.replace("_", "-").split("-"):
                pass
            parts = text.split("_")
            candidate = parts[-1] if parts else text
            if len(candidate) == 10 and candidate[4] == "-" and candidate[7] == "-":
                dates.append(candidate)
        return sorted(dates)

    def _forbidden_feature_hits(self, features: list[str]) -> list[str]:
        hits = []
        for feature in features:
            lower = feature.lower()
            if any(pattern in lower for pattern in FORBIDDEN_FEATURE_PATTERNS):
                hits.append(feature)
        return sorted(set(hits))

    def _leakage_risk(self, spec: dict[str, Any], forbidden_features: list[str], backtest_present: bool) -> str:
        if forbidden_features:
            return "high"
        if backtest_present and not spec["api_only"]:
            return "high_for_direct_retraining"
        if spec["api_only"]:
            return "low"
        return "medium"

    def _data_availability(self) -> dict[str, Any]:
        sources = [
            self._availability_from_dataset("stock_ml_dataset", STOCK_DATASET),
            self._availability_from_dir("feature_store", FEATURES_DIR),
            self._availability_from_dir("label_store", LABELS_DIR),
            self._availability_from_dataset("exit_ai_v2_dataset", EXIT_V2_DATASET),
            self._availability_from_dir("walk_forward_predictions", WALK_FORWARD_DIR),
            self._availability_from_dir("jquants_price_cache", PRICE_CACHE_DIR),
            self._availability_from_dir("jquants_topix_cache", TOPIX_CACHE_DIR),
            self._availability_from_dir("jquants_financial_cache", FINANCIAL_CACHE_DIR),
            self._availability_from_dir("jquants_listed_info_cache", LISTED_INFO_DIR),
        ]
        blocking = []
        if not any(row["available_from"] and str(row["available_from"]) <= "2021-06-01" for row in sources):
            blocking.append("no source clearly available from 2021-06-01")
        price = next((row for row in sources if row["source"] == "jquants_price_cache"), {})
        if not price.get("exists"):
            blocking.append("price cache missing")
        labels = next((row for row in sources if row["source"] == "label_store"), {})
        if not labels.get("exists"):
            blocking.append("label store missing")
        return {
            "sources": sources,
            "available_from": min([row["available_from"] for row in sources if row.get("available_from")] or [""]),
            "available_to": max([row["available_to"] for row in sources if row.get("available_to")] or [""]),
            "horizon_shortage_period": "latest horizon rows near 2026-05 may be unusable for 5d/10d/20d labels",
            "blocking_issues": blocking,
            "api_only_2021_retraining_data_possible": not blocking,
        }

    def _availability_from_dataset(self, source: str, rel_path: Path) -> dict[str, Any]:
        path = self._root(rel_path)
        info = self._dataset_info(path)
        date_from, date_to = self._split_range(info["date_range"])
        return {
            "source": source,
            "path": str(path),
            "exists": info["exists"],
            "available_from": date_from,
            "available_to": date_to,
            "usable_rows": info["rows"],
            "usable_codes": info["codes"],
            "missing_rate": info["missing_rate"],
            "days": None,
            "data_gaps": "not audited at row-level",
            "blocking_issues": "" if info["exists"] else "missing",
        }

    def _availability_from_dir(self, source: str, rel_path: Path) -> dict[str, Any]:
        path = self._root(rel_path)
        files = sorted(path.glob("*.parquet")) if path.exists() else []
        if not files:
            files = sorted(path.glob("*.csv")) if path.exists() else []
        dates = self._dates_from_filenames(files)
        sample = _read_parquet(files[0]) if files and files[0].suffix == ".parquet" else pd.DataFrame()
        return {
            "source": source,
            "path": str(path),
            "exists": path.exists() and bool(files),
            "available_from": dates[0] if dates else None,
            "available_to": dates[-1] if dates else None,
            "usable_rows": self._safe_row_count(files) if files and files[0].suffix == ".parquet" else len(files),
            "usable_codes": _count_codes(sample),
            "missing_rate": _missing_rate(sample),
            "days": len(set(dates)) if dates else None,
            "data_gaps": "not audited at row-level",
            "blocking_issues": "" if files else "missing_or_no_supported_files",
        }

    def _split_range(self, text: str) -> tuple[str | None, str | None]:
        if " to " not in text:
            return None, None
        start, end = text.split(" to ", 1)
        return start or None, end or None

    def _pm_dataset_audit(self, inventory: list[dict[str, Any]]) -> dict[str, Any]:
        pm = self._find(inventory, "Portfolio Manager AI")
        backtest_hits = pm.get("backtest_derived_columns", []) + pm.get("label_like_columns", [])
        return {
            "current_objective": pm.get("target_label"),
            "high_conviction_definition": "high_conviction probability from PM model",
            "avoid_proba_definition": "avoid probability from PM model",
            "pm_score_definition": "pm_score = high_conviction_proba - avoid_proba",
            "v2_82_pm_ai_effect": "PM-aware ordering, low-score skip, PM multiplier sizing, and cap38 interact in current best profile",
            "backtest_outcome_columns": sorted(set(backtest_hits)),
            "pm_dataset_retraining_safe": False,
            "pm_dataset_needs_rebuild": True,
            "api_only_rebuild_possible": True,
            "bear_alpha_cap38_as_teacher": "no; use Phase 6 findings as audit references only, not labels",
            "pm_retraining_priority": "medium_after_api_only_rebuild",
            "reason": "current PM dataset may include backtest-outcome/audit state; rebuild API-only labels before retraining",
        }

    def _stock_ai_audit(self, inventory: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
        stock = self._find(inventory, "Stock Selection AI")
        starts_2021 = bool(stock.get("dataset_date_range", "").startswith("2021") or data.get("available_from", "") <= "2021-06-01")
        return {
            "current_objective": stock.get("target_label"),
            "trained_on_2021_plus": starts_2021,
            "existing_dataset_api_only": stock.get("api_only"),
            "target_leakage_found": bool(stock.get("forbidden_feature_hits")),
            "feature_freshness": stock.get("dataset_date_range"),
            "possible_2023_weakness_link": "unknown; v2_82 improvement came from PM/cap layer, not proven stock selector weakness",
            "stock_ai_retraining_safe": bool(stock.get("retraining_allowed")),
            "stock_ai_retraining_priority": "low_to_medium",
            "reason": "stock selector has high blast radius and downstream PM/cap changes already improved results",
            "risk": "high",
        }

    def _exit_ai_audit(self, inventory: list[dict[str, Any]]) -> dict[str, Any]:
        current = self._find(inventory, "Exit AI current v2_66")
        v2 = self._find(inventory, "Exit AI v2 candidate")
        return {
            "exit_ai_current_state": "active_current_model; keep unchanged",
            "exit_ai_current_model_exists": current.get("model_exists"),
            "exit_ai_current_dataset_safe_for_retraining": current.get("retraining_allowed"),
            "exit_ai_v2_candidate_state": "trained_api_only_candidate; integration backtest underperformed",
            "exit_ai_v2_model_exists": v2.get("model_exists"),
            "phase5f_model_usable": bool(v2.get("model_exists") and v2.get("retraining_allowed")),
            "integration_failure_likely_cause": "integration/gating design more likely than raw model training safety",
            "exit_ai_retraining_priority": "medium_after_integration_redesign",
            "reason": "Exit AI v2 has safe dataset/model, but Phase 5-H profiles were worse than v2_78; redesign integration before more training",
        }

    def _leakage_audit(self, inventory: list[dict[str, Any]]) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        blocking: list[str] = []
        for row in inventory:
            name = row["model_name"]
            if row.get("forbidden_feature_hits"):
                checks.append({"check": f"{name} forbidden features", "status": "block", "detail": ", ".join(row["forbidden_feature_hits"])})
                blocking.append(f"{name}: forbidden feature columns detected")
            else:
                checks.append({"check": f"{name} forbidden features", "status": "pass", "detail": "none"})
            if row.get("backtest_derived_columns_present") and not row.get("api_only"):
                checks.append({"check": f"{name} direct retraining dataset", "status": "block_for_direct_retraining", "detail": "backtest/reference columns present or source is not API-only"})
            else:
                checks.append({"check": f"{name} direct retraining dataset", "status": "pass", "detail": "API-only or not a trained model"})
        generic = [
            ("future price leakage", "guard_required", "future returns are labels only; train/test must account for horizon cutoff"),
            ("same-day close leakage", "guard_required", "features must be known at prediction time"),
            ("selected_count_in_day", "pass", "explicitly forbidden in all future retraining plans"),
            ("backtest result leakage", "block_for_labels", "trades, realized profit, win/loss, and portfolio history cannot be teacher labels"),
            ("current model past prediction regeneration", "block", "use walk-forward artifacts only"),
            ("train/test period overlap", "guard_required", "use chronological walk-forward or strict 2021-2024/2025/2026 split"),
        ]
        checks.extend({"check": name, "status": status, "detail": detail} for name, status, detail in generic)
        return {
            "checks": checks,
            "blocking_issues": blocking,
            "safe_to_retrain_flags": {
                "stock_ai_direct_retrain_possible": self._find(inventory, "Stock Selection AI").get("retraining_allowed", False),
                "pm_ai_direct_retrain_possible": False,
                "exit_ai_current_direct_retrain_possible": False,
                "exit_ai_v2_direct_retrain_possible": self._find(inventory, "Exit AI v2 candidate").get("retraining_allowed", False),
                "api_only_dataset_rebuild_required_for_pm": True,
                "walk_forward_required": True,
            },
        }

    def _retraining_plans(
        self,
        pm: dict[str, Any],
        stock: dict[str, Any],
        exit_ai: dict[str, Any],
        leakage: dict[str, Any],
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        data_ok = bool(data.get("api_only_2021_retraining_data_possible"))
        return [
            {"plan": "Plan A: current models maintained", "expected_benefit": "low", "expected_risk": "low", "leakage_risk": "low", "implementation_cost": "low", "comparison_difficulty": "low", "recommended_order": 2},
            {"plan": "Plan B: Stock Selection AI only", "expected_benefit": "uncertain", "expected_risk": "high", "leakage_risk": "medium", "implementation_cost": "high", "comparison_difficulty": "high", "recommended_order": 5},
            {"plan": "Plan C: Exit AI v2 only retrain/improve", "expected_benefit": "medium", "expected_risk": "medium", "leakage_risk": "low" if exit_ai.get("phase5f_model_usable") else "medium", "implementation_cost": "medium", "comparison_difficulty": "medium", "recommended_order": 3},
            {"plan": "Plan D: Portfolio Manager AI only", "expected_benefit": "medium-high", "expected_risk": "medium-high", "leakage_risk": "high_until_api_only_rebuild", "implementation_cost": "medium-high", "comparison_difficulty": "high", "recommended_order": 4},
            {"plan": "Plan E: Stock Selection + PM AI", "expected_benefit": "high_uncertain", "expected_risk": "high", "leakage_risk": "high", "implementation_cost": "very_high", "comparison_difficulty": "very_high", "recommended_order": 6},
            {"plan": "Plan F: all AI retraining", "expected_benefit": "unknown", "expected_risk": "very_high", "leakage_risk": "high", "implementation_cost": "very_high", "comparison_difficulty": "very_high", "recommended_order": 7},
            {"plan": "Plan G: API-only dataset redesign first", "expected_benefit": "enables_safe_retraining", "expected_risk": "low", "leakage_risk": "low" if data_ok else "medium", "implementation_cost": "medium", "comparison_difficulty": "medium", "recommended_order": 1},
        ]

    def _safe_retraining_design(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "recommended_design": "walk-forward expanding window",
            "walk_forward_training": "train 2021 -> test 2022; train 2021-2022 -> test 2023; train 2021-2023 -> test 2024; train 2021-2024 -> test 2025; train 2021-2025 -> test 2026",
            "expanding_window": True,
            "static_split_viable": "secondary_baseline_only: train 2021-2024, validation 2025, test 2026",
            "horizon_cutoff": "drop rows whose t+horizon labels extend beyond available price data",
            "forbidden_designs": [
                "train on all 2021-2026 and evaluate 2022-2026",
                "use backtest outcomes as labels",
                "regenerate past predictions with current models",
                "include future_return labels as features",
            ],
            "reason": "walk-forward is fairest for yearly strategy comparison; static split is simpler but less representative",
        }

    def _final_verdict(
        self,
        inventory: list[dict[str, Any]],
        data: dict[str, Any],
        pm: dict[str, Any],
        stock: dict[str, Any],
        exit_ai: dict[str, Any],
        leakage: dict[str, Any],
        plans: list[dict[str, Any]],
    ) -> dict[str, Any]:
        blocking = list(data.get("blocking_issues", [])) + list(leakage.get("blocking_issues", []))
        order = [row["plan"] for row in sorted(plans, key=lambda row: row["recommended_order"])]
        return {
            "all_ai_retraining_ready": False,
            "recommended_first_retraining_target": "API-only dataset redesign, especially PM AI labels",
            "recommended_retraining_order": order,
            "retraining_should_start_now": False,
            "blocking_issues": blocking,
            "next_phase_recommended": "Phase 7-B PM AI API-only Dataset Rebuild",
            "keep_v282_during_retraining_design": True,
        }

    def _find(self, inventory: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
        return next((row for row in inventory if row["model_name"] == model_name), {})

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows_"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

