from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_dataset import CLEAN_FORBIDDEN_FEATURE_COLUMNS, LABEL_COLUMNS


DEFAULT_PM_MODEL_DIR = Path("models/ml/portfolio_manager/current_v2_73_phase3b_clean")
DEFAULT_PM_DATASET = Path("data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet")
DEFAULT_PM_V3_MODEL_DIR = Path("models/ml/portfolio_manager_v3/candidate_phase9d")
DEFAULT_PM_V3_DATASET = Path("data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_2023-01_to_2026-05.parquet")
DEFAULT_PM_V3_PM_SIZING_UNIVERSE_MODEL_DIR = Path("models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe")
DEFAULT_PM_V3_PM_SIZING_UNIVERSE_DATASET = Path("data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_pm_sizing_universe_2023-01_to_2026-05.parquet")
EXPECTED_CLEAN_FEATURE_COUNT = 68
EXPECTED_PM_V3_FEATURE_COUNT = 50
FORBIDDEN_FEATURE_FRAGMENTS = (
    "actual",
    "decision",
    "skip",
    "exit",
    "backtest",
    "audit",
    "profit",
    "return_label",
    "cash_",
    "final_",
)


@dataclass(frozen=True)
class PortfolioManagerSizingDecision:
    high_conviction_proba: float | None
    avoid_proba: float | None
    score: float
    multiplier: float
    feature_count: int
    model_version: str
    feature_found: bool
    warning: str = ""
    model_path: str = ""
    api_only_candidate_enabled: bool = False
    feature_missing_count: int | None = None
    calibration_rule: str = ""
    calibration_thresholds: dict[str, Any] | None = None
    raw_multiplier: float | None = None

    def as_fields(self) -> dict[str, Any]:
        status = "ok"
        missing_reason = ""
        if self.warning:
            missing_reason = self.warning
            status = "missing" if not self.feature_found else "warning"
        fields = {
            "pm_ai_enabled": True,
            "pm_status": status,
            "pm_missing_reason": missing_reason,
            "pm_feature_count": self.feature_count,
            "pm_high_conviction_proba": self.high_conviction_proba,
            "pm_avoid_proba": self.avoid_proba,
            "pm_score": self.score,
            "pm_multiplier": self.multiplier,
            "pm_model_version": self.model_version,
            "pm_feature_found": self.feature_found,
            "pm_warning": self.warning,
            "pm_model_path": self.model_path,
            "pm_api_only_candidate_enabled": self.api_only_candidate_enabled,
            "pm_calibration_rule": self.calibration_rule,
            "pm_calibration_thresholds": json.dumps(self.calibration_thresholds or {}, sort_keys=True),
        }
        if self.api_only_candidate_enabled:
            fields.update(
                {
                    "pm_candidate_high_conviction_proba": self.high_conviction_proba,
                    "pm_candidate_avoid_proba": self.avoid_proba,
                    "pm_candidate_score": self.score,
                    "pm_candidate_multiplier": self.multiplier,
                    "pm_candidate_multiplier_raw": self.raw_multiplier if self.raw_multiplier is not None else self.multiplier,
                    "pm_candidate_multiplier_calibrated": self.multiplier,
                    "pm_candidate_feature_missing_count": self.feature_missing_count,
                    "pm_candidate_prediction_available": self.feature_found,
                    "pm_candidate_fallback_reason": self.warning,
                }
            )
        return fields


class PortfolioManagerSizingAdvisor:
    """PM AI sizing lookup backed by the audited clean feature store and clean models."""

    def __init__(
        self,
        root: str | Path = ".",
        model_dir: str | Path = DEFAULT_PM_MODEL_DIR,
        dataset_path: str | Path = DEFAULT_PM_DATASET,
        expected_feature_count: int = EXPECTED_CLEAN_FEATURE_COUNT,
        calibration_rule: str = "",
        calibration_thresholds: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.model_dir = self._resolve(model_dir)
        self.dataset_path = self._resolve(dataset_path)
        self.expected_feature_count = int(expected_feature_count)
        self.calibration_rule = str(calibration_rule or "")
        self.calibration_thresholds = calibration_thresholds or {}
        self.feature_columns = self._load_feature_columns()
        self._assert_feature_columns()
        self.metadata = self._load_metadata()
        self.preprocess = self._load_preprocess()
        self.high_model = self._load_model(
            "high_conviction_target_classification.joblib",
            "high_conviction_target_classifier.joblib",
        )
        self.avoid_model = self._load_model(
            "avoid_target_classification.joblib",
            "avoid_target_classifier.joblib",
        )
        self.features = self._load_feature_store()

    def decision_for(self, signal_date: str, code: str) -> PortfolioManagerSizingDecision:
        key = (pd.Timestamp(signal_date).strftime("%Y-%m-%d"), str(code))
        row = self.features.get(key)
        if row is None:
            return PortfolioManagerSizingDecision(
                high_conviction_proba=None,
                avoid_proba=None,
                score=0.0,
                multiplier=1.0,
                feature_count=len(self.feature_columns),
                model_version=self.model_version,
                feature_found=False,
                warning="pm_feature_row_missing",
                model_path=str(self.model_dir),
                api_only_candidate_enabled=self.api_only_candidate_enabled,
                feature_missing_count=None,
                calibration_rule=self.calibration_rule,
                calibration_thresholds=self.calibration_thresholds,
                raw_multiplier=1.0,
            )
        feature_missing_count = sum(1 for column in self.feature_columns if pd.isna(row.get(column)))
        frame = self._transform_features(pd.DataFrame([row], columns=self.feature_columns), self.high_model)
        high = self._positive_probability(self.high_model, frame)
        avoid = self._positive_probability(self.avoid_model, frame)
        score = high - avoid
        raw_multiplier = multiplier_from_high_minus_avoid(high, avoid)
        multiplier = self._calibrated_multiplier(score, raw_multiplier)
        return PortfolioManagerSizingDecision(
            high_conviction_proba=high,
            avoid_proba=avoid,
            score=score,
            multiplier=multiplier,
            feature_count=len(self.feature_columns),
            model_version=self.model_version,
            feature_found=True,
            model_path=str(self.model_dir),
            api_only_candidate_enabled=self.api_only_candidate_enabled,
            feature_missing_count=feature_missing_count,
            calibration_rule=self.calibration_rule,
            calibration_thresholds=self.calibration_thresholds,
            raw_multiplier=raw_multiplier,
        )

    @property
    def model_version(self) -> str:
        value = self.metadata.get("model_profile") or self.metadata.get("created_at") or self.model_dir.name
        return str(value)

    @property
    def api_only_candidate_enabled(self) -> bool:
        profile = str(self.metadata.get("model_profile") or "")
        return "api_only" in profile or "candidate_v2_api_only" in str(self.model_dir)

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def _load_feature_columns(self) -> list[str]:
        path = self.model_dir / "feature_columns.json"
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        return [str(column) for column in payload]

    def _assert_feature_columns(self) -> None:
        if len(self.feature_columns) != self.expected_feature_count:
            raise ValueError(
                f"Portfolio Manager feature_count mismatch: expected {self.expected_feature_count}, "
                f"got {len(self.feature_columns)}"
            )
        if "selected_count_in_day" in self.feature_columns:
            raise ValueError("selected_count_in_day is forbidden for Portfolio Manager clean inference")
        forbidden = set(CLEAN_FORBIDDEN_FEATURE_COLUMNS + LABEL_COLUMNS)
        leaked = sorted(column for column in self.feature_columns if column in forbidden or _looks_forbidden(column))
        if leaked:
            raise ValueError(f"Forbidden Portfolio Manager feature columns: {', '.join(leaked)}")

    def _load_metadata(self) -> dict[str, Any]:
        path = self.model_dir / "model_metadata.json"
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}

    def _load_preprocess(self) -> dict[str, Any]:
        path = self.model_dir / "preprocess.json"
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}

    def _load_model(self, *filenames: str) -> Any:
        import joblib

        for filename in filenames:
            path = self.model_dir / filename
            if path.exists():
                return joblib.load(path)
        raise FileNotFoundError(f"Portfolio Manager model file not found in {self.model_dir}: {filenames}")

    def _load_feature_store(self) -> dict[tuple[str, str], dict[str, Any]]:
        date_column = self._feature_store_date_column()
        df = pd.read_parquet(self.dataset_path, columns=[date_column, "code", *self.feature_columns])
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
        df["code"] = df["code"].astype(str)
        df = df.dropna(subset=[date_column, "code"]).drop_duplicates([date_column, "code"], keep="last")
        return {
            (str(row[date_column]), str(row["code"])): {column: row.get(column) for column in self.feature_columns}
            for row in df.to_dict(orient="records")
        }

    def _feature_store_date_column(self) -> str:
        try:
            import pyarrow.parquet as pq

            names = set(pq.read_schema(self.dataset_path).names)
            if "signal_date" in names:
                return "signal_date"
            if "as_of_date" in names:
                return "as_of_date"
        except Exception:
            pass
        return "signal_date"

    def _transform_features(self, frame: pd.DataFrame, model: Any | None = None) -> pd.DataFrame:
        out = frame.copy()
        medians = self.preprocess.get("medians", {}) if isinstance(self.preprocess.get("medians"), dict) else {}
        missing_indicators = self.preprocess.get("missing_indicator_columns", [])
        missing_indicators = [str(column) for column in missing_indicators] if isinstance(missing_indicators, list) else []
        for column in self.feature_columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        for column in missing_indicators:
            if column in out.columns:
                out[f"{column}_missing"] = out[column].isna().astype(int)
        for column in self.feature_columns:
            if column in medians:
                out[column] = out[column].fillna(float(medians[column]))
        ordered_columns = list(self.feature_columns) + [
            f"{column}_missing" for column in missing_indicators if f"{column}_missing" in out.columns
        ]
        model_columns = [str(column) for column in getattr(model, "feature_names_in_", [])]
        if model_columns:
            for column in model_columns:
                if column not in out.columns:
                    out[column] = 0
            ordered_columns = model_columns
        return out[ordered_columns]

    def _positive_probability(self, model: Any, frame: pd.DataFrame) -> float:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(frame)
            return float(probabilities[0][1])
        return float(model.predict(frame)[0])

    def _calibrated_multiplier(self, score: float, raw_multiplier: float) -> float:
        rule = self.calibration_rule.strip().lower()
        if rule not in {"rule_e", "rule e", "quantile_match_current_pm_distribution"}:
            return raw_multiplier
        thresholds = {
            "pm130_score_min": -0.12284356890271281,
            "pm115_score_min": -0.1443989258328934,
            "pm080_score_max": -0.2072886007777547,
            **self.calibration_thresholds,
        }
        if score >= float(thresholds["pm130_score_min"]):
            return 1.30
        if score >= float(thresholds["pm115_score_min"]):
            return 1.15
        if score <= float(thresholds["pm080_score_max"]):
            return 0.80
        return 1.00


class PortfolioManagerV3SizingAdvisor:
    """Research-only PM AI v3 sizing lookup for Phase 9-F candidate backtests."""

    def __init__(
        self,
        root: str | Path = ".",
        model_dir: str | Path = DEFAULT_PM_V3_MODEL_DIR,
        dataset_path: str | Path = DEFAULT_PM_V3_DATASET,
        expected_feature_count: int = EXPECTED_PM_V3_FEATURE_COUNT,
        mapping_name: str = "mapping_a_rank_score_only",
    ) -> None:
        self.root = Path(root)
        self.model_dir = self._resolve(model_dir)
        self.dataset_path = self._resolve(dataset_path)
        self.expected_feature_count = int(expected_feature_count)
        self.mapping_name = str(mapping_name or "mapping_a_rank_score_only")
        self.feature_columns = self._load_feature_columns()
        self._assert_feature_columns()
        self.rank_model = self._load_model("model_a_candidate_ranking_regressor.joblib")
        self.downside_model = self._load_optional_model("model_b_downside_utility_regressor.joblib")
        self.top_model = self._load_optional_model("model_c_top_utility_classifier.joblib")
        self.decisions = self._load_decision_store()

    def decision_for(self, signal_date: str, code: str) -> PortfolioManagerSizingDecision:
        key = (pd.Timestamp(signal_date).strftime("%Y-%m-%d"), self._normalize_code(code))
        row = self.decisions.get(key)
        if row is None:
            return PortfolioManagerSizingDecision(
                high_conviction_proba=None,
                avoid_proba=None,
                score=0.0,
                multiplier=1.0,
                feature_count=len(self.feature_columns),
                model_version=self.model_version,
                feature_found=False,
                warning="pm_v3_feature_row_missing",
                model_path=str(self.model_dir),
                calibration_rule=self.mapping_name,
                calibration_thresholds=self.mapping_thresholds,
                raw_multiplier=1.0,
            )
        return PortfolioManagerSizingDecision(
            high_conviction_proba=row.get("pm_v3_top_utility_proba"),
            avoid_proba=None,
            score=float(row.get("pm_v3_rank_score_pred") or 0.0),
            multiplier=float(row.get("pm_multiplier") or 1.0),
            feature_count=len(self.feature_columns),
            model_version=self.model_version,
            feature_found=True,
            model_path=str(self.model_dir),
            calibration_rule=self.mapping_name,
            calibration_thresholds=self.mapping_thresholds,
            raw_multiplier=float(row.get("pm_multiplier") or 1.0),
        )

    @property
    def model_version(self) -> str:
        phase = "phase9d2_pm_sizing_universe" if "phase9d2" in str(self.model_dir) else "phase9d"
        return f"pm_ai_v3_{phase}_{self.mapping_name}"

    @property
    def mapping_thresholds(self) -> dict[str, float]:
        if self.mapping_name == "e_139_classifier_gate_recommended":
            return {"classifier_gate_threshold": 0.80, "rank_threshold": 0.75, "downside_threshold": 0.80}
        if self.mapping_name == "e_140_classifier_gate_more_pm130":
            return {"classifier_gate_threshold": 0.80, "rank_threshold": 0.75, "downside_threshold": 0.75}
        if self.mapping_name == "e_120_classifier_gate_wider":
            return {"classifier_gate_threshold": 0.75, "rank_threshold": 0.75, "downside_threshold": 0.75}
        if self.mapping_name == "mapping_a_rank_score_only":
            return {"pm130_rank_pct_min": 0.90, "pm115_rank_pct_min": 0.75, "pm080_rank_pct_max": 0.25, "pm060_rank_pct_max": 0.10}
        if self.mapping_name == "mapping_c_rank_plus_downside_blend":
            return {"pm130_blend_pct_min": 0.90, "pm115_blend_pct_min": 0.75, "pm080_blend_pct_max": 0.25, "pm060_blend_pct_max": 0.10}
        if self.mapping_name == "mapping_a_conservative_pm130_threshold":
            return {"pm130_rank_pct_min": 0.90, "pm130_downside_pct_min": 0.50, "pm115_rank_pct_min": 0.75, "pm080_rank_pct_max": 0.25, "pm060_rank_pct_max": 0.10}
        if self.mapping_name == "mapping_a_half_pm130_candidates":
            return {"pm130_rank_pct_min": 0.95, "pm115_rank_pct_min": 0.75, "pm080_rank_pct_max": 0.25, "pm060_rank_pct_max": 0.10}
        return {"pm130_rank_pct_min": 0.90, "pm115_rank_pct_min": 0.75, "pm080_rank_pct_max": 0.25, "pm060_rank_pct_max": 0.10}

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def _load_feature_columns(self) -> list[str]:
        path = self.model_dir / "feature_columns.json"
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        return [str(column) for column in payload]

    def _assert_feature_columns(self) -> None:
        if len(self.feature_columns) != self.expected_feature_count:
            raise ValueError(
                f"Portfolio Manager v3 feature_count mismatch: expected {self.expected_feature_count}, "
                f"got {len(self.feature_columns)}"
            )
        leaked = sorted(column for column in self.feature_columns if _looks_forbidden(column) or _looks_like_label(column))
        if leaked:
            raise ValueError(f"Forbidden Portfolio Manager v3 feature columns: {', '.join(leaked)}")

    def _load_model(self, filename: str) -> Any:
        import joblib

        return joblib.load(self.model_dir / filename)

    def _load_optional_model(self, filename: str) -> Any | None:
        path = self.model_dir / filename
        if not path.exists():
            return None
        import joblib

        return joblib.load(path)

    def _load_decision_store(self) -> dict[tuple[str, str], dict[str, Any]]:
        columns = ["prediction_date", "code", *self.feature_columns]
        df = pd.read_parquet(self.dataset_path, columns=columns)
        df["prediction_date"] = pd.to_datetime(df["prediction_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["code"] = df["code"].map(self._normalize_code)
        df = df.dropna(subset=["prediction_date", "code"]).drop_duplicates(["prediction_date", "code"], keep="last")
        if df.empty:
            return {}
        x = df[self.feature_columns]
        df["pm_v3_rank_score_pred"] = self.rank_model.predict(x)
        if self.downside_model is not None:
            df["pm_v3_downside_utility_pred"] = self.downside_model.predict(x)
            downside_pct = pd.to_numeric(df["pm_v3_downside_utility_pred"], errors="coerce").rank(method="first", pct=True)
        else:
            df["pm_v3_downside_utility_pred"] = pd.NA
            downside_pct = pd.Series(1.0, index=df.index)
        if self.top_model is not None and hasattr(self.top_model, "predict_proba"):
            df["pm_v3_top_utility_proba"] = self.top_model.predict_proba(x)[:, 1]
        elif self.top_model is not None:
            df["pm_v3_top_utility_proba"] = self.top_model.predict(x)
        else:
            df["pm_v3_top_utility_proba"] = pd.NA
        rank_pct = pd.to_numeric(df["pm_v3_rank_score_pred"], errors="coerce").rank(method="first", pct=True)
        top_pct = pd.to_numeric(df["pm_v3_top_utility_proba"], errors="coerce").rank(method="first", pct=True)
        blend_pct = 0.5 * rank_pct + 0.5 * downside_pct
        df["pm_v3_score_blend"] = 0.5 * pd.to_numeric(df["pm_v3_rank_score_pred"], errors="coerce") + 0.5 * pd.to_numeric(df["pm_v3_downside_utility_pred"], errors="coerce")
        df["pm_multiplier"] = self._map_multiplier(rank_pct, downside_pct, top_pct, blend_pct)
        return {
            (str(row["prediction_date"]), str(row["code"])): {
                "pm_v3_rank_score_pred": row.get("pm_v3_rank_score_pred"),
                "pm_v3_downside_utility_pred": row.get("pm_v3_downside_utility_pred"),
                "pm_v3_top_utility_proba": row.get("pm_v3_top_utility_proba"),
                "pm_v3_score_blend": row.get("pm_v3_score_blend"),
                "pm_multiplier": row.get("pm_multiplier"),
            }
            for row in df.to_dict(orient="records")
        }

    def _map_multiplier(self, rank_pct: pd.Series, downside_pct: pd.Series, top_pct: pd.Series, blend_pct: pd.Series) -> pd.Series:
        if self.mapping_name in {"e_139_classifier_gate_recommended", "e_140_classifier_gate_more_pm130", "e_120_classifier_gate_wider"}:
            thresholds = self.mapping_thresholds
            return self._map_classifier_gate(
                rank_pct,
                downside_pct,
                top_pct,
                blend_pct,
                gate=float(thresholds["classifier_gate_threshold"]),
                rank_t=float(thresholds["rank_threshold"]),
                downside_t=float(thresholds["downside_threshold"]),
            )
        if self.mapping_name == "mapping_c_rank_plus_downside_blend":
            return self._quantile_map(blend_pct)
        if self.mapping_name == "mapping_a_rank_score_only":
            return self._quantile_map(rank_pct)
        out = pd.Series(1.00, index=rank_pct.index)
        out.loc[rank_pct >= 0.75] = 1.15
        if self.mapping_name == "mapping_a_half_pm130_candidates":
            out.loc[rank_pct >= 0.95] = 1.30
        elif self.mapping_name == "mapping_a_conservative_pm130_threshold":
            out.loc[(rank_pct >= 0.90) & (downside_pct >= 0.50)] = 1.30
        else:
            out.loc[rank_pct >= 0.90] = 1.30
        out.loc[rank_pct <= 0.25] = 0.80
        out.loc[rank_pct <= 0.10] = 0.60
        return out

    def _quantile_map(self, pct: pd.Series) -> pd.Series:
        out = pd.Series(1.00, index=pct.index)
        out.loc[pct >= 0.90] = 1.30
        out.loc[(pct >= 0.75) & (pct < 0.90)] = 1.15
        out.loc[pct <= 0.25] = 0.80
        out.loc[pct <= 0.10] = 0.60
        return out

    def _map_classifier_gate(self, rank_pct: pd.Series, downside_pct: pd.Series, top_pct: pd.Series, blend_pct: pd.Series, *, gate: float, rank_t: float, downside_t: float) -> pd.Series:
        out = pd.Series(1.00, index=rank_pct.index)
        out.loc[(blend_pct >= 0.75) | (rank_pct >= rank_t)] = 1.15
        out.loc[(rank_pct >= rank_t) & (downside_pct >= downside_t) & (top_pct >= gate)] = 1.30
        out.loc[blend_pct <= 0.25] = 0.80
        out.loc[(blend_pct <= 0.10) | (top_pct <= 0.10)] = 0.60
        return out

    def _normalize_code(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits.zfill(4) if digits else text


def multiplier_from_high_minus_avoid(high_proba: float | None, avoid_proba: float | None) -> float:
    high = 0.5 if high_proba is None or pd.isna(high_proba) else float(high_proba)
    avoid = 0.5 if avoid_proba is None or pd.isna(avoid_proba) else float(avoid_proba)
    score = high - avoid
    if score >= 0.40:
        return 1.30
    if score >= 0.20:
        return 1.15
    if score >= 0.00:
        return 1.00
    if score >= -0.20:
        return 0.80
    return 0.60


def _looks_forbidden(column: str) -> bool:
    lowered = column.lower()
    return any(fragment in lowered for fragment in FORBIDDEN_FEATURE_FRAGMENTS)


def _looks_like_label(column: str) -> bool:
    lowered = column.lower()
    return lowered.startswith("future_") or "label" in lowered or "target" in lowered
