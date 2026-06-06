from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_dataset import CLEAN_FORBIDDEN_FEATURE_COLUMNS, LABEL_COLUMNS


DEFAULT_PM_MODEL_DIR = Path("models/ml/portfolio_manager/current_v2_73_phase3b_clean")
DEFAULT_PM_DATASET = Path("data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet")
EXPECTED_CLEAN_FEATURE_COUNT = 68
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

    def as_fields(self) -> dict[str, Any]:
        status = "ok"
        missing_reason = ""
        if self.warning:
            missing_reason = self.warning
            status = "missing" if not self.feature_found else "warning"
        return {
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
        }


class PortfolioManagerSizingAdvisor:
    """PM AI sizing lookup backed by the audited clean feature store and clean models."""

    def __init__(
        self,
        root: str | Path = ".",
        model_dir: str | Path = DEFAULT_PM_MODEL_DIR,
        dataset_path: str | Path = DEFAULT_PM_DATASET,
        expected_feature_count: int = EXPECTED_CLEAN_FEATURE_COUNT,
    ) -> None:
        self.root = Path(root)
        self.model_dir = self._resolve(model_dir)
        self.dataset_path = self._resolve(dataset_path)
        self.expected_feature_count = int(expected_feature_count)
        self.feature_columns = self._load_feature_columns()
        self._assert_feature_columns()
        self.metadata = self._load_metadata()
        self.high_model = self._load_model("high_conviction_target_classification.joblib")
        self.avoid_model = self._load_model("avoid_target_classification.joblib")
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
            )
        frame = pd.DataFrame([row], columns=self.feature_columns)
        for column in self.feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        high = self._positive_probability(self.high_model, frame)
        avoid = self._positive_probability(self.avoid_model, frame)
        score = high - avoid
        multiplier = multiplier_from_high_minus_avoid(high, avoid)
        return PortfolioManagerSizingDecision(
            high_conviction_proba=high,
            avoid_proba=avoid,
            score=score,
            multiplier=multiplier,
            feature_count=len(self.feature_columns),
            model_version=self.model_version,
            feature_found=True,
        )

    @property
    def model_version(self) -> str:
        value = self.metadata.get("model_profile") or self.metadata.get("created_at") or self.model_dir.name
        return str(value)

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

    def _load_model(self, filename: str) -> Any:
        import joblib

        return joblib.load(self.model_dir / filename)

    def _load_feature_store(self) -> dict[tuple[str, str], dict[str, Any]]:
        df = pd.read_parquet(self.dataset_path, columns=["signal_date", "code", *self.feature_columns])
        df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["code"] = df["code"].astype(str)
        df = df.dropna(subset=["signal_date", "code"]).drop_duplicates(["signal_date", "code"], keep="last")
        return {
            (str(row["signal_date"]), str(row["code"])): {column: row.get(column) for column in self.feature_columns}
            for row in df.to_dict(orient="records")
        }

    def _positive_probability(self, model: Any, frame: pd.DataFrame) -> float:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(frame)
            return float(probabilities[0][1])
        return float(model.predict(frame)[0])


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
