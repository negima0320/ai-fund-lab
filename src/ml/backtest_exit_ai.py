from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExitAISignal:
    enabled: bool
    triggered: bool
    probability: float | None
    threshold: float | None
    prediction_joined: bool
    reason: str | None = None
    warning: str | None = None


class ExitAIBacktestAdvisor:
    """Evaluate pre-trained Exit AI signals during backtests.

    This class never trains models and never regenerates buy-side predictions.
    It only reads the model files and the walk-forward prediction parquet for
    the current holding date.
    """

    def __init__(
        self,
        model_dir: str | Path,
        predictions_root: str | Path,
        threshold: float,
        bad_entry_weight: float = 0.5,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.predictions_root = Path(predictions_root)
        self.threshold = float(threshold)
        self.bad_entry_weight = float(bad_entry_weight)
        self.feature_columns = self._load_feature_columns()
        self.model = self._load_model(self.model_dir / "avoid_loss_5d_classification.joblib")
        self._prediction_cache: dict[str, pd.DataFrame | None] = {}

    def evaluate(
        self,
        *,
        position: dict[str, Any],
        market: dict[str, Any],
        trade_date: str,
        current_price: float,
        holding_days: int,
    ) -> ExitAISignal:
        code = str(position.get("code") or "")
        if not code:
            return ExitAISignal(False, False, None, self.threshold, False, warning="missing_code")
        prediction = self._prediction_for(code, trade_date)
        features = self._build_features(position, market, prediction, current_price, holding_days)
        frame = pd.DataFrame([{column: features.get(column) for column in self.feature_columns}])
        for column in self.feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        probability = self._predict_probability(frame)
        triggered = probability is not None and probability >= self.threshold
        return ExitAISignal(
            enabled=True,
            triggered=triggered,
            probability=probability,
            threshold=self.threshold,
            prediction_joined=bool(prediction),
            reason="Exit AI avoid_loss_5d" if triggered else None,
        )

    def _load_feature_columns(self) -> list[str]:
        path = self.model_dir / "feature_columns.json"
        return list(json.loads(path.read_text(encoding="utf-8")))

    def _load_model(self, path: Path) -> Any:
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("joblib is required to load Exit AI models. Install requirements.txt first.") from exc
        return joblib.load(path)

    def _prediction_for(self, code: str, date_text: str) -> dict[str, float]:
        if date_text not in self._prediction_cache:
            path = self.predictions_root / f"predictions_{date_text}.parquet"
            self._prediction_cache[date_text] = pd.read_parquet(path) if path.exists() else None
        df = self._prediction_cache[date_text]
        if df is None or df.empty or "code" not in df.columns:
            return {}
        if not pd.api.types.is_string_dtype(df["code"]):
            df = df.copy()
            df["code"] = df["code"].astype(str)
            self._prediction_cache[date_text] = df
        matched = df[df["code"].eq(str(code))]
        if matched.empty:
            return {}
        row = matched.iloc[0]
        columns = [
            "expected_return_10d",
            "expected_max_return_20d",
            "swing_success_probability_20d",
            "bad_entry_probability_10d",
        ]
        return {column: self._to_float(row.get(column), None) for column in columns}

    def _build_features(
        self,
        position: dict[str, Any],
        market: dict[str, Any],
        prediction: dict[str, float],
        current_price: float,
        holding_days: int,
    ) -> dict[str, float | None]:
        entry_price = self._to_float(position.get("entry_price"), None)
        unrealized_return = current_price / entry_price - 1 if entry_price else None
        max_so_far, min_so_far = update_unrealized_extrema(position, unrealized_return)
        risk_adjusted_score = None
        expected_return = prediction.get("expected_return_10d")
        bad_entry = prediction.get("bad_entry_probability_10d")
        if expected_return is not None and bad_entry is not None:
            risk_adjusted_score = expected_return - self.bad_entry_weight * bad_entry
        ma25 = self._to_float(market.get("ma25"), None)
        ma25_gap = current_price / ma25 - 1 if ma25 else self._to_float(market.get("ma25_gap"), None)
        high = self._to_float(market.get("high"), None)
        low = self._to_float(market.get("low"), None)
        daily_range_ratio = (high - low) / current_price if high is not None and low is not None and current_price else None
        return {
            "holding_days": float(holding_days),
            "entry_price": entry_price,
            "current_close": current_price,
            "unrealized_return": unrealized_return,
            "max_unrealized_return_so_far": max_so_far,
            "min_unrealized_return_so_far": min_so_far,
            "drawdown_from_peak": unrealized_return - max_so_far if unrealized_return is not None and max_so_far is not None else None,
            "expected_return_10d": expected_return,
            "expected_max_return_20d": prediction.get("expected_max_return_20d"),
            "swing_success_probability_20d": prediction.get("swing_success_probability_20d"),
            "bad_entry_probability_10d": bad_entry,
            "risk_adjusted_score": risk_adjusted_score,
            "volume": self._to_float(market.get("volume"), None),
            "turnover_value": self._to_float(market.get("turnover_value"), None),
            "return_5d": self._coalesce(market.get("return_5d"), market.get("stock_return_5d"), position.get("stock_return_5d")),
            "return_10d": self._coalesce(market.get("return_10d"), market.get("stock_return_10d"), position.get("stock_return_10d")),
            "ma25_gap": ma25_gap,
            "daily_range_ratio": daily_range_ratio,
        }

    def _predict_probability(self, features: pd.DataFrame) -> float | None:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)
            if len(probabilities) == 0:
                return None
            return float(probabilities[0][1])
        predictions = self.model.predict(features)
        if len(predictions) == 0:
            return None
        return float(predictions[0])

    def _coalesce(self, *values: Any) -> float | None:
        for value in values:
            parsed = self._to_float(value, None)
            if parsed is not None:
                return parsed
        return None

    def _to_float(self, value: Any, default: float | None = 0.0) -> float | None:
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default


def ml_exit_ai_enabled(config: dict[str, Any]) -> bool:
    cfg = config.get("ml_exit_ai") or {}
    return isinstance(cfg, dict) and bool(cfg.get("enabled", False))


def get_exit_ai_advisor(config: dict[str, Any], root: str | Path = ".") -> ExitAIBacktestAdvisor | None:
    if not ml_exit_ai_enabled(config):
        return None
    cfg = config.get("ml_exit_ai") or {}
    root_path = Path(root)
    model_dir = _resolve_path(root_path, cfg.get("model_dir") or "models/ml/exit/current_v2_66")
    predictions_root = _resolve_path(root_path, cfg.get("prediction_root") or "data/ml/walk_forward_predictions")
    threshold = float(cfg.get("threshold", 0.5))
    bad_entry_weight = float(cfg.get("risk_adjusted_bad_entry_weight", 0.5))
    return ExitAIBacktestAdvisor(model_dir, predictions_root, threshold, bad_entry_weight)


def apply_exit_ai_to_plan(
    exit_plan: dict[str, Any],
    advisor: ExitAIBacktestAdvisor | None,
    *,
    position: dict[str, Any],
    market: dict[str, Any],
    trade_date: str,
    current_price: float,
    holding_days: int,
) -> dict[str, Any]:
    if advisor is None or holding_days < 2:
        return exit_plan
    signal = advisor.evaluate(
        position=position,
        market=market,
        trade_date=trade_date,
        current_price=current_price,
        holding_days=holding_days,
    )
    updated = dict(exit_plan)
    updated.update(
        {
            "exit_ai_enabled": True,
            "exit_ai_probability": signal.probability,
            "exit_ai_threshold": signal.threshold,
            "exit_ai_prediction_joined": signal.prediction_joined,
            "exit_ai_signal": signal.triggered,
        }
    )
    if signal.warning:
        updated["exit_ai_warning"] = signal.warning
    if signal.triggered and not updated.get("exit_reason"):
        updated.update(
            {
                "exit_reason": signal.reason or "Exit AI avoid_loss_5d",
                "exit_price": current_price,
                "intended_exit_price": current_price,
                "execute_now": False,
                "exit_ai_triggered": True,
            }
        )
    else:
        updated["exit_ai_triggered"] = False
    return updated


def update_unrealized_extrema(position: dict[str, Any], unrealized_return: float | None) -> tuple[float | None, float | None]:
    previous_max = _optional_float(position.get("max_unrealized_return_so_far"))
    previous_min = _optional_float(position.get("min_unrealized_return_so_far"))
    if unrealized_return is None:
        return previous_max, previous_min
    max_so_far = unrealized_return if previous_max is None else max(previous_max, unrealized_return)
    min_so_far = unrealized_return if previous_min is None else min(previous_min, unrealized_return)
    return max_so_far, min_so_far


def exit_ai_trade_fields(exit_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "exit_ai_enabled": exit_plan.get("exit_ai_enabled"),
        "exit_ai_signal": exit_plan.get("exit_ai_signal"),
        "exit_ai_triggered": exit_plan.get("exit_ai_triggered"),
        "exit_ai_probability": exit_plan.get("exit_ai_probability"),
        "exit_ai_threshold": exit_plan.get("exit_ai_threshold"),
        "exit_ai_prediction_joined": exit_plan.get("exit_ai_prediction_joined"),
        "exit_ai_warning": exit_plan.get("exit_ai_warning"),
    }


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
