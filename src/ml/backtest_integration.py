from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import ML_DATA_ROOT
from ml.data_loader import JQuantsDataLoader


ML_BACKTEST_COLUMNS = [
    "expected_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "bad_entry_probability_10d",
    "risk_adjusted_score",
]


@dataclass(frozen=True)
class MLPredictionLoadResult:
    date: str
    path: Path
    exists: bool
    frame: pd.DataFrame
    source: str


class MLBacktestPredictionLoader:
    """Read one-day ML predictions for backtest-time report-only decisions."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else ML_DATA_ROOT / "walk_forward_predictions"

    def load_for_date(self, target_date: str) -> MLPredictionLoadResult:
        path = self.root / f"predictions_{target_date}.parquet"
        if not path.exists():
            return MLPredictionLoadResult(target_date, path, False, pd.DataFrame(), self.root.name)
        frame = pd.read_parquet(path)
        frame = _normalize_prediction_frame(frame)
        return MLPredictionLoadResult(target_date, path, True, frame, self.root.name)


def ml_backtest_enabled(config: dict[str, Any]) -> bool:
    ml_config = config.get("ml_backtest", {})
    return isinstance(ml_config, dict) and bool(ml_config.get("enabled", False))


def ml_backtest_mode(config: dict[str, Any]) -> str:
    ml_config = config.get("ml_backtest", {})
    if not isinstance(ml_config, dict) or not ml_config.get("enabled", False):
        return "disabled"
    return str(ml_config.get("mode") or "ranked")


def apply_ml_backtest_overlay(
    scoring_log: dict[str, Any],
    target_date: str,
    config: dict[str, Any],
    root: str | Path,
) -> dict[str, Any]:
    """Apply a profile-specific ML overlay to an existing scoring log."""

    if not ml_backtest_enabled(config):
        return scoring_log
    mode = ml_backtest_mode(config)
    if mode == "standalone":
        return build_ml_standalone_scoring_log(target_date, config, root)
    if mode == "ranked":
        return apply_ml_ranked_overlay(scoring_log, target_date, config, root)
    return scoring_log


def apply_ml_ranked_overlay(
    scoring_log: dict[str, Any],
    target_date: str,
    config: dict[str, Any],
    root: str | Path,
) -> dict[str, Any]:
    ml_config = _ml_config(config)
    load_result = _prediction_loader(ml_config, root).load_for_date(target_date)
    scores, audit = _attach_predictions(scoring_log.get("scores", []), load_result, ml_config)
    max_selected = int(config.get("selection", {}).get("max_selected", 10) or 10)

    eligible = [row for row in scores if row.get("ml_ranked_original_selected")]
    selected_codes = {
        str(row.get("code"))
        for row in _sort_ml_ranked_rows(eligible)[:max_selected]
        if row.get("code")
    }
    for row in scores:
        code = str(row.get("code") or "")
        row["selected"] = code in selected_codes
        if row["selected"]:
            row["selected_reason"] = _append_reason(row.get("selected_reason"), "ml_ranked")
        elif row.get("ml_ranked_original_selected"):
            row["rejected_reason"] = _append_reason(row.get("rejected_reason"), "ml_ranked_lower_priority")

    ordered_scores = _sort_ml_ranked_rows(scores)
    for rank, row in enumerate(ordered_scores, start=1):
        row["rank"] = rank
        row["daily_score_rank"] = rank

    selected = [row for row in ordered_scores if row.get("selected")]
    result = {
        **scoring_log,
        "scores": ordered_scores,
        "selected": selected,
        "selected_count": len(selected),
        "ml_backtest": {
            **audit,
            "enabled": True,
            "mode": "ranked",
            "ranking": "risk_adjusted_score",
            "missing_prediction_policy": "lower_rank",
            "note": "Existing v2_65 candidates are preserved; ML only changes priority for this profile.",
        },
    }
    return result


def build_ml_standalone_scoring_log(
    target_date: str,
    config: dict[str, Any],
    root: str | Path,
) -> dict[str, Any]:
    ml_config = _ml_config(config)
    load_result = _prediction_loader(ml_config, root).load_for_date(target_date)
    audit: dict[str, Any] = _base_audit(load_result, ml_config)
    if not load_result.exists or load_result.frame.empty:
        return _empty_standalone_log(target_date, config, audit, "missing_prediction")

    prices = _load_price_rows(target_date, root)
    if prices.empty:
        audit["warning"] = "price_cache_missing"
        return _empty_standalone_log(target_date, config, audit, "price_cache_missing")

    data = load_result.frame.merge(prices, on="code", how="inner", suffixes=("", "_price"))
    if "date_price" in data.columns:
        data["date"] = data["date_price"]
    price_columns = [column for column in ["open", "high", "low", "close"] if column in data.columns]
    if price_columns:
        data = data.dropna(subset=price_columns).copy()
    data = _apply_turnover_filter(data, ml_config)
    if data.empty:
        audit["warning"] = "no_predictions_after_turnover_filter"
        return _empty_standalone_log(target_date, config, audit, "turnover_filter")

    top_n = int(ml_config.get("top_n", config.get("selection", {}).get("max_selected", 10)) or 10)
    data = data.sort_values("risk_adjusted_score", ascending=False).head(top_n).reset_index(drop=True)
    scores = [_standalone_score_row(row, target_date, rank) for rank, (_, row) in enumerate(data.iterrows(), start=1)]
    audit.update(
        {
            "joined_count": int(len(data)),
            "selected_count": len(scores),
            "top_n": top_n,
        }
    )
    return {
        "date": target_date,
        "signal_date": target_date,
        "provider": "ml_walk_forward_predictions",
        "profile_id": config.get("profile_id"),
        "profile_name": config.get("profile_name"),
        "scores": scores,
        "selected": scores,
        "candidate_count": len(scores),
        "selected_count": len(scores),
        "selection_config": config.get("selection", {}),
        "market_context": {},
        "dynamic_exposure_context": {},
        "market_filter": {},
        "source_provider": "ml_walk_forward_predictions",
        "ml_backtest": {
            **audit,
            "enabled": True,
            "mode": "standalone",
            "ranking": "risk_adjusted_score",
            "note": "Candidates are built directly from walk-forward predictions for this date.",
        },
    }


def _prediction_loader(ml_config: dict[str, Any], root: str | Path) -> MLBacktestPredictionLoader:
    prediction_root = ml_config.get("prediction_root")
    if prediction_root:
        path = Path(prediction_root)
        if not path.is_absolute():
            path = Path(root) / path
        return MLBacktestPredictionLoader(path)
    return MLBacktestPredictionLoader(Path(root) / "data" / "ml" / "walk_forward_predictions")


def _normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    df = frame.copy()
    if "code" in df.columns:
        df["code"] = df["code"].astype(str)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["expected_return_10d", "expected_max_return_20d", "swing_success_probability_20d", "bad_entry_probability_10d", "turnover_value"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if {"expected_return_10d", "bad_entry_probability_10d"}.issubset(df.columns):
        df["risk_adjusted_score"] = df["expected_return_10d"] - 0.5 * df["bad_entry_probability_10d"]
    return df


def _attach_predictions(
    rows: list[dict[str, Any]],
    load_result: MLPredictionLoadResult,
    ml_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = {
        str(row.get("code")): row
        for row in load_result.frame.to_dict("records")
        if row.get("code") is not None
    }
    min_turnover = float(ml_config.get("min_turnover_value", 0) or 0)
    joined = 0
    missing = 0
    low_turnover = 0
    output: list[dict[str, Any]] = []
    for item in rows:
        row = dict(item)
        row["ml_ranked_original_selected"] = bool(item.get("selected"))
        code = str(row.get("code") or "")
        prediction = predictions.get(code)
        if prediction:
            joined += 1
            row["ml_prediction_found"] = True
            row["ml_prediction_source"] = load_result.source
            for column in ML_BACKTEST_COLUMNS:
                if column in prediction:
                    row[column] = _as_float_or_none(prediction.get(column))
        else:
            missing += 1
            row["ml_prediction_found"] = False
            row["ml_prediction_source"] = load_result.source
            row["risk_adjusted_score"] = None
        turnover = _as_float_or_none(row.get("turnover_value"))
        if min_turnover and (turnover is None or turnover < min_turnover):
            low_turnover += 1
            row["ml_turnover_filter_pass"] = False
            if row.get("selected"):
                row["selected"] = False
                row["rejected_reason"] = _append_reason(row.get("rejected_reason"), "ml_turnover_filter")
        else:
            row["ml_turnover_filter_pass"] = True
        output.append(row)
    audit = {
        **_base_audit(load_result, ml_config),
        "scored_count": len(rows),
        "joined_count": joined,
        "missing_count": missing,
        "low_turnover_count": low_turnover,
    }
    return output, audit


def _sort_ml_ranked_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, int, float, float, str]:
        found = 1 if row.get("ml_prediction_found") else 0
        turnover_pass = 1 if row.get("ml_turnover_filter_pass", True) else 0
        risk_score = _as_float_or_none(row.get("risk_adjusted_score"))
        total_score = _as_float_or_none(row.get("total_score")) or 0.0
        return (found, turnover_pass, risk_score if risk_score is not None else -999999.0, total_score, str(row.get("code") or ""))

    return sorted(rows, key=key, reverse=True)


def _load_price_rows(target_date: str, root: str | Path) -> pd.DataFrame:
    loader = JQuantsDataLoader(Path(root) / "data" / "cache" / "jquants")
    prices = loader.load_prices(target_date, target_date)
    if prices.empty:
        return prices
    columns = ["date", "code", "open", "high", "low", "close", "volume", "turnover_value"]
    prices = prices[[column for column in columns if column in prices.columns]].copy()
    prices["code"] = prices["code"].astype(str)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return prices


def _apply_turnover_filter(frame: pd.DataFrame, ml_config: dict[str, Any]) -> pd.DataFrame:
    min_turnover = float(ml_config.get("min_turnover_value", 0) or 0)
    if not min_turnover or "turnover_value" not in frame.columns:
        return frame.copy()
    return frame[pd.to_numeric(frame["turnover_value"], errors="coerce") >= min_turnover].copy()


def _standalone_score_row(row: pd.Series, target_date: str, rank: int) -> dict[str, Any]:
    code = str(row.get("code") or "")
    risk_score = _as_float_or_none(row.get("risk_adjusted_score"))
    name = _as_text(row.get("name")) or code
    sector_name = _as_text(row.get("sector_name"))
    market = _as_text(row.get("market"))
    return {
        "date": target_date,
        "signal_date": target_date,
        "code": code,
        "name": name,
        "sector_name": sector_name,
        "market": market,
        "market_section": market or "Unknown",
        "listing_market": market or "Unknown",
        "open": _as_float_or_none(row.get("open")),
        "high": _as_float_or_none(row.get("high")),
        "low": _as_float_or_none(row.get("low")),
        "close": _as_float_or_none(row.get("close")),
        "volume": _as_float_or_none(row.get("volume")),
        "turnover_value": _as_float_or_none(row.get("turnover_value")),
        "total_score": 100.0 + (risk_score or 0.0),
        "technical_score": 0.0,
        "confidence": 1.0,
        "selected": True,
        "rank": rank,
        "daily_score_rank": rank,
        "reason": "ml_standalone_risk_adjusted_score",
        "selected_reason": "ml_standalone_top_n",
        "ml_prediction_found": True,
        "ml_prediction_source": "walk_forward_predictions",
        "ml_turnover_filter_pass": True,
        **{column: _as_float_or_none(row.get(column)) for column in ML_BACKTEST_COLUMNS},
    }


def _empty_standalone_log(target_date: str, config: dict[str, Any], audit: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "date": target_date,
        "signal_date": target_date,
        "provider": "ml_walk_forward_predictions",
        "profile_id": config.get("profile_id"),
        "profile_name": config.get("profile_name"),
        "scores": [],
        "selected": [],
        "candidate_count": 0,
        "selected_count": 0,
        "selection_config": config.get("selection", {}),
        "market_context": {},
        "dynamic_exposure_context": {},
        "market_filter": {},
        "skip_reason": reason,
        "source_provider": "ml_walk_forward_predictions",
        "ml_backtest": {
            **audit,
            "enabled": True,
            "mode": "standalone",
            "warning": reason,
        },
    }


def _base_audit(load_result: MLPredictionLoadResult, ml_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction_path": str(load_result.path),
        "prediction_exists": load_result.exists,
        "prediction_rows": int(len(load_result.frame)),
        "prediction_source": load_result.source,
        "min_turnover_value": float(ml_config.get("min_turnover_value", 0) or 0),
    }


def _ml_config(config: dict[str, Any]) -> dict[str, Any]:
    ml_config = config.get("ml_backtest", {})
    return ml_config if isinstance(ml_config, dict) else {}


def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _append_reason(current: Any, reason: str) -> str:
    text = str(current or "").strip()
    if not text:
        return reason
    if reason in text.split("|"):
        return text
    return f"{text}|{reason}"
