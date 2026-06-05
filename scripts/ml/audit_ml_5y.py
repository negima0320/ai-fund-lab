#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 5y ML walk-forward predictions.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--predictions-root", default="data/ml/walk_forward_predictions")
    parser.add_argument("--features-root", default="data/ml/features")
    parser.add_argument("--labels-root", default="data/ml/labels")
    parser.add_argument("--model-root", default="models/ml/walk_forward/current/2026-05")
    parser.add_argument("--walk-forward-json", default="reports/ml/walk_forward_5y_enriched_2023-01_to_2026-05.json")
    parser.add_argument("--output-md", default="reports/ml/ml_audit_5y.md")
    parser.add_argument("--output-json", default="reports/ml/ml_audit_5y.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_audit(args)
    md_path = Path(args.output_md)
    json_path = Path(args.output_json)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(format_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved markdown report to {md_path.resolve()}")
    print(f"saved json report to {json_path.resolve()}")
    joined = result["joined_summary"]
    print(
        f"joined_rows={joined['rows']} dates={joined['dates']} codes={joined['codes']} "
        f"expected_return_10d_corr={result['prediction_quality']['expected_return_10d_corr']}"
    )


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    joined = load_joined_predictions(args.predictions_root, args.labels_root, start, end)
    features = load_features(args.features_root, sorted(joined["date"].dropna().dt.strftime("%Y-%m-%d").unique()))
    return {
        "period": {"start": args.start, "end": args.end},
        "data_leakage_check": data_leakage_check(args.walk_forward_json),
        "joined_summary": {
            "rows": int(len(joined)),
            "dates": int(joined["date"].nunique()) if not joined.empty else 0,
            "codes": int(joined["code"].nunique()) if not joined.empty else 0,
        },
        "feature_non_null_rate": feature_non_null_rate(features),
        "feature_non_null_summary": feature_non_null_summary(features),
        "label_distribution": label_distribution(joined),
        "prediction_quality": prediction_quality(joined),
        "top_bottom_expected_return_10d": top_bottom_analysis(joined, "expected_return_10d"),
        "prediction_deciles": prediction_deciles(joined, "expected_return_10d"),
        "feature_importance_top50": feature_importance(Path(args.model_root), 50),
        "judgement": judge_model(joined),
    }


def load_joined_predictions(predictions_root: str, labels_root: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    pred_root = Path(predictions_root)
    label_root = Path(labels_root)
    frames = []
    for path in sorted(pred_root.glob("predictions_*.parquet")):
        date_text = path.stem.replace("predictions_", "")
        date = pd.Timestamp(date_text)
        if date < start or date > end:
            continue
        label_path = label_root / f"labels_{date_text}.parquet"
        if not label_path.exists():
            continue
        predictions = pd.read_parquet(path)
        labels = pd.read_parquet(label_path)
        if predictions.empty or labels.empty:
            continue
        predictions = normalize_keys(predictions)
        labels = normalize_keys(labels)
        label_columns = [
            "date",
            "code",
            "future_5d_return",
            "future_10d_return",
            "future_max_return_10d",
            "future_max_return_20d",
            "future_swing_success_20d",
            "upside_10d",
            "bad_entry_10d",
        ]
        label_columns = [column for column in label_columns if column in labels.columns]
        frames.append(predictions.merge(labels[label_columns], on=["date", "code"], how="inner"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_features(features_root: str, date_texts: list[str]) -> pd.DataFrame:
    root = Path(features_root)
    frames = []
    for date_text in date_texts:
        path = root / f"features_{date_text}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype("string")
    return data


def data_leakage_check(walk_forward_json: str) -> dict[str, Any]:
    path = Path(walk_forward_json)
    if not path.exists():
        return {"status": "warning", "message": f"walk-forward json not found: {walk_forward_json}"}
    data = json.loads(path.read_text())
    violations = []
    gaps = []
    for fold in data.get("folds", []):
        requested = pd.Timestamp(fold["requested_train_end"])
        effective = pd.Timestamp(fold["effective_train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        if effective > requested:
            violations.append({"month": fold["month"], "reason": "effective_train_end_after_requested_train_end"})
        if effective >= test_start:
            violations.append({"month": fold["month"], "reason": "effective_train_end_reaches_test_period"})
        gaps.append(int((test_start - effective).days))
    return {
        "status": "pass" if not violations else "fail",
        "violations": violations,
        "folds": int(len(data.get("folds", []))),
        "min_calendar_gap_days_between_effective_train_end_and_test_start": min(gaps) if gaps else None,
        "notes": [
            "walk-forward training uses effective_train_end capped by LABEL_LOOKAHEAD_DAYS.",
            "FeatureBuilder loads prices only up to target_date and filters prices[date] <= target.",
            "financial statements are filtered to disclosure date <= target_date.",
            "TOPIX features are loaded only through target_date.",
        ],
    }


def feature_non_null_rate(features: pd.DataFrame) -> list[dict[str, Any]]:
    if features.empty:
        return []
    rows = []
    for column in features.columns:
        if column in {"date", "code"}:
            continue
        rows.append({"feature": column, "non_null_rate": float(features[column].notna().mean())})
    return sorted(rows, key=lambda row: row["non_null_rate"])


def feature_non_null_summary(features: pd.DataFrame) -> dict[str, Any]:
    if features.empty:
        return {"rows": 0, "columns": 0}
    rates = [row["non_null_rate"] for row in feature_non_null_rate(features)]
    return {
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "feature_count": int(max(len(features.columns) - 2, 0)),
        "mean_non_null_rate": float(pd.Series(rates).mean()) if rates else None,
        "min_non_null_rate": float(pd.Series(rates).min()) if rates else None,
        "features_below_10pct": int(sum(rate < 0.10 for rate in rates)),
        "features_above_90pct": int(sum(rate >= 0.90 for rate in rates)),
    }


def label_distribution(df: pd.DataFrame) -> dict[str, Any]:
    columns = ["future_10d_return", "future_max_return_20d", "bad_entry_10d", "future_swing_success_20d"]
    return {column: describe_column(df, column) for column in columns if column in df.columns}


def prediction_quality(df: pd.DataFrame) -> dict[str, Any]:
    pairs = {
        "expected_return_10d_corr": ("expected_return_10d", "future_10d_return"),
        "expected_max_return_20d_corr": ("expected_max_return_20d", "future_max_return_20d"),
        "swing_success_probability_20d_corr": ("swing_success_probability_20d", "future_swing_success_20d"),
        "bad_entry_probability_10d_corr": ("bad_entry_probability_10d", "bad_entry_10d"),
    }
    return {name: correlation(df, left, right) for name, (left, right) in pairs.items()}


def top_bottom_analysis(df: pd.DataFrame, score_column: str) -> dict[str, Any]:
    data = df.dropna(subset=[score_column]).copy()
    if data.empty:
        return {}
    low = data[score_column].quantile(0.10)
    high = data[score_column].quantile(0.90)
    bottom = summarize_group(data[data[score_column] <= low], "bottom_10pct")
    top = summarize_group(data[data[score_column] >= high], "top_10pct")
    return {"score_column": score_column, "bottom_10pct": bottom, "top_10pct": top, "spread": spread(top, bottom)}


def prediction_deciles(df: pd.DataFrame, score_column: str) -> list[dict[str, Any]]:
    data = df.dropna(subset=[score_column]).copy()
    if data.empty:
        return []
    data["_decile"] = pd.qcut(data[score_column].rank(method="first"), 10, labels=False)
    rows = []
    for decile, group in data.groupby("_decile", sort=True):
        label = f"{int(decile) * 10}-{(int(decile) + 1) * 10}%"
        row = summarize_group(group, label)
        row["score_min"] = numeric(group[score_column]).min()
        row["score_max"] = numeric(group[score_column]).max()
        rows.append(row)
    return rows


def summarize_group(group: pd.DataFrame, name: str) -> dict[str, Any]:
    return {
        "bucket": name,
        "count": int(len(group)),
        "expected_return_10d_mean": mean(group, "expected_return_10d"),
        "future_10d_return_mean": mean(group, "future_10d_return"),
        "future_10d_return_median": median(group, "future_10d_return"),
        "future_max_return_20d_mean": mean(group, "future_max_return_20d"),
        "future_max_return_20d_median": median(group, "future_max_return_20d"),
        "bad_entry_rate": mean_bool(group, "bad_entry_10d"),
        "swing_success_rate": mean_bool(group, "future_swing_success_20d"),
    }


def spread(top: dict[str, Any], bottom: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for column in ["future_10d_return_mean", "future_max_return_20d_mean", "bad_entry_rate", "swing_success_rate"]:
        if top.get(column) is not None and bottom.get(column) is not None:
            output[column] = top[column] - bottom[column]
    return output


def describe_column(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in df.columns:
        return {}
    if df[column].dropna().isin([True, False]).all() or str(df[column].dtype) == "bool":
        return {
            "count": int(df[column].notna().sum()),
            "rate": mean_bool(df, column),
        }
    values = numeric(df[column]).dropna()
    if values.empty:
        return {"count": 0}
    quantiles = values.quantile([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).to_dict()
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": {str(key): float(value) for key, value in quantiles.items()},
    }


def feature_importance(model_root: Path, top_n: int) -> list[dict[str, Any]]:
    path = model_root / "future_10d_return_regression.joblib"
    feature_path = model_root / "feature_columns.json"
    if not path.exists() or not feature_path.exists():
        return []
    import joblib

    model = joblib.load(path)
    features = json.loads(feature_path.read_text())
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    rows = [
        {"rank": rank + 1, "feature": feature, "importance": int(importance)}
        for rank, (feature, importance) in enumerate(
            sorted(zip(features, importances), key=lambda item: item[1], reverse=True)[:top_n]
        )
    ]
    return rows


def judge_model(df: pd.DataFrame) -> dict[str, Any]:
    quality = prediction_quality(df)
    top_bottom = top_bottom_analysis(df, "expected_return_10d")
    spread_data = top_bottom.get("spread", {})
    alive = (
        (quality.get("expected_return_10d_corr") or 0) > 0
        and (spread_data.get("future_10d_return_mean") or 0) > 0
        and (spread_data.get("future_max_return_20d_mean") or 0) > 0
    )
    return {
        "ai_is_alive": bool(alive),
        "summary": (
            "expected_return_10d ranking separates future returns in the right direction"
            if alive
            else "expected_return_10d ranking did not clearly separate future returns"
        ),
    }


def correlation(df: pd.DataFrame, left: str, right: str) -> float | None:
    if left not in df.columns or right not in df.columns:
        return None
    data = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 2:
        return None
    return float(data[left].corr(data[right]))


def numeric(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    value = numeric(df[column]).mean()
    return None if pd.isna(value) else float(value)


def median(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    value = numeric(df[column]).median()
    return None if pd.isna(value) else float(value)


def mean_bool(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    value = df[column].astype("boolean").mean()
    return None if pd.isna(value) else float(value)


def format_markdown(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ML Audit 5Y",
            "",
            f"- period: {result['period']['start']} to {result['period']['end']}",
            f"- joined_rows: {result['joined_summary']['rows']}",
            f"- joined_dates: {result['joined_summary']['dates']}",
            f"- joined_codes: {result['joined_summary']['codes']}",
            "",
            "## Data Leakage Check",
            "",
            table([result["data_leakage_check"]], ["status", "folds", "min_calendar_gap_days_between_effective_train_end_and_test_start"]),
            "",
            "## Feature Non-Null Summary",
            "",
            table([result["feature_non_null_summary"]], ["rows", "columns", "feature_count", "mean_non_null_rate", "min_non_null_rate", "features_below_10pct", "features_above_90pct"]),
            "",
            "## Feature Non-Null Rates",
            "",
            table(result["feature_non_null_rate"], ["feature", "non_null_rate"]),
            "",
            "## Label Distribution",
            "",
            "```json",
            json.dumps(result["label_distribution"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Prediction Quality",
            "",
            table([result["prediction_quality"]], ["expected_return_10d_corr", "expected_max_return_20d_corr", "swing_success_probability_20d_corr", "bad_entry_probability_10d_corr"]),
            "",
            "## Expected Return Top/Bottom 10%",
            "",
            table(
                [
                    result["top_bottom_expected_return_10d"].get("bottom_10pct", {}),
                    result["top_bottom_expected_return_10d"].get("top_10pct", {}),
                ],
                ["bucket", "count", "future_10d_return_mean", "future_max_return_20d_mean", "bad_entry_rate", "swing_success_rate"],
            ),
            "",
            "## Prediction Deciles",
            "",
            table(result["prediction_deciles"], ["bucket", "count", "score_min", "score_max", "future_10d_return_mean", "future_max_return_20d_mean", "bad_entry_rate", "swing_success_rate"]),
            "",
            "## Feature Importance Top 50",
            "",
            table(result["feature_importance_top50"], ["rank", "feature", "importance"]),
            "",
            "## Judgement",
            "",
            f"- ai_is_alive: {result['judgement']['ai_is_alive']}",
            f"- summary: {result['judgement']['summary']}",
            "",
        ]
    )


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(fmt(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return str(len(value))
    return str(value)


if __name__ == "__main__":
    main()
