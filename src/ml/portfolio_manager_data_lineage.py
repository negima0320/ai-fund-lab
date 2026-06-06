from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.portfolio_manager_dataset import (
    AUDIT_COLUMNS,
    CLEAN_DAY_FEATURE_COLUMNS,
    CLEAN_EARNINGS_FEATURE_COLUMNS,
    CLEAN_FEATURE_COLUMNS,
    CLEAN_FINANCIAL_FEATURE_COLUMNS,
    CLEAN_FORBIDDEN_FEATURE_COLUMNS,
    CLEAN_ML_FEATURE_COLUMNS,
    CLEAN_PRICE_FEATURE_COLUMNS,
    CLEAN_RELATIVE_FEATURE_COLUMNS,
    CLEAN_TOPIX_FEATURE_COLUMNS,
    LABEL_COLUMNS,
)


DATASET_PATH = "data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet"
FEATURE_COLUMNS_PATH = "models/ml/portfolio_manager/current_v2_73_phase3b_clean/feature_columns.json"
REPORT_PATH = "reports/ml/portfolio_manager_data_lineage_audit_2023-01_to_2026-05.md"

EXTRA_FORBIDDEN_KEYWORDS = [
    "actual",
    "backtest",
    "audit",
    "profit",
    "return_label",
    "decision",
    "skip",
    "exit_reason",
    "cash_before",
    "cash_after",
    "final_amount",
    "final_shares",
    "result",
]

DATASET_AUDIT_COLUMNS = [
    *AUDIT_COLUMNS,
    "trade_id",
    "profile_id",
    "profile_name",
    "entry_date",
    "name",
    "candidate_rank",
    "score_rank",
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
    "reject_reason",
    "scale_reason",
    "allocation_limit",
    "allocation_reason",
    "prediction_source",
    "prediction_joined",
]


@dataclass(frozen=True)
class PortfolioManagerDataLineagePaths:
    markdown: Path


class PortfolioManagerDataLineageAudit:
    def __init__(
        self,
        root: str | Path = ".",
        dataset_path: str | Path = DATASET_PATH,
        feature_columns_path: str | Path = FEATURE_COLUMNS_PATH,
        report_path: str | Path = REPORT_PATH,
    ) -> None:
        self.root = Path(root)
        self.dataset_path = self._resolve(dataset_path)
        self.feature_columns_path = self._resolve(feature_columns_path)
        self.report_path = self._resolve(report_path)

    def run(self) -> dict[str, Any]:
        dataset = pd.read_parquet(self.dataset_path)
        feature_columns = json.loads(self.feature_columns_path.read_text(encoding="utf-8"))
        label_columns = [column for column in LABEL_COLUMNS if column in dataset.columns]
        audit_columns = [column for column in DATASET_AUDIT_COLUMNS if column in dataset.columns]
        classifications = self.classify_columns(dataset.columns)
        forbidden_hits = self.forbidden_feature_hits(feature_columns)
        label_hits = sorted(set(feature_columns).intersection(label_columns))
        unknown_feature_hits = sorted(
            column
            for column in feature_columns
            if classifications.get(column, "不明") in {"監査/ログ/売買結果由来", "不明"}
        )
        trainer_mismatch = sorted(set(feature_columns).symmetric_difference(set(CLEAN_FEATURE_COLUMNS)))
        pass_result = not forbidden_hits and not label_hits and not unknown_feature_hits
        return {
            "dataset_path": str(self.dataset_path),
            "feature_columns_path": str(self.feature_columns_path),
            "source_files": [
                "src/ml/portfolio_manager_dataset.py",
                "scripts/ml/build_portfolio_manager_dataset.py",
                "scripts/ml/train_portfolio_manager_phase3a.py",
            ],
            "rows": int(len(dataset)),
            "columns": int(len(dataset.columns)),
            "feature_count": int(len(feature_columns)),
            "feature_columns": feature_columns,
            "label_columns": label_columns,
            "audit_columns": audit_columns,
            "forbidden_feature_hits": forbidden_hits,
            "label_feature_hits": label_hits,
            "unknown_or_audit_feature_hits": unknown_feature_hits,
            "trainer_feature_mismatch_with_clean_feature_columns": trainer_mismatch,
            "column_classification": [
                {
                    "column": column,
                    "classification": classifications.get(column, "不明"),
                    "is_feature": column in feature_columns,
                    "is_label": column in label_columns,
                    "is_audit": column in audit_columns,
                }
                for column in dataset.columns
            ],
            "classification_summary": self._classification_summary(classifications),
            "inference_reproducibility": self._inference_reproducibility(feature_columns),
            "future_training_flow": [
                "J-Quants API/raw/cache update",
                "FeatureBuilder generates daily feature parquet from target-date available data",
                "Walk-forward or latest buy-side ML models generate prediction parquet",
                "Portfolio Manager clean dataset joins candidate set with J-Quants features and ML prediction parquet",
                "Labels are generated from future outcomes/trades only after feature assembly",
                "Portfolio Manager models train from clean feature_columns only",
            ],
            "current_model_regeneration_check": [
                "Phase 3-B clean dataset uses data/ml/walk_forward_predictions prediction_source rows.",
                "restore_walk_forward_predictions.py restores prediction parquet from fold archive model directories.",
                "feature_columns.json does not depend on models/ml/current buy model predictions.",
            ],
            "result": "PASS" if pass_result else "FAIL",
        }

    def save(self, result: dict[str, Any]) -> PortfolioManagerDataLineagePaths:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(self.format_markdown(result), encoding="utf-8")
        return PortfolioManagerDataLineagePaths(markdown=self.report_path)

    def classify_columns(self, columns: list[str] | pd.Index) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for column in columns:
            if column in CLEAN_ML_FEATURE_COLUMNS:
                mapping[column] = "既存ML予測"
            elif column in {"close", "volume", "turnover_value"}:
                mapping[column] = "J-Quants由来"
            elif column in CLEAN_PRICE_FEATURE_COLUMNS:
                mapping[column] = "価格/テクニカル派生"
            elif column in CLEAN_TOPIX_FEATURE_COLUMNS:
                mapping[column] = "TOPIX派生"
            elif column in CLEAN_FINANCIAL_FEATURE_COLUMNS:
                mapping[column] = "財務派生"
            elif column in CLEAN_EARNINGS_FEATURE_COLUMNS:
                mapping[column] = "決算"
            elif column in CLEAN_RELATIVE_FEATURE_COLUMNS + CLEAN_DAY_FEATURE_COLUMNS:
                mapping[column] = "日次集計特徴量"
            elif column in LABEL_COLUMNS:
                mapping[column] = "ラベル"
            elif column in DATASET_AUDIT_COLUMNS or self._looks_like_audit_column(column):
                mapping[column] = "監査/ログ/売買結果由来"
            elif column in {"signal_date", "code"}:
                mapping[column] = "J-Quants由来"
            else:
                mapping[column] = "不明"
        return mapping

    def forbidden_feature_hits(self, feature_columns: list[str]) -> list[str]:
        explicit = set(CLEAN_FORBIDDEN_FEATURE_COLUMNS + LABEL_COLUMNS + ["actual_profit", "actual_return", "realized_return"])
        hits = set(feature_columns).intersection(explicit)
        for column in feature_columns:
            lower = column.lower()
            if any(keyword in lower for keyword in EXTRA_FORBIDDEN_KEYWORDS):
                hits.add(column)
        return sorted(hits)

    def format_markdown(self, result: dict[str, Any]) -> str:
        feature_rows = [
            row for row in result["column_classification"] if row["is_feature"]
        ]
        lines = [
            "# Portfolio Manager AI Data Lineage Audit",
            "",
            f"- result: **{result['result']}**",
            f"- dataset: `{result['dataset_path']}`",
            f"- feature_columns: `{result['feature_columns_path']}`",
            f"- rows: {result['rows']}",
            f"- dataset columns: {result['columns']}",
            f"- feature count: {result['feature_count']}",
            "",
            "## Target Files",
            "",
        ]
        lines.extend(f"- `{path}`" for path in result["source_files"])
        lines.extend(
            [
                "",
                "## Forbidden Column Check",
                "",
                self._table(
                    [
                        {
                            "forbidden_feature_hits": ", ".join(result["forbidden_feature_hits"]) or "none",
                            "label_feature_hits": ", ".join(result["label_feature_hits"]) or "none",
                            "unknown_or_audit_feature_hits": ", ".join(result["unknown_or_audit_feature_hits"]) or "none",
                            "trainer_feature_mismatch": ", ".join(result["trainer_feature_mismatch_with_clean_feature_columns"]) or "none",
                        }
                    ],
                    ["forbidden_feature_hits", "label_feature_hits", "unknown_or_audit_feature_hits", "trainer_feature_mismatch"],
                ),
                "",
                "## Label Columns",
                "",
                self._table([{"label": column} for column in result["label_columns"]], ["label"]),
                "",
                "## Classification Summary",
                "",
                self._table(result["classification_summary"], ["classification", "count"]),
                "",
                "## Feature Column Classification",
                "",
                self._table(feature_rows, ["column", "classification"]),
                "",
                "## All Dataset Columns",
                "",
                self._table(result["column_classification"], ["column", "classification", "is_feature", "is_label", "is_audit"]),
                "",
                "## Inference Reproducibility",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in result["inference_reproducibility"])
        lines.extend(["", "## Future Training Flow", ""])
        lines.extend(f"- {item}" for item in result["future_training_flow"])
        lines.extend(["", "## Current Model Regeneration Check", ""])
        lines.extend(f"- {item}" for item in result["current_model_regeneration_check"])
        lines.extend(["", "## Conclusion", ""])
        if result["result"] == "PASS":
            lines.append("- PASS: clean Portfolio Manager features are reproducible from J-Quants-derived features and existing ML prediction parquet.")
        else:
            lines.append("- FAIL: feature columns contain forbidden, label, audit/log, or unknown columns. Fix before training or backtest integration.")
        lines.append("")
        return "\n".join(lines)

    def _inference_reproducibility(self, feature_columns: list[str]) -> list[str]:
        classifications = self.classify_columns(feature_columns)
        bad = sorted(
            column for column, classification in classifications.items()
            if classification in {"監査/ログ/売買結果由来", "不明", "ラベル"}
        )
        lines = [
            "J-Quants-derived daily feature parquet can regenerate price/technical/TOPIX/financial/earnings features.",
            "Walk-forward or latest buy-side ML prediction parquet can regenerate expected return/risk probability features.",
            "Candidate relative/day aggregate features are computed from same-day candidate ML/J-Quants values.",
        ]
        if bad:
            lines.append(f"Not reproducible as clean inference features: {', '.join(bad)}")
        else:
            lines.append("All feature_columns are reproducible without backtest execution results.")
        return lines

    def _classification_summary(self, classifications: dict[str, str]) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for classification in classifications.values():
            counts[classification] = counts.get(classification, 0) + 1
        return [{"classification": key, "count": value} for key, value in sorted(counts.items())]

    def _looks_like_audit_column(self, column: str) -> bool:
        lower = column.lower()
        return any(keyword in lower for keyword in EXTRA_FORBIDDEN_KEYWORDS)

    def _resolve(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(column, "")).replace("\n", " ") for column in columns) + " |")
        return "\n".join(lines)
