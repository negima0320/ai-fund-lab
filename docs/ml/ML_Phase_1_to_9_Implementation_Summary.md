# ML Phase 1-9 Implementation Summary

This document summarizes the lightweight ML pipeline implemented for AI Fund Lab.

## Scope

Implemented:

- J-Quants cache DataLoader
- daily feature generation
- label generation
- feature/label dataset assembly
- LightGBM model training wrapper
- daily prediction wrapper
- daily ML pipeline orchestration
- smoke test CLI
- raw price to J-Quants cache sync CLI

Not implemented yet:

- production daily prediction integration
- existing backtest integration
- auto trading integration
- automatic retraining
- full-period heavy feature generation, labeling, dataset build, or training
- J-Quants API refetch from ML tools

The ML tools read local files only. They do not call OpenAI API, J-Quants API, or `data/processed/`.

## Paths

Input cache:

- `data/cache/jquants/prices/*.json`
- `data/cache/jquants/listed_info/*.json`
- `data/cache/jquants/topix_prices/*.json`
- `data/cache/jquants/earnings_calendar/*.json`
- `data/cache/jquants/trading_calendar/*.json`
- `data/cache/jquants/investor_types/*.json`
- `data/cache/jquants/financial_statements/*.json`

ML outputs:

- `data/ml/features/features_YYYY-MM-DD.parquet`
- `data/ml/labels/labels_YYYY-MM-DD.parquet`
- `data/ml/datasets/ml_dataset.parquet`
- `data/ml/datasets/train.parquet`
- `data/ml/datasets/valid.parquet`
- `data/ml/datasets/test.parquet`
- `data/ml/predictions/predictions_YYYY-MM-DD.parquet`

Models:

- `models/ml/archive/YYYYMMDD_HHMMSS/`
- `models/ml/current/`

## Phase 1: DataLoader

Files:

- `src/ml/config.py`
- `src/ml/data_loader.py`
- `src/ml/__init__.py`
- `tests/test_ml_data_loader.py`

Main class:

- `JQuantsDataLoader`

Methods:

- `load_prices(start_date, end_date)`
- `load_listed_info(as_of_date=None)`
- `load_topix(start_date, end_date)`
- `load_earnings_calendar(start_date, end_date)`
- `load_trading_calendar(start_date, end_date)`
- `load_investor_types(start_date, end_date)`
- `load_financial_statements(start_date, end_date)`

Normalization:

- `Date/date` to `date`
- `Code/LocalCode/code` to `code`
- price columns to `open/high/low/close/volume/turnover_value`
- adjusted price columns to `adjusted_open/adjusted_high/adjusted_low/adjusted_close/adjusted_volume`
- trading calendar `HolDiv` to `holiday_division`
- `is_business_day = holiday_division == "1"`

## Phase 2: FeatureBuilder

Files:

- `src/ml/feature_builder.py`
- `scripts/ml/build_features.py`
- `tests/test_ml_feature_builder.py`

Main class:

- `FeatureBuilder`

Methods:

- `build_daily_features(target_date)`
- `save_daily_features(df, target_date)`

Features:

- base: `date`, `code`, `close`, `volume`, `turnover_value`
- returns: `return_1d`, `return_3d`, `return_5d`, `return_10d`, `return_20d`
- moving averages: `ma5_gap`, `ma10_gap`, `ma25_gap`, `ma75_gap`, `ma5_slope`, `ma25_slope`
- volume: `volume_ratio_5d`, `volume_ratio_20d`
- turnover: `turnover_ratio_5d`, `turnover_ratio_20d`
- candlestick: `body_ratio`, `upper_shadow_ratio`, `lower_shadow_ratio`, `close_position`, `gap_up_ratio`, `daily_range_ratio`

Future leakage prevention:

- reads prices only up to `target_date`
- filters `prices["date"] <= target_date` after loading
- rolling calculations are grouped by `code`

CLI:

```bash
python3 scripts/ml/build_features.py --date 2026-05-15
```

## Phase 3: LabelGenerator

Files:

- `src/ml/label_generator.py`
- `scripts/ml/update_labels.py`
- `tests/test_ml_label_generator.py`

Main class:

- `LabelGenerator`

Methods:

- `generate_labels(target_date)`
- `save_labels(df, target_date)`
- `update_available_labels(as_of_date)`

Labels:

- `entry_price`
- `future_5d_return`
- `future_10d_return`
- `upside_10d`
- `bad_entry_10d`

Rules:

- `entry_price` is the next business day's `open`
- `future_5d_return = close_5_business_days_after_target / entry_price - 1`
- `future_10d_return = close_10_business_days_after_target / entry_price - 1`
- `upside_10d` is true if high within entry day through 10 business days reaches `+5%`
- `bad_entry_10d` is true if low within entry day through 10 business days reaches `-5%`

Business days:

- uses `trading_calendar.is_business_day`
- falls back to dates present in prices if calendar is missing

CLI:

```bash
python3 scripts/ml/update_labels.py --date 2026-05-15
python3 scripts/ml/update_labels.py --as-of 2026-05-29
```

## Phase 4: DatasetBuilder

Files:

- `src/ml/dataset_builder.py`
- `scripts/ml/build_dataset.py`
- `tests/test_ml_dataset_builder.py`

Main class:

- `DatasetBuilder`

Methods:

- `build_dataset(start_date, end_date)`
- `split_by_time(df, train_end, valid_end)`
- `save_dataset(df, name)`

Join:

- inner join features and labels on `date + code`

Filtering:

- missing `date`
- missing `code`
- missing `close`
- missing `entry_price`
- missing `future_5d_return`
- missing `future_10d_return`
- missing `upside_10d`
- missing `bad_entry_10d`
- `volume <= 0`
- `abs(future_5d_return) > 1.0`
- `abs(future_10d_return) > 1.0`

Time split:

- train: `date <= train_end`
- valid: `train_end < date <= valid_end`
- test: `valid_end < date`

CLI:

```bash
python3 scripts/ml/build_dataset.py \
  --start 2026-05-15 \
  --end 2026-05-15 \
  --train-end 2026-05-15 \
  --valid-end 2026-05-15
```

## Phase 5: ModelTrainer

Files:

- `src/ml/model_trainer.py`
- `scripts/ml/train_models.py`
- `tests/test_ml_model_trainer.py`

Main class:

- `ModelTrainer`

Methods:

- `load_dataset(path)`
- `train_all(train_df, valid_df)`
- `train_regression(target_col, train_df, valid_df)`
- `train_classification(target_col, train_df, valid_df)`
- `save_models(models, metrics)`

Models:

- `future_5d_return_regression.joblib`
- `future_10d_return_regression.joblib`
- `upside_10d_classification.joblib`
- `bad_entry_10d_classification.joblib`

Metadata:

- `feature_columns.json`
- `metrics.json`

Metrics:

- regression: `rmse`, `mae`
- classification: `auc`, `accuracy`, `precision`, `recall`

Excluded from features:

- `date`
- `code`
- `entry_price`
- `future_5d_return`
- `future_10d_return`
- `upside_10d`
- `bad_entry_10d`

Optional categorical features:

- `market`
- `sector_name`
- `scale_category`

CLI:

```bash
python3 scripts/ml/train_models.py \
  --train data/ml/datasets/train.parquet \
  --valid data/ml/datasets/valid.parquet
```

## Phase 6: Predictor

Files:

- `src/ml/predictor.py`
- `scripts/ml/predict_daily.py`
- `tests/test_ml_predictor.py`

Main class:

- `Predictor`

Methods:

- `load_current_models()`
- `predict_daily(target_date)`
- `save_predictions(df, target_date)`

Output columns:

- `date`
- `code`
- `expected_return_5d`
- `expected_return_10d`
- `upside_probability_10d`
- `bad_entry_probability_10d`
- `entry_risk_label`
- `ml_score`

`entry_risk_label`:

- `safe`: `bad_entry_probability_10d < 0.25`
- `watch`: `0.25 <= bad_entry_probability_10d < 0.40`
- `danger`: `bad_entry_probability_10d >= 0.40`

`ml_score`:

```text
expected_return_10d * 100
+ upside_probability_10d * 10
- bad_entry_probability_10d * 15
```

CLI:

```bash
python3 scripts/ml/predict_daily.py --date 2026-06-01
```

## Phase 7: Daily Pipeline

Files:

- `src/ml/pipeline.py`
- `scripts/ml/daily_pipeline.py`
- `tests/test_ml_pipeline.py`

Main class:

- `DailyMLPipeline`

Method:

- `run_daily_pipeline(target_date)`

Order:

1. `FeatureBuilder.build_daily_features(target_date)`
2. `FeatureBuilder.save_daily_features(...)`
3. if current models exist, `Predictor.predict_daily(target_date)`
4. if predicted, `Predictor.save_predictions(...)`
5. `LabelGenerator.update_available_labels(as_of_date=target_date)`

Model missing behavior:

- prediction is skipped
- `predictions_path` is `None`
- warning: `current ML models are missing; skipped prediction`

CLI:

```bash
python3 scripts/ml/daily_pipeline.py --date 2026-06-01
```

## Phase 8: Smoke Test

Files:

- `scripts/ml/smoke_ml_pipeline.py`
- `docs/ml_smoke_test.md`
- `tests/test_ml_smoke_cli.py`

CLI:

```bash
python3 scripts/ml/smoke_ml_pipeline.py --date 2026-05-15
```

Output sections:

- `target_date`
- `features_path`
- features rows and columns
- `updated_labels_paths`
- updated labels rows and columns
- `target_date_label_check` rows and columns
- `predictions_path` or `skipped`
- warnings

Important distinction:

- `updated_labels_paths` shows labels actually written by `update_available_labels(as_of_date=target_date)`
- `target_date_label_check` calls `LabelGenerator.generate_labels(target_date)` directly and does not save parquet

Observed real-data smoke after syncing May 2026 raw prices:

```text
target_date=2026-05-15
features_path=/Users/negishi/work/ai-fund-lab/data/ml/features/features_2026-05-15.parquet
features rows=1574 columns=26
updated_labels_paths=[]
updated_labels rows=0 columns=0
target_date_label_check rows=1570 columns=7
predictions_path=skipped
warning=current ML models are missing; skipped prediction
```

## Phase 9: Raw Price Cache Sync

Files:

- `scripts/ml/sync_raw_prices_to_jquants_cache.py`
- `docs/ml_data_cache_sync.md`
- `tests/test_sync_raw_prices_to_jquants_cache.py`

Purpose:

- converts `data/raw/prices_YYYY-MM-DD.json` to `data/cache/jquants/prices/YYYY-MM-DD.json`
- lets ML DataLoader use existing raw price snapshots without API refetch

Input:

```text
data/raw/prices_YYYY-MM-DD.json
```

Output:

```text
data/cache/jquants/prices/YYYY-MM-DD.json
```

Output JSON:

```json
{
  "records": [
    {
      "date": "2026-05-20",
      "code": "1001",
      "open": 100.0,
      "high": 105.0,
      "low": 99.0,
      "close": 104.0,
      "volume": 1000.0,
      "turnover_value": 104000.0
    }
  ]
}
```

CLI:

```bash
python3 scripts/ml/sync_raw_prices_to_jquants_cache.py --date 2026-05-20 --dry-run
python3 scripts/ml/sync_raw_prices_to_jquants_cache.py --date 2026-05-20
python3 scripts/ml/sync_raw_prices_to_jquants_cache.py --start 2026-05-01 --end 2026-05-29 --dry-run
```

Behavior:

- defaults to no overwrite
- `--overwrite` is required to overwrite existing cache
- `--dry-run` prints planned file syncs and counts only
- missing raw files are warnings, not hard failures

## Real-Data Checkpoints

May 2026 raw price sync:

```text
summary files=29 written=17 records=26768
```

One-day label generation:

```text
python3 scripts/ml/update_labels.py --date 2026-05-15
saved 1570 rows to data/ml/labels/labels_2026-05-15.parquet
```

One-day dataset build:

```text
data/ml/features/features_2026-05-15.parquet rows=1574 columns=26
data/ml/labels/labels_2026-05-15.parquet rows=1570 columns=7
data/ml/datasets/ml_dataset.parquet rows=1569 columns=31
data/ml/datasets/train.parquet rows=1569 columns=31
data/ml/datasets/valid.parquet rows=0 columns=31
data/ml/datasets/test.parquet rows=0 columns=31
```

The one-row reduction from labels to dataset is expected from dataset filtering. Required label and key columns had no missing values in the real-data check.

## Test Commands

Representative lightweight test commands used during implementation:

```bash
python3 -m pytest tests/test_ml_data_loader.py
python3 -m pytest tests/test_ml_feature_builder.py tests/test_ml_data_loader.py
python3 -m pytest tests/test_ml_label_generator.py tests/test_ml_feature_builder.py tests/test_ml_data_loader.py
python3 -m pytest tests/test_ml_dataset_builder.py tests/test_ml_label_generator.py tests/test_ml_feature_builder.py tests/test_ml_data_loader.py
python3 -m pytest tests/test_ml_model_trainer.py tests/test_ml_dataset_builder.py tests/test_ml_label_generator.py tests/test_ml_feature_builder.py tests/test_ml_data_loader.py
python3 -m pytest tests/test_ml_predictor.py tests/test_ml_model_trainer.py tests/test_ml_dataset_builder.py tests/test_ml_label_generator.py tests/test_ml_feature_builder.py tests/test_ml_data_loader.py
python3 -m pytest tests/test_ml_pipeline.py tests/test_ml_predictor.py tests/test_ml_model_trainer.py tests/test_ml_dataset_builder.py tests/test_ml_label_generator.py tests/test_ml_feature_builder.py tests/test_ml_data_loader.py
python3 -m pytest tests/test_ml_smoke_cli.py tests/test_ml_pipeline.py tests/test_ml_predictor.py tests/test_ml_model_trainer.py tests/test_ml_dataset_builder.py tests/test_ml_label_generator.py tests/test_ml_feature_builder.py tests/test_ml_data_loader.py
python3 -m pytest tests/test_sync_raw_prices_to_jquants_cache.py tests/test_ml_data_loader.py
```
