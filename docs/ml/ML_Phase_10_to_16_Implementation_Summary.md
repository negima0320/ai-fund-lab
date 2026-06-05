# ML Phase 10-16 Implementation Summary

This document summarizes the post-smoke ML work added after the initial
Phase 1-9 pipeline. These phases keep the existing trading logic unchanged.
They are report-only or offline ML pipeline extensions.

## Scope

Implemented:

- prediction-vs-label evaluation reports
- multi-day range smoke loop
- existing backtest trade join with ML predictions
- virtual ML filter simulation
- virtual ML position sizing simulation
- win/loss trade analysis using ML outputs
- swing-trade labels and 7-model training/prediction support
- ML generated data/model/report ignore rules

Still not implemented:

- production trading integration
- automatic retraining
- automatic order sizing changes
- existing strategy buy/sell logic changes
- J-Quants API refetch from ML tools

The ML tools continue to use local cache and generated parquet files only.
They do not call OpenAI API or J-Quants API.

## Ignored Generated Artifacts

The following generated ML artifacts are ignored:

- `data/ml/`
- `data/cache/jquants/prices/`
- `models/ml/`
- `reports/ml/`

Note: `.gitignore` does not remove files that were already tracked by Git.
If previously tracked ML parquet files should be removed from version control,
they need a separate index cleanup.

## Phase 10: Prediction Evaluation

Files:

- `src/ml/evaluator.py`
- `scripts/ml/evaluate_predictions.py`
- `tests/test_ml_evaluator.py`

Main class:

- `PredictionEvaluator`

Responsibilities:

- read `predictions_YYYY-MM-DD.parquet`
- read `labels_YYYY-MM-DD.parquet`
- inner join on `date + code`
- compute one-day evaluation metrics
- save Markdown reports to `reports/ml/evaluation_YYYY-MM-DD.md`

Core evaluation:

- joined row count
- top-N by `ml_score`
- entry risk label summary
- bad entry probability band summary
- `expected_return_10d` vs `future_10d_return` correlation

Phase 16 extended this evaluator with swing metrics:

- `expected_max_return_10d` vs `future_max_return_10d`
- `expected_max_return_20d` vs `future_max_return_20d`
- `swing_success_probability_20d` vs `future_swing_success_20d`
- swing success probability band summary

CLI:

```bash
python3 scripts/ml/evaluate_predictions.py --date 2026-05-15
```

## Phase 11: Multi-Day Range Smoke

Files:

- `src/ml/range_evaluator.py`
- `scripts/ml/smoke_ml_range.py`
- `tests/test_ml_smoke_range.py`
- `docs/ml_range_smoke_test.md`

Main class:

- `RangeSmokeRunner`

Pipeline:

1. build features for each date
2. generate labels for each date
3. build dataset
4. split train/valid/test by time
5. train models
6. predict each processed date
7. evaluate each date
8. save range summary report

CLI:

```bash
python3 scripts/ml/smoke_ml_range.py \
  --start 2025-06-01 \
  --end 2026-05-31 \
  --train-end 2026-02-28 \
  --valid-end 2026-04-30
```

Outputs:

- `data/ml/datasets/ml_dataset.parquet`
- `data/ml/datasets/train.parquet`
- `data/ml/datasets/valid.parquet`
- `data/ml/datasets/test.parquet`
- `models/ml/archive/YYYYMMDD_HHMMSS/`
- `models/ml/current/`
- `reports/ml/range_smoke_START_to_END.md`

## Phase 12-14: Backtest Join Analysis

Files:

- `src/ml/backtest_ml_analysis.py`
- `scripts/ml/analyze_backtest_with_ml.py`
- `tests/test_ml_backtest_ml_analysis.py`

Main class:

- `BacktestMLAnalyzer`

Responsibilities:

- read an existing `trades.csv`
- filter closed SELL trades for the requested period
- read ML predictions by `signal_date + code`
- keep existing trades even when ML predictions are missing
- produce report-only analysis
- never change trading logic

Join key:

- primary: `signal_date + code`
- fallback join date source in trade loading: `entry_date + code`

Report outputs:

- `reports/ml/backtest_ml_join_PROFILE_START_to_END.md`
- `reports/ml/backtest_ml_join_PROFILE_START_to_END.json`

Analysis tables:

- join summary
- existing trade performance by `entry_risk_label`
- bad entry probability band performance
- `ml_score` top/bottom performance
- `entry_risk_label x ml_score_band`
- `bad_entry_probability_10d x expected_return_10d`
- danger breakdown by expected return and upside probability
- trade details with ML columns

CLI:

```bash
python3 scripts/ml/analyze_backtest_with_ml.py \
  --profile rookie_dealer_02_v2_65 \
  --start 2025-06-01 \
  --end 2026-05-31
```

## Virtual Filter Simulation

The backtest join report includes hypothetical report-only filters.
They do not change trading behavior.

Filters:

- A: remove `entry_risk_label == "danger"`
- B: remove `bad_entry_probability_10d >= 0.70`
- C: remove danger and `ml_score < 0`
- D: remove danger and `upside_probability_10d < 0.80`
- E: remove `bad_entry_probability_10d >= 0.40` and `expected_return_10d < 0.03`
- F: remove `bad_entry_probability_10d >= 0.40` and `ml_score < 5`

Metrics:

- original/kept/removed trade count
- original/kept/removed net profit total
- original/kept/removed win rate
- `profit_delta`

In the one-year smoke validation, these filters did not improve the existing
strategy. Danger removal or reduction removed several profitable trades.

## Virtual Position Sizing Simulation

The backtest join report includes report-only position size simulations.
They do not change trading behavior.

Sizing rules:

- A: safe=1.0, watch=1.0, danger=0.5
- B: safe=1.2, watch=1.0, danger=0.5
- C: safe=1.2, watch=1.0, danger=0.7
- D: size by bad entry probability bands
- E: safe=1.2, watch=1.0, danger with high upside=1.0, other danger=0.5

Metrics:

- original net/gross profit
- adjusted net/gross profit
- profit delta
- average/min/max multiplier
- weighted win rate
- trade count

In the one-year smoke validation, position-size reduction also did not improve
the existing strategy.

## Phase 15: Win/Loss ML Trade Analysis

Added to `BacktestMLAnalyzer`.

Additional outputs:

- `reports/ml/backtest_ml_win_loss_analysis_PROFILE_START_to_END.md`
- `reports/ml/backtest_ml_win_loss_analysis_PROFILE_START_to_END.json`
- `reports/ml/backtest_ml_trades_PROFILE_START_to_END.csv`

Analysis:

- win trades vs loss trades ML averages
- top 10 profit trades
- bottom 10 profit trades
- `entry_risk_label x win/loss`
- danger win/loss difference
- watch distribution and trade details
- ML columns appended to trade detail CSV

Purpose:

- understand where the existing strategy wins despite danger labels
- avoid prematurely using danger as a hard exclusion signal

One-year result for `rookie_dealer_02_v2_65`:

- `trade_rows=134`
- `joined_count=133`
- `missing_count=1`
- `join_rate=0.9925`

Risk label performance after Phase 16 retraining:

- danger: `count=54`, `win_rate=0.4444`, `net_profit_total=652858.5`
- safe: `count=27`, `win_rate=0.5556`, `net_profit_total=95472.93`
- watch: `count=52`, `win_rate=0.4423`, `net_profit_total=-87829.41`

## Phase 16: Swing Labels And 7 Models

Files:

- `src/ml/config.py`
- `src/ml/label_generator.py`
- `src/ml/dataset_builder.py`
- `src/ml/model_trainer.py`
- `src/ml/predictor.py`
- `src/ml/evaluator.py`
- `src/ml/backtest_ml_analysis.py`

Additional labels:

- `future_max_return_10d`
- `future_max_return_20d`
- `future_swing_success_20d`

Definitions:

- `entry_price = next business day's open`
- `future_max_return_10d = max high within entry day through 10 business days / entry_price - 1`
- `future_max_return_20d = max high within entry day through 20 business days / entry_price - 1`
- `future_swing_success_20d = future_max_return_20d >= 0.10`

Compatibility:

- existing 4 labels remain required for the base dataset
- 20d swing labels are optional in dataset assembly
- ModelTrainer drops null target rows per model
- old 4-model current directories can still be loaded; missing new models produce null swing prediction columns

Additional models:

- `future_max_return_10d_regression`
- `future_max_return_20d_regression`
- `future_swing_success_20d_classification`

Total current model set:

- `future_5d_return_regression`
- `future_10d_return_regression`
- `upside_10d_classification`
- `bad_entry_10d_classification`
- `future_max_return_10d_regression`
- `future_max_return_20d_regression`
- `future_swing_success_20d_classification`

Additional prediction columns:

- `expected_max_return_10d`
- `expected_max_return_20d`
- `swing_success_probability_20d`

## One-Year Phase 16 Validation

Period:

- `2025-06-01` to `2026-05-31`

Train/valid split:

- train: `date <= 2026-02-28`
- valid: `2026-02-28 < date <= 2026-04-30`
- test: `2026-04-30 < date`

20d label availability:

- non-null date range: `2025-06-02` to `2026-04-27`
- non-null 20d label rows: `512239`

Dataset:

- full dataset: `523231 rows`, `34 columns`
- train: `444446 rows`
- valid: `66219 rows`
- test: `12566 rows`
- train non-null swing labels: `444446`
- valid non-null swing labels: `63074`
- test non-null swing labels: `0`

Prediction/evaluation:

- prediction dates: `232`
- prediction rows total: `554274`
- joined evaluation rows total: `524603`

Range correlations:

- `expected_return_10d` vs `future_10d_return`: `0.1651`
- `expected_max_return_10d` vs `future_max_return_10d`: `0.3518`
- `expected_max_return_20d` vs `future_max_return_20d`: `0.3324`
- `swing_success_probability_20d` vs `future_swing_success_20d`: `0.3235`

Metrics:

- `future_5d_return_regression`: RMSE `0.0622`, MAE `0.0366`
- `future_10d_return_regression`: RMSE `0.0927`, MAE `0.0560`
- `upside_10d_classification`: AUC `0.6717`
- `bad_entry_10d_classification`: AUC `0.6085`
- `future_max_return_10d_regression`: RMSE `0.0624`, MAE `0.0414`
- `future_max_return_20d_regression`: RMSE `0.0971`, MAE `0.0649`
- `future_swing_success_20d_classification`: AUC `0.6841`

Backtest join result:

- `trade_rows=134`
- `joined_count=133`
- `missing_count=1`
- `join_rate=0.9925`

Win/loss swing output averages:

- win `expected_max_return_20d_mean=0.0973`
- loss `expected_max_return_20d_mean=0.0999`
- win `swing_success_probability_20d_mean=0.3464`
- loss `swing_success_probability_20d_mean=0.3485`

Danger-only comparison:

- danger win `expected_max_return_20d_mean=0.1160`
- danger loss `expected_max_return_20d_mean=0.1164`
- danger win `swing_success_probability_20d_mean=0.4089`
- danger loss `swing_success_probability_20d_mean=0.3961`

Interpretation:

- swing labels appear useful at the all-stock prediction/ranking level
- they do not yet clearly separate existing strategy winners from losers
- danger remains profitable for the existing strategy, so hard exclusion remains unsupported
- next useful step is to test ML as a candidate-ranking overlay before changing trade logic

## Verification Commands

Unit tests:

```bash
python3 -m pytest \
  tests/test_ml_label_generator.py \
  tests/test_ml_dataset_builder.py \
  tests/test_ml_model_trainer.py \
  tests/test_ml_predictor.py \
  tests/test_ml_evaluator.py \
  tests/test_ml_backtest_ml_analysis.py \
  -q
```

One-day label check:

```bash
python3 scripts/ml/update_labels.py --date 2026-04-27
```

One-year range smoke:

```bash
python3 scripts/ml/smoke_ml_range.py \
  --start 2025-06-01 \
  --end 2026-05-31 \
  --train-end 2026-02-28 \
  --valid-end 2026-04-30
```

Dataset build:

```bash
python3 scripts/ml/build_dataset.py \
  --start 2025-06-01 \
  --end 2026-05-31 \
  --train-end 2026-02-28 \
  --valid-end 2026-04-30
```

Training:

```bash
python3 scripts/ml/train_models.py \
  --train data/ml/datasets/train.parquet \
  --valid data/ml/datasets/valid.parquet
```

Backtest ML join analysis:

```bash
python3 scripts/ml/analyze_backtest_with_ml.py \
  --profile rookie_dealer_02_v2_65 \
  --start 2025-06-01 \
  --end 2026-05-31
```
