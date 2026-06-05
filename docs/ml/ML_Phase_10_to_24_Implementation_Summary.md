# ML Phase 10-24 Implementation Summary

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
- all-stock ML top-N ranking analysis
- ML top-N paper portfolio simulation
- realistic ML portfolio simulation with cash, position, fee, slippage, and liquidity constraints
- expanding-window walk-forward evaluation
- walk-forward losing-month diagnostics
- bad-entry-aware walk-forward ranking comparison
- daily AI candidate export
- daily pipeline candidate export integration
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

## Phase 17: All-Stock ML Ranking Analysis

Files:

- `src/ml/ranking_analysis.py`
- `scripts/ml/analyze_ml_rankings.py`
- `tests/test_ml_ranking_analysis.py`

Main class:

- `MLRankingAnalyzer`

Purpose:

- evaluate whether ML-ranked all-stock candidates work better than using ML as
  a filter on existing strategy trades
- compare daily top-N ranking buckets against realized labels
- measure overlap between ML top-N candidates and existing strategy trades

Inputs:

- `data/ml/predictions/predictions_YYYY-MM-DD.parquet`
- `data/ml/labels/labels_YYYY-MM-DD.parquet`
- optional existing backtest `trades.csv` for overlap analysis

Daily rankings:

- `expected_max_return_20d_top10`
- `swing_success_probability_20d_top10`
- `ml_score_top10`
- `expected_return_10d_top10`

Metrics:

- average `future_10d_return`
- average `future_max_return_20d`
- `future_swing_success_20d` rate
- `bad_entry_10d` rate
- ranked row count
- processed date count
- monthly summary
- all-stock baseline
- overlap with existing strategy trades

Outputs:

- `reports/ml/ml_ranking_analysis_START_to_END.md`
- `reports/ml/ml_ranking_analysis_START_to_END.json`
- `reports/ml/ml_ranking_details_START_to_END.csv`

CLI:

```bash
python3 scripts/ml/analyze_ml_rankings.py \
  --start 2025-06-01 \
  --end 2026-05-31 \
  --top-n 10 \
  --profile rookie_dealer_02_v2_65
```

One-year result:

- processed dates: `232`
- baseline rows: `524603`
- baseline `future_max_return_20d_mean=0.0867`
- baseline swing success rate: `0.2621`
- baseline bad entry rate: `0.2606`

Ranking comparison:

- `expected_max_return_20d_top10`
  - `future_max_return_20d_mean=0.3538`
  - swing success rate `0.7527`
  - bad entry rate `0.6478`
- `swing_success_probability_20d_top10`
  - `future_max_return_20d_mean=0.2708`
  - swing success rate `0.7122`
  - bad entry rate `0.6198`
- `expected_return_10d_top10`
  - `future_max_return_20d_mean=0.2493`
  - swing success rate `0.6896`
  - bad entry rate `0.3922`
- `ml_score_top10`
  - `future_max_return_20d_mean=0.1280`
  - swing success rate `0.4356`
  - bad entry rate `0.2858`

Existing strategy overlap:

- all four top10 ranking buckets had `0` overlap with
  `rookie_dealer_02_v2_65` trades in this one-year run
- existing strategy trade top10 rate: `0.0`

Interpretation:

- all-stock ML top10 rankings strongly outperform the all-stock baseline on
  realized max return and swing success rate
- the strongest max-return rankings also have much higher bad-entry rates
- existing strategy trades are selecting a different part of the universe than
  ML top10 rankings
- the next useful experiment is an ML candidate-ranking overlay or parallel
  paper portfolio, not danger-based exclusion of existing trades

## Phase 18: ML Top-N Paper Portfolio Simulation

Files:

- `src/ml/paper_portfolio.py`
- `scripts/ml/simulate_ml_paper_portfolio.py`
- `tests/test_ml_paper_portfolio.py`

Main class:

- `MLPaperPortfolioSimulator`

Purpose:

- simulate report-only swing portfolios from all-stock ML top-N rankings
- check whether high-upside ML candidates can be profitable despite high
  `bad_entry_10d` rates
- avoid changing the existing strategy or rerunning existing backtests

Inputs:

- `data/ml/predictions/predictions_YYYY-MM-DD.parquet`
- `data/ml/labels/labels_YYYY-MM-DD.parquet`
- `data/cache/jquants/prices/YYYY-MM-DD.json`

Rankings:

- `expected_max_return_20d_top10`
- `swing_success_probability_20d_top10`
- `expected_return_10d_top10`
- `ml_score_top10`

Exit rules:

- `close_20d`: exit at 20th business day close from entry day
- `close_10d`: exit at 10th business day close from entry day
- `take_profit_10pct_or_close_20d`: +10% take profit, otherwise 20d close
- `stop_loss_5pct_or_close_20d`: -5% stop loss, otherwise 20d close
- `take_profit_10pct_stop_loss_5pct_or_close_20d`: first +10% or -5%, otherwise 20d close

Notes:

- entry is next available price date after signal date, using that day's open
- entry day is counted as day 1 of the holding window
- if both take-profit and stop-loss are hit on the same OHLC day, stop-loss is
  assumed first for the combined rule
- fees and taxes are not included
- duplicate holdings are allowed
- each ranking is evaluated as a separate paper portfolio

Outputs:

- `reports/ml/ml_paper_portfolio_START_to_END.md`
- `reports/ml/ml_paper_portfolio_START_to_END.json`
- `reports/ml/ml_paper_trades_START_to_END.csv`

CLI:

```bash
python3 scripts/ml/simulate_ml_paper_portfolio.py \
  --start 2025-06-01 \
  --end 2026-05-31 \
  --top-n 10
```

One-year result summary:

- all ranking/exit combinations had positive average returns
- best by average return:
  - `expected_return_10d_top10 + close_20d`
  - `total_trades=2200`
  - `win_rate=0.7323`
  - `average_return=0.1256`
  - `total_return_sum=276.2225`
  - `profit_factor=5.6411`
- best by profit factor:
  - `expected_return_10d_top10 + close_10d`
  - `win_rate=0.7077`
  - `average_return=0.0844`
  - `profit_factor=6.1706`
- weakest by average return:
  - `swing_success_probability_20d_top10 + take_profit_10pct_stop_loss_5pct_or_close_20d`
  - `win_rate=0.4284`
  - `average_return=0.0137`
  - `profit_factor=1.4814`

Selected ranking/exit results:

- `expected_max_return_20d_top10 + close_20d`
  - `average_return=0.1172`
  - `win_rate=0.6036`
  - `profit_factor=2.8800`
- `swing_success_probability_20d_top10 + close_20d`
  - `average_return=0.0770`
  - `win_rate=0.5883`
  - `profit_factor=2.1894`
- `ml_score_top10 + close_20d`
  - `average_return=0.0525`
  - `win_rate=0.6738`
  - `profit_factor=3.0407`

Interpretation:

- ML top-N paper portfolios were profitable in this one-year after-the-fact
  simulation, even though high-upside rankings also had high bad-entry rates
- simple fixed holding exits performed better than take-profit/stop-loss exits
  in this run
- `expected_return_10d_top10` was the strongest ranking for paper portfolio
  returns, despite Phase 17 showing `expected_max_return_20d_top10` had the
  largest realized max-return label
- this supports testing ML as a separate candidate-ranking paper portfolio
  before making any production trading changes

## Phase 19: Realistic ML Portfolio Simulation

Files:

- `src/ml/realistic_portfolio.py`
- `scripts/ml/simulate_ml_realistic_portfolio.py`
- `tests/test_ml_realistic_portfolio.py`

Main class:

- `MLRealisticPortfolioSimulator`

Purpose:

- extend the Phase 18 paper portfolio with report-only practical constraints
- keep existing trading logic and existing backtests unchanged
- evaluate whether ML top-N candidates still work after cash, position count,
  duplicate holding, fees, slippage, and liquidity filters

Default assumptions:

- `initial_cash=1,000,000`
- `position_size=100,000`
- no duplicate holding of the same code
- buy at next business day open with buy-side slippage
- sell at fixed 10d or 20d close with sell-side slippage
- fees are charged on both buy and sell notional
- liquidity is filtered by `turnover_value` from daily features

Grid:

- rankings:
  - `expected_return_10d`
  - `expected_max_return_20d`
  - `ml_score`
- `max_positions`: 5, 10
- exit rules: `close_10d`, `close_20d`
- `min_turnover_value`: 50,000,000 and 100,000,000

Outputs:

- `reports/ml/ml_realistic_portfolio_START_to_END.md`
- `reports/ml/ml_realistic_portfolio_START_to_END.json`
- `reports/ml/ml_realistic_trades_START_to_END.csv`

CLI:

```bash
python3 scripts/ml/simulate_ml_realistic_portfolio.py \
  --start 2025-06-01 \
  --end 2026-05-31 \
  --top-n 10 \
  --initial-cash 1000000 \
  --position-size 100000 \
  --fee-rate 0.001 \
  --slippage-rate 0.001 \
  --grid
```

One-year grid result summary:

- all 24 tested combinations remained profitable after realistic constraints
- best result:
  - `expected_return_10d_top10_pos10_close_10d_turnover50000000`
  - `final_assets=2,934,309.80`
  - `total_profit=1,934,309.80`
  - `total_return=1.9343`
  - `total_trades=239`
  - `win_rate=0.6904`
  - `profit_factor=6.3013`
  - `max_drawdown=-0.0190`
- weakest result:
  - `ml_score_top10_pos5_close_20d_turnover50000000`
  - `final_assets=1,469,181.83`
  - `total_profit=469,181.83`
  - `total_return=0.4692`
  - `total_trades=60`
  - `win_rate=0.7000`
  - `profit_factor=4.7495`

Liquidity filter notes:

- raising `min_turnover_value` from 50,000,000 to 100,000,000 increased
  liquidity rejects for every ranking
- for the best configuration, liquidity rejects increased from 52 to 101 and
  total profit decreased by 256,412.24
- stricter liquidity did not eliminate profitability, but it usually reduced
  trade count or total profit

Interpretation:

- Phase 18's idealized result survives a first round of practical constraints
- `expected_return_10d` remains the strongest ranking under both 5-position and
  10-position caps
- max-position capacity matters: 10 positions captured materially more return
  than 5 positions in this one-year run
- these are still after-the-fact report simulations, not production trading
  rules

## Phase 20: Expanding-Window Walk-Forward Evaluation

Files:

- `src/ml/walk_forward.py`
- `scripts/ml/run_walk_forward.py`
- `tests/test_ml_walk_forward.py`

Main class:

- `MLWalkForwardRunner`

Purpose:

- reduce future-data leakage and overfitting concerns from the single split
  validation
- train a new model for each test month using only past data
- evaluate `expected_return_10d_top10` with `close_10d` paper exits
- keep existing trading logic and existing backtests unchanged

Fold design:

- train start is fixed at `2025-06-01`
- test months are `2026-01` through `2026-05`
- each fold expands the train window to the month before the test month
- to avoid boundary-label leakage, the effective training end is capped to the
  date whose 20-business-day labels would already be knowable by the requested
  train end
- fold-specific models are saved under `models/ml/walk_forward/`
- fold predictions are saved under `data/ml/walk_forward_predictions/`

CLI:

```bash
python3 scripts/ml/run_walk_forward.py \
  --train-start 2025-06-01 \
  --test-start 2026-01-01 \
  --test-end 2026-05-31 \
  --ranking expected_return_10d \
  --top-n 10 \
  --exit-rule close_10d
```

Outputs:

- `reports/ml/walk_forward_2026-01_to_2026-05.md`
- `reports/ml/walk_forward_2026-01_to_2026-05.json`

Walk-forward result summary:

Overall:

- `total_trades=810`
- `win_rate=0.5802`
- `total_return=37.9293`
- `profit_factor=2.2779`
- `max_drawdown=-0.2968`

Monthly folds:

| month | train_rows | valid_rows | predicted_dates | trades | monthly_return | win_rate | profit_factor | max_drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-01 | 166698 | 56374 | 19 | 190 | 6.9671 | 0.6211 | 1.8593 | -0.2647 |
| 2026-02 | 219006 | 81799 | 18 | 166 | 7.4820 | 0.5723 | 2.1097 | -0.1950 |
| 2026-03 | 292586 | 82151 | 21 | 164 | 3.4480 | 0.5244 | 1.5796 | -0.4224 |
| 2026-04 | 378828 | 67185 | 21 | 210 | 21.5218 | 0.6714 | 5.4678 | -0.0391 |
| 2026-05 | 447580 | 31610 | 18 | 80 | -1.4897 | 0.3750 | 0.6334 | -0.1672 |

Interpretation:

- the walk-forward result remains positive overall, but it is much weaker and
  more uneven than the Phase 18/19 after-the-fact paper simulations
- May 2026 is negative under this strict fold setup
- this reduces confidence that the Phase 18/19 strength was fully robust, and
  suggests future work should focus on fold-stable ranking signals, transaction
  assumptions, and out-of-sample month-by-month risk control

## Phase 21: Walk-Forward Losing-Month Diagnostics

Files:

- `src/ml/walk_forward_diagnostics.py`
- `scripts/ml/analyze_walk_forward_diagnostics.py`
- `tests/test_ml_walk_forward_diagnostics.py`

Purpose:

- analyze why the 2026-05 walk-forward fold lost money
- reuse the existing walk-forward result without retraining
- join walk-forward predictions, labels, top10 paper trades, sector/market
  metadata, and TOPIX proxy data
- keep existing trading logic unchanged

CLI:

```bash
python3 scripts/ml/analyze_walk_forward_diagnostics.py \
  --walk-forward-json reports/ml/walk_forward_2026-01_to_2026-05.json \
  --start 2026-01-01 \
  --end 2026-05-31
```

Outputs:

- `reports/ml/walk_forward_diagnostics_2026-01_to_2026-05.md`
- `reports/ml/walk_forward_diagnostics_2026-01_to_2026-05.json`
- `reports/ml/walk_forward_losing_trades_2026-05.csv`

Diagnostics included:

- monthly ML prediction distributions
- monthly realized label distributions
- monthly top10 trade summaries
- monthly top10 trade details
- 2026-05 losing trades top 20
- code, sector, and market concentration
- TOPIX 10d/20d return proxy

2026-05 result:

- top10 trades:
  - `trades=80`
  - `win_rate=0.3750`
  - `return_sum=-1.4897`
  - `profit_factor=0.6334`
  - `bad_entry_rate=0.8000`
- top10 average predictions:
  - `expected_return_10d_mean=0.0443`
  - `bad_entry_probability_10d_mean=0.5739`
  - `expected_max_return_20d_mean=0.1920`
  - `swing_success_probability_20d_mean=0.6059`
- all-stock realized labels in 2026-05:
  - `future_10d_return_mean=0.0029`
  - `bad_entry_10d_rate=0.4411`
- concentration:
  - `unique_codes=41` across 80 trades
  - repeated names included `68570:7`, `68710:6`, `80350:5`, `69200:5`
  - sector concentration was high in `電気機器` with 41 of 80 trades
  - all 80 trades were `プライム`
- TOPIX proxy:
  - `month_return=0.0613`
  - `avg_10d=0.0198`
  - `avg_20d=0.0382`

Interpretation:

- 2026-05 was not simply a broad market selloff; TOPIX was positive
- the all-stock universe was weaker and riskier than January, but the selected
  top10 trades were much worse than the universe average
- the strongest warning sign was top10 `bad_entry_rate=0.80`, despite the
  model still assigning high expected returns and high swing probabilities
- there was notable concentration in electrical equipment names
- next useful checks are bad-entry-aware ranking, sector caps, duplicate/code
  caps, and ranking formulas that penalize high `bad_entry_probability_10d`

## Phase 22: Bad-Entry-Aware Walk-Forward Ranking Comparison

Files:

- `src/ml/walk_forward_ranking_compare.py`
- `scripts/ml/compare_walk_forward_rankings.py`
- `tests/test_ml_walk_forward_ranking_compare.py`

Purpose:

- reuse existing 2026-01 to 2026-05 walk-forward predictions
- compare ranking formulas and simple filters without retraining
- test whether 2026-05 can be improved by penalizing `bad_entry_probability_10d`
- keep existing trading logic and existing backtests unchanged

Compared strategies:

- `expected_return_10d`
- `risk_adjusted_return`
- `return_upside_combo`
- `swing_combo`
- `expected_return_10d_bad_entry_lt_0_60`
- `expected_return_10d_bad_entry_lt_0_70`
- `expected_return_10d_sector_cap_3`
- `expected_return_10d_bad_entry_lt_0_70_sector_cap_3`

Formulas:

- `risk_adjusted_return = expected_return_10d - 0.5 * bad_entry_probability_10d`
- `return_upside_combo = expected_return_10d + 0.5 * upside_probability_10d - 0.5 * bad_entry_probability_10d`
- `swing_combo = expected_return_10d + 0.5 * swing_success_probability_20d - 0.5 * bad_entry_probability_10d`

CLI:

```bash
python3 scripts/ml/compare_walk_forward_rankings.py \
  --start 2026-01-01 \
  --end 2026-05-31 \
  --top-n 10 \
  --exit-rule close_10d
```

Outputs:

- `reports/ml/walk_forward_ranking_compare_2026-01_to_2026-05.md`
- `reports/ml/walk_forward_ranking_compare_2026-01_to_2026-05.json`

Result summary:

| strategy | total_return | win_rate | profit_factor | max_drawdown | bad_entry_rate | 2026-05 return | 2026-05 PF | 2026-05 bad_entry_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expected_return_10d | 34.8503 | 0.5552 | 2.0500 | -0.4041 | 0.5851 | -1.4897 | 0.6334 | 0.8000 |
| risk_adjusted_return | -0.4723 | 0.4338 | 0.9347 | -0.2835 | 0.0725 | 0.3380 | 1.9765 | 0.0380 |
| return_upside_combo | 4.7147 | 0.4402 | 1.1627 | -0.4991 | 0.4954 | 0.6660 | 1.2365 | 0.5875 |
| swing_combo | 1.2360 | 0.4448 | 1.0366 | -0.8766 | 0.5253 | -0.5614 | 0.8480 | 0.6375 |
| expected_return_10d_bad_entry_lt_0_60 | 24.8456 | 0.5563 | 1.7752 | -0.4420 | 0.5655 | -1.2035 | 0.6503 | 0.7625 |
| expected_return_10d_bad_entry_lt_0_70 | 34.6071 | 0.5609 | 2.0519 | -0.3974 | 0.5793 | -1.3137 | 0.6702 | 0.8000 |
| expected_return_10d_sector_cap_3 | 33.4656 | 0.5552 | 1.9909 | -0.4138 | 0.5828 | -1.7661 | 0.5608 | 0.8250 |
| expected_return_10d_bad_entry_lt_0_70_sector_cap_3 | 33.1326 | 0.5563 | 1.9845 | -0.4168 | 0.5759 | -1.4502 | 0.6325 | 0.8125 |

Interpretation:

- `expected_return_10d_bad_entry_lt_0_70` is the best conservative candidate:
  it preserves nearly all baseline return while slightly improving PF and DD
- `risk_adjusted_return` fixes 2026-05 but destroys the overall edge, so the
  bad-entry penalty is too strong for a primary ranking
- `return_upside_combo` also fixes 2026-05, but total return and PF are much
  weaker than baseline
- sector cap 3 did not help in this run; it worsened 2026-05 and slightly
  reduced total return
- useful next experiments are softer bad-entry penalties, top-N blending, and
  bad-entry thresholds around 0.70 rather than hard low-risk-only ranking

## Phase 23: Daily AI Candidate Export

Files:

- `src/ml/daily_candidates.py`
- `scripts/ml/export_daily_ai_candidates.py`
- `tests/test_ml_daily_candidates.py`

Main class:

- `DailyAICandidateExporter`

Purpose:

- export a human-readable daily candidate list selected by AI predictions
- keep output report-only; no orders are placed
- avoid connecting to existing strategy or backtest trading logic

Selection rule:

- ranking: `expected_return_10d`
- filter: `bad_entry_probability_10d < 0.70`
- liquidity: `turnover_value >= 50,000,000`
- top_n: 10
- exit assumption: `close_10d`

Inputs:

- `data/ml/predictions/predictions_YYYY-MM-DD.parquet`
- `data/ml/features/features_YYYY-MM-DD.parquet`
- optional `data/cache/jquants/listed_info/*.json` for name, market, and sector

Outputs:

- `reports/ml/daily_candidates/ai_candidates_YYYY-MM-DD.md`
- `reports/ml/daily_candidates/ai_candidates_YYYY-MM-DD.csv`

CLI:

```bash
python3 scripts/ml/export_daily_ai_candidates.py \
  --date 2026-05-15 \
  --top-n 10 \
  --min-turnover-value 50000000 \
  --max-bad-entry-probability 0.70
```

Output columns:

- `rank`
- `date`
- `code`
- `name`
- `market`
- `sector_name`
- `close`
- `turnover_value`
- `expected_return_10d`
- `expected_max_return_20d`
- `swing_success_probability_20d`
- `bad_entry_probability_10d`
- `entry_risk_label`
- `ml_score`
- `reason`

2026-05-15 sample candidates:

| rank | code | name | expected_return_10d | bad_entry_probability_10d | turnover_value |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | 69620 | 大真空 | 0.0538 | 0.4834 | 473,946,300 |
| 2 | 65080 | 明電舎 | 0.0533 | 0.5400 | 3,890,531,000 |
| 3 | 69540 | ファナック | 0.0489 | 0.6670 | 120,932,628,300 |
| 4 | 57140 | ＤＯＷＡホールディングス | 0.0413 | 0.5601 | 18,326,731,500 |
| 5 | 68750 | メガチップス | 0.0394 | 0.6612 | 3,423,775,000 |
| 6 | 64900 | ＰＩＬＬＡＲ | 0.0368 | 0.6080 | 2,749,438,000 |
| 7 | 40220 | ラサ工業 | 0.0342 | 0.4506 | 9,261,927,100 |
| 8 | 40820 | 第一稀元素化学工業 | 0.0335 | 0.5089 | 2,582,755,100 |
| 9 | 69410 | 山一電機 | 0.0334 | 0.5427 | 6,582,607,000 |
| 10 | 35910 | ワコールホールディングス | 0.0329 | 0.4169 | 1,211,095,300 |

## Phase 24: Daily Pipeline Candidate Export Integration

Files:

- `src/ml/pipeline.py`
- `scripts/ml/daily_pipeline.py`
- `tests/test_ml_pipeline.py`

Purpose:

- automatically export daily AI candidates after daily prediction
- keep the export report-only; no orders are placed
- allow disabling candidate export with a CLI option

Default candidate export rule:

- ranking: `expected_return_10d`
- filter: `bad_entry_probability_10d < 0.70`
- liquidity: `turnover_value >= 50,000,000`
- `top_n=10`
- exit assumption: `close_10d`

CLI:

```bash
python3 scripts/ml/daily_pipeline.py \
  --date 2026-05-15 \
  --top-n 10 \
  --min-turnover-value 50000000 \
  --max-bad-entry-probability 0.70
```

Disable candidate export:

```bash
python3 scripts/ml/daily_pipeline.py \
  --date 2026-05-15 \
  --no-export-candidates
```

Pipeline return keys:

- `features_path`
- `predictions_path`
- `candidate_csv_path`
- `candidate_md_path`
- `labels_paths`
- `warnings`

Behavior:

- if current models are missing, prediction is skipped as before
- if prediction is skipped, candidate export is also skipped with a warning
- if prediction/features parquet are missing or empty, candidate export is
  skipped with a warning

2026-05-15 run:

- `features_path=data/ml/features/features_2026-05-15.parquet`
- `predictions_path=data/ml/predictions/predictions_2026-05-15.parquet`
- `candidate_csv_path=reports/ml/daily_candidates/ai_candidates_2026-05-15.csv`
- `candidate_md_path=reports/ml/daily_candidates/ai_candidates_2026-05-15.md`
- `labels_paths=[data/ml/labels/labels_2026-04-13.parquet]`
