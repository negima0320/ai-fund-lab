# Exit AI v2 Phase 5-A to 5-F Retraining Summary

Last updated: `2026-06-07`

This note summarizes the Exit AI v2 retraining-readiness work from Phase 5-A
through Phase 5-F.

The goal of Phase 5 was not to make a new production profile immediately. The
goal was to determine whether Exit AI can be retrained safely from API-origin
market data only, build the dataset, design a leakage-safe trainer, and train a
candidate model without touching the current deployed Exit AI.

## Current Decision

Exit AI v2 has a trained candidate model:

```text
models/ml/exit_ai_v2/candidate_v2_api_only/
```

The current Exit AI was not overwritten:

```text
models/ml/exit/current_v2_66/
```

No full backtest, profile addition, or live order path has been run for Exit AI
v2 yet. The next step is an Exit AI v2 Prediction / Integration Audit, not
adoption.

## Hard Data Policy

Allowed for retraining:

- API-origin price series
- API-origin financial data
- API-derived feature datasets
- mechanical future-return labels from API-origin price paths
- walk-forward artifacts only when produced without future feature leakage

Forbidden for retraining labels/features:

- `trades.csv`
- `backtest_summary.json`
- `summary.csv` / portfolio history
- realized P/L or win/loss
- v2_75 to v2_79 trading outcomes
- selected-only backtest universe
- `selected_count_in_day`
- current-model regenerated historical predictions
- backtest/profile/result/portfolio state columns

## Phase 5-A: Retraining Readiness Audit

Report:

```text
reports/ml/phase5a_retraining_readiness_audit_2023-01_to_2026-05.md
reports/ml/phase5a_retraining_readiness_audit_2023-01_to_2026-05.json
```

| AI | current path | retraining stance |
|---|---|---|
| Stock Selection AI | `models/ml/current_enriched_v2` | do not retrain first |
| Exit AI | `models/ml/exit/current_v2_66` | highest priority, but existing dataset is not retraining-safe |
| Portfolio Manager AI | `models/ml/portfolio_manager/current_v2_73_phase3b_clean` | not first; existing dataset is backtest-derived |

Key findings:

- Exit AI had the clearest improvement opportunity after Phase 4.
- Existing Exit AI dataset contains backtest/trade-path columns such as
  `trade_id`, `actual_exit_date`, `remaining_days_to_actual_exit`, and
  `holding_days`.
- PM AI dataset contains realized outcome and portfolio/backtest columns, so it
  is not suitable as an immediate retraining source.
- Stock Selection AI is broad and risky to disturb before Exit AI is addressed.

Recommended next phase:

```text
Phase 5-B Exit AI v2 API-Only Dataset Design
```

## Phase 5-B: Exit AI v2 API-Only Dataset Design

Report:

```text
reports/ml/phase5b_exit_ai_v2_dataset_design_2021-06_to_2026-05.md
reports/ml/phase5b_exit_ai_v2_dataset_design_2021-06_to_2026-05.json
```

Base dataset:

```text
data/ml/datasets/ml_dataset.parquet
```

| item | value |
|---|---:|
| candidate rows | `2,041,709` |
| unique codes | `4,236` |
| date range | `2021-06-01 to 2026-04-27` |
| 3d label available | `2,037,003` |
| 5d label available | `2,033,858` |
| 10d label available | `2,025,987` |
| 20d label available | `2,010,219` |
| leakage risk | `low` |
| blocking issues | `none` |

Design decisions:

- use API-derived all-stock rows;
- compute future-return labels mechanically from API price paths;
- exclude existing future/target/label-like columns from features;
- do not use the existing v2_66 Exit dataset for retraining.

Recommended label:

```text
exit_quality_score
```

The label is action-aligned because higher values mean a stronger case for
exiting.

## Phase 5-C: Exit AI v2 API-Only Dataset Builder

Builder:

```text
src/ml/phase5c_exit_ai_v2_dataset_builder.py
scripts/ml/build_phase5c_exit_ai_v2_dataset.py
```

Dataset output:

```text
data/ml/exit_ai_v2/exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet
```

| item | value |
|---|---:|
| rows | `1,957,321` |
| columns | `56` |
| final feature count | `41` |
| file size | `471.66 MB` |
| parquet read | `OK` |
| leakage risk | `low` |
| blocking issues | `none` |

Split:

| split | rows | date range |
|---|---:|---|
| train | `991,902` | `2021-06-01 to 2023-12-29` |
| validation | `386,869` | `2024-01-04 to 2024-12-30` |
| test | `578,550` | `2025-01-06 to 2026-03-30` |

Labels:

| label | definition |
|---|---|
| `future_return_Nd` | close at `t+N` business days / close at `t` - 1 |
| `avoid_loss_5d` | `future_return_5d <= -0.03` |
| `miss_profit_5d` | `future_return_5d >= 0.03` |
| `exit_quality_score` | `-future_return_5d` |
| `exit_quality_score_risk_adjusted` | `-future_return_5d + abs(min(0, future_max_drawdown_5d))` |

Label distribution:

| metric | value |
|---|---:|
| `future_return_5d` mean | `0.00247` |
| `future_return_5d` median | `0.00231` |
| `future_return_5d` p10 | `-0.04509` |
| `future_return_5d` p90 | `0.05003` |
| `avoid_loss_5d` positive rate | `17.22%` |
| `miss_profit_5d` positive rate | `20.27%` |
| `exit_quality_score` mean | `-0.00247` |

High-missing features excluded at builder stage:

```text
sector_name
days_after_earnings
scale_category
credit_category
margin_category
market
days_to_earnings
PayoutRatioAnn
```

## Phase 5-D: Exit AI v2 Training Design

Report:

```text
reports/ml/phase5d_exit_ai_v2_training_design_2021-06_to_2026-05.md
reports/ml/phase5d_exit_ai_v2_training_design_2021-06_to_2026-05.json
```

Recommended task:

```text
ranking-style exit_quality_score top decile
```

Recommended feature set:

```text
feature_set_drop_missing_30pct
```

Dropped features:

```text
BPS
OP_growth
FEPS_growth
FSales_growth
FOP_growth
```

Planned feature count:

```text
36
```

Imputation policy:

- numeric: median impute fit on train fold only;
- categorical: mode impute fit on train fold only;
- apply train-fitted imputer to validation/test;
- do not impute labels;
- do not fit imputer over the full period;
- add missing indicators for features with meaningful missingness.

Fair comparison policy:

- keep `models/ml/exit/current_v2_66` unchanged;
- save Exit AI v2 under a separate candidate path;
- keep v2_78 buy/PM logic fixed;
- swap only Exit AI when integration testing begins;
- do not regenerate historical predictions with a current model;
- use walk-forward training when generating prediction artifacts.

Leakage audit:

| item | result |
|---|---|
| forbidden columns | `[]` |
| label-like columns in features | `[]` |
| `future_return_*` in features | `[]` |
| target/label in features | `[]` |
| `selected_count_in_day` | `false` |
| backtest result columns | `[]` |
| split overlap | `false` |
| leakage risk | `low` |
| blocking issues | `none` |

## Phase 5-E: Trainer Prototype

Trainer:

```text
src/ml/phase5e_exit_ai_v2_trainer.py
scripts/ml/train_phase5e_exit_ai_v2.py
```

Task:

```text
exit_quality_top_decile = 1 if exit_quality_score >= train fold 90th percentile
```

Model:

```text
sklearn.ensemble.HistGradientBoostingClassifier
```

Dry-run result:

| item | value |
|---|---:|
| rows used | `1,957,321` |
| feature count | `36` |
| trained | `false` |
| model saved | `false` |
| leakage risk | `low` |
| blocking issues | `none` |

Train-only top-decile threshold:

```text
0.046277665995975825
```

Positive rates under train threshold:

| split | positive rate |
|---|---:|
| train | `10.00%` |
| validation | `9.90%` |
| test | `8.66%` |

## Phase 5-F: Full Train

Command executed:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 scripts/ml/train_phase5e_exit_ai_v2.py --train-full
```

Model output:

```text
models/ml/exit_ai_v2/candidate_v2_api_only/
```

Saved files:

```text
exit_quality_top_decile_classifier.joblib
model_metadata.json
feature_columns.json
preprocess.json
```

Training rows:

| split | rows | positive rate |
|---|---:|---:|
| train | `991,902` | `10.00%` |
| validation | `386,869` | `9.90%` |
| test | `578,550` | `8.66%` |

Metrics:

| split | AUC | PR-AUC | precision@top10% | recall@top10% | top decile lift |
|---|---:|---:|---:|---:|---:|
| validation | `0.5737` | `0.1283` | `0.1532` | `0.1548` | `1.5476` |
| test | `0.6524` | `0.1553` | `0.1956` | `0.2257` | `2.2574` |

Calibration by decile showed monotonic lift.

Validation actual positive rate by prediction decile:

```text
7.82%, 7.61%, 7.73%, 7.92%, 8.49%, 9.34%, 10.45%, 11.65%, 12.63%, 15.32%
```

Test actual positive rate by prediction decile:

```text
3.60%, 4.61%, 5.65%, 6.07%, 6.93%, 7.64%, 8.77%, 10.54%, 13.26%, 19.56%
```

Feature importance note:

`HistGradientBoostingClassifier` does not expose impurity-based
`feature_importances_`. A post-hoc permutation importance check on the first
20,000 validation rows ranked these highest:

| rank | feature | importance |
|---:|---|---:|
| 1 | `daily_range_ratio` | `0.02022` |
| 2 | `close` | `0.01617` |
| 3 | `entry_price` | `0.00638` |
| 4 | `ma75_gap` | `0.00305` |
| 5 | `ma25_slope` | `0.00234` |
| 6 | `turnover_value` | `0.00097` |
| 7 | `return_3d` | `0.00090` |
| 8 | `Sales_growth` | `0.00039` |
| 9 | `body_ratio` | `0.00031` |
| 10 | `gap_up_ratio` | `0.00025` |

Full-train leakage check:

| item | result |
|---|---|
| forbidden columns in features | `[]` |
| label-like columns in features | `[]` |
| `future_return_*` in features | `[]` |
| target/label in features | `[]` |
| `selected_count_in_day` | `false` |
| backtest/profile columns | `[]` |
| split overlap | `false` |
| train threshold only | `true` |
| leakage risk | `low` |
| blocking issues | `none` |

## Current Recommendation

Proceed to:

```text
Exit AI v2 Prediction / Integration Audit
```

Do not yet:

- replace `models/ml/exit/current_v2_66`;
- add a production/profile switch;
- run a full strategy backtest with Exit AI v2 as if it were adopted;
- connect to live order placement.

The next audit should check whether the candidate model can generate
walk-forward-safe prediction artifacts, how its signals map onto actual Exit AI
decision points, and whether integration can be tested by changing only the
Exit AI component while keeping v2_78 buy/PM logic fixed.
