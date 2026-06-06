# Storage Cleanup Delete Result

Generated: 2026-06-06

Scope: `/Users/negishi/work/ai-fund-lab`

## Summary

Capacity cleanup was executed for generated logs, old experiment outputs, intermediate prediction outputs, and non-finalist processed profile outputs.

Explicitly not deleted:

- `data/processed/common`
- `storage/ai_fund_lab.sqlite3`
- `data/raw`
- `data/cache`
- `data/ml/features`
- `data/ml/labels`
- `data/ml/datasets`
- `models`
- `src`
- `tests`
- `config`
- `.git`
- `.venv`

## Size Before

```text
du -sh .   -> 45G
du -shl .  -> 93G
```

Top-level physical usage before deletion:

```text
100K    ./.pytest_cache
392K    ./config
11M     ./articles
2.8M    ./tests
212M    ./models
256K    ./docs
1.2G    ./storage
5.3G    ./logs
494M    ./.venv
264K    ./scripts
4.0K    ./.github
628M    ./analysis_logs
489M    ./.git
36G     ./data
384M    ./reports
5.5M    ./src
45G     .
```

Key before-deletion directories:

```text
615M    data/cache
1.7G    data/ml
34G     data/processed
491M    data/raw
36G     data
535M    logs/backtests
4.6G    logs/scoring
153M    logs/screening
5.3G    logs
258M    reports/ml
384M    reports
212M    models/ml
212M    models
```

## Deleted Categories

- Generated `reports/ml` CSV files.
- Old experiment reports, if present.
- Old large backtest logs, if present.
- Non-finalist `logs/scoring` directories.
- Disposable ML prediction outputs.
- Non-finalist `data/processed/rookie_dealer_02_v2*` profile output directories.

## Deleted Major Paths

### `reports/ml` CSV

```text
reports/ml/portfolio_manager_phase1_trade_allocations_2023-01_to_2026-05.csv
reports/ml/portfolio_manager_phase1_daily_allocations_2023-01_to_2026-05.csv
```

### Old Experiment Reports

```text
reports/experiments
```

This path did not exist at dry-run time, so no bytes were removed from it in this run.

### Old Backtest Logs

```text
logs/backtests/rookie_dealer_02_v2_1
logs/backtests/rookie_dealer_02_v2_6
logs/backtests/rookie_dealer_02_v2_26
logs/backtests/rookie_dealer_02_v2_38
logs/backtests/rookie_dealer_02_v2_60
logs/backtests/rookie_dealer_02_v2_61
logs/backtests/rookie_dealer_02_v2_69
```

These paths did not exist at dry-run time, so no bytes were removed from them in this run.

### Non-Finalist `logs/scoring`

Removed all first-level directories under `logs/scoring` except the keep-list profiles. After deletion, the only remaining `logs/scoring` directory is:

```text
logs/scoring/rookie_dealer_02_v2_65
```

### Disposable ML Prediction Outputs

```text
data/ml/walk_forward_predictions
data/ml/walk_forward_predictions copy
data/ml/predictions
```

Dry-run sizes:

```text
126M    data/ml/walk_forward_predictions
55M     data/ml/walk_forward_predictions copy
46M     data/ml/predictions
```

### Non-Finalist `data/processed` Profile Outputs

Removed these non-finalist `data/processed/rookie_dealer_02_v2*` directories:

```text
data/processed/rookie_dealer_02_v2_68_ml_ranked_exit_ai_050
data/processed/rookie_dealer_02_v2_69_ml_ranked_exit_ai_055
data/processed/rookie_dealer_02_v2_66_ml_ranked_hold30
data/processed/rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue_asset_ratio_070
data/processed/rookie_dealer_02_v2_66_ml_ranked_hold25
data/processed/rookie_dealer_02_v2_26
data/processed/rookie_dealer_02_v2_66_ml_ranked_hold15
data/processed/rookie_dealer_02_v2_70_ml_ranked_exit_ai_060
data/processed/rookie_dealer_02_v2_74_ml_ranked_exit_ai_affordable_fallback_expected_002_bad_entry_lte_070
data/processed/rookie_dealer_02_v2_1
data/processed/rookie_dealer_02_v2_38
data/processed/rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue_fixed_500000
data/processed/rookie_dealer_02_v2_66_ml_ranked_hold5
data/processed/rookie_dealer_02_v2_74_ml_ranked_exit_ai_affordable_fallback_risk_adjusted_gte_005
data/processed/rookie_dealer_02_v2_74_ml_ranked_exit_ai_affordable_fallback_max_fallback_1_per_day
data/processed/rookie_dealer_02_v2_67_ml_standalone
data/processed/rookie_dealer_02_v2_66_ml_ranked_hold10
data/processed/rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue_fixed_1500000
data/processed/rookie_dealer_02_v2
data/processed/rookie_dealer_02_v2_66_ml_ranked_hold20
```

Largest dry-run logical sizes:

```text
4.6G    data/processed/rookie_dealer_02_v2_38
3.8G    data/processed/rookie_dealer_02_v2_26
3.4G    data/processed/rookie_dealer_02_v2_74_ml_ranked_exit_ai_affordable_fallback_risk_adjusted_gte_005
3.4G    data/processed/rookie_dealer_02_v2_74_ml_ranked_exit_ai_affordable_fallback_max_fallback_1_per_day
3.4G    data/processed/rookie_dealer_02_v2_74_ml_ranked_exit_ai_affordable_fallback_expected_002_bad_entry_lte_070
3.1G    data/processed/rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue_fixed_500000
3.1G    data/processed/rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue_fixed_1500000
3.1G    data/processed/rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue_asset_ratio_070
3.0G    data/processed/rookie_dealer_02_v2_67_ml_standalone
2.7G    data/processed/rookie_dealer_02_v2_70_ml_ranked_exit_ai_060
2.7G    data/processed/rookie_dealer_02_v2_69_ml_ranked_exit_ai_055
2.7G    data/processed/rookie_dealer_02_v2_68_ml_ranked_exit_ai_050
```

## Size After

```text
du -sh .   -> 33G
du -shl .  -> 48G
```

Exact post-delete totals:

```text
du -sk .   -> 34,478,356 KiB
du -skl .  -> 50,535,252 KiB
```

Top-level physical usage after deletion:

```text
100K    ./.pytest_cache
392K    ./config
11M     ./articles
2.8M    ./tests
212M    ./models
256K    ./docs
1.2G    ./storage
1.1G    ./logs
494M    ./.venv
264K    ./scripts
4.0K    ./.github
628M    ./analysis_logs
489M    ./.git
28G     ./data
381M    ./reports
5.5M    ./src
33G     .
```

Key after-deletion directories:

```text
615M    data/cache
1.5G    data/ml
26G     data/processed
491M    data/raw
28G     data
535M    logs/backtests
390M    logs/scoring
153M    logs/screening
1.1G    logs
255M    reports/ml
381M    reports
212M    models/ml
212M    models
```

## Reduction

Approximate reduction:

- Physical disk usage: **45G -> 33G**, about **12G reduced**.
- Logical / Finder-like size: **93G -> 48G**, about **45G reduced**.

Because `data/processed` uses many hard links, logical-size reduction is much larger than physical disk reduction.

## Remaining Large Areas

### `data/processed/common`

Not deleted in this run.

```text
17G     data/processed/common/indicators
17G     data/processed/common
4.3G    data/processed/common/indicators/3f7c96dc489b074d
3.7G    data/processed/common/indicators/3d22ac53b295d3a1
3.3G    data/processed/common/indicators/fdaa3dda41d370ff
2.9G    data/processed/common/indicators/24a1d45b90fb6e83
2.5G    data/processed/common/indicators/216fbbd34e3eda31
605M    data/processed/common/candidates
136M    data/processed/common/candidates/4b562b7a913497a0
110M    data/processed/common/candidates/3c093330c10d2206
107M    data/processed/common/candidates/ec78b0a4476d8fa5
107M    data/processed/common/candidates/00303ab57b291368
99M     data/processed/common/candidates/d8687be8b945845c
```

### Large Files Over 100M

```text
1.2G    ./storage/ai_fund_lab.sqlite3
413M    ./data/ml/datasets/ml_dataset.parquet
406M    ./data/ml/datasets/train.parquet
```

SQLite size:

```text
1.2G    storage/ai_fund_lab.sqlite3
```

## Next Cleanup Candidates

Next candidates, if further cleanup is approved:

- Audit `data/processed/common/indicators/*` hash directories. This is the largest remaining area at **17G**, but it was explicitly protected in this run.
- Consider whether `storage/ai_fund_lab.sqlite3` can be vacuumed, archived, or rebuilt. It was explicitly protected and not deleted.
- Review `analysis_logs` at **628M** if they are generated and not needed.
- Review old profile report directories under `reports/rookie_dealer_02_v2_*`; these are small individually but numerous.

## Verification

- `reports/ml` CSV files remaining: none found.
- Non-finalist `data/processed/rookie_dealer_02_v2*` directories remaining: none found.
- Remaining `logs/scoring` directory: `logs/scoring/rookie_dealer_02_v2_65`.
- `git status --short`: no output.

No source-code changes were detected by `git status --short`.
