# Storage Cleanup Result - 2026-06-06

## Summary

- Scope: safe deletion only, based on the requested cleanup targets.
- Note: `storage_cleanup_audit.md` was not found in the repository root during execution. The cleanup proceeded using the explicit targets from the instruction.
- Deleted:
  - 26 generated CSV files under `reports/ml`
  - 7 old backtest log directories under `logs/backtests`
  - `reports/experiments`
- Not touched:
  - `data/raw`
  - `data/cache`
  - `data/ml/features`
  - `data/ml/labels`
  - `data/ml/datasets`
  - `data/processed`
  - `models`
  - `src`
  - `tests`
  - `config`
  - `.git`
  - `.venv`
  - `storage/ai_fund_lab.sqlite3`
  - `logs/scoring`
  - `data/ml/walk_forward_predictions`

## Pre-Cleanup Size Snapshot

### `du -hd 1 .`

```text
100K	./.pytest_cache
392K	./config
 11M	./articles
2.8M	./tests
212M	./models
248K	./docs
1.2G	./storage
7.5G	./logs
494M	./.venv
260K	./scripts
4.0K	./.github
628M	./analysis_logs
489M	./.git
 36G	./data
419M	./reports
5.5M	./src
 47G	.
```

### `du -hd 1 data logs reports models`

```text
615M	data/cache
1.7G	data/ml
 34G	data/processed
491M	data/raw
 36G	data
152K	logs/reflections
2.7G	logs/backtests
816K	logs/trades
 24M	logs/market_context
  0B	logs/ai_decision
 96K	logs/safety
4.6G	logs/scoring
460K	logs/portfolio
153M	logs/screening
7.5G	logs
278M	reports/ml
 15M	reports/experiments
419M	reports
212M	models/ml
212M	models
```

## Dry Run

### `find reports/ml -type f -name '*.csv' -print`

```text
reports/ml/ml_realistic_trades_5y_enriched_2023-01-01_to_2026-05-31.csv
reports/ml/ml_exit_analysis_trades_v2_66_2023-01_to_2026-05.csv
reports/ml/exit_ai_trigger_trades_v2_68_2023-01_to_2026-05.csv
reports/ml/position_sizing_phase1_trades_2023-01_to_2026-05.csv
reports/ml/v2_74_fallback_trades_2023-01_to_2026-05.csv
reports/ml/ml_paper_trades_2025-06-01_to_2026-05-31.csv
reports/ml/ml_backtest_diagnostics_2023-01_to_2026-05_monthly.csv
reports/ml/walk_forward_losing_trades_2026-05.csv
reports/ml/v2_66_holding_period_sensitivity_2023-01_to_2026-05_trades.csv
reports/ml/daily_candidates/ai_candidates_2026-05-15.csv
reports/ml/daily_candidates/2026-05-15.csv
reports/ml/ml_ranking_details_2025-06-01_to_2026-05-31.csv
reports/ml/realistic_portfolio_bought_vs_rejected_5y_enriched.csv
reports/ml/capital_allocation_phase6_daily_buy_limit_sensitivity_2023-01_to_2026-05_summary.csv
reports/ml/ml_realistic_trades_5y_enriched_v2_2023-01-01_to_2026-05-31.csv
reports/ml/backtest_ml_trades_rookie_dealer_02_v2_65_2025-06-01_to_2026-05-31.csv
reports/ml/position_sizing_phase2_soft_rules_2023-01_to_2026-05_summary.csv
reports/ml/ml_realistic_trades_2023-01-01_to_2026-05-31.csv
reports/ml/ml_backtest_diagnostics_2023-01_to_2026-05_code.csv
reports/ml/scaled_buy_trades_2023-01_to_2026-05.csv
reports/ml/capital_allocation_phase8_fallback_filter_2023-01_to_2026-05_summary.csv
reports/ml/exit_avoid_loss_simulation_trades_v2_66_2023-01_to_2026-05.csv
reports/ml/exit_ai_trade_delta_v2_66_vs_v2_68_2023-01_to_2026-05.csv
reports/ml/ml_realistic_trades_2025-06-01_to_2026-05-31.csv
reports/ml/scaled_buy_audit_trades_v2_71_2023-01_to_2026-05.csv
reports/ml/ml_realistic_trades_5y_enriched_v2.csv
```

CSV target total from `du -ch`: `23M`.

### Backtest Logs And Experiments

```text
474M	logs/backtests/rookie_dealer_02_v2_1
479M	logs/backtests/rookie_dealer_02_v2_6
268M	logs/backtests/rookie_dealer_02_v2_26
404M	logs/backtests/rookie_dealer_02_v2_38
 63M	logs/backtests/rookie_dealer_02_v2_60
 91M	logs/backtests/rookie_dealer_02_v2_61
446M	logs/backtests/rookie_dealer_02_v2_69
 15M	reports/experiments
```

## Deletion Notes

- CSV files were removed with explicit file paths.
- Backtest log directories and `reports/experiments` were removed with explicit directory paths.
- `logs/backtests/rookie_dealer_02_v2_26` initially left a `.DS_Store`; that file and then the empty directory were removed explicitly.

## Post-Cleanup Size Snapshot

### `du -hd 1 .`

```text
100K	./.pytest_cache
392K	./config
 11M	./articles
2.8M	./tests
212M	./models
248K	./docs
1.2G	./storage
5.3G	./logs
494M	./.venv
260K	./scripts
4.0K	./.github
628M	./analysis_logs
489M	./.git
 36G	./data
381M	./reports
5.5M	./src
 45G	.
```

### `du -hd 1 data logs reports models`

```text
615M	data/cache
1.7G	data/ml
 34G	data/processed
491M	data/raw
 36G	data
152K	logs/reflections
535M	logs/backtests
816K	logs/trades
 24M	logs/market_context
  0B	logs/ai_decision
 96K	logs/safety
4.6G	logs/scoring
460K	logs/portfolio
153M	logs/screening
5.3G	logs
255M	reports/ml
381M	reports
212M	models/ml
212M	models
```

## Result

- Repository total: `47G` -> `45G`
- `logs`: `7.5G` -> `5.3G`
- `logs/backtests`: `2.7G` -> `535M`
- `reports`: `419M` -> `381M`
- `reports/ml`: `278M` -> `255M`
- Estimated visible reduction from `du`: about `2.2G` in `logs` plus about `38M` in `reports`.

## Verification

- `find reports/ml -type f -name '*.csv' -print` returned no files after cleanup.
- All specified old backtest log directories were absent after cleanup.
- `reports/experiments` was absent after cleanup.
- Protected directories and files listed in the request still existed after cleanup.
