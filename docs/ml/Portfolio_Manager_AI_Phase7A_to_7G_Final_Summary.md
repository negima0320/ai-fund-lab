# Portfolio Manager AI Phase 7-A to 7-G Final Summary

This note summarizes the final AI-state audit, PM AI API-only rebuild path,
final championship check, and pytest triage after `v2_82_cap38` became the
strongest full-backtested research candidate.

## Current Decision

Version 1.0 Candidate:

```text
rookie_dealer_02_v2_82_cap38
```

Aliases:

```text
rookie_dealer_02_v2_82
rookie_dealer_02_v2.82
```

Decision:

- Use `rookie_dealer_02_v2_82_cap38` as the current Version 1.0 Candidate.
- Keep `rookie_dealer_02_v2_78_pm_aware_order_fallback_w025` as the
  conservative fallback/reference.
- Do not integrate the PM AI API-only candidate model yet.
- Do not integrate Exit AI v2 candidate profiles.
- Do not adopt Bear Booster.
- Do not overwrite current model directories.

## Phase 7-A: Full AI State and Retraining Readiness Audit

Report:

```text
reports/ml/phase7a_full_ai_state_audit_2021_to_2026.md
```

Main findings:

| AI | current state | dataset rows | features | leakage risk | retraining recommendation |
|---|---|---:|---:|---|---|
| Stock Selection AI | current enriched model available | `2,041,709` | `48` | low | defer; safe but high risk to disturb |
| Portfolio Manager AI | current v2_73 clean model operational | `1,375` | `68` | high for direct retraining | rebuild dataset first |
| Exit AI current v2_66 | operational but old dataset unsafe for retraining | `1,194` | `18` | high | defer |
| Exit AI v2 candidate | API-only candidate trained | `1,957,321` | `36` | low | integration redesign first |

Final judgement:

- `all_ai_retraining_ready=false`
- first target should be API-only dataset redesign, especially PM AI labels
- recommended next phase was PM AI leakage forensics / API-only rebuild

Important constraint:

- Backtest outcomes, trades.csv, realized profit, win/loss, portfolio history,
  and selected_count_in_day must not be used as retraining labels.

## Phase 7-B and 7-B': PM AI Leakage Forensics

Reports:

```text
reports/ml/phase7b_pm_ai_leakage_forensics_2023-01_to_2026-05.md
reports/ml/phase7b_prime_pm_ai_leakage_fix_2023-01_to_2026-05.md
```

The first audit was intentionally strict and flagged possible leakage. The
follow-up fixed false positives such as `close_position`, which is a price
feature rather than portfolio position state.

Final 7-B' judgement:

| item | value |
|---|---|
| feature_leakage_confirmed | `false` |
| feature_leakage_suspected | `true` |
| feature_leakage_not_confirmed | `true` |
| current_pm_model_safe_to_use | `true` |
| v2_82_result_trust_level | `medium_trust` |
| pm_ai_direct_retraining_allowed | `false` |
| pm_ai_dataset_rebuild_required | `true` |

Candidate-list dependent features were not treated as immediate production
leakage, but they were judged unsuitable for direct PM AI retraining:

- `candidate_count_in_day`
- `rank_in_day`
- `score_rank_in_day`
- `day_avg_*`
- `day_max_*`
- `*_percentile_in_day`
- `*_gap_to_best`

`candidate_count_in_day` was a top feature in the current PM model, so a clean
API-only rebuild was required before retraining.

## Phase 7-C: PM AI API-Only Dataset Design

Report:

```text
reports/ml/phase7c_pm_ai_api_only_dataset_design_2021_to_2026.md
```

Result:

| item | value |
|---|---|
| api_only_pm_dataset_feasible | `true` |
| candidate_feature_removal_required | `true` |
| recommended_retraining_plan | `Plan B: PM AI API-only rebuild + complete candidate-list feature removal` |
| recommended_feature_set | `api_price_financial_market_plus_stock_walk_forward_predictions_no_candidate_list_features` |
| ready_for_phase7d | `true` |

The design explicitly banned candidate-list dependent features and portfolio
state columns such as `max_positions_remaining_before`.

Recommended labels:

- `future_5d_return`
- `future_10d_return`
- `future_max_return_20d`
- `future_max_drawdown_20d`
- `risk_adjusted_future_return`
- `high_conviction_target`
- `avoid_target`

## Phase 7-D: PM AI API-Only Dataset Builder

Report:

```text
reports/ml/phase7d_pm_ai_api_only_dataset_builder_2021_to_2026.md
```

Full dataset:

```text
data/ml/portfolio_manager_api_only/pm_ai_api_only_dataset_2021-06_to_2026-05.parquet
```

Full dataset result:

| item | value |
|---|---:|
| total_rows | `2,041,709` |
| final_rows | `1,999,421` |
| dropped_rows | `42,288` |
| final_feature_count | `40` |
| final_label_count | `5` |
| train rows | `991,902` |
| validation rows | `386,869` |
| test rows | `620,650` |

High-missing features removed from the final feature set:

- `days_to_earnings`
- `PayoutRatioAnn`

Leakage result:

- leakage risk: low
- blocking issues: none
- `ready_for_phase7e=true`

## Phase 7-E: PM AI API-Only Trainer

Report:

```text
reports/ml/phase7e_pm_ai_api_only_trainer_2021_to_2026.md
```

Candidate model output:

```text
models/ml/portfolio_manager/candidate_v2_api_only
```

Current PM AI was not overwritten:

```text
models/ml/portfolio_manager/current_v2_73_phase3b_clean
```

Training setup:

- features: Phase 7-D API-only `feature_columns`
- feature count: `40`
- primary targets:
  - `high_conviction_target`
  - `avoid_target`
- model:
  - `sklearn.ensemble.HistGradientBoostingClassifier`
- candidate-list dependent features: forbidden
- leakage risk: low
- blocking issues: none

Full-train metrics:

| target | validation AUC | test AUC | validation PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| high_conviction_target | `0.6527` | `0.6472` | `0.2397` | `0.2032` |
| avoid_target | `0.5483` | `0.6345` | `0.2424` | `0.2641` |

Interpretation:

- The API-only PM candidate is technically trainable and leakage-safe.
- It is not integrated into v2_82.
- It needs a separate integration audit before any profile change.

## Phase 7-F: Final Championship Audit

Report:

```text
reports/final/v2_82_cap38/final_summary.md
reports/final/v2_82_cap38/final_summary.json
```

Snapshot:

```text
reports/final/v2_82_cap38/core_2023-01_to_2026-05/
```

Core period:

```text
2023-01-01 to 2026-05-31
```

Core comparison:

| metric | v2_78 | v2_82 cap38 | delta |
|---|---:|---:|---:|
| net_profit | `3,054,794` | `3,777,545` | `+722,751` |
| PF | `2.6194` | `2.7309` | `+0.1115` |
| DD | `-7.47%` | `-6.54%` | `+0.93pt` |
| win_rate | `53.78%` | `55.11%` | `+1.33pt` |
| monthly_win_rate | `75.61%` | `78.05%` | `+2.44pt` |
| total_trades | `502` | `499` | `-3` |
| average_holding_days | `3.97` | `4.01` | `+0.04` |
| average_capital_utilization | `38.31%` | `40.49%` | `+2.18pt` |
| final_assets | `4,813,588` | `5,720,597` | `+907,010` |
| CAGR | `58.51%` | `66.74%` | `+8.23pt` |

v2_82 capital constraints:

| metric | value |
|---|---:|
| selected_but_not_affordable | `257` |
| insufficient_available_cash | `71` |
| per_code_cap_skip_or_reduction_count | `188` |
| per_code_cap_skip_count | `16` |
| per_code_cap_reduction_amount | `80,709,710` |

Concentration:

| metric | value |
|---|---:|
| single-code abs profit concentration | `6.83%` |
| top5 abs profit concentration | `14.12%` |
| top10 abs profit concentration | `18.44%` |

Max DD:

| item | value |
|---|---|
| max DD | `-6.54%` |
| start | `2023-03-06` |
| trough | `2023-04-25` |
| recovery | `2023-05-30` |
| recovery days | `35` |

Extended period `2021-06-01 to 2026-05-31` was not used as the decisive
comparison because Stock Selection walk-forward prediction files begin at
`2023-01-04`. Therefore, 2021-2022 cannot be treated as the same-condition
championship comparison without rebuilding historical walk-forward predictions.

Phase 7-F verdict:

| item | value |
|---|---|
| production_candidate | `true` |
| recommended_profile | `rookie_dealer_02_v2_82_cap38` |
| confidence_level | `medium` |
| fix_recommended | `true` |

## Phase 7-G: Final Test Failure Triage

Report:

```text
reports/ml/phase7g_final_test_failure_triage.md
reports/ml/phase7g_final_test_failure_triage.json
```

Initial full pytest:

| result | count |
|---|---:|
| passed | `811` |
| failed | `14` |
| warnings | `15` |

Failure categories:

- earnings filter expectation mismatch
- SELL trade `market_regime` inheritance
- J-Quants / no-data cache tests under temporary ROOT
- operations docs missing expected sections

Fixes:

- `market_filter.enabled=false` now disables market-section filtering in scoring.
- Bear PM booster disabled state no longer overwrites `market_regime` with an
  empty string.
- SELL rows inherit buy-time `market_regime` via `entry_market_regime`.
- Price history fetch uses `prime_codes` fallback only when the active listed
  stock master file is missing.
- Operations docs now include Japanese runbook sections and safety/schedule
  guidance.

Final full pytest:

```text
825 passed, 15 warnings
```

Phase 7-G verdict:

- all failures fixed
- Version 1.0 Candidate can proceed
- v2_82 final result impact risk: low
- no full backtest rerun was required after the test triage

## Current Ranking After Phase 7-G

| priority | profile / model | status |
|---:|---|---|
| 1 | `rookie_dealer_02_v2_82_cap38` | Version 1.0 Candidate; strongest full-backtested profile |
| 2 | `rookie_dealer_02_v2_78_pm_aware_order_fallback_w025` | conservative fallback/reference |
| candidate model | `models/ml/portfolio_manager/candidate_v2_api_only` | trained, leakage-safe, not integrated |
| rejected | v2_80 Exit AI v2 gate profiles | underperformed v2_78 |
| deferred | v2_81 Bear Booster | booster fired but cap absorbed effective benefit |
| on hold | v2_79 high-PM minimum hold | numerical improvement was side-effect path, not direct minimum-hold effect |

## Next Recommended Work

Recommended next phase:

1. Freeze v2_82 cap38 as Version 1.0 Candidate.
2. Keep v2_78 as fallback.
3. Do not integrate PM AI API-only candidate until a dedicated integration
   audit proves it improves v2_82.
4. If robustness beyond 2023 is needed, rebuild true 2021-2022 walk-forward
   predictions first.
5. Consider PM AI API-only integration audit as the next research step, not as
   a Version 1.0 blocker.
