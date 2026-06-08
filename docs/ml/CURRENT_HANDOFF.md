# Current ML Handoff

Last updated: `2026-06-08`

This document is the short handoff for continuing the AI / ML work in a fresh
chat. It intentionally summarizes only the current state, key constraints, and
next useful actions. For the full history, see
`docs/ml/ML_Phase_25_to_Portfolio_Manager_AI_Summary.md`,
`docs/ml/Exit_AI_v2_Phase5A_to_5F_Retraining_Summary.md`,
`docs/ml/Portfolio_Manager_AI_Phase7A_to_7G_Final_Summary.md`, and
`docs/ml/Portfolio_Manager_AI_Phase8A_to_8H_PM_AI_Redesign_Summary.md`.

## Current State

The latest full-backtested Version 1.0 Candidate remains:

```text
rookie_dealer_02_v2_82_cap38
```

Important reference profiles are:

```text
rookie_dealer_02_v2_82_cap38
rookie_dealer_02_v2_78_pm_aware_order_fallback_w025
rookie_dealer_02_v2_79_high_pm_min_hold_5d
rookie_dealer_02_v2_79_high_pm_min_hold_7d
rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38
rookie_dealer_02_v2_91_pm_ai_v2_calibrated_rule_e_cap38
rookie_dealer_02_v2_92_relative_allocator_cap38
rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030
rookie_dealer_02_v2_76_pm_ai_low_score_skip
rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing
rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue
```

Profile lineage:

- v2_73: ML-ranked + Exit AI + scaled buy continue; prior baseline.
- v2_75: adds Portfolio Manager AI sizing.
- v2_76: derives from v2_75 and skips very low PM score trades.
- v2_77 cap 0.30: derives from v2_76 and adds per-code exposure cap `0.30`.
- v2_78 w0.25: derives from v2_77 cap 0.30 and adds PM-aware selected ordering plus selected fallback.
- v2_79 5d/7d: derives from v2_78 w0.25 and suppresses Exit AI early exits only for high-PM positions; numerically stronger in Phase 4-C, but held back after Phase 4-F/G because the intended minimum-hold guard did not directly fire.
- v2_82 cap38: derives from v2_78 behavior and relaxes per-code exposure cap to `0.38`; current Version 1.0 Candidate.
- v2_90: integrates API-only PM AI v2 raw; rejected because PM 1.30 disappeared and utilization collapsed.
- v2_91: calibrates PM AI v2 to recover PM 1.30 count; rejected because PM 1.30 quality did not recover.
- v2_92: rule-based same-day relative allocator using Stock Selection ranks; operationally valid but rejected after underperforming v2_82.

Current v2_82 core result:

| metric | value |
|---|---:|
| net_profit | `3,777,545` |
| PF | `2.7309` |
| DD | `-6.54%` |
| win_rate | `55.11%` |
| monthly_win_rate | `78.05%` |
| average_capital_utilization | `40.49%` |
| final_assets | `5,720,597` |
| CAGR | `66.74%` |

Phase 8 decision:

- Keep v2_82 as the Version 1.0 Candidate.
- Do not promote PM AI v2 candidate.
- Do not promote calibrated PM AI v2.
- Do not promote v2_92 relative allocator.
- Do not overwrite current PM AI or Exit AI model directories.

Portfolio Manager AI score:

```text
pm_score = high_conviction_proba - avoid_proba
```

Multiplier rule:

| condition | multiplier |
|---|---:|
| `pm_score >= 0.40` | `1.30` |
| `pm_score >= 0.20` | `1.15` |
| `pm_score >= 0.00` | `1.00` |
| `pm_score >= -0.20` | `0.80` |
| otherwise | `0.60` |

The multiplier is applied before existing constraints:

- cash
- `daily_buy_limit = 900000`
- round lot
- scaled buy
- max positions
- Exit AI

v2_76 additionally skips the lowest PM-score trades. v2_77 keeps that behavior
and adds a per-code exposure cap to address v2_76's drawdown concentration.

## Latest Commit and Working Tree

This handoff was updated after Phase 8-H and is intended to be committed with
the Phase 8 PM AI redesign / relative allocator work.

Recent active work includes:

- Phase 8-A System Understanding Audit
- Phase 8-B PM AI Candidate Integration Audit
- Phase 8-C PM AI v2 raw integration backtest
- Phase 8-D PM AI v2 calibration audit
- Phase 8-E calibrated PM AI v2 backtest
- Phase 8-F PM AI label redesign audit
- Phase 8-G ranking / relative allocation audit
- Phase 8-H rule-based relative allocator implementation and backtest

Current generated reports of interest:

```text
reports/ml/phase8a_system_understanding_audit_2023-01_to_2026-05.md
reports/ml/phase8b_pm_candidate_integration_audit_2023-01_to_2026-05.md
reports/ml/phase8c_pm_ai_v2_backtest_2023-01_to_2026-05.md
reports/ml/phase8d_pm_ai_v2_calibration_audit_2023-01_to_2026-05.md
reports/ml/phase8e_pm_ai_v2_calibrated_backtest_2023-01_to_2026-05.md
reports/ml/phase8f_pm_ai_label_redesign_audit_2023-01_to_2026-05.md
reports/ml/phase8g_pm_ai_ranking_audit_2023-01_to_2026-05.md
reports/ml/phase8h_relative_allocator_backtest_2023-01_to_2026-05.md
```

## Important Artifacts

Models:

```text
models/ml/current_enriched_v2/
models/ml/exit/current_v2_66/
models/ml/exit_ai_v2/candidate_v2_api_only/
models/ml/portfolio_manager/current_v2_73_phase3b_clean/
```

Exit AI v2 API-only dataset:

```text
data/ml/exit_ai_v2/exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet
```

Historical walk-forward predictions:

```text
data/ml/walk_forward_predictions/predictions_YYYY-MM-DD.parquet
```

Portfolio Manager clean dataset:

```text
data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet
```

Backtest logs:

```text
logs/backtests/rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue/2023-01-01_to_2026-05-31/
logs/backtests/rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing/2023-01-01_to_2026-05-31/
logs/backtests/rookie_dealer_02_v2_76_pm_ai_low_score_skip/2023-01-01_to_2026-05-31/
logs/backtests/rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030/2023-01-01_to_2026-05-31/
logs/backtests/rookie_dealer_02_v2_78_pm_aware_order_fallback_w025/2023-01-01_to_2026-05-31/
```

Key reports:

```text
reports/ml/portfolio_manager_phase3d_full_backtest_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3d_detail_audit_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3d_detail_audit_2023-01_to_2026-05.json
reports/ml/portfolio_manager_phase3f_drawdown_audit_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3g_per_code_cap_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3h_capital_utilization_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3i_candidate_pool_expansion_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3j_affordability_audit_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3k_candidate_ranking_audit_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3l_pm_aware_order_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase4b_high_pm_min_hold_audit_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase4c_high_pm_min_hold_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase4d_v278_vs_v279_diff_audit_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase4f_side_effect_audit_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase4g_exit_delay_candidate_hold_audit_2023-01_to_2026-05.md
reports/ml/phase5a_retraining_readiness_audit_2023-01_to_2026-05.md
reports/ml/phase5b_exit_ai_v2_dataset_design_2021-06_to_2026-05.md
reports/ml/phase5c_exit_ai_v2_dataset_builder_2021-06_to_2026-05.md
reports/ml/phase5d_exit_ai_v2_training_design_2021-06_to_2026-05.md
reports/ml/phase5e_exit_ai_v2_trainer_prototype_2021-06_to_2026-05.md
```

Generated `data/ml`, `models/ml`, `reports/ml`, and `logs/backtests` artifacts
are intentionally not committed when ignored.

## Latest Results

Period:

```text
2023-01-01 to 2026-05-31
```

Latest comparison:

| profile | net_profit | PF | DD | win_rate | trades |
|---|---:|---:|---:|---:|---:|
| v2_73 baseline | `1,169,366` | `1.5683` | `-19.13%` | `43.01%` | `481` |
| v2_75 | `2,679,727` | `2.2054` | `-9.17%` | `48.53%` | `546` |
| v2_76 | `3,812,364` | `2.5720` | `-19.38%` | `55.05%` | `507` |
| v2_77 cap 0.20 | `1,343,154` | `2.2436` | `-7.58%` | `52.95%` | `477` |
| v2_77 cap 0.30 | `2,914,686` | `2.5430` | `-7.54%` | `53.15%` | `511` |
| v2_78 w0.25 | `3,054,794` | `2.6194` | `-7.47%` | `53.78%` | `505` |

Interpretation:

- v2_76 has the highest profit/PF/win rate, but DD is too large.
- Phase 3-F found v2_76 DD was mainly a specific-code exposure issue.
- v2_77 cap 0.30 was the best balance before PM-aware ordering.
- v2_78 w0.25 improved net profit, PF, DD, win rate, and affordability skips
  versus v2_77 cap 0.30.
- v2_75 remains the simpler PM-sizing reference.

Phase 3-L PM-aware ordering:

| variant | net_profit | PF | DD | selected_but_not_affordable |
|---|---:|---:|---:|---:|
| v2_77 cap0.30 | `2,914,686` | `2.5430` | `-7.54%` | `369` |
| v2_78 w0.25 | `3,054,794` | `2.6194` | `-7.47%` | `253` |

Phase 4-B high-PM minimum-hold audit on v2_78 w0.25:

| rule | profit_delta | virtual PF | virtual win_rate |
|---|---:|---:|---:|
| min hold 3d | `+24,095` | `3.1407` | `68.45%` |
| min hold 5d | `+344,508` | `3.7796` | `71.43%` |
| min hold 7d | `+1,508,389` | `9.3080` | `77.38%` |

Phase 4-C ran these profiles:

```text
rookie_dealer_02_v2_79_high_pm_min_hold_5d
rookie_dealer_02_v2_79_high_pm_min_hold_7d
```

Both v2_79 variants had the same headline result:

| profile | net_profit | PF | DD | win_rate | trades |
|---|---:|---:|---:|---:|---:|
| v2_78 w0.25 | `3,054,794` | `2.6194` | `-7.47%` | `53.78%` | `505` |
| v2_79 5d/7d | `3,544,602` | `2.7219` | `-6.49%` | `55.25%` | `517` |

The minimum-hold guard only blocks Exit AI exits (`exit_ai_triggered=True`) for
`pm_multiplier >= 1.15`; stop loss, take profit, max holding, and forced exits
are not suppressed.

Phase 4-F/G decision:

- `high_pm_min_hold_blocked_exit_count=0`; minimum hold was not directly effective.
- First path divergence was `71570` on `2023-01-24`: v2_78 sold by Exit AI, v2_79 did not sell until `2023-01-25`.
- v2_79's improvement is a path-divergence side effect, not an adoptable minimum-hold result.
- Phase 4-G found blanket Exit AI 1-day delay was harmful (`profit_delta=-81,700`).
- The only positive clean rule candidate was a narrow high-PM delay: `pm_multiplier >= 1.15`, `profit_delta=+11,200`.
- Keep v2_78 w0.25 as main candidate; keep v2_79 on hold.
- See `docs/ml/Portfolio_Manager_AI_Phase4C_to_4G_Audit_Summary.md`.

## Exit AI v2 Phase 5 Result

Phase 5-A through 5-F created a separate API-only Exit AI v2 candidate.

Important distinction:

- Current Exit AI remains `models/ml/exit/current_v2_66`.
- Candidate Exit AI v2 is saved separately under
  `models/ml/exit_ai_v2/candidate_v2_api_only`.
- No production profile or full strategy backtest has adopted Exit AI v2 yet.

Dataset:

```text
data/ml/exit_ai_v2/exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet
```

Dataset summary:

| item | value |
|---|---:|
| rows | `1,957,321` |
| columns | `56` |
| final feature count before Phase 5-D drop | `41` |
| file size | `471.66 MB` |
| leakage risk | `low` |
| blocking issues | `none` |

Training design:

| item | value |
|---|---|
| task | ranking-style `exit_quality_score` top decile |
| model | `HistGradientBoostingClassifier` |
| feature set | `feature_set_drop_missing_30pct` |
| feature count | `36` |
| dropped features | `BPS`, `OP_growth`, `FEPS_growth`, `FSales_growth`, `FOP_growth` |
| threshold rule | train split 90th percentile only |

Phase 5-F full train:

| split | rows | positive rate |
|---|---:|---:|
| train | `991,902` | `10.00%` |
| validation | `386,869` | `9.90%` |
| test | `578,550` | `8.66%` |

Target threshold:

```text
0.046277665995975825
```

Metrics:

| split | AUC | PR-AUC | precision@top10% | recall@top10% | top decile lift |
|---|---:|---:|---:|---:|---:|
| validation | `0.5737` | `0.1283` | `0.1532` | `0.1548` | `1.5476` |
| test | `0.6524` | `0.1553` | `0.1956` | `0.2257` | `2.2574` |

Full-train leakage check:

- forbidden columns in features: none
- label-like columns in features: none
- `future_return_*` in features: none
- target/label in features: none
- `selected_count_in_day`: false
- backtest/profile columns: none
- split overlap: false
- train threshold only: true
- leakage risk: low
- blocking issues: none

Historical next step, now completed in the later Phase 5-G section:

```text
Exit AI v2 Prediction / Integration Audit
```

The integration audit and full profile checks are summarized below. The current
decision remains: do not replace the current Exit AI, do not overwrite
`models/ml/exit/current_v2_66`, and do not adopt Exit AI v2 profiles.

## Phase 5-G to 6-G Result

The Exit AI v2 integration audit and subsequent market/cap audits have now been
completed. The detailed summary is:

```text
docs/ml/Portfolio_Manager_AI_Phase5G_to_6G_Audit_Summary.md
```

High-level decisions:

- Do not adopt Exit AI v2 integration profiles.
- Do not adopt the Bear Booster profile yet.
- The main bottleneck was per-code exposure cap, not Exit AI v2 or Bear
  booster logic.
- Promote `rookie_dealer_02_v2_82_cap38` to the strongest current
  full-backtested research candidate.
- Keep `rookie_dealer_02_v2_78_pm_aware_order_fallback_w025` as conservative
  fallback/reference.

Phase 5-G prediction audit:

| metric | value |
|---|---:|
| v2_78 sell rows audited | `505` |
| prediction coverage | `94.65%` |
| Exit AI v2 top-decile rows | `48` |
| agreement with existing Exit AI / non-exit | `438` |
| disagreement | `67` |

Phase 5-H integration profiles:

| variant | net_profit | PF | DD | win_rate | decision |
|---|---:|---:|---:|---:|---|
| v2_78 baseline | `3,054,794` | `2.6194` | `-7.47%` | `53.78%` | reference |
| v2_80 conservative gate | `2,318,919` | `2.2956` | `-10.12%` | `53.36%` | rejected |
| v2_80 high PM safe | `2,859,266` | `2.5615` | `-8.75%` | `54.06%` | rejected |

Phase 6-F cap audit:

| metric | value |
|---|---:|
| current cap hit count | `338` |
| prevented buy amount | `140,475,510` |
| cap 35% profit approximation | `+325,938` |
| cap 40% profit approximation | `+511,396` |
| cap 50% profit approximation | `+820,943` |
| cap_is_current_bottleneck | `true` |
| cap_relaxation_worth_testing | `true` |

Phase 6-G cap38 full backtest:

| metric | v2_78 | v2_82 cap38 | delta |
|---|---:|---:|---:|
| net_profit | `3,054,794` | `3,777,545` | `+722,751` |
| PF | `2.6194` | `2.7309` | `+0.1115` |
| DD | `-7.47%` | `-6.54%` | `+0.93pt` |
| win_rate | `53.78%` | `55.11%` | `+1.33pt` |
| monthly_win_rate | `75.61%` | `78.05%` | `+2.44pt` |
| total_trades | `505` | `502` | `-3` |
| average_capital_utilization | `38.31%` | `40.49%` | `+2.18pt` |
| per_code_cap_skip_or_reduction_count | `50` | `16` | `-34` |

Concentration:

| metric | v2_78 | v2_82 cap38 |
|---|---:|---:|
| single-code profit concentration | `8.11%` | `6.83%` |
| top5-code profit concentration | `15.20%` | `14.12%` |

Current profile ranking:

| priority | profile | status |
|---:|---|---|
| 1 | `rookie_dealer_02_v2_82_cap38` | current strongest full-backtested research candidate |
| 2 | `rookie_dealer_02_v2_78_pm_aware_order_fallback_w025` | conservative fallback/reference |
| deferred | `rookie_dealer_02_v2_81_bear_pm115_booster_50` | booster fired but performance unchanged due cap absorption |
| rejected | v2_80 Exit AI v2 profiles | underperformed v2_78 |

Coverage caveat:

- Backtests requested `2023-01-01` to `2026-05-31`.
- Cached prices ended at `2026-05-29`, so the engine reports
  `coverage_ok=false`.
- The compared logs were generated on the same cached-data basis.

v2_77 cap 0.30 capital utilization:

| metric | value |
|---|---:|
| average capital utilization | `50.995%` |
| median capital utilization | `53.121%` |
| days below 50% | `355` |
| cash idle days | `63` |
| average holding count | about `2.44` |

Low-utilization dominant reasons:

| reason | days |
|---|---:|
| `no_candidates` | `114` |
| `exit_only_day` | `67` |
| `candidates_all_low_pm_skipped` | `35` |

Candidate pool expansion result:

| variant | max_selected | net_profit | PF | DD | avg utilization | no_candidates |
|---|---:|---:|---:|---:|---:|---:|
| current | `10` | `2,914,686` | `2.5430` | `-7.54%` | `50.995%` | `114` |
| pool x2 | `20` | `2,520,271` | `2.3186` | `-8.04%` | `51.010%` | `114` |
| pool x3 | `30` | `2,520,271` | `2.3186` | `-8.04%` | `51.010%` | `114` |

Decision:

- Do not adopt candidate pool expansion for now.
- It did not reduce `no_candidates`, barely improved utilization, and worsened
  profit/PF/DD.

## Phase 7-A to 7-G Final Readiness Result

The final AI-state audit, PM AI API-only rebuild path, Final Championship
Audit, and final pytest triage have now been completed.

Detailed summary:

```text
docs/ml/Portfolio_Manager_AI_Phase7A_to_7G_Final_Summary.md
```

Final Version 1.0 Candidate:

```text
rookie_dealer_02_v2_82_cap38
```

Phase 7-F Final Championship core result:

| metric | v2_78 | v2_82 cap38 | delta |
|---|---:|---:|---:|
| net_profit | `3,054,794` | `3,777,545` | `+722,751` |
| PF | `2.6194` | `2.7309` | `+0.1115` |
| DD | `-7.47%` | `-6.54%` | `+0.93pt` |
| win_rate | `53.78%` | `55.11%` | `+1.33pt` |
| monthly_win_rate | `75.61%` | `78.05%` | `+2.44pt` |
| final assets | `4,813,588` | `5,720,597` | `+907,010` |
| CAGR | `58.51%` | `66.74%` | `+8.23pt` |

Phase 7-F verdict:

| item | value |
|---|---|
| production_candidate | `true` |
| recommended_profile | `rookie_dealer_02_v2_82_cap38` |
| confidence_level | `medium` |
| fix_recommended | `true` |

The confidence is `medium` because 2021-2022 cannot be evaluated as the same
condition championship period from the current prediction artifacts:

- `data/ml/walk_forward_predictions/` begins at `2023-01-04`.
- Extended `2021-06-01 to 2026-05-31` should be treated as unavailable for a
  decisive comparison until true historical walk-forward predictions are
  rebuilt.

PM AI API-only candidate:

```text
models/ml/portfolio_manager/candidate_v2_api_only
```

Status:

- trained from API-only PM dataset;
- current PM AI was not overwritten;
- high conviction test AUC `0.6472`;
- avoid target test AUC `0.6345`;
- not integrated into v2_82.

Phase 7-G final test result:

```text
825 passed, 15 warnings
```

Fixed before the final green run:

- earnings filter expectation mismatch;
- SELL trade `market_regime` inheritance;
- J-Quants / no-data cache fixture stability;
- operations docs missing expected sections.

Current profile/model ranking after Phase 7-G:

| priority | profile / model | status |
|---:|---|---|
| 1 | `rookie_dealer_02_v2_82_cap38` | Version 1.0 Candidate; strongest full-backtested profile |
| 2 | `rookie_dealer_02_v2_78_pm_aware_order_fallback_w025` | conservative fallback/reference |
| candidate model | `models/ml/portfolio_manager/candidate_v2_api_only` | trained, leakage-safe, not integrated |
| rejected | v2_80 Exit AI v2 profiles | underperformed v2_78 |
| deferred | v2_81 Bear Booster | booster fired but cap absorbed effective benefit |
| on hold | v2_79 high-PM minimum hold | numerical improvement was side-effect path, not direct minimum-hold effect |

## Hard Constraints

Do not do these unless explicitly asked:

- Do not call OpenAI API.
- Do not refetch J-Quants API data.
- Do not regenerate historical predictions with `models/ml/current`.
- Do not use `selected_count_in_day`.
- Do not use actual/backtest/audit/decision/skip/exit/cash/final/result/profit
  columns as Portfolio Manager AI features.
- Do not destructively modify existing profiles such as v2_73.
- Do not connect to live order placement.
- Do not change Exit AI or `daily_buy_limit` unless the task explicitly says so.

Historical backtests should use:

```text
data/ml/walk_forward_predictions/
```

not current-model regenerated predictions.

## Useful Commands

Run v2_75 full backtest:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 src/main.py \
  --mode backtest \
  --provider jquants \
  --profile rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing \
  --start-date 2023-01-01 \
  --end-date 2026-05-31 \
  --skip-price-fetch \
  --quiet \
  --summary-only \
  --no-daily-logs
```

Run v2_77 cap 0.30 full backtest:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 src/main.py \
  --mode backtest \
  --provider jquants \
  --profile rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030 \
  --start-date 2023-01-01 \
  --end-date 2026-05-31 \
  --skip-price-fetch \
  --quiet \
  --summary-only \
  --no-daily-logs
```

Run v2_75 detail audit:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 scripts/ml/audit_portfolio_manager_phase3d_detail.py
```

Run latest capital utilization and candidate-pool audits:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 scripts/ml/audit_portfolio_manager_phase3h_capital_utilization.py
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 scripts/ml/audit_portfolio_manager_phase3i_candidate_pool.py
```

Relevant tests:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 -m pytest -q \
  tests/test_ml_portfolio_manager_dataset.py \
  tests/test_ml_portfolio_manager_trainer.py \
  tests/test_ml_portfolio_manager_data_lineage.py \
  tests/test_ml_portfolio_manager_phase3c.py \
  tests/test_ml_portfolio_manager_phase3d.py \
  tests/test_ml_portfolio_manager_phase3d_detail_audit.py \
  tests/test_ml_portfolio_manager_phase3e.py \
  tests/test_ml_portfolio_manager_phase3f_drawdown_audit.py \
  tests/test_ml_portfolio_manager_phase3g_per_code_cap.py \
  tests/test_ml_portfolio_manager_phase3h_capital_utilization.py \
  tests/test_ml_portfolio_manager_phase3i_candidate_pool.py
```

Latest known full pytest result:

```text
825 passed, 15 warnings
```

## Next Good Tasks

Recommended next experiments:

1. Freeze v2_82 cap38 as the Version 1.0 Candidate.
2. Keep v2_78 w0.25 as conservative fallback/reference.
3. Do not integrate PM AI API-only candidate until a dedicated integration
   audit proves it improves v2_82.
4. If robustness beyond 2023 is required, rebuild true 2021-2022 walk-forward
   predictions first.
5. Keep v2_82 clean:
   - no Bear Booster;
   - no Exit AI v2 candidate;
   - no PM AI API-only candidate integration yet.
4. Do not continue candidate-pool expansion unless the upstream candidate
   shortage definition changes.
6. Try utilization-improvement paths that do not dilute candidate quality:
   - low-score skip threshold tuning
   - replacement candidate filling after cap / affordability blocks
   - total-assets-linked `daily_buy_limit`
   - fallback quality improvement
7. Keep v2_78, v2_77, v2_75, and v2_73 as fallback references.
8. Continue monitoring top-code contribution and DD-period concentration.

Do not promote v2_76 directly without an exposure guard because its DD is too
large.

Exit AI v2 integration-audit starting point:

```text
models/ml/exit_ai_v2/candidate_v2_api_only/
data/ml/exit_ai_v2/exit_ai_v2_dataset_api_only_2021-06_to_2026-05.parquet
reports/ml/phase5e_exit_ai_v2_trainer_prototype_2021-06_to_2026-05.json
```

## Documentation Map

Use these documents:

- `docs/ml/README.md`: ML documentation index
- `docs/ml/ML_Phase_25_to_Portfolio_Manager_AI_Summary.md`: full recent history
- `docs/ml/Portfolio_Manager_AI_Phase7A_to_7G_Final_Summary.md`: final readiness and Version 1.0 Candidate summary
- `docs/ml/Portfolio_Manager_AI_Phase4C_to_4G_Audit_Summary.md`: why v2_79 remains on hold
- `docs/ml/Exit_AI_v2_Phase5A_to_5F_Retraining_Summary.md`: Exit AI v2 API-only retraining work
- `docs/ml/v2_73_adoption_notes.md`: why v2_73 became the prior baseline
- `docs/ml/daily_ai_candidate_operation.md`: human-review daily AI candidates
