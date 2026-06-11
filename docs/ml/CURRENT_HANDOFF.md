# Current ML Handoff

Last updated: `2026-06-11`

This document is the short handoff for continuing the AI / ML work in a fresh
chat. It intentionally summarizes only the current state, key constraints, and
next useful actions. For the full history, see
`docs/ml/ML_Phase_25_to_Portfolio_Manager_AI_Summary.md`,
`docs/ml/Exit_AI_v2_Phase5A_to_5F_Retraining_Summary.md`,
`docs/ml/Portfolio_Manager_AI_Phase7A_to_7G_Final_Summary.md`,
`docs/ml/Portfolio_Manager_AI_Phase8A_to_8H_PM_AI_Redesign_Summary.md`,
`docs/ml/Portfolio_Manager_AI_Phase10_Stop_and_Hold_Summary.md`,
`docs/ml/Portfolio_Manager_AI_Phase11_Valuation_Allocation_Plan.md`,
`docs/ml/Portfolio_Manager_AI_Phase12_Dynamic_Capital_Allocation_Summary.md`,
and `docs/ml/Portfolio_Manager_AI_Phase11_12_Research_Summary.md`.

## Latest Phase 12-E2 Handoff

Phase 12-E2 Stock Selection Architecture Audit is implemented:

```text
src/ml/phase12e2_stock_selection_architecture_audit.py
scripts/ml/run_phase12e2_stock_selection_architecture_audit.py
tests/test_ml_phase12e2_stock_selection_architecture_audit.py
```

Latest generated report:

```text
reports/ml/phase12e2_stock_selection_architecture_audit.md
reports/ml/phase12e2_stock_selection_architecture_audit.json
```

Scope and constraints:

- 既存code / artifact / model metadata / report JSONのみ監査
- 新規AI学習なし
- prediction再生成なし
- full backtestなし
- profile追加/変更なし
- 既存model上書きなし
- historical prediction再生成なし
- future系は評価指標のみ
- leakage_risk `low`, blocking_issues `0`

Current Stock Selection architecture:

| item | value |
| --- | --- |
| model family | LightGBM `LGBMRegressor` + `LGBMClassifier` |
| training code | `src/ml/model_trainer.py::ModelTrainer.train_all` |
| prediction code | `src/ml/predictor.py::Predictor.predict_daily` |
| walk-forward code | `src/ml/walk_forward.py::MLWalkForwardRunner` |
| feature count | `48` |
| strict_oos_for_2025 | `true` |

Output interpretation:

| column | meaning |
| --- | --- |
| `expected_return` | alias of `expected_return_10d` |
| `risk_adjusted_score` | `expected_return_10d - 0.5 * bad_entry_probability_10d` |
| `stock_selection_rank_score` | derived from `ml_score`; not a direct 20d Opportunity target |
| `candidate_strength` | `expected_max_return_20d + swing_success_probability_20d - bad_entry_probability_10d` |

Phase 12-E1 Reality Audit immediately before E2:

| item | value |
| --- | --- |
| `stock_selection_adds_value` | `false` |
| `stock_selection_top5_valid` | `false` |
| `stock_selection_prefilter_hurts_valuation` | `true` |
| `stock_selection_rank_score_top5_top_decile_rate` | `0.0885` |
| `candidate_universe_top_decile_rate` | `0.1053` |
| `candidate_strength_top5_top_decile_rate` | `0.2000` |
| `opportunity_top5_reference` | `0.2400` |

Interpretation:

- Stock Selection AI is clean from a lineage/leakage perspective, but its
  objective is not aligned with Phase 12.
- It is a short-horizon composite selector: 5d/10d returns, 10d upside,
  10d bad-entry risk, 10d/20d max-return, and 20d swing success.
- Phase 12 is evaluating 20d Opportunity + Downside + allocation/exit behavior.
- `stock_selection_rank_score` comes from `ml_score`, so it is not a direct
  20d Opportunity model.
- `candidate_strength` performed better because it includes
  `expected_max_return_20d` and `swing_success_probability_20d`.
- Phase 12-D3 already confirmed Phase 12 inputs are strict OOS and
  `phase12_results_trustworthy=true`.

Current decision:

```text
ready_for_phase13 = false
phase12_results_trustworthy = true
stock_selection_prefilter_hurts_valuation = true
recommended_next_phase = Phase12-E3 Remove Stock Selection Prefilter Test
```

Do not run broad/full backtests yet. The next useful step is a 2025-limited
test that removes Stock Selection prefilter, or compares Candidate Strength
rebasing against direct Valuation + Downside candidate generation.

## Current State

The latest full-backtested Version 1.0 Candidate remains:

```text
rookie_dealer_02_v2_82_cap38
```

Phase 10 / PM AI redevelopment is stopped and held. Phase 11 starts a new
research direction:

```text
Valuation Engine
↓
Capital Allocation Engine
```

The goal is not to mimic current PM multipliers. Phase 11 will first audit
whether API-derived, prediction-time-safe features can explain opportunity:

- `opportunity_score`
- `expected_upside`
- `expected_downside`
- `confidence`

`v2_82_cap38` is a reference record for comparison, not the Phase 11 adoption
target.

Phase 11-A Valuation Engine Dataset Audit is implemented:

```text
src/ml/phase11a_valuation_dataset_audit.py
scripts/ml/audit_phase11a_valuation_dataset.py
tests/test_ml_phase11a_valuation_dataset_audit.py
```

Latest generated report:

```text
reports/ml/phase11a_valuation_dataset_audit_2023-01_to_2026-05.md
reports/ml/phase11a_valuation_dataset_audit_2023-01_to_2026-05.json
```

Core result:

- rows: `930,243`
- unique_codes: `4,234`
- date range: `2023-01-04` to `2026-04-23`
- feature_count: `55`
- label_count: `5`
- leakage_risk: `low`
- blocking_issues: `0`
- ready_for_phase11b: `true`

Phase 11-B Valuation Engine Prototype is implemented:

```text
src/ml/phase11b_valuation_engine_prototype.py
scripts/ml/train_phase11b_valuation_engine_prototype.py
tests/test_ml_phase11b_valuation_engine_prototype.py
```

Latest generated report and candidate model:

```text
reports/ml/phase11b_valuation_engine_prototype_2025_holdout.md
reports/ml/phase11b_valuation_engine_prototype_2025_holdout.json
models/ml/valuation_engine/candidate_phase11b/
```

Core Phase 11-B result:

- train rows: `250,000`
- test rows: `310,618`
- feature_count: `54`
- regression target: `opportunity_value_20d`
- regression MAE/RMSE: `0.0865` / `0.1407`
- regression Pearson/Spearman: `0.0559` / `-0.0028`
- classification target: `opportunity_top_decile_20d`
- AUC / PR-AUC: `0.6478` / `0.1600`
- precision@top10%: `0.1998`
- base positive rate: `0.0997`
- leakage_risk: `low`
- blocking_issues: `0`
- ready_for_phase11c: `true`

Valuation output:

- `opportunity_score`
- `predicted_opportunity_value`
- `opportunity_top_decile_proba`
- `confidence`

`expected_upside` and `expected_downside` are intentionally not modeled yet in
Phase 11-B.

Phase 11-C Capital Allocation Engine Prototype is implemented:

```text
src/ml/phase11c_capital_allocation_prototype.py
scripts/ml/run_phase11c_capital_allocation_prototype.py
tests/test_ml_phase11c_capital_allocation_prototype.py
```

Latest generated report:

```text
reports/ml/phase11c_capital_allocation_prototype_2025.md
reports/ml/phase11c_capital_allocation_prototype_2025.json
data/ml/valuation_engine/phase11c_allocation_simulation_2025.parquet
```

Core Phase 11-C result:

- period: `2025-01-07` to `2025-12-29`
- rows: `310,618`
- candidate_days: `165`
- leakage_risk: `low`
- blocking_issues: `0`
- best rule: `equal_weight_top5`
- weighted opportunity top-decile rate: `0.2403`
- weighted opportunity value: `0.0587`
- average budget usage: `20.7%`
- ready_for_phase11d: `true`

Important interpretation:

- `equal_weight_top5`, `proba_rank_weighted`, and `conservative_top_only`
  converged to the same allocated candidates under the current daily budget,
  max positions, round-lot, and affordability assumptions.
- Budget usage is low, so Phase 11-D should start with strict limited-scope
  design rather than a broad full backtest.

Phase 11-C2 Budget Usage Constraint Audit is implemented:

```text
src/ml/phase11c2_budget_usage_constraint_audit.py
scripts/ml/audit_phase11c2_budget_usage_constraints.py
tests/test_ml_phase11c2_budget_usage_constraint_audit.py
```

Latest generated report:

```text
reports/ml/phase11c2_budget_usage_constraint_audit_2025.md
reports/ml/phase11c2_budget_usage_constraint_audit_2025.json
```

Core Phase 11-C2 result:

- period: `2025-01-07` to `2025-12-29`
- rows: `310,618`
- candidate_days: `165`
- leakage_risk: `low`
- blocking_issues: `0`
- main bottleneck: `round_lot_and_top_candidate_affordability_limit_daily_budget_usage`
- constraint reasons: `rank_filter_too_strict=128 days`, `top_candidates_too_expensive=37 days`
- top5 lot cost median / p90: `195,100` / `1,658,300`
- top5 affordable rate under `300,000`: `61.45%`
- budget usage sensitivity: `20.7%` at `300,000`, `28.9%` at `500,000`, `40.9%` at `900,000`
- recommended_daily_budget: `900,000`
- recommended_max_positions: `5`
- recommended_candidate_threshold: `top5`
- ready_for_phase11d: `true`

Important interpretation:

- The low usage is not caused by lack of candidate days or missing affordable
  names in the full universe.
- It is mainly caused by the interaction between Valuation top candidates,
  round lots, and a `300,000` daily budget.
- Loosening candidate thresholds alone did not improve usage because the base
  budget and max-position constraints still select the same affordable top
  names.

Phase 11-D Limited Combined Backtest is implemented:

```text
src/ml/phase11d_combined_backtest.py
scripts/ml/run_phase11d_combined_backtest.py
tests/test_ml_phase11d_combined_backtest.py
```

Latest generated report:

```text
reports/ml/phase11d_combined_backtest_2025.md
reports/ml/phase11d_combined_backtest_2025.json
```

Core Phase 11-D result:

- period: `2025-01-01` to `2025-12-31` entries only
- source rows: `310,618`
- candidate_days: `165`
- leakage_risk: `low`
- blocking_issues: `0`
- full_period_backtest_executed: `false`
- historical_predictions_regenerated: `false`
- profile_changed: `false`

Strategy comparison:

| strategy | net_profit | PF | DD | win_rate | trades | final_assets | utilization |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline equal allocation | `88,578` | `1.5829` | `-5.33%` | `50.91%` | `55` | `1,088,578` | `50.07%` |
| valuation top5 | `187,018` | `1.6990` | `-16.83%` | `60.34%` | `58` | `1,187,018` | `49.89%` |

BUY quality comparison:

- baseline future_return_20d mean: `0.0104`
- valuation future_return_20d mean: `0.0314`
- baseline opportunity_value_20d mean: `0.0190`
- valuation opportunity_value_20d mean: `0.0631`
- baseline top-decile BUY rate: `7.27%`
- valuation top-decile BUY rate: `29.31%`

Important interpretation:

- Phase 11-D confirms that Valuation improves candidate quality in the 2025
  limited combined setup.
- Net profit, PF, win rate, and BUY quality all improved versus the non-
  Valuation baseline.
- DD worsened materially, so Phase 11-E should proceed only as a limited
  exit/risk guard experiment, not as a broad full-period backtest.

Phase 11-E Limited Exit / DD Guard is implemented:

```text
src/ml/phase11e_exit_dd_guard.py
scripts/ml/run_phase11e_exit_dd_guard.py
tests/test_ml_phase11e_exit_dd_guard.py
```

Latest generated report:

```text
reports/ml/phase11e_exit_dd_guard_2025.md
reports/ml/phase11e_exit_dd_guard_2025.json
```

Core Phase 11-E result:

- period: `2025-01-01` to `2025-12-31` entries only
- leakage_risk: `low`
- blocking_issues: `0`
- full_backtest_executed: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`
- stop_loss_uses_future_low: `false`

Variant comparison:

| variant | net_profit | PF | DD | win_rate | trades | avg holding |
|---|---:|---:|---:|---:|---:|---:|
| E0 no guard | `181,730` | `1.6869` | `-23.43%` | `58.62%` | `58` | `19.41` |
| E1 stop -8% | `82,540` | `1.1993` | `-22.94%` | `47.83%` | `69` | `15.96` |
| E2 stop -5% | `49,110` | `1.1185` | `-26.94%` | `42.11%` | `76` | `14.28` |
| E3 opportunity disappeared | `529,880` | `2.8160` | `-7.85%` | `54.74%` | `137` | `7.16` |
| E4 stop -8% + opportunity | `615,110` | `2.6219` | `-6.02%` | `52.26%` | `155` | `6.26` |

Important interpretation:

- Simple stop loss alone was harmful in this lightweight 2025 test.
- Opportunity Disappeared Exit improved DD below `-10%` while preserving PF
  and net profit.
- E4 had the best DD, but it also increased trade count and shortened holding
  days, so the next step should test transaction cost, slippage, and threshold
  robustness before expanding scope.

Phase 11-F Limited Robustness Check is implemented:

```text
src/ml/phase11f_robustness_check.py
scripts/ml/run_phase11f_robustness_check.py
tests/test_ml_phase11f_robustness_check.py
```

Latest generated report:

```text
reports/ml/phase11f_robustness_check_2025.md
reports/ml/phase11f_robustness_check_2025.json
```

Core Phase 11-F result:

- period: `2025-01-01` to `2025-12-31` entries only
- base strategy: `E4_stop_loss_8pct_plus_opportunity`
- leakage_risk: `low`
- blocking_issues: `0`
- full_backtest_executed: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`

Cost sensitivity:

| one-way cost | net_profit | PF | DD | trades | avg holding | cost paid |
|---:|---:|---:|---:|---:|---:|---:|
| `0.0%` | `615,110` | `2.6219` | `-6.02%` | `155` | `6.26` | `0` |
| `0.1%` | `568,935` | `2.4304` | `-6.59%` | `155` | `6.26` | `43,075` |
| `0.2%` | `473,578` | `2.0551` | `-6.36%` | `157` | `6.21` | `85,882` |
| `0.3%` | `394,497` | `1.8790` | `-8.50%` | `158` | `6.07` | `128,063` |

Threshold sensitivity:

| threshold | net_profit | PF | DD | trades | avg holding |
|---|---:|---:|---:|---:|---:|
| loose | `355,440` | `2.0448` | `-9.45%` | `112` | `9.32` |
| baseline | `615,110` | `2.6219` | `-6.02%` | `155` | `6.26` |
| strict | `496,270` | `2.0189` | `-9.14%` | `229` | `3.91` |

Overtrading notes:

- same_code_reentry_count: `115`
- reentry_within_5_days_count: `88`
- median_holding_days: `4.00`
- December trade count: `25`

Important interpretation:

- E4 passes the 2025-only robustness checks, including `0.2%` one-way cost.
- Overtrading risk remains meaningful; Phase 11-G should not jump to a broad
  full-period backtest yet.
- Next check should be limited out-of-sample year validation plus same-code
  reentry cooldown / minimum holding guard sensitivity.

Phase 11-G Limited Out-of-Sample Year Check is implemented:

```text
src/ml/phase11g_out_of_sample_check.py
scripts/ml/run_phase11g_out_of_sample_check.py
tests/test_ml_phase11g_out_of_sample_check.py
```

Latest generated report:

```text
reports/ml/phase11g_out_of_sample_check_2024.md
reports/ml/phase11g_out_of_sample_check_2024.json
```

Core Phase 11-G result:

- period: `2024-01-01` to `2024-12-31`
- rows: `262,224`
- candidate_days: `166`
- leakage_risk: `low`
- blocking_issues: `0`
- full_backtest_executed: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`

Important model-OOS limitation:

- Phase 11-B candidate model train period is `2023-01-04` to `2024-12-31`.
- Therefore, 2024 overlaps the model training period.
- Phase 11-G supports strategy/path robustness on an additional year, but it
  is not strict model out-of-sample proof.

Strategy comparison:

| strategy | net_profit | PF | DD | trades | avg holding | reentry within 5d |
|---|---:|---:|---:|---:|---:|---:|
| baseline equal allocation | `156,650` | `2.2226` | `-9.12%` | `56` | `19.66` | `10` |
| valuation top5 no guard | `668,360` | `3.3827` | `-14.85%` | `61` | `19.30` | `41` |
| valuation top5 E4 | `699,520` | `2.7918` | `-8.25%` | `172` | `5.18` | `111` |
| valuation top5 E4 cost 0.2% | `574,984` | `2.3421` | `-9.11%` | `168` | `5.23` | `107` |

Important interpretation:

- E4 passes the 2024 limited year check versus baseline and remains robust
  under `0.2%` one-way cost.
- No-guard valuation is strong but DD is worse, matching the Phase 11-D/E
  pattern that Opportunity Exit is needed.
- Overtrading persists in 2024, with more than 100 reentries within 5 business
  days for E4.
- Next step should combine cooldown/minimum-hold guard with a strict
  walk-forward OOS design where 2024 is not inside the training window.

Phase 11-H Cooldown / Minimum Holding Guard is implemented:

```text
src/ml/phase11h_cooldown_minhold_guard.py
scripts/ml/run_phase11h_cooldown_minhold_guard.py
tests/test_ml_phase11h_cooldown_minhold_guard.py
```

Latest generated report:

```text
reports/ml/phase11h_cooldown_minhold_guard_2024_2025.md
reports/ml/phase11h_cooldown_minhold_guard_2024_2025.json
```

Core Phase 11-H result:

- years: `2024`, `2025`
- base strategy: E4 with `0.2%` one-way cost
- leakage_risk: `low`
- blocking_issues: `0`
- full_backtest_executed: `false`
- walk_forward_retraining_executed: `false`
- historical_predictions_regenerated: `false`

2024 guard comparison:

| variant | net_profit | PF | DD | trades | avg hold | reentry within 5d |
|---|---:|---:|---:|---:|---:|---:|
| H0 baseline E4 | `574,984` | `2.3421` | `-9.11%` | `168` | `5.23` | `107` |
| H1 cooldown 5d | `448,303` | `2.1910` | `-8.13%` | `136` | `5.82` | `48` |
| H2 cooldown 10d | `451,707` | `2.5265` | `-6.18%` | `115` | `6.17` | `36` |
| H3 min hold 3d | `610,157` | `2.3592` | `-9.01%` | `138` | `6.73` | `89` |
| H4 cooldown 5d + min hold 3d | `623,772` | `3.1105` | `-4.82%` | `118` | `7.43` | `44` |

2025 guard comparison:

| variant | net_profit | PF | DD | trades | avg hold | reentry within 5d |
|---|---:|---:|---:|---:|---:|---:|
| H0 baseline E4 | `473,578` | `2.0551` | `-6.36%` | `157` | `6.21` | `90` |
| H1 cooldown 5d | `255,308` | `1.6062` | `-8.72%` | `147` | `5.84` | `41` |
| H2 cooldown 10d | `389,740` | `2.1260` | `-6.94%` | `137` | `5.88` | `38` |
| H3 min hold 3d | `397,786` | `1.9285` | `-7.65%` | `138` | `7.11` | `78` |
| H4 cooldown 5d + min hold 3d | `170,482` | `1.4359` | `-14.14%` | `120` | `7.61` | `41` |

Important interpretation:

- H2 cooldown 10d and H3 minimum holding 3d passed both years.
- H2 gives the strongest reentry reduction while preserving PF/DD in both
  years.
- H3 preserves more profit but reduces 5-day reentries less than H2.
- H4 is attractive in 2024 but unstable in 2025, so it should not be promoted
  without retuning.
- Strict OOS design was documented but not executed. Recommended first split:
  train `2023`, validation `2024`, test `2025`, with a separate research-only
  model directory.

Phase 11-I Strict Walk-Forward OOS Prototype is implemented:

```text
src/ml/phase11i_strict_oos.py
scripts/ml/run_phase11i_strict_oos.py
tests/test_ml_phase11i_strict_oos.py
```

Latest generated report and research-only model:

```text
reports/ml/phase11i_strict_walk_forward_oos_2025.md
reports/ml/phase11i_strict_walk_forward_oos_2025.json
models/ml/valuation_engine/research_phase11i_strict_oos/
```

Core Phase 11-I setup:

- train: `2023-01-04` to `2023-12-31`
- validation: `2024-01-01` to `2024-12-31`
- test: `2025-01-01` to `2025-12-31`
- classification target: `opportunity_top_decile_20d`
- model: `HistGradientBoostingClassifier`
- feature_count: `54`
- strict_model_oos: `true`
- train_validation_test_overlap: `false`
- existing_model_overwritten: `false`
- profile_changed: `false`
- full_backtest_executed: `false`
- historical_predictions_regenerated: `false`
- leakage_risk: `low`
- blocking_issues: `0`

Model quality:

| split | AUC | PR-AUC | precision@top10% | base positive rate |
|---|---:|---:|---:|---:|
| validation 2024 | `0.6055` | `0.1445` | `0.1754` | `0.0996` |
| test 2025 | `0.6297` | `0.1514` | `0.1837` | `0.0997` |
| Phase 11-B reference 2025 | `0.6478` | `0.1600` | `0.1998` | `0.0997` |

2025 strict OOS strategy check with `0.2%` one-way cost:

| strategy | net_profit | PF | DD | trades | avg hold | reentry within 5d |
|---|---:|---:|---:|---:|---:|---:|
| baseline equal allocation | `180,876` | `2.2930` | `-6.39%` | `55` | `19.93` | `22` |
| strict OOS valuation top5 no guard | `-85,275` | `0.8169` | `-22.35%` | `60` | `19.70` | `26` |
| strict OOS E4 | `116,049` | `1.2501` | `-13.14%` | `141` | `7.67` | `74` |
| strict OOS H2 cooldown 10d | `-75,507` | `0.8230` | `-14.27%` | `114` | `8.45` | `37` |
| strict OOS H3 min hold 3d | `42,411` | `1.0923` | `-15.26%` | `129` | `8.71` | `67` |

Important interpretation:

- Strict split still preserves classification lift, but the 2023-only model is
  weaker than the Phase 11-B model trained through 2024.
- Strategy checks do not pass strict OOS criteria. E4 remains positive, but it
  does not beat the baseline and DD is worse than the `-12%` provisional guard.
- H2 reduces short reentries but breaks profit/PF under the strict OOS model.
- H3 remains positive but also fails PF/DD and does not beat baseline.
- Phase 11-D/E/F/H remain useful strategy/path evidence, but Phase 11-I shows
  the valuation model itself needs improvement before broader adoption work.

Phase 11-B2 Strict OOS Failure Diagnosis is implemented:

```text
src/ml/phase11b2_strict_oos_failure_diagnosis.py
scripts/ml/audit_phase11b2_strict_oos_failure_diagnosis.py
tests/test_ml_phase11b2_strict_oos_failure_diagnosis.py
```

Latest generated report:

```text
reports/ml/phase11b2_strict_oos_failure_diagnosis_2025.md
reports/ml/phase11b2_strict_oos_failure_diagnosis_2025.json
```

Core Phase 11-B2 result:

- scope: 2025 diagnosis only
- Phase 11-I research model read only; no overwrite
- profile_changed: `false`
- full_backtest_executed: `false`
- historical_predictions_regenerated: `false`
- leakage_risk: `low`
- blocking_issues: `0`
- main failure reason: high-downside candidate concentration plus feature drift
- recommended next phase: `Phase11-B3 expected_downside model prototype`

Top candidate diagnosis:

| candidate set | future return | max return | max drawdown | opportunity value | top-decile rate | downside bad rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline top5 | `0.0162` | `0.0699` | `-0.0514` | `0.0185` | `0.0885` | `0.1358` |
| strict OOS valuation top5 | `0.0063` | `0.1311` | `-0.1042` | `0.0269` | `0.2400` | `0.3794` |

Important interpretation:

- Strict OOS valuation top5 still enriches top-decile candidates and future
  max return.
- It also concentrates large downside risk. Downside bad rate rises from
  `13.58%` to `37.94%`.
- Stop-loss exits have only a small average proba drop, so the classifier is
  not directly recognizing drawdown risk.
- Opportunity Exit was frequent but not classified as pure overreaction,
  because `opportunity_proba_drop` exits had positive average realized return.
- Feature drift is meaningful in Stock Selection score features and in
  `Sales_growth` / `topix_return_20d`.
- Daily range filtering reduced downside in a quick audit, but it removed too
  many candidates to be treated as a final rule.

Phase 11-B3 Expected Downside Model Prototype is implemented:

```text
src/ml/phase11b3_expected_downside_model.py
scripts/ml/run_phase11b3_expected_downside_model.py
tests/test_ml_phase11b3_expected_downside_model.py
```

Latest generated report and research-only model:

```text
reports/ml/phase11b3_expected_downside_model_2025.md
reports/ml/phase11b3_expected_downside_model_2025.json
models/ml/valuation_engine/research_phase11b3_downside/
```

Core Phase 11-B3 result:

- strict split: train `2023`, validation `2024`, test `2025`
- downside target: `downside_bad_20d = future_max_drawdown_20d <= -0.10`
- strategy_backtest_executed: `false`
- existing_model_overwritten: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`
- strict_model_oos: `true`
- leakage_risk: `low`
- blocking_issues: `0`
- recommended next phase: `Phase11-B4 combined ranking threshold tuning`

Downside model quality:

| split | AUC | PR-AUC | precision@top10% | base downside rate |
|---|---:|---:|---:|---:|
| validation 2024 | `0.6288` | `0.2788` | `0.3318` | `0.1942` |
| test 2025 | `0.6180` | `0.2323` | `0.2992` | `0.1495` |

Combined ranking audit:

| set | future return | max return | max drawdown | opportunity value | top-decile rate | downside bad rate |
|---|---:|---:|---:|---:|---:|---:|
| opportunity only top5 | `0.0063` | `0.1311` | `-0.1042` | `0.0269` | `0.2400` | `0.3794` |
| score_v1 top5 | `0.0155` | `0.0911` | `-0.0653` | `0.0258` | `0.1527` | `0.1976` |
| score_v2 top5 | `0.0172` | `0.1196` | `-0.0863` | `0.0332` | `0.2267` | `0.2921` |
| score_v3 top5 | `0.0133` | `0.0647` | `-0.0471` | `0.0177` | `0.0618` | `0.1297` |

Important interpretation:

- Downside model works as a separate risk axis; top10% downside precision is
  about 2x the 2025 base downside rate.
- `score_v1 = opportunity - downside` passes the downside target
  (`19.76%`) but loses too much opportunity top-decile rate (`15.27%`).
- `score_v2 = opportunity * (1 - downside)` retains more opportunity
  (`22.67%`) but downside remains above target (`29.21%`).
- `score_v3 = opportunity_rank - downside_rank` controls downside strongly
  (`12.97%`) but removes too much opportunity (`6.18%`).
- B3 validates the need for Opportunity + Downside, but does not yet provide a
  final ranking rule.

Phase 12-A Dynamic Capital Allocation Research is implemented:

```text
src/ml/phase12a_dynamic_capital_allocation.py
scripts/ml/run_phase12a_dynamic_capital_allocation.py
tests/test_ml_phase12a_dynamic_capital_allocation.py
```

Latest generated report and artifact:

```text
reports/ml/phase12a_dynamic_capital_allocation_2025.md
reports/ml/phase12a_dynamic_capital_allocation_2025.json
data/ml/valuation_engine/phase12a_dynamic_capital_allocation_2025.parquet
```

Core Phase 12-A result:

- scope: 2025 allocation quality audit only
- model source: `models/ml/valuation_engine/research_phase11b3_downside/`
- strategy_backtest_executed: `false`
- existing_model_overwritten: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`
- leakage_risk: `low`
- blocking_issues: `0`
- ready_for_phase12b: `false`
- recommended next phase: `Phase12-A2 allocation score refinement`

Allocation quality:

| rule | weighted top-decile | weighted downside bad | weighted opportunity value | weighted max drawdown |
|---|---:|---:|---:|---:|
| baseline equal top5 | `0.0885` | `0.1358` | `0.0185` | `-0.0514` |
| downside safe top5 | `0.0436` | `0.0412` | `0.0103` | `-0.0241` |
| opportunity only top5 | `0.2400` | `0.3794` | `0.0269` | `-0.1042` |
| score_a weighted | `0.1537` | `0.2260` | `0.0282` | `-0.0724` |
| score_b weighted | `0.0814` | `0.1209` | `0.0209` | `-0.0483` |
| score_c weighted | `0.1374` | `0.1933` | `0.0267` | `-0.0655` |
| score_d weighted | `0.0780` | `0.1166` | `0.0203` | `-0.0481` |
| score_e weighted | `0.1349` | `0.1958` | `0.0267` | `-0.0659` |

Important interpretation:

- No rule met the Phase 12-A minimum line:
  weighted top-decile rate `>= 0.20` and weighted downside bad rate `<= 0.25`.
- `score_a_weighted` reduced downside to `22.60%`, but top-decile rate was
  only `15.37%`.
- `score_c_weighted` / `score_e_weighted` kept opportunity value close to
  opportunity-only while controlling downside near `20%`, but top-decile rate
  remained too low.
- The p70-or-higher weight design selects too many candidates per day and
  dilutes opportunity concentration.
- Do not move to Phase 12-B strategy checks yet.

Phase 12-A2 Allocation Score Refinement is implemented:

```text
src/ml/phase12a2_allocation_score_refinement.py
scripts/ml/run_phase12a2_allocation_score_refinement.py
tests/test_ml_phase12a2_allocation_score_refinement.py
```

Latest generated report:

```text
reports/ml/phase12a2_allocation_score_refinement_2025.md
reports/ml/phase12a2_allocation_score_refinement_2025.json
```

Core Phase 12-A2 result:

- scope: 2025 allocation score refinement only
- source artifact: `data/ml/valuation_engine/phase12a_dynamic_capital_allocation_2025.parquet`
- strategy_backtest_executed: `false`
- full_backtest_executed: `false`
- existing_model_overwritten: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`
- leakage_risk: `low`
- blocking_issues: `0`
- ready_for_phase12b: `false`
- recommended next phase: `Phase12-A3 allocation refinement`

Best near-pass rule:

| rule | avg candidates/day | weighted top-decile | weighted downside bad | weighted opportunity value | budget usage proxy |
|---|---:|---:|---:|---:|---:|
| `opportunity_top5__penalty_rank_medium` | `5.0` | `0.2454` | `0.2664` | `0.0453` | `0.1907` |

Other useful references:

| rule | weighted top-decile | weighted downside bad | note |
|---|---:|---:|---|
| `opportunity_top5__penalty_none` | `0.2400` | `0.3794` | original opportunity-only risk |
| `opportunity_top5__penalty_rank_soft` | `0.2517` | `0.2970` | improves value but still high downside |
| `opportunity_top10__penalty_rank_medium` | `0.2122` | `0.2721` | broader, still misses downside target |
| `opportunity_p95__penalty_rank_soft` | `0.1773` | `0.2452` | downside passes but universe is too broad |

Important interpretation:

- The A2 direction is right: keep Opportunity as the candidate universe and use
  Downside only as a sizing penalty.
- `opportunity_top5__penalty_rank_medium` preserves Opportunity concentration
  and almost reaches the downside target, but `26.64%` is still above the
  required `25%`.
- p95/p90 universes are overbroad and should not be used to declare readiness.
- Do not move to Phase 12-B yet. A3 should test slightly stronger top5/top10
  rank penalties before any strategy check.

Phase 12-A3 Top5 Penalty Refinement is implemented:

```text
src/ml/phase12a3_top5_penalty_refinement.py
scripts/ml/run_phase12a3_top5_penalty_refinement.py
tests/test_ml_phase12a3_top5_penalty_refinement.py
```

Latest generated report:

```text
reports/ml/phase12a3_top5_penalty_refinement_2025.md
reports/ml/phase12a3_top5_penalty_refinement_2025.json
```

Core Phase 12-A3 result:

- scope: 2025 top5 penalty refinement only
- source artifact: `data/ml/valuation_engine/phase12a_dynamic_capital_allocation_2025.parquet`
- candidate universe: `opportunity_top5` fixed
- strategy_backtest_executed: `false`
- full_backtest_executed: `false`
- existing_model_overwritten: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`
- leakage_risk: `low`
- blocking_issues: `0`
- ready_for_phase12b: `true`
- recommended next phase: `Phase12-B limited allocation strategy check`

Best rule:

| rule | weighted top-decile | weighted downside bad | weighted opportunity value | average weight |
|---|---:|---:|---:|---:|
| `A3_3_rank_medium_floor_zero` | `0.2614` | `0.1432` | `0.0683` | `0.1168` |

Rules meeting the minimum target:

- `A3_1_rank_medium_plus`
- `A3_2_rank_medium_stronger_tail`
- `A3_3_rank_medium_floor_zero`

Important interpretation:

- A3 reached the Phase 12 minimum line without broadening beyond Opportunity
  top5.
- `A3_3_rank_medium_floor_zero` also met the ideal line, but this is still BUY
  quality / allocation audit only.
- Phase 12-B should be a 2025-only limited allocation strategy check using
  `A3_3_rank_medium_floor_zero`; it should not be a full backtest or profile
  change.

Phase 12-B Limited Allocation Strategy Check is implemented:

```text
src/ml/phase12b_limited_allocation_strategy_check.py
scripts/ml/run_phase12b_limited_allocation_strategy_check.py
tests/test_ml_phase12b_limited_allocation_strategy_check.py
```

Latest generated report:

```text
reports/ml/phase12b_limited_allocation_strategy_check_2025.md
reports/ml/phase12b_limited_allocation_strategy_check_2025.json
```

Core Phase 12-B result:

- scope: 2025 limited allocation strategy check only
- source artifact: `data/ml/valuation_engine/phase12a_dynamic_capital_allocation_2025.parquet`
- strategies: S0 baseline, S1 opportunity equal, S2 opportunity E4, S3a dynamic raw, S3b dynamic normalized
- cost_rate: `0.2%` one-way
- full_backtest_executed: `false`
- existing_model_overwritten: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`
- leakage_risk: `low`
- blocking_issues: `0`
- ready_for_phase12c: `false`
- recommended next phase: `Phase12-B2 allocation execution adjustment`

Strategy results:

| strategy | net profit | PF | DD | utilization |
|---|---:|---:|---:|---:|
| `S0_baseline_equal_allocation` | `66,840` | `1.2506` | `-13.56%` | `0.8646` |
| `S2_opportunity_top5_E4` | `7,724` | `1.0094` | `-23.86%` | `0.8771` |
| `S3a_dynamic_raw_weight` | `39,770` | `1.5971` | `-2.66%` | `0.1007` |
| `S3b_dynamic_normalized_weight` | `135,752` | `1.2712` | `-19.16%` | `0.7506` |

BUY quality:

| strategy | top-decile | downside bad | opportunity value |
|---|---:|---:|---:|
| `S2_opportunity_top5_E4` | `0.2282` | `0.3423` | `0.0378` |
| `S3a_dynamic_raw_weight` | `0.2321` | `0.1786` | `0.0616` |
| `S3b_dynamic_normalized_weight` | `0.2533` | `0.1867` | `0.0650` |

Important interpretation:

- Dynamic allocation improved BUY quality and improved PF/DD versus S2.
- It did not satisfy Phase 12-B forward criteria because S3a did not beat the
  baseline net profit, and S3b failed PF/DD thresholds.
- Raw weighting is too capital-light; normalized weighting is too aggressive.
- Phase 12-B2 should test a small number of execution adjustments between raw
  and normalized weighting, not new models or broad backtests.

Phase 12-B2 Allocation Execution Adjustment is implemented:

```text
src/ml/phase12b2_allocation_execution_adjustment.py
scripts/ml/run_phase12b2_allocation_execution_adjustment.py
tests/test_ml_phase12b2_allocation_execution_adjustment.py
```

Latest generated report:

```text
reports/ml/phase12b2_allocation_execution_adjustment_2025.md
reports/ml/phase12b2_allocation_execution_adjustment_2025.json
```

Core Phase 12-B2 result:

- scope: 2025 allocation execution adjustment only
- source artifact: `data/ml/valuation_engine/phase12a_dynamic_capital_allocation_2025.parquet`
- tested: partial normalization, min usage guard, capped normalized execution
- full_backtest_executed: `false`
- existing_model_overwritten: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`
- leakage_risk: `low`
- blocking_issues: `0`
- ready_for_phase12c: `false`
- recommended next phase: `Phase12-B3 execution adjustment or Phase12-A4 risk score refinement`

Key results:

| strategy | net profit | PF | DD | utilization |
|---|---:|---:|---:|---:|
| `S3a_dynamic_raw_weight` | `39,770` | `1.5971` | `-2.66%` | `0.1007` |
| `S3b_dynamic_normalized_weight` | `135,752` | `1.2712` | `-19.16%` | `0.7506` |
| `S5_partial_normalized_30` | `2,434` | `1.0094` | `-11.75%` | `0.3910` |
| `S8_capped_normalized` | `-34,065` | `0.9120` | `-17.16%` | `0.5922` |

Important interpretation:

- No B2 strategy met the minimum target of net_profit > 0, PF >= 1.5, DD >=
  -12%, and capital_utilization >= 0.20.
- Partial normalization increased utilization but lost PF.
- Usage guards and caps did not recover PF.
- The issue is no longer only allocation amount; the dynamic high-quality BUY
  set may need exit/churn adjustment or a richer risk score before Phase 12-C.

Phase 12-B3 Exit / Hold Decision Audit is implemented:

```text
src/ml/phase12b3_exit_hold_audit.py
scripts/ml/run_phase12b3_exit_hold_audit.py
tests/test_ml_phase12b3_exit_hold_audit.py
```

Latest generated report:

```text
reports/ml/phase12b3_exit_hold_audit_2025.md
reports/ml/phase12b3_exit_hold_audit_2025.json
```

Core Phase 12-B3 result:

- scope: 2025 exit / hold decision audit only
- audited strategies: `S3a_dynamic_raw_weight` and `S2_opportunity_top5_E4`
- full_backtest_executed: `false`
- existing_model_overwritten: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`
- leakage_risk: `low`
- blocking_issues: `0`
- main_exit_problem: `late_exit`
- early_exit_detected: `true`
- late_exit_detected: `true`
- opportunity_exit_effective: `false`
- recommended next phase: `Phase12-B4 trailing_exit_prototype`

Key audit findings:

| audit | S2 | S3a |
|---|---:|---:|
| trade count | `149` | `56` |
| avg post-exit 20d return | `0.0799` | `0.0653` |
| post-exit 10%+ count | `46` | `13` |
| late exit trade count | `24` | `5` |
| avg profit decay before late exit | `0.1232` | `0.1703` |
| opportunity exit avg post-exit 20d return | `0.0768` | `0.0577` |

Important interpretation:

- Dynamic allocation did not solve the hold/exit problem by itself.
- S3a has fewer bad trades, but the remaining stop-loss cases often had profit
  first and then decayed into loss.
- Opportunity Exit is not clearly effective; many exits still leave meaningful
  post-exit upside.
- Next should be a 2025-only B4 trailing exit / profit-decay guard prototype,
  not a broad backtest or profile change.

Phase 12-B4 Trailing Exit Prototype is implemented:

```text
src/ml/phase12b4_trailing_exit_prototype.py
scripts/ml/run_phase12b4_trailing_exit_prototype.py
tests/test_ml_phase12b4_trailing_exit_prototype.py
```

Latest generated report:

```text
reports/ml/phase12b4_trailing_exit_prototype_2025.md
reports/ml/phase12b4_trailing_exit_prototype_2025.json
```

Core Phase 12-B4 result:

- scope: 2025 trailing exit prototype only
- base BUY/allocation: `S3a_dynamic_raw_weight`
- full_backtest_executed: `false`
- existing_model_overwritten: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`
- leakage_risk: `low`
- blocking_issues: `0`
- best_variant: `T5_opportunity_plus_trailing_8pct`
- trailing_exit_improved_vs_opportunity_exit: `false`
- ready_for_phase12c: `false`
- recommended next phase: `Phase12-B5 exit threshold recalibration`

Key results:

| variant | net profit | PF | DD | utilization | avg holding days |
|---|---:|---:|---:|---:|---:|
| `T0_current_opportunity_plus_stop` | `39,770` | `1.5971` | `-2.66%` | `0.1007` | `6.20` |
| `T1_stop_loss_only` | `14,489` | `1.1587` | `-6.83%` | `0.1856` | `17.49` |
| `T3_trailing_8pct` | `9,149` | `1.1017` | `-5.50%` | `0.1806` | `16.84` |
| `T5_opportunity_plus_trailing_8pct` | `43,962` | `1.7044` | `-2.50%` | `0.0998` | `6.09` |

Important interpretation:

- Trailing exit alone did not improve over the current Opportunity Exit setup.
- Disabling Opportunity Exit raised holding days and utilization but hurt PF and
  profit.
- Opportunity + trailing 8% is a small improvement over T0 but does not solve
  utilization.
- The next useful step is not broader testing; it is recalibrating Opportunity
  Exit thresholds so the strategy can hold winners longer without normalizing
  into high DD.

Phase 12-B5 Exit Threshold Recalibration is implemented:

```text
src/ml/phase12b5_exit_threshold_recalibration.py
scripts/ml/run_phase12b5_exit_threshold_recalibration.py
tests/test_ml_phase12b5_exit_threshold_recalibration.py
```

Latest generated report:

```text
reports/ml/phase12b5_exit_threshold_recalibration_2025.md
reports/ml/phase12b5_exit_threshold_recalibration_2025.json
```

Core Phase 12-B5 result:

- scope: 2025 exit threshold recalibration only
- base BUY/allocation: `S3a_dynamic_raw_weight`
- stop_loss: `-8%` fixed
- full_backtest_executed: `false`
- existing_model_overwritten: `false`
- profile_changed: `false`
- historical_predictions_regenerated: `false`
- leakage_risk: `low`
- blocking_issues: `0`
- best_variant: `B5_2_proba_drop_larger`
- variants meeting minimum target: `B5_2_proba_drop_larger`, `B5_3_both_relaxed`
- ready_for_phase12c: `true`
- recommended next phase: `Phase12-C dynamic allocation + recalibrated exit`

Key results:

| variant | net profit | PF | DD | utilization | avg holding days | opportunity exits |
|---|---:|---:|---:|---:|---:|---:|
| `B5_0_baseline` | `39,770` | `1.5971` | `-2.66%` | `0.1007` | `6.20` | `47` |
| `B5_2_proba_drop_larger` | `71,922` | `2.1827` | `-3.24%` | `0.1613` | `13.34` | `19` |
| `B5_3_both_relaxed` | `71,450` | `2.0482` | `-4.02%` | `0.1755` | `14.15` | `17` |

Important interpretation:

- The main early-exit lever was not rank floor; lowering rank floor to `0.30`
  had no effect.
- Increasing opportunity_drop_threshold from `0.15` to `0.30` improved profit,
  PF, holding days, and capital utilization together.
- Confirmation-day and profit/loss-only variants were worse.
- Phase 12-C can now test the integrated 2025-only configuration:
  `A3_3 Dynamic Allocation + B5_2 recalibrated Opportunity Exit`.

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
- v2_95: PM-disabled equal-weight baseline; rejected because PM/allocator behavior is necessary.
- v2_96 / v2_97: score-based PM rule research; not promoted because bucket quality and headline metrics were insufficient.

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

Phase 10 / Phase 11 decision:

- Stop PM AI multiplier redevelopment for now.
- Do not promote PM AI v3, PM-disabled baseline, or score-based PM rules.
- Phase 11-A Valuation Engine Dataset Audit is complete.
- Phase 11-B Valuation Engine Prototype is complete.
- Phase 11-C Capital Allocation Engine Prototype is complete.
- Phase 11-C2 Budget Usage Constraint Audit is complete.
- Phase 11-D Limited Combined Backtest is complete.
- Phase 11-E Limited Exit / DD Guard is complete.
- Phase 11-F Limited Robustness Check is complete.
- Phase 11-G Limited Out-of-Sample Year Check is complete.
- Phase 11-H Cooldown / Minimum Holding Guard is complete.
- Phase 11-I Strict Walk-Forward OOS Prototype is complete.
- Phase 11-B2 Strict OOS Failure Diagnosis is complete.
- Phase 11-B3 Expected Downside Model Prototype is complete.
- Phase 12-A Dynamic Capital Allocation Research is complete.
- Phase 12-A2 Allocation Score Refinement is complete.
- Phase 12-A3 Top5 Penalty Refinement is complete.
- Phase 12-B Limited Allocation Strategy Check is complete.
- Phase 12-B2 Allocation Execution Adjustment is complete.
- Phase 12-B3 Exit / Hold Decision Audit is complete.
- Phase 12-B4 Trailing Exit Prototype is complete.
- Phase 12-B5 Exit Threshold Recalibration is complete.
- Do not proceed to broader backtests or adoption from Phase 11-I results.
- Recommended next step is Phase 12-C dynamic allocation + recalibrated exit.
- Do not overwrite current PM AI, current Exit AI, or v2_82.
- Do not use backtest results, trades, profit, cash, portfolio, selected,
  bought, affordable, or current PM multiplier as Phase 11 / Phase 12 features.

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

This handoff was updated at the start of Phase 11 and should be kept aligned
with the Phase 11 Valuation + Allocation plan.

Recent active work includes:

- Phase 9 PM AI v3 research and rejection
- Phase 10-A score-based PM rule research
- Phase 10 stop-and-hold decision
- Phase 11 Valuation Engine + Capital Allocation Engine plan
- Phase 11-A Valuation Engine Dataset Audit implementation
- Phase 11-B Valuation Engine Prototype implementation
- Phase 11-C Capital Allocation Engine Prototype implementation
- Phase 11-C2 Budget Usage Constraint Audit implementation
- Phase 11-D Limited Combined Backtest implementation
- Phase 11-E Limited Exit / DD Guard implementation
- Phase 11-F Limited Robustness Check implementation
- Phase 11-G Limited Out-of-Sample Year Check implementation
- Phase 11-H Cooldown / Minimum Holding Guard implementation

Current generated reports of interest:

```text
reports/ml/phase9a_pm_ai_rearchitecture_audit_2023-01_to_2026-05.md
reports/ml/phase9g_pm_disabled_equal_weight_backtest_2023-01_to_2026-05.md
reports/ml/phase10a_score_based_pm_rule_backtest_2023-01_to_2026-05.md
reports/ml/phase11a_valuation_dataset_audit_2023-01_to_2026-05.md
reports/ml/phase11b_valuation_engine_prototype_2025_holdout.md
reports/ml/phase11c_capital_allocation_prototype_2025.md
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
