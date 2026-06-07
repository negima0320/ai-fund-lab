# Portfolio Manager AI Phase 5-G to 6-G Audit Summary

This note summarizes the integration checks and market/capital-allocation audits
that followed Exit AI v2 full training. The main outcome is that Exit AI v2 was
not adopted, while a pure `per_code_exposure_cap` relaxation to `0.38` produced
the strongest current full-backtested research candidate.

## Current Decision

Current strongest full-backtested research candidate:

```text
rookie_dealer_02_v2_82_cap38
```

Previous main candidate:

```text
rookie_dealer_02_v2_78_pm_aware_order_fallback_w025
```

Decision:

- Promote `v2_82_cap38` to main research candidate.
- Keep `v2_78 w0.25` as the conservative fallback/reference.
- Do not adopt Exit AI v2 integration profiles.
- Do not adopt Bear Booster yet; its intended effect was mostly absorbed by
  per-code cap constraints.

## Phase 5-G: Exit AI v2 Prediction / Integration Audit

Report:

```text
reports/ml/phase5g_exit_ai_v2_prediction_audit_2023-01_to_2026-05.md
```

Scope:

- audit only;
- no full backtest;
- no profile creation;
- no current model overwrite;
- Exit AI v2 candidate model read-only.

Target summary:

| item | value |
|---|---:|
| sell rows | `505` |
| existing Exit AI sell rows | `23` |
| early-sell suspect rows | `369` |
| loss-avoidance rows | `232` |
| high-PM rows | `168` |

Prediction coverage:

| item | value |
|---|---:|
| prediction available | `478` |
| prediction missing | `27` |
| coverage rate | `94.65%` |
| top-decile count | `48` |
| average score | `0.1387` |

Existing Exit AI comparison:

| metric | value |
|---|---:|
| agreement count | `438` |
| disagreement count | `67` |
| existing Exit AI only | `21` |
| Exit AI v2 only | `46` |

Interpretation:

- Exit AI v2 score was available for most v2_78 sell rows.
- Top-decile rows tended to have weaker post-exit returns than non-top rows,
  which is directionally useful for an exit-risk score.
- High-PM non-top-decile rows still had positive post-exit returns, suggesting
  a possible high-PM suppression rule.
- The direct integration signal was not strong enough to adopt without a
  controlled profile backtest.

Recommended next phase from the audit:

```text
Phase 5-H Exit AI v2 Suppression Rule for high PM
```

Integrity:

- API-only dataset rows were used for prediction.
- No forbidden or label-like feature columns were detected.
- `selected_count_in_day` was not used.
- The candidate model was loaded from `models/ml/exit_ai_v2/candidate_v2_api_only`.
- `models/ml/exit/current_v2_66` was not overwritten.
- Feature schema matched training metadata.

## Phase 5-H: Exit AI v2 Conservative Gate Backtests

Report:

```text
reports/ml/portfolio_manager_phase5h_exit_ai_v2_gate_2023-01_to_2026-05.md
```

Profiles:

| profile | purpose |
|---|---|
| `rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate` | conservative Exit AI v2 gate |
| `rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate_high_pm_safe` | high-PM-safe variant |

Result:

| variant | net_profit | PF | DD | win_rate | monthly_win_rate | total_trades | avg_holding_days | avg_capital_utilization |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_78 baseline | `3,054,794` | `2.6194` | `-7.47%` | `53.78%` | `75.61%` | `505` | `3.97` | `38.31%` |
| v2_80 conservative gate | `2,318,919` | `2.2956` | `-10.12%` | `53.36%` | `73.17%` | `509` | `3.67` | `35.70%` |
| v2_80 high PM safe | `2,859,266` | `2.5615` | `-8.75%` | `54.06%` | `75.61%` | `509` | `3.78` | `36.75%` |

Decision:

- Do not adopt v2_80.
- Keep `models/ml/exit/current_v2_66` as the active Exit AI.
- Exit AI v2 remains a candidate model, not a production/profile component.

## Phase 6-A: Market Regime Audit

Report:

```text
reports/ml/phase6a_market_regime_audit_2023-01_to_2026-05.md
```

Purpose:

- determine which regimes v2_78 was strong or weak in;
- test the hypothesis "attack when TOPIX is strong, defend when TOPIX is weak."

Important result:

- v2_78 was not weak in Bear regimes.
- The Bear bucket contributed substantial profit despite limited trade count.
- Naive Bear suppression was harmful.

Selected result from the initial regime audit:

| regime | trades | net_profit | PF | win_rate |
|---|---:|---:|---:|---:|
| Bull | `340` | `1,146,203` | `1.7673` | n/a |
| Bear | `73` | `1,032,000` | `3.2560` | n/a |

Decision:

- Do not implement a simple Bull-only or Bear-off filter.
- Continue with Bear winner diagnostics instead.

## Phase 6-B: Bear Market Winner Audit

Report:

```text
reports/ml/phase6b_bear_market_winner_audit_2023-01_to_2026-05.md
```

After deduped TOPIX regime assignment, Bear trades were:

| metric | value |
|---|---:|
| Bear trades | `40` |
| Bear net_profit | `286,996` |
| Bear PF | `3.3164` |
| Bear win_rate | `70.00%` |

Observed Bear winner patterns:

- `volume_ratio >= 2` was broadly common.
- `pm_multiplier=1.30` was strong.
- `pm_multiplier=0.80` also contributed positive profit and should not be cut.
- Short holding periods around `3-5` days were common.
- Retail, service, and information/communication names appeared frequently.

Decision:

- PM AI appears to capture part of the Bear alpha.
- Do not add a Bear-only mode yet.
- Audit booster-style sizing before any logic change.

## Phase 6-C: Bear Alpha Booster Audit

Report:

```text
reports/ml/phase6c_bear_alpha_booster_audit_2023-01_to_2026-05.md
```

Key condition results:

| condition | trades | net_profit | PF | win_rate |
|---|---:|---:|---:|---:|
| Bear | `40` | `286,996` | `3.3164` | `70.00%` |
| Bear + PM 1.30 | `11` | `175,226` | `10.3965` | `90.91%` |
| Bear + PM>=1.15 | `15` | `202,119` | `11.8387` | `93.33%` |
| Bear + holding_days>=3 | `32` | `333,461` | `9.7225` | `84.38%` |

Booster approximation:

| rule | profit_delta | note |
|---|---:|---|
| Bear & PM 1.30 buy amount +25% | `+43,806` | positive |
| Bear & PM>=1.15 buy amount +50% | `+101,060` | best simple booster candidate |
| keep only Bear & PM 1.30 | `-111,771` | harmful because other Bear winners are removed |

PM 0.80 Bear contribution:

| metric | value |
|---|---:|
| trades | `13` |
| profit | `88,217` |
| PF | `3.0773` |
| win_rate | `69.23%` |

Decision:

- Bear alpha exists.
- PM AI captures part of it, but not all of it.
- Do not delete PM 0.80 trades.
- Test only additive sizing rules.

## Phase 6-D: Bear Booster Design Audit

Report:

```text
reports/ml/phase6d_bear_booster_design_audit_2023-01_to_2026-05.md
```

Recommended audit candidate:

```text
Rule_B_Bear_PM_gte_1_15_buy_amount_plus_50pct
```

Key result:

| item | value |
|---|---:|
| matched trades | `15` |
| profit_delta approximation | `+101,060` |
| PF approximation | `3.9128` |
| DD risk | `medium` |
| capital utilization impact | `+19.77%` |
| PM 0.80 removal loss | `-88,217` |

Decision:

- A safe booster candidate exists.
- Keep v2_78 as primary until full backtest proves otherwise.
- Implement a narrow Bear & PM>=1.15 booster profile for validation.

## Phase 6-E: Bear Alpha Booster Profile

Profile:

```text
rookie_dealer_02_v2_81_bear_pm115_booster_50
```

Report:

```text
reports/ml/portfolio_manager_phase6e_bear_booster_2023-01_to_2026-05.md
```

Implementation:

- Base: `rookie_dealer_02_v2_78_pm_aware_order_fallback_w025`
- Applies only when:
  - Bear regime by TOPIX MA25/MA75;
  - `pm_multiplier >= 1.15`;
  - profile booster flag is enabled.
- Multiplies desired post-PM sizing by `1.5`.
- Existing caps still apply: available cash, daily buy limit, round lot,
  max positions, and per-code exposure cap.
- Does not affect PM 0.80 / PM 1.0, Bull / Neutral / Unknown, sell logic, or
  Exit AI.

Full backtest result:

| metric | v2_78 | v2_81 |
|---|---:|---:|
| net_profit | `3,054,794` | `3,054,794` |
| PF | `2.6194` | `2.6194` |
| DD | `-7.47%` | `-7.47%` |
| win_rate | `53.78%` | `53.78%` |
| average capital utilization | `38.31%` | `38.31%` |

Booster diagnostics:

| metric | value |
|---|---:|
| booster applied count | `10` |
| desired incremental amount | `3,566,800` |
| boosted trades profit | `164,070` |
| boosted trades win_rate | `90.00%` |

Decision:

- Do not adopt v2_81.
- The booster signal itself was promising, but effective P/L did not change.
- The likely blocker was per-code exposure cap absorption.

## Phase 6-F: Cap Constraint Audit

Report:

```text
reports/ml/phase6f_cap_constraint_audit_2023-01_to_2026-05.md
```

Current setting:

```text
per_code_exposure_cap_rate = 0.30
```

Cap audit:

| metric | value |
|---|---:|
| purchase rows | `1,375` |
| cap hit count | `338` |
| cap reduction amount | `140,475,510` |
| cap prevented buy amount | `140,475,510` |

Virtual cap relaxation:

| cap_rate | newly_allowed_amount | profit approximation |
|---:|---:|---:|
| `0.35` | `91,319,916` | `+325,938` |
| `0.40` | `114,782,084` | `+511,396` |
| `0.50` | `137,457,586` | `+820,943` |

PM high-score cap stop:

| bucket | trade_count | reduction_amount | estimated_profit |
|---|---:|---:|---:|
| PM>=1.15 | `124` | `54,704,980` | `+573,568` |
| PM>=1.30 | `86` | `38,440,930` | `+449,411` |

Verdict:

| item | value |
|---|---|
| cap_is_current_bottleneck | `true` |
| cap_relaxation_worth_testing | `true` |
| safest_cap_candidate | `Rule_A_cap_35pct` |
| expected_profit_direction | `positive` |
| expected_dd_direction | `worse_or_uncertain` |
| ready_for_phase6g | `true` |

Decision:

- The current `30%` per-code cap likely suppresses profitable sizing.
- A pure cap change should be tested before more complex booster logic.

## Phase 6-G: per_code_cap 38% Profile

Profile:

```text
rookie_dealer_02_v2_82_cap38
```

Aliases:

```text
rookie_dealer_02_v2_82
rookie_dealer_02_v2.82
```

Report:

```text
reports/ml/portfolio_manager_phase6g_cap38_2023-01_to_2026-05.md
```

Implementation:

- Base: `rookie_dealer_02_v2_78_pm_aware_order_fallback_w025`
- Only changed:

```yaml
portfolio_manager_ai_sizing:
  per_code_exposure_cap_rate: 0.38
```

- No Bear Booster.
- No Exit AI v2.
- Current Exit AI remains `models/ml/exit/current_v2_66`.
- Current Portfolio Manager model remains
  `models/ml/portfolio_manager/current_v2_73_phase3b_clean`.

Full backtest comparison:

| metric | v2_78 | v2_82 cap38 | delta |
|---|---:|---:|---:|
| net_profit | `3,054,794` | `3,777,545` | `+722,751` |
| PF | `2.6194` | `2.7309` | `+0.1115` |
| DD | `-7.47%` | `-6.54%` | `+0.93pt` |
| win_rate | `53.78%` | `55.11%` | `+1.33pt` |
| monthly_win_rate | `75.61%` | `78.05%` | `+2.44pt` |
| total_trades | `505` | `502` | `-3` |
| average_holding_days | `3.97` | `4.01` | `+0.04` |
| average_capital_utilization | `38.31%` | `40.49%` | `+2.18pt` |
| selected_but_not_affordable | `253` | `257` | `+4` |
| insufficient_available_cash | `74` | `71` | `-3` |
| per_code_cap_skip_or_reduction_count | `50` | `16` | `-34` |

Concentration:

| metric | v2_78 | v2_82 cap38 | direction |
|---|---:|---:|---|
| single-code profit concentration | `8.11%` | `6.83%` | improved |
| top5-code profit concentration | `15.20%` | `14.12%` | improved |

Assets:

| metric | v2_78 | v2_82 cap38 |
|---|---:|---:|
| initial assets | `1,000,000` | `1,000,000` |
| final assets | `4,813,588` | `5,720,597` |
| CAGR, profile initial capital basis | `58.51%` | `66.74%` |

Date coverage caveat:

- Backtest requested `2023-01-01` to `2026-05-31`.
- Last cached price/trading day was `2026-05-29`.
- The engine reports coverage audit `ERROR / coverage_ok=false` because
  `2026-05-31` was beyond the cached price end; the comparison logs and summary
  were still generated from the same cached-data basis.

Decision:

- Promote `v2_82_cap38` to the strongest current full-backtested research
  candidate.
- Keep `v2_78 w0.25` as conservative fallback/reference.
- The next validation should focus on robustness rather than adding another
  logic layer immediately:
  - year-by-year result comparison;
  - regime-by-regime comparison;
  - cap-hit residual analysis under `0.38`;
  - DD-period and top-code concentration checks;
  - optional full pytest when a release-style checkpoint is needed.

## Current Profile Ranking After Phase 6-G

| priority | profile | status |
|---:|---|---|
| 1 | `rookie_dealer_02_v2_82_cap38` | strongest full-backtested candidate; cap 0.38 improved profit/PF/DD/win rate without concentration worsening |
| 2 | `rookie_dealer_02_v2_78_pm_aware_order_fallback_w025` | conservative fallback/reference; prior main candidate |
| deferred | `rookie_dealer_02_v2_81_bear_pm115_booster_50` | booster fired but effective performance was unchanged due cap absorption |
| rejected | `rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate` | worse profit/PF/DD than v2_78 |
| rejected | `rookie_dealer_02_v2_80_exit_ai_v2_conservative_gate_high_pm_safe` | improved over v2_80 but still below v2_78 |
| on hold | `rookie_dealer_02_v2_79_high_pm_min_hold_5d` | Phase 4-F showed the intended minimum-hold guard did not directly fire |

