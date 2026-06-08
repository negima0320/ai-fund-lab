# Portfolio Manager AI Phase 8-A to 8-H PM AI Redesign Summary

This note summarizes the Phase 8 work after `rookie_dealer_02_v2_82_cap38`
became the Version 1.0 Candidate. The purpose of Phase 8 was not to improve the
headline score immediately, but to understand why v2_82 wins and whether PM AI
can be rebuilt or replaced safely.

## Current Decision

Keep the main candidate unchanged:

```text
rookie_dealer_02_v2_82_cap38
```

Decision:

- Keep v2_82 as the Version 1.0 Candidate.
- Do not promote PM AI v2 candidate.
- Do not promote calibrated PM AI v2.
- Do not promote the rule-based relative allocator v2_92.
- Do not overwrite current PM AI or current Exit AI model directories.
- Treat PM AI research as unresolved model-design work, not as a production
  replacement path yet.

## Baseline: v2_82

Core full-backtest period:

```text
2023-01-01 to 2026-05-31
```

v2_82 core result:

| metric | value |
|---|---:|
| net_profit | `3,777,545` |
| PF | `2.7309` |
| DD | `-6.54%` |
| win_rate | `55.11%` |
| monthly_win_rate | `78.05%` |
| total_trades | `499` |
| average_capital_utilization | `40.49%` |
| final_assets | `5,720,597` |
| CAGR | `66.74%` |

## Phase 8-A: System Understanding and Logic Audit

Report:

```text
reports/ml/phase8a_system_understanding_audit_2023-01_to_2026-05.md
```

Main finding:

- PM AI is one of the core alpha sources in v2_82.
- v2_82 wins through a combination of Stock Selection AI, current PM AI,
  cap38, affordable fallback, and current Exit AI v2_66.
- The system should not be simplified by deleting current PM AI.
- Legacy cleanup remains useful, but not at the expense of v2_82 behavior.

Important conclusion:

```text
pm_ai_contribution_score: high
pm_ai_is_core_alpha: true
```

## Phase 8-B: PM AI Candidate Integration Audit

Report:

```text
reports/ml/phase8b_pm_candidate_integration_audit_2023-01_to_2026-05.md
```

Target:

```text
models/ml/portfolio_manager/candidate_v2_api_only
```

Main finding:

- The API-only PM AI candidate is operationally clean and leakage risk is low.
- It removes candidate-list dependent features.
- But its score distribution and multiplier behavior differ materially from the
  current PM AI.
- Backtest integration was worth testing, but not safe to adopt without a full
  profile check.

## Phase 8-C: PM AI v2 Integration Backtest

Report:

```text
reports/ml/phase8c_pm_ai_v2_backtest_2023-01_to_2026-05.md
```

Profile:

```text
rookie_dealer_02_v2_90_pm_ai_v2_api_only_cap38
```

Result:

| profile | net_profit | PF | DD | win_rate | avg capital utilization |
|---|---:|---:|---:|---:|---:|
| v2_82 | `3,777,545` | `2.7309` | `-6.54%` | `55.11%` | `40.49%` |
| v2_90 | `791,720` | `1.9050` | `-7.36%` | `46.47%` | `21.04%` |

Main failure:

- PM 1.30 count became `0`.
- high conviction rate effectively disappeared.
- capital utilization was roughly halved.
- PM AI v2 was too conservative as a direct replacement.

Decision:

- Do not adopt v2_90.
- Keep current PM AI in v2_82.

## Phase 8-D: PM AI v2 Calibration Redesign Audit

Report:

```text
reports/ml/phase8d_pm_ai_v2_calibration_audit_2023-01_to_2026-05.md
```

Main question:

Could PM AI v2 become usable by changing only the mapping from prediction
scores to PM multipliers?

Main finding:

- PM AI v2 scores can be quantile-calibrated to recreate PM 1.30 counts.
- But calibration alone does not guarantee PM 1.30 profit quality.
- Rule E, which matched the current PM multiplier distribution, became the
  strongest calibration candidate for a real backtest.

Decision:

- Implement a calibrated PM AI v2 profile for empirical testing.
- Do not overwrite current PM AI.

## Phase 8-E: PM AI v2 Calibrated Backtest

Report:

```text
reports/ml/phase8e_pm_ai_v2_calibrated_backtest_2023-01_to_2026-05.md
```

Profile:

```text
rookie_dealer_02_v2_91_pm_ai_v2_calibrated_rule_e_cap38
```

Result:

| profile | net_profit | PF | DD | win_rate | avg capital utilization |
|---|---:|---:|---:|---:|---:|
| v2_82 | `3,777,545` | `2.7309` | `-6.54%` | `55.11%` | `40.49%` |
| v2_90 | `791,720` | `1.9050` | `-7.36%` | `46.47%` | `21.04%` |
| v2_91 | `1,030,079` | `1.6642` | `-14.98%` | `45.29%` | `35.67%` |

PM 1.30 comparison:

| profile | PM 1.30 trades | PM 1.30 profit | PF | win_rate |
|---|---:|---:|---:|---:|
| v2_82 | `122` | `1,326,228` | `3.9239` | `68.03%` |
| v2_90 | `0` | `0` | n/a | n/a |
| v2_91 | `131` | `154,435` | `1.2714` | `46.56%` |

Main finding:

- v2_91 recovered PM 1.30 count, but not PM 1.30 quality.
- PM AI v2 is not merely under-calibrated; it is not learning the current PM
  model's high-quality sizing behavior.

Decision:

- Do not adopt v2_91.
- Keep v2_82.

## Phase 8-F: PM AI v2 Label Redesign Audit

Report:

```text
reports/ml/phase8f_pm_ai_label_redesign_audit_2023-01_to_2026-05.md
```

Main question:

Is PM AI v2 failing because the label design is wrong?

Audit result:

| item | value |
|---|---|
| pm_ai_true_objective | `capital_allocation_for_position_sizing` |
| current_label_is_correct | `true` |
| pm_ai_v2_problem_is_label | `false` |
| pm_ai_v2_problem_is_calibration | `true` |
| ready_for_phase8g_retraining | `false` |
| next_phase_recommended | `Stay current PM` |

Label correlations were not obviously broken:

| label | correlation_to_trade_profit | correlation_to_pm130_profit |
|---|---:|---:|
| future_5d_return | `0.6791` | `0.3598` |
| future_10d_return | `0.4805` | `0.2479` |
| risk_adjusted_future_return | `0.4897` | `0.2488` |
| high_conviction_target | `0.4018` | `0.2428` |
| avoid_target | `-0.3901` | `-0.1617` |

Main finding:

- The labels are not clearly wrong.
- The deeper issue is that PM AI's true role is not plain future-return
  prediction. It is position sizing and relative capital allocation.

Decision:

- Do not retrain PM AI v2 yet.
- First audit whether the PM problem should be reframed as ranking / relative
  allocation.

## Phase 8-G: PM AI Ranking / Relative Allocation Audit

Report:

```text
reports/ml/phase8g_pm_ai_ranking_audit_2023-01_to_2026-05.md
```

Main result:

```text
pm_problem_type_recommended: candidate_ranking_and_relative_capital_allocation
relative_allocation_worth_testing: true
best_next_approach: Phase 8-H Rule-Based Relative Allocator Backtest
```

Daily candidate reconstruction:

| run | days | avg candidate count | avg bought count | avg affordable count |
|---|---:|---:|---:|---:|
| v2_82 | `554` | `2.48` | `0.62` | `0.91` |
| v2_90 | `554` | `2.48` | `0.44` | `0.44` |
| v2_91 | `554` | `2.48` | `0.70` | `0.71` |

Current PM 1.30 relative rank audit:

| score | median rank | median percentile | top candidate rate |
|---|---:|---:|---:|
| expected_return_10d | `2.0` | `0.25` | `32.22%` |
| bad_entry_probability_10d | `2.0` | `0.33` | `28.89%` |
| risk_adjusted_score | `2.0` | `0.33` | `28.89%` |

API-only relative features:

Allowed if computed from the prediction-time candidate pool before cash /
portfolio / trade decisions:

- `candidate_count_in_day`
- `rank_in_day`
- `score_rank_in_day`
- `percentile_in_day`
- `gap_to_best`
- `candidate_strength`

Still forbidden:

- `selected_count_in_day`
- `bought_count_in_day`
- `affordable_count_in_day`
- `cash_after`
- `portfolio_state`
- `position_state`
- `backtest_outcome`

Decision:

- Test a simple rule-based relative allocator before designing a new ranking
  model.

## Phase 8-H: Rule-Based Relative Allocator Backtest

Report:

```text
reports/ml/phase8h_relative_allocator_backtest_2023-01_to_2026-05.md
```

Profile:

```text
rookie_dealer_02_v2_92_relative_allocator_cap38
```

Aliases:

```text
rookie_dealer_02_v2_92
rookie_dealer_02_v2.92
```

Implemented rule variants:

- Rule A: `risk_adjusted_score_rank`
- Rule B: `expected_return_10d_rank`
- Rule C: `blended_relative_score`
- Rule D: `conservative_blend`
- Rule E: `no_pm_baseline`

The actual backtest used Rule C:

```text
relative_allocator.rule: blended_relative_score
```

Rule C:

```text
relative_score =
  0.45 * risk_adjusted_score_percentile
+ 0.35 * expected_return_10d_percentile
+ 0.20 * inverse_bad_entry_probability_percentile
```

Multiplier:

| condition | multiplier |
|---|---:|
| rank 1 or top 10% | `1.30` |
| top 25% | `1.15` |
| middle | `1.00` |
| bottom 25% | `0.80` |

Result:

| profile | net_profit | PF | DD | win_rate | monthly_win_rate | avg utilization |
|---|---:|---:|---:|---:|---:|---:|
| v2_82 | `3,777,545` | `2.7309` | `-6.54%` | `55.11%` | `78.05%` | `40.49%` |
| v2_90 | `791,720` | `1.9050` | `-7.36%` | `46.47%` | `56.10%` | `21.04%` |
| v2_91 | `1,030,079` | `1.6642` | `-14.98%` | `45.29%` | `63.41%` | `35.67%` |
| v2_92 | `940,647` | `1.5118` | `-11.56%` | `43.89%` | `60.98%` | `42.75%` |

PM multiplier quality in v2_92:

| multiplier | trades | profit | PF | win_rate |
|---:|---:|---:|---:|---:|
| `1.30` | `315` | `511,617` | `1.3790` | `46.67%` |
| `1.15` | `19` | `-55,071` | `0.4121` | `26.32%` |
| `1.00` | `121` | `-2,143` | `0.9959` | `43.80%` |
| `0.80` | `71` | `17,719` | `1.0530` | `35.21%` |

Relative allocator diagnostics:

| item | value |
|---|---:|
| relative allocator rows | `1,375` |
| PM 1.30 relative rank median | `1.0` |
| PM 0.80 relative rank median | `4.0` |
| candidate_count median | `4.0` |
| relative_score median | `0.65` |

Risk / affordability:

| profile | selected_but_not_affordable | insufficient_available_cash | cap skip/reduction |
|---|---:|---:|---:|
| v2_82 | `257` | `71` | `188` |
| v2_92 | `530` | `271` | `193` |

Phase 8-H judgement:

| item | value |
|---|---|
| relative_allocator_operationally_valid | `true` |
| relative_allocator_result_acceptable | `false` |
| relative_allocator_beats_pm_ai_v2 | `false` |
| relative_allocator_close_to_v282 | `false` |
| pm_ai_ranking_hypothesis_supported | `false` |
| next_phase_recommended | `Stay v2_82 current PM` |

Important interpretation:

- The implementation is operationally valid.
- PM 1.30 was correctly concentrated at same-day relative rank 1.
- However, PM 1.30 quality was poor compared with current PM AI.
- Simple Stock Selection relative rank is not enough to replace current PM AI.
- v2_92 also worsened affordability pressure.

Final decision:

- Do not adopt v2_92.
- Keep v2_82 as the Version 1.0 Candidate.
- Do not continue PM AI replacement by simple relative score rules alone.

## Overall Phase 8 Conclusion

Current PM AI remains valuable even though it is less clean than the API-only
candidate.

The API-only PM AI candidate is technically clean, but it fails as a production
replacement because it does not reproduce high-quality PM 1.30 sizing.

The rule-based relative allocator confirms that same-day relative ranking is a
useful diagnostic frame, but a simple rank rule does not recover current PM AI's
alpha.

The safest current state is:

```text
Version 1.0 Candidate: rookie_dealer_02_v2_82_cap38
Fallback/reference: rookie_dealer_02_v2_78_pm_aware_order_fallback_w025
Do not promote: v2_90, v2_91, v2_92
```

## Next Work Candidates

Recommended:

- Freeze v2_82 as the Version 1.0 Candidate unless a clearly superior profile
  appears.
- Treat Phase 8 PM AI v2 work as research, not a near-term replacement.
- If PM AI research resumes, design a true ranking / allocation dataset using
  API-only future utility labels rather than simple standalone classification.

Not recommended now:

- Current PM AI replacement.
- PM AI v2 promotion.
- v2_92 promotion.
- Full PM AI retraining without a better objective.
