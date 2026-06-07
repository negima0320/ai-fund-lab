# Portfolio Manager AI Phase 4-C to 4-G Audit Summary

Last updated: `2026-06-07`

This note summarizes the v2_78 vs v2_79 high-PM minimum-hold investigation and
the clean v2_80 pre-audit. The current main candidate remains:

```text
rookie_dealer_02_v2_78_pm_aware_order_fallback_w025
```

v2_79 is kept on hold. Its headline backtest is better than v2_78, but the
improvement was not caused by the intended minimum-hold guard.

## Scope and Constraints

Period:

```text
2023-01-01 to 2026-05-31
```

Compared profiles:

```text
v2_78: rookie_dealer_02_v2_78_pm_aware_order_fallback_w025
v2_79: rookie_dealer_02_v2_79_high_pm_min_hold_5d
v2_79: rookie_dealer_02_v2_79_high_pm_min_hold_7d
```

Constraints kept during these audits:

- no full pytest unless explicitly requested
- no full backtest except the specifically requested v2_78 rerun
- no J-Quants API refetch
- no OpenAI API use
- no historical prediction regeneration with the current model
- no live order placement
- existing logs and cached prices were used for audits

## Phase 4-C Result

v2_79 5d and 7d produced identical results in the current logs.

| profile | net_profit | PF | DD | win_rate | trades |
|---|---:|---:|---:|---:|---:|
| v2_78 w0.25 | `3,054,794` | `2.6194` | `-7.47%` | `53.78%` | `505` |
| v2_79 5d | `3,544,602` | `2.7219` | `-6.49%` | `55.25%` | `517` |
| v2_79 7d | `3,544,602` | `2.7219` | `-6.49%` | `55.25%` | `517` |

Capital and affordability:

| profile | selected_but_not_affordable | insufficient_available_cash | cash_idle_days |
|---|---:|---:|---:|
| v2_78 w0.25 | `253` | `74` | `112` |
| v2_79 5d | `234` | `60` | `100` |

High-PM trades:

| profile | high PM net_profit | PF | win_rate |
|---|---:|---:|---:|
| v2_78 w0.25 | `1,314,581` | `3.2283` | `65.48%` |
| v2_79 5d | `1,622,577` | `3.5861` | `67.23%` |

However, both v2_79 variants had:

```text
high_pm_min_hold_blocked_exit_count = 0
```

This invalidated the first interpretation that v2_79 improved because the
minimum-hold guard directly blocked early Exit AI sells.

Report:

```text
reports/ml/portfolio_manager_phase4c_high_pm_min_hold_2023-01_to_2026-05.md
```

## Phase 4-D Difference Audit

Phase 4-D compared v2_78 and v2_79 trades and purchase audit rows.

Trade summary:

| metric | value |
|---|---:|
| only_v2_78_buy_count | `21` |
| only_v2_79_buy_count | `33` |
| common_buy_count | `486` |
| sell_date_changed_count | `19` |
| changed_trade_count | `186` |

Profit difference:

```text
v2_79 - v2_78 = +489,807
```

Contribution buckets:

| bucket | profit_delta |
|---|---:|
| buy_universe_change | `+161,364` |
| position_size_change | `+111,377` |
| sell_timing_change | `+99,883` |
| affordability_change | `+73,291` |
| unknown | `+43,893` |
| high_pm_change | `0` |

The largest bucket was `buy_universe_change`, not a direct high-PM hold effect.

Report:

```text
reports/ml/portfolio_manager_phase4d_v278_vs_v279_diff_audit_2023-01_to_2026-05.md
```

## Phase 4-E v2_78 Rerun Check

The current-code v2_78 w0.25 backtest was rerun under the same 2023-01 to
2026-05 condition. It matched the old v2_78 metrics and did not become v2_79.

This ruled out the possibility that the v2_79 improvement was only stale-log or
current-code drift.

Conclusion:

- v2_78 remains reproducible.
- v2_79 remains numerically stronger.
- v2_79 still cannot be adopted as a minimum-hold improvement because
  `blocked_exit_count=0`.

## Phase 4-F Side-Effect Root Cause Audit

Phase 4-F audited why v2_79 diverged despite no minimum-hold block.

Profile config diff:

- focused trading/config keys matched except high-PM minimum-hold settings
- raw/effective PM policy diff was limited to:
  - `high_pm_min_hold_enabled`
  - `high_pm_min_hold_days`
  - `high_pm_min_hold_min_multiplier`

Static code reference audit:

- `high_pm_min_hold` is referenced in policy extraction, trade-field logging,
  position snapshot propagation, and Exit guard logic.
- No intentional BUY ordering, sizing, affordability, or cash allocation branch
  was found that should use `high_pm_min_hold` as a BUY-side decision signal.

First path divergence:

| item | value |
|---|---|
| first_divergence_date | `2023-01-24` |
| divergence_type | `sell_difference` |
| code | `71570` |
| v2_78 | sold on `2023-01-24` by `Exit AI avoid_loss_5d` |
| v2_79 | did not sell on `2023-01-24`; later sold on `2023-01-25` by take profit |

Minimum-hold confirmation:

| metric | value |
|---|---:|
| high_pm_min_hold_enabled | `true` |
| high_pm_min_hold_days | `5` |
| high_pm_target_position_count | `177` |
| high_pm_exit_ai_signal_under_min_hold_count | `0` |
| blocked_exit_count | `0` |
| bug_candidate | `false` |

So `blocked_exit_count=0` is consistent. There were no high-PM positions with
`holding_days < 5` where Exit AI actually signaled an exit.

Side-effect judgement:

| flag | value |
|---|---|
| v279_improvement_explained | `true` |
| minimum_hold_directly_effective | `false` |
| v279_safe_to_adopt | `false` |
| should_create_clean_v280_from_discovered_effect | `true` |

Report:

```text
reports/ml/portfolio_manager_phase4f_side_effect_audit_2023-01_to_2026-05.md
```

## Phase 4-G Clean Exit Delay / Candidate Presence Hold Audit

Phase 4-G tested whether the v2_79 side effect could be turned into a clean
v2_80 rule.

Exit AI 1-day delay over all v2_78 Exit AI sells:

| metric | value |
|---|---:|
| trade_count | `23` |
| actual_net_profit | `125,495` |
| virtual_delay_1d_net_profit | `43,795` |
| profit_delta | `-81,700` |
| actual_pf | `4.8401` |
| virtual_pf | `1.5861` |
| actual_win_rate | `65.22%` |
| virtual_win_rate | `60.87%` |
| positive_delta_count | `8` |
| negative_delta_count | `15` |

Conclusion:

- blanket Exit AI 1-day delay is harmful and should not be implemented.

Candidate Presence Hold virtual rules:

| rule | held_trade_count | profit_delta |
|---|---:|---:|
| A: same-day candidate and `pm_score >= 0` | `0` | `0` |
| B: next-day candidate and `pm_score >= 0` | `0` | `0` |
| C: same/next candidate `pm_multiplier >= 1.0` | `0` | `0` |
| D: trade `pm_multiplier >= 1.15` | `2` | `+11,200` |

The candidate-presence rules did not fire with the available
`purchase_audit.csv` reconstruction. The only positive clean candidate was a
limited high-PM rule using the trade's own `pm_multiplier >= 1.15`.

71570 reproduction:

| item | value |
|---|---:|
| actual_sell_date | `2023-01-24` |
| actual_net_profit | `11,315` |
| virtual_hold_sell_date | `2023-01-25` |
| virtual_net_profit | `14,515` |
| profit_delta | `+3,200` |
| same_day_candidate_present | `false` |
| next_day_candidate_present | `false` |
| pm_score | `0.3405` |
| pm_multiplier | `1.15` |
| matched rule | `D_trade_pm_multiplier_gte_1_15` |

v2_79 Top Profit Delta reproducibility:

| metric | value |
|---|---:|
| target codes | `96970, 34960, 70120, 56310, 58050, 58380, 31970, 91010` |
| explained_count | `0` |
| unexplained_count | `8` |
| explained_profit_delta | `0` |
| unexplained_profit_delta | `318,912` |

Clean v2_80 judgement:

| flag | value |
|---|---|
| exit_delay_1d_recommended | `false` |
| candidate_presence_hold_recommended | `false` |
| pm_multiplier_presence_hold_recommended | `true` |
| clean_v280_worth_implementing | `true` |
| v279_side_effect_reproducible_by_clean_rule | `false` |
| best_rule | `D_trade_pm_multiplier_gte_1_15` |
| best_rule_profit_delta | `+11,200` |
| log_reliability | `medium_partial_purchase_audit_candidates` |

Report:

```text
reports/ml/portfolio_manager_phase4g_exit_delay_candidate_hold_audit_2023-01_to_2026-05.md
```

## Current Decision

Decision:

```text
Keep v2_78 w0.25 as the main candidate.
Keep v2_79 on hold.
Do not adopt v2_79 as the main profile.
```

Rationale:

- v2_79's headline metrics are better.
- The intended minimum-hold guard did not directly fire.
- The first improvement path was a sell-timing path divergence, not a proven
  deliberate minimum-hold behavior.
- Phase 4-G did not explain the top v2_79 profit deltas with a clean rule.
- A blanket Exit AI 1-day delay is harmful.

Possible next step:

```text
clean v2_80: a narrow high-PM Exit AI one-day delay candidate
```

This is only worth testing as a small, explicit experiment because Phase 4-G
found `+11,200` for `pm_multiplier >= 1.15`, including the 71570 reproduction.
It should not be treated as explaining or reproducing v2_79's overall gain.

## New Audit Implementations

Phase 4-D:

```text
scripts/ml/report_portfolio_manager_phase4d_v278_vs_v279_diff_audit.py
tests/test_ml_portfolio_manager_phase4d_diff_audit.py
```

Phase 4-F:

```text
src/ml/portfolio_manager_phase4f_side_effect_audit.py
scripts/ml/audit_portfolio_manager_phase4f_side_effect.py
tests/test_ml_portfolio_manager_phase4f_side_effect_audit.py
```

Phase 4-G:

```text
src/ml/portfolio_manager_phase4g_exit_delay_candidate_hold.py
scripts/ml/audit_portfolio_manager_phase4g_exit_delay_candidate_hold.py
tests/test_ml_portfolio_manager_phase4g_exit_delay_candidate_hold.py
```

Quick checks run:

```text
py_compile for Phase 4-F and 4-G scripts/modules: passed
pytest -q tests/test_ml_portfolio_manager_phase4f_side_effect_audit.py: 2 passed
pytest -q tests/test_ml_portfolio_manager_phase4g_exit_delay_candidate_hold.py: 2 passed
```

Earlier Phase 4-D quick check:

```text
py_compile scripts/ml/report_portfolio_manager_phase4d_v278_vs_v279_diff_audit.py: passed
pytest -q tests/test_ml_portfolio_manager_phase4d_diff_audit.py: 2 passed
```

