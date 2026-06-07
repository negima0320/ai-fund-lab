# ML Phase 25 to Portfolio Manager AI Summary

This document summarizes the ML work after Phase 24, including the 5-year
walk-forward validation, enriched features, ML-integrated backtest profiles,
Exit AI, capital allocation experiments, and Portfolio Manager AI through
Phase 4-C.

The overall direction changed from report-only ML analysis toward isolated
backtest profiles. Existing baseline profiles were kept intact, and every
trading-logic experiment was introduced as a new profile.

## Scope

Implemented:

- 5-year ML walk-forward preparation and validation
- enriched v2 feature set with financial statements and TOPIX relative features
- daily AI candidate operation mode
- walk-forward model audit
- `current_enriched_v2` model directory support
- ML prediction join into backtest candidates
- ML-ranked backtest profiles
- Exit AI dataset, training, and post-analysis
- Exit AI backtest profiles
- scaled-buy and capital allocation profile variants
- v2_73 adoption notes
- Portfolio Manager AI dataset, data-lineage audit, training, light evaluation, full backtest profiles, PM-aware ordering, and high-PM minimum hold profiles

Still not implemented:

- live trading integration
- automatic order placement
- automatic retraining
- broker execution connection
- using Portfolio Manager AI as a production allocator

Safety constraints kept:

- no OpenAI API use
- no J-Quants API refetch from backtest/profile tests
- no current model regeneration for historical predictions
- walk-forward predictions are used for historical backtests
- existing baseline profiles are not destructively modified

## Key Data and Model Artifacts

Important generated artifacts are intentionally not committed when ignored:

- `data/ml/`
- `models/ml/`
- `reports/ml/`
- `logs/backtests/`

Important model directories used during the later phases:

- `models/ml/current_enriched_v2/`
- `models/ml/exit/current_v2_66/`
- `models/ml/portfolio_manager/current_v2_73_phase3b_clean/`

Important report examples:

- `reports/ml/walk_forward_5y_enriched_v2_2023-01_to_2026-05.md`
- `reports/ml/ml_audit_5y_enriched_v2.md`
- `reports/ml/ml_exit_ai_backtest_comparison_2023-01_to_2026-05.md`
- `reports/ml/capital_allocation_phase5_v2_73_comparison_2023-01_to_2026-05.md`
- `reports/ml/portfolio_manager_phase3d_full_backtest_2023-01_to_2026-05.md`
- `reports/ml/portfolio_manager_phase3d_detail_audit_2023-01_to_2026-05.md`
- `reports/ml/portfolio_manager_phase3j_affordability_audit_2023-01_to_2026-05.md`
- `reports/ml/portfolio_manager_phase3k_candidate_ranking_audit_2023-01_to_2026-05.md`
- `reports/ml/portfolio_manager_phase3l_pm_aware_order_2023-01_to_2026-05.md`
- `reports/ml/portfolio_manager_phase4b_high_pm_min_hold_audit_2023-01_to_2026-05.md`

## Phase 25: 5-Year Walk-Forward

Purpose:

- verify whether the AI-only ranking edge survives a longer period
- avoid future leakage through expanding-window monthly walk-forward validation

Target:

- train start: `2021-06-01`
- test period: `2023-01-01` to `2026-05-31`
- price cache range available: `2021-06-01` to `2026-05-29`
- trading calendar fallback: prices-date fallback where calendar cache was insufficient

Key baseline result from the 24-feature model:

| ranking | result |
|---|---:|
| `expected_return_10d` PF | `1.0871` |
| `expected_return_10d` DD | `-95.73%` |
| `risk_adjusted_return` total_return | `35.4846` |
| `risk_adjusted_return` win_rate | `53.34%` |
| `risk_adjusted_return` PF | `1.6319` |
| `risk_adjusted_return` DD | `-14.33%` |

The main lesson was that risk-aware ranking is much more stable than raw
`expected_return_10d`.

## Phase 26 and Enriched v2 Features

FeatureBuilder was extended beyond the original price/volume/candlestick
features.

Added feature groups:

- financial statements from `/fins/summary`
- earnings calendar timing features
- TOPIX return and relative return features

Financial features:

- `EPS`
- `BPS`
- `EqAR`
- `Sales_growth`
- `OP_growth`
- `NP_growth`
- `FEPS_growth`
- `FSales_growth`
- `FOP_growth`
- `PayoutRatioAnn`

Earnings features:

- `days_to_earnings`
- `days_after_earnings`
- `is_near_earnings`

TOPIX features:

- `topix_return_5d`
- `topix_return_10d`
- `topix_return_20d`
- `relative_return_5d`
- `relative_return_10d`
- `relative_return_20d`

Listed info handling:

- `listed_info` display metadata is allowed for reports.
- Historical training does not force current `/equities/master` snapshots into past dates.
- This avoids future-information leakage through current market/sector metadata.

Enriched v2 validation memo:

| item | result |
|---|---:|
| 5-year walk-forward enriched v2 `risk_adjusted_return` PF | `1.7225` |
| 5-year walk-forward enriched v2 `risk_adjusted_return` DD | `-13.64%` |
| realistic portfolio main condition total_return | `+20.57%` |
| realistic portfolio main condition PF | `1.4376` |
| realistic portfolio main condition DD | `-7.33%` |

## Daily AI Candidate Operation

Daily AI candidate output was added as a human-review report, not an order
system.

Current candidate assumptions:

- ranking: `risk_adjusted_return`
- score: `expected_return_10d - 0.5 * bad_entry_probability_10d`
- top_n: `10`
- liquidity filter: `turnover_value >= 50,000,000`
- assumed exit: `close_20d`
- suggested position size: `200,000` JPY
- model profile: `enriched_v2`

Outputs:

- `reports/ml/daily_candidates/YYYY-MM-DD.md`
- `reports/ml/daily_candidates/YYYY-MM-DD.csv`

See also:

- `docs/ml/daily_ai_candidate_operation.md`

## Walk-Forward Model Audit

A model audit was added to confirm that walk-forward predictions were generated
with fold-specific models, not `models/ml/current`.

Audit checks:

- fold model id
- prediction parquet creation timestamp
- train start/end
- effective train end
- test start/end
- `effective_train_end < test_start`
- prediction metadata consistency
- current model reuse suspicion

Outcome:

- no current-model reuse suspicion was found in the enriched v2 walk-forward artifacts.
- `2026-05` predictions were confirmed to come from the corresponding `2026-05` fold model.

## Phase 28-29: ML Integrated Backtest Profiles

ML predictions were moved from report-only analysis into isolated backtest
profiles.

Important profiles:

| profile | purpose |
|---|---|
| `rookie_dealer_02_v2_65` | legacy baseline |
| `rookie_dealer_02_v2_66_ml_ranked` | existing strategy candidates ranked by ML |
| `rookie_dealer_02_v2_67_ml_standalone` | ML-only candidate source |

v2_66 uses:

```text
risk_adjusted_score = expected_return_10d - 0.5 * bad_entry_probability_10d
```

v2_66 became the first strong ML-integrated candidate:

- 25 months improved vs v2_65
- 14 months worsened vs v2_65
- all years improved
- ML join success rate reached 100% in the evaluated period

See also:

- `docs/ml/v2_66_ml_ranked_adoption_notes.md`

## Exit AI Phases

Exit AI was explored separately from buy-side ranking.

Exit dataset:

- one row per open-position business day
- target profile: `rookie_dealer_02_v2_66_ml_ranked`
- period: `2023-01-01` to `2026-05-31`
- output: `data/ml/exit_datasets/exit_dataset_v2_66_2023-01_to_2026-05.parquet`

Exit model targets:

- `future_remaining_return_5d`
- `future_remaining_return_10d`
- `hold_better_5d`
- `should_exit_now_5d`
- `avoid_loss_5d`

The most useful signal was:

- `avoid_loss_5d_classification`

Post-analysis suggested threshold `0.50` improved DD and profit versus baseline
in a counterfactual analysis:

| scenario | total_profit | PF | DD |
|---|---:|---:|---:|
| baseline existing exit | `424,724` | `1.1800` | `-35.97%` |
| avoid_loss threshold `0.50` | `650,308` | `1.2959` | `-22.93%` |

Exit AI profile family:

| profile | threshold |
|---|---:|
| `rookie_dealer_02_v2_68_ml_ranked_exit_ai_050` | `0.50` |
| `rookie_dealer_02_v2_69_ml_ranked_exit_ai_055` | `0.55` |
| `rookie_dealer_02_v2_70_ml_ranked_exit_ai_060` | `0.60` |

Exit AI improved drawdown, but trigger audits showed that early exits could
also change capital redeployment and miss large winners. Therefore, Exit AI is
useful but should not be evaluated in isolation.

## Capital Allocation Phases

Capital allocation work focused on how cash, daily buy limit, and order sizing
interact with ML-ranked candidates.

Important profiles:

| profile | change |
|---|---|
| `v2_71_ml_ranked_exit_ai_050_scaled_buy` | daily buy limit scaled-buy rescue |
| `v2_72_ml_ranked_exit_ai_scaled_buy_v2` | AI-first conservative allocation |
| `v2_73_ml_ranked_exit_ai_050_scaled_buy_continue` | v2_71 strength plus candidate continuation/auditability |
| `v2_74_ml_ranked_exit_ai_affordable_fallback` | affordability-aware fallback |

Key finding:

- Rejecting an order solely because it exceeded `daily_buy_limit` was too harsh.
- Scaling the order down to the largest valid 100-share lot inside the limit recovered important trades.

v2_71 result:

| metric | value |
|---|---:|
| net_profit | `1,502,024` |
| PF | `1.7053` |
| DD | `-16.14%` |

v2_72 was too conservative:

| metric | value |
|---|---:|
| net_profit | `617,630` |
| PF | `1.3845` |
| DD | `-13.40%` |

v2_74 improved profit but weakened PF/DD:

| metric | v2_73 | v2_74 |
|---|---:|---:|
| net_profit | `1,502,024` | `1,743,281` |
| PF | `1.7053` | `1.5617` |
| DD | `-16.14%` | `-22.38%` |
| capital_utilization | `47.12%` | `66.27%` |
| average_holding_count | `1.57` | `3.10` |

Conclusion:

- v2_73 was selected as the tentative main profile because it balances profit,
  PF, DD, and auditability better than v2_72/v2_74.

See also:

- `docs/ml/v2_73_adoption_notes.md`

## Position Sizing Phases

Post-analysis sizing rules were tested on v2_73/v2_74 trades.

Main result:

- larger profit often came with worse DD
- defensive rules improved PF/DD but cut too much profit
- no post-analysis sizing rule was strong enough to replace v2_73 sizing directly

This led to a more structured Portfolio Manager AI dataset instead of manual
position-size heuristics.

## Portfolio Manager AI Phase 1-2

Purpose:

- learn how much capital to allocate to each candidate
- decide how much cash to reserve
- use only J-Quants-derived features plus existing walk-forward ML predictions

Important leakage rule:

- `purchase_audit` and `trades` are allowed only for labels, result matching, and audit columns.
- They are not allowed as training features.

Dataset:

- `data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_2023-01_to_2026-05.parquet`
- clean dataset:
  `data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet`

Feature groups:

- existing ML predictions
- price/technical features
- volume features
- candlestick features
- TOPIX relative features
- financial features
- earnings features
- candidate-relative daily features
- day-level aggregate features
- current portfolio state features that are available at signal time

Labels:

- `realized_return`
- `positive_trade`
- `high_conviction_target`
- `avoid_target`
- `ideal_weight_bucket`
- `ideal_cash_reserve_bucket`

## Portfolio Manager AI Data Lineage Audit

The clean dataset and feature columns were audited before Phase 3-C/3-D.

Audited files:

- `data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet`
- `models/ml/portfolio_manager/current_v2_73_phase3b_clean/feature_columns.json`
- `src/ml/portfolio_manager_dataset.py`
- `scripts/ml/build_portfolio_manager_dataset.py`
- `scripts/ml/train_portfolio_manager_phase3a.py`

Audit result:

```text
PASS
```

Important checks:

- `selected_count_in_day` is not used.
- actual/profit/decision/skip/exit/cash/final/backtest/audit columns are not features.
- label columns are not features.
- feature count for the Phase 3-B clean model is `68`.
- current model was not used to regenerate historical predictions.

## Portfolio Manager AI Phase 3-B Clean Model

Model directory:

```text
models/ml/portfolio_manager/current_v2_73_phase3b_clean/
```

The Phase 3-B clean model provides probabilities used by Phase 3-C and 3-D:

- `high_conviction_target` probability
- `avoid_target` probability

These are combined as:

```text
pm_score = high_conviction_proba - avoid_proba
```

## Portfolio Manager AI Phase 3-C Light Evaluation

Phase 3-C was a lightweight fixed-trade position-size multiplier simulation.
It did not run the full backtest engine.

Clean-model re-evaluation result:

| rule | net_profit | PF | DD |
|---|---:|---:|---:|
| baseline | `959,058` | `1.3588` | `-29.05%` |
| high_minus_avoid | `1,219,924` | `1.5553` | `-14.86%` |
| high_strong | `1,265,470` | `1.4958` | `-15.34%` |
| avoid_strong | `905,920` | `1.5701` | `-8.62%` |

Phase 3-D candidate:

- `high_minus_avoid`

Reason:

- good profit improvement
- PF improvement
- DD improvement
- more balanced than `high_strong`

## Portfolio Manager AI Phase 3-D Full Backtest Profile

New profile:

```text
rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing
```

Base profile:

```text
rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue
```

Only intended difference:

- Portfolio Manager AI changes the planned buy amount multiplier.

PM rule:

```text
pm_score = high_conviction_proba - avoid_proba
```

Multiplier:

| condition | multiplier |
|---|---:|
| `pm_score >= 0.40` | `1.30` |
| `pm_score >= 0.20` | `1.15` |
| `pm_score >= 0.00` | `1.00` |
| `pm_score >= -0.20` | `0.80` |
| otherwise | `0.60` |

The multiplier is applied before the existing constraints:

- cash
- `daily_buy_limit = 900000`
- round lot
- scaled buy
- max positions
- Exit AI

New implementation files:

- `src/ml/portfolio_manager_sizing.py`
- `config/profiles/rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing.yaml`
- `scripts/ml/report_portfolio_manager_phase3d.py`
- `src/ml/portfolio_manager_phase3d.py`
- `tests/test_ml_portfolio_manager_phase3d.py`

Existing engine files touched:

- `src/paper_trade.py`
- `src/main.py`
- `src/profile_loader.py`

## Phase 3-D Full Backtest Result

Period:

- `2023-01-01` to `2026-05-31`

Comparison baseline:

- v2_73 was re-run in the same working tree and period.
- Older handoff numbers were not used as the comparison baseline.

Result:

| profile | final_assets | net_profit | PF | DD | win_rate | total_trades | monthly_win_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v2_73` | `2,449,386` | `1,169,366` | `1.5683` | `-19.13%` | `43.01%` | `481` | `51.22%` |
| `v2_75` | `4,344,801` | `2,679,727` | `2.2054` | `-9.17%` | `48.53%` | `546` | `70.73%` |

Delta:

| metric | delta |
|---|---:|
| net_profit | `+1,510,361` |
| PF | `+0.6371` |
| DD | `+9.96pt` |
| win_rate | `+5.52pt` |
| total_trades | `+65` |
| monthly_win_rate | `+19.51pt` |

67400 contribution:

| profile | contribution |
|---|---:|
| `v2_73` | `527,274` |
| `v2_75` | `539,696` |

The improvement was not just a simple 67400 concentration increase.

## PM Multiplier Analysis

From the full v2_75 backtest:

| multiplier | trade_count | net_profit | win_rate | PF | average_profit |
|---|---:|---:|---:|---:|---:|
| `0.60` | `110` | `-136,272` | `28.18%` | `0.7795` | `-1,239` |
| `0.80` | `152` | `244,615` | `46.05%` | `1.3123` | `1,609` |
| `1.00` | `145` | `512,770` | `44.83%` | `1.5592` | `3,536` |
| `1.15` | `40` | `363,848` | `65.00%` | `3.6735` | `9,096` |
| `1.30` | `99` | `1,128,016` | `72.73%` | `4.3629` | `11,394` |

This is a strong sanity check that the clean Portfolio Manager AI probabilities
are directionally useful inside the actual backtest engine.

## Phase 3-D Caveats

The first Phase 3-D run exposed a logging gap:

- `purchase_audit.csv` contained PM inference fields.
- closed rows in `trades.csv` did not initially carry the entry-side PM fields.

This was fixed by carrying PM fields from the buy decision into the pending
position and then into the closed trade row.

Fields now carried into trade logs:

- `pm_high_conviction_proba`
- `pm_avoid_proba`
- `pm_score`
- `pm_multiplier`
- `pm_model_version`
- `pm_feature_count`
- `pm_status`
- `pm_missing_reason`

Post-fix PM log audit:

| check | result |
|---|---:|
| `all_trades` BUY rows with PM values | `549 / 549` |
| `trades.csv` SELL rows with PM values | `546 / 546` |
| `purchase_audit.csv` PM match rate | `100%` |
| `pm_status` | `ok` for all closed trades |

Remaining caveat:

- The requested end date is `2026-05-31`, but price cache ends at
  `2026-05-29`, so the last candidate date cannot enter the next business day.
  The backtest emits a coverage warning for this period-end condition.

## Phase 3-D Detail Audit

A dedicated detail audit was added after the PM log carry-forward fix.

Implemented files:

- `src/ml/portfolio_manager_phase3d_detail_audit.py`
- `scripts/ml/audit_portfolio_manager_phase3d_detail.py`
- `tests/test_ml_portfolio_manager_phase3d_detail_audit.py`

Outputs:

- `reports/ml/portfolio_manager_phase3d_detail_audit_2023-01_to_2026-05.md`
- `reports/ml/portfolio_manager_phase3d_detail_audit_2023-01_to_2026-05.json`

Audit scope:

- v2_73 vs v2_75 monthly comparison
- code concentration
- 67400 dependency
- PM multiplier performance
- PM score band performance
- skip reason comparison
- capital utilization comparison
- promotion judgement

Key audit result:

| item | result |
|---|---:|
| promotion checks passed | `7 / 8` |
| v2_75 net_profit | `2,679,727` |
| v2_75 PF | `2.2054` |
| v2_75 DD | `-9.17%` |
| v2_75 monthly win rate | `70.73%` |
| v2_75 trades | `546` |

The single failed promotion check was strict monotonicity of absolute
`pm_score` band profit. This is mostly a trade-count issue: the higher score
bands clearly improved PF, win rate, and return on buy amount, but absolute
profit for the `0.20 to 0.40` band was lower than the `0 to 0.20` band because
it had fewer trades.

Monthly concentration check:

- v2_75 was not only a `2026-03` result.
- `2026-03` improved versus v2_73, but `2026-05` and `2025-12` also contributed
  strongly.
- Weak months remained, especially `2025-03` and `2024-04`.

Code concentration:

| metric | result |
|---|---:|
| top1 contribution rate | `25.54%` |
| top3 contribution rate | `36.91%` |
| top5 contribution rate | `45.86%` |
| largest contributor | `67400` |
| 67400 profit | `539,696` |
| 67400 contribution to v2_75 net_profit | `20.14%` |
| v2_75 excluding 67400 profit | `1,573,282` |
| v2_75 excluding 67400 PF | `1.5670` |
| v2_75 excluding 67400 DD | `-10.64%` |

Conclusion:

- 67400 is important but does not fully explain the improvement.
- v2_75 still beats v2_73 net profit after excluding 67400.

PM multiplier result:

| multiplier | trade_count | net_profit | PF | win_rate |
|---:|---:|---:|---:|---:|
| `0.60` | `110` | `-136,272` | `0.7795` | `28.18%` |
| `0.80` | `152` | `244,615` | `1.3123` | `46.05%` |
| `1.00` | `145` | `512,770` | `1.5592` | `44.83%` |
| `1.15` | `40` | `363,848` | `3.6735` | `65.00%` |
| `1.30` | `99` | `1,128,016` | `4.3629` | `72.73%` |

Interpretation:

- `1.30` and `1.15` are strongly positive.
- `0.60` is clearly weak and may deserve a future "skip instead of reduced buy"
  experiment.
- PM sizing did not simply increase risk; v2_75 improved PF and DD at the same
  time.

Skip reason comparison:

| skip_reason | v2_73 | v2_75 | note |
|---|---:|---:|---|
| `insufficient_available_cash` | `330` | `231` | improved |
| `selected_but_not_affordable` | `550` | `517` | improved |
| `daily_buy_limit_scaled_below_round_lot` | `3` | `74` | increased due PM downsizing |
| `max_positions_limit` | `1` | `0` | improved |

Capital utilization audit:

- v2_75 used less average capital than v2_73 in the approximate reconstruction.
- Average holding count increased.
- Drawdown improved despite the larger profit.

The capital utilization reconstruction is approximate because it is based on
closed-trade entry notional and the asset curve, not an exact daily mark-to-
market exposure ledger.

## Current Tentative Profile Ranking

| priority | profile | status |
|---:|---|---|
| 1 | `rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing` | strongest result so far; detail audit passed 7/8 promotion checks |
| 2 | `rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue` | prior tentative main baseline and fallback reference |
| 3 | `rookie_dealer_02_v2_66_ml_ranked` | simpler ML-ranked baseline |
| deferred | `v2_74` | higher profit than v2_73 but worse PF/DD |
| deferred | `v2_72` | too conservative |

## Recommended Next Steps

v2_75 can be treated as the current strongest candidate, with the following
follow-up checks before any production-like use:

1. Keep v2_73 as a fallback baseline until v2_75 survives more robustness tests.
2. Compare v2_75 against v2_73 on adverse periods only.
3. Run a robustness check with:
   - multiplier caps
   - no `1.30` boost
   - skip instead of `0.60` reduced buy
   - only defensive reduction
   - `high_strong` as a second full backtest if needed
4. Track whether `daily_buy_limit_scaled_below_round_lot` keeps increasing in
   live-like daily runs.
5. Continue checking top1/top3/top5 contribution so that the profile does not
   become silently dominated by one ticker.

## Test Commands

Portfolio Manager data lineage and Phase 3-C/3-D tests:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 -m pytest -q \
  tests/test_ml_portfolio_manager_dataset.py \
  tests/test_ml_portfolio_manager_trainer.py \
  tests/test_ml_portfolio_manager_data_lineage.py \
  tests/test_ml_portfolio_manager_phase3c.py \
  tests/test_ml_portfolio_manager_phase3d.py \
  tests/test_ml_portfolio_manager_phase3d_detail_audit.py
```

Latest result at the time of this document:

```text
20 passed, 1 warning
```

## Portfolio Manager AI Phase 3-E to 3-I Update

After the v2_75 detail audit, several robustness and capital-allocation checks
were added. The goal was to determine whether Portfolio Manager AI sizing could
be promoted safely, or whether the improved profit depended on hidden
concentration / drawdown risk.

### Phase 3-E: Low PM Score Skip

Profile:

```text
rookie_dealer_02_v2_76_pm_ai_low_score_skip
```

Change:

- Derived from v2_75.
- Trades with very low Portfolio Manager score were skipped instead of being
  bought at the lowest multiplier.

Result:

| profile | net_profit | PF | DD | win_rate | trades |
|---|---:|---:|---:|---:|---:|
| v2_75 | `2,679,727` | `2.2054` | `-9.17%` | `48.53%` | `546` |
| v2_76 | `3,812,364` | `2.5720` | `-19.38%` | `55.05%` | `507` |

Interpretation:

- v2_76 improved profit, PF, and win rate.
- It also removed the weakest low-score trades.
- However, DD worsened materially and required a root-cause audit before any
  promotion.

### Phase 3-F: v2_76 Drawdown Root Cause Audit

Report:

```text
reports/ml/portfolio_manager_phase3f_drawdown_audit_2023-01_to_2026-05.md
```

v2_76 maximum DD window:

| metric | value |
|---|---:|
| DD start | `2025-09-26` |
| DD trough | `2025-09-29` |
| recovery | `2025-10-24` |
| max DD | `-19.38%` |
| drawdown amount | `-743,700` |
| average capital utilization | `72.70%` |
| max capital utilization | `81.71%` |
| average holding count | `3.50` |

Root-cause flags:

| flag | result | note |
|---|---|---|
| specific code concentration | `True` | one code explained the DD-period loss |
| high multiplier concentration | `False` | not caused by `1.30` multiplier concentration |
| capital utilization spike | `False` | utilization was high but not an abnormal spike |
| holding count spike | `False` | holdings were not unusually high |
| market-regime-like drop | `False` | losses were not broad-based |
| exit delay suspected | `False` | loss holding days were not unusually long |

Conclusion:

- v2_76's DD problem was not caused by the PM multiplier itself.
- It was mainly a per-code exposure / concentration problem.
- The recommended next guard was a per-code exposure cap.

### Phase 3-G: Per-Code Exposure Cap

Profiles tested:

| profile | cap |
|---|---:|
| `rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_015` | `15%` |
| `rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap` | `20%` |
| `rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_025` | `25%` |
| `rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030` | `30%` |

Result:

| profile | net_profit | PF | DD | win_rate | trades | avg capital utilization |
|---|---:|---:|---:|---:|---:|---:|
| v2_75 | `2,679,727` | `2.2054` | `-9.17%` | `48.53%` | `546` | n/a |
| v2_76 | `3,812,364` | `2.5720` | `-19.38%` | `55.05%` | `507` | n/a |
| v2_77 cap 0.15 | `505,230` | `2.1495` | `-3.93%` | `50.81%` | `310` | `16.92%` |
| v2_77 cap 0.20 | `1,343,154` | `2.2436` | `-7.58%` | `52.95%` | `477` | `35.02%` |
| v2_77 cap 0.25 | `1,505,822` | `1.9677` | `-6.58%` | `50.70%` | `502` | `44.49%` |
| v2_77 cap 0.30 | `2,914,686` | `2.5430` | `-7.54%` | `53.15%` | `511` | `51.00%` |

Interpretation:

- Cap `0.15` and `0.20` were too conservative.
- Cap `0.25` improved DD but weakened PF.
- Cap `0.30` gave the best balance so far:
  - profit above v2_75
  - PF close to v2_76
  - DD better than both v2_75 and v2_76
  - 67400 / top-code concentration reduced relative to earlier profiles

Current best research candidate after Phase 3-G:

```text
rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030
```

### Phase 3-H: v2_77 Capital Utilization Audit

Report:

```text
reports/ml/portfolio_manager_phase3h_capital_utilization_2023-01_to_2026-05.md
```

v2_77 cap 0.30 utilization:

| metric | value |
|---|---:|
| average capital utilization | `50.995%` |
| median capital utilization | `53.121%` |
| days below 30% | `219` |
| days below 50% | `355` |
| days below 70% | `604` |
| days above 80% | `112` |
| cash idle days | `63` |
| average holding count | about `2.44` |

Low-utilization dominant reasons:

| reason | days |
|---|---:|
| `no_candidates` | `114` |
| `unknown` | `72` |
| `exit_only_day` | `67` |
| `selected_but_not_affordable` | `40` |
| `candidates_all_low_pm_skipped` | `35` |
| `below_round_lot_after_scaling` | `27` |

Skip counts:

| skip reason | count |
|---|---:|
| `selected_but_not_affordable` | `369` |
| `pm_low_score_skip` | `234` |
| `insufficient_available_cash` | `134` |
| `daily_buy_limit_scaled_below_round_lot` | `73` |
| `per_code_exposure_cap_scaled_below_round_lot` | `50` |
| `duplicate_holding` | `2` |

Next candidates proposed by the audit:

1. Keep per-code cap `0.30` and tune low-score skip / low multiplier behavior.
2. Test candidate pool expansion.
3. Test total-assets-linked `daily_buy_limit`.
4. Test replacement candidate filling after per-code cap blocks an order.

### Phase 3-I: Candidate Pool Expansion Audit

Report:

```text
reports/ml/portfolio_manager_phase3i_candidate_pool_expansion_2023-01_to_2026-05.md
```

Candidate pool settings:

| variant | max_selected |
|---|---:|
| current | `10` |
| candidate_pool_x2 | `20` |
| candidate_pool_x3 | `30` |

Result:

| variant | net_profit | PF | DD | win_rate | trades | avg utilization | no_candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | `2,914,686` | `2.5430` | `-7.54%` | `53.15%` | `511` | `50.995%` | `114` |
| x2 | `2,520,271` | `2.3186` | `-8.04%` | `51.59%` | `507` | `51.010%` | `114` |
| x3 | `2,520,271` | `2.3186` | `-8.04%` | `51.59%` | `507` | `51.010%` | `114` |

Interpretation:

- Candidate pool expansion did not reduce `no_candidates`.
- Capital utilization improved only by about `0.015pt`.
- Profit, PF, DD, win rate, and trade count all worsened versus current.
- x2 and x3 produced identical results, implying the bottleneck is not the raw
  candidate pool size after `max_selected=20`.

Decision:

- Do not adopt candidate pool expansion for now.
- Move to a different utilization-improvement path:
  - low-score skip threshold tuning
  - candidate replacement after cap/affordability blocks
  - cash / daily limit policy
  - fallback quality improvement

### Phase 3-J: Affordability Audit

Report:

```text
reports/ml/portfolio_manager_phase3j_affordability_audit_2023-01_to_2026-05.md
```

Target:

```text
rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030
```

Key finding:

- `selected_but_not_affordable` occurred `369` times.
- This was a major remaining reason for low utilization after v2_77.
- The audit attached hypothetical 3d/5d/10d returns to skipped candidates.
- Some skipped candidates had positive hypothetical returns, so a fallback path
  was worth auditing.

Interpretation:

- The issue was not simply lack of cash.
- The candidate loop could encounter an unaffordable selected candidate, while
  the surrounding candidate set still contained possible alternatives.
- Before changing logic, Phase 3-K checked whether existing candidate ordering
  and fallback behavior already handled this.

### Phase 3-K: Candidate Ranking / Fallback Path Audit

Report:

```text
reports/ml/portfolio_manager_phase3k_candidate_ranking_audit_2023-01_to_2026-05.md
```

Current ranking path before Phase 3-L:

| item | behavior |
|---|---|
| selected sorting | `daily_score_rank ASC`, then `risk_adjusted_score DESC`, then `code ASC` |
| PM AI timing | PM sizing was applied after selected ordering |
| affordability timing | cash / daily limit / round lot / PM sizing / per-code cap after ordering |
| `selected_count_in_day` | not used |

Path classification:

| classification | count |
|---|---:|
| `candidate_log_insufficient` | `79` |
| `top_candidate_unaffordable_and_no_buy` | `73` |
| `top_candidate_unaffordable_but_affordable_candidate_exists_not_bought` | `12` |
| `top_candidate_unaffordable_but_next_candidate_bought` | `40` |

Decision:

- A fallback path was still useful, but should stay inside audited selected
  candidates first.
- PM score should influence buy order because the original order was still
  mainly rule-score / risk-adjusted-score driven.

### Phase 3-L: PM-Aware Ordering and Selected Fallback

New profiles:

| profile | setting |
|---|---|
| `rookie_dealer_02_v2_78_pm_aware_order_fallback_w025` | PM-aware order weight `0.25` |
| `rookie_dealer_02_v2_78_pm_aware_order_fallback_w050` | PM-aware order weight `0.50` |
| `rookie_dealer_02_v2_78_pm_aware_order_fallback_w100` | PM-aware order weight `1.00` |

Report:

```text
reports/ml/portfolio_manager_phase3l_pm_aware_order_2023-01_to_2026-05.md
```

Result:

| variant | net_profit | PF | DD | win_rate | trades | selected_but_not_affordable |
|---|---:|---:|---:|---:|---:|---:|
| v2_77 current | `2,914,686` | `2.5430` | `-7.54%` | `53.15%` | `511` | `369` |
| v2_78 w0.25 | `3,054,794` | `2.6194` | `-7.47%` | `53.78%` | `505` | `253` |
| v2_78 w0.50 | `3,054,794` | `2.6194` | `-7.47%` | `53.78%` | `505` | `253` |
| v2_78 w1.00 | `3,054,794` | `2.6194` | `-7.47%` | `53.78%` | `505` | `253` |

Interpretation:

- PM-aware ordering plus selected fallback improved net profit, PF, DD, win
  rate, and affordability skips versus v2_77 cap0.30.
- w0.25 / w0.50 / w1.00 produced identical aggregate results in this run.
- v2_78 w0.25 became the next base for Exit quality audits because it was the
  least aggressive ordering weight with the same observed outcome.

### Phase 4-B: High PM Minimum Hold Audit

Report:

```text
reports/ml/portfolio_manager_phase4b_high_pm_min_hold_audit_2023-01_to_2026-05.md
```

Target:

```text
rookie_dealer_02_v2_78_pm_aware_order_fallback_w025
```

The audit found that high-PM trades, defined as `pm_multiplier >= 1.15`, were
already strong but often exited while post-exit returns remained positive.

Actual high-PM summary:

| metric | value |
|---|---:|
| trade_count | `168` |
| net_profit | `1,314,581` |
| PF | `3.2283` |
| win_rate | `65.48%` |
| average_holding_days | `4.17` |
| early_exit_rate | `60.71%` |

Minimum-hold counterfactual:

| rule | changed trades | profit_delta | virtual PF | virtual win_rate |
|---|---:|---:|---:|---:|
| min hold 3d | `27` | `+24,095` | `3.1407` | `68.45%` |
| min hold 5d | `64` | `+344,508` | `3.7796` | `71.43%` |
| min hold 7d | `168` | `+1,508,389` | `9.3080` | `77.38%` |

Important caveat:

- This was a lightweight audit and did not replay portfolio equity, so DD and
  capital lock-up impact still require a full backtest.

### Phase 4-C: High PM Minimum Hold Profiles

New profiles:

| profile | behavior |
|---|---|
| `rookie_dealer_02_v2_79_high_pm_min_hold_5d` | suppress Exit AI exits for high-PM positions before 5 business holding days |
| `rookie_dealer_02_v2_79_high_pm_min_hold_7d` | suppress Exit AI exits for high-PM positions before 7 business holding days |

Base profile:

```text
rookie_dealer_02_v2_78_pm_aware_order_fallback_w025
```

Implementation:

- Applies only when `pm_multiplier >= 1.15`.
- Suppresses only Exit AI generated exits while `holding_days < high_pm_min_hold_days`.
- Does not suppress existing stop loss, take profit, max holding, or forced
  exits because the guard only blocks plans with `exit_ai_triggered=True`.
- Adds audit fields such as:
  - `high_pm_min_hold_enabled`
  - `high_pm_min_hold_days`
  - `high_pm_min_hold_applied`
  - `high_pm_min_hold_blocked_exit`
  - `high_pm_min_hold_blocked_exit_count`
  - `high_pm_min_hold_exit_reason_original`
  - `high_pm_min_hold_release_date`
  - `holding_days_at_exit_signal`

Report script:

```text
scripts/ml/report_portfolio_manager_phase4c_high_pm_min_hold.py
```

Planned report output after full backtests:

```text
reports/ml/portfolio_manager_phase4c_high_pm_min_hold_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase4c_high_pm_min_hold_2023-01_to_2026-05.json
```

Quick Check result:

```text
tests/test_ml_portfolio_manager_phase4c_high_pm_min_hold.py
6 passed
```

Full backtests were intentionally not run during implementation.

### Current Profile Ranking After Phase 4-C

| priority | profile | status |
|---:|---|---|
| 1 | `rookie_dealer_02_v2_78_pm_aware_order_fallback_w025` | latest full-backtested best balance after PM-aware ordering |
| 2 | `rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030` | previous best balance: strong profit/PF with DD under 8% |
| experimental | `rookie_dealer_02_v2_79_high_pm_min_hold_5d` | implemented; needs full backtest |
| experimental | `rookie_dealer_02_v2_79_high_pm_min_hold_7d` | implemented; needs full backtest |
| reference | `rookie_dealer_02_v2_76_pm_ai_low_score_skip` | highest profit/PF/win rate before caps, but DD too large without exposure guard |
| reference | `rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing` | strong prior profile and simpler PM sizing reference |
| reference | `rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue` | prior baseline / fallback reference |
| deferred | candidate_pool_x2/x3 | no utilization benefit and worse PF/profit |

### Latest Test Command

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
  tests/test_ml_portfolio_manager_phase3i_candidate_pool.py \
  tests/test_ml_portfolio_manager_phase3j_affordability.py \
  tests/test_ml_portfolio_manager_phase3k_candidate_ranking.py \
  tests/test_ml_portfolio_manager_phase3l_pm_aware_order.py \
  tests/test_ml_portfolio_manager_phase4b_high_pm_min_hold.py \
  tests/test_ml_portfolio_manager_phase4c_high_pm_min_hold.py
```

Latest Phase 4-C quick-check result:

```text
tests/test_ml_portfolio_manager_phase4c_high_pm_min_hold.py
6 passed
```
