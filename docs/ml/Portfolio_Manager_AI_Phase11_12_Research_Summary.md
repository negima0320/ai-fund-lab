# Portfolio Manager AI Phase 11-12 Research Summary

Last updated: `2026-06-11`

This document summarizes the Phase 11 and Phase 12 research track. The central
constraint across both phases is unchanged:

```text
AI training and allocation decisions must not use backtest results, trade
outcomes, cash, portfolio state, selected/bought/affordable flags, or current
PM multiplier data as features.
```

Future columns are allowed only as labels, evaluation metrics, or audit-only
diagnostics. Existing production/reference profiles were not changed.

## Phase 11 Purpose

Phase 11 stopped trying to evolve the previous PM AI multiplier directly and
split the problem into two layers:

```text
Stock Selection AI
↓
Valuation Engine
↓
Capital Allocation Engine
```

The key question changed from:

```text
How much should PM AI multiply this trade?
```

to:

```text
Is this stock currently an attractive opportunity, and how much capital should
be allocated to it?
```

## Phase 11 Timeline

| phase | scope | result |
| --- | --- | --- |
| Phase 11-A | Valuation dataset audit | Built a leakage-safe dataset: `930,243` rows, `4,234` codes, `55` features, `5` labels, `leakage_risk=low`, `ready_for_phase11b=true`. |
| Phase 11-B | Valuation prototype | Regression was weak, but classification had useful lift: AUC `0.6478`, PR-AUC `0.1600`, precision@top10% `0.1998` vs base `0.0997`. |
| Phase 11-C | Allocation prototype | 2025-only allocation-quality simulation found `equal_weight_top5` best, top-decile rate `0.2403`, but budget usage only `20.7%`. |
| Phase 11-C2 | Budget usage constraint audit | Main bottleneck was round-lot and top-candidate affordability; recommended daily budget `900,000`, max positions `5`, threshold `top5`. |
| Phase 11-D | 2025 limited combined backtest | Valuation improved BUY quality and profit: Candidate net profit `187,018` vs baseline `88,578`, but DD worsened to `-16.83%`. |
| Phase 11-E | Exit / DD guard | Stop + Opportunity Exit E4 produced net profit `615,110`, PF `2.6219`, DD `-6.02%` on 2025, but turnover risk appeared. |
| Phase 11-F | Robustness check | E4 stayed strong under `0.2%` one-way cost: net profit `473,578`, PF `2.0551`, DD `-6.36%`; overtrading remained visible. |
| Phase 11-G | 2024 OOS year/path check | 2024 E4 was strong, but not strict model OOS because Phase 11-B training included 2024. |
| Phase 11-H | Cooldown/min-hold guards | `H2_cooldown_10d` and `H3_min_hold_3d` passed 2024/2025 checks; strict walk-forward OOS design was defined. |
| Phase 11-I | Strict walk-forward OOS prototype | Research-only strict split trained on 2023, validated on 2024, tested on 2025. Rank lift survived, but strategy checks failed adoption thresholds. |
| Phase 11-B2 | Strict OOS failure diagnosis | Strict OOS valuation top5 raised top-decile rate `8.85% -> 24.00%`, but downside_bad_rate worsened `13.58% -> 37.94%`. |
| Phase 11-B3 | Expected Downside model | Downside model detected risk: AUC `0.6180`, PR-AUC `0.2323`, precision@top10% `0.2992`; combined ranking could reduce downside but often sacrificed opportunity. |

## Phase 11 Conclusion

Phase 11 proved three things:

- Valuation Engine can find higher-opportunity candidates.
- Opportunity-only selection also captures high-downside candidates.
- Downside needs to be separated from Opportunity rather than hidden inside a
  single score.

Phase 11 did not produce an adoption-ready strategy. It produced the research
inputs for Phase 12:

```text
opportunity_proba
downside_bad_proba
opportunity_rank_percentile
downside_rank_percentile
```

## Phase 12 Purpose

Phase 12 researched Dynamic Capital Allocation using:

```text
Opportunity
+
Downside
+
Confidence / relative rank
```

The goal was not just to pick better candidates, but to size them:

```text
Opportunity high + Downside low  -> buy larger
Opportunity high + Downside high -> buy smaller
Opportunity low                  -> do not buy
```

All Phase 12 strategy checks remained 2025-only lightweight research unless
explicitly stated. No production profile was changed.

## Phase 12 Timeline

| phase | scope | result |
| --- | --- | --- |
| Phase 12-A | Dynamic allocation quality audit | Downside could be reduced, but Opportunity fell too much; no rule met the minimum line. |
| Phase 12-A2 | Allocation score refinement | Opportunity top5 with downside penalty nearly passed: top-decile `0.2454`, downside `0.2664`. |
| Phase 12-A3 | Top5 penalty refinement | `A3_3_rank_medium_floor_zero` passed: weighted top-decile `0.2614`, weighted downside `0.1432`, opportunity value `0.0683`. |
| Phase 12-B | Limited allocation strategy check | Raw dynamic weight was safe but underused capital; normalized weight used capital but DD worsened. |
| Phase 12-B2 | Execution adjustment | Partial normalization/min-usage/caps did not solve the tradeoff. |
| Phase 12-B3 | Exit / hold audit | Both early and late exits existed; Opportunity Exit often cut trades too early, and stop-loss trades showed profit decay. |
| Phase 12-B4 | Trailing exit prototype | Trailing alone did not solve utilization/PF; Opportunity + trailing 8% only modestly improved results. |
| Phase 12-B5 | Exit threshold recalibration | Relaxing proba-drop exit (`B5_2`) improved net profit `71,922`, PF `2.1827`, DD `-3.24%`, holding days `13.34`, utilization `16.13%`. |
| Phase 12-C | Dynamic allocation + improved exit | Raw allocation stayed safe but underused capital; normalized allocation had profit `306,382`, PF `2.0680`, but DD `-18.88%`. |
| Phase 12-C2 | DD attribution | Main DD cause was single-name concentration, not high downside proba exposure. |
| Phase 12-C3 | Position concentration guard | Simple per-name caps reduced concentration but broke PF/DD. |
| Phase 12-C4 | Concentration guard refinement | Redistributed caps, dynamic caps, and staged-buy proxies still failed minimum targets. |
| Phase 12-D1 | Winning trades turned losers audit | Found 7 trades with peak `>= +5%` that ended as losses; estimated recoverable peak-to-final leakage `328,000`. |
| Phase 12-D2 | Buy quality reality audit | Valuation and Downside add BUY-quality value. Stock Selection top5 alone did not beat the universe. Main bottleneck moved to Exit/risk control. |
| Phase 12-D3 | Prediction lineage / strict OOS integrity audit | Confirmed 2025 Stock Selection, Valuation, and Downside inputs are strict OOS by existing artifact evidence; `phase12_results_trustworthy=true`. |
| Phase 12-E1 | Stock Selection reality audit | Stock Selection adds value `false`, top5 valid `false`, and prefilter hurts Valuation `true`. |
| Phase 12-E2 | Stock Selection architecture audit | Stock Selection is a short-horizon LightGBM walk-forward composite. Its objective is misaligned with Phase 12's 20d opportunity/downside target. |

## Phase 12 Key Findings

### Buy Quality

Phase 12-D2 showed the buy-side research is real:

| layer | top-decile rate | downside bad rate |
| --- | ---: | ---: |
| Candidate universe | `0.1053` | `0.1650` |
| Stock Selection top5 | `0.0885` | `0.1358` |
| Opportunity top5 | `0.2400` | `0.3794` |
| A3_3 Opportunity + Downside weighted | `0.2614` | `0.1432` |

Interpretation:

- Opportunity finds winners.
- Opportunity alone also finds dangerous candidates.
- Downside penalty can preserve opportunity while reducing downside.
- Stock Selection top5 is not currently a good prefilter for the Phase 12
  objective.

### Strategy Bottleneck

Phase 12-C/C2/C3/C4 showed:

- Raw dynamic allocation is safe but capital-light.
- Normalized allocation raises profit and utilization but creates severe DD.
- The DD source is mostly single-name concentration.
- Blunt concentration caps remove good exposure too, causing PF/DD deterioration.

### Exit Bottleneck

Phase 12-B3/D1 showed:

- Opportunity Exit was often too early.
- Some stop-loss trades had meaningful peak profit before decaying into losses.
- Profit protection / break-even / trailing lock design is likely more useful
  than more allocation tweaks.

### Stock Selection Bottleneck

Phase 12-E1/E2 showed:

- `stock_selection_rank_score` is not a direct 20d Opportunity model.
- It is derived from `ml_score`, which combines 10d return, 10d upside, and 10d
  bad-entry probability.
- `candidate_strength` was stronger because it includes 20d max-return and
  swing-success signals:

```text
candidate_strength =
expected_max_return_20d
+
swing_success_probability_20d
-
bad_entry_probability_10d
```

The suspected failure reason is objective mismatch:

```text
Stock Selection AI = short-horizon composite selector
Phase 12 target    = 20d opportunity + downside-aware allocation
```

## Current Decision

Phase 11/12 produced useful research components but no adoption-ready live
strategy.

Current strongest full-backtested reference remains:

```text
rookie_dealer_02_v2_82_cap38
```

Current Phase 12 research decision:

```text
phase12_results_trustworthy = true
ready_for_phase13 = false
stock_selection_prefilter_hurts_valuation = true
recommended_next_phase = Phase12-E3 Remove Stock Selection Prefilter Test
```

Do not run broad/full backtests yet. The next clean step is to test whether
removing Stock Selection prefilter, or replacing it with Candidate Strength /
Valuation + Downside directly, improves 2025-limited strategy behavior without
reintroducing leakage.

## Important Generated Reports

```text
reports/ml/phase11a_valuation_dataset_audit_2023-01_to_2026-05.json
reports/ml/phase11b_valuation_engine_prototype_2025_holdout.json
reports/ml/phase11b3_expected_downside_model_2025.json
reports/ml/phase11i_strict_walk_forward_oos_2025.json
reports/ml/phase12a3_top5_penalty_refinement_2025.json
reports/ml/phase12b5_exit_threshold_recalibration_2025.json
reports/ml/phase12c4_concentration_guard_refinement_2025.json
reports/ml/phase12d1_winning_to_losing_audit_2025.json
reports/ml/phase12d2_buy_quality_reality_audit_2025.json
reports/ml/phase12d3_prediction_lineage_oos_audit.json
reports/ml/phase12e1_stock_selection_reality_audit_2025.json
reports/ml/phase12e2_stock_selection_architecture_audit.json
```

