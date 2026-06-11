# ML Documentation Index

This directory contains implementation notes, validation summaries, and
adoption notes for the AI / ML stack.

## Implementation Summaries

- `ML_Phase_1_to_9_Implementation_Summary.md`
  - J-Quants cache DataLoader
  - FeatureBuilder
  - LabelGenerator
  - DatasetBuilder
  - ModelTrainer
  - Predictor
  - daily pipeline
  - smoke tools

- `ML_Phase_10_to_24_Implementation_Summary.md`
  - prediction evaluation
  - range smoke
  - backtest join analysis
  - swing labels and 7-model support
  - paper portfolio and realistic portfolio
  - walk-forward ranking comparison
  - daily AI candidates

- `ML_Phase_25_to_Portfolio_Manager_AI_Summary.md`
  - 5-year walk-forward
  - enriched v2 financial / TOPIX features
  - ML-integrated backtest profiles
  - Exit AI
  - capital allocation profiles
  - v2_73 adoption
  - Portfolio Manager AI data lineage, training, Phase 3-C, Phase 3-D full backtest, v2_75 detail audit, v2_76 drawdown audit, v2_77 per-code cap, capital utilization, candidate-pool expansion, PM-aware ordering, affordability/ranking audits, and high-PM minimum-hold profiles

- `Portfolio_Manager_AI_Phase4C_to_4G_Audit_Summary.md`
  - v2_78 vs v2_79 high-PM minimum-hold verification
  - Phase 4-D difference audit
  - Phase 4-F side-effect root-cause audit
  - Phase 4-G clean Exit AI delay / candidate-presence hold audit
  - current decision: keep v2_78 as main candidate and keep v2_79 on hold

- `Exit_AI_v2_Phase5A_to_5F_Retraining_Summary.md`
  - Exit AI v2 API-only retraining readiness
  - API-only dataset design and builder
  - training design and leakage-safe top-decile trainer prototype
  - Phase 5-F candidate full-train results
  - current decision: keep current Exit AI unchanged and move next to Prediction / Integration Audit

- `Portfolio_Manager_AI_Phase5G_to_6G_Audit_Summary.md`
  - Exit AI v2 Prediction / Integration Audit and v2_80 integration rejection
  - Market Regime / Bear alpha audits
  - Bear Booster implementation and non-adoption
  - per-code cap bottleneck audit
  - v2_82 cap38 full backtest result and current main-candidate decision

- `Portfolio_Manager_AI_Phase7A_to_7G_Final_Summary.md`
  - full AI state and retraining-readiness audit
  - PM AI leakage forensics and API-only dataset rebuild/trainer
  - v2_82 Final Championship Audit
  - final pytest failure triage
  - current decision: `v2_82_cap38` is the Version 1.0 Candidate; PM AI API-only candidate is trained but not integrated

- `Portfolio_Manager_AI_Phase8A_to_8H_PM_AI_Redesign_Summary.md`
  - system understanding and PM AI contribution audit
  - PM AI API-only candidate integration, calibration, and label redesign audits
  - v2_90 / v2_91 PM AI v2 backtest rejection
  - Phase 8-G ranking / relative allocation audit
  - v2_92 rule-based relative allocator backtest rejection
  - current decision: keep `v2_82_cap38`; do not promote PM AI v2 or the relative allocator

- `Portfolio_Manager_AI_Phase9A_to_9G_PM_AI_v3_Research_Summary.md`
  - PM AI v3 re-architecture as market regime + candidate ranking + position sizing
  - clean dataset, PM sizing universe dataset, trainer, mapping stability, threshold, integration, and backtest candidate audits
  - v2_93 / v2_94 PM AI v3 candidate rejection and v2_95 PM-disabled equal-weight baseline rejection
  - current decision: keep `v2_82_cap38`; PM AI v3 and PM-disabled baseline remain research-only

- `Portfolio_Manager_AI_Phase10A_Score_Based_PM_Rule_Summary.md`
  - leakage-safe rule-based PM using only Stock Selection prediction-time scores
  - v2_96 Rule A/B/C score-based PM backtests versus v2_95 PM-disabled baseline and v2_82 reference
  - Phase 9 PM AI v3 generated artifact cleanup
  - current decision: Rule C is a review candidate versus v2_95 but not a strong adoption candidate; keep `v2_82_cap38`

- `Portfolio_Manager_AI_Phase10_Stop_and_Hold_Summary.md`
  - PM AI / Phase 10 development stop-and-hold decision
  - Phase 7 through Phase 10 PM research timeline and rejection rationale
  - current decision: stop PM AI redevelopment for now; keep `v2_82_cap38`; do not promote v2_96 / v2_97

- `Portfolio_Manager_AI_Phase11_Valuation_Allocation_Plan.md`
  - Phase 11 plan to replace direct PM multiplier redevelopment with a two-layer Valuation Engine and Capital Allocation Engine
  - defines allowed inputs, forbidden leakage sources, Phase 11-A to 11-E roadmap, and success criteria
  - Phase 11-A Valuation Engine Dataset Audit implemented with low leakage risk, no blocking issues, and `ready_for_phase11b=true`
  - Phase 11-B Valuation Engine Prototype implemented with low leakage risk, no blocking issues, candidate model saved, and `ready_for_phase11c=true`
  - Phase 11-C Capital Allocation Engine Prototype implemented as a 2025-only allocation-quality simulation with low leakage risk and `ready_for_phase11d=true`
  - Phase 11-C2 Budget Usage Constraint Audit identified round-lot / top-candidate affordability as the main budget usage bottleneck
  - Phase 11-D Limited Combined Backtest connected Valuation to 2025-only buy logic and improved profit/PF/BUY quality, but worsened DD
  - Phase 11-E Limited Exit / DD Guard found Opportunity Disappeared Exit variants that reduced DD below `-10%` in 2025-only testing
  - Phase 11-F Limited Robustness Check found E4 resilient to `0.2%` one-way cost and Opportunity Exit threshold sensitivity, with overtrading risk still present
  - Phase 11-G Limited 2024 Year Check supported E4 on an additional year, but flagged that 2024 overlaps the Phase 11-B model training period and is not strict model OOS
  - Phase 11-H Cooldown / Minimum Holding Guard found `H2_cooldown_10d` and `H3_min_hold_3d` passed 2024/2025 guard checks and produced a strict walk-forward OOS design
  - Phase 11-I Strict Walk-Forward OOS Prototype trained a research-only 2023 model and found strict OOS rank lift remained, but 2025 strategy checks did not beat the equal-allocation baseline
  - Phase 11-B2 Strict OOS Failure Diagnosis found the strict OOS top5 improved top-decile rate but also concentrated high-downside candidates, supporting an expected_downside model next
  - Phase 11-B3 Expected Downside Model Prototype showed downside risk can be reduced with combined ranking, but opportunity retention needs threshold / weight tuning
  - current decision: proceed to Phase 11-B4 combined ranking threshold tuning before any strategy backtest or adoption; keep `v2_82_cap38` as reference only

- `Portfolio_Manager_AI_Phase12_Dynamic_Capital_Allocation_Summary.md`
  - Phase 12 shifts from candidate replacement to Dynamic Capital Allocation using Opportunity + Downside + Confidence
  - Phase 12-A ran a 2025-only allocation quality audit with Phase 11-B3 research models and no strategy backtest
  - Phase 12-A found no rule meeting the minimum line of weighted top-decile rate `>= 0.20` and weighted downside bad rate `<= 0.25`
  - Phase 12-A2 narrowed candidate universes to opportunity top5/top10/top20 and used downside only as a penalty
  - Phase 12-A3 fixed the universe to opportunity top5 and found `A3_3_rank_medium_floor_zero` passed the ideal line with weighted top-decile rate `0.2614` and weighted downside bad rate `0.1432`
  - Phase 12-B connected A3_3 to a 2025-only lightweight strategy check; raw dynamic weighting improved PF/DD versus S2 but did not beat the baseline, while normalized weighting raised profit but worsened DD
  - Phase 12-B2 tested partial normalization, min usage guards, and capped normalized execution; no strategy met the minimum line
  - Phase 12-B3 audited S3a/S2 exits and found both early and late exit issues, with late profit decay before stop-loss the main problem
  - Phase 12-B4 tested stop-loss-only and trailing exits; trailing alone did not beat current Opportunity Exit, while Opportunity + trailing 8% gave only a small improvement without utilization gain
  - Phase 12-B5 relaxed Opportunity Exit proba drop threshold and found `B5_2_proba_drop_larger` passed minimum targets with PF `2.1827`, DD `-3.24%`, net profit `71,922`, and utilization `16.13%`
  - Phase 12-C integrated Dynamic Allocation with the recalibrated B5_2 exit; normalized allocation raised profit to `306,382` and PF `2.0680` but DD worsened to `-18.88%`, while raw allocation kept DD low but utilization stayed `16.13%`
  - current decision: do not proceed to Phase 13 yet; continue with Phase 12-C2 allocation utilization refinement, still 2025-limited before any broad/full backtests

## Adoption Notes

- `v2_66_ml_ranked_adoption_notes.md`
  - first ML-ranked existing-strategy candidate

- `v2_73_adoption_notes.md`
  - prior tentative main profile before Portfolio Manager AI Phase 3-D

Current strongest full-backtested research candidate:

- `rookie_dealer_02_v2_82_cap38`
  - v2_78 w0.25-derived profile with only `per_code_exposure_cap_rate` relaxed from `0.30` to `0.38`
  - Phase 7-F Final Championship core result: net profit `3,777,545`, PF `2.7309`, DD `-6.54%`, win rate `55.11%`, CAGR `66.74%`
  - improved profit, PF, DD, win rate, monthly win rate, capital utilization, and cap skip/reduction count versus v2_78
  - Phase 7-G full pytest after triage: `825 passed, 15 warnings`
  - latest audits are summarized in `Portfolio_Manager_AI_Phase7A_to_7G_Final_Summary.md`
  - PM AI / Phase 10 redevelopment is currently stopped and summarized in `Portfolio_Manager_AI_Phase10_Stop_and_Hold_Summary.md`
  - Phase 11 will explore Valuation Engine + Capital Allocation Engine as a new research direction, with v2_82 used as a reference record only
  - Phase 11-A confirmed a leakage-safe valuation dataset audit path and does not use backtest/trade outcomes as features
  - Phase 11-B trained a separate research-only valuation candidate model under `models/ml/valuation_engine/candidate_phase11b/`
  - Phase 11-C simulated allocation quality for 2025 only and did not run a strategy backtest
  - Phase 11-C2 audited the low `20.7%` budget usage proxy and recommended a `900,000` daily budget sensitivity path before broad combined backtesting
  - Phase 11-D ran a 2025-only limited combined backtest; Valuation top5 improved net profit from `88,578` to `187,018` and top-decile BUY rate from `7.27%` to `29.31%`, while DD worsened from `-5.33%` to `-16.83%`
  - Phase 11-E showed simple stop loss was harmful, while Opportunity Disappeared Exit improved DD to `-7.85%` and Stop + Opportunity Exit improved DD to `-6.02%`
  - Phase 11-F showed E4 remains above PF `2.0`, DD within `-10%`, and net profit above `300,000` under `0.2%` one-way cost, but has `115` same-code reentries and `88` reentries within 5 business days
  - Phase 11-G checked 2024 only: E4 net profit `699,520`, PF `2.7918`, DD `-8.25%`; E4 with `0.2%` cost net profit `574,984`, PF `2.3421`, DD `-9.11%`; strict model OOS remains false because Phase 11-B trained through 2024
  - Phase 11-H checked cooldown/min-hold guards on 2024/2025 with `0.2%` cost; `H2_cooldown_10d` and `H3_min_hold_3d` passed both years, while combined cooldown+min-hold was unstable in 2025
  - Phase 11-I strict OOS trained a separate research model on 2023 only, validated on 2024, and tested on 2025; test AUC `0.6297`, PR-AUC `0.1514`, precision@top10% `0.1837`, strict_model_oos `true`
  - Phase 11-I strategy check with `0.2%` cost did not pass strict OOS adoption criteria: baseline net profit `180,876`, PF `2.2930`, DD `-6.39%`; strict OOS E4 net profit `116,049`, PF `1.2501`, DD `-13.14%`
  - Phase 11-B2 diagnosed the failure: strict OOS valuation top5 top-decile rate `24.00%` versus baseline `8.85%`, but downside_bad_rate `37.94%` versus baseline `13.58%`; feature drift was also detected in score and market/financial features
  - Phase 11-B3 trained a strict OOS downside model; downside model test AUC `0.6180`, PR-AUC `0.2323`, precision@top10% `0.2992`; `score_v1` reduced downside_bad_rate to `19.76%` but top-decile rate fell to `15.27%`
  - Phase 12-A allocation audit found `score_a_weighted` reduced weighted downside_bad_rate to `22.60%` but weighted top-decile rate was only `15.37%`; no rule was ready for Phase 12-B
  - Phase 12-A2 found `opportunity_top5__penalty_rank_medium` kept weighted top-decile rate `24.54%` and reduced downside to `26.64%`, narrowly missing the `25%` downside target
  - Phase 12-A3 found `A3_3_rank_medium_floor_zero` improved weighted top-decile rate to `26.14%`, reduced weighted downside_bad_rate to `14.32%`, and set `ready_for_phase12b=true`
  - Phase 12-B found `S3a_dynamic_raw_weight` had net profit `39,770`, PF `1.5971`, DD `-2.66%`, and downside_bad_rate `17.86%`, but did not beat baseline net profit; `S3b_dynamic_normalized_weight` earned `135,752` but DD worsened to `-19.16%`
  - Phase 12-B2 found no execution adjustment met the minimum line; `S5_partial_normalized_30` improved utilization to `39.10%` and DD to `-11.75%` but PF stayed at `1.0094`
  - Phase 12-B3 found S3a stop-loss trades had avg profit decay before exit `17.03%`, while opportunity exits still had avg post-exit 20d return `5.77%`; recommended `Phase12-B4 trailing_exit_prototype`
  - Phase 12-B4 found `T5_opportunity_plus_trailing_8pct` improved net profit to `43,962`, PF to `1.7044`, and DD to `-2.50%`, but capital utilization stayed `9.98%`; `ready_for_phase12c=false`
  - Phase 12-B5 found `B5_2_proba_drop_larger` improved net profit to `71,922`, PF to `2.1827`, DD `-3.24%`, average holding days `13.34`, and reduced opportunity_exit_count from `47` to `19`; `ready_for_phase12c=true`
  - Phase 12-C found no integrated strategy met the minimum line of PF `>=1.8`, DD `>=-10%`, net profit positive, and utilization `>=20%`; `C2_dynamic_normalized_B5_2_exit` had high profit/PF but DD `-18.88%`, so `ready_for_phase13=false`
  - still a backtest/research profile; not connected to live order placement

Experimental next candidates:

- `rookie_dealer_02_v2_78_pm_aware_order_fallback_w025`
  - conservative fallback/reference after Phase 6-G
  - prior main candidate: net profit `3,054,794`, PF `2.6194`, DD `-7.47%`, win rate `53.78%`

- `rookie_dealer_02_v2_79_high_pm_min_hold_5d`
- `rookie_dealer_02_v2_79_high_pm_min_hold_7d`
  - Phase 4-C backtests were run and v2_79 outperformed v2_78 numerically
  - Phase 4-F confirmed `high_pm_min_hold_blocked_exit_count=0`; the intended minimum-hold guard was not directly effective
  - v2_79 remains on hold and should not be adopted as the main profile
  - see `Portfolio_Manager_AI_Phase4C_to_4G_Audit_Summary.md`

Possible clean v2_80 research direction:

- narrow high-PM Exit AI one-day delay for `pm_multiplier >= 1.15`
  - Phase 4-G found this reproduced 71570 and added `+11,200` in the virtual audit
  - blanket Exit AI 1-day delay was harmful and is not recommended

Exit AI v2 research candidate:

- `models/ml/exit_ai_v2/candidate_v2_api_only`
  - trained from API-only dataset, not from backtest outcomes
  - current Exit AI `models/ml/exit/current_v2_66` was not overwritten
  - test AUC `0.6524`, PR-AUC `0.1553`, top decile lift `2.2574`
  - Phase 5-G prediction audit completed
  - Phase 5-H v2_80 integration profiles underperformed v2_78 and were not adopted
  - keep current Exit AI `models/ml/exit/current_v2_66`

PM AI API-only research candidate:

- `models/ml/portfolio_manager/candidate_v2_api_only`
  - trained from API-only PM dataset, not from backtest outcomes
  - current PM AI `models/ml/portfolio_manager/current_v2_73_phase3b_clean` was not overwritten
  - high conviction test AUC `0.6472`; avoid target test AUC `0.6345`
  - Phase 8-B to 8-F confirmed the model is clean but not a viable replacement
    for the current PM AI in v2_82
  - v2_90 raw integration and v2_91 calibrated integration both underperformed
    v2_82 and were not adopted

Rule-based relative allocator research profile:

- `rookie_dealer_02_v2_92_relative_allocator_cap38`
  - uses same-day candidate relative rank / percentile instead of current PM AI
  - Rule C `blended_relative_score` backtest was operationally valid but
    underperformed
  - result: net profit `940,647`, PF `1.5118`, DD `-11.56%`, win rate `43.89%`
  - decision: do not adopt; keep v2_82
  - see `Portfolio_Manager_AI_Phase8A_to_8H_PM_AI_Redesign_Summary.md`

- `daily_ai_candidate_operation.md`
  - human-review daily AI candidate output
  - no order placement

## Design Notes

- `AI_Fund_Lab_AI_Design_v2_Complete.md`
  - original AI design and ML architecture

- `AI_Fund_Lab_ML_Implementation_Spec_v1.md`
  - implementation specification
