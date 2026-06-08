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
  - not integrated into v2_82; requires a separate integration audit

- `daily_ai_candidate_operation.md`
  - human-review daily AI candidate output
  - no order placement

## Design Notes

- `AI_Fund_Lab_AI_Design_v2_Complete.md`
  - original AI design and ML architecture

- `AI_Fund_Lab_ML_Implementation_Spec_v1.md`
  - implementation specification
