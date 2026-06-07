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
  - Portfolio Manager AI data lineage, training, Phase 3-C, Phase 3-D full backtest, v2_75 detail audit, v2_76 drawdown audit, v2_77 per-code cap, capital utilization, and candidate-pool expansion audit

## Adoption Notes

- `v2_66_ml_ranked_adoption_notes.md`
  - first ML-ranked existing-strategy candidate

- `v2_73_adoption_notes.md`
  - prior tentative main profile before Portfolio Manager AI Phase 3-D

Current strongest research candidate:

- `rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030`
  - v2_76-derived profile with low-PM-score skip and per-code exposure cap `0.30`
  - current best balance after Phase 3-G/3-H/3-I: profit/PF remain strong while DD stays below 8%
  - latest audits are summarized in `ML_Phase_25_to_Portfolio_Manager_AI_Summary.md`
  - still a backtest/research profile; not connected to live order placement

- `daily_ai_candidate_operation.md`
  - human-review daily AI candidate output
  - no order placement

## Design Notes

- `AI_Fund_Lab_AI_Design_v2_Complete.md`
  - original AI design and ML architecture

- `AI_Fund_Lab_ML_Implementation_Spec_v1.md`
  - implementation specification
