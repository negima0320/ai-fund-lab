# Portfolio Manager AI Phase 9-A to 9-D3 PM AI v3 Summary

作成日: 2026-06-09

## 目的

Phase 9では、現行チャンピオン `rookie_dealer_02_v2_82_cap38` を維持したまま、PM AIを

- 市場判断
- 同日候補内ランキング
- 資金配分

の3層として作り直すための研究系統を進めた。全Phaseを通じて、PM AI v3のfeatureにはJ-Quants/API由来データ、walk-forward prediction、予測時点で作れる同日候補relative featureのみを許可した。バックテスト結果、売買結果、損益、cash、portfolio、position、selected/bought/affordable、exit/skip/final_assets/current PM multiplier模倣はfeatureとして禁止した。

current PM AI、current Exit AI、`rookie_dealer_02_v2_82_cap38` profileは上書きしていない。

## 参照成果物

生成された詳細レポートは `reports/ml/` 配下にある。ただし `reports/ml/`、`data/ml/`、`models/ml/` は `.gitignore` 上の生成物領域であり、このsummaryでは主要結論と再現用コードの位置を記録する。

- `reports/ml/phase9a_pm_ai_rearchitecture_audit_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9b_pm_ai_v3_dataset_builder_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9c_pm_ai_v3_dataset_audit_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9d_pm_ai_v3_trainer_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9e_pm_ai_v3_integration_audit_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9f_pm_ai_v3_backtest_candidate_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9b2_pm_ai_v3_coverage_root_cause_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9b3_pm_ai_v3_pm_sizing_universe_dataset_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9d2_pm_ai_v3_trainer_pm_sizing_universe_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9d2b_mapping_stability_audit_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9d3_mapping_threshold_audit_2023-01_to_2026-05.{md,json}`

## Phase 9-A: Re-Architecture Audit

現行PM AIの役割を、v2_82のtrades/candidates/predictionsから監査した。v2_82ではPM1.30が利益品質の強いシグナルになっていたが、Phase 8のPM AI v2 raw/calibrated/relative allocatorではこれを再現できなかった。

主な結論:

- PM AIは単なる未来return予測ではなく、候補集合内での順位、同日の市場状態、資金配分を分けて扱うべき。
- 使用可能feature、条件付きrelative feature、禁止featureを明確に分離した。
- future return系はlabel/評価指標としてのみ使用可能。
- v2_82は維持し、PM AI v3は研究候補として別系統で進める。

## Phase 9-B / 9-C: Clean Dataset Builder and Quality Audit

PM AI v3用に、1行を `prediction_date x code` のStock Selection候補として構成するclean dataset builderと監査を追加した。

主な設計:

- 入力は `data/ml/walk_forward_predictions/` のhistorical predictionを使う。
- current modelで過去予測を再生成しない。
- 同日候補relative featureは、Stock Selection候補が出揃った後、cash/portfolio/backtest判断前に計算する。
- labelはJ-Quants価格由来のfuture utilityで作る。
- featureにfuture/label/target系を混入させない。

初期datasetはtop10固定寄りで、後続のPhase 9-F実候補集合とのcoverageが不足する問題が残った。

## Phase 9-D: Initial Trainer Prototype

初期PM AI v3 trainerを実装し、ranking/downside/classifier系のprototype modelを作った。dataset上では一定の候補ランキング効果が見えたが、実profile候補集合とのjoin coverage問題が残ったため、この段階ではintegration採用不可とした。

## Phase 9-E / 9-F: Integration and Backtest Candidate Audit

Phase 9-Fでは、v2_82のStock Selection AI、Exit AI、cap38、affordability、ordering、fallbackなどを維持し、PM部分のみをPM AI v3 candidateに差し替える研究用profileを追加した。

追加profile:

- `rookie_dealer_02_v2_93_pm_ai_v3_candidate`
- `rookie_dealer_02_v2_93b_pm_ai_v3_candidate_conservative`
- `rookie_dealer_02_v2_93c_pm_ai_v3_candidate_half_pm130`

結果:

| profile | net_profit | PF | DD | win_rate | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `rookie_dealer_02_v2_82_cap38` | 3,777,545 | 2.7309 | -6.54% | 55.11% | keep |
| `v2_93_a/b/c` | 783,708 | 1.4353 | -18.13% | 43.97% | reject |

重要な発見:

- PM v3 lookup coverageが0になり、v2_93系はPM multiplierが実質1.00にフォールバックした。
- したがってPhase 9-Fのv2_93 backtestはPM AI v3の実力評価としては不完全。
- ただし採用条件は満たさず、v2_82を超えていないため不採用。

## Phase 9-B2 / 9-B3: Coverage Root Cause and PM Sizing Universe Dataset

coverage 0の原因を監査し、top10固定datasetでは実際のPM sizing対象を十分に覆えないことを確認した。対策としてPM sizing universe datasetを作成した。

Phase 9-B3 dataset:

- path: `data/ml/portfolio_manager_v3/portfolio_manager_v3_dataset_pm_sizing_universe_2023-01_to_2026-05.parquet`
- rows: 947,438
- date range: 2023-01-04 to 2026-04-27
- code count: 4,234
- Phase 9-F PM sizing key coverage: 96.62%
- feature count: 58
- label count: 10
- forbidden feature count: 0
- leakage risk: low

結論:

- coverage問題はdataset設計で改善した。
- ただしPhase 9-Dの旧top10 universeモデルをそのまま使うべきではなく、PM sizing universeで再訓練する必要がある。

## Phase 9-D2: PM Sizing Universe Trainer

PM sizing universe datasetでPM AI v3を再訓練した。これはcurrent PM AIの置換ではなく、研究candidate modelである。

出力model:

- `models/ml/portfolio_manager_v3/candidate_phase9d2_pm_sizing_universe/`

学習概要:

- train rows: 513,469
- validation rows: 309,092
- test rows: 124,877
- feature count after drops: 50
- forbidden feature count: 0
- leakage risk: low

主なmapping候補:

- `mapping_a_rank_score_only`
- `mapping_b_downside_utility_only`
- `mapping_c_rank_plus_downside_blend`
- `mapping_d_conservative_high_conviction`
- `mapping_e_classifier_gate`

test上では `mapping_d_conservative_high_conviction` がPM1.30のdownside utilityで有望に見えたが、年次安定性の確認が必要になった。

## Phase 9-D2B: Mapping Stability Audit

`mapping_d_conservative_high_conviction` が2026年だけ偶然良かった/悪かったのかを監査した。

主要結果:

| mapping | yearly_positive_count | average_delta | worst_year_delta | consistency_score |
| --- | ---: | ---: | ---: | ---: |
| `mapping_a_rank_score_only` | 4 | 0.0162 | 0.0108 | 1.00 |
| `mapping_b_downside_utility_only` | 3 | 0.0354 | 0.0162 | 1.00 |
| `mapping_c_rank_plus_downside_blend` | 4 | 0.0429 | 0.0270 | 1.00 |
| `mapping_d_conservative_high_conviction` | 3 | 0.0346 | -0.0592 | 0.75 |
| `mapping_e_classifier_gate` | 3 | 0.0772 | 0.0429 | 1.00 |

結論:

- best_mapping_by_stability: `mapping_e_classifier_gate`
- best_mapping_by_performance: `mapping_e_classifier_gate`
- `mapping_d_is_stable`: false
- `mapping_e_classifier_gate` は品質が高いが、2026年のPM1.30 countが0であり件数不足が課題。

## Phase 9-D3: Mapping Threshold Optimization Audit

`mapping_e_classifier_gate` に固定し、閾値gridを監査した。再学習、integration、backtestは行っていない。

grid:

- classifier gate: 0.50 to 0.90
- rank threshold: top5% to top25%
- downside threshold: top10% to top25%
- total configs: 180

recommended config:

| config_id | classifier_gate_threshold | rank_threshold | downside_threshold | PM1.30 count | average_delta | consistency_score | 2026 PM1.30 count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e_139` | 0.80 | 0.75 | 0.80 | 1,229 | 0.0886 | 1.00 | 1 |

解釈:

- dataset上ではPM1.30候補の平均deltaとconsistencyは良い。
- ただし2026 PM1.30 countは1件だけで、2026の実用的な安定性はまだ弱い。
- Phase 9-E2へ進む価値はあるが、採用判断ではなくintegration監査に限定すべき。

## Leakage Checklist

Phase 9-AからPhase 9-D3まで一貫して以下をチェックした。

| item | result |
| --- | --- |
| forbidden feature count | 0 in generated audits |
| label/future columns in features | 0 in generated audits |
| backtest artifacts used as features | false |
| current PM multiplier imitation | false |
| OpenAI API usage | not used |
| J-Quants API refetch | not used |
| current PM AI overwrite | false |
| current Exit AI overwrite | false |
| v2_82 profile overwrite | false |

## Current Decision

現時点の本命は引き続き `rookie_dealer_02_v2_82_cap38`。

PM AI v3は、clean dataset、PM sizing universe、trainer、mapping stability、threshold auditまで進んだが、まだv2_82のPM部分を置換できる段階ではない。特にPhase 9-Fでは旧dataset/modelのcoverage問題によりcandidate profileがPM1.00 fallbackへ落ち、v2_82に大きく劣後した。

次に進むなら、Phase 9-E2として `candidate_phase9d2_pm_sizing_universe` とD3推薦閾値を使ったintegration auditを行う。ただし目的は採用ではなく、PM sizing universe由来モデルが実際のv2_82候補集合でPM multiplierを付与できるかを確認することである。

## Added Code Surface

主な追加モジュール:

- `src/ml/phase9a_pm_ai_rearchitecture_audit.py`
- `src/ml/portfolio_manager_v3_dataset_builder.py`
- `src/ml/portfolio_manager_v3_dataset_audit.py`
- `src/ml/portfolio_manager_v3_trainer.py`
- `src/ml/portfolio_manager_v3_integration_audit.py`
- `src/ml/portfolio_manager_v3_backtest_candidate_audit.py`
- `src/ml/portfolio_manager_v3_coverage_root_cause_audit.py`
- `src/ml/portfolio_manager_v3_pm_sizing_universe_builder.py`
- `src/ml/portfolio_manager_v3_pm_sizing_universe_trainer.py`
- `src/ml/portfolio_manager_v3_mapping_stability_audit.py`
- `src/ml/portfolio_manager_v3_mapping_threshold_audit.py`

主な追加script:

- `scripts/ml/audit_phase9a_pm_ai_rearchitecture.py`
- `scripts/ml/build_portfolio_manager_v3_dataset.py`
- `scripts/ml/audit_portfolio_manager_v3_dataset.py`
- `scripts/ml/train_portfolio_manager_v3.py`
- `scripts/ml/audit_portfolio_manager_v3_integration.py`
- `scripts/ml/audit_portfolio_manager_v3_backtest_candidate.py`
- `scripts/ml/audit_portfolio_manager_v3_coverage_root_cause.py`
- `scripts/ml/build_portfolio_manager_v3_pm_sizing_universe_dataset.py`
- `scripts/ml/train_portfolio_manager_v3_pm_sizing_universe.py`
- `scripts/ml/audit_portfolio_manager_v3_mapping_stability.py`
- `scripts/ml/audit_portfolio_manager_v3_mapping_threshold.py`

主な追加tests:

- `tests/test_ml_phase9a_pm_ai_rearchitecture_audit.py`
- `tests/test_ml_phase9b_pm_ai_v3_dataset_builder.py`
- `tests/test_ml_phase9c_pm_ai_v3_dataset_audit.py`
- `tests/test_ml_phase9d_pm_ai_v3_trainer.py`
- `tests/test_ml_phase9e_pm_ai_v3_integration_audit.py`
- `tests/test_ml_phase9f_pm_ai_v3_backtest_candidate.py`
- `tests/test_ml_phase9b2_pm_ai_v3_coverage_root_cause.py`
- `tests/test_ml_phase9b3_pm_ai_v3_pm_sizing_universe_dataset.py`
- `tests/test_ml_phase9d2_pm_ai_v3_trainer_pm_sizing_universe.py`
- `tests/test_ml_phase9d2b_mapping_stability_audit.py`
- `tests/test_ml_phase9d3_mapping_threshold_audit.py`

