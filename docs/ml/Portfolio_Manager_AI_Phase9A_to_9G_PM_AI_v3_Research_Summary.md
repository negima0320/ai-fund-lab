# Portfolio Manager AI Phase 9-A to 9-G PM AI v3 Research Summary

作成日: 2026-06-09

## 目的

Phase 9では、現行チャンピオン `rookie_dealer_02_v2_82_cap38` を維持したまま、PM AIをゼロベースで再設計した。

再設計の軸は以下の3層である。

- Layer 1: Market Regime Model
- Layer 2: Candidate Ranking Model
- Layer 3: Position Sizing Model

全Phaseを通じて、PM AI v3のfeatureにはJ-Quants/API由来データ、walk-forward prediction、予測時点で生成可能な同日候補relative featureのみを許可した。バックテスト結果、売買結果、損益、cash、portfolio、position、selected/bought/affordable、exit/skip/final_assets、current PM multiplier模倣はfeatureとして禁止した。

current PM AI、current Exit AI、`rookie_dealer_02_v2_82_cap38` profileは上書きしていない。

## 参照成果物

詳細レポートは `reports/ml/` 配下に生成されている。ただし `reports/ml/`、`data/ml/`、`models/ml/` は生成物領域であり、通常はgit管理対象外である。このsummaryでは主要結論と再現用コードの位置を記録する。

- `reports/ml/phase9a_pm_ai_rearchitecture_audit_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9b_pm_ai_v3_dataset_builder_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9c_pm_ai_v3_dataset_audit_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9d_pm_ai_v3_trainer_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9b2_pm_ai_v3_coverage_root_cause_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9b3_pm_ai_v3_pm_sizing_universe_dataset_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9d2_pm_ai_v3_trainer_pm_sizing_universe_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9d2b_mapping_stability_audit_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9d3_mapping_threshold_audit_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9e2_pm_ai_v3_integration_pm_sizing_universe_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9f2_pm_ai_v3_backtest_candidate_2023-01_to_2026-05.{md,json}`
- `reports/ml/phase9g_pm_disabled_equal_weight_backtest_2023-01_to_2026-05.{md,json}`

## Phase 9-A: PM AI Re-Architecture Audit

現行PM AIの役割を、v2_82のtrades/candidates/predictionsから監査した。

主な結論:

- v2_82ではPM1.30が利益品質の強いシグナルになっていた。
- Phase 8のPM AI v2 raw/calibrated/relative allocatorは、v2_82のPM効果を再現できなかった。
- PM AIは単純な未来return予測ではなく、市場判断、同日候補内ランキング、資金配分を分けて設計すべき。
- future return系はlabel/評価指標としてのみ使用可能。
- v2_82は維持し、PM AI v3は研究候補として別系統で進める。

## Phase 9-B / 9-C: Clean Dataset and Dataset Audit

PM AI v3用のclean dataset builderと監査を追加した。1行は `prediction_date x code` のStock Selection候補である。

設計:

- historical predictionは `data/ml/walk_forward_predictions/` を使う。
- current modelで過去予測を再生成しない。
- 同日候補relative featureは、候補が出揃った後、cash/portfolio/backtest判断前に計算する。
- labelはJ-Quants価格由来のfuture utilityで作る。
- featureにfuture/label/target系を混入させない。

初期datasetはtop10固定寄りで、実際のPM sizing対象とのcoverage不足が残った。

## Phase 9-D: Initial PM AI v3 Trainer

初期PM AI v3 trainerを実装し、ranking/downside/classifier系のprototype modelを作った。dataset上では候補ランキング効果が見えたが、実profile候補集合とのjoin coverage問題が残ったため、この段階ではintegration採用不可とした。

## Phase 9-E / 9-F: First Integration and Backtest Candidate Audit

v2_82のStock Selection AI、Exit AI、cap38、affordability、ordering、fallbackなどを維持し、PM部分のみをPM AI v3 candidateへ差し替える研究用profileを追加した。

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

- current PMとのjoin rowsが0であり、v2_93系はPM AI v3の実力評価として不完全だった。
- 実質的にはPM1.00 fallbackに近い挙動となり、v2_82に大きく劣後した。
- 採用条件を満たさないため不採用。

## Phase 9-B2 / 9-B3: Coverage Root Cause and PM Sizing Universe Dataset

coverage 0の原因を監査し、top10固定datasetでは実際のPM sizing対象を十分に覆えないことを確認した。対策として、PM sizing universe datasetを作成した。

PM sizing universe dataset:

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
- 旧top10 universeモデルは使わず、PM sizing universeで再訓練する必要がある。

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

複数年度と市場環境でmapping安定性を監査した。

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
- `mapping_e_classifier_gate` は品質が高いが、2026年のPM1.30 countが0で件数不足が課題。

## Phase 9-D3: Mapping Threshold Optimization Audit

`mapping_e_classifier_gate` に固定し、閾値gridを監査した。再学習、integration、backtestは行っていない。

recommended config:

| config_id | classifier_gate_threshold | rank_threshold | downside_threshold | PM1.30 count | average_delta | consistency_score | 2026 PM1.30 count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e_139` | 0.80 | 0.75 | 0.80 | 1,229 | 0.0886 | 1.00 | 1 |

解釈:

- dataset上ではPM1.30候補の平均deltaとconsistencyは良い。
- ただし2026 PM1.30 countは1件だけで、実用的な安定性は弱い。
- Phase 9-E2へ進む価値はあるが、採用判断ではなくintegration監査に限定すべき。

## Phase 9-E2: PM Sizing Universe Integration Audit

PM sizing universe由来のPhase 9-D2 modelとD3推薦閾値を、実際のv2_82 PM sizing候補集合に統合できるかを監査した。採用ではなく、coverageとruntime lookupの成立確認が目的である。

追加/確認したruntime surface:

- `PortfolioManagerV3SizingAdvisor`
- Phase 9-D2 model directory support
- D3推薦閾値 mapping support
- PM v3 lookup missing時のPM1.00 fallback

結論:

- PM sizing universeによりcoverageは大幅に改善した。
- ただしPM1.30発火はまだ少なく、採用判断にはbacktest candidate auditが必要。
- current PM AI、current Exit AI、v2_82は上書きしていない。

## Phase 9-F2: PM AI v3 Backtest Candidate Audit

Phase 9-E2で統合可能になったmappingを、v2_82のPM部分だけ差し替えた研究用profileとしてbacktest監査した。

追加profile:

- `rookie_dealer_02_v2_94_pm_ai_v3_e139_candidate`
- `rookie_dealer_02_v2_94b_pm_ai_v3_e140_candidate`
- `rookie_dealer_02_v2_94c_pm_ai_v3_e120_candidate`
- `rookie_dealer_02_v2_94d_pm_ai_v3_rank_score_candidate`
- `rookie_dealer_02_v2_94e_pm_ai_v3_rank_downside_blend_candidate`

結果:

- すべてv2_82に劣後した。
- coverageはおおむね96.6%から96.9%まで改善した。
- しかしPM1.30発火、特に2026年の発火が少なく、資金配分効果としては不十分だった。
- 採用条件を満たさず、全candidateを不採用。

判断:

- PM AI v3候補は研究としては前進したが、v2_82のPM部分を置換できない。
- v2_82は継続維持。

## Phase 9-G: PM Disabled / Equal Weight Baseline Audit

PM AIを完全無効化し、すべてのBUYをPM1.00固定にするbaselineを監査した。目的は「PMそのものが効いているのか」を切り分けることである。

追加profile:

- `rookie_dealer_02_v2_95_pm_disabled_equal_weight_cap38`
- alias: `rookie_dealer_02_v2_95`
- alias: `rookie_dealer_02_v2.95`

実装:

- `portfolio_manager_ai_sizing.rule: disabled_equal_weight`
- PM modelは読まない。
- PM AI v3 lookupはしない。
- current PM AI predictionはしない。
- `pm_multiplier=1.00`
- `pm_status=disabled`
- `pm_model_version=disabled`
- `pm_missing_reason=pm_disabled`
- PM low-score skipは無効。
- PM-aware orderingは無効。
- orderingはStock Selection側の `risk_adjusted_score` ベースに戻す。
- affordability、fallback、cap38、Exit AIは維持。

backtest比較:

| profile | net_profit | PF | DD | win_rate | total_trades | final_assets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rookie_dealer_02_v2_82_cap38` | 3,777,545 | 2.7309 | -6.54% | 55.11% | 502 | 5,720,597 |
| `rookie_dealer_02_v2_95_pm_disabled_equal_weight_cap38` | 801,879 | 1.4499 | -18.13% | 44.05% | 515 | 2,008,511 |

PM disabled correctness:

| item | result |
| --- | --- |
| BUY rows | 512 |
| PM1.00 BUY rows | 512 |
| non-PM1.00 BUY rows | 0 |
| `pm_status=disabled` | 512 |
| `pm_model_version=disabled` | 512 |
| `pm_missing_reason=pm_disabled` | 512 |
| PM model lookup used | false |
| PM AI v3 lookup used | false |
| current PM AI lookup used | false |
| PM low-score skip used | false |

結論:

- PM disabled baselineはv2_82に大きく劣後した。
- PM1.00固定ではprofit/PF/DD/win rateが悪化する。
- 現行v2_82のPM layerは、少なくともbacktest上では有意に効いている。
- v2_95は採用不可。ただし、PM効果のnegative controlとして価値がある。

## Leakage Checklist

| item | result |
| --- | --- |
| forbidden feature count | 0 in generated audits |
| label/future columns in features | 0 in generated audits |
| backtest artifacts used as features | false |
| current PM multiplier imitation | false |
| OpenAI API usage | not used |
| J-Quants API refetch | not used |
| live order | not used |
| current PM AI overwrite | false |
| current Exit AI overwrite | false |
| v2_82 profile overwrite | false |

## Current Decision

現時点の本命は引き続き `rookie_dealer_02_v2_82_cap38`。

Phase 9で確認できたこと:

- clean PM AI v3 dataset設計はできた。
- PM sizing universeによりcoverage問題は改善した。
- mapping stabilityとthreshold auditでは、dataset上の有望候補は見つかった。
- しかしintegration/backtest candidateではv2_82を超えなかった。
- PM disabled equal-weight baselineもv2_82に大きく劣後した。

したがって、current PM AIを置換しない。PM AI v3は研究継続候補だが、本番採用・v2_82置換・current artifact上書きはしない。

## Added Code Surface

Phase 9で追加/変更した主なモジュール:

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
- `src/ml/portfolio_manager_v3_integration_audit_pm_sizing_universe.py`
- `src/ml/portfolio_manager_v3_backtest_candidate_audit_pm_sizing_universe.py`
- `src/ml/portfolio_manager_pm_disabled_equal_weight_audit.py`

Phase 9で追加した主なprofile:

- `rookie_dealer_02_v2_93_pm_ai_v3_candidate`
- `rookie_dealer_02_v2_93b_pm_ai_v3_candidate_conservative`
- `rookie_dealer_02_v2_93c_pm_ai_v3_candidate_half_pm130`
- `rookie_dealer_02_v2_94_pm_ai_v3_e139_candidate`
- `rookie_dealer_02_v2_94b_pm_ai_v3_e140_candidate`
- `rookie_dealer_02_v2_94c_pm_ai_v3_e120_candidate`
- `rookie_dealer_02_v2_94d_pm_ai_v3_rank_score_candidate`
- `rookie_dealer_02_v2_94e_pm_ai_v3_rank_downside_blend_candidate`
- `rookie_dealer_02_v2_95_pm_disabled_equal_weight_cap38`

## Verification

Phase 9-Gで実行した軽量pytest:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 -m pytest -q tests/test_ml_phase9g_pm_disabled_equal_weight.py
```

結果:

- `4 passed in 0.12s`

Phase 9-G backtest:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 src/main.py --mode backtest --provider jquants --profile rookie_dealer_02_v2_95_pm_disabled_equal_weight_cap38 --start-date 2023-01-01 --end-date 2026-05-31 --skip-price-fetch --summary-only --no-daily-logs --quiet --progress-interval 50
```

注意:

- 価格再取得はskipし、cached pricesを使用した。
- OpenAI/news/live orderは無効。
- 最終価格日が2026-05-28のためbacktest coverage auditは `coverage_ok: False` となるが、summary/trades/purchase auditは生成され、比較監査に使用した。
