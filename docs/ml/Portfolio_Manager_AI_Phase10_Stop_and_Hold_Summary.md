# Portfolio Manager AI / Phase 10 Stop and Hold Summary

作成日: 2026-06-11

## 結論

Phase 10ならびにPM AI再開発は、いったん開発中止・保留とする。

現時点の判断は以下である。

- 現行チャンピオンは引き続き `rookie_dealer_02_v2_82_cap38`
- PM AI v2 / PM AI v3 / PM disabled baseline / score-based PM ruleはいずれもv2_82を置換しない
- Phase 10-B以降のPM threshold調整、追加backtest、再学習、strategy integrationは停止
- current PM AI、current Exit AI、v2_82 profileは上書きしない
- 研究成果は残すが、採用候補としては凍結する

## 背景

PM AI研究の主目的は、v2_82で効いているPM multiplierを、leakageなしに再現または改善することだった。

一貫して守った制約は以下である。

- featureはJ-Quants/API由来データ、walk-forward prediction、予測時点で生成可能な同日候補relative featureに限定
- バックテスト結果、売買結果、損益、cash、portfolio、position、selected/bought/affordable、exit/skip/final_assets、current PM multiplier模倣はfeature禁止
- future return系はlabelまたは評価指標としてのみ使用
- historical predictionは既存walk-forward predictionを使い、current modelで過去予測を再生成しない
- current PM AI、current Exit AI、v2_82 profileは上書きしない

## これまでの流れ

### Phase 7: v2_82がVersion 1.0候補になった

`rookie_dealer_02_v2_82_cap38` が、現時点で最も強いfull-backtested research candidateになった。

主要成績:

| profile | net_profit | PF | DD | win_rate | CAGR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rookie_dealer_02_v2_82_cap38` | 3,777,545 | 2.7309 | -6.54% | 55.11% | 66.74% |

この時点で、PM AIを含む既存構成は強いが、PM AIそのものにはleakageや再現性に関する懸念が残っていた。

### Phase 8: PM AI v2とrelative allocatorは不採用

PM AI API-only candidate、raw integration、calibrated integration、relative allocatorを試した。

結果:

- v2_90 raw integrationはv2_82に届かなかった
- v2_91 calibrated integrationもv2_82に届かなかった
- v2_92 rule-based relative allocatorもv2_82に届かなかった

判断:

- PM AI v2は採用しない
- relative allocatorは採用しない
- v2_82を維持する

### Phase 9: PM AI v3をゼロベースで再設計した

PM AIを以下の3層に分解して再設計した。

- Layer 1: Market Regime Model
- Layer 2: Candidate Ranking Model
- Layer 3: Position Sizing Model

実施内容:

- clean dataset builder
- dataset quality / label audit
- PM sizing universe dataset
- PM AI v3 trainer prototype
- mapping stability audit
- threshold optimization audit
- integration audit
- backtest candidate audit
- PM disabled / equal-weight baseline audit

主な成果:

- PM sizing universe datasetにより、実際のPM sizing候補へのcoverageは改善した
- dataset上では一部mappingにPM1.30品質の改善が見えた
- ただし実backtestではv2_82を置換できなかった

代表結果:

| profile | net_profit | PF | DD | win_rate | judgment |
| --- | ---: | ---: | ---: | ---: | --- |
| `rookie_dealer_02_v2_82_cap38` | 3,777,545 | 2.7309 | -6.54% | 55.11% | keep |
| `v2_93` PM AI v3 first candidate | 783,708 | 1.4353 | -18.13% | 43.97% | reject |
| `v2_94*` PM sizing universe candidates | v2_82未満 | v2_82未満 | v2_82未満 | v2_82未満 | reject |
| `v2_95` PM disabled equal weight | 801,879 | 1.4499 | -18.13% | 44.05% | reject |

判断:

- PM AI v3は研究としては前進した
- しかしPM1.30発火数、年度安定性、実backtest成績が不足した
- PM disabled baselineは大きく劣後し、PM機構そのものは必要と判断した
- v2_82を維持する

### Phase 10-A: score-based PM ruleを試した

新規学習やPM model推論を使わず、Stock Selection AIが予測時点で出しているスコアだけでPM multiplierを決めるルールを検証した。

使用featureは以下4つに限定した。

- `risk_adjusted_score`
- `expected_return`
- `stock_selection_rank_score`
- `candidate_strength`

追加profile:

- `rookie_dealer_02_v2_96_score_based_pm_rule_a`
- `rookie_dealer_02_v2_96b_score_based_pm_rule_b`
- `rookie_dealer_02_v2_96c_score_based_pm_rule_c`

結果:

| profile | net_profit | PF | DD | win_rate | monthly_win_rate | trades | final_assets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2_82_reference` | 3,777,545 | 2.7309 | -6.54% | 55.11% | 78.05% | 502 | 5,720,597 |
| `v2_95_pm_disabled` | 801,879 | 1.4499 | -18.13% | 44.05% | 53.66% | 515 | 2,008,511 |
| `v2_96_rule_a` | 685,289 | 1.3971 | -10.62% | 42.23% | 58.54% | 523 | 1,855,998 |
| `v2_96b_rule_b` | 706,051 | 1.4018 | -13.62% | 44.67% | 46.34% | 499 | 1,888,052 |
| `v2_96c_rule_c` | 923,134 | 1.4837 | -11.72% | 45.00% | 58.54% | 522 | 2,160,679 |

Rule CはPM disabled baselineを上回ったが、v2_82には大きく届かなかった。

Rule CのPM bucket品質:

| PM | buy_count | trade_count | profit | PF | win_rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.30 | 218 | 217 | 434,260 | 1.4248 | 46.54% |
| 1.15 | 44 | 46 | -64,023 | 0.7632 | 26.09% |
| 1.00 | 231 | 233 | 11,136 | 1.0108 | 46.78% |
| 0.80 | 17 | 17 | -32,717 | 0.4222 | 29.41% |
| 0.60 | 8 | 9 | 87,895 | 7.9824 | 77.78% |

重要な問題:

- PM0.60が非常に良く、PM1.15/0.80が悪い
- bucketの単調性が崩れている
- PM1.30品質もv2_82のPM1.30を置換できる水準ではない
- score-based ruleは単純でleakage riskは低いが、資金配分ロジックとしては未成熟

判断:

- v2_96c Rule Cはv2_95比較ではreview candidate
- ただし強い採用条件は満たさない
- v2_82は維持

### Phase 10-B: threshold robustness auditは途中停止

Rule Cのbucket不整合を調べるため、threshold / weight variantのrobustness auditを開始した。

追加された研究用profile:

- `rookie_dealer_02_v2_97_score_based_pm_rule_c_opt1`
- `rookie_dealer_02_v2_97b_score_based_pm_rule_c_opt2`
- `rookie_dealer_02_v2_97c_score_based_pm_rule_c_opt3`
- `rookie_dealer_02_v2_97d_score_based_pm_rule_c_opt4`
- `rookie_dealer_02_v2_97e_score_based_pm_rule_c_opt5`

途中までのbacktest結果:

| profile | variant | net_profit | DD | win_rate | total_trades |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2_97` | conservative_high / original | 949,378 | -12.82% | 44.66% | 515 |
| `v2_97b` | no_060 / original | 773,218 | -11.83% | 43.45% | 504 |
| `v2_97c` | no_115 / original | 923,134 | -11.72% | 45.00% | 520 |
| `v2_97d` | inverted_low_check / original | 976,830 | -12.11% | 45.30% | 532 |

これらはv2_95よりは改善する場合があるが、v2_82にはまったく届いていない。

`v2_97e` 実行中に方針変更が入り、Phase 10とPM AI開発を停止した。停止時点で継続中のbacktestプロセスは残っていないことを確認した。

## Phase 9 cleanup

Phase 10-Aで、Phase 9 PM AI v3の重い生成物を削除した。

削除対象:

- `data/ml/portfolio_manager_v3`
- `models/ml/portfolio_manager_v3`
- v2_93 / v2_94系backtest logs
- v2_93 / v2_94系research profiles

容量:

- saved: 500,684,097 bytes

残したもの:

- Phase 9 reports
- Phase 9 audit scripts/tests
- v2_95 PM disabled baseline profile/logs
- v2_82 logs
- current PM AI
- current Exit AI
- Stock Selection AI

## なぜ止めるか

PM AI研究は、leakageを避けた設計としてはかなり整理された。

一方で、採用に必要な実利が出ていない。

主な理由:

- v2_82のPM部分が非常に強く、単純な未来return予測やrelative rankingでは代替できない
- dataset上で良く見えるmappingが、実backtestではPM1.30発火数や年度安定性で不足する
- PM disabled baselineが大きく劣後し、PM機構そのものは必要だが、cleanな再実装ではまだ再現できない
- score-based ruleはPM disabledより改善したが、v2_82との差が大きすぎる
- threshold調整を続けても、現時点ではv2_82を超える根拠が薄い

## 現在の採用判断

| target | decision | reason |
| --- | --- | --- |
| `rookie_dealer_02_v2_82_cap38` | keep | 現時点の最強research candidate |
| current PM AI | keep unchanged | v2_82構成を壊さない |
| current Exit AI | keep unchanged | v2_82構成を壊さない |
| PM AI v2 | reject / frozen | v2_82を置換できない |
| PM AI v3 | reject / frozen | 実backtestでv2_82を置換できない |
| PM disabled equal weight | reject | v2_82に大きく劣後 |
| score-based PM rule | frozen | v2_95は上回るがv2_82に大きく劣後 |
| Phase 10-B以降 | stop | threshold探索を継続しない |

## 今後の扱い

PM AI / Phase 10系は、以下の扱いにする。

- 新規backtestを走らせない
- 追加threshold探索をしない
- 再学習しない
- strategy integrationしない
- current PM AI、current Exit AI、v2_82を上書きしない
- v2_96 / v2_97系profileは採用候補ではなく研究メモとして扱う
- PM AIを再開する場合は、新しい仮説と採用基準を先に定義する

再開する場合の最低条件:

- v2_82と同じ候補集合で比較できる
- PM1.30品質がv2_82以上
- PFがv2_82以上
- DDがv2_82以下
- profitがv2_82の95%以上
- 年度別、相場別、monthlyで安定
- leakage checklistがlow risk

## 参照ドキュメント

- `docs/ml/Portfolio_Manager_AI_Phase7A_to_7G_Final_Summary.md`
- `docs/ml/Portfolio_Manager_AI_Phase8A_to_8H_PM_AI_Redesign_Summary.md`
- `docs/ml/Portfolio_Manager_AI_Phase9A_to_9G_PM_AI_v3_Research_Summary.md`
- `docs/ml/Portfolio_Manager_AI_Phase10A_Score_Based_PM_Rule_Summary.md`

## Verification

今回の変更はドキュメント追加とREADME更新のみであるため、pytestは実行していない。

Phase 10-B停止確認:

- 2026-06-11時点で、`rookie_dealer_02_v2_97e` または `src/main.py --mode backtest` の継続プロセスは残っていないことを確認した。
