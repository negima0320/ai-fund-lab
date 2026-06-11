# Portfolio Manager AI Phase 11 Valuation + Capital Allocation Plan

作成日: 2026-06-11

## 結論

Phase 11では、PM AIを直接発展させるのではなく、資金配分を以下の2層に分解する。

```text
Valuation Engine
↓
Capital Allocation Engine
```

目的は、既存PM AIの倍率を模倣することではない。J-Quants/API由来かつ予測時点で取得可能な情報だけを使い、「この銘柄は現在価格に対して期待値が高いか」を評価し、その評価に応じて購入金額を決める。

`rookie_dealer_02_v2_82_cap38` は引き続き参考記録として扱うが、Phase 11の採用対象ではない。

## 背景

Phase 9からPhase 10までの検証で、以下が判明した。

### 現行PM AI

バックテスト上は非常に強い。

一方で、学習データの再現性と将来運用で同一条件を再現できる保証に課題があるため、新規採用対象にはしない。

### PM AI v3

リーク除去後に再設計し、coverage問題も改善したが、PF、利益、DDが未達だった。

判断:

- PM AI v3は不採用
- current PM AI、current Exit AI、v2_82 profileは上書きしない

### PM Disabled

PM倍率をすべて1.0固定にすると、利益、PF、DDが大幅に悪化した。

結論:

```text
資金配分機構そのものは必要
```

### Rule-Based PM

Phase 10-Aでは以下のStock Selection prediction-time scoreだけを使ったルールベースPMを検証した。

- `risk_adjusted_score`
- `expected_return`
- `stock_selection_rank_score`
- `candidate_strength`

Rule CはPM Disabledより改善したが、PM0.60 bucketが高品質になるなど、資金配分ロジックとしては未成熟だった。

## Phase 11の目的

Phase 11では、PM multiplierを直接予測または模倣しない。

代わりに以下を分離する。

1. Valuation Engine: 現在価格に対して期待値が高いかを評価する。
2. Capital Allocation Engine: Opportunityに応じていくら買うかを決める。

この分離により、Phase 9からPhase 10で問題になった「倍率bucketの品質不整合」や「PM1.30発火数の再現」を直接目的にしない。

## Layer 1: Valuation Engine

### 目的

判定したいのは以下である。

```text
この銘柄は今お得か？
```

重要なのは、単に「上がるか」ではない。

判定したいのは以下である。

```text
現在価格に対して期待値が高いか
```

### 出力

Valuation Engineは以下を出力する。

| output | meaning | example |
| --- | --- | --- |
| `opportunity_score` | 0から100の機会スコア | 95: 超お得、80: お得、60: 普通、30: 微妙、10: 危険 |
| `expected_upside` | 期待上昇余地 | `+35%`, `+20%`, `+5%` |
| `expected_downside` | 期待下落余地 | `-5%`, `-10%`, `-20%` |
| `confidence` | 判定への信頼度 | 90: 高信頼、50: 普通、20: 低信頼 |

### 利用可能な入力

絶対条件:

```text
J-Quants API由来
+
予測時点で取得可能
```

利用候補:

| group | examples |
| --- | --- |
| Market | TOPIX乖離、TOPIXリターン、TOPIXボラ、TOPIX高値圏/安値圏 |
| Price | 移動平均乖離、52週高値比、52週安値比、RS、ボラティリティ |
| Volume | 出来高、出来高急増率、流動性 |
| Financial | EPS、BPS、売上成長、利益成長、ROE、自己資本比率、利益率 |
| Stock Selection | `risk_adjusted_score`, `expected_return`, `stock_selection_rank_score`, `candidate_strength` |

### 利用禁止

絶対禁止:

- バックテスト結果
- 売買結果
- 損益
- cash
- portfolio
- selected
- bought
- affordable
- current PM multiplier

Future系の扱い:

- `future_return` や `future_profit` はlabelまたは評価指標としてのみ使用する。
- future系をfeatureとして使わない。

## Layer 2: Capital Allocation Engine

### 目的

Valuation Engineが出したOpportunityを元に、いくら買うかを決定する。

### 入力

- `opportunity_score`
- `expected_upside`
- `expected_downside`
- `confidence`
- 現金
- 保有銘柄
- 単元株
- cap制限

### 出力

- 購入金額
- 購入株数

### 配分イメージ

例えば候補が以下だった場合:

| code | opportunity_score |
| --- | ---: |
| A | 95 |
| B | 80 |
| C | 70 |
| D | 40 |

均等配分ではなく、以下のような相対配分を目指す。

| code | allocation |
| --- | ---: |
| A | 40% |
| B | 30% |
| C | 20% |
| D | 10% |

### Dynamic Cap

将来的には固定`38%`ではなく、OpportunityとConfidenceに応じて上限比率を変える。

例:

| opportunity | confidence | cap |
| ---: | ---: | ---: |
| 95 | 90 | 60% |
| 95 | 30 | 20% |

## Buy / Hold / Sell

### Buy

以下を満たす候補を優先する。

- Opportunityが高い
- Expected Upsideが高い
- Expected Downsideが低い

### Hold

Opportunityが維持される限り保有する。

### Sell

将来的には、Opportunityが消えたことを売却理由として検証する。

例:

| timing | opportunity_score |
| --- | ---: |
| 購入時 | 95 |
| 現在 | 40 |

この場合は売却候補とする。

## 理想アーキテクチャ

```text
Stock Selection AI
↓
候補抽出

Valuation Engine
↓
Opportunity Score
Expected Upside
Expected Downside
Confidence

Capital Allocation Engine
↓
購入額決定

保有中
↓
Valuation再評価

Exit判断
```

## Phase 11 Roadmap

### Phase 11-A: Valuation Engine Dataset Audit

目的:

```text
Opportunityを説明できる特徴量を調査する
```

初期監査観点:

- API由来かつas-of-safeなfeature候補の棚卸し
- future label候補の定義
- forbidden feature混入チェック
- current PM multiplier模倣になっていないことの確認
- Stock Selection候補集合とValuation対象集合のcoverage確認
- opportunity labelが単純なfuture return予測に潰れていないかの確認

### Phase 11-B: Valuation Engine Prototype

目的:

```text
Opportunity Score
Expected Upside
Expected Downside
Confidence
```

を生成する。

### Phase 11-C: Capital Allocation Engine Prototype

目的:

```text
Opportunityに応じて購入金額を決定する
```

### Phase 11-C2: Budget Usage Constraint Audit

目的:

```text
Phase 11-Cのbudget usage proxyが低い理由を特定する
```

Phase 11-Dへ進む前に、単元株、候補価格、daily budget、max positions、candidate thresholdのどれが資金利用率を抑えているかを軽量監査する。

### Phase 11-D: Combined Backtest

目的:

```text
Valuation
+
Allocation
```

を統合評価する。

### Phase 11-E: Exit Integration

目的:

```text
Opportunity消滅
↓
売却
```

を検証する。

## 成功条件

最低目標:

| metric | target |
| --- | ---: |
| PF | `>= 2.0` |
| DD | `>= -10%` |
| 資金利用率 | `>= 60%` |

理想目標:

| metric | target |
| --- | ---: |
| PF | `>= 2.3` |
| DD | `>= -8%` |
| 資金利用率 | `>= 80%` |

参考記録:

| profile | PF | DD |
| --- | ---: | ---: |
| `rookie_dealer_02_v2_82_cap38` | `2.7309` | `-6.54%` |

v2_82は採用対象ではなく、参考値として扱う。

## Safety Constraints

Phase 11でも以下を守る。

- current PM AIを上書きしない。
- current Exit AIを上書きしない。
- `rookie_dealer_02_v2_82_cap38` を破壊的に変更しない。
- historical predictionはwalk-forward predictionを使う。
- current modelで過去予測を再生成しない。
- J-Quants API refetchを行わない。
- OpenAI APIを使わない。
- live order placementに接続しない。

## Next Action

最初に実施するのは Phase 11-A Valuation Engine Dataset Audit である。

この段階ではprofile作成やbacktest統合は行わず、以下をレポートする。

- feature inventory
- label design
- leakage checklist
- coverage
- first-pass feature/label relationship
- Phase 11-Bへ進めるかどうか

## Phase 11-A Implementation Status

実装済み:

- `src/ml/phase11a_valuation_dataset_audit.py`
- `scripts/ml/audit_phase11a_valuation_dataset.py`
- `tests/test_ml_phase11a_valuation_dataset_audit.py`

生成report:

- `reports/ml/phase11a_valuation_dataset_audit_2023-01_to_2026-05.md`
- `reports/ml/phase11a_valuation_dataset_audit_2023-01_to_2026-05.json`

中間dataset:

- `data/ml/valuation_engine/phase11a_valuation_dataset_2023-01_to_2026-05.parquet`

実データ監査結果:

| item | value |
| --- | ---: |
| rows | `930,243` |
| unique_codes | `4,234` |
| candidate_days | `537` |
| date_range | `2023-01-04` to `2026-04-23` |
| feature_count | `55` |
| label_count | `5` |
| leakage_risk | `low` |
| blocking_issues | `0` |
| ready_for_phase11b | `true` |

Phase 11-Aでは、backtest結果、trades.csv、損益、cash、portfolio、selected/bought/affordable、current PM multiplierをfeatureとして使用していない。

推奨:

- primary label: `opportunity_value_20d`
- secondary labels: `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_top_decile_20d`
- next task: Phase 11-B Valuation Engine Prototype

## Phase 11-B Implementation Status

実装済み:

- `src/ml/phase11b_valuation_engine_prototype.py`
- `scripts/ml/train_phase11b_valuation_engine_prototype.py`
- `tests/test_ml_phase11b_valuation_engine_prototype.py`

生成report:

- `reports/ml/phase11b_valuation_engine_prototype_2025_holdout.md`
- `reports/ml/phase11b_valuation_engine_prototype_2025_holdout.json`

candidate model:

- `models/ml/valuation_engine/candidate_phase11b/`

Scope:

- Phase 11-A datasetを使用
- train: `2023-01-04` to `2024-12-31`
- test: `2025-01-01` to `2025-12-31`
- trainは長時間化回避のため deterministic sample `250,000` rows
- full backtestなし
- profile追加/変更なし
- 既存model上書きなし

実データholdout結果:

| item | value |
| --- | ---: |
| train_rows | `250,000` |
| test_rows | `310,618` |
| feature_count | `54` |
| regression target | `opportunity_value_20d` |
| regression MAE | `0.0865` |
| regression RMSE | `0.1407` |
| regression Pearson | `0.0559` |
| regression Spearman | `-0.0028` |
| classification target | `opportunity_top_decile_20d` |
| classification AUC | `0.6478` |
| classification PR-AUC | `0.1600` |
| precision@top10% | `0.1998` |
| base positive rate | `0.0997` |
| leakage_risk | `low` |
| blocking_issues | `0` |
| ready_for_phase11c | `true` |

Valuation output:

- `opportunity_score`: `predicted_opportunity_value` のtest内percentile rankを0-100化
- `predicted_opportunity_value`: `opportunity_value_20d` regression output
- `opportunity_top_decile_proba`: top decile classification probability
- `confidence`: `abs(opportunity_top_decile_proba - 0.5) * 200`
- `expected_upside`: Phase 11-Bでは未実装
- `expected_downside`: Phase 11-Bでは未実装

Phase 11-Bでも、backtest結果、trades.csv、損益、cash、portfolio、selected/bought/affordable、current PM multiplierをfeatureとして使用していない。

推奨:

- Phase 11-C Capital Allocation Engine Prototypeへ進む
- 初期入力は `opportunity_score`, `opportunity_top_decile_proba`, `confidence`
- `expected_upside` / `expected_downside` は後続で別target modelとして追加検討

## Phase 11-C Implementation Status

実装済み:

- `src/ml/phase11c_capital_allocation_prototype.py`
- `scripts/ml/run_phase11c_capital_allocation_prototype.py`
- `tests/test_ml_phase11c_capital_allocation_prototype.py`

生成report:

- `reports/ml/phase11c_capital_allocation_prototype_2025.md`
- `reports/ml/phase11c_capital_allocation_prototype_2025.json`

軽量artifact:

- `data/ml/valuation_engine/phase11c_allocation_simulation_2025.parquet`

Scope:

- Phase 11-B candidate modelを使用
- Phase 11-A datasetを使用
- 2025年のみのallocation quality simulation
- full backtestなし
- 売却ロジックなし
- Exit AI統合なし
- profile追加/変更なし
- 既存model上書きなし

Simulation条件:

| item | value |
| --- | ---: |
| initial_cash | `1,000,000` |
| daily_buy_budget | `300,000` |
| max_positions | `5` |
| per_code_cap_rate | `0.38` |
| round_lot | `100` |

実データsimulation結果:

| item | value |
| --- | ---: |
| rows | `310,618` |
| unique_codes | `4,225` |
| candidate_days | `165` |
| date_range | `2025-01-07` to `2025-12-29` |
| leakage_risk | `low` |
| blocking_issues | `0` |
| best_rule | `equal_weight_top5` |
| ready_for_phase11d | `true` |

Rule comparison:

| rule | allocated/day | budget_usage | weighted_return_20d | weighted_max_return_20d | weighted_max_drawdown_20d | weighted_opportunity_value | weighted_top_decile_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `equal_weight_top5` | `1.2485` | `0.2072` | `0.0415` | `0.1466` | `-0.0878` | `0.0587` | `0.2403` |
| `proba_rank_weighted` | `1.2485` | `0.2072` | `0.0415` | `0.1466` | `-0.0878` | `0.0587` | `0.2403` |
| `proba_confidence_weighted` | `0.6909` | `0.1104` | `0.0242` | `0.1131` | `-0.0771` | `0.0360` | `0.1973` |
| `conservative_top_only` | `1.2485` | `0.2072` | `0.0415` | `0.1466` | `-0.0878` | `0.0587` | `0.2403` |

Interpretation:

- `equal_weight_top5`, `proba_rank_weighted`, `conservative_top_only` は2025年条件では同じ候補集合に収束した。
- `opportunity_top_decile_20d` のweighted rateは `0.2403` で、母集団約 `0.10` に対して約2.4倍。
- ただしbudget usageは約 `20.7%` と低く、単元株・日次予算・候補価格により資金利用が強く制約されている。
- `proba_confidence_weighted` はより保守的だが、qualityとbudget usageが低下した。

推奨:

- Phase 11-Dへ進む場合は、いきなりfull backtestではなく、strict limited scopeのCombined Backtest設計から始める。
- Phase 11-C単体の改善案として、budget usage改善のために候補価格、単元株、daily budget、top-N制約の感度監査を行う余地がある。

## Phase 11-C2 Implementation Status

実装済み:

- `src/ml/phase11c2_budget_usage_constraint_audit.py`
- `scripts/ml/audit_phase11c2_budget_usage_constraints.py`
- `tests/test_ml_phase11c2_budget_usage_constraint_audit.py`

生成report:

- `reports/ml/phase11c2_budget_usage_constraint_audit_2025.md`
- `reports/ml/phase11c2_budget_usage_constraint_audit_2025.json`

Scope:

- Phase 11-C simulation parquetを使用
- 必要な `close` / `turnover_value` はPhase 11-A datasetから参照
- 2025年のみのbudget usage constraint audit
- full backtestなし
- profile追加/変更なし
- 既存model上書きなし
- historical prediction再生成なし

実データ監査結果:

| item | value |
| --- | ---: |
| rows | `310,618` |
| unique_codes | `4,225` |
| candidate_days | `165` |
| date_range | `2025-01-07` to `2025-12-29` |
| leakage_risk | `low` |
| blocking_issues | `0` |
| main_budget_bottleneck | `round_lot_and_top_candidate_affordability_limit_daily_budget_usage` |
| ready_for_phase11d | `true` |

Constraint reason summary:

| reason | days |
| --- | ---: |
| `rank_filter_too_strict` | `128` |
| `top_candidates_too_expensive` | `37` |

Lot cost distribution:

| subset | lot_cost_median | lot_cost_p90 | affordable_under_300k | affordable_under_500k | affordable_under_900k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `top5` | `195,100` | `1,658,300` | `0.6145` | `0.6836` | `0.7964` |
| `top10` | `244,350` | `1,545,600` | `0.5727` | `0.6545` | `0.7933` |
| `p90+` | `169,400` | `764,820` | `0.6897` | `0.8192` | `0.9288` |
| `p95+` | `172,250` | `867,600` | `0.6721` | `0.7890` | `0.9080` |

Budget sensitivity:

| daily_buy_budget | budget_usage | allocated_rows | allocated/day | weighted_opportunity_value | weighted_top_decile_rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `300,000` | `0.2072` | `206` | `1.2485` | `0.0587` | `0.2403` |
| `500,000` | `0.2895` | `283` | `1.7152` | `0.0396` | `0.2083` |
| `900,000` | `0.4092` | `388` | `2.3515` | `0.0621` | `0.2403` |

Max positions sensitivity:

| max_positions | threshold | budget_usage | allocated/day | weighted_opportunity_value | weighted_top_decile_rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| `5` | `top5` | `0.2072` | `1.2485` | `0.0587` | `0.2403` |
| `10` | `top10` | `0.0863` | `1.1212` | `0.0390` | `0.2724` |

Candidate threshold sensitivity:

- `top5`, `top10`, `p90+`, `p80+` は、base条件ではいずれも同じallocated candidatesに収束した。
- max_positions `5` とdaily budget `300,000` が先に効くため、thresholdを緩めるだけではbudget usageは改善しなかった。

Interpretation:

- 主因は、同日上位候補の単元株コストがdaily budgetに対して大きく、top候補が高価格の日に買付余地が詰まること。
- 全候補には十分なaffordable候補があるが、Valuation上位候補に絞ると買える候補数が少なくなる。
- daily budgetを `900,000` に上げるとbudget usage proxyは `20.7%` から `40.9%` へ改善するが、Phase 11の最低目標 `60%` にはまだ届かない。
- max_positionsを増やすだけでは、300k budget下では改善しない。

推奨:

- `recommended_daily_budget`: `900,000`
- `recommended_max_positions`: `5`
- `recommended_candidate_threshold`: `top5`
- Phase 11-Dへ進む場合は、full backtestではなく、strict limited-scope designでbudget 900k案とaffordability fallback案を分けて検証する。
- Phase 11-C3を行うなら、top5が高価格で買えない日にだけp90+ affordable候補へfallbackするルールを検証する。

## Phase 11-D Implementation Status

実装済み:

- `src/ml/phase11d_combined_backtest.py`
- `scripts/ml/run_phase11d_combined_backtest.py`
- `tests/test_ml_phase11d_combined_backtest.py`

生成report:

- `reports/ml/phase11d_combined_backtest_2025.md`
- `reports/ml/phase11d_combined_backtest_2025.json`

Scope:

- Phase 11-C simulation parquetを使用
- 必要な `close` / baseline rank scoreはPhase 11-A datasetから参照
- 2025年entryのみのlimited combined backtest
- 比較対象は2本のみ
- full period backtestなし
- profile追加/変更なし
- 既存model上書きなし
- historical prediction再生成なし

Backtest条件:

| item | value |
| --- | ---: |
| initial_cash | `1,000,000` |
| daily_buy_budget | `900,000` |
| max_positions | `5` |
| round_lot | `100` |
| holding_days | `20` |

比較対象:

| strategy | rank input | allocation |
| --- | --- | --- |
| `baseline_equal_allocation` | `stock_selection_rank_score` | equal allocation |
| `candidate_valuation_top5` | `opportunity_top_decile_proba` | `equal_weight_top5` |

実データ結果:

| strategy | net_profit | PF | DD | win_rate | total_trades | final_assets | capital_utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_equal_allocation` | `88,578` | `1.5829` | `-5.33%` | `50.91%` | `55` | `1,088,578` | `50.07%` |
| `candidate_valuation_top5` | `187,018` | `1.6990` | `-16.83%` | `60.34%` | `58` | `1,187,018` | `49.89%` |

BUY quality:

| strategy | future_return_20d | future_max_return_20d | future_max_drawdown_20d | opportunity_value_20d | top_decile_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_equal_allocation` | `0.0104` | `0.0691` | `-0.0501` | `0.0190` | `0.0727` |
| `candidate_valuation_top5` | `0.0314` | `0.1520` | `-0.0889` | `0.0631` | `0.2931` |

Valuation effect:

| metric | delta |
| --- | ---: |
| net_profit | `+98,440` |
| PF | `+0.1161` |
| DD | `-11.50%` |
| capital_utilization | `-0.18%pt` |
| opportunity_value_20d_mean | `+0.0441` |
| future_return_20d_mean | `+0.0210` |

Interpretation:

- Valuation接続により、2025年限定では利益、PF、勝率、BUY qualityが改善した。
- `opportunity_top_decile_20d` rateは `7.27%` から `29.31%` へ改善し、候補品質改善は確認できた。
- 一方で、`future_max_drawdown_20d` 平均とBacktest DDは悪化している。
- Phase 11-Dの目的である「Valuationが候補品質を改善するか」はyes。
- ただし、Phase 11-EはExit/Risk guard付きの限定検証にする必要がある。

推奨:

- `ready_for_phase11e`: `true`
- `dd_worsened`: `true`
- `recommended_next_phase`: `Phase11-E limited exit integration with DD guard`
- Phase 11-Eではfull backtestへ広げず、2025年限定でOpportunity消滅/expected downside/簡易DD guardのどれがDD悪化を抑えるかを先に検証する。

## Phase 11-E Implementation Status

実装済み:

- `src/ml/phase11e_exit_dd_guard.py`
- `scripts/ml/run_phase11e_exit_dd_guard.py`
- `tests/test_ml_phase11e_exit_dd_guard.py`

生成report:

- `reports/ml/phase11e_exit_dd_guard_2025.md`
- `reports/ml/phase11e_exit_dd_guard_2025.json`

Scope:

- Phase 11-D Candidateをbaseline referenceとして使用
- Phase 11-C simulation parquetとPhase 11-A closeを使用
- 2025年のみのExit / DD Guard軽量backtest
- full period backtestなし
- profile追加/変更なし
- 既存model上書きなし
- historical prediction再生成なし
- stop loss判定にfuture lowを使用しない

Variant:

| variant | definition |
| --- | --- |
| `E0_no_guard` | Phase 11-D Candidate相当のpath-based no guard |
| `E1_stop_loss_8pct` | entry closeから `-8%` で売却 |
| `E2_stop_loss_5pct` | entry closeから `-5%` で売却 |
| `E3_opportunity_disappeared` | current probaがentry probaから `0.15` 以上低下、またはcurrent rank `< 0.50` で売却 |
| `E4_stop_loss_8pct_plus_opportunity` | E1 + E3 |

Phase 11-D Candidate reference:

| net_profit | PF | DD | win_rate | trades | final_assets | utilization |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `187,018` | `1.6990` | `-16.83%` | `60.34%` | `58` | `1,187,018` | `49.89%` |

実データ結果:

| variant | net_profit | PF | DD | win_rate | trades | avg_holding_days |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `E0_no_guard` | `181,730` | `1.6869` | `-23.43%` | `58.62%` | `58` | `19.41` |
| `E1_stop_loss_8pct` | `82,540` | `1.1993` | `-22.94%` | `47.83%` | `69` | `15.96` |
| `E2_stop_loss_5pct` | `49,110` | `1.1185` | `-26.94%` | `42.11%` | `76` | `14.28` |
| `E3_opportunity_disappeared` | `529,880` | `2.8160` | `-7.85%` | `54.74%` | `137` | `7.16` |
| `E4_stop_loss_8pct_plus_opportunity` | `615,110` | `2.6219` | `-6.02%` | `52.26%` | `155` | `6.26` |

Delta vs Phase 11-D Candidate:

| variant | net_profit_delta | PF_delta | DD_delta | win_rate_delta |
| --- | ---: | ---: | ---: | ---: |
| `E1_stop_loss_8pct` | `-104,478` | `-0.4997` | `-6.11%pt` | `-12.52%pt` |
| `E2_stop_loss_5pct` | `-137,908` | `-0.5805` | `-10.11%pt` | `-18.24%pt` |
| `E3_opportunity_disappeared` | `+342,862` | `+1.1169` | `+8.98%pt` | `-5.60%pt` |
| `E4_stop_loss_8pct_plus_opportunity` | `+428,092` | `+0.9229` | `+10.81%pt` | `-8.09%pt` |

Exit reason counts:

| variant | time_exit_20d | stop_loss | opportunity_rank_below_floor | opportunity_proba_drop | forced_end |
| --- | ---: | ---: | ---: | ---: | ---: |
| `E1_stop_loss_8pct` | `39` | `25` | `0` | `0` | `5` |
| `E2_stop_loss_5pct` | `36` | `35` | `0` | `0` | `5` |
| `E3_opportunity_disappeared` | `24` | `0` | `9` | `99` | `5` |
| `E4_stop_loss_8pct_plus_opportunity` | `16` | `15` | `8` | `111` | `5` |

Interpretation:

- Simple Stop Loss単体のE1/E2は、利益、PF、DDを悪化させた。
- Opportunity Disappeared ExitのE3は、DDを `-7.85%` まで改善し、PFと利益も大きく改善した。
- E4はDDが最良の `-6.02%` で、Phase 11-Eの研究判定条件 `DD >= -10%`, `PF >= 1.5`, `net_profit >= 100,000` を満たした。
- E3/E4は平均保有日数が大きく短くなり、取引数も増えるため、次は手数料・スリッページ・過剰回転のrobustness checkが必要。

推奨:

- `best_variant`: `E4_stop_loss_8pct_plus_opportunity`
- `dd_improved_variant_found`: `true`
- `recommended_next_phase`: `Phase11-F limited robustness check`
- Phase 11-Fではfull periodへ広げず、2025年限定のままE3/E4について手数料、スリッページ、exit閾値感度、過剰回転を確認する。

## Phase 11-F Implementation Status

実装済み:

- `src/ml/phase11f_robustness_check.py`
- `scripts/ml/run_phase11f_robustness_check.py`
- `tests/test_ml_phase11f_robustness_check.py`

生成report:

- `reports/ml/phase11f_robustness_check_2025.md`
- `reports/ml/phase11f_robustness_check_2025.json`

Scope:

- Phase 11-E E4をbase strategyとして検証
- 2025年のみのlimited robustness check
- full period backtestなし
- profile追加/変更なし
- 既存model上書きなし
- historical prediction再生成なし
- future系は評価指標のみ

Cost sensitivity:

| one-way cost | net_profit | PF | DD | win_rate | trades | avg_holding_days | cost_paid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0%` | `615,110` | `2.6219` | `-6.02%` | `52.26%` | `155` | `6.26` | `0` |
| `0.1%` | `568,935` | `2.4304` | `-6.59%` | `51.61%` | `155` | `6.26` | `43,075` |
| `0.2%` | `473,578` | `2.0551` | `-6.36%` | `50.96%` | `157` | `6.21` | `85,882` |
| `0.3%` | `394,497` | `1.8790` | `-8.50%` | `50.00%` | `158` | `6.07` | `128,063` |

Opportunity Exit threshold sensitivity:

| threshold | drop | rank_floor | net_profit | PF | DD | trades | avg_holding_days |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `loose` | `0.25` | `0.40` | `355,440` | `2.0448` | `-9.45%` | `112` | `9.32` |
| `baseline` | `0.15` | `0.50` | `615,110` | `2.6219` | `-6.02%` | `155` | `6.26` |
| `strict` | `0.08` | `0.60` | `496,270` | `2.0189` | `-9.14%` | `229` | `3.91` |

Overtrading check:

| item | value |
| --- | ---: |
| average_holding_days | `6.26` |
| median_holding_days | `4.00` |
| same_code_reentry_count | `115` |
| reentry_within_5_days_count | `88` |
| opportunity_proba_drop exits | `111` |
| stop_loss exits | `15` |

Monthly notes:

- Negative months: January `-30,120`, November `-53,100`.
- December trade count is high at `25`, indicating possible turnover concentration.
- Baseline threshold is strongest on 2025 metrics, but strict threshold increases trades to `229` and average holding falls below `5` days.

Robustness判定:

| check | result |
| --- | --- |
| PF >= `2.0` | `true` |
| DD >= `-10%` | `true` |
| net_profit >= `300,000` | `true` |
| average_holding_days >= `5` | `true` |
| cost `0.2%` PF >= `1.8` | `true` |
| cost `0.2%` DD >= `-10%` | `true` |
| all_passed | `true` |

Interpretation:

- E4は2025年限定では片道 `0.2%` コストに耐え、片道 `0.3%` でもDDは `-10%` 以内に残った。
- Opportunity Exit閾値はbaselineが最も強く、loose/strictとも成立はするが利益が低下する。
- 過剰回転リスクは残る。特に同一銘柄再エントリーと5営業日以内の再エントリーが多い。

推奨:

- `robustness_passed`: `true`
- `recommended_next_phase`: `Phase11-G limited out-of-sample year check`
- Phase 11-Gではfull periodへは広げず、まず2024年または2026年一部などの限定out-of-sample year checkを行う。
- 併せて、same-code reentry cooldownとminimum holding guardの軽量感度を見る。

## Phase 11-G Implementation Status

実装済み:

- `src/ml/phase11g_out_of_sample_check.py`
- `scripts/ml/run_phase11g_out_of_sample_check.py`
- `tests/test_ml_phase11g_out_of_sample_check.py`

生成report:

- `reports/ml/phase11g_out_of_sample_check_2024.md`
- `reports/ml/phase11g_out_of_sample_check_2024.json`

Scope:

- 2024年のみのlimited year check
- 比較対象は4本のみ
- full period backtestなし
- profile追加/変更なし
- 既存model上書きなし
- historical prediction保存/再生成なし
- future系は評価指標のみ

重要な制約:

- Phase 11-B candidate modelのtrain periodは `2023-01-04` to `2024-12-31`。
- そのため2024年はmodel training periodと重なっている。
- Phase 11-Gはstrategy/pathの独立年確認として有用だが、strict model OOS proofではない。

2024 dataset:

| item | value |
| --- | ---: |
| rows | `262,224` |
| unique_codes | `1,586` |
| candidate_days | `166` |
| date_range | `2024-01-04` to `2024-12-27` |

Strategy comparison:

| strategy | net_profit | PF | DD | win_rate | trades | avg_holding | reentry_5d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_equal_allocation` | `156,650` | `2.2226` | `-9.12%` | `51.79%` | `56` | `19.66` | `10` |
| `valuation_top5_no_guard` | `668,360` | `3.3827` | `-14.85%` | `68.85%` | `61` | `19.30` | `41` |
| `valuation_top5_E4` | `699,520` | `2.7918` | `-8.25%` | `56.40%` | `172` | `5.18` | `111` |
| `valuation_top5_E4_cost_0.2pct` | `574,984` | `2.3421` | `-9.11%` | `52.98%` | `168` | `5.23` | `107` |

BUY quality:

| strategy | future_return_20d | future_max_return_20d | future_max_drawdown_20d | opportunity_value_20d | top_decile_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_equal_allocation` | `0.0225` | `0.0867` | `-0.0548` | `0.0319` | `0.1429` |
| `valuation_top5_no_guard` | `0.0889` | `0.2425` | `-0.0849` | `0.1576` | `0.5902` |
| `valuation_top5_E4` | `0.1096` | `0.2623` | `-0.0762` | `0.1861` | `0.5814` |
| `valuation_top5_E4_cost_0.2pct` | `0.1112` | `0.2637` | `-0.0767` | `0.1871` | `0.5774` |

OOS year check judgement:

| check | result |
| --- | --- |
| E4 beats baseline net profit | `true` |
| E4 PF >= `1.5` | `true` |
| E4 DD >= `-10%` | `true` |
| cost `0.2%` PF >= `1.3` | `true` |
| cost `0.2%` DD >= `-12%` | `true` |
| all_passed | `true` |

Interpretation:

- 2024年でもValuation top5はbaselineより大きく高いBUY qualityを示した。
- no guardは利益/PFが強い一方でDDが `-14.85%` まで悪化した。
- E4はDDを `-8.25%` へ抑え、cost `0.2%` でも `-9.11%` に収まった。
- ただしE4は2024年でも再エントリーが多い。5営業日以内reentryは `111`、cost `0.2%` でも `107`。
- strict model OOSではないため、正式な汎化確認にはwalk-forward設計が必要。

推奨:

- `oos_passed`: `true`
- `strict_model_oos`: `false`
- `recommended_next_phase`: `Phase11-H cooldown/min-hold guard plus strict walk-forward OOS design`
- Phase 11-Hではsame-code cooldown / minimum holding guardを限定検証し、その後に2024をtrain外にするstrict walk-forward OOS設計を行う。
