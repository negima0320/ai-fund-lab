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
