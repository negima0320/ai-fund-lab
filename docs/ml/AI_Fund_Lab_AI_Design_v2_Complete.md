# AI Fund Lab AI Design v2 (Complete Edition)

## 1. プロジェクト目的

### 現状

既存のルールベース戦略

- rookie_dealer_02_v2_26
- rookie_dealer_02_v2_65

を運用中。

### AI導入目的

AIで売買判断を置き換えることではない。

目的は以下。

- 高値掴み回避
- 期待上昇率予測
- 候補銘柄順位付け

---

## 2. 現在の分析結果

### v2.26

- net_profit: 86,472円
- PF: 1.167
- win_rate: 41.4%
- max_drawdown: -11.8%

### v2.65

- net_profit: 94,432円
- PF: 1.212
- win_rate: 47.3%
- max_drawdown: -11.4%

現時点では v2.65 が優勢。

---

## 3. 判明した重要事項

### スコア帯分析

45-49帯

- PF 1.88
- 利益 +113,808円

50-54帯

- PF 1.23
- 利益 +62,992円

55-59帯

- PF 0.69
- 利益 -61,375円

60-64帯

- PF 1.16
- 利益 +4,639円

### 仮説

高スコア = 強い

ではなく

高スコア = 過熱

の可能性が高い。

---

## 4. AIアーキテクチャ

既存J-Quants取得境界
`JQuantsDataProvider` / `JQuantsDataService`
↓
data/cache/jquants
↓
特徴量生成
↓
data/ml/features
↓
ラベル生成
↓
data/ml/labels
↓
学習データ
↓
data/ml/datasets
↓
LightGBM
↓
models/ml/current
↓
日次予測
↓
data/ml/predictions

---

## 5. J-Quants API一覧

API 呼び出し、保存先、ファイル命名規則の正は `docs/jquants_data_fetching.md` とする。  
AI / ML 側は J-Quants の URL、認証、ページネーション、保存処理を再実装しない。必要な取得は `src/data_provider.py` の `JQuantsDataService` または既存 CLI 取得処理に委譲する。

### 上場銘柄一覧

API:

/equities/master

保存:

- data/raw/listed_stocks_jquants.json
- data/raw/prime_stocks_jquants.json

キャッシュ:

- data/cache/jquants/listed_info/*.json

### 株価四本値

API:

/equities/bars/daily

保存:

- data/raw/prices_YYYY-MM-DD.json

キャッシュ:

- data/cache/jquants/prices/*.json

### 財務情報

API:

/fins/summary

保存:

- data/cache/jquants/financial_statements/*.json

### 決算予定

API:

/equities/earnings-calendar

保存:

- data/cache/jquants/earnings_calendar/*.json

### 取引カレンダー

API:

/markets/calendar

保存:

- data/cache/jquants/trading_calendar/*.json

### 投資部門別

API:

/equities/investor-types

保存:

- data/cache/jquants/investor_types/*.json

### TOPIX

API:

/indices/bars/daily/topix

保存:

- data/cache/jquants/topix_prices/*.json

---

## 6. 学習対象

対象:

- 全銘柄

理由:

- 将来ルール変更しても利用可能
- AIを汎用モデル化できる

将来的に流動性フィルタは検討。

---

## 7. モデル構成

### Model A

future_5d_return

回帰

### Model B

future_10d_return

回帰

### Model C

upside_10d

分類

### Model D

bad_entry_10d

分類

---

## 8. ラベル設計

### future_5d_return

5営業日後終値リターン

### future_10d_return

10営業日後終値リターン

### upside_10d

10営業日以内に +5%以上上昇

### bad_entry_10d

購入後10営業日以内に -5%以上含み損

entry_price = 翌営業日始値

max_adverse_excursion_10d =
future_min_low_10d / entry_price - 1

bad_entry_10d =
max_adverse_excursion_10d <= -0.05

---

## 9. 特徴量 v1

### 価格

- close
- return_1d
- return_3d
- return_5d
- return_10d
- return_20d

### 移動平均

- ma5_gap
- ma10_gap
- ma25_gap
- ma75_gap
- ma5_slope
- ma25_slope

### 出来高

- volume_ratio_5d
- volume_ratio_20d
- turnover_ratio_5d
- turnover_ratio_20d
- turnover_value

### ローソク足

- body_ratio
- upper_shadow_ratio
- lower_shadow_ratio
- close_position
- gap_up_ratio
- daily_range_ratio

### テクニカル

- RSI_14
- ATR_14
- MACD
- MACD_signal
- MACD_hist
- bollinger_position_20
- bollinger_bandwidth_20

### TOPIX

- topix_return_5d
- topix_return_10d
- topix_return_20d
- relative_return_5d
- relative_return_10d
- relative_return_20d

### 決算

- days_to_earnings
- days_after_earnings
- is_near_earnings

### 投資部門別

- overseas_net_buy_1w
- individual_net_buy_1w
- institution_net_buy_1w
- proprietary_net_buy_1w
- overseas_net_buy_4w_sum
- individual_net_buy_4w_sum

### 銘柄属性

- market
- sector_name
- scale_category

---

## 10. AI出力

```json
{
  "expected_return_5d": 0.018,
  "expected_return_10d": 0.041,
  "upside_probability_10d": 0.62,
  "bad_entry_probability_10d": 0.22
}
```

---

## 11. 売買判定案

- expected_return_10d >= 3%
- upside_probability_10d >= 55%
- bad_entry_probability_10d <= 30%

---

## 12. 日次パイプライン

### 毎日

1. ensure_jquants_cache
2. build_features
3. predict_today
4. update_labels
5. append_dataset

### 週1回

1. train_models
2. evaluate_models
3. deploy_best_model

---

## 13. ディレクトリ構成

data/ml/

- features/
- labels/
- datasets/
- predictions/

models/ml/

- current/
- archive/

reports/ml/

---

## 14. リポジトリ方針

現段階:

ai-fund-lab

内部に

- backtest
- trading
- ml

を配置

推奨。

AI専用リポジトリは将来分離。

---

## 15. 今後の実装ロードマップ

Phase 1

- 特徴量生成基盤
- ラベル生成基盤
- データセット生成

Phase 2

- LightGBM 4モデル学習
- 評価レポート

Phase 3

- 日次予測
- 売買候補への統合

Phase 4

- 自動再学習
- モデル監視
