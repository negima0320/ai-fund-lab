# AI Fund Lab ML 実装仕様書 v1

## 1. 目的

AI Fund Lab に、J-Quants の生データから機械学習用データセットを生成し、LightGBM で以下4モデルを学習・予測できる ML 基盤を追加する。

- future_5d_return 回帰モデル
- future_10d_return 回帰モデル
- upside_10d 分類モデル
- bad_entry_10d 分類モデル

AI は既存の売買ロジックを置き換えない。  
目的は、既存候補または全銘柄に対して以下を補助判断すること。

- 上がるか
- 下がるか
- 高値掴みか
- 期待値があるか

---

## 2. 基本方針

### OpenAI API は使わない

理由:

- バックテスト・日次運用で大量課金になる
- 再現性が落ちる
- テスト回数が制限される

### ローカル実行

Mac ローカルで完結する。

想定ライブラリ:

- pandas
- numpy
- scikit-learn
- lightgbm
- joblib
- pyarrow

---

## 3. 実装対象ディレクトリ

既存リポジトリ内に `ml` または `src/ml` を追加する。

推奨:

```text
src/ml/
├── __init__.py
├── config.py
├── data_loader.py
├── feature_builder.py
├── label_generator.py
├── dataset_builder.py
├── model_trainer.py
├── predictor.py
├── evaluator.py
└── pipeline.py

scripts/ml/
├── build_features.py
├── update_labels.py
├── build_dataset.py
├── train_models.py
├── predict_daily.py
└── daily_pipeline.py
```

出力先:

```text
data/ml/
├── features/
├── labels/
├── datasets/
├── predictions/
└── metadata/

models/ml/
├── current/
└── archive/

reports/ml/
```

---

## 4. 入力データ

J-Quants 生データを利用する。  
`data/processed/` は使用しない。

J-Quants の API 呼び出しと保存処理は既存の共通境界を利用する。

- HTTP/API 境界: `src/data_provider.py` の `JQuantsDataProvider`
- fetch/cache 境界: `src/data_provider.py` の `JQuantsDataService`
- API 一覧と保存先の正: `docs/jquants_data_fetching.md`

ML 側は原則として `data/cache/jquants/` と `data/raw/` の既存データを読み込む。キャッシュが不足している場合も、ML 側で J-Quants の URL や保存処理を再実装せず、`JQuantsDataService` または既存 CLI 取得処理を呼び出す。

### 4.1 上場銘柄一覧

API:

```text
/equities/master
```

保存先:

```text
data/raw/listed_stocks_jquants.json
data/raw/prime_stocks_jquants.json
data/cache/jquants/listed_info/*.json
```

利用項目:

- code
- name
- market
- market_section
- sector_name
- scale_category

### 4.2 株価四本値

API:

```text
/equities/bars/daily
```

保存先:

```text
data/raw/prices_YYYY-MM-DD.json
data/cache/jquants/prices/*.json
```

利用項目:

- date
- code
- open
- high
- low
- close
- volume
- turnover_value

### 4.3 財務情報

API:

```text
/fins/summary
```

保存先:

```text
data/cache/jquants/financial_statements/*.json
```

v1では学習特徴量から除外。  
v2以降で追加予定。

### 4.4 決算予定

API:

```text
/equities/earnings-calendar
```

保存先:

```text
data/cache/jquants/earnings_calendar/*.json
```

利用項目:

- Date
- Code
- FQ
- Section

### 4.5 取引カレンダー

API:

```text
/markets/calendar
```

保存先:

```text
data/cache/jquants/trading_calendar/*.json
```

利用項目:

- Date
- HolDiv

`HolDiv == "1"` を営業日として扱う。

### 4.6 投資部門別情報

API:

```text
/equities/investor-types
```

保存先:

```text
data/cache/jquants/investor_types/*.json
```

利用項目:

- PubDate
- StDate
- EnDate
- Section
- FrgnBal
- IndBal
- BrkBal
- PropBal
- InvTrBal
- TrstBnkBal

注意:

- 週次データ
- 公表遅れあり
- 必ず `PubDate` 以降の日付にだけ結合する

### 4.7 TOPIX四本値

API:

```text
/indices/bars/daily/topix
```

保存先:

```text
data/cache/jquants/topix_prices/*.json
```

利用項目:

- date
- open
- high
- low
- close

---

## 5. 主要クラス設計

## 5.1 MLConfig

ファイル:

```text
src/ml/config.py
```

責務:

- 入出力パス管理
- ラベル閾値管理
- 特徴量一覧管理
- モデル名管理

主な設定:

```python
FEATURE_WINDOWS = [1, 3, 5, 10, 20, 25, 75]
LABEL_HORIZONS = [5, 10]
UPSIDE_THRESHOLD = 0.05
BAD_ENTRY_THRESHOLD = -0.05
NEAR_EARNINGS_DAYS = 5
```

---

## 5.2 JQuantsDataLoader

ファイル:

```text
src/ml/data_loader.py
```

責務:

J-Quants 生キャッシュを読み込む。
API 取得が必要な場合は `JQuantsDataService` に委譲する。

主なメソッド:

```python
load_prices(start_date, end_date) -> pd.DataFrame
load_listed_info(as_of_date) -> pd.DataFrame
load_topix(start_date, end_date) -> pd.DataFrame
load_earnings_calendar(start_date, end_date) -> pd.DataFrame
load_trading_calendar(start_date, end_date) -> pd.DataFrame
load_investor_types(start_date, end_date) -> pd.DataFrame
```

出力 DataFrame の基本列:

```text
date
code
open
high
low
close
volume
turnover_value
```

注意:

- code は文字列で扱う
- date は datetime64 に正規化
- JSONのキー表記ゆれを吸収する
  - O/open
  - H/high
  - L/low
  - C/close
  - Vo/volume
  - Va/turnover_value

---

## 5.3 FeatureBuilder

ファイル:

```text
src/ml/feature_builder.py
```

責務:

指定日または期間に対して、全銘柄の特徴量を生成する。

主なメソッド:

```python
build_daily_features(target_date: str) -> pd.DataFrame
build_features(start_date: str, end_date: str) -> pd.DataFrame
```

入力:

- prices
- listed_info
- topix
- earnings_calendar
- investor_types
- trading_calendar

出力:

```text
data/ml/features/features_YYYY-MM-DD.parquet
```

必須列:

```text
date
code
```

### 5.3.1 価格特徴量

```text
close
return_1d
return_3d
return_5d
return_10d
return_20d
```

計算:

```text
return_Nd = close_today / close_N営業日前 - 1
```

### 5.3.2 移動平均特徴量

```text
ma5_gap
ma10_gap
ma25_gap
ma75_gap
ma5_slope
ma25_slope
```

計算:

```text
maN = close.rolling(N).mean()
maN_gap = close / maN - 1
maN_slope = maN / maN_5営業日前 - 1
```

### 5.3.3 出来高特徴量

```text
volume_ratio_5d
volume_ratio_20d
turnover_ratio_5d
turnover_ratio_20d
turnover_value
```

計算:

```text
volume_ratio_Nd = volume_today / average_volume_Nd
turnover_ratio_Nd = turnover_today / average_turnover_Nd
```

### 5.3.4 ローソク足特徴量

```text
body_ratio
upper_shadow_ratio
lower_shadow_ratio
close_position
gap_up_ratio
daily_range_ratio
```

計算:

```text
range = high - low

body_ratio = abs(close - open) / range
upper_shadow_ratio = (high - max(open, close)) / range
lower_shadow_ratio = (min(open, close) - low) / range
close_position = (close - low) / range
gap_up_ratio = open / prev_close - 1
daily_range_ratio = range / close
```

range が 0 の場合は 0 または NaN にし、後段で欠損処理する。

### 5.3.5 テクニカル指標

```text
rsi_14
atr_14
macd
macd_signal
macd_hist
bollinger_position_20
bollinger_bandwidth_20
```

計算方針:

- RSI: 14営業日
- ATR: 14営業日
- MACD: EMA12 - EMA26
- signal: EMA9
- Bollinger: 20営業日、±2σ

```text
bollinger_position_20 =
(close - lower_band) / (upper_band - lower_band)

bollinger_bandwidth_20 =
(upper_band - lower_band) / ma20
```

### 5.3.6 TOPIX・相対強度

```text
topix_return_5d
topix_return_10d
topix_return_20d
relative_return_5d
relative_return_10d
relative_return_20d
```

計算:

```text
relative_return_Nd = stock_return_Nd - topix_return_Nd
```

### 5.3.7 決算特徴量

```text
days_to_earnings
days_after_earnings
is_near_earnings
```

計算:

```text
days_to_earnings = 次回決算予定日 - target_date
days_after_earnings = target_date - 直近決算予定日
is_near_earnings = abs(days_to_earnings) <= 5
```

### 5.3.8 投資部門別特徴量

```text
overseas_net_buy_1w
individual_net_buy_1w
institution_net_buy_1w
proprietary_net_buy_1w
overseas_net_buy_4w_sum
individual_net_buy_4w_sum
```

結合ルール:

- `PubDate <= target_date` のデータだけ利用
- 銘柄の `market_section` に対応する `Section` を結合
- TSEPrime / TSEStandard / TSEGrowth を優先
- 該当がない場合は TokyoNagoya を利用

### 5.3.9 銘柄属性

```text
market
sector_name
scale_category
```

カテゴリ特徴量として扱う。

LightGBMではカテゴリ列として渡すか、One-Hot / category dtype に変換する。

---

## 5.4 LabelGenerator

ファイル:

```text
src/ml/label_generator.py
```

責務:

ラベル確定可能な過去日に対して教師ラベルを生成する。

主なメソッド:

```python
generate_labels(target_date: str) -> pd.DataFrame
update_available_labels(as_of_date: str) -> list[Path]
```

出力:

```text
data/ml/labels/labels_YYYY-MM-DD.parquet
```

必須列:

```text
date
code
future_5d_return
future_10d_return
upside_10d
bad_entry_10d
```

### 5.4.1 entry_price

実運用を想定し、当日終値ではなく翌営業日始値を購入価格とする。

```text
entry_price = next_business_day.open
```

### 5.4.2 future_5d_return

```text
future_5d_return = close_5営業日後 / entry_price - 1
```

### 5.4.3 future_10d_return

```text
future_10d_return = close_10営業日後 / entry_price - 1
```

### 5.4.4 upside_10d

```text
future_max_high_10d = 購入後10営業日の最高値

upside_10d =
future_max_high_10d / entry_price - 1 >= 0.05
```

### 5.4.5 bad_entry_10d

```text
future_min_low_10d = 購入後10営業日の最安値

bad_entry_10d =
future_min_low_10d / entry_price - 1 <= -0.05
```

注意:

- 未来データ混入防止のため、ラベル生成は対象日から10営業日以上経過した日だけ行う
- 株式分割など調整済み価格が利用可能なら、将来的に調整後価格へ切り替える

---

## 5.5 DatasetBuilder

ファイル:

```text
src/ml/dataset_builder.py
```

責務:

特徴量とラベルを結合し、学習可能なデータセットを作る。

主なメソッド:

```python
build_dataset(start_date: str, end_date: str) -> pd.DataFrame
save_dataset(df: pd.DataFrame, name: str) -> Path
```

出力:

```text
data/ml/datasets/ml_dataset.parquet
data/ml/datasets/train.parquet
data/ml/datasets/valid.parquet
data/ml/datasets/test.parquet
```

結合キー:

```text
date
code
```

除外条件:

- ラベル未確定
- close が欠損
- entry_price が欠損
- 出来高ゼロ
- 極端な異常値

将来的な流動性フィルタ候補:

```text
turnover_value >= 50,000,000
```

v1では設定だけ用意し、デフォルトOFFでもよい。

---

## 5.6 ModelTrainer

ファイル:

```text
src/ml/model_trainer.py
```

責務:

4モデルを学習する。

主なメソッド:

```python
train_all(train_df, valid_df) -> dict
train_regression(target_col, train_df, valid_df)
train_classification(target_col, train_df, valid_df)
save_models(models, metrics) -> Path
```

対象モデル:

```text
future_5d_return
future_10d_return
upside_10d
bad_entry_10d
```

保存先:

```text
models/ml/archive/YYYYMMDD_HHMMSS/
models/ml/current/
```

保存ファイル:

```text
future_5d_return.pkl
future_10d_return.pkl
upside_10d.pkl
bad_entry_10d.pkl
feature_columns.json
categorical_columns.json
metrics.json
```

### 5.6.1 LightGBMパラメータ初期案

回帰:

```python
{
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "verbose": -1
}
```

分類:

```python
{
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "verbose": -1
}
```

---

## 5.7 Predictor

ファイル:

```text
src/ml/predictor.py
```

責務:

当日特徴量を読み込み、4モデルで予測する。

主なメソッド:

```python
predict_daily(target_date: str) -> pd.DataFrame
load_current_models()
save_predictions(df, target_date) -> Path
```

出力:

```text
data/ml/predictions/predictions_YYYY-MM-DD.parquet
```

出力列:

```text
date
code
expected_return_5d
expected_return_10d
upside_probability_10d
bad_entry_probability_10d
entry_risk_label
ml_score
```

### 5.7.1 entry_risk_label

```text
safe:
bad_entry_probability_10d < 0.25

watch:
0.25 <= bad_entry_probability_10d < 0.40

danger:
bad_entry_probability_10d >= 0.40
```

### 5.7.2 ml_score 初期案

```text
ml_score =
expected_return_10d * 100
+ upside_probability_10d * 10
- bad_entry_probability_10d * 15
```

v1では仮スコア。  
バックテストで調整する。

---

## 5.8 Evaluator

ファイル:

```text
src/ml/evaluator.py
```

責務:

モデル性能と売買上の有効性を評価する。

主なメソッド:

```python
evaluate_regression(y_true, y_pred)
evaluate_classification(y_true, y_pred_proba)
evaluate_strategy(predictions, labels)
generate_report(metrics, output_path)
```

評価指標:

回帰:

- RMSE
- MAE
- Spearman相関
- 予測上位銘柄の平均future_return

分類:

- AUC
- Precision
- Recall
- F1
- bad_entry回避率

売買観点:

- 予測上位N件の平均リターン
- bad_entry_probability帯別の損益
- upside_probability帯別の損益
- expected_return帯別の損益

出力:

```text
reports/ml/model_eval_YYYY-MM-DD.md
```

---

## 5.9 Pipeline

ファイル:

```text
src/ml/pipeline.py
```

責務:

日次処理と学習処理をまとめる。

主なメソッド:

```python
run_daily_pipeline(target_date: str)
run_weekly_training(end_date: str)
```

日次処理:

```text
1. build_features
2. predict_daily
3. update_labels
4. append_dataset
```

注意:

J-Quants の取得処理は `JQuantsDataProvider` / `JQuantsDataService` を呼び出す。  
ML 側では原則としてキャッシュ済みデータを読むだけにし、API path、認証、ページネーション、保存処理を再実装しない。

---

## 6. CLI設計

### 6.1 特徴量生成

```bash
python scripts/ml/build_features.py --date 2026-06-01
```

### 6.2 ラベル更新

```bash
python scripts/ml/update_labels.py --as-of 2026-06-01
```

### 6.3 データセット生成

```bash
python scripts/ml/build_dataset.py \
  --start 2021-06-01 \
  --end 2026-05-31
```

### 6.4 モデル学習

```bash
python scripts/ml/train_models.py \
  --train-start 2021-06-01 \
  --train-end 2025-12-31 \
  --valid-start 2026-01-01 \
  --valid-end 2026-03-31 \
  --test-start 2026-04-01 \
  --test-end 2026-05-31
```

### 6.5 日次予測

```bash
python scripts/ml/predict_daily.py --date 2026-06-01
```

### 6.6 日次パイプライン

```bash
python scripts/ml/daily_pipeline.py --date 2026-06-01
```

---

## 7. 時系列分割

初期案:

```text
train:
2021-06-01 ～ 2025-12-31

valid:
2026-01-01 ～ 2026-03-31

test:
2026-04-01 ～ 2026-05-31
```

注意:

- ランダム分割は禁止
- 必ず時系列で分割
- future_10d_return が確定していない期間は除外

---

## 8. 未来データ混入防止ルール

### 8.1 株価

特徴量生成では target_date 以前の株価だけを使う。

### 8.2 ラベル

target_date の特徴量に対して、未来の価格を使うのはラベル生成時のみ。

### 8.3 財務情報

v1では未使用。  
v2以降で使う場合は `DiscDate` と `DiscTime` を基準にする。

### 8.4 投資部門別情報

`PubDate <= target_date` のレコードのみ使用する。

### 8.5 決算予定

取得時点で公表済みの予定として扱う。  
将来的に厳密化する場合は取得日・公表日を別管理する。

---

## 9. 既存バックテストとの接続

v1では直接売買ロジックに組み込まない。

まずは以下の形で予測結果を保存する。

```text
data/ml/predictions/predictions_YYYY-MM-DD.parquet
```

既存バックテスト側では、候補銘柄に対して `date + code` で ML 予測を join する。

join後に使う列:

```text
expected_return_5d
expected_return_10d
upside_probability_10d
bad_entry_probability_10d
entry_risk_label
ml_score
```

初期利用案:

```text
bad_entry_probability_10d >= 0.40 は除外
```

次の段階:

```text
expected_return_10d >= 0.03
upside_probability_10d >= 0.55
bad_entry_probability_10d <= 0.30
```

---

## 10. 実装順序

### Step 1

MLディレクトリと設定ファイルを追加する。

- src/ml/config.py
- scripts/ml/

### Step 2

DataLoaderを実装する。

- prices
- listed_info
- topix
- earnings
- trading_calendar
- investor_types

### Step 3

FeatureBuilderを実装する。

まずは以下だけでよい。

- 価格
- 移動平均
- 出来高
- ローソク足
- RSI
- TOPIX相対強度

### Step 4

LabelGeneratorを実装する。

- future_5d_return
- future_10d_return
- upside_10d
- bad_entry_10d

### Step 5

DatasetBuilderを実装する。

features + labels を結合して parquet 保存。

### Step 6

ModelTrainerを実装する。

LightGBM 4モデルを保存。

### Step 7

Predictorを実装する。

当日特徴量から予測 parquet を出力。

### Step 8

Evaluatorを実装する。

精度レポートを markdown 出力。

### Step 9

既存バックテストに join する。

まずは除外ルールのみ。

---

## 11. Codexへの実装指示で注意すること

- OpenAI API は使わない
- data/processed は使わない
- J-Quants 生キャッシュを入力にする
- J-Quants 取得が必要な場合は `JQuantsDataService` を使う
- 未来データ混入を防ぐ
- ランダム分割しない
- まずは長時間の全期間学習を強制しない
- 重い検証コマンドはユーザー確認後に提示する
- 既存バックテストを壊さない
- ML機能は疎結合で追加する
