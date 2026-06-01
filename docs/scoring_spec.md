# 銘柄選定スコア計算仕様

この資料は、現在の実装をもとに、銘柄選定に使うスコア計算の仕様と、各入力値がどのAPI・保存ファイルから来ているかを整理したものです。

対象実装:

- `src/scoring.py`
- `src/indicators.py`
- `src/technical_indicators.py`
- `src/real_screening.py`
- `src/data_provider.py`
- `config/profiles/rookie_dealer_02_v2_1.yaml`

## 全体の流れ

銘柄選定はおおまかに以下の流れです。

1. J-Quants APIから上場銘柄マスターと日次株価を取得する
2. 日次株価からテクニカル指標を計算する
3. テクニカル指標から候補銘柄を一次スクリーニングする
4. 候補銘柄にスコアを付ける
5. `total_score`、出来高、RSI、Market Filterなどで最終選定する

## 基本スコア式

現在の実スコア計算本体は `src/scoring.py` の `score_real_candidates()` です。

`rookie_dealer_02_v2_1` では実質的に以下の式です。

```text
total_score =
  technical_score
  + market_context_score
  + penalty_score
```

ただし、現在 `market_context_score` は `0` 固定です。

そのため、実質的には以下です。

```text
total_score = technical_score - RSI過熱ペナルティ
```

ただし `rookie_dealer_02_v2_1` では `reject_overheated_rsi: true` のため、RSIが閾値を超えた場合は減点ではなく新規買付除外になります。

## profile別の追加スコア

profileによって追加されるスコアが変わります。

| profile | 追加要素 | スコア式 |
| --- | --- | --- |
| `rookie_dealer_02_v2_1` | なし | `technical_score + market_context_score + penalty_score` |
| `rookie_dealer_02_v2_6` | Relative Strength | `technical_score + relative_strength_score + market_context_score + penalty_score` |
| `rookie_dealer_02_v2_8` | 投資部門別需給 | `technical_score + investor_context_score + market_context_score + penalty_score` |
| `rookie_dealer_02_v2_11` | 投資部門別需給をフィルター利用 | スコア加算せず、需給が悪い場合に除外 |
| `rookie_dealer_02_v2_9` | 財務情報capability検証 | 財務スコアは現在未使用 |
| `rookie_dealer_02_v2_10` | 決算予定フィルター | スコア加算せず、決算前後を除外 |

## technical_score

`technical_score` は最大50点です。

内訳は以下です。

| 項目 | 最大点 | 使う値 | 内容 |
| --- | ---: | --- | --- |
| `trend_score` | 15 | `close`, `ma5`, `ma25` | 終値が5日線より上、5日線が25日線より上、MA乖離が適度か |
| `volume_score` | 10 | `volume_ratio`, `turnover_value` | 出来高前日比と売買代金 |
| `rsi_score` | 10 | `rsi` | RSIが短期買い候補として適切な範囲か |
| `candlestick_score` | 15 | ローソク足シグナル | 陽線、強い陽線、下ヒゲ、5日線回復、出来高ブレイクなど |
| `sector_score` | +/-5 | `sector_momentum_score` | 業種モメンタムによる加減点 |

実装上は以下の形です。

```text
base_technical_score =
  trend_score
  + volume_score
  + rsi_score
  + candlestick_score

technical_score =
  clamp(base_technical_score + sector_adjustment, 0, 50)
```

## trend_score

最大15点です。

計算内容:

| 条件 | 加点 |
| --- | ---: |
| `close > ma5` | +5 |
| `ma5 > ma25` | +5 |
| `(ma5 - ma25) / ma25` が約3%に近い | 最大+5 |

MA乖離の加点は、3%付近を良い形として評価し、離れすぎると点数が下がります。

## volume_score

最大10点です。

計算内容:

| 条件 | 加点 |
| --- | ---: |
| `volume_ratio` が高い | 最大+6 |
| `turnover_value` が大きい | 最大+4 |

式のイメージ:

```text
volume_ratio_part = min(volume_ratio, 2.5) / 2.5 * 6
turnover_part = min(turnover_value / 2,000,000,000, 1.0) * 4
volume_score = clamp(round(volume_ratio_part + turnover_part), 0, 10)
```

## rsi_score

最大10点です。

RSI 50〜65を最も良い短期買いゾーンとして扱います。

| RSI | 点数 |
| --- | ---: |
| 50〜65 | 10 |
| 40〜50 | 6〜10 |
| 65〜70 | 8〜6 |
| 30〜40 | 0〜6 |
| 70〜80 | 6〜0 |
| 30未満または80超 | 0 |

なお、`rookie_dealer_02_v2_1` では `selection.max_rsi_for_new_position: 65` かつ `reject_overheated_rsi: true` なので、RSIが65を超えるとスコア以前に新規買付除外になります。

## candlestick_score

最大15点です。

ローソク足シグナルは `src/candlestick.py` で計算され、スコアでは以下のように使われます。

| シグナル | 加減点 |
| --- | ---: |
| `bullish_candle` | +4 |
| `strong_bullish_candle` | +5 |
| `long_lower_shadow_support` | +3 |
| `ma_reclaim` | +2 |
| `volume_confirmed_breakout` | +3 |
| `long_upper_shadow_warning` | -4 |
| `overheated_warning` | -5 |

ローソク足データが欠けている場合は、互換用に `candlestick_score = 12` として扱われます。

fallback候補の場合はさらに `-2` されます。

## sector_score

`sector_momentum_score` がある場合のみ、`technical_score` に加減点されます。

```text
sector_adjustment = clamp(round((sector_momentum_score - 50) / 10), -5, +5)
```

つまり、業種モメンタムが50なら中立、60なら約+1、40なら約-1です。

## relative_strength_score

`rookie_dealer_02_v2_6` など、`features.relative_strength: true` かつ `scoring.use_relative_strength_score: true` のprofileで有効です。

最大10点です。

計算対象:

- `stock_return_5d`
- `stock_return_10d`
- `stock_return_20d`
- `benchmark_return_5d`
- `benchmark_return_10d`
- `benchmark_return_20d`

個別株リターンからベンチマークリターンを引いた値が `relative_strength_*d` です。

```text
relative_strength_5d = stock_return_5d - benchmark_return_5d
relative_strength_10d = stock_return_10d - benchmark_return_10d
relative_strength_20d = stock_return_20d - benchmark_return_20d
```

加点ルール:

| 条件 | 加点 |
| --- | ---: |
| `relative_strength_5d > 0.03` | +3 |
| `relative_strength_10d > 0.05` | +4 |
| `relative_strength_20d > 0.08` | +3 |

合計最大10点です。

ベンチマークは設定上TOPIXを優先します。TOPIXが使えない場合は市場平均にフォールバックします。

## investor_context_score

`rookie_dealer_02_v2_8` など、`features.investor_context: true` かつ `scoring.use_investor_context_score: true` のprofileで有効です。

スコア範囲は `-3` 〜 `+5` です。

主に投資部門別売買情報から以下を見ます。

- 海外投資家の買越額
- 海外投資家の4週合計買越額
- 海外投資家需給トレンド
- 個人投資家の売買動向

加点・減点:

| 条件 | 加減点 |
| --- | ---: |
| 海外投資家4週合計が買い越し | +2 |
| 海外投資家4週合計が売り越し | -2 |
| 海外投資家需給トレンドが改善 | +2 |
| 海外投資家需給トレンドが悪化 | -1 |
| 個人が売り越し、海外が買い越し | +1 |

最終的に `-3` 〜 `+5` に丸められます。

## 一次スクリーニング条件

スコア計算の前に、`src/real_screening.py` の `screen_candidates()` で候補を絞ります。

通常条件:

| 条件 | 閾値 |
| --- | ---: |
| `turnover_value` | 500,000,000円以上 |
| `volume_ratio` | 1.5以上 |
| `close > ma5` | 必須 |
| `ma5 > ma25` | 必須 |
| `rsi` | 40〜70 |
| `five_day_volatility` | 0.12以下 |

候補が足りない場合は fallback 条件を使います。

fallback条件:

| 条件 | 閾値 |
| --- | ---: |
| `turnover_value` | 300,000,000円以上 |
| `volume_ratio` | 1.2以上 |
| `rsi` | 35〜75 |
| `five_day_volatility` | 0.16以下 |

候補のランキングでは以下を重視します。

1. `volume_ratio` が高い
2. `turnover_value` が大きい
3. `sector_momentum_score` が高い
4. MA乖離が約3%に近い
5. RSIが65を超えすぎていない

## 最終選定条件

`rookie_dealer_02_v2_1` の主な設定は以下です。

```yaml
selection:
  min_score: 45
  fallback_min_score: 40
  min_confidence: 0.70
  allow_top_pick_when_no_selection: true
  top_pick_min_score: 40
  max_selected: 5
  max_rsi_for_new_position: 65
  reject_overheated_rsi: true

market_filter:
  prime: true
  standard: false
  growth: false
  allow_unknown_market: false

volume_filter:
  enabled: true
  min_volume_ratio: 2.0
```

つまり、スコアが高くても以下の場合は除外されます。

- `market_section` が `TSEPrime` ではない
- `market_section` が `Unknown` / `None` / 空文字
- `volume_ratio < 2.0`
- `rsi > 65`
- `confidence < 0.70`
- 最大採用数5件を超える

通常基準で1件も選ばれない場合、`allow_top_pick_when_no_selection: true` により、`top_pick_min_score: 40` 以上の最上位候補を1件だけ採用する可能性があります。

## confidence

confidenceは以下の形で計算されます。

```text
confidence = 0.45 + technical_score / 100
```

ただし、以下で減点されます。

- fallback候補なら `-0.08`
- 必須フィールド欠損1つにつき `-0.04`

必須フィールド:

- `close`
- `volume`
- `ma5`
- `ma25`
- `rsi`
- `volume_ratio`
- `turnover_value`
- `five_day_volatility`

最終的に `0.10` 〜 `0.95` に丸められます。

## 入力値とAPI由来

| 値 | 用途 | API endpoint | 保存先 |
| --- | --- | --- | --- |
| `code` | 銘柄コード | `/equities/master`, `/equities/bars/daily` | `data/raw/listed_stocks_jquants.json`, `data/raw/prices_YYYY-MM-DD.json` |
| `name` | 銘柄名 | `/equities/master` | `data/raw/listed_stocks_jquants.json` |
| `market_section` | Prime/Standard/Growth判定 | `/equities/master` | `data/raw/listed_stocks_jquants.json` |
| `sector_name` | 業種モメンタム | `/equities/master` | `data/raw/listed_stocks_jquants.json` |
| `open` | ローソク足 | `/equities/bars/daily` | `data/raw/prices_YYYY-MM-DD.json` |
| `high` | ローソク足 | `/equities/bars/daily` | `data/raw/prices_YYYY-MM-DD.json` |
| `low` | ローソク足 | `/equities/bars/daily` | `data/raw/prices_YYYY-MM-DD.json` |
| `close` | MA/RSI/リターン/売買代金 | `/equities/bars/daily` | `data/raw/prices_YYYY-MM-DD.json` |
| `volume` | 出来高倍率/売買代金 | `/equities/bars/daily` | `data/raw/prices_YYYY-MM-DD.json` |
| `ma5` | トレンド評価 | raw pricesから計算 | `data/processed/.../indicators_YYYY-MM-DD.json` |
| `ma25` | トレンド評価 | raw pricesから計算 | `data/processed/.../indicators_YYYY-MM-DD.json` |
| `rsi` | RSIスコア/過熱除外 | raw pricesから計算 | `data/processed/.../indicators_YYYY-MM-DD.json` |
| `volume_ratio` | 出来高スコア/出来高フィルター | raw pricesから計算 | `data/processed/.../indicators_YYYY-MM-DD.json` |
| `turnover_value` | 出来高スコア/候補条件 | raw pricesから計算。API値があれば保持 | `data/processed/.../indicators_YYYY-MM-DD.json` |
| `candlestick_signals` | ローソク足スコア | raw pricesから計算 | `data/processed/.../indicators_YYYY-MM-DD.json` |
| `sector_momentum_score` | sector_score | raw prices + listed stock master | `data/processed/.../candidates_YYYY-MM-DD.json` |
| `relative_strength_score` | v2_6追加スコア | TOPIXまたは市場平均 | `data/processed/.../indicators_YYYY-MM-DD.json` |
| `investor_context_score` | v2_8追加スコア/ v2_11フィルター | `/equities/investor-types` | `data/cache/jquants/investor_types/...json` |
| `earnings_announcement_date` | 決算フィルター | `/equities/earnings-calendar` | `data/cache/jquants/earnings_calendar/...json` |
| 財務情報 | 現状スコア未使用 | `/fins/summary` | `data/cache/jquants/financial_statements/...json` |

## J-Quants API endpoint対応

実装上のJ-Quants API呼び出しは `src/data_provider.py` にまとまっています。

| 用途 | endpoint | 実装関数 | 主な保存先 |
| --- | --- | --- | --- |
| 銘柄マスター | `/equities/master` | `get_listed_stocks()` | `data/raw/listed_stocks_jquants.json` |
| 日次株価 | `/equities/bars/daily` | `get_daily_prices()` / `get_daily_prices_range()` | `data/raw/prices_YYYY-MM-DD.json` |
| TOPIX | `/indices/bars/daily/topix` | `get_topix_prices()` | `data/cache/jquants/topix_prices/...json` |
| 投資部門別売買 | `/equities/investor-types` | `fetch_investor_types()` | `data/cache/jquants/investor_types/...json` |
| 決算予定 | `/equities/earnings-calendar` | `fetch_earnings_calendar()` | `data/cache/jquants/earnings_calendar/...json` |
| 財務サマリー | `/fins/summary` | `fetch_financial_statements()` | `data/cache/jquants/financial_statements/...json` |

## 現在スコアに入っていないもの

以下は現在の主要profileでは銘柄選定スコアに直接入っていません。

- ニュース
- OpenAI判断
- 財務スコア
- PER/PBR/ROE
- 為替
- 米国市場サマリー

一部はレポートや将来拡張用のフィールドとして存在しますが、`rookie_dealer_02_v2_1` の選定スコアには加算されません。

## まとめ

現在の銘柄選定は、J-Quantsの日次株価から作った短期テクニカル指標を中心にしています。

特に `rookie_dealer_02_v2_1` は以下を強く重視します。

- 東証Primeのみ
- 出来高倍率2倍以上
- RSI 65超えを追わない
- `close > ma5 > ma25`
- 売買代金が十分ある
- ローソク足が短期買いに向いている
- 業種モメンタムが悪すぎない

つまり、現在の仕様は「短期で資金が入っているPrime銘柄を拾うが、過熱しすぎた銘柄は買わない」設計です。
