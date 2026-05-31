# Score Formula Audit

## 現在のscore算出式

`score_real_candidates()` の現在の算出式は以下です。

```text
technical_score = clamp(
  ma_score
  + rsi_score
  + volume_score
  + candlestick_score
  + sector_score,
  0,
  50
)

total_score =
  technical_score
  + relative_strength_score
  + market_context_score
  + penalty_score
```

`rookie_dealer_02_v2_1` / `rookie_dealer_02_v2_6` では、固定加点だったニュース・財務componentを廃止しました。新規backtestではDB列にも保存せず、`total_score` にも含めません。将来ニュースやファンダメンタルを本当に評価する場合だけ、実データ由来のcomponentとして再追加します。

`score_components` では `technical_score` を展開して、次の合計として保存します。

```text
component_total =
  ma_score
  + rsi_score
  + volume_score
  + candlestick_score
  + sector_score
  + market_context_score
  + relative_strength_score
  + penalty_score
```

`component_total` と `total_score` は `matches_total_score` で検証します。最新ロジックでは一致する設計です。

## v2.1のscore構成

`rookie_dealer_02_v2_1` は Relative Strength Score を使いません。

- `features.relative_strength`: 未指定
- `scoring.use_relative_strength_score`: 未指定
- `relative_strength_score`: `0.0`
- theoretical range: `0-50`
- effective observed range: 最新の `feature_analysis.md` の `Score Effective Range Audit` で確認

v2.1の構成は以下です。

```text
total_score =
  technical_score
  + market_context_score
  + penalty_score
```

現在の実装では `market_context_score` は `0` で、market regime は主に selection filter に使われます。

旧v2.1の `total_score 65-75` は、固定25点を抜いた新v2.1では概ね `40-50` と読み替えます。selection閾値も `min_score: 45`、`fallback_min_score: 40`、`top_pick_min_score: 40` に変更しました。

## v2.6のscore構成

`rookie_dealer_02_v2_6` は v2.1 に Relative Strength Score を追加した検証用profileです。

- `features.relative_strength: true`
- `scoring.use_relative_strength_score: true`
- `scoring.relative_strength_score_weight: 10`
- theoretical range: `0-60`
- effective observed range: 最新の `feature_analysis.md` の `Score Effective Range Audit` で確認

v2.6の構成は以下です。

```text
total_score =
  technical_score
  + relative_strength_score
  + market_context_score
  + penalty_score
```

`relative_strength_score` は `technical_score` には含まれず、独立したcomponentとして一度だけ加算されます。

## Relative Strength

各銘柄について、5日、10日、20日の個別リターンからbenchmarkリターンを差し引きます。

```text
relative_strength_5d = stock_return_5d - benchmark_return_5d
relative_strength_10d = stock_return_10d - benchmark_return_10d
relative_strength_20d = stock_return_20d - benchmark_return_20d
```

TOPIX専用データがない場合は、同じtarget_dateで計算できる銘柄ユニバースの平均リターンを代替benchmarkとして使います。

初期スコアは最大10点です。

- `relative_strength_5d > 3%`: `+3`
- `relative_strength_10d > 5%`: `+4`
- `relative_strength_20d > 8%`: `+3`

## 想定通りだった点

- v2.1ではRelative Strength Scoreは `0.0` になり、`total_score` に加算されません。
- v2.6ではRelative Strength Scoreが最大10点で `total_score` に加算されます。
- v2.1/v2.6ではニュース・財務の固定加点はなく、`total_score` に入りません。
- `score_components.component_total` と `total_score` は一致します。
- `relative_strength_score` は `technical_score` に含まれていないため、technical側との二重加算はありません。
- decision log、scoring_results、screening_results、trades、feature_analysis、selection_quality にRelative Strength関連フィールドが流れます。

## 想定と違った点

以前のv2.1/v2.6では、実評価していないニュース・財務の固定加点がありました。これは選定結果を保つための見かけ上の点数でしたが、スコアの意味を紛らわしくしていました。

厳密な二重加算はありませんでした。

ただし、以下は「スコア加算」と「selection filter」の両方に関わるため、採用可否への影響は強めです。

- `RSI`: `rsi_score` と RSI過熱フィルター
- `volume_ratio`: `volume_score` と volume filter

これは `total_score` への二重加算ではありませんが、銘柄選定のバイアスとして明示的に監査対象にしました。

## 修正した点

- `feature_analysis.md` に `Score Formula Audit` を追加しました。
- v2.1/v2.6のニュース・財務固定加点を廃止しました。
- v2.1のselection閾値を `70/65/65` から `45/40/40` に変更しました。
- v2.6のselection閾値も `45/40/40` に変更し、Relative Strength最大10点を一度だけ加算します。
- `Score Formula Audit` で以下を出力します。
  - `total_score formula`
  - `component averages`
  - `component min/max`
  - `total_score mismatch count`
  - `profiles using relative_strength_score`
  - `duplicated signal warning`
- READMEにScore Formula Auditとscore想定レンジを追記しました。
- `feature_analysis.md` に `Score Effective Range Audit` を追加しました。
- Score Effective Range Audit ではprofileごとに以下を出力します。
  - `theoretical_max_score`
  - `effective_max_score`
  - `observed_min_score`
  - `observed_max_score`
  - `observed_avg_score`
  - `selected_min_score`
  - `selected_max_score`
  - `selected_avg_score`
- componentごとに `configured_max`、`observed_min`、`observed_max`、`observed_avg`、`non_zero_count`、`zero_count`、`status` を出力します。
- `market_context_score`、`penalty_score` が全件0の場合は `inactive` と判定します。
- READMEのscoreレンジ表現を、theoretical range と effective observed range に分けました。
- v2.1/v2.6のRelative Strength加算差分、component合計一致、scoreレンジをテストに追加しました。

## 追加で検討すべき点

- RSIと出来高はスコアとselection filterの両方に効くため、将来的に「スコアで評価するのか」「フィルターで除外するのか」をprofileごとに明確化すると比較しやすくなります。
- `market_context_score` は現在0点で、market regimeはselection filter中心です。地合いをスコア化する場合は、filterとの役割分担を先に決める必要があります。
- v2.6は最大60点、v2.1は最大50点なので、profile比較ではscore閾値の意味が変わります。比較時は `relative_strength_score` あり/なしを分けて読むべきです。
- TOPIXデータを取得できるようになったら、ユニバース平均benchmarkとの比較差分を検証してください。
