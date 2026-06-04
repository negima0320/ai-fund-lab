# 銘柄選定スコア計算仕様

この資料は、現在の実装に基づくスコア計算仕様です。正は `src/scoring.py` の `score_real_candidates()` です。旧資料にあった「100点満点、ニュース30、財務20」の固定配点は現在の実装では使っていません。

## 全体式

現在の `total_score` は、候補ごとに次の形で計算されます。

```text
total_score =
  max(0,
    technical_score
    + relative_strength_score
    + investor_context_score
    + market_context_score
    + winner_loser_rule_score
    + penalty_score
  )
```

`penalty_score` は負値です。実装上は `rsi_selection_penalty` と `affordability_penalty` を合算し、`score_components.penalty_score` に負値として保存します。

```text
penalty_score = -(rsi_selection_penalty + affordability_penalty)
```

`market_context_score` は現在0固定です。market regimeやdynamic exposureは主に選定・資金配分・監査で使われ、現行のスコア加点ではありません。

## score_components

`scored_candidates_YYYY-MM-DD.json` と scoring logには、主に以下が保存されます。

| field | 意味 |
| --- | --- |
| `ma_score` / `trend_score` | 移動平均条件のスコア |
| `volume_score` | 出来高倍率と売買代金のスコア |
| `rsi_score` | RSIゾーンのスコア |
| `candlestick_score` | ローソク足シグナルのスコア |
| `technical_score` | technical合計。sector補正後に0〜50へclamp |
| `sector_score` | sector momentumによるtechnical内の補正差分 |
| `relative_strength_score` | profileで有効な場合のTOPIX等に対する相対強度 |
| `investor_context_score` | profileで有効な場合の投資部門別需給スコア |
| `market_context_score` | 現在0 |
| `winner_loser_rule_score` | 勝ち負け分析から作った実験的加減点 |
| `penalty_score` | RSI/affordabilityなどの減点 |
| `component_total` | component合計 |
| `total_score` | 最終スコア |
| `matches_total_score` | component合計とtotal_scoreの整合性 |

## technical_score

`technical_score` は最大50点です。

```text
technical_score =
  clamp(
    trend_score
    + volume_score
    + rsi_score
    + candlestick_score
    + sector_adjustment,
    0,
    50
  )
```

| component | 最大 | 主な入力 |
| --- | ---: | --- |
| `trend_score` | 15 | `close`, `ma5`, `ma25` |
| `volume_score` | 10 | `volume_ratio`, `turnover_value` |
| `rsi_score` | 10 | `rsi` |
| `candlestick_score` | 15 | `candlestick_signals`, candle shape |
| `sector_adjustment` | おおむね -5〜+5 | `sector_momentum_score` |

### trend_score

- `close > ma5`: +5
- `ma5 > ma25`: +5
- `(ma5 - ma25) / ma25` が3%付近に近いほど最大+5

### volume_score

```text
volume_ratio_part = min(volume_ratio, 2.5) / 2.5 * 6
turnover_part = min(turnover_value / 2,000,000,000, 1.0) * 4
volume_score = clamp(round(volume_ratio_part + turnover_part), 0, 10)
```

`turnover_value` はJ-Quants日次株価の `Va` がある場合はそれを優先し、欠損時は価格×出来高の推定値へfallbackします。どちらを使ったかは `direct_turnover_value_source` / API Field Usage Auditで確認します。

### rsi_score

短期買い候補として、RSI 50〜65を最も良いゾーンとして扱います。

| RSI | スコア |
| --- | ---: |
| 50〜65 | 10 |
| 40〜50 | 6〜10 |
| 65〜70 | 8〜6 |
| 30〜40 | 0〜6 |
| 70〜80 | 6〜0 |
| 30未満 / 80超 | 0 |

profileで `selection.reject_overheated_rsi: true` の場合、RSI過熱は減点ではなく選定除外になります。

### candlestick_score

ローソク足データがある場合、以下のようなシグナルを評価します。

| signal | 加減点 |
| --- | ---: |
| `bullish_candle` | +4 |
| `strong_bullish_candle` | +5 |
| `long_lower_shadow_support` | +3 |
| `ma_reclaim` | +2 |
| `volume_confirmed_breakout` | +3 |
| `long_upper_shadow_warning` | -4 |
| `overheated_warning` | -5 |

ローソク足データが欠ける場合は互換用に `candlestick_score=12` として扱います。fallback候補には追加で -2 が入ります。

## profile別の追加スコア

`features.*` と `scoring.use_*` は別物です。

- `features.*`: データ取得・特徴量生成を有効化する
- `scoring.use_*`: 生成済み特徴量を `total_score` に加算する

例:

| feature | data_enabled | scoring_enabled | 現在の扱い |
| --- | --- | --- | --- |
| `relative_strength` | `features.relative_strength` | `scoring.use_relative_strength_score` | 有効時は最大10点程度の追加スコア |
| `investor_context` | `features.investor_context` | `scoring.use_investor_context_score` | 有効時は -3〜+5 程度の補正 |
| `financial_context` | `features.financial_context` | `scoring.use_financial_score` | 現在は主にdata/audit用。通常の実験ではtotal_scoreへ加算しない |
| `market_context` | `features.market_context` | なし | regime、risk、dynamic exposure、監査に利用。scoreは0固定 |

Feature Activation Auditでは `data_enabled`、`scoring_enabled`、`actual_trigger_count` が分かれます。data-only profileは、APIや保存の検証だけを行い、scoreには足しません。

## relative_strength_score

`features.relative_strength: true` かつ `scoring.use_relative_strength_score: true` のprofileで有効です。

主な入力:

- `stock_return_5d`, `stock_return_10d`, `stock_return_20d`
- `benchmark_return_5d`, `benchmark_return_10d`, `benchmark_return_20d`
- `relative_strength_5d`, `relative_strength_10d`, `relative_strength_20d`

ベンチマークはLight planではTOPIXを優先します。利用不能な場合はmarket average等へfallbackします。fallback状態はfeature analysisのRelative Strength系監査で確認します。

## investor_context_score

`features.investor_context: true` かつ `scoring.use_investor_context_score: true` のprofileで有効です。J-Quants `/equities/investor-types` をもとに、海外投資家の買越、4週合計、トレンド、個人投資家との差などを補助的に評価します。

`investor_context_filter.enabled: true` のprofileでは、スコア加算ではなくマイナス需給の除外フィルターとして使えます。現在の整理済みregistryでは、この系統の旧検証profileは削除済みです。

## affordability / winner-loser adjustments

`affordability_filter` が有効なprofileでは、100株購入金額が `preferred_round_lot_amount` を超える候補に `price_band_penalty` を入れます。これは買付ロジックではなくスコア上の順位調整です。

`winner_loser_rule_adjustment` は、勝ち負け分析から作った実験的な加点/減点です。指定条件に合う候補だけ `winner_loser_rule_score` を加えます。

## 一次スクリーニング

スコア計算前に `src/real_screening.py` の `screen_candidates()` で候補を絞ります。通常の主な条件は以下です。

- 売買代金
- 出来高倍率
- `close > ma5`
- `ma5 > ma25`
- RSI範囲
- 短期ボラティリティ

Standard市場拡張profileでは、Primeの条件を維持したまま、Standardだけ売買代金・出来高・移動平均などを緩和する実験があります。Candidate Universe AuditとScreening Auditでは、market filter直後とscreening後を分けて集計します。

## 選定ルール

スコア計算後、`_apply_selection_rules()` が以下を判定します。

- `selection.min_score`
- `selection.market_min_score_overrides` / `selection.min_score_by_market_section`
- `selection.min_confidence`
- `selection.max_selected`
- `allow_top_pick_when_no_selection`
- `top_pick_min_score`
- volume filter
- RSI過熱フィルター
- RSI×出来高過熱ゾーン
- earnings filter
- investor context filter
- market filter

`market_filter.allowed_sections` に含まれない市場区分は除外されます。`allow_unknown_market: false` の場合、Unknown / None / 空文字も除外です。

`allow_top_pick_when_no_selection` によるtop-pick採用は、profile設定通りならScore Integrity上の異常とは扱いません。通常選定で市場別min scoreを下回った候補が選ばれた場合は、`invalid_below_threshold_selected_count` として警告対象です。

## APIと保存値の対応

| API / cache | 主なフィールド | processed/scoringでの利用 |
| --- | --- | --- |
| `/equities/master` | `Code`, `CoName`, `Mkt/MktNm`, `S17/S17Nm`, `S33/S33Nm`, `ScaleCat`, `Mrgn/MrgnNm`, `ProdCat` | code/name/market_section/sector/scale/margin/product category |
| `/equities/bars/daily` | `O/H/L/C/Vo`, `AdjO/AdjH/AdjL/AdjC/AdjVo`, `Va`, `UL`, `LL` | adjusted priceベースの指標、turnover、limit flag audit |
| `/indices/bars/daily/topix` | TOPIX daily bars | relative strength benchmark |
| `/equities/investor-types` | 投資部門別売買情報 | investor context score/filter |
| `/equities/earnings-calendar` | 決算予定 | earnings filter |
| `/fins/summary` | 財務サマリ | 現在は主にaudit/future candidate |

future data leakを避けるため、dynamic exposureの市場局面はsignal dateの前営業日以前のmarket contextを使います。同日終値由来contextを同日判断に使った場合はIntegrity側で検出対象です。
