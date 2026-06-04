# Rookie Dealer Decision Flow

この文書は、候補生成からPaperBroker買付までの現在フローです。

## 1. Data Load

主な入力:

- J-Quants listed info / `data/raw/listed_stocks_jquants.json`
- daily prices / `data/raw/prices_YYYY-MM-DD.json`
- processed indicators / `data/processed/common/` または `data/processed/<profile_id>/`
- market context / `market_context_YYYY-MM-DD.json`
- profile YAML / `config/profiles/<profile_id>.yaml`
- registry metadata / `config/profile_registry.yaml`

dynamic exposureが有効なprofileでは、signal date当日のmarket contextではなく、前営業日以前に確定したmarket contextを使います。

## 2. Candidate Universe

候補は日次価格と銘柄マスターから作られます。市場区分はrowにあればrow、なければmaster lookupから補完します。

Candidate Universe Auditでは以下を分けます。

- raw candidate count by market
- after market filter candidate count by market
- screening excluded count by market
- after screening candidate count by market
- market lookup source breakdown

`market_filter.allowed_sections` に含まれない市場は除外です。`allow_unknown_market: false` の場合、Unknown / None / 空文字も除外します。

## 3. Screening

`src/real_screening.py::screen_candidates()` が、売買代金、出来高倍率、移動平均、RSI、ボラティリティなどで一次screeningを行います。

Standard市場拡張profileでは、Primeの既存条件を保ちつつ、Standardだけ緩和条件を適用する実験があります。Growthは対象profileで許可されるまでaudit中心です。

Screening Auditでは主に以下を出します。

- `screening_excluded_count_by_market`
- `screening_excluded_reason_by_market`
- `screening_excluded_date_by_market`
- representative samples

代表的な理由:

- `trading_value_low`
- `volume_ratio_low`
- `close_below_ma5`
- `ma5_below_ma25`
- `rsi_out_of_range`
- `volatility_too_high`
- `missing_required_price_or_indicator`
- `ranking_drop`

## 4. Scoring

スコアは `src/scoring.py::score_real_candidates()` で付与します。

現在の概念式:

```text
total_score =
  technical_score
  + relative_strength_score
  + investor_context_score
  + market_context_score
  + winner_loser_rule_score
  + penalty_score
```

`market_context_score` は現在0です。詳細は [scoring_spec.md](scoring_spec.md) を参照してください。

Scored Candidate Auditでは、market別にscored count、平均/中央値/最高score、上位候補、selected countを見ます。

## 5. Selection

`_apply_selection_rules()` が、スコア順にselected候補を決めます。

主な条件:

- `selection.min_score`
- `selection.market_min_score_overrides` / `selection.min_score_by_market_section`
- `selection.min_confidence`
- `selection.max_selected`
- `selection.allow_top_pick_when_no_selection`
- `selection.top_pick_min_score`
- volume filter
- RSI過熱filter
- RSI×出来高hot zone
- earnings filter
- investor context filter
- market filter

`allow_top_pick_when_no_selection` による採用は、通常min score未満でもprofileで許可されたfallbackです。Score Integrityでは `fallback_top_pick_selected_count` として通常の閾値違反と分けます。

Standard専用min scoreを持つprofileでは、`market_min_score_overrides` によりStandardだけ有効min scoreを下げられます。Primeのmin scoreは変えません。

## 6. PaperBroker Buy

PaperBrokerは、selected候補を実際に買えるか確認します。

主な制約:

- cash
- min cash buffer
- max positions
- 100株単位
- `allocation_limit`
- `target_exposure`
- `max_position_value_rate`
- duplicate position / pending buy
- safety policy

`disable_single_order_amount_limit: true` のprofileでは、古い1銘柄固定上限を無効化し、`capital_utilization_policy` に基づいて買える最大100株単位を計算します。

## 7. Affordable Fallback Buy

`affordable_fallback_buy.enabled: true` のprofileでは、通常selectedが買えない場合、または通常選定後に余剰現金がある場合に、同日のscored candidatesから買える候補を探せます。

候補条件:

- regular min score以上
- 通常ランキング上位圏内
- `round_lot_amount <= allocation_limit`
- `round_lot_amount <= available_cash`
- `selected=false`
- 既に保有中でない
- 当日pending buyでない
- market filter通過済み

fallbackで買った候補は `selection_source=affordable_fallback_buy`、`affordable_fallback_buy_selected=true` として保存されます。Result Integrityでは未選定BUY扱いにしません。

Affordable Fallback Buy Auditでは以下を確認します。

- candidate / attempt / selected / rejected counts
- selected by market
- rejected reason counts
- available cash before/after fallback
- selected samples

## 8. Exit and Holding Revaluation

通常の売却判定はPaperBrokerの保有ポジションに対して行われ、主なexit reasonは損切り、利確、最大保有期間到達、market/risk exitです。損切りと利確は、保有延長系profileでも優先されます。

### Conditional Hold Extension

`conditional_hold_extension.enabled: true` のprofileでは、最大保有期間到達で売却する直前に、条件を満たす保有銘柄だけ最大保有期間を延長できます。これは銘柄選定やスコアリングそのものを変える機能ではなく、exit timingの検証機能です。

現在の判定に使う主な項目:

- `profit_rate_at_max_holding`: 最大保有期間到達時点の含み益率
- `relative_strength_score`
- `close`
- `ma25`
- `previous_ma25`
- `max_extension_count`
- extension用の `max_holding_days`

`require_ma25_uptrend: true` の場合は `ma25 > previous_ma25` が必要です。`skip_ma5_condition: true` のprofileではMA5条件を見ません。

延長されなかった場合は、以下のような理由が保存されます。

- `profit_below_threshold`
- `relative_strength_below_threshold`
- `below_ma25`
- `ma25_not_uptrend`
- `missing_indicator`
- `already_extended`

### Indicator Enrichment for Held Positions

保有銘柄が当日の通常候補に出ていない場合でも、`enrich_candidates_with_position_prices()` が `indicators_YYYY-MM-DD.json` から保有銘柄の市場スナップショットを補完します。条件付き保有延長では、この補完行に以下のfieldが必要です。

- `previous_ma25`
- `relative_strength_score`
- `relative_strength_5d` / `relative_strength_10d` / `relative_strength_20d`
- `stock_return_5d` / `stock_return_10d` / `stock_return_20d`
- `benchmark_return_5d` / `benchmark_return_10d` / `benchmark_return_20d`
- `benchmark_source`

これらがindicatorに存在する場合は補完行へコピーされます。indicator自体に存在しない場合は、延長判定のreject reasonが `missing_indicator` になることがあります。

### Recent Conditional Hold Profiles

- `rookie_dealer_02_v2_59`: v2.58と同じ条件付き保有延長設定。保有銘柄indicator補完修正後の検証用。
- `rookie_dealer_02_v2_60`: v2.59をベースに、`min_relative_strength_score` / `minimum_relative_strength_score` を `60` から `5` に緩和した発動確認用。

`compare_profiles.md/json` と `backtest_summary.json` には以下が出ます。

- `conditional_hold_extension_count`
- `conditional_hold_extension_profit_diff`
- `conditional_hold_extension_win_rate`
- `conditional_hold_extension_profit_factor`
- `conditional_hold_extension_rejected_count`
- `conditional_hold_extension_rejected_reason_breakdown`
- `conditional_hold_extension_rejected_samples`
- `Conditional Hold Extension Rejected Detail`

## 9. Integrity

重要な監査:

- `trade_without_selected_count`: selectedにないBUYが出ていないか
- `selected_without_trade_reason_breakdown`: selectedされたが買えなかった理由
- `market_filter_violation_count`: 市場filter違反
- `future_data_leak_count`: signal date以降の情報混入
- `signal_entry_date_violation_count`: signal date / entry date整合性
- `invalid_below_threshold_selected_count`: profile設定で許可されない閾値未満選定
