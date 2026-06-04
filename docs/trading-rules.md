# Trading Rules

この文書は現在実装されているPaperBroker上の売買ルール概要です。投資助言ではありません。

## Scope

- 対象は研究・検証・PaperBroker
- live自動売買は使わない
- 銘柄選定と買付条件はprofile YAMLで変わる

## Selection

候補はscreening後に `score_real_candidates()` でscored candidatesになります。その後、以下を満たしたものがselectedになります。

- `selection.min_score` または市場別min score
- `selection.min_confidence`
- `selection.max_selected`
- market filter
- volume filter
- RSI filter
- earnings / investor / hot-zone filters
- fallback/top-pick条件

市場区分はprofileの `market_filter.allowed_sections` で決まります。Prime固定ではありません。

## Buying

PaperBrokerの買付は以下を見ます。

- available cash
- minimum cash buffer
- max positions
- 100株単位
- target exposure
- allocation limit
- max position value rate
- duplicate position / pending buy
- safety limits

`disable_single_order_amount_limit` が有効なprofileでは、古い固定1注文上限ではなく、資金活用policyに基づく最大買付可能数量を使います。

## Affordable Fallback

`affordable_fallback_buy` が有効なprofileでは、通常選定後に余剰現金で買える高順位・高スコア候補を追加選定できます。fallback由来の取引は通常選定と区別して記録されます。

## Selling

売却はprofileのrisk設定に基づきます。

- stop loss
- take profit
- max holding business days
- market/risk exit

詳細なexit reasonの集計は `feature_analysis.md` のExit Reason系分析で確認します。

## Conditional Hold Extension

`conditional_hold_extension` は、最大保有期間到達時だけ判定されるexit timing実験です。通常の利確・損切りを無視して保有を伸ばすものではありません。利確・損切り条件に到達した場合は従来どおり売却します。

主な設定:

- `enabled`: 条件付き保有延長を有効化
- `min_unrealized_profit_rate` / `minimum_profit_for_extension`: 延長に必要な含み益率
- `min_relative_strength_score` / `minimum_relative_strength_score`: 延長に必要な相対強度score
- `require_ma25_uptrend`: `ma25 > previous_ma25` を要求
- `skip_ma5_condition`: MA5条件を使わない
- `max_holding_days`: 延長後の最大保有日数
- `max_extension_count`: 延長回数上限
- `extension_exit_guard`: 延長後だけ有効な失速撤退ガード。`max_profit_pullback_points` 以上の利益率悪化、または `min_remaining_profit_rate` 未満への低下で指定 `exit_reason` による売却候補にします

保有中銘柄が当日の候補リストにない場合でも、`indicators_YYYY-MM-DD.json` から `close`、`ma25`、`previous_ma25`、`relative_strength_score` などを補完して判定します。補完できない場合は `missing_indicator` として拒否理由に残します。

直近の検証profile:

- `rookie_dealer_02_v2_59`: v2.58同条件で、保有銘柄indicator補完修正後の検証
- `rookie_dealer_02_v2_60`: v2.59からrelative strength閾値だけを `60` から `5` に緩和
- `rookie_dealer_02_v2_61`: v2.60と同条件で、延長発動銘柄のbase比損益差分レポートを強化
- `rookie_dealer_02_v2_62`: v2.61と同条件に、延長後失速撤退ガードを追加

関連出力:

- `conditional_hold_extension_count`
- `conditional_hold_extension_profit_diff`
- `extension_profit_rate`
- `extension_exit_guard_triggered`
- `extension_exit_guard_reason`
- `extension_exit_guard_count`
- `extension_exit_guard_profit_diff_total`
- `extension_exit_guard_reasons`
- `conditional_hold_extension_rejected_reason_breakdown`
- `Conditional Hold Extension Rejected Detail`
- `Conditional Hold Extension Applied Detail`

## Integrity

売買結果は以下で監査します。

- `Backtest Result Integrity Audit`
- `Score Integrity Audit`
- `Compounding / Capital Flow Audit`
- `Trade Market Audit`
- `Monthly Performance Audit`
