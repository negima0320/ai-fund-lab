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

## Integrity

売買結果は以下で監査します。

- `Backtest Result Integrity Audit`
- `Score Integrity Audit`
- `Compounding / Capital Flow Audit`
- `Trade Market Audit`
- `Monthly Performance Audit`

