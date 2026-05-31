# J-Quants Plan Matrix

J-Quants Free / Light のどちらでも ai-fund-lab が停止せずに動くことを確認するための互換マトリクスです。Light専用機能はLight時のみ有効化し、Free時はfallbackまたは無効化して処理を継続します。

## Plan Capabilities

J-Quants連携はv2 endpointだけを使います。旧v1 endpoint（`/listed/info`, `/prices/daily_quotes`, `/indices/topix`, `/fins/statements`, `/markets/trades_spec`）は使用しません。

| capability | Free | Light | 用途 |
| --- | --- | --- | --- |
| listed_info | OK | OK | 上場銘柄一覧 |
| prices | OK | OK | 株価四本値 |
| financial_statements | OK | OK | 財務情報capability検証 |
| earnings_calendar | OK | OK | 決算発表予定日フィルター |
| trading_calendar | OK | OK | 営業日判定 |
| topix_prices | disabled | OK | TOPIX Relative Strength |
| investor_types | disabled | OK | 投資部門別情報 |

## Profile Matrix

| profile | earnings_calendar | topix_prices | investor_types | financial_statements | free対応 | light対応 |
| --- | --- | --- | --- | --- | --- | --- |
| rookie_dealer_02_v2_1 | - | - | - | - | OK | OK |
| rookie_dealer_02_v2_6 | - | preferred | - | - | OK: Prime平均/候補中央値へfallback | OK: TOPIX利用 |
| rookie_dealer_02_v2_8 | - | - | required | - | OK: investor_context_scoreを0にして無効化 | OK: investor_types利用 |
| rookie_dealer_02_v2_9 | - | - | - | required | OK | OK |
| rookie_dealer_02_v2_10 | required | - | - | - | OK | OK |

## Fallback Policy

- `topix_prices` がないFree planで `rookie_dealer_02_v2_6` を使う場合、TOPIX APIは呼ばず、Prime市場平均または候補中央値をbenchmarkにします。fallbackも使えない場合はRelative Strengthを0点扱いにします。
- `investor_types` がないFree planで `rookie_dealer_02_v2_8` を使う場合、投資部門別APIは呼ばず、`investor_context_score=0` として処理を継続します。
- `earnings_calendar` と `financial_statements` はFree / Lightの両方で利用可能なcapabilityとして扱います。
- capability不足はpreflightでwarning表示し、fallback可能な不足は `can_run_backtest: true` / `can_run_live/paper: true` のままにします。

## Preflight Checks

preflightでは以下を表示します。

- profile required capabilities
- current plan capabilities
- missing capabilities
- fallback applied
- can_run_backtest
- can_run_live/paper

例:

```text
J-Quants Plan: free
profile required capabilities: listed_info, prices, topix_prices
missing capabilities: topix_prices
fallback applied: topix_prices -> fallback to prime market average or candidate median benchmark
can_run_backtest: true
can_run_live/paper: true
```

## Recommended Plans

| profile | 推奨plan | 理由 |
| --- | --- | --- |
| rookie_dealer_02_v2_1 | Free | 基本の株価取得だけで検証可能 |
| rookie_dealer_02_v2_6 | Light | TOPIX Relative Strengthを正式benchmarkで検証できる |
| rookie_dealer_02_v2_8 | Light | investor_typesがLight専用 |
| rookie_dealer_02_v2_9 | Free | financial_statementsはFree/Light両対応 |
| rookie_dealer_02_v2_10 | Free | earnings_calendarはFree/Light両対応 |
