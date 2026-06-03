# J-Quants Plan Matrix

この文書は `config/jquants.yaml` と `src/jquants_plan.py` に基づくJ-Quants plan整理です。

## Plans

| plan | Earliest supported date | Rate limit setting | Parallel |
| --- | --- | ---: | --- |
| `free` | 2024-01-01 | 5 req/min | false |
| `light` | 2021-05-01 | 60 req/min | true, max 4 |

`investor_types` はFreeでは2024-01-01以降、Lightでは2021-05-31以降を想定します。

## Capabilities

| Capability | Free | Light | Notes |
| --- | --- | --- | --- |
| `listed_info` | yes | yes | 銘柄マスター、市場区分、業種など |
| `prices` | yes | yes | 日次株価。調整済み価格、売買代金、値幅制限flagを利用可能 |
| `financial_statements` | yes | yes | 現在は主にaudit/future candidate。通常scoreには加算しない |
| `earnings_calendar` | yes | yes | 決算予定フィルター |
| `trading_calendar` | no | yes | 営業日判定、前営業日fallbackなど |
| `topix_prices` | no | yes | relative strength benchmark |
| `investor_types` | no | yes | 投資部門別需給 |

## Profile Compatibility

`config/profile_registry.yaml` の `required_plan` が実験上の推奨planです。`validate-config` は現在planとprofile要件を照合し、fallback可能な不足はwarning、fallback不可な不足はfailureにします。

代表例:

| Profile family | Required plan | Reason |
| --- | --- | --- |
| `rookie_dealer_02_v2_1` | free | technical baseline |
| relative strength系 (`v2_6`, `v2_26`, `v2_51` など) | light | TOPIX benchmark / trading calendarを使うため |
| investor context系 (`v2_8`, `v2_11`) | light | `/equities/investor-types` を使うため |
| earnings filter系 | free | earnings calendarがfree capabilityに含まれる |

後続の実験profileはregistryの `required_plan` を確認してください。

## Fallback Behavior

| Missing capability | Fallback |
| --- | --- |
| `topix_prices` | Prime平均やcandidate medianなど、実装されたmarket average系benchmarkへfallback |
| `investor_types` | investor context score/filterを無効または中立扱い |
| `trading_calendar` | 利用可能な日付リストや既存価格日付から営業日を推定 |
| unsupported/no-data ranges | `data/cache/jquants/unsupported_ranges` / `empty_ranges` に記録 |

Fallbackは検証継続のための機構です。比較実験では、同じplan・同じfallback状態で揃えてください。

## Cache Paths

J-Quants response cache:

```text
data/cache/jquants/
  prices/
  topix_prices/
  earnings_calendar/
  investor_types/
  financial_statements/
  listed_info/
  trading_calendar/
```

Processed common cache:

```text
data/processed/common/
  indicators/<cache_key>/indicators_YYYY-MM-DD.json
  candidates/<cache_key>/candidates_YYYY-MM-DD.json
```

Profile runtime path:

```text
data/processed/<profile_id>/
  indicators_YYYY-MM-DD.json
  candidates_YYYY-MM-DD.json
  scored_candidates_YYYY-MM-DD.json
```

## API Field Notes

Daily prices use adjusted fields where available for technical indicators and relative strength. Turnover value uses API `Va` when present and falls back to price × volume only when necessary. Limit up/down flags are saved/audited but are disabled-by-default rule candidates unless a profile explicitly enables a guard.

Financial summary data is audited for freshness and future data leak risk. It is not part of the default short-term score.

