# Log and Cache Design

この文書は、現在のCLI helpと実装に基づくログ・キャッシュ階層です。

## Storage

```text
storage/
  ai_fund_lab.sqlite3
  STOP_TRADING
```

`STOP_TRADING` は新規買付停止の安全フラグです。

## Raw Data

```text
data/raw/
  prices_YYYY-MM-DD.json
  listed_stocks_jquants.json
  prime_stocks_jquants.json
  no_data_days_jquants.json
```

raw snapshotです。`prices_YYYY-MM-DD.json` や `listed_stocks_jquants.json` はrun-local memory cacheの対象でもあります。

## J-Quants API Cache

```text
data/cache/jquants/
  prices/
  topix_prices/
  earnings_calendar/
  investor_types/
  financial_statements/
  listed_info/
  trading_calendar/
  unsupported_ranges/
  empty_ranges/
```

API response cacheです。通常のcleanupでは保持対象にします。unsupported/no-data/rate limitの監査にも使います。

## Processed Data

```text
data/processed/
  indicators_YYYY-MM-DD.json
  market_context_YYYY-MM-DD.json
  common/
    indicators/<cache_key>/indicators_YYYY-MM-DD.json
    candidates/<cache_key>/candidates_YYYY-MM-DD.json
  <profile_id>/
    indicators_YYYY-MM-DD.json
    candidates_YYYY-MM-DD.json
    scored_candidates_YYYY-MM-DD.json
```

`data/processed/common/` は互換profile間で再利用するprocessed common cacheです。`data/processed/<profile_id>/` はprofile runtime I/O pathで、common cache利用後でもprofile固有のscoringやauditで読み書きされることがあります。

`scored_candidates` はprofile固有のscore/selection結果なので、基本的にcommon cache対象外です。

## Daily Logs

```text
logs/
  screening/<profile_id>/screening_YYYY-MM-DD.json
  scoring/<profile_id>/scoring_YYYY-MM-DD.json
  ai_decision/<profile_id>/ai_decision_YYYY-MM-DD.json
  market_context/<profile_id>/market_context_YYYY-MM-DD.json
  trades/<profile_id>/trades_YYYY-MM-DD.json
  portfolio/<profile_id>/portfolio_YYYY-MM-DD.json
  portfolio/<profile_id>/state.json
  safety/<profile_id>/safety_YYYY-MM-DD.json
  reflections/<profile_id>/reflections_YYYY-MM-DD.json
```

一部のlegacy/monthly log helperでは月階層のpathも残っていますが、通常の現在フローでは上記のprofile別日次JSONが主です。

## Backtest Logs

```text
logs/backtests/<profile_id>/<START>_to_<END>/
  backtest_summary.json
  scoring_YYYY-MM-DD.json
  trades_YYYY-MM-DD.json
  portfolio_YYYY-MM-DD.json
  summary.csv
  trades.csv
```

`backtest_summary.json` はsummary生成やfeature analysisの重要なsourceです。ただし、feature analysisで再計算されたintegrity値がある場合、experiment summaryはそれを優先する実装になっています。

## Reports

```text
reports/
  <profile_id>/backtests/
    analysis_latest.md
    analysis_latest.json
    feature_analysis.md
    feature_analysis.json
    selection_quality.md
    selection_quality.json
  backtests/<profile_id>/<START>_to_<END>/
  experiments/<START>_to_<END>/<base_profile>/
    experiment_summary.md
    experiment_summary.json
    compare_profiles.md
    compare_profiles.json
  profile_comparisons/
  articles/daily/YYYY/MM/<profile_id>/
```

`feature_analysis.md/json` は監査の中心です。主な見出し:

- `Backtest Result Integrity Audit`
- `Score Integrity Audit`
- `Market Filter Audit`
- `Candidate Universe Audit`
- `Screening Audit`
- `Scored Candidate Audit`
- `Selected Candidate Audit`
- `Trade Market Audit`
- `Capital Utilization Audit`
- `Allocation Strategy Audit`
- `Affordable Fallback Buy Audit`
- `Compounding / Capital Flow Audit`
- `Monthly Performance Audit`
- `Performance Audit`

## Cache Cleanup Caution

削除してよいかは用途で変わります。

- `data/cache/jquants/`: API再取得を避けるため通常保持
- `data/processed/common/`: 長期backtest高速化に重要。削除すると再計算が重い
- `data/processed/<profile_id>/`: profile固有のruntime成果物。対象profileを再生成できるなら削除候補
- `logs/backtests/<profile_id>/<period>/`: 既存成果物分析に必要。再実行しない期間のログは保持
- `reports/experiments/`: summary再生成や比較確認に必要。不要な古い実験だけ削除候補

削除前は `storage-audit` や `inspect-cache` を使ってください。

