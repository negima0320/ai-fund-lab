# Rule-based 90d Backtest Summary 2021-06-01 to 2026-05-29

## 実行条件

- 期間: 2021-06-01 〜 2026-05-29
- profile: rookie_dealer_02_v2_8 新人ディーラー2号 v2.8
- provider: jquants
- ChatGPT/OpenAI: disabled
- broker: paper
- config_version: cfg_45b90cc

## Backtest Date Range Audit

- requested_start_date: 2021-06-01
- requested_end_date: 2026-05-29
- effective_trade_start_date: 2021-06-01
- effective_trade_end_date: 2026-05-29
- indicator_fetch_start_date: 2021-05-01
- price_fetch_requested_start: 2021-05-01
- price_fetch_clamped_start: 2021-05-01
- first_fetch_attempt_date: None
- raw_price_first_date: 2021-06-01
- raw_price_last_date: 2026-05-29
- first_price_date: 2021-06-01
- last_price_date: 2026-05-29
- first_trading_day: 2021-06-01
- last_trading_day: 2026-05-29
- target_trading_days: 1304
- target_trading_days_source: raw_price_cache
- processed_first_date: 2021-06-01
- processed_last_date: 2026-05-28
- missing_processed_dates_count: 1
- first_missing_processed_date: 2026-05-29
- last_missing_processed_date: 2026-05-29
- processed_days: 1220
- skipped_days: 1
- last_processed_day: 2026-05-28
- first_trade_date: 2021-06-02
- last_trade_date: 2026-05-29

### Data Coverage Audit

- prices.requested_end_date: 2026-05-29
- prices.latest_available_price_date: 2026-05-29
- prices.coverage_ok: true
- prices.warning: -

### Requested vs Effective Period

- requested_period: 2021-06-01 to 2026-05-29
- effective_period: 2021-06-01 to 2026-05-28
- effective_range_warning: processed days end before requested_end_date; check latest_available_price_date and fetch-period-prices logs

### Hardcoded Date Audit

- target: 2026-03-06
- match_count: 288
- warning: 2026-03-06 remains in config/src/docs/reports/README

### Processed Data Audit

- indicators_last_date: 2026-03-06
- candidates_last_date: 2026-05-29
- scored_candidates_last_date: 2026-05-29
- indicators_count: 42
- candidates_file_count: 1221
- scored_candidates_file_count: 1221
- dates_with_indicators_but_no_candidates: 0
- dates_with_candidates_but_no_scored: 0
- match: config/provider.yaml
- match: src/db.py
- match: src/main.py
- match: src/__pycache__/main.cpython-312.pyc
- match: src/__pycache__/db.cpython-312.pyc
- match: docs/runbook.md
- match: reports/backtest_2026-03-06_to_2026-03-06.json
- match: reports/backtest_2026-03-06_to_2026-03-06.md
- match: reports/day_2026-03-06.md
- match: reports/rookie_dealer_02_v2_2/backtest_2025-03-01_to_2026-03-06.json

## 結果サマリ

- 初期資金: 1,000,000円
- 最終資産: 1,081,236円
- 税引前損益: 79,336円
- 税引後損益: 63,219円
- 税引後損益率: 6.32%
- 勝率: 40.52%
- profit factor: 1.08
- 最大ドローダウン: -6.95%
- 総取引数: 422
- 利確回数: 51
- 損切り回数: 202
- 最大保有期間売却回数: 175
- no trade日数: 524
- selected_count合計: 1559

## 新人ディーラー1号コメント

検証期間では初期資金を上回りました。感情は考慮せず、同じ条件で再現性を確認します。

## 次に見るべき改善ポイント

- selected_count_total と no_trade_days を見て、スクリーニングが厳しすぎないか確認する。
- profit factor と勝率を合わせて、利幅と損切り幅のバランスを確認する。
- 最大ドローダウンがrisk_marginの許容範囲内か確認する。
- 利確、損切り、最大保有期間売却の比率を見て、出口ルールが短期売買に合っているか確認する。
- J-Quants Freeプランの12週間遅延データ前提で、複数期間でも再現するか確認する。

## 注意

OpenAI / ChatGPT APIは使わず、ルールベースのみで検証しています。PaperBrokerによる仮想売買であり、実売買は行いません。
