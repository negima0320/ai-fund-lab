# Rule-based 90d Backtest Summary 2025-03-07 to 2026-03-06

## 実行条件

- 期間: 2025-03-07 〜 2026-03-06
- profile: rookie_dealer_02_v2_68 新人ディーラー2号 v2.68
- provider: jquants
- ChatGPT/OpenAI: disabled
- broker: paper
- config_version: cfg_7992ad8

## Backtest Date Range Audit

- requested_start_date: 2025-03-07
- requested_end_date: 2026-03-06
- effective_trade_start_date: 2025-03-07
- effective_trade_end_date: 2026-03-06
- indicator_fetch_start_date: 2024-09-08
- price_fetch_requested_start: 2024-09-08
- price_fetch_clamped_start: 2025-03-07
- first_fetch_attempt_date: 2025-03-07
- raw_price_first_date: 2024-09-09
- raw_price_last_date: 2026-03-06
- first_price_date: 2024-09-09
- last_price_date: 2026-03-06
- first_trading_day: 2025-03-07
- last_trading_day: 2026-03-06
- target_trading_days: 261
- target_trading_days_source: raw_price_cache
- processed_first_date: 2025-03-07
- processed_last_date: 2026-03-05
- missing_processed_dates_count: 1
- first_missing_processed_date: 2026-03-06
- last_missing_processed_date: 2026-03-06
- processed_days: 243
- skipped_days: 1
- last_processed_day: 2026-03-05
- first_trade_date: 2025-03-10
- last_trade_date: 2026-03-06

### Data Coverage Audit

- prices.requested_end_date: 2026-03-06
- prices.latest_available_price_date: 2026-03-06
- prices.coverage_ok: true
- prices.warning: -

### Requested vs Effective Period

- requested_period: 2025-03-07 to 2026-03-06
- effective_period: 2025-03-07 to 2026-03-05
- effective_range_warning: processed days end before requested_end_date; check latest_available_price_date and fetch-period-prices logs

### Hardcoded Date Audit

- target: 2026-03-06
- match_count: 674
- warning: 2026-03-06 remains in config/src/docs/reports/README

### Processed Data Audit

- indicators_last_date: 2026-05-29
- candidates_last_date: 2026-05-29
- scored_candidates_last_date: 2026-03-06
- indicators_count: 1060
- candidates_file_count: 1221
- scored_candidates_file_count: 244
- dates_with_indicators_but_no_candidates: 0
- dates_with_candidates_but_no_scored: 0
- match: config/provider.yaml
- match: src/db.py
- match: src/main.py
- match: src/__pycache__/main.cpython-312.pyc
- match: src/__pycache__/db.cpython-312.pyc
- match: reports/backtest_2026-03-06_to_2026-03-06.json
- match: reports/backtest_2026-03-06_to_2026-03-06.md
- match: reports/day_2026-03-06.md
- match: reports/rookie_dealer_02_v2_67/backtest_2025-03-07_to_2026-03-06.json
- match: reports/rookie_dealer_02_v2_67/trades.csv

## 結果サマリ

- 初期資金: 1,000,000円
- 最終資産: 1,116,809円
- 税引前損益: 116,809円
- 税引後損益: 93,079円
- 税引後損益率: 9.31%
- 勝率: 44.68%
- profit factor: 1.18
- 最大ドローダウン: -12.93%
- 総取引数: 141
- 利確回数: 16
- 損切り回数: 58
- 最大保有期間売却回数: 68
- no trade日数: 76
- selected_count合計: 454

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
