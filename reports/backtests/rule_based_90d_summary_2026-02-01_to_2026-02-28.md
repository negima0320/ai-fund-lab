# Rule-based 90d Backtest Summary 2026-02-01 to 2026-02-28

## 実行条件

- 期間: 2026-02-01 〜 2026-02-28
- profile: rookie_dealer_02_v2_66 新人ディーラー2号 v2.66
- provider: jquants
- ChatGPT/OpenAI: disabled
- broker: paper
- config_version: cfg_f3a7400

## Backtest Date Range Audit

- requested_start_date: 2026-02-01
- requested_end_date: 2026-02-28
- effective_trade_start_date: 2026-02-01
- effective_trade_end_date: 2026-02-28
- indicator_fetch_start_date: 2025-08-05
- price_fetch_requested_start: 2025-08-05
- price_fetch_clamped_start: 2025-08-05
- first_fetch_attempt_date: None
- raw_price_first_date: 2025-08-05
- raw_price_last_date: 2026-02-27
- first_price_date: 2025-08-05
- last_price_date: 2026-02-27
- first_trading_day: 2026-02-02
- last_trading_day: 2026-02-27
- target_trading_days: 20
- target_trading_days_source: raw_price_cache
- processed_first_date: 2026-02-02
- processed_last_date: 2026-02-26
- missing_processed_dates_count: 1
- first_missing_processed_date: 2026-02-27
- last_missing_processed_date: 2026-02-27
- processed_days: 17
- skipped_days: 1
- last_processed_day: 2026-02-26
- first_trade_date: 2026-02-03
- last_trade_date: 2026-02-27

### Data Coverage Audit

- prices.requested_end_date: 2026-02-28
- prices.latest_available_price_date: 2026-02-27
- prices.coverage_ok: false
- prices.warning: price data ends before requested_end_date; backtest can only process cached/fetched price dates

### Requested vs Effective Period

- requested_period: 2026-02-01 to 2026-02-28
- effective_period: 2026-02-02 to 2026-02-26
- effective_range_warning: processed days end before requested_end_date; check latest_available_price_date and fetch-period-prices logs

### Hardcoded Date Audit

- target: 2026-03-06
- match_count: 664
- warning: 2026-03-06 remains in config/src/docs/reports/README

### Processed Data Audit

- indicators_last_date: 2026-05-29
- candidates_last_date: 2026-05-29
- scored_candidates_last_date: 2026-03-06
- indicators_count: 1060
- candidates_file_count: 1221
- scored_candidates_file_count: 42
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
- match: reports/rookie_dealer_02_v2_60/backtest_2025-09-01_to_2026-03-06.md
- match: reports/rookie_dealer_02_v2_60/backtest_2025-09-01_to_2026-03-06.json

## 結果サマリ

- 初期資金: 1,000,000円
- 最終資産: 1,040,564円
- 税引前損益: 31,464円
- 税引後損益: 25,072円
- 税引後損益率: 2.51%
- 勝率: 57.14%
- profit factor: 2.00
- 最大ドローダウン: -2.63%
- 総取引数: 7
- 利確回数: 2
- 損切り回数: 2
- 最大保有期間売却回数: 3
- no trade日数: 5
- selected_count合計: 41

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
