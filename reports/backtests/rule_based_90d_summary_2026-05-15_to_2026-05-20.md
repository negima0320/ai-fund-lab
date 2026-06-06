# Rule-based 90d Backtest Summary 2026-05-15 to 2026-05-20

## 実行条件

- 期間: 2026-05-15 〜 2026-05-20
- profile: rookie_dealer_02_v2_68_ml_ranked_exit_ai_050 新人ディーラー2号 v2.68 ML ranked + Exit AI 0.50
- provider: jquants
- ChatGPT/OpenAI: disabled
- broker: paper
- config_version: cfg_13bd34d

## Backtest Date Range Audit

- requested_start_date: 2026-05-15
- requested_end_date: 2026-05-20
- effective_trade_start_date: 2026-05-15
- effective_trade_end_date: 2026-05-20
- indicator_fetch_start_date: 2025-11-16
- price_fetch_requested_start: 2025-11-16
- price_fetch_clamped_start: 2025-11-16
- first_fetch_attempt_date: None
- raw_price_first_date: 2025-11-17
- raw_price_last_date: 2026-05-20
- first_price_date: 2025-11-17
- last_price_date: 2026-05-20
- first_trading_day: 2026-05-15
- last_trading_day: 2026-05-20
- target_trading_days: 4
- target_trading_days_source: raw_price_cache
- processed_first_date: 2026-05-15
- processed_last_date: 2026-05-19
- missing_processed_dates_count: 1
- first_missing_processed_date: 2026-05-20
- last_missing_processed_date: 2026-05-20
- processed_days: 3
- skipped_days: 1
- last_processed_day: 2026-05-19
- first_trade_date: 2026-05-18
- last_trade_date: 2026-05-20

### Data Coverage Audit

- prices.requested_end_date: 2026-05-20
- prices.latest_available_price_date: 2026-05-20
- prices.coverage_ok: true
- prices.warning: -

### Requested vs Effective Period

- requested_period: 2026-05-15 to 2026-05-20
- effective_period: 2026-05-15 to 2026-05-19
- effective_range_warning: processed days end before requested_end_date; check latest_available_price_date and fetch-period-prices logs

### Hardcoded Date Audit

- target: 2026-03-06
- match_count: 756
- warning: 2026-03-06 remains in config/src/docs/reports/README

### Processed Data Audit

- indicators_last_date: 2026-05-29
- candidates_last_date: 2026-05-29
- scored_candidates_last_date: 2026-05-20
- indicators_count: 1060
- candidates_file_count: 1221
- scored_candidates_file_count: 4
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
- match: reports/rookie_dealer_02_v2_69/backtest_2021-06-01_to_2026-05-31.json
- match: reports/rookie_dealer_02_v2_69/backtest_2026-01-05_to_2026-03-06.json

## 結果サマリ

- 初期資金: 1,000,000円
- 最終資産: 1,013,300円
- 税引前損益: 0円
- 税引後損益: 0円
- 税引後損益率: 0.00%
- 勝率: N/A
- profit factor: N/A
- 最大ドローダウン: 0.00%
- 総取引数: 0
- 利確回数: 0
- 損切り回数: 0
- 最大保有期間売却回数: 0
- no trade日数: 1
- selected_count合計: 5

## 新人ディーラー1号コメント

売却済み取引がないため、勝率や売却ルールの評価はまだ保留します。ルールに従い、検証期間を広げます。

## 次に見るべき改善ポイント

- selected_count_total と no_trade_days を見て、スクリーニングが厳しすぎないか確認する。
- profit factor と勝率を合わせて、利幅と損切り幅のバランスを確認する。
- 最大ドローダウンがrisk_marginの許容範囲内か確認する。
- 利確、損切り、最大保有期間売却の比率を見て、出口ルールが短期売買に合っているか確認する。
- J-Quants Freeプランの12週間遅延データ前提で、複数期間でも再現するか確認する。

## 注意

OpenAI / ChatGPT APIは使わず、ルールベースのみで検証しています。PaperBrokerによる仮想売買であり、実売買は行いません。
