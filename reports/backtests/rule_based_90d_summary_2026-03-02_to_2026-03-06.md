# Rule-based 90d Backtest Summary 2026-03-02 to 2026-03-06

## 実行条件

- 期間: 2026-03-02 〜 2026-03-06
- profile: rookie_dealer_01 新人ディーラー1号
- provider: jquants
- ChatGPT/OpenAI: disabled
- broker: paper
- config_version: cfg_d5be056

## 結果サマリ

- 初期資金: 1,000,000円
- 最終資産: 1,000,000円
- 税引前損益: 0円
- 税引後損益: 0円
- 税引後損益率: 0.00%
- 勝率: N/A
- profit factor: N/A
- 最大ドローダウン: 0.00%
- 総取引数: 3
- 利確回数: 0
- 損切り回数: 0
- 最大保有期間売却回数: 0
- no trade日数: 3
- selected_count合計: 2

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
