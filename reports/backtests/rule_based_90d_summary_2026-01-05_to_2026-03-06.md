# Rule-based 90d Backtest Summary 2026-01-05 to 2026-03-06

## 実行条件

- 期間: 2026-01-05 〜 2026-03-06
- profile: rookie_dealer_02_v2_4 新人ディーラー2号 v2.4
- provider: jquants
- ChatGPT/OpenAI: disabled
- broker: paper
- config_version: cfg_acdb280

## 結果サマリ

- 初期資金: 1,000,000円
- 最終資産: 1,035,218円
- 税引前損益: 35,218円
- 税引後損益: 28,063円
- 税引後損益率: 2.81%
- 勝率: 70.00%
- profit factor: 3.35
- 最大ドローダウン: -1.10%
- 総取引数: 10
- 利確回数: 3
- 損切り回数: 3
- 最大保有期間売却回数: 4
- no trade日数: 21
- selected_count合計: 53

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
