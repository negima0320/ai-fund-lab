# Rule-based 90d Backtest Summary 2024-09-01 to 2026-03-06

## 実行条件

- 期間: 2024-09-01 〜 2026-03-06
- profile: rookie_dealer_02_v2_6 新人ディーラー2号 v2.6
- provider: jquants
- ChatGPT/OpenAI: disabled
- broker: paper
- config_version: cfg_250220d

## 結果サマリ

- 初期資金: 1,000,000円
- 最終資産: 1,191,891円
- 税引前損益: 191,891円
- 税引後損益: 152,908円
- 税引後損益率: 15.29%
- 勝率: 55.56%
- profit factor: 2.10
- 最大ドローダウン: -2.36%
- 総取引数: 90
- 利確回数: 16
- 損切り回数: 34
- 最大保有期間売却回数: 41
- no trade日数: 159
- selected_count合計: 429

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
